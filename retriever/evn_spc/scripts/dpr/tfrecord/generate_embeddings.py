import argparse
import logging
import json
import os
from typing import Text, Dict, List, Union
import tensorflow as tf

from libs.data_helpers.tfio.dual_encoder.loader import load_corpus_dataset
from libs.data_helpers.data_utils import load_corpus_to_list
from libs.nn.configuration.dual_encoder import DualEncoderConfig
from libs.nn.constants import ARCHITECTURE_MAPPINGS
from libs.nn.modeling.dual_encoder import DualEncoder
from libs.utils.logging import add_color_formater
from libs.utils.setup import setup_memory_growth, setup_distribute_strategy
from libs.faiss_indexer import DenseFlatIndexer


logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger()


def generate_embeddings(
    context_encoder: tf.keras.Model,
    dataset: Union[tf.data.Dataset, tf.distribute.DistributedDataset],
    strategy: tf.distribute.Strategy
):
    @tf.function
    def step_fn(features):
        outputs = context_encoder(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            return_dict=True,
            training=False
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return pooled_output
    
    embeddings = []
    for idx, features in enumerate(dataset):
        per_replicas_embeddings = strategy.run(step_fn, args=(features,))
        
        if strategy.num_replicas_in_sync > 1:
            batch_embeddings = tf.concat(per_replicas_embeddings.values, axis=0)
        else:
            batch_embeddings = per_replicas_embeddings
        logger.info("Done generate embeddings for {} articles".format((idx + 1) * batch_embeddings.shape[0]))

        embeddings.extend(batch_embeddings)
    return embeddings


def load_corpus(corpus_path):
    with open(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    return corpus


def load_context_encoder(checkpoint_dir, model_arch, pretrained_model_path):
    ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    encoder_class = ARCHITECTURE_MAPPINGS[model_arch]['model_class']
    context_encoder = encoder_class.from_pretrained(pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(context_encoder=context_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(ckpt_path).expect_partial()
    return context_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-max-seq-length", required=True, type=int)
    parser.add_argument("--eval-batch-size", required=True, type=int)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-arch", required=True, choices=['roberta', 'bert', 'distilbert'])
    parser.add_argument("--pretrained-model-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--corpus-tfrecord-dir", required=True)
    args = parser.parse_args()
    
    # setup environment
    setup_memory_growth()
    strategy = setup_distribute_strategy(use_tpu=False, tpu_name=None)

    dataset, num_examples = load_corpus_dataset(
        tfrecord_dir=args.corpus_tfrecord_dir,
        context_max_seq_length=args.context_max_seq_length
    )
    if num_examples % (args.eval_batch_size * strategy.num_replicas_in_sync) != 0:
        num_forwards = num_examples // (args.eval_batch_size * strategy.num_replicas_in_sync)
        num_fake_examples = (num_forwards + 1) * args.eval_batch_size * strategy.num_replicas_in_sync - num_examples
        fake_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tf.zeros([num_fake_examples, args.context_max_seq_length], dtype=tf.int32),
            'attention_mask': tf.ones([num_fake_examples, args.context_max_seq_length], dtype=tf.int32)
        })
        dataset = dataset.concatenate(fake_dataset)

    dataset = dataset.batch(batch_size=args.eval_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    # instantiate model
    with strategy.scope():
        context_encoder = load_context_encoder(args.checkpoint_dir, args.model_arch, args.pretrained_model_path)
    
    embeddings = generate_embeddings(context_encoder, dist_dataset, strategy)
    corpus = load_corpus(args.corpus_path)
    embeddings = embeddings[:len(corpus)]
    embeddings = [e.numpy() for e in embeddings]

    data_to_be_indexed = []
    for idx in range(len(corpus)):
        data_to_be_indexed.append((
            corpus[idx],
            embeddings[idx]
        ))

    # index data
    indexer = DenseFlatIndexer()
    indexer.init_index(vector_sz=embeddings[0].shape[0])
    indexer.index_data(data_to_be_indexed)

    if os.path.isfile(args.index_path):
        index_dir = os.path.basename(args.index_path)
    else:
        index_dir = args.index_path
    
    if tf.io.gfile.exists(index_dir):
        tf.io.gfile.makedirs(index_dir)

    indexer.serialize(args.index_path)


if __name__ == "__main__":
    main()