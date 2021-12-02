import argparse
import logging
import json
import os
from typing import Text, Dict, List, Union
import tensorflow as tf

from dataprocessor.loader import load_corpus_dataset

from dual_encoder.configuration import DualEncoderConfig
from dual_encoder.constants import ARCHITECTURE_MAPPINGS
from dual_encoder.modeling import DualEncoder
from utils.logging import add_color_formater
from utils.setup import setup_memory_growth, setup_distribute_strategy
from dataprocessor.dumper import load_corpus_to_list, tensorize_context
from indexing.faiss_indexer import DenseFlatIndexer


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


def load_context_encoder(config: DualEncoderConfig):
    ckpt_path = tf.train.latest_checkpoint(config.checkpoint_path)
    encoder_class = ARCHITECTURE_MAPPINGS[config.model_arch]
    context_encoder = encoder_class.from_pretrained(config.pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(context_encoder=context_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(ckpt_path).expect_partial()
    return context_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()
    config = DualEncoderConfig.from_json_file(args.config_file)
    
    # setup environment
    setup_memory_growth()
    strategy = setup_distribute_strategy(use_tpu=config.use_tpu, tpu_name=config.tpu_name)

    dataset, num_examples = load_corpus_dataset(config)
    if num_examples % (config.eval_batch_size * strategy.num_replicas_in_sync) != 0:
        num_forwards = num_examples // (config.eval_batch_size * strategy.num_replicas_in_sync)
        num_fake_examples = (num_forwards + 1) * config.eval_batch_size * strategy.num_replicas_in_sync - num_examples
        fake_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tf.zeros([num_fake_examples, config.context_max_seq_length], dtype=tf.int32),
            'attention_mask': tf.ones([num_fake_examples, config.context_max_seq_length], dtype=tf.int32)
        })
        dataset = dataset.concatenate(fake_dataset)

    dataset = dataset.batch(batch_size=config.eval_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    # instantiate model
    with strategy.scope():
        context_encoder = load_context_encoder(config)
    
    embeddings = generate_embeddings(context_encoder, dist_dataset, strategy)
    corpus = load_corpus_to_list(os.path.join(config.data_dir, 'legal_corpus.json'))
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