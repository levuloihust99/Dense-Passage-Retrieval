import argparse
import logging
import json
import os
from typing import Text, Dict, List, Union, Literal
import tensorflow as tf

from libs.data_helpers.corpus_data import load_corpus_dataset
from libs.constants import MODEL_MAPPING
from libs.nn.modeling import DualEncoder
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


def load_context_encoder(
    checkpoint_dir: Text,
    architecture: Literal["bert", "roberta"],
    pretrained_model_path: Text,
    checkpoint_name=None
):
    if checkpoint_name:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt_path:
        encoder_class = MODEL_MAPPING[architecture]
        context_encoder = encoder_class.from_pretrained(pretrained_model_path)
        dual_encoder = tf.train.Checkpoint(context_encoder=context_encoder)
        ckpt = tf.train.Checkpoint(model=dual_encoder)
        ckpt.restore(ckpt_path).expect_partial()
        logger.info("Restored checkpoint from {}".format(ckpt_path))
    else:
        raise Exception("Checkpoint not found in the given directory '{}'".format(checkpoint_dir))
    return context_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    with tf.io.gfile.GFile(args.config_file, "r") as reader:
        config = json.load(reader)
    config = argparse.Namespace(**config)
    
    # setup environment
    setup_memory_growth()
    strategy = setup_distribute_strategy(
        use_tpu=config.use_tpu,
        tpu_name=config.tpu_name,
        zone=config.tpu_zone if hasattr(config, "tpu_zone") else None,
        project=config.gcp_project if hasattr(config, "gcp_project") else None
    )

    dataset, num_examples = load_corpus_dataset(
        data_source=config.corpus_tfrecord_dir,
        max_context_length=config.max_context_length
    )
    if num_examples % (config.eval_batch_size * strategy.num_replicas_in_sync) != 0:
        num_forwards = num_examples // (config.eval_batch_size * strategy.num_replicas_in_sync)
        num_fake_examples = (num_forwards + 1) * config.eval_batch_size * strategy.num_replicas_in_sync - num_examples
        fake_dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tf.zeros([num_fake_examples, config.max_context_length], dtype=tf.int32),
            'attention_mask': tf.ones([num_fake_examples, config.max_context_length], dtype=tf.int32)
        })
        dataset = dataset.concatenate(fake_dataset)

    dataset = dataset.batch(batch_size=config.eval_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dist_dataset = strategy.distribute_datasets_from_function(
        lambda _: dataset
    )

    # instantiate model
    with strategy.scope():
        context_encoder = load_context_encoder(
            checkpoint_dir=config.checkpoint_dir,
            architecture=config.architecture,
            pretrained_model_path=config.pretrained_model_path,
            checkpoint_name=config.checkpoint_name
        )
    
    embeddings = generate_embeddings(context_encoder, dist_dataset, strategy)
    with tf.io.gfile.GFile(config.corpus_path, "r") as reader:
        corpus = json.load(reader)
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

    # config.index_path is path to a file
    if tf.io.gfile.exists(config.index_path) and not tf.io.gfile.isdir(config.index_path):
        index_dir = os.path.basename(config.index_path)
    else:
        index_dir = config.index_path
    
    if not tf.io.gfile.exists(index_dir):
        tf.io.gfile.makedirs(index_dir)

    indexer.serialize(config.index_path)


if __name__ == "__main__":
    main()