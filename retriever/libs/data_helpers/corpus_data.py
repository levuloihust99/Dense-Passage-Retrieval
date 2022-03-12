import os
import glob
import tensorflow as tf
import logging
import json
from typing import List, Tuple, Dict, Text
import argparse
from transformers import PhobertTokenizer, BertTokenizer

from .build_tfrecord import tokenize_context
from libs.constants import TOKENIZER_MAPPING

logger = logging.getLogger(__name__)


def dump_corpus(
    corpus,
    tokenizer,
    max_context_length: int,
    tfrecord_dir: Text,
    num_examples_per_file: int = 5000,
):
    counter = 0
    example_writer = tf.io.TFRecordWriter(os.path.join(
        tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(counter)))
    for idx, article in enumerate(corpus):
        context = {
            'title': article["title"],
            'text':  article["text"]
        }
        context_inputs = tokenize_context(
            tokenizer, context, max_context_length)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_inputs["input_ids"])
            ),
            'attention_mask': tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_inputs["attention_mask"])
            )
        }))

        if (idx + 1) % num_examples_per_file == 0:
            example_writer.close()
            logger.info("Done writing {} examples".format(idx + 1))
            counter += 1
            example_writer = tf.io.TFRecordWriter(os.path.join(
                tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(counter)))

        example_writer.write(tf_example.SerializeToString())

    example_writer.close()
    logger.info("Done writing {} examples".format(counter))


def load_corpus_dataset(
    data_source: Text,
    max_context_length: int
):
    tfrecord_files = sorted(glob.glob(os.path.join(data_source, "*")))
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    count = 0
    for _ in dataset:
        count += 1

    feature_description = {
        "input_ids": tf.io.FixedLenFeature(shape=[max_context_length], dtype=tf.int64),
        "attention_mask": tf.io.FixedLenFeature(shape=[max_context_length], dtype=tf.int64)
    }
    def _parse_ex(ex):
        return tf.io.parse_example(ex, feature_description)

    dataset = dataset.map(_parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda x: {
            "input_ids": tf.cast(x["input_ids"], dtype=tf.int32),
            "attention_mask": tf.cast(x["attention_mask"], dtype=tf.int32)
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset, count
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--tokenizer-type", default="roberta")
    parser.add_argument("--tokenizer-path", default="vinai/phobert-base")
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--num-examples-per-file", type=int, default=5000)
    args = parser.parse_args()

    with open(args.corpus_path, "r") as reader:
        corpus = json.load(reader)
    tokenizer = TOKENIZER_MAPPING[args.tokenizer_type].from_pretrained(args.tokenizer_path)
    dump_corpus(
        corpus=corpus,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        tfrecord_dir=args.tfrecord_dir,
        num_examples_per_file=args.num_examples_per_file
    )


if __name__ == "__main__":
    main()