import os
import glob
import tensorflow as tf
import logging
import json
import jsonlines
from typing import List, Tuple, Dict, Text
import argparse
from transformers import PhobertTokenizer, BertTokenizer
import queue
import multiprocessing
from multiprocessing import Process, Queue

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
    if not os.path.exists(tfrecord_dir):
        os.makedirs(tfrecord_dir)

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
    logger.info("Done writing {} examples".format(idx + 1))


def load_corpus_dataset(
    data_source: Text,
    max_context_length: int,
    skip_size: int = None
):
    tfrecord_files = tf.io.gfile.listdir(data_source)
    tfrecord_files = [os.path.join(data_source, f) for f in tfrecord_files]
    tfrecord_files = sorted(tfrecord_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x),
    )
    if skip_size is not None:
        dataset = dataset.skip(skip_size)

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
    return dataset


def worker_job(from_master_queue, to_master_queue):
    while True:
        item = from_master_queue.get()
        context = {
            'title': item["title"],
            'text':  item["text"]
        }
        context_inputs = tokenize_context(
            tokenizer, context, args.max_context_length)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_inputs["input_ids"])
            ),
            'attention_mask': tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_inputs["attention_mask"])
            )
        }))

        to_master_queue.put(tf_example)


def feed_process(feed_queues):
    num_jobs = args.num_processes
    for idx, item in enumerate(corpus):
        queue_idx = idx % num_jobs
        feed_queues[queue_idx].put(item)


def fetch_process(fetch_queues):
    if not os.path.exists(args.tfrecord_dir):
        os.makedirs(args.tfrecord_dir)
    
    idx = 0
    counter = 0
    example_writer = tf.io.TFRecordWriter(os.path.join(
        args.tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(counter)))

    is_done = False
    while not is_done:
        for q in fetch_queues:
            try:
                tf_example = q.get(timeout=50)
            except queue.Empty:
                is_done = True
                break

            example_writer.write(tf_example.SerializeToString())

            if (idx + 1) % args.num_examples_per_file == 0:
                example_writer.close()
                logger.info("Done writing {} examples".format(idx + 1))
                counter += 1
                example_writer = tf.io.TFRecordWriter(os.path.join(
                    args.tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(counter)))

            idx += 1
            
    example_writer.close()
    logger.info("Done writing {} examples".format(idx))


def parallel_processing():
    num_jobs = args.num_processes

    feed_queues = [Queue() for _ in range(num_jobs)]
    fetch_queues = [Queue() for _ in range(num_jobs)]
    jobs = [Process(target=worker_job, args=(feed_queues[idx], fetch_queues[idx])) for idx in range(num_jobs)]

    # start feeding job
    feed_job = Process(target=feed_process, args=(feed_queues,))
    feed_job.start()

    # start processing job
    for job in jobs:
        job.start()
    
    # fetching job
    fetch_process(fetch_queues)

    # cleanup
    for job in jobs:
        job.kill()


def single_processing():
    dump_corpus(
        corpus=corpus,
        tokenizer=tokenizer,
        max_context_length=args.max_context_length,
        tfrecord_dir=args.tfrecord_dir,
        num_examples_per_file=args.num_examples_per_file
    )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--tokenizer-type", default="roberta")
    parser.add_argument("--tokenizer-path", default="vinai/phobert-base")
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--num-examples-per-file", type=int, default=5000)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--data-format", choices=["json", "jsonlines"], default="json")

    global args
    args = parser.parse_args()

    global corpus
    if args.data_format == "json":
        with open(args.corpus_path, "r") as reader:
            corpus = json.load(reader)
    else:
        corpus = jsonlines.open(args.corpus_path, "r")

    global tokenizer
    tokenizer = TOKENIZER_MAPPING[args.tokenizer_type].from_pretrained(args.tokenizer_path)

    if args.num_processes > 1:
        parallel_processing()
    else:
        single_processing()
    
    if args.data_format == "jsonlines":
        corpus.close()


if __name__ == "__main__":
    main()
