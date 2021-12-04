"""Split train/test data. This module is used for data from Legal Text Retrieval competition only."""

import json
import os
import argparse
import logging
import time
from typing import Text

import numpy as np
import tensorflow as tf

from utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def load_data(data_path: Text):
    with tf.io.gfile.GFile(data_path, 'r') as reader:
        all_data = json.load(reader)
    return all_data


def train_test_split(all_data, train_ratio: float=0.8):
    all_items = all_data.get('items')
    indices = np.arange(len(all_items))
    np.random.shuffle(indices)
    indices = indices.tolist()

    train_size = int(len(indices) * train_ratio)

    train_items = []
    test_items = []
    for idx in indices[:train_size]:
        train_items.append(all_items[idx])
    for idx in indices[train_size:]:
        test_items.append(all_items[idx])
    return {'_name_': 'train', '_count_': len(train_items), 'items': train_items}, \
        {'name': '_test_', '_count_': len(test_items), 'items': test_items}, indices[:train_size], indices[train_size:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--qa-file", default="data/train_question_answer.json")
    parser.add_argument("--out-train-path", default="data/json/train/data.json")
    parser.add_argument("--out-test-path", default="data/json/test/data.json")
    args = parser.parse_args()

    train_dir = os.path.dirname(args.out_train_path)
    if not tf.io.gfile.exists(train_dir):
        tf.io.gfile.makedirs(train_dir)
    
    test_dir = os.path.dirname(args.out_test_path)
    if not tf.io.gfile.exists(test_dir):
        tf.io.gfile.makedirs(test_dir)

    logger.info("Load question-context pairs...")
    start_time = time.perf_counter()
    all_data = load_data(data_path=args.qa_file)
    logger.info("Done loading question-context pairs in {}s".format(time.perf_counter() - start_time))

    logger.info("Splitting train/test data...")
    start_time = time.perf_counter()
    train_data, test_data, train_indices, test_indices = train_test_split(all_data, args.train_ratio)
    logger.info("Done splitting train/test data in {}s".format(time.perf_counter() - start_time))

    with tf.io.gfile.GFile(args.out_train_path, 'w') as writer:
        writer.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    with tf.io.gfile.GFile(os.path.join(train_dir, 'indices.json'), 'w') as writer:
        writer.write(json.dumps(train_indices, ensure_ascii=False))
    with tf.io.gfile.GFile(args.out_test_path, 'w') as writer:
        writer.write(json.dumps(test_data, indent=4, ensure_ascii=False))
    with tf.io.gfile.GFile(os.path.join(test_dir, 'indices.json'), 'w') as writer:
        writer.write(json.dumps(test_indices, ensure_ascii=False))


if __name__ == "__main__":
    main()
