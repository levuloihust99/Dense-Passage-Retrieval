"""Split train/test data. This module is used for data from Legal Text Retrieval competition only."""

import json
import os
import argparse
from typing import Text

import numpy as np
import tensorflow as tf


def load_data(data_path: Text, qa_file: Text):
    all_data_path = os.path.join(data_path, qa_file)
    with tf.io.gfile.GFile(all_data_path, 'r') as reader:
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
    for idx in indices:
        if idx < train_size:
            train_items.append(all_items[idx])
        else:
            test_items.append(all_items[idx])
    return {'_name_': 'train', '_count_': len(train_items), 'items': train_items}, \
        {'name': '_test_', '_count_': len(test_items), 'items': test_items}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--data-path", default='data')
    parser.add_argument("--output-file-name", default='data.json')
    parser.add_argument("--qa-file", default="train_question_answer.json")
    args = parser.parse_args()

    train_data_path = os.path.join(args.data_path, 'train', args.output_file_name)
    train_dir = os.path.basename(train_data_path)
    if not tf.io.gfile.exists(train_dir):
        tf.io.gfile.makedirs(train_dir)
    
    test_data_path = os.path.join(args.data_path, 'test', args.output_file_name)
    test_dir = os.path.basename(test_data_path)
    if not tf.io.gfile.exists(test_dir):
        tf.io.gfile.makedirs(test_dir)

    all_data = load_data(data_path=args.data_path, qa_file=args.qa_file)
    train_data, test_data = train_test_split(all_data, args.train_ratio)
    with tf.io.gfile.GFile(train_data_path, 'w') as writer:
        writer.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    with tf.io.gfile.GFile(test_data_path, 'w') as writer:
        writer.write(json.dumps(test_data, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
