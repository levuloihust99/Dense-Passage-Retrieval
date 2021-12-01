import logging
import argparse
import os
import json
from typing import Text
import tensorflow as tf

from transformers import TFDistilBertModel

from dual_encoder.configuration import DualEncoderConfig
from utils.logging import add_color_formater
from dataprocessor.dumper import load_corpus_to_dict


logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger()

def load_test_questions(path: Text):
    with tf.io.gfile.GFile(path, 'r') as reader:
        data = json.load(reader)
    return data.get('items')


def load_query_encoder(config: DualEncoderConfig):
    ckpt_path = tf.train.latest_checkpoint(config.checkpoint_path)
    query_encoder = TFDistilBertModel.from_pretrained(config.pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(query_encoder=query_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(ckpt_path).expect_partial()
    return query_encoder


def evaluate(config: DualEncoderConfig):
    corpus = load_corpus_to_dict(os.path.join(config.data_dir, 'legal_corpus.json'))
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--questions-path", required=True)
    args = parser.parse_args()
    config = DualEncoderConfig.from_json_file(args.config_file)
