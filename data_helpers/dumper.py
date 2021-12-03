from ast import dump
import tensorflow as tf
import json
import os
import logging
import argparse
from typing import Text, Dict, List, Any

from dual_encoder.configuration import DualEncoderConfig
from dual_encoder.constants import ARCHITECTURE_MAPPINGS
from data_helpers.data_utils import (
    load_corpus_to_dict,
    load_corpus_to_list,
    load_qa_data,
    build_query_context_pairs,
    tensorize_question,
    tensorize_context,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  return feature


def dump_qa(
    query_context_pairs,
    tokenizer,
    query_max_seq_length: int,
    context_max_seq_length: int,
    tfrecord_dir,
    num_examples_per_file: int = 1000
):
    counter = 0
    idx = 0
    example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'data_{:03d}.tfrecord'.format(idx)))
    for pair in query_context_pairs:
        question = pair.get('question')
        question_inputs = tensorize_question(question, tokenizer, query_max_seq_length)
        context = pair.get('context')
        context_inputs = tensorize_context(context, tokenizer, context_max_seq_length)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'query_input_ids': create_int_feature(question_inputs.get('input_ids')),
            'query_attention_mask': create_int_feature(question_inputs.get('attention_mask')),
            'context_input_ids': create_int_feature(context_inputs.get('input_ids')),
            'context_attention_mask': create_int_feature(context_inputs.get('attention_mask'))
        }))

        if counter % num_examples_per_file == 0 and counter > 0:
            example_writer.close()
            logger.info("Done writing {} examples".format(counter))
            idx += 1
            example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'data_{:03d}.tfrecord'.format(idx)))
        
        example_writer.write(tf_example.SerializeToString())
        counter += 1

    example_writer.close()
    logger.info("Done writing {} examples".format(counter))


def dump_corpus(
    corpus,
    tokenizer,
    context_max_seq_length: int,
    tfrecord_dir,
    num_examples_per_file: int = 5000
):
    counter = 0
    idx = 0
    example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'data_{:03d}.tfrecord'.format(idx)))
    for article in corpus:
        context = {
            'title': article.get('title'),
            'text':  article.get('text')
        }
        context_inputs = tensorize_context(context, tokenizer, context_max_seq_length)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': create_int_feature(context_inputs.get('input_ids')),
            'attention_mask': create_int_feature(context_inputs.get('attention_mask'))
        }))

        if counter % num_examples_per_file == 0 and counter > 0:
            example_writer.close()
            logger.info("Done writing {} examples".format(counter))
            idx += 1
            example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'data_{:03d}.tfrecord'.format(idx)))
        
        example_writer.write(tf_example.SerializeToString())
        counter += 1
    
    example_writer.close()
    logger.info("Done writing {} examples".format(counter))


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--dump-corpus", action='store_const', const=True, default=False)
    parser.add_argument("--dump-qa", action='store_const', const=True, default=False)
    parser.add_argument("--query-max-seq-length", type=int)
    parser.add_argument("--context-max-seq-length", type=int)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--data-dir")
    parser.add_argument("--architecture", default="distilbert", choices=['roberta', 'distilbert', 'bert'])
    parser.add_argument("--tokenizer-path", default='pretrained/NlpHUST/vibert4news-base-cased')
    parser.add_argument("--corpus-path", default='legal_corpus.json')
    parser.add_argument("--qa-file", default='train/data.json')
    args = parser.parse_args()

    if args.config_path:
        config = DualEncoderConfig.from_json_file(args.config_path)
    else:
        hparams = {}
        if hasattr(args, 'query_max_seq_length'):
            hparams['query_max_seq_length'] = args.query_max_seq_length
        if hasattr(args, 'context_max_seq_length'):
            hparams['context_max_seq_length'] = args.context_max_seq_length
        if hasattr(args, 'data_dir'):
            hparams['data_dir'] = args.data_dir
        if hasattr(args, 'tokenizer_path'):
            hparams['tokenizer_path'] = args.tokenizer_path
        config = DualEncoderConfig(**hparams)
    
    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(config.tokenizer_path)

    if args.dump_qa:
        qa_tfrecord_dir = config.data_tfrecord_dir
        if not tf.io.gfile.exists(qa_tfrecord_dir):
            tf.io.gfile.makedirs(qa_tfrecord_dir)

        corpus_path = os.path.join(config.data_dir, args.corpus_path)
        corpus = load_corpus_to_dict(corpus_path)
        qa_data_path = os.path.join(config.data_dir, args.qa_file)
        train_data = load_qa_data(qa_data_path)
        query_context_pairs = build_query_context_pairs(corpus, train_data)
        
        dump_qa(
            query_context_pairs=query_context_pairs,
            tokenizer=tokenizer,
            query_max_seq_length=config.query_max_seq_length,
            context_max_seq_length=config.context_max_seq_length,
            tfrecord_dir=qa_tfrecord_dir,
        )
    if args.dump_corpus:
        corpus_tfrecord = os.path.join(config.data_dir, 'tfrecord' ,'corpus')
        if not tf.io.gfile.exists(corpus_tfrecord):
            tf.io.gfile.makedirs(corpus_tfrecord)
        
        corpus_path = os.path.join(config.data_dir, args.corpus_path)
        corpus = load_corpus_to_list(corpus_path)

        dump_corpus(
            corpus=corpus,
            tokenizer=tokenizer,
            context_max_seq_length=config.context_max_seq_length,
            tfrecord_dir=corpus_tfrecord,
        )


if __name__ == "__main__":
    main()
