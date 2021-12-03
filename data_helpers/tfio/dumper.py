import tensorflow as tf
import os
import logging
from typing import Text, Dict, List

from data_helpers.data_utils import tokenize_qa, tokenize_context
from data_helpers.tfio.feature_utils import (
    create_byte_feature,
    create_int_feature
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serialize_qa_pair(
    qa_pair: Dict[Text, List],
):
    features = {}
    for key, value in qa_pair.items():
        value_serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
        value_feature = create_byte_feature(value_serialized)
        features[key] = value_feature
    
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


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
        pair_tokenized = tokenize_qa(
            qa_pair=pair,
            tokenizer=tokenizer,
            query_max_seq_length=query_max_seq_length,
            context_max_seq_length=context_max_seq_length
        )
        tf_example_serialized = serialize_qa_pair(pair_tokenized)

        if counter % num_examples_per_file == 0 and counter > 0:
            example_writer.close()
            logger.info("Done writing {} examples".format(counter))
            idx += 1
            example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'data_{:03d}.tfrecord'.format(idx)))
        
        example_writer.write(tf_example_serialized)
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
    example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(idx)))
    for article in corpus:
        context = {
            'title': article.get('title'),
            'text':  article.get('text')
        }
        context_inputs = tokenize_context(context, tokenizer, context_max_seq_length)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': create_int_feature(context_inputs.get('input_ids')),
            'attention_mask': create_int_feature(context_inputs.get('attention_mask'))
        }))

        if counter % num_examples_per_file == 0 and counter > 0:
            example_writer.close()
            logger.info("Done writing {} examples".format(counter))
            idx += 1
            example_writer = tf.io.TFRecordWriter(os.path.join(tfrecord_dir, 'corpus_{:03d}.tfrecord'.format(idx)))
        
        example_writer.write(tf_example.SerializeToString())
        counter += 1
    
    example_writer.close()
    logger.info("Done writing {} examples".format(counter))
