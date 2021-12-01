import tensorflow as tf
import json
import os
import logging
from typing import Text, Dict, List, Any
from transformers import BertTokenizer

from dual_encoder.configuration import DualEncoderConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_corpus(corpus_path: Text):
    with tf.io.gfile.GFile(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    corpus_restructured = {}
    for doc in corpus:
        articles_restructured = {}
        for article in doc.get('articles'):
            articles_restructured[article.get('article_id')] = {
                'title': article.get('title'),
                'text': article.get('text')
            }
        corpus_restructured[doc.get('law_id')] = articles_restructured

    return corpus_restructured


def load_train_data(train_data_path: Text):
    with tf.io.gfile.GFile(train_data_path, 'r') as reader:
        train_data = json.load(reader)
    return train_data.get('items')


def build_query_context_pairs(corpus, train_data):
    query_context_pairs = []
    for record in train_data:
        question = record.get('question')
        relevant_articles = record.get('relevant_articles')
        context_article = relevant_articles[0]
        context = corpus.get(context_article.get('law_id')).get(context_article.get('article_id'))
        query_context_pairs.append({
            'question': question,
            'context': context
        })
    return query_context_pairs


def tensorize_question(question, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(question)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    mask = [1] * len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = max_seq_length - len(tokens)
    token_ids = token_ids + [tokenizer.pad_token_id] * pad_length
    mask = mask + [0] * pad_length
    return {
        'input_ids': token_ids,
        'attention_mask': mask
    }


def tensorize_context(context, tokenizer, max_seq_length):
    title = context.get('title')
    text = context.get('text')
    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    tokens = title_tokens + [tokenizer.sep_token] + text_tokens
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    mask = [1] * len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = max_seq_length - len(tokens)
    token_ids = token_ids + [tokenizer.pad_token_id] * pad_length
    mask = mask + [0] * pad_length

    return {
        'input_ids': token_ids,
        'attention_mask': mask
    }


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  return feature


def dump(
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


def main():
    config = DualEncoderConfig()
    tfrecord_dir = config.data_tfrecord_dir
    if not tf.io.gfile.exists(tfrecord_dir):
        tf.io.gfile.makedirs(tfrecord_dir)

    corpus_path = os.path.join(config.data_dir, 'legal_corpus.json')
    corpus = load_corpus(corpus_path)
    train_data_path = os.path.join(config.data_dir, 'train_question_answer.json')
    train_data = load_train_data(train_data_path)
    query_context_pairs = build_query_context_pairs(corpus, train_data)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)
    dump(
        query_context_pairs=query_context_pairs,
        tokenizer=tokenizer,
        query_max_seq_length=config.query_max_seq_length,
        context_max_seq_length=config.context_max_seq_length,
        tfrecord_dir=tfrecord_dir,
    )


if __name__ == "__main__":
    main()