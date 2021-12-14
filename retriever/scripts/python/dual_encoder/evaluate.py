"""This code can not run on TPU"""

import logging
import argparse
import os
import time
import json
from typing import Text, Dict, List, Any

import numpy as np
import tensorflow as tf

from libs.nn.configuration.dual_encoder import DualEncoderConfig
from libs.nn.constants import ARCHITECTURE_MAPPINGS
from libs.utils.logging import add_color_formater
from libs.utils.setup import setup_memory_growth
from libs.utils.evaluation import calculate_metrics
from libs.data_helpers.data_utils import load_corpus_to_dict, tokenize_question
from libs.faiss_indexer import DenseFlatIndexer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_test_data(path: Text):
    with tf.io.gfile.GFile(path, 'r') as reader:
        data = json.load(reader)
    return data.get('items')


def load_query_encoder(config: DualEncoderConfig):
    logger.info("Loading query encoder...")
    start_time = time.perf_counter()

    ckpt_path = tf.train.latest_checkpoint(config.checkpoint_dir)
    encoder_class = ARCHITECTURE_MAPPINGS[config.model_arch]['model_class']
    query_encoder = encoder_class.from_pretrained(config.pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(query_encoder=query_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(ckpt_path).expect_partial()

    logger.info("Done loading query encoder in {}s".format(
        time.perf_counter() - start_time))
    return query_encoder


def create_query_dataset(
    queries: List[Text],
    tokenizer,
    query_max_seq_length: int,
    batch_size: int
):
    logger.info("Creating query dataset...")
    start_time = time.perf_counter()
    query_tensors = []
    for query in queries:
        query_tensors.append(tokenize_question(
            query, tokenizer, query_max_seq_length))
    query_tensors = {
        'input_ids': [tf.convert_to_tensor(q.get('input_ids')) for q in query_tensors],
        'attention_mask': [tf.convert_to_tensor(q.get('attention_mask')) for q in query_tensors]
    }
    logger.info("Done creating query dataset in {}s".format(
        time.perf_counter() - start_time))
    return tf.data.Dataset.from_tensor_slices(query_tensors).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def generate_query_embeddings(
    query_dataset: tf.data.Dataset,
    query_encoder: tf.keras.Model
) -> np.ndarray:
    logger.info("Generating query embeddings...")
    start_time = time.perf_counter()
    query_embeddings = []
    for batch in query_dataset:
        outputs = query_encoder(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_dict=True, training=False)
        batch_embeddings = outputs.last_hidden_state[:, 0, :]
        query_embeddings.extend(batch_embeddings.numpy())
    logger.info("Done generating query embeddings in {}s".format(
        time.perf_counter() - start_time))
    return np.array(query_embeddings)


def evaluate(
    qa_test_data: Dict[Text, Any],
    query_encoder: tf.keras.Model,
    indexer: DenseFlatIndexer,
    tokenizer,
    result_dir: str,
    query_max_seq_length: int,
    batch_size: int = 256,
    top_docs: int = 1,
    debug=True,
    write_out_results=False
) -> np.ndarray:
    queries = [q.get('question') for q in qa_test_data]
    query_dataset = create_query_dataset(
        queries, tokenizer, query_max_seq_length, batch_size)
    query_embeddings = generate_query_embeddings(query_dataset, query_encoder)

    num_queries_per_search = 16
    eval_results = []
    for batch_idx in range(0, len(query_embeddings), num_queries_per_search):
        query_vectors = query_embeddings[batch_idx: batch_idx +
                                         num_queries_per_search]
        search_results = indexer.search_knn(query_vectors, top_docs=top_docs)
        for idx, (metas, scores) in enumerate(search_results):
            relevant_articles = []
            for meta in metas:
                record = {
                    'law_id': meta.get('law_id'),
                    'article_id': meta.get('article_id')
                }
                if write_out_results:
                    record['title'] = meta.get('title')
                    record['text'] = meta.get('text')
                relevant_articles.append(record)

            if write_out_results:
                output_record = {
                    'question_id': qa_test_data[batch_idx + idx].get('question_id'),
                    'question': qa_test_data[batch_idx + idx].get('question'),
                    'relevant_articles': relevant_articles
                }
            else:
                output_record = {
                    'question_id': qa_test_data[batch_idx + idx].get('question_id'),
                    'relevant_articles': relevant_articles
                }
            eval_results.append(output_record)

    if write_out_results:
        if not tf.io.gfile.exists(result_dir):
            tf.io.gfile.makedirs(result_dir)
        result_file = os.path.join(result_dir, 'retrieval_results.json')
        with tf.io.gfile.GFile(result_file, 'w') as writer:
            writer.write(json.dumps(
                eval_results, indent=4, ensure_ascii=False))

    if debug:
        corpus = load_corpus_to_dict('data/legal_corpus.json')
        precision, recall, f2_score = calculate_metrics(
            eval_results, qa_test_data, corpus)
        metric_file = os.path.join(result_dir, 'metrics.txt')
        with tf.io.gfile.GFile(metric_file, 'w') as writer:
            writer.write(
                "Precision\t= {}\n"
                "Recall\t\t= {}\n"
                "F2-score\t= {}".format(precision, recall, f2_score)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--index-path", default='indexes', required=True)
    parser.add_argument(
        "--qa-path", default='data/test/data.json', required=True)
    parser.add_argument("--result-dir", default='results', required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-docs", type=int, default=1)
    parser.add_argument("--debug", action='store_const',
                        const=True, default=False)
    parser.add_argument("--write-out-results",
                        action='store_const', const=True, default=False)
    args = parser.parse_args()

    # setup logger
    add_color_formater(logging.root)

    # setup environment
    setup_memory_growth()

    config = DualEncoderConfig.from_json_file(args.config_file)
    qa_test_data = load_test_data(args.qa_path)
    tokenizer_class = ARCHITECTURE_MAPPINGS[config.model_arch]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)
    query_encoder = load_query_encoder(config)

    # evaluating
    evaluate(
        qa_test_data=qa_test_data,
        query_encoder=query_encoder,
        indexer=indexer,
        tokenizer=tokenizer,
        result_dir=args.result_dir,
        query_max_seq_length=config.query_max_seq_length,
        top_docs=args.top_docs,
        debug=args.debug,
        write_out_results=args.write_out_results
    )


if __name__ == "__main__":
    main()
