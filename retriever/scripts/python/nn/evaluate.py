"""This code can not run on TPU"""

import logging
import argparse
import os
import time
import json
from typing import Text, Dict, List, Any

import numpy as np
import tensorflow as tf

from libs.nn.configuration import DualEncoderConfig
from libs.constants import TOKENIZER_MAPPING, MODEL_MAPPING
from libs.utils.logging import add_color_formater
from libs.utils.setup import setup_memory_growth
from libs.utils.evaluation import calculate_metrics
from libs.faiss_indexer import DenseFlatIndexer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def load_test_data(path: Text):
    with tf.io.gfile.GFile(path, 'r') as reader:
        data = json.load(reader)
    return data


def load_query_encoder(checkpoint_path: Text, architecture: Text, pretrained_model_path: Text):
    logger.info("Loading query encoder...")
    start_time = time.perf_counter()

    encoder_class = MODEL_MAPPING[architecture]
    query_encoder = encoder_class.from_pretrained(pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(query_encoder=query_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(checkpoint_path).expect_partial()

    logger.info("Done loading query encoder in {}s".format(
        time.perf_counter() - start_time))
    return query_encoder


def create_query_dataset(
    queries: List[Text],
    tokenizer,
    max_query_length: int,
    batch_size: int
):
    logger.info("Creating query dataset...")
    start_time = time.perf_counter()
    query_tensors = tokenizer(
        queries, padding='max_length',
        max_length=max_query_length,
        truncation=True,
        return_tensors="tf"
    )
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
    max_query_length: int,
    batch_size: int = 256,
    top_docs: int = 10,
) -> np.ndarray:
    queries = [q.get('question') for q in qa_test_data]
    query_dataset = create_query_dataset(
        queries, tokenizer, max_query_length, batch_size)
    query_embeddings = generate_query_embeddings(query_dataset, query_encoder)

    num_queries_per_search = 16
    eval_results = []
    for batch_idx in range(0, len(query_embeddings), num_queries_per_search):
        query_vectors = query_embeddings[batch_idx: batch_idx +
                                         num_queries_per_search]
        search_results = indexer.search_knn(query_vectors, top_docs=top_docs)
        for idx, (metas, scores) in enumerate(search_results):
            eval_results.append({
                "question": queries[batch_idx + idx],
                "relevant_articles": metas
            })

    metrics = calculate_metrics(qa_test_data, eval_results)
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    top_hits = metrics['top_hits']
    if not tf.io.gfile.exists(result_dir):
        tf.io.gfile.makedirs(result_dir)
    metric_file = os.path.join(result_dir, 'metrics.txt')
    with tf.io.gfile.GFile(metric_file, 'w') as writer:
        writer.write("Precision\t= {}\n".format(precision))
        writer.write("Recall\t= {}\n".format(recall))
        writer.write("F1 score\t= {}\n".format(f1_score))
        writer.write("\n********************** Top hits **********************\n")
        for idx, hit_score in enumerate(top_hits):
            writer.write("Top {:03d}: {}\n".format(idx + 1, hit_score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--pretrained-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-docs", type=int, default=10)
    parser.add_argument("--max-query-length", type=int, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args()

    # setup logger
    add_color_formater(logging.root)

    # setup environment
    setup_memory_growth()

    qa_test_data = load_test_data(args.qa_path)
    tokenizer_class = TOKENIZER_MAPPING[args.architecture]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)
    query_encoder = load_query_encoder(
        args.checkpoint_path, args.architecture, args.pretrained_model_path
    )

    # evaluating
    evaluate(
        qa_test_data=qa_test_data,
        query_encoder=query_encoder,
        indexer=indexer,
        tokenizer=tokenizer,
        result_dir=args.result_dir,
        max_query_length=args.max_query_length,
        top_docs=args.top_docs
    )


if __name__ == "__main__":
    main()
