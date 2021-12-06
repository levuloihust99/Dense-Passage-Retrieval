import json
import argparse
import os
import re
import multiprocessing
import logging

from rank_bm25 import BM25, BM25Okapi
from tensorflow.python.keras.utils.generic_utils import to_snake_case
from data_helpers.data_utils import load_corpus_to_dict, load_corpus_to_list
from evaluate import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_data(path):
    with open(path, 'r') as reader:
        data = json.load(reader)['items']
    return data


def remove_stopwords(text):
    for word in stop_words:
        text = re.sub(rf'\b{word}\b', '', text)
    return text


def init_worker_for_search():
    global bm25
    bm25 = BM25Okapi(tokenized_corpus)


def get_top_n(query):
    top_n_docs = bm25.get_top_n(query, corpus_raw, args.top_docs)
    with shared_counter.get_lock():
        shared_counter.value += 1
        if shared_counter.value % 100 == 0:
            logger.info("Done {} queries".format(shared_counter.value))
    return top_n_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-raw-path", required=True)
    parser.add_argument("--corpus-processed-path", required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--top-docs", default=20, type=int)
    parser.add_argument("--result-dir", default='results/bm25')
    parser.add_argument("--num-processes", default=1, type=int)
    global args
    args = parser.parse_args()

    global corpus_raw
    corpus_raw = load_corpus_to_list(args.corpus_raw_path)
    corpus_dict = load_corpus_to_dict(args.corpus_raw_path)
    with open(args.corpus_processed_path) as reader:
        corpus = json.load(reader)
    global tokenized_corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    global stop_words
    with open('bm25/vietnamese-stopwords.txt', 'r') as reader:
        stop_words = reader.read().split('\n')
    if stop_words[-1] == '':
        stop_words.pop()

    ground_truth = load_qa_data(args.qa_path)
    queries = [qa['question'] for qa in ground_truth]

    global shared_counter
    shared_counter = multiprocessing.Value('i', 0)

    jobs = multiprocessing.Pool(processes=args.num_processes, initializer=init_worker_for_search)
    queries = jobs.map(remove_stopwords, queries)
    queries_tokenized = [q.split(' ') for q in queries]

    retrieval_results = jobs.map(get_top_n, queries_tokenized)
    with shared_counter.get_lock():
        logger.info("Done {} queries".format(shared_counter.value))

    retrieval_results = [{
        'question': ground_truth[idx]['question'],
        'question_id': ground_truth[idx]['question_id'],
        'relevant_articles': retrieval_results[idx]
    } for idx in range(len(ground_truth))]

    precision, recall, f2_score = calculate_metrics(retrieval_results, ground_truth, corpus_dict)
    metric_file = os.path.join(args.result_dir, 'metrics.txt')
    with open(metric_file, 'w') as writer:
        writer.write(
            "Precision\t= {}\n"
            "Recall\t\t= {}\n"
            "F2-score\t= {}".format(precision, recall, f2_score)
        )


if __name__ == "__main__":
    main()
