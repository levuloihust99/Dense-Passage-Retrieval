import json
import argparse
import os
import multiprocessing
import logging
from functools import partial

from rank_bm25 import BM25Okapi
from term_search.utils import remove_stopwords_wrapper, shared_counter
from data_helpers.data_utils import load_corpus_to_dict, load_corpus_to_list, load_qa_data
from utils.evaluation import calculate_metrics
from utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_worker_for_search():
    global bm25
    bm25 = BM25Okapi(tokenized_corpus)


def get_top_n(query):
    top_n_docs = bm25.get_top_n(query, corpus_raw, args.top_docs)
    with query_shared_counter.get_lock():
        query_shared_counter.value += 1
        if query_shared_counter.value % 100 == 0:
            logger.info("Done {} queries".format(query_shared_counter.value))
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

    # setup logger
    add_color_formater(logging.root)

    global corpus_raw
    corpus_raw = load_corpus_to_list(args.corpus_raw_path)
    corpus_dict = load_corpus_to_dict(args.corpus_raw_path)
    with open(args.corpus_processed_path) as reader:
        corpus = json.load(reader)
    global tokenized_corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    global stop_words
    with open('term_search/vietnamese-stopwords.txt', 'r') as reader:
        stop_words = reader.read().split('\n')
    if stop_words[-1] == '':
        stop_words.pop()

    ground_truth = load_qa_data(args.qa_path)
    queries = [qa['question'] for qa in ground_truth]

    global query_shared_counter
    query_shared_counter = multiprocessing.Value('i', 0)
    jobs = multiprocessing.Pool(processes=args.num_processes, initializer=init_worker_for_search)
    remove_stopwords = partial(remove_stopwords_wrapper, stop_words=stop_words)

    queries = jobs.map(remove_stopwords, queries)
    queries_tokenized = [q.split(' ') for q in queries]

    retrieval_results = jobs.map(get_top_n, queries_tokenized)
    with query_shared_counter.get_lock():
        logger.info("Done {} queries".format(shared_counter.value))

    retrieval_results = [{
        'question': ground_truth[idx]['question'],
        'question_id': ground_truth[idx]['question_id'],
        'relevant_articles': retrieval_results[idx]
    } for idx in range(len(ground_truth))]

    precision, recall, f2_score = calculate_metrics(retrieval_results, ground_truth, corpus_dict)
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    metric_file = os.path.join(args.result_dir, 'metrics.txt')
    with open(metric_file, 'w') as writer:
        writer.write(
            "Precision\t= {}\n"
            "Recall\t\t= {}\n"
            "F2-score\t= {}".format(precision, recall, f2_score)
        )


if __name__ == "__main__":
    main()
