import json
import argparse
import os
import multiprocessing
import logging
from typing import Text, List
from elasticsearch import Elasticsearch

from term_search.utils import remove_stopwords, add_logging_info
from data_helpers.data_utils import load_corpus_to_dict, load_qa_data
from utils.evaluation import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def es_search(queries: List[Text], index_name: Text, top_docs: int):
    retrieval_results = []
    for query in queries:
        search_query = {
            'query': {
                'match': {
                    'search_text': {
                        'query': query
                    }
                }
            }
        }
        search_results = es.search(index_name=index_name, body=search_query)
        hits = search_results['hits']['hits']
        hits = hits[:top_docs]
        hits = [hit['_source'] for hit in hits]
        retrieval_results.append(hits)
    return retrieval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-raw-path", required=True)
    parser.add_argument("--corpus-processed-path", required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--top-docs", default=20, type=int)
    parser.add_argument("--result-dir", default='results/bm25')
    parser.add_argument("--num-processes", default=1, type=int)
    parser.add_argument("--index-name", default="legal")
    parser.add_argument("--es-host", default='localhost')
    parser.add_argument("--es-port", default='9200')
    global args
    args = parser.parse_args()

    global es
    es = Elasticsearch(HOST=args.es_host, PORT=args.es_port)
    corpus_dict = load_corpus_to_dict(args.corpus_raw_path)

    with open('term_search/vietnamese-stopwords.txt', 'r') as reader:
        stop_words = reader.read().split('\n')
    if stop_words[-1] == '':
        stop_words.pop()

    ground_truth = load_qa_data(args.qa_path)
    queries = [qa['question'] for qa in ground_truth]

    global shared_counter
    shared_counter = multiprocessing.Value('i', 0)

    jobs = multiprocessing.Pool(processes=args.num_processes)
    remove_stopwords_wrapper = add_logging_info(shared_counter=shared_counter, logger=logger)(remove_stopwords)
    queries = jobs.map(remove_stopwords_wrapper, queries)

    with shared_counter.get_lock():
        logger.info("Done {} queries".format(shared_counter.value))

    retrieval_results = es_search(queries)
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
