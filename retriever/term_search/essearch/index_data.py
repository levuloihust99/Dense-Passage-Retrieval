import argparse
import json
from typing import Text, List

from elasticsearch import Elasticsearch
import logging

from utils.logging import add_color_formater
from data_helpers.data_utils import load_corpus_to_list

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def create_index(index_name):
    es.indices.create(index=index_name, ignore=400)
    logger.info("Index `{}` created".format(index_name))


def index_data(index_name, corpus):
    num_indexed = 0
    for i, doc in enumerate(corpus):
        es.index(index=index_name, body=doc, id=i + 1)
        num_indexed += 1
    logger.info("Done indexing {} records".format(num_indexed))


def delete_index(index_name):
    es.indices.delete(index=index_name, ignore=[400, 404])
    logger.info("Index `{}` deleted".format(index_name))


def build_corpus_to_be_indexed(
    corpus_raw_path: Text,
    corpus_processed_path: Text
):
    corpus_raw = load_corpus_to_list(corpus_raw_path)
    with open(corpus_processed_path, 'r') as reader:
        corpus_processed = json.load(reader)
    assert len(corpus_raw) == len(corpus_processed)
    L = len(corpus_raw)
    corpus_to_be_indexed = []
    for i in range(L):
        doc_processed = corpus_processed[i]
        doc_raw = corpus_raw[i]
        corpus_to_be_indexed.append({
            'law_id': doc_raw['law_id'],
            'article_id': doc_raw['article_id'],
            'search_text': doc_processed,
            'title': doc_raw['title'],
            'text': doc_raw['text']
        })
    return corpus_to_be_indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--corpus-processed-path", required=True)
    parser.add_argument("--corpus-raw-path", required=True)
    parser.add_argument("--mode", choices=['index', 'delete'], default='index')
    parser.add_argument("--es-host", default='localhost')
    parser.add_argument("--es-port", default='9200')
    args = parser.parse_args()

    global es
    es = Elasticsearch(HOST='localhost', PORT='9200')

    if args.mode == 'index':
        create_index(args.index_name)
        corpus = build_corpus_to_be_indexed(args.corpus_raw_path, args.corpus_processed_path)
        index_data(args.index_name, corpus)
    else:
        delete_index(index_name=args.index_name)


if __name__ == "__main__":
    main()
