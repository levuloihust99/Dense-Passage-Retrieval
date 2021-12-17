import argparse
import json
import os
from typing import Text, List

import logging
import requests

from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.DEBUG)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def create_index(index_name):
    resp = requests.put(URL + "/{}".format(index_name))
    acknowledged = resp.json().get('acknowledged')
    if acknowledged:
        logger.info("Index `{}` created".format(index_name))
    else:
        logger.info("Failed to create index `{}`, response: {}".format(index_name, json.dumps(resp.json())))


def index_data(index_name, corpus):
    num_indexed = 0
    path = os.path.join(URL, index_name, "_doc")
    headers = {'Content-Type': 'application/json'}
    for i, doc in enumerate(corpus):
        payload = json.dumps(doc)
        requests.post(path, headers=headers, data=payload)
        num_indexed += 1
    logger.info("Done indexing {} records".format(num_indexed))


def delete_index(index_name):
    resp = requests.delete(os.path.join(URL, index_name))
    acknowledged = resp.json().get('acknowledged')
    if acknowledged:
        logger.info("Index `{}` deleted".format(index_name))
    else:
        logger.info("Failed to delete index `{}`, response = {}".format(index_name, json.dumps(resp.json())))


def load_corpus(corpus_path):
    with open(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--mode", choices=['index', 'delete'], default='index')
    parser.add_argument("--es-host", default='localhost')
    parser.add_argument("--es-port", default='9200')
    args = parser.parse_args()

    global es, URL
    URL = 'http://{}:{}'.format(args.es_host, args.es_port)

    if args.mode == 'index':
        create_index(args.index_name)
        corpus = load_corpus(args.corpus_path)
        index_data(args.index_name, corpus)
    else:
        delete_index(index_name=args.index_name)


if __name__ == "__main__":
    main()
