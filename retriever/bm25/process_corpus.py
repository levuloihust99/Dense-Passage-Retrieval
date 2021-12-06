import json
import argparse
import re
import logging
import multiprocessing

from tensorflow.python.ops.gen_math_ops import mul
from rank_bm25 import BM25Okapi
from data_helpers.data_utils import load_corpus_to_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_qa_data(path):
    with open(path, 'r') as reader:
        data = json.load(reader)['items']
    return data


def remove_stopwords(text):
    for word in stop_words:
        text = re.sub(rf'\b{word}\b', '', text)
    with shared_counter.get_lock():
        shared_counter.value += 1
        if shared_counter.value % 100 == 0:
            logger.info("Done processing {} docs".format(shared_counter.value))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    corpus = load_corpus_to_list(args.corpus_path)
    corpus_texts = [
        "{} . {}".format(c['title'], c['text'])
        for c in corpus
    ]

    global stop_words
    with open('bm25/vietnamese-stopwords.txt', 'r') as reader:
        stop_words = reader.read().split('\n')
    if stop_words[-1] == '':
        stop_words.pop()

    global shared_counter
    shared_counter = multiprocessing.Value('i', 0)

    jobs = multiprocessing.Pool(processes=args.num_processes)
    corpus_texts_no_stopwords = jobs.map(remove_stopwords, corpus_texts)

    with shared_counter.get_lock():
        logger.info("Done processing {} docs".format(shared_counter.value))

    with open("bm25/corpus_processed.json", 'w') as writer:
        json.dump(corpus_texts_no_stopwords, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
