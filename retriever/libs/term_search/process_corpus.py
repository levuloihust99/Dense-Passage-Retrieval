import json
import argparse
import logging
import multiprocessing
from functools import partial

from libs.data_helpers.data_utils import load_corpus_to_list
from libs.term_search.utils import remove_stopwords_wrapper, shared_counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    corpus = load_corpus_to_list(args.corpus_path)
    corpus_texts = [
        "{} . {}".format(c['title'], c['text'])
        for c in corpus
    ]

    global stop_words
    with open('libs/term_search/vietnamese-stopwords.txt', 'r') as reader:
        stop_words = reader.read().split('\n')
    if stop_words[-1] == '':
        stop_words.pop()

    jobs = multiprocessing.Pool(processes=args.num_processes)
    remove_stopwords = partial(remove_stopwords_wrapper, stop_words=stop_words)
    corpus_texts_no_stopwords = jobs.map(remove_stopwords, corpus_texts)

    with shared_counter.get_lock():
        logger.info("Done processing {} docs".format(shared_counter.value))

    with open(args.output_path, 'w') as writer:
        json.dump(corpus_texts_no_stopwords, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
