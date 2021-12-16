import argparse
import logging
import json
from typing import Text

from libs.data_helpers.tfio.dual_encoder.dumper import dump_corpus
from libs.nn.constants import ARCHITECTURE_MAPPINGS
from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Text):
    with open(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--context-max-seq-length", type=int, required=True)
    parser.add_argument("--architecture", choices=['roberta', 'distilbert', 'bert'], required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    args = parser.parse_args()

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    corpus = load_corpus(corpus_path=args.corpus_path)

    dump_corpus(
        corpus=corpus,
        tokenizer=tokenizer,
        context_max_seq_length=args.context_max_seq_length,
        tfrecord_dir=args.tfrecord_dir,
        add_law_id=False
    )


if __name__ == "__main__":
    main()
