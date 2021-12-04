import argparse
import logging
from data_helpers.data_utils import load_corpus_to_list

from data_helpers.tfio.dumper import dump_corpus
from dual_encoder.constants import ARCHITECTURE_MAPPINGS
from utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--context-max-seq-length", type=int, required=True)
    parser.add_argument("--architecture", choices=['roberta', 'distilbert', 'bert'], required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--add-law-id", type=eval, required=True)
    args = parser.parse_args()

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    corpus = load_corpus_to_list(corpus_path=args.corpus_path)

    dump_corpus(
        corpus=corpus,
        tokenizer=tokenizer,
        context_max_seq_length=args.context_max_seq_length,
        tfrecord_dir=args.tfrecord_dir,
        add_law_id=args.add_law_id
    )


if __name__ == "__main__":
    main()
