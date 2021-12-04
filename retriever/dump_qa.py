import random
import json
import argparse
import logging

from data_helpers.datasets import (
    vlsp_dataset, zalo_dataset, mailong_dataset
)
from data_helpers.tfio.dumper import dump_qa
from dual_encoder.constants import ARCHITECTURE_MAPPINGS
from utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def combine(datasets):
    combined_datasets = []
    for dataset in datasets:
        combined_datasets += dataset
    random.shuffle(combined_datasets)
    return combined_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--load-vlsp", type=eval, default=True)
    parser.add_argument("--load-zalo", type=eval, default=True)
    parser.add_argument("--load-mailong25", type=eval, default=True)
    parser.add_argument("--query-max-seq-length", type=int, required=True)
    parser.add_argument("--context-max-seq-length", type=int, required=True)
    parser.add_argument("--architecture", choices=['roberta', 'distilbert', 'bert'], required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--path-mapping", default='configs/path_mapping.json')
    args = parser.parse_args()

    random.seed(args.random_seed)

    vlsp_qa_data = []
    zalo_qa_data = []
    mailong_qa_data = []

    with open(args.path_mapping, 'r') as reader:
        path_mapping = json.load(reader)

    if args.load_vlsp:
        vlsp_qa_data = vlsp_dataset.load_data(path_mapping['vlsp'])
    if args.load_zalo:
        zalo_qa_data = zalo_dataset.load_data(path_mapping['zalo'])
    if args.load_mailong25:
        mailong_qa_data = mailong_dataset.load_data(path_mapping['mailong25'])

    combined_dataset = combine([vlsp_qa_data, zalo_qa_data, mailong_qa_data])

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)

    dump_qa(
        query_context_pairs=combined_dataset,
        tokenizer=tokenizer,
        query_max_seq_length=args.query_max_seq_length,
        context_max_seq_length=args.context_max_seq_length,
        tfrecord_dir=args.tfrecord_dir
    )


if __name__ == "__main__":
    main()
