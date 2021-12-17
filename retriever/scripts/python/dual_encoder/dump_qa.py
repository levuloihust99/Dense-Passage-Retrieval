import random
import json
import argparse
import logging

from libs.data_helpers.datasets import (
    atd_dataset, vlsp_dataset, zalo_dataset, mailong_dataset, evnspc_dataset
)
from libs.data_helpers.tfio.dual_encoder.dumper import dump_qa, dump_qa_with_hardneg
from libs.nn.constants import ARCHITECTURE_MAPPINGS
from libs.utils.logging import add_color_formater

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
    parser.add_argument("--load-evn-spc", type=eval, default=True)
    parser.add_argument("--load-atd", type=eval, default=True)
    parser.add_argument("--query-max-seq-length", type=int, required=True)
    parser.add_argument("--context-max-seq-length", type=int, required=True)
    parser.add_argument("--architecture", choices=['roberta', 'distilbert', 'bert'], required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--data-config", default='configs/data_config.json')
    args = parser.parse_args()

    random.seed(args.random_seed)

    vlsp_qa_data = []
    zalo_qa_data = []
    mailong_qa_data = []
    evn_spc_qa_data = []
    atd_qa_data = []

    with open(args.data_config, 'r') as reader:
        data_config = json.load(reader)

    if args.load_vlsp:
        vlsp_qa_data = vlsp_dataset.load_data(**data_config['vlsp'])
    if args.load_zalo:
        zalo_qa_data = zalo_dataset.load_data(**data_config['zalo'])
    if args.load_mailong25:
        mailong_qa_data = mailong_dataset.load_data(**data_config['mailong25'])
    if args.load_evn_spc:
        evn_spc_qa_data = evnspc_dataset.load_data(**data_config['evn_spc'])
    if args.load_atd:
        atd_qa_data = atd_dataset.load_data(**data_config['atd'])

    combined_dataset = combine([vlsp_qa_data, zalo_qa_data, mailong_qa_data, evn_spc_qa_data, atd_qa_data])

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)

    dump_func = dump_qa
    if data_config['zalo'].get('load_hardneg', False):
        dump_func = dump_qa_with_hardneg
    dump_func(
        query_context_pairs=combined_dataset,
        tokenizer=tokenizer,
        query_max_seq_length=args.query_max_seq_length,
        context_max_seq_length=args.context_max_seq_length,
        tfrecord_dir=args.tfrecord_dir
    )


if __name__ == "__main__":
    main()
