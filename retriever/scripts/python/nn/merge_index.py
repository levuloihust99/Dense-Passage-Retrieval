import os
import pickle
import argparse
import json
import logging
from ruamel.yaml import YAML
from libs.faiss_indexer import DenseFlatIndexer
import numpy as np
from libs.utils.logging import add_color_formater

yaml = YAML()

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as reader:
        config = yaml.load(reader)

    out_path = config['output_index_path']
    out_indexer = DenseFlatIndexer()
    out_indexer.init_index(768)

    data_tobe_indexed = []
    for index_cfg in config['input_indexes']:
        index_path = index_cfg['index_path']
        indexer = DenseFlatIndexer()
        indexer.deserialize(index_path)
        ntotal = indexer.index.ntotal
        vectors = indexer.index.reconstruct_n(0, ntotal)
        for meta, v in zip(indexer.meta, vectors):
            data_tobe_indexed.append((meta, v))

    out_indexer.index_data(data_tobe_indexed)
    out_indexer.serialize(out_path)
        

if __name__ == "__main__":
    main()
