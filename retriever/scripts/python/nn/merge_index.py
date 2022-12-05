import os
import pickle
import json
import logging
from ruamel.yaml import YAML
from libs.faiss_indexer import DenseFlatIndexer
import numpy as np
from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)


def main():
    yaml = YAML()
    with open("kms/configs/merge_index.yml") as reader:
        config = yaml.load(reader)

    out_path = config['output_index_path']
    config = config['input_indexes']
    base_indexer = DenseFlatIndexer()
    base_indexer.deserialize(config[0]['index_path'])

    for cfg in config[1:]:
        if cfg['embedding_path']:
            files = os.listdir(cfg['embedding_path'])
            files = sorted(files)
            all_embs = []
            for f in files:
                with open(os.path.join(cfg['embedding_path'], f), "rb") as reader:
                    embs = pickle.load(reader)
                    all_embs.extend(embs)
            all_embs = [emb[1] for emb in all_embs]
            all_embs = np.stack(all_embs, axis=0)
        else:
            tmp_indexer = DenseFlatIndexer()
            tmp_indexer.deserialize(cfg['index_path'])
            all_embs = tmp_indexer.index.reconstruct_n(0, tmp_indexer.index.ntotal)
        with open(cfg['corpus_meta'], "r") as reader:
            meta = json.load(reader)
        data_tobe_indexed = [(meta[i], all_embs[i]) for i in range(len(meta))]
        base_indexer.index_data(data_tobe_indexed)
    
    base_indexer.serialize(out_path)
        

if __name__ == "__main__":
    main()
