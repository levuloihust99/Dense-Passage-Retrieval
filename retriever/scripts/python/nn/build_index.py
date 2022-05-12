from operator import index
import os
import pickle
import argparse
import logging
import tensorflow as tf
from libs.faiss_indexer import DenseFlatIndexer

from libs.utils.logging import add_color_formater


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
add_color_formater(logging.root)


def indexing(embedding_dir, index_path, hidden_size):
    # create index directory
    if not tf.io.gfile.exists(index_path):
        tf.io.gfile.makedirs(index_path)

    # create index
    indexer = DenseFlatIndexer()
    indexer.init_index(vector_sz=hidden_size)

    # load embedding vectors and index
    embedding_files = tf.io.gfile.listdir(embedding_dir)
    embedding_files = sorted(embedding_files)
    embedding_files = [os.path.join(embedding_dir, f) for f in embedding_files]
    for idx, f in enumerate(embedding_files):
        with open(f, "rb") as reader:
            embeddings = pickle.load(reader)
        data_to_be_indexed = [(e[0], e[1].numpy()) for e in embeddings]
        indexer.index_data(data_to_be_indexed)
        logger.info("Indexed {} embeddings.".format(len(embeddings)))
    indexer.serialize(index_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--hidden-size", type=int, default=768)

    args = parser.parse_args()
    indexing(embedding_dir=args.embedding_dir,
             index_path=args.index_path, hidden_size=args.hidden_size)


if __name__ == "__main__":
    main()
