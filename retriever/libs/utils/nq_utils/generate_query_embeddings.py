import argparse
import jsonlines
import pickle
import os
import logging
import tensorflow as tf

from libs.constants import TOKENIZER_MAPPING
from scripts.python.nn.evaluate import (
    load_query_encoder,
    create_query_dataset,
    generate_query_embeddings
)
from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def load_queries(qas_path):
    read_stream = tf.io.gfile.GFile(qas_path)
    reader = jsonlines.Reader(read_stream)
    queries = [item["question"] for item in reader]
    read_stream.close()
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--architecture", choices=["bert", "roberta"], default="bert")
    parser.add_argument("--pretrained-model-path", default="bert-base-uncased")
    parser.add_argument("--max-query-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--qas-path", required=True)
    parser.add_argument("--embedding-path", required=True)

    args = parser.parse_args()

    cache_dir = os.path.dirname(args.embedding_path)
    if not tf.io.gfile.exists(cache_dir):
        tf.io.gfile.makedirs(cache_dir)

    tokenizer = TOKENIZER_MAPPING[args.architecture].from_pretrained(args.pretrained_model_path)

    query_encoder = load_query_encoder(
        checkpoint_path=args.checkpoint_path,
        architecture=args.architecture,
        pretrained_model_path=args.pretrained_model_path
    )

    queries = load_queries(qas_path=args.qas_path)
    query_dataset = create_query_dataset(
        queries=queries, tokenizer=tokenizer,
        max_query_length=args.max_query_length, batch_size=args.batch_size
    )

    query_embeddings = generate_query_embeddings(
        query_dataset=query_dataset,
        query_encoder=query_encoder
    )
    writer = tf.io.gfile.GFile(args.embedding_path, "wb")
    pickler = pickle.Pickler(writer)
    pickler.dump(query_embeddings)
    writer.close()


if __name__ == "__main__":
    main()
