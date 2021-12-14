import os
import argparse
import json
from typing import Text, List, Dict, Tuple, Union, Any
import tensorflow as tf

from libs.data_helpers.tfio.dual_encoder.loader import load_corpus_dataset
from libs.nn.constants import ARCHITECTURE_MAPPINGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--architecture", choices=['roberta', 'bert', 'distilbert'], required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--context-max-seq-length", required=True, type=int)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    dataset, num_examples = load_corpus_dataset(
        tfrecord_dir=args.tfrecord_dir,
        context_max_seq_length=args.context_max_seq_length
    )

    iterator = iter(dataset)
    corpus = []
    for i in range(args.num_examples):
        record = next(iterator)
        input_ids = record['input_ids']
        attention_mask = record['attention_mask']
        non_pad_input_ids = tf.boolean_mask(input_ids, attention_mask).numpy().tolist()
        corpus.append(tokenizer.decode(non_pad_input_ids))
    
    with open(args.output_file, 'w') as writer:
        json.dump(corpus, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()