import os
import argparse
import json
from typing import Text, List, Dict, Tuple, Union, Any
import tensorflow as tf

from libs.data_helpers.corpus_data import load_corpus_dataset
from libs.constants import TOKENIZER_MAPPING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--tokenizer-type", choices=['roberta', 'bert', 'distilbert'], default="roberta")
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--tokenizer-path", default="vinai/phobert-base")
    parser.add_argument("--max-context-length", default=256, type=int)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    tokenizer_class = TOKENIZER_MAPPING[args.tokenizer_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    dataset, _ = load_corpus_dataset(
        data_source=args.tfrecord_dir,
        max_context_length=args.max_context_length
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