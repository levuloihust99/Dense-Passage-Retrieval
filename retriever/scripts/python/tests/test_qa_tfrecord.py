import os
import argparse
import json
from typing import Text, List, Dict, Tuple, Union, Any
import tensorflow as tf

from libs.data_helpers.tfio.dual_encoder.loader import deserialize_qa_pairs, deserialize_qa_pairs_with_hardneg
from libs.nn.configuration.dual_encoder import DualEncoderConfig
from libs.utils.setup import setup_memory_growth
from libs.nn.constants import ARCHITECTURE_MAPPINGS
setup_memory_growth()

from transformers import BertTokenizer, PhobertTokenizer


def load_dataset(
    tfrecord_dir: Text,
    load_hardneg: bool
):
    list_files = tf.io.gfile.listdir(tfrecord_dir)
    list_files.sort()
    list_files = [os.path.join(tfrecord_dir, tfrecord_file) for tfrecord_file in list_files]
    dataset = tf.data.Dataset.from_tensor_slices(list_files)
    dataset = dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x)
    )

    name_to_features = {
        'query_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'query_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    if load_hardneg:
        name_to_features=dict(
            **name_to_features,
            **{
                'hardneg_context_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                'hardneg_context_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
            }
        )

    deserialize_func = deserialize_qa_pairs_with_hardneg if load_hardneg else deserialize_qa_pairs
    dataset = dataset.map(
        lambda record: deserialize_func(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--architecture", choices=['roberta', 'bert', 'distilbert'], required=True)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--load-hardneg", default=False, type=eval)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    dataset = load_dataset(
        tfrecord_dir=args.tfrecord_dir,
        load_hardneg=args.load_hardneg
    )
    iterator = iter(dataset)
    qa_pairs = []
    for i in range(args.num_examples):
        record = next(iterator)
        query_input_ids = record['query_input_ids']
        query_input_ids = tf.RaggedTensor.from_tensor(query_input_ids, padding=tokenizer.pad_token_id).to_list()
        context_input_ids = record['context_input_ids']
        context_input_ids = tf.RaggedTensor.from_tensor(context_input_ids, padding=tokenizer.pad_token_id).to_list()
        if args.load_hardneg:
            hardneg_context_input_ids = record['hardneg_context_input_ids']
            hardneg_context_input_ids = tf.RaggedTensor.from_tensor(hardneg_context_input_ids, padding=tokenizer.pad_token_id).to_list()
        else:
            hardneg_context_input_ids = None

        questions = []
        contexts = []
        hardneg_contexts = []
        for query_ids in query_input_ids:
            questions.append(tokenizer.decode(query_ids))
        for context_ids in context_input_ids:
            contexts.append(tokenizer.decode(context_ids))
        if args.load_hardneg:
            for hardneg_context_ids in hardneg_context_input_ids:
                hardneg_contexts.append(tokenizer.decode(hardneg_context_ids))
        
        out_record = {
            'question': questions,
            'context': contexts
        }
        if args.load_hardneg:
            out_record['hardneg_context'] = hardneg_contexts

        qa_pairs.append(out_record)
    
    with open(args.output_file, 'w') as writer:
        json.dump(qa_pairs, writer, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    main()