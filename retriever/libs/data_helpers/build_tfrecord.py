import tensorflow as tf
import os
import json
import argparse
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Text, Any, Union

from libs.utils.logging import add_color_formater
from libs.constants import TOKENIZER_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def tokenize_context(
    tokenizer,
    context: Dict[Text, Text],
    max_context_length: int
) -> Dict[Text, List[int]]:
    title_tokens = tokenizer.tokenize(context["title"])
    text_tokens = tokenizer.tokenize(context["text"])
    tokens = title_tokens + [tokenizer.sep_token] + text_tokens
    # truncate
    if len(tokens) > max_context_length - 2:
        tokens = tokens[:max_context_length - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    attention_mask = [1] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = max_context_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
    attention_mask = attention_mask + [0] * pad_length
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def tokenize_batched_contexts(
    tokenizer,
    contexts: List[Dict[Text, Text]],
    max_context_length: int
) -> Dict[Text, List[List[int]]]:
    ret = {
        "input_ids": [],
        "attention_mask": []
    }
    for context in contexts:
        context_tokenized = tokenize_context(
            tokenizer, context, max_context_length)
        ret["input_ids"].append(context_tokenized["input_ids"])
        ret["attention_mask"].append(context_tokenized["attention_mask"])

    return ret


def create_feature(inputs):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_ids_serialized = tf.io.serialize_tensor(
        tf.convert_to_tensor(input_ids, dtype=tf.int32)
    )
    attention_mask_serialized = tf.io.serialize_tensor(
        tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    )
    input_ids_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[input_ids_serialized.numpy()])
    )
    attention_mask_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[attention_mask_serialized.numpy()])
    )
    return {
        'input_ids_feature': input_ids_feature,
        'attention_mask_feature': attention_mask_feature
    }


def create_pos_example(
    item: Dict[Text, List[Union[Text, Dict[Text, Text]]]],
    tokenizer,
    max_query_length: int,
    max_context_length: int,
    **kwargs
) -> tf.train.Example:
    # question processing
    questions_tokenized = tokenizer(
        item["questions"],
        padding='max_length',
        max_length=max_query_length,
        truncation=True
    )
    questions_features = create_feature(questions_tokenized)

    # positive contexts processing
    positive_contexts_tokenized = tokenize_batched_contexts(
        tokenizer,
        item["positive_contexts"],
        max_context_length=max_context_length
    )
    positive_contexts_feature = create_feature(positive_contexts_tokenized)

    features = {
        'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item["sample_id"]])),
        'question/input_ids': questions_features["input_ids_feature"],
        'question/attention_mask': questions_features["attention_mask_feature"],
        'positive_context/input_ids': positive_contexts_feature["input_ids_feature"],
        'positive_context/attention_mask': positive_contexts_feature["attention_mask_feature"],
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def create_poshard_example(
    item: Dict[Text, List[Union[Text, Dict[Text, Text]]]],
    tokenizer,
    max_query_length: int,
    max_context_length: int,
    **kwargs
) -> tf.train.Example:
    # question processing
    questions_tokenized = tokenizer(
        item["questions"],
        padding='max_length',
        max_length=max_query_length,
        truncation=True
    )
    questions_features = create_feature(questions_tokenized)

    # positive contexts processing
    positive_contexts_tokenized = tokenize_batched_contexts(
        tokenizer,
        item["positive_contexts"],
        max_context_length=max_context_length
    )
    positive_contexts_feature = create_feature(positive_contexts_tokenized)

    # hard negative contexts processing
    hardneg_contexts_tokenized = tokenize_batched_contexts(
        tokenizer,
        item["hardneg_contexts"],
        max_context_length=max_context_length
    )
    hardneg_contexts_feature = create_feature(hardneg_contexts_tokenized)

    # hard negative masking

    features = {
        'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item["sample_id"]])),
        'question/input_ids': questions_features["input_ids_feature"],
        'question/attention_mask': questions_features["attention_mask_feature"],
        'positive_context/input_ids': positive_contexts_feature["input_ids_feature"],
        'positive_context/attention_mask': positive_contexts_feature["attention_mask_feature"],
        'hardneg_context/input_ids': hardneg_contexts_feature["input_ids_feature"],
        'hardneg_context/attention_mask': hardneg_contexts_feature["attention_mask_feature"],
        'num_hardneg': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(item["hardneg_contexts"])]))
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def create_nonhard_example(
    item: Dict[Text, List[Union[Text, Dict[Text, Text]]]],
    tokenizer,
    max_context_length: int,
    **kwargs
) -> tf.train.Example:
    positive_contexts_tokenized = tokenize_batched_contexts(
        tokenizer,
        item["positive_contexts"],
        max_context_length=max_context_length
    )
    positive_contexts_feature = create_feature(positive_contexts_tokenized)
    features = {
        'sample_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item["sample_id"]])),
        'negative_context/input_ids': positive_contexts_feature["input_ids_feature"],
        'negative_context/attention_mask': positive_contexts_feature["attention_mask_feature"]
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_examples(
    data: List,
    output_dir: Text,
    tokenizer,
    create_example_func,
    **kwargs
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    counter = 0
    example_writer = tf.io.TFRecordWriter(
        os.path.join(output_dir, 'data_{:03d}.tfrecord'.format(counter))
    )
    for idx, item in tqdm(enumerate(data)):
        example = create_example_func(
            item, tokenizer, **kwargs
        )
        example_writer.write(example.SerializeToString())
        if (idx + 1) % kwargs["num_examples_per_file"] == 0:
            example_writer.close()
            logger.info("Written {} examples".format(idx + 1))
            counter += 1
            example_writer = tf.io.TFRecordWriter(
                os.path.join(output_dir, 'data_{:03d}.tfrecord'.format(counter))
            )

    example_writer.close()
    logger.info("Written {} examples".format(idx + 1))    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", default="configs/pipeline_training_config.json")
    parser.add_argument("--data-path", default="data/v2/train_data.json")
    parser.add_argument("--num-examples-per-file", default=1000, type=int)
    parser.add_argument("--output-dir", default="data/v2/tfrecord/train/pos")
    parser.add_argument("--pipeline-type", choices=["pos", "poshard", "hard"], default="pos")
    args = parser.parse_args()

    with open(args.config_file, "r") as reader:
        pipeline_config = json.load(reader)
    with open(args.data_path, "r") as reader:
        data = json.load(reader)

    tokenizer = TOKENIZER_MAPPING[pipeline_config["tokenizer_type"]].from_pretrained(
        pipeline_config["tokenizer_path"]
    )

    if args.pipeline_type == "pos":
        write_examples(
            data=data,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            create_example_func=create_pos_example,
            num_examples_per_file=args.num_examples_per_file,
            max_query_length=pipeline_config["max_query_length"],
            max_context_length=pipeline_config["max_context_length"]
        )
    elif args.pipeline_type == "poshard":
        only_hard_data = [item for item in data if len(item["hardneg_contexts"]) > 0]
        if only_hard_data:
            write_examples(
                data=only_hard_data,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                create_example_func=create_poshard_example,
                num_examples_per_file=args.num_examples_per_file,
                max_query_length=pipeline_config["max_query_length"],
                max_context_length=pipeline_config["max_context_length"],
            )
    else:
        only_hard_data = []
        non_hard_data = []
        for item in data:
            if len(item["hardneg_contexts"]) > 0:
                only_hard_data.append(item)
            else:
                non_hard_data.append(item)

        if only_hard_data:
            write_examples(
                data=only_hard_data,
                output_dir=os.path.join(args.output_dir, "onlyhard"),
                tokenizer=tokenizer,
                create_example_func=create_poshard_example,
                num_examples_per_file=args.num_examples_per_file,
                max_query_length=pipeline_config["max_query_length"],
                max_context_length=pipeline_config["max_context_length"],
            )
        if non_hard_data:
            write_examples(
                data=non_hard_data,
                output_dir=os.path.join(args.output_dir, "nonhard"),
                tokenizer=tokenizer,
                create_example_func=create_nonhard_example,
                num_examples_per_file=args.num_examples_per_file,
                max_query_length=pipeline_config["max_query_length"],
                max_context_length=pipeline_config["max_context_length"]
            )


if __name__ == "__main__":
    main()
