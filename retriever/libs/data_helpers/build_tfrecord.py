import multiprocessing
import queue
import tensorflow as tf
import os
import json
import jsonlines
import argparse
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Text, Any, Union
from functools import partial
from multiprocessing import Process, Pool as ProcessPool, Queue

from libs.utils.logging import add_color_formater
from libs.constants import TOKENIZER_MAPPING
from libs.data_helpers.constants import (
    TOKENIZER_TYPE,
    TOKENIZER_PATH,
    MAX_QUERY_LENGTH,
    MAX_CONTEXT_LENGTH
)
from libs.data_helpers.constants import DataSourceType

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


def create_proper_example(
    item: Dict[Text, List[Union[Text, Dict[Text, Text]]]],
    tokenizer,
    max_query_length: int,
    max_context_length: int,
    **kwargs
):
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
    if len(item["hardneg_contexts"]) > 0:
        hardneg_contexts_tokenized = tokenize_batched_contexts(
            tokenizer,
            item["hardneg_contexts"],
            max_context_length=max_context_length
        )
        hardneg_contexts_feature = create_feature(hardneg_contexts_tokenized)
    else:
        pseudo_tensor = tf.zeros([0, max_context_length], dtype=tf.int32)
        pseudo_tensor_serialized = tf.io.serialize_tensor(pseudo_tensor)
        pseudo_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[pseudo_tensor_serialized.numpy()]))
        hardneg_contexts_feature = {
            "input_ids_feature": pseudo_feature,
            "attention_mask_feature": pseudo_feature
        }

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


def create_hard_none_example(
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


def worker_job(from_master_queue, to_master_queue):
    while True:
        item = from_master_queue.get()
        example = create_example_func(item)
        to_master_queue.put(example)


def feed_process(feed_queues):
    num_jobs = args.num_processes
    idx = 0
    for item in data:
        if (args.data_type == DataSourceType.HARD_ONLY and len(item["hardneg_contexts"]) == 0) \
            or (args.data_type == DataSourceType.HARD_NONE and len(item["hardneg_contexts"]) > 0):
            continue
        queue_idx = idx % num_jobs
        idx += 1
        feed_queues[queue_idx].put(item)


def fetch_process(fetch_queues):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    idx = 0
    counter = 0
    example_writer = tf.io.TFRecordWriter(
        os.path.join(args.output_dir, 'data_{:03d}.tfrecord'.format(counter))
    )
    is_done = False
    while not is_done:
        for q in fetch_queues:
            try:
                example = q.get(timeout=50)
            except queue.Empty:
                is_done = True
                break
            example_writer.write(example.SerializeToString())
            if (idx + 1) % args.num_examples_per_file == 0:
                example_writer.close()
                logger.info("Written {} examples".format(idx + 1))
                counter += 1
                example_writer = tf.io.TFRecordWriter(
                    os.path.join(args.output_dir, 'data_{:03d}.tfrecord'.format(counter))
                )
            idx += 1

    example_writer.close()
    logger.info("Written {} examples".format(idx))


def parallel_processing():
    num_jobs = args.num_processes
    if num_jobs < 2:
        sequential_processing()
        return

    global create_example_func
    if args.data_type == DataSourceType.ALL:
        create_example_func = partial(
            create_proper_example,
            tokenizer=tokenizer,
            max_query_length=pipeline_config[MAX_QUERY_LENGTH], 
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH])
    elif args.data_type == DataSourceType.ALL_POS_ONLY:
        create_example_func = partial(
            create_pos_example,
            tokenizer=tokenizer,
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH])
    elif args.data_type == DataSourceType.HARD_ONLY:
        create_example_func = partial(
            create_poshard_example,
            tokenizer=tokenizer,
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH])
    elif args.data_type == DataSourceType.HARD_NONE:
        create_example_func = partial(
            create_hard_none_example,
            tokenizer=tokenizer,
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH])
    else:
        raise Exception("Data type `{}` is not supported.".format(args.data_type))

    feed_queues = [Queue() for _ in range(num_jobs)]
    fetch_queues = [Queue() for _ in range(num_jobs)]
    jobs = [Process(target=worker_job, args=(feed_queues[idx], fetch_queues[idx])) for idx in range(num_jobs)]

    # start feeding job
    feed_job = Process(target=feed_process, args=(feed_queues,))
    feed_job.start()

    # start processing job
    for job in jobs:
        job.start()

    # fetching job
    fetch_process(fetch_queues)

    # cleanup
    for job in jobs:
        job.kill()


def sequential_processing():
    if args.data_type == DataSourceType.ALL:
        write_examples(
            data=data,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            create_example_func=create_proper_example,
            num_examples_per_file=args.num_examples_per_file,
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH]
        )
    if args.data_type == DataSourceType.ALL_POS_ONLY:
        write_examples(
            data=data,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            create_example_func=create_pos_example,
            num_examples_per_file=args.num_examples_per_file,
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH]
        )
    elif args.data_type == DataSourceType.HARD_ONLY:
        hard_only_data = [item for item in data if len(item["hardneg_contexts"]) > 0]
        if hard_only_data:
            write_examples(
                data=hard_only_data,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                create_example_func=create_poshard_example,
                num_examples_per_file=args.num_examples_per_file,
                max_query_length=pipeline_config[MAX_QUERY_LENGTH],
                max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
            )
    elif args.data_type == DataSourceType.HARD_NONE:
        hard_none_data = [item for item in data if len(item["hardneg_contexts"]) == 0]
        if hard_none_data:
            write_examples(
                data=hard_none_data,
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                create_example_func=create_hard_none_example,
                num_examples_per_file=args.num_examples_per_file,
                max_query_length=pipeline_config[MAX_QUERY_LENGTH],
                max_context_length=pipeline_config[MAX_CONTEXT_LENGTH]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", default="configs/pipeline_training_config.json")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-format", choices=["json", "jsonlines"], default="json")
    parser.add_argument("--num-examples-per-file", default=1000, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-type", choices=[DataSourceType.ALL_POS_ONLY, DataSourceType.ALL,
                        DataSourceType.HARD_ONLY, DataSourceType.HARD_NONE], required=True)
    parser.add_argument("--parallelize", type=eval, default=False)
    parser.add_argument("--num-processes", type=int, default=multiprocessing.cpu_count())

    global args
    args = parser.parse_args()

    global pipeline_config, data
    with open(args.config_file, "r") as reader:
        pipeline_config = json.load(reader)
    if args.data_format == "json":
        with open(args.data_path, "r") as reader:
            data = json.load(reader)
    else:
        data = jsonlines.open(args.data_path, "r")

    global tokenizer
    tokenizer = TOKENIZER_MAPPING[pipeline_config[TOKENIZER_TYPE]].from_pretrained(
        pipeline_config[TOKENIZER_PATH]
    )

    if args.parallelize:
        parallel_processing()
    else:
        sequential_processing()

    if args.data_format == "jsonlines":
        data.close()


if __name__ == "__main__":
    main()
