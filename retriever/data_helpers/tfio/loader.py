import os
from typing import Text
import tensorflow as tf


def deserialize_qa_pairs(
    example_proto,
    name_to_features
):
    element = tf.io.parse_single_example(example_proto, name_to_features)

    query_input_ids = element['query_input_ids']
    query_attention_mask = element['query_attention_mask']
    context_input_ids = element['context_input_ids']
    context_attention_mask = element['context_attention_mask']

    query_input_ids = tf.io.parse_tensor(query_input_ids, out_type=tf.int32)
    query_attention_mask = tf.io.parse_tensor(query_attention_mask, out_type=tf.int32)
    context_input_ids = tf.io.parse_tensor(context_input_ids, out_type=tf.int32)
    context_attention_mask = tf.io.parse_tensor(context_attention_mask, out_type=tf.int32)

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask
    }


def sample(element):
    query_input_ids = element['query_input_ids']
    query_attention_mask = element['query_attention_mask']

    composite_query = tf.stack([query_input_ids, query_attention_mask], axis=-1)
    composite_query = tf.random.shuffle(composite_query)
    composite_query = composite_query[0]
    
    query_input_ids = composite_query[:, 0]
    query_attention_mask = composite_query[:, 1]

    context_input_ids = element['context_input_ids']
    context_attention_mask = element['context_attention_mask']

    composite_context = tf.stack([context_input_ids, context_attention_mask], axis=-1)
    composite_context = tf.random.shuffle(composite_context)
    composite_context = composite_context[0]

    context_input_ids = composite_context[:, 0]
    context_attention_mask = composite_context[:, 1]

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask
    }


def load_qa_dataset(
    tfrecord_dir: Text,
    query_max_seq_length: int,
    context_max_seq_length: int,
    train_batch_size: int
):
    dataset = tf.data.Dataset.list_files(tfrecord_dir + "/*.tfrecord")
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    num_examples = _get_num_examples(dataset)

    name_to_features = {
        'query_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'query_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }

    dataset = dataset.map(
        lambda record: deserialize_qa_pairs(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda element: sample(element)
    )
    dataset = dataset.map(
        lambda element: {
            'query_input_ids': tf.reshape(element['query_input_ids'], [query_max_seq_length]),
            'query_attention_mask': tf.reshape(element['query_attention_mask'], [query_max_seq_length]),
            'context_input_ids': tf.reshape(element['context_input_ids'], [context_max_seq_length]),
            'context_attention_mask': tf.reshape(element['context_attention_mask'], [context_max_seq_length])
        }
    )
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=train_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_examples


def load_corpus_dataset(
    tfrecord_dir: Text,
    context_max_seq_length: int,
):
    list_files = tf.io.gfile.listdir(tfrecord_dir)
    list_files.sort()
    list_files = [os.path.join(tfrecord_dir, tfrecord_file) for tfrecord_file in list_files]
    dataset = tf.data.Dataset.from_tensor_slices(list_files)
    dataset = dataset.flat_map(
        lambda x: tf.data.TFRecordDataset(x)
    )
    num_examples = _get_num_examples(dataset)

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([context_max_seq_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([context_max_seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32
        # So cast all int64 to int32
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        
        return example
    
    dataset = dataset.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset, num_examples


def _get_num_examples(dataset):
    if tf.data.experimental.INFINITE_CARDINALITY == tf.data.experimental.cardinality(dataset):
        return -1
    num_examples = 0
    for _ in dataset:
        num_examples += 1
    return num_examples


def deserialize_qa_pairs_with_hardneg(
    example_proto,
    name_to_features
):
    element = tf.io.parse_single_example(example_proto, name_to_features)

    query_input_ids = element['query_input_ids']
    query_attention_mask = element['query_attention_mask']
    context_input_ids = element['context_input_ids']
    context_attention_mask = element['context_attention_mask']
    hardneg_context_input_ids = element['hardneg_context_input_ids']
    hardneg_context_attention_mask = element['hardneg_context_attention_mask']

    query_input_ids = tf.io.parse_tensor(query_input_ids, out_type=tf.int32)
    query_attention_mask = tf.io.parse_tensor(query_attention_mask, out_type=tf.int32)
    context_input_ids = tf.io.parse_tensor(context_input_ids, out_type=tf.int32)
    context_attention_mask = tf.io.parse_tensor(context_attention_mask, out_type=tf.int32)
    hardneg_context_input_ids = tf.io.parse_tensor(hardneg_context_input_ids, out_type=tf.int32)
    hardneg_context_attention_mask = tf.io.parse_tensor(hardneg_context_attention_mask, out_type=tf.int32)

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask,
        'hardneg_context_input_ids': hardneg_context_input_ids,
        'hardneg_context_attention_mask': hardneg_context_attention_mask
    }


def sample_with_hardneg(element):
    # query portion
    query_input_ids = element['query_input_ids']
    query_attention_mask = element['query_attention_mask']

    composite_query = tf.stack([query_input_ids, query_attention_mask], axis=-1)
    composite_query = tf.random.shuffle(composite_query)
    composite_query = composite_query[0]
    
    query_input_ids = composite_query[:, 0]
    query_attention_mask = composite_query[:, 1]

    # context portion
    context_input_ids = element['context_input_ids']
    context_attention_mask = element['context_attention_mask']

    composite_context = tf.stack([context_input_ids, context_attention_mask], axis=-1)
    composite_context = tf.random.shuffle(composite_context)
    composite_context = composite_context[0]

    context_input_ids = composite_context[:, 0]
    context_attention_mask = composite_context[:, 1]

    # hard negative context portion
    hardneg_context_input_ids = element['hardneg_context_input_ids']
    hardneg_context_attention_mask = element['hardneg_context_attention_mask']

    composite_hardneg_context = tf.stack([hardneg_context_input_ids, hardneg_context_attention_mask], axis=-1)
    composite_hardneg_context = tf.random.shuffle(composite_hardneg_context)
    composite_hardneg_context = composite_hardneg_context[0]

    hardneg_context_input_ids = composite_hardneg_context[:, 0]
    hardneg_context_attention_mask = composite_hardneg_context[:, 1]

    return {
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask,
        'hardneg_context_input_ids': hardneg_context_input_ids,
        'hardneg_context_attention_mask': hardneg_context_attention_mask
    }


def load_qa_dataset_with_hardneg(
    tfrecord_dir: Text,
    query_max_seq_length: int,
    context_max_seq_length: int,
    train_batch_size: int
):
    dataset = tf.data.Dataset.list_files(tfrecord_dir + "/*.tfrecord")
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    num_examples = _get_num_examples(dataset)

    name_to_features = {
        'query_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'query_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_input_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'context_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'hardneg_context_input_ids':  tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'hardneg_context_attention_mask': tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }

    dataset = dataset.map(
        lambda record: deserialize_qa_pairs_with_hardneg(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        sample_with_hardneg,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda element: {
            'query_input_ids': tf.reshape(element['query_input_ids'], [query_max_seq_length]),
            'query_attention_mask': tf.reshape(element['query_attention_mask'], [query_max_seq_length]),
            'context_input_ids': tf.reshape(element['context_input_ids'], [context_max_seq_length]),
            'context_attention_mask': tf.reshape(element['context_attention_mask'], [context_max_seq_length]),
            'hardneg_context_input_ids': tf.reshape(element['hardneg_context_input_ids'], [context_max_seq_length]),
            'hardneg_context_attention_mask': tf.reshape(element['hardneg_context_attention_mask'], [context_max_seq_length])
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=train_batch_size, drop_remainder=True)

    def _combine_hardneg(element):
        context_input_ids = element['context_input_ids']
        context_attention_mask = element['context_attention_mask']
        hardneg_context_input_ids = element['hardneg_context_input_ids']
        hardneg_context_attention_mask = element['hardneg_context_attention_mask']

        combined_context_input_ids = tf.concat([context_input_ids, hardneg_context_input_ids], axis=0)
        combined_context_attention_mask = tf.concat([context_attention_mask, hardneg_context_attention_mask], axis=0)

        return {
            'query_input_ids': element['query_input_ids'],
            'query_attention_mask': element['query_attention_mask'],
            'context_input_ids': combined_context_input_ids,
            'context_attention_mask': combined_context_attention_mask
        }

    dataset = dataset.map(
        _combine_hardneg,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_examples
