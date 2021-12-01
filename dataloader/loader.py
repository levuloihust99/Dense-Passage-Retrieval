import os
import tensorflow as tf

from dual_encoder.configuration import DualEncoderConfig


def load_qa_dataset(
    config: DualEncoderConfig,
):
    dataset = tf.data.Dataset.list_files(config.data_tfrecord_dir + "/*")
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    num_examples = _get_num_examples(dataset)

    name_to_features = {
        "query_input_ids": tf.io.FixedLenFeature([config.query_max_seq_length], tf.int64),
        "query_attention_mask": tf.io.FixedLenFeature([config.query_max_seq_length], tf.int64),
        "context_input_ids": tf.io.FixedLenFeature([config.context_max_seq_length], tf.int64),
        "context_attention_mask": tf.io.FixedLenFeature([config.context_max_seq_length], tf.int64)
    }

    dataset = dataset.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=config.train_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_examples


def load_corpus_dataset(
    config: DualEncoderConfig,
):
    input_pattern = os.path.join(config.data_dir, 'tfrecord/corpus') + "/*"
    dataset = tf.data.Dataset.list_files(input_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    num_examples = _get_num_examples(dataset)

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([config.context_max_seq_length], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([config.context_max_seq_length], tf.int64),
    }

    dataset = dataset.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset, num_examples


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


def _get_num_examples(dataset):
    if tf.data.experimental.INFINITE_CARDINALITY == tf.data.experimental.cardinality(dataset):
        return -1
    num_examples = 0
    for _ in dataset:
        num_examples += 1
    return num_examples