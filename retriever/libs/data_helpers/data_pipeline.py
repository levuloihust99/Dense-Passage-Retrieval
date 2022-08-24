import json
import glob
import os
from re import M
from typing import Text, List, Dict, Tuple, Any
from tqdm import tqdm
import tensorflow as tf
from libs.data_helpers.constants import (
    CONTRASTIVE_SIZE,
    DATA_SOURCE,
    DETERMINISTIC,
    FORWARD_BATCH_SIZE,
    INBATCH_PIPELINE_NAME,
    MAX_CONTEXT_LENGTH,
    MAX_QUERY_LENGTH,
    PIPELINE_SEPERATE_SYMBOL,
    POS_PIPELINE_NAME,
    POSHARD_PIPELINE_NAME,
    HARD_PIPELINE_NAME,
    LIMIT_HARDNEGS,
    TRAIN_MODE,
    USE_HARD_NONE,
    USE_HARDNEG_INBATCH,
    USE_NUM_HARDNEGS_INBATCH,
    SHUFFLE_BUFFER_SIZE,
    SHUFFLE_POSITIVE,
    DataSourceType
)


class Pipeline(object):
    def build(self):
        raise NotImplementedError


class PosPipeline(Pipeline):
    def __init__(
        self,
        max_query_length: int,
        max_context_length: int,
        forward_batch_size: int,
        contrastive_size: int,
        data_source: Text,
        deterministic: bool = False,
        shuffle_buffer_size: int = 70000,
        shuffle_positive = False
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.data_source = data_source
        self.deterministic = deterministic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.dataset_size = -1
        self.shuffle_positive = shuffle_positive

        self.feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        }

    def parse_ex(self, ex):
        return tf.io.parse_example(ex, self.feature_description)

    def decode(self, item):
        questions_input_ids = tf.io.parse_tensor(
            item["question/input_ids"], out_type=tf.int32)
        questions_attention_mask = tf.io.parse_tensor(
            item["question/attention_mask"], out_type=tf.int32)
        positive_contexts_input_ids = tf.io.parse_tensor(
            item["positive_context/input_ids"], out_type=tf.int32)
        positive_contexts_attention_mask = tf.io.parse_tensor(
            item["positive_context/attention_mask"], out_type=tf.int32)
        return {
            "sample_id": item["sample_id"],
            "question/input_ids": tf.reshape(
                questions_input_ids, [-1, self.max_query_length]),
            "question/attention_mask": tf.reshape(
                questions_attention_mask, [-1, self.max_query_length]),
            "positive_context/input_ids": tf.reshape(
                positive_contexts_input_ids, [-1, self.max_context_length]),
            "positive_context/attention_mask": tf.reshape(
                positive_contexts_attention_mask, [-1, self.max_context_length]),
        }

    @staticmethod
    def sample_attribute(input_ids, attention_mask):
        compact = tf.stack([input_ids, attention_mask], axis=-1) # [batch_size, seq_length]
        shuffled = tf.random.shuffle(compact)
        compact_sampled = shuffled[0]
        input_ids_sampled = compact_sampled[:, 0] # [seq_length]
        attention_mask_sampled = compact_sampled[:, 1] # [seq_length]
        return input_ids_sampled, attention_mask_sampled

    def sample(self, item):
        question_input_ids, question_attention_mask = PosPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        if self.shuffle_positive:
            positive_context_input_ids, positive_context_attention_mask = PosPipeline.sample_attribute(
                item["positive_context/input_ids"], item["positive_context/attention_mask"])
        else:
            positive_context_input_ids = item["positive_context/input_ids"][0]
            positive_context_attention_mask = item["positive_context/attention_mask"][0]
        return {
            "sample_id": item["sample_id"],
            "question/input_ids": question_input_ids,
            "question/attention_mask": question_attention_mask,
            "positive_context/input_ids": positive_context_input_ids,
            "positive_context/attention_mask": positive_context_attention_mask,
        }

    def build_contrastive_sample(self, item):
        grouped_data = {
            "sample_id": item["sample_id"][:self.forward_batch_size],
            "question/input_ids": item["question/input_ids"][:self.forward_batch_size],
            "question/attention_mask": item["question/attention_mask"][:self.forward_batch_size],
            "positive_context/input_ids": item["positive_context/input_ids"][:self.forward_batch_size],
            "positive_context/attention_mask": item["positive_context/attention_mask"][:self.forward_batch_size]
        }
        negatives_sampled = {
            "sample_id": item["sample_id"][self.forward_batch_size:],
            "negative_context/input_ids": item["positive_context/input_ids"][self.forward_batch_size:],
            "negative_context/attention_mask": item["positive_context/attention_mask"][self.forward_batch_size:]
        }
        return {
            "grouped_data": grouped_data,
            "negative_samples": negatives_sampled
        }
    
    @staticmethod
    def compute_duplicate_mask(item):
        grouped_ids = item["grouped_data"]["sample_id"] # [B]
        negative_ids = item["negative_samples"]["sample_id"] # [C]
        B = tf.shape(grouped_ids)[0]
        C = tf.shape(negative_ids)[0]
        duplicate_mask = ~(
            tf.tile(tf.expand_dims(grouped_ids, axis=1), [1, C]) ==
            tf.tile(tf.expand_dims(negative_ids, axis=0), [B, 1])
        )
        duplicate_mask = tf.cast(duplicate_mask, dtype=tf.int32)
        duplicate_mask = tf.concat(
            [tf.ones([B, 1], dtype=tf.int32), duplicate_mask],
            axis=1
        )
        return {**item, "duplicate_mask": duplicate_mask}


    def build(self):
        tfrecord_files = sorted(tf.io.gfile.listdir(self.data_source))
        tfrecord_files = [os.path.join(self.data_source, f)
                          for f in tfrecord_files]
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if not self.deterministic:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.flat_map(lambda x: tf.data.TFRecordDataset(x))

        # < calculate dataset size
        count = 0
        for item in dataset:
            count += 1
        self.dataset_size = count
        # />

        dataset = dataset.map(
            self.parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sample, num_parallel_calls=tf.data.AUTOTUNE)
        if not self.deterministic:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.window(
            self.forward_batch_size + self.contrastive_size,
            shift=self.forward_batch_size
        )
        dataset = dataset.flat_map(lambda window: tf.data.Dataset.zip(window))
        dataset = dataset.batch(
            self.forward_batch_size + self.contrastive_size
        )
        dataset = dataset.map(self.build_contrastive_sample,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(PosPipeline.compute_duplicate_mask,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class PosHardPipeline(Pipeline):
    def __init__(
        self,
        max_query_length: int,
        max_context_length: int,
        forward_batch_size: int,
        contrastive_size: int,
        limit_hardnegs: int,
        data_source: Text,
        deterministic: bool = False,
        shuffle_buffer_size: int = 70000,
        shuffle_positive: bool = False
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.data_source = data_source
        self.limit_hardnegs = limit_hardnegs
        self.deterministic = deterministic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_positive = shuffle_positive

        self.feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "num_hardneg": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
        }
        self.dataset_size = -1

    def parse_ex(self, ex):
        return tf.io.parse_example(ex, self.feature_description)

    def decode(self, item):
        questions_input_ids = tf.io.parse_tensor(
            item["question/input_ids"], out_type=tf.int32)
        questions_attention_mask = tf.io.parse_tensor(
            item["question/attention_mask"], out_type=tf.int32)
        positive_contexts_input_ids = tf.io.parse_tensor(
            item["positive_context/input_ids"], out_type=tf.int32)
        positive_contexts_attention_mask = tf.io.parse_tensor(
            item["positive_context/attention_mask"], out_type=tf.int32)
        hardneg_contexts_input_ids = tf.io.parse_tensor(
            item["hardneg_context/input_ids"], out_type=tf.int32)
        hardneg_contexts_attention_mask = tf.io.parse_tensor(
            item["hardneg_context/attention_mask"], out_type=tf.int32)

        hardneg_mask = tf.cond(
            item["num_hardneg"] >= self.contrastive_size,
            lambda: tf.ones(self.contrastive_size, dtype=tf.int32),
            lambda: tf.concat([tf.ones(item["num_hardneg"], dtype=tf.int32), tf.zeros(self.contrastive_size - item["num_hardneg"], dtype=tf.int32)], axis=0)
        )

        return {
            "sample_id": item["sample_id"],
            "question/input_ids": tf.reshape(
                questions_input_ids, [-1, self.max_query_length]),
            "question/attention_mask": tf.reshape(
                questions_attention_mask, [-1, self.max_query_length]),
            "positive_context/input_ids": tf.reshape(
                positive_contexts_input_ids, [-1, self.max_context_length]),
            "positive_context/attention_mask": tf.reshape(
                positive_contexts_attention_mask, [-1, self.max_context_length]),
            "hardneg_context/input_ids": tf.reshape(
                hardneg_contexts_input_ids, [-1, self.max_context_length]),
            "hardneg_context/attention_mask": tf.reshape(
                hardneg_contexts_attention_mask, [-1, self.max_context_length]),
            "hardneg_mask": hardneg_mask,
            "num_hardneg": item["num_hardneg"]
        }

    def sample_and_pad(self, item):
        question_input_ids, question_attention_mask = PosPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        if self.shuffle_positive:
            positive_context_input_ids, positive_context_attention_mask = PosPipeline.sample_attribute(
                item["positive_context/input_ids"], item["positive_context/attention_mask"])
        else:
            positive_context_input_ids = item["positive_context/input_ids"][0]
            positive_context_attention_mask = item["positive_context/attention_mask"][0]

        def _sample_hardneg():
            hardneg_context_input_ids = item["hardneg_context/input_ids"]
            hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
            hardneg_context_compact = tf.stack(
                [hardneg_context_input_ids, hardneg_context_attention_mask],
                axis=-1
            )
            if self.limit_hardnegs > 0:
                hardneg_context_compact = hardneg_context_compact[:self.limit_hardnegs]
            hardneg_context_compact_shuffled = tf.random.shuffle(
                hardneg_context_compact)
            hardneg_context_input_ids_sampled = hardneg_context_compact_shuffled[
                :self.contrastive_size, :, 0]
            hardneg_context_attention_mask_sampled = hardneg_context_compact_shuffled[
                :self.contrastive_size, :, 1]
            return hardneg_context_input_ids_sampled, hardneg_context_attention_mask_sampled

        def _pad_hardneg():
            hardneg_context_input_ids_padded = tf.pad(
                item["hardneg_context/input_ids"],
                paddings=[[0, self.contrastive_size -
                           item["num_hardneg"]], [0, 0]]
            )
            hardneg_context_attention_mask_padded = tf.pad(
                item["hardneg_context/attention_mask"],
                paddings=[[0, self.contrastive_size -
                           item["num_hardneg"]], [0, 0]]
            )
            return hardneg_context_input_ids_padded, hardneg_context_attention_mask_padded

        hardneg_context_input_ids, hardneg_context_attention_mask = tf.cond(
            item["num_hardneg"] > self.contrastive_size,
            _sample_hardneg,
            _pad_hardneg
        )

        return {
            "question/input_ids": question_input_ids,
            "question/attention_mask": question_attention_mask,
            "positive_context/input_ids": positive_context_input_ids,
            "positive_context/attention_mask": positive_context_attention_mask,
            "hardneg_context/input_ids": hardneg_context_input_ids,
            "hardneg_context/attention_mask": hardneg_context_attention_mask,
            "hardneg_mask": item["hardneg_mask"],
            "num_hardneg": item["num_hardneg"]
        }

    def build(self):
        tfrecord_files = sorted(tf.io.gfile.listdir(self.data_source))
        tfrecord_files = [os.path.join(self.data_source, f)
                          for f in tfrecord_files]
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if not self.deterministic:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = dataset.flat_map(lambda x: tf.data.TFRecordDataset(x))

        # < calculate dataset size
        count = 0
        for item in dataset:
            count += 1
        self.dataset_size = count
        # />

        dataset = dataset.map(
            self.parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sample_and_pad,
                              num_parallel_calls=tf.data.AUTOTUNE)
        if not self.deterministic:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.forward_batch_size)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class HardPipeline(Pipeline):
    def __init__(
        self,
        max_query_length: int,
        max_context_length: int,
        forward_batch_size: int,
        contrastive_size: int,
        limit_hardnegs: int,
        hard_only_data_source: Text,
        hard_none_data_source: Text,
        max_samplings: int = 100,
        use_hard_none: bool = True,
        deterministic: bool = False,
        shuffle_buffer_size: int = 70000
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.limit_hardnegs = limit_hardnegs
        self.hard_only_data_source = hard_only_data_source
        self.hard_none_data_source = hard_none_data_source
        self.deterministic = deterministic
        self.shuffle_buffer_size = shuffle_buffer_size

        self.hard_only_feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }

        self.hard_none_feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "negative_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "negative_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
        self.max_samplings = max_samplings
        self.hard_only_dataset_size = -1
        self.hard_none_dataset_size = -1
        self.use_hard_none = use_hard_none

    def parse_ex_hard_only(self, ex):
        return tf.io.parse_example(ex, self.hard_only_feature_description)

    def parse_ex_hard_none(self, ex):
        return tf.io.parse_example(ex, self.hard_none_feature_description)

    def decode_hard_only(self, item):
        questions_input_ids = tf.io.parse_tensor(
            item["question/input_ids"], out_type=tf.int32)
        questions_attention_mask = tf.io.parse_tensor(
            item["question/attention_mask"], out_type=tf.int32)
        positive_contexts_input_ids = tf.io.parse_tensor(
            item["positive_context/input_ids"], out_type=tf.int32)
        positive_contexts_attention_mask = tf.io.parse_tensor(
            item["positive_context/attention_mask"], out_type=tf.int32)
        hardneg_contexts_input_ids = tf.io.parse_tensor(
            item["hardneg_context/input_ids"], out_type=tf.int32)
        hardneg_contexts_attention_mask = tf.io.parse_tensor(
            item["hardneg_context/attention_mask"], out_type=tf.int32)
        return {
            "sample_id": tf.cast(item["sample_id"], dtype=tf.int32),
            "question/input_ids": tf.reshape(
                questions_input_ids, [-1, self.max_query_length]),
            "question/attention_mask": tf.reshape(
                questions_attention_mask, [-1, self.max_query_length]),
            "positive_context/input_ids": tf.reshape(
                positive_contexts_input_ids, [-1, self.max_context_length]),
            "positive_context/attention_mask": tf.reshape(
                positive_contexts_attention_mask, [-1, self.max_context_length]),
            "hardneg_context/input_ids": tf.reshape(
                hardneg_contexts_input_ids, [-1, self.max_context_length]),
            "hardneg_context/attention_mask": tf.reshape(
                hardneg_contexts_attention_mask, [-1, self.max_context_length]),
        }

    def decode_hard_none(self, item):
        negative_contexts_input_ids = tf.io.parse_tensor(
            item["negative_context/input_ids"], out_type=tf.int32)
        negative_contexts_attention_mask = tf.io.parse_tensor(
            item["negative_context/attention_mask"], out_type=tf.int32)
        return {
            "sample_id": tf.cast(item["sample_id"], dtype=tf.int32),
            "negative_context/input_ids": tf.reshape(
                negative_contexts_input_ids, [-1, self.max_context_length]),
            "negative_context/attention_mask": tf.reshape(
                negative_contexts_attention_mask, [-1, self.max_context_length]),
        }

    @staticmethod
    def sample_attribute(input_ids, attention_mask):
        compact = tf.stack([input_ids, attention_mask], axis=-1)
        shuffled = tf.random.shuffle(compact)
        compact_sampled = shuffled[0]
        input_ids_sampled = compact_sampled[:, 0]
        attention_mask_sampled = compact_sampled[:, 1]
        return input_ids_sampled, attention_mask_sampled

    def sample_hard_only(self, item):
        question_input_ids, question_attention_mask = HardPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        if self.limit_hardnegs > 0:
            hardneg_context_input_ids = hardneg_context_input_ids[:self.limit_hardnegs]
            hardneg_context_attention_mask = hardneg_context_attention_mask[:self.limit_hardnegs]
        hardneg_context_input_ids, hardneg_context_attention_mask = HardPipeline.sample_attribute(
            hardneg_context_input_ids, hardneg_context_attention_mask)
        combine_input_ids = tf.concat(
            [item["positive_context/input_ids"],
                item["hardneg_context/input_ids"]],
            axis=0
        )
        combine_attention_mask = tf.concat(
            [item["positive_context/attention_mask"],
                item["hardneg_context/attention_mask"]],
            axis=0
        )
        combine_compact = tf.stack(
            [combine_input_ids, combine_attention_mask], axis=-1)
        combine_compact_shuffled = tf.random.shuffle(combine_compact)
        combine_compact_sampled = combine_compact_shuffled[0]
        combine_input_ids_sampled = combine_compact_sampled[:, 0]
        combine_attention_mask_sampled = combine_compact_sampled[:, 1]

        return {
            "sample_id": item["sample_id"],
            "question/input_ids": question_input_ids,
            "question/attention_mask": question_attention_mask,
            "hardneg_context/input_ids": hardneg_context_input_ids,
            "hardneg_context/attention_mask": hardneg_context_attention_mask,
            "combine/input_ids": combine_input_ids_sampled,
            "combine/attention_mask": combine_attention_mask_sampled
        }

    @staticmethod
    def sample_hard_none(item):
        negative_context_input_ids, negative_context_attention_mask = HardPipeline.sample_attribute(
            item["negative_context/input_ids"], item["negative_context/attention_mask"])
        return {
            "sample_id": item["sample_id"],
            "negative_context/input_ids": negative_context_input_ids,
            "negative_context/attention_mask": negative_context_attention_mask
        }

    def build(self):
        # < hard only pipeline
        tfrecord_files = sorted(tf.io.gfile.listdir(self.hard_only_data_source))
        tfrecord_files = [os.path.join(self.hard_only_data_source, f)
                          for f in tfrecord_files]
        hard_only_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if not self.deterministic:
            hard_only_dataset = hard_only_dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            hard_only_dataset = hard_only_dataset.flat_map(lambda x: tf.data.TFRecordDataset(x))
        # < dataset size
        count = 0
        for item in hard_only_dataset:
            count += 1
        hard_only_dataset_size = count
        # dataset size />
        # hard only pipeline />

        if not self.use_hard_none:
            # < hard only pipeline transformations
            hard_only_dataset = hard_only_dataset.map(
                self.parse_ex_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
            hard_only_dataset = hard_only_dataset.map(
                self.decode_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
            hard_only_dataset = hard_only_dataset.map(
                self.sample_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
            if not self.deterministic:
                hard_only_dataset = hard_only_dataset.shuffle(buffer_size=self.shuffle_buffer_size)
            hard_only_dataset = hard_only_dataset.repeat()
            hard_only_dataset = hard_only_dataset.window(
                self.forward_batch_size + self.contrastive_size, shift=self.forward_batch_size
            )
            hard_only_dataset = hard_only_dataset.flat_map(
                lambda x: tf.data.Dataset.zip(x))
            hard_only_dataset = hard_only_dataset.batch(
                self.forward_batch_size + self.contrastive_size)
            hard_only_dataset = hard_only_dataset.map(
                self.build_contrastive_sample_hard_only)
            hard_only_dataset = hard_only_dataset.map(HardPipeline.compute_duplicate_mask,
                                            num_parallel_calls=tf.data.AUTOTUNE)
            return hard_only_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            # hard only pipeline transformations />

        # < hard none pipeline
        tfrecord_files = sorted(tf.io.gfile.listdir(self.hard_none_data_source))
        tfrecord_files = [os.path.join(
            self.hard_none_data_source, f) for f in tfrecord_files]
        hard_none_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if not self.deterministic:
            hard_none_dataset = hard_none_dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            hard_none_dataset = hard_none_dataset.flat_map(lambda x: tf.data.TFRecordDataset(x))
        # < dataset size
        count = 0
        for item in hard_none_dataset:
            count += 1
        hard_none_dataset_size = count
        # dataset size />
        # hard none pipeline />

        # < configuration
        window_shift = self.forward_batch_size
        if hard_only_dataset_size + hard_none_dataset_size < self.forward_batch_size + self.contrastive_size:
            raise Exception(
                "Not allow to train with hard pipeline because number of sample is too small.")
        if hard_only_dataset_size <= self.forward_batch_size:
            sample_from_hard_only = 0
            sample_from_hard_none = self.contrastive_size
        elif hard_only_dataset_size < self.forward_batch_size + self.contrastive_size:
            sample_from_hard_only = hard_only_dataset_size - self.forward_batch_size
            window_shift = hard_only_dataset_size
            sample_from_hard_none = min(hard_none_dataset_size, self.max_samplings)
        else:
            if hard_only_dataset_size - self.forward_batch_size < hard_none_dataset_size:
                sample_from_hard_only = self.contrastive_size
                sample_from_hard_none = int(
                    hard_none_dataset_size /
                    (hard_only_dataset_size - self.forward_batch_size) *
                    self.contrastive_size
                )
                sample_from_hard_none = min(
                    self.max_samplings, sample_from_hard_none)
            else:
                sample_from_hard_none = min(
                    self.contrastive_size, hard_none_dataset_size)
                sample_from_hard_only = int(
                    (hard_only_dataset_size - self.forward_batch_size) /
                    hard_none_dataset_size * sample_from_hard_none
                )
                sample_from_hard_only = min(self.max_samplings, sample_from_hard_only)
                sample_from_hard_only = max(sample_from_hard_only, self.contrastive_size)
        # configuration />

        # < hard only pipeline transformations
        hard_only_dataset = hard_only_dataset.map(
            self.parse_ex_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
        hard_only_dataset = hard_only_dataset.map(
            self.decode_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
        hard_only_dataset = hard_only_dataset.map(
            self.sample_hard_only, num_parallel_calls=tf.data.AUTOTUNE)
        if not self.deterministic:
            hard_only_dataset = hard_only_dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        hard_only_dataset = hard_only_dataset.repeat()
        hard_only_dataset = hard_only_dataset.window(
            self.forward_batch_size + sample_from_hard_only, shift=window_shift)
        hard_only_dataset = hard_only_dataset.flat_map(lambda x: tf.data.Dataset.zip(x))
        hard_only_dataset = hard_only_dataset.batch(
            self.forward_batch_size + sample_from_hard_only)
        # hard only pipeline transformations />

        # < hard none pipeline transformations
        hard_none_dataset = hard_none_dataset.map(
            self.parse_ex_hard_none, num_parallel_calls=tf.data.AUTOTUNE)
        hard_none_dataset = hard_none_dataset.map(
            self.decode_hard_none, num_parallel_calls=tf.data.AUTOTUNE)
        hard_none_dataset = hard_none_dataset.map(
            self.sample_hard_none, num_parallel_calls=tf.data.AUTOTUNE)
        if not self.deterministic:
            hard_none_dataset = hard_none_dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        hard_none_dataset = hard_none_dataset.repeat()
        hard_none_dataset = hard_none_dataset.batch(sample_from_hard_none)
        # hard none pipeline transformations />

        dataset = tf.data.Dataset.zip((hard_only_dataset, hard_none_dataset))
        dataset = dataset.map(self.build_contrastive_sample,
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(HardPipeline.compute_duplicate_mask,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_contrastive_sample(self, *item):
        hard_only_batch = item[0]
        grouped_data = {
            "sample_id": hard_only_batch["sample_id"][:self.forward_batch_size],
            "question/input_ids": hard_only_batch["question/input_ids"][:self.forward_batch_size],
            "question/attention_mask": hard_only_batch["question/attention_mask"][:self.forward_batch_size],
            "hardneg_context/input_ids": hard_only_batch["hardneg_context/input_ids"][:self.forward_batch_size],
            "hardneg_context/attention_mask": hard_only_batch["hardneg_context/attention_mask"][:self.forward_batch_size],
        }

        hard_none_batch = item[1]
        negative_samples = {
            "attach/sample_id": tf.concat(
                [hard_only_batch["sample_id"][self.forward_batch_size:],
                    hard_none_batch["sample_id"]], axis=0
            ),
            "negative_context/input_ids": tf.concat(
                [hard_only_batch["combine/input_ids"][self.forward_batch_size:],
                    hard_none_batch["negative_context/input_ids"]],
                axis=0
            ),
            "negative_context/attention_mask": tf.concat(
                [hard_only_batch["combine/attention_mask"][self.forward_batch_size:],
                    hard_none_batch["negative_context/attention_mask"]],
                axis=0
            )
        }
        negative_samples_compact = tf.stack(
            [
                negative_samples["negative_context/input_ids"],
                negative_samples["negative_context/attention_mask"],
                tf.tile(tf.expand_dims(
                    negative_samples["attach/sample_id"], axis=1), [1, self.max_context_length])
            ],
            axis=-1
        )
        negative_samples_shuffled = tf.random.shuffle(negative_samples_compact)
        negative_samples = {
            "sample_id": negative_samples_shuffled[:self.contrastive_size, :, 2][:, 0],
            "negative_context/input_ids": negative_samples_shuffled[:self.contrastive_size, :, 0],
            "negative_context/attention_mask": negative_samples_shuffled[:self.contrastive_size, :, 1]
        }
        return {
            "grouped_data": grouped_data,
            "negative_samples": negative_samples
        }

    def build_contrastive_sample_hard_only(self, item):
        grouped_data = {
            "sample_id": item["sample_id"][:self.forward_batch_size],
            "question/input_ids": item["question/input_ids"][:self.forward_batch_size],
            "question/attention_mask": item["question/attention_mask"][:self.forward_batch_size],
            "hardneg_context/input_ids": item["hardneg_context/input_ids"][:self.forward_batch_size],
            "hardneg_context/attention_mask": item["hardneg_context/attention_mask"][:self.forward_batch_size]
        }
        negatives_sampled = {
            "sample_id": item["sample_id"][self.forward_batch_size:],
            "negative_context/input_ids": item["hardneg_context/input_ids"][self.forward_batch_size:],
            "negative_context/attention_mask": item["hardneg_context/attention_mask"][self.forward_batch_size:]
        }
        return {
            "grouped_data": grouped_data,
            "negative_samples": negatives_sampled
        }
    
    @staticmethod
    def compute_duplicate_mask(item):
        grouped_ids = item["grouped_data"]["sample_id"] # [B]
        negative_ids = item["negative_samples"]["sample_id"] # [C]
        B = tf.shape(grouped_ids)[0]
        C = tf.shape(negative_ids)[0]
        duplicate_mask = ~(
            tf.tile(tf.expand_dims(grouped_ids, axis=1), [1, C]) ==
            tf.tile(tf.expand_dims(negative_ids, axis=0), [B, 1])
        )
        duplicate_mask = tf.cast(duplicate_mask, dtype=tf.int32)
        duplicate_mask = tf.concat(
            [tf.ones([B, 1], dtype=tf.int32), duplicate_mask],
            axis=1
        )
        return {**item, "duplicate_mask": duplicate_mask}


class InbatchPipeline(Pipeline):
    def __init__(
        self,
        max_query_length: int,
        max_context_length: int,
        forward_batch_size: int,
        data_source: Text,
        deterministic: bool = False, # for debug
        use_hardneg: bool = False,
        use_num_hardnegs: int = 1,
        shuffle_buffer_size: int = 70000,
        shuffle_positive: bool = False
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.data_source = data_source
        self.deterministic = deterministic
        self.use_hardneg = use_hardneg
        self.use_num_hardnegs = use_num_hardnegs
        self.shuffle_buffer_size = shuffle_buffer_size
        self.dataset_size = -1
        self.shuffle_positive = shuffle_positive

        self.feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        }
        if self.use_hardneg:
            self.feature_description.update({
                "hardneg_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                "hardneg_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
            })

    def parse_ex(self, ex):
        return tf.io.parse_example(ex, self.feature_description)

    def decode(self, item):
        questions_input_ids = tf.io.parse_tensor(
            item["question/input_ids"], out_type=tf.int32)
        questions_attention_mask = tf.io.parse_tensor(
            item["question/attention_mask"], out_type=tf.int32)
        positive_contexts_input_ids = tf.io.parse_tensor(
            item["positive_context/input_ids"], out_type=tf.int32)
        positive_contexts_attention_mask = tf.io.parse_tensor(
            item["positive_context/attention_mask"], out_type=tf.int32)
        if self.use_hardneg:
            hardneg_context_input_ids = tf.io.parse_tensor(
                item["hardneg_context/input_ids"], out_type=tf.int32)
            hardneg_context_attention_mask = tf.io.parse_tensor(
                item["hardneg_context/attention_mask"], out_type=tf.int32)

        ret = {
            "sample_id": item["sample_id"],
            "question/input_ids": tf.reshape(
                questions_input_ids, [-1, self.max_query_length]),
            "question/attention_mask": tf.reshape(
                questions_attention_mask, [-1, self.max_query_length]),
            "positive_context/input_ids": tf.reshape(
                positive_contexts_input_ids, [-1, self.max_context_length]),
            "positive_context/attention_mask": tf.reshape(
                positive_contexts_attention_mask, [-1, self.max_context_length]),
        }
        if self.use_hardneg:
            ret.update({
                "hardneg_context/input_ids": tf.reshape(hardneg_context_input_ids,
                                                        [-1, self.max_context_length]),
                "hardneg_context/attention_mask": tf.reshape(hardneg_context_attention_mask,
                                                             [-1, self.max_context_length])
            })
        return ret

    @staticmethod
    def sample_attribute(input_ids, attention_mask):
        compact = tf.stack([input_ids, attention_mask], axis=-1)
        shuffled = tf.random.shuffle(compact)
        compact_sampled = shuffled[0]
        input_ids_sampled = compact_sampled[:, 0]
        attention_mask_sampled = compact_sampled[:, 1]
        return input_ids_sampled, attention_mask_sampled

    def sample(self, item):
        question_input_ids, question_attention_mask = InbatchPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        if self.shuffle_positive:
            positive_context_input_ids, positive_context_attention_mask = InbatchPipeline.sample_attribute(
                item["positive_context/input_ids"], item["positive_context/attention_mask"])
        else:
            positive_context_input_ids = item["positive_context/input_ids"]
            positive_context_attention_mask = item["positive_context/attention_ask"]
        if self.use_hardneg:
            hardneg_context_input_ids = item["hardneg_context/input_ids"]
            hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
            num_actual_hardnegs = tf.shape(hardneg_context_input_ids)[0]
            hardneg_mask = tf.concat(
                [tf.ones([num_actual_hardnegs], dtype=tf.int32), tf.zeros([self.use_num_hardnegs], dtype=tf.int32)],
                axis=0)[:self.use_num_hardnegs]
            padding = tf.zeros([self.use_num_hardnegs, self.max_context_length], dtype=tf.int32)
            hardneg_context_input_ids = tf.concat([hardneg_context_input_ids, padding], axis=0)[:self.use_num_hardnegs]
            hardneg_context_attention_mask = tf.concat([hardneg_context_attention_mask, padding], axis=0)[:self.use_num_hardnegs]

        ret = {
            "sample_id": item["sample_id"],
            "question/input_ids": question_input_ids,
            "question/attention_mask": question_attention_mask,
            "positive_context/input_ids": positive_context_input_ids,
            "positive_context/attention_mask": positive_context_attention_mask
        }
        if self.use_hardneg:
            ret.update({
                "hardneg_context/input_ids": hardneg_context_input_ids,
                "hardneg_context/attention_mask": hardneg_context_attention_mask,
                "hardneg_mask": hardneg_mask
            })
        return ret
    
    @staticmethod
    def compute_duplicate_mask(item):
        ids = item["sample_id"]
        B = tf.shape(ids)[0]
        replicate_row = tf.tile(tf.expand_dims(ids, axis=0), [B, 1])
        replicate_col = tf.tile(tf.expand_dims(ids, axis=1), [1, B])
        duplicate_mask = ~(replicate_row == replicate_col)
        duplicate_mask = duplicate_mask | tf.eye(B, dtype=tf.bool)
        duplicate_mask = tf.cast(duplicate_mask, dtype=tf.int32)
        return {**item, "duplicate_mask": duplicate_mask}
    
    def flatten(self, item):
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        
        compact_input_ids = tf.concat(
            [positive_context_input_ids,
                tf.reshape(hardneg_context_input_ids, [-1, self.max_context_length])],
            axis=0
        )
        compact_attention_mask = tf.concat(
            [positive_context_attention_mask,
                tf.reshape(hardneg_context_attention_mask, [-1, self.max_context_length])],
            axis=0
        )
        return {
            "sample_id": item["sample_id"],
            "question/input_ids": item["question/input_ids"],
            "question/attention_mask": item["question/attention_mask"],
            "context/input_ids": compact_input_ids,
            "context/attention_mask": compact_attention_mask,
            "hardneg_mask": item["hardneg_mask"]
        }
    
    def rename(self, item):
        return {
            "sample_id": item["sample_id"],
            "question/input_ids": item["question/input_ids"],
            "question/attention_mask": item["question/attention_mask"],
            "context/input_ids": item["positive_context/input_ids"],
            "context/attention_mask": item["positive_context/attention_mask"]
        }

    def build(self):
        tfrecord_files = sorted(tf.io.gfile.listdir(self.data_source))
        tfrecord_files = [os.path.join(self.data_source, f)
                          for f in tfrecord_files]
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        if not self.deterministic:
            dataset = dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.flat_map(
                lambda x: tf.data.TFRecordDataset(x)
            )

        # < calculate dataset size
        count = 0
        for item in dataset:
            count += 1
        self.dataset_size = count
        # />

        dataset = dataset.map(
            self.parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sample, num_parallel_calls=tf.data.AUTOTUNE)
        if not self.deterministic:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.forward_batch_size)
        if self.use_hardneg:
            dataset = dataset.map(self.flatten, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(self.rename, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(InbatchPipeline.compute_duplicate_mask, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def measure_fetch_time():
    with open("configs/pipeline_config.json", "r") as reader:
        pipeline_config = json.load(reader)

    pos_pipeline = PosPipeline(
        max_query_length=pipeline_config[MAX_QUERY_LENGTH],
        max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
        forward_batch_size=pipeline_config[POS_PIPELINE_NAME][FORWARD_BATCH_SIZE],
        contrastive_size=pipeline_config[POS_PIPELINE_NAME][CONTRASTIVE_SIZE],
        data_source="data/v4/evnspc+zalo/all_pos_only"
    )
    pos_dataset = pos_pipeline.build()

    hard_pipeline = HardPipeline(
        max_query_length=pipeline_config[MAX_QUERY_LENGTH],
        max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
        forward_batch_size=pipeline_config[HARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
        contrastive_size=pipeline_config[HARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
        limit_hardnegs=pipeline_config[LIMIT_HARDNEGS],
        hard_only_data_source="data/v4/evnspc+zalo/hard_only",
        hard_none_data_source="data/v4/evnspc+zalo/hard_none"
    )
    hard_dataset = hard_pipeline.build()

    poshard_pipeline = PosHardPipeline(
        max_query_length=pipeline_config[MAX_QUERY_LENGTH],
        max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
        forward_batch_size=pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
        contrastive_size=pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
        limit_hardnegs=pipeline_config[LIMIT_HARDNEGS],
        data_source="data/v4/evnspc+zalo/hard_only"
    )
    poshard_dataset = poshard_pipeline.build()

    import time
    import logging
    from libs.utils.logging import add_color_formater

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    add_color_formater(logging.root)

    start_time = time.perf_counter()
    for idx, _ in tqdm(enumerate(pos_dataset)):
        if idx == 10000:
            break
    logger.info("Elapsed time (Pos): {}s".format(
        time.perf_counter() - start_time))

    start_time = time.perf_counter()
    for idx, _ in tqdm(enumerate(hard_dataset)):
        if idx == 10000:
            break
    logger.info("Elapsed time (Hard): {}s".format(
        time.perf_counter() - start_time))

    start_time = time.perf_counter()
    for idx, _ in tqdm(enumerate(poshard_dataset)):
        if idx == 10000:
            break
    logger.info("Elapsed time (PosHard): {}s".format(
        time.perf_counter() - start_time))


def get_pipelines(pipeline_config: Dict[Text, Any]):
    pipelines_to_build = pipeline_config[TRAIN_MODE].split(PIPELINE_SEPERATE_SYMBOL)
    datasets = {}

    if INBATCH_PIPELINE_NAME in pipelines_to_build:
        inbatch_pipeline = InbatchPipeline(
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
            forward_batch_size=pipeline_config[INBATCH_PIPELINE_NAME][FORWARD_BATCH_SIZE],
            data_source=pipeline_config[DATA_SOURCE][DataSourceType.ALL],
            deterministic=pipeline_config[INBATCH_PIPELINE_NAME][DETERMINISTIC],
            use_hardneg=pipeline_config[INBATCH_PIPELINE_NAME][USE_HARDNEG_INBATCH],
            use_num_hardnegs=pipeline_config[INBATCH_PIPELINE_NAME][USE_NUM_HARDNEGS_INBATCH],
            shuffle_buffer_size=pipeline_config[SHUFFLE_BUFFER_SIZE],
            shuffle_positive=pipeline_config[INBATCH_PIPELINE_NAME][SHUFFLE_POSITIVE]
        )
        datasets[INBATCH_PIPELINE_NAME] = inbatch_pipeline.build()

    if POS_PIPELINE_NAME in pipelines_to_build:
        pos_pipeline = PosPipeline(
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
            forward_batch_size=pipeline_config[POS_PIPELINE_NAME][FORWARD_BATCH_SIZE],
            contrastive_size=pipeline_config[POS_PIPELINE_NAME][CONTRASTIVE_SIZE],
            data_source=pipeline_config[DATA_SOURCE][DataSourceType.ALL_POS_ONLY],
            deterministic=pipeline_config[POS_PIPELINE_NAME][DETERMINISTIC],
            shuffle_buffer_size=pipeline_config[SHUFFLE_BUFFER_SIZE],
            shuffle_positive = pipeline_config[POS_PIPELINE_NAME][SHUFFLE_POSITIVE],
        )
        datasets[POS_PIPELINE_NAME] = pos_pipeline.build()

    if POSHARD_PIPELINE_NAME in pipelines_to_build:
        poshard_pipeline = PosHardPipeline(
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
            forward_batch_size=pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
            contrastive_size=pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
            limit_hardnegs=pipeline_config[LIMIT_HARDNEGS],
            data_source=pipeline_config[DATA_SOURCE][DataSourceType.HARD_ONLY],
            deterministic=pipeline_config[POSHARD_PIPELINE_NAME][DETERMINISTIC],
            shuffle_buffer_size=pipeline_config[SHUFFLE_BUFFER_SIZE],
            shuffle_positive=pipeline_config[POSHARD_PIPELINE_NAME][SHUFFLE_POSITIVE]
        )
        datasets[POSHARD_PIPELINE_NAME] = poshard_pipeline.build()

    if HARD_PIPELINE_NAME in pipelines_to_build:
        hard_pipeline = HardPipeline(
            max_query_length=pipeline_config[MAX_QUERY_LENGTH],
            max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
            forward_batch_size=pipeline_config[HARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
            contrastive_size=pipeline_config[HARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
            limit_hardnegs=pipeline_config[LIMIT_HARDNEGS],
            hard_only_data_source=pipeline_config[DATA_SOURCE][DataSourceType.HARD_ONLY],
            hard_none_data_source=pipeline_config[DATA_SOURCE][DataSourceType.HARD_NONE],
            use_hard_none=pipeline_config[HARD_PIPELINE_NAME][USE_HARD_NONE],
            deterministic=pipeline_config[HARD_PIPELINE_NAME][DETERMINISTIC],
            shuffle_buffer_size=pipeline_config[SHUFFLE_BUFFER_SIZE]
        )
        datasets[HARD_PIPELINE_NAME] = hard_pipeline.build()
    
    return datasets


def test_pipeline_duplicate():
    with open("configs/pipeline_config.json", "r") as reader:
        pipeline_config = json.load(reader)

    hard_pipeline = HardPipeline(
        max_query_length=pipeline_config[MAX_QUERY_LENGTH],
        max_context_length=pipeline_config[MAX_CONTEXT_LENGTH],
        forward_batch_size=pipeline_config[HARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
        contrastive_size=pipeline_config[HARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
        limit_hardnegs=pipeline_config[LIMIT_HARDNEGS],
        hard_only_data_source="data/v4/evnspc+zalo/hard_only",
        hard_none_data_source="data/v4/evnspc+zalo/hard_none"
    )
    dataset = hard_pipeline.build()

    def _take_id(*item):
        grouped_ids = item[0]["sample_id"]
        negative_ids = item[1]["sample_id"]
        return {
            "grouped_ids": grouped_ids,
            "negative_ids": negative_ids
        }

    dataset = dataset.map(_take_id)
    num_duplicated = 0
    num_steps = 500000
    for idx, item in tqdm(enumerate(dataset)):
        grouped_ids = set(item["grouped_ids"].numpy().tolist())
        negative_ids = set(item["negative_ids"].numpy().tolist())
        common_ids = grouped_ids.intersection(negative_ids)
        num_duplicated += len(common_ids)
        if idx == num_steps - 1:
            break

    print("Number of duplicated per {} steps: {}".format(
        num_steps, num_duplicated))


def main():
    test_pipeline_duplicate()


if __name__ == "__main__":
    main()
