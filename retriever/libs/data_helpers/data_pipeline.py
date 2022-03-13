import json
import glob
import os
from typing import Text, List, Dict, Tuple, Any
from tqdm import tqdm
import tensorflow as tf


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
        data_source: Text
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.data_source = data_source
        self.dataset_size = -1

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
        compact = tf.stack([input_ids, attention_mask], axis=-1)
        shuffled = tf.random.shuffle(compact)
        compact_sampled = shuffled[0]
        input_ids_sampled = compact_sampled[:, 0]
        attention_mask_sampled = compact_sampled[:, 1]
        return input_ids_sampled, attention_mask_sampled

    @staticmethod
    def sample(item):
        question_input_ids, question_attention_mask = PosPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        positive_context_input_ids, positive_context_attention_mask = PosPipeline.sample_attribute(
            item["positive_context/input_ids"], item["positive_context/attention_mask"])
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
    
    def build(self):
        tfrecord_files = sorted(glob.glob(os.path.join(self.data_source, "*")))
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # < calculate dataset size
        count = 0
        for item in dataset:
            count += 1
        self.dataset_size = count
        # />

        dataset = dataset.map(self.parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=10000).repeat()
        dataset = dataset.map(self.sample, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.window(
            self.forward_batch_size + self.contrastive_size,
            shift=self.forward_batch_size
        )
        dataset = dataset.flat_map(lambda window: tf.data.Dataset.zip(window))
        dataset = dataset.batch(
            self.forward_batch_size + self.contrastive_size
        )
        dataset = dataset.map(self.build_contrastive_sample, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class PosHardPipeline(Pipeline):
    def __init__(
        self,
        max_query_length: int,
        max_context_length: int,
        forward_batch_size: int,
        contrastive_size: int,
        limit_hardnegs: int,
        data_source: Text
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.data_source = data_source
        self.limit_hardnegs = limit_hardnegs

        self.feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_mask": tf.io.FixedLenFeature(shape=[self.contrastive_size], dtype=tf.int64),
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
            "hardneg_mask": item["hardneg_mask"],
            "num_hardneg": item["num_hardneg"]
        }
    
    def sample_and_pad(self, item):
        question_input_ids, question_attention_mask = PosPipeline.sample_attribute(
            item["question/input_ids"], item["question/attention_mask"])
        positive_context_input_ids, positive_context_attention_mask = PosPipeline.sample_attribute(
            item["positive_context/input_ids"], item["positive_context/attention_mask"])

        def _sample_hardneg():
            hardneg_context_input_ids = item["hardneg_context/input_ids"]
            hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
            hardneg_context_compact = tf.stack(
                [hardneg_context_input_ids, hardneg_context_attention_mask],
                axis=-1
            )
            if self.limit_hardnegs > 0:
                hardneg_context_compact = hardneg_context_compact[:self.limit_hardnegs]
            hardneg_context_compact_shuffled = tf.random.shuffle(hardneg_context_compact)
            hardneg_context_input_ids_sampled = hardneg_context_compact_shuffled[:self.contrastive_size, :, 0]
            hardneg_context_attention_mask_sampled = hardneg_context_compact_shuffled[:self.contrastive_size, :, 1]
            return hardneg_context_input_ids_sampled, hardneg_context_attention_mask_sampled
        
        def _pad_hardneg():
            hardneg_context_input_ids_padded = tf.pad(
                item["hardneg_context/input_ids"],
                paddings=[[0, self.contrastive_size - item["num_hardneg"]], [0, 0]]
            )
            hardneg_context_attention_mask_padded = tf.pad(
                item["hardneg_context/attention_mask"],
                paddings=[[0, self.contrastive_size - item["num_hardneg"]], [0, 0]]
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
        tfrecord_files = sorted(glob.glob(os.path.join(self.data_source, "*")))
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # < calculate dataset size
        count = 0
        for item in dataset:
            count += 1
        self.dataset_size = count
        # />

        dataset = dataset.map(self.parse_ex, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=10000).repeat()
        dataset = dataset.map(self.sample_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
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
        hard_data_source: Text,
        nonhard_data_source: Text,
        max_samplings: int = 100,
        use_nonhard: bool = True,
    ):
        self.max_query_length = max_query_length
        self.max_context_length = max_context_length
        self.forward_batch_size = forward_batch_size
        self.contrastive_size = contrastive_size
        self.limit_hardnegs = limit_hardnegs
        self.hard_data_source = hard_data_source
        self.nonhard_data_source = nonhard_data_source

        self.hard_feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "question/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "question/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "positive_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "hardneg_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }

        self.nonhard_feature_description = {
            "sample_id": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "negative_context/input_ids": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            "negative_context/attention_mask": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
        }
        self.max_samplings = max_samplings
        self.hard_dataset_size = -1
        self.nonhard_dataset_size = -1
        self.use_nonhard = use_nonhard

    def parse_ex_hard(self, ex):
        return tf.io.parse_example(ex, self.hard_feature_description)
    
    def parse_ex_nonhard(self, ex):
        return tf.io.parse_example(ex, self.nonhard_feature_description)
    
    def decode_hard(self, item):
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
    
    def decode_nonhard(self, item):
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

    def sample_hard(self, item):
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
            [item["positive_context/input_ids"], item["hardneg_context/input_ids"]],
            axis=0
        )
        combine_attention_mask = tf.concat(
            [item["positive_context/attention_mask"], item["hardneg_context/attention_mask"]],
            axis=0
        )
        combine_compact = tf.stack([combine_input_ids, combine_attention_mask], axis=-1)
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
    def sample_nonhard(item):
        negative_context_input_ids, negative_context_attention_mask = HardPipeline.sample_attribute(
            item["negative_context/input_ids"], item["negative_context/attention_mask"])
        return {
            "sample_id": item["sample_id"],
            "negative_context/input_ids": negative_context_input_ids,
            "negative_context/attention_mask": negative_context_attention_mask
        }

    def build(self):
        # < hard pipeline
        tfrecord_files = sorted(glob.glob(os.path.join(self.hard_data_source, "*")))
        hard_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        hard_dataset = hard_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # < dataset size
        count = 0
        for item in hard_dataset:
            count += 1
        hard_dataset_size = count
        # dataset size />
        # hard pipeline />

        if not self.use_nonhard:
            # < hard pipeline transformations
            hard_dataset = hard_dataset.map(self.parse_ex_hard, num_parallel_calls=tf.data.AUTOTUNE)
            hard_dataset = hard_dataset.map(self.decode_hard, num_parallel_calls=tf.data.AUTOTUNE)
            hard_dataset = hard_dataset.shuffle(buffer_size=10000).repeat()
            hard_dataset = hard_dataset.map(self.sample_hard, num_parallel_calls=tf.data.AUTOTUNE)
            hard_dataset = hard_dataset.window(
                self.forward_batch_size + self.contrastive_size, shift=self.forward_batch_size
            )
            hard_dataset = hard_dataset.flat_map(lambda x: tf.data.Dataset.zip(x))
            hard_dataset = hard_dataset.batch(self.forward_batch_size + self.contrastive_size)
            hard_dataset = hard_dataset.map(self.build_contrastive_sample_onlyhard)
            return hard_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            # hard pipeline transformations />

        # < nonhard pipeline
        tfrecord_files = sorted(glob.glob(os.path.join(self.nonhard_data_source, "*")))
        nonhard_dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        nonhard_dataset = nonhard_dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # < dataset size
        count = 0
        for item in nonhard_dataset:
            count += 1
        nonhard_dataset_size = count
        # dataset size />
        # nonhard pipeline />

        # < configuration
        window_shift = self.forward_batch_size
        if hard_dataset_size + nonhard_dataset_size < self.forward_batch_size + self.contrastive_size:
            raise Exception("Not allow to train with hard pipeline because number of sample is too small.")
        if hard_dataset_size <= self.forward_batch_size:
            sample_from_hard = 0
            sample_from_nonhard = self.contrastive_size
        elif hard_dataset_size < self.forward_batch_size + self.contrastive_size:
            sample_from_hard = hard_dataset_size - self.forward_batch_size
            window_shift = hard_dataset_size
            sample_from_nonhard = min(nonhard_dataset_size, self.max_samplings)
        else:
            if hard_dataset_size - self.forward_batch_size < nonhard_dataset_size:
                sample_from_hard = self.contrastive_size
                sample_from_nonhard = int(
                    nonhard_dataset_size / (hard_dataset_size - self.forward_batch_size) * self.contrastive_size
                )
                sample_from_nonhard = min(self.max_samplings, sample_from_nonhard)
            else:
                sample_from_nonhard = min(self.contrastive_size, nonhard_dataset_size)
                sample_from_hard = int(
                    (hard_dataset_size - self.forward_batch_size) / nonhard_dataset_size * sample_from_nonhard
                )
                sample_from_hard = min(self.max_samplings, sample_from_hard)
                sample_from_hard = max(sample_from_hard, self.contrastive_size)
        # configuration />
        
        # < hard pipeline transformations
        hard_dataset = hard_dataset.map(self.parse_ex_hard, num_parallel_calls=tf.data.AUTOTUNE)
        hard_dataset = hard_dataset.map(self.decode_hard, num_parallel_calls=tf.data.AUTOTUNE)
        hard_dataset = hard_dataset.shuffle(buffer_size=10000).repeat()
        hard_dataset = hard_dataset.map(self.sample_hard, num_parallel_calls=tf.data.AUTOTUNE)
        hard_dataset = hard_dataset.window(self.forward_batch_size + sample_from_hard, shift=window_shift)
        hard_dataset = hard_dataset.flat_map(lambda x: tf.data.Dataset.zip(x))
        hard_dataset = hard_dataset.batch(self.forward_batch_size + sample_from_hard)
        # hard pipeline transformations />

        # < nonhard pipeline transformations
        nonhard_dataset = nonhard_dataset.map(self.parse_ex_nonhard, num_parallel_calls=tf.data.AUTOTUNE)
        nonhard_dataset = nonhard_dataset.map(self.decode_nonhard, num_parallel_calls=tf.data.AUTOTUNE)
        nonhard_dataset = nonhard_dataset.shuffle(buffer_size=10000).repeat()
        nonhard_dataset = nonhard_dataset.map(self.sample_nonhard, num_parallel_calls=tf.data.AUTOTUNE)
        nonhard_dataset = nonhard_dataset.batch(sample_from_nonhard)
        # nonhard pipeline transformations />

        dataset = tf.data.Dataset.zip((hard_dataset, nonhard_dataset))
        dataset = dataset.map(self.build_contrastive_sample, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    def build_contrastive_sample(self, *item):
        hard_batch = item[0]
        grouped_data = {
            "sample_id": hard_batch["sample_id"][:self.forward_batch_size],
            "question/input_ids": hard_batch["question/input_ids"][:self.forward_batch_size],
            "question/attention_mask": hard_batch["question/attention_mask"][:self.forward_batch_size],
            "hardneg_context/input_ids": hard_batch["hardneg_context/input_ids"][:self.forward_batch_size],
            "hardneg_context/attention_mask": hard_batch["hardneg_context/attention_mask"][:self.forward_batch_size],
        }

        nonhard_batch = item[1]
        negative_samples = {
            "attach/sample_id": tf.concat(
                [hard_batch["sample_id"][self.forward_batch_size:], nonhard_batch["sample_id"]]
                , axis=0
            ),
            "negative_context/input_ids": tf.concat(
                [hard_batch["combine/input_ids"][self.forward_batch_size:], nonhard_batch["negative_context/input_ids"]],
                axis=0
            ),
            "negative_context/attention_mask": tf.concat(
                [hard_batch["combine/attention_mask"][self.forward_batch_size:], nonhard_batch["negative_context/attention_mask"]],
                axis=0
            )
        }
        negative_samples_compact = tf.stack(
            [
                negative_samples["negative_context/input_ids"],
                negative_samples["negative_context/attention_mask"],
                tf.tile(tf.expand_dims(negative_samples["attach/sample_id"], axis=1), [1, self.max_context_length])
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
    
    def build_contrastive_sample_onlyhard(self, item):
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


def measure_fetch_time():
    with open("configs/pipeline_training_config.json", "r") as reader:
        pipeline_config = json.load(reader)

    pos_pipeline = PosPipeline(
        max_query_length=pipeline_config["max_query_length"],
        max_context_length=pipeline_config["max_context_length"],
        forward_batch_size=pipeline_config["forward_batch_size_pos_neg"],
        contrastive_size=pipeline_config["contrastive_size_pos_neg"],
        data_source="data/v2/tfrecord/train/pos"
    )
    pos_dataset = pos_pipeline.build()
    
    hard_pipeline = HardPipeline(
        max_query_length=pipeline_config["max_query_length"],
        max_context_length=pipeline_config["max_context_length"],
        forward_batch_size=pipeline_config["forward_batch_size_hardneg_neg"],
        contrastive_size=pipeline_config["contrastive_size_hardneg_neg"],
        limit_hardnegs=pipeline_config["limit_hardnegs"],
        hard_data_source="data/v2/tfrecord/train/hard/onlyhard",
        nonhard_data_source="data/v2/tfrecord/train/hard/nonhard"
    )
    hard_dataset = hard_pipeline.build()

    poshard_pipeline = PosHardPipeline(
        max_query_length=pipeline_config["max_query_length"],
        max_context_length=pipeline_config["max_context_length"],
        forward_batch_size=pipeline_config["forward_batch_size_pos_hardneg"],
        contrastive_size=pipeline_config["contrastive_size_pos_hardneg"],
        limit_hardnegs=pipeline_config["limit_hardnegs"],
        data_source="data/v2/tfrecord/train/poshard"
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
    logger.info("Elapsed time (Pos): {}s".format(time.perf_counter() - start_time))

    start_time = time.perf_counter()
    for idx, _ in tqdm(enumerate(hard_dataset)):
        if idx == 10000:
            break
    logger.info("Elapsed time (Hard): {}s".format(time.perf_counter() - start_time))

    start_time = time.perf_counter()
    for idx, _ in tqdm(enumerate(poshard_dataset)):
        if idx == 10000:
            break
    logger.info("Elapsed time (PosHard): {}s".format(time.perf_counter() - start_time))


def get_pipelines(pipeline_config: Dict[Text, Any], use_hardneg=True):
    pos_pipeline = PosPipeline(
        max_query_length=pipeline_config["max_query_length"],
        max_context_length=pipeline_config["max_context_length"],
        forward_batch_size=pipeline_config["forward_batch_size_pos_neg"],
        contrastive_size=pipeline_config["contrastive_size_pos_neg"],
        data_source=pipeline_config["pos_data_source"]
    )
    pos_dataset = pos_pipeline.build()

    if use_hardneg:
        hard_pipeline = HardPipeline(
            max_query_length=pipeline_config["max_query_length"],
            max_context_length=pipeline_config["max_context_length"],
            forward_batch_size=pipeline_config["forward_batch_size_hardneg_neg"],
            contrastive_size=pipeline_config["contrastive_size_hardneg_neg"],
            limit_hardnegs=pipeline_config["limit_hardnegs"],
            hard_data_source=pipeline_config["hard_data_source"]["onlyhard"],
            nonhard_data_source=pipeline_config["hard_data_source"]["nonhard"],
            use_nonhard=pipeline_config["use_nonhard"]
        )
        hard_dataset = hard_pipeline.build()

        poshard_pipeline = PosHardPipeline(
            max_query_length=pipeline_config["max_query_length"],
            max_context_length=pipeline_config["max_context_length"],
            forward_batch_size=pipeline_config["forward_batch_size_pos_hardneg"],
            contrastive_size=pipeline_config["contrastive_size_pos_hardneg"],
            limit_hardnegs=pipeline_config["limit_hardnegs"],
            data_source=pipeline_config["poshard_data_source"]
        )
        poshard_dataset = poshard_pipeline.build()

        return {
            "pos_dataset": pos_dataset,
            "hard_dataset": hard_dataset,
            "poshard_dataset": poshard_dataset,
            "pos_dataset_size": pos_pipeline.dataset_size,
            "poshard_dataset_size": poshard_pipeline.dataset_size
        }
    else:
        return {
            "pos_dataset": pos_dataset,
            "pos_dataset_size": pos_pipeline.dataset_size
        }


def test_pipeline():
    with open("configs/pipeline_training_config.json", "r") as reader:
        pipeline_config = json.load(reader)

    # pos_pipeline = PosPipeline(
    #     max_query_length=pipeline_config["max_query_length"],
    #     max_context_length=pipeline_config["max_context_length"],
    #     forward_batch_size=pipeline_config["forward_batch_size_pos_neg"],
    #     contrastive_size=pipeline_config["contrastive_size_pos_neg"],
    #     data_source="data/v2/tfrecord/train/pos"
    # )
    # dataset = pos_pipeline.build()

    hard_pipeline = HardPipeline(
        max_query_length=pipeline_config["max_query_length"],
        max_context_length=pipeline_config["max_context_length"],
        forward_batch_size=pipeline_config["forward_batch_size_hardneg_neg"],
        contrastive_size=pipeline_config["contrastive_size_hardneg_neg"],
        limit_hardnegs=pipeline_config["limit_hardnegs"],
        hard_data_source="data/v2/tfrecord/train/hard/onlyhard",
        nonhard_data_source="data/v2/tfrecord/train/hard/nonhard"
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

    print("Number of duplicated per {} steps: {}".format(num_steps, num_duplicated))


def main():
    test_pipeline()
    

if __name__ == "__main__":
    main()