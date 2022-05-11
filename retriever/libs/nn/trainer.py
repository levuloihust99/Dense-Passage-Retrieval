import pip
import tensorflow as tf
import logging
import json
from typing import Union
import time

from official import nlp
import official.nlp.optimization
from tensorflow.python.distribute.values import PerReplica
from typing import List, Dict

from libs.nn.configuration import DualEncoderConfig
from libs.nn.modeling import DualEncoder
from libs.nn.losses import LossCalculator
from libs.nn.utils import flat_gradients
from libs.data_helpers.constants import (
    CONTEXT_SUB_BATCH,
    CONTRASTIVE_SIZE,
    FORWARD_BATCH_SIZE,
    GRADIENT_CACHE_CONFIG,
    INBATCH_PIPELINE_NAME,
    LOGGING_STEPS,
    MAX_CONTEXT_LENGTH,
    POS_PIPELINE_NAME,
    POSHARD_PIPELINE_NAME,
    HARD_PIPELINE_NAME,
    NUM_BACKWARD_ACCUMULATE_STEPS,
    USE_GRADIENT_CACHE,
    QUERY_SUB_BATCH,
    REGULATE_FACTOR,
    USE_HARDNEG_INBATCH,
    USE_GRADIENT_ACCUMULATE
)


logger = logging.getLogger(__name__)


class DualEncoderTrainer(object):
    def __init__(
        self,
        config: DualEncoderConfig,
        dual_encoder: DualEncoder,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_calculator: LossCalculator,
        datasets: Union[tf.data.Dataset, tf.distribute.DistributedDataset],
        strategy: tf.distribute.Strategy,
        ckpt_manager: tf.train.CheckpointManager,
        summary_writer: tf.summary.SummaryWriter
    ):
        """This class contains everything needed for training dual encoder.

        Args:
            config (DualEncoderConfig): Configuration for training model.
            dual_encoder (DualEncoder): The model to be trained.
            optimizer (tf.keras.optimizers.Optimizer): Used to update model parameters.
            loss_calculator (LossCalculator): Used to calculate loss.
            datasets (Union[tf.data.Dataset, tf.distribute.DistributedDataset]): Training data, possibly including multiple pipelines.
            strategy (tf.distribute.Strategy): Distributed strategy used when training model.
            ckpt_manager (tf.train.CheckpointManager): Manage checkpoint saving.
            summary_writer (tf.summary.SummaryWriter): Used to log training information to be visualized in tensorboard.
        """

        self.dual_encoder = dual_encoder
        self.optimizer = optimizer
        self.loss_calculator = loss_calculator
        self.strategy = strategy
        self.config = config
        self.ckpt_manager = ckpt_manager
        self.summary_writer = summary_writer
        self.iterators = {k: iter(v) for k, v in datasets.items()}
        self._setup_pipeline()
        self._setup_logging()
        self.pipeline_accumuate_computation_mapping = {
            INBATCH_PIPELINE_NAME: {
                "gc": self.inbatch_step_fn_gc,
                "base": self.inbatch_step_fn
            },
            POS_PIPELINE_NAME: {
                "gc": self.pos_step_fn_gc,
                "base": self.pos_step_fn
            },
            POSHARD_PIPELINE_NAME: {
                "gc": self.poshard_step_fn_gc,
                "base": self.poshard_step_fn
            },
            HARD_PIPELINE_NAME: {
                "gc": self.hard_step_fn_gc,
                "base": self.hard_step_fn
            }
        }
        self.pipeline_biward_computation_mapping = {
            INBATCH_PIPELINE_NAME: {
                "gc": self.inbatch_biward_step_fn_gc,
                "base": self.inbatch_biward_step_fn
            },
            POS_PIPELINE_NAME: {
                "gc": self.pos_biward_step_fn_gc,
                "base": self.pos_biward_step_fn
            },
            POSHARD_PIPELINE_NAME: {
                "gc": self.poshard_biward_step_fn_gc,
                "base": self.poshard_biward_step_fn
            },
            HARD_PIPELINE_NAME: {
                "gc": self.hard_biward_step_fn_gc,
                "base": self.hard_biward_step_fn
            }            
        }
        self.graph_cache = {}
        self._prebuild_graph_if_needed()
    
    def _prebuild_graph_if_needed(self):
        """Build computation graphs in advanced."""

        should_build_graph = {pipeline_type: self.config.pipeline_config[pipeline_type][USE_GRADIENT_CACHE]
                for pipeline_type in self._available_pipelines}
        
        if len(should_build_graph) == 0:
            return

        # params backward graph for query encoder
        self.get_params_backward_graph(is_query_encoder=True)

        # params backward graph for context encoder
        self.get_params_backward_graph(is_query_encoder=False)

        # no tracking gradient graph for query encoder
        self.get_no_tracking_gradient_forward_graph(is_query_encoder=True)

        # no tracking gradient graph for context encoder
        self.get_no_tracking_gradient_forward_graph(is_query_encoder=False)

        # tracking gradient biward graph for query encoder
        self.get_tracking_gradient_biward_graph(is_query_encoder=True)

        # tracking gradient biward graph for context encoder
        self.get_tracking_gradient_biward_graph(is_query_encoder=False)

        for pipeline_type in self._available_pipelines:
            # query batch forward graph
            self.get_batch_forward_graph(
                batch_size=self.config.pipeline_config[pipeline_type][FORWARD_BATCH_SIZE],
                sub_batch_size=self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH],
                is_query_encoder=True
            )
            
            # context batch forward graph
            self.get_batch_forward_graph(
                batch_size=self.config.pipeline_config[pipeline_type][FORWARD_BATCH_SIZE],
                sub_batch_size=self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH],
                is_query_encoder=False
            )

    def _setup_pipeline(self):
        """Setup data pipelines for fetching data item."""

        regulate_factor = self.config.pipeline_config[REGULATE_FACTOR]
        possible_pipelines = [
            INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME, POSHARD_PIPELINE_NAME, HARD_PIPELINE_NAME]
        available_pipelines = [
            p for p in possible_pipelines if p in self.iterators]
        both_inbatch_and_pos = all([p in available_pipelines for p in [
                                   INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME]])
        if both_inbatch_and_pos:
            logger.warn("You shouldn't include both '{}' and '{}' pipelines!".format(
                INBATCH_PIPELINE_NAME, POS_PIPELINE_NAME))
            inbatch_num_continuous_steps = regulate_factor // 2
        else:
            inbatch_num_continuous_steps = (
                regulate_factor
                if INBATCH_PIPELINE_NAME in available_pipelines else 0
            )
        pos_num_continuous_steps = regulate_factor - inbatch_num_continuous_steps
        poshard_and_hard_num_continuous_steps = [
            int(p in available_pipelines) for p in [POSHARD_PIPELINE_NAME, HARD_PIPELINE_NAME]]
        num_continuous_steps = [inbatch_num_continuous_steps,
                                pos_num_continuous_steps] + poshard_and_hard_num_continuous_steps
        truth_array = [p in available_pipelines for p in possible_pipelines]
        num_continuous_steps = [num_continuous_steps[i] for i in range(
            len(num_continuous_steps)) if truth_array[i]]
        self._available_pipelines = available_pipelines
        self._index_boundaries = [sum(num_continuous_steps[:idx])
                                  for idx in range(len(num_continuous_steps) + 1)]
        self._cycle_walk = regulate_factor + \
            sum(poshard_and_hard_num_continuous_steps)

    def _pipeline_name_from_index(self, index):
        """Get pipeline type from index, i.e. the relative position in a cycle walk."""

        for i in range(len(self._index_boundaries)):
            if self._index_boundaries[i] <= index < self._index_boundaries[i + 1]:
                break
        return self._available_pipelines[i]

    def _fetch_items(self, step):
        """Fetch items from an appropriate data pipeline."""

        index = step % self._cycle_walk
        pipeline_name = self._pipeline_name_from_index(index)
        backward_accumulate_steps = self.config.pipeline_config[
            pipeline_name][NUM_BACKWARD_ACCUMULATE_STEPS]
        items = [next(self.iterators[pipeline_name])
                 for _ in range(backward_accumulate_steps)]
        return pipeline_name, items

    def get_step_fn_accumulate(self, pipeline_type, computation_type=None):
        """Get computation function when using gradient accumulation from pipeline type."""

        if computation_type is None:
            computation_type = "gc" if self.config.pipeline_config[
                pipeline_type][USE_GRADIENT_CACHE] else "base"
        return self.pipeline_accumuate_computation_mapping[pipeline_type][computation_type]
    
    def get_step_fn_biward(self, pipeline_type, computation_type=None):
        """Get computation function from pipeline type when not using gradient accumulation."""

        if computation_type is None:
            computation_type = "gc" if self.config.pipeline_config[
                pipeline_type][USE_GRADIENT_CACHE] else "base"
        return self.pipeline_biward_computation_mapping[pipeline_type][computation_type]

    def _setup_logging(self):
        """Setup logging state that is used during training process."""

        self._logging_cache = {k: -1.0 for k in self._available_pipelines}
        self._pipeline_logging_count = {
            k: 0 for k in self._available_pipelines}
        
        length_of_longest_pipeline_name = max(
            [len(k) for k in self._available_pipelines])
        self._format_options = {
            "num_spaces": {
                k: length_of_longest_pipeline_name - len(k)
                for k in self._available_pipelines
            }
        }

    def train_step(self, pipeline_type, items: List):
        """A train step can combine multiple forwards and backwards but only one parameter update."""

        should_accumulate = self.config.pipeline_config[pipeline_type][NUM_BACKWARD_ACCUMULATE_STEPS] == 1 \
            and self.config.pipeline_config[pipeline_type][USE_GRADIENT_ACCUMULATE]
        
        if should_accumulate:
            step_fn = self.get_step_fn_accumulate(pipeline_type)
            loss, grads = self.accumulate_step(step_fn, items)
            self.strategy.run(
                self.update_params,
                args=(grads,)
            )
            return loss
        else:
            step_fn = self.get_step_fn_biward(pipeline_type)
            item = items[0]
            loss = self.strategy.run(step_fn, args=(item,))
            loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
            return loss

    def accumulate_step(self, step_fn, items):
        """Run multiple forward-backward passes, return the accumulated loss and grads.

        Args:
            step_fn: The computation function that does forward and backward on one minibatch
            items: Collections of minibatches to be fed into neural network
        
        Returns:
            Tuple of (loss, grads), computed on the input minibatches. The result is
            the same as if there is a bigger batch that is equivalent to multiple minibatches.
        """

        accumulate_grads = [tf.zeros_like(
            var) for var in self.dual_encoder.trainable_variables]
        accumulate_loss = 0.0
        for item in items:
            loss, grads = step_fn(item)
            accumulate_loss += loss
            accumulate_grads = [
                accum_grad + flat_gradients(grad) for accum_grad, grad in zip(accumulate_grads, grads)]

        backward_accumulate_steps = len(items)
        accumulate_grads = [
            grad / backward_accumulate_steps for grad in accumulate_grads]
        return accumulate_loss / backward_accumulate_steps, accumulate_grads

    def log(self, step, pipeline_type, loss):
        """Log training information, e.g. loss value, time elapsed, current step.

        Args:
            step (int): current training step
            pipeline_type (Text): current pipeline type
            loss: current loss value
        """

        self._logging_cache[pipeline_type] = loss
        self._pipeline_logging_count[pipeline_type] += 1
        if self._pipeline_logging_count[pipeline_type] \
                % self.config.pipeline_config[pipeline_type][LOGGING_STEPS] == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    pipeline_type, loss, step=self._pipeline_logging_count[pipeline_type])

        if (step + 1) % self.config.logging_steps == 0:
            log_string = "\nStep: {}/{}".format(step + 1,
                                                self.config.num_train_steps)
            log_string += "\nTime elapsed: {}s".format(
                time.perf_counter() - self._mark_time)
            log_string += "\nNum graphs: {}".format(self.get_num_graphs())
            log_string += "\nLoss:"
            self._mark_time = time.perf_counter()
            log_string += "\n\t- {}{}: {} *\n".format(
                pipeline_type, " " * self._format_options["num_spaces"][pipeline_type], loss)
            current_state_as_string = \
                ["\t- {}{}: {}\n".format(k, " " * self._format_options["num_spaces"][k], v)
                 for k, v in self._logging_cache.items() if k != pipeline_type]
            current_state_as_string = "".join(current_state_as_string)
            log_string += current_state_as_string
            logger.info(log_string)
    
    def get_num_graphs(self):
        num_graphs = 0
        for g in self.graph_cache.values():
            num_graphs += len(g._list_all_concrete_functions())
        return num_graphs

    def save_checkpoint(self):
        """Save checkpoint."""

        ckpt_save_path = self.ckpt_manager.save()
        logger.info('Saving checkpoint to {}'.format(ckpt_save_path))

    @tf.function
    def update_params(self, grads):
        """Update computation that is run on each training device."""

        self.optimizer.apply_gradients(zip(
            grads, self.dual_encoder.trainable_variables), experimental_aggregate_gradients=False)
    
    def train(self):
        """Training loop using gradient accumulation."""

        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training (with accumulation) ************************")
        self._mark_time = time.perf_counter()
        for step in range(trained_steps, self.config.num_train_steps):
            pipeline_type, items = self._fetch_items(step)
            loss = self.train_step(pipeline_type, items)
            self.log(step, pipeline_type, loss)
            if (step + 1) % self.config.save_checkpoint_freq == 0:
                self.save_checkpoint()

    @tf.function    
    def inbatch_biward_step_fn(self, item):
        """Forward, backward and update computation of inbatch pipeline without gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        context_input_ids = item["context/input_ids"]
        context_attention_mask = item["context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]
        if self.config.pipeline_config[INBATCH_PIPELINE_NAME][USE_HARDNEG_INBATCH]:
            hardneg_mask = item["hardneg_mask"]
        else:
            hardneg_mask = None

        with tf.GradientTape() as tape:
            query_embedding, context_embedding = self.dual_encoder(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                training=True
            )

            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "context_embedding": context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=INBATCH_PIPELINE_NAME,
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))
        return loss

    @tf.function
    def inbatch_biward_step_fn_gc(self, item):
        """Forward, backward and update computation of inbatch pipeline with gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        context_input_ids = item["context/input_ids"]
        context_attention_mask = item["context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]
        if self.config.pipeline_config[INBATCH_PIPELINE_NAME][USE_HARDNEG_INBATCH]:
            hardneg_mask = item["hardneg_mask"]
        else:
            hardneg_mask = None

        query_batch_size = query_input_ids.shape.as_list()[0]
        context_batch_size = context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            query_batch_forward_graph(query_input_ids, query_attention_mask)

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        context_input_ids_3d, context_attention_mask_3d, \
            context_embedding_tensor = \
                context_batch_forward_graph(context_input_ids, context_attention_mask)

        # backward from loss to embeddings
        query_embedding = query_embedding_tensor[:query_batch_size]
        context_embedding = context_embedding_tensor[
            :context_batch_size]

        loss, embedding_grads = self.embedding_backward_inbatch(
            query_embedding, context_embedding, duplicate_mask, hardneg_mask)

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = query_params_backward_graph(
            query_input_ids_3d, query_attention_mask_3d, query_embedding_grads_padded)

        context_embedding_grads = embedding_grads["context"]
        context_multiplier = (
            context_batch_size - 1) // context_sub_batch_size + 1
        context_padding = context_multiplier * \
            context_sub_batch_size - context_batch_size
        context_embedding_grads_padded = tf.pad(
            context_embedding_grads,
            [[0, context_padding], [0, 0]]
        )
        context_params_backward_graph = self.get_params_backward_graph(False)
        context_grads = context_params_backward_graph(
            context_input_ids_3d, context_attention_mask_3d, context_embedding_grads_padded)
        grads = query_grads + context_grads  # concatenate list

        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))
        return loss

    def inbatch_step_fn(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        loss, grads = self.strategy.run(
            self.inbatch_step_fn_computation, args=(item,))
        is_parallel_training = isinstance(
            loss, tf.distribute.DistributedValues)
        if is_parallel_training:
            grads = [grad.values[0] for grad in grads]
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None)
        return loss, grads

    @tf.function
    def inbatch_step_fn_computation(self, item):
        """Forward and backward computation of inbatch pipeline without gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        context_input_ids = item["context/input_ids"]
        context_attention_mask = item["context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]
        if self.config.pipeline_config[INBATCH_PIPELINE_NAME][USE_HARDNEG_INBATCH]:
            hardneg_mask = item["hardneg_mask"]
        else:
            hardneg_mask = None

        with tf.GradientTape() as tape:
            query_embedding, context_embedding = self.dual_encoder(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                training=True
            )

            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "context_embedding": context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=INBATCH_PIPELINE_NAME,
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return loss, ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)

    def inbatch_step_fn_gc(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.
        
        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        # possibly distributed values
        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        context_input_ids = item["context/input_ids"]
        context_attention_mask = item["context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]
        hardneg_mask = item["hardneg_mask"]

        is_parallel_training = isinstance(
            query_input_ids, tf.distribute.DistributedValues)
        if is_parallel_training:
            query_batch_size = query_input_ids.values[0].shape.as_list()[0]
            context_batch_size = context_input_ids.values[0].shape.as_list()[0]
        else:
            query_batch_size = query_input_ids.shape.as_list()[0]
            context_batch_size = context_attention_mask.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            self.strategy.run(
                query_batch_forward_graph,
                args=(query_input_ids, query_attention_mask)
            )

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        context_input_ids_3d, context_attention_mask_3d, \
            context_embedding_tensor = \
            self.strategy.run(
                context_batch_forward_graph,
                args=(context_input_ids, context_attention_mask)
            )

        # backward from loss to embeddings
        if is_parallel_training:
            # query
            query_embedding_gather = query_embedding_tensor.values
            query_embedding = [q[:query_batch_size]
                               for q in query_embedding_gather]
            query_embedding = PerReplica(query_embedding)
            # positive context
            context_embedding_gather = context_embedding_tensor.values
            context_embedding = [
                c[:context_batch_size] for c in context_embedding_gather]
            context_embedding = PerReplica(context_embedding)
        else:
            query_embedding = query_embedding_tensor[:query_batch_size]
            context_embedding = context_embedding_tensor[
                :context_batch_size]

        loss, embedding_grads = self.strategy.run(
            self.embedding_backward_inbatch,
            args=(query_embedding,
                  context_embedding, duplicate_mask, hardneg_mask)
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        if is_parallel_training:
            query_embedding_grads_gather = query_embedding_grads.values
            query_embedding_grads_padded = [
                tf.pad(q_grad, [[0, query_padding], [0, 0]])
                for q_grad in query_embedding_grads_gather
            ]
            query_embedding_grads_padded = PerReplica(
                query_embedding_grads_padded)
        else:
            query_embedding_grads_padded = tf.pad(
                query_embedding_grads,
                [[0, query_padding], [0, 0]]
            )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = self.strategy.run(
            query_params_backward_graph,
            args=(query_input_ids_3d, query_attention_mask_3d,
                  query_embedding_grads_padded)
        )

        context_embedding_grads = embedding_grads["context"]
        context_multiplier = (
            context_batch_size - 1) // context_sub_batch_size + 1
        context_padding = context_multiplier * \
            context_sub_batch_size - context_batch_size
        if is_parallel_training:
            context_embedding_grads_gather = context_embedding_grads.values
            context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, context_padding], [0, 0]])
                for c_grad in context_embedding_grads_gather
            ]
            context_embedding_grads_padded = PerReplica(
                context_embedding_grads_padded)
        else:
            context_embedding_grads_padded = tf.pad(
                context_embedding_grads,
                [[0, context_padding], [0, 0]]
            )
        context_params_backward_graph = self.get_params_backward_graph(False)
        context_grads = self.strategy.run(
            context_params_backward_graph,
            args=(context_input_ids_3d, context_attention_mask_3d,
                  context_embedding_grads_padded)
        )
        grads = query_grads + context_grads  # concatenate list

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None), \
            [self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                  grad, axis=None) for grad in grads]

    @tf.function    
    def pos_biward_step_fn(self, item):
        """Forward, backward and update computation of pos pipeline without gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]
        with tf.GradientTape() as tape:
            query_embedding, positive_context_embedding = self.dual_encoder(
                query_input_ids=grouped_data["question/input_ids"],
                query_attention_mask=grouped_data["question/attention_mask"],
                context_input_ids=grouped_data["positive_context/input_ids"],
                context_attention_mask=grouped_data["positive_context/attention_mask"],
                training=True
            )
            negative_context_embedding = self.dual_encoder.context_encoder(
                input_ids=negative_samples["negative_context/input_ids"],
                attention_mask=negative_samples["negative_context/attention_mask"],
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type=POS_PIPELINE_NAME,
                duplicate_mask=item["duplicate_mask"]
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))
        return loss

    def pos_biward_step_fn_gc(self, item):
        """Forward, backward and update computation of pos pipeline with gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        query_input_ids = grouped_data["question/input_ids"]
        query_attention_mask = grouped_data["question/attention_mask"]
        positive_context_input_ids = grouped_data["positive_context/input_ids"]
        positive_context_attention_mask = grouped_data["positive_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        query_batch_size = query_input_ids.shape.as_list()[0]
        positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
        negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            query_batch_forward_graph(query_input_ids, query_attention_mask)

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        positive_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=positive_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
            positive_context_batch_forward_graph(positive_context_input_ids, positive_context_attention_mask)

        negative_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=negative_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
            negative_context_batch_forward_graph(negative_context_input_ids, negative_context_attention_mask)

        # backward from loss to embeddings
        query_embedding = query_embedding_tensor[:query_batch_size]
        positive_context_embedding = positive_context_embedding_tensor[
            :positive_context_batch_size]
        negative_context_embedding = negative_context_embedding_tensor[
            :negative_context_batch_size]

        loss, embedding_grads = self.embedding_backward_pos(
            query_embedding, positive_context_embedding, negative_context_embedding, duplicate_mask)

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = query_params_backward_graph(
            query_input_ids_3d, query_attention_mask_3d, query_embedding_grads_padded)

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (
            positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * \
            context_sub_batch_size - positive_context_batch_size
        positive_context_embedding_grads_padded = tf.pad(
            positive_context_embedding_grads,
            [[0, positive_context_padding], [0, 0]]
        )
        context_params_backward_graph = self.get_params_backward_graph(False)
        positive_context_grads = context_params_backward_graph(
            positive_context_input_ids_3d, positive_context_attention_mask_3d, positive_context_embedding_grads_padded)
        
        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (
            negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * \
            context_sub_batch_size - negative_context_batch_size
        negative_context_embedding_grads_padded = tf.pad(
            negative_context_embedding_grads,
            [[0, negative_context_padding], [0, 0]]
        )
        negative_context_grads = context_params_backward_graph(
            negative_context_input_ids_3d, negative_context_attention_mask_3d, negative_context_embedding_grads_padded)

        context_grads = [positive_grad + negative_grad
                         for positive_grad, negative_grad in zip(positive_context_grads, negative_context_grads)]
        grads = query_grads + context_grads  # concatenate list
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))

        return loss

    def pos_step_fn(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        loss, grads = self.strategy.run(
            self.pos_step_fn_computation, args=(item,))
        is_parallel_training = isinstance(
            loss, tf.distribute.DistributedValues)
        if is_parallel_training:
            grads = [grad.values[0] for grad in grads]
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None)
        return loss, grads

    @tf.function
    def pos_step_fn_computation(self, item):
        """Forward and backward computation of pos pipeline without gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]
        with tf.GradientTape() as tape:
            query_embedding, positive_context_embedding = self.dual_encoder(
                query_input_ids=grouped_data["question/input_ids"],
                query_attention_mask=grouped_data["question/attention_mask"],
                context_input_ids=grouped_data["positive_context/input_ids"],
                context_attention_mask=grouped_data["positive_context/attention_mask"],
                training=True
            )
            negative_context_embedding = self.dual_encoder.context_encoder(
                input_ids=negative_samples["negative_context/input_ids"],
                attention_mask=negative_samples["negative_context/attention_mask"],
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type=POS_PIPELINE_NAME,
                duplicate_mask=item["duplicate_mask"]
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return loss, ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)

    def pos_step_fn_gc(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        # possibly distributed values
        query_input_ids = grouped_data["question/input_ids"]
        query_attention_mask = grouped_data["question/attention_mask"]
        positive_context_input_ids = grouped_data["positive_context/input_ids"]
        positive_context_attention_mask = grouped_data["positive_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        is_parallel_training = isinstance(
            query_input_ids, tf.distribute.DistributedValues)
        if is_parallel_training:
            query_batch_size = query_input_ids.values[0].shape.as_list()[0]
            positive_context_batch_size = positive_context_input_ids.values[0].shape.as_list()[0]
            negative_context_batch_size = negative_context_input_ids.values[0].shape.as_list()[0]
        else:
            query_batch_size = query_input_ids.shape.as_list()[0]
            positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
            negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            self.strategy.run(
                query_batch_forward_graph,
                args=(query_input_ids, query_attention_mask)
            )

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        positive_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=positive_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
            self.strategy.run(
                positive_context_batch_forward_graph,
                args=(positive_context_input_ids, positive_context_attention_mask)
            )
        negative_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=negative_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
            self.strategy.run(
                negative_context_batch_forward_graph,
                args=(negative_context_input_ids, negative_context_attention_mask)
            )

        # backward from loss to embeddings
        if is_parallel_training:
            # query
            query_embedding_gather = query_embedding_tensor.values
            query_embedding = [q[:query_batch_size]
                               for q in query_embedding_gather]
            query_embedding = PerReplica(query_embedding)
            # positive context
            positive_context_embedding_gather = positive_context_embedding_tensor.values
            positive_context_embedding = [
                c[:positive_context_batch_size] for c in positive_context_embedding_gather]
            positive_context_embedding = PerReplica(positive_context_embedding)
            # negative context
            negative_context_embedding_gather = negative_context_embedding_tensor.values
            negative_context_embedding = [
                c[:negative_context_batch_size] for c in negative_context_embedding_gather]
            negative_context_embedding = PerReplica(negative_context_embedding)
        else:
            query_embedding = query_embedding_tensor[:query_batch_size]
            positive_context_embedding = positive_context_embedding_tensor[
                :positive_context_batch_size]
            negative_context_embedding = negative_context_embedding_tensor[
                :negative_context_batch_size]

        loss, embedding_grads = self.strategy.run(
            self.embedding_backward_pos,
            args=(query_embedding, positive_context_embedding,
                  negative_context_embedding, duplicate_mask)
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        if is_parallel_training:
            query_embedding_grads_gather = query_embedding_grads.values
            query_embedding_grads_padded = [
                tf.pad(q_grad, [[0, query_padding], [0, 0]])
                for q_grad in query_embedding_grads_gather
            ]
            query_embedding_grads_padded = PerReplica(
                query_embedding_grads_padded)
        else:
            query_embedding_grads_padded = tf.pad(
                query_embedding_grads,
                [[0, query_padding], [0, 0]]
            )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = self.strategy.run(
            query_params_backward_graph,
            args=(query_input_ids_3d, query_attention_mask_3d,
                  query_embedding_grads_padded)
        )

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (
            positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * \
            context_sub_batch_size - positive_context_batch_size
        if is_parallel_training:
            positive_context_embedding_grads_gather = positive_context_embedding_grads.values
            positive_context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, positive_context_padding], [0, 0]])
                for c_grad in positive_context_embedding_grads_gather
            ]
            positive_context_embedding_grads_padded = PerReplica(
                positive_context_embedding_grads_padded)
        else:
            positive_context_embedding_grads_padded = tf.pad(
                positive_context_embedding_grads,
                [[0, positive_context_padding], [0, 0]]
            )
        context_params_backward_graph = self.get_params_backward_graph(False)
        positive_context_grads = self.strategy.run(
            context_params_backward_graph,
            args=(positive_context_input_ids_3d, positive_context_attention_mask_3d,
                  positive_context_embedding_grads_padded)
        )

        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (
            negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * \
            context_sub_batch_size - negative_context_batch_size
        if is_parallel_training:
            negative_context_embedding_grads_gather = negative_context_embedding_grads.values
            negative_context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, negative_context_padding], [0, 0]])
                for c_grad in negative_context_embedding_grads_gather
            ]
            negative_context_embedding_grads_padded = PerReplica(
                negative_context_embedding_grads_padded)
        else:
            negative_context_embedding_grads_padded = tf.pad(
                negative_context_embedding_grads,
                [[0, negative_context_padding], [0, 0]]
            )
        negative_context_grads = self.strategy.run(
            context_params_backward_graph,
            args=(negative_context_input_ids_3d, negative_context_attention_mask_3d,
                  negative_context_embedding_grads_padded)
        )

        # reduce to one device
        query_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in query_grads]
        positive_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in positive_context_grads]
        negative_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in negative_context_grads]

        context_grads = [positive_grad + negative_grad
                         for positive_grad, negative_grad in zip(positive_context_grads, negative_context_grads)]
        grads = query_grads + context_grads  # concatenate list

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None), grads

    @tf.function    
    def poshard_biward_step_fn(self, item):
        """Forward, backward and update computation of poshard pipeline without gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        hardneg_context_input_ids = tf.reshape(
            hardneg_context_input_ids, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
        hardneg_context_attention_mask = tf.reshape(
            hardneg_context_attention_mask, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])

        with tf.GradientTape() as tape:
            query_embedding = self.dual_encoder.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            positive_context_embedding = self.dual_encoder.context_encoder(
                input_ids=positive_context_input_ids,
                attention_mask=positive_context_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]

            # forward hard negative contexts with mask
            hardneg_context_embedding = self.dual_encoder.context_encoder(
                input_ids=hardneg_context_input_ids,
                attention_mask=hardneg_context_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            hardneg_context_embedding = tf.reshape(
                hardneg_context_embedding,
                [
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
                    -1
                ]
            )
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=POSHARD_PIPELINE_NAME,
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))
        return loss

    @tf.function
    def poshard_biward_step_fn_gc(self, item):
        """Forward, backward and update computation of poshard pipeline with gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        # query batch size
        query_batch_size = query_input_ids.shape.as_list()[0]
        # positive context batch size
        positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
        # reshape hardneg input ids
        hardneg_context_input_ids = tf.reshape(
            hardneg_context_input_ids, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
        # reshape hardneg attention mask
        hardneg_context_attention_mask = tf.reshape(
            hardneg_context_attention_mask, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
        # hardneg context batch size
        hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            query_batch_forward_graph(query_input_ids, query_attention_mask)

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        positive_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=positive_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
            positive_context_batch_forward_graph(positive_context_input_ids, positive_context_attention_mask)
        
        hardneg_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=hardneg_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
            hardneg_context_batch_forward_graph(hardneg_context_input_ids, hardneg_context_attention_mask)

        # query
        query_embedding = query_embedding_tensor[:query_batch_size]
        # positive context
        positive_context_embedding = positive_context_embedding_tensor[
            :positive_context_batch_size]
        # hardneg context
        hardneg_context_embedding_3d = tf.reshape(
            hardneg_context_embedding_tensor[:hardneg_context_batch_size],
            [
                self.config.pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
                self.config.pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
                -1
            ]
        )

        # backward from loss to embeddings
        loss, embedding_grads = self.embedding_backward_poshard(
            query_embedding, positive_context_embedding, hardneg_context_embedding_3d, hardneg_mask)

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = query_params_backward_graph(
            query_input_ids_3d, query_attention_mask_3d, query_embedding_grads_padded)

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (
            positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * \
            context_sub_batch_size - positive_context_batch_size
        positive_context_embedding_grads_padded = tf.pad(
            positive_context_embedding_grads,
            [[0, positive_context_padding], [0, 0]]
        )
        context_params_backward_graph = self.get_params_backward_graph(False)
        positive_context_grads = context_params_backward_graph(
            positive_context_input_ids_3d, positive_context_attention_mask_3d, positive_context_embedding_grads_padded)

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (
            hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * \
            context_sub_batch_size - hardneg_context_batch_size
        hardneg_context_embedding_grads_2d = tf.reshape(
            hardneg_context_embedding_grads,
            [hardneg_context_batch_size, -1]
        )
        hardneg_context_embedding_grads_padded = tf.pad(
            hardneg_context_embedding_grads_2d,
            [[0, hardneg_context_padding], [0, 0]]
        )
        hardneg_context_grads = context_params_backward_graph(
            hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, hardneg_context_embedding_grads_padded)

        context_grads = [
            positive_grad + hardneg_grad
            for positive_grad, hardneg_grad
            in zip(positive_context_grads, hardneg_context_grads)
        ]
        grads = query_grads + context_grads  # concatenate list
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))

        return loss

    def poshard_step_fn(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        loss, grads = self.strategy.run(
            self.poshard_step_fn_computation, args=(item,))
        is_parallel_training = isinstance(
            loss, tf.distribute.DistributedValues)
        if is_parallel_training:
            grads = [grad.values[0] for grad in grads]
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None)
        return loss, grads

    @tf.function
    def poshard_step_fn_computation(self, item):
        """Forward and backward computation of poshard pipeline without gradient cache, run on each training device."""

        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        hardneg_context_input_ids = tf.reshape(
            hardneg_context_input_ids, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
        hardneg_context_attention_mask = tf.reshape(
            hardneg_context_attention_mask, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])

        with tf.GradientTape() as tape:
            query_embedding = self.dual_encoder.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            positive_context_embedding = self.dual_encoder.context_encoder(
                input_ids=positive_context_input_ids,
                attention_mask=positive_context_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]

            # forward hard negative contexts with mask
            hardneg_context_embedding = self.dual_encoder.context_encoder(
                input_ids=hardneg_context_input_ids,
                attention_mask=hardneg_context_attention_mask,
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            hardneg_context_embedding = tf.reshape(
                hardneg_context_embedding,
                [
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
                    -1
                ]
            )
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=POSHARD_PIPELINE_NAME,
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return loss, ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)

    def poshard_step_fn_gc(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """
        
        query_input_ids = item["question/input_ids"] # possibly distributed values
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        is_parallel_training = isinstance(
            query_input_ids, tf.distribute.DistributedValues)
        if is_parallel_training:
            # query batch size
            query_batch_size = query_input_ids.values[0].shape.as_list()[0]
            # positive context batch size
            positive_context_batch_size = positive_context_input_ids.values[0].shape.as_list()[0]
            # reshape hardneg input ids
            hardneg_context_input_ids_gather = hardneg_context_input_ids.values
            hardneg_context_input_ids = [tf.reshape(
                input_ids, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
                for input_ids in hardneg_context_input_ids_gather]
            hardneg_context_input_ids = PerReplica(hardneg_context_input_ids)
            # reshape hardneg attention_mask
            hardneg_context_attention_mask_gather = hardneg_context_attention_mask.values
            hardneg_context_attention_mask = [tf.reshape(
                attention_mask, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
                for attention_mask in hardneg_context_attention_mask_gather]
            hardneg_context_attention_mask = PerReplica(
                hardneg_context_attention_mask)
            # hardneg context batch size
            hardneg_context_batch_size = hardneg_context_input_ids.values[0].shape.as_list()[0]
        else:
            # query batch size
            query_batch_size = query_input_ids.shape.as_list()[0]
            # positive context batch size
            positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
            # reshape hardneg input ids
            hardneg_context_input_ids = tf.reshape(
                hardneg_context_input_ids, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
            # reshape hardneg attention mask
            hardneg_context_attention_mask = tf.reshape(
                hardneg_context_attention_mask, [-1, self.config.pipeline_config[MAX_CONTEXT_LENGTH]])
            # hardneg context batch size
            hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            self.strategy.run(
                query_batch_forward_graph,
                args=(query_input_ids, query_attention_mask)
            )

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        positive_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=positive_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
            self.strategy.run(
                positive_context_batch_forward_graph,
                args=(positive_context_input_ids, positive_context_attention_mask)
            )
        
        hardneg_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=hardneg_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
            self.strategy.run(
                hardneg_context_batch_forward_graph,
                args=(hardneg_context_input_ids, hardneg_context_attention_mask)
            )
        if is_parallel_training:
            # query
            query_embedding_gather = query_embedding_tensor.values
            query_embedding = [q[:query_batch_size]
                               for q in query_embedding_gather]
            query_embedding = PerReplica(query_embedding)
            # positive context
            positive_context_embedding_gather = positive_context_embedding_tensor.values
            positive_context_embedding = [
                c[:positive_context_batch_size] for c in positive_context_embedding_gather]
            positive_context_embedding = PerReplica(positive_context_embedding)
            # hardneg context
            hardneg_context_embedding_gather = hardneg_context_embedding_tensor.values
            hardneg_context_embedding_3d = \
                [tf.reshape(emb[:hardneg_context_batch_size],
                            [self.config.pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
                            self.config.pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE], -1])
                 for emb in hardneg_context_embedding_gather]
            hardneg_context_embedding_3d = PerReplica(
                hardneg_context_embedding_3d)
        else:
            # query
            query_embedding = query_embedding_tensor[:query_batch_size]
            # positive context
            positive_context_embedding = positive_context_embedding_tensor[
                :positive_context_batch_size]
            # hardneg context
            hardneg_context_embedding_3d = tf.reshape(
                hardneg_context_embedding_tensor[:hardneg_context_batch_size],
                [
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][FORWARD_BATCH_SIZE],
                    self.config.pipeline_config[POSHARD_PIPELINE_NAME][CONTRASTIVE_SIZE],
                    -1
                ]
            )

        # backward from loss to embeddings
        loss, embedding_grads = self.strategy.run(
            self.embedding_backward_poshard,
            args=(query_embedding, positive_context_embedding,
                  hardneg_context_embedding_3d, hardneg_mask)
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        if is_parallel_training:
            query_embedding_grads_gather = query_embedding_grads.values
            query_embedding_grads_padded = [
                tf.pad(q_grad, [[0, query_padding], [0, 0]])
                for q_grad in query_embedding_grads_gather
            ]
            query_embedding_grads_padded = PerReplica(
                query_embedding_grads_padded)
        else:
            query_embedding_grads_padded = tf.pad(
                query_embedding_grads,
                [[0, query_padding], [0, 0]]
            )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = self.strategy.run(
            query_params_backward_graph,
            args=(query_input_ids_3d, query_attention_mask_3d,
                  query_embedding_grads_padded)
        )

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (
            positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * \
            context_sub_batch_size - positive_context_batch_size
        if is_parallel_training:
            positive_context_embedding_grads_gather = positive_context_embedding_grads.values
            positive_context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, positive_context_padding], [0, 0]])
                for c_grad in positive_context_embedding_grads_gather
            ]
            positive_context_embedding_grads_padded = PerReplica(
                positive_context_embedding_grads_padded)
        else:
            positive_context_embedding_grads_padded = tf.pad(
                positive_context_embedding_grads,
                [[0, positive_context_padding], [0, 0]]
            )
        context_params_backward_graph = self.get_params_backward_graph(False)
        positive_context_grads = self.strategy.run(
            context_params_backward_graph,
            args=(positive_context_input_ids_3d, positive_context_attention_mask_3d,
                  positive_context_embedding_grads_padded)
        )

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (
            hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * \
            context_sub_batch_size - hardneg_context_batch_size
        if is_parallel_training:
            hardneg_context_embedding_grads_gather = hardneg_context_embedding_grads.values
            hardneg_context_embedding_grads_2d = \
                [tf.reshape(emb_grads, [hardneg_context_batch_size, -1])
                 for emb_grads in hardneg_context_embedding_grads_gather]
            hardneg_context_embedding_grads_padded = \
                [tf.pad(emb_grads, [[0, hardneg_context_padding], [0, 0]])
                 for emb_grads in hardneg_context_embedding_grads_2d]
            hardneg_context_embedding_grads_padded = PerReplica(hardneg_context_embedding_grads_padded)
        else:
            hardneg_context_embedding_grads_2d = tf.reshape(
                hardneg_context_embedding_grads,
                [hardneg_context_batch_size, -1]
            )
            hardneg_context_embedding_grads_padded = tf.pad(
                hardneg_context_embedding_grads_2d,
                [[0, hardneg_context_padding], [0, 0]]
            )
        hardneg_context_grads = self.strategy.run(
            context_params_backward_graph,
            args=(hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d,
                  hardneg_context_embedding_grads_padded)
        )

        # reduce to one device
        query_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in query_grads]
        positive_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in positive_context_grads]
        hardneg_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in hardneg_context_grads]

        context_grads = [
            positive_grad + hardneg_grad
            for positive_grad, hardneg_grad
            in zip(positive_context_grads, hardneg_context_grads)
        ]
        grads = query_grads + context_grads  # concatenate list

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None), grads

    @tf.function
    def hard_biward_step_fn(self, item):
        """Forward, backward and update computation of poshard pipeline without gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]
        with tf.GradientTape() as tape:
            query_embedding, hardneg_context_embedding = self.dual_encoder(
                query_input_ids=grouped_data["question/input_ids"],
                query_attention_mask=grouped_data["question/attention_mask"],
                context_input_ids=grouped_data["hardneg_context/input_ids"],
                context_attention_mask=grouped_data["hardneg_context/attention_mask"],
                training=True
            )
            negative_context_embedding = self.dual_encoder.context_encoder(
                input_ids=negative_samples["negative_context/input_ids"],
                attention_mask=negative_samples["negative_context/attention_mask"],
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type="hard",
                duplicate_mask=item["duplicate_mask"]
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))

        return loss

    def hard_biward_step_fn_gc(self, item):
        """Forward, backward and update computation of poshard pipeline with gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        query_input_ids = grouped_data["question/input_ids"]
        query_attention_mask = grouped_data["question/attention_mask"]
        hardneg_context_input_ids = grouped_data["hardneg_context/input_ids"]
        hardneg_context_attention_mask = grouped_data["hardneg_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        query_batch_size = query_input_ids.shape.as_list()[0]
        hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]
        negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            query_batch_forward_graph(query_input_ids, query_attention_mask)

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        hardneg_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=hardneg_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
            hardneg_context_batch_forward_graph(hardneg_context_input_ids, hardneg_context_attention_mask)

        negative_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=negative_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
            negative_context_batch_forward_graph(negative_context_input_ids, negative_context_attention_mask)

        # backward from loss to embeddings
        query_embedding = query_embedding_tensor[:query_batch_size]
        hardneg_context_embedding = hardneg_context_embedding_tensor[
            :hardneg_context_batch_size]
        negative_context_embedding = negative_context_embedding_tensor[
            :negative_context_batch_size]

        loss, embedding_grads = self.embedding_backward_hard(
            query_embedding, hardneg_context_embedding, negative_context_embedding, duplicate_mask)

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = query_params_backward_graph(
            query_input_ids_3d, query_attention_mask_3d, query_embedding_grads_padded)

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (
            hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * \
            context_sub_batch_size - hardneg_context_batch_size
        hardneg_context_embedding_grads_padded = tf.pad(
            hardneg_context_embedding_grads,
            [[0, hardneg_context_padding], [0, 0]]
        )
        hardneg_context_params_backward_graph = self.get_params_backward_graph(False)
        hardneg_context_grads = hardneg_context_params_backward_graph(
            hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, hardneg_context_embedding_grads_padded)

        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (
            negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * \
            context_sub_batch_size - negative_context_batch_size
        negative_context_embedding_grads_padded = tf.pad(
            negative_context_embedding_grads,
            [[0, negative_context_padding], [0, 0]]
        )
        negative_context_grads = hardneg_context_params_backward_graph(
            negative_context_input_ids_3d, negative_context_attention_mask_3d, negative_context_embedding_grads_padded)

        context_grads = [hardneg_grad + negative_grad
                         for hardneg_grad, negative_grad in zip(hardneg_context_grads, negative_context_grads)]
        grads = query_grads + context_grads  # concatenate list
        # self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))

        return loss

    def hard_step_fn(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        loss, grads = self.strategy.run(
            self.hard_step_fn_computation, args=(item,))
        is_parallel_training = isinstance(
            loss, tf.distribute.DistributedValues)
        if is_parallel_training:
            grads = [grad.values[0] for grad in grads]
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss, axis=None)
        return loss, grads

    @tf.function
    def hard_step_fn_computation(self, item):
        """Forward and backward computation of hard pipeline without gradient cache, run on each training device."""

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]
        with tf.GradientTape() as tape:
            query_embedding, hardneg_context_embedding = self.dual_encoder(
                query_input_ids=grouped_data["question/input_ids"],
                query_attention_mask=grouped_data["question/attention_mask"],
                context_input_ids=grouped_data["hardneg_context/input_ids"],
                context_attention_mask=grouped_data["hardneg_context/attention_mask"],
                training=True
            )
            negative_context_embedding = self.dual_encoder.context_encoder(
                input_ids=negative_samples["negative_context/input_ids"],
                attention_mask=negative_samples["negative_context/attention_mask"],
                return_dict=True,
                training=True
            ).last_hidden_state[:, 0, :]
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type="hard",
                duplicate_mask=item["duplicate_mask"]
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return loss, ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)

    def hard_step_fn_gc(self, item):
        """One of step_fn functions, receive an item, then return loss and grads corresponding to that item.

        Args:
            item: An element from the data pipeline.

        Returns:
            loss, grads: loss value and gradients corresponding to the input item.
        """

        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        query_input_ids = grouped_data["question/input_ids"] # possibly distributed values
        query_attention_mask = grouped_data["question/attention_mask"]
        hardneg_context_input_ids = grouped_data["hardneg_context/input_ids"]
        hardneg_context_attention_mask = grouped_data["hardneg_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        is_parallel_training = isinstance(
            query_input_ids, tf.distribute.DistributedValues)
        if is_parallel_training:
            query_batch_size = query_input_ids.values[0].shape.as_list()[0]
            hardneg_context_batch_size = hardneg_context_input_ids.values[0].shape.as_list()[0]
            negative_context_batch_size = negative_context_input_ids.values[0].shape.as_list()[0]
        else:
            query_batch_size = query_input_ids.shape.as_list()[0]
            hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]
            negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][QUERY_SUB_BATCH]

        query_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=query_batch_size,
            sub_batch_size=query_sub_batch_size,
            is_query_encoder=True
        )
        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            self.strategy.run(
                query_batch_forward_graph,
                args=(query_input_ids, query_attention_mask)
            )

        context_sub_batch_size = \
            self.config.pipeline_config[GRADIENT_CACHE_CONFIG][CONTEXT_SUB_BATCH]

        hardneg_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=hardneg_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
            self.strategy.run(
                hardneg_context_batch_forward_graph,
                args=(hardneg_context_input_ids, hardneg_context_attention_mask)
            )

        negative_context_batch_forward_graph = self.get_batch_forward_graph(
            batch_size=negative_context_batch_size,
            sub_batch_size=context_sub_batch_size,
            is_query_encoder=False
        )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
            self.strategy.run(
                negative_context_batch_forward_graph,
                args=(negative_context_input_ids, negative_context_attention_mask)
            )

        # backward from loss to embeddings
        if is_parallel_training:
            # query
            query_embedding_gather = query_embedding_tensor.values
            query_embedding = [q[:query_batch_size]
                               for q in query_embedding_gather]
            query_embedding = PerReplica(query_embedding)
            # hardneg context
            hardneg_context_embedding_gather = hardneg_context_embedding_tensor.values
            hardneg_context_embedding = [
                c[:hardneg_context_batch_size] for c in hardneg_context_embedding_gather]
            hardneg_context_embedding = PerReplica(hardneg_context_embedding)
            # negative context
            negative_context_embedding_gather = negative_context_embedding_tensor.values
            negative_context_embedding = [
                c[:negative_context_batch_size] for c in negative_context_embedding_gather]
            negative_context_embedding = PerReplica(negative_context_embedding)
        else:
            query_embedding = query_embedding_tensor[:query_batch_size]
            hardneg_context_embedding = hardneg_context_embedding_tensor[
                :hardneg_context_batch_size]
            negative_context_embedding = negative_context_embedding_tensor[
                :negative_context_batch_size]

        loss, embedding_grads = self.strategy.run(
            self.embedding_backward_hard,
            args=(query_embedding, hardneg_context_embedding,
                  negative_context_embedding, duplicate_mask)
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        if is_parallel_training:
            query_embedding_grads_gather = query_embedding_grads.values
            query_embedding_grads_padded = [
                tf.pad(q_grad, [[0, query_padding], [0, 0]])
                for q_grad in query_embedding_grads_gather
            ]
            query_embedding_grads_padded = PerReplica(
                query_embedding_grads_padded)
        else:
            query_embedding_grads_padded = tf.pad(
                query_embedding_grads,
                [[0, query_padding], [0, 0]]
            )
        query_params_backward_graph = self.get_params_backward_graph(True)
        query_grads = self.strategy.run(
            query_params_backward_graph,
            args=(query_input_ids_3d, query_attention_mask_3d,
                  query_embedding_grads_padded)
        )

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (
            hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * \
            context_sub_batch_size - hardneg_context_batch_size
        if is_parallel_training:
            hardneg_context_embedding_grads_gather = hardneg_context_embedding_grads.values
            hardneg_context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, hardneg_context_padding], [0, 0]])
                for c_grad in hardneg_context_embedding_grads_gather
            ]
            hardneg_context_embedding_grads_padded = PerReplica(
                hardneg_context_embedding_grads_padded)
        else:
            hardneg_context_embedding_grads_padded = tf.pad(
                hardneg_context_embedding_grads,
                [[0, hardneg_context_padding], [0, 0]]
            )
        hardneg_context_params_backward_graph = self.get_params_backward_graph(False)
        hardneg_context_grads = self.strategy.run(
            hardneg_context_params_backward_graph,
            args=(hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d,
                  hardneg_context_embedding_grads_padded)
        )

        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (
            negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * \
            context_sub_batch_size - negative_context_batch_size
        if is_parallel_training:
            negative_context_embedding_grads_gather = negative_context_embedding_grads.values
            negative_context_embedding_grads_padded = [
                tf.pad(c_grad, [[0, negative_context_padding], [0, 0]])
                for c_grad in negative_context_embedding_grads_gather
            ]
            negative_context_embedding_grads_padded = PerReplica(
                negative_context_embedding_grads_padded)
        else:
            negative_context_embedding_grads_padded = tf.pad(
                negative_context_embedding_grads,
                [[0, negative_context_padding], [0, 0]]
            )
        negative_context_grads = self.strategy.run(
            hardneg_context_params_backward_graph,
            args=(negative_context_input_ids_3d, negative_context_attention_mask_3d,
                  negative_context_embedding_grads_padded)
        )

        # reduce to one device
        query_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in query_grads]
        hardneg_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in hardneg_context_grads]
        negative_context_grads = [self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, grad, axis=None) for grad in negative_context_grads]

        context_grads = [hardneg_grad + negative_grad
                         for hardneg_grad, negative_grad in zip(hardneg_context_grads, negative_context_grads)]
        grads = query_grads + context_grads  # concatenate list

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None), grads

    @tf.function
    def embedding_backward_inbatch(
        self,
        query_embedding: tf.Tensor,
        context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor,
        hardneg_mask: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            tape.watch(query_embedding)
            tape.watch(context_embedding)
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "context_embedding": context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=INBATCH_PIPELINE_NAME,
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        embedding_grads = tape.gradient(
            loss,
            {
                "query": query_embedding,
                "context": context_embedding
            }
        )
        return loss, embedding_grads

    @tf.function
    def embedding_backward_pos(
        self,
        query_embedding: tf.Tensor,
        positive_context_embedding: tf.Tensor,
        negative_context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            tape.watch(query_embedding)
            tape.watch(positive_context_embedding)
            tape.watch(negative_context_embedding)
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type=POS_PIPELINE_NAME,
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        embedding_grads = tape.gradient(
            loss,
            {
                "query": query_embedding,
                "positive_context": positive_context_embedding,
                "negative_context": negative_context_embedding
            }
        )
        return loss, embedding_grads

    @tf.function
    def embedding_backward_poshard(
        self,
        query_embedding: tf.Tensor,
        positive_context_embedding: tf.Tensor,
        hardneg_context_embedding: tf.Tensor,
        hardneg_mask: tf.Tensor,
    ):
        with tf.GradientTape() as tape:
            tape.watch(query_embedding)
            tape.watch(positive_context_embedding)
            tape.watch(hardneg_context_embedding)
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "hardneg_mask": hardneg_mask
                },
                sim_func=self.config.sim_score,
                type=POSHARD_PIPELINE_NAME
            )
            loss = loss / self.strategy.num_replicas_in_sync

        embedding_grads = tape.gradient(
            loss,
            {
                "query": query_embedding,
                "positive_context": positive_context_embedding,
                "hardneg_context": hardneg_context_embedding
            }
        )
        return loss, embedding_grads

    @tf.function
    def embedding_backward_hard(
        self,
        query_embedding: tf.Tensor,
        hardneg_context_embedding: tf.Tensor,
        negative_context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            tape.watch(query_embedding)
            tape.watch(hardneg_context_embedding)
            tape.watch(negative_context_embedding)
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "hardneg_context_embedding": hardneg_context_embedding,
                    "negative_context_embedding": negative_context_embedding
                },
                sim_func=self.config.sim_score,
                type=HARD_PIPELINE_NAME,
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        embedding_grads = tape.gradient(
            loss,
            {
                "query": query_embedding,
                "hardneg_context": hardneg_context_embedding,
                "negative_context": negative_context_embedding
            }
        )
        return loss, embedding_grads

    def get_batch_embeddings(
        self,
        input_ids: tf.Tensor,
        attention_mask: tf.Tensor,
        sub_batch_size: int,
        is_query_encoder: bool
    ):
        batch_size = input_ids.shape.as_list()[0]
        multiplier = (batch_size - 1) // sub_batch_size + 1
        padding = multiplier * sub_batch_size - batch_size
        input_ids_padded, attention_mask_padded = \
            self.padding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding=padding
            )
        input_ids_3d = tf.reshape(
            input_ids_padded,
            shape=[multiplier, sub_batch_size, -1]
        )
        attention_mask_3d = tf.reshape(
            attention_mask_padded,
            shape=[multiplier, sub_batch_size, -1]
        )
        embedding = [self.no_tracking_gradient_encoder_forward(
            inputs={
                "input_ids": input_ids_3d[idx],
                "attention_mask": attention_mask_3d[idx]
            },
            is_query_encoder=is_query_encoder
        ) for idx in range(multiplier)]
        embedding_tensor = tf.concat(embedding, axis=0)
        return input_ids_3d, attention_mask_3d, embedding_tensor

    def get_batch_forward_graph(
        self,
        batch_size: int,
        sub_batch_size: int,
        is_query_encoder: bool
    ):
        graph_key = "get_batch_forward_graph::" + json.dumps({
            "batch_size": batch_size,
            "sub_batch_size": sub_batch_size,
            "is_query_encoder": is_query_encoder
        })
        if graph_key in self.graph_cache:
            return self.graph_cache[graph_key]

        encoder = self.dual_encoder.query_encoder if is_query_encoder else self.dual_encoder.context_encoder
        multiplier = (batch_size - 1) // sub_batch_size + 1
        padding = multiplier * sub_batch_size - batch_size
        no_tracking_gradient_graph = self.get_no_tracking_gradient_forward_graph(is_query_encoder)

        def batch_forward_graph(
            input_ids: tf.Tensor,
            attention_mask: tf.Tensor
        ):
            input_ids_padded, attention_mask_padded = \
                self.padding(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    padding=padding
                )
            input_ids_3d = tf.reshape(
                input_ids_padded,
                shape=[multiplier, sub_batch_size, -1]
            )
            attention_mask_3d = tf.reshape(
                attention_mask_padded,
                shape=[multiplier, sub_batch_size, -1]
            )

            def loop_func(idx, container_emb, indices):
                emb = no_tracking_gradient_graph(
                    input_ids=input_ids_3d[idx],
                    attention_mask=attention_mask_3d[idx]
                )
                updated_container_emb = tf.tensor_scatter_nd_update(container_emb, indices, emb)
                updated_indices = indices + sub_batch_size
                return idx + 1, updated_container_emb, updated_indices

            idx = tf.constant(0)
            hidden_size = encoder.config.hidden_size
            container_embedding = tf.zeros([multiplier * sub_batch_size, hidden_size])
            initial_indices = tf.expand_dims(tf.range(sub_batch_size), axis=-1)
            _, embedding_tensor, _ = tf.while_loop(
                cond=lambda idx, emb, indices: tf.less(idx, multiplier),
                body=loop_func,
                loop_vars=(idx, container_embedding, initial_indices),
                maximum_iterations=multiplier
            )

            return input_ids_3d, attention_mask_3d, embedding_tensor
        
        self.graph_cache[graph_key] = tf.function(batch_forward_graph)
        return self.graph_cache[graph_key]

    def params_backward(
        self,
        input_ids_3d: tf.Tensor,
        attention_mask_3d: tf.Tensor,
        gradient_cache: tf.Tensor,
        is_query_encoder: bool
    ):
        if is_query_encoder is True:
            encoder = self.dual_encoder.query_encoder
        else:
            encoder = self.dual_encoder.context_encoder
        multiplier, sub_batch_size, _ = input_ids_3d.shape.as_list()

        grads = [tf.zeros_like(var) for var in encoder.trainable_variables]
        for idx in range(multiplier):
            sub_grads = self.tracking_gradient_encoder_forward(
                inputs={
                    "input_ids": input_ids_3d[idx],
                    "attention_mask": attention_mask_3d[idx]
                },
                gradient_cache=gradient_cache[
                    sub_batch_size * idx: sub_batch_size * (idx + 1)
                ],
                is_query_encoder=is_query_encoder
            )
            sub_grads = [tf.convert_to_tensor(grad) for grad in sub_grads]
            grads = [grad + sub_grad for grad,
                     sub_grad in zip(grads, sub_grads)]

        return grads

    def get_params_backward_graph(self, is_query_encoder: bool):
        graph_key = "get_params_backward_graph::" + str(is_query_encoder)
        if graph_key in self.graph_cache:
            return self.graph_cache[graph_key]
        
        encoder = self.dual_encoder.query_encoder if is_query_encoder else self.dual_encoder.context_encoder
        tracking_gradient_biward_graph = self.get_tracking_gradient_biward_graph(is_query_encoder)
        
        def params_backward_graph(input_ids_3d, attention_mask_3d, gradient_cache):
            multiplier, sub_batch_size, _ = input_ids_3d.shape.as_list()
            gradient_cache = tf.reshape(gradient_cache, [multiplier, sub_batch_size, -1])

            def loop_func(idx, grads):
                sub_grads = tracking_gradient_biward_graph(
                    input_ids=input_ids_3d[idx],
                    attention_mask=attention_mask_3d[idx],
                    gradient_cache=gradient_cache[idx]
                )
                sub_grads = [tf.convert_to_tensor(grad) for grad in sub_grads]
                grads = [grad + tf.reshape(sub_grad, grad.get_shape())
                        for grad, sub_grad in zip(grads, sub_grads)]
                return idx + 1, grads

            idx = tf.constant(0, dtype=tf.int32)
            init_grads = [tf.zeros_like(var)
                        for var in encoder.trainable_variables]
            _, grads = tf.while_loop(
                cond=lambda idx, _: tf.less(idx, multiplier),
                body=loop_func,
                loop_vars=(idx, init_grads),
                maximum_iterations=multiplier
            )

            return grads
        
        self.graph_cache[graph_key] = tf.function(params_backward_graph)
        return self.graph_cache[graph_key]

    def get_no_tracking_gradient_forward_graph(self, is_query_encoder: bool):
        graph_key = "get_no_tracking_gradient_forward_graph::" + str(is_query_encoder)
        if graph_key in self.graph_cache:
            return self.graph_cache[graph_key]
        
        encoder = self.dual_encoder.query_encoder if is_query_encoder else self.dual_encoder.context_encoder

        def no_tracking_gradient_forward_graph(input_ids, attention_mask):
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                training=True
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
            return pooled_output
        
        self.graph_cache[graph_key] = tf.function(no_tracking_gradient_forward_graph)
        return self.graph_cache[graph_key]

    def get_tracking_gradient_biward_graph(self, is_query_encoder: bool):
        graph_key = "get_tracking_gradient_biward_graph::" + str(is_query_encoder)
        if graph_key in self.graph_cache:
            return self.graph_cache[graph_key]
        
        encoder = self.dual_encoder.query_encoder if is_query_encoder else self.dual_encoder.context_encoder

        def tracking_gradient_biward_graph(input_ids, attention_mask, gradient_cache):
            with tf.GradientTape() as tape:
                outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    training=True
                )
                sequence_output = outputs.last_hidden_state
                pooled_output = sequence_output[:, 0, :]

            grads = tape.gradient(
                pooled_output, encoder.trainable_variables, output_gradients=gradient_cache)
            return grads
        
        self.graph_cache[graph_key] = tf.function(tracking_gradient_biward_graph)
        return self.graph_cache[graph_key]

    def padding(self, input_ids, attention_mask, padding):
        compact = tf.stack(
            [input_ids, attention_mask],
            axis=-1
        )
        compact_padded = tf.pad(
            compact, [[0, padding], [0, 0], [0, 0]]
        )
        input_ids_padded = compact_padded[:, :, 0]
        attention_mask_padded = compact_padded[:, :, 1]
        return input_ids_padded, attention_mask_padded
