from numpy import True_
import tensorflow as tf
import logging
from typing import Union
import time

from official import nlp
import official.nlp.optimization

from libs.nn.configuration import DualEncoderConfig
from libs.nn.modeling import DualEncoder
from libs.nn.losses import LossCalculator
from libs.nn.utils import flat_gradients


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
        self.dual_encoder = dual_encoder
        self.optimizer = optimizer
        self.loss_calculator = loss_calculator
        self.datasets = datasets
        self.strategy = strategy
        self.config = config
        self.ckpt_manager = ckpt_manager
        self.summary_writer = summary_writer
        self.iterators = {k: iter(v) for k, v in self.datasets.items()}

    def train(self):
        if self.config.pipeline_config["train_mode"] == "hard":
            self.train_hard()
        elif self.config.pipeline_config["train_mode"] == "poshard":
            self.train_poshard()
        elif self.config.pipeline_config["train_mode"] == "pos":
            self.train_pos()
        elif self.config.pipeline_config["train_mode"] == "inbatch":
            self.train_inbatch()
        elif self.config.pipeline_config["train_mode"] == "inbatch++":
            self.train_inbatch_plus()

    def train_pos(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        start_time = time.perf_counter()
        step_fn = (
            self.pos_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.pos_step_fn
        )
        while global_step < self.config.num_train_steps:
            loss = self.accumulate_step(
                iterator=self.iterators["pos_dataset"],
                step_fn=step_fn,
                backward_accumulate_steps=self.config.pipeline_config["backward_accumulate_pos_neg"]
            )
            global_step += 1
            if global_step % self.config.logging_steps == 0:
                logger.info("Step: {} || Loss: {} || Time elapsed: {}s".format(
                    global_step, loss, time.perf_counter() - start_time))
                with self.summary_writer.as_default():
                    tf.summary.scalar("Pos", loss, step=global_step)
                start_time = time.perf_counter()
            if global_step % self.config.save_checkpoint_freq == 0:
                self.save_checkpoint()

    def train_hard(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        pos_loss = -1.0
        poshard_loss = -1.0
        hard_loss = -1.0
        start_time = time.perf_counter()
        pos_step_fn = (
            self.pos_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.pos_step_fn
        )
        poshard_step_fn = (
            self.poshard_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.poshard_step_fn
        )
        hard_step_fn = (
            self.hard_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.hard_step_fn
        )
        while global_step < self.config.num_train_steps:
            # < pos pipeline
            if (global_step + 2) % (self.config.regulate_factor + 2) != 0:
                pos_loss = self.accumulate_step(
                    iterator=self.iterators["pos_dataset"],
                    step_fn=pos_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config["backward_accumulate_pos_neg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": pos_loss, "type": "Pos"},
                        "previous": [
                            {"value": poshard_loss, "type": "PosHard"},
                            {"value": hard_loss, "type": "Hard"}
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # pos pipeline />
            # < hard pipelines
            else:
                poshard_loss = self.accumulate_step(
                    iterator=self.iterators["poshard_dataset"],
                    step_fn=poshard_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config[
                        "backward_accumulate_pos_hardneg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": poshard_loss, "type": "PosHard"},
                        "previous": [
                            {"value": pos_loss, "type": "Pos"},
                            {"value": hard_loss, "type": "Hard"},
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
                hard_loss = self.accumulate_step(
                    iterator=self.iterators["hard_dataset"],
                    step_fn=hard_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config[
                        "backward_accumulate_hardneg_neg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": hard_loss, "type": "Hard"},
                        "previous": [
                            {"value": pos_loss, "type": "Pos"},
                            {"value": poshard_loss, "type": "PosHard"}
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # hard pipelines />

    def train_poshard(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        pos_loss = -1.0
        poshard_loss = -1.0
        start_time = time.perf_counter()
        pos_step_fn = (
            self.pos_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.pos_step_fn
        )
        poshard_step_fn = (
            self.poshard_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.poshard_step_fn
        )
        while global_step < self.config.num_train_steps:
            # < pos pipeline
            if (global_step + 1) % (self.config.regulate_factor + 1) != 0:
                pos_loss = self.accumulate_step(
                    iterator=self.iterators["pos_dataset"],
                    step_fn=pos_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config["backward_accumulate_pos_neg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": pos_loss, "type": "Pos"},
                        "previous": [
                            {"value": poshard_loss, "type": "PosHard"},
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # pos pipeline />
            # < hard pipelines
            else:
                poshard_loss = self.accumulate_step(
                    iterator=self.iterators["poshard_dataset"],
                    step_fn=poshard_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config[
                        "backward_accumulate_pos_hardneg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": poshard_loss, "type": "PosHard"},
                        "previous": [
                            {"value": pos_loss, "type": "Pos"},
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # hard pipelines />

    def train_inbatch(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        inbatch_loss = -1.0
        start_time = time.perf_counter()
        step_fn = (
            self.inbatch_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.inbatch_step_fn
        )
        while global_step < self.config.num_train_steps:
            inbatch_loss = self.accumulate_step(
                iterator=self.iterators["inbatch_dataset"],
                step_fn=step_fn,
                backward_accumulate_steps=self.config.pipeline_config["backward_accumulate_inbatch"]
            )
            global_step += 1
            if global_step % self.config.logging_steps == 0:
                logger.info("Step: {} || Inbatch loss: {} || Time elapsed: {}s".format(
                    global_step, inbatch_loss, time.perf_counter() - start_time))
                with self.summary_writer.as_default():
                    tf.summary.scalar("Pos", inbatch_loss, step=global_step)
                start_time = time.perf_counter()
            if global_step % self.config.save_checkpoint_freq == 0:
                self.save_checkpoint()
    
    def train_inbatch_plus(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        inbatch_loss = -1.0
        poshard_loss = -1.0
        hard_loss = -1.0
        start_time = time.perf_counter()
        inbatch_step_fn = (
            self.inbatch_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.inbatch_step_fn
        )
        poshard_step_fn = (
            self.poshard_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.poshard_step_fn
        )
        hard_step_fn = (
            self.hard_step_fn_gc
            if self.config.pipeline_config["use_gradient_cache"] is True
            else self.hard_step_fn
        )
        while global_step < self.config.num_train_steps:
            # < inbatch pipeline
            if (global_step + 2) % (self.config.regulate_factor + 2) != 0:
                inbatch_loss = self.accumulate_step(
                    iterator=self.iterators["inbatch_dataset"],
                    step_fn=inbatch_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config["backward_accumulate_inbatch"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": inbatch_loss, "type": "Inbatch"},
                        "previous": [
                            {"value": poshard_loss, "type": "PosHard"},
                            {"value": hard_loss, "type": "Hard"}
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # inbatch pipeline />
            # < hard pipelines
            else:
                poshard_loss = self.accumulate_step(
                    iterator=self.iterators["poshard_dataset"],
                    step_fn=poshard_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config[
                        "backward_accumulate_pos_hardneg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": poshard_loss, "type": "PosHard"},
                        "previous": [
                            {"value": inbatch_loss, "type": "Inbatch"},
                            {"value": hard_loss, "type": "Hard"},
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
                hard_loss = self.accumulate_step(
                    iterator=self.iterators["hard_dataset"],
                    step_fn=hard_step_fn,
                    backward_accumulate_steps=self.config.pipeline_config[
                        "backward_accumulate_hardneg_neg"]
                )
                global_step += 1
                if global_step % self.config.logging_steps == 0:
                    info = {
                        "previous_time": start_time,
                        "global_step": global_step,
                        "current": {"value": hard_loss, "type": "Hard"},
                        "previous": [
                            {"value": inbatch_loss, "type": "Inbatch"},
                            {"value": poshard_loss, "type": "PosHard"}
                        ]
                    }
                    self.log(info)
                    start_time = time.perf_counter()
                if global_step % self.config.save_checkpoint_freq == 0:
                    self.save_checkpoint()
            # hard pipelines />

    def log(self, info):
        previous_info = info["previous"]
        previous_logs = ["\t\t- Value = {} || Type = {}".format(
            item["value"], item["type"]) for item in previous_info]
        previous_logs = "\n".join(previous_logs)
        logger.info(
            ("\nStep: {}/{}"
             "\nTime elapsed: {}s"
             "\nLoss:"
             "\n\tCurrent:"
             "\n\t\t- Value = {} || Type = {}"
             "\n\tPrevious:"
             "\n{}\n").format(
                 info["global_step"],
                 self.config.num_train_steps,
                 time.perf_counter() - info["previous_time"],
                 info["current"]["value"],
                 info["current"]["type"],
                 previous_logs
            )
        )
        with self.summary_writer.as_default():
            tf.summary.scalar(
                info["current"]["type"], info["current"]["value"], step=info["global_step"])

    def save_checkpoint(self):
        ckpt_save_path = self.ckpt_manager.save()
        logger.info('Saving checkpoint to {}'.format(ckpt_save_path))

    def accumulate_step(
        self,
        iterator,
        step_fn,
        backward_accumulate_steps: int,
    ):
        accumulate_grads = None
        accumulate_loss = 0
        for _ in range(backward_accumulate_steps):
            item = next(iterator)
            per_replica_results = self.strategy.run(
                step_fn, args=(item,)
            )
            loss = self.strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_results["loss"], axis=None)
            accumulate_loss += loss
            grads = per_replica_results["grads"]
            if not isinstance(self.strategy, tf.distribute.get_strategy().__class__) and self.strategy.num_replicas_in_sync > 1:
                grads = [grad.values[0] for grad in grads]
            if accumulate_grads is None:
                accumulate_grads = [flat_gradients(grad) for grad in grads]
            else:
                accumulate_grads = [flat_gradients(
                    grad) + acc_grad for grad, acc_grad in zip(grads, accumulate_grads)]

        accumulate_grads = [
            grad / backward_accumulate_steps for grad in accumulate_grads]
        self.strategy.run(
            self.update_params,
            args=(accumulate_grads,)
        )
        return accumulate_loss / backward_accumulate_steps

    @tf.function
    def update_params(self, grads):
        self.optimizer.apply_gradients(zip(
            grads, self.dual_encoder.trainable_variables), experimental_aggregate_gradients=False)

    @tf.function
    def pos_step_fn(self, item):
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
                type="pos",
                duplicate_mask=item["duplicate_mask"]
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }

    @tf.function
    def hard_step_fn(self, item):
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
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }

    @tf.function
    def poshard_step_fn(self, item):
        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        hardneg_context_input_ids = tf.reshape(
            hardneg_context_input_ids, [-1, self.config.pipeline_config["max_context_length"]])
        hardneg_context_attention_mask = tf.reshape(
            hardneg_context_attention_mask, [-1, self.config.pipeline_config["max_context_length"]])

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
                    self.config.pipeline_config["forward_batch_size_pos_hardneg"],
                    self.config.pipeline_config["contrastive_size_pos_hardneg"],
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
                type="poshard",
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }

    @tf.function
    def inbatch_step_fn(self, item):
        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        with tf.GradientTape() as tape:
            query_embedding, positive_context_embedding = self.dual_encoder(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                context_input_ids=positive_context_input_ids,
                context_attention_mask=positive_context_attention_mask,
                training=True
            )

            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": positive_context_embedding,
                },
                sim_func=self.config.sim_score,
                type="inbatch",
                duplicate_mask=duplicate_mask
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }
    
    def inbatch_step_fn_gc(self, item):
        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["query_sub_batch"]

        query_input_ids_3d, query_attention_mask_3d, query_embedding_tensor = \
            self.get_batch_embeddings(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                sub_batch_size=query_sub_batch_size,
                is_query_encoder=True
            )
        
        context_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["context_sub_batch"]

        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=positive_context_input_ids,
                    attention_mask=positive_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )

        # backward from loss to embeddings
        query_batch_size = query_input_ids.shape.as_list()[0]
        positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
        loss, embedding_grads = self.embedding_backward_inbatch(
            query_embedding=query_embedding_tensor[:query_batch_size],
            context_embedding=positive_context_embedding_tensor[:positive_context_batch_size],
            duplicate_mask=duplicate_mask
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads, [[0, query_padding], [0, 0]]
        )
        query_grads = self.params_backward(
            query_input_ids_3d,
            query_attention_mask_3d,
            gradient_cache=query_embedding_grads_padded,
            is_query_encoder=True
        )

        positive_context_embedding_grads = embedding_grads["context"]
        positive_context_multiplier = (positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * context_sub_batch_size - positive_context_batch_size
        positive_context_embedding_grads_padded = tf.pad(
            positive_context_embedding_grads, [[0, positive_context_padding], [0, 0]]
        )
        context_grads = self.params_backward(
            positive_context_input_ids_3d,
            positive_context_attention_mask_3d,
            gradient_cache=positive_context_embedding_grads_padded,
            is_query_encoder=False
        )
        grads = query_grads + context_grads # concatenate list
        ctx = ctx = tf.distribute.get_replica_context()

        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }

    @tf.function
    def embedding_backward_inbatch(
        self,
        query_embedding: tf.Tensor,
        context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor
    ):
        with tf.GradientTape() as tape:
            tape.watch(query_embedding)
            tape.watch(context_embedding)
            loss = self.loss_calculator.compute(
                inputs={
                    "query_embedding": query_embedding,
                    "positive_context_embedding": context_embedding,
                },
                sim_func=self.config.sim_score,
                type="inbatch",
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

    def pos_step_fn_gc(self, item):
        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        query_input_ids = grouped_data["question/input_ids"]
        query_attention_mask = grouped_data["question/attention_mask"]
        positive_context_input_ids = grouped_data["positive_context/input_ids"]
        positive_context_attention_mask = grouped_data["positive_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["query_sub_batch"]

        query_input_ids_3d, query_attention_mask_3d, \
            query_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    sub_batch_size=query_sub_batch_size,
                    is_query_encoder=True
                )
        
        context_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["context_sub_batch"]

        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=positive_context_input_ids,
                    attention_mask=positive_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=negative_context_input_ids,
                    attention_mask=negative_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )

        # backward from loss to embeddings
        query_batch_size = query_input_ids.shape.as_list()[0]
        positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
        negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]
        loss, embedding_grads = self.embedding_backward_pos(
            query_embedding=query_embedding_tensor[:query_batch_size],
            positive_context_embedding=positive_context_embedding_tensor[:positive_context_batch_size],
            negative_context_embedding=negative_context_embedding_tensor[:negative_context_batch_size],
            duplicate_mask=duplicate_mask
        )

        # backward from embeddings to parameters]
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_grads = self.params_backward(
            query_input_ids_3d,
            query_attention_mask_3d,
            gradient_cache=query_embedding_grads_padded,
            is_query_encoder=True
        )

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * context_sub_batch_size - positive_context_batch_size
        positive_context_embedding_grads_padded = tf.pad(
            positive_context_embedding_grads,
            [[0, positive_context_padding], [0, 0]]
        )
        positive_context_grads = self.params_backward(
            positive_context_input_ids_3d,
            positive_context_attention_mask_3d,
            gradient_cache=positive_context_embedding_grads_padded,
            is_query_encoder=False
        )

        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * context_sub_batch_size - negative_context_batch_size
        negative_context_embedding_grads_padded = tf.pad(
            negative_context_embedding_grads,
            [[0, negative_context_padding], [0, 0]]
        )
        negative_context_grads = self.params_backward(
            negative_context_input_ids_3d,
            negative_context_attention_mask_3d,
            gradient_cache=negative_context_embedding_grads_padded,
            is_query_encoder=False
        )

        context_grads = [positive_grad + negative_grad \
            for positive_grad, negative_grad in zip(positive_context_grads, negative_context_grads)]
        grads = query_grads + context_grads # concatenate list

        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }
    
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
                type="pos",
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
    
    def poshard_step_fn_gc(self, item):
        query_input_ids = item["question/input_ids"]
        query_attention_mask = item["question/attention_mask"]
        positive_context_input_ids = item["positive_context/input_ids"]
        positive_context_attention_mask = item["positive_context/attention_mask"]
        hardneg_context_input_ids = item["hardneg_context/input_ids"]
        hardneg_context_attention_mask = item["hardneg_context/attention_mask"]
        hardneg_mask = item["hardneg_mask"]

        hardneg_context_input_ids = tf.reshape(
            hardneg_context_input_ids, [-1, self.config.pipeline_config["max_context_length"]])
        hardneg_context_attention_mask = tf.reshape(
            hardneg_context_attention_mask, [-1, self.config.pipeline_config["max_context_length"]])
        
        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["query_sub_batch"]

        query_input_ids_3d, query_attention_mask_3d, \
            query_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    sub_batch_size=query_sub_batch_size,
                    is_query_encoder=True
                )

        context_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["context_sub_batch"]

        positive_context_input_ids_3d, positive_context_attention_mask_3d, \
            positive_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=positive_context_input_ids,
                    attention_mask=positive_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )
        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=hardneg_context_input_ids,
                    attention_mask=hardneg_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )
        hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]
        hardneg_context_embedding_tensor_3d = tf.reshape(
            hardneg_context_embedding_tensor[:hardneg_context_batch_size],
            [
                self.config.pipeline_config["forward_batch_size_pos_hardneg"],
                self.config.pipeline_config["contrastive_size_pos_hardneg"],
                -1
            ]
        )

        # backward from loss to embeddings
        query_batch_size = query_input_ids.shape.as_list()[0]
        positive_context_batch_size = positive_context_input_ids.shape.as_list()[0]
        loss, embedding_grads = self.embedding_backward_poshard(
            query_embedding=query_embedding_tensor[:query_batch_size],
            positive_context_embedding=positive_context_embedding_tensor[:positive_context_batch_size],
            hardneg_context_embedding=hardneg_context_embedding_tensor_3d,
            hardneg_mask=hardneg_mask
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads, [[0, query_padding], [0, 0]]
        )
        query_grads = self.params_backward(
            query_input_ids_3d,
            query_attention_mask_3d,
            gradient_cache=query_embedding_grads_padded,
            is_query_encoder=True
        )

        positive_context_embedding_grads = embedding_grads["positive_context"]
        positive_context_multiplier = (positive_context_batch_size - 1) // context_sub_batch_size + 1
        positive_context_padding = positive_context_multiplier * context_sub_batch_size - positive_context_batch_size
        positive_context_embedding_grads_padded = tf.pad(
            positive_context_embedding_grads,
            [[0, positive_context_padding], [0, 0]]
        )
        positive_context_grads = self.params_backward(
            positive_context_input_ids_3d,
            positive_context_attention_mask_3d,
            gradient_cache=positive_context_embedding_grads_padded,
            is_query_encoder=False
        )

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * context_sub_batch_size - hardneg_context_batch_size
        hardneg_context_embedding_grads_2d = tf.reshape(
            hardneg_context_embedding_grads,
            [hardneg_context_batch_size, -1]
        )
        hardneg_context_embedding_grads_padded = tf.pad(
            hardneg_context_embedding_grads_2d,
            [[0, hardneg_context_padding], [0, 0]]
        )
        hardneg_context_grads = self.params_backward(
            hardneg_context_input_ids_3d,
            hardneg_context_attention_mask_3d,
            gradient_cache=hardneg_context_embedding_grads_padded,
            is_query_encoder=False
        )

        context_grads = [
            positive_grad + hardneg_grad
            for positive_grad, hardneg_grad 
            in zip(positive_context_grads, hardneg_context_grads)
        ]
        grads = query_grads + context_grads # concatenate list

        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }

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
                type="poshard"
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
    
    def hard_step_fn_gc(self, item):
        grouped_data = item["grouped_data"]
        negative_samples = item["negative_samples"]

        query_input_ids = grouped_data["question/input_ids"]
        query_attention_mask = grouped_data["question/attention_mask"]
        hardneg_context_input_ids = grouped_data["hardneg_context/input_ids"]
        hardneg_context_attention_mask = grouped_data["hardneg_context/attention_mask"]
        negative_context_input_ids = negative_samples["negative_context/input_ids"]
        negative_context_attention_mask = negative_samples["negative_context/attention_mask"]
        duplicate_mask = item["duplicate_mask"]

        # no tracking gradient forward
        query_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["query_sub_batch"]

        query_input_ids_3d, query_attention_mask_3d, \
            query_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask,
                    sub_batch_size=query_sub_batch_size,
                    is_query_encoder=True
                )

        context_sub_batch_size = \
            self.config.pipeline_config["gradient_cache_config"]["context_sub_batch"]

        hardneg_context_input_ids_3d, hardneg_context_attention_mask_3d, \
            hardneg_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=hardneg_context_input_ids,
                    attention_mask=hardneg_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )
        negative_context_input_ids_3d, negative_context_attention_mask_3d, \
            negative_context_embedding_tensor = \
                self.get_batch_embeddings(
                    input_ids=negative_context_input_ids,
                    attention_mask=negative_context_attention_mask,
                    sub_batch_size=context_sub_batch_size,
                    is_query_encoder=False
                )

        # backward from loss to embeddings
        query_batch_size = query_input_ids.shape.as_list()[0]
        hardneg_context_batch_size = hardneg_context_input_ids.shape.as_list()[0]
        negative_context_batch_size = negative_context_input_ids.shape.as_list()[0]
        loss, embedding_grads = self.embedding_backward_hard(
            query_embedding=query_embedding_tensor[:query_batch_size],
            hardneg_context_embedding=hardneg_context_embedding_tensor[:hardneg_context_batch_size],
            negative_context_embedding=negative_context_embedding_tensor[:negative_context_batch_size],
            duplicate_mask=duplicate_mask
        )

        # backward from embeddings to parameters
        query_embedding_grads = embedding_grads["query"]
        query_multiplier = (query_batch_size - 1) // query_sub_batch_size + 1
        query_padding = query_multiplier * query_sub_batch_size - query_batch_size
        query_embedding_grads_padded = tf.pad(
            query_embedding_grads,
            [[0, query_padding], [0, 0]]
        )
        query_grads = self.params_backward(
            query_input_ids_3d,
            query_attention_mask_3d,
            gradient_cache=query_embedding_grads_padded,
            is_query_encoder=True
        )

        hardneg_context_embedding_grads = embedding_grads["hardneg_context"]
        hardneg_context_multiplier = (hardneg_context_batch_size - 1) // context_sub_batch_size + 1
        hardneg_context_padding = hardneg_context_multiplier * context_sub_batch_size - hardneg_context_batch_size
        hardneg_context_embedding_grads_padded = tf.pad(
            hardneg_context_embedding_grads,
            [[0, hardneg_context_padding], [0, 0]]
        )
        hardneg_context_grads = self.params_backward(
            hardneg_context_input_ids_3d,
            hardneg_context_attention_mask_3d,
            gradient_cache=hardneg_context_embedding_grads_padded,
            is_query_encoder=False
        )

        negative_context_embedding_grads = embedding_grads["negative_context"]
        negative_context_multiplier = (negative_context_batch_size - 1) // context_sub_batch_size + 1
        negative_context_padding = negative_context_multiplier * context_sub_batch_size - negative_context_batch_size
        negative_context_embedding_grads_padded = tf.pad(
            negative_context_embedding_grads,
            [[0, negative_context_padding], [0, 0]]
        )
        negative_context_grads = self.params_backward(
            negative_context_input_ids_3d,
            negative_context_attention_mask_3d,
            gradient_cache=negative_context_embedding_grads_padded,
            is_query_encoder=False
        )

        context_grads = [hardneg_grad + negative_grad \
            for hardneg_grad, negative_grad in zip(hardneg_context_grads, negative_context_grads)]
        grads = query_grads + context_grads # concatenate list

        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }
    
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
                type="hard",
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

    @tf.function
    def no_tracking_gradient_encoder_forward(self, inputs, is_query_encoder: bool = True):
        if is_query_encoder:
            encoder = self.dual_encoder.query_encoder
        else:
            encoder = self.dual_encoder.context_encoder

        outputs = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            training=True
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        return pooled_output

    @tf.function
    def tracking_gradient_encoder_forward(self, inputs, gradient_cache, is_query_encoder: bool = True):
        if is_query_encoder is True:
            encoder = self.dual_encoder.query_encoder
        else:
            encoder = self.dual_encoder.context_encoder

        with tf.GradientTape() as tape:
            outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
                training=True
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
        
        grads = tape.gradient(pooled_output, encoder.trainable_variables, output_gradients=gradient_cache)
        return grads

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
            inputs = {
                "input_ids": input_ids_3d[idx],
                "attention_mask": attention_mask_3d[idx]
            },
            is_query_encoder=is_query_encoder
        ) for idx in range(multiplier)]
        embedding_tensor = tf.concat(embedding, axis=0)
        return input_ids_3d, attention_mask_3d, embedding_tensor

    @tf.function    
    def get_batch_embeddings_graph(
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

        def loop_func(idx, concatenated_emb):
            emb = self.no_tracking_gradient_encoder_forward(
                inputs = {
                    "input_ids": input_ids_3d[idx],
                    "attention_mask": attention_mask_3d[idx]
                },
                is_query_encoder=is_query_encoder
            )
            return idx + 1, tf.concat([concatenated_emb, emb], axis=0)

        idx = tf.constant(0)
        encoder = self.dual_encoder.query_encoder if is_query_encoder else self.dual_encoder.context_encoder
        hidden_size = encoder.config.hidden_size
        init_embedding = tf.zeros([0, hidden_size])
        _, embedding_tensor = tf.while_loop(
            cond=lambda idx, emb: tf.less(idx, multiplier),
            body=loop_func,
            loop_vars=(idx, init_embedding),
            shape_invariants=(idx.get_shape(), tf.TensorShape([None, hidden_size]))
        )
        embedding_tensor = tf.reshape(embedding_tensor, [batch_size, hidden_size])

        return input_ids_3d, attention_mask_3d, embedding_tensor

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
                inputs = {
                    "input_ids": input_ids_3d[idx],
                    "attention_mask": attention_mask_3d[idx]
                },
                gradient_cache=gradient_cache[
                    sub_batch_size * idx : sub_batch_size * (idx + 1)
                ],
                is_query_encoder=is_query_encoder
            )
            sub_grads = [tf.convert_to_tensor(grad) for grad in sub_grads]
            grads = [grad + sub_grad for grad, sub_grad in zip(grads, sub_grads)]
        
        return grads
    
    @tf.function
    def params_backward_graph(
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

        def loop_func(idx, grads):
            sub_grads = self.tracking_gradient_encoder_forward(
                inputs = {
                    "input_ids": input_ids_3d[idx],
                    "attention_mask": attention_mask_3d[idx]
                },
                gradient_cache=gradient_cache[
                    sub_batch_size * idx : sub_batch_size * (idx + 1)
                ],
                is_query_encoder=is_query_encoder
            )
            sub_grads = [tf.convert_to_tensor(grad) for grad in sub_grads]
            grads = [grad + tf.reshape(sub_grad, grad.get_shape()) for grad, sub_grad in zip(grads, sub_grads)]
            return idx + 1, grads

        idx = tf.constant(0, dtype=tf.int32)
        init_grads = [tf.zeros_like(var) for var in encoder.trainable_variables]
        _, grads = tf.while_loop(
            cond=lambda idx, _: tf.less(idx, multiplier),
            body=loop_func,
            loop_vars=(idx, init_grads)
        )

        return grads

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
