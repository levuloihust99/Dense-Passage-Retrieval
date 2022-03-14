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

    def train_pos(self):
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        global_step = trained_steps
        start_time = time.perf_counter()
        while global_step < self.config.num_train_steps:
            loss = self.accumulate_step(
                iterator=self.iterators["pos_dataset"],
                step_fn=self.pos_step_fn,
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
        while global_step < self.config.num_train_steps:
            # < pos pipeline
            if (global_step + 2) % (self.config.regulate_factor + 2) != 0:
                pos_loss = self.accumulate_step(
                    iterator=self.iterators["pos_dataset"],
                    step_fn=self.pos_step_fn,
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
                    step_fn=self.poshard_step_fn,
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
                    step_fn=self.hard_step_fn,
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
        while global_step < self.config.num_train_steps:
            # < pos pipeline
            if (global_step + 1) % (self.config.regulate_factor + 1) != 0:
                pos_loss = self.accumulate_step(
                    iterator=self.iterators["pos_dataset"],
                    step_fn=self.pos_step_fn,
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
                    step_fn=self.poshard_step_fn,
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
        while global_step < self.config.num_train_steps:
            inbatch_loss = self.accumulate_step(
                iterator=self.iterators["inbatch_dataset"],
                step_fn=self.inbatch_step_fn,
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
            grads = self.strategy.experimental_local_results(
                per_replica_results["grads"])[0]
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
                type="pos"
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
                type="hard"
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
            )
            loss = loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        ctx = ctx = tf.distribute.get_replica_context()
        return {
            "loss": loss,
            "grads": ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
        }
