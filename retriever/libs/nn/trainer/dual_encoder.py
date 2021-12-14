import tensorflow as tf
import logging
from typing import Union

from official import nlp
import official.nlp.optimization

from libs.nn.configuration.dual_encoder import DualEncoderConfig
from libs.nn.modeling.dual_encoder import DualEncoder
from libs.nn.losses.dual_encoder import (
    LossCalculator,
    InBatchLoss,
    StratifiedLoss
)


logger = logging.getLogger(__name__)


class DualEncoderTrainer(object):
    def __init__(
        self,
        config: DualEncoderConfig,
        dual_encoder: DualEncoder,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_calculator: LossCalculator,
        dataset: Union[tf.data.Dataset, tf.distribute.DistributedDataset],
        strategy: tf.distribute.Strategy,
        ckpt_manager: tf.train.CheckpointManager,
        summary_writer: tf.summary.SummaryWriter
    ):
        self.dual_encoder = dual_encoder
        self.optimizer = optimizer
        self.loss_calculator = loss_calculator
        self.dataset = dataset
        self.strategy = strategy
        self.config = config
        self.ckpt_manager = ckpt_manager
        self.summary_writer = summary_writer

    def train(self):
        data_iterator = iter(self.dataset)
        trained_steps = self.optimizer.iterations.numpy()
        logger.info(
            "************************ Start training ************************")
        for step in range(trained_steps, self.config.num_train_steps):
            features = next(data_iterator)
            per_replica_losses = self.strategy.run(
                self.train_step_fn, args=(features,))
            loss_record = {
                k: self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses[k], axis=None)
                for k in per_replica_losses
            }

            if (step + 1) % self.config.logging_steps == 0:
                loss_report = ["{}: {}".format(
                    k.capitalize(), v) for k, v in loss_record.items()]
                loss_report = " -- ".join(loss_report)
                logger.info("Step {:d}/{} --- {}".format(step + 1,
                            self.config.num_train_steps, loss_report))

                with self.summary_writer.as_default():
                    for k, v in loss_record.items():
                        tf.summary.scalar(k.capitalize(), v, step=step)

            if (step + 1) % self.config.save_checkpoint_freq == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logger.info('Saving checkpoint to {}'.format(ckpt_save_path))

    @tf.function
    def train_step_fn(self, features):
        with tf.GradientTape() as tape:
            query_embedding, context_embedding = self.dual_encoder(
                query_input_ids=features['query_input_ids'],
                query_attention_mask=features['query_attention_mask'],
                context_input_ids=features['context_input_ids'],
                context_attention_mask=features['context_attention_mask'],
                training=True
            )
            loss_record = self.loss_calculator.compute(
                query_embedding, context_embedding)
            loss = loss_record['loss'] / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dual_encoder.trainable_variables))
        return {
            k: v / self.strategy.num_replicas_in_sync
            for k, v in loss_record.items()
        }
