import tensorflow as tf
import logging
from typing import Union

from official import nlp
import official.nlp.optimization

from dual_encoder.configuration import DualEncoderConfig
from dual_encoder.modeling import DualEncoder


logger = logging.getLogger(__name__)

class DualEncoderTrainer(object):
    def __init__(
        self,
        config: DualEncoderConfig,
        dual_encoder: DualEncoder,
        optimizer: tf.keras.optimizers.Optimizer,
        dataset: Union[tf.data.Dataset, tf.distribute.DistributedDataset],
        strategy: tf.distribute.Strategy,
        ckpt_manager: tf.train.CheckpointManager,
        summary_writer: tf.summary.SummaryWriter
    ):
        self.dual_encoder = dual_encoder
        self.optimizer = optimizer
        self.dataset = dataset
        self.strategy = strategy
        self.config = config
        self.ckpt_manager = ckpt_manager
        self.summary_writer = summary_writer


    def train(self):
        data_iterator = iter(self.dataset)
        trained_steps = self.optimizer.iterations.numpy()
        logger.info("************************ Start training ************************")
        for step in range(trained_steps, self.config.num_train_steps):
            features = next(data_iterator)
            per_replica_losses = self.strategy.run(self.train_step_fn, args=(features,))
            loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            if (step + 1) % self.config.logging_steps == 0:
                logger.info("Step {:d}/{}: {}".format(step + 1, self.config.num_train_steps, loss))

                with self.summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=step)

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

            batch_size, embedding_size = query_embedding.shape.as_list()
            similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True) # batch_size x batch_size
            logits = tf.nn.log_softmax(similarity_matrix, axis=-1) # batch_size x batch_size
            ground_truth = tf.eye(batch_size)
            loss = -tf.reduce_sum(ground_truth * logits) / (self.strategy.num_replicas_in_sync * batch_size)

        grads = tape.gradient(loss, self.dual_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dual_encoder.trainable_variables))
        return loss
