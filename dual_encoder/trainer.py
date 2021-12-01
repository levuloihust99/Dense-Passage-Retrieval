import tensorflow as tf
import logging


logger = logging.getLogger(__name__)

class DualEncoderTrainer(object):
    def __init__(
        self,
        config,
        dual_encoder,
        optimizer,
        dataset,
        strategy,
        ckpt_manager,
        metrics,
        summary_writer
    ):
        self.dual_encoder = dual_encoder
        self.optimizer = optimizer
        self.dataset = dataset
        self.strategy = strategy
        self.config = config
        self.ckpt_manager = ckpt_manager
        self.metrics = metrics
        self.summary_writer = summary_writer


    def train(self):
        data_iterator = iter(self.dataset)
        trained_steps = self.optimizer.iterations.numpy()
        logger.info("************************ Start training ************************")
        for step in range(trained_steps, self.config.num_train_steps):
            features = next(data_iterator)
            per_replica_losses = self.strategy.run(self.train_step_fn, args=(features,))
            loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            loss = loss / (self.strategy.num_replicas_in_sync * self.config.batch_size)

            if (step + 1) % self.config.logging_steps == 0:
                logger.info("Step {d}/{}: {}".format(step + 1, self.config.num_train_steps, loss))

                with self.summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=step)

            if (step + 1) % self.config.save_checkpoints_steps == 0:
                ckpt_save_path = self.ckpt_manager.save()
                logger.info('Saving checkpoint to {}'.format(ckpt_save_path))

    @tf.function
    def train_step_fn(self, inputs):
        inputs = mask_with_strategy(
            config=self.config,
            inputs=inputs,
            mask_prob=self.config.mask_prob,
            max_masked_positions=self.config.max_masked_positions
        )

        with tf.GradientTape() as tape:
            outputs = self.distil_student(
                input_ids=inputs.input_ids,
                attention_mask=inputs.input_mask,
                masked_lm_ids=inputs.masked_lm_ids,
                masked_lm_positions=inputs.masked_lm_positions,
                masked_lm_weights=inputs.masked_lm_weights,
                training=True
            )
            loss = outputs.total_loss / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.distil_student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.distil_student.trainable_variables))
        return {
            'total_loss': loss,
            'cosine_loss': (outputs.cosine_loss / self.strategy.num_replicas_in_sync
                            if self._should_calculate_cosine_loss else None),
            'distil_loss': outputs.distil_loss / self.strategy.num_replicas_in_sync,
            'masked_lm_loss': outputs.masked_lm_loss / self.strategy.num_replicas_in_sync
        }

    def eval(self):
        data_iterator = iter(self.dataset)
        logger.info("************************ Start evaluating ************************")
        for step in range(self.config.num_eval_steps):
            features = next(data_iterator)
            inputs = features_to_inputs(features)
            self.strategy.run(self.eval_step, args=(inputs,))

        eval_results = {}
        for metric_name, metric in self.metrics.items():
            eval_results[metric_name] = float(metric.result().numpy())
        eval_json_string = "\n"
        for metric_name, metric_result in eval_results.items():
            eval_json_string += f'\t{metric_name} = {metric_result}\n'

        logger.info("************************ Evaluation results ************************")
        logger.info(eval_json_string)


    @tf.function
    def eval_step(self, inputs):
        inputs = dynamic_masking(
            config=self.config,
            inputs=inputs,
            mask_prob=self.config.mask_prob,
            max_masked_positions=self.config.max_masked_positions
        )

        outputs = self.distil_student(
            input_ids=inputs.input_ids,
            attention_mask=inputs.input_mask,
            masked_lm_ids=inputs.masked_lm_ids,
            masked_lm_positions=inputs.masked_lm_positions,
            masked_lm_weights=inputs.masked_lm_weights,
            training=False
        )

        pred_probs = outputs.student_probs # B x L x vocab_size
        B, L, vocab_size = pred_probs.shape.as_list()
        shift = tf.expand_dims(L * tf.range(B), -1) # B x 1
        flat_positions = tf.reshape(inputs.masked_lm_positions + shift, [-1, 1]) # [B * max_masked_positions]
        flat_probs = tf.reshape(pred_probs, [B * L, vocab_size])
        pred_masked_probs = tf.gather_nd(flat_probs, flat_positions)
        pred_masked_probs = tf.reshape(pred_masked_probs, [B, -1, vocab_size])
        pred_masked_ids = tf.argmax(pred_masked_probs, axis=-1)

        for metric in self.metrics.values():
            metric.update_state(
                y_true=inputs.masked_lm_ids,
                y_pred=pred_masked_ids,
                sample_weight=inputs.masked_lm_weights
            )
