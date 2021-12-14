import tensorflow as tf


class LossCalculator(object):
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        raise NotImplementedError


class InBatchLoss(LossCalculator):
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        batch_size, hidden_size = query_embedding.shape.as_list()
        similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True)
        logits = tf.nn.log_softmax(similarity_matrix, axis=-1)
        loss = tf.gather_nd(
            logits, tf.where(tf.eye(batch_size, dtype=tf.bool))
        )
        loss = -tf.reduce_sum(loss) / batch_size
        return {"loss": loss}


class StratifiedLoss(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        batch_size, hidden_size = query_embedding.shape.as_list()
        within_size = context_embedding.shape[0] // batch_size
        # restructure context_embedding
        # origin: positives (all) + hard negatives (sample 1) + hard negatives (sample 2) + ...
        # out: [positive + hard negatives] (sample 1) + [positive + hard negatives] (sample 2) + ...
        context_embedding_positive = context_embedding[:batch_size] # [batch_size, hidden_size]
        context_embedding_hard_negatives = context_embedding[batch_size:] # [batch_size * (within_size - 1), hidden_size]
        context_embedding_positive_batch = tf.expand_dims(context_embedding_positive, axis=1) # [batch_size, 1, hidden_size]
        context_embedding_hard_negatives_batch = tf.reshape(context_embedding_hard_negatives, [batch_size, within_size - 1, -1]) # [batch_size, within_size, hidden_size]
        context_embedding_expand = tf.concat([context_embedding_positive_batch, context_embedding_hard_negatives_batch], axis=1)
        context_embedding = tf.reshape(context_embedding_expand, [batch_size * within_size, -1])

        # Positive vs hard negative (of the same sample)
        query_embedding_expand = tf.expand_dims(query_embedding, axis=1)
        positive_hardneg_scores = tf.matmul(query_embedding_expand, context_embedding_expand, transpose_b=True)
        positive_hardneg_scores = tf.squeeze(positive_hardneg_scores)
        positive_hardneg_log_probs = tf.nn.log_softmax(positive_hardneg_scores, axis=-1)
        positive_hardneg_weights = tf.concat(
            [tf.ones([batch_size, 1]), tf.zeros([batch_size, within_size - 1])],
            axis=-1
        )
        positive_hardneg_loss = -tf.reduce_sum(positive_hardneg_weights * positive_hardneg_log_probs)

        # Hard negative vs inbatch negative
        positive_context_embedding = context_embedding_expand[:, 0, :] # take positive contexts, [batch_size, hidden_size]
        inbatch_scores = tf.matmul(query_embedding, positive_context_embedding, transpose_b=True)
        positive_inbatch_mask = tf.eye(batch_size, dtype=tf.bool)
        negative_inbatch_mask = ~positive_inbatch_mask
        negative_inbatch_scores = tf.reshape(tf.boolean_mask(inbatch_scores, negative_inbatch_mask), [batch_size, -1]) # [batch_size, batch_size - 1]
        negative_within_scores = positive_hardneg_scores[:, 1:] # [batch_size, within_size - 1]
        hardneg_inbatchneg_scores = tf.concat([
            tf.expand_dims(negative_within_scores, axis=-1),
            tf.tile(tf.expand_dims(negative_inbatch_scores, axis=1), [1, within_size - 1, 1])
        ], axis=-1)
        hardneg_inbatchneg_log_probs = tf.nn.log_softmax(hardneg_inbatchneg_scores, axis=-1)
        hardneg_inbatchneg_weights = tf.tile(
            tf.expand_dims(
                tf.concat([tf.ones([within_size - 1, 1]), tf.zeros([within_size - 1, batch_size - 1])], axis=1),
                axis=0
            ),
            [batch_size, 1, 1]
        )
        hardneg_inbatchneg_loss = -tf.reduce_sum(hardneg_inbatchneg_weights * hardneg_inbatchneg_log_probs)
        total_loss = (
            (positive_hardneg_loss + hardneg_inbatchneg_loss) / 
            (batch_size * within_size)
        )
        return {
            "loss": total_loss,
            "pos_hardneg_loss": positive_hardneg_loss / batch_size,
            "hardneg_neg_loss": hardneg_inbatchneg_loss / (batch_size * (within_size - 1))
        }
