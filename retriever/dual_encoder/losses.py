import tensorflow as tf


class LossCalculator(object):
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        raise NotImplementedError


class InBatchLoss(LossCalculator):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True) # batch_size x batch_size
        logits = tf.nn.log_softmax(similarity_matrix, axis=-1) # batch_size x batch_size
        ground_truth = tf.eye(self.batch_size)
        loss = -tf.reduce_sum(ground_truth * logits) / self.batch_size


class StratifiedLoss(LossCalculator):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        """Compute score of query with respect to positive, hard negative and negative contexts.

        Using softmax to rank positive higher than hard negatives and in turn higher than negatives.
        Args:
            query_embedding (tf.Tensor): [batch_size, hidden_size]
            context_embedding (tf.Tensor): [2 * batch_size, hidden_size]
        """
        # Compute crossentropy loss between positive and hard negative
        mask = tf.tile(tf.eye(self.batch_size, dtype=tf.bool), [2, 1])
        similarity_matrix = tf.matmul(context_embedding, query_embedding, transpose_b=True) # [2 * batch_size, batch_size]
        pos_hardneg_scores = tf.boolean_mask(similarity_matrix, mask)
        pos_hardneg_scores = tf.reshape(pos_hardneg_scores, [2, self.batch_size])
        pos_hardneg_log_probs = tf.nn.log_softmax(pos_hardneg_scores, axis=0)
        pos_hardneg_weights = tf.concat([
            tf.ones([1, self.batch_size]), tf.zeros([1, self.batch_size])
        ], axis=0)
        pos_hardneg_loss = -tf.reduce_sum(pos_hardneg_weights * pos_hardneg_log_probs)

        # Compute crossentropy loss between hard negative and inbatch negatives
        masked_indices = tf.where(tf.eye(self.batch_size, dtype=tf.bool))
        masked_values = [-1e5] * self.batch_size
        hardneg_neg_scores = tf.tensor_scatter_nd_update(
            similarity_matrix, masked_indices, masked_values
        )
        hardneg_neg_log_probs = tf.nn.log_softmax(hardneg_neg_scores, axis=0)
        hardneg_neg_weights = tf.concat([
            tf.zeros([self.batch_size, self.batch_size]),
            tf.eye(self.batch_size)
        ], axis=0)
        hardneg_neg_loss = -tf.reduce_sum(hardneg_neg_weights * hardneg_neg_log_probs)

        total_loss = (
            (pos_hardneg_loss + hardneg_neg_loss) / 
            (tf.reduce_sum(pos_hardneg_weights) + tf.reduce_sum(hardneg_neg_weights))
        )
        return total_loss
