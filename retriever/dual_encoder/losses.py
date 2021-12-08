import tensorflow as tf


class LossCalculator(object):
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        raise NotImplementedError


class InBatchLoss(LossCalculator):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True)
        logits = tf.nn.log_softmax(similarity_matrix, axis=-1)
        loss = tf.gather_nd(
            logits, tf.where(tf.eye(self.batch_size, dtype=tf.bool))
        )
        loss = -tf.reduce_sum(loss) / self.batch_size
        return {"loss": loss}


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
        return {
            "loss": total_loss,
            "pos_hardneg_loss": pos_hardneg_loss / tf.reduce_sum(pos_hardneg_weights),
            "hardneg_neg_loss": hardneg_neg_loss / tf.reduce_sum(hardneg_neg_weights)
        }


class StratifiedLoss(object):
    def __init__(self, batch_size, within_size):
        self.batch_size = batch_size
        self.within_size = within_size
    
    def compute(self, query_embedding: tf.Tensor, context_embedding: tf.Tensor):
        # restructure context_embedding
        # origin: positives (all) + hard negatives (sample 1) + hard negatives (sample 2) + ...
        # out: [positive + hard negatives] (sample 1) + [positive + hard negatives] (sample 2) + ...
        context_embedding_positive = context_embedding[:self.batch_size] # [batch_size, hidden_size]
        context_embedding_hard_negatives = context_embedding[self.batch_size:] # [batch_size * (within_size - 1), hidden_size]
        context_embedding_positive_batch = tf.expand_dims(context_embedding_positive, axis=1) # [batch_size, 1, hidden_size]
        context_embedding_hard_negatives_batch = tf.reshape(context_embedding_hard_negatives, [self.batch_size, self.within_size - 1, -1]) # [batch_size, within_size, hidden_size]
        context_embedding_expand = tf.concat([context_embedding_positive_batch, context_embedding_hard_negatives_batch], axis=1)
        context_embedding = tf.reshape(context_embedding_expand, [self.batch_size * self.within_size, -1])

        # Positive vs hard negative (of the same sample)
        query_embedding_expand = tf.expand_dims(query_embedding, axis=1)
        positive_hardneg_scores = tf.matmul(query_embedding_expand, context_embedding_expand, transpose_b=True)
        positive_hardneg_scores = tf.squeeze(positive_hardneg_scores)
        positive_hardneg_log_probs = tf.nn.log_softmax(positive_hardneg_scores, axis=-1)
        positive_hardneg_weights = tf.concat(
            [tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, self.within_size - 1])],
            axis=-1
        )
        positive_hardneg_loss = -tf.reduce_sum(positive_hardneg_weights * positive_hardneg_log_probs)

        # Hard negative vs inbatch negative
        positive_context_embedding = context_embedding_expand[:, 0, :] # take positive contexts, [batch_size, hidden_size]
        inbatch_scores = tf.matmul(query_embedding, positive_context_embedding, transpose_b=True)
        positive_inbatch_mask = tf.eye(self.batch_size, dtype=tf.bool)
        negative_inbatch_mask = ~positive_inbatch_mask
        negative_inbatch_scores = tf.reshape(tf.boolean_mask(inbatch_scores, negative_inbatch_mask), [self.batch_size, -1]) # [batch_size, batch_size - 1]
        negative_within_scores = positive_hardneg_scores[:, 1:] # [batch_size, within_size - 1]
        hardneg_inbatchneg_scores = tf.concat([
            tf.expand_dims(negative_within_scores, axis=-1),
            tf.tile(tf.expand_dims(negative_inbatch_scores, axis=1), [1, self.within_size - 1, 1])
        ], axis=-1)
        hardneg_inbatchneg_log_probs = tf.nn.log_softmax(hardneg_inbatchneg_scores, axis=-1)
        hardneg_inbatchneg_weights = tf.tile(
            tf.expand_dims(
                tf.concat([tf.ones([self.within_size - 1, 1]), tf.zeros([self.within_size - 1, self.batch_size - 1])], axis=1),
                axis=0
            ),
            [self.batch_size, 1, 1]
        )
        hardneg_inbatchneg_loss = -tf.reduce_sum(hardneg_inbatchneg_weights * hardneg_inbatchneg_log_probs)
        total_loss = (
            (positive_hardneg_loss + hardneg_inbatchneg_loss) / 
            (self.batch_size * self.within_size)
        )
        return {
            "loss": total_loss,
            "pos_hardneg_loss": positive_hardneg_loss / self.batch_size,
            "hardneg_neg_loss": hardneg_inbatchneg_loss / tf.reduce_sum(hardneg_inbatchneg_weights)
        }
