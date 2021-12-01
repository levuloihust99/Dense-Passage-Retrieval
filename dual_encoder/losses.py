import tensorflow as tf


class InBatchLoss(object):
    def __call__(
        self,
        query_embedding: tf.Tensor, # batch_size x embedding_size
        context_embedding: tf.Tensor # batch_size x embedding_size
    ):
        batch_size, embedding_size = query_embedding.shape.as_list()
        similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True) # batch_size x batch_size
        logits = tf.nn.log_softmax(similarity_matrix, axis=-1) # batch_size x batch_size
        ground_truth = tf.eye(batch_size)
        return tf.reduce_sum(ground_truth * logits)
