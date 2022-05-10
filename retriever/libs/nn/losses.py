from typing import Dict, Literal, Text, Optional
import tensorflow as tf


class LossCalculator(object):
    def compute(
        self,
        inputs: Dict[Text, tf.Tensor],
        sim_func: Literal["cosine", "dot_product"],
        type: Literal["pos", "hard", "poshard"],
        duplicate_mask: Optional[tf.Tensor] = None,
        **kwargs
    ):
        if sim_func == "cosine":
            inputs = {k: self.normalize(v) for k, v in inputs.items()}
        if type == "pos":
            return self.compute_contrastive_neg(
                query_embedding=inputs["query_embedding"],
                context_embedding=inputs["positive_context_embedding"],
                negative_context_embedding=inputs["negative_context_embedding"],
                duplicate_mask=duplicate_mask,
            )
        elif type == "hard":
            return self.compute_contrastive_neg(
                query_embedding=inputs["query_embedding"],
                context_embedding=inputs["hardneg_context_embedding"],
                negative_context_embedding=inputs["negative_context_embedding"],
                duplicate_mask=duplicate_mask
            )
        elif type == "poshard":
            return self.compute_contrastive_hardneg(
                query_embedding=inputs["query_embedding"],
                positive_context_embedding=inputs["positive_context_embedding"],
                hardneg_context_embedding=inputs["hardneg_context_embedding"],
                hardneg_mask=inputs["hardneg_mask"],
            )
        elif type == "inbatch":
            return self.compute_inbatch(
                query_embedding=inputs["query_embedding"],
                context_embedding=inputs["context_embedding"],
                duplicate_mask=duplicate_mask,
                hardneg_mask=inputs["hardneg_mask"]
            )
        else:
            raise Exception("Type '{}' is not supported.".format(type))
    
    def normalize(self, tensor: tf.Tensor):
        return tensor / tf.norm(tensor, axis=-1, keepdims=True)

    def compute_contrastive_neg(
        self,
        query_embedding: tf.Tensor,
        context_embedding: tf.Tensor,
        negative_context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor
    ):
        batch_size, hidden_size = query_embedding.shape.as_list()
        positive_sim_scores = tf.reduce_sum(query_embedding * context_embedding, axis=-1, keepdims=True)
        negative_sim_scores = tf.matmul(query_embedding, negative_context_embedding, transpose_b=True)
        sim_matrix = tf.concat(
            [positive_sim_scores, negative_sim_scores],
            axis=-1
        )
        sim_matrix_masked = tf.where(
            tf.cast(duplicate_mask, dtype=tf.bool),
            sim_matrix, -1e9
        )
        logits = tf.nn.log_softmax(sim_matrix_masked, axis=-1)
        loss = logits[:, 0]
        loss = -tf.reduce_sum(loss) / batch_size
        return loss

    def compute_contrastive_hardneg(
        self,
        query_embedding: tf.Tensor,
        positive_context_embedding: tf.Tensor,
        hardneg_context_embedding: tf.Tensor,
        hardneg_mask: tf.Tensor,
    ):
        batch_size, hidden_size = query_embedding.shape.as_list()
        positive_sim_scores = tf.reduce_sum(query_embedding * positive_context_embedding, axis=-1, keepdims=True)
        hardneg_sim_scores = tf.reduce_sum(tf.expand_dims(query_embedding, axis=1) * hardneg_context_embedding, axis=-1)
        sim_matrix = tf.concat(
            [positive_sim_scores, hardneg_sim_scores],
            axis=-1
        )
        mask_matrix = tf.fill(tf.shape(sim_matrix), -1e9)
        hardneg_mask_padded = tf.pad(hardneg_mask, paddings=[[0, 0], [1, 0]], constant_values=1)
        sim_matrix_masked = tf.where(
            tf.cast(hardneg_mask_padded, dtype=tf.bool),
            sim_matrix, mask_matrix
        )
        logits = tf.nn.log_softmax(sim_matrix_masked, axis=-1)
        loss = logits[:, 0]
        loss = -tf.reduce_sum(loss) / batch_size
        return loss

    def compute_inbatch(
        self,
        query_embedding: tf.Tensor,
        context_embedding: tf.Tensor,
        duplicate_mask: tf.Tensor,
        hardneg_mask: Optional[tf.Tensor] = None
    ):
        batch_size, hidden_size = query_embedding.shape.as_list()
        similarity_matrix = tf.matmul(query_embedding, context_embedding, transpose_b=True)
        if hardneg_mask is not None:
            hardneg_mask_replicate = tf.tile(tf.expand_dims(tf.reshape(
                hardneg_mask, [-1]), axis=0), multiples=[batch_size, 1])
            mask = tf.concat([duplicate_mask, hardneg_mask_replicate], axis=1)
        else:
            mask = duplicate_mask
        similarity_matrix_masked = tf.where(
            tf.cast(mask, dtype=tf.bool),
            similarity_matrix, -1e9
        )
        logits = tf.nn.log_softmax(similarity_matrix_masked, axis=-1)
        loss = tf.gather_nd(
            logits, tf.where(tf.eye(batch_size, dtype=tf.bool))
        )
        loss = -tf.reduce_sum(loss) / batch_size
        return loss
