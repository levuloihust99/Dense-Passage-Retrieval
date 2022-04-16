from typing import Dict, Literal, Text
import tensorflow as tf


class LossCalculator(object):
    def compute(
        self,
        inputs: Dict[Text, tf.Tensor],
        sim_func: Literal["cosine", "dot_product"],
        type: Literal["pos", "hard", "poshard"],
        padding_mask: tf.Tensor
    ):
        if sim_func == "cosine":
            inputs = {k: self.normalize(v) for k, v in inputs.items()}
        if type == "pos":
            return self.compute_contrastive_neg(
                query_embedding=inputs["query_embedding"],
                context_embedding=inputs["positive_context_embedding"],
                negative_context_embedding=inputs["negative_context_embedding"],
                padding_mask=padding_mask
            )
        elif type == "hard":
            return self.compute_contrastive_neg(
                query_embedding=inputs["query_embedding"],
                context_embedding=inputs["hardneg_context_embedding"],
                negative_context_embedding=inputs["negative_context_embedding"],
                padding_mask=padding_mask
            )
        elif type == "poshard":
            return self.compute_contrastive_hardneg(
                query_embedding=inputs["query_embedding"],
                positive_context_embedding=inputs["positive_context_embedding"],
                hardneg_context_embedding=inputs["hardneg_context_embedding"],
                hardneg_mask=inputs["hardneg_mask"],
                padding_mask=padding_mask
            )
        elif type == "inbatch":
            return self.compute_inbatch(
                query_embedding=inputs["query_embedding"],
                positive_context_embedding=inputs["positive_context_embedding"],
                padding_mask=padding_mask
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
        padding_mask: tf.Tensor
    ):
        positive_sim_scores = tf.reduce_sum(query_embedding * context_embedding, axis=-1, keepdims=True)
        negative_sim_scores = tf.matmul(query_embedding, negative_context_embedding, transpose_b=True)
        sim_matrix = tf.concat(
            [positive_sim_scores, negative_sim_scores],
            axis=-1
        )
        logits = tf.nn.log_softmax(sim_matrix, axis=-1)
        loss = logits[:, 0] * padding_mask
        loss = -tf.reduce_sum(loss) / tf.reduce_sum(padding_mask)
        return loss

    def compute_contrastive_hardneg(
        self,
        query_embedding: tf.Tensor,
        positive_context_embedding: tf.Tensor,
        hardneg_context_embedding: tf.Tensor,
        hardneg_mask: tf.Tensor,
        padding_mask: tf.Tensor
    ):
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
        loss = logits[:, 0] * padding_mask
        loss = -tf.reduce_sum(loss) / tf.reduce_sum(padding_mask)
        return loss

    def compute_inbatch(
        self,
        query_embedding: tf.Tensor,
        positive_context_embedding: tf.Tensor,
        padding_mask: tf.Tensor
    ):
        batch_size, hidden_size = query_embedding.shape.as_list()
        similarity_matrix = tf.matmul(query_embedding, positive_context_embedding, transpose_b=True)
        logits = tf.nn.log_softmax(similarity_matrix, axis=-1)
        loss = tf.gather_nd(
            logits, tf.where(tf.eye(batch_size, dtype=tf.bool))
        )
        loss = loss * padding_mask
        loss = -tf.reduce_sum(loss) / tf.reduce_sum(padding_mask)
        return loss
