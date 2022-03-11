import tensorflow as tf


class DualEncoder(tf.keras.Model):
    def __init__(
        self,
        query_encoder: tf.keras.Model,
        context_encoder: tf.keras.Model,
        name='dual_encoder',
        **kwargs
    ):
        super(DualEncoder, self).__init__(name=name, **kwargs)
        self.query_encoder = query_encoder
        self.context_encoder = context_encoder

    def call(
        self,
        query_input_ids: tf.Tensor,
        query_attention_mask: tf.Tensor,
        context_input_ids: tf.Tensor,
        context_attention_mask: tf.Tensor,
        return_dict=True,
        **kwargs
    ):
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            return_dict=return_dict,
            **kwargs
        )

        context_outputs = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            return_dict=return_dict,
            **kwargs
        )

        if return_dict:
            query_sequence_embeddings = query_outputs.last_hidden_state
            context_sequence_embeddings = context_outputs.last_hidden_state
        else:
            query_sequence_embeddings = query_outputs[0]
            context_sequence_embeddings = context_outputs[0]

        query_embedding = query_sequence_embeddings[:, 0, :]
        context_embedding = context_sequence_embeddings[:, 0, :]
        return query_embedding, context_embedding
