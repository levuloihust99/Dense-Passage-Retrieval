import tensorflow as tf
tf.random.set_seed(42)
from transformers import BertTokenizer, TFBertModel

SEQ_LENGTH = 256
BATCH_SIZE = 256
SUB_BATCH  = 16
MULTIPLIER = 16


def get_batch_forward_graph(sub_batch):
    global graph_cache
    graph_key = "get_batch_forward_graph::{}".format(sub_batch)
    if graph_key in graph_cache:
        return graph_cache[graph_key]

    @tf.function
    def forward_graph(input_ids, attention_mask):
        _, hidden_size = input_ids.shape.as_list()
        input_ids_3d = tf.reshape(input_ids, [-1, SUB_BATCH, hidden_size])
        attention_mask_3d = tf.reshape(attention_mask, [-1, SUB_BATCH, hidden_size])

        def loop_func(idx, container_emb, indices):
            outputs = encoder(input_ids=input_ids_3d[idx], attention_mask=attention_mask_3d[idx], training=False, return_dict=True)
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
            updated_container_emb = tf.tensor_scatter_nd_update(container_emb, indices, pooled_output)
            updated_indices = indices + SUB_BATCH
            return idx + 1, updated_container_emb, updated_indices

        idx = 0
        container_embedding = tf.zeros([SUB_BATCH * MULTIPLIER])
        initial_indices = tf.expand_dims(tf.range(SUB_BATCH), axis=-1)
        _, embedding_tensor, _ = tf.while_loop(
            cond=lambda idx, emb, indices: tf.less(idx, MULTIPLIER),
            body=loop_func,
            loop_vars=(idx, container_embedding, initial_indices),
            maximum_iterations=MULTIPLIER
        )
        return embedding_tensor
    
    graph_cache[graph_key] = forward_graph
    return forward_graph


def main():
    global encoder
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "Hôm nay trời đẹp lắm"
    inputs = tokenizer(text, padding='max_length', max_length=SEQ_LENGTH, return_token_type_ids=False, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids_replicate = tf.tile(tf.expand_dims(input_ids, axis=0), [BATCH_SIZE, 1])
    attention_mask_replicate = tf.tile(tf.expand_dims(attention_mask, axis=0), [BATCH_SIZE, 1])
    global graph_cache
    graph_cache = {}

    batch_forward_graph = get_batch_forward_graph(16)
    embedding = batch_forward_graph(input_ids_replicate, attention_mask_replicate)
    print(embedding[:10, :10])
    print("done")


if __name__ == "__main__":
    main()
