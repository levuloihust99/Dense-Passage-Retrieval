import tensorflow as tf
from typing import Dict, Text, List


def decode_feature(
    feature: Dict[Text, tf.Tensor],
    tokenizer
):
    questions = tf.RaggedTensor.from_tensor(feature['query_input_ids'], padding=tokenizer.pad_token_id).to_list()
    questions = [tokenizer.decode(q) for q in questions]
    contexts = tf.RaggedTensor.from_tensor(feature['context_input_ids'], padding=tokenizer.pad_token_id).to_list()
    contexts = [tokenizer.decode(c) for c in contexts]

    question_attention_masks = feature['query_attention_mask']
    context_attention_masks = feature['context_attention_mask']

    return {
        'questions': questions,
        'contexts': contexts,
        'question_attention_masks': question_attention_masks,
        'context_attention_masks': context_attention_masks
    }
