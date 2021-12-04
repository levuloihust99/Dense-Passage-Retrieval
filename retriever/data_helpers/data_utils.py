import json
import tensorflow as tf
from typing import Text, Dict, List, Any, Union


def load_corpus_to_dict(corpus_path: Text):
    """Load the Legal corpus to a dictionary. An article in the corpus can be lookup using `law_id` and `article_id`"""

    with tf.io.gfile.GFile(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    corpus_restructured = {}
    for doc in corpus:
        articles_restructured = {}
        for article in doc.get('articles'):
            articles_restructured[article.get('article_id')] = {
                'title': article.get('title'),
                'text': article.get('text')
            }
        corpus_restructured[doc.get('law_id')] = articles_restructured

    return corpus_restructured


def load_corpus_to_list(corpus_path: Text) -> List[Dict[Text, Text]]:
    """Load the Legal corpus to a list. An article can be retrieved given its position in the corpus."""

    with tf.io.gfile.GFile(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    corpus_res = []
    for doc in corpus:
        for article in doc.get('articles'):
            corpus_res.append(dict(law_id=doc.get('law_id'), **article))
    return corpus_res


def tokenize_question(question: Text, tokenizer, max_seq_length: int) -> Dict[Text, List[int]]:
    """Tokenize input question and create attention mask.
    
    Args
        question (Text): The input question.
        tokenizer (Text): Used for tokenizing the input question.
        max_seq_length (int): Maximum tokens of a question.
    
    Return:
        {
            'input_ids': List[int],
            'attention_mask': List[int]
        }
    """
    tokens = tokenizer.tokenize(question)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    mask = [1] * len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = max_seq_length - len(tokens)
    token_ids = token_ids + [tokenizer.pad_token_id] * pad_length
    mask = mask + [0] * pad_length
    return {
        'input_ids': token_ids,
        'attention_mask': mask
    }


def tokenize_context(context: Dict[Text, Text], tokenizer, max_seq_length: int) -> Dict[Text, List]:
    """Tokenize input context and create attention mask.

    Args:
        context (Dict[Text, Text]): The input context, must be a dictionary with keys `title`
            and `text`. For example, {"title": "Thể thao", "text": "Lịch thi đấu vòng loại 3 World Cup"}
        tokenizer (Text): Used for tokenizing the input context.
        max_seq_length (int): Maximum tokens of a context.
    
    Return:
        {
            'input_ids': List[int],
            'attention_mask': List[int]
        }
    """
    title = context.get('title')
    text = context.get('text')
    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    tokens = title_tokens + [tokenizer.sep_token] + text_tokens
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    mask = [1] * len(tokens)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_length = max_seq_length - len(tokens)
    token_ids = token_ids + [tokenizer.pad_token_id] * pad_length
    mask = mask + [0] * pad_length

    return {
        'input_ids': token_ids,
        'attention_mask': mask
    }


def tokenize_qa(
    qa_pair: Dict[Text, Any],
    tokenizer,
    query_max_seq_length: int,
    context_max_seq_length: int
) -> Dict[Text, List]:
    questions = qa_pair['question']
    questions_tokenized = [tokenize_question(question, tokenizer, query_max_seq_length) \
        for question in questions]

    contexts = qa_pair['context']
    contexts_tokenized = [tokenize_context(context, tokenizer, context_max_seq_length) \
        for context in contexts]

    return {
        'query_input_ids': [question['input_ids'] for question in questions_tokenized],
        'query_attention_mask': [question['attention_mask'] for question in questions_tokenized],
        'context_input_ids': [context['input_ids'] for context in contexts_tokenized],
        'context_attention_mask': [context['attention_mask'] for context in contexts_tokenized]
    }
