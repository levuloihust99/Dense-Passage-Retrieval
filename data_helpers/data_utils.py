import json
import tensorflow as tf
from typing import Text, Dict, List, Any, Union


def load_corpus_to_dict(corpus_path: Text):
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
    with tf.io.gfile.GFile(corpus_path, 'r') as reader:
        corpus = json.load(reader)
    corpus_res = []
    for doc in corpus:
        for article in doc.get('articles'):
            corpus_res.append(dict(law_id=doc.get('law_id'), **article))
    return corpus_res


def load_qa_data(qa_data_path: Text) -> List[Dict[Text, Any]]:
    with tf.io.gfile.GFile(qa_data_path, 'r') as reader:
        qa_data = json.load(reader)
    return qa_data.get('items')


def build_query_context_pairs(
    corpus: Dict[Text, Dict[Text, Text]],
    qa_data: List[Dict[Text, Any]]
) -> List[Dict[Text, Union[Text, Dict[Text, Text]]]]:
    query_context_pairs = []
    for record in qa_data:
        question = record.get('question')
        relevant_articles = record.get('relevant_articles')
        context_article = relevant_articles[0]
        context = corpus.get(context_article.get('law_id')).get(context_article.get('article_id'))
        query_context_pairs.append({
            'question': question,
            'context': context
        })
    return query_context_pairs


def tensorize_question(question: Text, tokenizer, max_seq_length: int) -> Dict[Text, List]:
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


def tensorize_context(context: Dict[Text, Text], tokenizer, max_seq_length: int) -> Dict[Text, List]:
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
