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
