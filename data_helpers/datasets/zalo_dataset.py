"""Data from Legal Text Retrieval Zalo AI 2021."""

import os
import json
import time
from typing import List, Dict, Text, Any
import logging

from data_helpers.data_utils import load_corpus_to_dict

logger = logging.getLogger(__name__)


def load_data(data_dir: Text, segmented=False) -> List[Dict[Text, Any]]:
    """Load data into a list of question, context pair."""

    logger.info("Load VLSP Legal Text Retrieval question-context pairs...")
    start_time = time.perf_counter()
    if not segmented:
        corpus_path = 'legal_corpus.json'
        qa_path = 'train_question_answer.json'
    else:
        corpus_path = 'legal_corpus_segmented.json'
        qa_path = 'train_question_answer_segmented.json'
    
    corpus_path = os.path.join(data_dir, corpus_path)
    qa_path = os.path.join(data_dir, qa_path)

    corpus_dict = load_corpus_to_dict(corpus_path)
    with open(qa_path, 'r') as reader:
        qa_data = json.load(reader)['items']

    qa_pairs = []
    for record in qa_data:
        question = record.get('question')
        relevant_articles = record.get('relevant_articles')
        contexts = []
        for article in relevant_articles:
            context = corpus_dict[article['law_id']][article['article_id']]
            contexts.append(context)
        qa_pairs.append({
            'question': [question],
            'context': contexts
        })
    logger.info("Done loading VLSP Legal Text Retrieval question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs