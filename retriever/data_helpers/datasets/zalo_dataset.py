"""Data from Legal Text Retrieval Zalo AI 2021."""

import os
import json
import time
import copy
from typing import List, Dict, Text, Any
import logging

from data_helpers.data_utils import load_corpus_to_dict

logger = logging.getLogger(__name__)


def load_data(
    data_dir: Text,
    qa_file: Text,
    corpus_file: Text,
    add_law_id=False
) -> List[Dict[Text, Any]]:
    """Load data into a list of question, context pair."""

    logger.info("Load VLSP Legal Text Retrieval question-context pairs...")
    start_time = time.perf_counter()
    qa_path = os.path.join(data_dir, qa_file)
    corpus_path = os.path.join(data_dir, corpus_file)
    
    corpus_dict = load_corpus_to_dict(corpus_path)
    with open(qa_path, 'r') as reader:
        qa_data = json.load(reader)['items']

    qa_pairs = []
    for record in qa_data:
        question = record.get('question')
        relevant_articles = record.get('relevant_articles')
        contexts = []
        for article in relevant_articles:
            context = copy.deepcopy(corpus_dict[article['law_id']][article['article_id']])
            if add_law_id:
                law_id = article['law_id']
                context['title'] = law_id + " " + context['title']
            contexts.append(context)
        qa_pairs.append({
            'question': [question],
            'context': contexts
        })
    logger.info("Done loading VLSP Legal Text Retrieval question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs