"""Data from VLSP Machine Reading Comprehension 2021."""

import os
import json
import time
import logging
from typing import List, Dict, Text, Any

logger = logging.getLogger(__name__)


def load_data(data_dir: Text, segmented=False) -> List[Dict[Text, Any]]:
    """Load data into a list of question, context pair."""
    
    logger.info("Loading VLSP MRC 2021 question-context pairs...")
    start_time = time.perf_counter()

    if not segmented:
        data_path = os.path.join(data_dir, 'train.json')
    else:
        data_path = os.path.join(data_dir, 'train_segmented.json')

    with open(data_path, 'r') as reader:
        data = json.load(reader)['data']
    
    qa_pairs = []
    for item in data:
        title = item['title']
        paragraphs = item['paragraphs']

        for para in paragraphs:
            context = para['context']
            qas = para['qas']
            questions = []

            for qa in qas:
                question = qa['question']
                answers = qa.get('answers', [])
                plausible_answers = qa.get('plausible_answers', [])
                all_answers = answers + plausible_answers

                if len(all_answers) > 0:
                    questions.append(question)
            
            qa_pairs.append({
                'question': questions,
                'context': [{
                    'title': title,
                    'text': context
                }]
            })
    
    logger.info("Done loading VLSP MRC 2021 question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs
