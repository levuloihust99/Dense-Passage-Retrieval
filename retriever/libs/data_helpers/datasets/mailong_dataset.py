import os
import json
import logging
import time
from typing import List, Dict, Text, Any

import random

logger = logging.getLogger(__name__)


def load_data(data_dir: Text, segmented=False) -> List[Dict[Text, Any]]:
    """Load data into a list of question, context pair."""

    logger.info("Loading mailong25 dataset's question-context pairs...")
    start_time = time.perf_counter()

    if not segmented:
        train_v2_path = os.path.join(data_dir, 'train-v2.0.json')
        train_ir_path = os.path.join(data_dir, 'train_IR.json')
    else:
        train_v2_path = os.path.join(data_dir, 'train-v2.0_segmented.json')
        train_ir_path = os.path.join(data_dir, 'train_IR_segmented.json')

    qa_pairs_v2 = _load_train_v2(train_v2_path)
    qa_pairs_ir = _load_train_ir(train_ir_path)
    qa_pairs = qa_pairs_v2 + qa_pairs_ir
    random.shuffle(qa_pairs)

    logger.info("Done loading mailong25 dataset's question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs


def _load_train_v2(path: Text):
    with open(path, 'r') as reader:
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
            
            if questions:
                qa_pairs.append({
                    'question': questions,
                    'context': [{
                        'title': title,
                        'text': context
                    }]
                })

    return qa_pairs


def _load_train_ir(path: Text):
    with open(path, 'r') as reader:
        data = json.load(reader)
    
    qa_pairs = []
    for item in data:
        if item['label'] is False:
            continue
        question = item['question']
        title = item['title']
        text = item['text']

        qa_pairs.append({
            'question': [question],
            'context': [{
                'title': title,
                'text': text
            }]
        })
    
    return qa_pairs
