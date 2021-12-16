"""Data from Legal Text Retrieval Zalo AI 2021."""

import os
import json
import time
import copy
from typing import List, Dict, Text, Any, Optional
import logging

from libs.data_helpers.data_utils import load_corpus_to_dict

logger = logging.getLogger(__name__)


def load_data(
    data_dir: Text,
    qa_file: Text,
    corpus_file: Text,
    hardneg_file: Optional[Text]=None,
    add_law_id=True,
    load_hardneg=True,
    num_hardnegs=10,
    cached=True,
    cached_filename="cached_hardneg_train_data.json",
    lowercase_and_remove_indicating_words=False
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

    if load_hardneg:
        logger.info("Loading hard negative samples...")
        if cached:
            cached_file = [f for f in os.listdir(data_dir) if f.startswith('cached')]
            if cached_file:
                assert len(cached_file) == 1, "There are multiple cached files, which is confused."
                cached_file_path = os.path.join(data_dir, cached_file[0])
                with open(cached_file_path, 'r') as reader:
                    hardneg_qa_pairs = json.load(reader)
                qa_pairs = hardneg_qa_pairs
            else:
                hardneg_path = os.path.join(data_dir, hardneg_file)
                hardneg_qa_pairs =_load_hardneg(hardneg_path, qa_pairs, corpus_dict, num_hardnegs, add_law_id)
                qa_pairs = hardneg_qa_pairs
            with open(os.path.join(data_dir, cached_filename), "w") as writer:
                json.dump(qa_pairs, writer, indent=4, ensure_ascii=False)
        else:
            hardneg_path = os.path.join(data_dir, hardneg_file)
            hardneg_qa_pairs =_load_hardneg(hardneg_path, qa_pairs, corpus_dict, num_hardnegs, add_law_id)
            qa_pairs = hardneg_qa_pairs

    if lowercase_and_remove_indicating_words:
        qa_pairs = _remove_indicating_words(qa_pairs)
    logger.info("Done loading VLSP Legal Text Retrieval question-context pairs in {}s".format(time.perf_counter() - start_time))
    return qa_pairs


def _load_hardneg(
    hardneg_path: Text,
    qa_pairs: List[Dict[Text, List]],
    corpus_dict: Dict[Text, Dict[Text, Text]],
    num_hardnegs: int,
    add_law_id: bool,
):
    with open(hardneg_path, 'r') as reader:
        hardneg_data = json.load(reader)
    
    out_hardneg_data = []
    for idx, item in enumerate(hardneg_data):
        assert item['question'] == qa_pairs[idx]['question'][0]
        hardneg_articles = []
        hardneg_idx = 0
        ground_truth_texts = set([
            c['text'] for c in qa_pairs[idx]['context']
        ])
        retrieval_articles = item['relevant_articles']
        retrieval_articles_idx = 0
        while True:
            hardneg_article = copy.deepcopy(retrieval_articles[retrieval_articles_idx])
            if hardneg_article['text'] not in ground_truth_texts:
                if add_law_id:
                    title = hardneg_article['law_id'] + " " + hardneg_article['title']
                else:
                    title = hardneg_article['title']
                hardneg_articles.append({
                    'title': title,
                    'text': hardneg_article['text']
                })
                hardneg_idx += 1
            retrieval_articles_idx += 1
            if hardneg_idx == num_hardnegs or retrieval_articles_idx == len(retrieval_articles):
                break
        
        out_hardneg_data.append({
            'question': copy.deepcopy(qa_pairs[idx]['question']),
            'context': copy.deepcopy(qa_pairs[idx]['context']),
            'hardneg_context': hardneg_articles
        })
    return out_hardneg_data


def _remove_indicating_words(qa_pairs):
    output_qa_pairs = copy.deepcopy(qa_pairs)
    import re
    regex = re.compile((r"(\b[Pp][Hh][Ụụ] *[Ll][Ụụ][Cc](?: *)?(?:[0-9][A-Za-z]*)?(?: *)?(?:[:).])?)|"
                        r"(\b[Cc][Hh][Ưư][Ơơ][Nn][Gg](?: *)?[0-9IVX]+?(?: *)?(?:[:).])?)|"
                        r"(\b[Mm][Ụụ][Cc](?: *)?[0-9IVXabc]+(?: *)?(?:[:).])?)|"
                        r"(\b[Đđ][Ii][Ềề][Uu](?: *)?[0-9]+(?: *)?(?:[:).])?)|(\b[0-9a-z] ?[).])"
                    ))
    for doc in output_qa_pairs:
        doc['question'] = [q.lower() for q in doc['question']]
        contexts = doc['context']
        for context in contexts:
            context['title'] = regex.sub(" ", context['title']).lower().strip()
            context['text'] = regex.sub(" ", context['text']).lower().strip()
        
        hardneg_contexts = doc.get('hardneg_context', None)
        if hardneg_contexts:
            for context in hardneg_contexts:
                context['title'] = regex.sub(" ", context['title']).lower().strip()
                context['text'] = regex.sub(" ", context['text']).lower().strip()
        
    return output_qa_pairs