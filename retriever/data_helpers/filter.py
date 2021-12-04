import os
import json
import argparse
import copy
from typing import Text, Dict, List
import tensorflow as tf

from data_helpers.data_utils import load_corpus_to_dict
from dual_encoder.constants import ARCHITECTURE_MAPPINGS


def filter_long_contexts(
    input_path: Text,
    output_path: Text,
    corpus: Dict[Text, Dict[Text, Dict[Text, Text]]],
    tokenizer,
    context_max_seq_length: int=512
):
    with open(input_path, 'r') as reader:
        train_data = json.load(reader)
    train_data = train_data.get('items')
    filtered_train_data = []
    for qa_pair in train_data:
        qa_pair = copy.deepcopy(qa_pair)
        filtered_relevant_articles = []
        relevant_articles = qa_pair.get('relevant_articles')
        for item in relevant_articles:
            article = corpus.get(item.get('law_id')).get(item.get('article_id'))
            title = article.get('title')
            text = article.get('text')
            title_tokens = tokenizer.tokenize(title)
            text_tokens = tokenizer.tokenize(text)
            if len(title_tokens) + len(text_tokens) + 3 <= context_max_seq_length:
                filtered_relevant_articles.append(item)
        if filtered_relevant_articles:
            qa_pair['relevant_articles'] = filtered_relevant_articles
            filtered_train_data.append(qa_pair)

    filtered_train_data = {
        '_name_': 'train',
        '_count_': len(filtered_train_data),
        'items': filtered_train_data
    }
    output_dir = os.path.basename(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, 'w') as writer:
        json.dump(filtered_train_data, writer, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default='data/train_question_answer.json')
    parser.add_argument("--output-path", default='data/train_question_answer_filtered.json')
    parser.add_argument("--corpus-path", default='data/legal_corpus.json')
    parser.add_argument("--tokenizer-path", default='pretrained/NlpHUST/vibert4news-base-cased')
    parser.add_argument("--architecture", default='distilbert', choices=['bert', 'distilbert', 'roberta'])
    parser.add_argument("--context-max-seq-length", type=int, required=True)
    args = parser.parse_args()

    corpus = load_corpus_to_dict(args.corpus_path)
    tokenizer_class = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class']
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)
    filter_long_contexts(
        input_path=args.input_path,
        output_path=args.output_path,
        corpus=corpus,
        tokenizer=tokenizer,
        context_max_seq_length=args.context_max_seq_length
    )


if __name__ == "__main__":
    main()