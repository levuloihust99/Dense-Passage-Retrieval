"""
1. Reading EXCEL files and turn into json files.
2. Filter out wrong docs for later processing.
"""


import argparse
import os
import pandas as pd
import json
from typing import Text, List


def get_qa_pairs(xlsx_filename: Text, json_filename: Text):
    with open(json_filename, 'r') as reader:
        json_corpus = json.load(reader)['document']
    df = pd.read_excel(xlsx_filename)
    df = df.dropna()

    qa_pairs = []
    missing_parents = []
    for _, row in df.iterrows():
        text_value = row.text_value
        question = row.question
        parent_index = row.parent_index
        if parent_index == '0.0.0.0.0.0.0.0.0.0':
            parents = [{'textValue': ''}]
            continue
        parents = []
        for doc in json_corpus:
            if doc['paraIndex'] == parent_index:
                parents.append(doc)
        assert len(parents) <= 1
        if len(parents) == 0:
            missing_parents.append({
                'file_id': row.file_id,
                'file_name': row.file_name,
                'para_index': row.para_index,
                'parent_index': row.parent_index,
                'text_value': text_value,
                'question': row.question
            })
        else:
            title = parents[0]['textValue']
            qa_pairs.append({
                'question': question,
                'context': {
                    'title': title,
                    'text': text_value
                }
            })
    return qa_pairs, missing_parents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="evn_spc/scripts/annotated_data_config.json")
    args = parser.parse_args()

    with open(args.data_config, 'r') as reader:
        data_config = json.load(reader)
    qa_pairs = []
    missing_parents = []
    for item in data_config:
        res = get_qa_pairs(
            xlsx_filename=item['xlsx'],
            json_filename=item['json']
        )
        qa_pairs.extend(res[0])
        missing_parents.extend(res[1])

    if missing_parents:
        with open("/home/levuloi/Data/evnspc/processed/missing_parents_added.json", "w") as writer:
            json.dump(missing_parents, writer, indent=4, ensure_ascii=False)
    with open("/home/levuloi/Data/evnspc/processed/evn_spc_question_answer.json", "w") as writer:
        json.dump(qa_pairs, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
