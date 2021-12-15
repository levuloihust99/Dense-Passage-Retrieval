"""
1. Merge the processed data with manually fixed data.
"""


import json
import copy
import os


def main():
    prefix = "/home/levuloi/Data/evnspc/processed"
    with open(os.path.join(prefix, "evn_spc_question_answer.json")) as reader:
        avail_qa_data = json.load(reader)
    with open(os.path.join(prefix, "missing_parents_added.json")) as reader:
        added_qa_data = json.load(reader)
    out_qa_data = []
    for doc in avail_qa_data:
        out_qa_data.append(copy.deepcopy(doc))
    for doc in added_qa_data:
        out_qa_data.append({
            'question': doc['question'],
            'context': {
                'title': doc['title'],
                'text': doc['text_value']
            }
        })
    with open(os.path.join(prefix, "evn_spc_question_answer_full.json"), "w") as writer:
        json.dump(out_qa_data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
