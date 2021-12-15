"""
1. Split questions + merge contexts and questions that connect together
"""
import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="evn_spc_question_answer_full.json")
    parser.add_argument("--output-file", default="evn_spc_question_answer_list_format.json")
    args = parser.parse_args()

    prefix = "/home/levuloi/Data/evnspc/processed"
    with open(os.path.join(prefix, args.input_file), 'r') as reader:
        input_qa_data = json.load(reader)
    
    out_qa_data = []
    is_visited = {}
    idx = 0
    for doc in input_qa_data:
        question = doc['question']
        if question not in is_visited:
            is_visited[question] = idx
            out_qa_data.append({
                'question': question,
                'context': [doc['context']]
            })
            idx += 1
        else:
            out_qa_data[is_visited[question]]['context'].append(doc['context'])
    for doc in out_qa_data:
        question = doc['question']
        questions = question.split('\n')
        questions = [q for q in questions if q]
        doc['question'] = questions
    
    with open(os.path.join(prefix, args.output_file), 'w') as writer:
        json.dump(out_qa_data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
