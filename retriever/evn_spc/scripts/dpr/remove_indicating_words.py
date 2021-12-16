"""
1. Remove words indicating heading like `Chương`, `Mục`, `Điều` and lowercase
"""
import os
import re
import json
import argparse

regex = re.compile((r"(\b[Pp][Hh][Ụụ] *[Ll][Ụụ][Cc](?: *)?(?:[0-9][A-Za-z]*)?(?: *)?(?:[:).])?)|"
         r"(\b[Cc][Hh][Ưư][Ơơ][Nn][Gg](?: *)?(?:[0-9ABCDEIVXivxabcde]+)?(?: *)?(?:[:).])?)|"
         r"(\b[Mm][Ụụ][Cc](?: *)?(?:[0-9IVXivxABCDEabcde]+)?(?: *)?(?:[:).])?)|"
         r"(\b[Đđ][Ii][Ềề][Uu](?: *)?(?:[0-9]+)?(?: *)?(?:[:).])?)|(\b(?:[0-9a-z]+) ?[).])"
))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="evn_spc_question_answer_list_format.json")
    parser.add_argument("--output-file", default="evn_spc_question_answer_lowercase_no_indicatingwords.json")
    args = parser.parse_args()

    prefix = "/home/levuloi/Data/evnspc/processed"
    with open(os.path.join(prefix, args.input_file), 'r') as reader:
        data = json.load(reader)
    for doc in data:
        contexts = doc['context']
        for context in contexts:
            context['title'] = regex.sub(" ", context['title'])
            context['text'] = regex.sub(" ", context['text'])
    for doc in data:
        doc['question'] = [q.lower().strip() for q in doc['question']]
        contexts = doc['context']
        for context in contexts:
            context['title'] = context['title'].lower().strip()
            context['text'] = context['text'].lower().strip()
    with open(os.path.join(prefix, args.output_file), 'w') as writer:
        json.dump(data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
