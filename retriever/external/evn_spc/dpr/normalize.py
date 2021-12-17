import unicodedata
import logging
import argparse
import json
import os

logger = logging.getLogger(__name__)


def normalize(data):
    counter = 0
    stack = [(None, -1, data)]  # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(
                list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            parent_node[index] = unicodedata.normalize('NFKC', node)
            counter += 1
            logger.info("Normalized {} times".format(counter))
        else:
            continue
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="evn_spc_question_answer_lowercase_no_indicatingwords.json")
    parser.add_argument("--output-file", default="evn_spc_question_answer_lowercase_no_indicatingwords_NFKC.json")
    args = parser.parse_args()

    prefix = "/home/levuloi/Data/evnspc/processed"

    with open(os.path.join(prefix, args.input_file), 'r') as reader:
        data = json.load(reader)
    data = normalize(data)
    with open(os.path.join(prefix, args.output_file), 'w') as writer:
        json.dump(data, writer, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()