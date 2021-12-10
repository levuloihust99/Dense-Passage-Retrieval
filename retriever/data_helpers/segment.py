import json
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

segment_keys = {'question', 'context', 'title', 'text'}


def segment_recursive(data):
    stack = [(None, -1, data)] # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            if index in segment_keys:
                if node:
                    parent_node[index] = ' '.join(rdrsegmenter.tokenize(node)[0])
                else:
                    parent_node[index] = node
        else:
            continue
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    with open(args.input_file, 'r') as reader:
        data = json.load(reader)
    data = segment_recursive(data)
    with open(args.output_file, 'w') as writer:
        json.dump(data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
