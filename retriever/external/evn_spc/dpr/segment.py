import json
import logging
import argparse
import requests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
URL = 'http://localhost:8080/segment'


def should_segment(key):
    if key in {'title', 'text'} or isinstance(key, int):
        return True
    return False


def segment_recursive(data):
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
            if should_segment(index):
                if node:
                    headers = {'Content-Type': 'application/json'}
                    payload = {'sentence': node}
                    resp = requests.post(
                        URL, headers=headers, data=json.dumps(payload))
                    parent_node[index] = resp.json()['sentence']
                    logger.info("Segmentation #{}".format(counter))
                    counter += 1
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
        input_data = json.load(reader)
    output_data = segment_recursive(input_data)
    with open(args.output_file, 'w') as writer:
        json.dump(output_data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
