import json
import logging
import requests

logger = logging.getLogger(__name__)


def segment_recursive(data, segment_host, segment_keys=None):
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
            if segment_keys is None or index in segment_keys or isinstance(index, int):
                if node:
                    headers = {'Content-Type': 'application/json'}
                    payload = {'sentence': node}
                    resp = requests.post(
                        segment_host, headers=headers, data=json.dumps(payload))
                    parent_node[index] = resp.json()['sentence']
                    logger.info("Segmentation #{}".format(counter))
                    counter += 1
                else:
                    parent_node[index] = node
        else:
            continue
    return data
