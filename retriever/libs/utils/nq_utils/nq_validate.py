import json
import os
import pickle
import base64
import requests
import argparse
import jsonlines
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from typing import List, Tuple
from libs.utils.nq_utils.qa_validation import calculate_matches
from libs.utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def load_answers(qas_path):
    answers_list = []
    with tf.io.gfile.GFile(qas_path, "r") as fr:
        reader = jsonlines.Reader(fr)
        for idx, item in tqdm(enumerate(reader)):
            answers_list.append(item["answers"])
    return answers_list


def get_cache_query_embeddings(cache_path):
    with tf.io.gfile.GFile(cache_path, "rb") as reader:
        loader = pickle.Unpickler(reader)
        query_embeddings = loader.load()
    return query_embeddings


def query_index(query_embeddings, index_hosts, num_queries_per_request, top_docs):
    request_headers = {"Content-Type": "application/json"}
    search_results = [[] for _ in range(len(index_hosts))]

    progress_bar = tqdm(desc="Num queries", total=query_embeddings.shape[0])
    while True:
        query_embs_to_be_executed = query_embeddings[:num_queries_per_request]
        if query_embs_to_be_executed.shape[0] == 0:
            break

        query_embeddings = query_embeddings[num_queries_per_request:]
        message = base64.b64encode(pickle.dumps(query_embs_to_be_executed)).decode()
        payload = {"message": message, "top_docs": top_docs}

        for idx, index_host in enumerate(index_hosts):
            resp = requests.post("{}/search".format(index_host), data=json.dumps(payload), headers=request_headers, timeout=1000)
            try:
                output_message = resp.json()["message"]
            except Exception as e:
                logger.info("There is some exception")

            output_message = base64.b64decode(output_message.encode())
            per_host_search_results = pickle.loads(output_message)
            search_results[idx].extend(per_host_search_results)
        
        progress_bar.update(query_embs_to_be_executed.shape[0])
    
    search_results = [merge_results(items, top_docs) for items in zip(*search_results)]
    return search_results


def merge_results(items: Tuple[Tuple[List, List]], top_docs) -> Tuple[List, List]:
    metas, scores = zip(*items)
    flatten_metas = []
    for meta in metas:
        flatten_metas.extend(meta)
    flatten_scores = np.concatenate(scores, axis=0)
    compact = list(zip(flatten_metas, flatten_scores))
    compact = sorted(compact, key=lambda x: x[1], reverse=True)

    output = []
    visited = set()
    for item in compact:
        if len(output) == top_docs:
            break
        if item[0] in visited:
            continue
        else:
            visited.add(item[0])
            output.append(item)

    output = tuple(zip(*output))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--index-hosts", required=True)
    parser.add_argument("--num-queries-per-request", type=int, default=50)
    parser.add_argument("--top-docs", default=100, type=int)
    parser.add_argument("--corpus-size", type=int, default=None)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--qas-path", required=True)
    parser.add_argument("--result-dir", required=True)

    global args
    args = parser.parse_args()

    if not tf.io.gfile.exists(args.result_dir):
        tf.io.gfile.makedirs(args.result_dir)

    global query_embeddings, index_hosts
    query_embeddings = get_cache_query_embeddings(args.cache_path)
    index_hosts = args.index_hosts.split(',')

    global corpus_dict
    corpus_dict = {}
    with tf.io.gfile.GFile(args.corpus_path) as fr:
        reader = jsonlines.Reader(fr)
        for idx, item in tqdm(enumerate(reader), total=args.corpus_size):
            corpus_dict[item["article_id"]] = (item["text"], item["title"])

    search_results = query_index(query_embeddings, index_hosts, args.num_queries_per_request, args.top_docs)
    answers_list = load_answers(args.qas_path)

    stats = calculate_matches(
        all_docs=corpus_dict,
        answers=answers_list,
        closest_docs=search_results,
        workers_num=4,
        match_type="string"
    )
    top_hits = stats.top_k_hits
    top_hits = [hit / len(answers_list) * 100 for hit in top_hits]
    pretty_results = ["Top {:03d}: {}".format(idx + 1, hit) for idx, hit in enumerate(top_hits)]
    pretty_results = "\n".join(pretty_results)
    with tf.io.gfile.GFile(os.path.join(args.result_dir, "top_hits.txt"), "w") as writer:
        writer.write(pretty_results + "\n")
    with tf.io.gfile.GFile(os.path.join(args.result_dir, "match_matrix.txt"), "w") as writer:
        writer.write(json.dumps(stats.questions_doc_hit) + "\n")


if __name__ == "__main__":
    main()
