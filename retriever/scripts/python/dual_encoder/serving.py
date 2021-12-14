from sanic import Sanic
from sanic_cors import CORS
import logging
import time
import argparse
import tensorflow as tf
from vncorenlp import VnCoreNLP

from indexing.faiss_indexer import DenseFlatIndexer
from dual_encoder.configuration import DualEncoderConfig
from dual_encoder.constants import ARCHITECTURE_MAPPINGS
from utils.setup import setup_memory_growth
from utils.logging import add_color_formater

logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)
logger = logging.getLogger(__name__)

app = Sanic(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

from datetime import datetime
from sanic.response import json as sanic_json

import argparse
import json
import glob
import numpy as np


def load_query_encoder(config: DualEncoderConfig):
    logger.info("Loading query encoder...")
    start_time = time.perf_counter()

    ckpt_path = tf.train.latest_checkpoint(config.checkpoint_path)
    encoder_class = ARCHITECTURE_MAPPINGS[config.model_arch]['model_class']
    query_encoder = encoder_class.from_pretrained(config.pretrained_model_path)
    dual_encoder = tf.train.Checkpoint(query_encoder=query_encoder)
    ckpt = tf.train.Checkpoint(model=dual_encoder)
    ckpt.restore(ckpt_path).expect_partial()
    if ckpt_path:
        logger.info("Latest checkpoint restored from {}".format(ckpt_path))
    logger.info("Done loading query encoder in {}s".format(time.perf_counter() - start_time))
    return query_encoder


@app.middleware("request")
def before_request_func(request):
    request.ctx.start_time = datetime.now()


@app.middleware("response")
def after_request_func(request, response):
    # just append timings to the output response:
    print(f"Total processing time ASR: {datetime.now() - request.ctx.start_time}")
    return response


@app.route('/retrieve', methods=['POST'])
def retrieve(request):
    data = request.json
    query = data['query']
    query = ' '.join(rdrsegmenter.tokenize(query)[0])
    top_docs = data.get('top_docs', 10)
    query_tokens = tokenizer.tokenize(query)
    query_tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token]
    query_ids = [tokenizer.convert_tokens_to_ids(query_tokens)]
    outputs = query_encoder(input_ids=tf.convert_to_tensor(query_ids), return_dict=True, training=False)
    query_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    search_results = indexer.search_knn(query_embedding, top_docs=top_docs)
    metas = search_results[0][0]
    scores = search_results[0][1]
    scores = [float(score) for score in scores]
    res = [dict(**meta, score=score) for meta, score in zip(metas, scores)]
    return sanic_json(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default='logs/phobert-base_ZL_neg_FrScr/config.json')
    parser.add_argument("--index-path", default='indexes/phobert-base_ZL_neg_FrScr')
    parser.add_argument("--tokenizer-path", default='pretrained/vinai/phobert-base')
    parser.add_argument("--architecture", default='roberta', choices=['roberta', 'bert', 'distilbert'])

    global args
    args = parser.parse_args()

    setup_memory_growth()

    global query_encoder, tokenizer, indexer, rdrsegmenter
    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)
    config = DualEncoderConfig.from_json_file(args.config_file)
    query_encoder = load_query_encoder(config)
    tokenizer = ARCHITECTURE_MAPPINGS[args.architecture]['tokenizer_class'].from_pretrained(args.tokenizer_path)
    rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')


if __name__ == '__main__':
    main()
    HOST = '0.0.0.0'
    PORT = 5051
    DEBUG = True
    THREADED = False
    app.run(host=HOST, port=PORT, debug=DEBUG, workers=1)
