import argparse
import logging
import tensorflow as tf

from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json as sanic_json
from datetime import datetime

from libs.utils.logging import add_color_formater
from libs.utils.setup import setup_memory_growth
from libs.faiss_indexer import DenseFlatIndexer
from scripts.python.nn.evaluate import load_query_encoder
from libs.constants import TOKENIZER_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)

app = Sanic(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.middleware("request")
def before_request_func(request):
    request.ctx.start_time = datetime.now()


@app.middleware("response")
def after_request_func(request, response):
    logger.info(f"Total processing time: {datetime.now() - request.ctx.start_time}")
    return response


@app.route("/retrieve", methods=["POST"])
def retrieve(request):
    data = request.json
    query = data['query']
    logger.debug("\x1b[38;5;11;1mReceived query: {}\x1b[0m".format(query))

    top_docs = data.get('top_docs', 10)
    query_tokens = tokenizer.tokenize(query)
    logger.debug("\x1b[38;5;11;1mTokenized query: {}\x1b[0m".format(query_tokens))
    query_tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token]
    query_ids = tokenizer.convert_tokens_to_ids(query_tokens)

    outputs = query_encoder(input_ids=tf.convert_to_tensor([query_ids]), return_dict=True, training=False)
    query_embedding = outputs.last_hidden_state[:, 0, :].numpy()

    retrieval_results = indexer.search_knn(query_embedding, top_docs=top_docs)
    metas = retrieval_results[0][0]
    scores = retrieval_results[0][1]
    scores = [float(score) for score in scores]
    retrieval_results = [{**meta, "score": score} for meta, score in zip(metas, scores)]

    return sanic_json(retrieval_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["bert", "roberta"], default="bert")
    parser.add_argument("--tokenizer-path", default="bert-base-uncased")
    parser.add_argument("--pretrained-model-path", default="bert-base-uncased")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--port", type=int, default=5053)

    global args
    args = parser.parse_args()

    setup_memory_growth()

    global tokenizer
    tokenizer_class = TOKENIZER_MAPPING[args.architecture]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path)

    global indexer
    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)

    global query_encoder
    query_encoder = load_query_encoder(checkpoint_path=args.checkpoint_path,
        architecture=args.architecture, pretrained_model_path=args.pretrained_model_path)


if __name__ == "__main__":
    main()
    HOST = "0.0.0.0"
    PORT = args.port
    DEBUG = False
    THREADED = False
    app.run(host=HOST, port=PORT, debug=DEBUG, workers=1)
