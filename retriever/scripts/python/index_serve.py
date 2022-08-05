import base64
import argparse
import logging
import pickle
import numpy as np
from datetime import datetime

from sanic import Sanic
from sanic.response import json as sanic_json
from sanic_cors import CORS

from libs.faiss_indexer import DenseFlatIndexer
from libs.utils.logging import add_color_formater

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
add_color_formater(logging.root)

app = Sanic(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.middleware("request")
def before_request_func(request):
    request.ctx.start_time = datetime.now()


@app.middleware("response")
def after_request_func(request, response):
    # just append timings to the output response:
    print(f"Total processing time: {datetime.now() - request.ctx.start_time}")
    return response


@app.route('/search', methods=['POST'])
def serve(request):
    top_docs = request.json["top_docs"]
    message = request.json["message"]
    message = base64.b64decode(message.encode())
    query_embedding = pickle.loads(message)
    search_results = indexer.search_knn(query_embedding, top_docs)
    output_message = pickle.dumps(search_results)
    output_message = base64.b64encode(output_message).decode()
    return sanic_json({"message": output_message})

@app.route('/health', methods=['GET'])
def health(request):
    return sanic_json({"success": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--port", default='5555')

    global args
    args = parser.parse_args()

    global indexer
    indexer = DenseFlatIndexer()
    indexer.deserialize(args.index_path)


if __name__ == "__main__":
    main()
    HOST = '0.0.0.0'
    DEBUG = False
    THREADED = False
    app.run(host=HOST, port=args.port, debug=DEBUG, workers=1)
