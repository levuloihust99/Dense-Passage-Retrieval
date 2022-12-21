import argparse
import logging

from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json as sanic_json
from datetime import datetime
from elasticsearch import Elasticsearch

from libs.utils.logging import add_color_formater

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

@app.route("/retrieve_es", methods=["POST"])
def retrieve(request):
    data = request.json
    query = data['query']
    logger.debug("\x1b[38;5;11;1mReceived query: {}\x1b[0m".format(query))

    top_docs = data.get('top_docs', 10)
    es_query = {
        "match": {
            "text": query
        }
    }
    resp = es.search(index=args.index_name, query=es_query, request_timeout=20)
    items = []
    for item in resp['hits']['hits']:
        score = item['_score']
        text = item['_source']['text']
        items.append({'score': score, 'text': text})
    return sanic_json(items)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-endpoint", default="http://localhost:9200")
    parser.add_argument("--index-name", default="kms-cmc")
    parser.add_argument("--port", type=int, default=5054)

    global args
    args = parser.parse_args()
    
    global es
    es = Elasticsearch(args.es_endpoint)


if __name__ == "__main__":
    main()
    HOST = "0.0.0.0"
    PORT = args.port
    DEBUG = False
    THREADED = False
    app.run(host=HOST, port=PORT, debug=DEBUG, workers=1)
