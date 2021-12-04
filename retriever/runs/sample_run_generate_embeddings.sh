#!/bin/bash
python generate_embeddings.py \
    --config-file logs/bert4newsQ64C512/config.json \
    --index-path indexes/corpus61425 \
    --corpus-path data/legal_corpus.json \
    --corpus-tfrecord-dir data/named_data/vibertQ64C512/tfrecord/corpus
