#!/bin/bash
python generate_embeddings.py \
    --config-file logs/phobert-base_ZL_neg_FrScr/config.json \
    --index-path indexes/phobert-base_ZL_neg_FrScr \
    --corpus-path data/named_data/phobert-base_ZL_neg_FrScr/json/train/legal_corpus_segmented.json \
    --corpus-tfrecord-dir data/named_data/phobert-base_ZL_neg_FrScr/tfrecord/corpus
