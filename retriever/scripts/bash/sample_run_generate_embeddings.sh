#!/bin/bash
python -m scripts.python.dual_encoder.generate_embeddings \
    --config-file logs/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/config.json \
    --index-path indexes/phobert_ZL-EVN_NonLID_Dat02 \
    --corpus-path data/legal_corpus_segmented.json \
    --corpus-tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/corpus
