#!/bin/bash
python -m scripts.python.dual_encoder.generate_embeddings \
    --config-file logs/dual_encoder/phobert_ZL_HN7_B8_NonLID_02/config.json \
    --index-path indexes/phobert_ZL_HN7_B8_NonLID_02 \
    --corpus-path data/legal_corpus_segmented.json \
    --corpus-tfrecord-dir data/named_data/dual_encoder/phobert_ZL_HN7_NonLID_02/tfrecord/corpus
