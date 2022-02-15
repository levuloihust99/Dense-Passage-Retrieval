#!/bin/bash
python -m scripts.python.evaluate \
    --config-file logs/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/config.json \
    --index-path indexes/phobert_ZL-EVN_NonLID_Dat02 \
    --qa-path data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/json/test/test_data.json \
    --tokenizer-path pretrained/vinai/phobert-base \
    --result-dir results/phobert_ZL-EVN_NonLID_Dat02 \
    --batch-size 256 \
    --top-docs 100 \
    --debug \
    --write-out-results
