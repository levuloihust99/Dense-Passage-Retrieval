#!/bin/bash
python evaluate.py \
    --config-file logs/phobert-base_ZL_neg_FrScr/config.json \
    --index-path indexes/phobert-base_ZL_neg_FrScr \
    --qa-path data/named_data/phobert-base_ZL_neg_FrScr/json/test/test_data.json \
    --tokenizer-path pretrained/vinai/phobert-base \
    --result-dir results/phobert-base_ZL_neg_FrScr \
    --batch-size 256 \
    --top-docs 100 \
    --debug \
    --write-out-results