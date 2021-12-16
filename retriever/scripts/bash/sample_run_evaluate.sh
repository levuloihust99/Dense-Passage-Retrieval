#!/bin/bash
python -m scripts.python.evaluate \
    --config-file logs/dual_encoder/phobert_ZL_HN7_B8_LID_02/config.json \
    --index-path indexes/phobert_ZL_HN7_B8_LID_02 \
    --qa-path data/named_data/dual_encoder/phobert_ZL_HN7_LID_02/json/test/test_data.json \
    --tokenizer-path pretrained/vinai/phobert-base \
    --result-dir results/phobert_ZL_HN7_B8_LID_02 \
    --batch-size 256 \
    --top-docs 1 \
    --debug \
    --write-out-results
