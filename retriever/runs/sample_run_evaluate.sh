#!/bin/bash
python evaluate.py \
    --config-file logs/bert4newsQ64C512/config.json \
    --index-path indexes/corpus61425 \
    --qa-path data/named_data/vibertQ64C512/json/test/data.json \
    --result-dir results \
    --batch-size 256 \
    --top-docs 100 \
    --debug