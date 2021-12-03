#!/bin/bash
python train.py \
    --model-name phobert-base \
    --model-arch roberta \
    --query-max-seq-length 64 \
    --context-max-seq-length 258 \
    --num-train-epochs 50 \
    --train-batch-size 16 \
    --tokenizer-path pretrained/vinai/phobert-base \
    --pretrained-model-path pretrained/vinai/phobert-base