#!/bin/bash
python train.py \
    --model-name vibert \
    --model-arch bert \
    --query-max-seq-length 64 \
    --context-max-seq-length 512 \
    --num-train-epochs 50 \
    --train-batch-size 16 \
    --tokenizer-path pretrained/NlpHUST/vibert4news-base-cased \
    --pretrained-model-path pretrained/NlpHUST/vibert4news-base-cased