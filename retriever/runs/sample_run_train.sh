#!/bin/bash
python train.py \
    --model-name bert4news_ZL_neg_FrScr \
    --pretrained-model-path pretrained/NlpHUST/vibert4news-base-cased \
    --data-tfrecord-dir data/named_data/bert4news_ZL_neg_FrScr/tfrecord/train \
    --model-arch bert \
    --query-max-seq-length 64 \
    --context-max-seq-length 512 \
    --train-batch-size 16 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
