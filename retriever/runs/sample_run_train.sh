#!/bin/bash
python train.py \
    --model-name bert4newsQ64C512 \
    --pretrained-model-path pretrained/NlpHUST/vibert4news-base-cased \
    --data-tfrecord-dir data/named_data/vibertQ64C512/tfrecord/train \
    --model-arch bert \
    --query-max-seq-length 64 \
    --context-max-seq-length 512 \
    --train-batch-size 16 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
