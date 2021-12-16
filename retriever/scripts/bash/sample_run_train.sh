#!/bin/bash
python -m scripts.python.dual_encoder.train \
    --model-name phobert_ZL_HN7_B8_NonLID_02 \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL_HN7_NonLID_02/tfrecord/train \
    --model-arch roberta \
    --query-max-seq-length 64 \
    --context-max-seq-length 256 \
    --train-batch-size 8 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
    --use-hardneg True \
    --use-stratified-loss True
