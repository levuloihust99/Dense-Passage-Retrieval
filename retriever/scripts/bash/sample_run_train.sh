#!/bin/bash
python -m scripts.python.dual_encoder.train \
    --model-name phobert_ZL-EVN_B16_NonLID \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID/tfrecord/train \
    --model-arch roberta \
    --sim-score dot_product \
    --query-max-seq-length 64 \
    --context-max-seq-length 256 \
    --train-batch-size 16 \
    --num-train-epochs 50 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
    --use-hardneg False \
    --use-stratified-loss False
