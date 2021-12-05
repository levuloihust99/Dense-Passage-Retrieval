#!/bin/bash
python train.py \
    --model-name phobert-base_ZL_hardneg_FrScr \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --data-tfrecord-dir data/named_data/phobert-base_ZL_hardneg_FrScr/tfrecord/train \
    --model-arch roberta \
    --query-max-seq-length 64 \
    --context-max-seq-length 258 \
    --train-batch-size 12 \
    --num-train-epochs 100 \
    --logging-steps 100 \
    --save-checkpoint-freq epoch \
    --use-hardneg True
