#!/bin/bash
python -m scripts.python.dual_encoder.dump_qa \
    --query-max-seq-length 64 \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/train \
    --load-vlsp False \
    --load-zalo True \
    --load-mailong25 False \
    --load-evn-spc True \
    --load-atd True