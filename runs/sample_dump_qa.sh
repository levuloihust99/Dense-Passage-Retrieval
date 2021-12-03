#!/bin/bash
python dump_qa.py \
    --query-max-seq-length 64 \
    --context-max-seq-length 512 \
    --architecture bert \
    --tokenizer-path pretrained/NlpHUST/vibert4news-base-case \
    --tfrecord-dir data/tfrecord/train