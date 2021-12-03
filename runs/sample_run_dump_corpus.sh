#!/bin/bash
python dump_corpus.py \
    --corpus-path data/legal_corpus.json \
    --context-max-seq-length 512 \
    --architecture bert \
    --tokenizer-path /home/levuloi/models/pretrained-lm/NlpHUST/vibert4news-base-cased \
    --tfrecord-dir data/tfrecord/corpus