#!/bin/bash
python dump_corpus.py \
    --corpus-path "/media/levuloi/Storage2/CIST/Data/MRC/Legal Text retrieval/legal_corpus.json" \
    --context-max-seq-length 512 \
    --architecture bert \
    --tokenizer-path /home/levuloi/models/pretrained-lm/NlpHUST/vibert4news-base-cased \
    --tfrecord-dir data/tfrecord/corpus