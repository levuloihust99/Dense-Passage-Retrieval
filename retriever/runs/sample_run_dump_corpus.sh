#!/bin/bash
python dump_corpus.py \
    --corpus-path data/named_data/phobert-base_ZL_neg_FrScr/json/train/legal_corpus_segmented.json \
    --context-max-seq-length 258 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/phobert-base_ZL_neg_FrScr/tfrecord/corpus \
    --add-law-id True