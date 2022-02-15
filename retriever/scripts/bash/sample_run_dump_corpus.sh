#!/bin/bash
python -m scripts.python.dual_encoder.dump_corpus \
    --corpus-path data/legal_corpus_segmented.json \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/named_data/dual_encoder/phobert_ZL-EVN_NonLID_Dat02/tfrecord/corpus \
    --add-law-id False
