#!/bin/bash
python -m external.evn_spc.dpr.tfrecord.dump_corpus \
    --corpus-path data/evn_spc/corpus/luat_dau_thau/corpus_noitemize_lower_segmented.json \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir data/evn_spc/tfrecord/luat_dau_thau