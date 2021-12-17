#!/bin/bash
python -m evn_spc.scripts.dpr.tfrecord.dump_corpus \
    --corpus-path evn_spc/data/corpus/an_toan_thong_tin/dpr/corpus_concat_siblings_with_closest_parent.json \
    --context-max-seq-length 256 \
    --architecture roberta \
    --tokenizer-path pretrained/vinai/phobert-base \
    --tfrecord-dir evn_spc/data/tfrecord/an_toan_thong_tin/concat_siblings_with_closest_parent