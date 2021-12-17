#!/bin/bash
python -m evn_spc.scripts.dpr.tfrecord.generate_embeddings \
    --context-max-seq-length 256 \
    --model-arch roberta \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --checkpoint-dir checkpoints/dual_encoder/phobert_ZL-EVN_B16_NonLID \
    --eval-batch-size 512 \
    --index-path evn_spc/indexes/phobert_ZL-EVN_B16_NonLID/ldt/concat_with_closest_parent \
    --corpus-path evn_spc/data/corpus/luat_dau_thau/dpr/corpus_concat_with_closest_parent.json \
    --corpus-tfrecord-dir evn_spc/data/tfrecord/luat_dau_thau/concat_with_closest_parent
