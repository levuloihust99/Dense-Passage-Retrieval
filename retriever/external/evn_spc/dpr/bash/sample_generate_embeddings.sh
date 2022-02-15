#!/bin/bash
python -m external.evn_spc.dpr.tfrecord.generate_embeddings \
    --context-max-seq-length 256 \
    --model-arch roberta \
    --pretrained-model-path pretrained/vinai/phobert-base \
    --checkpoint-dir checkpoints/dual_encoder/phobert_ZL-EVN_B16_NonLID_Dat02 \
    --eval-batch-size 4 \
    --index-path indexes/evn_spc/luat_dau_thau \
    --corpus-path data/evn_spc/corpus/luat_dau_thau/corpus.json \
    --corpus-tfrecord-dir data/evn_spc/tfrecord/luat_dau_thau
