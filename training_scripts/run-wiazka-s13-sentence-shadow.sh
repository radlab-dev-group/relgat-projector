#!/bin/sh
# S13 = S12 (rich LU identity texts) + a REAL sentence shadow manifold.
#
# Why.  docs/IDEAS/00-INDEKS.md condition 1: the sentence-level ideas (1, 2, 4,
# 12a) need F4 with shadow vectors = sentence vectors, and both benchmarks are
# scored on sentences.  Until now F4 regularised on jittered node embeddings,
# and the measured contribution was zero (S12 log: L_shadow_cycle=0.0000,
# L_shadow_commut=0.0001) for two independent reasons: the manifold was not
# made of text, and only the first 64 rows of the cache were ever read
# (ShadowSampler was constructed and never called).  Fixed here and in the
# trainer.
#
# Corpus: the 153 448 usage examples of Słowosieć, embedded with the same
# embedder as the nodes (ipipan/silver-retriever-base-v1.1, norm 23.67 against
# the nodes' 23.67, so the shadow losses need no rescaling).
#
# Everything else is the S12 recipe verbatim, so the comparison is controlled.
set -eu

DATA=data/same-embedder-rich
OUT=wiazka-models/S13_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python3 -u -m relgat_projector_apps.trainers.wiazka_trainer \
    --nodes-embeddings-path=$DATA/lexical_units_embedding.pickle \
    --relations-mapping=$DATA/relation_to_idx.json \
    --relations-triplets=$DATA/relations_triplets.json \
    --shadow-vectors=$DATA/shadow_sentences.npy \
    --shadow-batch=256 \
    --operator=givens --op-dim=256 --base-center --geometry-auto \
    --num-generators=32 \
    --epochs=40 --batch-size=2048 --num-neg=8 \
    --train-ratio=0.9 --calib-ratio=0.5 \
    --lr=0.0003 --weight-decay=0.0001 --seed=42 --device=cuda \
    --log-every-n-steps=400 \
    --balance-power=0.5 --pull-weight=1.0 --pull-batch=256 \
    --anchor-weight=0.5 --anchor-margin=0.05 \
    --contrast-hard-negatives=64 \
    --knn-weight=1.0 --knn-k=8 --knn-slack=0.02 --probe-size=1024 \
    --holonomy-squares-only \
    --order-weight=24.0 --order-join-weight=4.0 --order-join-slack=0.02 \
    --order-margin=0.06 --order-batch=1024 --order-scan-nodes=120000 \
    --commutation-from-order \
    --pull-hard-negatives=64 --pull-temperature=0.00404 \
    --qr-max-queries=256 \
    --qr-relations=synonimia,antonimia\ właściwa,fuzzynimia,derywacyjność \
    --save-dir="$OUT" 2>&1 | tee "$OUT.log"
