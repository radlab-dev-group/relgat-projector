#!/bin/sh
# S14 = S13 + sentence-level transport supervision (L_shadow_sent).
#
# Why.  S13 put real sentences in the F4 shadow manifold and moved prod_mrr
# 0.5421 -> 0.5767, but 12a-PL regressed 0.757 -> 0.705 and the reason is in
# the loss breakdown: L_shadow_cycle = 0.0000 through the whole run.  It is
# tautological -- `inverse()` in core/operator/givens.py is the exact analytic
# inverse of `apply()` at the same base point, so T_r^-1(T_r z) == z carries no
# gradient, and L_shadow_commut (0.0003) is two orders under the terms that do.
# F4 therefore regularises the operator off the graph and teaches it nothing:
# nothing in the curriculum ever asks T_r to carry *text* along a relation,
# which is exactly what the 12a/01 benchmarks measure.
#
# The fix is supervision, not a stronger regulariser.  Słowosieć attaches usage
# examples to 111 536 lexical units; for a graph edge (a, r, b) whose endpoints
# both own an example, the pair of sentences IS a labelled transport pair in
# the space the product works in.  L_shadow_sent scores
# production_query(sentence_of_a, r) against the sentence vectors of the batch
# with the same InfoNCE (and the same false-negative / subject masking) as
# L_pull, so the operator is trained on text, on the served path.
# 102 224 of the 313 960 train edges qualify; the corpus and its LU index are
# built by relgat_projector_apps.diagnostics.build_shadow_sentence_corpus.
#
# Everything else is the S12/S13 recipe verbatim, so the comparison is
# controlled: S14 vs S13 isolates the sentence supervision.
set -eu

DATA=data/same-embedder-rich
OUT=wiazka-models/S14_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

python3 -u -m relgat_projector_apps.trainers.wiazka_trainer \
    --nodes-embeddings-path=$DATA/lexical_units_embedding.pickle \
    --relations-mapping=$DATA/relation_to_idx.json \
    --relations-triplets=$DATA/relations_triplets.json \
    --shadow-vectors=$DATA/shadow_sentences_lu.npy \
    --shadow-sentences=$DATA/shadow_sentences_lu.npy \
    --shadow-sentence-lus=$DATA/shadow_sentences_lu_ids.npy \
    --shadow-sent-weight=1.0 --shadow-sent-batch=256 \
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
