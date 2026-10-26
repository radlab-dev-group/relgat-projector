#!/bin/bash
# =============================================================================
# WIĄZKA / RGT — the SAME-EMBEDDER control run
# =============================================================================
#
# Why this exists.  The best run (F4) is trained in the space of the exporter's
# own 1152-d embedder, which nobody can query with arbitrary text.  That left
# the consistency diagnostics (docs/IDEAS/12) with a hole: 12b could measure
# the *learned* operator (graph vectors are in the checkpoint), but 12a could
# only measure a lexicon-derived translation baseline, because document
# sentences live in a sentence-embedder space and the operator lives in
# Słowosieć's.  Comparing "learned" against "baseline" across two different
# spaces measures the spaces, not the method.
#
# This run closes that hole: the graph is re-embedded with
# ``ipipan/silver-retriever-base-v1.1`` — the exact model the diagnostics use
# for documents — so the trained ``T_syn``/``T_ant`` act on the very vectors
# the documents are scored in, and 12a becomes a measurement of the operator.
#
# Build the dataset first (lemma of every lexical unit, embedded once):
#
#   python -m relgat_projector_apps.diagnostics.build_same_embedder_dataset
#
# Then train, and finally compare the learned operator with the baseline:
#
#   python -m relgat_projector_apps.diagnostics.consistency_report \
#       --ckpt wiazka-models/<this-run>/best
#
# Geometry flags are NOT optional here.  That space is strongly anisotropic
# (mean random-pair cosine far above 0), so
#   --base-center    puts the chart origin at the centroid, which is what makes
#                    the operator see directions instead of the dominant cone;
#   --geometry-auto  rescales the cosine knobs to this space's cosine spread;
#   --op-dim 0       derives the chart dimension from this space's spectrum
#                    (256 was a number for the 1152-d space, not a law).
#
DEVICE="cuda"
CUDA_DEVICES="0"

DATASET_DIR="data/same-embedder"
LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

OPERATOR="givens"
OP_DIM=0
NUM_GENERATORS=32

EPOCHS=60
BATCH_SIZE=2048
NUM_NEG=8
TRAIN_RATIO=0.90
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.0001
SEED=42
LOG_EVERY_N_STEPS=400

BALANCE_POWER=0.5
PULL_WEIGHT=1.0
PULL_BATCH=256
ANCHOR_WEIGHT=0.5
ANCHOR_MARGIN=0.05
KNN_WEIGHT=1.0
KNN_K=8
KNN_SLACK=0.02
CALIB_RATIO=0.5
PROBE_SIZE=1024
QR_RELATIONS="synonimia,antonimia właściwa,fuzzynimia,derywacyjność"
QR_MAX_QUERIES=256

START_DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_DIR="wiazka-models/S1_${START_DATE_STR}"

ARGS=(
  --nodes-embeddings-path="${LU_EMBEDDING}"
  --relations-mapping="${RELS_MAPPING}"
  --relations-triplets="${RELS_TRIPLETS}"
  --operator="${OPERATOR}"
  --op-dim="${OP_DIM}"
  --base-center
  --geometry-auto
  --num-generators="${NUM_GENERATORS}"
  --epochs="${EPOCHS}"
  --batch-size="${BATCH_SIZE}"
  --num-neg="${NUM_NEG}"
  --train-ratio="${TRAIN_RATIO}"
  --calib-ratio="${CALIB_RATIO}"
  --lr="${LEARNING_RATE}"
  --weight-decay="${WEIGHT_DECAY}"
  --seed="${SEED}"
  --device="${DEVICE}"
  --log-every-n-steps="${LOG_EVERY_N_STEPS}"
  --balance-power="${BALANCE_POWER}"
  --pull-weight="${PULL_WEIGHT}"
  --pull-batch="${PULL_BATCH}"
  --anchor-weight="${ANCHOR_WEIGHT}"
  --anchor-margin="${ANCHOR_MARGIN}"
  --knn-weight="${KNN_WEIGHT}"
  --knn-k="${KNN_K}"
  --knn-slack="${KNN_SLACK}"
  --probe-size="${PROBE_SIZE}"
  --qr-max-queries="${QR_MAX_QUERIES}"
  --qr-relations="${QR_RELATIONS}"
  --save-dir="${OUT_DIR}"
)

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" relgat-projector-wiazka "${ARGS[@]}"

echo "[POST] saved to ${OUT_DIR}; diagnostics:"
echo "  python -m relgat_projector_apps.diagnostics.consistency_report --ckpt ${OUT_DIR}/best"
