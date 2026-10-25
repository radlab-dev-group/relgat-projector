#!/bin/bash
# =============================================================================
# WIĄZKA / RGT — relational gauge transport trainer
# (PLAN.md: F1 nodes -> F2 relations -> F3 composition -> F4 shadows)
# =============================================================================

# Device: {cuda, cpu, cuda:x}
DEVICE="cuda"
CUDA_DEVICES="0"

# ---------------------------------------------------------------- dataset
# Same exporter output as the legacy trainer (plwordnet-milvus).
DATASET_ROOT="/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers/o78zalgm"
DATASET_DIR="${DATASET_ROOT}/dataset_syn_two_way"
LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

# Optional: real text vectors from the retriever index (NOT graph nodes) used
# to supervise the operator off the graph (F4 / PLAN.md 2.6). Leave empty to
# synthesise shadows by jittering the node embeddings.
SHADOW_VECTORS=""

# ---------------------------------------------------------------- operator
# OPERATOR: {givens, distmult, transe}  (givens = rotation+scale, the method)
OPERATOR="givens"
OP_DIM=256
# K measured on the Slowosiec at 40 epochs (same seed, everything else equal):
# K=32 against K=16 is worth +0.005 mrr, +0.010 prod_mrr and +0.03 relation
# naming macro.  K only costs *training* time -- the served query is one
# D x D matvec whatever K is -- so this is bought with GPU hours, not latency.
NUM_GENERATORS=32

# ---------------------------------------------------------------- training
# The curriculum advances in *steps*, not epochs: F1 400, F2 800, F3 1600, so
# the stage budgets are only sensible for a batch that covers a decent slice of
# the 324k training edges.  At BATCH_SIZE=128 (2 530 steps/epoch) the whole
# curriculum runs out inside the third epoch and the export (L_pull) is trained
# from the very beginning of the schedule; at 2048 (158 steps/epoch) F4 starts
# around epoch 15 and gets the rest of the run, which is what the numbers below
# were measured with.
EPOCHS=60
BATCH_SIZE=2048
NUM_NEG=8
TRAIN_RATIO=0.90
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.0001
GAT_NUM_LAYERS=2
GAT_HEADS=2
DROPOUT=0.1
SEED=42
LOG_EVERY_N_STEPS=400

# ------------------------------------------------------------- loss weighting
# Sampling temper of the relation-balanced batch: p(r) ~ count**BALANCE_POWER.
#   1.0 = the raw edge distribution (synonimia is 49.7 % of the Słowosieć and
#         the 14 relations with <100 edges are never sampled at all)
#   0.0 = uniform over relations (those 14 relations then take 25 % of every
#         batch while the dominant one drops to 1.8 % and loses ~0.24 MRR)
BALANCE_POWER=0.5
# L_pull: the production term. The exported query is scored with a full softmax
# over every node of the retriever index, so this is what makes the deployed
# query rewriter better than sending the embedding unchanged.
PULL_WEIGHT=1.0
PULL_BATCH=256
# L_anchor: how far the refinement may move a node from its own base vector.
ANCHOR_WEIGHT=0.5
ANCHOR_MARGIN=0.05
# L_knn: keep chart neighbours neighbours after refinement. The refinement buys
# the semantic-role relations and pays with the index quality of the relations
# the base embeddings already encode (measured: id_mrr 0.106 on the refined index
# against chart_mrr 0.139 for the same vectors unrefined); this hinge is the
# targeted trade-off knob. Non-zero values are an experiment, not the default.
KNN_WEIGHT=0.0
# Held-out edges scored at every evaluation (relation-stratified).
PROBE_SIZE=1024
# Relations validated with QR-Recall at the end of the run.
QR_RELATIONS="synonimia,antonimia właściwa,fuzzynimia,derywacyjność"

# ---------------------------------------------------------------- output
START_DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_DIR="wiazka-models/wiazka-${OPERATOR}_${START_DATE_STR}"

# torch_scatter's wheel segfaults on import in some environments; uncomment
# to force the pure-PyTorch scatter fallback.
# export RELgat_NO_TORCH_SCATTER=1

# =============================================================================
ARGS=(
  --nodes-embeddings-path="${LU_EMBEDDING}"
  --relations-mapping="${RELS_MAPPING}"
  --relations-triplets="${RELS_TRIPLETS}"
  --operator="${OPERATOR}"
  --op-dim="${OP_DIM}"
  --num-generators="${NUM_GENERATORS}"
  --gat-num-layers="${GAT_NUM_LAYERS}"
  --gat-heads="${GAT_HEADS}"
  --dropout="${DROPOUT}"
  --epochs="${EPOCHS}"
  --batch-size="${BATCH_SIZE}"
  --num-neg="${NUM_NEG}"
  --train-ratio="${TRAIN_RATIO}"
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
  --probe-size="${PROBE_SIZE}"
  --qr-relations="${QR_RELATIONS}"
  --save-dir="${OUT_DIR}"
)
[[ -n "${SHADOW_VECTORS}" ]] && ARGS+=(--shadow-vectors="${SHADOW_VECTORS}")

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" relgat-projector-wiazka "${ARGS[@]}"

echo "====================================================================="
echo "[POST TRAINING INFO] started : ${START_DATE_STR}"
echo "[POST TRAINING INFO] saved to: ${OUT_DIR}"
echo "[POST TRAINING INFO] report  : ${OUT_DIR}/report.json"
echo "[POST TRAINING INFO] C_rs    : ${OUT_DIR}/commutation_C.csv"
echo "====================================================================="
