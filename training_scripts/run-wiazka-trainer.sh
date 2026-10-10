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
NUM_GENERATORS=16

# ---------------------------------------------------------------- training
EPOCHS=20
BATCH_SIZE=128
NUM_NEG=8
TRAIN_RATIO=0.90
LEARNING_RATE=0.0003
WEIGHT_DECAY=0.0001
GAT_NUM_LAYERS=2
GAT_HEADS=2
DROPOUT=0.1
SEED=42
LOG_EVERY_N_STEPS=10

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
