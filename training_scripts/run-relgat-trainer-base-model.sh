#!/bin/bash

# =============================================================================
# =============================================================================
# --------------------  GENERAL OPTIONS
# =============================================================================
# Device: {cuda, cpu, cuda:x}
DEVICE="cuda"
CUDA_DEVICES="0"
MAX_CHECKPOINTS=5

# Ratio of training data
TRAIN_EVAL_DATASET_RATIO="0.90"

# Which architecture will be trained {small, medium, medium}
ARCHITECTURE="small"

# =============================================================================
# =============================================================================
# --------------------  TRAINING PARAMETERS
# =============================================================================
if [[ "${ARCHITECTURE}" == "small" ]]
then
  EPOCHS=20
  BATCH_SIZE=256
  NUM_NEG_TO_POS=32

  GAT_OUT_DIM=128
  NUM_OF_LAYERS=2
  NUM_OF_HEADS=12

  LEARNING_RATE=0.00002

  SAVE_N_STEPS=100
  EVAL_N_STEPS=100
  LOG_EVERY_N_STEPS=10
elif [[ "${ARCHITECTURE}" == "medium" ]]
then
  EPOCHS=10
  BATCH_SIZE=128
  NUM_NEG_TO_POS=24

  LEARNING_RATE=0.0002

  GAT_OUT_DIM=256
  NUM_OF_LAYERS=1
  NUM_OF_HEADS=12

  SAVE_N_STEPS=200
  EVAL_N_STEPS=200
  LOG_EVERY_N_STEPS=5
elif [[ "${ARCHITECTURE}" == "large" ]]
then
  EPOCHS=50
  BATCH_SIZE=128
  NUM_NEG_TO_POS=32

  LEARNING_RATE=0.00005

  GAT_OUT_DIM=256
  NUM_OF_LAYERS=4
  NUM_OF_HEADS=12

  SAVE_N_STEPS=400
  EVAL_N_STEPS=400
  LOG_EVERY_N_STEPS=5
else
  echo "Supported architectures: [small, medium, large]"
  exit 1
fi

# =============================================================================
# Scorer, one of: [distmult, transe]
SCORER="transe"

# Dropout used while training (on embedder dimension)
DROPOUT=0.3

# Relation attention dropout
DROPOUT_REL_ATT=0.0

# Multiplicative factor applied to LR after warm‑up (default: 1.0 – no change)
LR_DECAY=1.0

# Learning rate scheduler, one of: [linear, cosine, constant]
LR_SCHEDULER="linear"

# Optional explicit warmup steps (comment out to auto-compute)
# WARMUP_STEPS=500
# weight decay (0.0 is default) - used in Adam optimizer
WEIGHT_DECAY=0.0

# If set, clips gradient norm to this value (default: None – no clipping)
#GRADIENT_CLIPPING=10.0

# Number of evaluation steps without improvement after which training stops
EARLY_STOP_PATIENCE_STEPS=10

# To enable Automatic Mixed Precision (AMP) training uncomment next line
#USE_AMP=1

# If set, the model will not mask edges by relation type - to use option
# uncomment next line
#DISABLE_EDGE_TYPES=1

# Enable self-adversarial negative sampling instead of margin ranking loss
USE_SELF_ADV_NEG=True

# =============================================================================#
# =============================================================================
# --------------------  DATASET CONFIGURATION
# =============================================================================

# Output directory to store the model and checkpoints during training
OUT_MODEL_DIR="relgat-models/relgat-${ARCHITECTURE}_$(date +%Y%m%d_%H%M%S)"

# This dataset have to prepared using the base embedder (the same as the used in application)
DATASET_ROOT="/mnt/data2/data/resources/plwordnet_handler/relgat/aligned-dataset-identifiers/wtcsnxj9"
# Available datasets:
#  - FULL: dataset_syn_two_way
#  - SAMPLE: dataset_syn_two_way__limit1k
DATASET_DIR="${DATASET_ROOT}/dataset_syn_two_way"
LU_EMBEDDING="${DATASET_DIR}/lexical_units_embedding.pickle"
RELS_MAPPING="${DATASET_DIR}/relation_to_idx.json"
RELS_TRIPLETS="${DATASET_DIR}/relations_triplets.json"

# =============================================================================
# =============================================================================
# --------------------  APPLICATION CALL
# =============================================================================

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" relgat-base-train \
  --architecture="${ARCHITECTURE}" \
  --lr="${LEARNING_RATE}" \
  --lr-decay="${LR_DECAY}" \
  --early-stop-patience="${EARLY_STOP_PATIENCE_STEPS}" \
  --weight-decay="${WEIGHT_DECAY}" \
  --lr-scheduler="${LR_SCHEDULER}" \
  --num-neg="${NUM_NEG_TO_POS}" \
  --gat-num-layers="${NUM_OF_LAYERS}" \
  --heads="${NUM_OF_HEADS}" \
  --epochs="${EPOCHS}" \
  --scorer="${SCORER}" \
  --dropout="${DROPOUT}" \
  --dropout-relation-attention="${DROPOUT_REL_ATT}" \
  --gat-out-dim="${GAT_OUT_DIM}" \
  --train-ratio="${TRAIN_EVAL_DATASET_RATIO}" \
  --batch-size="${BATCH_SIZE}" \
  --nodes-embeddings-path="${LU_EMBEDDING}" \
  --relations-mapping="${RELS_MAPPING}" \
  --relations-triplets="${RELS_TRIPLETS}" \
  --device="${DEVICE}" \
  --log-every-n-steps="${LOG_EVERY_N_STEPS}" \
  --save-dir="${OUT_MODEL_DIR}" \
  --save-every-n-steps="${SAVE_N_STEPS}" \
  --max-checkpoints="${MAX_CHECKPOINTS}" \
  ${EVAL_N_STEPS:+--eval-every-n-steps="${EVAL_N_STEPS}"} \
  ${WARMUP_STEPS:+--warmup-steps="${WARMUP_STEPS}"} \
  ${GRADIENT_CLIPPING:+--grad-clip-norm="${GRADIENT_CLIPPING}"} \
  ${USE_AMP:+--use-amp="${USE_AMP}"} \
  ${DISABLE_EDGE_TYPES:+--disable-edge-type-mask="${DISABLE_EDGE_TYPES}"} \
  ${USE_SELF_ADV_NEG:+--use-self-adv-neg}
