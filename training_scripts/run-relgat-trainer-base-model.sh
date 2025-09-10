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
TRAIN_EVAL_DATASET_RATIO="0.80"

# Which architecture will be trained {small, medium, medium}
ARCHITECTURE="small"

# =============================================================================
# =============================================================================
# --------------------  TRAINING PARAMETERS
# =============================================================================
if [[ "${ARCHITECTURE}" == "small" ]]
then
  EPOCHS=60
  BATCH_SIZE=128
  NUM_NEG_TO_POS=32

  GAT_OUT_DIM=128
  NUM_OF_LAYERS=2
  NUM_OF_HEADS=16

  LEARNING_RATE=0.00002

  SAVE_N_STEPS=300
  EVAL_N_STEPS=150
  LOG_EVERY_N_STEPS=10
#elif [[ "${ARCHITECTURE}" == "medium" ]]
#then
#  EPOCHS=10
#  BATCH_SIZE=128
#  NUM_NEG_TO_POS=24
#
#  LEARNING_RATE=0.0002
#
#  GAT_OUT_DIM=256
#  NUM_OF_LAYERS=2
#  NUM_OF_HEADS=12
#
#  SAVE_N_STEPS=200
#  EVAL_N_STEPS=200
#  LOG_EVERY_N_STEPS=5
#elif [[ "${ARCHITECTURE}" == "large" ]]
#then
#  EPOCHS=50
#  BATCH_SIZE=128
#  NUM_NEG_TO_POS=32
#
#  LEARNING_RATE=0.00005
#
#  GAT_OUT_DIM=256
#  NUM_OF_LAYERS=4
#  NUM_OF_HEADS=12
#
#  SAVE_N_STEPS=400
#  EVAL_N_STEPS=400
#  LOG_EVERY_N_STEPS=5
else
  echo "Supported architectures: [small, medium, large]"
  exit 1
fi

# =============================================================================
# Scorer, one of: [distmult, transe]
SCORER="distmult"

# Enable value above, to use the projection to base embedding size.
# Relation embeddings will be projected to the input embedding size
# (to disable projection, lets comment the next line)
PROJECTION_TO_BASE_EMB_SIZE=True
# Projection layers. 0 - Identity, 1 - Linear, >=2 - MLP
PROJECTION_LAYERS=2
# Projection dropout
PROJECTION_DROPOUT=0.1
# Dimension of hidden layers in projection (0 to the same as input dim)
PROJECTION_HIDDEN_DIM=0

# When using projection, multi objective loss function will be used.
# ... then following weights will be used:
RELGAT_WEIGHT=1.0
# Weigh for positive cosine
COSINE_WEIGHT_POS=1.0
# Weigh for negative cosine
COSINE_WEIGHT_NEG=1.0
# Weigh for mean square error
MSE_WEIGHT=0.0

# Project to input size embeddings (if option is given).
# If not set, then frozen-GAT will be learned
# PROJECT_TO_INPUT_SIZE=1

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
WEIGHT_DECAY=0.0001

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
START_DATE_STR=$(date +%Y%m%d_%H%M%S)
OUT_MODEL_DIR="relgat-models/relgat-${ARCHITECTURE}_${START_DATE_STR}"

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

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" relgat-projector-train \
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
  --relgat-weight="${RELGAT_WEIGHT}" \
  --pos-cosine-weight="${COSINE_WEIGHT_POS}" \
  --neg-cosine-weight="${COSINE_WEIGHT_NEG}" \
  --mse-weight="${MSE_WEIGHT}" \
  ${EVAL_N_STEPS:+--eval-every-n-steps="${EVAL_N_STEPS}"} \
  ${WARMUP_STEPS:+--warmup-steps="${WARMUP_STEPS}"} \
  ${GRADIENT_CLIPPING:+--grad-clip-norm="${GRADIENT_CLIPPING}"} \
  ${USE_AMP:+--use-amp="${USE_AMP}"} \
  ${DISABLE_EDGE_TYPES:+--disable-edge-type-mask="${DISABLE_EDGE_TYPES}"} \
  ${USE_SELF_ADV_NEG:+--use-self-adv-neg} \
  ${PROJECTION_LAYERS:+--projection-layers="${PROJECTION_LAYERS}"} \
  ${PROJECTION_DROPOUT:+--projection_dropout="${PROJECTION_DROPOUT}"} \
  ${PROJECTION_HIDDEN_DIM:+--projection-hidden-dim="${PROJECTION_HIDDEN_DIM}"} \
  ${PROJECTION_TO_BASE_EMB_SIZE:+--project-to-input-size}


END_DATE_STR=$(date +%Y%m%d_%H%M%S)
echo "====================================================================="
echo "[POST TRAINING INFO] Training stared on  : ${START_DATE_STR}"
echo "[POST TRAINING INFO] Training ended on   : ${END_DATE_STR}"
echo "[POST TRAINING INFO] Checkpoints saved to: ${OUT_MODEL_DIR}"
echo "====================================================================="
