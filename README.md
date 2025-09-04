# RekGAT Projector with RelGAT trainer
This repository provides a hybrid Relational-GAT model with a projection head back to the input embedding space, 
enabling relation-specific operators (TransE/DistMult) to be applied consistently inside and outside the graph. 
The model supports inductive embedding imputation for nodes without vectors and relation-driven query expansion 
by composing operators along graph paths—all in the same embedding space used within retriever. 
Includes multi-objective training (ranking + reconstruction), metrics (MRR/Hits, cosine, MSE), and utilities for inference and export.


A lightweight trainer for Relational Graph Attention Networks (RelGAT) that learns 
relation-aware node representations from a knowledge graph. It consumes precomputed 
node embeddings and relation triplets, and supports multi-head attention, 
negative sampling, LR scheduling, checkpointing, and optional Weights & Biases logging.

## Installation
- Requirements: Python 3.9+, pip, and a PyTorch-compatible environment (CPU or CUDA).
- Binary note: torch and torch-scatter must match your PyTorch/CUDA stack.

Install from local source:
```bash 
pip install .
```

Editable/development install:
```bash
pip install -e .
````

Install directly from a Git repository (replace URL accordingly):
```bash
pip install "git+[https://github.com/radlab-dev-group/relgat-llm.git#egg=relgat-llm](https://github.com/radlab-dev-group/relgat-llm.git#egg=relgat-llm)"
````

After installation, a console entry point is available:
```bash
relgat-train --help
```

If you run into issues with torch-scatter (especially on GPU), install the 
wheel matching your PyTorch/CUDA versions as per the project’s documentation.


## Highlights
- Relational multi-head GAT layers
- Flexible scorers (distmult, transe)
- Warmup/decay LR schedulers
- Periodic evaluation and checkpointing
- CPU/CUDA device selection

## Dataset preparation
This trainer expects:
- nodes embeddings file (lexical units)
- relations mapping (relation name → id)
- relation triplets (from_idx, to_idx, relation_name)

You can export a compatible dataset using the plwordnet-milvus CLI
(see its repository for details - 
https://github.com/radlab-dev-group/radlab-plwordnet).
Example:

````bash
plwordnet-milvus
    --milvus-config=resources/milvus-config.json
    --embedder-config=resources/embedder-config.json
    --nx-graph-dir=/path/to/nx/graphs
    --relgat-mapping-directory=resources/aligned-dataset-identifiers/
    --export-relgat-dataset
    --relgat-dataset-directory=resources/aligned-dataset-identifiers/dataset
    --log-level="DEBUG"
````

## Quick start
Train a model with your prepared files:

```bash
relgat-train
    --nodes-embeddings-path /path/to/nodes_embeddings.pt
    --relations-mapping /path/to/relations_mapping.json
    --relations-triplets /path/to/relations_triplets.tsv
    --gat-out-dim 200
    --gat-num-layers 2
    --heads 4
    --scorer distmult
    --epochs 10
    --batch-size 1024
    --lr 2e-4
    --lr-scheduler cosine
    --device cuda:0
    --save-dir outputs/relgat_run
    --save-every-n-steps 5000
    --max-checkpoints 5
```

Common flags:
- --train-ratio: train/eval split ratio
- --num-neg: negative samples per positive
- --dropout: GAT dropout
- --warmup-steps: warmup steps (auto if omitted)
- --weight-decay, --grad-clip-norm: regularization
- --eval-every-n-steps: step-based evaluation (or per-epoch by default)
- --run-name: optional W&B run name

## Outputs
- Checkpoints in save-dir/checkpoint-<step> (if enabled)
- Final model in save-dir

## Requirements
- Python 3.9+
- PyTorch
- torch-scatter

Install torch-scatter matching your PyTorch/CUDA version as per its documentation.
