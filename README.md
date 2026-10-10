# RelGAT Projector and RelGAT trainer with frozen embeddings
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

> **Prerequisite**: `radlab-ml-utils`
>
> This project uses the 
> [radlab-ml-utils](https://github.com/radlab-dev-group/ml-utils) 
> library for machine learning utilities 
> (e.g., experiment/result logging with Weights & Biases/wandb).
> Install it before working with ML-related parts:
>
> ```bash
> pip install git+https://github.com/radlab-dev-group/ml-utils.git
> ```
>
> For more options and details, see the library README: 
> https://github.com/radlab-dev-group/ml-utils

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
pip install "git+[https://github.com/radlab-dev-group/relgat-projector.git#egg=relgat-projector](https://github.com/radlab-dev-group/relgat-projector.git#egg=relgat-projector)"
````

After installation, a console entry point is available:
```bash
relgat-projector-train --help
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
(see its repository for details — https://github.com/radlab-dev-group/plwordnet).
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
relgat-projector-train
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

## WIĄZKA — relational transport trainer (new)

A second training path (`feature/wiazka-rgt`) treats a semantic graph
(Słowosieć / WordNet / ontology) as a **connection**: nodes carry a vector,
relations are *invertible linear maps in the same space*, and a chain of
relations compiles to a single matrix.

    T_r(x) = s_r · exp(θ_rᵀ B) x ∈ R⁺ × O(D)        T⁻¹ exact, cosine preserving

What it gives you:

- **`t → r₁ → … → rₖ → t′` in one matmul** — `conn.compile([r1, r2])` folds a
  whole chain; no graph traversal at query time;
- **`λ` dial** — `conn.apply(v, "hyper", lam=0.4)` = "40 % more general",
  with `λ = 0` exactly the identity;
- **path consistency as the training signal** — holonomy + a learned
  commutation matrix `C_rs` instead of heuristically supervised long paths; this
  is what makes the operator valid on vectors that were never graph nodes;
- **applicability head** — `conn.applicability(v, "ant")` says when a relation
  is meaningless for a vector, so a search pipeline can refuse instead of
  returning noise;
- **relation decoder** — `conn.decode(v, v3)` answers "which relation maps this
  query to those results?".

### Training

```bash
relgat-projector-wiazka \
  --nodes-embeddings-path=.../lexical_units_embedding.pickle \
  --relations-mapping=.../relation_to_idx.json \
  --relations-triplets=.../relations_triplets.json \
  --operator=givens --op-dim=256 --num-generators=16 \
  --epochs=20 --batch-size=128 --train-ratio=0.90 \
  --device=cuda --save-dir=wiazka-models/run1
```

`training_scripts/run-wiazka-trainer.sh` is a ready-to-edit template. The
trainer runs the `F1 nodes → F2 relations → F3 composition → F4 shadows`
curriculum; each stage advances only when its gate is met **on held-out
edges**. It prints the learned `C_rs` table and writes
`commutation_C.csv`, `config.json`, `metrics.json` and `report.json`.

Baselines for ablation share the same interface: `--operator=distmult|transe`.

### Inference

```python
from relgat_projector.inference import RelationFlow

conn = RelationFlow.load("wiazka-models/run1")
v  = conn.refine(base_vec)                    # (1) refined representation
v2 = conn.apply(v, "syn")                     # (3) one relation step
v3 = conn.apply_chain(v, ["ant", "syn"])      # (3) relation chain
op = conn.compile(["ant", "syn"])             # (3) one operator for the chain
v4 = conn.apply(v, "hyper", lam=0.4)          # (3) dial: "lightly generalise"
p  = conn.applicability(v, "ant")             # (4) is it defined at all?
r, conf = conn.decode(v, v3)                  # (5) "which relation was this?"
out = conn.to_retriever_space(v3)             # export for the vector database
```

### Metrics

Beyond MRR/Hits: **PCE(k)** (path compilation error — the headline curve),
**CCE** (cycle closure), **ISOT** (cosine distortion), **HD** (holonomy
defect), **OMD** (off-manifold degradation on real text vectors), and
applicability **AUC/ECE**. See `docs/GEOMETRY.md` for definitions, thresholds,
reference numbers and how to read `C_rs`.

### Notes

- The `WIĄZKA` path is self-contained: it does not import the legacy
  `RelGATTrainer` and works with the pure-PyTorch scatter fallback
  (`export RELgat_NO_TORCH_SCATTER=1`), which is required where the
  `torch_scatter` wheel segfaults on import.
- Tests: `pytest tests/` (`test_operator_givens.py`, `test_wiazka_e2e.py`,
  `test_commutation_learning.py`, `test_checkpoint_and_ablation.py`).

## Requirements
- Python 3.9+
- PyTorch
- torch-scatter

Install torch-scatter matching your PyTorch/CUDA version as per its documentation.
