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

`training_scripts/run-wiazka-trainer.sh` is the pinned recipe of the best
run so far (F4, selection score 0.3361) and is ready to edit.  The trainer
runs the `F1 nodes → F2 relations → F3 composition → F4 shadows`
curriculum; each stage advances only when its gate is met **on held-out
edges** (`F1`: the refinement stays near the base embeddings, `F2`: `T_r`
beats the identity transport **on the same index**, `F3`: two-hop transport
beats doing nothing on a fixed path pool — `eval/path_lift >= 0`), and the checkpoint with the best held-out score is kept under `<save-dir>/best`. At the end of the run that checkpoint is scored again with the same criterion as the last-epoch weights (`metrics_by_checkpoint`, `shipped_checkpoint` in `report.json`) and the better of the two is what the run ships, so the reported numbers describe the model `RelationFlow.load` returns. A stage that
only exhausted its `max_steps` says so explicitly (`max_steps reached (gate NOT met)`) — an unreachable threshold is
otherwise indistinguishable from a passed one; the previous `F3` threshold (`PCE(2)/PCE(1) <= 1.6`) was mathematically
unreachable (two independent angular errors give ratio ~2.0 by construction), so every run silently burned a third of
its steps before the production term started. `F4` also turns on base-point-dependent transport,
`θ(r, x) = θ_r + ε·P(relu(Wx))`, whose strength `ε` is learned; its inverse is taken at the point the transport
started from, so the cycle stays exact and the connection is not trained away. It prints the learned `C_rs` table, a
per-relation table and writes `commutation_C.csv`, `config.json`, `metrics.json` and `report.json`.

Knobs worth knowing:

| flag | default | why |
|------|---------|-----|
| `--num-generators` | 16 | dimension of the shared skew basis; the transport of every relation is a point in it. **32 measured better** on the Słowosieć (+0.005 `mrr`, +0.010 `prod_mrr`, +0.03 naming macro at equal epochs) and it costs training time only — the served query is one `D x D` matvec whatever K is, so `training_scripts/run-wiazka-trainer.sh` ships 32 |
| `--gat-attention` (default on) | on | relation-aware softmax over incoming edges; `--no-gat-attention` restores the legacy unnormalised sum |
| `--proj-pca` (default on) | on | base↔operator chart = frozen orthonormal principal subspace; `--no-proj-pca` makes it a trainable random map |
| `--anchor-weight/--anchor-margin` | 0.5 / 0.05 | how far the refinement may move a node from its base vector |
| `--probe-size` | 1024 | held-out edges scored per evaluation (relation-stratified) |
| `--balance-relations` | on | sample the relation type first, then an edge of that type |
| `--balance-power` | 0.5 | temper of that sampling: `p(r) ~ count**power`. `1.0` is the raw edge distribution (`synonimia` = 49.7 % of the graph, the 14 relations with <100 edges never appear); `0.0` is uniform and starves the dominant relation to 1.8 % of every batch. 0.5 → 16.3 % / 3.1 % |
| `--contrast-temperature` | 0.07 | softmax temperature of the contrastive terms (`L_con`, `L_pull`); queries and keys are unit-norm, so this is the only knob that sets how much gradient the *hard* negatives keep |
| `--pull-weight/--pull-batch` | 1.0 / 256 | `L_pull`, the production term: the exported query must retrieve the gold node from the base index (full softmax over every node, false negatives masked) |
| `--knn-weight` | 0.0 | `L_knn`: keep chart neighbours neighbours after refinement (`--knn-k`, `--knn-slack`); the best run trains with **1.0** |
| `--decoder-balance` | 0.5 | class weights of the relation-decoder CE: `w(r) ~ count**-power`. Plain CE on a graph where `synonimia` holds half the edges scored macro relation accuracy 0.019; tempered to match the sampling temper (`--balance-power`) it scores 0.43 in the same epoch |
| `--decoder-label-smoothing` | 0.05 | label smoothing of the same CE |
| `--epochs` | 20 | `L_pull` (the export) only exists from `F4` — a run whose `F4` does not last >= 15 epochs undertrains `prod_mrr`. **60 recommended** at `--batch-size=2048`: the curriculum advances in *steps* (F1 400, F2 800, F3 1600), so with 158 steps/epoch `F4` starts around epoch 15 and `prod_mrr` is still climbing at epoch 40. Shrinking the batch shrinks the curriculum in epochs the same way — 128 reaches `F4` inside epoch 3 |
| `--train-ratio` | 1.0 | fraction of edges used for training; the rest is held out and never enters the GAT topology (measured: 12.3 % of held-out edges have a source that is isolated in the training topology — for those the model is by construction the identity refinement, which is why `mrr` is read against `id_mrr` on the same index) |

Baselines for ablation share the same interface: `--operator=distmult|transe`.

### The base↔operator chart

`op_dim` < the retriever dimensionality means every vector is exported through
a projection. That projection is initialised from the principal subspace of the
base embeddings, **kept frozen**, and its transpose *is* the export map, so
`to_retriever(base_to_op(x))` is an orthogonal projection at every step and the
operator acts inside a rigid coordinate chart of the retriever space. A
trainable projection cannot hold this: its weights are ~`1/sqrt(op_dim)` while
an Adam step is ~`lr`, so after a few hundred steps `W Wᵀ` was 0.22 away from
the identity and every exported query had left the retriever space (QR-Recall
measured exactly 0.0 for all relations). `export_proj` is the gated escape
hatch: zero-effect at init, opened only by `L_pull`.

### Consistency diagnostics (ideas/12)

A label-free health check of the trained operator: does a document sit near
its own negation, and does the model know that `T_r T_s != T_s T_r`?

```bash
python -m relgat_projector_apps.diagnostics.make_consistency_fixture --out-dir data/consistency
python -m relgat_projector_apps.diagnostics.consistency_report --best
```

`--best` auto-resolves the best checkpoint by scanning `report.json` (no
pointer file to go stale).  The fixtures are bilingual (Polish + English),
the learned operator is scored next to an embedder-free translation
baseline whose order separation is exactly 0 by construction -- the floor
the operator has to beat.  Details and the first measured numbers:
`docs/IDEAS/12-consistency-diagnostics.md`.

### Reading the evaluation

`evaluate()` reports the same probe under several protocols so that a
measurement is never compared across two different indexes:

| metric | query | index | reads as |
|--------|-------|-------|----------|
| `eval/mrr` | `T_r(refined_a)` | refined | graph completion (headline) |
| `eval/id_mrr` | `refined_a` | refined | what refinement alone is worth |
| `eval/raw_mrr` | `base_a` | base | plain cosine baseline |
| `eval/chart_mrr` | `chart(base_a)` | chart | ceiling the refinement starts from; `id_mrr - chart_mrr` is pure refinement damage |
| `eval/prod_mrr` | `(1-λ_r)·base_a + λ_r·export(T_r(chart(base_a)))` | base | the deployed query rewriter |
| `eval/knn_cos_refined` | | | mean cosine of chart-neighbour pairs after refinement (`knn_cos_chart` before it) |
| `eval/rel_acc_*`, `eval/rel_acc_macro_*` | `decoder` / `decoder_cal` (train-prior corrected) / `transport` (calibrated `cos(T_r x_a, x_b)`) / `ensemble` | — | can the model name the relation; `_macro` averages per relation, the plain name stays the uncorrected decoder for compatibility |
| `eval/rel_prior_strength` | | | fitted strength of the training-frequency correction of the head (`0` = the class-balanced head needs none; `1` = classic Bayes correction) |
| `eval/rel_ensemble_alpha` | | | fitted weight of the transport evidence in the `ensemble` scorer (`z(head) + w·z(transport)`), fitted on train edges; `0` means the head alone wins |
| `eval/path_mrr`, `eval/path_mrr_ident`, `eval/path_lift` | chain `T_{r2}T_{r1}` over a fixed two-hop pool | refined | composition quality: `path_lift` is the `F3` gate (the two-hop transport must beat doing nothing) |
| `eval/eps` | | | learned strength of the base-point-dependent connection; `0` before `F4`, and anything near `0` after `F4` means the connection is not training |
| `eval/mrr_lift` | | | `mrr - id_mrr`: the transport itself |
| `eval/mrr_vs_raw` | | | `mrr - raw_mrr`: the whole model vs cosine |
| `eval/mrr_prod_lift` | | | `prod_mrr - raw_mrr`: value in the vector DB |
| `eval/prod_mrr_traffic`, `eval/raw_mrr_traffic`, `eval/mrr_prod_lift_traffic` | same production query, probe drawn as **traffic** is distributed (unstratified) | base | what a deployed rewriter is judged by: the stratified average gives `synonimia` (49.7 % of the edges) 1/56 of the vote, so a dial that ruins it can look like +0.004 |

The probe is **relation-stratified** (`stratified_edge_probe`) and unique per
`(source, relation)`: on a graph where `synonimia` holds 178k of 360k edges, a
uniform 128-edge probe scores one relation and never shows the ~30 relations
with fewer than 100 edges. `report.json` additionally carries a per-relation
table (`mrr_raw`, `mrr_id`, `mrr_model`, `mrr_prod`, `lift_over_raw`).

Every ranking masks two classes of columns, and one function (`WiazkaTrainer._rank_mask`)
defines them so that the metric, the λ fit and the report cannot disagree:

* **other true targets** of the same `(src, rel)` — correct answers must not count
  as misses (the standard filtered protocol);
* **the row of the query itself** — `a -r-> b` is never answered by `a` (0 self-loops
  in 359 818 Słowosieć edges) and a retriever does not return the query document. Left
  in the index it is a candidate certain to be wrong and, of all of them, the one the
  rewritten query resembles most, so it sat above the gold at small λ and rewarded the
  dial for rewriting more than the metric wanted. The contrast losses (`L_con`, `L_pull`)
  already masked the subject; QR-Recall already excluded it (`exclude_self=True`).

QR-Recall filters other golds against the **graph's** answer table (`other_map`), not
against whatever the sample happened to contain — with one answer listed per query, a
subject with 40 correct synonyms scored its other 39 as misses.

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
r, ev = conn.classify_relation(b_a, b_b)      # (5) names + evidence: decoder / transport / ensemble / prior
# (5) the two calibration scalars of a loaded model are read back from its run
# config: conn.model.relation_prior_strength, conn.model.relation_ensemble_weight
s = conn.transport_relation_scores(v_a, v_b)  # (5) cos(T_r v_a, v_b) for every relation
out = conn.to_retriever_space(v3)             # export for the vector database
q   = conn.production_query(base_vec, 'syn')  # the query a rewriter sends (λ_r dial included)
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
- [`docs/TRAINING_QUALITY.md`](docs/TRAINING_QUALITY.md) is the audit behind the
  current defaults: every defect found by measurement (a connection that could
  not move, a `C_rs` that never learned, two hardcoded calibrations, an
  unreachable stage gate), the metric that exposes it, and the measured effect.
- Tests: `pytest tests/` (`test_operator_givens.py`, `test_wiazka_e2e.py`,
  `test_commutation_learning.py`, `test_checkpoint_and_ablation.py`,
  `test_prediction_quality.py`).

## Requirements
- Python 3.9+
- PyTorch
- torch-scatter

Install torch-scatter matching your PyTorch/CUDA version as per its documentation.
