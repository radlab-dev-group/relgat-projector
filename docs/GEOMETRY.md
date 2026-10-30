# WIĄZKA / RGT — geometry of the relational transport

Design document for the operator that turns `N -> r -> K` into a *linear map in
the retriever space*, plus a guide to reading the artefacts it produces
(`PCE`, `HD`, `CCE`, `ISOT`, `OMD`, and the learned commutation matrix `C_rs`).

The user-facing requirement is small and very demanding:

> given a vector `t` (a text, not a graph node) and relations `r_1..r_k`
> (negation, synonymy, hypernymy, …) return `t' = T_{r_k}…T_{r_1}(t)` such that
> `t'` still means something in the **same** space, so a cosine search finds
> "the negated / anonymised / generalised version" of `t`.

Two properties make that possible: the operator must not destroy the cosine
geometry, and it must be *path consistent* — otherwise applying a relation to a
vector that was never in the graph is undefined behaviour.

---

## 1. The operator: relations as the similarity group

A relation is an element of `R+ x O(D)` (the similarity group — the smallest
group that preserves angles up to scale):

    T_r(x) = s_r * exp(Omega_r) x,      Omega_r = sum_m theta_{r,m} B_m

* `B_m in so(D)` — a learned, Frobenius-orthonormal basis of skew generators
  (`K = num_generators`, shared by all relations);
* `theta in R^{num_rel x K}` — relation coordinates in the Lie algebra;
* `s_r = exp(log_scale_r)` — a learned magnitude ("hypernymy narrows",
  "antonymy preserves size").

`exp` of a skew matrix is computed with the **Cayley transform**
`(I - O/2)(I + O/2)^{-1}` — exact, differentiable, no `expm` and no
orthogonality-drifting regulariser.

Why this and not a diagonal (DistMult) or translational (TransE) operator:

| property | `givens` | `distmult` | `transe` |
|---|---|---|---|
| `T^{-1}` exact | yes | yes (safe division) | yes |
| cosine geometry preserved | yes (rotation) | no (rescaling) | no (shift) |
| non-commutative `r1 r2 != r2 r1` | yes | **never** | never |
| chain folds to one `D x D` matmul | yes | yes | yes |
| `lambda` dial (0 = identity) | yes | yes | yes |

DistMult is *diagonal*, hence commutative: it cannot represent that
"negation of a synonym" differs from "synonym of a negation". That is the
structural reason the previous pipeline could not learn composition.

### 1.1 Dial (relation strength)

    T_r(lambda) = s_r^lambda * exp(lambda * Omega_r) x

`lambda = 0` is *exactly* the identity (contract tested for every operator),
`lambda = 1` is the full relation. This gives "make the query 40 % more
anonymised" without retraining.

### 1.2 Context (base-point dependence)

Relations do not mean the same thing on a verb as on a proper noun, so the
coordinates are modulated by the point of application:

    theta(r, x) = theta_r + eps * P(relu(W x)),     eps = exp(log_eps) ~ 0

The hyper-network (`W: D x 64`, `P: 64 x K`) is created **only at stage F4**
and `eps` starts at `exp(-6) ~ 2.5e-3` with a zeroed output layer, so turning
context on does not move the transport at all on the first step; it is kept
only if it reduces the loss. Because `T` is then no longer an exact rotation,
`CCE` (below) becomes a real measurement instead of a tautology.

### 1.3 Composition

    T_{r_k} … T_{r_1} = (prod_k exp(theta_{r_k}^T B)) * prod s_{r_i}

The product of rotations is a rotation, so a whole chain compiles to **one**
`D x D` matrix and one scalar — `RelationFlow.compile([...])`. Production cost
of a chain is one `matmul`, independent of `len(chain)` and of the graph.

---

## 2. Path consistency = zero curvature (the core idea)

If transport did not depend on the path between two nodes, the connection would
be flat. So instead of heuristically supervising long paths, we *penalise
curvature*:

* **Holonomy** around a closed square `A -r-> C -q-> D <-s- B <-p- A`:

      H = T_p^{-1} T_s^{-1} T_q T_r,      L_curv = || H(x_A) - x_A ||^2

* **Commutation** of relation pairs, weighted by a learned matrix
  `C_rs in [0,1]^{num_rel x num_rel}`:

      L_commut = w_c * C_rs * ||T_r T_s(x) - T_s T_r(x)||^2
               + w_a * (1 - C_rs) * margin(||T_r T_s(x) - T_s T_r(x)||)

The second term is what keeps the model honest: forcing everything to commute
would collapse the operator onto DistMult. Pairs that *do* commute are
supervised from the graph (a relation pair whose two routes provably meet in
the graph — a "witness" — is commuting), the rest is free.

Why this is what enables off-graph transforms: `t` is not in the graph, so the
only way to know where `T_{r_2} T_{r_1}(t)` should land is that transport
between the same two meanings is independent of the route taken. Flatness is
therefore not an aesthetic property, it *is* the generalisation property.

---

## 3. Metrics (how to tell whether it works)

| name | definition | good value | where |
|---|---|---|---|
| `eval/cos_edge` | `cos(T_r(x_A), x_B)` on edges | -> 1 | `WiazkaTrainer.evaluate` |
| `PCE(k)` `eval/pce_k*` | `1 - cos(T_compiled(x_A), x_B)` for chains of length k | flat curve in k | `core/metrics/geometry.py` |
| `CCE` `eval/cce` | `||T^{-1}(T(x)) - x||` | ~0 exact, grows with context | `cycle_closure_error` |
| `ISOT` `eval/isot` | `abs(cos(u,v) - cos(T u, T v))` | < 0.05 | `isometry_error` |
| `HD` | holonomy defect on closed squares | < 0.15 | `holonomy_defect` |
| `OMD` | `MRR_nodes / MRR_shadow_texts` | ~1.0 | `core/metrics/offline.py` |
| `APL AUC`, `ECE` | applicability quality / calibration | > 0.85, < 0.08 | `core/metrics/offline.py` |

**PCE is the headline metric of the whole project.** A method that learns
relations one edge at a time shows PCE growing with `k` (composition drifts);
a connection-consistent operator keeps the curve nearly flat. Reference run on
a `8x8` grid with `dx`, `dy` (40 epochs, `op_dim=48`):

    ep  1  stage=F1_nodes         cos_edge=0.070  pce_k2=0.999  pce_k3=1.014
    ep 17  stage=F3_composition   cos_edge=0.900  pce_k2=0.142  pce_k3=0.165
    ep 40  stage=F4_shadows       cos_edge=0.912  pce_k2=0.104  pce_k3=0.153
                                     cce=0.0001  isot=0.058

PCE(2) ~ PCE(3) means the two- and three-step transports are equally reliable:
the chain is a genuine operator, not a stack of approximations.

---

## 4. Reading `C_rs` (the interpretability artefact)

Every run writes `commutation_C.csv` / `.npy` next to the checkpoint, and the
CLI prints the table. `C_rs = 1` means "applying `r` then `s` is the same as
`s` then `r`"; `0` means the order changes the meaning; `0.5` is "the graph
never told me" (unsupervised — no diamond witnesses that pair).

Validation on structures with known answers:

* `Z_3 x Z_3` torus (`dx`, `dy` commute) plus a `swap` relation that does not
  commute with either: learned `C[dx,dy] = 0.71` vs `C[dx,swap] = C[dy,swap] = 0.29`;
* `D4` (non-abelian) — the graph contains no commuting squares, so no witness
  is found and `C` correctly stays at `0.5` instead of hallucinating structure.

Practical uses:

- **chain shortening / caching**: pairs with `C_rs > 0.9` can be sorted
  canonically, so `r,s,r,s` collapses to `r,r,s,s` and the compiled matrix can
  be cached;
- **debugging a vocabulary**: `C` tells you which relation names in the ontology
  are actually being treated as independent axes;
- **relation algebra**: `T_r o T_s^{-1}` is a new operator with no training.

`C_rs` is owned by `CurvatureLoss`, *not* by the model. The trainer therefore
registers loss-module parameters with the optimiser — a `model.parameters()`
only optimiser freezes `C_rs` at 0.5 silently (regression test
`test_loss_owned_parameters_are_optimised`).

---

## 5. Curriculum and gates (`trainer/stages`)

| stage | active losses | gate |
|---|---|---|
| `F1_nodes` | anchor, edge (operator frozen) | `cos_edge >= 0.9` |
| `F2_relations` | + contrast, isometry, cycle | `cce <= 0.02` |
| `F3_composition` | + path, curvature, commutation | `pce_k2 <= 0.10` |
| `F4_shadows` | + shadow cycle/commutation, pull, applicability, decoder, context | — |

Gates are evaluated on **held-out edges** (`--train-ratio`), and every metric
carries an explicit direction (`LOWER_IS_BETTER`) — a gate on a
"lower-is-better" metric compared with `>=` passes immediately and silently
skips the whole curriculum. Context (`eps`) is enabled exactly at F4, via
`_sync_stage_state`, which also registers the new parameters with the
optimiser.

---

## 6. Numerical rules (violating any of these reintroduced a bug)

1. `sqrt` in losses: `torch.sqrt(diff.pow(2).mean() + EPS)`. A plain
   `sqrt(mean(diff^2))` has an infinite gradient at `diff = 0`, which is the
   normal situation for a converged cycle loss.
2. `T_r^{-1}` uses the **same** generators in the **same** order with negated
   angles; reversing the order is wrong.
3. Chains fold as `acc = bmm(rot_step, acc)` iterating `step = 0..k-1`, i.e.
   `T_{r_1}` applied first.
4. Never `torch.empty() * scale` for initialisation — it keeps uninitialised
   memory (NaNs survive the multiply).
5. Parameters created during training (F4 context) must be added to the
   optimiser explicitly.
6. `torch_scatter` wheels segfault on import in some environments and that
   cannot be caught; import scatter primitives from
   `relgat_projector/utils/scatter.py` and set `RELgat_NO_TORCH_SCATTER=1`.

---

## 7. Honest limitations

- Rotations cannot shrink a concept, which hypernymy arguably does. Mitigation
  is `s_r` plus, if needed, a rank-4 term `T = s_r Rot (I + U V^T)`; not
  enabled by default.
- Curvature needs squares; wordnets are sparse in them, hence shadow squares
  and learned `C_rs` instead of hard constraints.
- The `Z_3 x Z_3` / `D4` checks validate the *geometry*; they do not yet
  validate the *semantics* of `ant`/`syn` on Słowosieć. That needs a gold
  query-rewriting set (`QR-Recall`) and is the next deliverable.
