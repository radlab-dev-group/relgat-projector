"""
WIĄZKA trainer: curriculum (F1-F4) over the WIĄZKA model.

This is a *self-contained* trainer (it does not depend on the legacy
``RelGATTrainer``) so that the new loss terms, samplers and stage manager
can be exercised end to end without touching the legacy path.  It consumes
the same dataset primitives (node embeddings, relation mapping, triplets)
that the plwordnet-milvus exporter produces.
"""

from __future__ import annotations

import random
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from relgat_projector.core.operator import build_operator
from relgat_projector.core.model.wiazka import WiazkaModel
from relgat_projector.core.loss import (
    TransportLoss,
    PathLoss,
    CurvatureLoss,
    ContrastLoss,
    StabilityLoss,
    ShadowLoss,
    DecoderLoss,
)
from relgat_projector.core.sampler import (
    PathSampler,
    QuadrupleSampler,
    NegativeSampler,
    ShadowSampler,
)
from relgat_projector.trainer.stages import StageManager
from relgat_projector.core.metrics.ranking import (
    filtered_ranking,
    stratified_edge_probe,
)
from relgat_projector.core.geometry.embedder import (
    EmbedderProfile,
    auto_op_dim,
)

#: random-pair cosine std of the space the shipped cosine knobs were tuned
#: on -- the *measured* Slowosiec profile (run F4), not a guess: the knobs
#: ``T=0.07``, ``anchor_margin=0.05``, ``knn_slack=0.02`` are the exact values
#: that run used with ``geometry_auto`` off, so the reference must be the
#: spread of that same run.  Two constants because the code selects
#: centred/raw spread the same way the run selected its chart: an as-stored
#: chart (``base_center=False``, which is how F4 ran) constrains the raw
#: spread, a centred chart the centred one.
#:
#: The previous single constant (0.047, "a sentence-embedder scale") was ~3x
#: below the reference's real spread (0.1477 raw / 0.1400 centred), which made
#: ``geometry_auto`` inflate every knob ~3x on any space: measured on the
#: same-embedder control run that put the contrast temperature at 0.1579
#: (against 0.053 at the reference ratio), the anchor margin at 0.1128, and
#: the contrast term at 75-80 % of the total loss near chance (10.5 vs
#: ln 182960 = 12.1) while the refinement destroyed the index
#: (anchor 0.999 -> 0.78, MRR 0.15 -> 0.013 in F2).
REFERENCE_COS_STD_CENTRED = 0.1400369554758072
REFERENCE_COS_STD_RAW = 0.14771294593811035

#: RMS per coordinate of the operator chart of the space the loss weights were
#: tuned on (run F4: ``base_to_op(node_emb_fixed)`` of the Slowosiec chart),
#: measured 0.02419 there against 0.99029 on ipipan/silver-retriever -- a
#: factor of 41, 1660 on the quadratic terms.  Every l2 loss (curvature,
#: shadow, cycle) is an absolute distance *inside* the chart and therefore
#: inherits this scale; the cosine losses do not.  Dividing the l2 terms by
#: ``l2_scale = rms(this chart) / REFERENCE_CHART_RMS`` puts the loss balance
#: back where the weights were tuned, and yields exactly 1.0 -- the historical
#: numbers -- on the reference space.  Same philosophy as
#: :data:`REFERENCE_COS_STD_RAW` above, applied to the terms that one works in
#: absolute units instead of cosines.
REFERENCE_CHART_RMS = 0.0242


@dataclass
class WiazkaConfig:
    #: operator dimension.  ``0`` = derive it from the base-embedding
    #: profile (:meth:`EmbedderProfile.op_dim_for_variance`), i.e. the
    #: smallest dimension that keeps ``op_dim_variance`` of the variance.
    #: A hand-picked 256 is a statement about *one* embedder: it raises
    #: ``ValueError`` on any space with at most 256 dimensions, and on a
    #: 1152-d space with effective rank ~60 it spends 4x the capacity the
    #: data uses.  ``0`` makes the method work on any dimension.
    op_dim: int = 0
    #: variance target of the automatic ``op_dim`` (see above)
    op_dim_variance: float = 0.95
    #: cap on the automatic ``op_dim``; ``0`` = no cap (the space dimension)
    op_dim_cap: int = 0
    #: subtract the column mean from the base embeddings before the PCA
    #: chart.  Anisotropy (a positive mean cosine of a random pair, e.g.
    #: +0.13 for the Słowosieć embedder) is a shared direction that every
    #: cosine is shifted by; centring removes it from the *chart* only --
    #: the export and the raw baseline still use the stored vectors, so
    #: the retriever space is untouched.  Measured: anisotropy
    #: +0.1323 -> +0.0017, identity-transport MRR +0.004.
    base_center: bool = False
    #: fraction of the *held-out* edges used to fit the served scalars
    #: (rewriting dial, relation prior, ensemble weight) instead of the
    #: supervised ones.  Fitting them on train edges is fitting them on the
    #: edges the decoder memorised: measured on F2 the whole ``alpha`` grid
    #: scored macro 0.969-0.973 -- a flat curve, so the argmax is sampling
    #: noise (shipped ``alpha`` 1.5 against a held-out optimum of 0.75).  The
    #: slice is carved out of ``eval_edges`` before anything reports on it, so
    #: the probe the metrics come from stays disjoint from the fit.
    #: ``0.0`` keeps the historical (train-edge) behaviour.
    calib_ratio: float = 0.0
    #: express the l2 losses (curvature, shadow, cycle) in the chart unit of
    #: the reference space, see :data:`REFERENCE_CHART_RMS`.  ``True`` measures
    #: the chart RMS of this space and rescales the terms by the ratio; the
    #: reference space measures 1.0 and reproduces the historical loss exactly.
    #: ``False`` keeps the raw absolute values -- the behaviour that, ported to
    #: a retriever-scale embedder, drove the total loss from 0.33 to 13.5 and
    #: collapsed the closed-loop holonomy to 0.002 (flat).
    l2_unit_auto: bool = True
    #: manual override of the measured ratio (``0.0`` = measure it); exists so
    #: a screening run can pin the unit instead of inheriting the measurement
    l2_scale: float = 0.0
    #: RMS per coordinate of this run's operator chart, measured at build time
    #: (``base_to_op(node_emb_fixed)``); persisted so a checkpoint says which
    #: unit its losses were expressed in.  ``0.0`` = not measured yet.
    chart_rms: float = 0.0
    #: scale the cosine-scale knobs off the measured random-pair cosine
    #: standard deviation of *this* space instead of assuming the SimCLR
    #: scale (0.07 temperature ~ 1.5x a 0.047 std).  The reference space is
    #: the one the knobs act on: centred when ``base_center`` is on,
    #: as-stored otherwise.
    geometry_auto: bool = False
    num_generators: int = 16
    operator_type: str = "givens"
    gat_num_layers: int = 2
    gat_heads: int = 2
    dropout: float = 0.1
    use_scale: bool = True
    use_context: bool = False
    #: master switch for the F4 connection (base-point-dependent transport).
    #: ``False`` keeps the whole curriculum but leaves ``theta(r, x) = theta_r``
    #: position-independent, which is the ablation that says what the connection
    #: is worth -- it became learnable in 0.3.1 (it could not move before), so
    #: every number quoted for it up to then was measured with it dead.
    enable_connection: bool = True
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_neg: int = 8
    epochs: int = 1
    #: fraction of edges used for training (topology + supervision);
    #: the rest is held out and never seen by the GAT (PLAN.md 6.6)
    train_ratio: float = 1.0
    device: str = "cpu"
    seed: int = 42
    # number of shadow vectors synthesised when none are supplied (PLAN.md 2.6)
    num_shadow: int = 512
    shadow_jitter: float = 0.35
    # F4: shadow vectors drawn per step (PLAN.md 2.6)
    shadow_batch: int = 64
    #: number of nearest distractors ``L_con`` scores instead of the whole
    #: key set.  A full softmax over 243k nodes spends almost all of its mass
    #: on nodes that were never candidates, so the term saturates (measured
    #: ``L_con`` ~3.4 with a lift of the transport that stopped growing); the
    #: top-K variant keeps the ranking signal that matters -- gold above its
    #: confusable rivals -- and is cheaper.  ``0`` keeps the historical full
    #: softmax, ``K >= N-1`` reproduces it exactly.
    contrast_hard_negatives: int = 0
    #: temperature of the *production* term ``L_pull`` only.  ``L_pull`` scores
    #: the exported query against ``node_emb_fixed`` -- the **raw**, uncentred
    #: retriever vectors -- while ``contrast_temperature`` is resolved for the
    #: centred chart.  The two spaces differ enormously: on this chart a random
    #: pair sits at cos 0.946 +- 0.0085 raw against 0.000 +- 0.107 centred, so
    #: the centred temperature is ~12.6x too flat here and the softmax over
    #: 183k keys is nearly uniform (measured ``L_pull`` 10.3 nats against the
    #: 11.5 a perfect retrieval can reach and the 12.1 of chance: the term is
    #: the largest single loss, 67 %% of the total, and barely differentiates).
    #: ``0`` inherits ``contrast_temperature`` (the historical behaviour); the
#: raw-spread equivalent this space measures is printed at startup as
#: ``raw-spread equivalent ...`` and persisted in ``config.json``.
    pull_temperature: float = 0.0
    #: number of nearest distractors ``L_pull`` scores instead of the whole base
    #: index; the same argument as ``contrast_hard_negatives`` but applied
    #: separately, because ``L_pull`` works on the raw index where the
    #: distractor distribution is much narrower.  ``0`` = full softmax.
    pull_hard_negatives: int = 0
    #: weight of the *separating* half of the order-dependence teacher inside
    #: ``L_curv``.  The anti-commutation term is a cap (gap must not exceed the
    #: margin), so nothing in the curvature loss ever asks two routes to
    #: differ; on this chart the cap is above the largest reachable gap and the
    #: whole term trains "stay flat" -- closed-loop holonomy 0.0003 against
    #: 0.120 on the reference.  This hinge is the mirror: where the graph
    #: proves the composed orders ``r.s`` and ``s.r`` to reach different
    #: neighbourhoods it asks for ``gap >= order_margin``
    #: (:meth:`QuadrupleSampler.order_dependence_examples`).
    #: ``0.0`` leaves the historical loss untouched.
    order_weight: float = 0.0
    #: weight of the *commuting* half: where the graph proves the two composed
    #: orders to reach the same neighbourhood, they must agree.  Without it the
    #: separating hinge alone would learn "everything differs", which is as
    #: useless as the flat transport it is here to replace.
    order_join_weight: float = 0.0
    #: route gap a proven-commutative pair is allowed to keep (see the loss)
    order_join_slack: float = 0.005
    #: smallest route gap demanded of a proven order-dependent pair, in
    #: *chart units* (the same unit as ``gap_l2``: per-coordinate RMS of the
    #: route difference divided by the chart unit, so ~``sqrt(2(1-cos))`` for
    #: unit-length vectors -- 0.05 is ~8 deg, 0.10 ~16 deg)
    order_margin: float = 0.0
    #: supervise ``C_rs`` from the *order-dependence* teacher (the same graph
    #: evidence that moves the operator) instead of the witness table.  The
    #: witness teacher counts base points where the two routes meet, which is a
    #: proxy for relation *frequency* -- on this graph ``synonimia`` (49.7 % of
    #: edges) is witnessed on most pairs, so ``C -> 0.93`` lands on exactly the
    #: pairs the operator separates most: measured ``spearman(C, order) =
    #: -0.29`` and agreement with the graph's verdict on only 26.7 % of pairs.
    #: Here ``C_rs`` is trained as ``P(order-dependent)`` from the per-pair
    #: order rate, so the matrix the 12b diagnostic reads describes the
    #: transport it is read on.  ``False`` keeps the historical witness
    #: supervision.
    commutation_from_order: bool = False
    #: weight of the order-supervised ``L_commut`` term (``False`` above keeps
    #: the historical unit weight)
    commutation_weight: float = 1.0
    #: gate the ``L_curv`` holonomy term to *genuine commutation squares*
    #: (the diamond's closing edges are the two opening ones, ``q == p`` and
    #: ``s == r``).  Measured on this graph, only **6.1 %** of the sampled
    #: diamonds are such squares; the other 93.9 % are two unrelated paths that
    #: happen to reconverge on one node.  Penalising ``||H(x_A) - x_A||`` on
    #: those does not encode "these two relations commute" -- it encodes "every
    #: route to that node is the same transport", which flattens the whole
    #: space.  That is the third, independent reason the closed-loop holonomy
    #: measured flat (0.0003): the term is not merely weak, on 19/20 of its
    #: samples it asks for the wrong thing.  ``False`` keeps every diamond.
    holonomy_squares_only: bool = False
    #: ``(a, r, s)`` examples of the order teacher drawn per step
    order_batch: int = 512
    #: feed the order teacher the base-point vectors *detached*.  The teacher
    #: is a statement about the transport -- about how two relations compose
    #: -- and a measured optimisation probe showed why that matters: fitted
    #: alone it raised the gap of the order-dependent pairs from 0.029 to 0.052
    #: (so the hinge is satisfiable, the operator can do it) but it did it by
    #: moving the node chart too, taking cos(refined, base) from 0.947 to 0.78.
    #: Detaching keeps that force on the rotations, where the curvature lives,
    #: and leaves the index to ``L_pull`` / ``L_anchor`` / ``L_knn``.
    order_detach_input: bool = True
    #: nodes probed for order dependence (see the sampler method)
    order_scan_nodes: int = 4000
    #: overlap below which the two composed successor sets count as disagreeing
    order_jaccard: float = 0.3
    #: base points needed before a pair counts at all
    order_min_support: int = 3
    #: fraction of those base points that must disagree
    order_min_fraction: float = 0.5
    #: softmax temperature of every contrastive term (``L_con``, ``L_pull``).
    #: Queries and keys are unit-normalised, so this is the only knob that sets
    #: how much gradient the *hard* negatives keep: 0.07 is the SimCLR default,
    #: a smaller value sharpens the ranking and weights the nearest distractors.
    contrast_temperature: float = 0.07
    #: queries used by the full-softmax contrast term.  Keys are *every* node,
    #: so the cost of this term is ``queries x num_nodes`` -- at batch_size
    #: 2048 and 243k nodes that is ~2 GB of logits per step.  Capping the
    #: queries keeps the exact full-softmax objective inside a sane budget.
    contrast_batch: int = 512
    #: sample the relation type uniformly instead of the edge uniformly
    balance_relations: bool = True
    #: temper exponent of relation-balanced sampling: a relation is drawn
    #: with probability ``count ** balance_power`` (renormalised).
    #: ``1.0`` = the raw edge distribution (``synonimia`` is half the graph, so
    #: the ~30 relations with fewer than 100 edges are never sampled and their
    #: transport stays at initialisation); ``0.0`` = uniform over relations
    #: (those relations train, but the one relation holding 50 % of the graph
    #: sees 1/56 of the batches it needs -- measured: ``synonimia`` MRR fell
    #: 0.59 -> 0.35 while rare relations improved).  ``0.5`` is the standard
    #: tempered-sampling compromise: synonimia 16.3 % of every batch (against
    #: 1.8 % uniform / 49.7 % raw) and the 14 relations with <100 edges 3.1 %
    #: (against 25 % / 0.15 %).
    balance_power: float = 0.5
    #: relation-aware softmax attention in the GAT aggregation (PLAN.md 3.2a).
    #: With it off, messages are summed over *all* incoming edges: the node
    #: vector is then an unnormalised blend of every relation type it touches,
    #: which measurably costs ``synonimia`` recall on a relation-skewed graph.
    gat_attention: bool = True
    #: initialise base_to_op from the principal subspace of the base embeddings
    #: and tie to_retriever to its transpose (see WiazkaModel).  Without it the
    #: export into the retriever space is an untrained random map until F4.
    proj_pca: bool = True
    #: probe size for evaluation; scored against all nodes, so keep this
    #: moderate (cost is O(probe x num_nodes))
    probe_size: int = 1024
    #: ``L_anchor = weight * relu((1 - cos(refined, base)) - margin)``.
    #: The refinement is what lets the transport beat plain similarity, but a
    #: refinement that moves too far destroys the neighbourhood structure the
    #: base embeddings already have (measured: identity transport over the
    #: refined index scores *below* the raw base ranking), so this is the knob
    #: that trades graph signal against index quality.
    anchor_weight: float = 0.5
    anchor_margin: float = 0.05
    #: weight of ``L_knn`` -- the *structure*-preserving counterpart of
    #: ``L_anchor``.  ``L_anchor`` keeps every vector near its own base
    #: position, but MRR is a ranking measure: what has to survive refinement
    #: is the *neighbourhood order*, not the absolute position.  ``L_knn``
    #: samples pairs that are near in the operator chart (the space the refined
    #: index actually lives in) and only penalises the refinement when it pulls
    #: them apart by more than ``knn_slack``.
    #: Measured on the Slowosiec: refinement moved the index so that identity
    #: transport over it scored 0.065 MRR against 0.091 for the raw base
    #: vectors, and morphology relations lost up to 0.25 MRR, while the same
    #: refinement gained 0.5 on semantic-role relations.
    knn_weight: float = 0.0
    #: neighbours (in the operator chart) kept per anchor node
    knn_k: int = 8
    #: anchor nodes whose base neighbourhoods are preserved
    knn_anchors: int = 4096
    #: tolerated drop of ``cos`` on a base-neighbour pair
    knn_slack: float = 0.02
    #: pairs sampled from the preserved neighbourhoods per training step
    knn_batch: int = 512
    #: weight of ``L_path`` (composition over chains of k >= 2).  PCE grows
    #: ~linearly with the hop count unless composition is pushed explicitly.
    path_weight: float = 1.0
    #: the relation decoder is trained with class weights proportional to
    #: ``count^-decoder_balance``: the Słowosieć relation distribution puts
    #: 49.7 % of the edges on ``synonimia`` and ~30 relations have fewer than
    #: 100 edges, so a plain cross-entropy head answers the frequent relations
    #: and its macro accuracy (the honest average) collapses.  0.5 matches the
    #: tempered edge sampling and 1.0 is full inverse-frequency.
    decoder_balance: float = 0.5
    #: label smoothing of the decoder CE (keeps the head from saturating on the
    #: 50 % ``synonimia`` class)
    decoder_label_smoothing: float = 0.05
    #: learning-rate multiplier of the per-relation rewriting dial.  It is 56
    #: scalars whose gradient comes from ``L_pull`` only, i.e. from the
    #: ``pull_batch`` queries of a step -- a handful per relation -- so at the
    #: base lr the sign flips more often than it points anywhere: after a full
    #: 35-epoch run every dial had moved 0.04 in logit space and sat at
    #: ``lam ~ 0.81``, the relation-blind answer it started with.
    dial_lr_scale: float = 20.0
    #: learning-rate multiplier of the commutation matrix ``C_rs``.  It is
    #: ``num_rel ** 2`` bounded logits taught by a *sample* of the witness
    #: table (16 pairs a step against ~2900 negative ordered pairs), so at the
    #: base lr its entries barely leave 0.5 inside one run: measured on the
    #: 12-epoch same-embedder screen every cell still sat at 0.500 +- 0.000,
    #: i.e. the commutation structure the method exists to learn never moved.
    comm_lr_scale: float = 20.0
    #: weight of ``L_pull`` -- the *production* term: an exported transformed
    #: query must retrieve the right node from the base index (what the vector
    #: database actually holds).  Scored with the same full softmax as the
    #: ``prod_*`` eval protocol, so this is the loss that closes the gap
    #: between the graph metric (``eval/mrr``) and the product metric.
    pull_weight: float = 1.0
    #: queries for ``L_pull``; keys are every node of the *base* index, so the
    #: cost is ``pull_batch x num_nodes x base_dim``.
    pull_batch: int = 256
    # learning rate schedule: {constant, cosine, linear} + warmup fraction
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    # where the best-checkpoint is written while training (set by the runner)
    save_dir: str = "wiazka-models"


#: candidate weights of the transport evidence in the relation ensemble,
#: ``z(head) + w * z(transport)``
ENSEMBLE_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

#: global rewriting strengths scanned by ``eval/prod_mrr_*_bestlam``.
#: ``0.0`` is "never rewrite" (i.e. the raw baseline), so the maximum over
#: this grid is a lower bound on what the *per-relation* dial can reach --
#: the dial can always imitate a global constant.
LAMBDA_PROBE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def spread_sample(rows: Sequence, n: int, seed: int = 0) -> List:
    """``n`` uniformly chosen elements of ``rows``, deterministically.

    Used wherever a calibration takes "a few examples per relation".  The edge
    list is ordered by source id, so the first 96 ``synonimia`` edges are 96
    adjacent nodes of one frequency band, and a dial fitted on them describes
    that neighbourhood rather than the relation.  A *fixed stride* is not the
    answer either -- a stride commensurate with a period in the data aliases
    (every 20th edge of a list that interleaves ten relations is the same
    relation) -- so this draws with a seeded generator and returns the indices
    sorted, which is uniform, deterministic and reproducible.
    """
    total = len(rows)
    n = int(n)
    if n <= 0 or total <= n:
        return list(rows)
    gen = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(total, generator=gen)[:n].tolist()
    return [rows[i] for i in sorted(idx)]


#: candidate strengths of the training-frequency correction of the relation head
def smooth_curve(values: Sequence[float], window: int = 3) -> List[float]:
    """Centred moving average of a scanned curve, ends included.

    A grid scan chooses the argmax of a *noisy* curve: the dial is fitted from
    ~100 probe edges per relation, so two neighbouring grid points differ by a
    handful of probes and an unsmoothed argmax locks onto spikes that are pure
    sampling noise.  Averaging kills single-point spikes, and because a centred
    average preserves monotone order it does not drag an optimum away from a
    boundary: a curve that only decreases still peaks at its first grid point,
    so "do not rewrite" stays reachable.
    """
    v = [float(x) for x in values]
    n = len(v)
    half = max(0, int(window) // 2)
    if n <= 1 or half == 0:
        return v
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


PRIOR_STRENGTH_GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def macro_relation_accuracy(pred: torch.Tensor, rel: torch.Tensor,
                            num_rel: int) -> float:
    """Mean over relations of top-1 accuracy, counting only seen relations."""
    correct = (pred == rel).float()
    per_rel_sum = torch.zeros(num_rel)
    per_rel_n = torch.zeros(num_rel)
    per_rel_sum.index_add_(0, rel.cpu(), correct.cpu())
    per_rel_n.index_add_(0, rel.cpu(), torch.ones_like(correct).cpu())
    seen = per_rel_n > 0
    if not bool(seen.any()):
        return 0.0
    return float((per_rel_sum[seen] / per_rel_n[seen]).mean().item())


def fit_ensemble_weight(
    head: torch.Tensor,
    transport: torch.Tensor,
    rel: torch.Tensor,
    num_rel: int,
    grid: Sequence[float] = ENSEMBLE_WEIGHT_GRID,
) -> float:
    """Grid argmax of macro accuracy for ``z(head) + w * z(transport)``.

    Ties go to the *smaller* weight: when the transport adds nothing the head is
    kept alone rather than diluted by it.
    """
    zh = WiazkaModel._row_zscore(head)
    zt = WiazkaModel._row_zscore(transport)
    best_alpha, best_acc = float(grid[0]), -1.0
    for alpha in grid:
        acc = macro_relation_accuracy(
            (zh + float(alpha) * zt).argmax(dim=-1), rel, num_rel)
        if acc > best_acc + 1e-9:
            best_alpha, best_acc = float(alpha), acc
    return best_alpha


def _build_adjacency(
    edges: List[Tuple[int, int, int]], num_nodes: int
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    out_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    in_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    for s, d, r in edges:
        out_adj[s].append((d, r))
        in_adj[d].append((s, r))
    return out_adj, in_adj


def principal_projection(
    node_emb: torch.Tensor, dim: int, device: str = "cpu", seed: int = 42,
    n_iter: int = 4,
) -> torch.Tensor:
    """Orthonormal ``[dim, base_dim]`` basis of the base embedding subspace.

    The operator and the export share this basis, so at initialisation
    ``to_retriever(base_to_op(x))`` is an orthogonal projection: the transport
    starts inside a faithful low-dimensional copy of the retriever space and
    cosine search in the exported vectors is meaningful from step 0 (a random
    linear export made every pre-F4 QR-Recall measurement exactly 0.0).
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if dim >= min(node_emb.shape):
        raise ValueError(
            f"cannot take a {dim}-dimensional subspace of a "
            f"{tuple(node_emb.shape)} matrix"
        )
    x = node_emb.to(device)
    x = x - x.mean(dim=0, keepdim=True)
    q = min(dim, min(x.shape))
    try:
        # pca_lowrank draws its random test matrix from the global generator:
        # seed it so the projection (and therefore the run) is reproducible
        torch.manual_seed(seed)
        _, _, V = torch.pca_lowrank(x, q=q, center=False, n_iter=n_iter)
    except Exception:  # pragma: no cover - degenerate/tiny datasets
        V = torch.randn(x.shape[1], q, device=x.device)
    W = V.t()[:dim].contiguous()
    # re-orthonormalise: the export is the transpose, so orthonormality of the
    # rows is what makes the round trip an orthogonal projection
    return torch.linalg.qr(W.t())[0].t().contiguous()


class WiazkaTrainer:
    def __init__(
        self,
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edges: List[Tuple[int, int, str]],
        cfg: Optional[WiazkaConfig] = None,
        shadow_vectors=None,
    ):
        self.cfg = cfg or WiazkaConfig()
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.rel2idx = dict(rel2idx)
        self.num_rel = len(self.rel2idx)

        node_ids = sorted(node2emb.keys())
        self.id2idx = {nid: i for i, nid in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        D_in = int(next(iter(node2emb.values())).numel())

        self.node_emb = torch.stack(
            [torch.as_tensor(node2emb[nid], dtype=torch.float32) for nid in node_ids]
        )
        # one device copy shared by the eval baselines (an `.to(device)` inside
        # evaluate() re-uploaded 1.1 GB on every evaluation)
        self._node_emb_dev = self.node_emb.to(self.cfg.device)

        # ---------------------------------------------------- base geometry
        # Everything the method assumes about the base space is measured
        # here once, so no constant below has to be a guess about an
        # embedder nobody looked at (see EmbedderProfile).
        self.base_profile = EmbedderProfile.from_embeddings(
            self._node_emb_dev, seed=self.cfg.seed
        )
        self.cfg.op_dim = self._resolve_op_dim(self.base_profile, self.cfg)
        self.geometry = self._resolve_geometry(self.base_profile, self.cfg)
        print(
            f"[geometry] base space: dim={self.base_profile.dim} "
            f"vectors={self.base_profile.num_vectors} "
            f"norm={self.base_profile.norm_mean:.4f}+-{self.base_profile.norm_std:.4f}"
            f"{' (unit-norm)' if self.base_profile.unit_norm else ' (NOT unit-norm)'}\n"
            f"[geometry] random-pair cos "
            f"{self.base_profile.pair_cos_mean:+.4f}+-{self.base_profile.pair_cos_std:.4f} "
            f"(centred {self.base_profile.pair_cos_mean_centred:+.4f}"
            f"+-{self.base_profile.pair_cos_std_centred:.4f}), "
            f"anisotropy={self.base_profile.anisotropy:.4f}, "
            f"effective rank={self.base_profile.effective_rank:.1f}\n"
            f"[geometry] duplicates (1-NN cos>0.999): "
            f"{self.base_profile.dup_1nn_fraction:.4f} "
            f"(mean 1-NN cos {self.base_profile.dup_1nn_cos:.4f}) -- "
            f"the raw baseline is inflated by this much\n"
            f"[geometry] op_dim={self.cfg.op_dim} "
            f"(EV "
            + ", ".join(f"{k}:{v:.4f}" for k, v in
                        sorted(self.base_profile.explained_variance.items(), key=lambda kv: int(kv[0])))
            + f"), base_center={self.cfg.base_center}, "
            f"cosine knobs T={self.geometry['contrast_temperature']:.4f} "
            f"(L_pull T={float(self.cfg.pull_temperature) if float(self.cfg.pull_temperature) > 0.0 else self.geometry['contrast_temperature']:.4f} "
            f"on the raw index, std {self.geometry.get('raw_cos_std', 0.0):.4f}, "
            f"raw-spread equivalent {self.geometry.get('pull_temperature_suggested', 0.0):.4f}) "
            f"anchor_margin={self.geometry['anchor_margin']:.4f} "
            f"knn_slack={self.geometry['knn_slack']:.4f}"
        )

        # the chart origin: a mean-centred base.  ``base_to_op`` becomes affine
        # and the export adds the centre back, so the retriever index stays
        # exactly the stored vectors.
        center = None
        if self.cfg.base_center:
            center = self._node_emb_dev.mean(dim=0).detach()
            print(
                f"[geometry] chart centred on ||mu||={float(center.norm()):.4f} "
                f"({float(center.norm()) / max(1e-9, self.base_profile.norm_mean):.1%} "
                f"of a typical vector); anisotropy "
                f"{self.base_profile.anisotropy:.4f} -> "
                f"{abs(self.base_profile.pair_cos_mean_centred):.4f}"
            )
        self._base_center = center

        # compact edges
        self.edges: List[Tuple[int, int, int]] = [
            (self.id2idx[s], self.id2idx[d], self.rel2idx[r])
            for (s, d, r) in edges
            if s in self.id2idx and d in self.id2idx and r in self.rel2idx
        ]
        # PLAN.md 6.6: the GAT topology and the supervised edges must be an
        # explicit, reproducible split -- otherwise held-out edges leak into
        # message passing and every "eval" metric is measured on training data.
        self.eval_edges: List[Tuple[int, int, int]] = []
        ratio = float(self.cfg.train_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0, 1], got {ratio}")
        if ratio < 1.0 and len(self.edges) > 1:
            self.edges, self.eval_edges = self._split_edges(ratio)
            print(
                f"[split] train edges={len(self.edges)} "
                f"held-out edges={len(self.eval_edges)}"
            )
        # ``calib_ratio`` splits the held-out set in two: a slice the served
        # scalars are fitted on and a remainder the metrics are reported on.
        # It has to happen here, before ``_targets_by_src_rel`` and the probe
        # pools exist, so that a calibration edge can never be scored.
        self.calib_edges: List[Tuple[int, int, int]] = []
        calib = float(self.cfg.calib_ratio)
        if calib > 0.0:
            if not 0.0 < calib < 1.0:
                raise ValueError(f"calib_ratio must be in (0, 1), got {calib}")
            if not self.eval_edges:
                raise ValueError(
                    "calib_ratio > 0 needs a held-out split: set train_ratio < 1"
                )
            self.calib_edges, self.eval_edges = self._split_edges(
                calib, source=self.eval_edges, seed=self.cfg.seed + 101)
            print(
                f"[split] calibration edges={len(self.calib_edges)} "
                f"(fitted scalars) / scored held-out edges={len(self.eval_edges)}"
            )
        self.topology_endpoint_coverage = self._coverage_report()

        # Precomputed once: rebuilding these from a Python list of 324k tuples
        # on every training step was pure waste.
        self.edge_tensor = torch.tensor(self.edges, dtype=torch.long)
        self.edge_index = self.edge_tensor[:, [0, 1]].t().contiguous()
        self.edge_type = self.edge_tensor[:, 2].contiguous()
        # (src, rel) -> every true target: used both for the filtered MRR
        # protocol and for false-negative masking in the contrast loss.
        self._targets_by_src_rel: Dict[Tuple[int, int], List[int]] = {}
        for s, d, r in self.edges:
            self._targets_by_src_rel.setdefault((s, r), []).append(d)
        for s, d, r in self.eval_edges:
            self._targets_by_src_rel.setdefault((s, r), []).append(d)
        # node -> relation types it really participates in: the applicability
        # head may only be taught "not applicable" for a relation that is truly
        # absent from that node's neighbourhood
        # edges grouped by relation id, for relation-balanced sampling
        rel_order = torch.argsort(self.edge_tensor[:, 2])
        self._edge_tensor_by_rel = self.edge_tensor[rel_order].contiguous()
        self._rel_counts = torch.bincount(
            self._edge_tensor_by_rel[:, 2], minlength=self.num_rel
        )
        self._rel_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(self._rel_counts[:-1], dim=0),
            ]
        )
        #: inverse-CDF of the tempered relation sampling distribution
        #: (``count ** balance_power``, renormalised; absent relations weight 0
        #: so they can never be drawn).  ``searchsorted`` on a tie returns the
        #: *first* index, which is the preceding relation that does carry
        #: weight, so a zero-weight entry cannot be selected.
        power = float(self.cfg.balance_power)
        weights = self._rel_counts.to(torch.float64).clamp_min(0.0)
        if power != 1.0:
            weights = weights.pow(power)
        weights[self._rel_counts == 0] = 0.0
        total = float(weights.sum())
        if total <= 0.0:
            weights = self._rel_counts.to(torch.float64)
            total = float(weights.sum())
        self._rel_sample_cdf = torch.cumsum(weights / total, dim=0)
        self._rels_by_node: Dict[int, set] = {}
        for s, d, r in self.edges:
            self._rels_by_node.setdefault(s, set()).add(r)
            self._rels_by_node.setdefault(d, set()).add(r)

        # operator
        self.operator = build_operator(
            self.cfg.operator_type,
            dim=self.cfg.op_dim,
            num_rel=self.num_rel,
            num_generators=self.cfg.num_generators,
            use_scale=self.cfg.use_scale,
            use_context=False,  # context is enabled by the stage manager at F4
        )

        # a principal subspace only exists below the base dimensionality; an
        # operator wider than the embeddings keeps its random projection
        proj_weight = None
        if self.cfg.proj_pca and self.cfg.op_dim < min(self.node_emb.shape):
            proj_weight = principal_projection(
                self.node_emb, self.cfg.op_dim, device=self.cfg.device,
                seed=self.cfg.seed,
            )
        self.model = WiazkaModel(
            node_emb=self.node_emb,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_rel=self.num_rel,
            operator=self.operator,
            gat_num_layers=self.cfg.gat_num_layers,
            gat_heads=self.cfg.gat_heads,
            dropout=self.cfg.dropout,
            use_applicability=True,
            use_decoder=True,
            attention=self.cfg.gat_attention,
            proj_weight=proj_weight,
            tie_projections=True,
            center_mean=center,
            # an orthonormal chart cannot survive AdamW: the weights are ~1/sqrt(D)
            # in magnitude while an Adam step is ~lr, so after a few hundred steps
            # W W^T was 0.22 away from the identity and every exported query was
            # geometric garbage (norms 1.5 -> 8.3, QR-Recall 0.0).  Freezing keeps
            # base<->operator an exact isometry; export_proj is the learned escape
            # hatch, gated from L_pull.
            freeze_projection=proj_weight is not None,
        ).to(self.cfg.device)

        # ------------------------------------------------- chart loss unit
        # measured here, before the losses below are constructed: it is the
        # only place that knows both the chart and the weights tuned for it
        self.cfg.chart_rms = self._chart_rms(self.model)
        if float(self.cfg.l2_scale) > 0.0:
            l2_scale = float(self.cfg.l2_scale)
            unit_source = "override"
        elif self.cfg.l2_unit_auto:
            l2_scale = self.cfg.chart_rms / REFERENCE_CHART_RMS
            unit_source = "measured"
        else:
            l2_scale = 1.0
            unit_source = "off (absolute chart units)"
        self.cfg.l2_scale = float(l2_scale)
        print(
            f"[geometry] chart rms={self.cfg.chart_rms:.5f} "
            f"(reference {REFERENCE_CHART_RMS}) -> l2 unit={l2_scale:.4f} "
            f"[{unit_source}]: curvature/shadow/cycle losses divided by this, "
            f"so 1.0 reproduces the reference loss exactly"
        )

        # samplers
        out_adj, in_adj = _build_adjacency(self.edges, self.num_nodes)
        deg = np.array(
            [len(out_adj.get(i, [])) + len(in_adj.get(i, [])) for i in range(self.num_nodes)],
            dtype=np.float64,
        )
        self.neg_sampler = NegativeSampler(
            num_nodes=self.num_nodes, degrees=deg, rng=random.Random(self.cfg.seed)
        )
        self.path_sampler = PathSampler(
            adjacency=out_adj,
            rel_ids=list(range(self.num_rel)),
            num_nodes=self.num_nodes,
            max_depth=4,
            rng=random.Random(self.cfg.seed + 1),
        )
        self.quad_sampler = QuadrupleSampler(
            out_adj=out_adj, in_adj=in_adj, num_nodes=self.num_nodes,
            num_rel=self.num_rel,
            rng=random.Random(self.cfg.seed + 2),
        )

        # losses
        self.transport_loss = TransportLoss()
        self.path_loss = PathLoss()
        # CurvatureLoss owns C_rs as a parameter, so it has to live on the
        # training device like the model does; left on the CPU it raised
        # "indices should be either on cpu or on the same device" the first
        # time the curvature term indexed C_rs with CUDA relation ids.
        # margin_auto: the anti-commutation margin is an l2 distance between
        # two transported vectors, so a fixed constant is only meaningful for
        # one particular vector length.  At the default 0.25 with chart
        # vectors of norm ~1.3 the term is *never* violated (measured route
        # gaps 0.02-0.06) and the anti-commutation half of the curvature loss
        # silently trains nothing.
        self.curvature_loss = CurvatureLoss(
            self.num_rel, margin_auto=True, l2_scale=l2_scale,
            order_margin=self.cfg.order_margin,
            order_weight=self.cfg.order_weight,
            order_join_weight=self.cfg.order_join_weight,
            order_join_slack=self.cfg.order_join_slack,
        ).to(self.cfg.device)
        # the order-dependence teacher is taught from what the graph *proves*,
        # so it is read off the graph once, here: a run with a non-zero order
        # weight is self-contained, there is no hand-written pair list to ship
        # (and no invented non-commutativity either)
        self._order_pairs: List[Tuple[int, int]] = []
        self._order_rows: List[Tuple[int, int, int, int]] = []
        #: ``((r, s), rate)`` supervision of ``C_rs`` (see
        #: ``WiazkaConfig.commutation_from_order``); empty unless enabled
        self._comm_targets: List[Tuple[Tuple[int, int], float]] = []
        if ((self.cfg.order_weight > 0 or self.cfg.order_join_weight > 0
             or self.cfg.commutation_from_order)
                and self.cfg.order_margin >= 0):
            self._order_pairs, self._order_rows = (
                self.quad_sampler.order_dependence_examples(
                    scan_nodes=self.cfg.order_scan_nodes,
                    jaccard=self.cfg.order_jaccard,
                    min_support=self.cfg.order_min_support,
                    min_fraction=self.cfg.order_min_fraction,
                )
            )
            n_dep = sum(1 for row in self._order_rows if row[3] == 1)
            names = {v: k for k, v in self.rel2idx.items()}
            if self.cfg.order_weight > 0 or self.cfg.order_join_weight > 0:
                print(
                    f"[curvature] order teacher: {len(self._order_rows)} labelled "
                    f"example(s) over {len(self._order_pairs)} pair(s) "
                    f"({n_dep} order-dependent / "
                    f"{len(self._order_rows) - n_dep} commutative), "
                    f"margin={self.cfg.order_margin:.4f} "
                    f"sep_w={self.cfg.order_weight:.3f} "
                    f"join_w={self.cfg.order_join_weight:.3f}"
                )
                for pair in self._order_pairs:
                    print("    " + " x ".join(
                        names.get(r, str(r)) for r in pair))
            if self.cfg.commutation_from_order:
                # the SAME scan, but every scored pair and its rate, mixed ones
                # included -- see ``QuadrupleSampler.order_pair_rates``
                self._comm_targets = self.quad_sampler.order_pair_rates(
                    min_support=self.cfg.order_min_support)
                n_mixed = sum(1 for _, rate in self._comm_targets
                              if 0.0 < rate < 1.0)
                mean_rate = (sum(rate for _, rate in self._comm_targets)
                             / max(1, len(self._comm_targets)))
                print(
                    f"[curvature] C_rs <- order teacher: {len(self._comm_targets)} "
                    f"pair(s) supervised as P(order-dependent) "
                    f"({n_mixed} mixed, mean rate {mean_rate:.3f}), "
                    f"weight={self.cfg.commutation_weight:.3f} "
                    f"(instead of the witness table)"
                )
        # the resolved geometry, not the raw config: with ``geometry_auto`` the
        # numbers below are the configured ones scaled to this base space
        self.contrast_loss = ContrastLoss(self.geometry["contrast_temperature"])
        # ``L_pull`` scores the exported query against the raw index, so it gets
        # its own temperature (see ``WiazkaConfig.pull_temperature``)
        self.pull_contrast_loss = ContrastLoss(
            float(self.cfg.pull_temperature)
            if float(self.cfg.pull_temperature) > 0.0
            else float(self.geometry["contrast_temperature"]))
        self.stability_loss = StabilityLoss(
            anchor_weight=self.cfg.anchor_weight,
            anchor_margin=self.geometry["anchor_margin"],
            l2_scale=l2_scale,
        )
        self.shadow_loss = ShadowLoss(l2_scale=l2_scale)
        self.decoder_loss = DecoderLoss(
            label_smoothing=self.cfg.decoder_label_smoothing,
        )
        # class weights from the TRAIN edge counts (held-out edges never touch
        # the decoder, so they never enter the prior either)
        counts = torch.zeros(self.num_rel, dtype=torch.float64)
        for s_, d_, r_ in self.edges:
            counts[r_] += 1.0
        counts = counts.clamp_min(1.0)
        power = float(self.cfg.decoder_balance)
        weights = counts.pow(-power)
        weights = (weights / weights.mean()).to(torch.float32)
        self._decoder_class_weights = weights
        #: raw train-edge counts per relation: the prior the relation
        #: classifier has to correct for when queries are relation-uniform
        self._relation_counts = counts.to(torch.float32)
        # the loss modules are not part of ``self.model``, so they are not
        # moved by ``WiazkaModel.to(device)``: the weights have to follow the
        # training device explicitly
        self.decoder_loss.set_class_weights(weights.to(self.cfg.device))

        # shadow manifold: real text vectors from the retriever index.  When
        # none are provided we synthesise them by jittering the base node
        # embeddings, which approximates "a text near a node but not the node".
        if shadow_vectors is None:
            base_np = self.node_emb.detach().cpu().numpy()
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.integers(0, len(base_np), size=self.cfg.num_shadow)
            # Additive per-coordinate noise is wrong in a high-dimensional
            # cosine space: with D = 1152 and sigma = 0.35 the perturbation has
            # norm ~12 against a base vector of norm ~1, so the "shadow" is
            # almost orthogonal noise rather than a text close to a node, and
            # every off-manifold loss is measured on garbage.  Rotate the base
            # vector by a bounded angle instead, keeping its norm.
            anchor = base_np[idx]
            noise = rng.standard_normal(anchor.shape).astype(np.float32)
            anchor_n = anchor / np.linalg.norm(anchor, axis=-1, keepdims=True).clip(
                1e-8
            )
            # remove the component along the anchor, then renormalise: the two
            # parts are orthogonal unit directions, so the mix is an exact
            # rotation by ``theta`` with cos(theta) = 1 - shadow_jitter
            noise -= (noise * anchor_n).sum(-1, keepdims=True) * anchor_n
            noise /= np.linalg.norm(noise, axis=-1, keepdims=True).clip(1e-8)
            keep = float(np.clip(1.0 - self.cfg.shadow_jitter, 0.0, 1.0))
            theta = float(np.arccos(keep))
            direction = np.cos(theta) * anchor_n + np.sin(theta) * noise
            scale = np.linalg.norm(anchor, axis=-1, keepdims=True)
            shadow_np = (direction * scale).astype(np.float32)
        else:
            shadow_np = np.asarray(shadow_vectors, dtype=np.float32)
        self.shadow_sampler = ShadowSampler(shadow_np, rng=self.cfg.seed)
        self._shadow_cache = torch.as_tensor(shadow_np, dtype=torch.float32).to(
            self.cfg.device
        )

        # NOTE: CurvatureLoss owns the commutation matrix C_rs as a parameter,
        # so the optimizer must include the loss modules, not only the model.
        # Omitting them silently freezes C_rs at its initialisation.
        #
        # Weight decay must NOT touch the operator (PLAN.md 2.7): decaying the
        # rotation angles theta, the log-scales or the generator basis shrinks
        # the transport toward the identity and destroys the structure the
        # curriculum is learning.  Only biases and weight matrices get decay.
        op_param_ids = {id(p) for p in self.operator.parameters()}
        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in op_param_ids:
                no_decay_params.append(p)
            elif p.ndim <= 1:  # biases, gates, heads' last layers' biases
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        # C_rs is a bounded classification (logits -> sigmoid in [0, 1])
        # taught by witness supervision, not a weight matrix: decay would pull
        # it toward "everything unknown" and fight the supervision
        for p in self.curvature_loss.parameters():
            if p.requires_grad:
                no_decay_params.append(p)
        assert not any(
            id(p) in op_param_ids for p in decay_params
        ), "operator parameters must not be weight-decayed"
        # the rewriting dial is a 56-dim vector of sigmoid gates: same loss,
        # same update rule, but it needs a larger step than a weight matrix to
        # leave its initialisation inside one run (see WiazkaConfig.dial_lr_scale)
        dial_ids = {
            id(p) for n, p in self.model.named_parameters()
            if n == "transport_dial"
        }
        no_decay_params = [p for p in no_decay_params if id(p) not in dial_ids]
        dial_params = [p for p in self.model.parameters()
                       if id(p) in dial_ids and p.requires_grad]
        # C_rs is a bounded classification taught by a witness sample, not a
        # weight matrix: the same under-training story as the dial, one
        # param group with a larger step
        comm_ids = {id(self.curvature_loss.comm_logit)}
        no_decay_params = [p for p in no_decay_params if id(p) not in comm_ids]
        comm_params = [p for p in self.curvature_loss.parameters()
                       if id(p) in comm_ids and p.requires_grad]
        groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if dial_params:
            groups.append({
                "params": dial_params,
                "weight_decay": 0.0,
                "lr": self.cfg.lr * float(self.cfg.dial_lr_scale),
            })
        if comm_params:
            groups.append({
                "params": comm_params,
                "weight_decay": 0.0,
                "lr": self.cfg.lr * float(self.cfg.comm_lr_scale),
            })
        self.optimizer = torch.optim.AdamW(groups, lr=self.cfg.lr)
        self._scheduler = None
        self.stage_manager = StageManager()
        self.history: List[Dict] = []
        self._sync_stage_state()

    # ------------------------------------------------------------- sampling
    def _build_scheduler(self, steps_per_epoch: int, epochs: int) -> None:
        total = max(1, steps_per_epoch * epochs)
        warmup = max(1, int(self.cfg.warmup_ratio * total))
        if self.cfg.lr_scheduler == "cosine":
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total - warmup)
            )
        elif self.cfg.lr_scheduler == "linear":
            decay = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.05,
                total_iters=max(1, total - warmup),
            )
        else:
            decay = None
        if decay is None or warmup >= total:
            self._scheduler = decay
            return
        warm = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup
        )
        self._scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warm, decay], milestones=[warmup]
        )

    def _sync_stage_state(self) -> None:
        """Apply the current stage's structural switches to operator + optimizer.

        F4 turns on base-point-dependent transport (PLAN.md 1.3):
        ``theta(r, x) = theta_r + eps * P(relu(W x))``.  The hyper-network is
        created *at that moment*, so its parameters have to be registered with
        the optimizer as well -- otherwise context would exist but never train.
        """
        wanted = (bool(self.stage_manager.stage().enable_context)
                  and bool(self.cfg.enable_connection))
        enable = getattr(self.operator, "enable_context", None)
        if enable is not None and bool(self.operator.use_context) != wanted:
            enable(wanted)
        # F1 declares ``freeze_operator`` (the curriculum learns the node
        # refinement before any transport exists).  The flag existed but was
        # never applied, so the operator trained from step one and the stage
        # was decorative.
        frozen = bool(self.stage_manager.freeze_operator)
        for p in self.operator.parameters():
            p.requires_grad_(not frozen)
        known = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group["params"]
        }
        fresh = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in known and
            not frozen
        ]
        if fresh:
            self.optimizer.add_param_group({"params": fresh})

    def _split_edges(
        self,
        ratio: float,
        source: Optional[List[Tuple[int, int, int]]] = None,
        seed: Optional[int] = None,
    ):
        """Deterministic, position-disjoint split of an edge list.

        Splits on *positions*, so when the graph contains duplicate
        ``(src, dst, rel)`` triplets (multi-graph semantics) no position is
        used for training and held out at the same time.  ``source`` defaults
        to the training edges, which is what the historical caller wants.
        """
        rows = self.edges if source is None else source
        seed = self.cfg.seed if seed is None else seed
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        n_train = max(1, int(round(ratio * len(order))))
        train_ids = set(order[:n_train])
        train = [rows[i] for i in range(len(rows)) if i in train_ids]
        eval_ = [rows[i] for i in range(len(rows)) if i not in train_ids]
        return train, eval_

    def _coverage_report(self) -> float:
        """Share of held-out edges whose endpoints exist in the GAT topology."""
        if not self.eval_edges:
            return 1.0
        nodes = set()
        for s, d, _ in self.edges:
            nodes.add(s)
            nodes.add(d)
        visible = sum(1 for s, d, _ in self.eval_edges if s in nodes and d in nodes)
        coverage = visible / max(1, len(self.eval_edges))
        print(f"[split] held-out edges reachable in topology: {coverage:.1%}")
        return coverage

    @staticmethod
    def _resolve_op_dim(profile: EmbedderProfile, cfg: "WiazkaConfig") -> int:
        """Resolve the operator dimension from the configured value.

        ``op_dim == 0`` means "derive it from the space"; a positive value is
        kept as configured.  The check that used to live in ``__init__``
        raised when the operator was wider than the base embeddings, i.e. the
        method *could not run at all* on an embedder with at most 256
        dimensions.  Deriving the dimension from the measured spectrum makes
        that case ordinary: the operator is then the smallest principal
        subspace that keeps ``op_dim_variance`` of the variance.
        """
        configured = int(cfg.op_dim)
        if configured > 0:
            return configured
        cap = int(cfg.op_dim_cap) or profile.dim
        return auto_op_dim(profile, target=float(cfg.op_dim_variance), cap=cap)

    @staticmethod
    def _resolve_geometry(profile: EmbedderProfile, cfg: "WiazkaConfig") -> dict:
        """Cosine knobs expressed in the units of *this* base space.

        ``anchor_margin``, ``knn_slack`` and ``contrast_temperature`` are all
        statements about a cosine difference.  In a space whose random pairs
        already sit at cos 0.13 +- 0.05, a slack of 0.02 is 0.4 of a standard
        deviation; in an isotropic space with the same 0.02 it is a different
        constraint.  ``geometry_auto`` rescales every one of them by
        ``std(this space) / std(reference space)`` so the shipped defaults are
        reproduced on the reference space and travel as a *ratio* elsewhere.
        The reference is the centred space when the chart is centred, because
        that is the space the knobs then constrain.
        """
        if not cfg.geometry_auto:
            return {
                "contrast_temperature": float(cfg.contrast_temperature),
                "anchor_margin": float(cfg.anchor_margin),
                "knn_slack": float(cfg.knn_slack),
                "cos_std_reference": None,
                "cos_std_used": None,
                "pull_temperature_suggested": float(cfg.contrast_temperature),
                "raw_cos_std": float(profile.pair_cos_std),
            }
        # the reference spread is selected the same way the chart is: a
        # centred chart constrains centred cosines, an as-stored chart the
        # raw ones -- using the wrong one would offset every knob by the
        # anisotropy of the reference
        used = (profile.pair_cos_std_centred if cfg.base_center
                else profile.pair_cos_std)
        reference = (REFERENCE_COS_STD_CENTRED if cfg.base_center
                     else REFERENCE_COS_STD_RAW)
        scale = float(used) / float(reference)
        return {
            "contrast_temperature": 0.07 * scale,
            "anchor_margin": 0.05 * scale,
            "knn_slack": 0.02 * scale,
            "cos_std_reference": float(reference),
            "cos_std_used": float(used),
            # the production term lives in the raw index, so the temperature
            # that means the same thing there is the raw-spread one; measured,
            # not assumed -- ``--pull-temperature`` accepts it explicitly
            "pull_temperature_suggested": 0.07 * (float(profile.pair_cos_std)
                                                  / float(REFERENCE_COS_STD_RAW)),
            "raw_cos_std": float(profile.pair_cos_std),
        }

    @staticmethod
    def _chart_rms(model: "WiazkaModel", chunk: int = 32768) -> float:
        """RMS per coordinate of the operator chart, measured rather than assumed.

        This is the unit every l2 loss of the method is expressed in: the
        curvature, shadow and cycle terms are absolute distances *inside* the
        chart, so they scale with the embedder that produced it (0.0242 for the
        Slowosiec chart the shipped weights were tuned on, 0.9903 for a
        retriever embedder).  The chart map is frozen at the principal subspace
        and the node table is fixed, so measuring before training gives the
        same number as after; it depends on the embedder, not on the weights.
        """
        emb = model.node_emb_fixed
        with torch.no_grad():
            total = 0.0
            count = 0
            for i in range(0, emb.shape[0], chunk):
                block = model.base_to_op(emb[i:i + chunk]).double()
                total += float(block.pow(2).sum())
                count += block.numel()
        return float((total / max(1, count)) ** 0.5)

    def _calibration_edges(self) -> List[Tuple[int, int, int]]:
        """Edges the served scalars are fitted on -- never the scored ones.

        With ``calib_ratio`` set this is the held-out calibration slice (the
        decoder was never shown it, so the fit is not a memorisation curve).
        Without it the historical behaviour is kept: train edges.
        """
        return self.calib_edges if self.calib_edges else self.edges

    def _sample_edges(self, batch_size: int):
        if self.cfg.balance_relations:
            # Draw the relation type with probability proportional to
            # ``count ** balance_power`` *among the relations that exist*, then
            # an edge of that type.  Under the raw distribution (power 1) half
            # of every batch is `synonimia` and the ~30 relations with fewer
            # than a hundred edges are never seen at all, so their transport
            # stays at initialisation no matter how long the run goes; a fully
            # uniform draw (power 0) starves the dominant relation, which is
            # why the power is tempered rather than 0.
            cdf = self._rel_sample_cdf
            u = torch.rand(batch_size, dtype=torch.float64)
            rel_choice = torch.searchsorted(cdf, u).clamp_max_(cdf.numel() - 1)
            counts = self._rel_counts[rel_choice]
            uniform = torch.rand(batch_size) * counts.to(torch.float32)
            local = uniform.long().clamp_max_(torch.clamp(counts - 1, min=0))
            idx = self._rel_offsets[rel_choice] + local
            return self._edge_tensor_by_rel[idx]  # [B, 3]
        idx = torch.randint(0, len(self.edge_tensor), (batch_size,))
        return self.edge_tensor[idx]  # [B, 3]: src, dst, rel

    def _other_targets(self, probe):
        # For each probe edge, the *other* true targets of the same (src, rel):
        # they are correct answers, so the filtered protocol must not count
        # them as misses, and the contrast loss must not call them negatives.
        out = []
        for s, d, r in probe:
            out.append(
                [c for c in self._targets_by_src_rel.get((s, r), ()) if c != d]
            )
        return out

    def _rank_mask(self, probe):
        """The two column classes a ranking protocol has to mask, per probe.

        ``(other_targets, never_valid)`` for
        :func:`relgat_projector.core.metrics.ranking.filtered_ranking`:

        * ``other_targets`` -- other true targets of the same ``(src, rel)``,
          which are correct answers and must not count as misses;
        * ``never_valid`` -- the row of the query node itself.  ``a -r-> b`` is
          never answered by ``a``: the Słowosieć has **no self-loops** (measured:
          0 of 359 818 edges), and a retrieval pipeline does not return the query
          document as the answer.  Left in the index it is a candidate that is
          certain to be wrong and, of all of them, the one the query resembles
          most: at small ``lam_r`` the rewritten query still points at
          ``x_a``, so the source is ranked above the gold and every probe pays a
          position it should not have to pay.  The penalty is largest exactly
          where the raw embedding is already good (a synonym sits one position
          from its own row), so it systematically pushed the dial fit towards
          rewriting *more* than the served metric wants -- measured as
          ``fuzzynimia`` QR-Recall 0.316 -> 0.299 while ``prod_mrr`` rose.

        One helper, so that the metric, the dial fit and the report cannot drift
        into three protocols again -- QR-Recall already excluded the query
        (``exclude_self=True``), which is why it disagreed with the dial about
        the useful rewriting strength (fuzzynimia: 0.316 -> 0.299).
        """
        others = self._other_targets(probe)
        never_valid = [[int(s)] for s, _d, _r in probe]
        return others, never_valid

    def _false_negative_mask(self, src, rel, dst, device):
        """``[B, N]`` bool: True where a key is a *different* true target.

        The graph is a multi-graph: a relation usually has several correct
        objects for the same subject.  They must not be scored as negatives.
        """
        mask = torch.zeros(
            (src.shape[0], self.num_nodes), dtype=torch.bool, device=device
        )
        src_cpu = src.detach().cpu().tolist()
        rel_cpu = rel.detach().cpu().tolist()
        dst_cpu = dst.detach().cpu().tolist()
        rows = []
        cols = []
        for i, (s, r, d) in enumerate(zip(src_cpu, rel_cpu, dst_cpu)):
            for c in self._targets_by_src_rel.get((s, r), ()):
                if c != d:
                    rows.append(i)
                    cols.append(c)
        if rows:
            mask[
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(cols, dtype=torch.long, device=device),
            ] = True
        return mask

    def _sample_absent_relations(self, src, rel, tries: int = 8):
        """Batch rows whose sampled relation is *absent* from that node.

        Returns ``(rows, neg_rel)`` on the training device, or ``None`` when no
        absent relation could be found (e.g. every node already carries every
        relation type).
        """
        src_cpu = src.detach().cpu().tolist()
        rows, negs = [], []
        for i, s in enumerate(src_cpu):
            valid = self._rels_by_node.get(s)
            if not valid or len(valid) >= self.num_rel:
                continue
            for _ in range(tries):
                candidate = random.randrange(self.num_rel)
                if candidate not in valid:
                    rows.append(i)
                    negs.append(candidate)
                    break
        if not rows:
            return None
        device = self.cfg.device
        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(negs, dtype=torch.long, device=device),
        )

    def _base_neighbour_pairs(self):
        """``(anchor, neighbour, cos_reference)`` for the chart kNN graph.

        The reference space is the **operator chart** ``base_to_op(base)``,
        not the raw base embeddings: the refinement has ``op_dim`` dimensions
        at its disposal and a 256-d orthogonal projection of a 1152-d space
        cannot reproduce 1152-d cosines (the principal chart of the Slowosiec
        carries 22 % of the variance).  Demanding base-space cosines would add
        a constant, unattainable penalty to every pair.  Referenced to the
        chart, the hinge is exactly satisfied by the identity refinement, so it
        only fires when the refinement is what destroys the neighbourhood.

        Built once and lazily: the scan is ``O(knn_anchors x num_nodes)`` and
        is only paid for by runs that actually switch ``L_knn`` on.  Self
        similarity is masked out by node id (a chunk row is *not* the same
        index as its column).
        """
        cached = getattr(self, "_knn_pairs_cache", None)
        if cached is not None:
            return cached
        k = max(1, int(self.cfg.knn_k))
        anchors = max(1, min(int(self.cfg.knn_anchors), self.num_nodes))
        with torch.no_grad():
            base = self.model.base_to_op(self.model.node_emb_fixed)
        x = base / base.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gen = torch.Generator().manual_seed(self.cfg.seed + 11)
        rows = torch.randperm(self.num_nodes, generator=gen)[:anchors].to(
            base.device
        )
        src_parts, dst_parts, cos_parts = [], [], []
        with torch.no_grad():
            for start in range(0, rows.numel(), 512):
                ids = rows[start:start + 512]
                sim = x[ids] @ x.t()
                sim.scatter_(1, ids.unsqueeze(1), float("-inf"))
                values, indices = sim.topk(k, dim=-1)
                src_parts.append(ids.repeat_interleave(k))
                dst_parts.append(indices.reshape(-1))
                cos_parts.append(values.reshape(-1))
        cached = (
            torch.cat(src_parts),
            torch.cat(dst_parts),
            torch.cat(cos_parts).clamp(-1.0, 1.0),
        )
        self._knn_pairs_cache = cached
        return cached

    def _knn_loss(self, refined: torch.Tensor) -> torch.Tensor:
        """Hinge keeping base neighbours neighbours after refinement.

        One-sided on purpose: the refinement may separate anything it has a
        reason to separate, but a pair that the *base* index considered close
        may not collapse below ``cos_base - knn_slack``.  This is the ranking
        structure the retriever actually uses, which the pointwise ``L_anchor``
        cannot express (a vector can sit on the sphere at the right distance
        from its base position and still shuffle its whole neighbourhood).
        """
        src, dst, cos_base = self._base_neighbour_pairs()
        n = min(int(self.cfg.knn_batch), src.numel())
        if n == 0:
            return refined.new_zeros(())
        sel = torch.randint(0, src.numel(), (n,), device=src.device)
        cos_refined = torch.nn.functional.cosine_similarity(
            refined[src[sel]], refined[dst[sel]], dim=-1
        )
        gap = (cos_base[sel] - self.geometry["knn_slack"] - cos_refined).clamp_min(0.0)
        return self.cfg.knn_weight * gap.mean()

    def _pce_pools_build(self, size: int = 64, max_k: int = 3):
        """Nested pools of paths of exactly 1..max_k hops.

        The PCE *curve* is only meaningful if PCE(k) and PCE(k+1) are measured
        on comparable inputs: sampling the k=1 pool from the edge list weights
        it by relation frequency (``synonimia`` is half the graph, and its
        targets are the nearest of all), while the k>=2 pools come out of a
        random walk.  Here one walk per starting node contributes *every* one
        of its prefixes, so the k=1 point is exactly the first hop of the same
        walks that produce the k=2 and k=3 points.

        The walk mirrors :class:`PathSampler` (no immediate return, no repeated
        relation) with a private deterministic RNG.
        """
        adj = self.path_sampler.adjacency
        nodes = [n for n, nbrs in adj.items() if nbrs]
        if not nodes:
            return {}
        rng = random.Random(self.cfg.seed + 7)
        prefixes: Dict[int, List[Tuple[int, List[int], int]]] = {
            k: [] for k in range(1, max_k + 1)
        }
        tries = 0
        max_tries = size * 400
        while any(len(v) < size for v in prefixes.values()) and tries < max_tries:
            tries += 1
            start = nodes[rng.randrange(len(nodes))]
            node, prev, prev_rel, chain = start, None, None, []
            for _ in range(max_k):
                options = [
                    (d, r) for d, r in adj.get(node, ())
                    if d != prev and r != prev_rel
                ]
                if not options:
                    break
                nxt, rel = rng.choice(options)
                chain.append(rel)
                prev, node = node, nxt
                k = len(chain)
                if len(prefixes[k]) < size:
                    prefixes[k].append((start, list(chain), node))
        device = self.cfg.device
        pools = {}
        for k, rows in prefixes.items():
            if not rows:
                continue
            pools[k] = (
                torch.tensor([r[0] for r in rows], dtype=torch.long, device=device),
                torch.tensor([r[1] for r in rows], dtype=torch.long, device=device),
                torch.tensor([r[2] for r in rows], dtype=torch.long, device=device),
            )
        return pools

    def _pce_pool(self, k: int, size: int = 64):
        """Fixed nested pool of paths of exactly ``k`` hops (built once)."""
        cache = getattr(self, "_pce_pools", None)
        if cache is None:
            cache = self._pce_pools = self._pce_pools_build(size=size)
        return cache.get(k)

    def _sample_paths(self, batch_size: int):
        """A batch of chains, all of the *same* length -- never zero-padded.

        ``torch.zeros`` padding would put relation id 0 (``Pertainym`` in this
        export) on top of every shorter chain: ``L_path`` then optimises
        ``T_0 o T_0 o T_r2 o T_r1 (x_A) -> x_B`` for the ~75% of rows that were
        padded, which is supervision against a wrong target (and the reason
        PCE(2) stayed ~2.7x PCE(1)).
        """
        depth = self.path_sampler.pick_depth()
        paths = self.path_sampler.sample_batch(batch_size, depth=depth)
        if not paths:
            paths = self.path_sampler.sample_batch(batch_size)
        if not paths:
            return None
        k = min(len(p[1]) for p in paths)
        paths = [p for p in paths if len(p[1]) == k]
        srcs = torch.tensor([p[0] for p in paths], dtype=torch.long)
        rels = torch.tensor([list(p[1]) for p in paths], dtype=torch.long)
        dsts = torch.tensor([p[2] for p in paths], dtype=torch.long)
        return srcs, rels, dsts

    def _sample_quadruples(self, batch_size: int):
        quads = self.quad_sampler.sample_batch_tensor(batch_size)
        return quads  # [B, 8] long tensor or None (no closed quadruple)

    # ------------------------------------------------------------- forward
    def _step(self, edges, paths, quads, refined):
        device = self.cfg.device
        B = edges.shape[0]
        loss = torch.zeros((), device=device)
        log: Dict[str, float] = {}

        src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
        x_a = refined[src]
        x_b = refined[dst]
        # base embeddings carried into the operator chart (used by L_anchor and
        # by the production term below; computed once per step)
        chart = self.model.base_to_op(self.model.node_emb_fixed)

        # L_edge (+ applicability on positives)
        transformed = self.model.apply(x_a, rel)
        a_pos = self.model.applicability_score(x_a, rel)
        l_edge = self.transport_loss(transformed, x_b, a_pos)
        loss = loss + l_edge
        log["L_edge"] = float(l_edge.item())

        # L_anchor
        l_anchor = self.stability_loss(
            refined, None, refined=refined, base=chart
        )
        loss = loss + l_anchor
        log["L_anchor"] = float(l_anchor.item())

        if self.cfg.knn_weight > 0:
            l_knn = self._knn_loss(refined)
            loss = loss + l_knn
            log["L_knn"] = float(l_knn.item())

        if self.stage_manager.is_active("L_con"):
            # Full-softmax contrast: query = T_r(x_A), keys = every refined
            # node, target = x_B.  Sampling extra negatives on top of a key set
            # that already contains *all* nodes only duplicated rows, and
            # without the false-negative mask a second true target of the same
            # (src, rel) was pushed away as if it were a negative -- the main
            # source of gradient noise in this term (PLAN.md 2.4).
            n_con = min(B, self.cfg.contrast_batch)
            mask = self._false_negative_mask(
                src[:n_con], rel[:n_con], dst[:n_con], device
            )
            # the subject is never the answer to its own relation: masking it
            # keeps the loss from being trivially easy on self-loops of
            # similarity (the query is already near T_r(x_a) in most relations)
            mask[torch.arange(n_con, device=device), src[:n_con]] = True
            l_con = self.contrast_loss(
                transformed[:n_con], refined, dst[:n_con], mask=mask,
                hard_negatives=self.cfg.contrast_hard_negatives,
            )
            loss = loss + l_con
            log["L_con"] = float(l_con.item())

        if self.stage_manager.is_active("L_cycle"):
            t = self.model.apply(x_a, rel)
            back = self.model.inverse(t, rel, base_point=x_a)
            l_cyc = self.stability_loss(x_a, back)
            loss = loss + l_cyc
            log["L_cycle"] = float(l_cyc.item())

        if self.stage_manager.is_active("L_isom"):
            u = x_a
            v = x_b
            tu = self.model.apply(u, rel)
            tv = self.model.apply(v, rel)
            l_iso = self.stability_loss(refined, None, u=u, v=v, tu=tu, tv=tv)
            loss = loss + l_iso
            log["L_isom"] = float(l_iso.item())

        if self.stage_manager.is_active("L_path") and paths is not None:
            p_src, p_rels, p_dst = paths
            p_xa = refined[p_src]
            p_xb = refined[p_dst]
            chain_t = self.model.apply_chain(p_xa, p_rels)
            l_path = self.path_loss(chain_t, None, p_xb)
            loss = loss + self.cfg.path_weight * l_path
            log["L_path"] = float(l_path.item())

        if self.stage_manager.is_active("L_curv") and quads is not None:
            # diamond columns: (a, r, c, q, d, p, b, s) -- see sampler FIELDS
            cols = [quads[:, i].to(device) for i in (0, 1, 3, 5, 7)]
            a, r, q, p, s = cols
            xa = refined[a]
            # holonomy around the diamond: H = T_p^-1 o T_s^-1 o T_q o T_r
            h = self.model.apply(xa, r)
            h = self.model.apply(h, q)
            h = self.model.inverse(h, s)
            h = self.model.inverse(h, p)
            hol_mask = None
            if self.cfg.holonomy_squares_only:
                # keep only the diamonds that are commutation squares, where
                # closure is an identity the graph actually states
                hol_mask = ((q == p) & (s == r)).to(h.dtype)
                if float(hol_mask.mean()) == 0.0:
                    hol_mask = None
            # the two routes A->D of the diamond
            route1 = self.model.apply(self.model.apply(xa, r), q)
            route2 = self.model.apply(self.model.apply(xa, p), s)
            l_curv = self.curvature_loss(
                xa, h, r, p, route1, route2, holonomy_mask=hol_mask
            )
            loss = loss + l_curv
            log["L_curv"] = float(l_curv.item())

        if (self.stage_manager.is_active("L_curv") and self._order_rows
                and (self.cfg.order_weight > 0
                     or self.cfg.order_join_weight > 0)):
            # the order-dependence teacher.  Both routes here apply the *same
            # two relations to the same point in the opposite order*, so their
            # distance is the non-commutativity of the pair -- the one thing
            # the diamond terms cannot say (a closed quadruple ends both of its
            # routes at the same node, so it only ever witnesses equality).
            # The labels come from the graph, once, at construction.
            rows = self.quad_sampler.sample_order_batch(
                self.cfg.order_batch, device=device)
            if rows is not None:
                o_a, o_r, o_s, o_dep = (rows[:, i] for i in range(4))
                xo = refined[o_a]
                if self.cfg.order_detach_input:
                    xo = xo.detach()
                order_rs = self.model.apply(self.model.apply(xo, o_r), o_s)
                order_sr = self.model.apply(self.model.apply(xo, o_s), o_r)
                taught = self.curvature_loss.order_teacher(
                    order_rs, order_sr, o_dep.float())
                loss = loss + taught["total"]
                log["L_order"] = float(taught["total"].item())
                log["L_order_sep"] = float(taught["sep"].item())
                log["L_order_join"] = float(taught["join"].item())

        if self.stage_manager.is_active("L_commut"):
            if self.cfg.commutation_from_order:
                # ``C_rs`` as ``P(order-dependent)``, taught from the same graph
                # evidence the order term uses to move the operator.  The
                # witness teacher below counts base points where two routes
                # meet, which on this graph tracks relation *frequency*: the
                # measured result was ``spearman(C, order_rate) = -0.29`` and
                # agreement with the graph's verdict on 26.7 % of pairs, i.e.
                # the matrix 12b reads described something other than the
                # transport it is read on.  Every scored pair enters each step,
                # so the term is a full-batch least-squares fit of C to the
                # rate rather than a sampled contrast.
                if self._comm_targets:
                    l_comm = self.curvature_loss.supervision_soft(
                        self._comm_targets,
                        weight=float(self.cfg.commutation_weight),
                    )
                    loss = loss + l_comm
                    log["L_commut"] = float(l_comm.item())
            else:
                # the commutation matrix C_rs is taught exclusively from the
                # graph: relation pairs whose two routes meet (positives) vs
                # pairs with no witness (negatives), drawn from the EXHAUSTIVE
                # witness table and balanced per step.
                pos_pairs, neg_pairs = self.quad_sampler.sample_relation_pairs(
                    batch_size=16, max_neg=16
                )
                if pos_pairs or neg_pairs:
                    l_comm = self.curvature_loss.supervision(pos_pairs, neg_pairs)
                    loss = loss + l_comm
                    log["L_commut"] = float(l_comm.item())

        if self.stage_manager.is_active("L_shadow"):
            # cycle + commutation consistency OFF the graph (PLAN.md 2.6)
            z_base = self._shadow_cache[: min(self.cfg.shadow_batch, self._shadow_cache.shape[0])]
            z = self.model.base_to_op(z_base)
            r_sh = torch.randint(0, self.num_rel, (z.shape[0],), device=device)
            z_fwd = self.model.apply(z, r_sh)
            back = self.model.inverse(z_fwd, r_sh, base_point=z)
            l_sh_cyc = self.shadow_loss.cycle(z, back)
            s_sh = (r_sh + 1) % self.num_rel
            trst = self.model.apply(self.model.apply(z, r_sh), s_sh)
            tstr = self.model.apply(self.model.apply(z, s_sh), r_sh)
            c_sh = self.curvature_loss.commutation()[r_sh, s_sh]
            l_sh_com = self.shadow_loss.commute_gated(trst, tstr, c_sh)
            loss = loss + l_sh_cyc + l_sh_com
            log["L_shadow_cycle"] = float(l_sh_cyc.item())
            log["L_shadow_commut"] = float(l_sh_com.item())

        if self.stage_manager.is_active("L_pull"):
            # The *production* objective: the vector database holds the base
            # embeddings, so an exported transformed query has to retrieve the
            # gold node out of that index.  Scored with the same full softmax
            # (and the same false-negative / subject masking) as the ``prod_*``
            # eval protocol, and fed from the same input the product uses --
            # ``chart(base)`` rather than the refined node -- so that nothing
            # separates the trained objective from the deployed one.
            #
            # The previous version was a ``(1 - cos)`` margin against the gold
            # plus 8 *random* negatives: with 243k nodes, random negatives are
            # almost always trivially dissimilar, so the term nearly vanished
            # and left the whole gap between ``eval/mrr`` (0.19) and
            # ``eval/prod_mrr`` (0.11) unoptimised.
            n_pull = min(B, self.cfg.pull_batch)
            q_pull = self.model.production_query(
                self.model.node_emb_fixed[src[:n_pull]], rel[:n_pull]
            )
            mask_pull = self._false_negative_mask(
                src[:n_pull], rel[:n_pull], dst[:n_pull], device
            )
            mask_pull[torch.arange(n_pull, device=device), src[:n_pull]] = True
            l_pull = self.pull_contrast_loss(
                q_pull,
                self.model.node_emb_fixed,
                dst[:n_pull],
                mask=mask_pull,
                hard_negatives=self.cfg.pull_hard_negatives,
            )
            loss = loss + self.cfg.pull_weight * l_pull
            log["L_pull"] = float(l_pull.item())

        if self.stage_manager.is_active("L_decoder"):
            logits, conf = self.model.decoder(x_a, x_b)
            l_dec = self.decoder_loss(logits, conf, rel)
            loss = loss + l_dec
            log["L_decoder"] = float(l_dec.item())

        if self.stage_manager.is_active("L_applicability"):
            # Negatives are relation types the source node genuinely does NOT
            # participate in.  The previous rule took "any other relation id",
            # which for a wordnet labels perfectly valid relations as invalid
            # (one synset typically carries several relation types at once) and
            # so trained the head on mostly wrong labels.
            neg_pairs = self._sample_absent_relations(src, rel)
            if neg_pairs is not None:
                rows, neg_rel = neg_pairs
                a_neg = self.model.applicability_score(
                    x_a.index_select(0, rows), neg_rel
                )
                l_app = nn.functional.binary_cross_entropy_with_logits(
                    a_neg, torch.zeros_like(a_neg)
                )
                loss = loss + l_app
                log["L_applicability"] = float(l_app.item())

        return loss, log

    # ---------------------------------------------------------------- train
    def train(self, epochs: Optional[int] = None, log_every: int = 50) -> List[Dict]:
        epochs = epochs or self.cfg.epochs
        history: List[Dict] = []
        self.history = history
        global_step = 0
        steps_per_epoch = max(1, len(self.edges) // self.cfg.batch_size)
        self._build_scheduler(steps_per_epoch, epochs)
        best_score = float("-inf")
        best_step = 0
        for epoch in range(1, epochs + 1):
            epoch_log: Dict[str, float] = {}
            # an epoch walks the edge set once; dropout must be ON while we
            # actually train (evaluate() flips the model to eval mode)
            self.model.train()
            for _ in range(steps_per_epoch):
                edges = self._sample_edges(self.cfg.batch_size).to(self.cfg.device)
                paths = self._sample_paths(self.cfg.batch_size // 2)
                if paths is not None:
                    paths = tuple(t.to(self.cfg.device) for t in paths)
                quads = self._sample_quadruples(self.cfg.batch_size // 4)

                refined = self.model.refine()
                loss, log = self._step(edges, paths, quads, refined)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.curvature_loss.parameters()),
                    1.0,
                )
                self.optimizer.step()
                if self._scheduler is not None:
                    self._scheduler.step()

                global_step += 1
                self.stage_manager.step()
                for k, v in log.items():
                    epoch_log[k] = epoch_log.get(k, 0.0) + v
                if global_step % log_every == 0:
                    print(
                        f"[epoch {epoch} step {global_step}/{steps_per_epoch}] "
                        f"stage={self.stage_manager.order[self.stage_manager.current]} "
                        f"loss={float(loss.item()):.5f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )
            n = max(1, steps_per_epoch)
            avg = {k: v / n for k, v in epoch_log.items()}
            avg["epoch"] = epoch
            avg["stage"] = self.stage_manager.order[self.stage_manager.current]

            # stage acceptance is measured on eval metrics, not train losses
            eval_metrics = self.evaluate()
            avg.update(eval_metrics)
            history.append(avg)
            headline = [
                "eval/mrr", "eval/mrr_macro", "eval/id_mrr",
                "eval/mrr_lift", "eval/raw_mrr", "eval/mrr_vs_raw",
                "eval/prod_mrr", "eval/mrr_prod_lift", "eval/hits@1",
                "eval/anchor", "eval/knn_cos_refined",
                "eval/rel_acc_macro", "eval/rel_acc_macro_ensemble",
                "eval/prod_mrr_traffic", "eval/prod_mrr_traffic_bestlam",
                "eval/prod_mrr_traffic_calib",
                "eval/prod_lambda_traffic", "eval/raw_mrr_traffic",
                "eval/path_lift", "eval/eps",
                "eval/pce_k1", "eval/pce_k2", "eval/pce_ratio_k2",
            ]
            # one epoch per line is the only place a dead term shows up: keep
            # the line short when a metric does not exist yet
            headline = [k for k in headline if k in eval_metrics]
            print(
                f"[epoch {epoch}] "
                + " ".join(
                    f"{k}={eval_metrics[k]:.4f}"
                    for k in headline
                    if k in eval_metrics
                )
                + f" | lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
            # the loss-term breakdown, on its own line: a term that has gone
            # dead (or dominant) is the single most common failure mode of
            # this stack, and it is invisible in a total.  Porting the recipe
            # to a retriever-scale embedder put the anti-commutation hinge at
            # ~95 % of the loss while the total looked merely "large"; with
            # this line it is two epochs of reading instead of one whole run.
            terms = {k: v for k, v in avg.items() if k.startswith("L_")}
            if terms:
                total = sum(terms.values()) or 1e-12
                print(
                    f"[epoch {epoch} losses] "
                    + " ".join(
                        f"{k}={v:.4f}({100.0 * v / total:.0f}%)"
                        for k, v in sorted(
                            terms.items(), key=lambda kv: -kv[1]
                        )
                    )
                )
            score = self.selection_score(eval_metrics)
            if score > best_score:
                best_score, best_step = score, global_step
                try:
                    self.save(f"{self.cfg.save_dir}/best", metrics=eval_metrics)
                except Exception as exc:  # keep training even if disk fails
                    print(f"[ckpt] best save failed: {exc}")
            if self.stage_manager.should_stop(eval_metrics):
                name = self.stage_manager.order[self.stage_manager.current]
                reason = (
                    "gate passed"
                    if self.stage_manager.gate_passed(eval_metrics)
                    else "max_steps reached (gate NOT met)"
                )
                print(f"[stage] {name} -> {reason}, advancing")
                self.stage_manager.advance()
                self._sync_stage_state()
        self.best_step = best_step
        self.best_score = best_score
        print(f"[train] best held-out score {best_score:.4f} @ step {best_step}")
        return history

    # ------------------------------------------------------------- selection
    @staticmethod
    def selection_score(metrics: Dict[str, float]) -> float:
        """Best-checkpoint criterion: rank well, on every relation, and stay
        usable from the retriever.

        * ``eval/mrr``      -- filtered MRR of ``T_r`` over the refined index;
        * ``eval/mrr_macro``-- the same averaged per relation, so a checkpoint
          that trades a rare relation for ``synonimia`` cannot win;
        * ``eval/prod_mrr`` -- the production protocol (export into the base
          space the vector database actually holds).  Weighted lower because it
          is the downstream property, but it must be part of the criterion: a
          model that ranks well only inside the operator space and exports to
          noise is not shippable (measured: QR-Recall after = 0.0 for every
          relation while the graph metrics looked fine);
        * ``eval/rel_acc_macro`` -- macro top-1 accuracy of the relation
          decoder: can the model *name* the relation between two nodes, on
          every relation and not only on the dominant one;
        * the production term, in order of preference:
          ``eval/prod_mrr_traffic_calib`` -- the traffic MRR with the
          per-relation dial fitted on the calibration slice, i.e. the quantity
          the shipped artifact is calibrated to;
          ``eval/prod_mrr_traffic_bestlam`` -- the same probe with the best
          *global* rewriting strength (the historical fallback when no
          calibration slice exists).  Neither depends on the gradient-trained
          dial, which never leaves ``lam ~ 0.8``: measured on F2 the criterion
          that ranked on that dial chose weights whose traffic metric had
          already decayed (0.4696 at epoch 15 -> 0.4467 at epoch 60) as the
          best of the run.  The per-relation dial is what separates the two:
          on F2 no global lambda beat the raw query, while the calibrated
          dial scored 0.4902 against 0.4499.
        """
        traffic = metrics.get("eval/prod_mrr_traffic_calib")
        if traffic is None or traffic != traffic:
            traffic = metrics.get("eval/prod_mrr_traffic_bestlam")
        if traffic is None or traffic != traffic:
            traffic = metrics.get("eval/prod_mrr", 0.0)
        return (
            0.30 * metrics.get("eval/mrr", 0.0)
            + 0.18 * metrics.get("eval/mrr_macro", 0.0)
            + 0.10 * metrics.get("eval/prod_mrr", 0.0)
            + 0.24 * traffic
            + 0.18 * metrics.get("eval/rel_acc_macro_ensemble",
                                 metrics.get("eval/rel_acc_macro", 0.0))
        )

    #: deprecated alias kept for callers written against the private name
    _selection_score = selection_score

    # ---------------------------------------------------------------- storage
    def load_weights_from(self, ckpt_dir: str) -> Dict[str, Any]:
        """Restore weights saved by :meth:`save` into the live model.

        Training keeps the best-scoring checkpoint under ``<save-dir>/best``,
        but the run reported (and shipped) the weights the last epoch happened
        to leave behind.  This makes the reported numbers describe the artifact
        that is actually loaded by ``RelationFlow.load``.

        The graph buffers are part of the state dict and the base embeddings are
        frozen, so nothing else has to be rebuilt; keys the checkpoint predates
        keep their current value.
        """
        from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint

        model, _rel2idx, _spec, meta = load_checkpoint(
            ckpt_dir, device=self.cfg.device)
        # the calibration scalars are attributes, not parameters: copy them too,
        # or the live model mixes its evidences differently from the checkpoint
        for name in ("relation_ensemble_weight", "relation_prior_strength"):
            if hasattr(model, name):
                setattr(self.model, name, float(getattr(model, name)))
        result = self.model.load_state_dict(model.state_dict(), strict=False)
        del model
        if result.unexpected_keys:
            raise RuntimeError(
                f"checkpoint {ckpt_dir} does not match this model: "
                f"unexpected={list(result.unexpected_keys)[:8]}"
            )
        if result.missing_keys:
            print(f"[ckpt] {len(result.missing_keys)} tensors kept from "
                  f"initialisation: {list(result.missing_keys)[:6]}")
        self.model.eval()
        return meta

    def save(self, ckpt_dir: str, metrics: Optional[Dict[str, float]] = None) -> str:
        """Persist model + config + the learned commutation matrix C_rs."""
        from relgat_projector.handlers.wiazka_checkpoint import save_checkpoint

        history = getattr(self, "history", None) or []
        last = history[-1] if history else None
        return save_checkpoint(
            ckpt_dir,
            self.model,
            self.rel2idx,
            config={
                **dataclasses.asdict(self.cfg),
                # the curriculum switches the connection on at F4, so the
                # configured value is not the architecture of what is being
                # saved: a checkpoint written as ``use_context: False`` but
                # carrying a trained ``ctx_P`` rebuilds as a different model
                "use_context": bool(
                    getattr(self.model.operator, "use_context",
                            self.cfg.use_context)),
                "num_nodes": self.num_nodes,
                "num_edges": len(self.edges),
                "stage": self.stage_manager.order[self.stage_manager.current],
                # the ensemble weight was fitted on train edges and is part of
                # the served relation classifier, so it has to travel with the
                # checkpoint (RelationFlow.load restores it)
                "relation_ensemble_weight": float(
                    getattr(self.model, "relation_ensemble_weight", 1.0)),
                "relation_prior_strength": float(
                    getattr(self.model, "relation_prior_strength", 1.0)),
                # what the base space actually measured like, so a
                # checkpoint says which embedder scale its cosine knobs were
                # resolved against (``op_dim`` above is the *resolved* one)
                "base_profile": self.base_profile.as_dict(),
                "geometry": self.geometry,
                "num_calibration_edges": len(self.calib_edges),
            },
            commutation=self.curvature_loss.commutation().cpu(),
            metrics=metrics if metrics is not None else last,
        )

    # ------------------------------------------------------ query rewriting
    @torch.no_grad()
    def qr_eval(
        self,
        rel_names: List[str],
        ks: Tuple[int, ...] = (1, 5, 10),
        max_queries: int = 256,
        use_heldout: bool = True,
        filter_others: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """QR-Recall: does `T_r` actually improve retrieval of the gold answer?

        Evaluated the way the feature is used in production: the index is the
        base (retriever) embeddings -- what the vector database really holds --
        and a query is rewritten by
        ``to_retriever(T_r(base_to_op(x)))``.  Recall is compared with the
        untransformed query, so a relation that only looks good geometrically
        but does not help retrieval is exposed.  Scored on held-out edges when
        they exist (PLAN.md 5).
        """
        from relgat_projector.core.metrics.query_rewriting import gold_pairs, qr_recall

        scored = self.eval_edges if (use_heldout and self.eval_edges) else self.edges
        index = self._node_emb_dev
        gen = torch.Generator().manual_seed(self.cfg.seed)
        out: Dict[str, Dict[str, float]] = {}
        for name in rel_names:
            if name not in self.rel2idx:
                raise KeyError(f"unknown relation {name!r}")
            rid = self.rel2idx[name]
            # edges already reference rows of self.node_emb
            pairs = gold_pairs(scored, rid, max_queries, gen)
            if not pairs:
                out[name] = {"relation": name, "n_queries": 0.0}
                continue
            result = qr_recall(
                pairs,
                index,
                # the served path, dial included: QR-Recall has to measure what
                # the rewriter actually sends, not a neighbouring function
                transform=lambda x, rid=rid: self.model.production_query(
                    x, torch.full((x.shape[0],), rid, dtype=torch.long,
                                  device=index.device)
                ),
                ks=ks,
                filter_others=filter_others,
                # filter against the graph, not against the sample: a query
                # with 40 true synonyms that appears once in the probe had no
                # filtering at all and scored 39 correct answers as misses
                other_map={int(q): self._targets_by_src_rel.get((int(q), rid), ())
                           for q, _a in pairs} if filter_others else None,
                relation=name,
                chain=(rid,),
            )
            out[name] = result.as_dict()
        return out

    # ---------------------------------------------------------------- eval
    def _probe_edges(self, n_probe: Optional[int] = None):
        """Deterministic, relation-stratified, ``(src, rel)``-unique probe.

        Held-out edges only.  Stratification matters: ``synonimia`` carries
        half the graph, so a uniform probe scores one relation, caps hits@1
        near the multiplicity ceiling of that one relation, and never shows
        what happened to the ~30 relations with fewer than 100 edges.
        """
        scored = self.eval_edges if self.eval_edges else self.edges
        probe = stratified_edge_probe(
            scored, n_probe or self.cfg.probe_size, seed=self.cfg.seed
        )
        seen = set()
        deduped = []
        for s, d, r in probe:
            if (s, r) in seen:
                continue
            seen.add((s, r))
            deduped.append((s, d, r))
        return deduped

    @torch.no_grad()
    def _transport_prior_stats(self, refined: torch.Tensor, n: int = 512):
        """``(mean, std)`` per relation of ``cos(T_r u, v)`` over random pairs.

        A relation whose transport is nearly the identity scores high on
        *everything*, so raw transport fidelity ranks the easy relations first.
        Calibrating against the score a relation gives on random pairs is the
        transport counterpart of the class prior; measured on the Słowosieć it
        is worth +0.04 macro relation accuracy, and without it the transport
        scorer is worse than the trained head.

        Fitted on the **calibration** edges (see :meth:`_calibration_edges`):
        on the supervised edges the head has memorised the labels, so the
        grid curve is flat and its argmax is sampling noise.
        """
        gen = torch.Generator().manual_seed(self.cfg.seed + 21)
        idx_a = torch.randint(
            0, self.num_nodes, (min(n, self.num_nodes),), generator=gen
        ).to(refined.device)
        idx_b = torch.randint(
            0, self.num_nodes, (min(n, self.num_nodes),), generator=gen
        ).to(refined.device)
        scores = self.model.transport_relation_scores(
            refined[idx_a], refined[idx_b])
        return scores.mean(dim=0), scores.std(dim=0).clamp_min(1e-6)

    @torch.no_grad()
    def fit_relation_prior_strength(
        self,
        refined: torch.Tensor,
        train_counts: torch.Tensor,
        n_pairs: int = 4096,
        grid: Sequence[float] = PRIOR_STRENGTH_GRID,
    ) -> float:
        """Fit the strength of the training-frequency correction of the head.

        ``mode="prior"`` subtracts ``strength * log p_train(r)`` from the head's
        log-probabilities.  ``1.0`` is right for plain cross-entropy; a head
        trained with class weights ``w(r) ~ count**-beta`` fits
        ``p(r|x) * w(r)`` and therefore needs ``+beta * log count``, i.e. the
        opposite sign -- which is why the fixed ``1.0`` scored macro 0.058
        against 0.465 uncorrected once the decoder was balanced.  One scalar on
        **train** edges covers both regimes (``0`` = trust the balanced head).
        """
        source = self._calibration_edges()
        if not source:
            return float(self.model.relation_prior_strength)
        gen = torch.Generator().manual_seed(self.cfg.seed + 31)
        table = torch.tensor(source, dtype=torch.long)
        n = min(int(n_pairs), len(source))
        pick = torch.randperm(len(source), generator=gen)[:n]
        edges = table[pick].to(refined.device)
        src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
        logits = self.model.relation_logits(refined[src], refined[dst],
                                            mode="decoder")
        counts = train_counts.to(logits.device).clamp_min(1.0).float()
        log_prior = torch.log(counts / counts.sum())

        best_strength, best_acc = 0.0, -1.0
        for strength in grid:
            pred = (logits - float(strength) * log_prior).argmax(dim=-1)
            acc = macro_relation_accuracy(pred, rel, self.num_rel)
            if acc > best_acc + 1e-9:
                best_strength, best_acc = float(strength), acc
        self.model.relation_prior_strength = best_strength
        return best_strength

    @torch.no_grad()
    def fit_relation_ensemble_weight(
        self,
        refined: torch.Tensor,
        score_stats: Tuple[torch.Tensor, torch.Tensor],
        n_pairs: int = 4096,
        grid: Sequence[float] = ENSEMBLE_WEIGHT_GRID,
    ) -> float:
        """Fit the weight of the transport evidence in the relation ensemble.

        Which evidence deserves to win depends on how the head was trained.  With
        plain cross-entropy the transport was the better of the two (0.295 macro
        against 0.285 for the head) and an equal-weight ensemble beat both;
        class-balancing the head reversed the order (0.49 against 0.42), so the
        fixed equal weighting shipped a relation classifier *worse* than the head
        it blended.  One scalar fitted on the **calibration** edges (see
        :meth:`_calibration_edges`) settles it the same way
        :meth:`calibrate_transport_dial` settles the dial -- the held-out
        probe stays untouched.  On the supervised edges the head has
        memorised the labels, so the grid curve is flat and its argmax is
        sampling noise.
        """
        source = self._calibration_edges()
        if not source:
            return float(self.model.relation_ensemble_weight)
        gen = torch.Generator().manual_seed(self.cfg.seed + 33)
        table = torch.tensor(source, dtype=torch.long)
        n = min(int(n_pairs), len(source))
        pick = torch.randperm(len(source), generator=gen)[:n]
        edges = table[pick].to(refined.device)
        src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
        # the head evidence exactly as the ensemble sees it, i.e. with the prior
        # correction of :meth:`fit_relation_prior_strength` already applied
        head = self.model.relation_logits(
            refined[src], refined[dst], mode="prior",
            train_counts=self._relation_counts.to(refined.device))
        transport = self.model.relation_logits(
            refined[src], refined[dst], mode="transport",
            score_stats=score_stats)
        alpha = fit_ensemble_weight(head, transport, rel, self.num_rel, grid)
        self.model.relation_ensemble_weight = alpha
        return alpha

    @torch.no_grad()
    def calibrate_transport_dial(
        self,
        n_per_rel: int = 256,
        grid: Optional[Sequence[float]] = None,
        edges: Optional[List[Tuple[int, int, int]]] = None,
        smooth: int = 3,
        min_calib_per_rel: int = 24,
    ) -> Dict[int, float]:
        """Fit the per-relation rewriting strength on training edges.

        ``L_pull`` trains ``lam_r`` through 256 sampled queries per step, and
        measured over a whole 35-epoch run every dial ended where it started
        (``lam ~ 0.81`` for all 56 relations): one scalar per relation, a weak
        and noisy gradient, and a loss whose derivative against ``lam`` is close
        to flat in the middle of the mixture.  This is the direct version of the
        same fit -- for every relation scan a grid of ``lam`` and keep the value
        that ranks the gold node best in the base index, which is the quantity
        the product reports.

        It is fitted on the **calibration** edges (up to ``n_per_rel`` per
        relation) and is one parameter per relation, so it does not touch the
        probe the reported numbers come from.  The fit is on edges the decoder
        and the pull loss have never seen, so the curve is not a memorisation
        curve (measured on F2: fitting on train edges ships ``lam`` 0.28 on
        ``synonimia`` where the held-out data says ~0.05).  Returns
        ``{relation_id: lam}``.
        """
        if grid is None:
            grid = [round(0.1 * i, 1) for i in range(11)]
        source = self._calibration_edges() if edges is None else edges
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for s_, d_, r_ in source:
            grouped.setdefault(int(r_), []).append((int(s_), int(d_)))
        # A relation the calibration slice barely covers cannot have its dial
        # fitted there: one edge fixes the grid argmax by accident, and the
        # dial is the one parameter that decides whether the relation gets
        # rewritten at all.  Those relations fall back to the supervised edges
        # -- an honest fit needs data, and a biased fit on 400 edges still
        # beats a random one on 2.
        fallback: Dict[int, List[Tuple[int, int]]] = {}
        if source is not self.edges:
            for s_, d_, r_ in self.edges:
                fallback.setdefault(int(r_), []).append((int(s_), int(d_)))
        topped_up = 0
        for r, rows in list(grouped.items()):
            if len(rows) < max(2, int(min_calib_per_rel)) and fallback.get(r):
                grouped[r] = rows + fallback[r]
                topped_up += 1
        if topped_up:
            print(f"[dial] {topped_up} relations below {int(min_calib_per_rel)} "
                  f"calibration edges: fitted on supervised edges")
        # spread over each relation instead of taking its first edges (see
        # ``spread_sample``): the dial is one scalar per relation fitted from
        # ~100 examples, so which 100 decides what it describes
        by_rel = {r: spread_sample(rows, n_per_rel, seed=self.cfg.seed + r)
                  for r, rows in grouped.items()}
        base = self._node_emb_dev
        dial = self.model.transport_dial
        original = dial.detach().clone()
        chosen: Dict[int, float] = {}
        for r, rows in sorted(by_rel.items()):
            src = torch.tensor([p[0] for p in rows], device=base.device)
            dst = torch.tensor([p[1] for p in rows], device=base.device)
            rel = torch.full((len(rows),), r, dtype=torch.long, device=base.device)
            # exactly the protocol of ``evaluate()``/``_rank_mask`` (one source
            # of truth): the query row is not a candidate, so ``lam = 0`` is
            # reachable and the dial optimises the quantity that is reported
            others = [
                [c for c in self._targets_by_src_rel.get((int(s), r), ())
                 if c != int(d)]
                for s, d in rows
            ]
            never_valid = [[int(s)] for s, _d in rows]
            curve: List[float] = []
            for lam in grid:
                dial[r] = float(torch.logit(torch.tensor(float(lam))
                                            .clamp(1e-3, 1 - 1e-3)))
                query = self.model.production_query(base[src], rel)
                _, ranks = filtered_ranking(query, base, dst,
                                            other_targets=others,
                                            never_valid=never_valid)
                curve.append(float(ranks.float().reciprocal().mean().item()))
            # ties go to the first (smallest) lambda: rewriting is the risky
            # direction, so an equal score keeps the milder one
            scores = smooth_curve(curve, smooth)
            best_i = max(range(len(scores)), key=lambda i: (scores[i], -i))
            chosen[r] = float(grid[best_i])
            dial[r] = float(torch.logit(torch.tensor(chosen[r])
                                        .clamp(1e-3, 1 - 1e-3)))
        # relations with no training edge keep whatever they had
        for r in range(self.num_rel):
            if r not in chosen:
                dial[r] = original[r]
        return chosen

    @torch.no_grad()
    def _best_lambda_mrr(
        self,
        x: torch.Tensor,
        exported: torch.Tensor,
        rel: torch.Tensor,
        dst: torch.Tensor,
        others,
        never_valid,
        grid: Sequence[float] = LAMBDA_PROBE_GRID,
    ) -> Tuple[float, float]:
        """``(best mrr, argmax lam)`` over a *global* rewriting strength.

        Why this exists: the checkpoint a run ships is chosen by a score
        computed at every epoch, and until now that score contained the
        production metric measured with the **gradient-trained dial** -- a
        dial that measurably never moves from ``lam ~ 0.8``.  Measured on F2
        the traffic metric rose to 0.4696 around epoch 15 and then decayed to
        0.4467 while the loss kept improving, so the epoch-wise score ranked
        weights by a quantity the product never serves, and the "best"
        checkpoint was the best-looking-at-lam-0.8 one.  Scanning a global
        ``lam`` costs one extra ranking per grid point, is independent of the
        dial, and bounds the calibrated model from below.
        """
        best_mrr, best_lam = -1.0, 0.0
        for lam in grid:
            query = x if float(lam) == 0.0 else (1.0 - float(lam)) * x + float(lam) * exported
            stats, _ = filtered_ranking(
                query, self._node_emb_dev, dst,
                other_targets=others, never_valid=never_valid)
            if stats["mrr"] > best_mrr:
                best_mrr, best_lam = float(stats["mrr"]), float(lam)
        return best_mrr, best_lam

    @torch.no_grad()
    def _calibrated_traffic_mrr(
        self,
        x: torch.Tensor,
        exported: torch.Tensor,
        rel: torch.Tensor,
        dst: torch.Tensor,
        others,
        never_valid,
        n_per_rel: int = 64,
        grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    ) -> float:
        """Traffic MRR with a per-relation dial fitted on the calibration slice.

        ``prod_mrr_traffic`` measured with the gradient-trained dial is not the
        quantity the product ships: the dial is calibrated before serving and
        the calibration travels in the checkpoint.  A *global* lambda scan
        (:meth:`_best_lambda_mrr`) cannot substitute for it -- on F2 no global
        lambda beat the raw query on traffic, while the per-relation dial scored
        0.4902 against 0.4499 raw, because the useful answer is "rewrite
        ``antonimia`` fully and ``synonimia`` not at all" and no constant can
        express that.  So the epoch-wise checkpoint score has to be measured
        with a dial fitted the same way the served one is.

        The fit goes into a local buffer, never into ``model.transport_dial``:
        this is a measurement, and a measurement that mutates the weights it is
        about to score is how a training run starts optimising its own probe.
        Calibration edges only; the probe it scores is the disjoint remainder
        of the held-out set.
        """
        if not self.calib_edges:
            return float("nan")
        base = self._node_emb_dev
        source = self._calibration_edges()
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for s_, d_, r_ in source:
            grouped.setdefault(int(r_), []).append((int(s_), int(d_)))
        fallback: Dict[int, List[Tuple[int, int]]] = {}
        if source is not self.edges:
            for s_, d_, r_ in self.edges:
                fallback.setdefault(int(r_), []).append((int(s_), int(d_)))
        for r, rows in list(grouped.items()):
            if len(rows) < 24 and fallback.get(r):
                grouped[r] = rows + fallback[r]

        dial = torch.zeros(self.num_rel, device=base.device)
        for r, rows in sorted(grouped.items()):
            rows = spread_sample(rows, n_per_rel, seed=self.cfg.seed + r)
            src = torch.tensor([p[0] for p in rows], device=base.device)
            dstc = torch.tensor([p[1] for p in rows], device=base.device)
            relc = torch.full((len(rows),), r, dtype=torch.long, device=base.device)
            others_c = [
                [c for c in self._targets_by_src_rel.get((int(s_), r), ())
                 if c != int(d_)]
                for s_, d_ in rows
            ]
            never_c = [[int(s_)] for s_, _d in rows]
            x_row = base[src]
            exp_row = self.model._export_linear(
                self.model.apply(self.model.base_to_op(x_row), relc))
            best_mrr = -1.0
            for lam in grid:
                query = x_row if float(lam) == 0.0 else (
                    (1.0 - float(lam)) * x_row + float(lam) * exp_row
                )
                stats, _ = filtered_ranking(
                    query, base, dstc, other_targets=others_c,
                    never_valid=never_c)
                if stats["mrr"] > best_mrr:
                    best_mrr = float(stats["mrr"])
                    dial[r] = float(lam)
        lam = dial[rel].unsqueeze(-1)
        query = (1.0 - lam) * x + lam * exported
        stats, _ = filtered_ranking(
            query, self._node_emb_dev, dst,
            other_targets=others, never_valid=never_valid)
        return float(stats["mrr"])

    @torch.no_grad()
    def evaluate(self, n_probe: Optional[int] = None) -> Dict[str, float]:
        from relgat_projector.core.metrics import geometry
        from relgat_projector.core.metrics.ranking import (
            edge_probe,
            filtered_ranking,
            macro_mrr,
        )

        self.model.eval()
        refined = self.model.refine()
        device = self.cfg.device
        probe = self._probe_edges(n_probe)
        e = torch.tensor(probe, dtype=torch.long).to(device)
        src, dst, rel = e[:, 0], e[:, 1], e[:, 2]
        x_a = refined[src]
        x_b = refined[dst]
        transformed = self.model.apply(x_a, rel)
        cos = torch.nn.functional.cosine_similarity(transformed, x_b, dim=-1)
        base_op = self.model.base_to_op(self.model.node_emb_fixed)
        out: Dict[str, float] = {
            "eval/cos_edge": float(cos.mean().item()),
            # refinement must stay near the base embedding (F1 gate, PLAN 4)
            "eval/anchor": float(
                torch.nn.functional.cosine_similarity(
                    refined, base_op, dim=-1
                ).mean().item()
            ),
            "eval/cce": geometry.cycle_closure_error(
                self.model.operator, x_a, rel
            ),
            "eval/isot": geometry.isometry_error(
                self.model.operator, x_a[:64], x_b[:64], rel[:64]
            ),
        }
        # the answer list excludes the query itself and the other true targets
        # (the contrast term does the same), so every protocol -- graph,
        # production and QR-Recall -- compares like with like
        others, never_valid = self._rank_mask(probe)

        def score(query, index, prefix):
            ranking, ranks = filtered_ranking(
                query, index, dst, other_targets=others,
                never_valid=never_valid)
            for key, value in ranking.items():
                if key != "n":
                    out[f"eval/{prefix}{key}"] = value
            macro = macro_mrr(ranks, rel.detach().cpu().tolist())
            out[f"eval/{prefix}mrr_macro"] = macro["mrr_macro"]
            if not prefix:
                out["eval/num_relations"] = macro["num_relations"]
            return ranks

        # (1) graph protocol: T_r over the refined index, plus the identity
        # transport on the *same* index.  Comparing the model with a baseline
        # scored on a different index (the base embeddings) mixes "the
        # transport is good" with "refinement moved the index", which is what
        # made the old headline lift meaningless: refinement alone was worth
        # +0.19 MRR on antonimia and -0.03 on synonimia at the same time.
        score(transformed, refined, "")
        score(refined[src], refined, "id_")
        # (1b) the same identity transport over the *unrefined* chart.  Without
        # this row ``id_mrr`` conflates two very different losses -- what the
        # 256-d chart cannot represent at all, and what the refinement destroys
        # on top of it.  ``chart_mrr`` is the ceiling the refinement starts
        # from, so ``id_mrr - chart_mrr`` is pure refinement damage.
        score(base_op[src], base_op, "chart_")
        # (2) production protocol: the retriever index is the base embeddings
        # (what the vector database holds) and the query goes through
        # export(T_r(base_to_op(base_x)))  (PLAN.md 5)
        q_prod = self.model.production_query(self._node_emb_dev[src], rel)
        score(q_prod, self._node_emb_dev, "prod_")
        score(self._node_emb_dev[src], self._node_emb_dev, "raw_")

        # (2c) the same production query on a probe drawn the way traffic is
        # distributed.  The stratified probe is how a *rare* relation is judged;
        # it is not what a deployed rewriter is judged by, because ``synonimia``
        # carries half of the edges and the stratified average gives it 1/56 of
        # the vote.  A dial that over-wrote ``synonimia`` cost 0.20 MRR there and
        # 0.004 in the stratified number -- which is exactly how the defect
        # stayed invisible.  Both views are reported; neither is enough alone.
        #
        # The probe is stratified by (src, rel), NOT by (src, dst, rel): the
        # ranking protocol compares one query against a *set* of answers, so a
        # node with 40 true targets of one relation contributes one query, not
        # 40.  Stratifying on the triplet duplicated exactly those high-target
        # nodes and handed the metric to the relation whose targets are most
        # duplicated: on F3 the traffic probe then scored 0.4445 best-lambda
        # while the identical model scored 0.4902 on a triplet-stratified probe
        # of the same size, with the raw baseline moving 0.4426 -> 0.4499 in the
        # same probe change.  The metric was measuring the probe, not the model.
        traffic_source = self.eval_edges if self.eval_edges else self.edges
        traffic = edge_probe(
            traffic_source, self.cfg.probe_size,
            seed=self.cfg.seed + 7, unique_by="src_rel")
        if len(traffic) > 1:
            et = torch.tensor(traffic, dtype=torch.long).to(device)
            t_src, t_dst, t_rel = et[:, 0], et[:, 1], et[:, 2]
            t_others, t_never = self._rank_mask(traffic)
            t_prod = self.model.production_query(self._node_emb_dev[t_src], t_rel)
            stats, _ = filtered_ranking(
                t_prod, self._node_emb_dev, t_dst,
                other_targets=t_others, never_valid=t_never)
            raw_stats, _ = filtered_ranking(
                self._node_emb_dev[t_src], self._node_emb_dev, t_dst,
                other_targets=t_others, never_valid=t_never)
            out["eval/prod_mrr_traffic"] = stats["mrr"]
            out["eval/raw_mrr_traffic"] = raw_stats["mrr"]
            out["eval/mrr_prod_lift_traffic"] = (
                stats["mrr"] - raw_stats["mrr"])
            # dial-invariant view of the same probe (see _best_lambda_mrr)
            t_export = self.model._export_linear(self.model.apply(
                self.model.base_to_op(self._node_emb_dev[t_src]), t_rel))
            best, best_lam = self._best_lambda_mrr(
                self._node_emb_dev[t_src], t_export, t_rel, t_dst,
                t_others, t_never)
            out["eval/prod_mrr_traffic_bestlam"] = best
            out["eval/prod_lambda_traffic"] = best_lam
            out["eval/mrr_prod_lift_traffic_bestlam"] = (
                best - raw_stats["mrr"])
            # the quantity the product actually ships: the traffic MRR with
            # the per-relation dial fitted on the calibration slice (a few
            # seconds per call; NaN when there is no slice, i.e. the
            # historical train-ratio=1.0 / calib-ratio=0 configuration)
            out["eval/prod_mrr_traffic_calib"] = self._calibrated_traffic_mrr(
                self._node_emb_dev[t_src], t_export, t_rel, t_dst,
                t_others, t_never)

        out["eval/n_probe"] = float(len(probe))
        # the transport's own contribution, measured on one index
        out["eval/mrr_lift"] = out["eval/mrr"] - out["eval/id_mrr"]
        # the whole model against "just use the raw embedding"
        out["eval/mrr_vs_raw"] = out["eval/mrr"] - out["eval/raw_mrr"]
        # what the exported query adds in the retriever (the product property)
        out["eval/mrr_prod_lift"] = out["eval/prod_mrr"] - out["eval/raw_mrr"]

        # PCE curve on a FIXED pool of paths per depth.  Drawing new random
        # walks at every eval made the curve incomparable between epochs (and
        # cost thousands of Python walks per eval).
        for k in range(1, 4):
            pool = self._pce_pool(k)
            if pool is None:
                continue
            psrc, prels, pdst = pool
            chain_t = self.model.apply_chain(refined[psrc], prels)
            cos_k = torch.nn.functional.cosine_similarity(
                chain_t, refined[pdst], dim=-1
            )
            out[f"eval/pce_k{k}"] = float((1.0 - cos_k).mean().item())
            if k == 2:
                # cosine error says nothing about whether the composition is
                # *usable*: rank the two-hop target among all nodes, by the
                # chain transport and by the identity transport on the same
                # index.  ``path_lift`` is the F3 acceptance criterion -- the
                # old gate (pce_ratio_k2 <= 1.6) asks for composition error
                # *below* one step's, which no non-trivial operator achieves
                # (two independent angular errors add up to ratio ~2.0 by
                # construction), so F3 always exited through the max_steps
                # valve after 10 epochs and the export/decoder stage lost a
                # third of every run.
                chain_rank, _ = filtered_ranking(chain_t, refined, pdst)
                ident_rank, _ = filtered_ranking(refined[psrc], refined, pdst)
                out["eval/path_mrr"] = chain_rank["mrr"]
                out["eval/path_mrr_ident"] = ident_rank["mrr"]
                out["eval/path_lift"] = chain_rank["mrr"] - ident_rank["mrr"]
        # strength of the connection (PLAN.md 1.3): 0 when the operator has no
        # context.  Logged because an unlearnable eps is invisible everywhere
        # else -- the transport stays numerically the same and no loss complains.
        out["eval/eps"] = float(getattr(self.model.operator, "eps", 0.0))
        # composition must not degrade transport much faster than one step
        # does; the ratio is scale-free, unlike an absolute PCE threshold, so
        # it stays valid while the whole curve is still improving
        pce1 = out.get("eval/pce_k1")
        pce2 = out.get("eval/pce_k2")
        if pce1 is not None and pce2 is not None:
            out["eval/pce_ratio_k2"] = float(pce2 / max(pce1, 1e-6))

        # --- neighbourhood preservation (the property L_knn optimises) -----
        # ``knn_cos_chart`` is the mean cosine of chart-neighbour pairs before
        # refinement, ``knn_cos_refined`` after it.  A ratio below 1 says the
        # refinement is spending the index to buy edge accuracy -- which is
        # exactly the trade that made identity transport score *below* the raw
        # embeddings.  Measured on the reference run: 0.62 -> 0.41.
        if self.num_nodes > 1:
            src_n, dst_n, cos_chart_n = self._base_neighbour_pairs()
            n_pairs = min(8192, src_n.numel())
            out["eval/knn_cos_chart"] = float(
                cos_chart_n[:n_pairs].mean().item()
            )
            out["eval/knn_cos_refined"] = float(
                torch.nn.functional.cosine_similarity(
                    refined[src_n[:n_pairs]], refined[dst_n[:n_pairs]], dim=-1
                ).mean().item()
            )

        # --- relation prediction (PLAN.md 1.5) -----------------------------
        # Four scorers, because "can the model name the relation" has more than
        # one honest answer and the head alone used to be the only one:
        #   rel_acc*_decoder    the MLP head (raw logits, then corrected for the
        #                       train-set relation prior: the probe is relation
        #                       uniform, the head is not trained that way)
        #   rel_acc*_transport  transport fidelity cos(T_r x_a, x_b) calibrated
        #                       against the score every relation gives on random
        #                       pairs -- no head, no extra parameters
        #   rel_acc*_ensemble   the two evidences added after row z-scoring
        # The head is trained with tempered inverse-frequency class weights, so
        # its uncorrected macro is expected to be close to the corrected one.
        if self.model.use_decoder:
            with torch.no_grad():
                counts = self._relation_counts.to(device)
                stats = self._transport_prior_stats(refined)
                prior = self.fit_relation_prior_strength(refined, counts)
                alpha = self.fit_relation_ensemble_weight(refined, stats)
                out["eval/rel_prior_strength"] = float(prior)
                out["eval/rel_ensemble_alpha"] = float(alpha)
                evidence = {
                    "decoder": self.model.relation_logits(
                        x_a, x_b, mode="decoder"),
                    "decoder_cal": self.model.relation_logits(
                        x_a, x_b, mode="prior", train_counts=counts),
                    "transport": self.model.relation_logits(
                        x_a, x_b, mode="transport", score_stats=stats),
                    "ensemble": self.model.relation_logits(
                        x_a, x_b, mode="ensemble", score_stats=stats,
                        train_counts=counts, ensemble_weight=alpha),
                }
                for name, ev in evidence.items():
                    pred = ev.argmax(dim=-1)
                    correct = (pred == rel).float()
                    per_rel_sum = torch.zeros(
                        self.num_rel, device=correct.device, dtype=correct.dtype
                    )
                    per_rel_n = torch.zeros_like(per_rel_sum)
                    per_rel_sum.index_add_(0, rel, correct)
                    per_rel_n.index_add_(0, rel, torch.ones_like(correct))
                    seen = per_rel_n > 0
                    out[f"eval/rel_acc_{name}"] = float(correct.mean().item())
                    out[f"eval/rel_acc_macro_{name}"] = float(
                        (per_rel_sum[seen] / per_rel_n[seen]).mean().item()
                    )
                # backwards-compatible names: the decoder, uncorrected
                out["eval/rel_acc"] = out["eval/rel_acc_decoder"]
                out["eval/rel_acc_macro"] = out["eval/rel_acc_macro_decoder"]

        # evaluate() must not leave the model in eval mode (the caller resumes
        # training right after; dropout off silently halves the regularisation)
        return out

    @torch.no_grad()
    def per_relation_report(
        self, n_per_rel: int = 48
    ) -> List[Dict[str, float]]:
        """Filtered MRR per relation: model vs identity vs raw base.

        The aggregate hides everything that matters on a 56-relation graph:
        a single relation holding half the edges decides the micro average,
        while the interesting learning happens on the small ones.
        """
        from relgat_projector.core.metrics.ranking import filtered_ranking

        self.model.eval()
        scored = self.eval_edges if self.eval_edges else self.edges
        probe = stratified_edge_probe(
            scored, max(self.cfg.probe_size, n_per_rel * self.num_rel),
            seed=self.cfg.seed + 1, max_per_rel=n_per_rel,
        )
        if not probe:
            return []
        e = torch.tensor(probe, dtype=torch.long).to(self.cfg.device)
        src, dst, rel = e[:, 0], e[:, 1], e[:, 2]
        refined = self.model.refine()
        others, never_valid = self._rank_mask(probe)
        _, r_model = filtered_ranking(
            self.model.apply(refined[src], rel), refined, dst,
            other_targets=others, never_valid=never_valid,
        )
        _, r_id = filtered_ranking(
            refined[src], refined, dst,
            other_targets=others, never_valid=never_valid)
        q_prod = self.model.production_query(self._node_emb_dev[src], rel)
        _, r_prod = filtered_ranking(
            q_prod, self._node_emb_dev, dst,
            other_targets=others, never_valid=never_valid)
        _, r_raw = filtered_ranking(
            self._node_emb_dev[src], self._node_emb_dev, dst,
            other_targets=others, never_valid=never_valid,
        )

        # relation naming per relation (the ensemble scorer): which relations
        # can be *named* at all, alongside which can be retrieved
        stats = self._transport_prior_stats(refined)
        evidence = self.model.relation_logits(
            refined[src], refined[dst], mode="ensemble", score_stats=stats,
            train_counts=self._relation_counts.to(refined.device))
        rel_pred = evidence.argmax(dim=-1)
        dial = self.model.transport_strength(rel).detach().cpu().tolist()

        idx2rel = {v: k for k, v in self.rel2idx.items()}
        buckets: Dict[int, Dict[str, List[float]]] = {}
        for i, (_, _, r) in enumerate(probe):
            b = buckets.setdefault(int(r), {"mrr": [], "id": [], "prod": [],
                                            "raw": [], "cls": [], "dial": []})
            b["mrr"].append(float(1.0 / r_model[i]))
            b["id"].append(float(1.0 / r_id[i]))
            b["prod"].append(float(1.0 / r_prod[i]))
            b["raw"].append(float(1.0 / r_raw[i]))
            b["cls"].append(float(int(rel_pred[i]) == int(r)))
            b["dial"].append(float(dial[i]))
        rows = []
        for r, b in buckets.items():
            mean = lambda k: float(sum(b[k]) / len(b[k]))
            rows.append({
                "relation": idx2rel.get(r, str(r)),
                "n": len(b["mrr"]),
                "mrr_raw": mean("raw"),
                "mrr_id": mean("id"),
                "mrr_model": mean("mrr"),
                "mrr_prod": mean("prod"),
                "lift_over_raw": mean("mrr") - mean("raw"),
                "op_lift": mean("mrr") - mean("id"),
                "relation_acc": mean("cls"),
                "transport_lambda": mean("dial"),
            })
        rows.sort(key=lambda row: -row["lift_over_raw"])
        return rows
