"""
Vector augmentation by the relation operator (docs/IDEAS/03-vector-augmentation.md).

Text augmentation (back-translation, dropout, LLM paraphrase) moves the *norm
and register* of the vector, i.e. it deforms exactly the quantity cosine
measures; hard negatives then have to be mined.  The operator `T_r` is a
similarity up to scale, so `cos(T_r u, T_r v) = cos(u, v)` exactly -- the
augmented pair keeps the geometry, the only thing that moves is the meaning,
and the dial `lambda` makes the strength of the move a continuous parameter.

This module is a **diagnostic** of the three claims that idea 03 builds on,
each with a control the idea document asks for:

* **isometry of the move** -- `ISOT = mean_ij |cos(T_r u_i, T_r u_j) -
  cos(u_i, u_j)|`, gold 0.  A text augmentation cannot get below its own
  deformation, so the measured number is the whole claim: if it is 0, the
  pair carries the same geometry, whatever the embedder does afterwards.
* **the dial** -- the curve `cos(v, T_r(lambda) v)` in `lambda`; at
  `lambda -> 0` it must be the trivial pair (cos -> 1) and at `lambda = 1`
  the full step.  Reported as `dial_range` and the `lambda` where the pair
  becomes as far as it gets.
* **meaningful hard negatives** -- `T_ant(v)` must be *near in topic, far in
  meaning*, which is exactly: close to `v` in the chart, but the antonym
  orbit of `v` must not re-close onto `v`.  Measured as the separation of
  the two orders of the ant-syn chain against the commutation matrix (the
  generator of hard negatives the idea calls for), with the control of the
  document: the same numbers for a *random* relation family of equal size.

Nothing here needs a label: the golds are structural (0 for the isometry,
1 at lambda 0 for the dial).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", dtype=torch.float64).numpy()


def _off_diagonal(cos: torch.Tensor) -> torch.Tensor:
    out = cos.clone()
    out.fill_diagonal_(float("nan"))
    return out


def isometry_error(transport, vectors: torch.Tensor, rel: str) -> float:
    """Mean absolute drift of every pair's cosine under ``T_r``.

    Gold is ``0.0``: a map with ``T^T T = s^2 I`` keeps all cosines.  The
    number above zero is exactly how much the augmentation deforms the
    geometry the embedder is supposed to keep.
    """
    v = _unit(transport.prepare(vectors.float()))
    moved = _unit(transport.apply(v, rel))
    cos_in = _off_diagonal(v @ v.t())
    cos_out = _off_diagonal(moved @ moved.t())
    return float(np.nanmean(np.abs(_np(cos_out - cos_in))))


def dial_curve(transport, vectors: torch.Tensor, rel: str,
               lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, float]:
    """How much ``T_r(lambda)`` moves a vector, as a function of lambda.

    ``self_cos[l]`` is the mean ``cos(v, T_r(l) v)``; a good dial starts at 1
    (the trivial pair) and moves monotonically.  ``dial_range`` is the total
    travel and ``dial_at_05`` the step at the working point of the idea
    (lambda = 0.5).
    """
    v = _unit(transport.prepare(vectors.float()))
    self_cos: Dict[str, float] = {}
    for lam in lambdas:
        moved = transport.apply_at(v, rel, float(lam))
        self_cos[str(lam)] = float((_unit(moved) * v).sum(dim=-1).mean())
    first, last = self_cos[str(lambdas[0])], self_cos[str(lambdas[-1])]
    return {
        **{f"self_cos_{k}": val for k, val in self_cos.items()},
        "dial_range": float(first - last),
        "dial_at_05": float(self_cos.get("0.5", float("nan"))),
    }


def hard_negative_separation(
    transport,
    vectors: torch.Tensor,
    commutation: Optional[np.ndarray],
    rel: str,
    hard_rels: Sequence[str],
    control_rels: Sequence[str],
) -> Dict[str, float]:
    """Do the chains of a *meaningful* relation family separate the orders?

    The idea's hard negatives come from pairs ``r, s`` with low ``C_rs``:
    ``T_r T_s v`` and ``T_s T_r v`` are then two different points of the same
    orbit.  For the family of relations attached to ``rel`` (and for an equal-
    sized random control family) we measure the mean order separation
    ``1 - cos(T_s T_r v, T_r T_s v)`` and its mean ``C_rs``; a family that is
    a good generator must separate the orders more than a random one.
    """
    v = _unit(transport.prepare(vectors.float()))
    out: Dict[str, float] = {}
    for name, rels in (("hard", hard_rels), ("control", control_rels)):
        seps, cs = [], []
        for r in rels:
            for s in rels:
                if r == s:
                    continue
                u = transport.apply(transport.apply(v, r), s)
                w = transport.apply(transport.apply(v, s), r)
                seps.append(float((1.0 - (_unit(u) * _unit(w)).sum(dim=-1)).mean()))
                if commutation is not None:
                    cs.append(float(commutation[transport.rel2idx[r],
                                                transport.rel2idx[s]]))
        out[f"{name}_sep_mean"] = float(np.mean(seps)) if seps else float("nan")
        out[f"{name}_c_mean"] = float(np.mean(cs)) if cs else float("nan")
    out["sep_lift_over_control"] = (
        out["hard_sep_mean"] - out["control_sep_mean"]
        if out["hard_sep_mean"] == out["hard_sep_mean"]
        and out["control_sep_mean"] == out["control_sep_mean"]
        else float("nan"))
    return out


def augmentation_report(
    transport,
    vectors: torch.Tensor,
    commutation: Optional[np.ndarray],
    rel_pos: str,
    rel_neg: str,
    hard_rels: Sequence[str],
    control_rels: Sequence[str],
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, float]:
    """One row of the 03 benchmark: the three claims, each with its control."""
    out: Dict[str, float] = {}
    out["isot_pos"] = isometry_error(transport, vectors, rel_pos)
    out["isot_neg"] = isometry_error(transport, vectors, rel_neg)
    out.update({f"pos_{k}": val for k, val in
                dial_curve(transport, vectors, rel_pos, lambdas).items()})
    out.update({f"neg_{k}": val for k, val in
                dial_curve(transport, vectors, rel_neg, lambdas).items()})
    out.update(hard_negative_separation(
        transport, vectors, commutation, rel_pos, hard_rels, control_rels))
    # the negative must be *hard*: near the source in the chart, not a random
    # far point -- the distance of T_neg(v) to v versus a random vector
    v = _unit(transport.prepare(vectors.float()))
    moved_neg = _unit(transport.apply(v, rel_neg))
    rand = _unit(torch.randn_like(v))
    out["neg_self_dist"] = float((1.0 - (moved_neg * v).sum(dim=-1)).mean())
    out["rand_self_dist"] = float((1.0 - (rand * v).sum(dim=-1)).mean())
    return out
