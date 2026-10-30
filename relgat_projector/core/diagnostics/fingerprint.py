"""
Relational fingerprint of a space (docs/IDEAS/07-model-fingerprint-ci.md).

Embedder swaps, quantisation, aggressive fine-tunes and ontology bumps all
break a production index while the aggregate metrics stay quiet.  MMD/PSI/
Procrustes measure the *distribution* and say nothing about what broke; the
idea's claim is that the *relation structure* is the thing to diff, and that
its diff is readable:

    D_Theta[i, j] = angle( O(theta_i), O(theta_j) )   in [0, pi]

The angle between two relation rotations is invariant under a change of the
basis ``B`` (``<Q A Q^-1, Q B Q^-1>_F = <A, B>_F``), so it compares two
spaces without aligning them -- which is the whole reason it can be a gate.
Layer 2 is ``C_rs``, layer 3 the per-relation probe accuracy the CI actually
alarms on.

This module computes all three and the diff between two checkpoints, with
the two properties the document demands of the number:

* **it is zero for the same model** -- a self-diff must be 0, otherwise the
  gate alarms on its own noise;
* **the alarm names the culprit** -- ``regressed_relations`` lists the
  relations whose probe accuracy dropped, not a scalar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def theta_angle_matrix(rotation_matrices: torch.Tensor) -> np.ndarray:
    """``[R, R]`` matrix of angles between relation rotations.

    ``rotation_matrices`` is ``[R, D, D]`` -- the context-free rotation of
    every relation (:meth:`GivensRelationOperator.relation_matrices`).  The
    angle is the Frobenius one, ``arccos(<A,B>_F / (||A|| ||B||))``, which
    is invariant under ``A -> Q A Q^-1``, i.e. under a rotation of the
    chart basis -- the property the fingerprint rests on.
    """
    mats = rotation_matrices.detach().to("cpu", dtype=torch.float64)
    flat = mats.reshape(mats.shape[0], -1)
    norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    unit = flat / norms
    gram = unit @ unit.t()
    return np.arccos(np.clip(gram.numpy(), -1.0 + 1e-12, 1.0 - 1e-12))


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(matrix.shape[0], k=1)
    return matrix[iu]


def probe_accuracy_per_relation(
    model,
    pairs: Sequence[Tuple[int, int, int]],
    mode: str = "ensemble",
    score_stats=None,
    train_counts=None,
    node_repr: Optional[torch.Tensor] = None,
) -> Dict[int, Tuple[int, int]]:
    """Per-relation ``(correct, total)`` of the relation decoder on a probe set.

    ``pairs`` are ``(row_a, row_b, rel_id)`` indices into ``model.node_emb_fixed``.
    Per relation, never averaged across relations: in practice exactly one
    relation breaks (usually the negation) and a global mean hides it -- the
    reason the document asks for per-relation gates.

    ``node_repr`` is the matrix the pairs are read from, and it has to be the
    matrix the decoder was scored on: the trainer evaluates ``rel_acc`` on
    :meth:`WiazkaModel.refine`, so feeding it the raw chart instead measures a
    different model (measured: macro 0.070 on the chart against 0.476 on the
    refined vectors -- the same mistake as the 12a floor, in the other direction).
    ``None`` falls back to the chart, which is only correct for a checkpoint
    trained without refinement layers.
    """
    if not pairs:
        return {}
    rows_a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    rows_b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    rel = torch.tensor([p[2] for p in pairs], dtype=torch.long)
    device = next(model.parameters()).device
    with torch.no_grad():
        if node_repr is None:
            nodes = model.base_to_op(model.node_emb_fixed.to(device))
        else:
            nodes = node_repr.to(device)
        x, y = nodes[rows_a], nodes[rows_b]
        evidence = model.relation_logits(
            x, y, mode=mode, score_stats=score_stats, train_counts=train_counts)
    pred = evidence.argmax(dim=-1)
    hits = (pred == rel.to(device)).cpu().numpy()
    rels = rel.cpu().numpy()
    out: Dict[int, Tuple[int, int]] = {}
    for r, ok in zip(rels, hits):
        c, t = out.get(int(r), (0, 0))
        out[int(r)] = (c + int(ok), t + 1)
    return out


def applicability_profile(model, vectors: torch.Tensor,
                          rel_ids: torch.Tensor) -> np.ndarray:
    """``[B, R]`` applicability of every relation at every probe vector."""
    device = next(model.parameters()).device
    with torch.no_grad():
        x = model.base_to_op(vectors.to(device))
        out = torch.zeros(x.shape[0], rel_ids.numel(), device=device)
        for k, r in enumerate(rel_ids.tolist()):
            out[:, k] = torch.sigmoid(model.applicability_score(
                x, torch.full((x.shape[0],), int(r), device=device,
                              dtype=torch.long)))
    return out.cpu().numpy()


def fingerprint(model, commutation: Optional[np.ndarray],
                probes: Optional[np.ndarray] = None) -> Dict:
    """The three layers of the idea, as one comparable dict."""
    from relgat_projector.core.operator.givens import GivensRelationOperator

    op = model.operator
    if not isinstance(op, GivensRelationOperator):
        raise TypeError(f"fingerprint needs a Givens operator, got {type(op)}")
    with torch.no_grad():
        rotations = op.relation_matrices()
    out: Dict = {
        "theta_angles": theta_angle_matrix(rotations),
        "log_scale": op.log_scale.detach().cpu().numpy().astype("float64"),
    }
    if commutation is not None:
        out["commutation"] = np.asarray(commutation, dtype="float64")
    if probes is not None:
        out["probes"] = np.asarray(probes, dtype="float64")
    return out


def fingerprint_delta(old: Dict, new: Dict,
                      probe_accuracy_old: Optional[Dict[int, Tuple[int, int]]] = None,
                      probe_accuracy_new: Optional[Dict[int, Tuple[int, int]]] = None,
                      min_probes: int = 20,
                      tol: float = 0.05) -> Dict[str, object]:
    """The readable diff of two fingerprints.

    ``d_theta_angles`` is the Frobenius norm of the upper triangle's
    difference; ``d_commutation`` the same for ``C_rs``;
    ``regressed_relations`` lists ``(relation, old_acc, new_acc)`` for every
    relation whose accuracy dropped by more than ``tol``, only when the
    probe set is large enough to say anything (``min_probes``) -- the
    document's "not enough probes is *no data*, not OK".
    """
    d_theta = float(np.linalg.norm(_upper_triangle(
        new["theta_angles"] - old["theta_angles"])))
    out: Dict[str, object] = {"d_theta_angles": d_theta,
                              "theta_angles_rel": d_theta / max(
                                  1e-12, float(np.linalg.norm(
                                      _upper_triangle(old["theta_angles"]))))}
    if "commutation" in old and "commutation" in new:
        out["d_commutation"] = float(np.linalg.norm(
            new["commutation"] - old["commutation"]))
        out["c_moved_pairs"] = _moved_pairs(old["commutation"], new["commutation"])
    if probe_accuracy_old is not None and probe_accuracy_new is not None:
        regressed = []
        for r, (c_new, n_new) in sorted(probe_accuracy_new.items()):
            c_old, n_old = probe_accuracy_old.get(r, (0, 0))
            if n_new < min_probes or n_old < min_probes:
                continue
            a_old, a_new = c_old / n_old, c_new / n_new
            if a_new < a_old - tol:
                regressed.append((int(r), round(a_old, 4), round(a_new, 4)))
        out["regressed_relations"] = regressed
        out["probe_min_count"] = int(min_probes)
    return out


def _moved_pairs(old: np.ndarray, new: np.ndarray,
                 threshold: float = 0.2) -> List[Tuple[int, int, float, float]]:
    """Relation pairs whose commutation value moved by more than ``threshold``."""
    diff = np.abs(np.asarray(new) - np.asarray(old))
    iu = np.triu_indices(diff.shape[0], k=1)
    rows, cols = iu[0], iu[1]
    keep = diff[rows, cols] > float(threshold)
    return [(int(rows[k]), int(cols[k]),
             float(old[rows[k], cols[k]]), float(new[rows[k], cols[k]]))
            for k in np.flatnonzero(keep)]


def detection_auc(control: Sequence[float], alarm: Sequence[float]) -> float:
    """How well the fingerprint separates regressions from the noise floor.

    ``control`` is the self-diff of unchanged models (the noise level a gate
    has to sit above), ``alarm`` the diff under an injected regression.
    Reported per regression type, with the control always printed next to
    it -- a detection number without its noise floor is not a result.
    """
    from .consistency import roc_auc

    labels = [0.0] * len(control) + [1.0] * len(alarm)
    return roc_auc(labels, list(control) + list(alarm))
