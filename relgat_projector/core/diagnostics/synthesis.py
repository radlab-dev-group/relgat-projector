"""
Relational constraint synthesis (docs/IDEAS/11-constrained-synthesis.md).

"Give me the opposite of X, but not something that smells like Z" is the most
common requirement in retrieval that cosine cannot answer, and every system
implements it as a post-filter or as `q - z`, with no guarantee of anything.
The idea's claim is structural: because `T_r` is linear and a similarity,

    cos(T_r v, a) = cos(v, T_r^{-1} a)      **exactly**

so the relational constraint is an ordinary cosine constraint on the rotated
target, and the problem

    maximise cos(T_r v, a)   subject to   cos(T_s v, b) <= tau

has a closed form on the sphere:

    a' = T_r^{-1} a,  b' = T_s^{-1} b
    v* = a'                       if cos(a', b') <= tau
    v* ∝ a' - cos(a', b') b'      otherwise   (the part of a' perpendicular to b')

This module is interesting as a **diagnostic** for one reason: the identity
above is not a theorem about the learned operator, it is a claim about it being
conformal (`T^T T = s^2 I`), and its measured violation is a number nobody was
printing.  Three quantities, none of which needs a label:

``identity_error``  :`|cos(T_r v, a) - cos(v, T_r^{-1} a)|` -- gold is 0, and it
                    is exactly how much the closed-form reduction lies;
``satisfaction``    the constraint, measured with the *real* operator on `v*`;
``optimality_gap``  objective of `v*` against the best vector of a candidate
                    pool that also satisfies the constraint -- the closed form
                    is optimal in the reduced problem, so a gap here is the
                    reduction's fault, not the solver's.

The floor is `q - z`, the vector subtraction people actually ship, scored on
the same objective and the same constraint.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def identity_error(transport, probes: torch.Tensor, rel: str,
                   others: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """How badly ``cos(T_r v, a) == cos(v, T_r^{-1} a)`` fails.

    Gold is ``0.0``: for a map with ``T^T T = s^2 I`` the two cosines are the
    same number, so anything above zero is the error the closed-form synthesis
    inherits.  ``probes`` are the vectors ``v`` (rows), ``others`` the targets
    ``a`` (default: the probes themselves against each other).
    """
    v = _unit(transport.prepare(probes.float()))
    a = v if others is None else _unit(transport.prepare(others.float()))
    moved = _unit(transport.apply(v, rel))
    back = _unit(transport.inverse(a, rel))
    lhs = moved @ a.T                    # cos(T_r v_i, a_j)
    rhs = v @ back.T                     # cos(v_i, T_r^{-1} a_j)
    err = (lhs - rhs).abs()
    off = ~torch.eye(err.shape[0], dtype=torch.bool, device=err.device)
    take = err[off] if err.shape[0] == err.shape[1] and off.any() else err.flatten()
    t = take.detach().cpu().numpy()
    return {
        "identity_error_mean": float(np.mean(t)),
        "identity_error_p95": float(np.percentile(t, 95)),
        "identity_error_max": float(np.max(t)),
    }


def synthesize(transport, goal: torch.Tensor, forbid: torch.Tensor,
               rel: str, forbid_rel: str, tau: float = 0.5) -> Dict[str, torch.Tensor]:
    """Closed-form solution of the constraint problem, on the unit sphere.

    ``goal`` is ``a`` (what ``T_r v`` should look like), ``forbid`` is ``b``
    (what ``T_s v`` must not look like).  Returns the direction ``v`` plus the
    two rotated targets, so the caller can measure the objective and the
    constraint with the real operator instead of trusting the reduction.
    """
    a = _unit(transport.inverse(transport.prepare(goal.float()), rel))
    b = _unit(transport.inverse(transport.prepare(forbid.float()), forbid_rel))
    c = float((a * b).sum())
    v = a if c <= float(tau) else (a - c * b)
    return {"v": _unit(v), "a_rot": a, "b_rot": b,
            "cos_rot": torch.tensor(c),
            "constraint_active": torch.tensor(c > float(tau))}


def score_candidate(transport, v: torch.Tensor, goal: torch.Tensor,
                    forbid: torch.Tensor, rel: str, forbid_rel: str,
                    tau: float) -> Dict[str, float]:
    """Objective and constraint of one candidate, measured on the real operator."""
    vv = _unit(transport.prepare(v.float()))
    obj = float((_unit(transport.apply(vv, rel))
                 * _unit(transport.prepare(goal.float()))).sum())
    con = float((_unit(transport.apply(vv, forbid_rel))
                 * _unit(transport.prepare(forbid.float()))).sum())
    return {"objective": obj, "constraint": con,
            "satisfied": float(con <= float(tau))}


def synthesis_report(transport, rel: str, forbid_rel: str,
                     goals: torch.Tensor, forbids: torch.Tensor,
                     candidates: torch.Tensor,
                     taus: Sequence[float] = (0.3, 0.5, 0.7)) -> Dict[str, float]:
    """One row of the benchmark: closed form vs the `q - z` baseline.

    ``candidates`` is the pool standing in for an index: the brute-force best
    over it (restricted to the feasible ones) is the reference the closed form
    is compared with, and ``q - z`` is the baseline everybody ships.
    """
    out: Dict[str, float] = {}
    rows: List[Dict[str, float]] = []
    n = min(int(goals.shape[0]), int(forbids.shape[0]))
    for tau in taus:
        for k in range(n):
            sol = synthesize(transport, goals[k], forbids[k], rel, forbid_rel, tau)
            r = score_candidate(transport, sol["v"], goals[k], forbids[k],
                                rel, forbid_rel, tau)
            # baseline: the vector subtraction, on the same two measurements
            naive = _unit(transport.prepare((goals[k] - forbids[k]).float()))
            base = score_candidate(transport, naive, goals[k], forbids[k],
                                   rel, forbid_rel, tau)
            # brute force over the pool, among the feasible candidates only
            best = float("-inf")
            for c in range(int(candidates.shape[0])):
                sc = score_candidate(transport, candidates[c], goals[k],
                                     forbids[k], rel, forbid_rel, tau)
                if sc["satisfied"] >= 1.0 and sc["objective"] > best:
                    best = sc["objective"]
            r.update(baseline_objective=base["objective"],
                     baseline_satisfied=base["satisfied"],
                     brute_force_objective=(best if best != float("-inf")
                                            else float("nan")),
                     tau=float(tau))
            rows.append(r)
        for key, name in (("objective", "closed"), ("baseline_objective", "naive"),
                          ("satisfied", "satisfied"),
                          ("baseline_satisfied", "naive_satisfied")):
            vals = [r[key] for r in rows if r["tau"] == tau]
            out[f"{name}_objective@{tau}"] = float(np.mean(vals))
            out[f"{name}_satisfied@{tau}"] = float(np.mean(vals))
        gap = [r["brute_force_objective"] - r["objective"] for r in rows
               if r["tau"] == tau and np.isfinite(r["brute_force_objective"])]
        out[f"optimality_gap@{tau}"] = (float(np.mean(gap)) if gap
                                        else float("nan"))
    out.update(identity_error(transport, goals[:min(16, n)] if n else goals, rel))
    out["n_solutions"] = float(len(rows))
    return out
