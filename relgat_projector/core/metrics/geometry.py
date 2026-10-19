"""
Geometry metrics that make step (3) verifiable (PLAN.md section 5).

* **PCE** -- Path Compilation Error  ``1 - cos(T_compiled(x), target)``
  reported as a *curve* over path length ``k = 1..K``.
* **HD**  -- Holonomy Defect: mean deviation of the closed-loop transport
  from the identity.
* **CCE** -- Cycle Closure Error ``|| T^{-1}(T(x)) - x ||``.
* **ISOT** -- Isometry Error ``|cos(u, v) - cos(Tu, Tv)|``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


@torch.no_grad()
def pce_curve(
    operator,
    x: torch.Tensor,
    chains: Dict[int, torch.Tensor],   # k -> [B, k] relation ids
    targets: Dict[int, torch.Tensor],  # k -> [B, D]
    max_k: int = 4,
) -> List[float]:
    """Return PCE for each path length ``k = 1..max_k``."""
    out = []
    for k in range(1, max_k + 1):
        if k not in chains or k not in targets:
            out.append(float("nan"))
            continue
        direct = operator.apply_chain(x, chains[k])
        cos = torch.nn.functional.cosine_similarity(direct, targets[k], dim=-1)
        out.append(float((1.0 - cos).mean().item()))
    return out


@torch.no_grad()
def holonomy_defect(operator, x: torch.Tensor, quadruples) -> float:
    """Mean || H(x_A) - x_A || over closed quadruples."""
    total = 0.0
    n = 0
    for d in quadruples:
        xa = x[d.a:d.a + 1]
        # H = T_p^-1 o T_s^-1 o T_q o T_r  (loop based at A)
        h = operator.apply(xa, torch.tensor([d.r], device=x.device))
        h = operator.apply(h, torch.tensor([d.q], device=x.device))
        h = operator.inverse(h, torch.tensor([d.s], device=x.device))
        h = operator.inverse(h, torch.tensor([d.p], device=x.device))
        total += float((h - xa).pow(2).sum(dim=-1).sqrt().mean().item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def cycle_closure_error(operator, x: torch.Tensor, rel_ids: torch.Tensor) -> float:
    t = operator.apply(x, rel_ids)
    back = operator.inverse(t, rel_ids)
    return float((back - x).pow(2).sqrt().mean().item())


@torch.no_grad()
def isometry_error(
    operator, u: torch.Tensor, v: torch.Tensor, rel_id: int | torch.Tensor
) -> float:
    if not isinstance(rel_id, torch.Tensor):
        rel_id = torch.tensor(int(rel_id), device=u.device)
    ru = operator.apply(u, rel_id)
    rv = operator.apply(v, rel_id)
    cos_in = torch.nn.functional.cosine_similarity(u, v, dim=-1)
    cos_out = torch.nn.functional.cosine_similarity(ru, rv, dim=-1)
    return float((cos_in - cos_out).abs().mean().item())
