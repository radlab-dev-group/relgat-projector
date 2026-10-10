"""
Holonomy / curvature loss (PLAN.md 2.3) -- the heart of the method.

For a closed quadruple around ``A`` with transport holonomy

    H = T_p^{-1} o T_q o T_r o T_s

we penalise  ``|| H(x_A) - x_A ||^2``.  Additionally, a *learned commutation
matrix* ``C_{rs} in [0,1]`` (one scalar per unordered relation pair) decides
which relation pairs should commute:

    L_commut = w_com  * C_{rs}  * || T_r T_s x - T_s T_r x ||^2
             + w_anti * (1 - C) * margin( || T_r T_s x - T_s T_r x || )

The second term keeps non-commuting pairs from commuting *accidentally*; the
model only has to be *consistent*, not fully commutative.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CurvatureLoss(nn.Module):
    def __init__(self, num_rel: int, comm_weight: float = 0.5,
                 anti_weight: float = 0.25, margin: float = 0.25):
        super().__init__()
        self.num_rel = int(num_rel)
        self.comm_weight = comm_weight
        self.anti_weight = anti_weight
        self.margin = margin
        # C_{rs} as logits over the symmetric matrix; C = sigmoid(C_logit)
        self.comm_logit = nn.Parameter(torch.zeros(num_rel, num_rel))

    def commutation(self) -> torch.Tensor:
        """Symmetric commutation matrix in [0, 1]^2."""
        c = torch.sigmoid(self.comm_logit)
        c = 0.5 * (c + c.t())
        return c

    def forward(
        self,
        x_a: torch.Tensor,                # [B, D] base points
        holonomy_x_a: torch.Tensor,       # [B, D]  = H(x_A)
        r: torch.Tensor,                  # [B] first route's relation id
        s: torch.Tensor,                  # [B] second route's relation id
        ts_tr: torch.Tensor,              # [B, D]  first route result
        tr_ts: torch.Tensor,              # [B, D]  second route result
    ) -> torch.Tensor:
        hol = (holonomy_x_a - x_a).pow(2).mean(dim=-1)
        loss = hol.mean()

        c = self.commutation()
        c_rs = c[r, s]  # [B]
        gap = (ts_tr - tr_ts).pow(2).mean(dim=-1)  # [B]
        # eps inside sqrt: commuting pairs can give a difference of exactly 0
        gap_l2 = torch.sqrt(gap + 1e-12)
        comm_part = self.comm_weight * (c_rs * gap).mean()
        anti_part = self.anti_weight * (
            (1.0 - c_rs) * (gap_l2 - self.margin).clamp_min(0.0)
        ).mean()
        loss = loss + comm_part + anti_part
        return loss

    def supervision(
        self,
        pos_pairs: list,
        neg_pairs: list,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """Supervised learning of the commutation matrix ``C_rs``.

        ``C`` has no gradient that would keep it away from 0 in the unsupervised
        term above (it only multiplies a non-negative gap), so it must be
        taught from the graph: pairs whose two routes meet in the same node
        commute (label 1), pairs that were never witnessed commuting must not
        be assumed to commute (label 0).  Without this term ``C`` collapses to
        zero and the learned commutation map is meaningless.
        """
        dev = self.comm_logit.device
        terms = []
        if len(pos_pairs):
            r = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=dev)
            s = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=dev)
            terms.append(
                nn.functional.binary_cross_entropy_with_logits(
                    self.comm_logit[r, s], torch.ones_like(r, dtype=torch.float)
                )
            )
        if len(neg_pairs):
            r = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long, device=dev)
            s = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long, device=dev)
            terms.append(
                nn.functional.binary_cross_entropy_with_logits(
                    self.comm_logit[r, s], torch.zeros_like(r, dtype=torch.float)
                )
            )
        if not terms:
            return self.comm_logit.new_zeros(())
        return weight * torch.stack(terms).mean()
