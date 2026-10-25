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
                 anti_weight: float = 0.25, margin: float = 0.25,
                 margin_auto: bool = False, margin_floor: float = 0.02):
        super().__init__()
        self.num_rel = int(num_rel)
        self.comm_weight = comm_weight
        self.anti_weight = anti_weight
        self.margin = margin
        # ``margin_auto`` rescales the anti-commutation margin to the size of
        # the ambient vectors (see :meth:`effective_margin`): a fixed 0.25 in
        # a 256-d chart of ~unit vectors means "routes must differ by at
        # least 25% in every coordinate on average", which is far above what
        # genuinely non-commuting relation pairs manage (measured gaps
        # ~0.02-0.06) and so the anti term silently trains on nothing.
        self.margin_auto = bool(margin_auto)
        self.margin_floor = float(margin_floor)
        # C_{rs} as logits over the symmetric matrix; C = sigmoid(C_logit)
        self.comm_logit = nn.Parameter(torch.zeros(num_rel, num_rel))

    def commutation(self) -> torch.Tensor:
        """Symmetric commutation matrix in [0, 1]^2."""
        c = torch.sigmoid(self.comm_logit)
        c = 0.5 * (c + c.t())
        return c

    def effective_margin(self, x: torch.Tensor) -> torch.Tensor:
        """Anti-commutation margin, optionally scaled to the vector size.

        With ``margin_auto`` the margin is ``2 * margin_floor * ||x||``:
        the route difference ``||T_r T_s x - T_s T_r x||`` is at most
        ``~4 ||x||`` for a full-size ambient space, so the margin must grow
        with the vector norm or it is either always trivially violated (huge
        fixed margin: dead term) or always trivially satisfied (tiny margin:
        the anti term pushes commuting pairs apart).
        """
        if not self.margin_auto:
            return torch.as_tensor(self.margin, device=x.device)
        return (
            2.0 * self.margin_floor
            * x.detach().norm(dim=-1).clamp_min(1e-8)
        )

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

        # ``c_rs`` is *gated* with detach: it is a learned classification of
        # "do these relations commute?", and the only teacher for it is the
        # graph witness supervision (:meth:`supervision`).  If the unsupervised
        # gap term below could also move it, the two fight -- the anti term's
        # gradient is two orders of magnitude stronger than the comm term's, so
        # the matrix collapses toward whatever minimises the penalty of the
        # *current* transport instead of tracking the graph (measured: after a
        # full run every entry sat at 0.49 +- 0.01).
        c = self.commutation().detach()
        c_rs = c[r, s]  # [B]
        gap = (ts_tr - tr_ts).pow(2).mean(dim=-1)  # [B]
        # eps inside sqrt: commuting pairs can give a difference of exactly 0
        gap_l2 = torch.sqrt(gap + 1e-12)
        comm_part = self.comm_weight * (c_rs * gap).mean()
        margin = self.effective_margin(x_a)
        if margin.dim() == 0:
            margin = margin.expand_as(gap_l2)
        anti_part = self.anti_weight * (
            (1.0 - c_rs) * (gap_l2 - margin).clamp_min(0.0)
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

        ``C`` is the *only* thing taught by this term (the unsupervised gap
        penalty detaches it), so it has to be taught well:

        * witnesses are symmetric in ``(r, p)`` and the matrix is forced
          symmetric, so a positive (r, p) teaches BOTH cells;
        * positives and negatives are averaged over *cells*, not over the two
          batch terms -- otherwise a step with many positives and few
          negatives (the common case: the witness table has ~198 positive
          ordered pairs on the Slowosieć against ~2 900 negatives, but a step
          draws ~24 of the former and 16 of the latter) trains the two classes
          at different rates.
        """
        dev = self.comm_logit.device
        if not pos_pairs and not neg_pairs:
            return self.comm_logit.new_zeros(())

        def cells(pairs, label):
            idx = []
            for r, s in pairs:
                idx.append((r, s))
                idx.append((s, r))
            r = torch.tensor([i[0] for i in idx], dtype=torch.long, device=dev)
            s = torch.tensor([i[1] for i in idx], dtype=torch.long, device=dev)
            return nn.functional.binary_cross_entropy_with_logits(
                self.comm_logit[r, s],
                torch.full(r.shape, label, dtype=torch.float, device=dev),
            )

        parts = []
        if pos_pairs:
            parts.append(cells(pos_pairs, 1.0))
        if neg_pairs:
            parts.append(cells(neg_pairs, 0.0))
        # each class is already a mean over its own cells, so averaging the two
        # terms keeps them equally weighted whatever their sizes are
        return weight * torch.stack(parts).mean()

    def symmetrize_logits(self) -> None:
        """Make the parameter itself symmetric (idempotent, in-place)."""
        with torch.no_grad():
            self.comm_logit.copy_(0.5 * (self.comm_logit + self.comm_logit.t()))
