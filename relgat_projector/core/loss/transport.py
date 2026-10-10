"""
Edge (k=1) transport fidelity loss (PLAN.md 2.1).

    L_edge = w_d * (1 - cos(T_r(x_A), x_B)) + w_g * BCE(a_r(x_A), 1)

Replaces the old "ranking + cosine + MSE" trio for k=1; the ranking part
becomes the contrast (2.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransportLoss(nn.Module):
    def __init__(self, drift_weight: float = 1.0, gate_weight: float = 0.1):
        super().__init__()
        self.drift_weight = drift_weight
        self.gate_weight = gate_weight

    def forward(
        self,
        transformed: torch.Tensor,   # T_r(x_A)   [B, D]
        target: torch.Tensor,        # x_B        [B, D]
        applicability: torch.Tensor | None = None,  # a_r(x_A) [B]
    ) -> torch.Tensor:
        cos = torch.nn.functional.cosine_similarity(transformed, target, dim=-1)
        drift = (1.0 - cos).mean()
        loss = self.drift_weight * drift
        if applicability is not None and self.gate_weight > 0:
            loss = loss + self.gate_weight * nn.functional.binary_cross_entropy_with_logits(
                applicability, torch.ones_like(applicability)
            )
        return loss
