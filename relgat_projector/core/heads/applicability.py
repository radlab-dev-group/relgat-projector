"""
Applicability & drift-budget head (PLAN.md 1.4).

Predicts, for a relation ``r`` and a base point ``x``:

    a_r(x) in [0, 1]   -- probability the relation is defined at x
    b_r(x) in [0, 1]   -- remaining "drift budget" (how much more transform
                          the neighbourhood of x can tolerate)

Used both as supervision (real edges = 1, type-invalid / random pairs = 0)
and at inference to decide when a transformed query is no longer trustworthy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ApplicabilityHead(nn.Module):
    def __init__(self, dim: int, num_rel: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out_a = nn.Linear(hidden, num_rel)
        self.out_b = nn.Linear(hidden, num_rel)

    def forward(self, x: torch.Tensor):
        """Return ``(a, b)`` each ``[B, num_rel]`` (logits for a, sigmoid for b)."""
        h = self.net(x)
        a = self.out_a(h)
        b = torch.sigmoid(self.out_b(h))
        return a, b

    def score(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """Applicability of ``rel_ids`` at ``x`` -> logits ``[B]``."""
        a, _ = self(x)
        return a[torch.arange(x.shape[0], device=x.device), rel_ids]

    def budget(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        _, b = self(x)
        return b[torch.arange(x.shape[0], device=x.device), rel_ids]
