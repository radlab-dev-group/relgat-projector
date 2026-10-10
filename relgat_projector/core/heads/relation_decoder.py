"""
Relation decoder: the *inverse* problem (PLAN.md 1.5).

Given a pair of vectors ``(x, y)`` it recovers the coordinates of the
transport that maps ``x`` to ``y``, i.e. predicts which relation (or mixture
of relations) maps one vector to another.  Enables automatic query rewriting,
graph completion and relation arithmetic ``r_a o r_b^{-1}``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationDecoder(nn.Module):
    def __init__(self, dim: int, num_rel: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out_rel = nn.Linear(hidden, num_rel)
        self.out_conf = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Return ``(logits [B, num_rel], confidence [B])``."""
        h = self.net(torch.cat([x, y], dim=-1))
        logits = self.out_rel(h)
        conf = torch.sigmoid(self.out_conf(h)).squeeze(-1)
        return logits, conf

    @torch.no_grad()
    def decode(self, x: torch.Tensor, y: torch.Tensor):
        """Return ``(rel_ids [B], confidence [B])`` for the top relation."""
        logits, conf = self.forward(x, y)
        rel_ids = logits.argmax(dim=-1)
        return rel_ids, conf
