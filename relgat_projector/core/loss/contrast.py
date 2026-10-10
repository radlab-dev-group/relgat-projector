"""
In-batch contrastive loss in retriever space (PLAN.md 2.4).

    L_con = InfoNCE( pi(T_r(x_A)), {pi(x_j)}_{j in batch}, target = pi(x_B) )

with false-negative masking: a batch vector that is a *real* target of the
relation ``r`` from ``A`` is not treated as a negative.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,      # pi(T_r(x_A))  [B, D']
        keys: torch.Tensor,       # pi(x_j)       [N, D']  (N >= B)
        target_idx: torch.Tensor, # [B] index of x_B in keys
        mask: torch.Tensor | None = None,  # [B, N] True where a false negative
    ) -> torch.Tensor:
        q = F.normalize(query, p=2, dim=-1)
        k = F.normalize(keys, p=2, dim=-1)
        logits = q @ k.t() / self.temperature  # [B, N]
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        labels = target_idx.long()
        return F.cross_entropy(logits, labels)
