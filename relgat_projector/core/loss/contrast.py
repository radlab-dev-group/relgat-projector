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
        hard_negatives: int = 0,  # top-K distractors scored instead of all N
    ) -> torch.Tensor:
        q = F.normalize(query, p=2, dim=-1)
        k = F.normalize(keys, p=2, dim=-1)
        # clamp so that a near-duplicate (e.g. a synonym that is already almost
        # identical to the transformed query) cannot push its logit to -inf and
        # make softmax(0/0) -> NaN; 1e4 is still ~e^-139000, i.e. a hard floor
        logits = (q @ k.t() / self.temperature).clamp(min=-1e4)  # [B, N]
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        labels = target_idx.long()
        if hard_negatives <= 0 or hard_negatives >= logits.shape[1] - 1:
            return F.cross_entropy(logits, labels)
        # hard-negative variant: score the target against its ``K`` most
        # similar distractors only, instead of the full key set.  The full
        # softmax puts ~1e-4 of its mass on 243k irrelevant nodes, so after a
        # few epochs the gradient is dominated by "be a little more different
        # from everything" rather than "beat the confusable ones"; the top-K
        # cut keeps exactly the ranking signal (target above its nearest
        # rivals) and costs ``B x (K+1)`` instead of ``B x N``.  K >= N-1
        # reproduces the full softmax exactly, so the default (0) is the
        # historical path bit for bit.
        masked = logits.masked_fill(
            F.one_hot(labels, num_classes=logits.shape[1]).bool(),
            float("-inf")
        )
        rivals = masked.topk(int(hard_negatives), dim=1).values  # [B, K]
        pos = logits.gather(1, labels[:, None])                 # [B, 1]
        return F.cross_entropy(torch.cat([pos, rivals], dim=1),
                               torch.zeros(pos.shape[0], dtype=torch.long,
                                           device=logits.device))
