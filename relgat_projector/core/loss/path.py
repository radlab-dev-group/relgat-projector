"""
Path (k >= 2) composition loss (PLAN.md 2.2).

    L_path = mean 1 - cos( T_{r_k} o ... o T_{r_1}(x_A), x_B )

Also supports the *compiled* variant (same target, computed with the
single folded operator) so that the chain is equivalent to one matmul also
numerically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PathLoss(nn.Module):
    def __init__(self, compiled_weight: float = 0.5):
        super().__init__()
        self.compiled_weight = compiled_weight

    def forward(
        self,
        chain_transformed: torch.Tensor,   # [B, D] via apply_chain
        compiled_transformed: torch.Tensor | None,  # [B, D] via compiled
        target: torch.Tensor,              # [B, D]
    ) -> torch.Tensor:
        cos = torch.nn.functional.cosine_similarity(chain_transformed, target, dim=-1)
        loss = (1.0 - cos).mean()
        if compiled_transformed is not None and self.compiled_weight > 0:
            cos_c = torch.nn.functional.cosine_similarity(
                compiled_transformed, target, dim=-1
            )
            loss = loss + self.compiled_weight * (1.0 - cos_c).mean()
        return loss
