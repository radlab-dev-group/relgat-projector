"""DistMult baseline as a :class:`RelationalOperator`."""

from __future__ import annotations

import torch
import torch.nn as nn

from relgat_projector.core.operator.base import RelationalOperator


class DistMultOperator(nn.Module, RelationalOperator):
    """``T_r(x) = x * r`` (diagonal, commutative -- used as a baseline)."""

    closed_composition = True
    exact_inverse = False
    is_isometry = False

    def __init__(self, dim: int, num_rel: int, normalize: bool = True):
        super().__init__()
        self._dim = dim
        self._num_rel = num_rel
        self.normalize = normalize
        self.rel_emb = nn.Embedding(num_rel, dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_rel(self) -> int:
        return self._num_rel

    def apply(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
        out = x * r
        if lambdas is not None:
            out = (1.0 - lambdas).unsqueeze(-1) * x + lambdas.unsqueeze(-1) * out
        return out

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
        return x / r.clamp_min(1e-6) * torch.where(
            r.abs() < 1e-6, torch.ones_like(r), torch.ones_like(r)
        )

    def score(
        self, src: torch.Tensor, rel_ids: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            src = torch.nn.functional.normalize(src, p=2, dim=-1)
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
            dst = torch.nn.functional.normalize(dst, p=2, dim=-1)
        return (src * r * dst).sum(dim=-1)
