"""TransE baseline as a :class:`RelationalOperator`."""

from __future__ import annotations

import torch
import torch.nn as nn

from relgat_projector.core.operator.base import RelationalOperator


class TransEOperator(nn.Module, RelationalOperator):
    """``T_r(x) = x + r`` (translation -- used as a baseline)."""

    closed_composition = True
    exact_inverse = True
    is_isometry = True

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
        if lambdas is not None:
            r = r * lambdas.unsqueeze(-1)
        if self.normalize:
            return torch.nn.functional.normalize(x + r, p=2, dim=-1)
        return x + r

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            return torch.nn.functional.normalize(x - r, p=2, dim=-1)
        return x - r

    def score(
        self, src: torch.Tensor, rel_ids: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            src = torch.nn.functional.normalize(src, p=2, dim=-1)
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
            dst = torch.nn.functional.normalize(dst, p=2, dim=-1)
        return -torch.linalg.norm(src + r - dst, ord=2, dim=-1)
