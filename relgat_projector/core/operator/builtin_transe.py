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
        self.ctor_kwargs: dict = {
            "type": "transe",
            "kwargs": {"dim": dim, "num_rel": num_rel, "normalize": normalize},
        }

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
            # translate along a unit direction; re-projecting x + r onto the
            # sphere would break the isometry this operator claims
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
        out = x + r
        if lambdas is not None:
            lam = lambdas.reshape(-1, 1)
            out = (1.0 - lam) * x + lam * out
        return out

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor,
                base_point: torch.Tensor | None = None) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
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
