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
        self.ctor_kwargs: dict = {
            "type": "distmult",
            "kwargs": {"dim": dim, "num_rel": num_rel, "normalize": normalize},
        }

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_rel(self) -> int:
        return self._num_rel

    def _relation(self, rel_ids: torch.Tensor) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            # normalise the *relation* only: the vector's own norm survives, so
            # T^{-1} is exact and cosine geometry is not destroyed by rescaling
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
        return r

    def apply(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = x * self._relation(rel_ids)
        if lambdas is not None:
            # interpolate in the input space: lambda = 0 is exactly identity
            lam = lambdas.reshape(-1, 1)
            out = (1.0 - lam) * x + lam * out
        return out

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor,
                base_point: torch.Tensor | None = None) -> torch.Tensor:
        r = self._relation(rel_ids)
        # coordinates that a unit relation made (near-)zero carry no signal:
        # treat them as identity instead of dividing by them
        safe = torch.where(r.abs() > 1e-4, r, torch.ones_like(r))
        return x / safe

    def score(
        self, src: torch.Tensor, rel_ids: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        r = self.rel_emb(rel_ids)
        if self.normalize:
            src = torch.nn.functional.normalize(src, p=2, dim=-1)
            r = torch.nn.functional.normalize(r, p=2, dim=-1)
            dst = torch.nn.functional.normalize(dst, p=2, dim=-1)
        return (src * r * dst).sum(dim=-1)
