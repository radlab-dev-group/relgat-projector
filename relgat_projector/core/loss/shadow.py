"""
Shadow-manifold losses (PLAN.md 2.6).

The product requirement is that a relation operator applied to a *text*
vector (which is not a graph node) still lands somewhere meaningful.  These
losses supervise the operator off the graph, on vectors taken from the
retriever index:

* ``cycle``   -- ``T_r^{-1}(T_r(z)) ~= z`` for shadow vectors ``z``;
* ``commut``  -- the learned commutation structure holds off the graph too;
* ``pull``    -- the operator commutes (approximately) with the projection
  between the operator space and the retriever space, otherwise exporting a
  transformed vector to the vector database destroys the transformation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShadowLoss(nn.Module):
    def __init__(
        self,
        cycle_weight: float = 1.0,
        commut_weight: float = 0.5,
        pull_weight: float = 0.5,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.cycle_weight = cycle_weight
        self.commut_weight = commut_weight
        self.pull_weight = pull_weight
        self.eps = eps

    def cycle(self, z: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        """``back`` must be ``T_r^{-1}(T_r(z))`` for shadow vectors ``z``."""
        diff = back - z
        return self.cycle_weight * torch.sqrt(diff.pow(2).mean() + self.eps)

    def commute(self, trst: torch.Tensor, tstr: torch.Tensor) -> torch.Tensor:
        diff = trst - tstr
        return self.commut_weight * torch.sqrt(diff.pow(2).mean() + self.eps)

    def pull(
        self,
        projected_transformed: torch.Tensor,  # P(T_r(z))
        transformed_projected: torch.Tensor,  # T_r'(P(z)) in retriever space
    ) -> torch.Tensor:
        """Coherence of the operator with the projection to retriever space."""
        diff = projected_transformed - transformed_projected
        return self.pull_weight * torch.sqrt(diff.pow(2).mean() + self.eps)

    def forward(
        self,
        cycle_term: torch.Tensor | None = None,
        commute_term: torch.Tensor | None = None,
        pull_term: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = (cycle_term if cycle_term is not None else 0.0)
        if commute_term is not None:
            loss = loss + commute_term
        if pull_term is not None:
            loss = loss + pull_term
        return loss


class DecoderLoss(nn.Module):
    """Supervision for the relation decoder ``g(x, y) -> r`` (PLAN.md 1.5)."""

    def __init__(self, confidence_weight: float = 0.2):
        super().__init__()
        self.confidence_weight = confidence_weight

    def forward(
        self,
        logits: torch.Tensor,      # [B, num_rel]
        confidence: torch.Tensor,  # [B]
        rel_ids: torch.Tensor,     # [B] ground-truth relations
    ) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, rel_ids.long())
        conf = nn.functional.binary_cross_entropy_with_logits(
            confidence - confidence.mean().detach(), torch.ones_like(confidence)
        ) * self.confidence_weight
        return ce + conf
