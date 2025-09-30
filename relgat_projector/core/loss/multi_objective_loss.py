"""Composite loss for RelGAT combining ranking, cosine and MSE components."""

import torch
import torch.nn as nn
from typing import Any, Dict

from relgat_projector.core.loss.cosine import CosineLoss
from relgat_projector.core.loss.mse import MSELoss
from relgat_projector.core.loss.relgat_loss import RelGATLoss


class MultiObjectiveRelLoss(nn.Module):
    """
    Combines ranking loss (via :class:`RelGATLoss`) with optional
    cosine‑similarity and mean‑squared‑error reconstruction terms.

    All component weights are read from ``run_config`` but can be overridden
    via the constructor.  At least one weight must be non‑zero.
    """

    def __init__(
        self,
        *,
        relgat_loss: RelGATLoss,
        run_config: Dict[str, Any],
        pos_cosine_weight: float = 1.0,
        neg_cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
        relgat_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ranking_weight: float = float(
            run_config.get("relgat_weight", relgat_weight)
        )
        self.pos_cosine_weight: float = float(
            run_config.get("pos_cosine_weight", pos_cosine_weight)
        )
        self.neg_cosine_weight: float = float(
            run_config.get("neg_cosine_weight", neg_cosine_weight)
        )
        self.mse_weight: float = float(run_config.get("mse_weight", mse_weight))

        self.relgat_loss = relgat_loss

        # sanity check – at least one component must contribute
        if all(
            w == 0.0
            for w in [
                self.ranking_weight,
                self.pos_cosine_weight,
                self.neg_cosine_weight,
                self.mse_weight,
            ]
        ):
            raise ValueError("At least one loss weight must be non‑zero.")

    # --------------------------------------------------------------------- #
    # Helper utilities
    # --------------------------------------------------------------------- #
    def _weighted_part(self, weight: float, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``weight * tensor`` (or ``0`` if ``weight`` is zero)."""
        return (
            weight * tensor
            if weight != 0.0
            else torch.tensor(0.0, device=tensor.device)
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def relgat_ranking_loss(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> torch.Tensor:
        """Delegate to the underlying :class:`RelGATLoss`."""
        return self.relgat_loss.prepare_scores_and_compute_loss(
            pos_score=pos_score, neg_score=neg_score
        )

    def forward(
        self,
        *,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        transformed_src: torch.Tensor,  # f_r(A)
        dst_vec: torch.Tensor,  # B
        neg_dst_vec: torch.Tensor,  # [B, NumNeg]
    ) -> torch.Tensor:
        """Compute the weighted sum of all active loss components."""
        parts = []
        weights = []

        # ranking loss
        if self.ranking_weight != 0.0:
            loss = self.relgat_ranking_loss(pos_score, neg_score)
            parts.append(self._weighted_part(self.ranking_weight, loss))
            weights.append(self.ranking_weight)

        # positive cosine similarity loss (1 - cos)
        if self.pos_cosine_weight != 0.0:
            loss = CosineLoss.calculate(transformed_src, dst_vec)
            parts.append(self._weighted_part(self.pos_cosine_weight, loss))
            weights.append(self.pos_cosine_weight)

        # negative cosine similarity loss (1 - cos) – note the sign inversion
        if self.neg_cosine_weight != 0.0:
            loss = 1.0 - CosineLoss.calculate(transformed_src, neg_dst_vec)
            parts.append(self._weighted_part(self.neg_cosine_weight, loss))
            weights.append(self.neg_cosine_weight)

        # MSE reconstruction loss
        if self.mse_weight != 0.0:
            loss = MSELoss.calculate(transformed_src, dst_vec)
            parts.append(self._weighted_part(self.mse_weight, loss))
            weights.append(self.mse_weight)

        total = torch.stack(parts).sum()
        return total / sum(weights)
