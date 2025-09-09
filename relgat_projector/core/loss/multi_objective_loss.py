import torch

from typing import Any, Dict

from relgat_projector.core.loss.cosine import CosineLoss
from relgat_projector.core.loss.mse import MSELoss
from relgat_projector.core.loss.relgat_loss import RelGATLoss


class MultiObjectiveRelLoss:
    """
    Kombinuje klasyczny ranking loss z rekonstrukcyjnymi stratami A->r->B:
    - ranking_loss: margin/self-adversarial na score'ach
    - cosine_recon: 1 - cosine(f_r(A), B)
    - mse_recon: MSE(f_r(A), B)
    Wagi poszczególnych komponentów konfigurowalne.
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
    ):
        self.ranking_weight = run_config.get("relgat_weight", relgat_weight)
        self.pos_cosine_weight = run_config.get(
            "pos_cosine_weight", pos_cosine_weight
        )
        self.neg_cosine_weight = run_config.get(
            "neg_cosine_weight", neg_cosine_weight
        )
        self.mse_weight = run_config.get("mse_weight", mse_weight)

        self.relgat_loss = relgat_loss

    def relgat_ranking_loss(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> torch.Tensor:
        return self.relgat_loss.prepare_scores_and_compute_loss(
            pos_score=pos_score, neg_score=neg_score
        )

    def __call__(
        self,
        *,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        transformed_src: torch.Tensor,  # f_r(A)
        dst_vec: torch.Tensor,  # B
        neg_dst_vec: torch.Tensor,  # [B NumNeg]
    ) -> torch.Tensor:
        parts = []
        weights = []
        if self.ranking_weight != 0.0:
            weights.append(self.ranking_weight)
            parts.append(
                self.ranking_weight * self.relgat_ranking_loss(pos_score, neg_score)
            )
        if self.pos_cosine_weight != 0.0:
            weights.append(self.pos_cosine_weight)
            parts.append(
                self.pos_cosine_weight
                * CosineLoss.calculate(transformed_src, dst_vec)
            )

        if self.neg_cosine_weight != 0.0:
            weights.append(self.neg_cosine_weight)
            parts.append(
                self.neg_cosine_weight
                * (1.0 - CosineLoss.calculate(transformed_src, neg_dst_vec))
            )
        if self.mse_weight != 0.0:
            weights.append(self.mse_weight)
            parts.append(
                self.mse_weight * MSELoss.calculate(transformed_src, dst_vec)
            )
        if not parts:
            raise ValueError("At least one loss weight must be non-zero.")
        return torch.stack(parts).sum() / sum(weights)
