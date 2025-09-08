import torch

import torch.nn.functional as F

from typing import Any, Dict

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

    @staticmethod
    def cosine_reconstruction_loss(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        1 - cosine_similarity averaged over batch.
        """
        pred_n = F.normalize(pred, p=2, dim=-1)
        tgt_n = F.normalize(target, p=2, dim=-1)
        cos = (pred_n * tgt_n).sum(dim=-1)  # [B]
        return (1.0 - cos).mean()

    @staticmethod
    def mse_reconstruction_loss(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(pred, target)

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
        if self.ranking_weight != 0.0:
            parts.append(
                self.ranking_weight * self.relgat_ranking_loss(pos_score, neg_score)
            )
        if self.pos_cosine_weight != 0.0:
            parts.append(
                self.pos_cosine_weight
                * self.cosine_reconstruction_loss(transformed_src, dst_vec)
            )

        if self.neg_cosine_weight != 0.0:
            parts.append(
                self.neg_cosine_weight
                * (
                    1.0
                    - self.cosine_reconstruction_loss(
                        transformed_src, neg_dst_vec
                    )
                )
            )
        if self.mse_weight != 0.0:
            parts.append(
                self.mse_weight
                * self.mse_reconstruction_loss(transformed_src, dst_vec)
            )
        if not parts:
            raise ValueError("At least one loss weight must be non-zero.")
        return torch.stack(parts).sum()
