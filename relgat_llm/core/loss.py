import torch

import torch.nn.functional as F

from typing import Optional, Any, Dict


class RelGATLoss:

    def __init__(
        self,
        loss_type: str,
        self_adv_alpha: Optional[float],
        margin: Optional[float],
        clamp_limit: Optional[float],
        run_config: Dict[str, Any],
    ):
        self.loss_type = loss_type
        self.clamp_limit = clamp_limit

        self.margin = run_config.get("margin", margin)
        if self.margin is not None:
            self.margin = float(self.margin)

        self.self_adv_alpha = run_config.get("self_adv_alpha", self_adv_alpha)
        if self.self_adv_alpha is not None:
            self.self_adv_alpha = float(self.self_adv_alpha)

        self.use_self_adv_neg = False
        if loss_type == "self_adversarial_loss":
            self.use_self_adv_neg = True

    def prepare_scores_and_compute_loss(
        self,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
    ) -> torch.Tensor:
        # UNCOMMENT NOTE GRADIENT TODO
        # No change to finite scores
        # pos = torch.nan_to_num(
        #     pos_score, nan=0.0, neginf=-self.clamp_limit, posinf=self.clamp_limit
        # ).clamp(-self.clamp_limit, self.clamp_limit)
        # neg = torch.nan_to_num(
        #     neg_score, nan=0.0, neginf=-self.clamp_limit, posinf=self.clamp_limit
        # ).clamp(-self.clamp_limit, self.clamp_limit)

        if self.use_self_adv_neg:
            return self._self_adversarial_loss(pos_score, neg_score)
        else:
            return self._margin_ranking_loss(pos_score, neg_score)

    def _margin_ranking_loss(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        pos = pos_score.unsqueeze(1).expand_as(neg_score)
        loss = F.relu(self.margin + neg_score - pos)
        return loss.mean()

    def _self_adversarial_loss(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ):
        """
        pos_score: [B], neg_score: [B, K]
        L = -log σ(pos_score) - Σ_i softmax(α * neg_i) * log σ(-neg_i)
        Wagi liczone bez gradientu (detach) dla stabilności.
        """
        with torch.no_grad():
            # [B, K]
            weights = torch.softmax(self.self_adv_alpha * neg_score, dim=1)

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -(weights * F.logsigmoid(-neg_score)).sum(dim=1).mean()

        return pos_loss + neg_loss
