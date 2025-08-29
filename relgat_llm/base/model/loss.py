import torch

import torch.nn.functional as F


class RelGATLoss(torch.nn.Module):
    @staticmethod
    def margin_ranking_loss(
        pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 1.0
    ):
        pos = pos_score.unsqueeze(1).expand_as(neg_score)
        loss = F.relu(margin + neg_score - pos)
        return loss.mean()

    @staticmethod
    def self_adversarial_loss(
        pos_score: torch.Tensor, neg_score: torch.Tensor, alpha: float = 1.0
    ):
        """
        pos_score: [B], neg_score: [B, K]
        L = -log σ(pos) - Σ_i softmax(α * neg_i) * log σ(-neg_i)
        Wagi liczone bez gradientu (detach) dla stabilności.
        """
        # Klamrowanie dla stabilności i usunięcie niefinitych
        pos = torch.nan_to_num(pos_score, nan=0.0, neginf=-20.0, posinf=20.0).clamp(
            -20.0, 20.0
        )
        neg = torch.nan_to_num(neg_score, nan=0.0, neginf=-20.0, posinf=20.0).clamp(
            -20.0, 20.0
        )

        with torch.no_grad():
            weights = torch.softmax(alpha * neg, dim=1)  # [B, K]
        pos_loss = -F.logsigmoid(pos).mean()
        neg_loss = -(weights * F.logsigmoid(-neg)).sum(dim=1).mean()
        return pos_loss + neg_loss
