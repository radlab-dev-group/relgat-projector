# relgat_projector/core/loss_cosine.py
import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


class InBatchCosineLoss(nn.Module):
    """
    In-batch negatives (InfoNCE/NT-Xent style) z opcjonalnym marginesem
    kątowym i skalowaniem. Anchory i pozytywy mogą być tym samym tensorem
    (symetryczna NT-Xent) albo dwoma różnymi (asymetryczna).

    tryb margin:
      - "none": klasyczny InfoNCE z temperaturą tau (skala s = 1/tau)
      - "cosface": additive margin w przestrzeni cos (CosFace)
      - "arcface": additive angular margin w przestrzeni kątowej (ArcFace)

    Parametry:
    - scale: skala logitów (s). Jeśli podasz tau, przyjmij scale = 1/tau.
    - margin: wartość marginesu (m). Typowo 0.2–0.5 dla CosFace, 0.2–0.3 dla ArcFace.
    - symmetric: jeśli True i anchors==positives, liczy średnią z kierunków (i->j, j->i)
    - hard_negatives_topk: jeśli >0, ogranicza mianownik do top-k najtrudniejszych negatywów per anchor
    - label_smoothing: niewielkie wygładzenie etykiet (np. 0.0–0.1)

    Wejście:
    - anchors: [B, D]
    - positives: [B, D]
    - mask_valid: opcjonalna maska [B] (True=wykorzystaj)
    """

    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.0,
        margin_type: Literal["none", "cosface", "arcface"] = "none",
        symmetric: bool = False,
        hard_negatives_topk: int = 0,
        label_smoothing: float = 0.0,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.s = float(scale)
        self.m = float(margin)
        self.margin_type = margin_type
        self.symmetric = symmetric
        self.topk = int(hard_negatives_topk)
        self.ls = float(label_smoothing)
        self.normalize_inputs = normalize_inputs

    def _compute_logits(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A, B: [B, D], zwraca [B, B] macierz podobieństw
        if self.normalize_inputs:
            A = l2_normalize(A)
            B = l2_normalize(B)
        sim = A @ B.t()  # cosine przy L2-norm
        return sim

    def _apply_margin(self, sim: torch.Tensor) -> torch.Tensor:
        """
        Zastosuj margines do logitu pozytywnego (diagonali).
        sim: [B, B] (cosine)
        """
        if self.margin_type == "none" or self.m == 0.0:
            return sim

        B = sim.size(0)
        device = sim.device
        eye = torch.eye(B, device=device, dtype=torch.bool)

        if self.margin_type == "cosface":
            # CosFace: s*(cos(theta_y) - m)
            sim_m = sim.clone()
            sim_m[eye] = sim_m[eye] - self.m
            return sim_m

        if self.margin_type == "arcface":
            # ArcFace: s*cos(theta_y + m)
            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            cos_m = math.cos(self.m)
            sin_m = math.sin(self.m)
            # clip dla numeryki
            sim_clamped = sim.clamp(-1.0, 1.0)
            sin_theta = torch.sqrt((1.0 - sim_clamped.pow(2)).clamp_min(0.0))
            positive = sim_clamped * cos_m - sin_theta * sin_m

            sim_m = sim_clamped.clone()
            sim_m[eye] = positive[eye]
            return sim_m

        raise ValueError(f"Unknown margin_type: {self.margin_type}")

    def _cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # opcjonalne label smoothing
        if self.ls > 0.0:
            num_classes = logits.size(-1)
            smooth = self.ls / (num_classes - 1)
            one_hot = torch.full_like(logits, smooth)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0 - self.ls)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(one_hot * log_probs).sum(dim=-1).mean()
            return loss
        else:
            return F.cross_entropy(logits, labels)

    def _restrict_to_hard_negatives(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Ogranicz wagę negatywów do top-k najtrudniejszych na podstawie logitów.
        Zwraca nową macierz [B, B] gdzie nie-wybrane negatywy są -inf (wytną się w softmax).
        """
        if self.topk <= 0:
            return logits

        B = logits.size(0)
        device = logits.device
        # maska pozytywów
        pos_mask = torch.eye(B, device=device, dtype=torch.bool)

        # ustaw pozytywy niską wartością na czas wyboru negatywów
        neg_scores = logits.masked_fill(pos_mask, float("-inf"))

        # top-k po wierszach (największe logity = najtrudniejsze negatywy)
        k = min(self.topk, B - 1)
        topk_vals, topk_idx = torch.topk(neg_scores, k=k, dim=1)

        # maska wybranych negatywów
        hard_mask = torch.zeros_like(logits, dtype=torch.bool)
        hard_mask.scatter_(1, topk_idx, True)
        # dodaj pozytyw
        hard_mask |= pos_mask

        # pozostałe ustaw na -inf
        logits_new = logits.masked_fill(~hard_mask, float("-inf"))
        return logits_new

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        mask_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Zwraca skalarną stratę.
        """
        # podstawowy kierunek: anchor -> positive
        sim = self._compute_logits(anchors, positives)  # [B, B]
        sim = self._apply_margin(sim)
        logits = self.s * sim

        B = anchors.size(0)
        labels = torch.arange(B, device=anchors.device)

        logits = self._restrict_to_hard_negatives(logits, labels)
        loss1 = self._cross_entropy(logits, labels)

        if not self.symmetric:
            return loss1

        # kierunek symetryczny: positive -> anchor
        sim2 = self._compute_logits(positives, anchors)
        sim2 = self._apply_margin(sim2)
        logits2 = self.s * sim2
        logits2 = self._restrict_to_hard_negatives(logits2, labels)
        loss2 = self._cross_entropy(logits2, labels)

        return 0.5 * (loss1 + loss2)
