"""
Cycle, isometry and anchor stability losses (PLAN.md 2.5, 2.6).

    L_cycle   = || T_r^{-1}(T_r(x)) - x ||
    L_isom    = relu( |cos(u, v) - cos(Tu, Tv)| - delta )
    L_iso-nrm = | ||T_r(x)|| / ||x|| - s_r |
    L_anchor  = relu( (1 - cos(x_v, base_v)) - margin )
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StabilityLoss(nn.Module):
    def __init__(
        self,
        cycle_weight: float = 1.0,
        isom_weight: float = 0.5,
        isom_delta: float = 0.02,
        anchor_weight: float = 0.5,
        anchor_margin: float = 0.05,
    ):
        super().__init__()
        self.cycle_weight = cycle_weight
        self.isom_weight = isom_weight
        self.isom_delta = isom_delta
        self.anchor_weight = anchor_weight
        self.anchor_margin = anchor_margin

    def forward(
        self,
        x: torch.Tensor,                # [B, D]
        t_inv_t_x: torch.Tensor | None,  # [B, D]
        u: torch.Tensor | None = None,   # [B, D]
        v: torch.Tensor | None = None,   # [B, D]
        tu: torch.Tensor | None = None,
        tv: torch.Tensor | None = None,
        scale_expected: torch.Tensor | None = None,  # s_r [B]
        refined: torch.Tensor | None = None,         # x_v
        base: torch.Tensor | None = None,            # base_v
    ) -> torch.Tensor:
        loss = x.new_zeros(())
        if t_inv_t_x is not None:
            loss = loss + self.cycle_weight * (t_inv_t_x - x).pow(2).sqrt().mean()
        if None not in (u, v, tu, tv):
            cos_in = F_cos(u, v)
            cos_out = F_cos(tu, tv)
            loss = loss + self.isom_weight * (
                (cos_in - cos_out).abs() - self.isom_delta
            ).clamp_min(0.0).mean()
        if scale_expected is not None:
            ratio = x.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)
            loss = loss + (ratio - scale_expected).abs().mean()
        if None not in (refined, base):
            cos = F_cos(refined, base)
            loss = loss + self.anchor_weight * ((1.0 - cos) - self.anchor_margin).clamp_min(
                0.0
            ).mean()
        return loss


def F_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)
