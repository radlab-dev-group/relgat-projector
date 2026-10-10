"""
Curriculum stage manager (PLAN.md section 4, F0-F4).

Each stage activates a subset of loss terms and exposes a *gate*: the stage
may only be left when its acceptance criterion (measured on the eval set) is
met, otherwise the trainer keeps the current stage (with a max-stage-steps
safety valve).

Stage map (from PLAN.md):

    F1  nodes      : L_anchor, L_edge (operator frozen to identity)
    F2  relations  : + L_con, L_isom, L_cycle
    F3  composition: + L_path, L_curv, L_commut
    F4  shadows    : + L_shadow_*, L_pull, applicability, decoder (eps on)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class StageConfig:
    name: str
    active: List[str] = field(default_factory=list)
    freeze_operator: bool = False
    enable_context: bool = False
    max_steps: int = 10_000


STAGES: Dict[str, StageConfig] = {
    "F1_nodes": StageConfig(
        name="F1_nodes",
        active=["L_anchor", "L_edge"],
        freeze_operator=True,
        max_steps=5000,
    ),
    "F2_relations": StageConfig(
        name="F2_relations",
        active=["L_anchor", "L_edge", "L_con", "L_isom", "L_cycle"],
        freeze_operator=False,
        max_steps=20_000,
    ),
    "F3_composition": StageConfig(
        name="F3_composition",
        active=[
            "L_anchor", "L_edge", "L_con", "L_isom", "L_cycle",
            "L_path", "L_curv",
        ],
        freeze_operator=False,
        max_steps=30_000,
    ),
    "F4_shadows": StageConfig(
        name="F4_shadows",
        active=[
            "L_anchor", "L_edge", "L_con", "L_isom", "L_cycle",
            "L_path", "L_curv", "L_shadow", "L_pull",
            "L_applicability", "L_decoder",
        ],
        freeze_operator=False,
        enable_context=True,
        max_steps=40_000,
    ),
}

STAGE_ORDER = ["F1_nodes", "F2_relations", "F3_composition", "F4_shadows"]


class StageManager:
    def __init__(self, order: Optional[List[str]] = None,
                 gates: Optional[Dict[str, float]] = None):
        self.order = order or STAGE_ORDER
        # acceptance gates (metric -> threshold); default to "no gate"
        self.gates = gates or {}
        self.current = 0
        self.steps_in_stage = 0

    def stage(self) -> StageConfig:
        return STAGES[self.order[self.current]]

    def advance(self) -> None:
        if self.current < len(self.order) - 1:
            self.current += 1
            self.steps_in_stage = 0

    def step(self) -> None:
        self.steps_in_stage += 1

    def should_stop(self, metrics: Dict[str, float]) -> bool:
        """Return True when the current stage's gate is passed (advance)."""
        cfg = self.stage()
        if self.steps_in_stage >= cfg.max_steps:
            return self.current < len(self.order) - 1
        for metric, threshold in self.gates.get(self.order[self.current], {}).items():
            value = metrics.get(metric)
            if value is None:
                continue
            if value >= threshold:
                return self.current < len(self.order) - 1
        return False

    def is_active(self, loss_name: str) -> bool:
        return loss_name in self.stage().active

    @property
    def freeze_operator(self) -> bool:
        return self.stage().freeze_operator

    @property
    def enable_context(self) -> bool:
        return self.stage().enable_context
