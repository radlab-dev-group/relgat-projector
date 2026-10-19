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
        max_steps=400,
    ),
    "F2_relations": StageConfig(
        name="F2_relations",
        active=["L_anchor", "L_edge", "L_con", "L_isom", "L_cycle"],
        freeze_operator=False,
        max_steps=1_200,
    ),
    "F3_composition": StageConfig(
        name="F3_composition",
        active=[
            "L_anchor", "L_edge", "L_con", "L_isom", "L_cycle",
            "L_path", "L_curv", "L_commut",
        ],
        freeze_operator=False,
        max_steps=2_400,
    ),
    "F4_shadows": StageConfig(
        name="F4_shadows",
        active=[
            "L_anchor", "L_edge", "L_con", "L_isom", "L_cycle",
            "L_path", "L_curv", "L_commut", "L_shadow", "L_pull",
            "L_applicability", "L_decoder",
        ],
        freeze_operator=False,
        enable_context=True,
        max_steps=4_000,
    ),
}

STAGE_ORDER = ["F1_nodes", "F2_relations", "F3_composition", "F4_shadows"]


# metric -> True when *lower* is better (PLAN.md section 4)
LOWER_IS_BETTER = {
    "eval/cce": True,
    "eval/isot": True,
    "eval/anchor": False,
    "eval/pce_k1": True,
    "eval/pce_k2": True,
    "eval/pce_k3": True,
    "eval/cos_edge": False,
}

DEFAULT_GATES: Dict[str, Dict[str, float]] = {
    # stage -> {metric: threshold}; ALL listed metrics must pass
    # (PLAN.md section 4).  F1 is gated on the *anchor* -- refinement must stay
    # near the base embedding -- and NOT on cos_edge: relations such as
    # antonymy deliberately connect dissimilar nodes, so a cosine-over-edges
    # gate is unattainable at a stage where the operator is still frozen.
    "F1_nodes": {"eval/anchor": 0.9},
    "F2_relations": {"eval/cce": 0.02, "eval/isot": 0.05},
    "F3_composition": {"eval/pce_k2": 0.10},
}


class StageManager:
    def __init__(self, order: Optional[List[str]] = None,
                 gates: Optional[Dict[str, Dict[str, float]]] = None):
        self.order = order or STAGE_ORDER
        # acceptance gates (metric -> threshold); default to "no gate"
        self.gates = gates if gates is not None else {
            k: dict(v) for k, v in DEFAULT_GATES.items()
        }
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
        """True when the stage may be left: every gate metric passes.

        A single easy metric must not open the gate -- ``eval/cce`` sits at ~0
        for an exactly invertible operator at *every* stage, so an
        any-metric rule let F2 advance after one evaluation.  Metrics that were
        never computed are skipped (they cannot block progress), and
        ``max_steps`` remains the safety valve.
        """
        cfg = self.stage()
        if self.steps_in_stage >= cfg.max_steps:
            return self.current < len(self.order) - 1
        gate = self.gates.get(self.order[self.current], {})
        if not gate:
            return False
        # every gate metric must be *present* and pass; a missing metric means
        # "not yet demonstrated", never "good enough".  (The safety valve above
        # already guarantees progress when a metric is structurally absent.)
        for metric, threshold in gate.items():
            value = metrics.get(metric)
            if value is None:
                return False
            lower_better = LOWER_IS_BETTER.get(metric, False)
            ok = value <= threshold if lower_better else value >= threshold
            if not ok:
                return False
        return self.current < len(self.order) - 1

    def is_active(self, loss_name: str) -> bool:
        return loss_name in self.stage().active

    @property
    def freeze_operator(self) -> bool:
        return self.stage().freeze_operator

    @property
    def enable_context(self) -> bool:
        return self.stage().enable_context
