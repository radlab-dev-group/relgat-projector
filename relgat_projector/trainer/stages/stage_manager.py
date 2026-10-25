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
    #: a stage may not be left before it has actually optimised for a while:
    #: gates that are already satisfied at initialisation used to advance the
    #: curriculum after a single evaluation
    min_steps: int = 0


STAGES: Dict[str, StageConfig] = {
    "F1_nodes": StageConfig(
        name="F1_nodes",
        active=["L_anchor", "L_edge"],
        freeze_operator=True,
        max_steps=400,
        min_steps=200,
    ),
    "F2_relations": StageConfig(
        name="F2_relations",
        # L_decoder is here, not only in F4: naming a relation is defined as
        # soon as the nodes exist, and the head had ~15 of 35 epochs to learn
        # it while every other term had the whole run.
        active=["L_anchor", "L_edge", "L_con", "L_isom", "L_cycle", "L_decoder"],
        freeze_operator=False,
        max_steps=800,
        min_steps=400,
    ),
    "F3_composition": StageConfig(
        name="F3_composition",
        active=[
            "L_anchor", "L_edge", "L_con", "L_isom", "L_cycle",
            "L_path", "L_curv", "L_commut", "L_decoder",
        ],
        freeze_operator=False,
        # F3 does not *remove* anything when it ends -- F4 only adds -- so a
        # long stay here only delays the stage that trains the export
        # (``L_pull``).  With the 158 steps/epoch of the Slowosiec, 2400 steps
        # of safety valve pushed F4 past epoch 21 of a 35-epoch run and the
        # product-side losses were trained on the decaying tail of the LR
        # schedule.
        max_steps=1_600,
        min_steps=800,
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
        min_steps=1_200,
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
    # composition must not blow up relative to a single transport step
    "eval/pce_ratio_k2": True,
    "eval/cos_edge": False,
    # ranking lift over the raw-base baseline: the value the transport adds
    "eval/mrr_lift": False,
    # the same comparison done inside one index (transport vs identity), and
    # the two end-to-end numbers: both are higher-is-better
    "eval/mrr": False,
    "eval/mrr_macro": False,
    "eval/mrr_vs_raw": False,
    "eval/mrr_prod_lift": False,
}

DEFAULT_GATES: Dict[str, Dict[str, float]] = {
    # stage -> {metric: threshold}; ALL listed metrics must pass
    # (PLAN.md section 4).  F1 is gated on the *anchor* -- refinement must stay
    # near the base embedding -- and NOT on cos_edge: relations such as
    # antonymy deliberately connect dissimilar nodes, so a cosine-over-edges
    # gate is unattainable at a stage where the operator is still frozen.
    #
    # F2 is no longer gated on cce / isot.  The operator is exactly invertible
    # and exactly conformal *by construction*, so both sit near zero at every
    # stage and the gate opened on the first evaluation with the relations
    # having learned nothing at all.  F2 is gated on what it is actually for:
    # the transport must rank the true target better than raw similarity does.
    "F1_nodes": {"eval/anchor": 0.9},
    # ``eval/mrr_lift`` is T_r against the *identity transport on the same
    # refined index*: it isolates the relation from the refinement and from
    # the choice of index, so F2 opens exactly when the transport itself adds
    # ranking value (the pre-2026-10 gate compared the refined index with the
    # base index, which a refinement that moves nodes can never satisfy).
    "F2_relations": {"eval/mrr_lift": 0.0},
    # F3 is accepted on what composition is *for*: the two-hop transport must
    # retrieve the two-hop target better than doing nothing does.  The previous
    # gate (``pce_ratio_k2 <= 1.6``) demanded composition error below one
    # single-step error; two independent angular errors add up to ratio ~2.0 by
    # construction, so the gate was unreachable (best measured 2.07), every run
    # left F3 through the max_steps valve, and the export/decoder stage lost a
    # third of its steps.  ``pce_ratio_k2`` stays reported.
    "F3_composition": {"eval/path_lift": 0.0},
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
        return self._exit_reason(metrics) is not None

    def gate_passed(self, metrics: Dict[str, float]) -> bool:
        """True only when the acceptance criterion itself was satisfied.

        Kept apart from :meth:`should_stop`, which also fires on the
        ``max_steps`` safety valve: a log line saying "gate passed" for a stage
        whose gate was never met is how an unreachable threshold stays hidden
        for an entire training run (F3 asks for ``pce_ratio_k2 <= 1.6``; the
        lowest value ever measured on the Slowosiec is ~2.0).
        """
        return self._exit_reason(metrics) == "gate"

    def _exit_reason(self, metrics: Dict[str, float]) -> Optional[str]:
        """``"gate"``, ``"max_steps"``, or ``None`` for the current stage."""
        cfg = self.stage()
        last = self.current >= len(self.order) - 1
        if self.steps_in_stage >= cfg.max_steps:
            return None if last else "max_steps"
        # a gate that is already true at initialisation must not let the
        # curriculum run itself to completion on the first evaluation
        if self.steps_in_stage < cfg.min_steps:
            return None
        gate = self.gates.get(self.order[self.current], {})
        if not gate:
            return None
        # every gate metric must be *present* and pass; a missing metric means
        # "not yet demonstrated", never "good enough".  (The safety valve above
        # already guarantees progress when a metric is structurally absent.)
        for metric, threshold in gate.items():
            value = metrics.get(metric)
            if value is None:
                return None
            lower_better = LOWER_IS_BETTER.get(metric, False)
            ok = value <= threshold if lower_better else value >= threshold
            if not ok:
                return None
        return None if last else "gate"

    def is_active(self, loss_name: str) -> bool:
        return loss_name in self.stage().active

    @property
    def freeze_operator(self) -> bool:
        return self.stage().freeze_operator

    @property
    def enable_context(self) -> bool:
        return self.stage().enable_context
