"""
Inference API for WIĄZKA (PLAN.md 3.4).

Wraps a trained :class:`WiazkaModel` (or a bare
:class:`RelationalOperator`) behind the small user-facing surface:

    conn = RelationFlow.load(ckpt)
    v   = conn.refine(base_vec)            # (1) refined representation
    v2  = conn.apply(v, rel_id)            # (3) single relation step
    v3  = conn.apply_chain(v, [r1, r2])    # (3) relation chain
    op  = conn.compile([r1, r2])           # (3) fold chain into one operator
    v4  = conn.apply(v, rel_id, lam=0.4)   # (3) dial: "40% of the relation"
    p   = conn.applicability(v, rel_id)    # (4) is this even defined?
    r, c = conn.decode(v, v3)              # (5) "which relation is this?"
    out = conn.to_retriever_space(v3)      # export to the retriever space
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Union

import torch


class RelationFlow:
    def __init__(self, model, rel2idx: dict, projector=None, device: str = "cpu"):
        self.model = model
        self.rel2idx = dict(rel2idx)
        self.idx2rel = {v: k for k, v in rel2idx.items()}
        self.projector = projector
        self.device = device
        model.eval()

    # ------------------------------------------------------------- factories
    @classmethod
    def from_model(cls, model, rel2idx: dict, projector=None,
                   device: str = "cpu") -> "RelationFlow":
        return cls(model, rel2idx, projector=projector, device=device)

    @classmethod
    def load(cls, ckpt_dir: str, projector=None,
             device: str = "cpu") -> "RelationFlow":
        """Load from a directory written by :meth:`save`."""
        state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"),
                           map_location=device)
        cfg = torch.load(os.path.join(ckpt_dir, "config.pt"), map_location=device)
        model = cfg["model_factory"]()
        model.load_state_dict(state["model"])
        model = model.to(device)
        return cls(model, state["rel2idx"], projector=projector, device=device)

    def save(self, ckpt_dir: str) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))

    # ----------------------------------------------------------------- names
    def _resolve(self, rel_id: Union[int, str]) -> int:
        if isinstance(rel_id, str):
            if rel_id not in self.rel2idx:
                raise KeyError(f"unknown relation {rel_id!r}")
            return self.rel2idx[rel_id]
        return int(rel_id)

    # --------------------------------------------------------------- (1) refine
    def refine(self, base_vec: torch.Tensor) -> torch.Tensor:
        """Refine a base vector (single node) to operator space."""
        if base_vec.dim() == 1:
            base_vec = base_vec.unsqueeze(0)
        base_vec = base_vec.to(self.device)
        return self.model.base_to_op(base_vec)

    # --------------------------------------------------------------- (3) apply
    def apply(self, vec: torch.Tensor, rel_id: Union[int, str],
              lam: Optional[float] = None) -> torch.Tensor:
        vec = vec.to(self.device)
        rid = self._resolve(rel_id)
        lambdas = (
            torch.full((vec.shape[0],), float(lam), device=self.device)
            if lam is not None else None
        )
        return self.model.apply(vec, torch.full((vec.shape[0],), rid,
                                                dtype=torch.long, device=self.device),
                                lambdas)

    def apply_chain(self, vec: torch.Tensor,
                    rel_ids: Sequence[Union[int, str]],
                    lams: Optional[Sequence[float]] = None) -> torch.Tensor:
        vec = vec.to(self.device)
        ids = torch.tensor([self._resolve(r) for r in rel_ids], dtype=torch.long,
                           device=self.device).unsqueeze(0).expand(vec.shape[0], -1)
        lambdas = None
        if lams is not None:
            lambdas = torch.tensor(lams, device=self.device).unsqueeze(0).expand(
                vec.shape[0], -1)
        return self.model.apply_chain(vec, ids, lambdas)

    def compile(self, rel_ids: Sequence[Union[int, str]],
                lams: Optional[Sequence[float]] = None):
        ids = torch.tensor([self._resolve(r) for r in rel_ids], dtype=torch.long,
                           device=self.device).unsqueeze(0)
        lambdas = (
            torch.tensor(lams, device=self.device).unsqueeze(0)
            if lams is not None else None
        )
        return self.model.compiled(ids, lambdas)

    def to_retriever_space(self, vec: torch.Tensor) -> torch.Tensor:
        if self.projector is None:
            return vec
        vec = vec.to(self.device)
        return self.projector(vec)

    # --------------------------------------------------------------- (4) applicability
    def applicability(self, vec: torch.Tensor,
                      rel_id: Union[int, str]) -> torch.Tensor:
        vec = vec.to(self.device)
        rid = self._resolve(rel_id)
        logits = self.model.applicability_score(
            vec, torch.full((vec.shape[0],), rid, dtype=torch.long, device=self.device)
        )
        return torch.sigmoid(logits)

    # --------------------------------------------------------------- (5) decode
    def decode(self, vec_a: torch.Tensor,
               vec_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vec_a = vec_a.to(self.device)
        vec_b = vec_b.to(self.device)
        ids, conf = self.model.decode(vec_a, vec_b)
        names = [self.idx2rel.get(int(i), str(int(i))) for i in ids.tolist()]
        return names, conf
