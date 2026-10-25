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
    r, ev = conn.classify_relation(b1, b2) # (5b) relation of a node PAIR
    out = conn.to_retriever_space(v3)      # export to the retriever space
    q   = conn.production_query(base, rel) # (6) served query, lam_r included
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
        self.meta: dict = {}
        model.eval()

    # ------------------------------------------------------------- factories
    @classmethod
    def from_model(cls, model, rel2idx: dict, projector=None,
                   device: str = "cpu") -> "RelationFlow":
        return cls(model, rel2idx, projector=projector, device=device)

    @classmethod
    def load(cls, ckpt_dir: str, projector=None,
             device: str = "cpu") -> "RelationFlow":
        """Load from a directory written by :meth:`save` (see ``wiazka_checkpoint``)."""
        from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint

        # ``load_checkpoint`` restores the fitted calibration scalars (see
        # ``wiazka_checkpoint.apply_calibrated_scalars``); doing it here as well
        # would be two definitions of "what the run actually fitted".
        model, rel2idx, _spec, meta = load_checkpoint(ckpt_dir, device=device)
        flow = cls(model, rel2idx, projector=projector, device=device)
        flow.meta = meta
        return flow

    def save(self, ckpt_dir: str, config: Optional[dict] = None,
             commutation: Optional[torch.Tensor] = None) -> str:
        """Write a full checkpoint (model + declarative spec + ``rel2idx``)."""
        from relgat_projector.handlers.wiazka_checkpoint import save_checkpoint

        return save_checkpoint(ckpt_dir, self.model, self.rel2idx,
                               config=config, commutation=commutation)

    def commutation(self) -> Optional[torch.Tensor]:
        """Learned commutation matrix ``C_rs`` of the *training* loss, if attached."""
        loss = getattr(self, "curvature_loss", None)
        return loss.commutation() if loss is not None else None

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

    # --------------------------------------------------- (6) query rewriting
    @torch.no_grad()
    def production_query(self, base_vec: torch.Tensor,
                         rel_id: Union[int, str]) -> torch.Tensor:
        """The vector to send to the retriever for "apply ``rel_id`` to this query".

        ``(1 - lam_r) * x + lam_r * export(T_r(chart(x)))`` with the learned
        per-relation strength ``lam_r``: the exact function the trainer scores as
        ``eval/prod_mrr`` and QR-Recall measure.  Use this instead of composing
        :meth:`apply` and :meth:`to_retriever_space` by hand -- those two are the
        *internal* path (operator space, no learned strength), and the served
        query is what the held-out numbers were measured on.
        """
        if base_vec.dim() == 1:
            base_vec = base_vec.unsqueeze(0)
        base_vec = base_vec.to(self.device)
        rid = self._resolve(rel_id)
        rel = torch.full((base_vec.shape[0],), rid, dtype=torch.long,
                         device=self.device)
        return self.model.production_query(base_vec, rel)

    @torch.no_grad()
    def transport_strength(self, rel_id: Union[int, str]) -> float:
        """Learned strength ``lam_r`` in ``[0, 1]`` of rewriting by ``rel_id``.

        ``~0`` means the model decided this relation is better served by the
        plain embedding; a search pipeline can use it to skip rewriting.
        """
        rid = self._resolve(rel_id)
        dial = getattr(self.model, "transport_dial", None)
        if dial is None:          # checkpoint written before the dial existed
            return 1.0
        return float(torch.sigmoid(dial[rid]))

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

    # -------------------------------------------------- (5b) relation of a pair
    def transport_relation_scores(self, vec_a: torch.Tensor,
                                  vec_b: torch.Tensor) -> torch.Tensor:
        """``cos(T_r(a), b)`` for every relation -- transport-only classification.

        Both vectors are *refined* (operator-space) node vectors, e.g. from
        :meth:`refine`.  No head is involved: the relation is scored by how well
        transporting ``a`` by it lands on ``b``.
        """
        vec_a = vec_a.to(self.device)
        vec_b = vec_b.to(self.device)
        with torch.no_grad():
            return self.model.transport_relation_scores(vec_a, vec_b)

    def classify_relation(
        self,
        base_a: torch.Tensor,
        base_b: torch.Tensor,
        mode: str = "ensemble",
        score_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """Which relation holds between two nodes -- the inverse problem.

        ``base_a`` / ``base_b`` are *base* (retriever-space) vectors, mapped
        through the same chart the refinement trains on, so a caller holding
        raw embeddings can use this directly.  ``mode`` selects the evidence:
        ``"decoder"`` (the head), ``"transport"`` (transport fidelity),
        ``"ensemble"`` (both, default: the strongest measured), or
        ``"prior"`` (head corrected for the training-set relation frequency).
        Returns ``(relation names, evidence matrix [B, num_rel])``.

        ``score_stats`` is the ``(mean, std)`` of the transport scores over
        random pairs used to calibrate the transport evidence; without it the
        transport scores are used raw.
        """
        charted_a = self.refine(base_a)
        charted_b = self.refine(base_b)
        with torch.no_grad():
            ids, evidence = self.model.classify_relation(
                charted_a, charted_b, mode=mode, score_stats=score_stats)
        names = [self.idx2rel.get(int(i), str(int(i))) for i in ids.tolist()]
        return names, evidence
