"""
Shadow-manifold losses (PLAN.md 2.6).

The product requirement is that a relation operator applied to a *text*
vector (which is not a graph node) still lands somewhere meaningful.  These
losses supervise the operator off the graph, on vectors taken from the
retriever index:

* ``cycle``   -- ``T_r^{-1}(T_r(z)) ~= z`` for shadow vectors ``z``;
* ``commut``  -- the learned commutation structure holds off the graph too;
* ``pull``    -- the operator commutes (approximately) with the projection
  between the operator space and the retriever space, otherwise exporting a
  transformed vector to the vector database destroys the transformation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShadowLoss(nn.Module):
    def __init__(
        self,
        cycle_weight: float = 1.0,
        commut_weight: float = 0.5,
        pull_weight: float = 0.5,
        eps: float = 1e-12,
        l2_scale: float = 1.0,
    ):
        super().__init__()
        self.cycle_weight = cycle_weight
        self.commut_weight = commut_weight
        self.pull_weight = pull_weight
        self.eps = eps
        #: Every term here is an l2 distance inside the operator chart, so it
        #: carries the scale of the embedder that produced the chart.
        #: Quadratic terms divide by ``l2_scale ** 2``, linear ones by
        #: ``l2_scale``; see :class:`~relgat_projector.core.loss.curvature.CurvatureLoss`
        #: for the measurement that motivated it.  1.0 = historical behaviour.
        self.l2_scale = float(l2_scale)

    def cycle(self, z: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        """``back`` must be ``T_r^{-1}(T_r(z))`` for shadow vectors ``z``."""
        diff = back - z
        return self.cycle_weight * torch.sqrt(diff.pow(2).mean() + self.eps) / self.l2_scale

    def commute(self, trst: torch.Tensor, tstr: torch.Tensor) -> torch.Tensor:
        diff = trst - tstr
        return self.commut_weight * torch.sqrt(diff.pow(2).mean() + self.eps) / self.l2_scale

    def commute_gated(
        self,
        trst: torch.Tensor,
        tstr: torch.Tensor,
        c_rs: torch.Tensor,
        margin: float | None = None,
    ) -> torch.Tensor:
        """Commutation consistency *as the learned structure prescribes*.

        Forcing ``T_r T_s = T_s T_r`` on every pair would collapse the operator
        onto a diagonal (DistMult-like) one -- exactly the degenerate model the
        method exists to avoid, and in direct conflict with the anti-commuting
        entries of ``C_rs``.  So off the graph, just as in
        :class:`CurvatureLoss`, pairs the model believes commute are pulled
        together and pairs it believes do not are only required to be
        *consistently* non-commuting.
        """
        q = self.l2_scale ** 2
        gap = (trst - tstr).pow(2).mean(dim=-1) / q
        gap_l2 = torch.sqrt(gap + self.eps)
        comm = (c_rs * gap).mean()
        # ``margin`` is an l2 distance between two transported vectors, so it
        # must scale with the norm of the vectors it compares: a fixed 0.25 is
        # dead in a chart whose vectors have norm ~1.3 (measured gaps for
        # genuinely non-commuting relation pairs are 0.02-0.06).  ``None``
        # derives it from the batch, matching CurvatureLoss.effective_margin.
        if margin is None:
            m = (2.0 * 0.02 * trst.detach().norm(dim=-1).clamp_min(1e-8)
                 / self.l2_scale)
        else:
            m = (torch.as_tensor(margin, device=trst.device)
                 / self.l2_scale).expand_as(gap_l2)
        anti = ((1.0 - c_rs) * (gap_l2 - m).clamp_min(0.0)).mean()
        return self.commut_weight * (comm + anti)

    def pull(
        self,
        projected_transformed: torch.Tensor,  # P(T_r(z))
        transformed_projected: torch.Tensor,  # T_r'(P(z)) in retriever space
    ) -> torch.Tensor:
        """Coherence of the operator with the projection to retriever space."""
        diff = projected_transformed - transformed_projected
        return self.pull_weight * torch.sqrt(diff.pow(2).mean() + self.eps) / self.l2_scale

    def forward(
        self,
        cycle_term: torch.Tensor | None = None,
        commute_term: torch.Tensor | None = None,
        pull_term: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = (cycle_term if cycle_term is not None else 0.0)
        if commute_term is not None:
            loss = loss + commute_term
        if pull_term is not None:
            loss = loss + pull_term
        return loss


class DecoderLoss(nn.Module):
    """Supervision for the relation decoder ``g(x, y) -> r`` (PLAN.md 1.5).

    ``class_weights`` exists because the relation distribution of a WordNet is
    nowhere near uniform (``synonimia`` carries 49.7 % of the Słowosieć edges,
    ~30 relations carry fewer than 100): trained with a plain cross-entropy the
    head learns to answer the frequent relations and its *macro* accuracy -- the
    only honest average here -- collapses.  The trainer passes inverse-frequency
    weights tempered by ``decoder_balance``.
    """

    def __init__(self, confidence_weight: float = 0.2, label_smoothing: float = 0.0,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.confidence_weight = confidence_weight
        self.label_smoothing = float(label_smoothing)
        self.register_buffer(
            "class_weights",
            torch.ones(0) if class_weights is None else class_weights.detach(),
        )

    def set_class_weights(self, class_weights: torch.Tensor | None) -> None:
        """Weights keep the device they are given on.

        The loss modules are not submodules of the model, so ``model.to(cuda)``
        never touches them: pinning the buffer to its own (CPU) device here made
        every CUDA run die in ``cross_entropy`` with "weight is on cpu".
        """
        if class_weights is None:
            self.class_weights = torch.ones(0, device=self.class_weights.device)
            return
        self.class_weights = class_weights.detach()

    def forward(
        self,
        logits: torch.Tensor,      # [B, num_rel]
        confidence: torch.Tensor,  # [B]
        rel_ids: torch.Tensor,     # [B] ground-truth relations
    ) -> torch.Tensor:
        weight = self.class_weights if self.class_weights.numel() else None
        ce = nn.functional.cross_entropy(
            logits, rel_ids.long(), weight=weight,
            label_smoothing=self.label_smoothing,
        )
        conf = nn.functional.binary_cross_entropy_with_logits(
            confidence - confidence.mean().detach(), torch.ones_like(confidence)
        ) * self.confidence_weight
        return ce + conf
