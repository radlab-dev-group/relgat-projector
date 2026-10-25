"""
Common interface for relational operators (WIĄZKA / RGT).

A *relational operator* maps a batch of node vectors by a relation:

    T_r : R^D -> R^D

The interface is intentionally small so that the training pipeline, the
inference API and the metrics all work with any concrete operator:

* :class:`~relgat_projector.core.operator.givens.GivensRelationOperator`
  (the WIĄZKA operator: product of Givens rotations + log-scale),
* :class:`~relgat_projector.core.operator.builtin_distmult.DistMultOperator`
  (baseline: element-wise product by the relation vector),
* :class:`~relgat_projector.core.operator.builtin_transe.TransEOperator`
  (baseline: translation by the relation vector).

All operators expose:

* :meth:`apply`       -- transform a batch of vectors by relations,
* :meth:`apply_chain` -- transform by a sequence of relations (composition),
* :meth:`inverse`     -- exact (or approximate) inverse of the transform,
* :meth:`dial`        -- continuous strength in [0, 1] (identity at 0),
* :meth:`apply_to_matrix` -- batched transform of a full [N, D] matrix by a
  single relation id (used by the GAT transport-aware aggregation).
"""

from __future__ import annotations

import abc

import torch


class RelationalOperator(abc.ABC):
    """Abstract relational operator acting in a D-dimensional space."""

    #: whether composition is closed (chain of operators is again one operator)
    closed_composition: bool = False
    #: whether the inverse is exact (T^{-1} T = I) by construction
    exact_inverse: bool = False
    #: whether the operator is an isometry (norm/cosine preserving) by construction
    is_isometry: bool = False

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Ambient dimension D."""

    @property
    @abc.abstractmethod
    def num_rel(self) -> int:
        """Number of distinct relation types."""

    @abc.abstractmethod
    def apply(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the relation operator(s) to a batch of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Vectors of shape ``[B, D]``.
        rel_ids : torch.Tensor
            Relation indices of shape ``[B]``.
        lambdas : torch.Tensor, optional
            Per-vector strength in ``[0, 1]`` (shape ``[B]``).  ``lambda=0``
            must yield the identity (no transformation).  Defaults to 1.

        Returns
        -------
        torch.Tensor
            Transformed vectors of shape ``[B, D]``.
        """

    def apply_chain(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a sequence of relations ``[B, k]`` (row-major order).

        The *i*-th row of ``rel_ids`` is applied left-to-right:
        ``T_{r_{i,k-1}} o ... o T_{r_{i,1}}(x_i)``.  The default
        implementation loops; operators with closed composition should
        override with a compiled single operator.
        """
        B, D = x.shape
        k = rel_ids.shape[1]
        out = x
        for step in range(k):
            out = self.apply(out, rel_ids[:, step],
                             lambdas[:, step] if lambdas is not None else None)
        return out

    @abc.abstractmethod
    def inverse(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        base_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the inverse transform of ``rel_ids`` to ``x``.

        For operators with :attr:`exact_inverse` this satisfies
        ``T_r^{-1}(T_r(x)) == x`` up to floating-point round-off.

        ``base_point`` is the point the forward transform was applied *at*.  A
        connection with base-point-dependent coefficients maps ``x`` by
        ``A(x) = exp(Omega(theta + eps * P(x)))``, whose inverse is
        ``A(x)^{-1}`` -- the coefficients of the origin of the transport, not of
        the destination.  Callers that go forward and back from the same point
        have to pass it, otherwise the round trip is not a round trip and the
        cycle loss spends its whole gradient pushing the connection back to
        zero.
        """

    def dial(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lam: float | torch.Tensor,
    ) -> torch.Tensor:
        """Continuous strength of a relation (identity at 0, full at 1)."""
        B = x.shape[0]
        if isinstance(lam, float):
            lambdas = x.new_full((B,), lam)
        else:
            lambdas = lam
        return self.apply(x, rel_ids, lambdas)

    def apply_to_matrix(
        self, x: torch.Tensor, rel_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Apply one relation to an entire ``[N, D]`` matrix.

        Default implementation chunks through :meth:`apply`.  Operators should
        override for speed (used by the transport-aware GAT aggregation).
        """
        rel_ids = x.new_full((x.shape[0],), rel_id, dtype=torch.long)
        return self.apply(x, rel_ids)

    def to(self, *args, **kwargs) -> "RelationalOperator":
        return super().to(*args, **kwargs)

    def eval(self) -> "RelationalOperator":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "RelationalOperator":
        super().train(mode)
        return self
