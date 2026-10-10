"""
Path compilation: turn a relation chain ``r_1..r_k`` into one operator.

The exact product of ``k`` similarity transforms is a single similarity
transform ``M = T_k ... T_1`` (a ``D x D`` matrix plus a scalar).  A compiled
operator therefore applies a whole chain with ONE matrix-vector product,
independent of the chain length (PLAN.md, section 3.3).
"""

from __future__ import annotations

import torch


class CompiledGivensOperator:
    """A single similarity transform produced by composing relation steps."""

    def __init__(self, matrix: torch.Tensor, scale: torch.Tensor):
        self.matrix = matrix  # [B, D, D]
        self.scale = scale    # [B]

    @property
    def num_vectors(self) -> int:
        return int(self.matrix.shape[0])

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the compiled transform to ``[B, D]`` (one bmm)."""
        out = torch.bmm(self.matrix, x.unsqueeze(-1)).squeeze(-1)
        return out * self.scale.unsqueeze(-1)

    @staticmethod
    def from_product(rotations: list, scales: list) -> "CompiledGivensOperator":
        """Build the compiled operator from a list of per-step rotations."""
        raise NotImplementedError

    def as_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.matrix, self.scale

    def contract(self, x: torch.Tensor, n_target: int | None = None):
        """Project the compiled matrix back onto a low-rank generator basis.

        Returns ``(coefficients, residual)`` where ``coefficients`` are the
        best-fit coordinates in the operator's generator basis and ``residual``
        is the mean L2 error of the reconstructed action on ``x``.  Used to
        measure how well a long chain can be *shortened* (PLAN.md H3).
        """
        raise NotImplementedError(
            "contract requires the generator basis; use GivensRelationOperator."
        )
