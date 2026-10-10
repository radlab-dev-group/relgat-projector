"""
WIĄZKA relational operator: relations as elements of the similarity group.

A relation ``r`` is parametrised as

    T_r(x) = s_r * Rot_r(x),    Rot_r = exp(Omega_r) in O(D)

where ``Omega_r = sum_m theta_{r,m} * B_m`` is a linear combination of a
*learned basis of skew-symmetric generators* ``B_m`` (``m = 1..K``), and
``s_r = exp(log_s_r)`` is a learned log-scale.

Design notes (PLAN.md, section 1.2):

* **Non-commutativity**: the generators ``B_m`` span a Lie subalgebra of
  ``so(D)``; products ``exp(Omega_a) exp(Omega_b)`` do not commute in
  general, so ``r_1 o r_2 != r_2 o r_1`` -- a property the diagonal
  (DistMult) baseline fundamentally cannot have.
* **Exact, cheap inverse**: ``T_r^{-1} = exp(-Omega_r) / s_r``.
  ``exp`` of a skew matrix is computed once per (relation, step) via the
  Cayley transform, cached, and reused for apply/inverse.
* **Dial**: ``T_r(lambda) = exp(lambda * Omega_r) * s_r^lambda`` --
  ``lambda = 0`` is the identity, so relations interpolate continuously.
* **Context dependence**: optionally the coefficients are modulated by the
  base point, ``theta(r, x) = theta_r + eps * P(relu(W x))`` (eps learned,
  starts at 0), making the transport a true connection (PLAN.md 1.3).
* **Composition is closed**: a chain of relations is again an element of the
  same group; :meth:`compiled` folds a chain into a *single* transform.

Cost: one ``expm`` per relation (``D^3``, done once and cached) plus
``O(D^2)`` per vector application.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from relgat_projector.core.operator.base import RelationalOperator


def _random_skew_basis(
    dim: int,
    num_gen: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random orthonormalised basis of skew-symmetric matrices ``[K, D, D]``."""
    K = num_gen
    gens = torch.randn(K, dim, dim, device=device, dtype=dtype)
    gens = gens - gens.transpose(-1, -2)  # skew part
    # orthonormalise w.r.t. the Frobenius inner product
    for i in range(K):
        for j in range(i + 1, K):
            g = gens[j] - (gens[j] @ gens[i].t()).trace() / max(
                1e-12, (gens[i] @ gens[i].t()).trace()
            ) * gens[i]
            gens[j] = g
        n = torch.norm(gens[i].flatten())
        if n > 1e-8:
            gens[i] = gens[i] / n
    return gens


class GivensRelationOperator(nn.Module, RelationalOperator):
    """Relations as ``exp(theta^T B) * s`` in a learned skew-symmetric basis."""

    closed_composition = True
    exact_inverse = True
    is_isometry = True  # up to the scalar s_r (similarity)

    def __init__(
        self,
        dim: int,
        num_rel: int,
        num_generators: int = 16,
        learn_basis: bool = True,
        use_scale: bool = True,
        use_context: bool = False,
        context_hidden: int = 64,
        init_angle_scale: float = 0.1,
        expm_cache: bool = True,
    ):
        super().__init__()
        if not 0 < num_generators <= dim * (dim - 1) // 2:
            raise ValueError(f"bad num_generators={num_generators} for dim={dim}")
        self._dim = dim
        self._num_rel = num_rel
        self._num_gen = num_generators
        self.learn_basis = learn_basis
        self.use_scale = use_scale
        self.use_context = use_context
        self.expm_cache = expm_cache

        # Generator basis B_m, [K, D, D] (skew-symmetric).
        if learn_basis:
            self.basis = nn.Parameter(
                _random_skew_basis(dim, num_generators)
            )
        else:
            self.register_buffer("basis", _random_skew_basis(dim, num_generators))

        # Relation coordinates: theta_{r,m} and log s_r.
        self.theta = nn.Parameter(torch.randn(num_rel, num_generators) * init_angle_scale)
        if use_scale:
            self.log_scale = nn.Parameter(torch.zeros(num_rel))
        else:
            self.register_buffer("log_scale", torch.zeros(num_rel))

        # Context hyper-network: theta(r, x) = theta_r + eps * P(relu(W x)).
        if use_context:
            self.ctx_W = nn.Linear(dim, context_hidden, bias=False)
            self.ctx_P = nn.Linear(context_hidden, num_generators, bias=False)
            self.log_eps = nn.Parameter(torch.tensor(-6.0))
            nn.init.zeros_(self.ctx_P.weight)
        else:
            self.ctx_W = None
            self.ctx_P = None
            self.log_eps = None

        self._expm_cache: dict[tuple, torch.Tensor] = {}
        #: enough to rebuild this exact operator from a checkpoint
        self.ctor_kwargs: dict = {
            "type": "givens",
            "kwargs": {
                "dim": dim,
                "num_rel": num_rel,
                "num_generators": num_generators,
                "learn_basis": learn_basis,
                "use_scale": use_scale,
                "use_context": use_context,
                "context_hidden": context_hidden,
                "init_angle_scale": init_angle_scale,
                "expm_cache": expm_cache,
            },
        }

    # ------------------------------------------------------------- properties
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_rel(self) -> int:
        return self._num_rel

    @property
    def num_generators(self) -> int:
        return self._num_gen

    @property
    def eps(self) -> float:
        return float(self.log_eps.exp().detach()) if self.log_eps is not None else 0.0

    # ------------------------------------------------------------ context
    def enable_context(self, enabled: bool) -> None:
        """(Re)enable base-point-dependent transport (PLAN.md 1.3).

        Called by the stage manager when F4 starts: the hyper-network
        ``theta(r, x) = theta_r + eps * P(relu(W x))`` is created at this
        point (``eps`` starts near 0 so transport stays exactly invertible
        until context is earned).  Disabling removes the parameters so the
        checkpoint does not carry dead state.
        """
        enabled = bool(enabled)
        if enabled == self.use_context:
            return
        self.use_context = enabled
        if enabled:
            self.ctx_W = nn.Linear(self._dim, 64, bias=False)
            self.ctx_P = nn.Linear(64, self._num_gen, bias=False)
            self.log_eps = nn.Parameter(torch.tensor(-6.0))
            nn.init.zeros_(self.ctx_P.weight)
        else:
            del self.ctx_W
            del self.ctx_P
            del self.log_eps
            self.ctx_W = None
            self.ctx_P = None
            self.log_eps = None
        self.ctor_kwargs["kwargs"]["use_context"] = enabled
        self._expm_cache.clear()

    # ------------------------------------------------------------ matrix ops
    def _omega(self, theta: torch.Tensor) -> torch.Tensor:
        """Contract ``[*, K]`` coefficients with the basis -> ``[..., D, D]``."""
        return torch.einsum("...m,mij->...ij", theta, self.basis)

    def _rotation_matrix(self, omega: torch.Tensor) -> torch.Tensor:
        """Exponentiate a batch of skew matrices via the Cayley transform."""
        # Cayley: expm(Ω) = (I - Ω/2) (I + Ω/2)^{-1}  for skew Ω.
        I = torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype)
        A = I - omega / 2.0
        B = I + omega / 2.0
        return torch.linalg.solve(B, A)

    def _relation_rotations(
        self, rel_ids: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Full rotation matrices ``[B, D, D]`` for the given relation ids."""
        theta = self.theta[rel_ids]  # [B, K]
        if self.use_context and x is not None and self.ctx_W is not None:
            theta = theta + self.eps * self.ctx_P(torch.relu(self.ctx_W(x)))
        omega = self._omega(theta)  # [B, D, D]
        return self._rotation_matrix(omega)

    # ------------------------------------------------------------------ apply
    def apply(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(1).expand(B)
        if rel_ids.numel() == 1 and B > 1:
            rel_ids = rel_ids.expand(B)
        if lambdas is None:
            lambdas = x.new_ones(B)
        elif lambdas.dim() == 0:
            lambdas = lambdas.view(1).expand(B)

        # dial: scale the rotation by lambda (expm(lambda * Omega) approximated
        # by the Cayley transform of lambda * Omega), and the scale by s^lambda
        theta = self.theta[rel_ids]
        if self.use_context and self.ctx_W is not None:
            theta = theta + self.eps * self.ctx_P(torch.relu(self.ctx_W(x)))
        omega = self._omega(theta) * lambdas.unsqueeze(-1).unsqueeze(-1)  # [B,D,D]
        rot = self._rotation_matrix(omega)  # [B, D, D]

        out = torch.bmm(rot, x.unsqueeze(-1)).squeeze(-1)  # [B, D]
        if self.use_scale:
            s = (self.log_scale[rel_ids] * lambdas).exp().unsqueeze(-1)
            out = out * s
        return out

    def apply_to_matrix(
        self, x: torch.Tensor, rel_id: int | torch.Tensor
    ) -> torch.Tensor:
        B = x.shape[0]
        rel_ids = (
            torch.as_tensor(rel_id, device=x.device, dtype=torch.long)
            .view(1)
            .expand(B)
        )
        return self.apply(x, rel_ids)

    # ---------------------------------------------------------------- inverse
    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """Exact inverse: ``exp(-Omega_r) / s_r``."""
        B = x.shape[0]
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(1).expand(B)
        if rel_ids.numel() == 1 and B > 1:
            rel_ids = rel_ids.expand(B)

        theta = self.theta[rel_ids]
        omega = -self._omega(theta)  # [B, D, D]
        rot_inv = self._rotation_matrix(omega)
        out = torch.bmm(rot_inv, x.unsqueeze(-1)).squeeze(-1)
        if self.use_scale:
            out = out / self.log_scale[rel_ids].exp().unsqueeze(-1)
        return out

    # ------------------------------------------------------------------ chain
    def chain_transform(
        self, rel_ids: torch.Tensor, lambdas: torch.Tensor | None = None
    ) -> tuple:
        """Exact product of the chain rotations: ``[B, D, D]``."""
        B, k = rel_ids.shape
        D = self._dim
        acc = torch.eye(D, device=self.theta.device, dtype=self.theta.dtype).expand(B, D, D).clone()
        # x' = M_k ... M_2 M_1 x  =>  M = M_k ... M_1, so prepend M_k first
        for step in range(k):
            ids = rel_ids[:, step]
            if lambdas is not None:
                th = self.theta[ids] * lambdas[:, step].unsqueeze(-1)
            else:
                th = self.theta[ids]
            rot = self._rotation_matrix(self._omega(th))  # [B, D, D]
            acc = torch.bmm(rot, acc)
        s = torch.ones(B, device=acc.device)
        if self.use_scale:
            log_s = self.log_scale[rel_ids].sum(dim=1)
            if lambdas is not None:
                log_s = (self.log_scale[rel_ids] * lambdas).sum(dim=1)
            s = log_s.exp()
        return acc, s

    def compiled(
        self, rel_ids: torch.Tensor, lambdas: torch.Tensor | None = None
    ) -> "CompiledGivensOperator":
        """Fold a chain into one transform (exact, independent of chain length)."""
        from relgat_projector.core.operator.compiled import (
            CompiledGivensOperator,
        )

        acc, s = self.chain_transform(rel_ids, lambdas)
        # acc is [B, D, D] similarity matrix; s is [B]
        return CompiledGivensOperator(matrix=acc, scale=s)

    # ------------------------------------------------------------------- misc
    def get_relation_matrices(
        self, rel_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Materialise the full ``[R, D, D]`` similarity matrices (debug/eval)."""
        if rel_ids is None:
            rel_ids = torch.arange(self._num_rel, device=self.theta.device)
        R = rel_ids.shape[0]
        mats = []
        for r in range(R):
            m = self.apply(
                torch.eye(self._dim, device=self.theta.device), rel_ids[r : r + 1]
            ).reshape(self._dim, self._dim)
            mats.append(m)
        return torch.stack(mats, dim=0)

    def extra_repr(self) -> str:
        return (
            f"dim={self._dim}, num_rel={self._num_rel}, "
            f"K={self._num_gen}, learn_basis={self.learn_basis}, "
            f"scale={self.use_scale}, context={self.use_context}"
        )
