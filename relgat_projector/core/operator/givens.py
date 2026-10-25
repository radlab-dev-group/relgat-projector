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


class _GroupedRotationApply(torch.autograd.Function):
    """``y_i = R_{g(i)} x_i`` for rows pre-sorted by group, with a fused backward.

    Written naively, the per-group loop is a chain of ~``num_rel`` slice
    assignments, and autograd clones the whole ``[B, D]`` buffer once *per
    assignment* to build the backward graph: on the 324k-edge Słowosieć split
    that turned a 8 ms forward into a 147 ms backward (measured).  Doing the
    same arithmetic with no-grad tensors inside a custom ``Function`` keeps the
    result identical and makes the backward cost the same as the forward.
    """

    @staticmethod
    def forward(ctx, x, rot, order, counts):
        with torch.no_grad():
            xs = x.index_select(0, order)
            out = xs.new_empty(xs.shape)
            begin = 0
            for u, n in enumerate(counts):
                if n == 0:
                    continue
                rows = slice(begin, begin + n)
                out[rows] = xs[rows] @ rot[u].transpose(0, 1)
                begin += n
            y = out.new_empty(out.shape)
            y[order] = out
        ctx.save_for_backward(x, rot, order)
        ctx.counts = counts
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, rot, order = ctx.saved_tensors
        counts = ctx.counts
        need_x = ctx.needs_input_grad[0]
        need_rot = ctx.needs_input_grad[1]
        with torch.no_grad():
            gys = grad_y.index_select(0, order)
            grad_x = None
            grad_rot = torch.zeros_like(rot) if need_rot else None
            xs = x.index_select(0, order) if need_rot else None
            if need_x:
                grad_x = torch.empty_like(gys)
            begin = 0
            for u, n in enumerate(counts):
                if n == 0:
                    continue
                rows = slice(begin, begin + n)
                if need_x:
                    grad_x[rows] = gys[rows] @ rot[u]
                if need_rot:
                    grad_rot[u] = gys[rows].transpose(0, 1) @ xs[rows]
                begin += n
            if need_x:
                restored = torch.empty_like(grad_x)
                restored[order] = grad_x
                grad_x = restored
        return grad_x, grad_rot, None, None


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

        # Generator basis B_m, [K, D, D], skew-symmetric *by construction*:
        # the stored tensor is an unconstrained raw matrix and B = (R - R^T)/2
        # is what every consumer sees.  Keeping the raw tensor learnable without
        # this projection lets gradients drift the basis out of so(D), which
        # silently destroys the three headline properties (orthogonality, exact
        # inverse, cosine preservation) -- measured ISOT was ~4e-2 after a few
        # hundred steps, where it must be ~1e-7.
        raw = _random_skew_basis(dim, num_generators)
        if learn_basis:
            self.basis_raw = nn.Parameter(raw)
        else:
            self.register_buffer("basis_raw", raw)

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
            # eps = exp(log_eps), started at exp(-2) ~ 0.135 and NOT at the
            # exp(-6) ~ 0.0025 the plan suggested.  Adam moves the log of a
            # scalar by at most ~lr per step, so a whole run (5k steps at
            # lr 3e-4) explores a window of about e^1.5 = 4.5x around its
            # initialisation: from -6 the strength of the connection could never
            # exceed 0.011, and a finished 35-epoch run confirmed it -- log_eps
            # ended at -5.43, i.e. the modulation was ~1 % of theta and the F4
            # connection was numerically a no-op.  -2 is small enough to leave
            # the transport undisturbed at stage entry (ctx_P starts at std 1e-3,
            # so eps * P(...) ~ 1e-4) and large enough to be learnable both ways.
            self.log_eps = nn.Parameter(torch.tensor(-2.0))
            # NOT exactly zero.  theta(r, x) = theta_r + eps * P(...) makes the
            # gradient of log_eps proportional to P(...), so a zero-initialised
            # P gives log_eps no gradient at all while P itself only moves at
            # the rate eps * lr -- the strength of the connection then never
            # leaves its initialisation (measured with the old zero init after a
            # full run: log_eps == -6.0 exactly, ctx_P.std() == 0.0).  A small
            # random P keeps the transport numerically the identity at the start
            # and makes both factors trainable.
            nn.init.normal_(self.ctx_P.weight, std=1e-3)
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
    def basis(self) -> torch.Tensor:
        """The skew-symmetric generators ``[K, D, D]`` (exact by projection)."""
        r = self.basis_raw
        return 0.5 * (r - r.transpose(-1, -2))

    @property
    def eps(self) -> float:
        """Strength of the connection, detached (diagnostics / logging)."""
        return float(self.log_eps.exp().detach()) if self.log_eps is not None else 0.0

    def eps_tensor(self) -> torch.Tensor | None:
        """Differentiable strength of the connection.

        ``eps`` above is a Python float by design (it is compared, logged and
        put into ``ctor_kwargs``), but it must not be the value the transport
        uses: ``float(log_eps.exp().detach())`` has no gradient, so ``log_eps``
        -- the single parameter that decides *how much* the transport depends on
        its base point -- could never be learned and stayed at its initialisation.
        """
        return None if self.log_eps is None else self.log_eps.exp()

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
            # built on the operator's own device: the stage manager turns
            # context on *during* training, and a freshly constructed nn.Linear
            # lands on the CPU, which then dies on the first CUDA matmul
            device = self.theta.device
            self.ctx_W = nn.Linear(self._dim, 64, bias=False, device=device)
            self.ctx_P = nn.Linear(64, self._num_gen, bias=False, device=device)
            # see __init__: exp(-2) is the smallest strength that Adam can still
            # move measurably inside one training run
            self.log_eps = nn.Parameter(torch.tensor(-2.0, device=device))
            # see __init__: a zero P freezes the strength of the connection
            nn.init.normal_(self.ctx_P.weight, std=1e-3)
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

    #: ``torch.linalg.solve`` returns invalid LU pivots for *batched* inputs on
    #: multi-threaded Intel oneMKL builds ("Pivots given to lu_solve must all be
    #: greater or equal to 1", oneMKL: "Parameter 6 was incorrect on entry to
    #: SLASWP"), which kills every CPU inference of a 256-d model with more than
    #: one relation in the batch while the same code is fine on GPU and on a
    #: single-threaded CPU.  Measured with torch 2.9.0 + MKL at 2 threads.
    #: The first failure switches the process to ``lstsq`` -- a different LAPACK
    #: driver that is correct there and agrees with ``solve`` to 1e-6 -- so the
    #: fast path is kept wherever it works.
    _LU_SOLVE_OK = True

    def _rotation_matrix(self, omega: torch.Tensor) -> torch.Tensor:
        """Exponentiate a batch of skew matrices via the Cayley transform."""
        # Cayley: expm(Ω) = (I - Ω/2) (I + Ω/2)^{-1}  for skew Ω.
        I = torch.eye(omega.shape[-1], device=omega.device, dtype=omega.dtype)
        A = I - omega / 2.0
        B = I + omega / 2.0
        cls = type(self)
        if cls._LU_SOLVE_OK:
            try:
                return torch.linalg.solve(B, A)
            except RuntimeError as exc:
                if "lu_solve" not in str(exc) and "Pivots" not in str(exc):
                    raise
                cls._LU_SOLVE_OK = False
                print("[operator] batched linalg.solve is broken in this "
                      "environment (multi-threaded MKL); using lstsq")
        try:
            # different LAPACK driver, correct where lu_solve is broken
            return torch.linalg.lstsq(B, A).solution
        except Exception:
            # last resort: one matrix at a time, which no BLAS threading bug
            # of this kind affects
            return torch.stack([torch.linalg.solve(b, a) for b, a in zip(B, A)])

    def _relation_rotations(
        self, rel_ids: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Full rotation matrices ``[B, D, D]`` for the given relation ids."""
        theta = self.theta[rel_ids]  # [B, K]
        if self.use_context and x is not None and self.ctx_W is not None:
            theta = theta + self.eps_tensor() * self.ctx_P(
                torch.relu(self.ctx_W(x))
            )
        omega = self._omega(theta)  # [B, D, D]
        return self._rotation_matrix(omega)

    def relation_matrices(self) -> torch.Tensor:
        """The ``[num_rel, D, D]`` rotation of every relation, batched.

        ``theta`` holds one coefficient vector per relation, so all of them
        exponentiate in a single batched solve.  That is what makes scoring
        *every* relation (the inverse problem: which relation maps ``x`` to
        ``y``?) cost ``O(R D^3 + B R D^2)`` instead of ``R`` separate operator
        calls.  This is the context-free part of the transport: with the
        connection on, the coefficients depend on the base point and the caller
        must go through :meth:`apply` per row.
        """
        return self._rotation_matrix(self._omega(self.theta))

    # ------------------------------------------------------------------ apply
    # ------------------------------------------------------- grouped action
    #: rows per dense ``[B, D, D]`` block on the ungrouped fallback path.
    #: ``num_generators`` coefficients describe ONE matrix per relation, so a
    #: dense per-row matrix is pure waste: with ``D = 256`` and the ~324k edges
    #: of a real graph it asks for 79 GiB and dies in the CUDA allocator.
    #: Keeping the bound small makes memory ``O(num_rel * D^2 + B * D)``.
    MAX_DENSE_ROWS = 4096

    def _apply_dense_rows(
        self,
        x: torch.Tensor,
        theta_rows: torch.Tensor,
        log_scale_rows: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-row transform through explicit matrices, in bounded chunks."""
        B = x.shape[0]
        if B <= self.MAX_DENSE_ROWS:
            rot = self._rotation_matrix(self._omega(theta_rows))
            out = torch.bmm(rot, x.unsqueeze(-1)).squeeze(-1)
            if log_scale_rows is not None:
                out = out * log_scale_rows.exp().unsqueeze(-1)
            return out
        parts = []
        for i in range(0, B, self.MAX_DENSE_ROWS):
            j = i + self.MAX_DENSE_ROWS
            parts.append(
                self._apply_dense_rows(
                    x[i:j],
                    theta_rows[i:j],
                    None if log_scale_rows is None else log_scale_rows[i:j],
                )
            )
        return torch.cat(parts, dim=0)

    def _apply_grouped(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        sign: float,
        lam: float,
    ) -> torch.Tensor:
        """``x_i -> exp(sign * lam * Omega_{r_i}) x_i`` without per-row matrices.

        One matrix per *distinct relation* (at most ``num_rel`` of them) is
        built and applied to that relation's rows by
        :class:`_GroupedRotationApply`, so memory is ``O(num_rel * D^2 + B * D)``
        instead of ``O(B * D^2)`` and the backward is not a per-relation clone
        of the activations.
        """
        num_rel = self._num_rel
        flat = rel_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=num_rel)
        present = counts > 0
        uniq = torch.nonzero(present, as_tuple=False).reshape(-1)

        theta = self.theta[uniq]
        if lam != 1.0:
            theta = theta * lam
        if sign < 0:
            theta = -theta
        rot = self._rotation_matrix(self._omega(theta))  # [U, D, D]
        # `order` sorts the rows by relation, so the groups land in the same
        # order as the rows of `rot` (which are renumbered to 0..U-1)
        restored = _GroupedRotationApply.apply(
            x, rot, order, counts[uniq].tolist()
        )
        if self.use_scale:
            scale = self.log_scale  # [R]
            if lam != 1.0:
                scale = scale * lam
            if sign < 0:
                scale = -scale
            restored = restored * scale[flat].exp().unsqueeze(-1)
        return restored

    @staticmethod
    def _broadcast_rel(rel_ids: torch.Tensor, B: int) -> torch.Tensor:
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(1).expand(B)
        if rel_ids.numel() == 1 and B > 1:
            rel_ids = rel_ids.expand(B)
        return rel_ids

    def _is_uniform_lambda(self, lambdas: torch.Tensor | None) -> bool:
        """A single dial value for the whole batch keeps the grouped path open."""
        if lambdas is None:
            return True
        return bool(torch.all(lambdas == lambdas.reshape(-1)[0]).item())

    def apply(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        lambdas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        rel_ids = self._broadcast_rel(rel_ids, B)
        if lambdas is not None and lambdas.dim() == 0:
            lambdas = lambdas.view(1).expand(B)

        # dial: scale the rotation by lambda (expm(lambda * Omega) approximated
        # by the Cayley transform of lambda * Omega), and the scale by s^lambda
        context = self.use_context and self.ctx_W is not None
        if not context and self._is_uniform_lambda(lambdas):
            lam = 1.0 if lambdas is None else float(lambdas.reshape(-1)[0].item())
            return self._apply_grouped(x, rel_ids, sign=1.0, lam=lam)

        theta = self.theta[rel_ids]
        if context:
            theta = theta + self.eps_tensor() * self.ctx_P(
                torch.relu(self.ctx_W(x))
            )
        if lambdas is not None:
            theta = theta * lambdas.unsqueeze(-1)
        log_scale = None
        if self.use_scale:
            log_scale = self.log_scale[rel_ids]
            if lambdas is not None:
                log_scale = log_scale * lambdas
        return self._apply_dense_rows(x, theta, log_scale)

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
    def inverse(
        self,
        x: torch.Tensor,
        rel_ids: torch.Tensor,
        base_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact inverse ``exp(-Omega_r) / s_r`` of the transform used forward.

        With a connection (``use_context``) the coefficients depend on where the
        transform was applied, so the caller has to name that point:
        ``inverse(apply(x, r), r, base_point=x) == x`` exactly, while
        ``base_point=None`` inverts the *unmodulated* relation (the transport of
        the chart origin).  Ignoring the modulation would turn every cycle loss
        into a measure of "how much context is there" and train the connection
        away.
        """
        B = x.shape[0]
        rel_ids = self._broadcast_rel(rel_ids, B)
        context = self.use_context and self.ctx_W is not None
        if not context or base_point is None:
            return self._apply_grouped(x, rel_ids, sign=-1.0, lam=1.0)

        theta = self.theta[rel_ids] + self.eps_tensor() * self.ctx_P(
            torch.relu(self.ctx_W(base_point))
        )
        log_scale = -self.log_scale[rel_ids] if self.use_scale else None
        return self._apply_dense_rows(x, -theta, log_scale)

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
