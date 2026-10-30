"""
WIĄZKA model: transport-aware GAT + relational operator + heads.

Wraps the existing :class:`RelGATLayer` (node refinement) together with a
:class:`RelationalOperator` (relations as transport) and the applicability /
relation-decoder heads, exposing the single interface used by the training
stages and the inference API.

Node refinement is *transport-aware*: the message from ``u`` to ``v`` over
relation ``r`` is ``T_r(proj(x_u))`` (PLAN.md 3.2a) instead of a bare
attention-weighted projection, and a residual + ``L_anchor`` keeps nodes in
the neighbourhood of their base embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from relgat_projector.utils.scatter import scatter_add
from relgat_projector.core.operator.base import RelationalOperator
from relgat_projector.core.heads.applicability import ApplicabilityHead
from relgat_projector.core.heads.relation_decoder import RelationDecoder


class TransportGATLayer(nn.Module):
    """One transport-aware message-passing layer (PLAN.md 3.2a).

    Aggregation is *relation-aware and normalised*.  A plain
    ``scatter_add`` of the transported messages over every incoming edge is
    both unnormalised (a node with 17 neighbours gets a message ~17x longer
    than a node with one) and blind to the edge type (a verb's derivational
    neighbours dilute its synonym neighbourhood, and one shared refined
    vector has to serve all 56 relation types at once).  Measured on the
    Slowosiec: unnormalised summing cost ``synonimia`` 0.034 MRR against the
    base embedding while the very same layer was the main source of the
    ``antonimia`` gain (0.009 -> 0.203).

    The attention logits are bounded with ``tanh`` so the softmax cannot
    overflow and no segment-wise maximum is needed::

        s_uv = <q_h(x_v), T_r(phi_h(x_u))> / sqrt(D) + b_h[r]
        alpha_uv = exp(tanh(s_uv / 4) * 4)_uv / sum_{u\' in N(v)} exp(...)_{u\'v}
    """

    #: attention logit bound: exp(+-LOGIT_BOUND) is the largest possible
    #: up-weighting ratio of one neighbour over another (~e^8).
    LOGIT_BOUND = 4.0

    def __init__(self, in_dim: int, out_dim: int, operator: RelationalOperator,
                 heads: int = 1, dropout: float = 0.1, use_context: bool = False,
                 attention: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        # Base-point-dependent transport is deliberately OFF in message
        # passing: context makes the operator row-dependent, which turns the
        # cheap "one matrix per relation" aggregation into a dense per-edge
        # matrix and OOMs on a full-sized graph (324k x 256 x 256).  The
        # per-sample losses in the trainer still run with context on.
        self.use_context = bool(use_context)
        self.attention = bool(attention)
        self.operator = operator
        # project the operator space to the GAT space (per head)
        self.proj = nn.ModuleList(
            [nn.Linear(out_dim, out_dim, bias=False) for _ in range(heads)]
        )
        # project base features to operator space (when dims differ)
        self.base_proj = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )
        if self.attention:
            num_rel = int(getattr(operator, "num_rel", 0))
            # query vector of the *receiving* node (per head)
            self.att_q = nn.ModuleList(
                [nn.Linear(out_dim, out_dim, bias=False) for _ in range(heads)]
            )
            # per-relation prior: lets a head ignore relation types whose
            # messages are noise for it (b_h[r] = 0 -> uniform at init)
            self.att_rel = (
                nn.Parameter(torch.zeros(heads, num_rel)) if num_rel else None
            )
            self.att_scale = out_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        # gated residual: the refinement starts as the *identity* (tanh(0)=0)
        # and may only move as far as its gate opens.  An ungated residual with
        # LayerNorm/GELU on the node path lets the refinement collapse
        # neighbouring nodes into one blob and the operator to learn the
        # identity -- a degenerate optimum with a good edge cosine and no
        # relation at all (measured: refined cos 0.96 on *every* edge type).
        self.gate = nn.Parameter(torch.zeros(1))

    def _messages(self, refined, edge_index, edge_type):
        """Transport of the incoming messages, context-free by design.

        Base-point-dependent transport makes the operator row-dependent, which
        turns one matrix per relation into one matrix per *edge* and is not
        representable on a 324k-edge graph.  The layer therefore runs the
        operator with context switched off -- but it must hand the operator
        back **exactly as it found it**.  Restoring ``context`` (always False,
        because this layer never uses context) instead of the previous value
        switched the F4 connection off for the rest of the step, and since
        ``refine()`` runs first, every per-sample loss in the trainer then saw
        a context-free operator: the hyper-network received exactly zero
        gradient for a whole training run (``ctx_P.std() == 0.0`` in the saved
        checkpoint).
        """
        src, _ = edge_index
        previous = bool(getattr(self.operator, "use_context", False))
        self.operator.use_context = previous and self.use_context
        try:
            return [
                self.operator.apply(self.proj[h](refined)[src], edge_type)
                for h in range(self.heads)
            ]
        finally:
            self.operator.use_context = previous

    def _aggregate_uniform(self, msgs, edge_index, N):
        """Degree-normalised mean over incoming edges (``attention=False``)."""
        _, dst = edge_index
        ones = torch.ones(dst.shape[0], 1, device=dst.device, dtype=msgs[0].dtype)
        deg = scatter_add(ones, dst, dim=0, dim_size=N).clamp_min(1.0)
        heads_out = []
        for m in msgs:
            heads_out.append(scatter_add(m, dst, dim=0, dim_size=N) / deg)
        return heads_out

    def _aggregate_attention(self, msgs, refined, edge_index, edge_type, N):
        _, dst = edge_index
        logits = []
        for h in range(self.heads):
            q = self.att_q[h](refined)[dst]           # [E, D]
            s = (q * msgs[h]).sum(dim=-1) * self.att_scale
            if self.att_rel is not None:
                s = s + self.att_rel[h][edge_type]
            logits.append(s)
        # [E, heads], bounded so exp() cannot overflow
        S = torch.tanh(torch.stack(logits, dim=-1) / self.LOGIT_BOUND)
        S = S * self.LOGIT_BOUND
        w = torch.exp(S)
        denom = scatter_add(w, dst, dim=0, dim_size=N).clamp_min(1e-9)
        heads_out = []
        for h in range(self.heads):
            alpha_h = w[:, h] / (denom[dst, h] + 1e-9)     # [E]
            heads_out.append(
                scatter_add(alpha_h.unsqueeze(-1) * msgs[h], dst, dim=0, dim_size=N)
            )
        return heads_out

    def forward(
        self,
        base_emb: torch.Tensor,   # [N, in_dim]  (fixed base vectors)
        refined: torch.Tensor,    # [N, out_dim] (current refinement, op space)
        edge_index: torch.Tensor, # [2, E]
        edge_type: torch.Tensor,  # [E]
    ) -> torch.Tensor:
        _, dst = edge_index
        N = base_emb.size(0)

        # transport-aware messages: T_r(proj(x_u))
        msgs = self._messages(refined, edge_index, edge_type)

        if self.attention:
            heads_out = self._aggregate_attention(msgs, refined, edge_index, edge_type, N)
        else:
            heads_out = self._aggregate_uniform(msgs, edge_index, N)

        # fold heads back to out_dim
        out = torch.stack(heads_out, dim=1).mean(dim=1)  # [N, out_dim]
        # non-linearity on the *update*, never on the node representation:
        # GELU/LayerNorm applied to the node vector destroys its cosine
        # geometry with the base embedding (and with it the retriever space)
        update = self.act(self.dropout(out))
        return refined + torch.tanh(self.gate) * update


class WiazkaModel(nn.Module):
    """Full WIĄZKA model (refinement + operator + heads)."""

    def __init__(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_rel: int,
        operator: RelationalOperator,
        gat_num_layers: int = 2,
        gat_heads: int = 2,
        dropout: float = 0.1,
        use_applicability: bool = True,
        use_decoder: bool = True,
        attention: bool = True,
        proj_weight: torch.Tensor | None = None,
        tie_projections: bool = True,
        freeze_projection: bool = False,
        center_mean: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("node_emb_fixed", node_emb)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)
        self.num_rel = num_rel
        self.operator = operator
        self._op_dim = operator.dim

        D = self._op_dim
        self.layers = nn.ModuleList(
            [
                TransportGATLayer(
                    in_dim=node_emb.size(1),
                    out_dim=D,
                    operator=operator,
                    heads=gat_heads,
                    dropout=dropout,
                    use_context=False,
                    attention=attention,
                )
                for _ in range(max(1, gat_num_layers))
            ]
        )

        # the refined node vectors live in operator space; initialise them as
        # a projection of the base embeddings (so that T_r is well-defined on
        # them from step 0).
        # ``proj_weight`` is the principal subspace of the base embeddings
        # (rows orthonormal, ``[D, base_dim]``).  With ``tie_projections`` the
        # export map is the *transpose* of the same matrix, so
        # ``to_retriever(base_to_op(x))`` is an orthogonal projection at
        # initialisation: the operator starts acting inside a faithful
        # low-dimensional copy of the retriever space instead of behind a
        # random, untrained linear map (an untied export was a random
        # projection until ``L_pull`` switched on at F4, which made every
        # QR-Recall number measured before that stage exactly 0.0).
        base_dim = node_emb.size(1)
        self.tie_projections = bool(tie_projections) and base_dim != D
        #: origin of the operator chart.  ``base_to_op`` is affine when a
        #: centre is supplied, so *every* call site -- refinement, shadow
        #: manifold, production query, relation logits -- sees the centred
        #: chart without knowing about it, and ``_export_linear`` adds the
        #: centre back so the retriever space is exactly the stored vectors.
        self.register_buffer(
            "center_mean",
            torch.zeros(base_dim) if center_mean is None
            else center_mean.detach().to(dtype=torch.float32).clone(),
        )
        self.centered = center_mean is not None
        if base_dim != D:
            self.base_to_op = nn.Linear(base_dim, D, bias=self.centered)
        elif self.centered:
            # same dimension, but the chart still needs an origin: an
            # identity linear map with the centering bias
            identity = nn.Linear(base_dim, D, bias=True)
            with torch.no_grad():
                identity.weight.copy_(torch.eye(D, dtype=identity.weight.dtype))
                identity.bias.zero_()
            self.base_to_op = identity
        else:
            self.base_to_op = nn.Identity()
        if self.centered:
            # the caller may pass the centre from the training device (the
            # trainer computes it on the device copy of the embeddings) while
            # the module is still being built on the CPU; the bias is the one
            # place both operands must meet, so they meet there
            with torch.no_grad():
                center = self.center_mean.to(self.base_to_op.weight.device)
                self.base_to_op.bias.copy_(
                    -self.base_to_op.weight @ center
                )
                self.base_to_op.bias.requires_grad_(False)
        if proj_weight is not None:
            if not isinstance(self.base_to_op, nn.Linear):
                raise ValueError("proj_weight requires base_dim != op_dim")
            want = tuple(self.base_to_op.weight.shape)
            given = tuple(proj_weight.shape)
            if given != want:
                raise ValueError(f"proj_weight shape {given} != {want}")
            with torch.no_grad():
                self.base_to_op.weight.copy_(
                    proj_weight.to(self.base_to_op.weight.device,
                                   self.base_to_op.weight.dtype)
                )
        #: the base<->operator map is kept orthonormal (see __init__ docs)
        self.frozen_projection = False
        if freeze_projection:
            if proj_weight is None:
                raise ValueError("freeze_projection requires proj_weight")
            self.base_to_op.weight.requires_grad_(False)
            self.frozen_projection = True

        # projection of the refined/operator space back to the base
        # (retriever) space, so that a transformed vector can be exported to
        # the vector database (PLAN.md 2.6, L_pull)
        if self.tie_projections:
            self.to_retriever = None
        else:
            self.to_retriever = (
                nn.Linear(D, base_dim, bias=False) if D != base_dim else nn.Identity()
            )
        self.base_dim = base_dim
        # a small gate (tanh, starts at 0) keeps the export a controlled
        # deformation of the base space: the vector database holds *base*
        # vectors, and a large learned linear map would move an exported query
        # far from every index entry, destroying cosine search.  The export is
        # base + g * Linear(x - base) in the common case where base_to_op is
        # the identity; otherwise it is base + g * Linear(x) projected back.
        # ``export_gate`` is exactly 0 (export = the chart adjoint), while
        # ``export_proj`` must NOT be zero: the product gate*proj has a
        # vanishing gradient in both factors at a double zero, so a zero
        # projection freezes the gate forever and L_pull can never open it.
        self.export_gate = nn.Parameter(torch.zeros(1))
        self.export_proj = nn.Linear(D, base_dim, bias=False)
        nn.init.normal_(self.export_proj.weight, std=1e-3)

        #: Per-relation strength of query rewriting, in ``[0, 1]`` (logits).
        #: ``production_query`` returns ``(1-lam_r) * x + lam_r * export(T_r(x))``,
        #: so ``lam_r = 0`` is "do not rewrite this relation" and the query stays
        #: the raw embedding.  Without it the model has no way to say "this
        #: relation is better served by the base similarity": on the Slowosiec the
        #: transport is a large win on ~40 relations but *loses* to plain
        #: similarity on the ones the base embeddings already encode (``synonimia``
        #: 0.59 raw vs 0.42 refined; QR-Recall@1 0.50 -> 0.38), and a single
        #: global answer has to be wrong for one side or the other.
        #:
        #: Initialised at 0 (``lam_r = 0.5``), NOT below 1: with every relation
        #: far from the boundary the dial's gradient is
        #: ``dL/dlogit = lam * (1 - lam) * dL/dlam <= 0.25 * dL/dlam`` and it
        #: takes tens of thousands of steps to move at all (measured: after a
        #: full 35-epoch run the logit had moved 1.5 -> 1.46, i.e. the dial was
        #: still at 0.81 for every relation and synonimia's export was still
        #: below the raw query).  At the boundary the same loss signal moves the
        #: dial in both directions from step one, and ``lam = 0.5`` is exactly
        #: the neutral point: the query is half-raw, half-transport, so opening
        #: the dial can only be earned, and so can closing it.
        self.transport_dial = nn.Parameter(torch.zeros(num_rel))

        self.use_applicability = use_applicability
        self.use_decoder = use_decoder
        #: weight of the transport evidence in ``mode="ensemble"``:
        #: ``z(head) + w * z(transport)``.  1.0 is the unweighted combination;
        #: :meth:`WiazkaTrainer.fit_relation_ensemble_weight` fits it on train
        #: edges, because which evidence dominates changes with how the head is
        #: trained (unbalanced head: transport wins, class-balanced head: the
        #: head wins and transport only dilutes it).
        self.relation_ensemble_weight = 1.0
        #: strength of the training-frequency correction in ``mode="prior"``:
        #: ``log_softmax(logits) - strength * log p_train(r)``.  ``1.0`` is the
        #: textbook Bayes correction, valid for a head trained with plain
        #: cross-entropy.  A head trained with class weights ``w(r) ~ count**-beta``
        #: already fits ``p(r|x) * w(r)``, so the correction it needs is
        #: ``-log w(r) = +beta * log count`` -- the *opposite* sign -- and applying
        #: the plain one to a balanced head is catastrophic (measured macro
        #: accuracy 0.058 against 0.465 uncorrected).  The trainer fits this one
        #: scalar on train edges instead of assuming a value.
        self.relation_prior_strength = 1.0
        self._ctor = {
            "gat_num_layers": max(1, gat_num_layers),
            "gat_heads": gat_heads,
            "dropout": dropout,
            "use_applicability": use_applicability,
            "use_decoder": use_decoder,
            # defaults keep checkpoints written before these options existed
            # loadable with exactly the architecture they were trained with
            "attention": bool(attention),
            "tie_projections": bool(tie_projections),
            "frozen_projection": bool(self.frozen_projection),
            # a centred chart needs a bias in base_to_op, so the shape of the
            # module -- and therefore the state dict -- depends on it
            "centered": bool(self.centered),
        }
        if use_applicability:
            self.applicability = ApplicabilityHead(D, num_rel)
        if use_decoder:
            self.decoder = RelationDecoder(D, num_rel)

    # ------------------------------------------------------------- refinement
    def refine(self) -> torch.Tensor:
        """Return refined node vectors ``[N, D_op]`` in operator space."""
        x = self.base_to_op(self.node_emb_fixed)
        for layer in self.layers:
            x = layer(self.node_emb_fixed, x, self.edge_index, self.edge_type)
        return x

    def get_node_repr(self) -> torch.Tensor:
        return self.refine()

    # ------------------------------------------------------------------ apply
    def apply(self, x: torch.Tensor, rel_ids: torch.Tensor,
              lambdas: torch.Tensor | None = None) -> torch.Tensor:
        return self.operator.apply(x, rel_ids, lambdas)

    def apply_chain(self, x: torch.Tensor, rel_ids: torch.Tensor,
                    lambdas: torch.Tensor | None = None) -> torch.Tensor:
        return self.operator.apply_chain(x, rel_ids, lambdas)

    def compiled(self, rel_ids: torch.Tensor,
                 lambdas: torch.Tensor | None = None):
        return self.operator.compiled(rel_ids, lambdas)

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor,
                base_point: torch.Tensor | None = None) -> torch.Tensor:
        return self.operator.inverse(x, rel_ids, base_point=base_point)

    def applicability_score(self, x: torch.Tensor,
                            rel_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_applicability:
            return x.new_zeros(x.shape[0])
        return self.applicability.score(x, rel_ids)

    def _export_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Operator space -> base (retriever) space, respecting the tie.

        ``export_proj`` is a gated residual correction on top of the adjoint
        (zero-initialised, gate starts closed), so the export is the exact
        isometry of the chart until ``L_pull`` earns a deformation.
        """
        if self.tie_projections:
            # rows of base_to_op.weight are orthonormal directions of the base
            # space, so x @ W is exactly the adjoint of base_to_op
            out = x @ self.base_to_op.weight
        else:
            out = self.to_retriever(x)
        out = out + torch.tanh(self.export_gate) * self.export_proj(x)
        if self.centered:
            # the chart was built on x - mu, so its adjoint lands in the
            # centred space; the vector database holds the stored vectors
            out = out + self.center_mean
        return out

    def to_retriever_space(self, x: torch.Tensor) -> torch.Tensor:
        """Export operator-space vectors into the base (retriever) space.

        The base vectors are what the vector database actually holds, so the
        export is anchored to the base space: ``base + g * (P(x) - base)``
        where ``base`` is the *base embedding of the same node* when the input
        carries one (node vectors), and the origin otherwise.  ``g`` starts at
        0 (export = base) and opens only if it helps the export losses.
        """
        g = torch.tanh(self.export_gate)
        if x.shape[0] == self.node_emb_fixed.shape[0]:
            base = self.node_emb_fixed
            # row alignment: caller passes the full node matrix in order
            return base + g * (self._export_linear(x) - base)
        return self._export_linear(x)

    def transport_strength(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """Learned rewriting strength ``lam_r`` in ``[0, 1]`` per row."""
        return torch.sigmoid(self.transport_dial[rel_ids])

    def production_query(self, x_base: torch.Tensor,
                         rel_ids: torch.Tensor) -> torch.Tensor:
        """The query a deployed rewriter actually sends to the vector database.

        ``x_base`` is a *base* (retriever-space) vector -- that is what an
        incoming query is -- and the result is again in base space::

            q = (1 - lam_r) * x + lam_r * export(T_r(chart(x)))

        Every protocol that claims to measure the product has to use this path
        and nothing else: training a different function from the one that is
        scored (or served) is how ``eval/mrr`` reached 0.19 while the exported
        query scored 0.11.
        """
        charted = self.base_to_op(x_base)
        exported = self._export_linear(self.apply(charted, rel_ids))
        lam = self.transport_strength(rel_ids).unsqueeze(-1)
        return (1.0 - lam) * x_base + lam * exported

    # ------------------------------------------------- relation scoring (1.5)
    def transport_relation_scores(
        self, x_a: torch.Tensor, x_b: torch.Tensor, chunk: int = 256
    ) -> torch.Tensor:
        """Transport fidelity of *every* relation: ``cos(T_r(x_a), x_b)``.

        The relation is by definition the group element that maps ``x_a`` onto
        ``x_b``, so the transport itself answers the inverse problem: score all
        ``num_rel`` candidates and take the best.  It needs no head, no extra
        parameters and no training of its own -- it is exactly the quantity the
        edge loss optimises -- which is why on the Słowosieć it already matches
        the trained MLP decoder (macro 0.295 vs 0.285) and, calibrated, beats it
        (0.330).

        Returns ``[B, num_rel]``.  Without a connection every relation is one
        matrix and the whole thing is two einsums; with the connection the
        coefficients depend on the base point, so it falls back to the exact
        per-row transport in row chunks.
        """
        R = self.num_rel
        B = x_a.shape[0]
        op = self.operator
        matrices = getattr(op, "relation_matrices", None)
        if matrices is not None and not bool(getattr(op, "use_context", False)):
            rot = matrices()                                  # [R, D, D]
            # row-vector form of ``x -> R_r x``: identical to apply()
            transported = torch.einsum("bi,rji->brj", x_a, rot)  # [B, R, D]
            num = (transported * x_b.unsqueeze(1)).sum(-1)
            den = (transported.norm(dim=-1) * x_b.norm(dim=-1).unsqueeze(1))
            return num / den.clamp_min(1e-8)
        # context on: the coefficients depend on the base point, so the exact
        # score goes through the per-row transport in bounded row chunks
        out = x_a.new_empty(B, R)
        for start in range(0, B, chunk):
            stop = min(start + chunk, B)
            rows = x_a[start:stop]
            cols = []
            for r in range(R):
                ids = rows.new_full((rows.shape[0],), r, dtype=torch.long)
                t = self.apply(rows, ids)
                cols.append(torch.nn.functional.cosine_similarity(
                    t, x_b[start:stop], dim=-1))
            out[start:stop] = torch.stack(cols, dim=-1)
        return out

    @staticmethod
    def _row_zscore(scores: torch.Tensor) -> torch.Tensor:
        return (scores - scores.mean(dim=-1, keepdim=True)) / \
            scores.std(dim=-1, keepdim=True).clamp_min(1e-6)

    def relation_logits(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        mode: str = "prior",
        train_counts: torch.Tensor | None = None,
        transport_scores: torch.Tensor | None = None,
        score_stats: tuple | None = None,
        ensemble_weight: float | None = None,
    ) -> torch.Tensor:
        """Comparable ``[B, num_rel]`` evidence per relation for one pair.

        ``mode``
            * ``"decoder"``   -- raw head logits (what the head was trained on),
            * ``"prior"``     -- head log-probabilities minus the training-set
              log-frequency of the relation.  The head is trained on the raw
              relation distribution (``synonimia`` is half of the Słowosieć), so
              it systematically over-predicts the frequent relations; on a
              relation-uniform query the Bayes rule is exactly this correction,
              and measured it is worth +0.096 macro accuracy,
            * ``"transport"`` -- transport fidelity, calibrated against the score
              every relation gives on *random* pairs (``score_stats``),
            * ``"ensemble"``  -- row-z-scored head evidence + ``ensemble_weight``
              times the calibrated transport evidence (default: the model's
              :attr:`relation_ensemble_weight`).

        ``score_stats`` is ``(mean, std)`` of the transport scores over random
        pairs (``[R]`` each); without it the transport modes use no calibration.
        """
        if not self.use_decoder and mode != "transport":
            raise RuntimeError("decoder disabled")
        if mode == "transport":
            s = (transport_scores if transport_scores is not None
                 else self.transport_relation_scores(x_a, x_b))
            if score_stats is not None:
                mean, std = score_stats
                s = (s - mean) / std.clamp_min(1e-6)
            return s
        logits, _ = self.decoder(x_a, x_b)
        if mode == "decoder":
            return logits
        head = torch.log_softmax(logits, dim=-1)
        prior = None
        if train_counts is not None:
            freq = train_counts.to(head.device).clamp_min(1.0)
            prior = self.relation_prior_strength * torch.log(
                freq / float(freq.sum()))
        if mode == "prior" and prior is not None:
            head = head - prior
        if mode == "ensemble":
            t = self.relation_logits(
                x_a, x_b, mode="transport", score_stats=score_stats,
                transport_scores=transport_scores)
            w = (self.relation_ensemble_weight if ensemble_weight is None
                 else float(ensemble_weight))
            # the same prior correction the head alone is scored with, so the
            # ensemble cannot be dragged by a relation frequency the head has
            # already been corrected for (or, with a balanced head, by the
            # correction itself)
            head_evidence = head if prior is None else head - prior
            return (self._row_zscore(head_evidence)
                    + w * self._row_zscore(t))
        return head

    def classify_relation(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        mode: str = "prior",
        train_counts: torch.Tensor | None = None,
        score_stats: tuple | None = None,
    ) -> tuple:
        """Predict the relation of a pair: ``(rel_ids [B], evidence [B, R])``."""
        evidence = self.relation_logits(
            x_a, x_b, mode=mode, train_counts=train_counts,
            score_stats=score_stats)
        return evidence.argmax(dim=-1), evidence

    def decode(self, x: torch.Tensor, y: torch.Tensor):
        if not self.use_decoder:
            raise RuntimeError("decoder disabled")
        return self.decoder.decode(x, y)

    # ------------------------------------------------------------ (de)serialisation
    def spec(self) -> dict:
        """Everything needed to rebuild this exact model (checkpoint metadata)."""
        return {
            "base_dim": self.base_dim,
            "num_rel": self.num_rel,
            "operator": getattr(self.operator, "ctor_kwargs", None),
            **self._ctor,
        }

    @classmethod
    def from_spec(cls, spec: dict, node_emb: torch.Tensor | None = None,
                  edge_index: torch.Tensor | None = None,
                  edge_type: torch.Tensor | None = None,
                  operator=None) -> "WiazkaModel":
        """Rebuild a model from :meth:`spec`.

        ``node_emb`` / ``edge_index`` are *structure*, not weights: when they
        are not supplied the checkpoint buffers (saved in the state dict) are
        re-created with placeholder shapes taken from the checkpoint, so that
        ``load_state_dict`` restores the real graph afterwards.
        """
        from relgat_projector.core.operator import build_operator

        op_spec = spec["operator"]
        if operator is None:
            if not op_spec:
                raise ValueError("spec has no operator construction information")
            operator = build_operator(op_spec["type"], **op_spec["kwargs"])
        if node_emb is None:
            node_emb = torch.zeros(0, int(spec["base_dim"]))
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        if edge_type is None:
            edge_type = torch.empty(0, dtype=torch.long)
        model = cls(
            node_emb=node_emb,
            edge_index=edge_index,
            edge_type=edge_type,
            num_rel=int(spec["num_rel"]),
            operator=operator,
            gat_num_layers=int(spec.get("gat_num_layers", 2)),
            gat_heads=int(spec.get("gat_heads", 2)),
            dropout=float(spec.get("dropout", 0.1)),
            use_applicability=bool(spec.get("use_applicability", True)),
            use_decoder=bool(spec.get("use_decoder", True)),
            attention=bool(spec.get("attention", False)),
            tie_projections=bool(spec.get("tie_projections", False)),
            center_mean=(
                torch.zeros(int(spec["base_dim"]))
                if bool(spec.get("centered", False)) else None
            ),
        )
        # the projection is rebuilt from the state dict, so "frozen" has to be
        # re-applied after construction rather than passed through the ctor
        if bool(spec.get("frozen_projection", False)):
            model.base_to_op.weight.requires_grad_(False)
            model.frozen_projection = True
        return model
