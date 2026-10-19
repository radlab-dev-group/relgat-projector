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
    """One transport-aware message-passing layer (PLAN.md 3.2a)."""

    def __init__(self, in_dim: int, out_dim: int, operator: RelationalOperator,
                 heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.operator = operator
        # project the operator space to the GAT space (per head)
        self.proj = nn.ModuleList(
            [nn.Linear(out_dim, out_dim, bias=False) for _ in range(heads)]
        )
        # project base features to operator space (when dims differ)
        self.base_proj = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        # gated residual: the refinement starts as the *identity* (tanh(0)=0)
        # and may only move as far as its gate opens.  An ungated residual with
        # LayerNorm/GELU on the node path lets the refinement collapse
        # neighbouring nodes into one blob and the operator to learn the
        # identity -- a degenerate optimum with a good edge cosine and no
        # relation at all (measured: refined cos 0.96 on *every* edge type).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        base_emb: torch.Tensor,   # [N, in_dim]  (fixed base vectors)
        refined: torch.Tensor,    # [N, out_dim] (current refinement, op space)
        edge_index: torch.Tensor, # [2, E]
        edge_type: torch.Tensor,  # [E]
    ) -> torch.Tensor:
        src, dst = edge_index
        N = base_emb.size(0)

        # base features projected to operator space
        b = self.base_proj(base_emb)  # [N, out_dim]

        # transport-aware messages: T_r(proj(x_u))
        msgs = []
        for h in range(self.heads):
            proj_src = self.proj[h](refined)[src]  # [E, out_dim]
            transported = self.operator.apply(proj_src, edge_type)  # [E, out_dim]
            msgs.append(transported)

        out = torch.cat(
            [scatter_add(m, dst, dim=0, dim_size=N) for m in msgs], dim=-1
        )  # [N, heads*out_dim]

        # fold heads back to out_dim
        out = out.view(N, self.heads, self.out_dim).mean(dim=1)
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
                )
                for _ in range(max(1, gat_num_layers))
            ]
        )

        # the refined node vectors live in operator space; initialise them as
        # a projection of the base embeddings (so that T_r is well-defined on
        # them from step 0)
        self.base_to_op = (
            nn.Linear(node_emb.size(1), D, bias=False)
            if node_emb.size(1) != D
            else nn.Identity()
        )

        # projection of the refined/operator space back to the base
        # (retriever) space, so that a transformed vector can be exported to
        # the vector database (PLAN.md 2.6, L_pull)
        self.to_retriever = (
            nn.Linear(D, node_emb.size(1), bias=False)
            if D != node_emb.size(1)
            else nn.Identity()
        )
        self.base_dim = node_emb.size(1)
        # a small gate (tanh, starts at 0) keeps the export a controlled
        # deformation of the base space: the vector database holds *base*
        # vectors, and a large learned linear map would move an exported query
        # far from every index entry, destroying cosine search.  The export is
        # base + g * Linear(x - base) in the common case where base_to_op is
        # the identity; otherwise it is base + g * Linear(x) projected back.
        self.export_gate = nn.Parameter(torch.zeros(1))
        self.export_proj = nn.Linear(D, node_emb.size(1), bias=False)
        nn.init.zeros_(self.export_proj.weight)

        self.use_applicability = use_applicability
        self.use_decoder = use_decoder
        self._ctor = {
            "gat_num_layers": max(1, gat_num_layers),
            "gat_heads": gat_heads,
            "dropout": dropout,
            "use_applicability": use_applicability,
            "use_decoder": use_decoder,
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

    def inverse(self, x: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        return self.operator.inverse(x, rel_ids)

    def applicability_score(self, x: torch.Tensor,
                            rel_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_applicability:
            return x.new_zeros(x.shape[0])
        return self.applicability.score(x, rel_ids)

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
            return base + g * (self.to_retriever(x) - base)
        return self.to_retriever(x)

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
        return cls(
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
        )
