"""
WIĄZKA trainer: curriculum (F1-F4) over the WIĄZKA model.

This is a *self-contained* trainer (it does not depend on the legacy
``RelGATTrainer``) so that the new loss terms, samplers and stage manager
can be exercised end to end without touching the legacy path.  It consumes
the same dataset primitives (node embeddings, relation mapping, triplets)
that the plwordnet-milvus exporter produces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from relgat_projector.core.operator import build_operator
from relgat_projector.core.model.wiazka import WiazkaModel
from relgat_projector.core.loss import (
    TransportLoss,
    PathLoss,
    CurvatureLoss,
    ContrastLoss,
    StabilityLoss,
)
from relgat_projector.core.sampler import (
    PathSampler,
    QuadrupleSampler,
    NegativeSampler,
)
from relgat_projector.trainer.stages import StageManager


@dataclass
class WiazkaConfig:
    op_dim: int = 256
    num_generators: int = 16
    operator_type: str = "givens"
    gat_num_layers: int = 2
    gat_heads: int = 2
    dropout: float = 0.1
    use_scale: bool = True
    use_context: bool = False
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_neg: int = 8
    epochs: int = 1
    device: str = "cpu"
    seed: int = 42


def _build_adjacency(
    edges: List[Tuple[int, int, int]], num_nodes: int
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    out_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    in_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_nodes)}
    for s, d, r in edges:
        out_adj[s].append((d, r))
        in_adj[d].append((s, r))
    return out_adj, in_adj


class WiazkaTrainer:
    def __init__(
        self,
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edges: List[Tuple[int, int, str]],
        cfg: Optional[WiazkaConfig] = None,
    ):
        self.cfg = cfg or WiazkaConfig()
        torch.manual_seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.rel2idx = dict(rel2idx)
        self.num_rel = len(self.rel2idx)

        node_ids = sorted(node2emb.keys())
        self.id2idx = {nid: i for i, nid in enumerate(node_ids)}
        self.num_nodes = len(node_ids)
        D_in = int(next(iter(node2emb.values())).numel())

        self.node_emb = torch.stack(
            [torch.as_tensor(node2emb[nid], dtype=torch.float32) for nid in node_ids]
        )

        # compact edges
        self.edges: List[Tuple[int, int, int]] = [
            (self.id2idx[s], self.id2idx[d], self.rel2idx[r])
            for (s, d, r) in edges
            if s in self.id2idx and d in self.id2idx and r in self.rel2idx
        ]
        src = torch.tensor([e[0] for e in self.edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in self.edges], dtype=torch.long)
        et = torch.tensor([e[2] for e in self.edges], dtype=torch.long)
        self.edge_index = torch.stack([src, dst], dim=0)
        self.edge_type = et

        # operator
        self.operator = build_operator(
            self.cfg.operator_type,
            dim=self.cfg.op_dim,
            num_rel=self.num_rel,
            num_generators=self.cfg.num_generators,
            use_scale=self.cfg.use_scale,
            use_context=False,  # context is enabled by the stage manager at F4
        )

        self.model = WiazkaModel(
            node_emb=self.node_emb,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_rel=self.num_rel,
            operator=self.operator,
            gat_num_layers=self.cfg.gat_num_layers,
            gat_heads=self.cfg.gat_heads,
            dropout=self.cfg.dropout,
            use_applicability=True,
            use_decoder=True,
        ).to(self.cfg.device)

        # samplers
        out_adj, in_adj = _build_adjacency(self.edges, self.num_nodes)
        deg = np.array(
            [len(out_adj.get(i, [])) + len(in_adj.get(i, [])) for i in range(self.num_nodes)],
            dtype=np.float64,
        )
        self.neg_sampler = NegativeSampler(
            num_nodes=self.num_nodes, degrees=deg, rng=random.Random(self.cfg.seed)
        )
        self.path_sampler = PathSampler(
            adjacency=out_adj,
            rel_ids=list(range(self.num_rel)),
            num_nodes=self.num_nodes,
            max_depth=4,
            rng=random.Random(self.cfg.seed + 1),
        )
        self.quad_sampler = QuadrupleSampler(
            out_adj=out_adj, in_adj=in_adj, num_nodes=self.num_nodes,
            rng=random.Random(self.cfg.seed + 2),
        )

        # losses
        self.transport_loss = TransportLoss()
        self.path_loss = PathLoss()
        self.curvature_loss = CurvatureLoss(self.num_rel)
        self.contrast_loss = ContrastLoss()
        self.stability_loss = StabilityLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.stage_manager = StageManager()

    # ------------------------------------------------------------- sampling
    def _sample_edges(self, batch_size: int):
        idx = torch.randint(0, len(self.edges), (batch_size,))
        e = torch.tensor(self.edges, dtype=torch.long)
        return e[idx]  # [B, 3]: src, dst, rel

    def _sample_paths(self, batch_size: int):
        paths = self.path_sampler.sample_batch(batch_size)
        if not paths:
            return None
        srcs = torch.tensor([p[0] for p in paths], dtype=torch.long)
        k = max(len(p[1]) for p in paths)
        rels = torch.zeros(len(paths), k, dtype=torch.long)
        dsts = torch.tensor([p[2] for p in paths], dtype=torch.long)
        for i, p in enumerate(paths):
            rels[i, : len(p[1])] = torch.tensor(p[1])
        return srcs, rels, dsts

    def _sample_quadruples(self, batch_size: int):
        quads = self.quad_sampler.sample_batch(batch_size)
        if not quads:
            return None
        return quads

    # ------------------------------------------------------------- forward
    def _step(self, edges, paths, quads, refined):
        device = self.cfg.device
        B = edges.shape[0]
        loss = torch.zeros((), device=device)
        log: Dict[str, float] = {}

        src, dst, rel = edges[:, 0], edges[:, 1], edges[:, 2]
        x_a = refined[src]
        x_b = refined[dst]

        # L_edge (+ applicability on positives)
        transformed = self.model.apply(x_a, rel)
        a_pos = self.model.applicability_score(x_a, rel)
        l_edge = self.transport_loss(transformed, x_b, a_pos)
        loss = loss + l_edge
        log["L_edge"] = float(l_edge.item())

        # L_anchor
        base = self.model.base_to_op(self.model.node_emb_fixed)
        l_anchor = self.stability_loss(
            refined, None, refined=refined, base=base
        )
        loss = loss + l_anchor
        log["L_anchor"] = float(l_anchor.item())

        if self.stage_manager.is_active("L_con"):
            # in-batch contrast: query = T_r(x_A) (op space), keys = refined
            neg_idx = self.neg_sampler.sample(list(dst.cpu().numpy()), k=self.cfg.num_neg)
            neg_nodes = torch.tensor(neg_idx, dtype=torch.long).to(device)
            keys = torch.cat([refined, refined[neg_nodes]], dim=0)  # [N + B*K, D]
            q = transformed
            target_idx = torch.zeros(B, dtype=torch.long, device=device)  # x_B at row dst?
            # x_B lives at refined[dst]; but keys rows 0..N-1 are all nodes, so
            # target row index == dst value
            target_idx = dst.clone()
            l_con = self.contrast_loss(q, keys, target_idx)
            loss = loss + l_con
            log["L_con"] = float(l_con.item())

        if self.stage_manager.is_active("L_cycle"):
            t = self.model.apply(x_a, rel)
            back = self.model.inverse(t, rel)
            l_cyc = self.stability_loss(x_a, back)
            loss = loss + l_cyc
            log["L_cycle"] = float(l_cyc.item())

        if self.stage_manager.is_active("L_isom"):
            u = x_a
            v = x_b
            tu = self.model.apply(u, rel)
            tv = self.model.apply(v, rel)
            l_iso = self.stability_loss(refined, None, u=u, v=v, tu=tu, tv=tv)
            loss = loss + l_iso
            log["L_isom"] = float(l_iso.item())

        if self.stage_manager.is_active("L_path") and paths is not None:
            p_src, p_rels, p_dst = paths
            p_xa = refined[p_src]
            p_xb = refined[p_dst]
            chain_t = self.model.apply_chain(p_xa, p_rels)
            l_path = self.path_loss(chain_t, None, p_xb)
            loss = loss + l_path
            log["L_path"] = float(l_path.item())

        if self.stage_manager.is_active("L_curv") and quads is not None:
            a, r, c, s, b, p, q = (
                torch.tensor([t[i] for t in quads], dtype=torch.long, device=device)
                for i in range(7)
            )
            xa = refined[a]
            # H = T_p^{-1} o T_q o T_r o T_s  (around A)
            h = self.model.apply(xa, s)
            h = self.model.apply(h, r)
            h = self.model.apply(h, q)
            h = self.model.inverse(h, p)
            # commutation sample: use (r, s) with x_A
            trs = self.model.apply(self.model.apply(xa, r), s)
            tsr = self.model.apply(self.model.apply(xa, s), r)
            l_curv = self.curvature_loss(xa, h, r, s, tsr, trs)
            loss = loss + l_curv
            log["L_curv"] = float(l_curv.item())

        if self.stage_manager.is_active("L_applicability"):
            # negatives: applicability should be low on type-invalid pairs
            neg_rel = (rel + torch.randint(1, self.num_rel, (B,), device=device)) % self.num_rel
            a_neg = self.model.applicability_score(x_a, neg_rel)
            l_app = nn.functional.binary_cross_entropy_with_logits(
                a_neg, torch.zeros_like(a_neg)
            )
            loss = loss + l_app
            log["L_applicability"] = float(l_app.item())

        return loss, log

    # ---------------------------------------------------------------- train
    def train(self, epochs: Optional[int] = None, log_every: int = 50) -> List[Dict]:
        epochs = epochs or self.cfg.epochs
        history: List[Dict] = []
        self.model.train()
        global_step = 0
        for epoch in range(1, epochs + 1):
            epoch_log: Dict[str, float] = {}
            for _ in range(self.cfg.epochs if epochs == self.cfg.epochs else 1):
                edges = self._sample_edges(self.cfg.batch_size).to(self.cfg.device)
                paths = self._sample_paths(self.cfg.batch_size // 2)
                if paths is not None:
                    paths = tuple(t.to(self.cfg.device) for t in paths)
                quads = self._sample_quadruples(self.cfg.batch_size // 4)

                refined = self.model.refine()
                loss, log = self._step(edges, paths, quads, refined)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                global_step += 1
                self.stage_manager.step()
                for k, v in log.items():
                    epoch_log[k] = epoch_log.get(k, 0.0) + v
                if global_step % log_every == 0:
                    print(
                        f"[epoch {epoch} step {global_step}] "
                        f"stage={self.stage_manager.order[self.stage_manager.current]} "
                        f"loss={float(loss.item()):.5f}"
                    )
            n = max(1, self.cfg.epochs)
            avg = {k: v / n for k, v in epoch_log.items()}
            avg["epoch"] = epoch
            avg["stage"] = self.stage_manager.order[self.stage_manager.current]
            history.append(avg)
            if self.stage_manager.should_stop(avg):
                self.stage_manager.advance()
        return history

    # ---------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, n_probe: int = 256) -> Dict[str, float]:
        from relgat_projector.core.metrics import geometry

        self.model.eval()
        refined = self.model.refine()
        device = self.cfg.device
        probe = torch.randint(0, len(self.edges), (n_probe,))
        e = torch.tensor(self.edges, dtype=torch.long)[probe].to(device)
        src, dst, rel = e[:, 0], e[:, 1], e[:, 2]
        x_a = refined[src]
        x_b = refined[dst]
        transformed = self.model.apply(x_a, rel)
        cos = torch.nn.functional.cosine_similarity(transformed, x_b, dim=-1)
        out: Dict[str, float] = {
            "eval/cos_edge": float(cos.mean().item()),
            "eval/cce": geometry.cycle_closure_error(
                self.model.operator, x_a, rel
            ),
            "eval/isot": geometry.isometry_error(
                self.model.operator, refined[src[:64]], refined[dst[:64]], rel[0]
            ),
        }
        # PCE curve
        chains = {}
        targets = {}
        for k in range(1, 4):
            paths = self.path_sampler.sample_batch(min(64, n_probe))
            if not paths:
                continue
            psrc = torch.tensor([p[0] for p in paths], dtype=torch.long, device=device)
            pk = max(len(p[1]) for p in paths)
            prels = torch.zeros(len(paths), pk, dtype=torch.long, device=device)
            pdst = torch.tensor([p[2] for p in paths], dtype=torch.long, device=device)
            for i, p in enumerate(paths):
                prels[i, : len(p[1])] = torch.tensor(p[1], device=device)
            chains[len(paths) and pk] = prels
            targets[pk] = refined[pdst]
        if chains:
            curve = geometry.pce_curve(self.model.operator, refined, chains, targets,
                                       max_k=3)
            out["eval/pce_k1"] = curve[0] if len(curve) > 0 else float("nan")
            out["eval/pce_k2"] = curve[1] if len(curve) > 1 else float("nan")
            out["eval/pce_k3"] = curve[2] if len(curve) > 2 else float("nan")
        return out
