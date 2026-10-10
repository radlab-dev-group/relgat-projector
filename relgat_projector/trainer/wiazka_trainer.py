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
import dataclasses
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
    ShadowLoss,
    DecoderLoss,
)
from relgat_projector.core.sampler import (
    PathSampler,
    QuadrupleSampler,
    NegativeSampler,
    ShadowSampler,
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
    #: fraction of edges used for training (topology + supervision);
    #: the rest is held out and never seen by the GAT (PLAN.md 6.6)
    train_ratio: float = 1.0
    device: str = "cpu"
    seed: int = 42
    # number of shadow vectors synthesised when none are supplied (PLAN.md 2.6)
    num_shadow: int = 512
    shadow_jitter: float = 0.35
    # F4: shadow vectors drawn per step (PLAN.md 2.6)
    shadow_batch: int = 64


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
        shadow_vectors=None,
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
        # PLAN.md 6.6: the GAT topology and the supervised edges must be an
        # explicit, reproducible split -- otherwise held-out edges leak into
        # message passing and every "eval" metric is measured on training data.
        self.eval_edges: List[Tuple[int, int, int]] = []
        ratio = float(self.cfg.train_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"train_ratio must be in (0, 1], got {ratio}")
        if ratio < 1.0 and len(self.edges) > 1:
            self.edges, self.eval_edges = self._split_edges(ratio)
            print(
                f"[split] train edges={len(self.edges)} "
                f"held-out edges={len(self.eval_edges)}"
            )
        self.topology_endpoint_coverage = self._coverage_report()

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
            num_rel=self.num_rel,
            rng=random.Random(self.cfg.seed + 2),
        )

        # losses
        self.transport_loss = TransportLoss()
        self.path_loss = PathLoss()
        self.curvature_loss = CurvatureLoss(self.num_rel)
        self.contrast_loss = ContrastLoss()
        self.stability_loss = StabilityLoss()
        self.shadow_loss = ShadowLoss()
        self.decoder_loss = DecoderLoss()

        # shadow manifold: real text vectors from the retriever index.  When
        # none are provided we synthesise them by jittering the base node
        # embeddings, which approximates "a text near a node but not the node".
        if shadow_vectors is None:
            base_np = self.node_emb.detach().cpu().numpy()
            rng = np.random.default_rng(self.cfg.seed)
            idx = rng.integers(0, len(base_np), size=self.cfg.num_shadow)
            jitter = rng.normal(
                0.0, self.cfg.shadow_jitter, size=base_np[idx].shape
            ).astype(np.float32)
            shadow_np = base_np[idx] + jitter
        else:
            shadow_np = np.asarray(shadow_vectors, dtype=np.float32)
        self.shadow_sampler = ShadowSampler(shadow_np, rng=self.cfg.seed)
        self._shadow_cache = torch.as_tensor(shadow_np, dtype=torch.float32).to(
            self.cfg.device
        )

        # NOTE: CurvatureLoss owns the commutation matrix C_rs as a parameter,
        # so the optimizer must include the loss modules, not only the model.
        # Omitting them silently freezes C_rs at its initialisation.
        params = list(self.model.parameters())
        for module in (self.curvature_loss,):
            params += [
                p for p in module.parameters() if p.requires_grad
            ]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.stage_manager = StageManager()
        self.history: List[Dict] = []
        self._sync_stage_state()

    # ------------------------------------------------------------- sampling
    def _sync_stage_state(self) -> None:
        """Apply the current stage's structural switches to operator + optimizer.

        F4 turns on base-point-dependent transport (PLAN.md 1.3):
        ``theta(r, x) = theta_r + eps * P(relu(W x))``.  The hyper-network is
        created *at that moment*, so its parameters have to be registered with
        the optimizer as well -- otherwise context would exist but never train.
        """
        wanted = self.stage_manager.stage().enable_context
        enable = getattr(self.operator, "enable_context", None)
        if enable is not None and bool(self.operator.use_context) != wanted:
            enable(wanted)
        known = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group["params"]
        }
        fresh = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in known
        ]
        if fresh:
            self.optimizer.add_param_group({"params": fresh})

    def _split_edges(self, ratio: float):
        """Deterministic, triplet-disjoint train/holdout split of the edges.

        Splits on *positions* of the edge list, so when the graph contains
        duplicate ``(src, dst, rel)`` triplets (multi-graph semantics) no
        position is used for training and held out at the same time.
        """
        order = list(range(len(self.edges)))
        random.Random(self.cfg.seed).shuffle(order)
        n_train = max(1, int(round(ratio * len(order))))
        train_ids, eval_ids = set(order[:n_train]), set(order[n_train:])
        train, eval_ = [], []
        for i in range(len(self.edges)):
            (train if i in train_ids else eval_).append(self.edges[i])
        return train, eval_

    def _coverage_report(self) -> float:
        """Share of held-out edges whose endpoints exist in the GAT topology."""
        if not self.eval_edges:
            return 1.0
        nodes = set()
        for s, d, _ in self.edges:
            nodes.add(s)
            nodes.add(d)
        visible = sum(1 for s, d, _ in self.eval_edges if s in nodes and d in nodes)
        coverage = visible / max(1, len(self.eval_edges))
        print(f"[split] held-out edges reachable in topology: {coverage:.1%}")
        return coverage

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
            neg_idx = self.neg_sampler.sample_pool(
                list(dst.cpu().numpy()), k=self.cfg.num_neg
            )
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

        if self.stage_manager.is_active("L_curv") and quads:
            a = torch.tensor([d.a for d in quads], dtype=torch.long, device=device)
            r = torch.tensor([d.r for d in quads], dtype=torch.long, device=device)
            q = torch.tensor([d.q for d in quads], dtype=torch.long, device=device)
            p = torch.tensor([d.p for d in quads], dtype=torch.long, device=device)
            s = torch.tensor([d.s for d in quads], dtype=torch.long, device=device)
            xa = refined[a]
            # holonomy around the diamond: H = T_p^-1 o T_s^-1 o T_q o T_r
            h = self.model.apply(xa, r)
            h = self.model.apply(h, q)
            h = self.model.inverse(h, s)
            h = self.model.inverse(h, p)
            # the two routes A->D of the diamond
            route1 = self.model.apply(self.model.apply(xa, r), q)
            route2 = self.model.apply(self.model.apply(xa, p), s)
            l_curv = self.curvature_loss(xa, h, r, p, route1, route2)
            loss = loss + l_curv
            log["L_curv"] = float(l_curv.item())

        if self.stage_manager.is_active("L_commut"):
            # the commutation matrix C_rs has no gradient that keeps it away
            # from 0 in the unsupervised penalty, so it is taught from the
            # graph: relation pairs whose two routes meet are commuting.
            pos_pairs, neg_pairs = self.quad_sampler.sample_relation_pairs(
                batch_size=16, max_neg=16
            )
            if pos_pairs:
                l_comm = self.curvature_loss.supervision(pos_pairs, neg_pairs)
                loss = loss + l_comm
                log["L_commut"] = float(l_comm.item())

        if self.stage_manager.is_active("L_shadow"):
            # cycle + commutation consistency OFF the graph (PLAN.md 2.6)
            z_base = self._shadow_cache[: min(self.cfg.shadow_batch, self._shadow_cache.shape[0])]
            z = self.model.base_to_op(z_base)
            r_sh = torch.randint(0, self.num_rel, (z.shape[0],), device=device)
            back = self.model.inverse(self.model.apply(z, r_sh), r_sh)
            l_sh_cyc = self.shadow_loss.cycle(z, back)
            s_sh = (r_sh + 1) % self.num_rel
            trst = self.model.apply(self.model.apply(z, r_sh), s_sh)
            tstr = self.model.apply(self.model.apply(z, s_sh), r_sh)
            l_sh_com = self.shadow_loss.commute(trst, tstr)
            loss = loss + l_sh_cyc + l_sh_com
            log["L_shadow_cycle"] = float(l_sh_cyc.item())
            log["L_shadow_commut"] = float(l_sh_com.item())

        if self.stage_manager.is_active("L_pull"):
            # exporting a transformed vector must preserve its neighbourhood
            # in the retriever space (base vectors are the index)
            p_trans = self.model.to_retriever_space(transformed)
            base = self.model.node_emb_fixed
            cos_pos = torch.nn.functional.cosine_similarity(
                p_trans, base[dst], dim=-1
            )
            neg_rows = torch.randint(
                0, base.shape[0], (B, self.cfg.num_neg), device=device
            )
            cos_neg = torch.nn.functional.cosine_similarity(
                p_trans.unsqueeze(1), base[neg_rows], dim=-1
            )
            l_pull = (1.0 - cos_pos).mean() + cos_neg.mean().clamp_min(0.0) * 0.5
            loss = loss + l_pull
            log["L_pull"] = float(l_pull.item())

        if self.stage_manager.is_active("L_decoder"):
            logits, conf = self.model.decoder(x_a, x_b)
            l_dec = self.decoder_loss(logits, conf, rel)
            loss = loss + l_dec
            log["L_decoder"] = float(l_dec.item())

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
        self.history = history
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

            # stage acceptance is measured on eval metrics, not train losses
            eval_metrics = self.evaluate(n_probe=min(128, max(16, len(self.edges))))
            avg.update(eval_metrics)
            history.append(avg)
            if self.stage_manager.should_stop(eval_metrics):
                print(
                    f"[stage] {self.stage_manager.order[self.stage_manager.current]}"
                    f" -> gate passed, advancing"
                )
                self.stage_manager.advance()
                self._sync_stage_state()
        return history

    # ---------------------------------------------------------------- storage
    def save(self, ckpt_dir: str, metrics: Optional[Dict[str, float]] = None) -> str:
        """Persist model + config + the learned commutation matrix C_rs."""
        from relgat_projector.handlers.wiazka_checkpoint import save_checkpoint

        history = getattr(self, "history", None) or []
        last = history[-1] if history else None
        return save_checkpoint(
            ckpt_dir,
            self.model,
            self.rel2idx,
            config={
                **dataclasses.asdict(self.cfg),
                "num_nodes": self.num_nodes,
                "num_edges": len(self.edges),
                "stage": self.stage_manager.order[self.stage_manager.current],
            },
            commutation=self.curvature_loss.commutation(),
            metrics=metrics if metrics is not None else last,
        )

    # ---------------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, n_probe: int = 256) -> Dict[str, float]:
        from relgat_projector.core.metrics import geometry

        self.model.eval()
        refined = self.model.refine()
        device = self.cfg.device
        # held-out edges when available: gates must not fire on training data
        scored = self.eval_edges if self.eval_edges else self.edges
        probe = torch.randint(0, len(scored), (min(n_probe, len(scored)),))
        e = torch.tensor(scored, dtype=torch.long)[probe].to(device)
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
        # PCE curve: for each k, sample paths of *exactly* length k
        for k in range(1, 4):
            paths = []
            tries = 0
            while len(paths) < 32 and tries < 2000:
                tries += 1
                p = self.path_sampler.sample()
                if p is not None and len(p[1]) == k:
                    paths.append(p)
            if not paths:
                continue
            psrc = torch.tensor([p[0] for p in paths], dtype=torch.long, device=device)
            prels = torch.tensor([p[1] for p in paths], dtype=torch.long, device=device)
            pdst = torch.tensor([p[2] for p in paths], dtype=torch.long, device=device)
            chain_t = self.model.apply_chain(refined[psrc], prels)
            cos = torch.nn.functional.cosine_similarity(
                chain_t, refined[pdst], dim=-1
            )
            out[f"eval/pce_k{k}"] = float((1.0 - cos).mean().item())
        return out
