"""Diagnose the commutation matrix C_rs against its two (conflicting) teachers.

C_rs is the object the idea-12b diagnostic reads: ``C = 1`` claims the pair
commutes, so the two composed routes must AGREE.  It is supervised only by
``CurvatureLoss.supervision`` (witness positives vs "no witness" negatives),
while the operator's *actual* separation is driven by the order-dependence
teacher.  This script measures all four quantities per relation pair on one
checkpoint, so the disagreement is visible instead of inferred:

    C_rs              what the matrix claims
    witness pos/neg   the C supervision's own labels
    order rate        fraction of graph examples that are order-dependent
    sep               the operator's measured separation (the truth 12b reads)

Usage:
    PYTHONPATH=. python3 scripts/diag_c_rs.py --run S8_20261027_212926
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from relgat_projector.core.sampler.quadruples import QuadrupleSampler
from relgat_projector.handlers.wiazka_checkpoint import (
    load_checkpoint, load_commutation)
from relgat_projector.handlers.wiazka_dataset import load_graph_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under --runs-dir")
    ap.add_argument("--runs-dir", default="wiazka-models")
    ap.add_argument("--nodes", default="data/same-embedder/lexical_units_embedding.pickle")
    ap.add_argument("--rels", default="data/same-embedder/relation_to_idx.json")
    ap.add_argument("--edges", default="data/same-embedder/relations_triplets.json")
    ap.add_argument("--top", type=int, default=20, help="rows to print")
    ap.add_argument("--n-vectors", type=int, default=4096)
    args = ap.parse_args()

    ckpt = os.path.join(args.runs_dir, args.run, "best")
    model, rel2idx, _spec, _meta = load_checkpoint(ckpt, device="cpu")
    C = load_commutation(ckpt)
    model.eval()
    idx2rel = {i: n for n, i in rel2idx.items()}

    node2emb, _r2i, edges_raw = load_graph_dataset(args.nodes, args.rels, args.edges)
    node_ids = sorted(node2emb.keys())
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    edges = [(id2idx[s], id2idx[d], rel2idx[r]) for s, d, r in edges_raw]
    num_nodes, num_rel = len(node_ids), len(rel2idx)

    out_adj = {i: [] for i in range(num_nodes)}
    in_adj = {i: [] for i in range(num_nodes)}
    for s, d, r in edges:
        out_adj[s].append((d, r))
        in_adj[d].append((s, r))

    sampler = QuadrupleSampler(
        out_adj=out_adj, in_adj=in_adj, num_nodes=num_nodes, num_rel=num_rel,
        rng=random.Random(42))

    # teacher 1: the C supervision's own labels
    pos, neg = sampler.commutation_witnesses(min_support=3)
    witness = {}
    for pair in pos:
        witness[pair] = 1
    for pair in neg:
        witness.setdefault(pair, 0)
    n_witness_pos = len(pos)

    # teacher 2: the order-dependence labels that actually move the operator
    order_pairs, order_rows = sampler.order_dependence_examples()
    stats = {}
    for a, r, s, label in order_rows:
        key = (min(r, s), max(r, s))
        bucket = stats.setdefault(key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(label)

    # the truth 12b reads: separation of the two composed orders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.no_grad():
        x = model.node_emb_fixed[:args.n_vectors].to(device)
    from relgat_projector.core.diagnostics.consistency import CheckpointTransport
    transport = CheckpointTransport(model, rel2idx, device=device)
    xp = transport.prepare(x)

    names = list(transport.relations)
    rows = []
    with torch.no_grad():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                u = transport.apply_chain(xp, [b, a])
                w = transport.apply_chain(xp, [a, b])
                sep = float(1.0 - torch.nn.functional.cosine_similarity(u, w, dim=-1).mean())
                ia, ib = rel2idx[a], rel2idx[b]
                key = (min(ia, ib), max(ia, ib))
                n, dep = stats.get(key, (0, 0))
                rows.append({
                    "pair": f"{a}|{b}",
                    "C": float(C[ia, ib]),
                    "witness": witness.get(key, witness.get((ib, ia), None)),
                    "n_order": n,
                    "order_rate": dep / n if n else float("nan"),
                    "sep": sep,
                })

    rows.sort(key=lambda r: -r["sep"])
    print(f"\n=== {args.run}: C_rs vs its two teachers ({len(rows)} pairs) ===")
    print(f"witness positives (C supervision): {n_witness_pos}; "
          f"pairs with order labels: {len(stats)}")
    hdr = f"{'pair':<52}{'C':>7}{'wit':>5}{'n_ord':>7}{'ord_rate':>9}{'sep':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows[:args.top]:
        w = "-" if r["witness"] is None else r["witness"]
        print(f"{r['pair']:<52}{r['C']:>7.3f}{str(w):>5}{r['n_order']:>7}"
              f"{r['order_rate']:>9.3f}{r['sep']:>8.4f}")

    have = [r for r in rows if r["n_order"] > 0]
    if have:
        sep = np.array([r["sep"] for r in have])
        c = np.array([r["C"] for r in have])
        rate = np.array([r["order_rate"] for r in have])
        print(f"\n--- on the {len(have)} pairs with order evidence ---")
        print(f"spearman(sep, C)        = {spearman(sep, c):+.4f}")
        print(f"spearman(sep, order_rate)= {spearman(sep, rate):+.4f}")
        print(f"spearman(C, order_rate) = {spearman(c, rate):+.4f}   <-- C tracks the graph?")
        agree = float(np.mean((c > 0.5) == (rate > 0.5)))
        print(f"agreement of C with the order verdict: {agree:.1%}")
        inverted = [r for r in have if (r["C"] > 0.5) != (r["order_rate"] > 0.5)]
        print(f"inverted pairs: {len(inverted)} / {len(have)}")
        for r in sorted(inverted, key=lambda r: -r["sep"])[:args.top]:
            print(f"  {r['pair']:<50} C={r['C']:.2f} order_rate={r['order_rate']:.2f} "
                  f"sep={r['sep']:.3f}")


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    main()
