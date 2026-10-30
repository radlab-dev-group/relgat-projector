"""
Operator-level diagnostics of a WIĄZKA checkpoint: ideas 03, 05, 06, 07.

The four ideas in this file share one property: they are claims about the
*transport*, not about documents, so they run on the checkpoint's own node
matrix and need no embedder, no fixture and no annotation.  Each section is
therefore reported next to the floor the idea document names:

* **03** -- the operator as an augmentation: ``ISOT`` (gold 0), the dial
  curve, and the ``C_rs``-generated hard negatives against a *random* family
  of the same size (the control the document demands: if a random family
  separates as well, the negatives are not meaningful).
* **05** -- holonomy as a ripple probe: the coverage gate, then ``dHD`` of a
  rank-one edit of the **connection** aimed at an entity, scored near vs far.
  The zero edit (same code, magnitude 0) is the noise floor.
* **06** -- reversible pseudonymisation: the privacy--usefulness grid of
  ``T_r(lambda)`` against a random-key control and a gaussian-noise floor,
  with the key check (``cce``) that decides whether the scheme is reversible
  at all.
* **07** -- the relational fingerprint: ``D_Theta`` (basis-invariant), the
  self-diff of the same model as the floor, and the per-relation probe table
  a CI gate would alarm on.

Usage::

    python -m relgat_projector_apps.diagnostics.operator_report --best
    python -m relgat_projector_apps.diagnostics.operator_report \\
        --ckpt wiazka-models/S14_.../best --section 03 06 --json /tmp/op.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from relgat_projector.core.diagnostics.augmentation import augmentation_report
from relgat_projector.core.diagnostics.consistency import (
    CheckpointTransport,
    REL_ANT,
    REL_SYN,
)
from relgat_projector.core.diagnostics.fingerprint import (
    fingerprint,
    fingerprint_delta,
    probe_accuracy_per_relation,
    theta_angle_matrix,
)
from relgat_projector.core.diagnostics.pseudonymization import (
    cce,
    k_anonymity,
    linkage_auc,
    retrieval_mrr,
)
from relgat_projector.core.diagnostics.pseudonymization import anonymity_radius
from relgat_projector.core.diagnostics.ripple import (
    adjacency,
    connection_edit,
    concentration_auc,
    coverage,
    incident_relations,
    loop_pairs,
    measure_ripple,
)

DEFAULT_RUNS_DIR = "wiazka-models"
SECTIONS = ("03", "05", "06", "07")


# ------------------------------------------------------------------------ data
def load_graph(model, rel2idx: Dict[str, int], device: str):
    """The checkpoint's own graph, as the diagnostics see it."""
    rel_names = [n for n, _ in sorted(rel2idx.items(), key=lambda kv: int(kv[1]))]
    edge_index = model.edge_index.to(device)
    edge_type = model.edge_type.to(device)
    vectors = model.node_emb_fixed.to(device)
    return rel_names, edge_index, edge_type, vectors


def sample_rows(n_nodes: int, n: int, min_degree: int,
                edge_index: torch.Tensor, seed: int) -> List[int]:
    """Deterministic sample of node rows, optionally restricted by degree."""
    rng = np.random.default_rng(seed)
    if min_degree <= 0:
        return rng.choice(n_nodes, size=min(n, n_nodes), replace=False).tolist()
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    deg = np.bincount(np.concatenate([src, dst]), minlength=n_nodes)
    ok = np.flatnonzero(deg >= min_degree)
    return rng.choice(ok, size=min(n, len(ok)), replace=False).tolist()


# ----------------------------------------------------------------- section 03
def run_augmentation(transport, vectors: torch.Tensor, commutation,
                     rows: List[int], n_hard: int, seed: int) -> Dict:
    x = vectors[torch.as_tensor(rows, device=vectors.device)]
    idx = {n: i for i, n in enumerate(transport.relations)}
    if REL_SYN not in idx or REL_ANT not in idx:
        return {"skipped": f"needs {REL_SYN!r} and {REL_ANT!r} in the checkpoint"}

    # hard family: the relations C_rs says do not commute with the negative;
    # control: the same number of relations drawn at random, which is the
    # document's control -- separation a random family also achieves is not
    # a property of the ontology
    ant = idx[REL_ANT]
    order = np.argsort(np.asarray(commutation)[ant])
    hard = [transport.relations[int(r)] for r in order if int(r) != ant][:n_hard]
    rng = np.random.default_rng(seed)
    control = list(rng.choice(
        [n for n in transport.relations if n != REL_ANT],
        size=n_hard, replace=False))
    out = augmentation_report(transport, x, commutation, REL_SYN, REL_ANT,
                              hard, control)
    out["hard_family"] = hard
    out["control_family"] = sorted(control)
    return out


# ----------------------------------------------------------------- section 05
def run_ripple(transport, model, vectors, edge_index, edge_type, rel_names,
               rows: List[int], edits: int, magnitude: float,
               far_size: int, seed: int) -> Dict:
    cov = coverage(vectors, edge_index, edge_type, rel_names, sample=rows)
    inc = incident_relations(edge_index, edge_type, rel_names, rows)
    usable = [r for r, rels in inc.items() if len(loop_pairs(rels)) >= 1]
    if len(usable) < 4:
        return {"coverage": cov, "skipped": "not enough entities with loops"}

    adj = adjacency(edge_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(usable)
    # every probe has to be an entity with a loop of its own: an entity with
    # one relation has no HD (not zero -- undefined), and mixing those in is
    # what made the zero edit score 0.35 instead of the 0.5 it must be
    usable_set = set(usable)
    near_all: List[float] = []
    far_all: List[float] = []
    zero_near: List[float] = []
    zero_far: List[float] = []
    device = vectors.device
    for e in usable[:edits]:
        near_rows = [e] + [n for n in adj.get(e, [])
                           if n in usable_set and n != e][:8]
        pool = np.array([r for r in usable_set if r not in set(near_rows)])
        far_rows = rng.choice(pool, size=min(far_size, len(pool)),
                              replace=False).tolist()
        probes = []
        for r in near_rows + far_rows:
            probes.append((int(r), loop_pairs(inc[r])))
        near_n = len(near_rows)

        before = measure_ripple(transport, vectors, probes)
        # the connection is a function of the *charted* vector: ctx_W reads
        # the 256-d operator space, not the 768-d retriever space
        with torch.no_grad():
            edited_vector = transport.prepare(vectors[e:e + 1].float())
        gen = torch.Generator(device=device).manual_seed(seed)
        with connection_edit(model, edited_vector, magnitude, generator=gen):
            after = measure_ripple(transport, vectors, probes)
        delta = after - before
        near_all.extend(delta[:near_n].tolist())
        far_all.extend(delta[near_n:].tolist())

        zero_before = measure_ripple(transport, vectors, probes)
        with connection_edit(model, edited_vector, 0.0):
            zero_after = measure_ripple(transport, vectors, probes)
        zd = zero_after - zero_before
        zero_near.extend(zd[:near_n].tolist())
        zero_far.extend(zd[near_n:].tolist())

    return {
        "coverage": cov,
        "edits": min(edits, len(usable)),
        "magnitude": magnitude,
        "n_near": len(near_all),
        "n_far": len(far_all),
        "near_mean": float(np.mean(near_all)),
        "far_mean": float(np.mean(far_all)),
        "dHD_mean": float(np.mean(near_all) - np.mean(far_all)),
        "concentration_auc": concentration_auc(near_all, far_all),
        "zero_edit_concentration_auc": concentration_auc(zero_near, zero_far),
        "zero_edit_dHD_mean": float(np.mean(zero_near + zero_far)),
    }


# ----------------------------------------------------------------- section 06
def run_protection(transport, vectors: torch.Tensor, rows: List[int],
                   lambdas: Sequence[float], seed: int) -> Dict:
    """One privacy--usefulness grid per arm: key, random key, noise floor."""
    corpus = _unit(transport.prepare(
        vectors[torch.as_tensor(rows, device=vectors.device)].float()))
    n = int(corpus.shape[0])
    targets = list(range(n))

    rng = np.random.default_rng(seed)
    key_rotation = _haar_deterministic(corpus.shape[-1], rng, corpus.device)
    eps = anonymity_radius(corpus, share=0.01)

    def semantic(v, lam):
        if lam == 0.0:
            return v
        return transport.apply_dial(
            v, REL_SYN, torch.full((v.shape[0],), lam, device=v.device))

    def invert_semantic(v, lam):
        if lam == 0.0:
            return v
        return transport.inverse_dial(
            v, REL_SYN, torch.full((v.shape[0],), lam, device=v.device))

    def random_key(v, lam):
        # a rotation with the same norm profile and no semantics: whatever
        # this arm gets, the semantic key must not merely match but explain
        if lam == 0.0:
            return v
        return (1.0 - lam) * v + lam * (v @ key_rotation.t())

    # gaussian floor matched to the *displacement* of the semantic step, so
    # the two arms are compared at equal distortion, not equal lambda
    step = float((semantic(corpus, 1.0) - corpus).norm(dim=-1).mean())

    def noise(v, lam):
        g = torch.randn(v.shape, device=v.device, generator=torch.Generator(
            device=v.device).manual_seed(seed))
        return v + float(lam) * step * g / float(g.norm(dim=-1, keepdim=True).mean())

    def grid(protect, invertible=None) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for lam in lambdas:
            pub = protect(corpus, float(lam))
            row = k_anonymity(corpus, pub, epsilon=eps)
            row["linkage_auc"] = linkage_auc(corpus, pub, top_k=64)
            row["mrr"] = retrieval_mrr(corpus, pub, corpus, targets=targets, k=10)
            if invertible is not None:
                back = invertible(pub, float(lam))
                row["cce"] = float((back - corpus).abs().max())
            out[str(lam)] = row
        return out

    return {
        "n_vectors": n,
        "step_norm": step,
        "anonymity_epsilon": eps,
        "key_semantic": grid(semantic, invert_semantic),
        "key_random": grid(random_key),
        "noise_gaussian": grid(noise),
    }


def _haar_deterministic(dim: int, rng: np.random.Generator,
                        device: torch.device) -> torch.Tensor:
    """A fixed random orthogonal matrix, seeded (the random-key control)."""
    g = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(g)
    return torch.as_tensor(q, dtype=torch.float32, device=device)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


# ----------------------------------------------------------------- section 07
def run_fingerprint(model, rel2idx, commutation, vectors, edge_index,
                    edge_type, rows: List[int], n_per_rel: int,
                    baseline: Optional[str], device: str) -> Dict:
    rel_names = [n for n, _ in sorted(rel2idx.items(), key=lambda kv: int(kv[1]))]
    fp_new = fingerprint(model, commutation)

    # floor: the same model measured twice.  Anything a gate alarms on has to
    # sit above this, and it is 0 only because the map is deterministic --
    # which is itself worth printing next to the regression numbers
    out: Dict[str, object] = {
        "self_diff_d_theta": float(np.linalg.norm(
            fp_new["theta_angles"] - fingerprint(model, commutation)["theta_angles"])),
        "theta_angle_mean": float(
            theta_angle_matrix(model.operator.relation_matrices())[
                np.triu_indices(len(rel_names), 1)].mean()),
        "log_scale_mean": float(np.mean(fp_new["log_scale"])),
    }

    # probe set: gold edges, sampled per relation, scored exactly as the
    # trainer scores rel_acc (head + train-frequency prior)
    counts = torch.bincount(edge_type, minlength=len(rel_names)).float()
    pairs = _probe_pairs(edge_index, edge_type, n_per_rel, len(rel_names))
    with torch.no_grad():
        refined = model.refine()
    acc = probe_accuracy_per_relation(
        model, pairs, mode="prior", train_counts=counts, node_repr=refined)
    macro = [(c / t) for (c, t) in acc.values() if t > 0]
    out["probe_pairs"] = len(pairs)
    out["probe_macro_accuracy"] = float(np.mean(macro)) if macro else float("nan")
    # honest label: the probe edges are the checkpoint's own training edges.
    # For a fingerprint that is the right choice -- it has to be a stable
    # property of *this* model, and a held-out split would make the diff of
    # two checkpoints a diff of two generalisation errors -- but it is a fit
    # measure, not a generalisation estimate, and it must not be read as the
    # 0.476 macro the trainer reports on edges the model never saw.
    out["probe_is_in_sample"] = True
    out["trainer_macro_accuracy_held_out"] = None
    out["probe_per_relation"] = {
        rel_names[r]: {"accuracy": round(c / t, 4), "n": t}
        for r, (c, t) in sorted(acc.items()) if t > 0}

    # injected regressions: does the fingerprint see what MMD would miss?
    out["injected"] = _injected_deltas(model, commutation, fp_new, device)

    if baseline:
        from relgat_projector.handlers.wiazka_checkpoint import (
            load_checkpoint, load_commutation)
        base_model, base_rel2idx, _, _ = load_checkpoint(baseline, device=device)
        base_C = load_commutation(baseline)
        fp_old = fingerprint(base_model, base_C)
        delta = fingerprint_delta(fp_old, fp_new)
        base_acc = probe_accuracy_per_relation(
            base_model, pairs, mode="prior",
            train_counts=torch.bincount(base_model.edge_type.to(device),
                                        minlength=len(rel_names)).float(),
            node_repr=base_model.refine())
        delta.update(fingerprint_delta(fp_old, fp_new, base_acc, acc))
        out["baseline"] = {"ckpt": baseline, **delta}
    return out


def _probe_pairs(edge_index, edge_type, n_per_rel: int,
                 num_rel: int) -> List[Tuple[int, int, int]]:
    """Up to ``n_per_rel`` gold ``(a, b, rel)`` probes per relation."""
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    types = edge_type.cpu().numpy()
    pairs: List[Tuple[int, int, int]] = []
    for r in range(num_rel):
        take = np.flatnonzero(types == r)[:n_per_rel]
        pairs.extend((int(src[i]), int(dst[i]), r) for i in take)
    return pairs


def _injected_deltas(model, commutation, fp_new, device: str) -> Dict[str, float]:
    """``D_Theta`` moved by a known regression vs. the untouched fingerprint."""
    op = model.operator
    out: Dict[str, float] = {}
    with torch.no_grad():
        original = op.theta.detach().clone()
    try:
        for scale in (0.01, 0.05, 0.2):
            with torch.no_grad():
                op.theta.mul_(1.0 + scale)
            moved = theta_angle_matrix(op.relation_matrices())
            out[f"d_theta_at_scale_{scale}"] = float(np.linalg.norm(
                _upper(moved - fp_new["theta_angles"])))
            with torch.no_grad():
                op.theta.copy_(original)
        with torch.no_grad():
            op.theta.copy_(op.theta.detach().half().float())
        moved = theta_angle_matrix(op.relation_matrices())
        out["d_theta_at_fp16"] = float(np.linalg.norm(_upper(
            moved - fp_new["theta_angles"])))
    finally:
        # an interrupted run must not leave a perturbed checkpoint in memory:
        # every later section reads the same object
        with torch.no_grad():
            op.theta.copy_(original)
    return out

def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], k=1)]


# --------------------------------------------------------------------- report
def print_report(results: Dict) -> None:
    block = results.get("03")
    if block:
        print("\n=== 03 operator as augmentation [ideas/03] ===")
        print(f"ISOT syn={block['isot_pos']:.6f} ant={block['isot_neg']:.6f}"
              "   (gold 0.000000: an augmented pair must keep every cosine)")
        print(f"dial cos(v, T_syn(l)v): "
              + " ".join(f"l={k.split('_')[-1]}:{block[k]:.3f}"
                         for k in block if k.startswith("pos_self_cos_")))
        print(f"antonym step: self dist={block['neg_self_dist']:.3f} vs "
              f"random vector {block['rand_self_dist']:.3f}"
              "   (hard negative = closer than random, same topic, other meaning)")
        print(f"order separation of the C_rs family = {block['hard_sep_mean']:.4f}"
              f" (mean C={block['hard_c_mean']:.3f}) vs random family "
              f"{block['control_sep_mean']:.4f} (mean C={block['control_c_mean']:.3f})"
              f" -> lift {block['sep_lift_over_control']:+.4f}")

    block = results.get("05")
    if block:
        print("\n=== 05 holonomy as a ripple probe [ideas/05] ===")
        if "skipped" in block:
            print(f"skipped: {block['skipped']}")
        else:
            cov = block["coverage"]
            print(f"coverage: {cov['coverage']:.1%} of {cov['n_entities']} entities "
                  f"have a loop to measure (median {cov['pairs_median']:.0f} pairs); "
                  "the idea's gate is ~30%")
            print(f"dHD near={block['near_mean']:.6f} far={block['far_mean']:.6f} "
                  f"-> {block['dHD_mean']:+.6f}; concentration AUC = "
                  f"{block['concentration_auc']:.3f}")
            print(f"zero edit: concentration AUC = "
                  f"{block['zero_edit_concentration_auc']:.3f}, dHD = "
                  f"{block['zero_edit_dHD_mean']:+.6f}   <- the floor of the procedure")

    block = results.get("06")
    if block:
        print("\n=== 06 reversible pseudonymisation [ideas/06] ===")
        print(f"corpus={block['n_vectors']} chart vectors, semantic step norm="
              f"{block['step_norm']:.3f}")
        header = (f"{'arm':<16}{'lambda':>7}{'linkage AUC':>13}{'k-anon med':>12}"
                  f"{'MRR':>8}{'cce':>10}")
        print(header)
        print("-" * (len(header) - 2))
        for arm in ("key_semantic", "key_random", "noise_gaussian"):
            for lam, row in block[arm].items():
                cce_v = row.get("cce", float("nan"))
                print(f"{arm:<16}{lam:>7}{row['linkage_auc']:>13.3f}"
                      f"{row['k_anon_median']:>12.0f}{row['mrr']:>8.3f}"
                      f"{cce_v:>10.1e}")
        print("  linkage AUC 1.0 = the attack identifies every vector, 0.5 = blind;"
              "\n  cce is the key check: a reversible scheme must be < 1e-4.")

    block = results.get("07")
    if block:
        print("\n=== 07 relational fingerprint [ideas/07] ===")
        print(f"mean angle between relations = {block['theta_angle_mean']:.3f} rad;"
              f" self-diff of the fingerprint = {block['self_diff_d_theta']:.1e}"
              "   <- the floor a CI gate sits on")
        print(f"probe: {block['probe_pairs']} gold edges, macro naming accuracy = "
              f"{block['probe_macro_accuracy']:.3f}  (IN SAMPLE -- the model's own "
              "training edges: a fingerprint, not a generalisation estimate)")
        print("injected regressions (D_Theta moved):")
        for k, v in block.get("injected", {}).items():
            print(f"  {k:<26}{v:>10.4f}")
        base = block.get("baseline")
        if base:
            print(f"vs baseline {base['ckpt']}: d_Dtheta={base['d_theta_angles']:.4f}"
                  f" d_C={base.get('d_commutation', float('nan')):.4f}")
            reg = base.get("regressed_relations") or []
            print(f"  probe accuracy dropped on {len(reg)} relations"
                  + (": " + ", ".join(f"r{r} {a:.2f}->{b:.2f}"
                                      for r, a, b in reg[:6]) if reg else ""))


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operator-level diagnostics of ideas 03/05/06/07")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--best", action="store_true",
                        help="best checkpoint under --runs-dir")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--baseline", default=None,
                        help="checkpoint to diff the fingerprint against (07)")
    parser.add_argument("--section", nargs="+", default=list(SECTIONS),
                        choices=list(SECTIONS))
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vectors", type=int, default=4096,
                        help="node rows sampled per section")
    parser.add_argument("--hard-relations", type=int, default=4,
                        help="size of the C_rs hard-negative family (03)")
    parser.add_argument("--edits", type=int, default=8,
                        help="connection edits to measure (05)")
    parser.add_argument("--magnitude", type=float, default=0.5,
                        help="strength of the injected connection edit (05)")
    parser.add_argument("--n-per-rel", type=int, default=24,
                        help="gold probe pairs per relation (07)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)
    from relgat_projector.handlers.wiazka_checkpoint import (
        load_checkpoint, load_commutation)
    from relgat_projector_apps.diagnostics.consistency_report import (
        find_best_checkpoint)

    ckpt = args.ckpt or (find_best_checkpoint(args.runs_dir) if args.best else None)
    if not ckpt:
        raise SystemExit("pass --ckpt or --best")
    model, rel2idx, _, _ = load_checkpoint(ckpt, device=args.device)
    commutation = load_commutation(ckpt)
    print(f"[model] {ckpt}: {len(rel2idx)} relations, "
          f"base_dim={model.base_dim}, C={commutation.shape}")

    rel_names, edge_index, edge_type, vectors = load_graph(
        model, rel2idx, args.device)
    transport = CheckpointTransport(model, rel2idx, device=args.device)
    rows = sample_rows(int(vectors.shape[0]), args.vectors, 1, edge_index, args.seed)
    print(f"[graph] {vectors.shape[0]} nodes, {edge_index.shape[1]} edges, "
          f"{len(rows)} sampled")

    results: Dict = {}
    if "03" in args.section:
        results["03"] = run_augmentation(transport, vectors, commutation,
                                         rows, args.hard_relations, args.seed)
    if "05" in args.section:
        results["05"] = run_ripple(transport, model, vectors, edge_index,
                                   edge_type, rel_names, rows, args.edits,
                                   args.magnitude, 24, args.seed)
    if "06" in args.section:
        results["06"] = run_protection(
            transport, vectors,
            sample_rows(int(vectors.shape[0]), 2048, 0, edge_index, args.seed),
            (0.0, 0.25, 0.5, 0.75, 1.0), args.seed)
    if "07" in args.section:
        results["07"] = run_fingerprint(model, rel2idx, commutation, vectors,
                                        edge_index, edge_type, rows,
                                        args.n_per_rel, args.baseline,
                                        args.device)

    print_report(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
