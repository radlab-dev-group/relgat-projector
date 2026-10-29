"""
Relational-clustering diagnostics of a WIĄZKA checkpoint (docs/IDEAS/01).

Groups the fixture's stance-items three ways and scores every grouping with two
numbers, because they answer different questions:

* **ARI / NMI** against the gold ``topic x stance`` clustering -- did the method
  find the clusters at all;
* **contradiction purity (CP)** -- how many of the clusters it produced hide a
  contradicting pair inside.  ARI cannot see this failure: a method can split
  the topics perfectly and still fuse "the battery lasts two days" with "the
  battery dies in hours" into one cluster, which is the exact defect idea 01 is
  about.

Methods measured, in the order the report prints them:

* ``cosine-kmeans`` -- the metric baseline, **handed the gold number of
  clusters**;
* ``cosine-dedup`` -- connected components of ``cos >= tau`` (SemDeDup-style),
  the no-``k`` baseline a practitioner actually reaches for;
* ``antonym-baseline`` -- embedder-free: the commutative translation transport
  of ideas/12 types the pairs, then the same signed clustering.  This is the
  floor -- no learned operator, only the geometry of the seed lexicon;
* ``decoder`` / ``transport`` -- the checkpoint's relation head, and the
  head-free transport fidelity, each typed into a signed graph and clustered
  with **no k supplied**.

The go/no-go test of the idea document (section 8) is printed first: the AUC of
the confidence separating the ``ant`` pairs from the ``syn`` pairs.  Below ~0.6
the family of ideas 1/2/4/12a needs sentence-level training before any of it can
work, and that is worth knowing before building on top of it.

Usage::

    python -m relgat_projector_apps.diagnostics.clustering_report --best
    python -m relgat_projector_apps.diagnostics.clustering_report \\
        --ckpt wiazka-models/S11_20261028/best --langs pl --json /tmp/c1.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from relgat_projector.core.diagnostics import (
    AdditiveTransport,
    AntonymBaselineTyper,
    DecoderTyper,
    TransportTyper,
    candidate_pairs,
    clustering_objective,
    cosine_kmeans,
    cosine_matrix,
    cosine_threshold_components,
    evaluate_partition,
    gold_pair_index,
    signed_greedy_clustering,
    signed_weights,
    typing_auc,
)
from relgat_projector_apps.diagnostics.consistency_report import (
    DEFAULT_RUNS_DIR,
    Embedder,
    find_best_checkpoint,
    read_jsonl,
)
from relgat_projector_apps.diagnostics.make_consistency_fixture import LEXICON

DEFAULT_EMBEDDER = "ipipan/silver-retriever-base-v1.1"

#: relation names typed as attraction / repulsion, as the Słowosieć exports them
ATTRACTING = ("synonimia", "fuzzynimia", "podobieństwo", "strong equivalence",
              "weak equivalence", "regular equivalence")
REPELLING = ("antonimia właściwa", "antonimia komplementarna")


# ----------------------------------------------------------------------- input
def load_fixture(fixture_dir: str) -> Tuple[List[Dict], List[Dict]]:
    items = read_jsonl(os.path.join(fixture_dir, "stances_pl_en.jsonl"))
    pairs = read_jsonl(os.path.join(fixture_dir, "pairs_pl_en.jsonl"))
    if not items:
        raise SystemExit(f"no items in {fixture_dir}/stances_pl_en.jsonl "
                         "(run relgat_projector_apps.diagnostics."
                         "make_clustering_fixture first)")
    return items, pairs


def texts_of(items: List[Dict], lang: str) -> List[str]:
    return [it["text"][lang] for it in items]


# ------------------------------------------------------------------ evaluation
def score(items: List[Dict], labels: Sequence[int], method: str) -> Dict:
    """One row of the table: gold agreement plus contradiction purity."""
    report = evaluate_partition(items, labels, method)
    return report.as_row()


def typed_clustering(items: List[Dict], vectors: np.ndarray, typer,
                     ant_ids: Sequence[int], syn_ids: Sequence[int], lang: str,
                     tau_attr: float, tau_appl: float, top_k: int,
                     min_cos: float, top_fraction: float, method: str,
                     notes: Dict) -> Dict:
    """Candidate pairs -> typed signed graph -> correlation clustering, no ``k``.

    The go/no-go AUC of the same typing is recorded in ``notes`` next to the
    clustering row, so a run can never show a purity number without the
    pair-level evidence that produced it.
    """
    sim = cosine_matrix(vectors)
    rows, cols = candidate_pairs(sim, top_k=top_k, min_cos=min_cos)
    a = torch.as_tensor(vectors[rows], dtype=torch.float32)
    b = torch.as_tensor(vectors[cols], dtype=torch.float32)
    rel_id, conf, appl = typer.type_pairs(a, b)

    gold = gold_pair_index(items)
    pair_list = list(zip(rows.tolist(), cols.tolist()))
    # the go/no-go question of the idea document: does the confidence alone rank
    # the contradicting pairs above the synonymous ones?
    notes[f"go_no_go_{method}_{lang}"] = typing_auc(
        np.abs(conf), pair_list, gold["ant"], gold["syn"])
    # ... and the same evidence read the way the graph reads it: a repelling
    # relation must score the pair *below* an attracting one
    signed = np.where(np.isin(rel_id, list(ant_ids)), -conf, conf)
    notes[f"go_no_go_signed_{method}_{lang}"] = typing_auc(
        signed, pair_list, gold["syn"], gold["ant"])

    s_rows, s_cols, w = signed_weights(
        rows, cols, rel_id, conf, appl,
        ant_ids=list(ant_ids), syn_ids=list(syn_ids),
        tau_attr=tau_attr, tau_appl=tau_appl, top_fraction=top_fraction)
    notes[f"edges_{method}_{lang}"] = float(len(w))
    notes[f"edges_repelling_{method}_{lang}"] = float((w < 0).sum())
    notes[f"edges_attracting_{method}_{lang}"] = float((w > 0).sum())
    labels = signed_greedy_clustering(len(items), s_rows, s_cols, w)
    row = score(items, labels, method)
    row.update(clustering_objective(len(items), s_rows, s_cols, w, labels))
    return row


def ids_of(names: Sequence[str], rel_ids: Optional[Dict[str, int]]) -> List[int]:
    """Relation ids of ``names``; for the fixture's own 0/1 typer pass ``None``."""
    if rel_ids is None:
        return []
    return [rel_ids[n] for n in names if n in rel_ids]


# -------------------------------------------------------------------- printing
def print_report(results: Dict) -> None:
    go = results.get("go_no_go") or {}
    if go:
        print("\n=== 01a go/no-go: does the typed evidence separate ant from syn? ===")
        for key, value in go.items():
            print(f"  {key:34s} AUC = {value:.3f}"
                  + ("   (0.5 = no signal, >0.8 = idea 01 stands)"
                     if key.endswith("_abs") or "_signed" not in key else ""))
    for lang, rows in (results.get("lang") or {}).items():
        print(f"\n=== 01b relational clustering [{lang}] ===")
        header = (f"{'method':<20}{'clusters':>9}{'gold':>6}{'ARI':>8}{'NMI':>8}"
                  f"{'CP':>8}{'hidden':>8}{'singletons':>12}{'':>12}")
        print(header)
        print("-" * (len(header) - 12))
        gold_k = results.get("gold_clusters", {}).get(lang, "")
        n_items = results.get("n_items", "")
        for row in rows:
            # a partition that merged nothing scores CP 1.0 without knowing
            # anything, and ARI 0 says so -- flag it instead of letting the
            # purity column be read alone
            flag = "  (merged nothing)" if row["n_clusters"] >= n_items else ""
            print(f"{row['method']:<20}{row['n_clusters']:>9}{gold_k:>6}"
                  f"{row['ari']:>8.3f}{row['nmi']:>8.3f}{row['cp']:>8.3f}"
                  f"{row['hidden']:>8}{row['singletons']:>12.2f}{flag}")
        print("  CP = share of clusters with no contradicting pair inside "
              "(1.0 = none merged); hidden = ant pairs swallowed.")
        print("  Read CP together with ARI: CP alone rewards not clustering.")
        edges = {k: v for k, v in (results.get("notes") or {}).items()
                 if k.startswith("edges_") and k.endswith(lang)}
        for key, value in edges.items():
            print(f"  {key} = {value:.0f}")
        if not edges:
            print("  (no edge counters for this language)")


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relational-clustering diagnostics (ideas/01)")
    parser.add_argument("--fixture-dir", default="data/clustering")
    parser.add_argument("--lexicon-dir", default="data/consistency",
                        help="where the ideas/12 lexicon lives (it defines the "
                             "embedder-free baseline transport)")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--best", action="store_true",
                        help="use the best checkpoint under --runs-dir")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--langs", nargs="+", default=["pl", "en"],
                        choices=["pl", "en"])
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=12,
                        help="candidate neighbours per item (the ANN pruning)")
    parser.add_argument("--min-cos", type=float, default=0.0)
    parser.add_argument("--top-fraction", type=float, default=0.2,
                        help="share of the typed candidates that becomes an edge, "
                             "ranked by |confidence|: a quantile, because the "
                             "sigmoid head and the transport score are not on one "
                             "scale (this index has mean pairwise cos 0.946, so an "
                             "absolute cut is either everything or nothing)")
    parser.add_argument("--tau-attr", type=float, default=0.0,
                        help="extra absolute minimum confidence (0 = off)")
    parser.add_argument("--tau-appl", type=float, default=0.0,
                        help="extra absolute minimum applicability (0 = off)")
    parser.add_argument("--dedup-threshold", type=float, default=0.995,
                        help="cosine threshold of the dedup baseline; this space "
                             "sits at mean pairwise cos 0.946, so a 0.9 cut merges "
                             "the whole corpus into one cluster")
    parser.add_argument("--translation-strength", type=float, default=1.0)
    parser.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)
    items, pairs = load_fixture(args.fixture_dir)
    gold_k = len({it["group"] for it in items})
    print(f"[fixture] {len(items)} stance-items, {gold_k} gold clusters, "
          f"{len(pairs)} gold pairs ({args.fixture_dir})")

    model = rel2idx = None
    ckpt = args.ckpt or (find_best_checkpoint(args.runs_dir) if args.best else None)
    if ckpt:
        from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint
        model, rel2idx, _spec, _meta = load_checkpoint(ckpt, device=args.device)
        print(f"[model] {ckpt}: {len(rel2idx)} relations, "
              f"base_dim={model.base_dim}")
    else:
        print("[model] no checkpoint: baselines and the embedder-free floor only")

    embed = Embedder(args.embedder, device=args.device)
    if model is not None and embed.dim != model.base_dim:
        print(f"[warn] embedder dim {embed.dim} != checkpoint base dim "
              f"{model.base_dim}: the learned typer needs the checkpoint's own "
              "space and is skipped")
        model = None

    # the embedder-free transport of ideas/12 defines the floor of this test
    lexicon_path = os.path.join(args.lexicon_dir, "lexicon_pl_en.json")
    transports: Dict[str, AdditiveTransport] = {}
    if os.path.isfile(lexicon_path):
        for lang in args.langs:
            transports[lang] = AdditiveTransport.from_pairs(
                {rel: LEXICON[rel][lang] for rel in LEXICON}, embed.encode,
                strength=args.translation_strength)
    else:
        print(f"[baseline] {lexicon_path} not found: no embedder-free floor")

    results: Dict = {"gold_clusters": {}, "lang": {}, "notes": {}, "go_no_go": {},
                     "n_items": len(items)}
    for lang in args.langs:
        vecs = embed.encode(texts_of(items, lang)).detach().cpu().numpy()
        results["gold_clusters"][lang] = gold_k
        rows: List[Dict] = []

        # the metric baselines
        rows.append(score(items, cosine_kmeans(vecs, k=gold_k, seed=0),
                          "cosine-kmeans"))
        rows.append(score(items, cosine_threshold_components(
            vecs, args.dedup_threshold), "cosine-dedup"))

        # the embedder-free floor: typed by the translation transport only
        if lang in transports:
            floor = AntonymBaselineTyper(transports[lang])
            charted = floor.prepare(torch.as_tensor(vecs, dtype=torch.float32))
            rows.append(typed_clustering(
                items, _np(charted), floor, [1], [0], lang,
                args.tau_attr, args.tau_appl, args.top_k, args.min_cos,
                args.top_fraction, "antonym-baseline", results["notes"]))

        # the checkpoint
        if model is not None:
            device = str(next(model.parameters()).device)
            for cls in (DecoderTyper, TransportTyper):
                typer = cls(model, rel2idx, device=device)
                charted = typer.prepare(torch.as_tensor(vecs, dtype=torch.float32))
                rows.append(typed_clustering(
                    items, _np(charted), typer,
                    ids_of(REPELLING, rel2idx), ids_of(ATTRACTING, rel2idx), lang,
                    args.tau_attr, args.tau_appl, args.top_k, args.min_cos,
                    args.top_fraction, typer.name, results["notes"]))

        results["lang"][lang] = rows
        results["notes"][f"gold_k_{lang}"] = float(gold_k)
    results["go_no_go"] = {k: v for k, v in results["notes"].items()
                           if k.startswith("go_no_go")}

    print_report(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n[json] {args.json_out}")


def _np(x) -> np.ndarray:
    return x.detach().to("cpu", dtype=torch.float64).numpy()


if __name__ == "__main__":
    main()
