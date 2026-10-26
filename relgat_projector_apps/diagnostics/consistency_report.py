"""
Consistency diagnostics of a WIĄZKA checkpoint (docs/IDEAS/12-consistency-diagnostics.md).

Two label-free tests:

* **12a** -- is a document close to its own negation?  Per document: ``S``
  (mean cosine of the antonym-transported claims to the document), ``K``
  (claims are mutually synonymic rather than antonymic), ``HD`` (how badly the
  ``syn o ant`` loop fails to close around the document), plus the two cheap
  baselines the idea document asks to beat: a surface negation heuristic and
  the raw claim-to-claim cosine.
* **12b** -- does the model know that ``T_r T_s != T_s T_r``?  The order
  separation ``1 - cos(T_s T_r x, T_r T_s x)`` is measured for relation pairs
  and compared with the learned commutation matrix ``C_rs``.

The learned model is optional: ``--best`` picks the best checkpoint under
``--runs-dir``, and the embedder-free translation baseline is always reported
next to it, in Polish and in English, so a run without any checkpoint still
produces a number.

Usage::

    python -m relgat_projector_apps.diagnostics.consistency_report --best
    python -m relgat_projector_apps.diagnostics.consistency_report \\
        --ckpt wiazka-models/F4_20261026_024039/best --section 12b --json /tmp/c.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from relgat_projector.core.diagnostics import (
    AdditiveTransport,
    CheckpointTransport,
    Transport,
    document_scores,
    order_description_separation,
    order_separation,
    pair_contradiction_scores,
    relative_holonomy,
    roc_auc,
)
from relgat_projector_apps.diagnostics.make_consistency_fixture import (
    LEXICON,
    ORDER_PAIRS,
    POSITIVE_LABELS,
)

DEFAULT_EMBEDDER = "ipipan/silver-retriever-base-v1.1"
DEFAULT_RUNS_DIR = "wiazka-models"

#: 12a columns: (header, metric key, +1 when a larger value means "less coherent")
DOCUMENT_METRICS = [
    ("S", "self_negation", +1),
    ("K", "coherence_k", -1),
    ("HD", "holonomy", +1),
    ("raw", "raw_incoherence", +1),
    ("neg", "negation_baseline", +1),
]


# ----------------------------------------------------------------------- inputs
def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_best_checkpoint(runs_dir: str) -> Optional[str]:
    """``<run>/best`` of the run with the highest ``checkpoint_scores.best``.

    Every run writes its own ``report.json``, so scanning them beats keeping a
    pointer file that goes stale the moment a new run finishes.
    """
    candidates: List[tuple] = []
    if not os.path.isdir(runs_dir):
        return None
    for name in sorted(os.listdir(runs_dir)):
        report = os.path.join(runs_dir, name, "report.json")
        if not os.path.isfile(report):
            continue
        try:
            with open(report, encoding="utf-8") as f:
                scores = (json.load(f).get("checkpoint_scores") or {})
            score = float(scores.get("best", float("nan")))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if score == score:
            candidates.append((score, os.path.join(runs_dir, name, "best")))
    return max(candidates)[1] if candidates else None


def negate(text: str, lang: str) -> str:
    """Surface negation -- the cheap baseline 12a has to beat."""
    prefix = "Nieprawdą jest, że " if lang == "pl" else "It is not the case that "
    return prefix + text[0].lower() + text[1:]


class Embedder:
    """One sentence-embedding space shared by both languages of the fixture."""

    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        print(f"[embedder] loading {name} on {device}")
        self.model = SentenceTransformer(name, device=device)
        self.device = torch.device(device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            out = self.model.encode(
                missing, convert_to_tensor=True, normalize_embeddings=False,
                show_progress_bar=False).to(self.device)
            for text, vec in zip(missing, out):
                self._cache[text] = vec
        return torch.stack([self._cache[t] for t in texts])


# ------------------------------------------------------------------------- 12a
def run_documents(documents: List[Dict], embed: Embedder, lang: str,
                  transport: Transport) -> Dict:
    rows: List[Dict] = []
    for doc in documents:
        texts = [claim[lang] for claim in doc["claims"]]
        claims = embed.encode(texts)
        negated = embed.encode([negate(t, lang) for t in texts])
        scores = document_scores(claims, transport, negated=negated)
        gold = pair_contradiction_scores(claims, transport, doc["contradictions"])
        rows.append({"id": doc["id"], "label": doc["label"],
                     "n_claims": len(texts),
                     "gold_pair_auc": gold["auc"], "n_gold_pairs": gold["n_gold"],
                     **scores})

    labels = [1.0 if r["label"] in POSITIVE_LABELS else 0.0 for r in rows]
    aucs = {}
    for name, key, sign in DOCUMENT_METRICS:
        vals = [r[key] for r in rows]
        aucs[name] = float("nan") if any(v != v for v in vals) else roc_auc(labels, [sign * v for v in vals])

    def mean(subset: List[Dict], key: str) -> float:
        vals = [r[key] for r in subset if r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    coherent = [r for r in rows if r["label"] == "coherent"]
    contra = [r for r in rows if r["label"] in POSITIVE_LABELS]
    return {
        "lang": lang,
        "documents": rows,
        "auc": aucs,
        "mean_by_label": {
            "coherent": {name: mean(coherent, key) for name, key, _ in DOCUMENT_METRICS},
            "contradictory": {name: mean(contra, key) for name, key, _ in DOCUMENT_METRICS},
        },
        "gold_pair_auc_mean": float(np.nanmean([r["gold_pair_auc"] for r in rows])),
    }


# ------------------------------------------------------------------------- 12b
def run_order(tasks: List[Dict], embed: Embedder, langs: Sequence[str],
              transports: Dict[str, AdditiveTransport], model, rel2idx,
              commutation, n_vectors: int, all_pairs: bool) -> Dict:
    out: Dict[str, Dict] = {}

    # the space itself: can two readings of the same chain be told apart at all?
    for lang in langs:
        desc_a = embed.encode([t["desc_a"][lang] for t in tasks])
        desc_b = embed.encode([t["desc_b"][lang] for t in tasks])
        out[f"descriptions:{lang}"] = order_description_separation(desc_a, desc_b)

    # the embedder-free transport: commutative by construction, so this is the
    # baseline the learned operator has to beat (and a bug detector: a non-zero
    # number here means the transport is not a pure translation)
    for lang in langs:
        transport = transports[lang]
        words = [t["word"][lang] for t in tasks]
        dummy_c = np.full((len(transport.relations),) * 2, 0.5)
        report = order_separation(embed.encode(words), transport, dummy_c)
        out[f"translation:{lang}"] = {
            "n_pairs": report.n_pairs,
            "sep_median": report.sep_median,
            "sep_max": report.sep_max,
        }

    # the learned operator, on real graph vectors
    if model is None or commutation is None:
        return out
    device = str(next(model.parameters()).device)
    with torch.no_grad():
        x = model.node_emb_fixed[:n_vectors].to(device)
    checkpoint_transport = CheckpointTransport(model, rel2idx, device=device)
    wanted = {rel for pair in ORDER_PAIRS for rel in pair}
    subset = None
    if not all_pairs:
        subset = [r for r in checkpoint_transport.relations if r in wanted]
        if len(subset) < 2:
            subset = None
    report = order_separation(x, checkpoint_transport, commutation,
                              relation_subset=subset)
    out["checkpoint"] = {**report.__dict__,
                         "relations": "ALL" if subset is None else ",".join(subset)}

    # the closed-loop version of the same question: walk r then s and undo it.
    # Order separation says the two routes end elsewhere; holonomy says whether
    # the space around x is curved at all (ideas/12a, third variant).
    names = list(subset) if subset is not None else list(checkpoint_transport.relations)
    loops = {}
    with torch.no_grad():
        # ``x`` is in the base space; the loop has to be walked inside the chart
        probe = checkpoint_transport.prepare(x[:min(64, x.shape[0])])
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                loops[f"{first}|{second}"] = relative_holonomy(
                    checkpoint_transport, probe, [first, second])
    out["checkpoint"]["holonomy"] = {
        "mean": float(np.mean(list(loops.values()))),
        "max": float(np.max(list(loops.values()))),
        "worst": sorted(loops, key=loops.get, reverse=True)[:3],
        "loops": loops,
    }
    return out


# --------------------------------------------------------------------- reporting
#: label of the learned-operator 12a block inside results["lang"][lang]
LEARNED = "12a_learned"


def _print_document_report(report: Dict, lang: str, title: str) -> None:
    print(f"\n=== 12a {title} [{lang}] ===")
    header = f"{'document':<22}{'label':<24}{'S':>7}{'K':>7}{'HD':>7}{'raw':>7}{'neg':>7}{'goldAUC':>9}"
    print(header)
    print("-" * len(header))
    for row in report["documents"]:
        print(f"{row['id']:<22}{row['label']:<24}"
              f"{row['self_negation']:>7.3f}{row['coherence_k']:>7.3f}"
              f"{row['holonomy']:>7.3f}{row['raw_incoherence']:>7.3f}"
              f"{row['negation_baseline']:>7.3f}{row['gold_pair_auc']:>9.3f}")
    coherent = report["mean_by_label"]["coherent"]
    contra = report["mean_by_label"]["contradictory"]
    print(f"{'MEAN coherent':<22}{'':<24}"
          + "".join(f"{coherent[name]:>7.3f}" for name, _, _ in DOCUMENT_METRICS))
    print(f"{'MEAN contradictory':<22}{'':<24}"
          + "".join(f"{contra[name]:>7.3f}" for name, _, _ in DOCUMENT_METRICS))
    print("AUC contradictory-vs-coherent: " + "  ".join(
        f"{name}={report['auc'][name]:.3f}" for name, _, _ in DOCUMENT_METRICS))
    print(f"gold contradiction pairs: mean AUC = {report['gold_pair_auc_mean']:.3f}")


def _auc_row(report: Dict) -> Dict[str, float]:
    row = dict(report["auc"])
    row["goldPair"] = report["gold_pair_auc_mean"]
    return row


def _print_transport_comparison(lang: str, additive: Dict, learned: Dict) -> None:
    """AUC table of the same 12a metrics under both transports, plus delta."""
    columns = [name for name, _, _ in DOCUMENT_METRICS] + ["goldPair"]
    base, model_auc = _auc_row(additive), _auc_row(learned)
    print(f"\n=== 12a learned operator vs translation baseline [{lang}] ===")
    header = f"{'variant':<26}" + "".join(f"{c:>10}" for c in columns)
    print(header)
    print("-" * len(header))
    for label, values in (("translation baseline", base), ("learned operator", model_auc)):
        print(f"{label:<26}" + "".join(
            f"{values[c]:>10.3f}" if values[c] == values[c] else f"{'nan':>10}"
            for c in columns))
    delta = {c: model_auc[c] - base[c] for c in columns
             if base[c] == base[c] and model_auc[c] == model_auc[c]}
    print(f"{'learned - baseline':<26}" + "".join(
        f"{delta[c]:>+10.3f}" if c in delta else f"{'-':>10}" for c in columns))
    print("  (AUC 0.5 = no signal; >0.5 = the metric rises on self-contradicting text)")


def print_report(results: Dict) -> None:
    for lang, block in results.get("lang", {}).items():
        report = block.get("12a")
        learned = block.get(LEARNED)
        if report:
            _print_document_report(report, lang, "self-consistency [translation baseline]")
        if learned:
            _print_document_report(learned, lang, "self-consistency [learned operator]")
        if report and learned:
            _print_transport_comparison(lang, report, learned)

    block = results.get("12b") or {}
    for key, value in block.items():
        if key.startswith("descriptions:"):
            print(f"\n=== 12b order sensitivity [{key.split(':')[1]}] ===")
            print(f"description separation (embedder space) = {value['desc_separation']:.4f}"
                  "   -- how far the two readings of the same chain land apart")
        elif key.startswith("translation:"):
            print(f"translation baseline [{key.split(':')[1]}]: median order separation = "
                  f"{value['sep_median']:.6f} over {value['n_pairs']} pairs"
                  "   -- additive transport is commutative, so this is the floor")
        elif key == "checkpoint":
            print("\n=== 12b order sensitivity [learned operator] ===")
            print(f"relations={value['relations']} pairs={value['n_pairs']} "
                  f"vectors={value['n_vectors']}")
            print(f"separation min/median/max = {value['sep_min']:.4f} / "
                  f"{value['sep_median']:.4f} / {value['sep_max']:.4f}")
            print(f"C_rs range [{value['c_min']:.3f}, {value['c_max']:.3f}]: "
                  f"{value['n_c_evidence']} pairs left the 0.5 prior "
                  f"({value['c_low']} low, {value['c_high']} high)")
            print(f"spearman(sep, C) = {value['corr_sep_c']:.4f}   "
                  f"spearman(sep, 1-C) = {value['corr_sep_1mc']:.4f}")
            for name, rows in (("most order-dependent", value["strongest"]),
                               ("least order-dependent", value["weakest"])):
                print(f"  {name}: " + "; ".join(
                    f"{p['a']}|{p['b']} sep={p['sep']:.3f} C={p['c']:.2f}" for p in rows[:3]))
            holonomy = value.get("holonomy")
            if holonomy:
                print(f"closed loop r,s,r^-1,s^-1: mean defect = {holonomy['mean']:.4f}, "
                      f"max = {holonomy['max']:.4f} (0 = flat connection)")
                print("  worst loops: " + "; ".join(holonomy["worst"]))


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consistency diagnostics of a WIĄZKA checkpoint (ideas/12)")
    parser.add_argument("--fixture-dir", default="data/consistency")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint directory (a run's best/ or last/)")
    parser.add_argument("--best", action="store_true",
                        help="use the best checkpoint found under --runs-dir")
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--langs", nargs="+", default=["pl", "en"], choices=["pl", "en"])
    parser.add_argument("--section", nargs="+", default=["12a", "12b"],
                        choices=["12a", "12b"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--order-vectors", type=int, default=256,
                        help="graph vectors used to measure order separation")
    parser.add_argument("--order-all-pairs", action="store_true",
                        help="sweep every relation pair, not just the fixture relations")
    parser.add_argument("--translation-strength", type=float, default=1.0)
    parser.add_argument("--json", dest="json_out", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)
    documents = read_jsonl(os.path.join(args.fixture_dir, "documents_pl_en.jsonl"))
    tasks = read_jsonl(os.path.join(args.fixture_dir, "order_tasks_pl_en.jsonl"))
    print(f"[fixture] {len(documents)} documents, {len(tasks)} order tasks "
          f"({args.fixture_dir})")

    model = rel2idx = commutation = None
    ckpt = args.ckpt or (find_best_checkpoint(args.runs_dir) if args.best else None)
    if ckpt:
        from relgat_projector.handlers.wiazka_checkpoint import (
            load_checkpoint, load_commutation)
        model, rel2idx, _spec, _meta = load_checkpoint(ckpt, device=args.device)
        commutation = load_commutation(ckpt)
        print(f"[model] {ckpt}: {len(rel2idx)} relations, base_dim={model.base_dim}, "
              f"C={None if commutation is None else commutation.shape}")
    else:
        print("[model] no checkpoint: measuring the embedder-free baseline only")

    embed = Embedder(args.embedder, device=args.device)
    if model is not None and embed.dim != model.base_dim:
        print(f"[warn] embedder dim {embed.dim} != checkpoint base dim "
              f"{model.base_dim}: documents are scored in the embedder space, "
              f"the learned operator only in 12b on graph vectors")

    transports = {
        lang: AdditiveTransport.from_pairs(
            {rel: LEXICON[rel][lang] for rel in LEXICON}, embed.encode,
            strength=args.translation_strength)
        for lang in args.langs
    }

    # 12a on the *learned* operator: only possible when the documents are
    # embedded in the checkpoint's own base space (the same-embedder control
    # run).  Comparing a learned T_ant against a baseline in another space
    # would measure the spaces, not the method.
    learned = None
    if model is not None and "12a" in args.section:
        if embed.dim == model.base_dim:
            learned = CheckpointTransport(
                model, rel2idx, device=str(next(model.parameters()).device))
            print(f"[12a] measuring the learned operator too "
                  f"(embedder dim {embed.dim} == checkpoint base dim)")
        else:
            print(f"[12a] learned operator skipped for documents: it lives in "
                  f"{model.base_dim}-d, documents in {embed.dim}-d")

    results: Dict = {"lang": {}}
    for lang in args.langs:
        entry: Dict = {"12a": None, "12a_learned": None}
        if "12a" in args.section:
            entry["12a"] = run_documents(documents, embed, lang, transports[lang])
            if learned is not None:
                entry["12a_learned"] = run_documents(documents, embed, lang, learned)
        results["lang"][lang] = entry
    if "12b" in args.section:
        results["12b"] = run_order(tasks, embed, args.langs, transports, model,
                                   rel2idx, commutation, args.order_vectors,
                                   args.order_all_pairs)

    print_report(results)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=float)
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
