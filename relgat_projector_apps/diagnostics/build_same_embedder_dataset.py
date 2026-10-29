"""
Rebuild the node vectors of a same-embedder dataset from richer LU texts.

Why this exists.  ``data/same-embedder`` was built by embedding the bare
*lemma* of every graph node, and that is the weakest text a node can have.
Słowosieć carries a definition and usage examples for every sense, and
dropping them costs exactly the signal the operator needs:

* **54.4 % of the nodes share their lemma with another node** (116 273 distinct
  strings over 182 960 nodes), so over half the graph sits on bit-identical
  vectors and the 1-NN duplicate share is 53.7 % -- the defect the trainer's
  duplicate-mask has to patch over;
* a bare lemma and a document sentence sit in different parts of the
  embedder's input distribution, so a transport trained on lemmas has to
  cross a gap it was never supervised across.

This module recomputes the node vectors from the unit's **identity text** --
``lemma (pos) | [domain] | definition | examples`` -- composed by
``lu_identity_text`` imported from the plwordnet repo, which is the single
source of truth for the recipe: the Milvus pipeline and this builder cannot
drift apart.

The graph itself (node ids, relation mapping, relation triplets) is copied
unchanged from ``--from-dataset``, so two datasets built this way differ *only*
in the node vectors and every comparison between them is controlled.

Writes to ``--out-dir``:

* ``lexical_units_embedding.pickle`` -- ``{int node_id: float32 vector}``;
* ``node_id_to_lemma.json`` -- copied (provenance);
* ``node_id_to_text.json`` -- the exact text every node was embedded from, so
  the vectors are auditable without running the embedder again;
* ``relation_to_idx.json``, ``relations_triplets.json`` -- copied unchanged.

Usage::

    python -m relgat_projector_apps.diagnostics.build_same_embedder_dataset \\
        --from-dataset data/same-embedder \\
        --out-dir data/same-embedder-rich \\
        --plwordnet-repo ../plwordnet
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

DEFAULT_EMBEDDER = "ipipan/silver-retriever-base-v1.1"
#: default location of the NetworkX export inside a plwordnet checkout
NX_GRAPH_SUBPATH = os.path.join(
    "resources", "plwordnet_4_5", "full", "graphs", "full", "nx", "graphs")


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch missing or broken
        return False


def _import_from_plwordnet(repo: str, member: str):
    """Import ``member`` out of a plwordnet checkout added to ``sys.path``."""
    repo = os.path.abspath(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        module = __import__(
            "plwordnet_ml.embedder.generator.base_embeddings.lexical_unit",
            fromlist=[member])
        return getattr(module, member)
    except Exception as exc:  # noqa: BLE001 - the cause is the actionable part
        raise SystemExit(
            f"cannot import {member!r} from {repo!r}: {exc}\n"
            "point --plwordnet-repo at a checkout that has the LU identity-text "
            "helper (see the ``lu_identity_text`` function)") from exc


def load_texts(node_ids: List[int], plwordnet_repo: str, nx_graph_dir: str,
               max_examples: int) -> List[str]:
    """Identity text of every node id, in the order they were given."""
    units_dir = nx_graph_dir or os.path.join(plwordnet_repo, NX_GRAPH_SUBPATH)
    _import_from_plwordnet(plwordnet_repo, "lu_identity_text")
    from plwordnet_handler.base.structure.polishwordnet import PolishWordnet

    print(f"[plwordnet] loading lexical units from {units_dir}")
    plwn = PolishWordnet(nx_graph_dir=units_dir)
    plwn.connect()
    units: Dict[int, object] = {int(u.ID): u for u in plwn.get_lexical_units()}
    print(f"[plwordnet]   {len(units)} lexical units")

    missing = [i for i in node_ids if i not in units]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(node_ids)} node ids are not lexical units of "
            f"this graph (e.g. {missing[:5]}); --from-dataset and "
            "--nx-graph-dir describe different exports")

    lu_identity_text = _import_from_plwordnet(plwordnet_repo, "lu_identity_text")
    return [lu_identity_text(units[i], max_examples=max_examples)
            for i in node_ids]


def embed(texts: List[str], embedder: str, device: str,
          batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[embedder] loading {embedder} on {device}")
    model = SentenceTransformer(embedder, device=device)
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=False, convert_to_numpy=True)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"[embedder]   {len(texts)} texts, dim {dim}")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def report_geometry(name: str, vectors: np.ndarray, device: str) -> Dict:
    """Measure the space with the trainer's own profile, so the numbers match
    what a run will print as its ``base_profile``."""
    from relgat_projector.core.geometry.embedder import EmbedderProfile

    profile = EmbedderProfile.from_embeddings(
        torch.as_tensor(vectors, dtype=torch.float32, device=device))
    print(f"[geometry] {name}: pair cos {profile.pair_cos_mean:.4f} "
          f"(centred {profile.pair_cos_mean_centred:.4f}), "
          f"1-NN duplicates {profile.dup_1nn_fraction:.1%}, "
          f"effective rank {profile.effective_rank:.1f}, "
          f"EV@256 {profile.explained_variance.get('256', 0.0):.4f}")
    return {
        "pair_cos_mean": profile.pair_cos_mean,
        "pair_cos_mean_centred": profile.pair_cos_mean_centred,
        "pair_cos_std": profile.pair_cos_std,
        "dup_1nn_fraction": profile.dup_1nn_fraction,
        "dup_1nn_cos": profile.dup_1nn_cos,
        "effective_rank": profile.effective_rank,
        "explained_variance": profile.explained_variance,
    }


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild same-embedder node vectors from LU identity texts")
    parser.add_argument("--from-dataset", default="data/same-embedder",
                        help="dataset to take the node ids and relations from")
    parser.add_argument("--out-dir", default="data/same-embedder-rich")
    parser.add_argument("--plwordnet-repo", default="../plwordnet",
                        help="plwordnet checkout: source of the units and of "
                             "the identity-text recipe")
    parser.add_argument("--nx-graph-dir", default=None,
                        help="NetworkX graph dir; default "
                             "<plwordnet-repo>/" + NX_GRAPH_SUBPATH)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                        help="must stay the model the diagnostics use, or the "
                             "operator stops being comparable with 12/01")
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=2,
                        help="usage examples folded into the identity text")
    parser.add_argument("--limit", type=int, default=0,
                        help="rebuild only the first N nodes (smoke test)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = get_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    lemma_path = os.path.join(args.from_dataset, "node_id_to_lemma.json")
    if not os.path.isfile(lemma_path):
        raise SystemExit(f"{lemma_path} not found; --from-dataset must be a "
                         "dataset built by this module or its predecessor")
    with open(lemma_path, encoding="utf-8") as f:
        node_id_to_lemma = {int(k): v for k, v in json.load(f).items()}
    node_ids = list(node_id_to_lemma)
    if args.limit:
        node_ids = node_ids[:args.limit]
    print(f"[dataset] {len(node_ids)} nodes from {args.from_dataset}")

    texts = load_texts(node_ids, args.plwordnet_repo, args.nx_graph_dir,
                       args.max_examples)
    lemma_dups = 1.0 - len({node_id_to_lemma[i] for i in node_ids}) / len(node_ids)
    text_dups = 1.0 - len(set(texts)) / len(texts)
    print(f"[text] duplicate share: lemma {lemma_dups:.1%} -> identity text "
          f"{text_dups:.1%}")
    print(f"[text] example: {texts[0][:150]!r}")

    vectors = embed(texts, args.embedder, args.device, args.batch_size)
    geometry = report_geometry("identity-text nodes", vectors, args.device)

    with open(os.path.join(args.out_dir, "lexical_units_embedding.pickle"),
              "wb") as f:
        pickle.dump({i: vectors[k] for k, i in enumerate(node_ids)}, f)
    with open(os.path.join(args.out_dir, "node_id_to_text.json"), "w",
              encoding="utf-8") as f:
        json.dump({str(i): texts[k] for k, i in enumerate(node_ids)}, f,
                  ensure_ascii=False)
    if len(node_ids) != len(node_id_to_lemma):
        keep = {str(i): node_id_to_lemma[i] for i in node_ids}
        with open(os.path.join(args.out_dir, "node_id_to_lemma.json"), "w",
                  encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False)
    else:
        src = os.path.join(args.from_dataset, "node_id_to_lemma.json")
        with open(src, encoding="utf-8") as f_in, \
                open(os.path.join(args.out_dir, "node_id_to_lemma.json"), "w",
                     encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    for name in ("relation_to_idx.json", "relations_triplets.json"):
        src = os.path.join(args.from_dataset, name)
        with open(src, "rb") as f_in, \
                open(os.path.join(args.out_dir, name), "wb") as f_out:
            f_out.write(f_in.read())

    with open(os.path.join(args.out_dir, "build_report.json"), "w",
              encoding="utf-8") as f:
        json.dump({"from_dataset": args.from_dataset,
                   "plwordnet_repo": os.path.abspath(args.plwordnet_repo),
                   "nx_graph_dir": args.nx_graph_dir or os.path.join(
                       args.plwordnet_repo, NX_GRAPH_SUBPATH),
                   "embedder": args.embedder, "max_examples": args.max_examples,
                   "num_nodes": len(node_ids),
                   "lemma_duplicate_share": lemma_dups,
                   "text_duplicate_share": text_dups,
                   "geometry": geometry}, f, indent=2, ensure_ascii=False)
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
