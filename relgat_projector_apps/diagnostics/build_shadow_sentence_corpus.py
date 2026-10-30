"""Build the sentence corpus the sentence-level F4 supervision trains on.

Why.  The sentence-level ideas (01, 02, 04, 12) are scored on *sentences*, and
the F4 stage is where the operator is supposed to learn to work off the graph.
Until now F4 regularised on jittered node vectors, and the measured
contribution was exactly zero -- not because off-graph consistency is easy,
but because the cycle term of an exactly invertible operator is an identity
(``L_shadow_cycle = 0.0000`` through a whole run).  Regularisers can say where
not to go; they cannot say where to go.

Słowosieć attaches usage examples to 111 536 of its 514 016 lexical units
(156 315 examples in total).  Those examples are text from the same embedder,
and every one of them belongs to a node of the graph.  For an edge ``(a, r, b)``
whose endpoints both own an example, the pair of sentences is therefore a
**labelled transport pair in the space the product works in** -- which turns
F4 from a regulariser into supervision (see ``L_shadow_sent`` in the trainer).

Writes three files next to each other, rows aligned:

* ``shadow_sentences.jsonl``  -- ``{"text", "lu_id"}`` per line, auditable;
* ``shadow_sentences.npy``     -- ``[S, D]`` float32, same embedder as the nodes;
* ``shadow_sentences_lu_ids.npy`` -- ``[S]`` int64 LU id of each sentence.

The LU ids are graph node ids, so the trainer joins them with its own node
index and drops the sentences whose unit is not in the dataset.

Usage::

    python -m relgat_projector_apps.diagnostics.build_shadow_sentence_corpus \\
        --out-dir data/same-embedder-rich --plwordnet-repo ../plwordnet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_EMBEDDER = "ipipan/silver-retriever-base-v1.1"
#: default location of the NetworkX export inside a plwordnet checkout
NX_GRAPH_SUBPATH = os.path.join(
    "resources", "plwordnet_4_5", "full", "graphs", "full", "nx", "graphs")


def collect_usage_examples(nx_graph_dir: str) -> List[Tuple[str, int]]:
    """Every non-empty usage example of the export, with the LU it illustrates.

    ``lu.comment.usage_examples`` is the field that holds them; ``lu.examples``
    is empty in the NetworkX export, which is the trap this function exists to
    keep anybody from falling into.
    """
    repo_root = os.path.abspath(os.path.join(nx_graph_dir, *([".."] * 0)))
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from plwordnet_handler.base.structure.polishwordnet import PolishWordnet

    print(f"[plwordnet] reading usage examples from {nx_graph_dir}")
    plwn = PolishWordnet(nx_graph_dir=nx_graph_dir)
    plwn.connect()
    rows: List[Tuple[str, int]] = []
    units = 0
    for lu in plwn.get_lexical_units():
        units += 1
        comment = lu.comment
        if not comment or not comment.usage_examples:
            continue
        for example in comment.usage_examples:
            text = (example.text or "").strip()
            if text:
                rows.append((text, int(lu.ID)))
    print(f"[plwordnet]   {units} lexical units, {len(rows)} examples "
          f"over {len({i for _, i in rows})} units")
    return rows


def embed(texts: List[str], embedder: str, device: str,
          batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[embedder] loading {embedder} on {device}")
    model = SentenceTransformer(embedder, device=device)
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                           normalize_embeddings=False, convert_to_numpy=True)
    print(f"[embedder]   {len(texts)} texts, dim "
          f"{model.get_sentence_embedding_dimension()}")
    return np.ascontiguousarray(vectors, dtype=np.float32)


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the LU-indexed sentence corpus for sentence-level F4")
    parser.add_argument("--out-dir", default="data/same-embedder-rich")
    parser.add_argument("--plwordnet-repo", default="../plwordnet",
                        help="plwordnet checkout holding the NetworkX export")
    parser.add_argument("--nx-graph-dir", default=None,
                        help="default <plwordnet-repo>/" + NX_GRAPH_SUBPATH)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                        help="must stay the model the nodes and the "
                             "diagnostics use, or the operator stops being "
                             "comparable with 12/01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0,
                        help="embed only the first N examples (smoke test)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = get_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    nx_dir = args.nx_graph_dir or os.path.join(args.plwordnet_repo,
                                               NX_GRAPH_SUBPATH)

    rows = collect_usage_examples(nx_dir)
    if args.limit:
        rows = rows[:args.limit]
    texts = [t for t, _ in rows]
    lu_ids = np.asarray([i for _, i in rows], dtype=np.int64)

    vectors = embed(texts, args.embedder, args.device, args.batch_size)

    jsonl_path = os.path.join(args.out_dir, "shadow_sentences.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for text, lu_id in rows:
            f.write(json.dumps({"text": text, "lu_id": lu_id},
                               ensure_ascii=False) + "\n")
    np.save(os.path.join(args.out_dir, "shadow_sentences.npy"), vectors)
    np.save(os.path.join(args.out_dir, "shadow_sentences_lu_ids.npy"), lu_ids)

    norm = float(np.linalg.norm(vectors, axis=-1).mean())
    print(f"[done] {vectors.shape} -> {args.out_dir} "
          f"(mean norm {norm:.2f}; the node vectors of same-embedder-rich "
          "have 23.67, so the shadow losses need no rescaling)")


if __name__ == "__main__":
    main()
