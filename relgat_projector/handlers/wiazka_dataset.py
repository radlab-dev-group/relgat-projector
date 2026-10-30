"""
Dataset loading for the WIĄZKA trainer.

Consumes the same exporter output as the legacy trainer (produced by
``plwordnet-milvus --export-relgat-dataset``):

* ``lexical_units_embedding.pickle``  -- ``{int node_id: vector}``
* ``relation_to_idx.json``            -- ``{relation_name: int}``
* ``relations_triplets.json``         -- ``[[from_id, to_id, rel_name], ...]``

Optionally a file of *shadow vectors* (real text embeddings that are NOT
graph nodes) for the F4 stage: ``.npy`` array or a ``.pickle`` dict like the
node embeddings.
"""

from __future__ import annotations

import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def load_graph_dataset(
    nodes_path: str,
    rels_path: str,
    edges_path: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, int], List[Tuple[int, int, str]]]:
    """Load node embeddings, the relation mapping and the edge triplets."""
    print(f"[data] loading nodes: {nodes_path}")
    with open(nodes_path, "rb") as f:
        node2emb = pickle.load(f)
    node2emb = {int(k): torch.as_tensor(v, dtype=torch.float32) for k, v in node2emb.items()}
    print(f"[data]   nodes: {len(node2emb)}")

    print(f"[data] loading relations: {rels_path}")
    with open(rels_path, "r", encoding="utf-8") as f:
        rel2idx = {str(k): int(v) for k, v in json.load(f).items()}
    print(f"[data]   relations: {len(rel2idx)}")

    print(f"[data] loading triplets: {edges_path}")
    with open(edges_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    edges = [
        (int(s), int(d), str(r))
        for s, d, r in raw
        if int(s) in node2emb and int(d) in node2emb and str(r) in rel2idx
    ]
    print(f"[data]   edges: {len(edges)} (after filtering)")
    return node2emb, rel2idx, edges


def load_shadow_vectors(path: str, dim: int) -> Optional[np.ndarray]:
    """Load shadow (off-graph text) vectors; ``None`` when the path is absent."""
    if not path:
        return None
    print(f"[data] loading shadow vectors: {path}")
    if path.endswith(".npy"):
        vec = np.load(path)
    elif path.endswith((".pickle", ".pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            vec = np.stack([np.asarray(v, dtype=np.float32) for v in data.values()])
        else:
            vec = np.asarray(data, dtype=np.float32)
    else:
        raise ValueError(f"unsupported shadow vector format: {path}")
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if vec.shape[1] != dim:
        raise ValueError(
            f"shadow vectors have dim {vec.shape[1]}, expected {dim}"
        )
    print(f"[data]   shadow vectors: {vec.shape[0]}")
    return vec


def load_shadow_sentences(path: str, lu_ids_path: str, dim: int):
    """Sentence corpus for the sentence-level F4 supervision.

    ``path`` is a ``.npy`` ``[S, D]`` array (or a ``.pickle`` dict) of *text*
    vectors -- the Słowosieć usage examples, embedded with the same embedder
    as the nodes.  ``lu_ids_path`` is a ``.npy`` ``[S]`` int64 array naming, for
    every sentence, the LU (graph node) it illustrates; the two files are rows
    of the same corpus, produced together by
    ``relgat_projector_apps.diagnostics.build_shadow_sentence_corpus``.

    Returns ``(vectors [S, D] float32, lu_ids [S] int64)`` or ``(None, None)``
    when no path is given.
    """
    if not path:
        return None, None
    if not lu_ids_path:
        raise ValueError(
            "--shadow-sentences needs --shadow-sentence-lus: a sentence corpus "
            "without the LU it belongs to cannot supervise a relation")
    print(f"[data] loading shadow sentences: {path}")
    if path.endswith(".npy"):
        vec = np.load(path)
    elif path.endswith((".pickle", ".pkl")):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            vec = np.stack([np.asarray(v, dtype=np.float32) for v in data.values()])
        else:
            vec = np.asarray(data, dtype=np.float32)
    else:
        raise ValueError(f"unsupported shadow sentence format: {path}")
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if vec.shape[1] != dim:
        raise ValueError(f"shadow sentences have dim {vec.shape[1]}, expected {dim}")
    lus = np.load(lu_ids_path)
    lus = np.ascontiguousarray(lus, dtype=np.int64).reshape(-1)
    if len(lus) != vec.shape[0]:
        raise ValueError(
            f"{len(lus)} LU ids for {vec.shape[0]} sentence vectors: "
            f"{lu_ids_path} and {path} are not the same corpus")
    print(f"[data]   shadow sentences: {vec.shape[0]}")
    return vec, lus
