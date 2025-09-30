"""Dataset utilities for RelGAT – handling node embeddings, edge splits and loaders."""

import copy
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from relgat_projector.dataset.edge import EdgeDataset


class RelGATDataset:
    """
    Prepares train / evaluation splits, PyTorch ``DataLoader`` objects and
    adjacency tensors for a RelGAT model.
    """

    def __init__(
        self,
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edge_index_raw: List[Tuple[int, int, str]],
        train_ratio: float,
        num_neg: int,
        train_batch_size: int,
        eval_batch_size: int,
        device: str,
        run_config: Dict[str, Any],
    ) -> None:
        # ----------------------------------------------------------------- #
        # Raw inputs
        # ----------------------------------------------------------------- #
        self.node2emb = node2emb
        self.rel2idx = rel2idx
        self.edge_index_raw = edge_index_raw

        # ----------------------------------------------------------------- #
        # Configuration (with possible overrides from ``run_config``)
        # ----------------------------------------------------------------- #
        self.device = device
        self.train_ratio = float(run_config.get("train_ratio", train_ratio))
        self.num_neg = int(run_config.get("num_neg", num_neg))
        self.train_batch_size = int(
            run_config.get("train_batch_size", train_batch_size)
        )
        self.eval_batch_size = int(
            run_config.get("eval_batch_size", eval_batch_size)
        )
        self._seed = int(run_config.get("seed", random.randint(0, 2**31 - 1)))

        # ----------------------------------------------------------------- #
        # Internals – populated later
        # ----------------------------------------------------------------- #
        self.id2idx: Dict[int, int] = {}
        self.all_node_ids: List[int] = []
        self.node_emb_matrix: torch.Tensor

        self.train_edges: List[Tuple[int, int, str]] = []
        self.eval_edges: List[Tuple[int, int, str]] = []
        self.train_dataset: EdgeDataset
        self.eval_dataset: EdgeDataset
        self.train_loader: DataLoader
        self.eval_loader: DataLoader

        self.edge_index: torch.Tensor
        self.edge_type: torch.Tensor

        # ----------------------------------------------------------------- #
        # Build everything
        # ----------------------------------------------------------------- #
        self._prepare_base_mappings()
        self._prepare_nodes_matrix()
        self._prepare_dataset()
        self._prepare_loaders()
        self._prepare_edge_dataset()

        # --------------------------------------------------------------------- #
        # Private helpers
        # --------------------------------------------------------------------- #

    def _prepare_base_mappings(self) -> None:
        """Create ``id -> compact index`` mapping."""
        self.all_node_ids = sorted(self.node2emb.keys())
        self.id2idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

    def _prepare_nodes_matrix(self) -> None:
        """Stack node embeddings into a single tensor (no gradient required)."""
        with torch.no_grad():
            self.node_emb_matrix = torch.stack(
                [torch.as_tensor(self.node2emb[nid]) for nid in self.all_node_ids],
                dim=0,
            ).to(self.device)

    def _prepare_dataset(self) -> None:
        """Randomly split edges into training / evaluation sets."""
        # Work on a copy to keep the original list intact
        edges_copy = copy.deepcopy(self.edge_index_raw)

        rng = random.Random(self._seed)
        rng.shuffle(edges_copy)

        n_edges = len(edges_copy)
        n_train = int(self.train_ratio * n_edges)

        # Map original node IDs to compact indices
        def _map_edge(e: Tuple[int, int, str]) -> Tuple[int, int, str]:
            s, d, r = e
            return self.id2idx[s], self.id2idx[d], r

        mapped = [_map_edge(e) for e in edges_copy]
        self.train_edges = mapped[:n_train]
        self.eval_edges = mapped[n_train:]

        # Simple diagnostic output – can be silenced by setting LOG_LEVEL env var
        print(
            f"Number of edges (relations): {n_edges}\n"
            f" - train: {len(self.train_edges)} ({self.train_ratio * 100:.1f} %)\n"
            f" - eval: {len(self.eval_edges)} ({100 - self.train_ratio * 100:.1f} %)"
        )

    def _prepare_loaders(self) -> None:
        """Create ``EdgeDataset`` and ``DataLoader`` objects."""
        common_kwargs = dict(
            node2emb=self.node2emb,
            rel2idx=self.rel2idx,
            num_neg=self.num_neg,
            all_node_ids=list(range(len(self.all_node_ids))),
        )
        self.train_dataset = EdgeDataset(
            edge_index=self.train_edges, **common_kwargs
        )
        self.eval_dataset = EdgeDataset(edge_index=self.eval_edges, **common_kwargs)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

    def _prepare_edge_dataset(self) -> None:
        """Create tensors ``edge_index`` and ``edge_type`` for the training graph."""
        train_src, train_dst, train_rel = zip(*self.train_edges)

        self.edge_index = torch.tensor(
            [train_src, train_dst], dtype=torch.long, device=self.device
        )
        self.edge_type = torch.tensor(
            [self.rel2idx[r] if isinstance(r, str) else int(r) for r in train_rel],
            dtype=torch.long,
            device=self.device,
        )

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def get_edge_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(edge_index, edge_type)`` for the training graph."""
        return self.edge_index, self.edge_type
