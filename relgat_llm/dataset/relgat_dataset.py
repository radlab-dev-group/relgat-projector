import torch
import random

from torch.utils.data import DataLoader
from typing import Tuple, List, Any, Dict

from relgat_llm.dataset.edge import EdgeDataset


class RelGATDataset:
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
    ):
        # Raw dataset
        self.node2emb = node2emb
        self.rel2idx = rel2idx
        self.num_rel = len(self.rel2idx)
        self.edge_index_raw = edge_index_raw

        self.device = device
        self.train_ratio = float(run_config.get("train_ratio", train_ratio))
        self.num_neg = int(run_config.get("num_neg", num_neg))
        self.train_batch_size = int(
            run_config.get("train_batch_size", train_batch_size)
        )
        self.eval_batch_size = int(
            run_config.get("eval_batch_size", eval_batch_size)
        )

        # Embeddings id to proper idx (node)
        self.id2idx = None
        self.all_node_ids = None
        self.node_emb_matrix = None

        # Raw mapping after train/test split
        self.train_edges = None
        self.eval_edges = None
        self.train_dataset = None
        self.eval_dataset = None
        self.train_loader = None
        self.eval_loader = None

        self.edge_index = None
        self.edge_type = None

        self._prepare_base_mappings()
        self._prepare_nodes_matrix()
        self._prepare_dataset()
        self._prepare_loaders()
        self._prepare_edge_dataset()

    def _prepare_base_mappings(self):
        self.all_node_ids = sorted(self.node2emb.keys())
        self.id2idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

    def _prepare_nodes_matrix(self):
        self.node_emb_matrix = torch.stack(
            [torch.as_tensor(self.node2emb[nid]) for nid in self.all_node_ids], dim=0
        ).to(self.device)

    def _prepare_dataset(self):
        # random division of edges to train/test
        random.shuffle(self.edge_index_raw)
        n_edges = len(self.edge_index_raw)
        n_train = int(self.train_ratio * n_edges)

        # Remap edges on compact indexes
        def _map_edge(e):
            s, d, r = e
            return self.id2idx[s], self.id2idx[d], r

        mapped_edges = [_map_edge(e) for e in self.edge_index_raw]
        self.train_edges = mapped_edges[:n_train]
        self.eval_edges = mapped_edges[n_train:]
        print(f"Number of edges (relations): {n_edges}")
        print(f" - train: {len(self.train_edges)} ({self.train_ratio * 100:.1f} %)")
        print(
            f" - eval: {len(self.eval_edges)} ({100 - self.train_ratio * 100:.1f} %)"
        )

    def _prepare_loaders(self):
        # Dataset / DataLoader
        self.train_dataset = EdgeDataset(
            edge_index=self.train_edges,
            node2emb=self.node2emb,
            rel2idx=self.rel2idx,
            num_neg=self.num_neg,
            all_node_ids=list(range(len(self.all_node_ids))),
        )
        self.eval_dataset = EdgeDataset(
            edge_index=self.eval_edges,
            node2emb=self.node2emb,
            rel2idx=self.rel2idx,
            num_neg=self.num_neg,
            all_node_ids=list(range(len(self.all_node_ids))),
        )

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
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

    def _prepare_edge_dataset(self):
        # Mapping lu from triples on compact indexes
        train_src_list, train_dst_list, train_rel_list = zip(*self.train_edges)

        self.edge_index = torch.tensor(
            [train_src_list, train_dst_list], dtype=torch.long
        ).to(self.device)

        self.edge_type = torch.tensor(
            [
                self.rel2idx[r] if isinstance(r, str) else int(r)
                for r in train_rel_list
            ],
            dtype=torch.long,
        ).to(self.device)
