import os
import json
import torch
import torch.nn as nn

from typing import Tuple, Optional, List, Dict, Any

from relgat_projector.core.scorer import DistMultScorer, TransEScorer
from relgat_projector.core.model.relgat_base.layer import RelGATLayer
from relgat_projector.core.model.relgat_base.projection import ProjectionHead


class RelGATModel(nn.Module):
    def __init__(
        self,
        node_emb: torch.Tensor,  # [N, 1152] (frozen)
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,  # [E]
        num_rel: int,
        scorer_type: str = "distmult",  # "transe"
        gat_out_dim: int = 200,
        gat_heads: int = 4,
        dropout: float = 0.2,
        relation_attn_dropout: float = 0.0,
        gat_num_layers: int = 1,
        project_to_input_size: bool = False,
        projection_layers: int = 1,
        projection_dropout: float = 0.0,
        projection_hidden_dim: int = 0,
    ):
        super().__init__()
        self.register_buffer("node_emb_fixed", node_emb)  # not a Parameter

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.gat_num_layers = gat_num_layers
        self.projection_layers = projection_layers
        self.project_to_input_size = project_to_input_size
        if project_to_input_size and self.projection_layers < 1:
            raise ValueError(
                "projection_layers must be >= 1 when project_to_input_size=True"
            )

        if gat_num_layers == 1:
            self.gat_layer = RelGATLayer(
                in_dim=node_emb.size(1),
                out_dim=gat_out_dim,
                num_rel=num_rel,
                heads=gat_heads,
                dropout=dropout,
                relation_attn_dropout=relation_attn_dropout,
                use_bias=True,
            )
            # No activation between layers
            self.act = None
        else:
            self.gat_layers = nn.ModuleList()
            in_dim = node_emb.size(1)
            for layer_idx in range(max(1, gat_num_layers)):
                self.gat_layers.append(
                    RelGATLayer(
                        in_dim=in_dim,
                        out_dim=gat_out_dim,
                        num_rel=num_rel,
                        heads=gat_heads,
                        dropout=dropout,
                        relation_attn_dropout=relation_attn_dropout,
                        use_bias=True,
                    )
                )
                # subsequent layers now receive the dimension heads*out_dim
                in_dim = gat_out_dim * gat_heads
            self.act = nn.ELU()

        # Optional projection back to the input-embedding dimension
        _scorer_gat_dim = gat_out_dim * gat_heads
        if self.project_to_input_size:
            self.projection = ProjectionHead(
                in_dim=_scorer_gat_dim,
                out_dim=node_emb.size(1),
                hidden_dim=projection_hidden_dim,
                num_layers=self.projection_layers,
                dropout=projection_dropout,
            )
            _scorer_gat_dim = node_emb.size(1)
        else:
            self.projection = None

        self.scorer_type = scorer_type
        if scorer_type.lower() == "distmult":
            self.scorer = DistMultScorer(num_rel, rel_dim=_scorer_gat_dim)
        elif scorer_type.lower() == "transe":
            self.scorer = TransEScorer(num_rel, rel_dim=_scorer_gat_dim)
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}")

    def forward(
        self,
        src_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        transform_to_input_if_possible: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute scores for a batch of triples. And optionally transform the
        GAT output dimension to input embedding-space.
        src_ids, rel_ids, dst_ids : [B]  (int64)

        Parameters
        ----------
        src_ids : torch.LongTensor [B]
            Source node indices.
        rel_ids : torch.LongTensor [B]
            Relation indices.
        dst_ids : torch.LongTensor [B]
            Destination node indices.
        transform_to_input_if_possible: bool:
            Whether to transform the GAT output dimension
            to input embedding-space if projection is available.

        Returns
        -------
        Tuple[
            torch.Tensor [B],
            Optional[torch.Tensor] [B, D_sc],
            torch.Tensor [B, D']
        ]
        Compatibility scores (higher ⇒ more plausible), optional projection
        to embedding space and destination vector


        """
        x = self.single_gat_step()
        src_vec = x[src_ids]  # [B, D']
        dst_vec = x[dst_ids]  # [B, D']
        transformed = None
        if self.project_to_input_size and transform_to_input_if_possible:
            transformed = self.scorer.transform(src_vec, rel_ids)  # [B, D_sc]
        scores = self.scorer(src_vec, rel_ids, dst_vec)  # [B]
        return scores, transformed, dst_vec

    @torch.no_grad()
    def get_node_repr(self) -> torch.Tensor:
        """
        Return full matrix of node representations used by the scorer.
        Useful for exporting or offline usage.
        """
        return self.single_gat_step()

    @torch.no_grad()
    def transform(
        self, src_ids: torch.Tensor, rel_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform source node embeddings by relation operator and return transformed vectors.
        Shapes:
          - src_ids: [B]
          - rel_ids: [B] or [1] (will broadcast if needed)
        Returns:
          - transformed vectors: [B, D_sc]
        """
        x = self.single_gat_step()
        src_vec = x[src_ids]
        # support broadcasting single rel_id over batch
        return self.transform_from_vectors(src_vectors=src_vec, rel_ids=rel_ids)

    @torch.no_grad()
    def transform_from_vectors(
        self, src_vectors: torch.Tensor, rel_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply relation operator to arbitrary vectors in the scorer's space.
        If project_to_input=True, this space is the original input embedding space.
        Shapes:
          - src_vectors: [B, D_sc]
          - rel_ids: [B] or [1]
        Returns:
          - transformed vectors: [B, D_sc]
        """
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(1)
        if rel_ids.numel() == 1 and src_vectors.size(0) > 1:
            rel_ids = rel_ids.expand(src_vectors.size(0))
        return self.scorer.transform(src_vectors, rel_ids)

    def get_config(self) -> dict:
        """
        Zwraca konfigurację potrzebną do odtworzenia modelu.
        Uwaga: node_emb, edge_index i edge_type nie są zapisywane,
        bo zależą od danych – należy je dostarczyć przy ładowaniu.
        """
        return dict(self._config)

    def save_pretrained(
        self, output_dir: str, add_files: Optional[List[Tuple[str, Dict[str, Any]]]]
    ) -> None:
        """
        Zapisuje wagi modelu i konfigurację do folderu:
         - {output_dir}/config.json
         - {output_dir}/pytorch_model.bin
        """
        add_files = [] if add_files is None else add_files
        os.makedirs(output_dir, exist_ok=True)
        # Config files
        add_files.append(("config.json", self.get_config()))
        for _d_file_name, _f_content in add_files:
            _f_path = os.path.join(output_dir, _d_file_name)
            with open(_f_path, "w", encoding="utf-8") as f:
                json.dump(_f_content, f, ensure_ascii=False, indent=2)

        # Model's weights
        w_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), w_path)

    @staticmethod
    def load_from_pretrained(
        input_dir: str,
        *,
        node_emb: torch.Tensor,  # [N, D_in]
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,  # [E]
        map_location: str | torch.device | None = None,
    ) -> "RelGATModel":
        """
        Ładuje model do inferencji:
        - odczytuje config.json
        - konstruuje RelGATModel z podanymi: node_emb, edge_index, edge_type
        - wczytuje wagi z pytorch_model.bin
        Zwraca model w trybie eval().
        """
        cfg_path = os.path.join(input_dir, "config.json")
        w_path = os.path.join(input_dir, "pytorch_model.bin")

        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        if not os.path.isfile(w_path):
            raise FileNotFoundError(f"Weights file not found: {w_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Walidacja wymiarów
        if int(cfg.get("input_dim")) != int(node_emb.size(1)):
            raise ValueError(
                f"Input dim mismatch: "
                f"config={cfg.get('input_dim')} vs node_emb={node_emb.size(1)}"
            )

        model = RelGATModel(
            node_emb=node_emb,
            edge_index=edge_index,
            edge_type=edge_type,
            num_rel=int(cfg["num_rel"]),
            scorer_type=str(cfg["scorer_type"]),
            gat_out_dim=int(cfg["gat_out_dim"]),
            gat_heads=int(cfg["gat_heads"]),
            dropout=float(cfg["dropout"]),
            relation_attn_dropout=float(cfg["relation_attn_dropout"]),
            gat_num_layers=int(cfg["gat_num_layers"]),
            project_to_input_size=bool(cfg["project_to_input_size"]),
            projection_layers=int(cfg["projection_layers"]),
            projection_dropout=float(cfg["projection_dropout"]),
            projection_hidden_dim=int(cfg["projection_hidden_dim"]),
        )

        state = torch.load(w_path, map_location=map_location)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def single_gat_step(self) -> torch.Tensor:
        """
        Compute node representations (optionally projected to input dim).
        """
        if self.gat_num_layers == 1:
            x = self.gat_layer(
                self.node_emb_fixed, self.edge_index, self.edge_type
            )  # [N, D_gat]
        else:
            x = self.node_emb_fixed
            for li, gat in enumerate(self.gat_layers):
                x = gat(x, self.edge_index, self.edge_type)  # [N, D_gat]
                if self.act is not None and li < len(self.gat_layers) - 1:
                    x = self.act(x)

        if self.project_to_input_size:
            x = self.projection(x)  # [N, D_in]

        return x
