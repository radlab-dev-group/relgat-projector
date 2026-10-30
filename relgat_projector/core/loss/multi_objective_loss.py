import torch

from typing import Any, Dict, Optional, List, Tuple

from relgat_projector.core.loss.cosine import CosineLoss
from relgat_projector.core.loss.mse import MSELoss
from relgat_projector.core.loss.relgat_loss import RelGATLoss


class MultiObjectiveRelLoss:
    """
    Kombinuje klasyczny ranking loss z rekonstrukcyjnymi stratami A->r->B:
    - ranking_loss: margin/self-adversarial na score'ach
    - cosine_recon: 1 - cosine(f_r(A), B)
    - mse_recon: MSE(f_r(A), B)
    - ontology_regularization: regularizacja na podstawie ontologii

    Wagi poszczególnych komponentów konfigurowalne.
    """

    def __init__(
        self,
        *,
        relgat_loss: RelGATLoss,
        run_config: Dict[str, Any],
        pos_cosine_weight: float = 1.0,
        neg_cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
        relgat_weight: float = 1.0,
        ontology_reg_weight: float = 0.0,
        ontology_similarity: Optional[torch.Tensor] = None,
    ):
        self.ranking_weight = run_config.get("relgat_weight", relgat_weight)
        self.pos_cosine_weight = run_config.get(
            "pos_cosine_weight", pos_cosine_weight
        )
        self.neg_cosine_weight = run_config.get(
            "neg_cosine_weight", neg_cosine_weight
        )
        self.mse_weight = run_config.get("mse_weight", mse_weight)
        self.ontology_reg_weight = run_config.get(
            "ontology_reg_weight", ontology_reg_weight
        )

        self.relgat_loss = relgat_loss

        # Ontology-aware components
        self.ontology_similarity = ontology_similarity
        self.ontology_reg_active = (
            ontology_reg_weight > 0.0 and ontology_similarity is not None
        )

    def relgat_ranking_loss(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> torch.Tensor:
        return self.relgat_loss.prepare_scores_and_compute_loss(
            pos_score=pos_score, neg_score=neg_score
        )

    def __call__(
        self,
        *,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        transformed_src: torch.Tensor,  # f_r(A)
        dst_vec: torch.Tensor,  # B
        neg_dst_vec: torch.Tensor,  # [B NumNeg]
        rel_ids: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        rel_ids: [B] – IDs relacji dla pozytywów
        rel_embeddings: [num_rel, D] – embeddingi wszystkich relacji (do regularizacji)
        """
        parts = []
        weights = []

        # 1. Ranking loss (naprawione: bez podwójnego mnożenia wagi)
        if self.ranking_weight != 0.0:
            weights.append(self.ranking_weight)
            parts.append(
                self.relgat_ranking_loss(
                    pos_score, neg_score
                )  # ← NAPRAWA: bez powtórzenia wagi
            )

        # 2. Pozitywna cosine similarity (rekonstrukcja A->r->B)
        if self.pos_cosine_weight != 0.0:
            weights.append(self.pos_cosine_weight)
            parts.append(
                self.pos_cosine_weight
                * CosineLoss.calculate(transformed_src, dst_vec)
            )

        # 3. Negatywna cosine similarity (dla negatywnych dest)
        if self.neg_cosine_weight != 0.0:
            weights.append(self.neg_cosine_weight)
            # neg_dst_vec ma kształt [K, B, D] lub [B, K, D] – dopasuj
            if neg_dst_vec.dim() == 3:
                # [K, B, D] -> [B, K, D]
                neg_dst_vec = neg_dst_vec.permute(1, 0, 2)
            # broadcast transformed_src [B, D] z neg_dst_vec [B, K, D]
            neg_dst_vec_flat = neg_dst_vec.view(-1, neg_dst_vec.shape[-1])
            transformed_src_broadcast = (
                transformed_src.unsqueeze(1)
                .expand(-1, neg_dst_vec.shape[1], -1)
                .reshape(-1, transformed_src.shape[-1])
            )
            parts.append(
                self.neg_cosine_weight
                * (
                    1.0
                    - CosineLoss.calculate(
                        transformed_src_broadcast, neg_dst_vec_flat
                    )
                )
            )

        # 4. MSE loss
        if self.mse_weight != 0.0:
            weights.append(self.mse_weight)
            parts.append(
                self.mse_weight * MSELoss.calculate(transformed_src, dst_vec)
            )

        # 5. Ontology regularizacja (dodane ulepszenie)
        ontology_loss = torch.tensor(0.0, device=transformed_src.device)
        if (
            self.ontology_reg_active
            and rel_ids is not None
            and rel_embeddings is not None
        ):
            weights.append(self.ontology_reg_weight)
            ontology_loss = self._ontology_regularization(
                rel_embeddings=rel_embeddings,
                rel_ids=rel_ids,
                ontology_similarity=self.ontology_similarity,
            )
            parts.append(self.ontology_reg_weight * ontology_loss)

        if not parts:
            raise ValueError("At least one loss weight must be non-zero.")

        total_loss = torch.stack(parts).sum()
        return total_loss / sum(weights) if weights else total_loss

    def _ontology_regularization(
        self,
        rel_embeddings: torch.Tensor,  # [num_rel, D]
        rel_ids: torch.Tensor,  # [B]
        ontology_similarity: torch.Tensor,  # [num_rel, num_rel]
    ) -> torch.Tensor:
        """
        Regularizacja zachęcająca embeddingi semantycznie podobnych relacji
        do bycia podobnymi w przestrzeni embeddingów.

        L_ont = -sum_{i,j} S_{ij} * cos(emb_i, emb_j)
        gdzie S_{ij} to podobieństwo semantyczne z ontologii
        """
        # Wydobądź embeddingi użytych relacji
        used_rels = rel_embeddings[rel_ids.unique()]  # [n_unique, D]
        sim_matrix = ontology_similarity[rel_ids.unique(), :][
            :, rel_ids.unique()
        ]  # [n_unique, n_unique]

        # Oblicz cosine similarity dla embeddingów
        used_rels_norm = torch.nn.functional.normalize(used_rels, p=2, dim=-1)
        cosine_sim = torch.matmul(
            used_rels_norm, used_rels_norm.t()
        )  # [n_unique, n_unique]

        # Penalizuj rozbieżność między semantyką ontologiczną a embeddingami
        loss = -torch.sum(sim_matrix * cosine_sim) / (sim_matrix.numel() + 1e-8)

        return loss
