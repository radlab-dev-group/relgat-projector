import torch

from typing import Tuple, Dict


class RelgatEval:
    @staticmethod
    def compute_mrr_hits(
        scores: torch.Tensor, true_idx: int = 0, ks: Tuple[int, ...] = (1, 3, 10)
    ):
        # Zabezpieczenie:
        #   NaN/±inf traktujemy jako "bardzo słabe" wyniki,
        #   by nie windowały MRR do 1.0
        s = torch.nan_to_num(scores, nan=-1e9, neginf=-1e9, posinf=1e9)
        if not torch.isfinite(s[true_idx]):
            s[true_idx] = -1e9
        rank = (s > s[true_idx]).sum().item() + 1
        mrr = 1.0 / max(1, rank)
        hits = {k: 1.0 if rank <= k else 0.0 for k in ks}
        return mrr, hits

    @staticmethod
    def compute_batch_metrics_vectorized(
        *,
        pos_score: torch.Tensor,
        neg_score: torch.Tensor,
        ks: Tuple[int, ...],
        pessimistic: bool = True,
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute MRR i Hits@K wektorowo z polityką remisów pesymistyczną:
        - rank_i = 1 + count(neg_i >= pos_i)   (remisy liczone jako gorsze)
        Sanitizacja jak w implementacji referencyjnej:
        - NaN/-Inf/+Inf mapowane odpowiednio do (-1e9, -1e9, 1e9).
        """
        B = pos_score.shape[0]
        if B == 0:
            return 0.0, {k: 0.0 for k in ks}

        pos_s = torch.nan_to_num(pos_score, nan=-1e9, neginf=-1e9, posinf=1e9)
        neg_s = torch.nan_to_num(neg_score, nan=-1e9, neginf=-1e9, posinf=1e9)

        if pessimistic:
            worse_or_equal = (neg_s >= pos_s.unsqueeze(1)).to(pos_s.dtype)
        else:
            worse_or_equal = (neg_s > pos_s.unsqueeze(1)).to(pos_s.dtype)

        ranks = 1.0 + worse_or_equal.sum(dim=1)  # [B]

        mrr = (1.0 / torch.clamp(ranks, min=1.0)).mean().item()
        hits = {k: (ranks <= float(k)).to(pos_s.dtype).mean().item() for k in ks}
        return mrr, hits
