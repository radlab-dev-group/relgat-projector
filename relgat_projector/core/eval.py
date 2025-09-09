import torch

from typing import Tuple, Dict


class RelgatEval:
    @staticmethod
    def compute_mrr_hits(
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
