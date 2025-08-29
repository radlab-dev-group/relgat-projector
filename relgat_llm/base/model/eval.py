import torch

from typing import Tuple


class RelgatEval:
    # Helper methods (loss, ranking, evaluation)
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
