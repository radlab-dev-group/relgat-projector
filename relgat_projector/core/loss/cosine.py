import torch


class CosineLoss:
    @staticmethod
    def calculate(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        1 - cosine_similarity averaged over batch.
        """
        pred_n = torch.nn.functional.normalize(pred, p=2, dim=-1)
        tgt_n = torch.nn.functional.normalize(target, p=2, dim=-1)
        cos = (pred_n * tgt_n).sum(dim=-1)  # [B]
        return (1.0 - cos).mean()
