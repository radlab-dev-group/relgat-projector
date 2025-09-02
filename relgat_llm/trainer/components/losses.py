import torch

from relgat_llm.base.model.loss import RelGATLoss


def prepare_scores_and_compute_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
    use_self_adv_neg: bool,
    self_adv_alpha: float,
    margin: float,
    clamp_limit: float = 20.0,
) -> torch.Tensor:
    # Clamp ranges preserved exactly as before
    pos_s = pos_score.clamp(-clamp_limit, clamp_limit)
    neg_s = neg_score.clamp(-clamp_limit, clamp_limit)

    if use_self_adv_neg:
        return RelGATLoss.self_adversarial_loss(pos_s, neg_s, alpha=self_adv_alpha)
    else:
        return RelGATLoss.margin_ranking_loss(pos_s, neg_s, margin=margin)
