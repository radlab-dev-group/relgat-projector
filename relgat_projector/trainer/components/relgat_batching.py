import torch
from typing import Tuple


def concat_pos_negs_to_tensors(
    pos: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    negs: Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], ...],
    device: str,
):
    src_ids = torch.cat(
        [p[0] for p in pos] + [n[0] for n in sum(negs, ())], dim=0
    ).to(device)
    rel_ids = torch.cat(
        [p[1] for p in pos] + [n[1] for n in sum(negs, ())], dim=0
    ).to(device)
    dst_ids = torch.cat(
        [p[2] for p in pos] + [n[2] for n in sum(negs, ())], dim=0
    ).to(device)
    return src_ids, rel_ids, dst_ids
