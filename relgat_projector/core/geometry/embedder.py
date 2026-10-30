"""
What a base-embedding space *is*, measured.

The WIĄZKA pipeline makes several statements about the base space it is
given -- an operator dimension, cosine margins, a contrast temperature --
and every one of them is a statement about *this* embedder, not about the
method.  This module measures the space so those statements can be checked
or derived instead of assumed:

* **unit-norm?** -- most sentence embedders ship unit vectors; this one
  does not (mean norm 0.82).  Cosines are invariant to scale, but any loss
  that mixes raw norms (anchor L2 terms, shadow jitter) is not.
* **anisotropy** -- the mean cosine of a random pair.  A positive mean
  shifts *every* cosine up, so a fixed margin (``anchor_margin``,
  ``knn_slack``) means a different thing in a different space.
* **effective rank / explained-variance curve** -- how many dimensions the
  data actually uses.  An ``op_dim`` above the effective rank spends
  capacity on noise; one below it throws away structure.  ``auto_op_dim``
  derives the smallest dimension that keeps ``op_dim_variance`` of the
  variance -- this is also what makes ``op_dim`` meaningful for embedders
  whose dimension is *below* the hand-picked default (256), where the old
  code raised ``ValueError`` instead of working.
* **1-NN cosine** -- how much of the index is exact duplicates.  Duplicate
  nodes score 1.0 in the raw baseline for free, inflating it; the profile
  makes that inflation visible.

Everything here is deterministic (seeded) and runs on whatever device the
input lives on, so a profile can be computed in ``__init__`` of the
trainer without touching the training loop.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class EmbedderProfile:
    """Measured facts about a base-embedding space.

    All cosine statistics are computed on *unit-normalised* vectors, so
    they describe the angular geometry independent of the stored scale.
    """

    dim: int
    num_vectors: int
    #: mean / std / min / max of ||x|| over the stored vectors
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float
    #: True when the stored vectors are unit-norm up to a tight tolerance
    unit_norm: bool
    #: mean cosine of a random pair, as stored
    pair_cos_mean: float
    pair_cos_std: float
    pair_cos_p99: float
    #: same statistics after subtracting the column mean (anisotropy of a
    #: centred space is what a *directional* retrieval space should have)
    pair_cos_mean_centred: float
    pair_cos_std_centred: float
    pair_cos_p99_centred: float
    #: |mean pair cosine| as stored: the anisotropy magnitude
    anisotropy: float
    #: singular-value spectrum of the centred, norm-matrix
    effective_rank: float
    #: cumulative explained variance of the principal directions
    explained_variance: Dict[str, float]
    #: fraction of vectors whose nearest neighbour is (near-)identical:
    #: exact-duplicate mass in the index
    dup_1nn_fraction: float
    dup_1nn_cos: float

    @property
    def isotropic(self) -> bool:
        return self.anisotropy < 0.05

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _cos_stats(x: torch.Tensor, n: int, seed: int) -> tuple:
        """(mean, std, p99) of cosines between seeded random pairs of rows."""
        g = torch.Generator(device=x.device).manual_seed(seed)
        a = x[torch.randint(x.shape[0], (n,), generator=g, device=x.device)]
        b = x[torch.randint(x.shape[0], (n,), generator=g, device=x.device)]
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return float(c.mean()), float(c.std()), float(c.quantile(0.99))

    @staticmethod
    def from_embeddings(
        x: torch.Tensor,
        seed: int = 42,
        cos_pairs: int = 60000,
        svd_rows: int = 60000,
        dup_rows: int = 16384,
        dup_chunk: int = 256,
    ) -> "EmbedderProfile":
        """Measure ``x`` (``[N, D]``).  Deterministic in ``seed``.

        The 1-NN scan is ``O(rows x N)`` -- against a 243k-node index that is
        only affordable for a sample, so the duplicate statistic is measured
        on ``dup_rows`` seeded rows against the whole index.
        """
        x = x.detach().to(dtype=torch.float32)
        n, d = x.shape
        norms = x.norm(dim=-1)
        unit_norm = bool(norms.std() < 1e-3 and abs(norms.mean() - 1.0) < 1e-3)

        u = torch.nn.functional.normalize(x, dim=-1)
        mc, ms, m99 = EmbedderProfile._cos_stats(u, cos_pairs, seed)
        xc = u - u.mean(dim=0, keepdim=True)
        xc = torch.nn.functional.normalize(xc, dim=-1)
        mc_c, ms_c, m99_c = EmbedderProfile._cos_stats(xc, cos_pairs, seed + 1)

        # spectrum of the centred *stored* vectors: the scale of the stored
        # vectors is part of what the downstream L2 losses see, so the
        # variance curve is measured there, not on the unit sphere
        xs = x - x.mean(dim=0, keepdim=True)
        rows = min(n, svd_rows)
        sv = torch.linalg.svdvals(xs[:rows])
        var = sv.pow(2)
        total = float(var.sum())
        if total <= 0.0:
            ev = {str(k): 1.0 for k in (32, 64, 128, 256, 512)}
            eff_rank = 0.0
        else:
            cum = torch.cumsum(var / total, dim=0)
            ev = {}
            for k in (32, 64, 128, 256, 512):
                ev[str(k)] = 1.0 if k >= sv.shape[0] else float(cum[k - 1])
            # effective rank: exp(entropy of the normalised spectrum) --
            # the dimensionality the data *behaves* like, not its span
            p = var / total
            nz = p > 0
            eff_rank = float(torch.exp(-(p[nz] * p[nz].log()).sum()))

        # 1-NN: chunked so a 243k x 243k Gram matrix is never materialised
        rows_ids = torch.arange(n, device=x.device)
        if n > dup_rows:
            g = torch.Generator(device="cpu").manual_seed(seed + 3)
            rows_ids = torch.randperm(n, generator=g)[:int(dup_rows)].to(x.device)
        dup_cos = []
        chunk = max(1, int(dup_chunk))
        for i in range(0, rows_ids.numel(), chunk):
            ids = rows_ids[i:i + chunk]
            g = u[ids] @ u.t()
            g.scatter_(1, ids.unsqueeze(1), -1.0)
            dup_cos.append(g.max(dim=-1).values)
        nn_cos = torch.cat(dup_cos)
        dup_1nn_cos = float(nn_cos.mean())
        dup_1nn_fraction = float((nn_cos > 0.999).float().mean())

        return EmbedderProfile(
            dim=int(d),
            num_vectors=int(n),
            norm_mean=float(norms.mean()),
            norm_std=float(norms.std()),
            norm_min=float(norms.min()),
            norm_max=float(norms.max()),
            unit_norm=unit_norm,
            pair_cos_mean=mc,
            pair_cos_std=ms,
            pair_cos_p99=m99,
            pair_cos_mean_centred=mc_c,
            pair_cos_std_centred=ms_c,
            pair_cos_p99_centred=m99_c,
            anisotropy=abs(mc),
            effective_rank=eff_rank,
            explained_variance=ev,
            dup_1nn_fraction=dup_1nn_fraction,
            dup_1nn_cos=dup_1nn_cos,
        )

    def op_dim_for_variance(self, target: float, cap: int) -> int:
        """Smallest ``d <= cap`` whose principal directions keep ``target``
        of the variance, per the measured explained-variance curve.

        The curve is sampled at 32/64/128/256/512, so this is exact up to
        the coarsest grid point below the answer -- cheap and stable, which
        is the point: an *operator* dimension only needs to be the right
        order of magnitude, not the exact eigen-cutoff.
        """
        if target <= 0.0:
            target = 1.0
        if target >= 1.0:
            return min(cap, self.dim)
        # numeric order, not lexicographic: '128' < '32' as strings
        for k in sorted(self.explained_variance, key=int):
            if int(k) > cap:
                continue
            if self.explained_variance[str(k)] >= target:
                return int(k)
        # nothing below the cap reaches the target: take the largest that
        # still fits -- the operator may never be wider than the space
        return min(cap, self.dim)


def embedder_profile(
    x: torch.Tensor,
    seed: int = 42,
    cos_pairs: int = 60000,
    svd_rows: int = 60000,
    dup_rows: int = 16384,
) -> EmbedderProfile:
    """Convenience wrapper around :meth:`EmbedderProfile.from_embeddings`."""
    return EmbedderProfile.from_embeddings(
        x, seed=seed, cos_pairs=cos_pairs, svd_rows=svd_rows,
        dup_rows=dup_rows,
    )


def auto_op_dim(
    profile: EmbedderProfile,
    target: float = 0.95,
    cap: Optional[int] = None,
) -> int:
    """Resolve an automatic ``op_dim`` from a profile.

    ``cap`` bounds the dimension (defaults to the space dimension), so an
    embedder whose *whole* dimension is smaller than the hand-picked
    default still yields a usable operator instead of raising.
    """
    cap = profile.dim if cap is None else min(int(cap), profile.dim)
    d = profile.op_dim_for_variance(target, cap)
    # an operator needs at least a couple of dimensions to be a rotation
    # of anything; and it must stay strictly below the base dimension for
    # the PCA chart to exist (base_to_op has to be a real projection)
    d = max(2, d)
    if d >= profile.dim:
        d = max(2, profile.dim // 2)
    return d
