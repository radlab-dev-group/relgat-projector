import torch

import torch.nn as nn
from typing import Optional

try:
    from torch_scatter import scatter_add
except Exception:
    scatter_add = None

from relgat_llm.inference.inference_i import InferenceI


class _RelationalDrift(nn.Module):
    """
    Wrapper around a trained RelGAT to be used as a velocity field (drift) in ODE.

    Supports:
      - model with attribute .gat (single layer),
      - model with attribute .gat_layers (+ .act) (multi-layer),
      - a single RelGATLayer instance.
    """

    def __init__(
        self,
        model_or_layers: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.act = act

        self.gat_single = getattr(model_or_layers, "gat", None)
        self.gat_layers = getattr(model_or_layers, "gat_layers", None)

        if self.gat_single is not None:
            heads = self.gat_single.heads
            out_dim = self.gat_single.out_dim
        elif self.gat_layers is not None and len(self.gat_layers) > 0:
            last = self.gat_layers[-1]
            heads = last.heads
            out_dim = last.out_dim
        elif hasattr(model_or_layers, "heads") and hasattr(
            model_or_layers, "out_dim"
        ):
            # treat it as a single RelGATLayer
            self.gat_single = model_or_layers
            heads = model_or_layers.heads
            out_dim = model_or_layers.out_dim
        else:
            raise ValueError(
                "Unsupported RelGAT container – expected .gat, "
                ".gat_layers, or a RelGATLayer"
            )

        self.out_total_dim = heads * out_dim

    @torch.no_grad()
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        if self.gat_single is not None and self.gat_layers is None:
            return self.gat_single(H, self.edge_index, self.edge_type)
        if self.gat_layers is not None:
            x = H
            for li, gat in enumerate(self.gat_layers):
                x = gat(x, self.edge_index, self.edge_type)
                if self.act is not None and li < len(self.gat_layers) - 1:
                    x = self.act(x)
            return x
        return self.gat_single(H, self.edge_index, self.edge_type)


class _GraphLaplacianDiffusion(nn.Module):
    """
    Laplacian diffusion term:
      S(H) = H - mean_{u in N_in(v)} H_u

    Uses incoming edges (destination-centric accumulation).
    """

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        super().__init__()
        if scatter_add is None:
            raise RuntimeError(
                "torch_scatter is required for diffusion "
                "term (install torch-scatter)"
            )
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        dst = edge_index[1]
        deg = torch.bincount(dst, minlength=num_nodes).clamp_min(1)
        self.register_buffer("deg_in", deg.to(torch.float32))

    @torch.no_grad()
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        src, dst = self.edge_index
        neigh_sum = scatter_add(
            H[src], dst, dim=0, dim_size=self.num_nodes
        )  # [N, D]
        neigh_mean = neigh_sum / self.deg_in.unsqueeze(-1)
        return H - neigh_mean


class _RDGNODERHS(nn.Module):
    """
    ODE right-hand side:
      dH/dt = alpha * Drift(H) - beta * Diffusion(H)
    """

    def __init__(
        self,
        drift: _RelationalDrift,
        diffusion: _GraphLaplacianDiffusion,
        alpha: float,
        beta: float,
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def forward(self, t: float, H: torch.Tensor) -> torch.Tensor:
        D = self.drift(H)
        S = self.diffusion(H)
        return self.alpha * D - self.beta * S


def _rk4(rhs: _RDGNODERHS, H0: torch.Tensor, T: float, steps: int) -> torch.Tensor:
    """
    Classic fourth-order Runge–Kutta integrator (inference-only).
    """
    with torch.no_grad():
        h = H0
        dt = T / max(1, steps)
        t = 0.0
        for _ in range(max(1, steps)):
            k1 = rhs(t, h)
            k2 = rhs(t + 0.5 * dt, h + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, h + 0.5 * dt * k2)
            k4 = rhs(t + dt, h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
        return h


class RDGNODEInference(InferenceI):
    """
    RD-GNODE: continuous-time propagation using
    RelGAT-driven drift + Laplacian diffusion and RK4 solver (inference-only).
    """

    def __init__(
        self,
        model_or_layers: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.2,
        T: float = 1.0,
        steps: int = 6,
        gamma: float = 0.3,
        clip_norm: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(edge_index=edge_index, edge_type=edge_type, device=device)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.T = float(T)
        self.steps = int(steps)
        self.gamma = float(gamma)
        self.clip_norm = clip_norm

        # Build drift and diffusion
        act = getattr(model_or_layers, "act", None)
        self._drift = _RelationalDrift(
            model_or_layers=model_or_layers,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            act=act,
        )
        self._diffusion = _GraphLaplacianDiffusion(
            edge_index=self.edge_index, num_nodes=self._infer_num_nodes(edge_index)
        )
        self._rhs = _RDGNODERHS(
            self._drift, self._diffusion, self.alpha, self.beta
        ).to(self.device)

        # Optional projection to match drift output dimension
        # to Din (initialized lazily in run)
        self._proj: Optional[nn.Linear] = None

    def _infer_num_nodes(self, edge_index: torch.Tensor) -> int:
        return int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

    def _ensure_proj(self, Din: int) -> None:
        Dout = self._drift.out_total_dim
        if Dout != Din and self._proj is None:
            self._proj = nn.Linear(Dout, Din, bias=False).to(self.device)
            with torch.no_grad():
                nn.init.xavier_uniform_(self._proj.weight)

        # Monkey-patch drift.forward so it always returns [N, Din]
        if self._proj is not None:
            orig_forward = self._drift.forward

            @torch.no_grad()
            def _wrapped(H: torch.Tensor) -> torch.Tensor:
                return self._proj(orig_forward(H))

            self._drift.forward = _wrapped  # type: ignore

    @torch.no_grad()
    def run(self, base_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        base_emb : torch.Tensor [N, D]
            Base embeddings; unknown nodes can be zeros or small noise.

        Returns
        -------
        torch.Tensor [N, D]
            Final embeddings after ODE integration, norm clipping, and EMA fusion.
        """
        assert base_emb.dim() == 2, "base_emb must be [N, D]"
        base_emb = base_emb.to(self.device)
        _, Din = base_emb.shape

        self._ensure_proj(Din)

        H_T = _rk4(self._rhs, base_emb, T=self.T, steps=self.steps)

        if self.clip_norm is not None and self.clip_norm > 0:
            norms = H_T.norm(dim=1, keepdim=True).clamp_min(1e-8)
            H_T = H_T / norms.clamp_max(self.clip_norm) * self.clip_norm

        H_tilde = self.gamma * base_emb + (1.0 - self.gamma) * H_T
        return H_tilde
