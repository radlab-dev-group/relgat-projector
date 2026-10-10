import torch
import torch.nn as nn

from typing import Optional

from relgat_projector.inference.inference_i import InferenceI


class IterativeRelGATInference(InferenceI):
    """
    Discrete iterative propagation using a trained RelGAT.
    Each iteration:
      H <- gamma * base + (1 - gamma) * GAT(H),
    with optional norm clipping and projection back to the input dimension.
    """

    def __init__(
        self,
        model_or_layers: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        iters: int = 4,
        gamma: float = 0.3,
        clip_norm: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(edge_index=edge_index, edge_type=edge_type, device=device)
        self.iters = int(iters)
        self.gamma = float(gamma)
        self.clip_norm = clip_norm

        # Resolve RelGAT handles
        self.gat_single = getattr(model_or_layers, "gat", None)
        self.gat_layers = getattr(model_or_layers, "gat_layers", None)
        self.act = getattr(model_or_layers, "act", None)
        if self.gat_single is None and self.gat_layers is None:
            # Maybe it's a single RelGATLayer
            if hasattr(model_or_layers, "forward") and hasattr(
                model_or_layers, "heads"
            ):
                self.gat_single = model_or_layers
            else:
                raise ValueError(
                    "Unsupported RelGAT container – expected "
                    ".gat, .gat_layers, or a RelGATLayer"
                )

        self.eval()

    def eval(self):
        if self.gat_single is not None and hasattr(self.gat_single, "eval"):
            self.gat_single.eval()
        if self.gat_layers is not None:
            for g in self.gat_layers:
                g.eval()
        if self.act is not None and hasattr(self.act, "eval"):
            self.act.eval()

    @torch.no_grad()
    def _gat_forward(self, H: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def run(self, base_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        base_emb : torch.Tensor [N, D]
            Base node embeddings.

        Returns
        -------
        torch.Tensor [N, D]
            Embeddings after K discrete GAT steps with EMA stabilization.
        """
        assert base_emb.dim() == 2, "base_emb must be [N, D]"
        H = base_emb.to(self.device)
        base = H.clone()

        # Detect output dim of GAT and, if needed, define a projection back to D.
        D_in = H.size(1)
        D_out_probe = self._gat_forward(H[:1, :]).size(1)
        proj_back = None
        if D_out_probe != D_in:
            proj_back = nn.Linear(D_out_probe, D_in, bias=False).to(self.device)
            with torch.no_grad():
                nn.init.xavier_uniform_(proj_back.weight)

        for _ in range(self.iters):
            G = self._gat_forward(H)
            if proj_back is not None:
                G = proj_back(G)

            # EMA blend with the base embedding
            H = self.gamma * base + (1.0 - self.gamma) * G

            # Optional norm clipping for stability
            if self.clip_norm is not None and self.clip_norm > 0:
                norms = H.norm(dim=1, keepdim=True).clamp_min(1e-8)
                H = H / norms.clamp_max(self.clip_norm) * self.clip_norm

        return H
