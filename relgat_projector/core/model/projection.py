import torch
import torch.nn as nn

from typing import Optional


class ProjectionHead(nn.Module):
    """
    Projection head:
    - num_layers == 0:
        - Identity (if in_dim == out_dim),
        - otherwise Linear(in_dim, out_dim)
    - num_layers == 1:
        - Linear(in_dim, out_dim)
    - num_layers >= 2:
        - MLP with (num_layers - 1) layers in/out=in_dim
        and Linear(..., out_dim) on the end

    Each layer: Linear -> GELU -> LayerNorm.
    Last layer without activation and normalization
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.net = None

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None and hidden_dim > 0 else in_dim
        )

        self.num_layers = max(0, int(num_layers))

        self.dropout = (
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        )

        self.__prepare_net()

    def __prepare_net(self):

        if self.num_layers == 0 and self.in_dim == self.out_dim:
            self.net = nn.Identity()
        elif self.num_layers == 0 or self.num_layers == 1:
            self.net = nn.Linear(self.in_dim, self.out_dim, bias=False)
        else:
            layers = [
                nn.Linear(self.in_dim, self.hidden_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            ]
            for _ in range(self.num_layers - 2):
                layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                )
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.Linear(self.hidden_dim, self.out_dim, bias=False))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.dropout(x)
        return x
