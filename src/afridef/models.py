"""Models: EdgeGraphSAGE and GAT-variant backbones for AfriDef.

Pure-PyTorch implementation — no torch_geometric required.

EdgeGraphSAGE is the default backbone for the adversarially-trained
classifier (paper §4.2).  GATEdge is provided as an ablation backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-PyTorch building blocks
# ═══════════════════════════════════════════════════════════════════════════════
class SAGEConv(nn.Module):
    """Mean-pooling SAGEConv layer (no torch_geometric required).

    Aggregates neighbour embeddings via scatter-add + degree normalisation,
    then concatenates with self embedding and projects.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        deg = torch.bincount(dst, minlength=x.size(0)).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return self.lin(torch.cat([x, agg], dim=1))


class GATConv(nn.Module):
    """Single-head graph attention layer (no torch_geometric required).

    Computes attention weights α_{ij} = softmax(LeakyReLU(a^T [Wh_i ‖ Wh_j]))
    and returns the attended neighbourhood mean.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(out_dim * 2, 1, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h   = self.W(x)
        e   = torch.cat([h[src], h[dst]], dim=-1)
        attn = F.leaky_relu(self.att(e), negative_slope=0.2).squeeze(-1)
        # Softmax per destination node
        attn_max = torch.zeros(x.size(0), device=x.device)
        attn_max.scatter_reduce_(0, dst, attn, reduce="amax", include_self=True)
        attn_exp = torch.exp(attn - attn_max[dst])
        attn_sum = torch.zeros(x.size(0), device=x.device)
        attn_sum.scatter_add_(0, dst, attn_exp)
        alpha = attn_exp / (attn_sum[dst] + 1e-8)
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.size(1)),
                         alpha.unsqueeze(1) * h[src])
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Edge-level GraphSAGE  (primary AfriDef backbone)
# ═══════════════════════════════════════════════════════════════════════════════
class EdgeGraphSAGE(nn.Module):
    """GraphSAGE encoder + per-transaction edge classifier.

    For each transaction edge (u→v):
        score = MLP( h_u ‖ h_v ‖ edge_feat )

    This is the model trained by TRADES (Layer 2) and evaluated by the
    Stackelberg outer loop (Layer 3).

    Parameters
    ----------
    node_in_dim : int   — node feature dimension (4 in our PaySim setup)
    edge_in_dim : int   — edge feature dimension (4 in our PaySim setup)
    hidden_dim  : int   — SAGEConv hidden width  (default 128)
    num_layers  : int   — number of SAGEConv layers (default 2)
    dropout     : float — dropout rate in the edge head (default 0.5)
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        s, d = edge_index
        if mask is not None:
            s, d, edge_attr = s[mask], d[mask], edge_attr[mask]
        return self.edge_head(
            torch.cat([h[s], h[d], edge_attr], dim=-1)
        ).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge-level GAT  (ablation backbone)
# ═══════════════════════════════════════════════════════════════════════════════
class EdgeGAT(nn.Module):
    """GAT encoder + per-transaction edge classifier (ablation).

    Same interface as EdgeGraphSAGE — swap in via configs/default.yaml
    ``gnn.backbone: gat``.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_in_dim, hidden_dim, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, dropout=dropout))

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        s, d = edge_index
        if mask is not None:
            s, d, edge_attr = s[mask], d[mask], edge_attr[mask]
        return self.edge_head(
            torch.cat([h[s], h[d], edge_attr], dim=-1)
        ).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(
    backbone: str,
    node_in_dim: int,
    edge_in_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5,
) -> nn.Module:
    """Instantiate a backbone by name.

    Parameters
    ----------
    backbone : "graphsage" | "gat"
    """
    name = backbone.lower()
    kwargs = dict(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    if name == "graphsage":
        return EdgeGraphSAGE(**kwargs)
    if name == "gat":
        return EdgeGAT(**kwargs)
    raise ValueError(f"Unknown backbone '{backbone}'. Choose 'graphsage' or 'gat'.")
