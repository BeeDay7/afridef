"""Models: GraphSAGE, GAT, and MLP/XGBoost baselines.

GraphSAGE is the default backbone for the adversarially-trained classifier
(paper §4.2). GAT is the ablation backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGEClassifier(nn.Module):
    """Vanilla GraphSAGE for node-level (or edge-level via projection) fraud
    classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.head(x).squeeze(-1)


class GATClassifier(nn.Module):
    """GAT ablation backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return self.head(x).squeeze(-1)


def build_model(name: str, in_dim: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "graphsage":
        return GraphSAGEClassifier(in_dim, **kwargs)
    if name == "gat":
        return GATClassifier(in_dim, **kwargs)
    raise ValueError(f"unknown model: {name}")
