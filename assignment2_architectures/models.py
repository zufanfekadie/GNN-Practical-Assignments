"""
Model Architectures for Assignment 2.

Implements custom model classes using PyTorch Geometric layers:
1. MLPClassifier (ignores graph structure)
2. GCNClassifier (Kipf & Welling GCNConv)
3. GraphSAGEClassifier (Hamilton et al. SAGEConv)
4. GATClassifier (Veličković et al. multi-head GATConv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class MLPClassifier(nn.Module):
    """
    MLP Baseline: Classifies nodes using word features only.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = F.relu(layer(h))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.layers[-1](h)
        return out


class GCNClassifier(nn.Module):
    """
    GCN Architecture with configurable depth.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out


class GraphSAGEClassifier(nn.Module):
    """
    GraphSAGE Architecture with configurable depth and mean aggregation.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr="mean"))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="mean"))

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h, edge_index))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out


class GATClassifier(nn.Module):
    """
    Graph Attention Network with multi-head attention.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5, heads: int = 8):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Hidden dimension per head
        head_dim = max(1, hidden_channels // heads)
        actual_hidden = head_dim * heads

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, head_dim, heads=heads, concat=True, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(actual_hidden, head_dim, heads=heads, concat=True, dropout=dropout))
            # Final output layer uses single head averaging for classification logits
            self.convs.append(GATConv(actual_hidden, out_channels, heads=1, concat=False, dropout=dropout))

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = F.elu(conv(h, edge_index))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index)
            h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out
