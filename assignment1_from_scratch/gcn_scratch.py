"""
GCN from Scratch using Pure PyTorch Tensor Operations (Assignment 1).

Requirements addressed:
- Self-loops: A_tilde = A + I
- Degree normalization: D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
- Message passing via tensor multiplication
- Two-layer GCN architecture without any ready-made GNN packages
- Complete training loop (loss, backward, optimizer step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Computes symmetric normalized adjacency matrix:
    A_hat = D_tilde^(-1/2) * (A + I) * D_tilde^(-1/2)
    """
    num_nodes = adj.shape[0]

    # 1. Add self-loops
    adj_tilde = adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)

    # 2. Compute degree of each node: d_i = sum_j(A_tilde_ij)
    deg = torch.sum(adj_tilde, dim=1)

    # 3. Compute D^(-1/2)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 4. Symmetric normalization using broadcasting:
    # A_norm[i, j] = deg_inv_sqrt[i] * adj_tilde[i, j] * deg_inv_sqrt[j]
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj_tilde * deg_inv_sqrt.unsqueeze(0)
    return norm_adj


class GCNLayerScratch(nn.Module):
    """
    Single GCN Layer implemented purely with tensor operations.
    Operation: H_out = norm_adj @ H_in @ W + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Trainable weight matrix
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Glorot (Xavier) uniform initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # 1. Feature transformation: X @ W
        support = torch.matmul(x, self.weight)

        # 2. Neighborhood message aggregation: norm_adj @ support
        output = torch.matmul(norm_adj, support)

        # 3. Add bias if present
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN2LayerScratch(nn.Module):
    """
    Two-layer GCN implemented purely from scratch.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.layer1 = GCNLayerScratch(in_features, hidden_features)
        self.layer2 = GCNLayerScratch(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Precompute normalized adjacency matrix
        norm_adj = normalize_adjacency(adj)

        # First GCN Layer + ReLU + Dropout
        h1 = self.layer1(x, norm_adj)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Second GCN Layer (Raw Logits)
        out = self.layer2(h1, norm_adj)
        return out


def train_gcn(model, data, epochs: int = 150, lr: float = 0.01, weight_decay: float = 5e-4):
    """
    Standard training loop for GCN.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x = data["x"]
    adj = data["adj"]
    y = data["y"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    history = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x, adj)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (preds[test_mask] == y[test_mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

    return history
