"""
MLP Baseline for Assignment 1.

Operates purely on node features without utilizing any graph connectivity.
Uses the identical hidden dimensions, layer structure, and training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h)
        return out


def train_mlp(model, data, epochs: int = 150, lr: float = 0.01, weight_decay: float = 5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x = data["x"]
    y = data["y"]
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    history = {"train_loss": [], "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (preds[test_mask] == y[test_mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

    return history
