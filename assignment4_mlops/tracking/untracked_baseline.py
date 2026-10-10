"""
Untracked Baseline Script for Assignment 4.

Limitations of this untracked baseline:
1. No parameter provenance: Hyperparameters (lr, dropout, hidden_dim) are hardcoded.
   If someone changes a number, there is no audit log or historical record of what worked.
2. Silent drift & lack of reproducibility: No random seed logging or dataset versioning.
   Re-running the script may yield different metrics with no way to explain the discrepancy.
3. No artifact versioning: Weights are either overwritten or printed to stdout.
   Downstream deployment cannot trace which code commit generated the running weights.
4. No gatekeeper: Regressive models can be deployed into production without validation checks.
"""

import torch
import torch.nn.functional as F
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment2_architectures.models import GCNClassifier

dataset, data = load_cora_dataset()
model = GCNClassifier(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Bare uninstrumented training loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(data.x, data.edge_index)[data.test_mask].argmax(dim=-1)
    acc = (preds == data.y[data.test_mask]).float().mean().item()

print(f"[UNTRACKED BASELINE] Final Test Accuracy: {acc:.2%}")
print("Limitation: No parameters, metrics, or weights were logged or packaged.")
