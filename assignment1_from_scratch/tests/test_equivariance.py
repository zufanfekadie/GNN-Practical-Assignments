"""
Test 2 for Assignment 1:
Verify permutation equivariance: consistently reordering nodes, features,
and connections produces correspondingly reordered predictions.
"""

import torch
from assignment1_from_scratch.dataset import generate_synthetic_graph
from assignment1_from_scratch.gcn_scratch import GCN2LayerScratch


def test_permutation_equivariance():
    torch.manual_seed(42)

    # 1. Load synthetic graph
    data = generate_synthetic_graph(num_nodes=120, seed=42)
    x = data["x"]
    adj = data["adj"]
    num_nodes = x.shape[0]

    # 2. Instantiate GCN model in eval mode
    model = GCN2LayerScratch(in_features=3, hidden_features=16, out_features=2, dropout=0.0)
    model.eval()

    with torch.no_grad():
        original_preds = model(x, adj)

    # 3. Create a random permutation of nodes
    perm = torch.randperm(num_nodes)

    # Permute features: x_perm[i] = x[perm[i]]
    x_perm = x[perm]

    # Permute adjacency matrix: adj_perm = P * adj * P^T
    # Node i in permuted graph corresponds to node perm[i] in original
    adj_perm = adj[perm][:, perm]

    # 4. Run forward pass on permuted graph
    with torch.no_grad():
        perm_preds = model(x_perm, adj_perm)

    # 5. Check if perm_preds == original_preds[perm]
    expected_perm_preds = original_preds[perm]
    max_diff = torch.abs(perm_preds - expected_perm_preds).max().item()

    print(f"Max difference between permuted output and reordered original: {max_diff:.8e}")
    assert torch.allclose(perm_preds, expected_perm_preds, atol=1e-5), "Permutation equivariance violated!"
    print("SUCCESS: Model is strictly permutation-equivariant.")


if __name__ == "__main__":
    test_permutation_equivariance()
