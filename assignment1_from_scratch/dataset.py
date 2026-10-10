"""
Synthetic Graph Dataset Generator for Assignment 1.

Requirements addressed:
- At least 100 nodes (default: 120 nodes).
- Exactly 3 numerical features per node.
- Exactly 2 target classes (0 and 1).
- Graph connections provide useful information beyond individual node features
  (engineered via homophilic community structure + overlapping feature distributions).
- Target labels are strictly isolated from input features.
- Fixed, reproducible train/validation/test split masks.
"""

import torch
import numpy as np


def generate_synthetic_graph(
    num_nodes: int = 120,
    num_features: int = 3,
    p_same_class: float = 0.15,
    p_diff_class: float = 0.01,
    seed: int = 42,
):
    """
    Generates a synthetic graph where graph structure resolves feature ambiguity.

    Parameters:
    -----------
    num_nodes : int (>= 100)
    num_features : int (strictly 3)
    p_same_class : float, edge connection probability within the same class
    p_diff_class : float, edge connection probability across different classes
    seed : int, reproducibility seed

    Returns:
    --------
    data_dict containing:
        - 'x': Tensor of shape (num_nodes, 3), node features
        - 'adj': Tensor of shape (num_nodes, num_nodes), binary symmetric adjacency matrix
        - 'y': Tensor of shape (num_nodes,), class labels (0 or 1)
        - 'train_mask', 'val_mask', 'test_mask': Boolean tensors
    """
    assert num_nodes >= 100, "Requirement: Graph must have at least 100 nodes."
    assert num_features == 3, "Requirement: Exactly three numerical features per node."

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Generate balanced classes: 0 and 1
    half = num_nodes // 2
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[half:] = 1

    # 2. Generate overlapping features (so features alone are ambiguous: ~60-65% accuracy)
    # Class 0: centered around [0.0, 0.0, 0.0] with std 1.0
    # Class 1: centered around [0.6, 0.6, 0.6] with std 1.0
    feat_class_0 = torch.randn(half, num_features) * 1.0 + 0.0
    feat_class_1 = torch.randn(num_nodes - half, num_features) * 1.0 + 0.6
    x = torch.cat([feat_class_0, feat_class_1], dim=0)

    # 3. Generate graph connections via homophily (Stochastic Block Model)
    # Nodes connect frequently to their own class, rarely to the opposite class.
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = p_same_class if y[i] == y[j] else p_diff_class
            if np.random.rand() < prob:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # Undirected

    # 4. Create train / validation / test splits (60% train, 20% val, 20% test)
    indices = np.random.permutation(num_nodes)
    train_end = int(0.6 * num_nodes)
    val_end = int(0.8 * num_nodes)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return {
        "x": x,
        "adj": adj,
        "y": y,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


if __name__ == "__main__":
    data = generate_synthetic_graph()
    x, adj, y = data["x"], data["adj"], data["y"]
    print("Dataset verification successful:")
    print(f"- Nodes: {x.shape[0]}")
    print(f"- Features per node: {x.shape[1]}")
    print(f"- Target classes: {torch.unique(y).tolist()}")
    print(f"- Total undirected edges: {int(adj.sum().item() / 2)}")
    print(f"- Train / Val / Test split counts: {data['train_mask'].sum().item()} / {data['val_mask'].sum().item()} / {data['test_mask'].sum().item()}")
