"""
Edge Splitting, Negative Sampling, and Zero-Leakage Validation for Assignment 3.

Guarantees:
1. Positive edges split into Train (85%), Val (5%), Test (10%).
2. Both directions (u, v) and (v, u) are kept in the same split.
3. Message-passing graph uses strictly Train positive edges.
4. Negative pairs exclude self-loops and any observed edges across all splits.
5. Automated assertions mathematically prove zero leakage into the message-passing graph.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from assignment2_architectures.dataset_cora import load_cora_dataset


def create_link_prediction_splits(val_ratio: float = 0.05, test_ratio: float = 0.10, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, data = load_cora_dataset()
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. Extract unique undirected edges where u < v
    mask = edge_index[0] < edge_index[1]
    unique_edges = edge_index[:, mask].t().cpu().numpy()  # shape (E_unique, 2)
    num_unique = len(unique_edges)

    # 2. Shuffle and split positive edges
    perm = np.random.permutation(num_unique)
    num_val = int(val_ratio * num_unique)
    num_test = int(test_ratio * num_unique)
    num_train = num_unique - num_val - num_test

    train_pos = unique_edges[perm[:num_train]]
    val_pos = unique_edges[perm[num_train : num_train + num_val]]
    test_pos = unique_edges[perm[num_train + num_val :]]

    # 3. Build sets of all observed positive edges (both directions) for fast negative sampling
    all_positive_set = set()
    for u, v in unique_edges:
        all_positive_set.add((u, v))
        all_positive_set.add((v, u))

    # 4. Negative sampling helper
    def sample_negatives(count, existing_neg_set):
        negatives = []
        while len(negatives) < count:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in all_positive_set or pair in existing_neg_set:
                continue
            existing_neg_set.add(pair)
            negatives.append(list(pair))
        return np.array(negatives)

    used_neg_set = set()
    train_neg = sample_negatives(len(train_pos), used_neg_set)
    val_neg = sample_negatives(len(val_pos), used_neg_set)
    test_neg = sample_negatives(len(test_pos), used_neg_set)

    # 5. Construct the strictly leak-free message passing graph:
    # Contains ONLY train positive edges + their reverse copies!
    train_pos_tensor = torch.tensor(train_pos, dtype=torch.long)
    train_pos_reversed = torch.stack([train_pos_tensor[:, 1], train_pos_tensor[:, 0]], dim=1)
    train_message_passing_edges = torch.cat([train_pos_tensor, train_pos_reversed], dim=0).t()

    splits = {
        "x": data.x,
        "num_nodes": num_nodes,
        "train_msg_edge_index": train_message_passing_edges,
        "train_pos": torch.tensor(train_pos, dtype=torch.long),
        "train_neg": torch.tensor(train_neg, dtype=torch.long),
        "val_pos": torch.tensor(val_pos, dtype=torch.long),
        "val_neg": torch.tensor(val_neg, dtype=torch.long),
        "test_pos": torch.tensor(test_pos, dtype=torch.long),
        "test_neg": torch.tensor(test_neg, dtype=torch.long),
        # Deliberately leaky edge index containing ALL edges (for leakage experiment)
        "leaky_msg_edge_index": edge_index,
    }

    # Run automated zero-leakage verification
    verify_zero_leakage(splits)
    return splits


def verify_zero_leakage(splits):
    """
    Automated check proving held-out links and their reverse copies
    are strictly absent from the message-passing graph.
    """
    msg_edges = set()
    msg_tensor = splits["train_msg_edge_index"].t().tolist()
    for u, v in msg_tensor:
        msg_edges.add((u, v))

    for split_name in ["val_pos", "test_pos"]:
        edges = splits[split_name].tolist()
        leaked = []
        for u, v in edges:
            if (u, v) in msg_edges or (v, u) in msg_edges:
                leaked.append((u, v))

        assert len(leaked) == 0, f"DATA LEAKAGE DETECTED! {len(leaked)} edges from {split_name} found in message-passing graph!"

    print("[SUCCESS] Zero-leakage verified: 0 held-out edges exist in the message-passing graph.")


if __name__ == "__main__":
    splits = create_link_prediction_splits()
    print("Link Prediction Splits Summary:")
    print(f"- Total Nodes: {splits['num_nodes']}")
    print(f"- Train Positive Edges: {len(splits['train_pos'])} (Message-passing edges: {splits['train_msg_edge_index'].shape[1]})")
    print(f"- Val Positive / Negative:  {len(splits['val_pos'])} / {len(splits['val_neg'])}")
    print(f"- Test Positive / Negative: {len(splits['test_pos'])} / {len(splits['test_neg'])}")
