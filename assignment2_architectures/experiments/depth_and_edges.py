"""
Diagnostic Experiments for Assignment 2:
1. GNN Depth vs. Oversmoothing (Cosine Similarity collapse)
2. Edge Removal Ablation (0% to 100% edge drop vs MLP control)
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment2_architectures.models import GCNClassifier, MLPClassifier, GraphSAGEClassifier, GATClassifier
from assignment2_architectures.train_benchmark import train_single_run


def compute_edge_cosine_similarity(embeddings: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    Computes average cosine similarity between connected node pairs (oversmoothing metric).
    """
    # Normalize node embeddings to unit sphere
    norm_emb = F.normalize(embeddings, p=2, dim=-1)
    src_emb = norm_emb[edge_index[0]]
    dst_emb = norm_emb[edge_index[1]]
    cos_sim = (src_emb * dst_emb).sum(dim=-1).mean().item()
    return cos_sim


def run_depth_experiment(depths=(2, 4, 8, 16)):
    dataset, data = load_cora_dataset()
    in_dim, out_dim = dataset.num_features, dataset.num_classes

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: GNN DEPTH VS. OVERSMOOTHING (GCN)")
    print("=" * 70)
    print(f"{'Depth (Layers)':<15} | {'Test Accuracy':<15} | {'Avg Edge Cosine Similarity':<25}")
    print("-" * 70)

    depth_results = {"depths": list(depths), "accuracies": [], "similarities": []}

    for depth in depths:
        torch.manual_seed(42)
        model = GCNClassifier(in_dim, 64, out_dim, num_layers=depth, dropout=0.5)
        metrics = train_single_run(model, data, epochs=150)

        # Compute cosine similarity of best checkpoint embeddings
        model.eval()
        with torch.no_grad():
            emb = model.embed(data.x, data.edge_index)
            sim = compute_edge_cosine_similarity(emb, data.edge_index)

        depth_results["accuracies"].append(metrics["test_acc"])
        depth_results["similarities"].append(sim)
        print(f"{depth:<15} | {metrics['test_acc']:<15.2%} | {sim:<25.4f}")

    print("=" * 70)
    return depth_results


def run_edge_removal_experiment(drop_ratios=(0.0, 0.25, 0.50, 0.75, 1.0)):
    dataset, data = load_cora_dataset()
    in_dim, out_dim = dataset.num_features, dataset.num_classes

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: EDGE REMOVAL SWEEP (VS MLP CONTROL)")
    print("=" * 70)
    print(f"{'Drop Ratio':<12} | {'MLP (Control)':<14} | {'GCN':<12} | {'GraphSAGE':<12} | {'GAT':<12}")
    print("-" * 70)

    num_edges = data.edge_index.shape[1]
    edge_results = {"drop_ratios": list(drop_ratios), "MLP": [], "GCN": [], "GraphSAGE": [], "GAT": []}

    # MLP is independent of edges
    torch.manual_seed(42)
    mlp = MLPClassifier(in_dim, 64, out_dim)
    mlp_acc = train_single_run(mlp, data, epochs=150)["test_acc"]

    for ratio in drop_ratios:
        # Create pruned graph
        data_pruned = copy.copy(data)
        if ratio == 1.0:
            # Completely remove all edges (empty edge index)
            data_pruned.edge_index = torch.empty((2, 0), dtype=torch.long)
        elif ratio > 0.0:
            keep_count = int(num_edges * (1.0 - ratio))
            perm = torch.randperm(num_edges)[:keep_count]
            data_pruned.edge_index = data.edge_index[:, perm]

        # Train models on pruned graph
        torch.manual_seed(42)
        gcn = GCNClassifier(in_dim, 64, out_dim)
        gcn_acc = train_single_run(gcn, data_pruned, epochs=150)["test_acc"]

        torch.manual_seed(42)
        sage = GraphSAGEClassifier(in_dim, 64, out_dim)
        sage_acc = train_single_run(sage, data_pruned, epochs=150)["test_acc"]

        torch.manual_seed(42)
        gat = GATClassifier(in_dim, 64, out_dim, heads=4)
        gat_acc = train_single_run(gat, data_pruned, epochs=150)["test_acc"]

        edge_results["MLP"].append(mlp_acc)
        edge_results["GCN"].append(gcn_acc)
        edge_results["GraphSAGE"].append(sage_acc)
        edge_results["GAT"].append(gat_acc)

        print(f"{ratio * 100:>9.0f}%   | {mlp_acc:<14.2%} | {gcn_acc:<12.2%} | {sage_acc:<12.2%} | {gat_acc:<12.2%}")

    print("=" * 70)
    return edge_results


if __name__ == "__main__":
    run_depth_experiment()
    run_edge_removal_experiment()
