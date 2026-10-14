"""
Cora Dataset Loader and Inspection Module for Assignment 2.

Inspects:
- Total nodes, edges, features, classes.
- Node feature sparsity and semantics.
- Label distribution across standard splits (train/val/test).
- Graph topological properties (density, average degree, homophily).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_cora_dataset(root_dir: str = r"D:\gnn-practical-assignments\data"):
    """
    Downloads and caches the Cora citation dataset.
    """
    os.makedirs(root_dir, exist_ok=True)
    dataset = Planetoid(root=root_dir, name="Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    return dataset, data


def inspect_cora(data, dataset):
    """
    Prints comprehensive dataset statistics and analysis.
    """
    num_nodes = data.num_nodes
    num_edges = data.num_edges // 2  # undirected
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Class distribution
    classes, counts = torch.unique(data.y, return_counts=True)

    # Homophily calculation (fraction of edges connecting nodes with the same label)
    edge_index = data.edge_index
    same_class_edges = (data.y[edge_index[0]] == data.y[edge_index[1]]).float().sum().item()
    homophily_ratio = same_class_edges / data.edge_index.shape[1]

    print("=" * 65)
    print("CORA CITATION DATASET INSPECTION")
    print("=" * 65)
    print(f"Total Nodes (Research Papers):      {num_nodes}")
    print(f"Total Undirected Edges (Citations): {num_edges}")
    print(f"Node Features (Vocabulary Words):   {num_features}")
    print(f"Target Classes (Research Topics):   {num_classes}")
    print(f"Graph Homophily Ratio:              {homophily_ratio:.2%}")
    print(f"Average Node Degree:                {data.edge_index.shape[1] / num_nodes:.2f}")
    print("-" * 65)
    print("Splits Distribution:")
    print(f"- Training set nodes:               {data.train_mask.sum().item()} (20 per class)")
    print(f"- Validation set nodes:             {data.val_mask.sum().item()}")
    print(f"- Test set nodes:                   {data.test_mask.sum().item()}")
    print("-" * 65)
    print("Class label counts across whole graph:")
    for c, cnt in zip(classes.tolist(), counts.tolist()):
        print(f"  Class {c}: {cnt} papers")
    print("=" * 65)


if __name__ == "__main__":
    dataset, data = load_cora_dataset()
    inspect_cora(data, dataset)
