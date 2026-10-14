"""
Graph Structure and Embedding Drift Detection Module (Assignment 4).

Monitors:
1. Graph Structure Statistic: Average Node Degree (Threshold: 30% relative deviation)
2. Embedding Statistic: Cosine Distance from Reference Centroid (Threshold: > 0.20)

Demonstrates:
- Normal case (in-distribution snapshot)
- Drifted case (distribution shift in citations or vocabulary)
- Theoretical rationale for why raw coordinates across different model versions cannot be directly compared.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment4_mlops.app.main import state, load_production_model


class GNNDriftDetector:
    def __init__(self, degree_threshold: float = 0.30, embedding_threshold: float = 0.20):
        self.degree_threshold = degree_threshold
        self.embedding_threshold = embedding_threshold

        # Compute baseline reference statistics on training data
        load_production_model()
        _, data = load_cora_dataset()
        self.ref_avg_degree = data.edge_index.shape[1] / data.num_nodes

        # Compute baseline embedding centroid using fixed production model
        with torch.no_grad():
            ref_emb = state["model"].embed(data.x, data.edge_index)
            # Reference centroid normalized to unit sphere
            self.ref_centroid = F.normalize(ref_emb.mean(dim=0, keepdim=True), p=2, dim=-1)

    def check_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor, snapshot_name: str):
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # 1. Structural Statistic: Average Degree
        batch_avg_degree = (num_edges / num_nodes) if num_nodes > 0 else 0.0
        degree_shift = abs(batch_avg_degree - self.ref_avg_degree) / self.ref_avg_degree
        degree_alert = degree_shift > self.degree_threshold

        # 2. Embedding Statistic: Cosine Distance from Reference Centroid
        with torch.no_grad():
            batch_emb = state["model"].embed(x, edge_index)
            batch_centroid = F.normalize(batch_emb.mean(dim=0, keepdim=True), p=2, dim=-1)
            cosine_similarity = (self.ref_centroid * batch_centroid).sum().item()
            cosine_distance = 1.0 - cosine_similarity

        embedding_alert = cosine_distance > self.embedding_threshold
        overall_drift = degree_alert or embedding_alert

        print(f"\n--- Drift Assessment: Snapshot '{snapshot_name}' ---")
        print(f"Graph Nodes / Edges:       {num_nodes} / {num_edges}")
        print(f"Average Degree:            {batch_avg_degree:.2f} (Baseline: {self.ref_avg_degree:.2f}) -> Relative Shift: {degree_shift:.1%}")
        print(f"Structural Drift Alert:    {'[ALERT: DEGREE SHIFT]' if degree_alert else '[NORMAL]'}")
        print(f"Embedding Cosine Distance: {cosine_distance:.4f} (Threshold: {self.embedding_threshold:.2f})")
        print(f"Embedding Drift Alert:     {'[ALERT: EMBEDDING SHIFT]' if embedding_alert else '[NORMAL]'}")
        print(f"Final Decision:            {'*** DRIFT DETECTED ***' if overall_drift else 'NO DRIFT DETECTED'}")

        return {
            "snapshot": snapshot_name,
            "degree_shift": degree_shift,
            "cosine_distance": cosine_distance,
            "drift_detected": overall_drift,
        }


def run_drift_demonstration():
    print("=" * 80)
    print("MONITORING: GRAPH-STRUCTURE & EMBEDDING DRIFT DETECTION")
    print("=" * 80)

    detector = GNNDriftDetector()
    _, base_data = load_cora_dataset()

    # --- Scenario 1: Normal In-Distribution Snapshot (e.g. Validation split) ---
    normal_x = base_data.x[:500]
    mask_normal = (base_data.edge_index[0] < 500) & (base_data.edge_index[1] < 500)
    normal_edges = base_data.edge_index[:, mask_normal]
    detector.check_snapshot(normal_x, normal_edges, snapshot_name="Normal_Production_Batch")

    # --- Scenario 2: Severe Structural & Semantic Drift ---
    # A batch where papers are almost completely unlinked (low degree) and vocabulary is shifted
    drift_x = base_data.x[:500] * 0.05 + torch.randn(500, base_data.x.shape[1]) * 0.5
    # Sparse edge index (only 20 edges for 500 nodes)
    drift_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    detector.check_snapshot(drift_x, drift_edges, snapshot_name="Drifted_Out_Of_Distribution_Batch")

    print("\n" + "=" * 80)
    print("EXPLANATION: WHY RAW EMBEDDINGS ACROSS DIFFERENT MODEL VERSIONS CANNOT BE COMPARED:")
    print("1. Rotational & Permutation Invariance: Deep neural networks have many equivalent global minima.")
    print("   Two training runs produce weight matrices that are rotations or permutations of one another.")
    print("2. Orthogonal Misalignment: Even if both models capture the identical semantic manifold, the raw")
    print("   latent basis vectors (e.g. coordinate index 5) do not correspond to the same underlying feature.")
    print("3. Therefore, drift monitoring across versions must use relative topological metrics or aligned")
    print("   similarity metrics (e.g., Procrustes distance, Centered Kernel Alignment - CKA) rather than raw Euclidean coordinates.")
    print("=" * 80)


if __name__ == "__main__":
    run_drift_demonstration()
