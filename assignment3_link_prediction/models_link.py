"""
Link Prediction Models and Decoders for Assignment 3.

Components:
- GCNEncoder: 2-layer GCN producing node embeddings Z
- DotProductDecoder: s(u, v) = sigmoid(z_u^T * z_v)
- SymmetricMLPDecoder: s(u, v) = sigmoid(MLP([z_u * z_v || z_u + z_v]))
  Guarantees mathematical symmetry s(u, v) == s(v, u).
- CommonNeighborsBaseline: Heuristic counting shared 1-hop training neighbors.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        return z


class DotProductDecoder(nn.Module):
    """
    Computes link probability using inner product: s(u, v) = sigmoid(z_u^T * z_v)
    """
    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        # edge_pairs: shape (num_pairs, 2)
        u_emb = z[edge_pairs[:, 0]]
        v_emb = z[edge_pairs[:, 1]]
        scores = (u_emb * v_emb).sum(dim=-1)
        return torch.sigmoid(scores)


class SymmetricMLPDecoder(nn.Module):
    """
    Learned non-linear decoder that is strictly permutation-invariant for undirected pairs:
    s(u, v) = sigmoid(MLP([z_u * z_v || z_u + z_v]))
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 32):
        super().__init__()
        # Input features: emb_dim (product) + emb_dim (sum) = 2 * emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        u_emb = z[edge_pairs[:, 0]]
        v_emb = z[edge_pairs[:, 1]]

        # Permutation-invariant representations
        prod_term = u_emb * v_emb
        sum_term = u_emb + v_emb

        features = torch.cat([prod_term, sum_term], dim=-1)
        logits = self.mlp(features).squeeze(-1)
        return torch.sigmoid(logits)


def common_neighbors_score(train_msg_edge_index: torch.Tensor, num_nodes: int, edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    Heuristic baseline counting shared neighbors in the training graph.
    """
    # Build adjacency sets
    adj_sets = [set() for _ in range(num_nodes)]
    edges = train_msg_edge_index.t().cpu().numpy()
    for u, v in edges:
        adj_sets[u].add(v)

    scores = []
    pairs = edge_pairs.cpu().numpy()
    for u, v in pairs:
        shared = len(adj_sets[u].intersection(adj_sets[v]))
        scores.append(shared)

    return torch.tensor(scores, dtype=torch.float32)


def test_decoder_symmetry():
    torch.manual_seed(42)
    decoder = SymmetricMLPDecoder(emb_dim=16)
    z = torch.randn(10, 16)

    # Pair (3, 7) vs Pair (7, 3)
    pair_forward = torch.tensor([[3, 7]], dtype=torch.long)
    pair_reversed = torch.tensor([[7, 3]], dtype=torch.long)

    score_fwd = decoder(z, pair_forward).item()
    score_rev = decoder(z, pair_reversed).item()

    print(f"MLP Decoder Score (u, v): {score_fwd:.8f}")
    print(f"MLP Decoder Score (v, u): {score_rev:.8f}")
    assert abs(score_fwd - score_rev) < 1e-6, "Symmetric MLP Decoder failed undirected symmetry test!"
    print("[SUCCESS] MLP Decoder produces identical scores when node pair is reversed.")


if __name__ == "__main__":
    test_decoder_symmetry()
