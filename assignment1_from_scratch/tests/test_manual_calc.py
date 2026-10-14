"""
Test 1 for Assignment 1:
Verify one node's output from one GCN layer against an explicit manual calculation.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly from any folder or play button
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from assignment1_from_scratch.gcn_scratch import GCNLayerScratch, normalize_adjacency


def test_manual_calculation_single_node():
    torch.manual_seed(101)

    # 1. Create a tiny toy graph with 4 nodes and 2 features
    num_nodes = 4
    in_features = 2
    out_features = 2

    # Node features
    x = torch.tensor([
        [1.0, 2.0],  # Node 0
        [3.0, 1.0],  # Node 1
        [0.0, 4.0],  # Node 2
        [2.0, 2.0],  # Node 3
    ], dtype=torch.float32)

    # Undirected graph edges: 0-1, 0-2, 2-3
    adj = torch.tensor([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=torch.float32)

    # Instantiate layer with known fixed weights and bias
    layer = GCNLayerScratch(in_features, out_features, bias=True)
    w = torch.tensor([[0.5, -0.5], [1.0, 2.0]], dtype=torch.float32)
    b = torch.tensor([0.1, -0.2], dtype=torch.float32)
    layer.weight.data = w
    layer.bias.data = b

    # Model computation
    norm_adj = normalize_adjacency(adj)
    model_output = layer(x, norm_adj)
    node_0_model_out = model_output[0]

    # --- MANUAL STEP-BY-STEP CALCULATION FOR NODE 0 ---
    # With self-loops:
    # Node 0 has neighbors: {0, 1, 2}
    # Degrees with self-loops:
    # d_tilde[0] = 3 (edges to 0, 1, 2)
    # d_tilde[1] = 2 (edges to 1, 0)
    # d_tilde[2] = 3 (edges to 2, 0, 3)
    # d_tilde[3] = 2 (edges to 3, 2)

    # Transformation x @ W for each relevant node:
    xw_0 = torch.matmul(x[0], w)  # [1*0.5 + 2*1, 1*-0.5 + 2*2] = [2.5, 3.5]
    xw_1 = torch.matmul(x[1], w)  # [3*0.5 + 1*1, 3*-0.5 + 1*2] = [2.5, 0.5]
    xw_2 = torch.matmul(x[2], w)  # [0*0.5 + 4*1, 0*-0.5 + 4*2] = [4.0, 8.0]

    # Normalization factors 1 / sqrt(d_i * d_j):
    norm_00 = 1.0 / (3.0**0.5 * 3.0**0.5)  # 1/3
    norm_01 = 1.0 / (3.0**0.5 * 2.0**0.5)  # 1/sqrt(6)
    norm_02 = 1.0 / (3.0**0.5 * 3.0**0.5)  # 1/3

    # Neighborhood aggregation + bias:
    manual_node_0 = (norm_00 * xw_0) + (norm_01 * xw_1) + (norm_02 * xw_2) + b

    # Assert model output matches manual computation
    difference = torch.abs(node_0_model_out - manual_node_0).max().item()
    print(f"Node 0 Model Output:  {node_0_model_out.detach().numpy()}")
    print(f"Node 0 Manual Output: {manual_node_0.numpy()}")
    print(f"Max absolute difference: {difference:.8e}")

    assert torch.allclose(node_0_model_out, manual_node_0, atol=1e-6), "Manual calculation mismatch!"
    print("SUCCESS: Model output exactly matches manual hand-calculation.")


if __name__ == "__main__":
    test_manual_calculation_single_node()
