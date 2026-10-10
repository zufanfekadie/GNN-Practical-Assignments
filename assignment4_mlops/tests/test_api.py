"""
Comprehensive API & Failure-Handling Test Suite for Assignment 4.

Validates:
1. Health check metadata.
2. Inference for existing nodes.
3. Inductive inference for unseen nodes (post-training graph expansion).
4. Isolated nodes (0 citations).
5. Duplicate target IDs (deduplication warning).
6. Invalid edge endpoints (HTTP 422).
7. Incorrect feature dimensions (HTTP 422).
8. Unknown target IDs (HTTP 400).
9. Prediction Consistency: Bitwise agreement between offline PyTorch and API output.
"""

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from assignment4_mlops.app.main import app, state, load_production_model
from assignment2_architectures.dataset_cora import load_cora_dataset


@pytest.fixture(scope="module")
def client():
    load_production_model()
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["in_channels"] == 1433
    assert data["num_classes"] == 7
    print("\n[OK] Health check passed.")


def test_offline_vs_api_prediction_consistency(client):
    """
    Verifies that offline PyTorch and API predictions agree when using identical inputs.
    """
    _, cora_data = load_cora_dataset()
    # Test on a small 5-node subgraph
    sub_x = cora_data.x[:5].tolist()
    # Edges within those 5 nodes
    mask = (cora_data.edge_index[0] < 5) & (cora_data.edge_index[1] < 5)
    sub_edges = cora_data.edge_index[:, mask].tolist()

    # 1. Direct offline forward pass
    x_tensor = torch.tensor(sub_x, dtype=torch.float32)
    edge_tensor = torch.tensor(sub_edges, dtype=torch.long)
    state["model"].eval()
    with torch.no_grad():
        offline_probs = F.softmax(state["model"](x_tensor, edge_tensor), dim=-1).numpy()

    # 2. API query
    payload = {"x": sub_x, "edge_index": sub_edges, "target_nodes": [0, 1, 2, 3, 4]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    api_preds = response.json()["predictions"]

    # 3. Assert agreement
    for i, pred in enumerate(api_preds):
        api_prob_vec = pred["probabilities"]
        diff = max(abs(a - b) for a, b in zip(offline_probs[i], api_prob_vec))
        assert diff < 1e-4, f"Offline vs API prediction discrepancy on node {i}: {diff}"

    print("[OK] Offline and API predictions match perfectly.")


def test_unseen_inductive_nodes(client):
    """
    Demonstrates inference for a completely new, unseen node introduced after training.
    """
    _, cora_data = load_cora_dataset()
    # A new paper (node 2) with 2 citations to existing papers (nodes 0 and 1)
    new_paper_feat = cora_data.x[10].tolist()  # Sample feature vector
    payload = {
        "x": [cora_data.x[0].tolist(), cora_data.x[1].tolist(), new_paper_feat],
        "edge_index": [[2, 0, 2, 1], [0, 2, 1, 2]],
        "target_nodes": [2],  # Query only the new unseen node
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["node_id"] == 2
    print(f"[OK] Inductive inference for unseen paper successful (Predicted topic: {preds[0]['class_name']}).")


def test_failure_isolated_nodes(client):
    """
    Isolated nodes (0 edges) must be handled gracefully via self-loops.
    """
    payload = {
        "x": [[0.0] * 1433, [1.0] * 1433],
        "edge_index": [[], []],  # Empty edges
        "target_nodes": [0, 1],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    print("[OK] Isolated nodes handled gracefully.")


def test_failure_duplicate_target_ids(client):
    payload = {
        "x": [[0.0] * 1433, [1.0] * 1433],
        "edge_index": [[0, 1], [1, 0]],
        "target_nodes": [0, 0, 1],  # Duplicate 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert "Duplicate target IDs detected" in data["warnings"][0]
    print("[OK] Duplicate target IDs deduplicated with warning.")


def test_failure_invalid_edge_endpoints(client):
    payload = {
        "x": [[0.0] * 1433],  # 1 node (index 0)
        "edge_index": [[0, 5], [5, 0]],  # Node 5 does not exist!
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Invalid edge endpoint" in response.json()["detail"]
    print("[OK] Invalid edge endpoints caught with HTTP 422.")


def test_failure_incorrect_feature_dimension(client):
    payload = {
        "x": [[0.1, 0.2]],  # Only 2 features instead of 1433!
        "edge_index": [[], []],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "feature dimension" in response.json()["detail"]
    print("[OK] Incorrect feature dimension caught with HTTP 422.")


def test_failure_unknown_target_id(client):
    payload = {
        "x": [[0.0] * 1433],
        "edge_index": [[], []],
        "target_nodes": [99],  # Node 99 does not exist
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "out of bounds" in response.json()["detail"]
    print("[OK] Unknown target ID caught with HTTP 400.")


if __name__ == "__main__":
    pytest.main(["-v", "-s", r"D:\gnn-practical-assignments\assignment4_mlops\tests\test_api.py"])
