"""
FastAPI Serving Application for GNN Inference (Assignment 4).

Features:
- Loads active production model from artifacts directory.
- Endpoints:
  - GET /health: Service and model metadata
  - POST /predict: Node classification over subgraphs or full graphs
- Robust Failure Handling:
  - Isolated nodes: Handled gracefully via GCN self-loops
  - Duplicate target IDs: Deduplicated with informational warning
  - Invalid edge endpoints: HTTP 422 with precise validation message
  - Incorrect feature dimensions: HTTP 422
  - Unknown target IDs: HTTP 400
- Supports serving existing nodes and inductive unseen nodes.
"""

import os
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, status
from assignment4_mlops.app.schemas import InferenceRequest, InferenceResponse, PredictionItem, HealthResponse
from assignment2_architectures.models import GCNClassifier

PRODUCTION_DIR = r"D:\gnn-practical-assignments\assignment4_mlops\artifacts\production_model"

app = FastAPI(
    title="GNN Model Serving API",
    description="Production-grade GNN inference service for citation document classification.",
    version="1.0.0"
)

# Global model state
state = {
    "model": None,
    "schema": None,
    "version": None,
}


def load_production_model():
    schema_path = os.path.join(PRODUCTION_DIR, "schema.json")
    weights_path = os.path.join(PRODUCTION_DIR, "weights.pt")

    if not os.path.exists(schema_path) or not os.path.exists(weights_path):
        state["model"] = None
        state["schema"] = None
        state["version"] = "no_model_loaded"
        return

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    model = GCNClassifier(
        in_channels=schema["in_channels"],
        hidden_channels=schema["hidden_channels"],
        out_channels=schema["out_channels"],
        dropout=schema.get("dropout", 0.5),
    )
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    state["model"] = model
    state["schema"] = schema
    state["version"] = schema.get("run_id", "v1.0.0")


# Load model at startup
load_production_model()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=state["model"] is not None,
        model_version=state["version"],
        in_channels=state["schema"]["in_channels"] if state["schema"] else None,
        num_classes=state["schema"]["out_channels"] if state["schema"] else None,
    )


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )

    # 1. Feature Dimension Validation
    num_nodes = len(request.x)
    if num_nodes == 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Feature matrix x cannot be empty.")

    expected_dim = state["schema"]["in_channels"]
    for i, row in enumerate(request.x):
        if len(row) != expected_dim:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Node {i} has feature dimension {len(row)}, but model expects {expected_dim}."
            )

    # 2. Edge Index Structure & Endpoint Validation
    if len(request.edge_index) != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"edge_index must have shape (2, num_edges), got {len(request.edge_index)} rows."
        )

    num_edges = len(request.edge_index[0])
    if len(request.edge_index[1]) != num_edges:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source and target edge rows must have identical lengths."
        )

    # Validate that edge endpoints do not exceed node boundaries
    for src, dst in zip(request.edge_index[0], request.edge_index[1]):
        if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid edge endpoint ({src}, {dst}). All node IDs must be in range [0, {num_nodes - 1}]."
            )

    # 3. Target Node Validation & Deduplication
    warnings = []
    if request.target_nodes is None:
        target_nodes = list(range(num_nodes))
    else:
        # Check for unknown target IDs
        for tid in request.target_nodes:
            if tid < 0 or tid >= num_nodes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Target node ID {tid} is out of bounds for graph with {num_nodes} nodes."
                )
        # Check duplicate target IDs
        if len(request.target_nodes) != len(set(request.target_nodes)):
            warnings.append("Duplicate target IDs detected in request; returning deduplicated predictions.")
            seen = set()
            target_nodes = [x for x in request.target_nodes if not (x in seen or seen.add(x))]
        else:
            target_nodes = request.target_nodes

    # 4. Convert to PyTorch Tensors & Execute Forward Pass
    x_tensor = torch.tensor(request.x, dtype=torch.float32)
    edge_index_tensor = torch.tensor(request.edge_index, dtype=torch.long)

    with torch.no_grad():
        logits = state["model"](x_tensor, edge_index_tensor)
        probs = F.softmax(logits, dim=-1)

    class_names = state["schema"]["class_names"]
    predictions = []
    for tid in target_nodes:
        node_probs = probs[tid].tolist()
        pred_class = int(torch.argmax(probs[tid]).item())
        predictions.append(
            PredictionItem(
                node_id=tid,
                predicted_class=pred_class,
                class_name=class_names[pred_class],
                probabilities=[round(p, 6) for p in node_probs],
            )
        )

    return InferenceResponse(
        model_version=state["version"],
        num_nodes=num_nodes,
        num_edges=num_edges,
        predictions=predictions,
        warnings=warnings if warnings else None,
    )
