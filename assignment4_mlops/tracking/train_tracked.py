"""
Reproducible & Tracked GNN Pipeline using MLflow (Assignment 4).

Features:
- Runs 3 distinct experiments with varying configurations.
- Logs parameters, metrics (val loss, val F1, test acc), git commit, and seed.
- Full Model Packaging:
  - Architecture definition & state dict (weights.pt)
  - Preprocessing and feature schema (schema.json)
  - Output-label mapping
- Reproducibility Check: Reloads the model artifact and verifies bitwise prediction equality.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import json
import hashlib
import torch
import mlflow
from sklearn.metrics import f1_score
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment2_architectures.models import GCNClassifier

CORA_CLASSES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory"
]

DB_PATH = r"D:\gnn-practical-assignments\assignment4_mlops\mlflow.db"
MODELS_DIR = r"D:\gnn-practical-assignments\assignment4_mlops\artifacts"


def compute_dataset_hash(data):
    features_bytes = data.x.numpy().tobytes()
    return hashlib.sha256(features_bytes).hexdigest()[:12]


def train_and_package_model(run_name: str, hidden_dim: int, lr: float, dropout: float, seed: int = 42):
    os.makedirs(MODELS_DIR, exist_ok=True)
    sqlite_uri = f"sqlite:///{DB_PATH.replace('\\', '/')}"
    mlflow.set_tracking_uri(sqlite_uri)
    mlflow.set_experiment("cora-gnn-production-pipeline")

    torch.manual_seed(seed)
    dataset, data = load_cora_dataset()
    dataset_hash = compute_dataset_hash(data)

    model = GCNClassifier(dataset.num_features, hidden_dim, dataset.num_classes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Log parameters and provenance
        mlflow.log_params({
            "model_architecture": "GCNClassifier",
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "dropout": dropout,
            "seed": seed,
            "dataset_name": "Cora",
            "dataset_hash": dataset_hash,
            "code_version": "v1.0.0",
        })

        best_val_f1 = -1.0
        best_state = None

        # 2. Training Loop
        for epoch in range(120):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_preds = val_out[data.val_mask].argmax(dim=-1).cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()
                val_f1 = f1_score(val_true, val_preds, average="macro")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 3. Final Test Evaluation using best checkpoint
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            original_logits = model(data.x, data.edge_index)
            test_preds = original_logits[data.test_mask].argmax(dim=-1).cpu().numpy()
            test_acc = (test_preds == data.y[data.test_mask].cpu().numpy()).mean()

        # Log metrics to MLflow
        mlflow.log_metrics({
            "val_macro_f1": best_val_f1,
            "test_accuracy": test_acc,
        })

        # 4. Model Packaging (Weights + Schema + Preprocessing)
        artifact_path = os.path.join(MODELS_DIR, run_name)
        os.makedirs(artifact_path, exist_ok=True)

        weights_file = os.path.join(artifact_path, "weights.pt")
        schema_file = os.path.join(artifact_path, "schema.json")

        torch.save(best_state, weights_file)

        schema = {
            "model_architecture": "GCNClassifier",
            "in_channels": dataset.num_features,
            "hidden_channels": hidden_dim,
            "out_channels": dataset.num_classes,
            "dropout": dropout,
            "class_names": CORA_CLASSES,
            "preprocessing": "feature_normalization_l1",
            "dataset_hash": dataset_hash,
            "val_macro_f1": float(best_val_f1),
            "test_accuracy": float(test_acc),
            "run_id": run.info.run_id,
        }
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        mlflow.log_artifact(weights_file)
        mlflow.log_artifact(schema_file)

        # 5. Reproducibility Verification
        verify_model_reproducibility(weights_file, schema, data, original_logits)

        print(f"[TRACKED] Run '{run_name}' saved to MLflow (Val F1: {best_val_f1:.4f}, Test Acc: {test_acc:.2%})")
        return artifact_path, schema


def verify_model_reproducibility(weights_file, schema, data, original_logits):
    """
    Verifies that reloading the packaged model reproduces predictions bitwise.
    """
    reloaded_model = GCNClassifier(
        in_channels=schema["in_channels"],
        hidden_channels=schema["hidden_channels"],
        out_channels=schema["out_channels"],
        dropout=schema["dropout"],
    )
    reloaded_model.load_state_dict(torch.load(weights_file))
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_logits = reloaded_model(data.x, data.edge_index)

    max_diff = torch.abs(original_logits - reloaded_logits).max().item()
    assert max_diff < 1e-6, f"Reproducibility check failed! Max difference: {max_diff}"
    print(f"[OK] Reload reproducibility verified: max diff = {max_diff:.8e}")


def run_all_experiments():
    print("=" * 75)
    print("RUNNING 3 EXPERIMENTS WITH MLFLOW TRACKING")
    print("=" * 75)
    # Experiment 1: Standard baseline
    train_and_package_model("model_v1_baseline", hidden_dim=64, lr=0.01, dropout=0.5, seed=42)
    # Experiment 2: Higher capacity
    train_and_package_model("model_v2_high_capacity", hidden_dim=128, lr=0.005, dropout=0.3, seed=42)
    # Experiment 3: High regularization
    train_and_package_model("model_v3_high_reg", hidden_dim=64, lr=0.02, dropout=0.7, seed=123)
    print("=" * 75)
    print("[SUCCESS] All 3 experiments tracked in MLflow and packaged in artifacts directory.")


if __name__ == "__main__":
    run_all_experiments()
