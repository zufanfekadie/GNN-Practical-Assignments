"""
Benchmark Runner for Assignment 2.

Evaluates MLP, GCN, GraphSAGE, and GAT across 3 random seeds:
- Tracks Test Accuracy, Macro F1, Training Time, and Parameter Count.
- Strict protocol: Checkpoint selection using validation loss/F1.
- Test set evaluated once per final model checkpoint.
- Reports Mean +/- Std across seeds.
"""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment2_architectures.models import MLPClassifier, GCNClassifier, GraphSAGEClassifier, GATClassifier


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_single_run(model, data, epochs: int = 200, lr: float = 0.01, weight_decay: float = 5e-4, patience: int = 30):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_preds = val_out[data.val_mask].argmax(dim=-1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(val_true, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    training_time = time.time() - start_time

    # Evaluate best checkpoint on test set
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        test_preds = test_out[data.test_mask].argmax(dim=-1).cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()

        test_acc = (test_preds == test_true).mean()
        test_f1 = f1_score(test_true, test_preds, average="macro")

    return {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time": training_time,
        "best_weights": best_weights,
    }


def run_architecture_benchmarks(seeds=(42, 123, 999)):
    dataset, data = load_cora_dataset()
    in_dim = dataset.num_features
    out_dim = dataset.num_classes
    hidden_dim = 64

    model_factories = {
        "MLP": lambda: MLPClassifier(in_dim, hidden_dim, out_dim),
        "GCN": lambda: GCNClassifier(in_dim, hidden_dim, out_dim),
        "GraphSAGE": lambda: GraphSAGEClassifier(in_dim, hidden_dim, out_dim),
        "GAT": lambda: GATClassifier(in_dim, hidden_dim, out_dim, heads=8),
    }

    results = {}

    print("=" * 80)
    print("RUNNING GNN ARCHITECTURE BENCHMARKS (3 RANDOM SEEDS)")
    print("=" * 80)

    for name, factory in model_factories.items():
        results[name] = {
            "params": count_parameters(factory()),
            "accs": [],
            "f1s": [],
            "times": [],
        }

        print(f"\nEvaluating {name} (Params: {results[name]['params']:,})...")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = factory()
            metrics = train_single_run(model, data)
            results[name]["accs"].append(metrics["test_acc"])
            results[name]["f1s"].append(metrics["test_f1"])
            results[name]["times"].append(metrics["train_time"])
            print(f"  Seed {seed}: Test Acc = {metrics['test_acc']:.2%}, Macro F1 = {metrics['test_f1']:.4f}, Time = {metrics['train_time']:.2f}s")

    # Format summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<12} | {'Params':<8} | {'Test Acc (Mean ± Std)':<22} | {'Macro F1 (Mean ± Std)':<22} | {'Train Time (s)':<12}")
    print("-" * 80)
    for name, r in results.items():
        acc_mean, acc_std = np.mean(r["accs"]), np.std(r["accs"])
        f1_mean, f1_std = np.mean(r["f1s"]), np.std(r["f1s"])
        time_mean = np.mean(r["times"])
        print(f"{name:<12} | {r['params']:<8} | {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}%     | {f1_mean:.4f} ± {f1_std:.4f}         | {time_mean:.2f}s")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_architecture_benchmarks()
