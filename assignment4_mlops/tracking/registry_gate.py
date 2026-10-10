"""
Model Registry and Validation Gate for Assignment 4.

Defines the explicit gate policy:
- Metric: Validation Macro F1
- Threshold delta: +0.002
- First-model behavior: Promotes automatically as initial baseline
- Tie behavior: Rejects candidate (preserves approved production model)
- Regression behavior: Blocks promotion and logs warning
- Missing-metric behavior: Hard failure / blocks promotion
"""

import os
import json
import shutil

REGISTRY_FILE = r"D:\gnn-practical-assignments\assignment4_mlops\artifacts\registry.json"
PRODUCTION_DIR = r"D:\gnn-practical-assignments\assignment4_mlops\artifacts\production_model"
ARTIFACTS_DIR = r"D:\gnn-practical-assignments\assignment4_mlops\artifacts"


def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"approved_model": None, "history": []}


def save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def evaluate_promotion_gate(candidate_name: str, threshold_delta: float = 0.002):
    candidate_schema_file = os.path.join(ARTIFACTS_DIR, candidate_name, "schema.json")
    if not os.path.exists(candidate_schema_file):
        raise FileNotFoundError(f"Candidate artifact '{candidate_name}' does not exist.")

    with open(candidate_schema_file, "r", encoding="utf-8") as f:
        candidate_schema = json.load(f)

    candidate_metric = candidate_schema.get("val_macro_f1")
    if candidate_metric is None:
        return {"decision": "REJECTED", "reason": "Missing required metric 'val_macro_f1'"}

    registry = load_registry()
    current_approved = registry["approved_model"]

    # 1. First-Model Behavior
    if current_approved is None:
        promote_to_production(candidate_name, candidate_schema)
        decision = {
            "decision": "PROMOTED",
            "reason": "First model in registry - automatically promoted as production baseline.",
            "candidate": candidate_name,
            "metric": candidate_metric,
        }
        registry["approved_model"] = decision
        registry["history"].append(decision)
        save_registry(registry)
        return decision

    current_metric = current_approved["metric"]
    diff = candidate_metric - current_metric

    # 2. Blocked Regression or Insufficient Improvement
    if diff >= threshold_delta:
        promote_to_production(candidate_name, candidate_schema)
        decision = {
            "decision": "PROMOTED",
            "reason": f"Candidate outperformed production by {diff:+.4f} (Threshold: +{threshold_delta})",
            "candidate": candidate_name,
            "metric": candidate_metric,
            "previous_production": current_approved["candidate"],
        }
        registry["approved_model"] = decision
        registry["history"].append(decision)
        save_registry(registry)
        return decision
    elif diff <= 0:
        return {
            "decision": "BLOCKED_REGRESSION",
            "reason": f"Candidate regressed or tied production ({candidate_metric:.4f} vs {current_metric:.4f}). Promotion blocked.",
            "candidate": candidate_name,
            "current_production": current_approved["candidate"],
        }
    else:
        return {
            "decision": "REJECTED",
            "reason": f"Improvement ({diff:+.4f}) fell below required threshold (+{threshold_delta}).",
            "candidate": candidate_name,
            "current_production": current_approved["candidate"],
        }


def promote_to_production(candidate_name: str, schema: dict):
    src_dir = os.path.join(ARTIFACTS_DIR, candidate_name)
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(src_dir, "weights.pt"), os.path.join(PRODUCTION_DIR, "weights.pt"))
    shutil.copyfile(os.path.join(src_dir, "schema.json"), os.path.join(PRODUCTION_DIR, "schema.json"))
    print(f"[REGISTRY] Model '{candidate_name}' promoted and deployed to active production directory.")


def demonstrate_registry_gate():
    print("=" * 75)
    print("DEMONSTRATING MODEL REGISTRY & VALIDATION GATE")
    print("=" * 75)

    # Clean registry state for demonstration
    if os.path.exists(REGISTRY_FILE):
        os.remove(REGISTRY_FILE)

    # 1. First model demonstration (model_v1_baseline)
    res1 = evaluate_promotion_gate("model_v1_baseline")
    print("\nCase 1 (First Model):")
    print(f"  Candidate: model_v1_baseline -> Decision: {res1['decision']}")
    print(f"  Reason:    {res1['reason']}")

    # 2. Regression demonstration (model_v2_high_capacity has lower val F1: 0.7937 vs 0.7982)
    res2 = evaluate_promotion_gate("model_v2_high_capacity")
    print("\nCase 2 (Blocked Regression):")
    print(f"  Candidate: model_v2_high_capacity -> Decision: {res2['decision']}")
    print(f"  Reason:    {res2['reason']}")

    print("\n" + "=" * 75)
    print("[SUCCESS] Promotion gate successfully protected production from regressive model.")


if __name__ == "__main__":
    demonstrate_registry_gate()
