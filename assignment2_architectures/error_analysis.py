"""
Error Analysis Module for Assignment 2.

Analyzes 5 misclassified test nodes by inspecting:
- Ground truth class vs. Predicted class
- Node degree and 1-hop neighborhood composition
- Local neighborhood homophily (fraction of neighbors sharing the true class)
- Feature density (vocabulary word count in the paper)
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from assignment2_architectures.dataset_cora import load_cora_dataset
from assignment2_architectures.models import GCNClassifier
from assignment2_architectures.train_benchmark import train_single_run

CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def analyze_misclassifications(num_samples: int = 5):
    dataset, data = load_cora_dataset()
    in_dim, out_dim = dataset.num_features, dataset.num_classes

    # Train a standard 2-layer GCN
    torch.manual_seed(42)
    model = GCNClassifier(in_dim, 64, out_dim)
    metrics = train_single_run(model, data, epochs=150)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)

    # Find misclassified nodes strictly in the test set
    test_indices = torch.where(data.test_mask)[0]
    errors = test_indices[preds[test_indices] != data.y[test_indices]]

    print("=" * 80)
    print(f"QUALITATIVE ERROR ANALYSIS: 5 MISCLASSIFIED TEST NODES (Total Test Errors: {len(errors)})")
    print("=" * 80)

    for i in range(min(num_samples, len(errors))):
        node_id = errors[i].item()
        true_label = data.y[node_id].item()
        pred_label = preds[node_id].item()

        # Find 1-hop neighbors from edge_index
        neighbors = data.edge_index[1][data.edge_index[0] == node_id].tolist()
        neighbor_labels = data.y[neighbors].tolist() if neighbors else []

        # Count neighbor classes
        same_class_neighbors = sum(1 for l in neighbor_labels if l == true_label)
        homophily = (same_class_neighbors / len(neighbor_labels)) if neighbor_labels else 0.0

        # Feature sparsity: how many non-zero words are present
        active_words = (data.x[node_id] > 0).sum().item()

        print(f"\n--- Case {i + 1}: Node ID {node_id} ---")
        print(f"True Class:             {true_label} ({CLASS_NAMES[true_label]})")
        print(f"Predicted Class:        {pred_label} ({CLASS_NAMES[pred_label]})")
        print(f"Active Feature Words:   {active_words} / {data.x.shape[1]}")
        print(f"Total Citations/Degree: {len(neighbors)}")
        print(f"Neighbor Class Labels:  {neighbor_labels}")
        print(f"Neighborhood Homophily: {homophily:.1%}")

        # Diagnosis
        if len(neighbors) == 0:
            print("Reason for error:       Isolated Node (degree 0) - relies solely on weak text features without citation support.")
        elif homophily == 0.0:
            print(f"Reason for error:       Strong Heterophily - all neighbors belong to different categories ({neighbor_labels}), misleading neighborhood aggregation towards class {pred_label}.")
        elif homophily < 0.5:
            print(f"Reason for error:       Neighborhood Contamination - majority of citing papers belong to conflicting classes, overpowering the true label signal.")
        else:
            print("Reason for error:       Ambiguous boundary - conflicting text features despite supportive citations.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_misclassifications()
