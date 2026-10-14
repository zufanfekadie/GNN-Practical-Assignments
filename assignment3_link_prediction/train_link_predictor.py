"""
Complete Link Prediction Pipeline, Leaky Comparison & Candidate Discovery (Assignment 3).

Evaluates:
1. Common Neighbors Heuristic Baseline
2. GNN Encoder + Dot-Product Decoder (Leak-Free)
3. GNN Encoder + Symmetric MLP Decoder (Leak-Free)
4. Deliberately Leaky GNN (Includes test edges in message passing)
5. Extracts and analyzes 10 high-scoring unobserved citation candidates.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the file can be run directly with the play button
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from assignment3_link_prediction.split_utils import create_link_prediction_splits
from assignment3_link_prediction.models_link import GCNEncoder, DotProductDecoder, SymmetricMLPDecoder, common_neighbors_score


def evaluate_link_model(encoder, decoder, x, msg_edge_index, pos_pairs, neg_pairs):
    encoder.eval()
    if decoder is not None:
        decoder.eval()

    with torch.no_grad():
        z = encoder(x, msg_edge_index)
        pos_scores = decoder(z, pos_pairs)
        neg_scores = decoder(z, neg_pairs)

        scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    return auc, ap


def train_link_pipeline(decoder_type="dot", leaky=False, epochs=120, lr=0.01, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = create_link_prediction_splits(seed=seed)
    x = splits["x"]
    # Choose message-passing graph: leak-free vs deliberately leaky
    msg_edge_index = splits["leaky_msg_edge_index"] if leaky else splits["train_msg_edge_index"]

    train_pos = splits["train_pos"]
    train_neg = splits["train_neg"]
    val_pos = splits["val_pos"]
    val_neg = splits["val_neg"]
    test_pos = splits["test_pos"]
    test_neg = splits["test_neg"]

    in_dim = x.shape[1]
    hidden_dim = 64
    emb_dim = 32

    encoder = GCNEncoder(in_dim, hidden_dim, emb_dim)
    if decoder_type == "dot":
        decoder = DotProductDecoder()
        params = list(encoder.parameters())
    else:
        decoder = SymmetricMLPDecoder(emb_dim=emb_dim, hidden_dim=32)
        params = list(encoder.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    criterion = nn.BCELoss()

    best_val_auc = -1.0
    best_weights = None

    for epoch in range(epochs):
        encoder.train()
        if decoder_type != "dot":
            decoder.train()
        optimizer.zero_grad()

        z = encoder(x, msg_edge_index)
        pos_scores = decoder(z, train_pos)
        neg_scores = decoder(z, train_neg)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))])

        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        # Validation check every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_auc, val_ap = evaluate_link_model(encoder, decoder, x, msg_edge_index, val_pos, val_neg)
            if val_auc > best_val_auc:
                best_val_auc = val_auc

    # Final test evaluation
    test_auc, test_ap = evaluate_link_model(encoder, decoder, x, msg_edge_index, test_pos, test_neg)
    return {
        "test_auc": test_auc,
        "test_ap": test_ap,
        "encoder": encoder,
        "decoder": decoder,
        "splits": splits,
    }


def find_top_unobserved_candidates(encoder, decoder, splits, top_k=10):
    """
    Ranks unobserved pairs and presents top-k candidates.
    """
    x = splits["x"]
    msg_edges = splits["train_msg_edge_index"]
    num_nodes = splits["num_nodes"]

    # Collect all existing edges
    all_known = set()
    for u, v in splits["train_pos"].tolist() + splits["val_pos"].tolist() + splits["test_pos"].tolist():
        all_known.add((min(u, v), max(u, v)))

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(x, msg_edges)

    # Sample a candidate pool of unobserved pairs
    np.random.seed(42)
    candidate_pairs = []
    while len(candidate_pairs) < 5000:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair not in all_known:
            candidate_pairs.append(list(pair))

    candidate_tensor = torch.tensor(candidate_pairs, dtype=torch.long)
    with torch.no_grad():
        scores = decoder(z, candidate_tensor).cpu().numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\n" + "=" * 80)
    print(f"TOP {top_k} HIGH-CONFIDENCE UNOBSERVED CANDIDATE CITATIONS (OPEN-WORLD DISCOVERY)")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Candidate Pair (Paper u, Paper v)':<35} | {'Predicted Link Probability':<28}")
    print("-" * 80)
    for rank, idx in enumerate(top_indices, 1):
        pair = candidate_pairs[idx]
        score = scores[idx]
        print(f"{rank:<6} | Node {pair[0]:<5} <---> Node {pair[1]:<5}                | {score * 100:.2f}%")
    print("=" * 80)
    print("ANALYSIS OF UNOBSERVED LINKS:")
    print("In real-world citation graphs, the absence of an edge is NOT a confirmed negative.")
    print("An unobserved link simply means the authors may not have read each other's paper yet,")
    print("or published concurrently. High-confidence predictions represent potential missing citations")
    print("or natural candidates for future inter-paper collaboration.")


def run_assignment3_experiments():
    print("=" * 85)
    print("ASSIGNMENT 3: LINK PREDICTION BENCHMARKS & LEAKAGE INVESTIGATION")
    print("=" * 85)

    # 1. Baseline: Common Neighbors
    splits = create_link_prediction_splits(seed=42)
    cn_pos = common_neighbors_score(splits["train_msg_edge_index"], splits["num_nodes"], splits["test_pos"]).numpy()
    cn_neg = common_neighbors_score(splits["train_msg_edge_index"], splits["num_nodes"], splits["test_neg"]).numpy()
    cn_scores = np.concatenate([cn_pos, cn_neg])
    cn_labels = np.concatenate([np.ones(len(cn_pos)), np.zeros(len(cn_neg))])
    cn_auc = roc_auc_score(cn_labels, cn_scores)
    cn_ap = average_precision_score(cn_labels, cn_scores)

    # 2. GNN + Dot Product Decoder (Valid / Leak-Free)
    res_dot = train_link_pipeline(decoder_type="dot", leaky=False, seed=42)

    # 3. GNN + Symmetric MLP Decoder (Valid / Leak-Free)
    res_mlp = train_link_pipeline(decoder_type="mlp", leaky=False, seed=42)

    # 4. GNN + Deliberately Leaky Evaluation
    res_leaky = train_link_pipeline(decoder_type="dot", leaky=True, seed=42)

    print("\nRESULTS SUMMARY TABLE:")
    print(f"{'Model / Approach':<40} | {'Test ROC-AUC':<15} | {'Test Average Precision (AP)':<25}")
    print("-" * 85)
    print(f"{'1. Common Neighbors (Baseline Heuristic)':<40} | {cn_auc:<15.4f} | {cn_ap:<25.4f}")
    print(f"{'2. GCN + Dot-Product Decoder (Valid)':<40} | {res_dot['test_auc']:<15.4f} | {res_dot['test_ap']:<25.4f}")
    print(f"{'3. GCN + Symmetric MLP Decoder (Valid)':<40} | {res_mlp['test_auc']:<15.4f} | {res_mlp['test_ap']:<25.4f}")
    print(f"{'4. GCN + Dot-Product (DELIBERATELY LEAKY)':<40} | {res_leaky['test_auc']:<15.4f} | {res_leaky['test_ap']:<25.4f}")
    print("=" * 85)

    print("\nWHY THE LEAKY EXPERIMENT IS INVALID:")
    print(f"Notice that the leaky model scores {res_leaky['test_auc']:.2%} ROC-AUC vs {res_dot['test_auc']:.2%} for the valid model.")
    print("This +{:.2f}% jump is completely artificial! Because test edges were left in the message-passing graph,".format((res_leaky['test_auc'] - res_dot['test_auc']) * 100))
    print("the GCN directly passed node features across the test edges before the decoder was even queried.")
    print("In production, future links do not exist yet; using leaky metrics results in catastrophic failure in real-world deployment.")

    # Present 10 Unobserved Candidates
    find_top_unobserved_candidates(res_dot["encoder"], res_dot["decoder"], splits, top_k=10)


if __name__ == "__main__":
    run_assignment3_experiments()
