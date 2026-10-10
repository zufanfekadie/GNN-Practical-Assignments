# GNN Comprehensive Technical Report & Evaluation Defense

**Author**: GNN Engineering Team  
**Dataset**: Synthetic Homophilic Graph & Cora Citation Network  
**Environment**: PyTorch 2.11, PyG 2.8, FastAPI, MLflow  

---

## 1. Assignment 1: GCN from Scratch & Correctness Verification

### 1.1 Mathematical Formulation
The layer-wise propagation rule implemented purely using PyTorch tensor operations is:
$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W + b$$
where $\tilde{A} = A + I_N$ integrates self-loops, and $\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$ is the diagonal degree matrix.

### 1.2 Synthetic Graph Design
* **Nodes & Features**: 120 nodes partitioned equally into Class 0 and Class 1. Each node has 3 numerical features sampled from overlapping Gaussians ($\mathcal{N}(0.0, 1.0)$ for Class 0 vs $\mathcal{N}(0.6, 1.0)$ for Class 1). Because of this overlap, feature-only classifiers are noisy and achieve only ~60% accuracy.
* **Topology**: Stochastic Block Model with intra-class connection probability $p_{\text{intra}} = 0.15$ and inter-class connection probability $p_{\text{inter}} = 0.01$.

### 1.3 Correctness Verification Results
* **Manual Hand-Calculation**: An isolated node's feature transformation, neighborhood degree scaling, and bias aggregation was computed by hand and compared against PyTorch tensor outputs.
  $$\text{Max Absolute Difference} = 4.768 \times 10^{-7}$$
* **Permutation Equivariance**: A random node permutation $P$ was applied such that $X' = PX$ and $A' = PAP^T$.
  $$\text{Max Output Discrepancy} = 1.676 \times 10^{-8}$$

### 1.4 Ablation Results
| Configuration | Validation Acc | Test Acc | Interpretation |
| :--- | :---: | :---: | :--- |
| **GCN (True Graph)** | **100.00%** | **100.00%** | Graph convolutions perfectly resolve feature ambiguity. |
| **MLP Baseline** | 66.67% | 62.50% | Feature overlap caps accuracy around ~62%. |
| **GCN (Randomized Edges)** | 37.50% | 54.17% | False neighbors corrupt signal through message passing. |
| **GCN (Empty Graph)** | 66.67% | 58.33% | Collapses directly to the feature-only MLP baseline. |

---

## 2. Assignment 2: Architecture Comparison on Cora

### 2.1 3-Seed Benchmark Results
All models were trained under a uniform hyperparameter budget (hidden dimension 64, dropout 0.5, Adam lr 0.01) across random seeds `[42, 123, 999]`:

| Architecture | Parameters | Test Acc (Mean ± Std) | Macro F1 (Mean ± Std) | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **MLP** | 92,231 | 59.20% ± 0.96% | 0.5739 ± 0.0089 | **2.44s** |
| **GCN** | 92,231 | 82.30% ± 0.28% | 0.8096 ± 0.0028 | 3.19s |
| **GraphSAGE** | 184,391 | 79.10% ± 0.59% | 0.7847 ± 0.0061 | 7.27s |
| **GAT (8 heads)** | 92,373 | **82.43% ± 0.17%** | **0.8121 ± 0.0014** | 10.66s |

**Recommendation**: GCN is recommended for deployment. While GAT marginally leads in accuracy (+0.13%), GCN trains 3.3x faster and avoids dynamic attention computation over edges.

### 2.2 Depth vs. Oversmoothing
| Depth (Layers) | Test Accuracy | Average Edge Cosine Similarity |
| :---: | :---: | :---: |
| 2 | 82.70% | 0.9593 |
| 4 | 81.30% | 0.9764 |
| 8 | 16.10% | 0.9365 (Oversmoothed collapse to random guessing) |
| 16 | 17.50% | 0.9879 |

### 2.3 Edge Removal Sweep
| Edges Removed | MLP (Control) | GCN | GraphSAGE | GAT |
| :---: | :---: | :---: | :---: | :---: |
| 0% | 59.50% | 82.70% | 78.50% | 82.30% |
| 25% | 59.50% | 78.40% | 78.40% | 80.20% |
| 50% | 59.50% | 74.30% | 71.40% | 76.30% |
| 75% | 59.50% | 66.10% | 65.10% | 68.90% |
| 100% | 59.50% | 59.60% | 58.30% | 58.20% |

At 100% edge drop, all GNN models converge directly to the MLP baseline (~59%).

---

## 3. Assignment 3: Link Prediction & Data Leakage Investigation

### 3.1 Benchmark & Leakage Comparison
| Approach | Test ROC-AUC | Test Average Precision (AP) | Note |
| :--- | :---: | :---: | :--- |
| **Common Neighbors** | 0.7211 | 0.7181 | Unsupervised heuristic baseline |
| **GCN + Dot-Product (Valid)** | **0.7405** | **0.7441** | Strictly leak-free message passing |
| **GCN + Symmetric MLP (Valid)** | 0.6957 | 0.7224 | Guaranteed $s(u, v) = s(v, u)$ |
| **GCN + Dot-Product (LEAKY)** | **0.8143** | **0.8135** | **+7.38% Artificial metric inflation** |

### 3.2 Top Unobserved Candidate Citations
Using our valid trained GNN, we discovered top unobserved links with predicted citation probabilities exceeding 97%:
1. `Node 1358 <---> Node 2145` (99.99%)
2. `Node 436  <---> Node 1693` (99.33%)
3. `Node 132  <---> Node 1483` (97.79%)

**Open-World Rationale**: The absence of a citation in academic literature is an unobserved event, not a confirmed negative. Two papers working in the same niche may be written concurrently or unaware of one another.

---

## 4. Assignment 4: Production MLOps System

1. **Experiment Tracking**: MLflow tracking with SQLite backend (`mlflow.db`) logging parameters, metrics, seeds, dataset hashes, and artifacts across 3 configurations.
2. **Model Packaging**: Packaged model weights (`weights.pt`), schema metadata (`schema.json`), and class mappings with bitwise reload verification (`diff = 0.000`).
3. **Model Registry & Promotion Gate**: Validates candidate models against production. Candidate `model_v1_baseline` was promoted automatically, while regressive `model_v2_high_capacity` was blocked (`0.7937 < 0.7982`).
4. **Drift Detection**: Alerts triggered when batch average degree deviates by $>30\%$ from baseline (3.90) or embedding centroid cosine distance shifts by $>0.20$.
5. **CI/CD & Rollback**: GitHub Actions workflow running tests on push; automated script restoring previous approved models upon failure.

---

## 5. Significant Implementation Problem & Resolution

### Problem:
During the link prediction setup (Assignment 3), we initially observed a subtle data leakage issue where the GNN achieved an unexpectedly high ROC-AUC (~81.5%).

### Investigation:
We investigated the message-passing graph construction. In an undirected graph, every edge $(u, v)$ has a reverse copy $(v, u)$. When positive links were split into train/val/test, only the forward edge $(u, v)$ was held out from the message-passing graph, while the reverse copy $(v, u)$ accidentally remained in the PyG `edge_index`.

Because $(v, u)$ was still present in the message-passing graph, node $v$ passed its features directly into node $u$ during layer 1 and layer 2 convolutions. When the decoder was later queried for $(u, v)$, the dot-product easily detected the shared representation, creating a **false positive evaluation**.

### Resolution:
1. We modified `split_utils.py` to operate on unique undirected pairs ($u < v$).
2. Both $(u, v)$ and $(v, u)$ were strictly bound to the exact same split partition.
3. We implemented an automated verification function `verify_zero_leakage()` that executes mathematical assertions checking `set(test_edges).intersection(set(msg_edges)) == 0`.
4. This brought the valid test ROC-AUC to an honest **0.7405**, establishing a reliable, uncompromised benchmark.
