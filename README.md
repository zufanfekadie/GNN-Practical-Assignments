# Graph Neural Network (GNN) Practical Assignments & MLOps System

This repository provides complete, reproducible implementations, automated test suites, experimental benchmarks, and an end-to-end MLOps pipeline covering all four GNN assignments.

---

## Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI: Tests, linting, promotion gate
├── assignment1_from_scratch/          # Assignment 1: GCN from Scratch (Pure PyTorch)
│   ├── dataset.py                     # Synthetic graph with homophily & ambiguous features
│   ├── gcn_scratch.py                 # Pure tensor operations (self-loops, degree norm, forward pass)
│   ├── mlp.py                         # Feature-only baseline
│   ├── run_experiments.py             # Ablation study (True graph vs MLP vs Shuffled vs Empty)
│   ├── notebook_a1.ipynb              # Interactive notebook with outputs
│   └── tests/
│       ├── test_manual_calc.py        # Hand-calculation correctness verification
│       └── test_equivariance.py       # Permutation equivariance verification
├── assignment2_architectures/         # Assignment 2: Architectural Comparison on Cora
│   ├── dataset_cora.py                # Dataset loader & topological inspection
│   ├── models.py                      # Custom MLP, GCN, GraphSAGE, and GAT classes
│   ├── train_benchmark.py             # 3-seed benchmark with validation checkpointing
│   ├── experiments/
│   │   └── depth_and_edges.py         # Depth vs oversmoothing & edge removal sweeps
│   ├── error_analysis.py              # Qualitative breakdown of 5 misclassified nodes
│   └── notebook_a2.ipynb              # Interactive notebook with Matplotlib plots
├── assignment3_link_prediction/       # Assignment 3: Link Prediction & Data Leakage
│   ├── split_utils.py                 # Strict leak-free splitting & negative sampling
│   ├── models_link.py                 # GNN encoder, Dot-product & Symmetric MLP decoders
│   ├── train_link_predictor.py         # Benchmarks, leaky comparison, top-10 candidate discovery
│   └── notebook_a3.ipynb              # Interactive link prediction notebook
├── assignment4_mlops/                 # Assignment 4: Production MLOps Pipeline
│   ├── app/
│   │   ├── main.py                    # FastAPI service with failure handling
│   │   └── schemas.py                 # Pydantic request and response schemas
│   ├── tracking/
│   │   ├── untracked_baseline.py      # Bare script demonstrating operational flaws
│   │   ├── train_tracked.py           # MLflow tracking suite & model packaging
│   │   ├── registry_gate.py           # Model registry with validation promotion gate
│   │   └── rollback_demo.py           # Automated production rollback mechanism
│   ├── monitoring/
│   │   └── drift_detector.py          # Structural (degree) & embedding drift monitoring
│   ├── tests/
│   │   └── test_api.py                # Pytest suite for API, inductive nodes & edge cases
│   └── notebook_a4.ipynb              # Complete MLOps demonstration notebook
├── report/
│   └── TECHNICAL_REPORT.md            # Comprehensive technical report
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Container orchestration
├── requirements.txt                   # Locked dependencies
└── README.md
```

---

## Quick Setup & Reproduction

### 1. Environment Setup
Clone the repository and install dependencies:
```powershell
python -m pip install -r requirements.txt
```

Set the Python path in your terminal:
```powershell
$env:PYTHONPATH="D:\gnn-practical-assignments"
```

---

## How to Run Each Assignment

### Assignment 1: GCN from Scratch
1. Run Hand-Calculation Verification:
   ```powershell
   python assignment1_from_scratch/tests/test_manual_calc.py
   ```
2. Run Permutation Equivariance Verification:
   ```powershell
   python assignment1_from_scratch/tests/test_equivariance.py
   ```
3. Run Ablation Study & Comparison:
   ```powershell
   python assignment1_from_scratch/run_experiments.py
   ```

### Assignment 2: Architecture Comparison on Cora
1. Inspect Cora Dataset:
   ```powershell
   python assignment2_architectures/dataset_cora.py
   ```
2. Run 3-Seed Benchmark:
   ```powershell
   python assignment2_architectures/train_benchmark.py
   ```
3. Run Depth Oversmoothing & Edge Sparsity Sweeps:
   ```powershell
   python assignment2_architectures/experiments/depth_and_edges.py
   ```
4. Run Neighborhood Error Analysis:
   ```powershell
   python assignment2_architectures/error_analysis.py
   ```

### Assignment 3: Link Predictor & Data Leakage
1. Run Zero-Leakage Asserts:
   ```powershell
   python assignment3_link_prediction/split_utils.py
   ```
2. Run Decoder Symmetry Test:
   ```powershell
   python assignment3_link_prediction/models_link.py
   ```
3. Run Link Benchmarks, Leaky Comparison & Candidate Discovery:
   ```powershell
   python assignment3_link_prediction/train_link_predictor.py
   ```

### Assignment 4: MLOps Pipeline
1. Run MLflow Tracking & Model Packaging:
   ```powershell
   python assignment4_mlops/tracking/train_tracked.py
   ```
2. Test Model Registry Promotion Gate:
   ```powershell
   python assignment4_mlops/tracking/registry_gate.py
   ```
3. Run API and Failure Handling Tests:
   ```powershell
   python assignment4_mlops/tests/test_api.py
   ```
4. Run Drift Monitoring:
   ```powershell
   python assignment4_mlops/monitoring/drift_detector.py
   ```
5. Test Automated Rollback:
   ```powershell
   python assignment4_mlops/tracking/rollback_demo.py
   ```

---

## Docker Deployment
Build and run the container locally:
```powershell
docker build -t gnn-serving-api .
docker run -p 8000:8000 gnn-serving-api
```
Navigate to `http://localhost:8000/docs` to test interactive Swagger API endpoints.
