"""
Utility script to generate notebook_a1.ipynb with self-contained path resolution.
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 1: Build and Verify a GCN from Scratch\n",
                "\n",
                "## 1. Overview and Problem Statement\n",
                "In this assignment, we construct a 2-layer Graph Convolutional Network (GCN) entirely from scratch using pure PyTorch tensor operations—without ready-made GNN libraries.\n",
                "\n",
                "### Key Requirements Addressed:\n",
                "1. **Synthetic Graph**: At least 100 nodes (we use 120), exactly 3 numerical features, 2 target classes (0 and 1).\n",
                "2. **Graph Structure Utility**: Designed with overlapping feature distributions (making feature-only models ambiguous) and homophilic community connections (allowing graph convolutions to resolve the ambiguity).\n",
                "3. **Scratch Implementation**: Explicit self-loops ($\\tilde{A} = A + I$), degree normalization ($\\hat{A} = \\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2}$), message passing, and two-layer forward pass.\n",
                "4. **Correctness Checks**:\n",
                "   - Single-node output matched against explicit manual calculation.\n",
                "   - Permutation equivariance verified across random node reorderings.\n",
                "5. **Ablation Studies**: Rigorous comparison against an MLP baseline, randomized edges, and empty graph."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "# Automatically add repository root to sys.path\n",
                "repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "if repo_root not in sys.path:\n",
                "    sys.path.insert(0, repo_root)\n",
                "\n",
                "import torch\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from assignment1_from_scratch.dataset import generate_synthetic_graph\n",
                "from assignment1_from_scratch.gcn_scratch import GCN2LayerScratch, train_gcn, normalize_adjacency\n",
                "from assignment1_from_scratch.mlp import MLPBaseline, train_mlp\n",
                "from assignment1_from_scratch.tests.test_manual_calc import test_manual_calculation_single_node\n",
                "from assignment1_from_scratch.tests.test_equivariance import test_permutation_equivariance\n",
                "%matplotlib inline\n",
                "print(\"All imports and path resolution successful!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Generation & Verification\n",
                "We inspect the synthetic graph properties and verify the strict train/val/test splits."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "data = generate_synthetic_graph(num_nodes=120, num_features=3, seed=42)\n",
                "x, adj, y = data['x'], data['adj'], data['y']\n",
                "print(f\"Node feature matrix shape: {x.shape}\")\n",
                "print(f\"Adjacency matrix shape:     {adj.shape}\")\n",
                "print(f\"Total undirected edges:      {int(adj.sum().item() / 2)}\")\n",
                "print(f\"Train / Val / Test split:    {data['train_mask'].sum().item()} / {data['val_mask'].sum().item()} / {data['test_mask'].sum().item()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Correctness Checks\n",
                "### Test A: Single-Node Manual Calculation vs. Layer Output\n",
                "Calculates $H_0 = \\sum_{j \\in \\mathcal{N}(0) \\cup \\{0\\}} \\frac{1}{\\sqrt{\\tilde{d}_0 \\tilde{d}_j}} (x_j W + b)$ by hand and checks for exact equality."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_manual_calculation_single_node()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Test B: Permutation Equivariance Verification\n",
                "Proves that reordering $(X, A)$ by permutation $P$ produces output $\\hat{Y}_{\\text{perm}} = P \\hat{Y}$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_permutation_equivariance()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Training & Ablation Experiments\n",
                "We train the GCN on the true graph, compare against an MLP baseline, and run edge removal / randomization ablations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "from assignment1_from_scratch.run_experiments import run_assignment1_experiments\n",
                "run_assignment1_experiments(seed=42)"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"D:\gnn-practical-assignments\assignment1_from_scratch\notebook_a1.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated with sys.path fix.")
