"""
Builder script to generate notebook_a2.ipynb for Assignment 2.
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 2: Compare GNN Architectures on Cora\n",
                "\n",
                "## 1. Overview and Objectives\n",
                "In this assignment, we implement and empirically compare four distinct architectures on the Cora citation benchmark:\n",
                "1. **MLP** (Feature-only baseline)\n",
                "2. **GCN** (Symmetric neighborhood averaging)\n",
                "3. **GraphSAGE** (Sample-and-aggregate with representation concatenation)\n",
                "4. **GAT** (Multi-head attention over citations)\n",
                "\n",
                "### Experiments Conducted:\n",
                "- Dataset inspection (features, citation graph, homophily, and splits).\n",
                "- 3-seed benchmark tracking Test Accuracy, Macro F1, Training Time, and Parameter Counts.\n",
                "- Depth vs. Oversmoothing investigation (evaluating embedding cosine similarity collapse).\n",
                "- Edge removal sweep (0% to 100% edge drop vs. MLP control).\n",
                "- Qualitative neighborhood error analysis on 5 misclassified test papers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "source": [
                "import os\n",
                "import sys\n",
                "sys.path.insert(0, r\"D:\\gnn-practical-assignments\")\n",
                "\n",
                "import torch\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from assignment2_architectures.dataset_cora import load_cora_dataset, inspect_cora\n",
                "from assignment2_architectures.train_benchmark import run_architecture_benchmarks\n",
                "from assignment2_architectures.experiments.depth_and_edges import run_depth_experiment, run_edge_removal_experiment\n",
                "from assignment2_architectures.error_analysis import analyze_misclassifications\n",
                "%matplotlib inline\n",
                "print(\"Environment initialized.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Inspection\n",
                "We inspect the Cora graph topology, node vocabulary sparsity, and scientific topic distribution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "source": [
                "dataset, data = load_cora_dataset()\n",
                "inspect_cora(data, dataset)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Architecture Benchmarks (3 Random Seeds)\n",
                "We evaluate MLP, GCN, GraphSAGE, and GAT under a comparable hyperparameter budget, reporting mean and standard deviation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "source": [
                "benchmark_results = run_architecture_benchmarks(seeds=(42, 123, 999))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Diagnostic 1: GNN Depth vs. Oversmoothing\n",
                "We vary depth from 2 to 16 layers to observe performance collapse and embedding cosine similarity convergence."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "source": [
                "depth_data = run_depth_experiment(depths=(2, 4, 8, 16))\n",
                "\n",
                "# Plot Depth vs Accuracy and Cosine Similarity\n",
                "fig, ax1 = plt.subplots(figsize=(8, 4.5))\n",
                "color = 'tab:blue'\n",
                "ax1.set_xlabel('GCN Depth (Layers)', fontsize=12)\n",
                "ax1.set_ylabel('Test Accuracy', color=color, fontsize=12)\n",
                "ax1.plot(depth_data['depths'], depth_data['accuracies'], marker='o', color=color, linewidth=2, label='Test Accuracy')\n",
                "ax1.tick_params(axis='y', labelcolor=color)\n",
                "ax1.set_ylim(0, 1.0)\n",
                "\n",
                "ax2 = ax1.twinx()\n",
                "color = 'tab:red'\n",
                "ax2.set_ylabel('Edge Cosine Similarity', color=color, fontsize=12)\n",
                "ax2.plot(depth_data['depths'], depth_data['similarities'], marker='s', color=color, linestyle='--', linewidth=2, label='Cosine Sim')\n",
                "ax2.tick_params(axis='y', labelcolor=color)\n",
                "ax2.set_ylim(0.9, 1.01)\n",
                "\n",
                "plt.title('GCN Depth vs. Accuracy and Embedding Oversmoothing', fontsize=14)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Diagnostic 2: Edge Sparsity Sweep (vs. MLP Control)\n",
                "We systematically drop edges from 0% to 100% to prove dependency on graph topology."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "source": [
                "edge_data = run_edge_removal_experiment(drop_ratios=(0.0, 0.25, 0.50, 0.75, 1.0))\n",
                "\n",
                "plt.figure(figsize=(8, 4.5))\n",
                "drops = [r * 100 for r in edge_data['drop_ratios']]\n",
                "plt.plot(drops, [acc * 100 for acc in edge_data['MLP']], 'k--', label='MLP (Control - No Edges)', linewidth=2)\n",
                "plt.plot(drops, [acc * 100 for acc in edge_data['GCN']], 'b-o', label='GCN', linewidth=2)\n",
                "plt.plot(drops, [acc * 100 for acc in edge_data['GraphSAGE']], 'g-s', label='GraphSAGE', linewidth=2)\n",
                "plt.plot(drops, [acc * 100 for acc in edge_data['GAT']], 'm-^', label='GAT', linewidth=2)\n",
                "plt.xlabel('Edges Removed (%)', fontsize=12)\n",
                "plt.ylabel('Test Accuracy (%)', fontsize=12)\n",
                "plt.title('Effect of Edge Sparsity on GNN Architectures vs. MLP Control', fontsize=14)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.legend(fontsize=11)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Qualitative Neighborhood Error Analysis\n",
                "We inspect 5 specific test nodes misclassified by GCN to understand neighborhood homophily breakdown."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "source": [
                "analyze_misclassifications(num_samples=5)"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"D:\gnn-practical-assignments\assignment2_architectures\notebook_a2.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("notebook_a2.ipynb generated successfully.")
