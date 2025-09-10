# GNN-Node-Classification
**Node Classification with Graph Neural Networks (GCN, GraphSAGE, GAT) on Citation Datasets**\
This repository contains PyTorch Geometric implementations of three foundational GNN models — **GCN**, **GraphSAGE**, and **GAT** — applied to node classification tasks on citation networks (Cora, Citeseer).\
Includes:
-	**Reproducible training scripts** (--model gcn|sage|gat)
-	**Training/validation/test accuracy plots**
-	**Comparison table of results**
-	**2-page report analyzing performance, scalability, and trade-offs**

  # Node Classification with Graph Neural Networks (GCN, GraphSAGE, GAT)

Reproducible PyTorch Geometric implementations of three foundational Graph Neural Network models applied to **node classification** on citation networks (**Cora**, **Citeseer**).

This repo is part of my PhD preparation journey in Graph Neural Networks (GNNs).

---

## 🚀 Features
- Implementations of **GCN**, **GraphSAGE**, and **GAT**
- Training on **Cora** and **Citeseer** datasets
- Easy model selection with CLI flag (`--model gcn|sage|gat`)
- Logs training/validation/test accuracy
- Saves loss/accuracy curves to PNG
- Includes a **2-page report** analyzing results

---

## 📂 Repository Structure
gnn-node-classification/\
│\
├── data/ # Cora/Citeseer datasets (auto-downloaded)\
├── models/ # GCN, GraphSAGE, GAT definitions\
├── results/ # Saved plots + logs\
├── report/ # 2-page PDF report\
│\
├── run_gnn.py # Main training script\
├── requirements.txt # Dependencies\
└── README.md # This file\


---

## ⚙️ Installation

```
# Clone the repo
git clone https://github.com/UmarRazzaq/GNN-Node-Classification.git
cd gnn-node-classification

# Create environment
conda create -n gnn python=3.10 -y
conda activate gnn

# Install dependencies
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
torch
torch-geometric
matplotlib
```
---
## ▶️ Usage

Train a model with one command:
```
# Train GCN on Cora
python run_gnn.py --model gcn --dataset Cora

# Train GraphSAGE on Citeseer
python run_gnn.py --model sage --dataset Citeseer

# Train GAT on Cora
python run_gnn.py --model gat --dataset Cora
```
Options:
-	`--model`: `gcn`, `sage`, or `gat`
-	`--dataset`: `Cora` or `Citeseer`
-	`--epochs`: number of training epochs (default: 200)
---
## 📊 Results
| Model         | Cora (Test Acc) | Citeseer (Test Acc) |
| ------------- | --------------- | ------------------- |
| **GCN**       | \~81–83%        | \~70%               |
| **GraphSAGE** | \~82–84%        | \~71%               |
| **GAT**       | \~83–85%        | \~72–73%            |


Plots available in `results/`.

---
## 📄 Report

A concise **2-page PDF report** is included in `report/`, discussing:
-	Problem setup
-	Experimental setup
-	Accuracy comparison
-	Analysis of trade-offs
-	Conclusion + next steps
---
## 🔮 Next Steps

-	Extend to graph classification (e.g., molecules, OGB datasets)
-	Add link prediction experiments
-	Explore self-supervised GNNs (e.g., GraphCL, DGI)
---
## 📚 References
-	[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
-	Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks
-  Hamilton et al. (2017): Inductive Representation Learning on Large Graphs (GraphSAGE)
-  Veličković et al. (2018): Graph Attention Networks
