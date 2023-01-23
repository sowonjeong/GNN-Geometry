# GNN-Geometry

## 1. Overview
This repository contains the implementation and experiments of different graph operators suggested in \cite{}

## 2. Setup
### 2-1. Installation

**Dependencies**
    * Python 3
    * PyTorch 

**Installation**

1. `git clone https://github.com/donnate/GNN-Geometry.git`
2. `cd GNN-Geometry`
3. `pip install -r requirements.txt`

### 2-2. Datasets

The `data\` folder contains source file for:
* Amazon: Amazon Photo
* Coauthor: Coauthor CS
* Planetoid: Cora, Pubmed, Citeseer
* WebKB: Cornell, Wisconsin
* Benchmark: PATTERN, CLUSTER
* OGB-arxiv
* WikiCS

## 3. Usage

* Newly defined families of operators have been implemented in `operators.py`
* `experiments-datasets-demo.ipynb` gives a demo on how to apply these operators on node classification task. 
* `metrics.py` for the evaluation of properties of embedding space. 

## 4. Examples

All the examples are to show the effect of the type of operators in the resulting embedding space. The target task is limited to node prediction tasks. 

