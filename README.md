# Fake News Detection using Graph Neural Network Ensembles

This repository implements an ensemble approach to fake news detection using multiple graph neural networks. The ensemble methods combine different GNN architectures to achieve improved detection performance on news article graphs.

## Installation

### 1. Create and activate conda environment
```bash
# Create new conda environment with Python 3.10
conda create -n sc python=3.10
conda activate sc
```
install conda: https://www.anaconda.com/docs/getting-started/miniconda/install#linux

### 2. Install PyTorch
Choose one of the following methods:

#### Using conda (recommended):
```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

#### Using pip:
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install PyTorch Geometric (PyG)
```bash
pip install torch_geometric
```

You can also visits PyTorch and PyG's sites for other installation options:  
- PyTorch: https://pytorch.org/get-started/locally/
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## Ensemble Methods

The project implements four ensemble strategies that combine GAT, HAN, and RGCN models:

### 1. Voting Ensemble

Takes predictions from each model and selects the most common prediction (majority vote) for each graph.

```bash
python train.py --model_type Ensemble --dataset [politifact|gossipcop] \
  --ensemble_models GAT,HAN,RGCN \
  --ensemble_method voting \
  --hidden_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.5 \
  --weight_decay 5e-4 \
  --num_layers 2
```

### 2. Average Ensemble

Combines raw prediction logits (pre-softmax values) from each model by averaging them.

```bash
python train.py --model_type Ensemble --dataset [politifact|gossipcop] \
  --ensemble_models GAT,HAN,RGCN \
  --ensemble_method average \
  --hidden_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.5 \
  --weight_decay 5e-4 \
  --num_layers 2
```

### 3. Concatenation Ensemble

Extracts graph embeddings from each model, concatenates them, and applies a final classification layer.

```bash
python train.py --model_type Ensemble --dataset [politifact|gossipcop] \
  --ensemble_models GAT,HAN,RGCN \
  --ensemble_method concat \
  --hidden_dim 128 \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.6 \
  --weight_decay 5e-4 \
  --num_layers 2
```

### 4. Transform Ensemble

Concatenates embeddings, applies a transformation to a fixed hidden dimension, then makes the final prediction.

```bash
python train.py --model_type Ensemble --dataset [politifact|gossipcop] \
  --ensemble_models GAT,HAN,RGCN \
  --ensemble_method transform \
  --hidden_dim 256 \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.5 \
  --weight_decay 5e-4 \
  --num_layers 2
```

### Ensemble Parameters
- `--dataset`: Can be `gossipcop` or `politifact`
- `--ensemble_models`: Comma-separated list of models to combine (GAT, HAN, RGCN)
- `--ensemble_method`: Combination strategy (voting, average, concat, transform)
- `--hidden_dim`: Dimension of hidden layers/embeddings (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 0.005)
- `--dropout`: Dropout rate for regularization (default: 0.5)
- `--weight_decay`: L2 regularization parameter (default: 5e-4)
- `--num_layers`: Number of layers in each base model (default: 2)

## Base Models

The ensemble combines three different graph neural network architectures:

### GAT (Graph Attention Network)

```bash
python train.py --model_type GAT --dataset [politifact|gossipcop] \
  --hidden_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.5 \
  --weight_decay 5e-4 \
  --heads 8 \
  --num_layers 2 \
  --pooling mean
```

Additional parameters:
- `--heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of GAT layers (default: 2)
- `--pooling`: Node feature aggregation method (mean, max, add)

### HAN (Heterogeneous Graph Attention Network)

```bash
python train.py --model_type HAN --dataset [politifact|gossipcop] \
  --hidden_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.6 \
  --weight_decay 5e-4 \
  --heads 8 \
  --num_layers 2 \
  --use_self_loops
```

Additional parameters:
- `--heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformation layers (default: 2)
- `--use_self_loops`: Add self-loops to source nodes

### RGCN (Relational Graph Convolutional Network)

```bash
python train.py --model_type RGCN --dataset [politifact|gossipcop] \
  --hidden_dim 128 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.005 \
  --dropout 0.5 \
  --weight_decay 5e-4 \
  --num_layers 2 \
  --pooling mean
```

Additional parameters:
- `--num_layers`: Number of RGCN layers (default: 2)
- `--pooling`: Node feature aggregation method (mean, max, add)

## Datasets

The code works with the UPFD (User-Publication-Fake-Detection) datasets:

- **PolitiFact**: News articles from PolitiFact with user engagements
- **GossipCop**: News articles from GossipCop with user engagements

Each dataset represents news articles as heterogeneous graphs containing:
- News source nodes with article content
- User nodes with interaction history
- Edges representing user engagement patterns

## Output

The script creates a timestamped directory for each run under `./runs` containing:

- `best_model.pt`: Model with highest validation accuracy
- `training_history.json`: Detailed training metrics and configuration
- `summary.json`: Key results and configuration summary
- `training.log`: Full training log

## Additional Parameters

Common parameters available for all models:
```bash
--save_path   Directory to save model and results (default: './runs')
--seed        Random seed for reproducibility (default: 42)
--device      Device to use (cuda or cpu)
```

## Acknowledgements
This project contains modified code from [PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric).