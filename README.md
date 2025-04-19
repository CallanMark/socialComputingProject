# Fake News Detection using Graph Neural Networks

This repository contains code for detecting fake news using graph neural networks, where news articles and user interactions are modeled as graphs.

## Installation

### Required packages
Install PyTorch <= 2.5: https://pytorch.org/get-started/previous-versions/  
Install PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.

## Models

The repository implements several graph neural network architectures:

- **GAT (Graph Attention Network)**: Utilizes attention mechanisms to weigh the importance of node relationships.
- **HAN (Heterogeneous Graph Attention Network)**: Extends GAT to heterogeneous graphs with different node and edge types.
- **RGCN (Relational Graph Convolutional Network)**: Handles different types of relationships between nodes.
- **Ensemble**: Combines multiple models for improved performance.

## Datasets

The code works with the UPFD (User-Publication-Fake-Detection) datasets:

- **PolitiFact**: News articles from PolitiFact with user engagements
- **GossipCop**: News articles from GossipCop with user engagements

## Running the Training Script

The `train.py` script allows you to train different graph neural network models for fake news detection.

### Basic Usage

```bash
python train.py --model_type MODEL --dataset DATASET [OPTIONS]
```

Where:
- `MODEL` is one of: GAT, HAN, RGCN, or Ensemble
- `DATASET` is one of: politifact, gossipcop

## Ensemble Training

Ensemble models combine multiple graph neural networks to improve prediction accuracy. Several combination methods are supported:

### 1. Voting Ensemble

The **voting** method takes predictions from each model and selects the most common prediction (majority vote) for each graph.

```bash
python train.py --model_type Ensemble --dataset politifact --ensemble_models GAT,HAN,RGCN --ensemble_method voting --hidden_dim 128 --epochs 100 --batch_size 32 --dropout 0.5
```

### 2. Average Ensemble

The **average** method combines the raw prediction logits (pre-softmax values) from each model by averaging them.

```bash
python train.py --model_type Ensemble --dataset politifact --ensemble_models GAT,HAN,RGCN --ensemble_method average --hidden_dim 128 --epochs 100 --lr 0.005 --dropout 0.5
```

### 3. Concatenation Ensemble

The **concat** method extracts graph embeddings from each model, concatenates them, and applies a final classification layer.

```bash
python train.py --model_type Ensemble --dataset politifact --ensemble_models GAT,HAN,RGCN --ensemble_method concat --hidden_dim 128 --epochs 150 --weight_decay 1e-4 --dropout 0.6
```

### 4. Transform Ensemble

The **transform** method concatenates embeddings, applies a transformation to a fixed hidden dimension, then makes the final prediction.

```bash
python train.py --model_type Ensemble --dataset politifact --ensemble_models GAT,HAN,RGCN --ensemble_method transform --hidden_dim 256 --epochs 150 --lr 0.001 --dropout 0.5
```

## Individual Model Options

### GAT (Graph Attention Network)

GAT uses attention mechanisms to weigh neighboring nodes differently based on their features:

```bash
python train.py --model_type GAT --dataset politifact --hidden_dim 128 --heads 8 --num_layers 2 --dropout 0.5 --pooling mean --epochs 100 --batch_size 32
```

Key parameters:
- `--heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of GAT layers (default: 2)
- `--pooling`: Method to convert node features to graph features (`mean`, `max`, or `add`)
- `--dropout`: Dropout rate for regularization (default: 0.5)

### HAN (Heterogeneous Graph Attention Network)

HAN extends GAT to handle heterogeneous graphs with different node and edge types:

```bash
python train.py --model_type HAN --dataset politifact --hidden_dim 128 --heads 8 --dropout 0.6 --use_self_loops --epochs 100
```

Key parameters:
- `--heads`: Number of attention heads (default: 8)
- `--use_self_loops`: Add self-loops to source nodes
- `--dropout`: Dropout rate for regularization (default: 0.6)

Note: HAN always uses batch_size=1 because of how heterogeneous graphs are batched.

### RGCN (Relational Graph Convolutional Network)

RGCN handles different types of relationships between nodes:

```bash
python train.py --model_type RGCN --dataset politifact --hidden_dim 128 --num_layers 2 --dropout 0.5 --pooling mean --epochs 100 --batch_size 32
```

Key parameters:
- `--num_layers`: Number of RGCN layers (default: 2)
- `--pooling`: Method to convert node features to graph features (`mean`, `max`, or `add`)
- `--dropout`: Dropout rate for regularization (default: 0.5)

## Common Training Options

These options apply to all model types:

- `--hidden_dim`: Dimensionality of hidden layers (default: 128)
- `--lr`: Learning rate (default: 0.005)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--weight_decay`: L2 regularization parameter (default: 5e-4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save_path`: Directory to save model and results (default: './runs')

Example with all options:

```bash
python train.py --model_type GAT --dataset politifact \
  --hidden_dim 256 --heads 8 --num_layers 3 \
  --dropout 0.6 --pooling max \
  --lr 0.001 --epochs 200 --batch_size 64 \
  --weight_decay 1e-5 --seed 123 \
  --save_path ./custom_runs
```

## Output

The script creates a timestamped directory for each run under the specified `save_path` (default: `./runs`). This directory contains:

- `best_model.pt`: The model with the highest validation accuracy
- `training_history.json`: Detailed training metrics and configuration
- `summary.json`: Key results and configuration summary
- `training.log`: Full training log

## Acknowledgements
This project contains modified code from [PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric).