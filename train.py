#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import argparse
import torch
import numpy as np
import random
import datetime
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from modeling import (
    GATForGraphClassification, 
    HANForGraphClassification, 
    RGCNForGraphClassification,
    EnsembleGraphClassifier
)
from utils import convert_to_heterogeneous, get_edge_type, to_hetero_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GNN models for fake news detection')
    
    # Core parameters
    parser.add_argument('--model_type', type=str, default='GAT',
                        choices=['GAT', 'HAN', 'RGCN', 'Ensemble'],
                        help='Type of model to use')
    parser.add_argument('--dataset', type=str, default='politifact', 
                        choices=['politifact', 'gossipcop'],
                        help='Dataset name')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Optional parameters
    parser.add_argument('--ensemble_models', type=str, default='GAT,HAN,RGCN',
                        help='Models to include in ensemble, comma-separated')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./runs',
                        help='Directory to save model and results')
    
    # Model-specific parameters
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads for GAT/HAN')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'add'],
                        help='Pooling method for graph-level representations')
    parser.add_argument('--use_self_loops', action='store_true',
                        help='Add self-loops to source nodes (for HAN model)')
    parser.add_argument('--ensemble_method', type=str, default='voting',
                        choices=['voting', 'average', 'concat', 'transform'],
                        help='Method for combining ensemble models')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 penalty)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def get_device(args):
    """Determine the device to use."""
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def load_data(args):
    """Load and preprocess datasets."""
    logger.info(f"Loading {args.dataset} dataset with feature type 'bert'")
    
    data_dir = os.path.join('data', 'upfd')
    
    try:
        # Load datasets
        train_dataset = UPFD(data_dir, name=args.dataset, feature='bert', split='train')
        val_dataset = UPFD(data_dir, name=args.dataset, feature='bert', split='val')
        test_dataset = UPFD(data_dir, name=args.dataset, feature='bert', split='test')
        
        logger.info(f"Dataset loaded: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test graphs")
        
        # Process datasets based on model type
        if args.model_type == 'HAN':
            logger.info("Converting homogeneous graphs to heterogeneous for HAN model")
            train_dataset = convert_to_heterogeneous(train_dataset, add_source_self_loop=args.use_self_loops)
            val_dataset = convert_to_heterogeneous(val_dataset, add_source_self_loop=args.use_self_loops)
            test_dataset = convert_to_heterogeneous(test_dataset, add_source_self_loop=args.use_self_loops)
            
            # Create dataloaders for heterogeneous graphs
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        else:
            # Create dataloaders for homogeneous graphs
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        logger.info("Data loading and preprocessing completed")
        return train_loader, val_loader, test_loader, train_dataset
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_model(args, num_features, device, metadata=None):
    """Create model based on specified type."""
    logger.info(f"Creating {args.model_type} model")
    
    try:
        if args.model_type == 'GAT':
            model = GATForGraphClassification(
                in_channels=num_features,
                hidden_channels=args.hidden_dim,
                num_classes=2,  # Binary classification
                num_layers=args.num_layers,
                dropout=args.dropout,
                pooling=args.pooling,
                heads=args.heads,
                v2=False,  # Use standard GAT
                concat=True  # Concatenate attention heads
            )
        elif args.model_type == 'HAN':
            model = HANForGraphClassification(
                in_channels=num_features,
                hidden_channels=args.hidden_dim,
                out_channels=args.hidden_dim,
                num_classes=2,  # Binary classification
                heads=args.heads,
                metadata=metadata,
                dropout=args.dropout,
                num_layers=args.num_layers
            )
        elif args.model_type == 'RGCN':
            model = RGCNForGraphClassification(
                in_channels=num_features,
                hidden_channels=args.hidden_dim,
                num_classes=2,  # Binary classification
                num_relations=2,  # Source-to-user and user-to-user
                num_bases=None,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pooling=args.pooling
            )
        elif args.model_type == 'Ensemble':
            # Parse ensemble models
            model_types = args.ensemble_models.split(',')
            models = []
            
            for model_type in model_types:
                if model_type == 'GAT':
                    models.append(GATForGraphClassification(
                        in_channels=num_features,
                        hidden_channels=args.hidden_dim,
                        num_classes=2,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        pooling=args.pooling,
                        heads=args.heads,
                        v2=False,
                        concat=True
                    ))
                elif model_type == 'HAN':
                    # For simplicity, assume HAN models in ensemble will use converted data
                    models.append(HANForGraphClassification(
                        in_channels=num_features,
                        hidden_channels=args.hidden_dim,
                        out_channels=args.hidden_dim,
                        num_classes=2,
                        heads=args.heads,
                        metadata=(['source', 'user'], [('source', 'to', 'user'), ('user', 'to', 'user'), ('source', 'to', 'source')]),
                        dropout=args.dropout,
                        num_layers=args.num_layers
                    ))
                elif model_type == 'RGCN':
                    models.append(RGCNForGraphClassification(
                        in_channels=num_features,
                        hidden_channels=args.hidden_dim,
                        num_classes=2,
                        num_relations=2,
                        num_bases=None,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        pooling=args.pooling
                    ))
            
            model = EnsembleGraphClassifier(
                models=models,
                ensemble_method=args.ensemble_method,
                num_classes=2,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        model = model.to(device)
        logger.info(f"Model created: {args.model_type}")
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def train_epoch(model, loader, optimizer, criterion, device, model_type='GAT'):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass depends on model type
        if model_type == 'GAT':
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr if hasattr(batch, 'edge_attr') else None)
            y = batch.y
        elif model_type == 'HAN':
            # For HAN, we're using batch_size=1, so no need for batch vector
            out = model(batch.x_dict, batch.edge_index_dict)
            # For heterogeneous data, label is stored in the 'source' node type
            y = batch['source'].y
        elif model_type == 'RGCN':
            # Generate edge type based on source nodes
            edge_type = get_edge_type(batch.edge_index, 
                                      source_indices=batch.ptr[:-1].tolist() if hasattr(batch, 'ptr') else [0])
            out = model(batch.x, batch.edge_index, edge_type, batch.batch)
            y = batch.y
        elif model_type == 'Ensemble':
            # Ensemble model handles different input types internally
            out = model(batch)
            # Need to determine label source based on input type
            if hasattr(batch, 'x_dict'):  # Heterogeneous graph
                y = batch['source'].y
            else:  # Homogeneous graph
                y = batch.y
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Handle shape mismatch 
        if len(out.shape) == 1:  # If out is [2] shape
            out = out.unsqueeze(0)  # Make it [1, 2]
        
        # Squeeze y if needed
        if y.size(0) == 1 and len(y.shape) > 1:
            y = y.squeeze()
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, model_type='GAT'):
    """Evaluate model on given dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass depends on model type
            if model_type == 'GAT':
                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr if hasattr(batch, 'edge_attr') else None)
                y = batch.y
            elif model_type == 'HAN':
                # For HAN, we're using batch_size=1, so no need for batch vector
                out = model(batch.x_dict, batch.edge_index_dict)
                # For heterogeneous data, label is stored in the 'source' node type
                y = batch['source'].y
            elif model_type == 'RGCN':
                # Generate edge type based on source nodes
                edge_type = get_edge_type(batch.edge_index, 
                                          source_indices=batch.ptr[:-1].tolist() if hasattr(batch, 'ptr') else [0])
                out = model(batch.x, batch.edge_index, edge_type, batch.batch)
                y = batch.y
            elif model_type == 'Ensemble':
                # Ensemble model handles different input types internally
                out = model(batch)
                # Need to determine label source based on input type
                if hasattr(batch, 'x_dict'):  # Heterogeneous graph
                    y = batch['source'].y
                else:  # Homogeneous graph
                    y = batch.y
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Handle shape mismatch
            if len(out.shape) == 1:  # If out is [2] shape
                out = out.unsqueeze(0)  # Make it [1, 2]
            
            # Squeeze y if needed
            if y.size(0) == 1 and len(y.shape) > 1:
                y = y.squeeze()
            
            loss = criterion(out, y)
            total_loss += loss.item() * batch.num_graphs
            
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    set_seed(args.seed)
    device = get_device(args)
    
    # Create timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}_{args.dataset}_{timestamp}"
    run_dir = os.path.join(args.save_path, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up logging to file in the run directory
    file_handler = logging.FileHandler(os.path.join(run_dir, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Run directory created: {run_dir}")
    logger.info(f"Command-line arguments: {args}")
    
    # Load data
    train_loader, val_loader, test_loader, train_dataset = load_data(args)
    
    # Determine input features and metadata
    if args.model_type == 'HAN':
        # For HAN, we need to determine feature dimensions from the first graph
        # and extract metadata
        sample_graph = train_loader.dataset[0]
        num_features = {node_type: sample_graph[node_type].x.size(-1) 
                       for node_type in sample_graph.node_types}
        metadata = sample_graph.metadata()
    else:
        # For other models, feature dimension is the same for all nodes
        sample_graph = train_dataset[0]
        num_features = sample_graph.num_node_features
    
    # Create model
    model = create_model(args, num_features, device, 
                         metadata=metadata if args.model_type == 'HAN' else None)
    
    # Set up optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_accs = []
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.model_type)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.model_type)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}, "
                   f"Best Val Acc: {best_val_acc:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save best model
    if best_model_state:
        model_save_path = os.path.join(run_dir, "best_model.pt")
        torch.save(best_model_state, model_save_path)
        logger.info(f"Best model saved to {model_save_path}")
        
        # Load best model for testing
        model.load_state_dict(best_model_state['model_state_dict'])
    
    # Test best model
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, args.model_type)
    logger.info(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Save training history and results
    history = {
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'timestamp': timestamp
    }
    
    history_save_path = os.path.join(run_dir, "training_history.json")
    with open(history_save_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    logger.info(f"Training history saved to {history_save_path}")
    
    # Create a summary file with key results
    summary = {
        'model_type': args.model_type,
        'dataset': args.dataset,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time': training_time,
        'timestamp': timestamp
    }
    
    summary_save_path = os.path.join(run_dir, "summary.json")
    with open(summary_save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Summary saved to {summary_save_path}")
    
    return test_acc

if __name__ == "__main__":
    main()