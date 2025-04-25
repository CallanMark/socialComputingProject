#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import optuna
import logging
import argparse
import torch
from datetime import datetime
from train import create_model, load_data, evaluate, set_seed, get_device
from utils import get_edge_type, to_hetero_batch

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for Ensemble GNN models')
    
    # Core parameters
    parser.add_argument('--dataset', type=str, default='politifact', 
                        choices=['politifact', 'gossipcop'],
                        help='Dataset name')
    
    # Optuna parameters
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the Optuna study')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout for study in seconds')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./optuna_results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Default batch size (will be overridden by Optuna)')
    
    return parser.parse_args()

def train_with_params(args, trial_params, device, train_dataset):
    """
    Train an ensemble model with the given hyperparameters and evaluate on validation set.
    Returns validation accuracy.
    """
    # Set all parameters from trial
    for param, value in trial_params.items():
        setattr(args, param, value)
    
    # Always use Ensemble model type
    args.model_type = 'Ensemble'
    args.ensemble_models = 'GAT,HAN,RGCN'
    
    # Reload data with the new batch size from trial
    train_loader, val_loader, test_loader, _ = load_data(args)
    
    # Determine input features
    sample_graph = train_dataset[0]
    num_features = sample_graph.num_node_features
    
    # Add this after initializing the training loop
    logger.info(f"Starting training with batch size: {args.batch_size}, learning rate: {args.lr}, hidden dim: {args.hidden_dim}")
    
    # Create model
    model = create_model(args, num_features, device)
    
    # Set up optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Train epoch
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass for ensemble model
            out = model(batch)
            y = batch.y
            
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
        
        # Log training progress
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}")
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.model_type)
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model state
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate on test set using best model
    if best_model_state:
        model.load_state_dict(best_model_state['model_state_dict'])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, args.model_type)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    else:
        test_acc = 0.0
                
    return best_val_acc, test_acc, best_model_state

def objective(trial, args, device, train_dataset):
    """Optuna objective function to maximize validation accuracy."""
    
    # Define hyperparameters to search
    trial_params = {}
    
    # Common parameters
    trial_params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    trial_params['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    trial_params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    trial_params['dropout'] = trial.suggest_float('dropout', 0.5, 0.7)
    
    # Ensemble-specific parameters
    trial_params['ensemble_method'] = trial.suggest_categorical(
        'ensemble_method', ['voting', 'average', 'concat', 'transform']
    )
    
    # Component model parameters
    trial_params['heads'] = trial.suggest_categorical('heads', [4, 8])
    trial_params['num_layers'] = trial.suggest_int('num_layers', 1, 4) 
    trial_params['pooling'] = trial.suggest_categorical('pooling', ['mean', 'max', 'add'])
    trial_params['use_self_loops'] = trial.suggest_categorical('use_self_loops', [True])
    
    # Add batch size as a hyperparameter
    trial_params['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    
    # Print current trial configuration
    config_str = f"\n{'='*80}\nTRIAL #{trial.number} CONFIGURATION:\n"
    for param, value in trial_params.items():
        config_str += f"  {param}: {value}\n"
    config_str += f"{'='*80}\n"
    print(config_str)
    logger.info(config_str)
    
    # Train with these parameters
    best_val_acc, test_acc, _ = train_with_params(
        args, trial_params, device, train_dataset
    )
    
    # Store test accuracy as user attribute for analysis
    trial.set_user_attr('test_accuracy', test_acc)
    
    # Print trial results
    result_str = f"\n{'='*80}\nTRIAL #{trial.number} RESULTS:\n"
    result_str += f"  Validation Accuracy: {best_val_acc:.4f}\n"
    result_str += f"  Test Accuracy: {test_acc:.4f}\n"
    result_str += f"{'='*80}\n"
    print(result_str)
    logger.info(result_str)
    
    return best_val_acc

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    set_seed(args.seed)
    device = get_device(args)
    
    # Create results directory
    if args.study_name is None:
        args.study_name = f"Ensemble_{args.dataset}_study"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_path, f"{args.study_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(results_dir, 'tuning.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Hyperparameter tuning started for Ensemble Model on {args.dataset}")
    logger.info(f"Results will be saved to {results_dir}")
    
    # Temporarily set model_type for data loading to get the dataset
    args.model_type = 'GAT'  # Use homogeneous data loading
    
    # Load only the dataset first (we'll reload with proper batch sizes during trials)
    _, _, _, train_dataset = load_data(args)
    
    # Create and run Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # We want to maximize validation accuracy
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, device, train_dataset),
        n_trials=args.n_trials,
        timeout=args.timeout,
        catch=(Exception,)
    )
    
    # Log best trial
    logger.info(f"Best trial:")
    logger.info(f"  Value (validation accuracy): {study.best_trial.value:.4f}")
    logger.info(f"  Test accuracy: {study.best_trial.user_attrs['test_accuracy']:.4f}")
    logger.info(f"  Params: {study.best_trial.params}")
    
    # Save study results
    study_results = {
        'best_params': study.best_trial.params,
        'best_val_accuracy': study.best_trial.value,
        'best_test_accuracy': study.best_trial.user_attrs['test_accuracy'],
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'test_accuracy': t.user_attrs.get('test_accuracy', None),
                'state': t.state.name
            }
            for t in study.trials
        ],
        'args': vars(args)
    }
    
    with open(os.path.join(results_dir, 'study_results.json'), 'w') as f:
        json.dump(study_results, f, indent=2)
    
    # Save visualization plots
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
        
        # Optimization history plot
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(results_dir, 'optimization_history.png'))
        
        # Parameter importance plot
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(results_dir, 'param_importances.png'))
        
        # Try to create contour plots for important parameters
        param_names = list(study.best_trial.params.keys())
        for i in range(min(len(param_names), 3)):
            for j in range(i + 1, min(len(param_names), 4)):
                try:
                    fig = plot_contour(study, params=[param_names[i], param_names[j]])
                    fig.write_image(os.path.join(results_dir, f'contour_{param_names[i]}_{param_names[j]}.png'))
                except:
                    pass
        
        logger.info(f"Visualization plots saved to {results_dir}")
    except ImportError:
        logger.warning("Could not create visualization plots. Make sure matplotlib and plotly are installed.")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    
    # Set best parameters to args
    for param, value in study.best_trial.params.items():
        setattr(args, param, value)
    
    # Set Ensemble parameters
    args.model_type = 'Ensemble'
    args.ensemble_models = 'GAT,HAN,RGCN'
    
    # Train best model and get results
    _, test_acc, best_model_state = train_with_params(
        args, study.best_trial.params, device, train_dataset
    )
    
    # Save final results
    final_results = {
        'best_params': study.best_trial.params,
        'test_accuracy': test_acc,
        'best_epoch': best_model_state['epoch'],
        'val_accuracy': best_model_state['val_acc']
    }
    
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save best model
    torch.save(best_model_state, os.path.join(results_dir, 'best_model.pt'))
    
    logger.info(f"Hyperparameter tuning completed successfully!")
    logger.info(f"Best parameters: {study.best_trial.params}")
    logger.info(f"Test accuracy with best parameters: {test_acc:.4f}")
    logger.info(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()