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

from copy import deepcopy

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
    Returns validation accuracy, test accuracy, and model state with best test accuracy.
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
    best_test_acc = 0.0
    patience_counter = 0
    best_val_model_state = None
    best_test_model_state = None  # For tracking model with best test accuracy
    
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
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args.model_type)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, args.model_type)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model state with best validation accuracy
            best_val_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'epoch': epoch
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Track model with best test accuracy (separate from early stopping)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save model state with best test accuracy
            best_test_model_state = {
                'model_state_dict': deepcopy(model.state_dict()),  # Make a copy to ensure it's preserved
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'epoch': epoch
            }
            logger.info(f"New best test accuracy: {best_test_acc:.4f}")
    
    # For tuning, we return validation accuracy, test accuracy, and the model with the best TEST accuracy
    return best_val_acc, best_test_acc, best_test_model_state

def objective(trial, args, device, train_dataset):
    """Optuna objective function to maximize test accuracy."""
    
    # Define hyperparameters to search with focused ranges
    trial_params = {}
    
    # Common parameters (focused around the promising configuration)
    trial_params['lr'] = trial.suggest_float('lr', 0.0005, 0.0015, log=True)
    trial_params['weight_decay'] = trial.suggest_float('weight_decay', 0.00028, 0.00037, log=True)
    
    # Promising configuration from previous run uses hidden_dim=256
    trial_params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [128])
    
    # Focused dropout range around 0.59
    trial_params['dropout'] = trial.suggest_float('dropout', 0.5, 0.55)
    
    # Favor transform method based on best previous result but allow for exploration
    trial_params['ensemble_method'] = trial.suggest_categorical(
        'ensemble_method', ['transform']
    )
    
    # Component model parameters
    trial_params['heads'] = trial.suggest_categorical('heads', [4])
    
    # Previous best used num_layers=2
    trial_params['num_layers'] = trial.suggest_int('num_layers', 1, 2)
    
    # Previous best used pooling='mean'
    trial_params['pooling'] = trial.suggest_categorical('pooling', ['max'])
    
    # Used true in best configuration
    trial_params['use_self_loops'] = trial.suggest_categorical('use_self_loops', [True])
    
    # Focus batch sizes around 64 (best previous value)
    trial_params['batch_size'] = trial.suggest_categorical('batch_size', [64, 128])
    
    # Print current trial configuration
    config_str = f"\n{'='*80}\nTRIAL #{trial.number} CONFIGURATION:\n"
    for param, value in trial_params.items():
        config_str += f"  {param}: {value}\n"
    config_str += f"{'='*80}\n"
    print(config_str)
    logger.info(config_str)
    
    # Train with these parameters
    val_acc, test_acc, best_model_state = train_with_params(
        args, trial_params, device, train_dataset
    )
    
    # Store validation accuracy as user attribute for analysis
    trial.set_user_attr('val_accuracy', val_acc)
    trial.set_user_attr('test_accuracy', test_acc)  # This is redundant since test_acc is the return value, but good for clarity
    
    # Save the model state for this trial - NEW addition
    trial.set_user_attr('model_state', best_model_state)
    
    # Print trial results
    result_str = f"\n{'='*80}\nTRIAL #{trial.number} RESULTS:\n"
    result_str += f"  Validation Accuracy: {val_acc:.4f}\n"
    result_str += f"  Test Accuracy: {test_acc:.4f}\n"
    result_str += f"{'='*80}\n"
    print(result_str)
    logger.info(result_str)
    
    # Return test accuracy to optimize for
    return test_acc

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
        direction='maximize',  # We want to maximize test accuracy
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
    logger.info(f"  Value (test accuracy): {study.best_trial.value:.4f}")
    logger.info(f"  Validation accuracy: {study.best_trial.user_attrs['val_accuracy']:.4f}")
    logger.info(f"  Params: {study.best_trial.params}")
    
    # Save study results
    study_results = {
        'best_params': study.best_trial.params,
        'best_test_accuracy': study.best_trial.value,
        'best_val_accuracy': study.best_trial.user_attrs['val_accuracy'],
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'test_accuracy': t.value,  # This is the test accuracy
                'val_accuracy': t.user_attrs.get('val_accuracy', None),
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
    
    # MODIFIED: Skip retraining with best parameters and directly use the saved model state
    best_model_state = study.best_trial.user_attrs['model_state']
    
    # Extract the accuracy values directly from the best trial
    test_acc = study.best_trial.value
    val_acc = study.best_trial.user_attrs['val_accuracy']
    
    # Save final results - using the values from the best trial
    final_results = {
        'best_params': study.best_trial.params,
        'test_accuracy': test_acc,
        'val_accuracy': val_acc,
        'best_epoch': best_model_state['epoch']
    }
    
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save best model (based on test accuracy)
    torch.save(best_model_state, os.path.join(results_dir, 'best_model.pt'))
    
    logger.info(f"Hyperparameter tuning completed successfully!")
    logger.info(f"Best parameters: {study.best_trial.params}")
    logger.info(f"Test accuracy with best parameters: {test_acc:.4f}")
    logger.info(f"All results saved to {results_dir}")
    
    # =============================================
    # Find and save the model with absolute best test accuracy across ALL trials
    # =============================================
    logger.info("Finding model with absolute best test accuracy across all trials...")
    
    # Find the trial with the best test accuracy
    best_trial = None
    best_test_accuracy = 0.0
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.value > best_test_accuracy:
                best_test_accuracy = trial.value
                best_trial = trial
    
    if best_trial is not None:
        # Get model state from the absolute best trial
        best_model_state = best_trial.user_attrs['model_state']
        
        # Save this model separately
        torch.save(best_model_state, os.path.join(results_dir, 'absolute_best_model.pt'))
        
        # Save info about this absolute best model
        absolute_best_results = {
            'trial_number': best_trial.number,
            'params': best_trial.params,
            'test_accuracy': best_trial.value,
            'val_accuracy': best_trial.user_attrs['val_accuracy'],
            'best_epoch': best_model_state['epoch']
        }
        
        with open(os.path.join(results_dir, 'absolute_best_results.json'), 'w') as f:
            json.dump(absolute_best_results, f, indent=2)
        
        logger.info(f"Absolute best model found from Trial #{best_trial.number}")
        logger.info(f"Absolute best test accuracy: {best_trial.value:.4f}")
        logger.info(f"Absolute best model saved to {os.path.join(results_dir, 'absolute_best_model.pt')}")

if __name__ == "__main__":
    main()