"""
Hyperparameter optimization for SoftTreeEnsemble using hyperopt library.
"""

import sys
import os
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
from functools import partial
import json
from datetime import datetime

# Hyperopt imports
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runners.single_tree import run_single_tree_experiment
from data_handling.utils.load_data import load_processed_classification_public_data

# Global variable to store the dataset with shuffled labels
global_train_dataset = None
global_test_dataset = None
global_input_dims = None
global_num_classes = None

def create_results_dir(dataset_name, base_dir="outputs/hyperopt_single"):
    """Create a timestamped results directory for the current run."""
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory
    run_dir = os.path.join(base_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    logging.info(f"Created results directory: {run_dir}")
    return run_dir

def load_and_prepare_data(dataset_name, noise_level):
    """Load and prepare data with shuffled labels once at the beginning."""
    global global_train_dataset, global_test_dataset, global_input_dims, global_num_classes
    
    if global_train_dataset is None:
        # Set fixed seed for reproducibility
        np.random.seed(42)  # Fixed seed for consistent label shuffling
        
        # Load data with specified noise level
        data = load_processed_classification_public_data(name=dataset_name, noise_level=noise_level, seed=42)
        
        # Convert to tensors
        train_X_tensor = torch.tensor(data.x_train_processed, dtype=torch.float32)
        train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
        test_X_tensor = torch.tensor(data.x_test_processed, dtype=torch.float32)
        test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32)
        
        # Create datasets
        global_train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        global_test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
        
        # Store dimensions
        global_input_dims = train_X_tensor.shape[1]
        global_num_classes = data.num_classes
        
        # Calculate and log class distributions
        train_labels = train_y_tensor.numpy()
        test_labels = test_y_tensor.numpy()
        
        # Training set distribution
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        train_percentages = (train_counts / len(train_labels)) * 100
        
        # Test set distribution
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        test_percentages = (test_counts / len(test_labels)) * 100
        
        # Log distributions
        logging.info("Class Distribution in Training Set:")
        for label, count, percentage in zip(train_unique, train_counts, train_percentages):
            logging.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
        
        logging.info("\nClass Distribution in Test Set:")
        for label, count, percentage in zip(test_unique, test_counts, test_percentages):
            logging.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
        
        # Log majority class information
        train_majority_idx = np.argmax(train_counts)
        test_majority_idx = np.argmax(test_counts)
        
        logging.info(f"\nMajority Class in Training Set: Class {train_unique[train_majority_idx]} ({train_percentages[train_majority_idx]:.2f}%)")
        logging.info(f"Majority Class in Test Set: Class {test_unique[test_majority_idx]} ({test_percentages[test_majority_idx]:.2f}%)")
        
        logging.info(f"\nLoaded dataset with {noise_level*100}% label noise")
        logging.info(f"Training samples: {len(global_train_dataset)}")
        logging.info(f"Test samples: {len(global_test_dataset)}")
    
    return global_train_dataset, global_test_dataset, global_input_dims, global_num_classes

def evaluate_model(params, dataset_name, device='cpu', results_dir=None, subset_selection=False):
    """Train and evaluate a SoftTreeEnsemble with given hyperparameters."""
    # Extract hyperparameters from params
    max_depth = int(params['max_depth'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    subset_share = params.get('subset_share', None)
    
    # Log current trial parameters
    logging.info(f"Trial parameters: depth={max_depth}, lr={learning_rate}, "
                f"batch={batch_size}, epochs={epochs}")
    
    # Create Args object
    class Args:
        def __init__(self):
            self.num_trees = 1
            self.max_depth = max_depth
            self.combine_output = True
            self.subset_selection = subset_selection
            self.subset_share = subset_share
            self.epochs = epochs
            self.lr = learning_rate
            self.batch_size = batch_size
            self.device = device
            self.bootstrap = False
            self.dataset_name = dataset_name
    
    args = Args()
    
    try:
        # Run experiment
        results = run_single_tree_experiment(
            global_train_dataset,
            global_test_dataset,
            global_input_dims,
            global_num_classes,
            args
        )
        
        # Return results
        return {
            'status': STATUS_OK,
            'loss': -results['test_accuracy'],  # Hyperopt minimizes, so we negate accuracy
            'accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'test_auc': results['test_auc']
        }
    
    except Exception as e:
        logging.error(f"Error in trial: {e}")
        return {
            'status': STATUS_OK,
            'loss': float('inf'),
            'accuracy': 0.0,
            'test_loss': float('inf'),
            'test_auc': 0.0
        }

def optimize_hyperparams(dataset_name, 
                         max_evals=30, 
                         device='cpu', 
                         base_results_dir="outputs/hyperopt_single", 
                         subset_selection=False,
                         noise_level=0.0,
                         learning_rate = None,
                         epochs = None,
                         batch_size = None,
                         max_depth = None,
                         ):
    """Run hyperopt to find optimal hyperparameters for SoftTreeEnsemble."""
    # Create results directory
    results_dir = create_results_dir(dataset_name, base_results_dir)
    
    # Setup logging
    log_file = os.path.join(results_dir, f"hyperopt.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # Load and prepare data once at the beginning
    load_and_prepare_data(dataset_name, noise_level)
    
    # Define hyperparameter search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.1)),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'epochs': hp.quniform('epochs', 5, 20, 5),
        'max_depth': hp.quniform('max_depth', 3, 8, 1),
    }
    
    if learning_rate is not None:
        space['learning_rate'] = learning_rate
    if epochs is not None:
        space['epochs'] = epochs
    if batch_size is not None:
        space['batch_size'] = batch_size
    if max_depth is not None:
        space['max_depth'] = max_depth

    # Add subset_share if subset_selection is enabled
    if subset_selection:
        space['subset_share'] = hp.uniform('subset_share', 0.1, 0.9)
    
    # Create objective function
    objective = partial(
        evaluate_model,
        dataset_name=dataset_name,
        device=device,
        results_dir=results_dir,
        subset_selection=subset_selection
    )
    
    # Set up trials object
    trials = Trials()
    
    # Run optimization
    logging.info(f"Starting hyperparameter optimization with {max_evals} trials")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Get best parameters
    best_params = space_eval(space, best)
    
    # Find the trial with best accuracy
    trial_accuracies = [trial['result']['accuracy'] if 'accuracy' in trial['result'] else 0 
                        for trial in trials.trials]
    best_trial_idx = np.argmax(trial_accuracies)
    best_trial = trials.trials[best_trial_idx]
    best_accuracy = best_trial['result']['accuracy']
    best_loss = best_trial['result']['test_loss']
    best_auc = best_trial['result']['test_auc']
    
    # Log results
    logging.info("=" * 50)
    logging.info("Hyperparameter optimization complete")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info(f"Best validation AUC: {best_auc:.4f}")
    logging.info("=" * 50)
    
    # Save results
    result = {
        'params': best_params,
        'accuracy': best_accuracy,
        'loss': best_loss,
        'auc': best_auc,
        'results_dir': results_dir,
        'noise_level': noise_level
    }
    
    result_file = os.path.join(results_dir, "results.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logging.info(f"All results saved to {results_dir}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SoftTreeEnsemble")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--max_evals", type=int, default=30, help="Maximum evaluations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="outputs/hyperopt_single", help="Base output directory")
    parser.add_argument("--subset_selection", action="store_true", help="Enable subset selection")
    parser.add_argument("--noise_level", type=float, default=0.0, help="Proportion of labels to shuffle (0-1)")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Run optimization
    logging.info(f"Starting hyperparameter search for dataset: {args.dataset}")
    result = optimize_hyperparams(
        dataset_name=args.dataset,
        max_evals=args.max_evals,
        device=args.device,
        base_results_dir=args.output_dir,
        subset_selection=args.subset_selection,
        noise_level=args.noise_level,
        learning_rate = 0.05,
        epochs = 10
    )
    
    logging.info(f"Optimization complete. Results saved to {result['results_dir']}")
