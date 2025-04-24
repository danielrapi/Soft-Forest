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
from data_handling import load_processed_classification_public_data


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


def evaluate_model(params, dataset_name, device='cpu', results_dir=None, subset_selection=False):
    """Train and evaluate a SoftTreeEnsemble with given hyperparameters."""
    # Extract hyperparameters from params
    max_depth = int(params['max_depth'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    subset_share = params.get('subset_share', None)
    
    # Log current trial parameters
    logging.info(f"Trial params: max_depth={max_depth}, lr={learning_rate:.6f}, "
                 f"batch_size={batch_size}, epochs={epochs}, subset_selection={subset_selection}")
    
    try:
        # Load data
        data = load_processed_classification_public_data(name=dataset_name)
        
        # Convert to PyTorch tensors
        train_X_tensor = torch.tensor(data.x_train_processed, dtype=torch.float32)
        train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
        test_X_tensor = torch.tensor(data.x_test_processed, dtype=torch.float32)
        test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32)
        
        # Create datasets
        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
        
        # Get dimensions
        input_dims = train_X_tensor.shape[1]
        num_classes = data.num_classes
        
        # Create args object to pass to run_single_tree_experiment
        class Args:
            pass
        
        args = Args()
        args.dataset_name = dataset_name
        args.batch_size = batch_size
        args.max_depth = max_depth
        args.lr = learning_rate
        args.epochs = epochs
        args.device = device
        args.combine_output = True
        args.subset_selection = subset_selection
        args.subset_share = subset_share
        args.num_trees = 1
        
        # Run the experiment
        results = run_single_tree_experiment(train_dataset, test_dataset, input_dims, num_classes, args)
        
        # Return results in format expected by hyperopt
        return {
            'loss': results['loss'],
            'status': STATUS_OK,
            'accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'test_auc': results['test_auc'],
            'baseline_accuracy': 0.0,  # Placeholder, not used for single tree
            'execution_time': results['execution_time']
        }
        
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            'loss': 1.0,
            'status': STATUS_OK,
            'accuracy': 0.0,
            'test_loss': 999.0,
            'error': str(e)
        }


def optimize_hyperparams(dataset_name, 
                         max_evals=30, 
                         device='cpu', 
                         base_results_dir="outputs/hyperopt_single", 
                         subset_selection=False, 
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
    
    # Define hyperparameter search space
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'epochs': hp.quniform('epochs', 5, 30, 5),
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
        'results_dir': results_dir
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
        learning_rate = 0.05,
        epochs = 10
    )
    
    logging.info(f"Optimization complete. Results saved to {result['results_dir']}")
