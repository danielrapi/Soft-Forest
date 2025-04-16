"""
Hyperparameter optimization for SoftTreeEnsemble using hyperopt library.
"""

import sys
import os
import logging
import numpy as np
import torch
import time
from functools import partial
import matplotlib.pyplot as plt
import json

# Hyperopt imports
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Import your model and training functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from softensemble import SoftTreeEnsemble
from engine import train_model
from Data_Prep.load_data import load_data
from Data_Prep.view_contents import grab_data_info


def create_results_dir(dataset_name, base_dir="hyperopt_results"):
    """
    Create a timestamped results directory for the current run.
    
    Returns:
        str: Path to the created directory
    """
    import datetime
    
    #set working directory to current file
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific directory
    run_dir = os.path.join(base_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "trees"), exist_ok=True)
    
    logging.info(f"Created results directory: {run_dir}")
    return run_dir


def evaluate_model(params, dataset_name, input_dims, num_classes, device='cpu', results_dir=None):
    """
    Train and evaluate a SoftTreeEnsemble with given hyperparameters.
    Returns a loss value to be minimized by hyperopt.
    """
    # Extract hyperparameters from params
    max_depth = int(params['max_depth'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    l2_reg = params.get('l2_reg', 0.0)  # Default to 0 if not provided
    
    # Log current trial parameters
    logging.info(f"Trial params: max_depth={max_depth}, lr={learning_rate:.6f}, "
                 f"batch_size={batch_size}, epochs={epochs}, l2_reg={l2_reg:.6f}")
    
    print(f"Softtrees with following params: ")
    print(f"max_depth: {max_depth}")
    print(f"learning_rate: {learning_rate}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"l2_reg: {l2_reg}")
    
    # Load data with the specified batch size
    try:
        train_dataloader, test_dataloader = load_data(
            dataset_name=dataset_name,
            framework='torch',
            file_path='Data_Prep/datasets.h5',
            batch_size=batch_size
        )
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        # Return a poor score with all required fields to avoid KeyError
        return {
            'loss': 1.0, 
            'status': STATUS_OK, 
            'accuracy': 0, 
            'test_loss': 999.0,  # Add this to avoid KeyError
            'error': str(e)
        }
    
    # Initialize model
    model = SoftTreeEnsemble(
        num_trees=1,
        max_depth=max_depth,
        leaf_dims=num_classes,
        input_dim=input_dims,
        combine_output=True,
        subset_selection=False
    )
    
    # Apply L2 regularization through weight decay in optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    # Train model
    start_time = time.time()
    try:
        test_loss, accuracy = train_model(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            optimizer=optimizer
        )
        train_time = time.time() - start_time
    except Exception as e:
        logging.error(f"Error during training: {e}")
        return {
            'loss': 1.0, 
            'status': STATUS_OK, 
            'accuracy': 0, 
            'test_loss': 999.0,  # Add this to avoid KeyError
            'error': str(e)
        }
    
    # Return result (negative accuracy for minimization)
    return {
        'loss': -accuracy,  # We want to maximize accuracy, so minimize negative accuracy
        'accuracy': accuracy,
        'test_loss': test_loss,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'l2_reg': l2_reg,
        'train_time': train_time,
        'status': STATUS_OK
    }


def optimize_hyperparams(dataset_name, max_evals=30, device='cpu', base_results_dir="hyperopt_results"):
    """
    Run hyperopt to find optimal hyperparameters for SoftTreeEnsemble.
    
    Args:
        dataset_name: Name of the dataset to use
        max_evals: Maximum number of evaluations for hyperopt
        device: Device to use (cpu/cuda)
        base_results_dir: Base directory for results
        
    Returns:
        Dictionary with the best hyperparameters
    """
    # Create timestamped results directory
    results_dir = create_results_dir(dataset_name, base_results_dir)
    
    # Setup logging to the results directory
    log_file = os.path.join(results_dir, f"hyperopt.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)
    
    # Get dataset info
    input_dims, num_classes = grab_data_info('Data_Prep/datasets.h5', dataset_name)
    
    logging.info(f"Dataset: {dataset_name}, Features: {input_dims}, Classes: {num_classes}")
    logging.info(f"Results will be saved to: {results_dir}")
    
    # Define hyperparameter search space based on the paper specifications
    space = {
        # Learning rate: Uniform over {10^-1, 10^-2, ..., 10^-5}
        'learning_rate': hp.choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
        
        # Batch size: Uniform over {32, 64, 128, 256, 512}
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
        
        # Number of Epochs: Discrete uniform over [5, 100]
        'epochs': hp.quniform('epochs', 5, 100, 1),
        
        # Tree Depth: Discrete uniform over [2, 8]
        'max_depth': hp.quniform('max_depth', 2, 8, 1),
        
        # L2 Regularization: Mixture model of 0 and log uniform over [10^-8, 10^2]
        'l2_reg': hp.choice('l2_reg_choice', [
            0.0,  # 50% chance of no regularization
            hp.loguniform('l2_reg_value', np.log(1e-8), np.log(1e2))  # 50% chance of log uniform
        ])
    }
    
    # Create objective function with fixed parameters
    objective = partial(
        evaluate_model,
        dataset_name=dataset_name,
        input_dims=input_dims,
        num_classes=num_classes,
        device=device,
        results_dir=results_dir  # Pass the results directory
    )
    
    # Set up trials object to store results
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
    
    # Get best parameters in their original form (not indices)
    best_params = space_eval(space, best)
    print("best_params: ", best_params)
    # Find the trial with best accuracy
    trial_accuracies = [trial['result']['accuracy'] if 'accuracy' in trial['result'] else 0 
                        for trial in trials.trials]
    best_trial_idx = np.argmax(trial_accuracies)
    best_trial = trials.trials[best_trial_idx]
    best_accuracy = best_trial['result']['accuracy']
    best_loss = best_trial['result']['test_loss']
    
    # Log results
    logging.info("=" * 50)
    logging.info("Hyperparameter optimization complete")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info("=" * 50)
    
    # Convert all_trials to a simpler format for JSON serialization
    result = {
        'params': best_params,
        'accuracy': best_accuracy,
        'loss': best_loss,
        'all_trials': trials.trials,
        'results_dir': results_dir
    }

    # Save results
    result_file = os.path.join(results_dir, "results.json")
    with open(result_file, 'w') as f:
        
        result_copy = result.copy()
        result_copy['all_trials'] = str(len(result['all_trials'])) + " trials (removed for serialization)"
        json.dump(result_copy, f, indent=2)
    
    # Plot results
    plot_dir = os.path.join(results_dir, "plots")
    plot_optimization_results(result['all_trials'], output_dir=plot_dir)
    
    # Save the best model
    try:
        best_model = create_best_model(best_params, input_dims, num_classes)
        torch.save(best_model.state_dict(), os.path.join(results_dir, "models", "best_model.pt"))
        
        # Also save a visualization of the best model
        if best_model.num_trees == 1:  # Only plot for single tree models
            fig = best_model.plot_tree()
            fig.savefig(os.path.join(results_dir, "trees", "best_model_tree.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logging.error(f"Error saving best model: {e}")
    
    logging.info(f"All results saved to {results_dir}")
    
    return result


def plot_optimization_results(trials, output_dir='plots'):
    """
    Plot the results of hyperparameter optimization.
    
    Args:
        trials: Trials object from hyperopt
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results into a DataFrame for easier manipulation
    results = []
    for trial in trials:
        if 'result' in trial and isinstance(trial['result'], dict):
            result = trial['result'].copy()
            # Add parameter values
            if 'misc' in trial and 'vals' in trial['misc']:
                for param, values in trial['misc']['vals'].items():
                    if values:  # Some might be empty
                        result[param] = values[0]
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Convert negative loss back to accuracy
    if 'loss' in df.columns:
        df['computed_accuracy'] = -df['loss']
    
    # Accuracy over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(df) + 1), df['accuracy'], 'o-', color='blue')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Trial Number')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_per_trial.png'))
    plt.close()
    
    # Hyperparameter importance plots
    important_params = ['max_depth', 'learning_rate', 'batch_size', 'epochs']
    
    for param in important_params:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[param], df['accuracy'], alpha=0.7)
            plt.xlabel(param)
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy vs {param}')
            plt.grid(True)
            
            # Special handling for parameters
            if param == 'learning_rate':
                plt.xscale('log')
            elif param == 'batch_size':
                plt.xscale('log', base=2)
            
            plt.savefig(os.path.join(output_dir, f'accuracy_vs_{param}.png'))
            plt.close()
    
    # For L2 regularization, which is a mixture model
    if 'l2_reg' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Split into zero and non-zero l2_reg
        zero_mask = df['l2_reg'] == 0
        nonzero_mask = ~zero_mask
        
        # Plot zero L2 values
        plt.scatter(df.loc[zero_mask].index, df.loc[zero_mask, 'accuracy'], 
                   color='blue', label='No Regularization', alpha=0.7)
        
        # Plot non-zero L2 values with a different color on a second y-axis
        if nonzero_mask.any():
            plt.scatter(df.loc[nonzero_mask].index, df.loc[nonzero_mask, 'accuracy'],
                       color='red', label='With Regularization', alpha=0.7)
        
        plt.xlabel('Trial Number')
        plt.ylabel('Accuracy')
        plt.title('Impact of L2 Regularization on Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'l2_regularization_impact.png'))
        plt.close()
    
    # Create a summary of best parameters
    best_row = df.loc[df['accuracy'].idxmax()]
    
    # Plot best parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    param_names = ['max_depth', 'learning_rate', 'batch_size', 'epochs', 'l2_reg']
    param_values = [best_row.get(p, 'N/A') for p in param_names]
    
    ax.barh(param_names, [1]*len(param_names), alpha=0.3)
    for i, (name, value) in enumerate(zip(param_names, param_values)):
        ax.text(0.5, i, f"{value}", va='center', ha='center', fontweight='bold')
    
    ax.set_title('Best Hyperparameters')
    ax.set_xlabel('Value')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_parameters.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


def create_best_model(params, input_dims, num_classes):
    """Create a model with the best parameters"""
    model = SoftTreeEnsemble(
        num_trees=1,
        max_depth=int(params['max_depth']),
        leaf_dims=num_classes,
        input_dim=input_dims,
        combine_output=True,
        subset_selection=False
    )
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for SoftTreeEnsemble")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--max_evals", type=int, default=30, help="Maximum evaluations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="hyperopt_results", help="Base output directory for results")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Run optimization (results dir is created inside the function)
    logging.info(f"Starting hyperparameter search for dataset: {args.dataset}")
    result = optimize_hyperparams(
        dataset_name=args.dataset,
        max_evals=args.max_evals,
        device=args.device,
        base_results_dir=args.output_dir
    )
    
    logging.info(f"Optimization complete. Results saved to {result['results_dir']}")
