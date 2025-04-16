"""
Hyperparameter optimization for multi-tree SoftTreeEnsemble using hyperopt library.
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
import pandas as pd

# Hyperopt imports
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Import your model and training functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from softensemble import SoftTreeEnsemble
from engine import train_model
from Data_Prep.load_data import load_data
from Data_Prep.view_contents import grab_data_info


def create_results_dir(dataset_name, base_dir="hyperopt_multi_results"):
    """
    Create a timestamped results directory for the current run.
    
    Returns:
        str: Path to the created directory
    """
    import datetime
    
    # Set working directory to current file
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
    Train and evaluate multiple SoftTreeEnsemble models and combine with majority voting.
    Returns a loss value to be minimized by hyperopt.
    """
    # Extract hyperparameters from params
    max_depth = int(params['max_depth'])
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    l2_reg = params.get('l2_reg', 0.0)  # Default to 0 if not provided
    num_trees = int(params['num_trees'])
    
    # Log current trial parameters
    logging.info(f"Trial params: num_trees={num_trees}, max_depth={max_depth}, lr={learning_rate:.6f}, "
                 f"batch_size={batch_size}, epochs={epochs}, l2_reg={l2_reg:.6f}")
    
    print(f"Multi-tree ensemble with parameters: ")
    print(f"num_trees: {num_trees}")
    print(f"max_depth: {max_depth}")
    print(f"learning_rate: {learning_rate}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"l2_reg: {l2_reg}")
    
    # Load data with the specified batch size
    try:
        test_dataloader = None  # This will be set in the loop
        
        # Store all tree predictions
        all_preds = []
        all_models = []
        
        start_time = time.time()
        
        # Train each tree
        for tree_idx in range(num_trees):
            logging.info(f"Training tree {tree_idx+1}/{num_trees}")
            
            # Load data with bootstrap if specified
            train_dataloader, test_dataloader = load_data(
                dataset_name=dataset_name,
                framework='torch',
                file_path='Data_Prep/datasets.h5',
                batch_size=batch_size,
                bootstrap=True  # Use bootstrap sampling for each tree
            )
            
            # Initialize model for this tree
            model = SoftTreeEnsemble(
                num_trees=1,
                max_depth=max_depth,
                leaf_dims=num_classes,
                input_dim=input_dims,
                combine_output=True,
                subset_selection=True  # Enable feature subset selection for random forest behavior
            )
            
            # Apply L2 regularization through weight decay in optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
            
            # Train model
            try:
                test_loss, test_accuracy = train_model(
                    model=model,
                    train_loader=train_dataloader,
                    test_loader=test_dataloader,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    device=device,
                    optimizer=optimizer
                )
                
                logging.info(f"Tree {tree_idx+1} trained. Test accuracy: {test_accuracy:.4f}")
                
                # Save the tree visualization
                if results_dir is not None:
                    try:
                        # Generate a unique filename for this tree
                        tree_filename = f"tree_{tree_idx+1}_depth{max_depth}_acc{test_accuracy:.4f}.png"
                        tree_path = os.path.join(results_dir, "trees", tree_filename)
                        
                        # Plot the tree and save it
                        fig = model.plot_tree()
                        fig.savefig(tree_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        logging.info(f"Tree visualization saved to {tree_path}")
                    except Exception as e:
                        logging.error(f"Error saving tree visualization: {e}")
                
                # Get predictions for this tree
                model.eval()
                tree_preds = []
                
                with torch.no_grad():
                    for X_batch, _ in test_dataloader:
                        X_batch = X_batch.to(device)
                        outputs = model(X_batch)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        tree_preds.append(preds.cpu().numpy())
                
                # Convert predictions to numpy arrays
                tree_preds = np.concatenate(tree_preds, axis=0)
                
                # Store predictions and model
                all_preds.append(tree_preds)
                all_models.append(model)
                
            except Exception as e:
                logging.error(f"Error training tree {tree_idx+1}: {e}")
                continue
        
        # If no trees were successfully trained, return error
        if not all_preds:
            return {
                'loss': 1.0, 
                'status': STATUS_OK, 
                'accuracy': 0, 
                'test_loss': 999.0,
                'error': "Failed to train any trees"
            }
        
        # Get true labels from test set
        true_labels = []
        with torch.no_grad():
            for _, y_batch in test_dataloader:
                if len(y_batch.shape) > 1:
                    y_batch = y_batch.squeeze()
                true_labels.append(y_batch.numpy())
        
        true_labels = np.concatenate(true_labels, axis=0)
        
        # Stack predictions from all trees
        all_preds = np.stack(all_preds, axis=0)  # shape: (num_trees, num_samples)
        
        # Majority voting: For each sample, count the most frequent class prediction
        from scipy.stats import mode
        ensemble_preds = mode(all_preds, axis=0, keepdims=False)[0]
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        ensemble_accuracy = accuracy_score(true_labels, ensemble_preds)
        
        train_time = time.time() - start_time
        logging.info(f"Ensemble training complete. Accuracy: {ensemble_accuracy:.4f}, Time: {train_time:.2f}s")
        
        # Save ensemble visualization showing accuracy contribution of each tree
        if results_dir is not None:
            try:
                # Plot individual tree accuracies vs ensemble accuracy
                plt.figure(figsize=(10, 6))
                
                # Calculate individual accuracies
                individual_accuracies = []
                for i, preds in enumerate(all_preds):
                    acc = accuracy_score(true_labels, preds)
                    individual_accuracies.append(acc)
                
                # Plot individual accuracies
                plt.bar(range(1, num_trees+1), individual_accuracies, alpha=0.6, color='skyblue', label='Individual Trees')
                
                # Plot ensemble accuracy
                plt.axhline(y=ensemble_accuracy, color='red', linestyle='-', label=f'Ensemble ({ensemble_accuracy:.4f})')
                
                plt.xlabel('Tree Number')
                plt.ylabel('Accuracy')
                plt.title(f'Individual Tree vs Ensemble Accuracy (n={num_trees})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                ensemble_viz_path = os.path.join(results_dir, "plots", f"ensemble_accuracy_n{num_trees}.png")
                plt.savefig(ensemble_viz_path, dpi=150)
                plt.close()
                
                logging.info(f"Ensemble visualization saved to {ensemble_viz_path}")
            except Exception as e:
                logging.error(f"Error saving ensemble visualization: {e}")
        
        # Save all models in the ensemble
        if results_dir is not None:
            try:
                ensemble_model_path = os.path.join(results_dir, "models", f"ensemble_n{num_trees}.pt")
                torch.save({f'tree_{i}': model.state_dict() for i, model in enumerate(all_models)}, 
                          ensemble_model_path)
                logging.info(f"Ensemble models saved to {ensemble_model_path}")
            except Exception as e:
                logging.error(f"Error saving ensemble models: {e}")
                
        return {
            'loss': -ensemble_accuracy,  # Negative because we want to maximize accuracy
            'accuracy': ensemble_accuracy,
            'num_trees': num_trees,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'l2_reg': l2_reg,
            'train_time': train_time,
            'status': STATUS_OK
        }
        
    except Exception as e:
        logging.error(f"Error during ensemble evaluation: {e}")
        return {
            'loss': 1.0, 
            'status': STATUS_OK, 
            'accuracy': 0, 
            'test_loss': 999.0,
            'error': str(e)
        }


def optimize_hyperparams(dataset_name, max_evals=30, device='cpu', base_results_dir="hyperopt_multi_results"):
    """
    Run hyperopt to find optimal hyperparameters for multi-tree SoftTreeEnsemble.
    
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
        # Number of Trees: Discrete uniform over [1, 100]
        'num_trees': hp.quniform('num_trees', 5, 50, 5),  # Reduce range for faster evaluation
        
        # Learning rate: Uniform over {10^-1, 10^-2, 10^-3}
        'learning_rate': hp.choice('learning_rate', [1e-1, 1e-2, 1e-3]),
        
        # Batch size: Uniform over {32, 64, 128, 256, 512}
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        
        # Number of Epochs: Discrete uniform over [5, 100]
        'epochs': hp.quniform('epochs', 5, 30, 5),  # Reduce range for faster evaluation
        
        # Tree Depth: Discrete uniform over [2, 8]
        'max_depth': hp.quniform('max_depth', 2, 6, 1),  # Reduce max depth for faster training
        
        # L2 Regularization: Mixture model of 0 and log uniform over [10^-8, 10^2]
        'l2_reg': hp.choice('l2_reg_choice', [
            0.0,  # 50% chance of no regularization
            hp.loguniform('l2_reg_value', np.log(1e-6), np.log(1e-2))  # 50% chance of log uniform
        ])
    }
    
    # Create objective function with fixed parameters
    objective = partial(
        evaluate_model,
        dataset_name=dataset_name,
        input_dims=input_dims,
        num_classes=num_classes,
        device=device,
        results_dir=results_dir
    )
    
    # Set up trials object to store results
    trials = Trials()
    
    # Run hyperopt to find best parameters
    logging.info("Starting hyperparameter optimization...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1
    )
    
    # Get best parameters in their original form (not indices)
    best_params = space_eval(space, best)
    print("best_params: ", best_params)
    
    # Find the trial with best accuracy
    trial_accuracies = [trial['result']['accuracy'] if 'accuracy' in trial['result'] else 0 
                       for trial in trials.trials]
    best_accuracy = max(trial_accuracies) if trial_accuracies else 0
    best_trial = trials.trials[trial_accuracies.index(best_accuracy)]
    best_loss = best_trial['result']['loss']
    
    # Log the best parameters
    logging.info("=" * 50)
    logging.info("Best hyperparameters found:")
    for param, value in best_params.items():
        logging.info(f"{param}: {value}")
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
    try:
        plot_optimization_results(trials.trials, output_dir=os.path.join(results_dir, "plots"))
    except Exception as e:
        logging.error(f"Error plotting optimization results: {e}")
    
    # Train and save the best model
    try:
        # Create and save best model
        logging.info("Training best model with optimal parameters...")
        
        # Re-evaluate with best parameters to get the ensemble
        objective(best_params)
        
        logging.info("Best model evaluation complete")
    except Exception as e:
        logging.error(f"Error training best model: {e}")
    
    logging.info(f"All results saved to {results_dir}")
    
    return result


def plot_optimization_results(trials, output_dir="plots"):
    """
    Create visualizations of hyperparameter optimization results.
    """
    import pandas as pd
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert trials to DataFrame for easier plotting
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
    
    if not results:
        logging.error("No results to plot")
        return
    
    df = pd.DataFrame(results)
    
    # Convert negative loss back to accuracy
    if 'loss' in df.columns:
        df['computed_accuracy'] = -df['loss']
    
    # Accuracy over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(df) + 1), df['accuracy'], 'o-', color='blue')
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Accuracy vs Trial Number')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_per_trial.png'))
    plt.close()
    
    # Hyperparameter importance plots
    important_params = ['num_trees', 'max_depth', 'learning_rate', 'batch_size', 'epochs']
    
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
        
        # Plot non-zero L2 values with a different color
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
    
    # Effect of num_trees on accuracy
    if 'num_trees' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Group by number of trees and get average accuracy
        tree_counts = df.groupby('num_trees')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
        
        plt.errorbar(tree_counts['num_trees'], tree_counts['mean'], 
                     yerr=tree_counts['std'], fmt='o-', capsize=5, 
                     color='blue', label='Mean ± Std Dev')
        
        # Add count as size of points
        for i, row in tree_counts.iterrows():
            plt.annotate(f"n={int(row['count'])}", 
                       (row['num_trees'], row['mean']),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center')
        
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')
        plt.title('Effect of Ensemble Size on Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'num_trees_vs_accuracy.png'))
        plt.close()
    
    # Create a summary of best parameters
    best_row = df.loc[df['accuracy'].idxmax()]
    
    # Plot best parameters
    fig, ax = plt.subplots(figsize=(10, 6))
    param_names = ['num_trees', 'max_depth', 'learning_rate', 'batch_size', 'epochs', 'l2_reg']
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for multi-tree SoftTreeEnsemble")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--max_evals", type=int, default=20, help="Maximum evaluations")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="hyperopt_multi_results", help="Base output directory for results")
    
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