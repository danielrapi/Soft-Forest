"""
Pipeline for running hyperparameter optimization and model evaluation across multiple datasets.
"""

import os
import logging
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
import h5py
import sys

# Import your model and training functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parameter_tuning import optimize_single_tree, optimize_multi_tree
from runners.ensemble import run_ensemble_experiment
from data_handling import load_processed_classification_public_data

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def create_results_dir(base_dir="outputs/pipeline", dataset_name=None, bagging=False, subset_selection=False):
    """Create a timestamped results directory for the current run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create configuration string
    config = []
    if bagging:
        config.append("bagging")
    if subset_selection:
        config.append("subset")
    if not config:
        config.append("base")
    config_str = "_".join(config)
    
    # Create directory structure: outputs/pipeline/dataset_name/config/run_timestamp
    run_dir = os.path.join(base_dir, dataset_name, config_str, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_predictions(predictions, labels, dataset_name, model_type, results_dir):
    """Save model predictions and labels to HDF5 file."""
    pred_file = os.path.join(results_dir, f"{dataset_name}_{model_type}_predictions.h5")
    with h5py.File(pred_file, 'w') as f:
        f.create_dataset('predictions', data=predictions)
        f.create_dataset('labels', data=labels)

def run_dataset_pipeline(dataset_name, noise_level=0.15, device='cpu', results_dir=None, bagging=False, subset_selection=False):
    """Run the complete pipeline for a single dataset with specified configuration."""
    logging.info(f"\n{'='*50}")
    logging.info(f"Starting pipeline for dataset: {dataset_name}")
    logging.info(f"Configuration: bagging={bagging}, subset_selection={subset_selection}")
    logging.info(f"{'='*50}\n")
    
    # Step 1: Hyperopt single tree with train/validation
    logging.info("Step 1: Running single tree hyperparameter optimization...")
    single_tree_results = optimize_single_tree(
        dataset_name=dataset_name,
        max_evals=1,
        device=device,
        noise_level=noise_level,
        base_results_dir=os.path.join(results_dir, "single_tree")
    )
    print(single_tree_results)
    # Extract best parameters from single tree optimization
    best_single_params = single_tree_results['params']
    logging.info(f"Best single tree parameters: {best_single_params}")
    
    # Step 2: Hyperopt multiple trees with train/test
    logging.info("\nStep 2: Running multi-tree hyperparameter optimization...")
    multi_tree_results = optimize_multi_tree(
        dataset_name=dataset_name,
        max_evals=1,
        device=device,
        noise_level=noise_level,
        base_results_dir=os.path.join(results_dir, "multi_tree"),
        # Use best parameters from single tree optimization
        learning_rate=best_single_params['learning_rate'],
        epochs=best_single_params['epochs'],
        batch_size=best_single_params['batch_size'],
        max_depth=best_single_params['max_depth'],
        # Only optimize num_trees and subset_share
        subset_selection=subset_selection,
        bootstrap=bagging,
        num_trees=None,
        subset_share=None
    )
    
    # Extract best parameters from multi-tree optimization
    best_multi_params = multi_tree_results['params']
    logging.info(f"Best multi-tree parameters: {best_multi_params}")
    
    # Step 3: Train final model with best parameters and save detailed results
    logging.info("\nStep 3: Training final model with best parameters...")
    
    # Load data
    data = load_processed_classification_public_data(name=dataset_name, noise_level=noise_level)
    
    # Convert to tensors
    train_X_tensor = torch.tensor(data.x_train_processed, dtype=torch.float32)
    train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
    test_X_tensor = torch.tensor(data.x_test_processed, dtype=torch.float32)
    test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_X_tensor, test_y_tensor)
    
    # Create args object
    class Args:
        def __init__(self):
            self.num_trees = int(best_multi_params['num_trees'])
            self.max_depth = int(best_multi_params['max_depth'])
            self.lr = best_multi_params['learning_rate']
            self.batch_size = int(best_multi_params['batch_size'])
            self.epochs = int(best_multi_params['epochs'])
            self.device = device
            self.combine_output = True
            self.subset_selection = subset_selection
            self.subset_share = best_multi_params.get('subset_share', 0.5)
            self.bootstrap = bagging
            self.dataset_name = dataset_name
    
    args = Args()
    
    # Run final experiment
    final_results = run_ensemble_experiment(
        train_dataset,
        test_dataset,
        train_X_tensor.shape[1],
        data.num_classes,
        args
    )
    
    # Convert NumPy arrays to lists for JSON serialization
    final_results_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in final_results.items()
    }
    
    # Save all results
    results = {
        'dataset': dataset_name,
        'noise_level': noise_level,
        'bagging': bagging,
        'subset_selection': subset_selection,
        'single_tree_params': best_single_params,
        'multi_tree_params': best_multi_params,
        'final_results': {
            'loss': final_results['loss'],
            'accuracy': final_results['accuracy'],
            'auc': final_results['auc'],
            'baseline': final_results['baseline'],
            'execution_time': final_results['execution_time']
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    print(final_results)
    
    # Save ensemble predictions and probabilities
    ensemble_file = os.path.join(results_dir, f"{dataset_name}_ensemble_results.h5")
    with h5py.File(ensemble_file, 'w') as f:
        f.create_dataset('ensemble_predictions', data=final_results['ensemble_predictions'])
        f.create_dataset('ensemble_probabilities', data=final_results['ensemble_probabilities'])
        f.create_dataset('labels', data=test_y_tensor.numpy())
    
    # Save individual tree predictions and probabilities
    trees_file = os.path.join(results_dir, f"{dataset_name}_trees_results.h5")
    with h5py.File(trees_file, 'w') as f:
        # Save predictions for each tree
        for i, tree_preds in enumerate(final_results['tree_predictions']):
            f.create_dataset(f'tree_{i}_predictions', data=tree_preds)
        
        # Save probabilities for each tree
        for i, tree_probs in enumerate(final_results['tree_probabilities']):
            f.create_dataset(f'tree_{i}_probabilities', data=tree_probs)
        
        f.create_dataset('labels', data=test_y_tensor.numpy())
    
    logging.info(f"\nPipeline completed for {dataset_name}")
    logging.info(f"Results saved to {results_file}")
    logging.info(f"Ensemble results saved to {ensemble_file}")
    logging.info(f"Individual tree results saved to {trees_file}")
    
    return results

def main():
    # Get list of datasets
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data_handling', 'storage')
    datasets = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    datasets = ['breast_cancer_wisconsin']
    # Define configurations
    configurations = [
        (False, False),  # Base
        (True, False),   # Bagging only
        (False, True),   # Subset selection only
        (True, True)     # Both
    ]
    
    # Run pipeline for each dataset and configuration
    for dataset in tqdm(datasets, desc="Processing datasets"):
        for bagging, subset_selection in configurations:
            try:
                # Create results directory for this configuration
                results_dir = create_results_dir(
                    dataset_name=dataset,
                    bagging=bagging,
                    subset_selection=subset_selection
                )
                
                # Setup logging for this configuration
                log_file = os.path.join(results_dir, "pipeline.log")
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ]
                )
                
                # Run pipeline with current configuration
                run_dataset_pipeline(
                    dataset,
                    noise_level=0.0,
                    device='cpu',
                    results_dir=results_dir,
                    bagging=bagging,
                    subset_selection=subset_selection
                )
            except Exception as e:
                logging.error(f"Error processing dataset {dataset} with bagging={bagging}, subset_selection={subset_selection}: {str(e)}")
                continue

if __name__ == "__main__":
    main() 