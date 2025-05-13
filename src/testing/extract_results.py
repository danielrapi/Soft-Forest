import os
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def load_tree_predictions(h5f):
    """Load and combine predictions from all trees in the HDF5 file."""
    tree_keys = sorted([k for k in h5f.keys() if k.endswith('_probabilities')])
    
    if not tree_keys:
        return None, None
    
    tree_probs = []
    for key in tree_keys:
        prob = h5f[key][:]
        tree_probs.append(prob)
    
    probabilities = np.stack(tree_probs)
    true_labels = h5f['labels'][:] if 'labels' in h5f else None
    
    return probabilities, true_labels

def load_results(base_path="outputs/pipeline"):
    """Load results from all datasets and configurations."""
    all_results = {}
    
    # Walk through the directory structure
    for dataset_dir in Path(base_path).glob("*"):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"\nProcessing dataset: {dataset_name}")
        
        dataset_results = {
            'base': [],
            'bagging': [],
            'subset': [],
            'bagging_subset': []
        }
        
        # Load results from each configuration
        for config_dir in dataset_dir.glob("*"):
            if not config_dir.is_dir():
                continue
                
            config_name = config_dir.name
            config_type = {
                'base': 'base',
                'bagging': 'bagging',
                'subset': 'subset',
                'bagging_subset': 'bagging_subset'
            }.get(config_name)
            
            if not config_type:
                continue
            
            # Find the most recent run directory
            run_dirs = list(config_dir.glob("run_*"))
            if not run_dirs:
                continue
            latest_run = max(run_dirs, key=lambda x: x.name)
            
            # Load results
            results_file = latest_run / f"{dataset_name}_results.json"
            trees_file = latest_run / f"{dataset_name}_trees_results.h5"
            
            if not results_file.exists():
                continue
                
            with open(results_file) as f:
                result = json.load(f)
                result['dataset'] = dataset_name
                
                if trees_file.exists():
                    try:
                        with h5py.File(trees_file, 'r') as h5f:
                            predictions, true_labels = load_tree_predictions(h5f)
                            if predictions is not None:
                                result['tree_probabilities'] = predictions
                            if true_labels is not None:
                                result['true_labels'] = true_labels
                    except Exception as e:
                        print(f"Error loading h5 file: {e}")
                
                dataset_results[config_type].append(result)
        
        all_results[dataset_name] = dataset_results
    
    return all_results

def create_ensemble_accuracy_table(results):
    """Create a table of ensemble accuracies per configuration and dataset."""
    data = []
    
    for dataset_name, dataset_results in results.items():
        row = {'Dataset': dataset_name}
        
        for config, config_results in dataset_results.items():
            if config_results:
                row[config] = config_results[0]['final_results']['accuracy']
            else:
                row[config] = None
                
        data.append(row)
    
    return pd.DataFrame(data)

def create_tree_accuracy_table(results):
    """Create a table of mean tree accuracies per configuration and dataset."""
    data = []
    
    for dataset_name, dataset_results in results.items():
        row = {'Dataset': dataset_name}
        
        for config, config_results in dataset_results.items():
            if config_results and 'tree_probabilities' in config_results[0] and 'true_labels' in config_results[0]:
                tree_probabilities = config_results[0]['tree_probabilities']
                true_labels = config_results[0]['true_labels']
                
                # Calculate accuracy for each tree
                tree_predictions = np.argmax(tree_probabilities, axis=-1)
                tree_accuracies = np.mean(tree_predictions == true_labels, axis=1)
                row[config] = np.mean(tree_accuracies)
            else:
                row[config] = None
                
        data.append(row)
    
    return pd.DataFrame(data)

def create_accuracy_uplift_table(ensemble_df, tree_df):
    """Create a table showing accuracy uplift between ensemble and mean tree accuracy."""
    # Create a combined table with both percentage and percentage points
    combined_df = ensemble_df.copy()
    combined_df.set_index('Dataset', inplace=True)
    tree_df_idx = tree_df.set_index('Dataset')
    
    # Create a new DataFrame for the formatted results
    formatted_df = pd.DataFrame()
    formatted_df['Dataset'] = ensemble_df['Dataset']
    
    # Calculate and format uplift for each configuration
    for col in combined_df.columns:
        # Calculate percentage uplift
        pct_uplift = ((combined_df[col] - tree_df_idx[col]) / tree_df_idx[col] * 100)
        
        # Calculate percentage points uplift
        pp_uplift = (combined_df[col] - tree_df_idx[col]) * 100
        
        # Format as "X% (Y.Zpp)"
        formatted_values = []
        for pct, pp in zip(pct_uplift, pp_uplift):
            if pd.isna(pct) or pd.isna(pp):
                formatted_values.append(None)
            else:
                formatted_values.append(f"{pct:.1f}% ({pp:.1f}pp)")
        
        formatted_df[col] = formatted_values
    
    return formatted_df

def create_distribution_metrics_table(results):
    """Create a table with distribution metrics for tree predictions."""
    data = []
    
    for dataset_name, dataset_results in results.items():
        for config, config_results in dataset_results.items():
            if not config_results or 'tree_probabilities' not in config_results[0]:
                continue
                
            probabilities = config_results[0]['tree_probabilities']
            num_classes = probabilities.shape[-1]
            
            for class_idx in range(num_classes):
                class_probs = probabilities[:, :, class_idx].flatten()
                
                # Calculate various distribution metrics
                metrics = {
                    'Dataset': dataset_name,
                    'Configuration': config,
                    'Class': class_idx,
                    'Mean': np.mean(class_probs),
                    'Std': np.std(class_probs),
                    'Skewness': stats.skew(class_probs),
                    'Kurtosis': stats.kurtosis(class_probs),
                    'Q1': np.percentile(class_probs, 25),
                    'Median': np.median(class_probs),
                    'Q3': np.percentile(class_probs, 75),
                    'IQR': np.percentile(class_probs, 75) - np.percentile(class_probs, 25)
                }
                
                data.append(metrics)
    
    return pd.DataFrame(data)

def create_ensemble_auc_table(results):
    """Create a table of ensemble AUC scores per configuration and dataset."""
    data = []
    
    for dataset_name, dataset_results in results.items():
        row = {'Dataset': dataset_name}
        
        for config, config_results in dataset_results.items():
            if config_results and 'tree_probabilities' in config_results[0] and 'true_labels' in config_results[0]:
                # Get ensemble probabilities (mean across trees)
                tree_probabilities = config_results[0]['tree_probabilities']
                true_labels = config_results[0]['true_labels']
                
                # Average probabilities across trees for ensemble prediction
                ensemble_probabilities = np.mean(tree_probabilities, axis=0)
                
                # Calculate AUC
                try:
                    if ensemble_probabilities.shape[1] > 2:  # Multi-class case
                        # Use macro averaging for multi-class
                        classes = np.arange(ensemble_probabilities.shape[1])
                        true_labels_bin = label_binarize(true_labels, classes=classes)
                        auc = roc_auc_score(true_labels_bin, ensemble_probabilities, multi_class='ovr', average='macro')
                    else:  # Binary case
                        # For binary classification, use the probability of class 1
                        auc = roc_auc_score(true_labels, ensemble_probabilities[:, 1])
                except Exception as e:
                    print(f"Error calculating AUC for {dataset_name}, {config}: {e}")
                    auc = None
                
                row[config] = auc
            else:
                row[config] = None
                
        data.append(row)
    
    return pd.DataFrame(data)

def create_tree_auc_table(results):
    """Create a table of mean tree AUC scores per configuration and dataset."""
    data = []
    
    for dataset_name, dataset_results in results.items():
        row = {'Dataset': dataset_name}
        
        for config, config_results in dataset_results.items():
            if config_results and 'tree_probabilities' in config_results[0] and 'true_labels' in config_results[0]:
                tree_probabilities = config_results[0]['tree_probabilities']
                true_labels = config_results[0]['true_labels']
                
                # Calculate AUC for each tree
                tree_aucs = []
                for tree_idx in range(tree_probabilities.shape[0]):
                    try:
                        if tree_probabilities.shape[2] > 2:  # Multi-class case
                            # Use macro averaging for multi-class
                            classes = np.arange(tree_probabilities.shape[2])
                            true_labels_bin = label_binarize(true_labels, classes=classes)
                            auc = roc_auc_score(true_labels_bin, tree_probabilities[tree_idx], multi_class='ovr', average='macro')
                        else:  # Binary case
                            # For binary classification, use the probability of class 1
                            auc = roc_auc_score(true_labels, tree_probabilities[tree_idx, :, 1])
                        tree_aucs.append(auc)
                    except Exception as e:
                        print(f"Error calculating AUC for tree {tree_idx} in {dataset_name}, {config}: {e}")
                        continue
                
                if tree_aucs:
                    row[config] = np.mean(tree_aucs)
                else:
                    row[config] = None
            else:
                row[config] = None
                
        data.append(row)
    
    return pd.DataFrame(data)

def create_auc_uplift_table(ensemble_df, tree_df):
    """Create a table showing AUC uplift between ensemble and mean tree AUC."""
    # Create a combined table with both percentage and percentage points
    combined_df = ensemble_df.copy()
    combined_df.set_index('Dataset', inplace=True)
    tree_df_idx = tree_df.set_index('Dataset')
    
    # Create a new DataFrame for the formatted results
    formatted_df = pd.DataFrame()
    formatted_df['Dataset'] = ensemble_df['Dataset']
    
    # Calculate and format uplift for each configuration
    for col in combined_df.columns:
        # Calculate percentage uplift
        pct_uplift = ((combined_df[col] - tree_df_idx[col]) / tree_df_idx[col] * 100)
        
        # Calculate percentage points uplift
        pp_uplift = (combined_df[col] - tree_df_idx[col]) * 100
        
        # Format as "X% (Y.Zpp)"
        formatted_values = []
        for pct, pp in zip(pct_uplift, pp_uplift):
            if pd.isna(pct) or pd.isna(pp):
                formatted_values.append(None)
            else:
                formatted_values.append(f"{pct:.1f}% ({pp:.1f}pp)")
        
        formatted_df[col] = formatted_values
    
    return formatted_df

def create_hyperparameters_table(results):
    """Create a table of final hyperparameters per dataset and configuration, with only selected columns."""
    selected_hyperparams = [
        'batch_size', 'epochs', 'learning_rate', 'max_depth', 'num_trees', 'subset_share'
    ]
    data = []
    for dataset_name, dataset_results in results.items():
        for config, config_results in dataset_results.items():
            if config_results:
                row = {'Dataset': dataset_name, 'Configuration': config}
                params = config_results[0]['multi_tree_params']
                if 'final_results' in config_results[0] and 'multi_tree_params' in config_results[0]['final_results']:
                    params = config_results[0]['final_results']['multi_tree_params']
                elif 'params' in config_results[0]:
                    params = config_results[0]['params']
                elif 'config' in config_results[0]:
                    params = config_results[0]['config']
                for hp in selected_hyperparams:
                    row[hp] = params.get(hp) if params and hp in params else None
                data.append(row)
    # Ensure columns are in the specified order
    columns = ['Dataset', 'Configuration'] + selected_hyperparams
    return pd.DataFrame(data, columns=columns)

def main():
    # Load all results
    results = load_results()
    
    # Create output directory
    output_dir = "outputs/tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate tables
    print("\nGenerating tables...")
    
    # 1. Ensemble accuracy table
    ensemble_df = create_ensemble_accuracy_table(results)
    ensemble_df.to_csv(os.path.join(output_dir, 'ensemble_accuracy.csv'), index=False)
    print("Generated ensemble accuracy table")
    
    # 2. Mean tree accuracy table
    tree_df = create_tree_accuracy_table(results)
    tree_df.to_csv(os.path.join(output_dir, 'tree_accuracy.csv'), index=False)
    print("Generated tree accuracy table")
    
    # 3. Accuracy uplift table (combined format)
    uplift_formatted = create_accuracy_uplift_table(ensemble_df, tree_df)
    uplift_formatted.to_csv(os.path.join(output_dir, 'accuracy_uplift_combined.csv'), index=False)
    print("Generated combined accuracy uplift table")
    
    # 4. Distribution metrics table
    dist_metrics_df = create_distribution_metrics_table(results)
    dist_metrics_df.to_csv(os.path.join(output_dir, 'distribution_metrics.csv'), index=False)
    print("Generated distribution metrics table")
    
    # 5. Ensemble AUC table
    ensemble_auc_df = create_ensemble_auc_table(results)
    ensemble_auc_df.to_csv(os.path.join(output_dir, 'ensemble_auc.csv'), index=False)
    print("Generated ensemble AUC table")
    
    # 6. Mean tree AUC table
    tree_auc_df = create_tree_auc_table(results)
    tree_auc_df.to_csv(os.path.join(output_dir, 'tree_auc.csv'), index=False)
    print("Generated tree AUC table")
    
    # 7. AUC uplift table
    auc_uplift_formatted = create_auc_uplift_table(ensemble_auc_df, tree_auc_df)
    auc_uplift_formatted.to_csv(os.path.join(output_dir, 'auc_uplift_combined.csv'), index=False)
    print("Generated AUC uplift table")
    
    # 8. Hyperparameters table
    hyperparams_df = create_hyperparameters_table(results)
    hyperparams_df.to_csv(os.path.join(output_dir, 'final_hyperparameters.csv'), index=False)
    print("Generated final hyperparameters table")

if __name__ == "__main__":
    main() 