import os
import json
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_tree_predictions(h5f):
    """Load and combine predictions from all trees in the HDF5 file."""
    # Get all tree prediction keys (sorted to ensure consistent order)
    tree_keys = sorted([k for k in h5f.keys() if k.endswith('_probabilities')])
    
    if not tree_keys:
        return None, None
    
    # Load predictions from each tree
    tree_probs = []
    for key in tree_keys:
        prob = h5f[key][:]
        tree_probs.append(prob)
    
    # Stack predictions into a single array (samples × trees × classes)
    probabilities = np.stack(tree_probs)
    # Load true labels if available
    true_labels = h5f['labels'][:] if 'labels' in h5f else None
    
    return probabilities, true_labels

def load_results(dataset_name, base_path="outputs/pipeline"):
    """Load results for a specific dataset from all experiment configurations."""
    results = {
        'base': [],
        'bagging': [],
        'subset': [],
        'bagging_subset': []
    }
    
    print(f"Searching for results for dataset {dataset_name} in: {base_path}")
    
    # Get the dataset directory
    dataset_dir = Path(base_path) / dataset_name
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}")
        return results
            
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load results from each configuration
    for config_dir in dataset_dir.glob("*"):
        if not config_dir.is_dir():
            continue
            
        config_name = config_dir.name
        print(f"  Processing config: {config_name}")
        
        # Map directory names to configuration types
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
        print(f"    Latest run: {latest_run}")
        
        # Load results.json
        results_file = latest_run / f"{dataset_name}_results.json"
        trees_file = latest_run / f"{dataset_name}_trees_results.h5"
        
        print(f"    Results file exists: {results_file.exists()}")
        print(f"    Trees file exists: {trees_file.exists()}")
        
        if not results_file.exists():
            continue
            
        with open(results_file) as f:
            result = json.load(f)
            result['dataset'] = dataset_name
            
            # Load tree predictions if available
            if trees_file.exists():
                try:
                    with h5py.File(trees_file, 'r') as h5f:
                        print(f"    Available keys in h5 file: {list(h5f.keys())}")
                        predictions, true_labels = load_tree_predictions(h5f)
                        if predictions is not None:
                            result['tree_probabilities'] = predictions
                            print(f"    Loaded predictions shape: {predictions.shape}")
                        if true_labels is not None:
                            result['true_labels'] = true_labels
                            print(f"    Loaded true labels shape: {true_labels.shape}")
                except Exception as e:
                    print(f"    Error loading h5 file: {e}")
            
            results[config_type].append(result)
    
    # Print summary of loaded results
    print(f"\nLoaded results summary for {dataset_name}:")
    for config, config_results in results.items():
        print(f"{config}: {len(config_results)} results")
        for result in config_results:
            print(f"  - Dataset: {result['dataset']}")
            print(f"    Has predictions: {'tree_probabilities' in result}")
            print(f"    Has true labels: {'true_labels' in result}")
    
    return results

def plot_accuracy_comparison(results, dataset_name, output_dir):
    """Plot accuracy comparison between different configurations."""
    # Prepare data
    data = []
    for config, config_results in results.items():
        for result in config_results:
            if 'tree_probabilities' not in result or 'true_labels' not in result:
                continue
                
            #compute std of accuracy by tree 
            tree_probabilities = result['tree_probabilities']
            tree_predictions = np.argmax(tree_probabilities, axis=-1)
            tree_accuracy = np.mean(tree_predictions == result['true_labels'])
            tree_accuracy_std = np.std(tree_predictions == result['true_labels'])

            data.append({
                'Configuration': config,
                'Ensemble Accuracy': result['final_results']['accuracy'],
                'Dataset': result['dataset'],
                'Tree Accuracy': tree_accuracy,
                'Tree Accuracy Std': tree_accuracy_std
            })
    
    if not data:
        print(f"No data available for accuracy comparison for {dataset_name}")
        return
        
    df = pd.DataFrame(data)
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Set positions for the bars
    x = np.arange(len(df))
    width = 0.35
    
    # Plot bars for ensemble and tree accuracy
    bars1 = ax1.bar(x - width/2, df['Ensemble Accuracy'], width, label='Ensemble Accuracy')
    bars2 = ax1.bar(x + width/2, df['Tree Accuracy'], width, label='Tree Accuracy')
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Create second axis for standard deviation
    ax2 = ax1.twinx()
    line = ax2.plot(x, df['Tree Accuracy Std'], 'r-', marker='o', label='Tree Accuracy Std')
    
    # Customize the plot
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Standard Deviation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Set x-axis ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Configuration'], rotation=45)
    
    # Set y-axis range based on minimum accuracy
    min_accuracy = min(df['Ensemble Accuracy'].min(), df['Tree Accuracy'].min())
    ax1.set_ylim(min_accuracy - 0.2, 1.0)
    
    # Add dataset labels as annotations
    for i, row in df.iterrows():
        ax1.annotate(row['Dataset'], 
                    (i, max(row['Ensemble Accuracy'], row['Tree Accuracy']) + 0.02),
                    ha='center', 
                    fontsize=8)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    plt.title(f'Accuracy Comparison Across Configurations - {dataset_name}')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add extra space at the bottom for the legend
    
    output_file = os.path.join(output_dir, f'{dataset_name}_accuracy_comparison.png')
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.close()

def plot_prediction_std(results, dataset_name, output_dir):
    """Plot histograms of probabilities for each class, with all configurations in one plot."""
    print(f"\nAttempting to create probability histograms for {dataset_name}...")
    
    # Find the first result with tree_probabilities
    first_result = None
    for config in results:
        if results[config] and 'tree_probabilities' in results[config][0]:
            first_result = results[config][0]
            break
    
    if not first_result:
        print(f"No tree probabilities found for {dataset_name}")
        return
    
    # Determine number of classes
    num_classes = first_result['tree_probabilities'].shape[2]
    
    # Create a separate figure for each class
    for class_idx in range(num_classes):
        plt.figure(figsize=(12, 8))
        
        # Plot histogram for each configuration
        for config, config_results in results.items():
            for result in config_results:
                if 'tree_probabilities' not in result:
                    continue
                
                tree_probabilities = result['tree_probabilities']
                print("SHAPE OF TREE PROBS for CLASS", class_idx, tree_probabilities.shape)
                
                # Get probabilities for this class
                class_probs = tree_probabilities[:, :, class_idx].flatten()
                
                # Normalize the histogram to show density instead of counts
                # This scales all histograms to the same area regardless of number of trees
                plt.hist(class_probs, bins=50, alpha=0.6, label=config, density=True)
                
                break  # Only use the first result for each configuration
        
        # Customize the plot
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.title(f'Probability Distribution for Class {class_idx}\nDataset: {dataset_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the figure
        output_file = os.path.join(output_dir, f'{dataset_name}_probability_histogram_class_{class_idx}.png')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved {output_file}")
        plt.close()

def plot_prediction_distribution(results, dataset_name, output_dir):
    """Plot the probability density of predictions across trees for each configuration."""
    plt.figure(figsize=(12, 6))
    
    colors = {
        'base': '#1f77b4',
        'bagging': '#ff7f0e',
        'subset': '#2ca02c',
        'bagging_subset': '#d62728'
    }
    
    # For each class
    num_classes = None
    for config, config_results in results.items():
        if config_results:
            probabilities = config_results[0]['tree_probabilities']
            num_classes = probabilities.shape[-1]
            break
    
    if num_classes is None:
        print("No probability data found")
        return
        
    for class_idx in range(num_classes):
        plt.figure(figsize=(12, 6))
        
        for config, config_results in results.items():
            if not config_results:
                continue
                
            # Get probabilities for this configuration and class
            probabilities = config_results[0]['tree_probabilities']
            class_probs = probabilities[:, :, class_idx].flatten()  # Flatten to get all tree predictions
            
            # Create KDE plot
            sns.kdeplot(data=class_probs, 
                       label=config, 
                       color=colors[config],
                       fill=True,
                       alpha=0.3)
        
        plt.title(f'Probability Distribution for Class {class_idx}\nDataset: {dataset_name}')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'probability_distribution_class_{class_idx}.png'))
        plt.close()

def main():
    # Set base paths
    base_path = "outputs/pipeline"
    output_base_dir = "outputs/plots"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of dataset directories
    dataset_dirs = [d for d in Path(base_path).glob("*") if d.is_dir()]

    print(f"Found {len(dataset_dirs)} dataset directories")
    
    # Set default style parameters
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Process each dataset
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Create output directory for this dataset
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Load results for this dataset
        results = load_results(dataset_name, base_path)
        
        # Skip if no results found
        if not any(results.values()):
            print(f"No results found for {dataset_name}, skipping...")
            continue
        
        # Generate plots
        print(f"\nGenerating plots for {dataset_name}...")
        plot_accuracy_comparison(results, dataset_name, dataset_output_dir)
        plot_prediction_std(results, dataset_name, dataset_output_dir)
        plot_prediction_distribution(results, dataset_name, dataset_output_dir)
        
        print(f"Completed processing for {dataset_name}")

if __name__ == "__main__":
    main() 