import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load the results CSV file
results_file = "outputs/single_run/results_04-18-09-35.csv"
results_df = pd.read_csv(results_file)
print(results_df)

# Create output directory for plots
plot_dir = "outputs/plots"
os.makedirs(plot_dir, exist_ok=True)

# Set the style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Function to create comparison plots
def create_comparison_plots(metric):
    # For each dataset
    for dataset in results_df['dataset_name'].unique():
        # Filter data for this dataset
        dataset_df = results_df[results_df['dataset_name'] == dataset]
        
        # Set up the figure
        plt.figure(figsize=(12, 6))
        
        # Get unique depths
        depths = sorted(dataset_df['depth'].unique())
        x = np.arange(len(depths))
        width = 0.35
        
        # Get data for subset_selection True and False, handling missing values
        true_values = []
        false_values = []
        
        for d in depths:
            # Get True values, use NaN if missing
            true_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == True)]
            if len(true_df) > 0:
                true_values.append(true_df[metric].values[0])
            else:
                true_values.append(np.nan)
            
            # Get False values, use NaN if missing
            false_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == False)]
            if len(false_df) > 0:
                false_values.append(false_df[metric].values[0])
            else:
                false_values.append(np.nan)
        
        # Create the grouped bar chart, skipping NaN values
        plt.bar(x - width/2, true_values, width, label='Subset Selection: True', color='#1f77b4')
        plt.bar(x + width/2, false_values, width, label='Subset Selection: False', color='#ff7f0e')
        
        # Add baseline if plotting accuracy
        if metric == 'accuracy':
            baseline = dataset_df['baseline'].iloc[0]
            plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline: {baseline:.2f}')
        
        # Add labels and title
        plt.xlabel('Tree Depth')
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'{metric.capitalize()} Comparison for {dataset.capitalize()}')
        plt.xticks(x, depths)
        plt.legend()
        
        # Add value labels on top of bars, skipping NaN values
        for i, v in enumerate(true_values):
            if not np.isnan(v):
                plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
        for i, v in enumerate(false_values):
            if not np.isnan(v):
                plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{dataset}_{metric}_comparison.png'), dpi=300)
        plt.close()

# Create plots for accuracy and AUC
create_comparison_plots('accuracy')
create_comparison_plots('AUC')

# Create a combined plot showing all datasets
datasets = results_df['dataset_name'].unique()
num_datasets = len(datasets)

# Handle the case where we only have one dataset
if num_datasets == 1:
    fig, axes = plt.subplots(num_datasets, 2, figsize=(15, 5))
    axes = np.array([axes])  # Make it 2D for consistent indexing
else:
    fig, axes = plt.subplots(num_datasets, 2, figsize=(15, 5*num_datasets))

for i, dataset in enumerate(datasets):
    dataset_df = results_df[results_df['dataset_name'] == dataset]
    depths = sorted(dataset_df['depth'].unique())
    x = np.arange(len(depths))
    width = 0.35
    
    # Accuracy plot with error handling
    true_acc = []
    false_acc = []
    
    for d in depths:
        # Handle True values
        true_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == True)]
        if len(true_df) > 0:
            true_acc.append(true_df['accuracy'].values[0])
        else:
            true_acc.append(np.nan)
        
        # Handle False values
        false_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == False)]
        if len(false_df) > 0:
            false_acc.append(false_df['accuracy'].values[0])
        else:
            false_acc.append(np.nan)
    
    axes[i, 0].bar(x - width/2, true_acc, width, label='Subset Selection: True', color='#1f77b4')
    axes[i, 0].bar(x + width/2, false_acc, width, label='Subset Selection: False', color='#ff7f0e')
    baseline = dataset_df['baseline'].iloc[0]
    axes[i, 0].axhline(y=baseline, color='r', linestyle='--', label=f'Baseline: {baseline:.2f}')
    axes[i, 0].set_title(f'Accuracy for {dataset.capitalize()}')
    axes[i, 0].set_xticks(x)
    axes[i, 0].set_xticklabels(depths)
    axes[i, 0].set_xlabel('Tree Depth')
    axes[i, 0].set_ylabel('Accuracy')
    
    # AUC plot with error handling
    true_auc = []
    false_auc = []
    
    for d in depths:
        # Handle True values
        true_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == True)]
        if len(true_df) > 0:
            true_auc.append(true_df['AUC'].values[0])
        else:
            true_auc.append(np.nan)
        
        # Handle False values
        false_df = dataset_df[(dataset_df['depth'] == d) & (dataset_df['subset_selection'] == False)]
        if len(false_df) > 0:
            false_auc.append(false_df['AUC'].values[0])
        else:
            false_auc.append(np.nan)
    
    axes[i, 1].bar(x - width/2, true_auc, width, label='Subset Selection: True', color='#1f77b4')
    axes[i, 1].bar(x + width/2, false_auc, width, label='Subset Selection: False', color='#ff7f0e')
    axes[i, 1].set_title(f'AUC for {dataset.capitalize()}')
    axes[i, 1].set_xticks(x)
    axes[i, 1].set_xticklabels(depths)
    axes[i, 1].set_xlabel('Tree Depth')
    axes[i, 1].set_ylabel('AUC')
    
    # Add legend to the first row only
    if i == 0:
        axes[i, 0].legend()
        axes[i, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'all_datasets_comparison.png'), dpi=300)
plt.close()

print(f"Plots saved to {plot_dir}")