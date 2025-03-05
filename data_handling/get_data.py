from pmlb import fetch_data, dataset_names
import os

# Get the list of available datasets from PMLB
available_datasets = set(dataset_names)

# List of datasets you want to download
requested_datasets = [
    "ann-thyroid", "breast-cancer-wisconsin", "car-evaluation", "churn",
    "crx", "dermatology", "diabetes", "dna", "ecoli", "flare", "heart-c",
    "hypothyroid", "nursery", "optdigits", "pima", "satimage", "sleep",
    "solar-flare_2", "spambase", "texture", "twonorm", "vehicle", "yeast"
]

# Directory to save datasets
save_dir = "pmlb_datasets"
os.makedirs(save_dir, exist_ok=True)

# Track successful and failed downloads
successful = []
failed = []

# Download and save datasets
for dataset in requested_datasets:
    print(f"Attempting to download: {dataset}")
    
    # Check if dataset exists in PMLB
    if dataset not in available_datasets:
        print(f"  - Warning: '{dataset}' not found in PMLB")
        failed.append(dataset)
        continue
        
    try:
        # Download dataset
        df = fetch_data(dataset, local_cache_dir=save_dir)
        
        # Save as CSV
        csv_path = os.path.join(save_dir, f"{dataset}.csv")
        df.to_csv(csv_path, index=False)
        
        # Print success message with dataset size
        print(f"  - Successfully downloaded '{dataset}' ({df.shape[0]} rows, {df.shape[1]} columns)")
        successful.append(dataset)
        
    except Exception as e:
        print(f"  - Error downloading '{dataset}': {str(e)}")
        failed.append(dataset)

# Print summary
print("\n=== Download Summary ===")
print(f"Successfully downloaded {len(successful)}/{len(requested_datasets)} datasets")

if failed:
    print("\nFailed downloads:")
    for dataset in failed:
        print(f"  - {dataset}")
    
    # Suggest similar dataset names for failed downloads
    print("\nSuggested alternatives for failed datasets:")
    for dataset in failed:
        # Find similar dataset names using fuzzy matching
        similar = [name for name in available_datasets 
                  if any(part in name for part in dataset.split("-"))][:5]
        
        if similar:
            print(f"  - Instead of '{dataset}', try: {', '.join(similar)}")
        else:
            print(f"  - No similar datasets found for '{dataset}'")

print("\nAvailable datasets can be viewed at: https://epistasislab.github.io/pmlb/")
