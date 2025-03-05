import os
import pandas as pd
import gzip
import glob

def convert_tsvgz_to_csv(input_dir="pmlb_datasets"):
    """
    Recursively walks through input_dir, finds all .tsv.gz files,
    and converts them to .csv if a corresponding .csv doesn't already exist.
    """
    # Count statistics
    total_files = 0
    converted = 0
    already_exists = 0
    errors = 0
    
    print(f"Scanning {input_dir} for .tsv.gz files...")
    
    # Walk through directory structure
    for root, dirs, files in os.walk(input_dir):
        # Find all .tsv.gz files in current directory
        tsvgz_files = [f for f in files if f.endswith('.tsv.gz')]
        
        for tsvgz_file in tsvgz_files:
            total_files += 1
            base_name = os.path.splitext(os.path.splitext(tsvgz_file)[0])[0]  # Remove both .tsv and .gz
            tsvgz_path = os.path.join(root, tsvgz_file)
            
            # Create csv filename in same directory as .tsv.gz
            csv_path = os.path.join(root, f"{base_name}.csv")
            
            # Check if CSV already exists
            if os.path.exists(csv_path):
                print(f"  - Skipping {tsvgz_file} (CSV already exists)")
                already_exists += 1
                continue
            
            try:
                print(f"  - Converting {tsvgz_file} to CSV...")
                # Read the .tsv.gz file
                df = pd.read_csv(tsvgz_path, compression='gzip', sep='\t')
                
                # Save as CSV
                df.to_csv(csv_path, index=False)
                converted += 1
                print(f"    ✓ Created {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
                
            except Exception as e:
                errors += 1
                print(f"    ✗ Error converting {tsvgz_file}: {str(e)}")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Total .tsv.gz files found: {total_files}")
    print(f"Files converted to CSV: {converted}")
    print(f"Files already had CSV: {already_exists}")
    print(f"Conversion errors: {errors}")
    
    return converted, already_exists, errors

# Alternative version that uses glob pattern matching instead of os.walk
def convert_tsvgz_to_csv_glob(input_dir="pmlb_datasets"):
    """
    Uses glob to find all .tsv.gz files in input_dir and subdirectories,
    and converts them to .csv if a corresponding .csv doesn't already exist.
    """
    # Count statistics
    converted = 0
    already_exists = 0
    errors = 0
    
    # Find all .tsv.gz files recursively
    tsvgz_pattern = os.path.join(input_dir, "**", "*.tsv.gz")
    tsvgz_files = glob.glob(tsvgz_pattern, recursive=True)
    
    print(f"Found {len(tsvgz_files)} .tsv.gz files in {input_dir}...")
    
    for tsvgz_path in tsvgz_files:
        # Get directory and filename
        directory = os.path.dirname(tsvgz_path)
        filename = os.path.basename(tsvgz_path)
        
        # Generate CSV filename (remove .tsv.gz extension)
        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove both .tsv and .gz
        csv_path = os.path.join(directory, f"{base_name}.csv")
        
        # Check if CSV already exists
        if os.path.exists(csv_path):
            print(f"  - Skipping {filename} (CSV already exists)")
            already_exists += 1
            continue
        
        try:
            print(f"  - Converting {filename} to CSV...")
            # Read the .tsv.gz file
            df = pd.read_csv(tsvgz_path, compression='gzip', sep='\t')
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            converted += 1
            print(f"    ✓ Created {csv_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
        except Exception as e:
            errors += 1
            print(f"    ✗ Error converting {filename}: {str(e)}")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Total .tsv.gz files found: {len(tsvgz_files)}")
    print(f"Files converted to CSV: {converted}")
    print(f"Files already had CSV: {already_exists}")
    print(f"Conversion errors: {errors}")
    
    return converted, already_exists, errors

if __name__ == "__main__":
    # Use the glob version which is simpler and more efficient
    convert_tsvgz_to_csv_glob() 