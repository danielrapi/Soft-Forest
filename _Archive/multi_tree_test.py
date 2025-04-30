#######################################  GOAL  ######################################
# This will be the file that runs everything in the HPC enviroment 
# Our goal is to make this as modular as possible with full flexibility

#####################################################################################
# Standard library imports
import os
import sys
import time
import argparse
import logging
from datetime import datetime
import json
# Third-party imports
import numpy as np
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

# Local imports
from engine import train_model
from softensemble import *

# Set the working directory to the parent directory 
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create the outputs directory if it doesn't exist
log_dir = "outputs/single_run"
os.makedirs(log_dir, exist_ok=True)

# Instead of changing directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Then import
from data import load_processed_classification_public_data


# Get the current date and time in the format month-day-hour-minute
current_time = datetime.now().strftime("%m-%d-%H-%M")

# Add this near the top of your file
def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        logging.info("CUDA not available, using CPU instead")
        return "cpu"

# Then in your main code, before the loops
device = get_device()
logging.info(f"Using device: {device}")

##################  DATA LOADING #####################################################
'''
# This section will be dedicated to selecting what data set that the user wants to use
# Flags:
#      1.) dataset_name
#      2.) framework: What framework that the user wants to use
#           a.  sklearn: will return a dictionary of numpy arrays {train_X, test_X, train_y, test_y}
            b. torch: Will return a tuple of (train_loader, test_loader)
# , file_path='datasets.h5', batch_size=32 

NEED TO TEST OTHER FUNCTIONALITIES.
''' 
######################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Experiment Set Up",
        description="This is meant to set up an environment on an HPC to test randomization ideas on soft trees.",
        epilog="Add for parameters later"
    )
    
    # # Create subparsers for different modes
    # subparsers = parser.add_subparsers(dest="mode", help="Select mode to run")

    # # Subcommand for torch
    # torch_parser = subparsers.add_parser("torch", help="Run with torch framework")
    # torch_parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    # torch_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for torch or tf loading data")
    # torch_parser.add_argument("--num_trees", type=int, required=True, help="Number of trees for the model")
    # torch_parser.add_argument("--leaf_dims", action="store_true", help="Number of classes if not included, otherwise extend to higher feature space")
    # # torch_parser.add_argument("--input_dims", type=int, required=True, help="Number of features")
    # torch_parser.add_argument("--max_depth", type=int, required=True, help="Max depth for the trees")
    # torch_parser.add_argument("--combine_output", action="store_true", default=True, help="Combine the output into leaf_dims")
    # torch_parser.add_argument("--subset_selection", action="store_true", help="Run soft trees with Hadamard product for random feature selection")
    # torch_parser.add_argument("--epochs", type=int,required=True,help="EPOCHS")
    # torch_parser.add_argument("--lr", type=float,required=True,help="learning_rate")
    # torch_parser.add_argument("--device", type = str, help = "cuda or cpu")
    # torch_parser.add_argument("--bootstrap", action="store_true", help = "Whether to boostrap or not")
    
    # Parse the arguments
    # args = parser.parse_args()

    depth_range = [3, 5, 7]
    dataset_names = os.listdir("data/storage/")
    dataset_names = ['ecoli', 'optdigits', 'churn']
    subset_selection = [True, False]
    print(dataset_names)

    
    results = []

    for dataset_name in tqdm(dataset_names, desc="Datasets"):
        for depth in tqdm(depth_range, desc=f"Depths for {dataset_name}", leave=False):
            for subset_selection_option in tqdm(subset_selection, desc=f"Subset selection for {dataset_name}, depth {depth}", leave=False):
                #load dataset   
                try:    

                    log_file = os.path.join(
                        log_dir,
                        f"run_logs_{dataset_name}_depth_{depth}_subsetselection_{subset_selection}_{current_time}.log"
                        )
                    logging.basicConfig(
                        filename=log_file,
                        filemode="a",  # Append mode
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        level=logging.INFO
                    )    

                    data = load_processed_classification_public_data(name = dataset_name)
                    
                    # Convert sparse matrices to dense numpy arrays if needed
                    if hasattr(data.x_train_processed, "toarray"):  # Check if it's a sparse matrix
                        x_train_dense = data.x_train_processed.toarray()
                        x_test_dense = data.x_test_processed.toarray()
                    else:
                        x_train_dense = data.x_train_processed
                        x_test_dense = data.x_test_processed
                    
                    # Convert to PyTorch tensors
                    train_X_tensor = torch.tensor(x_train_dense, dtype=torch.float32)
                    train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
                    test_X_tensor = torch.tensor(x_test_dense, dtype=torch.float32)
                    test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32)
                    
                    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
                    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
                    
                    # Create DataLoaders for batching
                    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                    
                    # Get dimensions from the actual data
                    input_dims = train_X_tensor.shape[1]
                    #print(f"Input dimensions: {input_dims}")
                    num_classes = data.num_classes
                except Exception as e:
                    logging.error(f"Error: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    exit()
            
            
            ############################################## MULTIPLE TREES #########################################################

                results_objects = {}

                logging.info(f"Running in torch mode with dataset: {dataset_name}")
                logging.info(f"Batch size: {32}")
                logging.info(f"Number of trees: {10}, Max depth: {depth}")
                logging.info(f"Combine output: {True}, Subset selection: {subset_selection}")

            
                # store all the predictions 
                all_preds = []
                start_time = time.time()
                for tree in range(10):

                    # FIGURE OUT HOW TO MAKE THIS EFFICIENT LATER IF SLOW
                    try:
                        # Create DataLoaders for batching
                        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                        
                    except Exception as e:
                        logging.error(f"Error loading data: {e}")
                        continue

                    logging.info(f"Running Tree Number {tree + 1}")
                
                    # create the model
                    model = SoftTreeEnsemble(
                        num_trees=1,max_depth=depth, 
                        leaf_dims=num_classes, input_dim=input_dims, 
                        combine_output=True, subset_selection=subset_selection
                    )

                    # then we train the model 
                    train_model(model=model, train_loader=train_dataloader, 
                                test_loader=test_dataloader, 
                                epochs=100, 
                                learning_rate=0.05, device=device)
                    
                    end_time = time.time()

                    execution_time = (end_time - start_time) / 60

                    logging.info(f"The single tree took {execution_time} to run.")

                    # Get predictions on test set
                    model.eval()

                    # we will have to store the predictions in order to count the majority 
                    tree_preds = []
                    
                    # potentially do this in the training loop 
                    # for efficiency as well
                    with torch.no_grad():
                        for X_batch, y_batch in test_dataloader:
                            X_batch = X_batch.to(device)
                            outputs = model(X_batch)
                            probs = torch.softmax(outputs, dim=1) 
                            tree_preds.append(probs.cpu().numpy())

                    # Convert lists to numpy arrays
                    tree_preds = np.concatenate(tree_preds, axis=0)
                    
                    # Store predictions
                    all_preds.append(tree_preds)

                # reshape to (samples, num_calsses)
                all_preds = np.stack(all_preds, axis=0)
                
                #take argmax by tree and then majority vote, shape (num_trees, samples, num_classes)
                argmax_all_preds = np.argmax(all_preds, axis = 2)
                #take the mode of the predictions
                final_predictions = stats.mode(argmax_all_preds, axis=0)[0]

                # get true labels 
                true_labels = np.concatenate([y_batch.numpy() for _, y_batch in test_dataloader], axis=0)

                accuracy =  accuracy_score(true_labels, final_predictions)
                try:
                    # For binary classification
                    if num_classes == 2:
                        # For binary classification, we need probabilities for the positive class
                        # Average probabilities across trees and take the probability of class 1
                        avg_probs = np.mean(all_preds, axis=0)
                        pos_class_probs = avg_probs[:, 1]  # Probability of positive class
                        AUC = roc_auc_score(true_labels, pos_class_probs)
                    # For multi-class classification
                    else:
                        # Get the one-hot encoded predictions for multi-class AUC
                        # We need probabilities for each class, not just the final prediction
                        # Use the average of probabilities from all trees
                        avg_probs = np.mean(all_preds, axis=0)
                        AUC = roc_auc_score(true_labels, avg_probs, multi_class='ovr', average='macro')
                        
                    logging.info(f"AUC: {AUC:.4f}")
                except Exception as e:
                    logging.error(f"Error calculating AUC: {e}")
                    AUC = None
                
                logging.info(f"Ensemble Accuracy: {accuracy:.4f}")
                if AUC is not None:
                    logging.info(f"Ensemble AUC: {AUC:.4f}")
                
                #predict majority of true labels
                baseline = accuracy_score(true_labels, np.full(true_labels.shape, stats.mode(true_labels, axis=0)[0]))
                logging.info(f"Baseline Accuracy: {baseline:.4f}")
                
                #results
                results.append({
                    "dataset_name": dataset_name,
                    "depth": depth,
                    "subset_selection": subset_selection_option,
                    "accuracy": accuracy,
                    "AUC": AUC,
                    "baseline": baseline
                })
                         
                results_df = pd.DataFrame(results)
                #save results object as csv file
                results_df.to_csv(os.path.join(log_dir, f"results_{current_time}.csv"), index=False)
                    