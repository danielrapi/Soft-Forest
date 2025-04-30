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

# Third-party imports
import numpy as np
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Local imports
from engine import train_model
from softensemble import *
from config import get_args, setup_logging
from runners.single_tree import run_single_tree_experiment
from runners.ensemble import run_ensemble_experiment
# Set the working directory to the parent directory 
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create the outputs directory if it doesn't exist
log_dir = "outputs/single_run"
os.makedirs(log_dir, exist_ok=True)

# Instead of changing directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Then import
from data_handling import load_processed_classification_public_data


# Get the current date and time in the format month-day-hour-minute
current_time = datetime.now().strftime("%m-%d-%H-%M")

######################################################################################
def main():
    
    # Parse the arguments
    args = get_args()
    log_file = setup_logging(args)

    #load dataset   
    try:    
        data = load_processed_classification_public_data(name = args.dataset_name, noise_level=args.noise_level)

        train_X_tensor = torch.tensor(data.x_train_processed, dtype=torch.float32)
        train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
        test_X_tensor = torch.tensor(data.x_test_processed, dtype=torch.float32)
        test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32)
            
        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
            
        # Get dimensions from the actual data
        input_dims = train_X_tensor.shape[1]
        #print(f"Input dimensions: {input_dims}")
        num_classes = data.num_classes
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        exit()
    
    try:
        if args.num_trees == 1:
            results = run_single_tree_experiment(train_dataset=train_dataset, test_dataset=test_dataset, input_dims=input_dims, num_classes=num_classes, args=args)
            logging.info(f"Single tree experiment completed")
            logging.info(f"Test accuracy: {results['test_accuracy']:.4f}")
            logging.info(f"Test AUC: {results['test_auc']:.4f}")
        else:
            results = run_ensemble_experiment(train_dataset=train_dataset, test_dataset=test_dataset, input_dims=input_dims, num_classes=num_classes, args=args)
            logging.info(f"Ensemble experiment completed")
        
        logging.info(f"Results saved to {log_file}")
        return 0
    
    except Exception as e:
        logging.error(f"Error in experiment: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
if __name__ == "__main__":
    sys.exit(main())