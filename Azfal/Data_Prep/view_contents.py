################################################ GOAL #############################################################
'''
    Just a file to see what all of the datasets look like to know how to adjust my params
'''
###################################################################################################################

import h5py
import numpy as np
import logging

def print_hdf5_structure(file_path):
    def recursively_print(name, obj):
        print(name)
    
    with h5py.File(file_path, 'r') as file:
        file.visititems(recursively_print)

# Replace 'your_file.h5' with the actual file path
# print_hdf5_structure('datasets.h5')



def grab_data_info(file_path, dataset):
    '''
        Returns information about the data set needed to run a softtree
            (num_features-->(input_dims), num_classes)
    '''
    with h5py.File(file_path, 'r') as file:
        if f"{dataset}/train_X" in file and f"{dataset}/train_y" in file:
            train_X = file[f"{dataset}/train_X"]
            train_y = file[f"{dataset}/train_y"]

            # (f"Dataset: {dataset}")
            logging.info(f"  train_X has : {train_X.shape[1]} features")
            logging.info(f"  The dataset is a : {len(np.unique(np.array(train_y)))} classification problem")
        else:
            logging.error(f"{dataset} had some error!")

        return (train_X.shape[1], len(np.unique(np.array(train_y))))
