######################################### GOAL #################################
# Put all of the datasets in a hdf5 file. 
# Makes it super easy to grab whatever we want 
################################################################################

import h5py
from sklearn.model_selection import train_test_split
from pmlb import dataset_names, fetch_data

# The 23 datasets used in TEL
pmlb_datasets = ['ann_thyroid', 'breast_cancer','car_evaluation','churn', 'crx', 'dermatology', 'diabetes', 
                 'dna', 'ecoli', 'flare', 'heart_c', 'hypothyroid', 'nursery', 'optdigits', 'pima', 'satimage', 'sleep', 
                 'solar_flare_2', 'spambase', 'texture', 'twonorm', 'vehicle', 'yeast']

# ones that need to be checked
failed_datasets = []

# Create an HDF5 file
with h5py.File('datasets.h5', 'w') as h5file:
    for classification_data_set in pmlb_datasets:

        print(f"Processing dataset: {classification_data_set}")

        try:
          # fetch the data 
          X, y = fetch_data(classification_data_set, return_X_y=True)
        except:
          print(f"Dataset {classification_data_set} failed to load")
          failed_datasets.append(classification_data_set)
          continue

        # Split the data into training and testing sets 

        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7)

        # Create a group for each dataset
        group = h5file.create_group(classification_data_set)

        # Save train and test data
        group.create_dataset('train_X', data=train_X)
        group.create_dataset('test_X', data=test_X)
        group.create_dataset('train_y', data=train_y)
        group.create_dataset('test_y', data=test_y)





