######################################### GOAL #################################
# Fetch from hdf5 file and then put the data into whatever format that is wantet
################################################################################

import h5py
from sklearn.model_selection import train_test_split
from pmlb import dataset_names,fetch_data
import torch
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import tensorflow as tf



# Write a function to retrieve the data set

def load_data(dataset_name, framework='sklearn', file_path='datasets.h5', batch_size=32, bootstrap = False):
  '''
    Function Goal:
      Loads in data from the hdf5 file containing all datasets 
    Parameters:
      dataset_name: Name of the data set 
      framework: 
        sklearn --> numpy arrays {train_X, test_X, train_y, test_y}
        torch --> (train_loader, test_loader)
        boost --> (train, test)
        tf --> (train, test)
  '''

    # Open the HDF5 file
  with h5py.File(file_path, 'r') as h5file:
        # Navigate to the group (folder) for the specified dataset
      try:
        group = h5file[dataset_name]
      except KeyError:
        raise ValueError(f"Dataset '{dataset_name}' not found in the HDF5 file.")
        
        # Load train and test data as NumPy arrays
      train_X = group['train_X'][:]
      test_X = group['test_X'][:]
      train_y = group['train_y'][:]
      test_y = group['test_y'][:]
        
    # Format data based on the specified framework
  if framework == 'sklearn':
        # Return NumPy arrays (compatible with scikit-learn)
      return {'train_X':train_X, 'test_X':test_X, 'train_y':train_y, 'test_y':test_y}

  elif framework == 'torch':
      train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
      train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
      test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
      test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

      # we will apply boostrapping if it is true
      if bootstrap:
          # Generate bootstrap indices with replacement
          indices = torch.randint(0, train_X_tensor.shape[0], (train_X_tensor.shape[0],))
          train_X_tensor = train_X_tensor[indices]
          train_y_tensor = train_y_tensor[indices]

      # Convert to PyTorch tensors
      train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
      test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
        
      # Create DataLoaders for batching
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      return train_loader, test_loader

  elif framework == 'boost':
        # Convert to DMatrix format (XGBoost's preferred format)
      train_dmatrix = xgb.DMatrix(train_X, label=train_y)
      test_dmatrix = xgb.DMatrix(test_X, label=test_y)
      return train_dmatrix, test_dmatrix

  elif framework == 'tf':
        # Create TensorFlow datasets
      train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
      test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        
        # Batch and shuffle data
      train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
      test_dataset = test_dataset.batch(batch_size)
      return train_dataset, test_dataset

  else:
      raise ValueError("Unsupported framework. Choose from 'sklearn', 'pytorch', 'xgboost', or 'tensorflow'.")
  