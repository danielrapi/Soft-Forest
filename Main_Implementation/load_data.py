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
import os
import pandas as pd

# make sure that files are found independent of the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Write a function to retrieve the data set

def load_data(file_name, framework='sklearn', file_path=script_dir + '/pmlb_datasets/', batch_size=32, train_test_ratio=0.7, train_val_ratio=0):
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
  # filename to include .csv ending
  if not file_name.endswith('.csv'):
    file_name = file_name + '.csv'
  df = pd.read_csv(file_path + file_name)
  
  # split the data into train,val and test
  #last column is the target variable
  X = df.drop(columns=df.columns[-1])
  y = df[df.columns[-1]]

  X = X.astype(float)
  y = y.astype(float)

  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1-train_test_ratio, random_state=42)

  if train_val_ratio > 0:
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=train_val_ratio, random_state=42)

    # Format data based on the specified framework
  if framework == 'sklearn':
        # Return NumPy arrays (compatible with scikit-learn)
      if train_val_ratio > 0:
        return {'train_X':train_X, 'val_X':val_X, 'test_X':test_X, 'train_y':train_y, 'val_y':val_y, 'test_y':test_y}
      else:
        return {'train_X':train_X, 'test_X':test_X, 'train_y':train_y, 'test_y':test_y}

  elif framework == 'torch':
      
      train_dataset = TensorDataset(torch.tensor(train_X.to_numpy(), dtype=torch.float32), 
                                      torch.tensor(train_y.to_numpy(), dtype=torch.float32))
      test_dataset = TensorDataset(torch.tensor(test_X.to_numpy(), dtype=torch.float32), 
                                     torch.tensor(test_y.to_numpy(), dtype=torch.float32))

      if train_val_ratio > 0:
        val_dataset = TensorDataset(torch.tensor(val_X.to_numpy(), dtype=torch.float32), 
                                     torch.tensor(val_y.to_numpy(), dtype=torch.float32))
        
      # Create DataLoaders for batching
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

      if train_val_ratio > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader,val_loader,test_loader
      
      else:
        return train_loader,test_loader 

  elif framework == 'boost':
        # Convert to DMatrix format (XGBoost's preferred format)
      train_dmatrix = xgb.DMatrix(train_X, label=train_y)
      test_dmatrix = xgb.DMatrix(test_X, label=test_y)

      if train_val_ratio > 0:
        val_dmatrix = xgb.DMatrix(val_X, label=val_y)
        return train_dmatrix, val_dmatrix, test_dmatrix
      else:
        return train_dmatrix, test_dmatrix

  elif framework == 'tf':
      # Create TensorFlow datasets
      train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
      test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
      
      # Batch and shuffle data
      train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
      test_dataset = test_dataset.batch(batch_size)

      if train_val_ratio > 0:
        val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y))
        val_dataset = val_dataset.batch(batch_size)
        return train_dataset, val_dataset, test_dataset
      else:
        return train_dataset, test_dataset
      
  else:
      raise ValueError("Unsupported framework. Choose from 'sklearn', 'pytorch', 'xgboost', or 'tensorflow'.")
  
def get_filenames(file_path = script_dir + '/pmlb_datasets/.') -> list[str]:
  #get all the files in the directory
  files = os.listdir(file_path)
  #return the files
  return files

#test the load_data function
if __name__ == "__main__":
  filenames = get_filenames()
  print("sklearn: ", str(load_data(filenames[0], framework='sklearn')))
  print("torch: ", str(load_data(filenames[0], framework='torch')))
  print("boost: ", str(load_data(filenames[0], framework='boost')))
  print("tf: ", str(load_data(filenames[0], framework='tf')))

