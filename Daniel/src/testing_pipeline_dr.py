
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class train_fastel_rf:
    def __init__(self, model = None, dataset_name = None, hyperparameters = None):
        self.model = model
        self.dataset = self.get_dataset(dataset_name)
        self.hyperparameters = self.get_hyperparameters()
        
    def get_dataset(self, dataset_name):
        #check if dataset is in pmlb
        #load files in pmlb_datasets
        files = os.listdir("pmlb_datasets")
        print(files)
        if dataset_name + '.csv' not in files:
            raise ValueError(f"Dataset {dataset_name + '.csv'} not found in PMLB")
        #load dataset

        dataset = pd.read_csv(f"pmlb_datasets/{dataset_name}.csv")
        print(dataset.head())
        return dataset

    def get_hyperparameters(self):
        
        hyperparameters = {
            "learning_rate": np.logspace(-1, -5, 5),
            "batch_size": [32, 64, 128, 256, 512],
            "num_epochs": range(5, 100),
            "gamma": np.logspace(-4, 0, 5),
            "tree_depth": range(2,8),
            "num_trees": range(1,100),
            "l2_regularization": np.logspace(-8, 2, 11)
        }

        print(hyperparameters)

        return hyperparameters

    def preprocess_data(self):
        #preprocess data
        #standardize data
        X = self.dataset.drop(columns=["target"])
        y = self.dataset["target"]

        # Store dataset info
        self.dataset_info = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y))
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test
        
    def draw_hyperparameters(self):
        #draw hyperparameters from hyperparameters
        #random draw from hyperparams dict
        params = {}
        for key, value in self.hyperparameters.items():
                params[key] = np.random.choice(value)
    
        return params

    def train_model(self, iterations = 10, num_folds = 5):
        #train model
        #use random search to find best hyperparameters
        for i in range(iterations):
            params = self.draw_hyperparameters()
            print(params)
        #train model with best hyperparameters
        #return model
        pass    

if __name__ == "__main__":
    ecoli = train_fastel_rf(dataset_name = "ecoli")
    ecoli.train_model()