"""Data Processing Utilities
"""
import collections
import copy
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import yaml
import logging

def load_processed_classification_public_data(
    name="human-activity-recognition",
    val_size=0.2,
    test_size=0.2,
    seed=8,
    noise_level=0.0
    ):
    
    #set path to parent directory of this file
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    path = os.path.join(path, "storage/")
          
    if name in ['ann_thyroid', 'breast_cancer_wisconsin', 'car_evaluation', 'churn','dermatology','dna','ecoli','hypothyroid',
                  'nursery','optdigits','sleep','vehicle']:

        full_path = path+f"{name}/{name}.csv"
        df = pd.read_csv(full_path)

        df_y = df['target']
        df_X = df.drop(columns='target')
        if name in ['sleep']:
            _, p = df_X.shape
            features = np.arange(p)
            features_to_permute = np.random.choice(features, p,  replace=False)
            num_permutes = 10
            cols = df_X.columns
            orig_f_names = []
            new_f_names = []
            for f in features_to_permute:
                f_name = cols[f]                
                for j in range(num_permutes):
                    orig_f_names.append(f_name)
                    new_f_name = '{}-Noise-{}'.format(f_name, j)
                    new_f_names.append(new_f_name)
                    df_X[new_f_name] = np.random.permutation(df_X.iloc[:, f].values)
    else:
        raise ValueError("Data: '{}' is not supported".format(name))

    np.random.seed(seed)
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(df_X, df_y, test_size=test_size, stratify=df_y, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=val_size, stratify=y_train_valid, random_state=seed)
    # print(x_train.nunique())
        
    # print("Number of samples in training set: ", x_train.shape[0], y_train.shape[0])
    # print("Number of samples in validation set: ", x_valid.shape[0], y_valid.shape[0])
    # print("Number of samples in train+validation set: ", x_train_valid.shape[0], y_train_valid.shape[0])
    # print("Number of samples in testing set: ", x_test.shape[0], y_test.shape[0])
    # print("Percentage of missing vals in training covariates: ", 100*np.count_nonzero(x_train.isna().values)/(x_train.values.size))
    # print("Percentage of missing vals in validation covariates: ", 100*np.count_nonzero(x_valid.isna().values)/(x_valid.values.size))
    # print("Percentage of missing vals in train+validation covariates: ", 100*np.count_nonzero(x_train_valid.isna().values)/(x_train_valid.values.size))
    # print("Percentage of missing vals in testing covariates: ", 100*np.count_nonzero(x_test.isna().values)/(x_test.values.size))
    # print("Number of NaNs in tasks responses in training set: ", y_train.isna().values.sum(axis=0))
    # print("Number of NaNs in tasks responses in validation set: ", y_valid.isna().values.sum(axis=0))
    # print("Number of NaNs in tasks responses in train+validation set: ", y_train_valid.isna().values.sum(axis=0))
    # print("Number of NaNs in tasks responses in train+validation set: ", y_test.isna().values.sum(axis=0))
    
    # w_train = np.ones((y_train.shape[0],))
    # w_valid = np.ones((y_valid.shape[0],))
    # w_train_valid = np.ones((y_train_valid.shape[0],))
    # w_test = np.ones((y_test.shape[0],))
    # print(x_train.shape, x_valid.shape, x_train_valid.shape, x_test.shape)
    # print(y_train.shape, y_valid.shape, y_train_valid.shape, y_test.shape)
    # print(w_train.shape, w_valid.shape, w_train_valid.shape, w_test.shape)
    
 
    if name in ['ann_thyroid', 'breast_cancer_wisconsin', 'car_evaluation', 'churn','dermatology','dna','ecoli','hypothyroid',
                  'nursery','optdigits','sleep','vehicle']:
        with open(os.path.join(path, f"{name}/metadata.yaml"), "r") as stream:
            try:
                metadata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        df_metadata = pd.DataFrame(metadata['features'])
        if  name in ['sleep']:  
            for orig_f_name, new_f_name in zip(orig_f_names, new_f_names):
                df_metadata_sub = df_metadata[df_metadata.name==orig_f_name]
                df_metadata_sub['name'] = new_f_name
                df_metadata = pd.concat([df_metadata, df_metadata_sub], axis=0)
        # from IPython.display import display
        # display(df_metadata)
        metadata = {
            'continuous_features': df_metadata[df_metadata['type']=='continuous'].name.astype(str).values,
            'categorical_features': df_metadata[df_metadata['type']=='categorical'].name.astype(str).values,
            'binary_features': df_metadata[df_metadata['type']=='binary'].name.astype(str).values,
            'nominal_features': df_metadata[df_metadata['type']=='nominal'].name.astype(str).values,
            'ordinal_features': df_metadata[df_metadata['type']=='ordinal'].name.astype(str).values,
        }
    #print(metadata)

    if metadata['ordinal_features'] is not None:
        df_X[metadata['ordinal_features']] = df_X[metadata['ordinal_features']].apply(pd.to_numeric)
    if metadata['continuous_features'] is not None:
        df_X[metadata['continuous_features']] = df_X[metadata['continuous_features']].apply(pd.to_numeric)

    if name in ['ann_thyroid', 'breast_cancer_wisconsin', 'car_evaluation', 'churn','dermatology','dna','ecoli','hypothyroid',
                  'nursery','optdigits','sleep','vehicle']:
        continuous_features = metadata['continuous_features']
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features =  metadata['categorical_features']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        binary_features =  metadata['binary_features']
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        nominal_features =  metadata['nominal_features']
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ordinal_features = metadata['ordinal_features']
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        x_preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_transformer, continuous_features),
                ('categorical', categorical_transformer, categorical_features),
                ('binary', binary_transformer, binary_features),
                ('nominal', nominal_transformer, nominal_features),
                ('ordinal', ordinal_transformer, ordinal_features),                
            ])

        # print(x_train.nunique().sort_values())
        x_train_processed = process_sparse_matrices(x_preprocessor.fit_transform(x_train))
        x_valid_processed = process_sparse_matrices(x_preprocessor.transform(x_valid))
        x_train_valid_processed = process_sparse_matrices(x_preprocessor.transform(x_train_valid))    
        x_test_processed = process_sparse_matrices(x_preprocessor.transform(x_test))
    
    
    input_dims = x_train_processed.shape[1]
    num_classes = len(set(df_y.values))

    y_preprocessor = LabelEncoder()
    y_train_processed = shuffle_labels(y_preprocessor.fit_transform(y_train), noise_level=noise_level, num_classes=num_classes)
    y_valid_processed = y_preprocessor.transform(y_valid)
    y_train_valid_processed = y_preprocessor.transform(y_train_valid)
    y_test_processed = y_preprocessor.transform(y_test)
    

    # print(x_train_processed.shape, x_valid_processed.shape, x_train_valid_processed.shape, x_test_processed.shape)
    # print(y_train_processed.shape, y_valid_processed.shape, y_train_valid_processed.shape, y_test_processed.shape)
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_train_valid', 'x_test',
                                                'y_train', 'y_valid', 'y_train_valid', 'y_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_train_valid_processed', 'x_test_processed',
                                                'y_train_processed', 'y_valid_processed',
                                                'y_train_valid_processed', 'y_test_processed', 
                                                'input_dims', 'num_classes'])
    data_processed = data_coll(x_train, x_valid, x_train_valid, x_test,
                               y_train, y_valid, y_train_valid, y_test,
                               x_train_processed, x_valid_processed,
                               x_train_valid_processed, x_test_processed,
                               y_train_processed, y_valid_processed, y_train_valid_processed, 
                               y_test_processed, input_dims, num_classes)
    return data_processed

def process_sparse_matrices(dataset):
    """Convert sparse matrices to dense if needed."""
    if hasattr(dataset, "toarray"):
        dataset = dataset.toarray()
    return dataset

def shuffle_labels(labels, noise_level=0.15, num_classes=None):
    """Shuffle a proportion of labels to test model robustness."""
    # Validate unique labels match num_classes
    unique_labels = np.unique(labels)
    if num_classes and len(unique_labels) != num_classes:
        logging.warning(f"Unique labels ({len(unique_labels)}) doesn't match num_classes ({num_classes})")
    
    # Shuffle labels
    num_samples = len(labels)
    num_to_shuffle = int(noise_level * num_samples)
    indices_to_shuffle = np.random.choice(num_samples, size=num_to_shuffle, replace=False)
    
    labels_to_shuffle = labels[indices_to_shuffle].copy()
    np.random.shuffle(labels_to_shuffle)
    
    noisy_labels = labels.copy()
    noisy_labels[indices_to_shuffle] = labels_to_shuffle
    
    logging.info(f"Shuffled {num_to_shuffle}/{num_samples} labels ({noise_level*100:.1f}%)")
    return noisy_labels