# Data manipulation and numerical operations
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt

# Machine learning modelsr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import shap

#hyperopt sklearn
from hpsklearn import HyperoptEstimator
from hyperopt import tpe, hp, fmin, Trials

#set working directory to current directory 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#import xgboost as xgb
## Function Import
#from Main_Implementation.load_data import load_data

#support function to pretty print the dictionary
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

class BasePredictor(BaseEstimator):  # Inherit from BaseEstimator for sklearn compatibility
    def __init__(self,  **kwargs):
        self.model = None
        # Default estimator type - will be overridden by subclasses
        self._estimator_type = "classifier"
        # print(f"[DEBUG] Initializing {self.__class__.__name__} with estimator_type: {self._estimator_type}")
        # print(f"[DEBUG] kwargs: {kwargs}")

    def fit(self, X, y):
        # print(f"[DEBUG] Fitting {self.__class__.__name__} - model type: {type(self.model)}")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # print(f"[DEBUG] Predicting with {self.__class__.__name__}")
        return self.model.predict(X)
    
    def predict_proba(self, X): 
        # print(f"[DEBUG] predict_proba called on {self.__class__.__name__}")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            print(f"WARNING: model {type(self.model)} doesn't have predict_proba method!")
            # Return a dummy probability output as fallback
            preds = self.predict(X)
            n_classes = len(np.unique(preds))
            proba = np.zeros((len(X), n_classes))
            for i, p in enumerate(preds):
                proba[i, int(p)] = 1.0
            return proba

    def get_params(self, deep=True):
        # print(f"[DEBUG] get_params called on {self.__class__.__name__}")
        params = self.model.get_params(deep=deep)
        # print(f"[DEBUG] model params: {params}")
        return params

    def set_params(self, **params):
        # print(f"[DEBUG] set_params called with: {params}")
        self.model.set_params(**params)
        return self

    def hyperparameter_tuning(self, X, y, param_distributions, n_iter=5, n_splits=5):
        print(f"\nStarting hyperparameter tuning for {self.__class__.__name__}")
        # print(f"[DEBUG] param_distributions: {param_distributions}")
        # print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
        
        # Check if this is a multi-class problem
        unique_classes = np.unique(y)
        is_multiclass = len(unique_classes) > 2
        print(f"Unique classes in y: {unique_classes}")
        print(f"Is multi-class problem: {is_multiclass}")
        
        # Define scoring metrics based on problem type
        if is_multiclass:
            print(f"Using multi-class scoring configuration")
            scoring = 'roc_auc_ovr'
        else:
            print(f"Using binary classification scoring configuration")
            scoring = 'roc_auc'
        
        #randomized search
        """
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring=scoring,
            refit='AUC',
            verbose=1
        )
        """

        def objective(params):
            model = self.model.set_params(**params)
            score = cross_val_score(model, 
                                    X, y, 
                                    scoring=scoring, 
                                    cv=5).mean()

            return -score

        trials = Trials()
        best = fmin(objective, 
                    param_distributions, 
                    algo=tpe.suggest, 
                    max_evals=n_iter, 
                    trials=trials,
                    )
        print(f"Best parameters: {best}")
        # Get the loss (negative score) of the best trial
        best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
        best_trial_loss = trials.trials[best_trial_idx]['result']['loss']
        best_score = -best_trial_loss  # Convert back to positive score
        
        print(f"Best trial index: {best_trial_idx}")
        print(f"Best trial loss: {best_trial_loss}")
        print(f"Best score (AUC): {best_score:.4f}")
 
        print(f"HyperoptSearch completed")
        # Store best estimator
        self.model = self.model.set_params(**best)
       
        # Extract results
        best_params = best
        cv_results = trials
        
        return {
            'best_params': best_params,
            'best_auc': best_score,
            'trials': trials
        }

class DecisionTreeClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeClassifier(**kwargs)

class RandomForestClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(**kwargs)
        self._estimator_type = "classifier"

class XGBoostClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = xgb.XGBClassifier(**kwargs)
        self._estimator_type = "classifier"

    def compute_shap_values(self, X):
        # Use TreeExplainer for computational efficiency
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

class LogisticRegressionModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs, penalty='l2', C = 1.0, max_iter=1000)
        self._estimator_type = "classifier"

    
def benchmark_pipelines(X_train, y_train, X_test, y_test, n_iter=5, n_splits=5):
    # Prepare scaled data for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"Data shapes - Original: {X_train.shape}, Scaled: {X_train_scaled_df.shape}")
    
    models = [
        DecisionTreeClassifierModel(),
        RandomForestClassifierModel(),
        #XGBoostClassifierModel(),  # Comment out XGBoost temporarily
        LogisticRegressionModel(),
        ]
    model_names = [
        "Decision Tree",    
        "Random Forest",
        #"XGBoost",  # Comment out XGBoost temporarily
        "Logistic Regression",
    ]
    
    param_distributions = {
        "Decision Tree": {
            "max_depth": hp.uniformint("max_depth", 2, 20),
        },
        "Random Forest": {
            "n_estimators": hp.uniformint("n_estimators", 1, 500),  # Integer uniform over [1, 500]
            "max_depth": hp.uniformint("max_depth", 2, 20),      # Integer uniform over [2, 20]
        }, 
        "XGBoost": {
            "n_estimators": hp.uniformint("n_estimators", 1, 500),  # Integer uniform over [1, 500]
            "max_depth": hp.uniformint("max_depth", 2, 20),      # Integer uniform over [2, 20]
            "learning_rate": hp.loguniform("learning_rate", np.log(10**-5), np.log(10**-1)),  # Log uniform over [10^-5, 10^-1]
            "reg_lambda": hp.loguniform("reg_lambda", np.log(10**-8), np.log(10**2)),  # Log uniform over [10^-8, 10^2]
            "min_child_weight": [0],               # min child weight = 0
        }, 
        "Logistic Regression": {
            "C": hp.loguniform("C", np.log(10**-8), np.log(10**4)),  # Log uniform over [10^-8, 10^4]
        },
    }
    
    # Check if all model names are in param_distributions
    for name in model_names:
        if name not in param_distributions:
            print(f"WARNING: {name} not found in param_distributions")
            
    #pretty(param_distributions)

    results = []
        
    for model, name in zip(models, model_names):
        print(f'\nValidating {name}')
        
        # Use scaled data for logistic regression, original data for others
        if name == "Logistic Regression":
            current_X_train = X_train_scaled_df
            #current_X_test = X_test_scaled_df
            print(f"Using SCALED data for {name}")
        else:
            current_X_train = X_train
            #current_X_test = X_test
            print(f"Using original data for {name}")
        
        # Hyperparameter tuning if applicable
        if name in param_distributions and param_distributions[name]:
            print(f'Hyperparameter tuning for {name}')
            try:
                model_results = model.hyperparameter_tuning(current_X_train, y_train,
                                                    param_distributions=param_distributions[name],
                                                    n_iter=n_iter, n_splits=n_splits)
                # Store results
                model_results["Model"] = name
                results.append(model_results)
            except Exception as e:
                print(f"ERROR during hyperparameter tuning for {name}: {e}")
                import traceback
                traceback.print_exc()
                model_results = {"Model": name, "best_params": {}, "best_auc": 0, "trials": None, "error": str(e)}
                results.append(model_results)
        else:
            print(f'No hyperparameter tuning for {name}')
            model_results = {"Model": name, "best_params": {}, "best_auc": 0, "trials": None, "note": "No tuning performed"}
            results.append(model_results)
        
        # Plot predictions
        # model.plot(y_test, y_pred)

    # Display the results
    results_df = pd.DataFrame(results)
    print("\nFinal results:")
    print(results_df)
    return results

def get_data(dataset_name):
    #check if dataset is in pmlb
    #load files in pmlb_datasets
    files = os.listdir("./pmlb_datasets")
    #print(files)
    if dataset_name not in files:
        raise ValueError(f"Dataset {dataset_name} not found in PMLB")
    #load dataset

    try:
        dataset = pd.read_csv(f"./pmlb_datasets/{dataset_name}")
        #print(dataset.head())
        return dataset

    except Exception as e:
        print(f"Dataset {dataset_name} failed to read.")
        print(e)
        return None

def preprocess_data(dataset):
        #preprocess data
        #standardize data
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        # Store dataset info
        dataset_info = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y))
        }
        #print(dataset_info)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test, dataset_info


if __name__ == "__main__":
    all_datasets_results = []  # Rename to avoid confusion

    for file_name in os.listdir("./pmlb_datasets"):
        if file_name != ".DS_Store":
            print(f"Running {file_name}")
            dataset_results = []  # Store results for this dataset
            
            for i in range(5):
                dataset = get_data(file_name)
                print(dataset.head())
                X_train, X_test, y_train, y_test, dataset_info = preprocess_data(dataset)
                
                # Get model results
                model_results = benchmark_pipelines(X_train, y_train, X_test, y_test, n_iter=10, n_splits=5)
                
                # Add dataset info and run number to each model result
                for result in model_results:
                    result['run'] = i
                    
                # Extend the dataset results with this run's results
                dataset_results.extend(model_results)
            
            # Create DataFrame for this dataset and save it
            print(f"Results for {file_name}:")
            results_df = pd.DataFrame(dataset_results)
            print(results_df['trials'].iloc[0].trials)
            print(results_df)
            
            # Create results directory if it doesn't exist
            os.makedirs("./results", exist_ok=True)
            os.makedirs("./results/trials", exist_ok=True)

            all_trials = results_df['trials'].map(lambda x: x.trials if x.trials is not None else None)
            results_df.drop(columns=['trials'], inplace=True)
            
            all_trials.to_csv(f"./results/trials/trials_{file_name.replace('.csv', '')}", index=False)
            results_df.to_csv(f"./results/result_{file_name.replace('.csv', '')}", index=False)
            
            # Add to all dataset results
            all_datasets_results.extend(dataset_results)
            
            # Remove this to process all datasets

    """
    SAVE COMBINED RESULTS - TO BE FIXED
    # Save combined results from all datasets
    if all_datasets_results:
        combined_df = pd.DataFrame(all_datasets_results)
        
        # Group by dataset and model to calculate statistics
        stats_df = combined_df.groupby(['dataset', 'Model']).agg({
            'best_auc': ['mean', 'median', 'max', 'min', 'std'],
            'best_acc': ['mean', 'median', 'max', 'min', 'std']
        }).reset_index()
        
        # Flatten column names
        stats_df.columns = ['dataset', 'Model'] + [
            f'{metric}_{stat}' 
            for metric in ['best_auc', 'best_acc']
            for stat in ['mean', 'median', 'max', 'min', 'std']
        ]
        
        # Save both detailed and summary results
        combined_df.to_csv("results/all_datasets_results", index=False)
        stats_df.to_csv("results/all_datasets_summary", index=False)
    """        
    


"""

def test_model(model_name, param_dist, X_train, y_train, X_test, y_test):

    params = param_dist[model_name]
    # Select the model
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression()#,
        #"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }

    model = models[model_name]
    # RandomizedSearchCV for hyperparameter optimization
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        scoring='accuracy',
        n_jobs=-1,
        cv=5,
        verbose=2,
        random_state=42
    )

    # Fit the model
    random_search.fit(X_train, y_train)

    # Best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return best_model, accuracy, report


"""