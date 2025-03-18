# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Machine learning models
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

import shap

## Function Import
from Azfal.Data_Prep.load_data import load_data

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

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def hyperparameter_tuning(self, X, y, param_distributions, n_iter=5, n_splits=5):
        scoring = {
            'AUC': 'roc_auc',
            'Accuracy': 'accuracy'
        }

        random_search = RandomizedSearchCV(
            estimator=self,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=5,
            scoring=scoring,
            refit='AUC',
            verbose=1
        )
        random_search.fit(X, y)
        self.model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_index = random_search.best_index_
        cv_results = random_search.cv_results_
        best_auc = -cv_results['AUC'][best_index]
        best_acc = cv_results['Accuracy'][best_index]

        print(f"Best parameters: {best_params}")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Best Accuracy: {best_acc:.4f}")

        return {
            'best_params': best_params,
            'best_auc': best_auc,
            'best_acc': best_acc
        }

class DecisionTreeClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = DecisionTreeClassifier(**kwargs)

class RandomForestClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(**kwargs)

class XGBoostClassifierModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = xgb.XGBClassifier(**kwargs)

    def compute_shap_values(self, X):
        # Use TreeExplainer for computational efficiency
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

class LogisticRegressionModel(BasePredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs, penalty='l2', C = 1.0)

    
def benchmark_pipelines(X_train, y_train, X_test, y_test, n_iter=5, n_splits=5):
    models = [
        DecisionTreeClassifierModel(),
        RandomForestClassifierModel(),
        # XGBoostRegressorModel(),  # Comment out XGBoost temporarily
        LogisticRegressionModel(),
        ]
    model_names = [
        "Decision Tree",    
        "Random Forest",
        # "XGBoost",  # Comment out XGBoost temporarily
        "Logistic Regression",
    ]
    
    param_distributions = {
        "Decision Tree": {
            "max_depth": list(range(2, 21)),  # Discrete uniform over [2, 20]
        },
        "Random Forest": {
            "n_estimators": list(range(1, 501)),  # Discrete uniform over [1, 500]
            "max_depth": list(range(2, 21)),      # Discrete uniform over [2, 20]
        }, 
        "XGBoost": {
            "n_estimators": list(range(1, 501)),  # Discrete uniform over [1, 500]
            "max_depth": list(range(2, 21)),      # Discrete uniform over [2, 20]
            "learning_rate": [10**-i for i in range(1, 6)],  # Uniform over {10^-1, 10^-2, ..., 10^-5}
            ### To be checked what the mixture coeff is and if log uniform is correct ###
            "reg_lambda": [0]*100 + [10**(np.random.uniform(-8, 2)) for _ in range(100)],  # Mixture model of 0 and log uniform over [10^-8, 10^2]
            "min_child_weight": [0],               # min child weight = 0
        }, 
        "Logistic Regression": {
            "C": [10**i for i in range(-8, 5)],  # Log uniform over [10^-8, 10^4]
            "max_iter": 1000,  # Increase maximum number of iterations to 1000
        },
    }
    #pretty(param_distributions)

    results = []
        
    for model, name in zip(models, model_names):
        print(f'Validating {name}')
        
        # Hyperparameter tuning if applicable
        if name in param_distributions and param_distributions[name]:
            print(f'Hyperparameter tuning for {name}')
            model_results = model.hyperparameter_tuning(X_train, y_train,
                                                    param_distributions=param_distributions[name],
                                                    n_iter=n_iter, n_splits=n_splits)
        else:
            best_params = {}
            print(f'No hyperparameter tuning for {name}')

        # Store results
        model_results["Model"] = name
        results.append(model_results)
        
        # Plot predictions
        # model.plot(y_test, y_pred)

    # Display the results
    results_df = pd.DataFrame(results)
    print(results_df)
    return results





if __name__ == "__main__":
    data = load_data(dataset_name="ecoli", framework="sklearn")
    X_train, X_test, y_train, y_test = data['train_X'], data['test_X'], data['train_y'], data['test_y']

    print("#### X_train ####")
    print(X_train.shape)
    print(X_train)  # X_train is a numpy array, so we print it directly
    print("#### y_train ####")
    print(y_train.shape)
    print(y_train)
    print("#### X_test ####")
    print(X_test.shape)
    print(X_test)    
    print("#### y_test ####")
    print(y_test.shape)
    print(y_test)
    benchmark_pipelines(X_train, y_train, X_test, y_test)