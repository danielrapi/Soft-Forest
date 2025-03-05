import os
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import json
from datetime import datetime
from soft_trees_rf import RandomForestEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rf_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RandomForestTester:
    """Testing pipeline for RandomForestEnsemble models on PMLB datasets."""
    
    def __init__(self, datasets_dir="pmlb_datasets", results_dir="results"):
        """
        Initialize the tester with dataset and results directories.
        
        Args:
            datasets_dir: Directory containing PMLB datasets
            results_dir: Directory to save testing results
        """
        self.datasets_dir = datasets_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Dictionary to store dataset metadata
        self.dataset_info = {}
        
        # Dictionary to store testing results
        self.results = {}
        
        logger.info(f"Initialized RandomForestTester with datasets from {datasets_dir}")
        
    def get_available_datasets(self):
        """Get list of available datasets in the datasets directory."""
        datasets = []
        for dataset_name in os.listdir(self.datasets_dir):
            dataset_path = os.path.join(self.datasets_dir, dataset_name)
            if os.path.isdir(dataset_path):
                csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
                if csv_files:
                    datasets.append(dataset_name)
        
        logger.info(f"Found {len(datasets)} datasets: {datasets}")
        return datasets
    
    def load_dataset(self, dataset_name):
        """
        Load a dataset from the datasets directory.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            X: Features
            y: Target
            is_classification: Whether the dataset is for classification
        """
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV file found in {dataset_path}")
        
        # Use the first CSV file found
        csv_path = os.path.join(dataset_path, csv_files[0])
        logger.info(f"Loading dataset from {csv_path}")
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Assume the last column is the target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Determine if classification or regression
        unique_values = np.unique(y)
        is_classification = len(unique_values) <= 10  # Heuristic: if ≤10 unique values, treat as classification
        
        # Store dataset info
        self.dataset_info[dataset_name] = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'num_classes': len(unique_values) if is_classification else None,
            'is_classification': is_classification
        }
        
        logger.info(f"Loaded dataset {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Task type: {'Classification' if is_classification else 'Regression'}")
        
        if is_classification:
            logger.info(f"Number of classes: {len(unique_values)}")
            # For classification, convert to one-hot encoding if more than 2 classes
            if len(unique_values) > 2:
                y_one_hot = np.zeros((y.size, len(unique_values)))
                y_one_hot[np.arange(y.size), y.astype(int)] = 1
                y = y_one_hot
        
        return X, y, is_classification
    
    def generate_hyperparameters(self, num_features, is_classification):
        """
        Generate random hyperparameters according to the specified distributions.
        
        Args:
            num_features: Number of features in the dataset
            is_classification: Whether the dataset is for classification
            
        Returns:
            Dictionary of hyperparameters
        """
        # Learning rate: Uniform over {10^-1, 10^-2, ..., 10^-5}
        learning_rate = 10 ** (-random.randint(1, 5))
        
        # Batch size: Uniform over {32, 64, 128, 256, 512}
        batch_size = random.choice([32, 64, 128, 256, 512])
        
        # Number of Epochs: Discrete uniform over [5, 100]
        epochs = random.randint(5, 100)
        
        # γ: Log uniform over [10^-4, 1]
        gamma = 10 ** (random.uniform(-4, 0))
        
        # Tree Depth: Discrete uniform over [2, 8]
        tree_depth = random.randint(2, 8)
        
        # Number of Trees: Discrete uniform over [1, 100]
        num_trees = random.randint(1, 100)
        
        # L2 Regularization: Mixture model of 0 and log uniform over [10^-8, 10^2]
        if random.random() < 0.5:
            l2_reg = 0
        else:
            l2_reg = 10 ** (random.uniform(-8, 2))
        
        # For classification, determine leaf_dims based on number of classes
        if is_classification:
            # Get number of classes from dataset_info
            num_classes = self.dataset_info.get(dataset_name, {}).get('num_classes', 2)
            leaf_dims = num_classes
        else:
            # For regression, leaf_dims is 1
            leaf_dims = 1
        
        # Features per node - use default RF logic (implemented in the model)
        features_per_node = None  # Let the model decide based on regression/classification
        
        hyperparams = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'gamma': gamma,
            'tree_depth': tree_depth,
            'num_trees': num_trees,
            'l2_reg': l2_reg,
            'leaf_dims': leaf_dims,
            'features_per_node': features_per_node
        }
        
        logger.info(f"Generated hyperparameters: {hyperparams}")
        return hyperparams
    
    def build_model(self, hyperparams, input_shape, is_classification):
        """
        Build a RandomForestEnsemble model with the given hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            input_shape: Shape of the input data
            is_classification: Whether the dataset is for classification
            
        Returns:
            Compiled TensorFlow model
        """
        # Create regularizer
        kernel_regularizer = tf.keras.regularizers.L2(hyperparams['l2_reg'])
        
        # Create input layer
        inputs = tf.keras.Input(shape=(input_shape,))
        
        # Create RandomForestEnsemble layer
        rf_layer = RandomForestEnsemble(
            num_trees=hyperparams['num_trees'],
            max_depth=hyperparams['tree_depth'],
            leaf_dims=hyperparams['leaf_dims'],
            features_per_node=hyperparams['features_per_node'],
            kernel_regularizer=kernel_regularizer,
            internal_eps=0.01  # Small epsilon to prevent numerical issues
        )
        
        # Connect the layer
        outputs = rf_layer(inputs)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        if is_classification:
            if hyperparams['leaf_dims'] > 1:  # Multi-class
                loss = 'categorical_crossentropy'
                metrics = ['accuracy']
                activation = 'softmax'
            else:  # Binary
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
                activation = 'sigmoid'
        else:  # Regression
            loss = 'mse'
            metrics = ['mae']
            activation = None
        
        # Add activation layer if needed
        if activation:
            model = tf.keras.Sequential([
                model,
                tf.keras.layers.Activation(activation)
            ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate']),
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Built model with {model.count_params()} parameters")
        return model
    
    def evaluate_model(self, model, X_test, y_test, is_classification):
        """
        Evaluate the model on test data.
        
        Args:
            model: Trained TensorFlow model
            X_test: Test features
            y_test: Test targets
            is_classification: Whether the dataset is for classification
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_classification:
            if y_test.ndim > 1 and y_test.shape[1] > 1:  # Multi-class one-hot encoded
                # Convert predictions to class indices
                y_pred_class = np.argmax(y_pred, axis=1)
                y_test_class = np.argmax(y_test, axis=1)
                accuracy = accuracy_score(y_test_class, y_pred_class)
                metrics = {
                    'accuracy': accuracy
                }
            else:  # Binary classification
                # Round predictions to 0 or 1
                y_pred_class = np.round(y_pred).astype(int)
                accuracy = accuracy_score(y_test, y_pred_class)
                metrics = {
                    'accuracy': accuracy
                }
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics = {
                'mse': mse,
                'r2': r2
            }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def run_test(self, dataset_name, num_trials=5):
        """
        Run a test on a dataset with random hyperparameters.
        
        Args:
            dataset_name: Name of the dataset to test
            num_trials: Number of random hyperparameter trials to run
            
        Returns:
            Dictionary of test results
        """
        logger.info(f"Running test on dataset {dataset_name} with {num_trials} trials")
        
        # Load dataset
        X, y, is_classification = self.load_dataset(dataset_name)
        
        # Split dataset into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Store results for this dataset
        dataset_results = []
        
        # Run trials
        for trial in range(num_trials):
            logger.info(f"Starting trial {trial+1}/{num_trials} for dataset {dataset_name}")
            
            # Generate hyperparameters
            hyperparams = self.generate_hyperparameters(X.shape[1], is_classification)
            
            # Build model
            model = self.build_model(hyperparams, X.shape[1], is_classification)
            
            # Create early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test, is_classification)
            
            # Store results
            trial_results = {
                'hyperparams': hyperparams,
                'metrics': metrics,
                'training_history': {
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            dataset_results.append(trial_results)
            
            logger.info(f"Completed trial {trial+1}/{num_trials} for dataset {dataset_name}")
        
        # Find best trial
        if is_classification:
            best_trial = max(dataset_results, key=lambda x: x['metrics']['accuracy'])
            best_metric = best_trial['metrics']['accuracy']
            metric_name = 'accuracy'
        else:
            best_trial = min(dataset_results, key=lambda x: x['metrics']['mse'])
            best_metric = best_trial['metrics']['mse']
            metric_name = 'mse'
        
        logger.info(f"Best trial for {dataset_name}: {metric_name}={best_metric}")
        
        # Store results
        self.results[dataset_name] = {
            'dataset_info': self.dataset_info[dataset_name],
            'trials': dataset_results,
            'best_trial_index': dataset_results.index(best_trial),
            'best_metric': {metric_name: best_metric}
        }
        
        # Save results to file
        self.save_results(dataset_name)
        
        return self.results[dataset_name]
    
    def save_results(self, dataset_name=None):
        """
        Save results to file.
        
        Args:
            dataset_name: Name of the dataset to save results for. If None, save all results.
        """
        if dataset_name:
            # Save results for a specific dataset
            results_path = os.path.join(self.results_dir, f"{dataset_name}_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.results[dataset_name], f, indent=2)
            logger.info(f"Saved results for dataset {dataset_name} to {results_path}")
        else:
            # Save all results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"all_results_{timestamp}.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Saved all results to {results_path}")
    
    def plot_results(self, dataset_name=None):
        """
        Plot results for a dataset or all datasets.
        
        Args:
            dataset_name: Name of the dataset to plot results for. If None, plot summary for all datasets.
        """
        if dataset_name:
            # Plot results for a specific dataset
            if dataset_name not in self.results:
                logger.error(f"No results found for dataset {dataset_name}")
                return
            
            dataset_results = self.results[dataset_name]
            is_classification = dataset_results['dataset_info']['is_classification']
            
            # Plot training history for best trial
            best_trial = dataset_results['trials'][dataset_results['best_trial_index']]
            
            plt.figure(figsize=(12, 5))
            
            # Plot training loss
            plt.subplot(1, 2, 1)
            plt.plot(best_trial['training_history']['loss'], label='Training Loss')
            plt.plot(best_trial['training_history']['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training History - {dataset_name}')
            plt.legend()
            
            # Plot hyperparameter importance
            plt.subplot(1, 2, 2)
            
            # Collect hyperparameters and metrics
            hyperparams_data = []
            for trial in dataset_results['trials']:
                trial_data = {k: v for k, v in trial['hyperparams'].items() 
                             if k not in ['leaf_dims', 'features_per_node']}
                
                if is_classification:
                    trial_data['metric'] = trial['metrics']['accuracy']
                    metric_name = 'accuracy'
                else:
                    trial_data['metric'] = trial['metrics']['mse']
                    metric_name = 'mse'
                
                hyperparams_data.append(trial_data)
            
            # Convert to DataFrame
            hyperparams_df = pd.DataFrame(hyperparams_data)
            
            # Calculate correlations
            correlations = hyperparams_df.corr()[metric_name].drop(metric_name)
            
            # Plot correlations
            correlations.plot(kind='bar')
            plt.title(f'Hyperparameter Importance - {dataset_name}')
            plt.ylabel(f'Correlation with {metric_name}')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_dir, f"{dataset_name}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved plot for dataset {dataset_name} to {plot_path}")
        else:
            # Plot summary for all datasets
            if not self.results:
                logger.error("No results found")
                return
            
            # Collect best metrics for each dataset
            classification_results = {}
            regression_results = {}
            
            for dataset_name, dataset_results in self.results.items():
                is_classification = dataset_results['dataset_info']['is_classification']
                
                if is_classification:
                    classification_results[dataset_name] = dataset_results['best_metric']['accuracy']
                else:
                    regression_results[dataset_name] = dataset_results['best_metric']['mse']
            
            plt.figure(figsize=(12, 10))
            
            # Plot classification results
            if classification_results:
                plt.subplot(2, 1, 1)
                classification_df = pd.Series(classification_results).sort_values(ascending=False)
                classification_df.plot(kind='bar')
                plt.title('Classification Datasets - Best Accuracy')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
            
            # Plot regression results
            if regression_results:
                plt.subplot(2, 1, 2)
                regression_df = pd.Series(regression_results).sort_values()
                regression_df.plot(kind='bar')
                plt.title('Regression Datasets - Best MSE')
                plt.ylabel('MSE')
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.results_dir, f"summary_plot_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved summary plot to {plot_path}")
    
    def run_all_tests(self, num_trials=5, max_datasets=None):
        """
        Run tests on all available datasets.
        
        Args:
            num_trials: Number of random hyperparameter trials to run per dataset
            max_datasets: Maximum number of datasets to test. If None, test all datasets.
        """
        datasets = self.get_available_datasets()
        
        if max_datasets:
            datasets = datasets[:max_datasets]
        
        logger.info(f"Running tests on {len(datasets)} datasets with {num_trials} trials each")
        
        for dataset_name in tqdm(datasets, desc="Testing datasets"):
            try:
                self.run_test(dataset_name, num_trials)
            except Exception as e:
                logger.error(f"Error testing dataset {dataset_name}: {str(e)}")
        
        # Save all results
        self.save_results()
        
        # Plot summary
        self.plot_results()
        
        logger.info("Completed all tests")


if __name__ == "__main__":
    # Create tester
    tester = RandomForestTester(datasets_dir="pmlb_datasets", results_dir="results")
    
    # Run tests on all datasets
    # Adjust num_trials and max_datasets as needed
    tester.run_all_tests(num_trials=5, max_datasets=None)
    
    # Alternatively, test specific datasets
    # datasets_to_test = ["diabetes", "heart-c", "breast-cancer-wisconsin"]
    # for dataset in datasets_to_test:
    #     tester.run_test(dataset, num_trials=10)
    #     tester.plot_results(dataset) 