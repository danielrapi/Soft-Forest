"""
Main package for the soft trees implementation.

This package provides a collection of modules for training, evaluating, and optimizing
soft decision tree models and ensembles for machine learning tasks.
"""

# Import core functionality to make it available at the package level
from .engine import train_model
from .data_handling import load_processed_classification_public_data, shuffle_labels
from .parameter_tuning import optimize_single_tree, optimize_multi_tree

__all__ = [
    'train_model',
    'load_processed_classification_public_data',
    'shuffle_labels',
    'optimize_single_tree',
    'optimize_multi_tree'
]
