"""
Parameter tuning module for hyperparameter optimization.

This module provides utilities for optimizing hyperparameters of machine learning models,
including single tree models and ensemble tree models.
"""

# Import directly from the module path
from .hyperopt_single_tree import optimize_hyperparams as optimize_single_tree
from .hyperopt_multi_tree import optimize_hyperparams as optimize_multi_tree

__all__ = ['optimize_single_tree', 'optimize_multi_tree']
