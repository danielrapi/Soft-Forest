"""
Data module for loading and processing datasets.

This module provides utilities for loading various datasets used in machine learning experiments,
including classification datasets from public repositories.
"""

# Import directly from the module path
from .single_tree import run_single_tree_experiment
from .ensemble import run_ensemble_experiment
__all__ = ['run_single_tree_experiment', 'run_ensemble_experiment']
