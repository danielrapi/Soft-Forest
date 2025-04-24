"""
Data module for loading and processing datasets.

This module provides utilities for loading various datasets used in machine learning experiments,
including classification datasets from public repositories.
"""

# Import directly from the module path
from .utils.load_data import load_processed_classification_public_data, shuffle_labels

__all__ = ['load_processed_classification_public_data', 'shuffle_labels']
