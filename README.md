# Soft Forests: Combining Random Forest Randomization with Differentiable Soft Tree Ensembles

This repository contains the implementation of Soft Forests, a novel approach that combines the randomization techniques of Random Forests with differentiable soft tree ensembles. This work was developed as part of an independent study (15.961) at MIT's Operations Research Center.

## Overview

Soft Forests explores the impact of incorporating classic random forest techniques, namely subset feature selection, bagging, and bootstrapping into soft ensembles. The goal is to understand whether these ideas, originally designed for hard decision trees, can enhance the performance, robustness, or diversity of soft ensemble methods.

Key features:
- Implementation of soft decision trees with differentiable sigmoid-based splits in pytorch
- Integration of random forest randomization techniques (both bootstrapping & feature subset selection)
- Support for classification tasks (incl. data preprocessing)
- Comprehensive hyperparameter tuning framework using Hyperopt
- Extensive evaluation across multiple datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soft-forests.git
cd Soft-Forest
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
src/                                # Main source code directory
├── config.py                       # Configuration and argument parsing
├── data_handling/                  # Data loading and preprocessing modules
│   ├── storage/                    # Data storage directory
│   └── utils/                      # Data utility functions
├── engine.py                       # Core training and evaluation logic
├── main.py                         # Main entry point for experiments
├── parameter_tuning/               # Hyperparameter optimization
│   ├── hyperopt_multi_tree.py      # Multi-tree hyperparameter tuning
│   ├── hyperopt_single_tree.py     # Single-tree hyperparameter tuning
│   └── load_tough_data.py          # Utilities for loading challenging datasets
├── runners/                        # Experiment runners
│   ├── ensemble.py                 # Ensemble experiment implementation
│   └── single_tree.py              # Single tree experiment implementation
├── softensemble.py                 # Soft tree ensemble implementation
└── testing/                        # Testing and evaluation scripts
    ├── extract_results.py          # Results extraction utilities
    ├── plot_results.py             # Visualization utilities
    ├── rf_ensemble_test.py         # Random forest comparison tests
    └── run_pipeline.py             # End-to-end testing pipeline
outputs/                            # Results and output files
├── pipeline/                       # Pipeline execution outputs
├── plots/                          # Visualization and performance plots
└── tables/                         # Performance metrics and comparison tables
requirements.txt                    # Project dependencies
.gitignore                          # Git ignore file
tree.png                            # Example visualization of a soft tree
```

## Usage

### Basic Usage (Single Tree and Multiple Trees)

Run a single tree or ensemble experiment using the main entry point:

```bash
python3 src/main.py torch \
    --dataset_name DATASET_NAME \
    --batch_size 32 \
    --num_trees 10 \
    --max_depth 5 \
    --epochs 10 \
    --lr 0.05 \
    --device cpu \
    --combine_output \
    --subset_selection \
    --subset_share 0.5 \
    --bootstrap \
    --noise_level 0.15
```

**Arguments:**
- `--dataset_name`: Name of the dataset (required)
- `--batch_size`: Batch size (default: 32)
- `--num_trees`: Number of trees (required)
- `--max_depth`: Maximum tree depth (required)
- `--epochs`: Number of training epochs (required)
- `--lr`: Learning rate (required)
- `--device`: Device to use (`cpu` or `cuda`, default: cpu)
- `--combine_output`: Combine output into leaf_dims (flag)
- `--subset_selection`: Use random feature selection (flag)
- `--subset_share`: Share of features to use for subset selection
- `--bootstrap`: Whether to use bootstrapping (flag)
- `--noise_level`: Proportion of labels to shuffle (default: 0.15)

You can omit flags like `--subset_selection`, `--bootstrap`, or `--subset_share` to run standard (non-randomized) soft trees.

### Hyperparameter Tuning

Run single-tree hyperparameter optimization with Hyperopt:

```bash
python3 src/parameter_tuning/hyperopt_single_tree.py \
    --dataset DATASET_NAME \
    --max_evals 30 \
    --device cpu \
    --output_dir outputs/hyperopt_single \
    --subset_selection \
    --noise_level 0.0
```

**Arguments:**
- `--dataset`: Name of the dataset (required)
- `--max_evals`: Maximum number of evaluations (default: 30)
- `--device`: Device to use (`cpu` or `cuda`, default: cpu)
- `--output_dir`: Output directory for results (default: outputs/hyperopt_single)
- `--subset_selection`: Enable subset selection (flag)
- `--noise_level`: Proportion of labels to shuffle (default: 0.0)

## Results

The implementation has been evaluated across 11 benchmark classification datasets, comparing four configurations:
1. Standard TEL (no randomization)
2. TEL with Bootstrapping
3. TEL with Random Feature Subset Selection
4. TEL with Bootstrapping + Feature Subset

Key findings:
- Stochastic configurations outperformed vanilla TEL in 9 out of 11 benchmarks
- Randomized variants achieved the largest accuracy uplift in 10 datasets
- The combination of bootstrapping and feature subset selection produced the best results in 5 tasks

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{softforests2024,
  title={Soft Forests: Combining Random Forest Randomization with Differentiable Soft Tree Ensembles},
  author={Peermohammed, Azfal and Rapoport, Daniel},
  journal={MIT Operations Research Center},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Supervised by Prof. Rahul Mazumder at MIT's Operations Research Center
- Inspired by the FASTEL implementation (Shibal Ibrahim)
- Theories developed on top of TEL framework (Hussein Hazimeh)