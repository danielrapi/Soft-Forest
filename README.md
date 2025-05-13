# Soft Forests: Combining Random Forest Randomization with Differentiable Soft Tree Ensembles

This repository contains the implementation of Soft Forests, a novel approach that combines the randomization techniques of Random Forests with differentiable soft tree ensembles. This work was developed as part of an independent study (15.961) at MIT's Operations Research Center.

## Overview

Soft Forests explores the impact of incorporating classic random forest techniques, namely subset feature selection, bagging, and bootstrapping into soft ensembles. The goal is to understand whether these ideas, originally designed for hard decision trees, can enhance the performance, robustness, or diversity of soft ensemble methods.

Key features:
- Implementation of soft decision trees with differentiable sigmoid-based splits
- Integration of random forest randomization techniques
- Support for both classification and regression tasks
- Comprehensive hyperparameter tuning framework
- Extensive evaluation across multiple datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soft-forests.git
cd soft-forests
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
soft-forests/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── engine/         # Core training engine
│   ├── softensemble/   # Soft tree ensemble implementation
│   └── utils/          # Utility functions
├── outputs/            # Results and model outputs
│   ├── hyperopt_multi/ # Multi-stage hyperparameter optimization results
│   ├── hyperopt_single/# Single-stage optimization results
│   ├── single_run/     # Individual run results
│   └── tables/         # Generated tables and metrics
└── tests/              # Unit tests
```

## Usage

### Basic Usage

```python
from src.softensemble import SoftForest
from src.data import load_dataset

# Load your dataset
X_train, y_train, X_test, y_test = load_dataset('your_dataset')

# Initialize and train the model
model = SoftForest(
    num_trees=10,
    max_depth=5,
    subset_share=0.5,  # Feature subset ratio
    use_bootstrapping=True
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Hyperparameter Tuning

The project includes a two-stage hyperparameter tuning process:

1. Single-tree parameter optimization:
   - Learning rate
   - Batch size
   - Training epochs
   - Maximum tree depth

2. Ensemble-level parameter optimization:
   - Number of trees
   - Subset share ratio

```python
from src.parameter_tuning import run_hyperopt

# Run hyperparameter optimization
best_params = run_hyperopt(
    dataset_name='your_dataset',
    num_trials=30,
    stage='single'  # or 'ensemble'
)
```

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
- Built upon the Tree Ensemble Layer (TEL) framework
- Inspired by the FASTEL implementation 
