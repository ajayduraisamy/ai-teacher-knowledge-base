# Concept: Grid Search

## Concept ID

DL-163

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the exhaustive nature of grid search
- Implement grid search in PyTorch
- Analyze the computational cost of grid search as dimensionality grows
- Identify when grid search is appropriate vs inefficient
- Compare grid search with random search

## Prerequisites

- Hyperparameter search (DL-162)
- Understanding of Cartesian products
- Cross-validation concepts
- Experiment tracking (DL-161)

## Definition

Grid search is a hyperparameter optimization method that exhaustively evaluates all combinations of specified hyperparameter values. For each hyperparameter, the practitioner defines a discrete set of candidate values, and the Cartesian product of all sets is evaluated. Grid search is simple, parallelizable, and guarantees finding the best combination within the predefined grid, but its computational cost grows exponentially with the number of hyperparameters (curse of dimensionality).

## Intuition

Grid search is like trying every combination of ingredients in a recipe systematically. If you have 3 types of flour, 4 sweeteners, and 5 leavening agents, you bake 3*4*5 = 60 loaves of bread. You will definitely find the best combination among your options. But if you add 2 more ingredients with 3 options each, suddenly you need 60*3*3 = 540 loaves. This exponential explosion makes grid search impractical beyond 3-4 hyperparameters.

## Why This Concept Matters

Grid search was the default hyperparameter tuning method in early machine learning and is still useful for: (1) exploring a small number of discrete hyperparameters, (2) reproducing published results where specific grid values were used, (3) initial exploration when the search space is small, and (4) as a baseline to compare with more advanced methods. Understanding grid search's limitations (curse of dimensionality) motivates the need for random search and Bayesian optimization.

## Code Examples

### Example 1: Basic Grid Search

`python
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.data import DataLoader, TensorDataset

def grid_search(param_grid, train_fn, eval_fn):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    results = []
    n_configs = len(list(product(*values)))
    
    for i, combo in enumerate(product(*values)):
        config = dict(zip(keys, combo))
        print(f"[{i+1}/{n_configs}] Testing: {config}")
        
        try:
            model = train_fn(config)
            val_metric = eval_fn(model)
            results.append({**config, 'metric': val_metric})
            print(f"  -> metric = {val_metric:.4f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
    
    return results

def train_fn(config):
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 5))
    opt = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    x = torch.randn(200, 20)
    y = torch.randint(0, 5, (200,))
    
    for epoch in range(5):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
    return model

def eval_fn(model):
    x = torch.randn(100, 20)
    y = torch.randint(0, 5, (100,))
    model.eval()
    with torch.no_grad():
        _, preds = torch.max(model(x), 1)
        return (preds == y).float().mean().item()

param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'wd': [0.0, 0.001],
}

results = grid_search(param_grid, train_fn, eval_fn)

best = max(results, key=lambda r: r['metric'])
print(f"\nBest: {best}")
# Output:
# [1/6] Testing: {'lr': 0.1, 'wd': 0.0}
#   -> metric = 0.2100
# [2/6] Testing: {'lr': 0.1, 'wd': 0.001}
#   -> metric = 0.2300
# ...
# Best: {'lr': 0.01, 'wd': 0.001, 'metric': 0.2500}
`

### Example 2: Grid Search with Cross-Validation

`python
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import numpy as np

def cross_validated_grid_search(param_grid, model_class, dataset, k_folds=3):
    n = len(dataset)
    fold_size = n // k_folds
    indices = list(range(n))
    np.random.shuffle(indices)
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []
    
    for combo in product(*values):
        config = dict(zip(keys, combo))
        fold_metrics = []
        
        for fold in range(k_folds):
            val_idx = indices[fold * fold_size: (fold+1) * fold_size]
            train_idx = [i for i in indices if i not in val_idx]
            
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            # Training (simplified)
            model = model_class()
            opt = torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.01))
            
            # Simulated training
            for epoch in range(3):
                for x, y in train_data:
                    opt.zero_grad()
                    loss = nn.CrossEntropyLoss()(model(x.unsqueeze(0)), y.unsqueeze(0))
                    loss.backward()
                    opt.step()
            
            # Validation
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for x, y in val_data:
                    pred = model(x.unsqueeze(0)).argmax(1)
                    correct += (pred == y).sum().item()
                    total += 1
            fold_metrics.append(correct / total)
        
        mean_metric = np.mean(fold_metrics)
        std_metric = np.std(fold_metrics)
        results.append({**config, 'mean_metric': mean_metric, 'std_metric': std_metric})
        print(f"Config: {config} -> mean={mean_metric:.4f} +/- {std_metric:.4f}")
    
    return results

# Simulated data
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n=100):
        self.x = torch.randn(n, 20)
        self.y = torch.randint(0, 5, (n,))
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

param_grid = {'lr': [0.1, 0.01, 0.001], 'wd': [0.0, 0.0001]}
dataset = SimpleDataset(300)

results = cross_validated_grid_search(
    param_grid, 
    lambda: nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5)), 
    dataset, k_folds=3
)

best = max(results, key=lambda r: r['mean_metric'])
print(f"\nBest config: lr={best['lr']}, wd={best['wd']}, "
      f"mean accuracy={best['mean_metric']:.4f}")
# Output:
# Config: {'lr': 0.1, 'wd': 0.0} -> mean=0.1950 +/- 0.0250
# Config: {'lr': 0.1, 'wd': 0.0001} -> mean=0.2100 +/- 0.0300
# Config: {'lr': 0.01, 'wd': 0.0} -> mean=0.2050 +/- 0.0200
# Config: {'lr': 0.01, 'wd': 0.0001} -> mean=0.2200 +/- 0.0150
# Config: {'lr': 0.001, 'wd': 0.0} -> mean=0.1850 +/- 0.0100
# Config: {'lr': 0.001, 'wd': 0.0001} -> mean=0.1900 +/- 0.0200
#
# Best config: lr=0.01, wd=0.0001, mean accuracy=0.2200
`

### Example 3: Grid Search Scalability Analysis

`python
import numpy as np
import matplotlib.pyplot as plt

def grid_search_cost(n_params, n_values_per_param):
    """Number of trials needed for grid search."""
    return n_values_per_param ** n_params

def random_search_coverage(n_trials, n_params):
    """Expected fraction of each parameter value sampled."""
    return 1 - (1 - 1/n_trials) ** n_params

print("Grid search trial count vs number of hyperparameters:")
print(f"{'Params':10s} {'Values/Param':15s} {'Trials':15s}")
print("-" * 45)

for n_params in [1, 2, 3, 4, 5, 10]:
    n_values = 10
    n_trials = grid_search_cost(n_params, n_values)
    print(f"{n_params:5d}      {n_values:10d}      {n_trials:15d}")
    if n_trials > 100000:
        print("  -> Impractical!")
        break
# Output:
# Params     Values/Param    Trials
# ---------------------------------------------
#     1               10             10
#     2               10            100
#     3               10           1000
#     4               10          10000
#     5               10         100000
#   -> Impractical!
`

## Common Mistakes

1. **Using too many hyperparameters**: Grid search cost grows exponentially. Limit to 2-4 hyperparameters.
2. **Using too many values per parameter**: 5-10 values per parameter is sufficient for most cases.
3. **Not using log scales**: For learning rate and weight decay, grid values should be logarithmically spaced.
4. **Overlapping grid density**: Having more values in one dimension than another creates imbalance.
5. **No validation in inner loop**: Each grid configuration must be evaluated on a held-out validation set, not the training set.

## Interview Questions

### Beginner

1. What is grid search?
2. How does grid search evaluate hyperparameter combinations?
3. What is the curse of dimensionality in grid search?
4. When is grid search appropriate?
5. How does grid search differ from random search?

### Intermediate

1. Explain why grid search becomes impractical beyond 4 hyperparameters.
2. How would you implement grid search with cross-validation?
3. Compare the coverage of grid search vs random search.
4. How many values per parameter are typically used in grid search?
5. Why should log-spaced values be used for certain hyperparameters?

### Advanced

1. Prove that random search is more efficient than grid search for hyperparameter optimization when the number of important parameters is small.
2. Design a multi-stage grid search (coarse to fine) and analyze its efficiency.
3. Implement a grid search that automatically prunes unpromising regions.

## Practice Problems

### Easy

1. How many trials for a grid with 3 parameters and 5 values each?
2. How many trials for 4 parameters and 3 values each?
3. Write a function to generate log-spaced grid values.
4. Implement grid search for learning rate and batch size.
5. Print the top 3 configurations from a grid search.

### Medium

1. Implement grid search with k-fold cross-validation.
2. Compare grid search with random search on a fixed budget.
3. Implement a coarse-to-fine grid search.
4. Add parallel execution to grid search (evaluate multiple configs simultaneously).
5. Visualize grid search results as a heatmap.

### Hard

1. Implement a distributed grid search that runs on multiple GPUs/machines.
2. Design an adaptive grid search that increases resolution near promising regions.
3. Prove that grid search with optimal adaptive resolution is equivalent to Bayesian optimization.

## Solutions

### Easy Solutions

1. 3^5 = 243 trials
2. 4^3 = 64 trials
3. np.logspace(np.log10(min_val), np.log10(max_val), num_values)
4. Iterate over product(lr_list, bs_list), train for each, record metric
5. Sort results by metric descending, print first 3

## Related Concepts

- Random Search (DL-164)
- Bayesian Hyperparameter Optimization (DL-165)
- Hyperparameter Search (DL-162)
- Hyperparameter Importance

## Next Concepts

- Random Search (DL-164)
- Bayesian Hyperparameter Optimization (DL-165)
- Learning Curves (DL-166)

## Summary

Grid search exhaustively evaluates all hyperparameter combinations in a predefined grid. It is simple and guarantees finding the best configuration within the grid, but suffers from the curse of dimensionality — cost grows exponentially with the number of hyperparameters. Grid search is best limited to 2-4 critical hyperparameters.

## Key Takeaways

- Exhaustive enumeration over all specified hyperparameter values
- Cost = product of number of values per parameter
- Curse of dimensionality: impractical for >4 parameters
- Guarantees finding best combination in grid
- Use log-spaced values for positive parameters (LR, WD)
- Best for small, discrete search spaces
- Random search is more efficient for higher dimensions
- Always use cross-validation for reliable evaluation
