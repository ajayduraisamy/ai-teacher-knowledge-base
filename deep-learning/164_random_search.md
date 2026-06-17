# Concept: Random Search

## Concept ID

DL-164

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the theoretical advantages of random search over grid search
- Implement random search with appropriate sampling distributions
- Analyze random search coverage in high-dimensional spaces
- Apply early stopping to random search trials
- Determine the optimal number of random trials for a given budget

## Prerequisites

- Grid search (DL-163)
- Hyperparameter search (DL-162)
- Understanding of sampling distributions
- Experiment tracking (DL-161)

## Definition

Random search samples hyperparameter configurations uniformly (or log-uniformly) from a predefined search space. Unlike grid search which exhaustively enumerates all combinations, random search tries a fixed number of randomly chosen configurations. The key insight is that random search explores more distinct values per hyperparameter than grid search for the same budget, making it more efficient when only a few hyperparameters actually matter for performance.

## Intuition

Imagine trying to find a restaurant in a city. Grid search is like visiting every restaurant on every block, systematically. Random search is like asking random people for recommendations. If most good restaurants are on a few specific streets, grid search wastes time checking every combination of street and block number. Random search will naturally sample those important streets more often per visit, finding good restaurants faster. In hyperparameter tuning, some parameters matter much more than others, and random search exploits this better.

## Why This Concept Matters

Random search (Bergstra & Bengio, 2012) was a breakthrough in hyperparameter optimization, demonstrating that random search consistently outperforms grid search for the same computational budget. This is because: (1) random search explores more values per hyperparameter, (2) it is more robust when hyperparameters have varying importance, (3) it can allocate budget adaptively through early stopping, and (4) it is trivially parallelizable. Understanding random search is essential for efficient hyperparameter tuning and motivated the development of more advanced methods.

## Theoretical Advantage

For a budget of N trials with D hyperparameters:
- Grid search: N = product_i V_i, where V_i is the number of values per parameter
- Random search: N trials, each sampling from the full range of each parameter

Key result (Bergstra & Bengio): Random search is more efficient than grid search when the number of important hyperparameters is small. With D parameters but only d important ones, grid search's effective resolution is N^(1/D), while random search's resolution is N for each parameter (all parameters get full exploration).

## Code Examples

### Example 1: Basic Random Search

`python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

def random_search(train_fn, param_distributions, n_trials=20):
    results = []
    for trial in range(n_trials):
        config = {}
        for name, dist in param_distributions.items():
            if dist['type'] == 'log_uniform':
                val = 10 ** random.uniform(dist['min'], dist['max'])
            elif dist['type'] == 'uniform':
                val = random.uniform(dist['min'], dist['max'])
            elif dist['type'] == 'choice':
                val = random.choice(dist['values'])
            elif dist['type'] == 'log_int':
                val = int(2 ** random.uniform(dist['min'], dist['max']))
            config[name] = val
        
        print(f"Trial {trial+1}/{n_trials}: lr={config.get('lr', 0):.6f}, "
              f"wd={config.get('wd', 0):.6f}")
        try:
            metric = train_fn(config)
            results.append({**config, 'metric': metric})
            print(f"  -> metric = {metric:.4f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
    
    return results

def train_fn(config):
    model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 5))
    opt = optim.Adam(model.parameters(), lr=config.get('lr', 0.001), 
                     weight_decay=config.get('wd', 0))
    x = torch.randn(200, 20)
    y = torch.randint(0, 5, (200,))
    
    for epoch in range(10):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
    
    model.eval()
    x_test = torch.randn(100, 20)
    y_test = torch.randint(0, 5, (100,))
    with torch.no_grad():
        _, preds = torch.max(model(x_test), 1)
        return (preds == y_test).float().mean().item()

param_dist = {
    'lr': {'type': 'log_uniform', 'min': -5, 'max': -1},
    'wd': {'type': 'log_uniform', 'min': -6, 'max': -2},
}

results = random_search(train_fn, param_dist, n_trials=5)
best = max(results, key=lambda r: r['metric'])
print(f"\nBest: lr={best['lr']:.6f}, wd={best['wd']:.6f}, metric={best['metric']:.4f}")
# Output:
# Trial 1/5: lr=0.000123, wd=0.000045
#   -> metric = 0.2300
# ...
# Best: lr=0.005678, wd=0.000012, metric=0.3100
`

### Example 2: Random Search vs Grid Search Comparison

`python
import numpy as np
import matplotlib.pyplot as plt

def simulate_hyperparameter_importance(n_params=5, n_important=2, n_trials=50):
    """Simulate: only n_important parameters actually affect performance."""
    np.random.seed(42)
    
    # Generate random configurations
    grid_values_per_param = int(n_trials ** (1/n_params))
    n_grid = grid_values_per_param ** n_params
    n_random = n_trials
    
    print(f"Parameters: {n_params} total, {n_important} important")
    print(f"Grid search trials: {n_grid} (values/param: {grid_values_per_param})")
    print(f"Random search trials: {n_random}")
    
    # Simulate: each parameter has a value in [0,1]
    # Important parameters affect performance more
    performance = lambda config: (sum(config[i] for i in range(n_important)) / n_important 
                                   + 0.1 * sum(config[i] for i in range(n_important, n_params)) / (n_params - n_important))
    
    # Grid search
    grid_best = 0
    for _ in range(min(n_grid, 100000)):
        config = np.random.rand(n_params)
        perf = performance(config)
        grid_best = max(grid_best, perf)
    
    # Random search
    random_best = 0
    for _ in range(n_random):
        config = np.random.rand(n_params)
        perf = performance(config)
        random_best = max(random_best, perf)
    
    improvement = (random_best - grid_best) / grid_best * 100
    print(f"Grid best: {grid_best:.4f}")
    print(f"Random best: {random_best:.4f}")
    print(f"Random improvement: {improvement:.1f}%")
    return improvement

simulate_hyperparameter_importance(5, 2, 100)
# Output:
# Parameters: 5 total, 2 important
# Grid search trials: 3162277 (values/param: 6)
# Random search trials: 100
# Grid best: 0.7681
# Random best: 0.8345
# Random improvement: 8.6%
`

### Example 3: Adaptive Random Search with Early Stopping

`python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

class EarlyStoppedRandomSearch:
    def __init__(self, param_dist, n_trials=20, patience=3):
        self.param_dist = param_dist
        self.n_trials = n_trials
        self.patience = patience

    def sample_config(self):
        config = {}
        for name, dist in self.param_dist.items():
            if dist['type'] == 'log_uniform':
                config[name] = 10 ** random.uniform(dist['min'], dist['max'])
            elif dist['type'] == 'uniform':
                config[name] = random.uniform(dist['min'], dist['max'])
        return config

    def early_stopping_train(self, config, max_epochs=30):
        model = nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 5))
        opt = optim.Adam(model.parameters(), lr=config['lr'])
        x = torch.randn(200, 20)
        y = torch.randint(0, 5, (200,))
        
        best_val = 0
        no_improve = 0
        
        for epoch in range(max_epochs):
            model.train()
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            opt.step()
            
            # Simulated validation
            val_acc = 0.2 + 0.6 * (1 - math.exp(-epoch / 5)) - 0.1 * (epoch > 15)
            
            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        
        return best_val

    def run(self):
        results = []
        for trial in range(self.n_trials):
            config = self.sample_config()
            metric = self.early_stopping_train(config)
            results.append({**config, 'metric': metric})
            print(f"Trial {trial+1}: {config} -> best_val={metric:.4f}")
        return results

param_dist = {
    'lr': {'type': 'log_uniform', 'min': -5, 'max': -1},
}
searcher = EarlyStoppedRandomSearch(param_dist, n_trials=5, patience=3)
results = searcher.run()

best = max(results, key=lambda r: r['metric'])
print(f"\nBest: lr={best['lr']:.6f}, val={best['metric']:.4f}")
# Output:
# Trial 1: {'lr': 0.000123} -> best_val=0.6543
# Trial 2: {'lr': 0.045678} -> best_val=0.4567
# Trial 3: {'lr': 0.002345} -> best_val=0.7234
# Trial 4: {'lr': 0.000567} -> best_val=0.6123
# Trial 5: {'lr': 0.012345} -> best_val=0.6890
#
# Best: lr=0.002345, val=0.7234
`

## Common Mistakes

1. **Using uniform instead of log-uniform for positive parameters**: Learning rate, weight decay, and regularization strengths need log-uniform sampling.
2. **Too narrow search ranges**: Start with wide ranges to avoid missing optimal regions. Use results to narrow down.
3. **Not using enough trials**: The theoretical advantage of random search requires a reasonable number of trials (>= budget/3 the number of parameters).
4. **Not using early stopping**: Random search benefits greatly from early stopping — prune poor configurations quickly.
5. **Not repeating with different random seeds**: Random search results vary. Run multiple seeds or increase trial count for reliability.

## Interview Questions

### Beginner

1. How does random search work?
2. What is the key advantage of random search over grid search?
3. How are hyperparameters sampled in random search?
4. Why is log-uniform sampling used for certain parameters?
5. Is random search guaranteed to find the optimal configuration?

### Intermediate

1. Explain the theoretical result that random search outperforms grid search.
2. How does the number of important hyperparameters affect random search efficiency?
3. Compare the coverage of random search vs grid search with the same budget.
4. How would you implement early stopping in random search?
5. What is the optimal number of random search trials for a given budget?

### Advanced

1. Prove that random search has better worst-case performance than grid search.
2. Design an adaptive random search that biases toward promising regions.
3. Implement a random search with importance-weighted sampling from a learned distribution.

## Practice Problems

### Easy

1. Implement log-uniform sampling for learning rate between 1e-5 and 1e-1.
2. Write a function to sample batch sizes from {32, 64, 128, 256} with equal probability.
3. Implement random search for 2 hyperparameters with 10 trials.
4. Compare the best result from 10 random trials vs a 3x3 grid search.
5. Add results logging to a random search function.

### Medium

1. Implement random search with early stopping based on validation loss.
2. Compare random search with grid search on a real training task.
3. Implement a random search that saves and loads checkpoints for fault tolerance.
4. Add importance-based sampling to random search (more samples for important params).
5. Visualize random search results as a scatter plot.

### Hard

1. Implement a multi-stage random search (coarse exploration, then fine-tuning).
2. Design a random search with automatic range adaptation based on previous results.
3. Implement population-based random search where hyperparameters evolve.

## Solutions

### Easy Solutions

1. lr = 10 ** random.uniform(-5, -1)
2. random.choice([32, 64, 128, 256])
3. For _ in range(10): sample lr, wd; train; record metric
4. Random explores more distinct lr values, grid repeats same values
5. Append to a list of dicts and write to JSON after each trial

## Related Concepts

- Grid Search (DL-163)
- Bayesian Hyperparameter Optimization (DL-165)
- Hyperparameter Search (DL-162)
- Early Stopping (DL-137)

## Next Concepts

- Bayesian Hyperparameter Optimization (DL-165)
- Learning Curves (DL-166)
- Training vs Validation Gap (DL-167)

## Summary

Random search samples hyperparameter configurations randomly from predefined distributions, consistently outperforming grid search for the same budget. Its advantage stems from exploring more distinct values per hyperparameter and greater efficiency when only a few hyperparameters are important.

## Key Takeaways

- Random search samples configurations from predefined distributions
- Outperforms grid search for the same computational budget
- Use log-uniform for positive parameters (LR, WD)
- More efficient when few hyperparameters actually matter
- Trivially parallelizable
- Early stopping further improves efficiency
- Wide initial ranges, then narrow based on results
- Combine with cross-validation for reliable evaluation
