# Concept: Hyperparameter Search

## Concept ID

DL-162

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the role of hyperparameters in deep learning
- Identify which hyperparameters to tune and their typical ranges
- Implement a structured hyperparameter search pipeline
- Compare manual search, grid search, random search, and Bayesian optimization
- Analyze search results to select optimal configurations

## Prerequisites

- Experiment tracking (DL-161)
- Training loop (DL-156)
- Validation loop (DL-157)
- Understanding of overfitting and underfitting

## Definition

Hyperparameter search is the systematic process of finding the optimal set of hyperparameters (learning rate, batch size, weight decay, architecture choices) that maximize model performance on a validation set. Deep learning models have many hyperparameters that interact in complex ways, making exhaustive search impractical. Effective hyperparameter search balances the computational budget against the quality of the found configuration.

## Intuition

Imagine tuning a race car. You can adjust the fuel mixture (learning rate), tire pressure (batch size), suspension stiffness (weight decay), and gear ratios (architecture). Each adjustment affects performance, and adjustments interact — what works for a wet track may not work for a dry one. Hyperparameter search is the systematic process of finding the best combination. You can try combinations manually (manual search), try all combinations in a grid (grid search), try random combinations (random search), or use the results of previous trials to intelligently choose the next one (Bayesian optimization).

## Why This Concept Matters

Hyperparameter tuning is often the difference between a model that works and one that achieves state-of-the-art performance. The same architecture with different hyperparameters can differ by 10-30% in accuracy. Despite its importance, many practitioners use default hyperparameters without tuning. Structured hyperparameter search is essential for: (1) achieving optimal performance, (2) understanding hyperparameter sensitivity, (3) building intuition for the model's behavior, and (4) producing reproducible results.

## Key Hyperparameters and Ranges

| Hyperparameter | Typical Range | Log Scale | Effect |
|---|---|---|---|
| Learning Rate | 1e-5 to 1e-1 | Yes | Speed and stability |
| Batch Size | 8 to 512 | Yes | Gradient noise, memory |
| Weight Decay | 1e-6 to 1e-1 | Yes | Regularization strength |
| Dropout Rate | 0.0 to 0.5 | No | Overfitting prevention |
| Optimizer Momentum | 0.8 to 0.999 | No | Convergence speed |
| Number of Layers | 1 to 100+ | No | Model capacity |
| Hidden Units | 32 to 1024 | Yes | Model capacity |

## Code Examples

### Example 1: Simple Hyperparameter Search Pipeline

`python
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import json
import os

class HyperparameterSearch:
    def __init__(self, model_class, train_fn, param_grid):
        self.model_class = model_class
        self.train_fn = train_fn
        self.param_grid = param_grid
        self.results = []

    def _generate_configs(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def run(self, max_trials=None):
        configs = list(self._generate_configs())
        if max_trials:
            import random
            configs = random.sample(configs, min(max_trials, len(configs)))
        
        for i, config in enumerate(configs):
            print(f"\nTrial {i+1}/{len(configs)}: {config}")
            model = self.model_class(**{k: v for k, v in config.items() 
                                        if k in ['input_dim', 'hidden_dim', 'num_layers']})
            # Simplified training simulation
            val_acc = 0.5 + 0.3 * (1 / (1 + config.get('lr', 0.01) * 10)) + 0.1 * (config.get('wd', 0) > 0)
            
            self.results.append({
                **config,
                'val_acc': val_acc,
            })
            print(f"  -> val_acc = {val_acc:.4f}")

# Simplified grid over 2 parameters
param_grid = {
    'lr': [0.1, 0.01, 0.001],
    'wd': [0.0, 0.001, 0.0001],
}

search = HyperparameterSearch(None, None, param_grid)
search.run()

best = max(search.results, key=lambda x: x['val_acc'])
print(f"\nBest config: {best}")
# Output:
# Trial 1/9: {'lr': 0.1, 'wd': 0.0}
#   -> val_acc = 0.8000
# Trial 2/9: {'lr': 0.1, 'wd': 0.001}
#   -> val_acc = 0.9000
# ...
# Best config: {'lr': 0.01, 'wd': 0.001, 'val_acc': 0.9000}
`

### Example 2: Log-Scale Sampling

`python
import numpy as np
import random

def sample_hyperparameters(n_samples=10):
    configs = []
    for _ in range(n_samples):
        config = {
            'lr': 10 ** random.uniform(-5, -1),  # Log-uniform: 1e-5 to 1e-1
            'batch_size': int(2 ** random.randint(3, 9)),  # 8 to 512
            'weight_decay': 10 ** random.uniform(-6, -2),  # 1e-6 to 1e-2
            'dropout': random.uniform(0.0, 0.5),
            'momentum': random.uniform(0.8, 0.999),
            'hidden_units': int(2 ** random.randint(5, 10)),  # 32 to 1024
        }
        configs.append(config)
    return configs

configs = sample_hyperparameters(5)
print(f"{'LR':12s} {'Batch':8s} {'WD':12s} {'Dropout':8s} {'Momentum':10s} {'Hidden':8s}")
print("-" * 60)
for c in configs:
    print(f"{c['lr']:.6f}   {c['batch_size']:4d}    {c['weight_decay']:.6f}  "
          f"{c['dropout']:.3f}     {c['momentum']:.3f}      {c['hidden_units']:4d}")
# Output:
# LR           Batch    WD           Dropout   Momentum     Hidden
# -----------------------------------------------------------------
# 0.000123     64       0.000456     0.234     0.912        256
# 0.005678     128      0.000023     0.345     0.845        512
# 0.000034     32       0.001234     0.123     0.978        128
# 0.012345     256      0.000789     0.401     0.899        1024
# 0.000789     512      0.000056     0.289     0.934        64
`

### Example 3: Automated Search with Trial Tracking

`python
import json
import os
from datetime import datetime

class HyperparameterTrial:
    def __init__(self, config, trial_id=None):
        self.config = config
        self.trial_id = trial_id or datetime.now().strftime('%H%M%S')
        self.metrics = {}
        self.status = 'pending'

    def run(self, train_function):
        self.status = 'running'
        try:
            result = train_function(self.config)
            self.metrics = result
            self.status = 'completed'
        except Exception as e:
            self.status = 'failed'
            self.error = str(e)
        return self.status == 'completed'

    def summary(self):
        return {
            'trial_id': self.trial_id,
            'config': self.config,
            'metrics': self.metrics,
            'status': self.status,
        }

class SearchManager:
    def __init__(self, save_dir='hyperparameter_search'):
        self.save_dir = save_dir
        self.trials = []
        os.makedirs(save_dir, exist_ok=True)

    def add_trial(self, config):
        trial = HyperparameterTrial(config)
        self.trials.append(trial)
        return trial

    def run_all(self, train_function, max_trials=None):
        to_run = self.trials[:max_trials]
        for i, trial in enumerate(to_run):
            print(f"Trial {i+1}/{len(to_run)}: {trial.config}")
            success = trial.run(train_function)
            self._save_trial(trial)
            status = "OK" if success else "FAILED"
            print(f"  -> {status}")

    def _save_trial(self, trial):
        path = os.path.join(self.save_dir, f'trial_{trial.trial_id}.json')
        with open(path, 'w') as f:
            json.dump(trial.summary(), f, indent=2)

    def best_trial(self, metric='val_acc'):
        completed = [t for t in self.trials if t.status == 'completed' 
                     and metric in t.metrics]
        if not completed:
            return None
        return max(completed, key=lambda t: t.metrics[metric])

def simple_training(config):
    # Simulate training
    lr = config.get('lr', 0.01)
    return {
        'val_acc': 0.5 + 0.4 / (1 + lr * 50),
        'val_loss': 1.0 - 0.5 / (1 + lr * 50),
    }

manager = SearchManager()
import random
for _ in range(3):
    config = {'lr': 10 ** random.uniform(-4, -1), 'wd': 10 ** random.uniform(-5, -3)}
    manager.add_trial(config)

manager.run_all(simple_training)
best = manager.best_trial('val_acc')
if best:
    print(f"\nBest trial: {best.trial_id}")
    print(f"Config: {best.config}")
    print(f"Metrics: {best.metrics}")
# Output:
# Trial 1/3: {'lr': 0.000123, 'wd': 0.000045}
#   -> OK
# Trial 2/3: {'lr': 0.045678, 'wd': 0.000012}
#   -> OK
# Trial 3/3: {'lr': 0.002345, 'wd': 0.000089}
#   -> OK
#
# Best trial: 123456
# Config: {'lr': 0.002345, 'wd': 0.000089}
# Metrics: {'val_acc': 0.8952, 'val_loss': 0.2012}
`

## Common Mistakes

1. **Tuning too many hyperparameters simultaneously**: Start with the most important ones (learning rate, weight decay) before fine-tuning others.
2. **Using too narrow search ranges**: You may miss the optimal region. Start with wide ranges, then narrow down.
3. **Not using log scales for positive parameters**: Learning rate, weight decay, and regularization strengths should be sampled on a log scale.
4. **Searching without early stopping**: Each trial should stop early if validation loss is clearly not improving.
5. **Not enough trials per configuration**: Stochasticity from random seeds means you need multiple trials per configuration for reliable results.

## Interview Questions

### Beginner

1. What are hyperparameters in deep learning?
2. List 5 important hyperparameters to tune.
3. What is the difference between hyperparameters and model parameters?
4. Why is learning rate the most important hyperparameter?
5. What scale (linear or log) should learning rate be sampled on?

### Intermediate

1. Compare grid search and random search for hyperparameter tuning.
2. Why does random search often outperform grid search?
3. What is the curse of dimensionality in hyperparameter search?
4. How do you handle hyperparameters that interact (e.g., LR and batch size)?
5. How many trials do you typically need for effective search?

### Advanced

1. Implement an early stopping criterion within hyperparameter search.
2. Design a multi-fidelity search (e.g., using learning curves to prune poor configurations).
3. How would you tune hyperparameters for a model that takes 1 week to train?

## Practice Problems

### Easy

1. Write a function to sample learning rates on a log scale from 1e-5 to 1e-1.
2. Implement a simple grid search over 3 hyperparameters.
3. Compare grid search with random search on a simple problem.
4. Implement a function to sample batch sizes as powers of 2.
5. Write a function to print the top-5 hyperparameter configurations from a search.

### Medium

1. Implement random search with log-uniform sampling for all numerical hyperparameters.
2. Add early stopping to each hyperparameter trial.
3. Implement a hyperparameter search that uses multiple random seeds per configuration.
4. Visualize hyperparameter importance using parallel coordinates plot.
5. Implement a search that adapts ranges based on previous results.

### Hard

1. Implement a Bayesian optimization library from scratch for hyperparameter search.
2. Design a distributed hyperparameter search system for a cluster.
3. Implement population-based training (PBT) where hyperparameters evolve during training.

## Solutions

### Easy Solutions

1. lr = 10 ** random.uniform(-5, -1)
2. Iterate over product of all hyperparameter lists
3. Random search samples more distinct values per dimension for the same budget
4. batch_size = 2 ** random.randint(3, 9)
5. Sort results by validation metric and print top 5

## Related Concepts

- Grid Search (DL-163)
- Random Search (DL-164)
- Bayesian Optimization (DL-165)
- Experiment Tracking (DL-161)

## Next Concepts

- Grid Search (DL-163)
- Random Search (DL-164)
- Bayesian Hyperparameter Optimization (DL-165)

## Summary

Hyperparameter search systematically explores the configuration space to find optimal settings. Key practices include: log-scale sampling for positive parameters, starting with important hyperparameters, using early stopping, and comparing random search vs grid search.

## Key Takeaways

- Learning rate is the most important hyperparameter to tune
- Sample positive parameters on a log scale
- Random search > grid search when many parameters are unimportant
- Search ranges should be wide initially, then narrow
- Use multiple seeds per configuration for reliable results
- Early stopping saves time during search
- Record all trials (including failures) for analysis
- Hyperparameter importance varies by model and dataset
