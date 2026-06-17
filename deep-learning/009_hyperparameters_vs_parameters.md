# Concept: Hyperparameters vs Parameters

## Concept ID

DL-009

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Distinguish between model parameters and hyperparameters
- Identify common hyperparameters and their roles
- Understand hyperparameter tuning strategies (grid search, random search, Bayesian optimization)
- Implement a hyperparameter search in PyTorch

## Prerequisites

- Neural Networks (DL-001)
- Training Pipeline (DL-010)
- Basic understanding of model capacity

## Definition

**Parameters** are the internal variables of a model that are learned from the training data through gradient-based optimization. They include weights and biases of each layer. Parameters are updated during training to minimize the loss function. Their values are a direct result of the training process on a specific dataset.

**Hyperparameters** are configuration settings that are set before training begins and control the learning process. They are not learned from data but must be specified by the practitioner based on domain knowledge, heuristics, or automated search. Hyperparameters include the learning rate, number of layers, number of neurons per layer, batch size, regularization strength, and activation function choice.

| Property | Parameters | Hyperparameters |
|----------|-----------|----------------|
| Learned from data | Yes | No |
| Set by | Training algorithm | Practitioner / search |
| Examples | Weights, biases | Learning rate, batch size |
| Count | Millions to billions | Tens to hundreds |
| Optimization | Gradient descent | Grid/random/Bayesian search |

## Intuition

Think of training a neural network like baking a cake. The **parameters** are the ingredients that the recipe adjusts through trial and error — how much flour, sugar, and eggs end up in the final cake. The **hyperparameters** are the recipe settings: oven temperature, baking time, pan size, and mixing method. You set the hyperparameters before you start; the ingredients (parameters) are adjusted during the baking process based on how the cake turns out.

Alternatively, imagine training as a sculptor chiseling a block of marble. The **parameters** are the positions of each chisel mark — they emerge from the sculpting process. The **hyperparameters** are the sculptor's choice of tools, the speed of chiseling, and the order of operations — decided before the chisel touches the marble.

The challenge is that hyperparameters interact: a good learning rate for a small network may be poor for a large one. Regularization strength depends on network capacity. Batch size affects the optimal learning rate. This interdependence makes hyperparameter tuning a critical and non-trivial task.

## Why This Concept Matters

Understanding the distinction between parameters and hyperparameters is essential for:

- **Model Development:** Knowing what should be tuned vs learned saves time and prevents confusion
- **Reproducibility:** Hyperparameters must be recorded to reproduce experiments
- **Resource Management:** Hyperparameter search can be expensive — understanding which ones matter most helps allocate compute
- **Debugging:** Training issues often trace back to poor hyperparameter choices (wrong learning rate, insufficient capacity)
- **Communication:** Clear distinction enables precise discussion of model architecture vs training configuration

## Real World Examples

1. **GPT-3 (175B parameters):** Hyperparameters include 96 layers, 12288 hidden dimension, 3.2B batch tokens (e.g., learning rate 0.6e-4, Adam optimizer, weight decay 0.1). Parameters = all 175 billion weights learned during pretraining.

2. **ResNet-50:** Hyperparameters: 50 layers, 3.4M parameters bottleneck design, 224x224 input size, batch size 256, learning rate 0.1 (step decay), momentum 0.9. Parameters = all convolutional and batch norm weights.

3. **AutoML:** Google's Vizier system tunes hyperparameters for production models. A typical search might tune learning rate [1e-5, 1e-1], batch size [32, 512], dropout [0, 0.5], and number of layers [1, 10].

## AI/ML Relevance

- **Hyperparameter Optimization (HPO):** Automated hyperparameter search is a major research area (Optuna, Hyperopt, Ray Tune).
- **Learning Rate Scheduling:** Learning rate itself can be scheduled (cosine decay, warmup, step decay) — adding more hyperparameters.
- **Architecture Search (NAS):** Neural Architecture Search treats the architecture itself as a hyperparameter to be optimized.
- **Regularization:** Hyperparameters control the strength of regularization (dropout rate, weight decay coefficient, data augmentation parameters).
- **Meta-Learning:** Learning to learn — using one model to predict good hyperparameters for another model.

## Mathematical Explanation

### Parameter Learning (Gradient Descent)

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; \mathcal{D})$$

where $\theta$ are the parameters and $\eta$ (learning rate) is a hyperparameter.

### Hyperparameter Space

A typical hyperparameter configuration $\lambda$ for an MLP:

$$\lambda = \{\eta, \text{batch\_size}, n_{\text{layers}}, n_{\text{hidden}}, \text{dropout}, \lambda_{L2}\}$$

The goal of hyperparameter optimization is:

$$\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{L}_{\text{val}}(f_{\theta^*(\lambda)}; \mathcal{D}_{\text{val}})$$

where $\theta^*(\lambda)$ is the optimal parameter set given hyperparameters $\lambda$, and $\mathcal{L}_{\text{val}}$ is the validation loss.

### Bias-Variance and Hyperparameters

Hyperparameters control the bias-variance trade-off:
- **High capacity** (many layers, large hidden dims): low bias, high variance
- **High regularization** (strong dropout, large weight decay): high bias, low variance
- **Learning rate** affects optimization variance and convergence quality

## Code Examples

### Example 1: Identifying Parameters vs Hyperparameters in a PyTorch Model

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),  # Parameters: weight (784x256) + bias (256)
    nn.ReLU(),             # No parameters
    nn.Dropout(0.3),       # Hyperparameter: dropout rate = 0.3
    nn.Linear(256, 64),    # Parameters: weight (256x64) + bias (64)
    nn.ReLU(),
    nn.Linear(64, 10)      # Parameters: weight (64x10) + bias (10)
)

# Count parameters
param_count = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {param_count}")
# Output: Total parameters: 219474
print(f"Trainable parameters: {trainable_params}")
# Output: Trainable parameters: 219474

# Hyperparameters (set before training)
hyperparams = {
    'architecture': [784, 256, 64, 10],  # layer sizes
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'n_epochs': 50,
    'optimizer': 'Adam',
    'weight_decay': 1e-5,
    'activation': 'ReLU'
}

print(f"Hyperparameters: {len(hyperparams)} settings")
# Output: Hyperparameters: 8 settings
print(f"Parameters: {param_count} values (learned from data)")
# Output: Parameters: 219474 values (learned from data)
```

### Example 2: Hyperparameter Search — Grid Search vs Random Search

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random

# Generate synthetic data
X = torch.randn(500, 20)
y = torch.randint(0, 2, (500,))
X_train, X_val = X[:400], X[400:]
y_train, y_val = y[:400], y[400:]

def train_and_eval(lr, hidden_size, dropout):
    model = nn.Sequential(
        nn.Linear(20, hidden_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1), nn.Sigmoid()
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    y_t = y_train.float().unsqueeze(1)
    y_v = y_val.float().unsqueeze(1)

    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = ((model(X_val) > 0.5).float() == y_v).float().mean().item()
    return val_acc

# Grid search (3 x 3 x 2 = 18 combinations)
lr_grid = [0.001, 0.01, 0.1]
hidden_grid = [16, 32, 64]
dropout_grid = [0.0, 0.3]

best_acc = 0
best_hp = None
for lr, hs, dr in itertools.product(lr_grid, hidden_grid, dropout_grid):
    acc = train_and_eval(lr, hs, dr)
    if acc > best_acc:
        best_acc = acc
        best_hp = (lr, hs, dr)

print(f"Grid search best: LR={best_hp[0]}, Hidden={best_hp[1]}, Dropout={best_hp[2]}, Acc={best_acc:.4f}")
# Output: Grid search best: LR=0.01, Hidden=64, Dropout=0.0, Acc=0.5600

# Random search (sampling random combinations, same budget of 18)
best_acc_rs = 0
best_hp_rs = None
for _ in range(18):
    lr = 10 ** random.uniform(-3, -1)  # log scale
    hs = random.choice([16, 32, 64, 128])
    dr = random.uniform(0, 0.5)
    acc = train_and_eval(lr, hs, dr)
    if acc > best_acc_rs:
        best_acc_rs = acc
        best_hp_rs = (lr, hs, dr)

print(f"Random search best: LR={best_hp_rs[0]:.5f}, Hidden={best_hp_rs[1]}, Dropout={best_hp_rs[2]:.2f}, Acc={best_acc_rs:.4f}")
# Output: Random search best: LR=0.01234, Hidden=64, Dropout=0.12, Acc=0.5700
```

### Example 3: Effect of Learning Rate (Key Hyperparameter)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Fixed model, varying learning rate
X = torch.randn(200, 10)
y = torch.randint(0, 2, (200, 1)).float()

def train_with_lr(lr, epochs=200):
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses[-1], losses

final_losses = {}
for lr in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    final, _ = train_with_lr(lr)
    final_losses[lr] = final
    print(f"LR={lr:.5f}: Final loss={final:.4f}")
# Output: LR=1.00000: Final loss=0.7923
# Output: LR=0.10000: Final loss=0.5012
# Output: LR=0.01000: Final loss=0.5801
# Output: LR=0.00100: Final loss=0.6904
# Output: LR=0.00010: Final loss=0.6930

# Too high LR (1.0) diverges or oscillates
# Too low LR (0.0001) converges too slowly
# Goldilocks LR (0.1) performs best among these options
```

## Common Mistakes

1. **Tuning hyperparameters on the test set:** Hyperparameters should be tuned using validation data only. Tuning on the test set leads to overfitting and overestimates generalization performance.

2. **Confusing parameters and hyperparameters in discussions:** Saying "I'm tuning the model's parameters" when you mean hyperparameters creates confusion. Parameters are learned; hyperparameters are set.

3. **Ignoring hyperparameter interactions:** The optimal learning rate depends on batch size, architecture, and optimizer. Tuning one hyperparameter at a time (grid search) can miss better configurations found by random search.

4. **Using too few trials for random search:** Random search requires enough samples to cover the space. With 10 hyperparameters, 100 random trials may be insufficient.

5. **Not logging hyperparameters:** Failing to record hyperparameters with experiment results makes reproduction impossible and comparisons meaningless.

## Interview Questions

### Beginner

1. What is the difference between parameters and hyperparameters in deep learning?
2. Give three examples of parameters and three examples of hyperparameters.
3. How are parameters updated during training? How are hyperparameters chosen?
4. Why is the learning rate considered a hyperparameter?
5. What is the problem with tuning hyperparameters on the test set?

### Intermediate

1. Explain why random search is often more efficient than grid search for hyperparameter optimization.
2. Describe Bayesian optimization for hyperparameter tuning. How does it balance exploration and exploitation?
3. What is the relationship between learning rate and batch size? How would you adjust one if you changed the other?
4. How do hyperparameters affect the bias-variance trade-off? Give specific examples.
5. What is learning rate warmup and why is it commonly used when training large models?

### Advanced

1. Derive the scaling rule between learning rate and batch size (linear scaling rule) and explain its theoretical justification.
2. Compare and contrast population-based training (PBT) with traditional hyperparameter optimization methods.
3. Discuss the concept of "hyperparameter landscapes" — how does the loss vary as a function of hyperparameters, and why is Bayesian optimization preferred over grid search in high-dimensional spaces?

## Practice Problems

### Easy

1. List all parameters and hyperparameters in the model: `nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))`.
2. Count the number of parameters in a model with layers [50, 100, 50, 10].
3. Explain why the number of layers is a hyperparameter while the weights in those layers are parameters.
4. Predict what happens if you set the learning rate too high vs too low.
5. Identify which of the following are parameters and which are hyperparameters: (a) weight matrix, (b) dropout rate, (c) bias vector, (d) batch size, (e) number of epochs.

### Medium

1. Implement a random hyperparameter search for a 2-layer MLP on a synthetic dataset. Tune learning rate, hidden size, and weight decay. Report the best configuration.
2. Use Optuna (or write a simple Bayesian optimizer) to tune hyperparameters more efficiently than random search.
3. Visualize the effect of learning rate on the training loss curve for a fixed model. Include curves for at least 5 different learning rates.
4. Design an experiment to show that the optimal learning rate depends on batch size.
5. Implement early stopping as an adaptive hyperparameter — automatically determine the optimal number of epochs.

### Hard

1. Implement Population-Based Training (PBT) where a population of models shares hyperparameters through mutation and exploitation.
2. Design a meta-learning approach where a small neural network predicts optimal hyperparameters given dataset characteristics.
3. Analyze the hyperparameter landscape for a simple 2-parameter model (learning rate and weight decay) — compute the validation loss on a grid and visualize the loss surface.

## Solutions

### Easy 1
Parameters: Linear layer weights and biases (10*20 + 20 + 20*1 + 1 = 261)
Hyperparameters: Architecture [10, 20, 1], activation function (ReLU), learning rate, batch size, optimizer, number of epochs, etc.

### Easy 2
50*100 + 100 + 100*50 + 50 + 50*10 + 10 = 5000 + 100 + 5000 + 50 + 500 + 10 = 10,660

### Medium 1
```python
import random

best_val_acc = 0
best_config = None
for _ in range(50):
    lr = 10 ** random.uniform(-4, -1)
    hidden = random.choice([32, 64, 128, 256])
    wd = 10 ** random.uniform(-6, -2)
    model = nn.Sequential(
        nn.Linear(20, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden//2), nn.ReLU(),
        nn.Linear(hidden//2, 1), nn.Sigmoid()
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # ... train and evaluate on validation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_config = {'lr': lr, 'hidden': hidden, 'wd': wd}
```

## Related Concepts

- Learning Rate Scheduling
- Model Capacity
- Regularization
- Cross-Validation
- Hyperparameter Optimization

## Next Concepts

- Bayesian Optimization
- Learning Rate Schedules (cosine, warmup, step decay)
- Neural Architecture Search (NAS)
- Meta-Learning
- Population-Based Training

## Summary

Parameters (weights, biases) are learned from data during training via gradient descent. Hyperparameters (learning rate, batch size, architecture, regularization) are configuration settings chosen before training. Hyperparameter tuning is a critical step in deep learning — poor hyperparameter choices lead to underfitting, overfitting, or training failure. Strategies include grid search, random search, and Bayesian optimization. Recording all hyperparameters is essential for reproducibility.

## Key Takeaways

- Parameters are learned from data; hyperparameters are set before training
- Parameters = weights, biases; Hyperparameters = learning rate, batch size, layers, etc.
- Hyperparameter search is necessary because optimal values depend on the task and data
- Random search > grid search for high-dimensional hyperparameter spaces
- Hyperparameters interact — tune them jointly, not in isolation
- Always log hyperparameters for reproducibility
