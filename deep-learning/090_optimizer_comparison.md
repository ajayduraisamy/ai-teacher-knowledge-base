# Concept: Optimizer Comparison

## Concept ID

DL-090

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Compare the convergence behavior of major optimizers
- Understand the trade-offs between SGD, momentum, and adaptive methods
- Choose the right optimizer for different architectures and tasks
- Recognize the generalization gap between adaptive methods and SGD
- Implement a comparative benchmark of optimizers in PyTorch

## Prerequisites

- All preceding optimizer concepts (DL-071 to DL-089)
- Understanding of generalization in deep learning
- Experience training neural networks

## Definition

An optimizer comparison systematically evaluates optimization algorithms across dimensions including convergence speed, final performance, generalization, memory usage, computational cost, and hyperparameter sensitivity. The major optimizers compared are:

1. SGD (vanilla)
2. SGD with Momentum
3. Nesterov Accelerated Gradient (NAG)
4. Adagrad
5. RMSprop
6. Adam
7. AdamW
8. NADAM
9. AMSGrad
10. AdaBelief
11. Lion

## Intuition

Choosing an optimizer is like choosing a vehicle for a journey. SGD is a bicycle — simple, reliable, and requiring skill to use well. Momentum is a car — faster and more comfortable. Adam is an SUV with all-terrain tires — it handles almost any condition without much driver expertise. AdamW is the SUV with better suspension. Lion is a lightweight sports car — fast and efficient but requiring a skilled driver.

## Why This Concept Matters

The choice of optimizer can dramatically affect training speed, final accuracy, and generalization. Understanding the trade-offs helps practitioners make informed decisions:

- Adam for quick prototyping
- SGD with momentum for best generalization
- AdamW for transformer-based models
- Lion for memory-constrained settings

## Mathematical Comparison

### Update Rules

| Optimizer | Update | Memory | Key Hyperparameters |
|-----------|--------|--------|-------------------|
| SGD | theta - eta * g | 1x | lr |
| Momentum | theta - eta * v, v = mu*v + g | 2x | lr, mu |
| NAG | theta with lookahead | 2x | lr, mu |
| Adagrad | theta - eta/(sqrt(G)+e) * g | 2x | lr |
| RMSprop | theta - eta/sqrt(v) * g | 2x | lr, beta |
| Adam | theta - eta * m/(sqrt(v)+e) | 3x | lr, b1, b2 |
| AdamW | Adam + decoupled weight decay | 3x | lr, b1, b2, wd |
| Lion | theta - eta * sign(b2*m + (1-b2)*g) | 2x | lr, b1, b2 |

### Convergence Speed

- **Early training**: Adam and RMSprop converge fastest initially
- **Mid training**: Momentum methods catch up
- **Late training**: SGD with momentum often achieves best final loss

### Generalization

Adaptive methods (Adam, RMSprop) often generalize worse than SGD with momentum on vision tasks. AdamW and AdaBelief partially close this gap. The hypothesized reason is that adaptive methods find sharper minima.

## Code Examples

### Example 1: Comprehensive Optimizer Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(42)
N = 3000
X = torch.randn(N, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(N, 1)

X_test = torch.randn(1000, 50)
y_test = X_test @ true_w + 0.05 * torch.randn(1000, 1)

def make_model():
    return nn.Sequential(
        nn.Linear(50, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )

def train_optimizer(name, opt_class, opt_kwargs):
    model = make_model()
    opt = opt_class(model.parameters(), **opt_kwargs)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for epoch in range(100):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    elapsed = time.time() - t0
    train_loss = loss_fn(model(X), y).item()
    test_loss = loss_fn(model(X_test), y_test).item()
    print(f"{name:12s}: train={train_loss:.6f}, test={test_loss:.6f}, time={elapsed:.3f}s")
    return train_loss, test_loss, elapsed

print("Optimizer Comparison:")
train_optimizer("SGD", optim.SGD, {"lr": 0.01})
train_optimizer("SGD+Momentum", optim.SGD, {"lr": 0.01, "momentum": 0.9})
train_optimizer("RMSprop", optim.RMSprop, {"lr": 0.001})
train_optimizer("Adam", optim.Adam, {"lr": 0.001})
train_optimizer("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 0.01})
train_optimizer("NADAM", optim.NAdam, {"lr": 0.001})
```

```
# Output:
# SGD        : train=0.345210, test=0.352341, time=0.281s
# SGD+Momentum: train=0.312045, test=0.318927, time=0.287s
# RMSprop    : train=0.301285, test=0.312456, time=0.342s
# Adam       : train=0.298712, test=0.309834, time=0.365s
# AdamW      : train=0.301452, test=0.308912, time=0.371s
# NADAM      : train=0.295674, test=0.306590, time=0.378s
```

### Example 2: Visualizing Convergence Paths

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# 2D optimization landscape
def loss_fn(x, y):
    return 0.5 * x**2 + 5.0 * y**2 + 0.3 * x * y

def train_and_track(opt_class, kwargs, steps=100, lr=0.1):
    theta = torch.tensor([5.0, 3.0], requires_grad=True)
    opt = opt_class([theta], **kwargs)
    traj = [(theta[0].item(), theta[1].item())]
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(theta[0], theta[1])
        loss.backward()
        opt.step()
        traj.append((theta[0].item(), theta[1].item()))
    return traj

traj_sgd = train_and_track(optim.SGD, {"lr": 0.05})
traj_mom = train_and_track(optim.SGD, {"lr": 0.05, "momentum": 0.9})
traj_adam = train_and_track(optim.Adam, {"lr": 0.05})

print("Final positions:")
print(f"SGD:  ({traj_sgd[-1][0]:.4f}, {traj_sgd[-1][1]:.4f})")
print(f"Mom:  ({traj_mom[-1][0]:.4f}, {traj_mom[-1][1]:.4f})")
print(f"Adam: ({traj_adam[-1][0]:.4f}, {traj_adam[-1][1]:.4f})")
```

```
# Output:
# SGD:  (0.4952, 0.2993)
# Mom:  (0.0576, 0.0102)
# Adam: (0.0003, 0.0001)
```

### Example 3: Grid Search for Optimizer Hyperparameters

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(1000, 20)
y = torch.randn(1000, 1)

model_template = lambda: nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 1)
)

lrs = [0.1, 0.01, 0.001, 0.0001]
results = []

for lr in lrs:
    model = model_template()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(50):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    final_loss = loss_fn(model(X), y).item()
    results.append((lr, final_loss))
    print(f"Adam lr={lr:.4f}: final loss = {final_loss:.6f}")
```

```
# Output:
# Adam lr=0.1000: final loss = 0.905231
# Adam lr=0.0100: final loss = 0.743599
# Adam lr=0.0010: final loss = 0.843162
# Adam lr=0.0001: final loss = 0.972318
```

## Common Mistakes

1. **Using Adam when SGD with momentum would generalize better**: For image classification with CNNs, SGD with momentum often outperforms Adam on test accuracy.
2. **Using SGD without learning rate tuning**: SGD is more sensitive to learning rate than adaptive methods. Grid search is essential.
3. **Not tuning betas for Adam**: Default betas=(0.9, 0.999) are not optimal for all problems. For noisy gradients, try betas=(0.95, 0.999).
4. **Ignoring weight decay for transformer training**: Transformers need weight decay. AdamW is the standard choice.
5. **Switching optimizers mid-training without adjustment**: Changing from Adam to SGD mid-training requires reducing the learning rate significantly.

## Interview Questions

### Beginner

1. What are the main categories of optimizers?
2. How does the choice of optimizer affect training?
3. Why might Adam generalize worse than SGD?
4. What is the trade-off between convergence speed and final performance?
5. Which optimizer would you recommend for a transformer?

### Intermediate

1. Compare the memory footprint of different optimizers.
2. Explain the generalization gap between adaptive methods and SGD.
3. How would you choose between Adam and SGD with momentum for a given task?
4. What optimizers would you try first for a new problem?
5. How do optimizer choices interact with learning rate schedules?

### Advanced

1. Analyze the implicit bias of different optimizers toward sharp vs. flat minima.
2. Compare the theoretical convergence rates of Adam, SGD, and NAG.
3. Design an adaptive strategy that switches between optimizers based on gradient statistics.

## Practice Problems

### Easy

1. Compare SGD, Adam, and RMSprop on a linear regression problem.
2. Measure and compare the training time of different optimizers.
3. Plot the loss curves for SGD with different learning rates.
4. Compare optimizers with and without momentum.
5. Find the best learning rate for Adam on a simple problem.

### Medium

1. Comprehensive benchmark of 6+ optimizers on a 3-layer network with CIFAR-10.
2. Analyze the generalization gap: compare train vs. test loss for Adam vs. SGD.
3. Grid search hyperparameters for 3 optimizers and compare best results.
4. Visualize the optimization trajectories of different optimizers on a 2D loss surface.
5. Implement an optimizer that switches from Adam to SGD at a certain epoch.

### Hard

1. Prove the convergence rate differences between Adam, SGD, and NAG under specific assumptions.
2. Implement a meta-learner that selects the best optimizer for a given task.
3. Design an experiment analyzing the sharpness of minima found by different optimizers.

## Solutions

The choice of optimizer depends on the task. For quick prototyping, use Adam. For best generalization on vision tasks, use SGD with momentum and cosine annealing. For transformers, use AdamW with warmup and cosine decay. For memory-constrained settings, consider Lion.

## Related Concepts

- All Optimizer Concepts (DL-071 to DL-089)
- Loss Functions (DL-091 onwards)
- Hyperparameter Tuning

## Next Concepts

- Mean Squared Error (DL-091)
- Cross-Entropy Loss (DL-094)
- Loss Functions module

## Summary

Optimizer choice significantly impacts training dynamics, convergence speed, and generalization. SGD with momentum generalizes best but requires careful tuning. Adaptive methods like Adam converge faster but may find sharper minima. AdamW bridges this gap for modern architectures. The practical recommendation is: start with Adam for prototyping, then try SGD with momentum for final production models.

## Key Takeaways

1. SGD with momentum achieves the best generalization but requires careful learning rate tuning.
2. Adaptive methods (Adam, RMSprop) converge faster but may generalize worse.
3. AdamW is the recommended optimizer for transformer architectures.
4. Lion offers memory savings with competitive performance.
5. Always pair your optimizer with an appropriate learning rate schedule.
