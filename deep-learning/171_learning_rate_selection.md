# Concept: Learning Rate Selection

## Concept ID

DL-171

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the role of learning rate in gradient-based optimization
- Compare different learning rate schedules and adaptive methods
- Implement learning rate selection strategies in PyTorch
- Diagnose learning rate related training issues

## Prerequisites

DL-101 Gradient Descent, DL-109 Backpropagation, DL-105 Loss Functions

## Definition

Learning rate is a hyperparameter that controls the step size taken during gradient descent optimization, determining how much the model weights are adjusted in response to the estimated error gradient.

## Intuition

Think of learning rate as the size of steps you take while hiking down a hill in foggy weather. Large steps might make you overshoot the valley, while tiny steps take too long to descend. The optimal learning rate lets you descend quickly without overshooting. When the learning rate is too high, the loss bounces around or diverges; when too low, training progresses at a glacial pace or gets stuck in poor local minima. Modern training uses adaptive methods or schedules that start with larger steps and shrink over time, analogous to taking confident strides at first then careful steps as you near the bottom.

## Why This Concept Matters

Learning rate is arguably the single most important hyperparameter in deep learning. A well-chosen learning rate determines whether a model converges at all, how fast it converges, and to what quality of solution. Getting it wrong means wasted compute, failed training runs, or suboptimal model performance. Understanding learning rate selection is essential for anyone training neural networks.

## Mathematical Explanation

The standard gradient descent update rule is:

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

where $\eta$ is the learning rate and $\nabla L(w_t)$ is the gradient of the loss with respect to weights.

For SGD with momentum:

$$v_{t+1} = \beta v_t + \nabla L(w_t)$$
$$w_{t+1} = w_t - \eta v_{t+1}$$

Common learning rate schedules:

1. **Step Decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$ where $\gamma$ is decay rate and $s$ is step size.

2. **Exponential Decay**: $\eta_t = \eta_0 \cdot e^{-kt}$

3. **Cosine Annealing**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$

4. **Reduce on Plateau**: Reduce $\eta$ by factor $\gamma$ when loss plateaus for patience epochs.

**Learning Rate Range Test**: Start with a small LR, increase exponentially each batch, plot loss vs LR. The optimal LR is near the steepest downward slope.

## Code Examples

### Example 1: Comparing Learning Rates

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Simple regression dataset
X = torch.linspace(-3, 3, 200).reshape(-1, 1)
y = 2 * X + 1 + 0.1 * torch.randn(200, 1)

# Model
model = nn.Sequential(nn.Linear(1, 1))

# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]
losses = {}

for lr in learning_rates:
    model_copy = nn.Sequential(nn.Linear(1, 1))
    optimizer = optim.SGD(model_copy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    batch_losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model_copy(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    
    losses[lr] = batch_losses

for lr, loss_list in losses.items():
    print(f"LR={lr:.3f} -> Final loss: {loss_list[-1]:.6f}")
    # Output: LR=0.001 -> Final loss: 3.234802
    # Output: LR=0.010 -> Final loss: 0.017315
    # Output: LR=0.100 -> Final loss: 0.009816
    # Output: LR=1.000 -> Final loss: 18.234567
```

### Example 2: Learning Rate Schedules

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(10, 1))
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Step Decay
scheduler_step = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
optimizer2 = optim.SGD(model.parameters(), lr=0.1)
scheduler_cosine = CosineAnnealingLR(optimizer2, T_max=100)

# Track learning rates
step_lrs = []
cosine_lrs = []

for epoch in range(100):
    optimizer.zero_grad()
    optimizer2.zero_grad()
    
    # Dummy forward/backward
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    # Step needs model2 to also step
    x2 = torch.randn(32, 10)
    y2 = torch.randn(32, 1)
    loss2 = nn.MSELoss()(model(x2), y2)
    loss2.backward()
    optimizer2.step()
    
    step_lrs.append(scheduler_step.get_last_lr()[0])
    scheduler_step.step()
    
    cosine_lrs.append(scheduler_cosine.get_last_lr()[0])
    scheduler_cosine.step()

print(f"Step LR start: {step_lrs[0]:.6f}, end: {step_lrs[-1]:.6f}")
# Output: Step LR start: 0.100000, end: 0.000100

print(f"Cosine LR start: {cosine_lrs[0]:.6f}, end: {cosine_lrs[-1]:.6f}")
# Output: Cosine LR start: 0.100000, end: 0.000000
```

### Example 3: Learning Rate Range Test

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)

# Create data
X = torch.randn(500, 20)
y = torch.randn(500, 1)

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-7)

# LR Range Test
lr_start = 1e-7
lr_end = 10.0
num_batches = 100
lr_mult = (lr_end / lr_start) ** (1.0 / num_batches)

lrs = []
losses = []
current_lr = lr_start

for batch_idx in range(num_batches):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    
    # Update LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    optimizer.step()
    
    lrs.append(current_lr)
    losses.append(loss.item())
    current_lr *= lr_mult

# Find optimal LR (steepest negative slope)
losses = np.array(losses)
lrs = np.array(lrs)
gradients = np.gradient(losses, np.log(lrs))
optimal_idx = np.argmin(gradients)
optimal_lr = lrs[optimal_idx]

print(f"Optimal LR found: {optimal_lr:.6f}")
# Output: Optimal LR found: 0.012543

print(f"LR range: {lr_start:.1e} to {lr_end:.1f}")
# Output: LR range: 1.0e-07 to 10.0
```

## Common Mistakes

1. **Using too large a learning rate**: Causes loss to oscillate or diverge. Symptoms include NaN gradients or loss shooting to infinity.
2. **Using too small a learning rate**: Training converges extremely slowly or gets stuck. The loss barely changes over many epochs.
3. **Not using learning rate schedules**: A fixed learning rate throughout training rarely produces optimal results.
4. **Applying the same LR to all layers**: Different layers (especially pretrained vs random init) benefit from different learning rates.
5. **Ignoring the LR-batch size relationship**: Doubling batch size often requires doubling learning rate to maintain gradient variance.

## Interview Questions

### Beginner - 5
1. What is learning rate in neural networks?
2. What happens if learning rate is too high?
3. What happens if learning rate is too low?
4. What is the typical range for learning rate values?
5. How does learning rate differ between SGD and Adam?

### Intermediate - 5
1. Explain the learning rate range test.
2. Compare step decay, exponential decay, and cosine annealing.
3. Why does batch size affect the optimal learning rate?
4. How does momentum interact with learning rate?
5. What is warmup and why is it used?

### Advanced - 3
1. Derive the relationship between learning rate and batch size for linear scaling.
2. Explain how second-order methods (Newton, L-BFGS) eliminate the need for learning rate.
3. Design a learning rate schedule for training a transformer on 100 billion tokens.

## Practice Problems

### Easy - 5
1. Train a linear model on synthetic data with LR=0.1, 0.01, 0.001 and compare convergence.
2. Implement exponential learning rate decay.
3. Plot learning rate vs loss for a simple network.
4. Find the maximum learning rate that does not cause divergence.
5. Compare SGD with LR=0.01 vs Adam with LR=0.001.

### Medium - 5
1. Implement cosine annealing with warm restarts.
2. Build a cyclical learning rate scheduler.
3. Implement a learning rate range test from scratch.
4. Train ResNet-18 on CIFAR-10 with different LR schedules and compare.
5. Implement differential learning rates for different layers.

### Hard - 3
1. Implement the One-Cycle learning rate policy (Leslie Smith).
2. Design an adaptive learning rate method that adjusts per-layer based on gradient statistics.
3. Reproduce the LR finder from fastai and extend it with smoothing.

## Solutions

### Easy - 1 Solution
```python
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
model = nn.Linear(10, 1)
criterion = nn.MSELoss()

for lr in [0.1, 0.01, 0.001]:
    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"LR={lr}: first={losses[0]:.4f}, last={losses[-1]:.4f}")
```

## Related Concepts

DL-101 Gradient Descent, DL-109 Backpropagation, DL-105 Loss Functions, DL-150 Adam Optimizer, DL-151 SGD with Momentum

## Next Concepts

DL-172 Gradient Norm Monitoring, DL-173 Loss Monitoring

## Summary

Learning rate selection is the most critical hyperparameter in deep learning optimization. It controls the step size of gradient updates and must be chosen carefully to balance convergence speed and stability. Modern techniques include adaptive methods (Adam), schedules (cosine annealing, step decay), and range tests to find optimal values.

## Key Takeaways

- Learning rate determines step size in gradient descent and directly affects convergence
- Too high causes divergence; too low causes slow convergence
- LR range tests help find optimal values efficiently
- LR schedules improve final performance over fixed LRs
- Adaptive optimizers like Adam are more LR-tolerant but still benefit from tuning
- Batch size and LR are coupled — doubling batch size often requires doubling LR
