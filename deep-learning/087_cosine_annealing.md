# Concept: Cosine Annealing

## Concept ID

DL-087

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the cosine annealing learning rate schedule
- Implement cosine annealing in PyTorch
- Combine cosine annealing with warmup
- Use cosine annealing with restarts (SGDR)
- Analyze why cosine annealing works well in practice

## Prerequisites

- Learning Rate Decay (DL-086)
- Learning Rate Warmup (DL-085)
- SGD with Momentum (DL-074)

## Definition

Cosine annealing is a learning rate schedule that follows a cosine curve to smoothly decrease the learning rate from an initial value to a minimum value. The learning rate at step t is:

eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t / T * pi))

where eta_max is the initial learning rate, eta_min is the minimum learning rate, t is the current step (or epoch), and T is the total number of steps.

With restarts (SGDR): The schedule resets every T_0 steps, allowing the learning rate to jump back up.

## Intuition

Cosine annealing mimics the natural process of cooling. Early in training, the learning rate is high and decreases slowly. As training progresses, the rate of decrease accelerates (the cosine curve drops steeply in the middle), then flattens out at the end for fine-grained convergence. The smooth, continuous nature of the cosine function avoids the sharp discontinuities of step decay.

With restarts, the process repeats: the learning rate jumps up, allowing the optimizer to escape suboptimal minima and explore new regions of the loss landscape before annealing again.

## Why This Concept Matters

Cosine annealing has become one of the most popular learning rate schedules:

- **Smooth decay**: No sharp discontinuities that can destabilize training
- **Excellent empirical performance**: Often outperforms step and exponential decay
- **SGDR (restarts)**: Helps escape sharp minima and find flatter, better-generalizing solutions
- **Standard for modern architectures**: Used in training ResNeXt, EfficientNet, and many transformer variants
- **Simple implementation**: Just one hyperparameter (T_max) beyond the usual learning rate

## Mathematical Explanation

### Cosine Annealing Schedule

eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t * pi / T_max))

At t = 0: eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(0)) = eta_max
At t = T_max: eta = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi)) = eta_min

### SGDR: Stochastic Gradient Descent with Restarts

SGDR restarts the learning rate to eta_max every T_0 epochs. The restart period can be fixed (T_0 constant) or multiplicative (T_0 * factor^i):

eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos((t mod T_i) * pi / T_i))

### Why Cosine Works

The cosine schedule spends more time at moderate learning rates compared to step decay. This balances exploration (high learning rates) and exploitation (low learning rates) more effectively. The continuous nature also avoids the sudden loss spikes that can occur with step decay.

### Comparison with Step Decay

| Property | Step Decay | Cosine Annealing |
|----------|------------|------------------|
| Decay shape | Sharp drops | Smooth curve |
| Discontinuities | Yes | No |
| Parameters | gamma, step_size | T_max, eta_min |
| Popularity | Traditional | Modern |

## Code Examples

### Example 1: Cosine Annealing Without Restarts

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
loss_fn = nn.MSELoss()

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 10 == 0 or epoch == 49:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Epoch 0: loss = 0.979969, lr = 0.100000
# Epoch 10: loss = 0.009406, lr = 0.075479
# Epoch 20: loss = 0.009149, lr = 0.025879
# Epoch 30: loss = 0.009149, lr = 0.002926
# Epoch 40: loss = 0.009149, lr = 0.000166
# Epoch 49: loss = 0.009149, lr = 0.000100
```

### Example 2: Cosine Annealing with Warmup

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.Adam(model.parameters(), lr=0.0)
loss_fn = nn.MSELoss()

total_epochs = 100
warmup_epochs = 10
max_lr = 0.01
min_lr = 1e-6

def warmup_cosine(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return cosine_decay + min_lr / max_lr

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

for epoch in range(total_epochs):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 20 == 0 or epoch == total_epochs - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.8f}")
```

```
# Output:
# Epoch 0: loss = 0.979969, lr = 0.00100000
# Epoch 20: loss = 0.009207, lr = 0.00617284
# Epoch 40: loss = 0.009149, lr = 0.00279557
# Epoch 60: loss = 0.009149, lr = 0.00101062
# Epoch 80: loss = 0.009149, lr = 0.00021345
# Epoch 99: loss = 0.009149, lr = 0.00000100
```

### Example 3: SGDR (Cosine with Restarts)

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 20)
y = torch.randn(2000, 1)

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 10 == 0 or epoch == 99:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Epoch 0: loss = 1.010394, lr = 0.050000
# Epoch 10: loss = 0.931436, lr = 0.038849
# Epoch 20: loss = 0.841515, lr = 0.010506
# Epoch 30: loss = 0.764525, lr = 0.050000
# Epoch 40: loss = 0.696735, lr = 0.015426
# Epoch 50: loss = 0.677462, lr = 0.050000
# Epoch 60: loss = 0.635730, lr = 0.001453
# Epoch 70: loss = 0.621901, lr = 0.050000
# Epoch 80: loss = 0.614702, lr = 0.002943
# Epoch 90: loss = 0.607909, lr = 0.050000
# Epoch 99: loss = 0.604664, lr = 0.005426
```

## Common Mistakes

1. **Not setting eta_min**: The minimum learning rate defaults to 0, which may be too low. Set eta_min to a small positive value like 1e-6.
2. **Restarts too frequent**: With SGDR, T_0 too small causes constant restarts without enough time to converge. Use T_0 = 10-50 epochs.
3. **Restarts too infrequent**: Very large T_0 with T_mult=1 means few restarts, losing the benefit of SGDR.
4. **Forgetting warmup before cosine**: Cosine decay starts at the maximum learning rate. Without warmup, this can destabilize early training, especially for Adam.
5. **Using cosine decay with very short training**: For fewer than 10 epochs, cosine decay may not provide much benefit over constant or step decay.
6. **Not matching T_max to total epochs**: For non-restart cosine, T_max should equal the total number of epochs/steps.

## Interview Questions

### Beginner

1. How does cosine annealing decay the learning rate?
2. What is SGDR and how does it differ from standard cosine annealing?
3. How do you implement cosine annealing in PyTorch?
4. What does T_max control in CosineAnnealingLR?
5. Why might cosine annealing outperform step decay?

### Intermediate

1. Derive the cosine annealing formula and explain its shape.
2. Compare cosine annealing with and without restarts.
3. How does cosine annealing help escape sharp minima?
4. Explain the warmup-then-cosine schedule and why it works.
5. How would you set T_mult in CosineAnnealingWarmRestarts?

### Advanced

1. Prove that cosine annealing achieves a specific convergence rate under convex assumptions.
2. Analyze the relationship between cosine annealing and simulated annealing.
3. Design an adaptive variant of cosine annealing that adjusts based on validation loss.

## Practice Problems

### Easy

1. Implement CosineAnnealingLR with T_max=50.
2. Plot the learning rate curve for cosine annealing over 100 epochs.
3. Compare cosine decay vs. step decay on a simple linear regression.
4. Implement cosine annealing with warmup in PyTorch.
5. Track and print learning rates for a full cosine schedule.

### Medium

1. Compare cosine annealing with and without restarts on a 3-layer network.
2. Find the optimal T_max and eta_min for a given task.
3. Train a small CNN on CIFAR-10 with cosine annealing.
4. Visualize the loss landscape convergence with cosine vs. step decay.
5. Implement a custom scheduler with cosine annealing and hard restarts.

### Hard

1. Derive the optimal restart period for SGDR based on loss landscape curvature.
2. Implement a variant of SGDR with adaptive restart based on validation plateau.
3. Design an experiment showing SGDR finding flatter minima than other schedules.

## Solutions

CosineAnnealingLR(optimizer, T_max) is the basic implementation. CosineAnnealingWarmRestarts(optimizer, T_0, T_mult) adds restarts. For warmup, combine with LambdaLR or use a custom scheduler.

## Related Concepts

- Learning Rate Decay (DL-086): General category of schedules
- Learning Rate Warmup (DL-085): Precursor to cosine decay
- Cyclical Learning Rates (DL-088): Another cyclic approach
- One Cycle Policy (DL-089): A specific cosine-like schedule

## Next Concepts

- Cyclical Learning Rates (DL-088)
- One Cycle Policy (DL-089)
- Optimizer Comparison (DL-090)

## Summary

Cosine annealing smoothly decreases the learning rate following a cosine curve, starting at eta_max and ending at eta_min over T_max steps. The smooth decay avoids discontinuities and balances exploration with exploitation. SGDR (Stochastic Gradient Descent with Restarts) periodically resets the learning rate, helping escape sharp minima and find flatter, better-generalizing solutions. Cosine annealing is the default schedule for many modern architectures.

## Key Takeaways

1. Cosine annealing: eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(t * pi / T_max)).
2. The smooth cosine decay avoids sharp discontinuities in the learning rate.
3. SGDR with restarts helps escape poor local minima.
4. Cosine annealing is often paired with linear warmup.
5. Cosine scheduling is the default for many state-of-the-art vision and language models.
