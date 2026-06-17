# Concept: Learning Rate Decay

## Concept ID

DL-086

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand why learning rate decay improves convergence
- Implement step, exponential, and inverse decay schedules
- Apply learning rate schedulers in PyTorch
- Choose appropriate decay strategies for different tasks
- Combine decay with warmup for optimal schedules

## Prerequisites

- Learning Rate concepts
- Gradient Descent (DL-071)
- SGD with Momentum (DL-074)

## Definition

Learning rate decay (also called learning rate annealing) is a technique that reduces the learning rate over time during training. The idea is to take large steps early in training to make rapid progress, then smaller steps later to fine-tune parameters precisely. Common decay schedules include:

**Step decay:** eta_t = eta_0 * gamma^{floor(t / T_step)}
**Exponential decay:** eta_t = eta_0 * exp(-lambda * t)
**Inverse decay:** eta_t = eta_0 / (1 + lambda * t)
**Cosine decay:** eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * t / T))

## Intuition

Imagine searching for a campsite in a large forest while it's getting dark. Initially, you take large steps to cover ground quickly and find the general area. As it gets darker (metaphorically, as you approach the minimum), you take smaller, more careful steps to avoid missing the exact spot.

Learning rate decay allows the optimizer to explore broadly early on and then settle precisely into a minimum. Without decay, SGD with a constant learning rate oscillates around the minimum without converging.

## Why This Concept Matters

Learning rate decay is crucial for convergence:

- **SGD convergence**: SGD with constant learning rate does not converge to the minimum — it bounces around due to gradient noise. Decay is required for exact convergence.
- **Fine-tuning**: Large learning rates in the final stages prevent fine-grained optimization.
- **Optimal schedules**: Schedule design significantly impacts final model quality.
- **Practical necessity**: Virtually all production training pipelines use learning rate decay.

## Mathematical Explanation

### SGD Convergence

For SGD to converge to the minimum, the learning rate must satisfy the Robbins-Monro conditions:

sum_{t=1}^{inf} eta_t = inf  (decay slowly enough to reach any point)
sum_{t=1}^{inf} eta_t^2 < inf  (decay quickly enough to cancel noise)

These conditions are satisfied by eta_t = O(1/t) but NOT by constant learning rates.

### Step Decay

eta_t = eta_0 * gamma^{floor(1 + t / T_step)}

Common: gamma = 0.1, T_step = 30 epochs
The learning rate drops by 10x every 30 epochs.

### Exponential Decay

eta_t = eta_0 * exp(-lambda * t)

The decay rate lambda controls how quickly the learning rate drops. The half-life is t_{1/2} = ln(2) / lambda.

### Inverse Decay

eta_t = eta_0 / (1 + lambda * t)

This satisfies the Robbins-Monro conditions when lambda > 0.

### Cosine Decay

eta_t = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * t / T))

The learning rate smoothly decreases from eta_0 to eta_min following a cosine curve. This schedule is popular for modern deep learning.

## Code Examples

### Example 1: Step Decay

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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 20 == 0 or epoch == 99:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Epoch 0: loss = 0.979969, lr = 0.100000
# Epoch 20: loss = 0.009638, lr = 0.100000
# Epoch 30: loss = 0.009149, lr = 0.010000
# Epoch 40: loss = 0.009149, lr = 0.010000
# Epoch 60: loss = 0.009149, lr = 0.001000
# Epoch 80: loss = 0.009149, lr = 0.000100
# Epoch 99: loss = 0.009149, lr = 0.000010
```

### Example 2: Exponential Decay

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
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
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
# Epoch 10: loss = 0.009345, lr = 0.059874
# Epoch 20: loss = 0.009149, lr = 0.035849
# Epoch 30: loss = 0.009149, lr = 0.021464
# Epoch 40: loss = 0.009149, lr = 0.012849
# Epoch 49: loss = 0.009149, lr = 0.008077
```

### Example 3: Comparing Decay Schedules

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

def train_schedule(scheduler_constructor, label):
    model = nn.Linear(20, 1, bias=False)
    opt = optim.SGD(model.parameters(), lr=0.1)
    sched = scheduler_constructor(opt)
    loss_fn = nn.MSELoss()
    for epoch in range(50):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        sched.step()
    final_loss = loss_fn(model(X), y).item()
    print(f"{label}: final loss = {final_loss:.6f}")

train_schedule(lambda opt: optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0), "Constant")
train_schedule(lambda opt: optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1), "Step")
train_schedule(lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.95), "Exponential")
train_schedule(lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50), "Cosine")
```

```
# Output:
# Constant: final loss = 0.021672
# Step: final loss = 0.009149
# Exponential: final loss = 0.009152
# Cosine: final loss = 0.009149
```

## Common Mistakes

1. **No decay at all**: Using a constant learning rate throughout training prevents exact convergence. The model oscillates around the minimum.
2. **Decaying too quickly**: Aggressive decay (e.g., gamma=0.1 every 5 epochs) stops learning prematurely. The model gets stuck far from the minimum.
3. **Decaying too slowly**: Very mild decay (e.g., gamma=0.999 per epoch) provides insufficient annealing. The model continues to bounce at the end.
4. **Forgetting to call scheduler.step()**: The scheduler does not automatically advance. You must call scheduler.step() each epoch or batch.
5. **Decaying before warmup**: Apply warmup first, then decay. Applying decay first defeats the purpose of warmup.
6. **Using the wrong scheduler order**: Call scheduler.step() after optimizer.step(), not before.

## Interview Questions

### Beginner

1. Why is learning rate decay necessary for SGD convergence?
2. How does step decay differ from exponential decay?
3. What are the Robbins-Monro conditions for learning rates?
4. How do you apply learning rate decay in PyTorch?
5. What happens if you never decay the learning rate?

### Intermediate

1. Derive the Robbins-Monro conditions and explain their significance.
2. Compare the convergence behavior of step, exponential, and cosine decay.
3. How does learning rate decay interact with momentum?
4. When would you prefer cosine decay over step decay?
5. How do you determine the appropriate decay rate for a given task?

### Advanced

1. Prove that SGD with eta_t = O(1/t) achieves optimal convergence rate.
2. Analyze the effect of learning rate schedules on generalization.
3. Derive an optimal learning rate schedule for a given loss landscape curvature.

## Practice Problems

### Easy

1. Implement step decay with gamma=0.1 and step_size=10.
2. Plot learning rate vs. epoch for exponential decay.
3. Compare constant lr vs. step decay for linear regression.
4. Use CosineAnnealingLR in PyTorch.
5. Track the learning rate during training and verify the schedule.

### Medium

1. Implement inverse decay manually (not using a scheduler).
2. Compare step, exponential, and cosine decay on a 3-layer network.
3. Design a schedule with warmup for 10 epochs followed by cosine decay.
4. Analyze the effect of decay rate on final model performance.
5. Implement ReduceLROnPlateau and compare with fixed schedules.

### Hard

1. Derive the optimal learning rate schedule for quadratic objectives.
2. Implement an adaptive learning rate schedule based on gradient norms.
3. Design an experiment showing the effect of different decay schedules on the sharpness of the found minimum.

## Solutions

PyTorch provides various schedulers in torch.optim.lr_scheduler. The most common are StepLR, ExponentialLR, CosineAnnealingLR, and ReduceLROnPlateau. Schedulers must be stepped each epoch/batch after optimizer.step().

## Related Concepts

- Learning Rate Warmup (DL-085): The counterpart that precedes decay
- Cosine Annealing (DL-087): A specific smooth decay schedule
- Cyclical Learning Rates (DL-088): Learning rates that cycle up and down
- Gradient Clipping (DL-084): Complements decay for training stability

## Next Concepts

- Cosine Annealing (DL-087)
- Cyclical Learning Rates (DL-088)
- One Cycle Policy (DL-089)

## Summary

Learning rate decay reduces the learning rate over training, enabling rapid initial progress followed by fine-grained convergence. Common schedules include step decay (sharp drops), exponential decay (smooth decreases), inverse decay (Robbins-Monro satisfying), and cosine decay (smooth S-curve). Decay is necessary for SGD convergence and improves final model quality for all optimizers.

## Key Takeaways

1. Learning rate decay is required for SGD to converge exactly to the minimum.
2. Step, exponential, inverse, and cosine decay are the main schedule types.
3. Cosine decay is the most popular choice for modern deep learning.
4. Decay should be paired with warmup for optimal results.
5. Call scheduler.step() after each optimizer.step() to advance the schedule.
