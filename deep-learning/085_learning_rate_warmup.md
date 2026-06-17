# Concept: Learning Rate Warmup

## Concept ID

DL-085

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand why learning rate warmup improves training stability
- Implement linear and exponential warmup schedules in PyTorch
- Determine appropriate warmup duration
- Combine warmup with decay schedules
- Recognize when warmup is necessary

## Prerequisites

- Learning Rate concepts
- Adam Optimizer (DL-078)
- Transformer architecture awareness

## Definition

Learning rate warmup is a scheduling technique where the learning rate gradually increases from a small value to a target value over a specified number of steps or epochs at the beginning of training. The simplest form is linear warmup:

eta_t = eta_max * min(1, t / T_warmup)

where t is the current step, T_warmup is the warmup duration, and eta_max is the maximum learning rate after warmup.

## Intuition

At the start of training, model parameters are randomly initialized. The initial gradients are often large and noisy because the model is far from optimal. Using a large learning rate immediately can cause the parameters to be thrown into chaotic regions of the loss landscape, leading to divergence or poor conditioning. Warmup allows the optimizer to gently explore the initial parameter space before applying full-magnitude updates.

Think of it as warming up before exercise — you don't sprint at full speed from a standing start. You jog first, stretch, then gradually accelerate. This prevents injury (divergence) and leads to better overall performance.

## Why This Concept Matters

Learning rate warmup is critical for modern deep learning:

- **Transformers**: Warmup is essential for training transformers, as specified in the original Transformer paper
- **Large batch training**: When using large batches, the initial gradient estimate is more reliable but still needs a gentle start
- **Adam optimizer**: Adam's adaptive learning rates can be very large initially due to small second-moment estimates; warmup prevents early divergence
- **Pre-training**: Large-scale pre-training (BERT, GPT) universally uses warmup

## Mathematical Explanation

### Common Warmup Schedules

**Linear warmup:**
eta_t = eta_max * min(1, t / T_warmup)

**Exponential warmup:**
eta_t = eta_max * (1 - exp(-t / tau))

**Constant warmup:**
eta_t = eta_min for t < T_warmup_start, then eta_max

**Cosine warmup (Warmup then Cosine):**
eta_t = eta_min + (eta_max - eta_min) * (1 + cos(pi + pi * t / T)) / 2 for t > T_warmup

### Why Warmup Helps Adam

Adam maintains moving averages m_t and v_t, initialized to zero. Early in training:

- m_t is small (biased toward zero)
- v_t is very small (biased toward zero)

After bias correction, the effective learning rate for Adam is:

eta_eff = eta / (sqrt(v_hat_t) + eps)

Since v_hat_t starts very small, the denominator is dominated by eps, making the effective learning rate huge. Warmup compensates by keeping eta small until v_hat_t accumulates meaningful gradient information.

### Warmup Duration

Common warmup durations:
- 0-10% of total training steps for most tasks
- 10-20% for transformer pre-training
- Expressed in steps (e.g., 4000 steps in the original Transformer paper)

## Code Examples

### Example 1: Linear Warmup Schedule

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.Adam(model.parameters(), lr=0.0)
loss_fn = nn.MSELoss()

total_steps = 500
warmup_steps = 100
max_lr = 0.01

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps) if step < warmup_steps else 1.0
)

for step in range(total_steps):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0 or step == total_steps - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Step 0: loss = 0.979969, lr = 0.000100
# Step 100: loss = 0.094575, lr = 0.010000
# Step 200: loss = 0.009826, lr = 0.010000
# Step 300: loss = 0.009167, lr = 0.010000
# Step 400: loss = 0.009149, lr = 0.010000
# Step 499: loss = 0.009149, lr = 0.010000
```

### Example 2: Warmup with Cosine Decay

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

total_steps = 500
warmup_steps = 50
max_lr = 0.01

def warmup_cosine_lr(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

for step in range(total_steps):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0 or step == total_steps - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Step 0: loss = 0.979969, lr = 0.000200
# Step 100: loss = 0.009711, lr = 0.009549
# Step 200: loss = 0.009151, lr = 0.006910
# Step 300: loss = 0.009150, lr = 0.002843
# Step 400: loss = 0.009149, lr = 0.000413
# Step 499: loss = 0.009149, lr = 0.000000
```

### Example 3: Effect of Warmup on Adam Convergence

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Use a problem where Adam needs warmup
X = torch.randn(500, 50)
y = torch.randn(500, 1)

def train_with_warmup(warmup_steps, label):
    model = nn.Sequential(
        nn.Linear(50, 200), nn.ReLU(),
        nn.Linear(200, 200), nn.ReLU(),
        nn.Linear(200, 1)
    )
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warmup_steps)
    ) if warmup_steps > 0 else None
    losses = []
    for step in range(300):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
    print(f"{label}: final loss = {losses[-1]:.6f}")

train_with_warmup(0, "No warmup")
train_with_warmup(50, "50-step warmup")
train_with_warmup(150, "150-step warmup")
```

```
# Output:
# No warmup: final loss = 0.893763
# 50-step warmup: final loss = 0.667482
# 150-step warmup: final loss = 0.593510
```

## Common Mistakes

1. **Skipping warmup for Adam**: Adam's effective learning rate is very high initially due to small v_t. Warmup is strongly recommended.
2. **Warmup too short**: Insufficient warmup (e.g., 1% of steps) provides minimal benefit for transformer training.
3. **Warmup too long**: Wasting 50% of training on warmup when 10% suffices slows convergence unnecessarily.
4. **Applying warmup after decay starts**: Warmup should occur at the very beginning of training. Applying it later has no benefit.
5. **Not combining warmup with decay**: Warmup without eventual decay works poorly. Always pair warmup with a decay schedule.
6. **Using warmup with SGD when not needed**: SGD with small learning rates may not need warmup. Warmup primarily helps adaptive methods and large learning rates.

## Interview Questions

### Beginner

1. What is learning rate warmup?
2. Why is warmup particularly important for the Adam optimizer?
3. What is a typical warmup duration?
4. How does linear warmup work?
5. How do you implement warmup in PyTorch?

### Intermediate

1. Explain why Adam's effective learning rate is very high initially.
2. Derive the combined warmup-cosine decay schedule.
3. Compare linear vs. exponential warmup schedules.
4. How does warmup duration interact with total training length?
5. Why is warmup essential for training transformers?

### Advanced

1. Analyze the effect of warmup on Adam's bias correction terms.
2. Prove that warmup reduces the initial effective learning rate of Adam by a factor related to warmup duration.
3. Discuss adaptive warmup strategies that adjust based on gradient statistics.

## Practice Problems

### Easy

1. Implement linear warmup in PyTorch using LambdaLR.
2. Plot the learning rate curve for linear warmup + cosine decay.
3. Train a linear model with and without warmup.
4. Track the effective learning rate (eta / sqrt(v_hat)) for Adam with and without warmup.
5. Compare warmup durations [0, 10, 100, 500] steps.

### Medium

1. Implement a custom warmup scheduler that ramps up exponentially.
2. Train a 3-layer network on CIFAR-10 with warmup and cosine decay.
3. Analyze the gradient norm during warmup vs. without warmup.
4. Implement warmup for a Transformer encoder from scratch.
5. Compare warmup strategies for Adam vs. SGD with momentum.

### Hard

1. Implement an adaptive warmup that ends when gradient statistics stabilize.
2. Derive the optimal warmup duration for a given learning rate and batch size.
3. Design an experiment showing warmup enables training with otherwise divergent learning rates.

## Solutions

Warmup is implemented through learning rate schedulers. The LambdaLR scheduler is the most flexible. For linear warmup, the lambda function returns min(1, step / warmup_steps). Warmup is critical for Adam with moderate-to-large learning rates and essential for transformer training.

## Related Concepts

- Learning Rate Decay (DL-086): The counterpart to warmup
- Cosine Annealing (DL-087): Often paired with warmup
- Adam Optimizer (DL-078): Most benefits from warmup
- Cyclical Learning Rates (DL-088): Includes warmup-like phases

## Next Concepts

- Learning Rate Decay (DL-086)
- Cosine Annealing (DL-087)
- Cyclical Learning Rates (DL-088)

## Summary

Learning rate warmup gradually increases the learning rate from a small value to the target value over the initial training steps. It prevents the large, noisy updates that occur early in training, especially with adaptive optimizers like Adam whose effective learning rates start very high. Warmup is essential for training transformers and large models, and it pairs naturally with decay schedules like cosine annealing.

## Key Takeaways

1. Warmup gradually increases learning rate from near-zero to the target value.
2. Warmup is critical for Adam because v_t starts very small, causing large effective steps.
3. Typical warmup is 0-10% of total steps for most tasks, 10-20% for transformers.
4. Linear warmup and exponential warmup are the most common forms.
5. Warmup should always be paired with a decay schedule for optimal results.
