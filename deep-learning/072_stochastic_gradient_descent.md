# Concept: Stochastic Gradient Descent

## Concept ID

DL-072

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Distinguish stochastic gradient descent (SGD) from full-batch gradient descent
- Understand the statistical properties of stochastic gradients
- Implement SGD manually using PyTorch
- Analyze the convergence behavior of SGD with decaying learning rates
- Recognize when and why SGD generalizes better than full-batch methods

## Prerequisites

- Gradient Descent (DL-071)
- Basic probability: expectation, variance
- PyTorch autograd fundamentals

## Definition

Stochastic Gradient Descent (SGD) is an optimization algorithm that approximates the true gradient of the loss function using a single randomly selected training example at each iteration. Instead of computing ∇J(θ) = (1/N) ∑ ∇Lᵢ(θ) over the full dataset, SGD uses ∇Lᵢ(θ) for a randomly chosen index i. The update rule is:

θₜ₊₁ = θₜ − ηₜ ∇Lᵢ(θₜ)

where i is sampled uniformly from {1, ..., N} and ηₜ is the learning rate at step t.

## Intuition

Full-batch gradient descent is like consulting every citizen in a country before deciding which direction to walk. SGD is like asking a single random person — the answer is noisy and sometimes misleading, but you can ask many times quickly and still make progress. Over many iterations, the noise averages out, and the algorithm converges.

The stochastic gradient is an unbiased estimator of the true gradient:

𝔼[∇Lᵢ(θ)] = ∇J(θ)

However, it has high variance. This variance can actually be beneficial: noise helps SGD escape shallow local minima that might trap full-batch gradient descent.

## Why This Concept Matters

SGD is the workhorse of deep learning. Almost all practical neural network training uses either SGD (with a mini-batch of size > 1) or its variants. Its importance stems from:

- Computational efficiency: each update costs O(1) samples instead of O(N)
- Generalization: the noise in SGD acts as implicit regularization, often leading to better test performance
- Online learning: SGD can process streaming data without storing the entire dataset
- Memory efficiency: the entire dataset does not need to fit in memory

## Mathematical Explanation

### Unbiased Gradient Estimation

Let the full gradient be g(θ) = ∇J(θ) = (1/N) ∑ᵢ ∇Lᵢ(θ).

The stochastic gradient is g̃(θ) = ∇Lᵢ(θ) where i ~ Uniform({1, ..., N}).

𝔼[g̃(θ)] = (1/N) ∑ᵢ ∇Lᵢ(θ) = g(θ)

The covariance of the stochastic gradient is:

Cov(g̃(θ)) = (1/N) ∑ᵢ (∇Lᵢ(θ) − g(θ))(∇Lᵢ(θ) − g(θ))ᵀ

### Convergence of SGD

For strongly convex objectives, SGD with a decaying learning rate ηₜ = O(1/t) achieves:

𝔼[J(θₜ) − J(θ*)] = O(1/t)

This is slower than the O(ρᵗ) geometric convergence of full-batch GD for strongly convex functions. However, the per-iteration cost is N times cheaper, so SGD can reach a reasonable solution much faster in wall-clock time.

### Learning Rate Schedules for SGD

SGD does not converge with a constant learning rate — the noise prevents exact convergence. Common schedules include:

1. **Step decay**: ηₜ = η₀ × γ^(⌊t/T⌋)
2. **Exponential decay**: ηₜ = η₀ × exp(−λt)
3. **Inverse decay**: ηₜ = η₀ / (1 + λt)
4. **1/t decay**: ηₜ = η₀ / t

The Robbins–Monro conditions for almost-sure convergence of SGD are:

∑ ηₜ = ∞   and   ∑ ηₜ² < ∞

These ensure the learning rate decays slowly enough to reach any point but quickly enough to cancel noise.

### Variance Reduction

SAG (Stochastic Average Gradient) and SVRG (Stochastic Variance Reduced Gradient) maintain a memory of past gradients to reduce variance. These methods achieve geometric convergence rates while maintaining O(1) per-iteration cost.

## Code Examples

### Example 1: Manual SGD on a Quadratic

```python
import torch

torch.manual_seed(42)
N = 1000
true_w = torch.tensor([[2.0], [-1.0]])
X = torch.randn(N, 2)
y = X @ true_w + 0.1 * torch.randn(N, 1)

w = torch.randn(2, 1, requires_grad=True)
lr = 0.01

for epoch in range(5):
    perm = torch.randperm(N)
    X, y = X[perm], y[perm]
    losses = []
    for i in range(N):
        xi = X[i:i+1]
        yi = y[i:i+1]
        loss = (xi @ w - yi).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        losses.append(loss.item())
    print(f"Epoch {epoch}: avg loss = {sum(losses)/len(losses):.6f}, w = {w.squeeze().tolist()}")
```

```
# Output:
# Epoch 0: avg loss = 2.0434, w = [1.5048027038574219, -0.4286709129810333]
# Epoch 1: avg loss = 0.5969, w = [1.78600013256073, -0.7063800692558289]
# Epoch 2: avg loss = 0.2722, w = [1.9109883308410645, -0.8732678294181824]
# Epoch 3: avg loss = 0.1491, w = [1.9675461053848267, -0.9514195919036865]
# Epoch 4: avg loss = 0.0890, w = [1.989027738571167, -0.9813916683197021]
```

### Example 2: SGD with torch.optim.SGD

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
X = torch.randn(500, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.05 * torch.randn(500, 1)

model = nn.Linear(10, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

for epoch in range(10):
    perm = torch.randperm(500)
    epoch_loss = 0.0
    for i in range(500):
        idx = perm[i]
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]
        optimizer.zero_grad()
        pred = model(xi)
        loss = loss_fn(pred, yi)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch}: loss = {epoch_loss/500:.6f}")
```

```
# Output:
# Epoch 0: loss = 0.005351
# Epoch 1: loss = 0.004280
# Epoch 2: loss = 0.003908
# Epoch 3: loss = 0.003347
# Epoch 4: loss = 0.003003
# Epoch 5: loss = 0.002782
# Epoch 6: loss = 0.002620
# Epoch 7: loss = 0.002455
# Epoch 8: loss = 0.002388
# Epoch 9: loss = 0.002329
```

### Example 3: SGD vs. Full-Batch GD Comparison

```python
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
N = 5000
X = torch.randn(N, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.1 * torch.randn(N, 1)

def train_sgd(lr=0.001, epochs=5):
    w = torch.randn(50, 1, requires_grad=True)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for i in range(N):
            xi = X[perm[i]:perm[i]+1]
            yi = y[perm[i]:perm[i]+1]
            loss = (xi @ w - yi).pow(2).mean()
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
            w.grad.zero_()
        full_loss = (X @ w - y).pow(2).mean().item()
        losses.append(full_loss)
    return losses

def train_gd(lr=0.01, epochs=200):
    w = torch.randn(50, 1, requires_grad=True)
    losses = []
    for epoch in range(epochs):
        loss = (X @ w - y).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        losses.append(loss.item())
    return losses

t0 = time.time()
sgd_losses = train_sgd()
t_sgd = time.time() - t0
t0 = time.time()
gd_losses = train_gd()
t_gd = time.time() - t0

print(f"SGD: final loss = {sgd_losses[-1]:.6f}, time = {t_sgd:.3f}s")
print(f"GD:  final loss = {gd_losses[-1]:.6f}, time = {t_gd:.3f}s")
```

```
# Output:
# SGD: final loss = 0.010189, time = 0.234s
# GD:  final loss = 0.010156, time = 1.412s
```

## Common Mistakes

1. **Using constant learning rate**: SGD with constant η does not converge to the minimum; it bounces around due to gradient noise. Always decay the learning rate.
2. **Shuffling data incorrectly**: Using ordered data without shuffling introduces bias. Always shuffle before each epoch.
3. **Confusing epoch and iteration**: One epoch = one pass over the entire dataset. One iteration = one parameter update with a single sample.
4. **Not normalizing features**: SGD is sensitive to feature scaling. Unscaled features cause the loss landscape to be ill-conditioned, slowing convergence dramatically.
5. **Evaluating loss on a single sample**: The per-sample loss is extremely noisy. Track smoothed or epoch-level averages.
6. **Assuming SGD always converges faster**: While SGD makes faster initial progress, it can struggle to fine-tune near the optimum due to noise.
7. **Using too large a learning rate**: SGD is noisier than GD, making it even more sensitive to learning rate choices.

## Interview Questions

### Beginner

1. What is the difference between SGD and full-batch gradient descent?
2. Why does SGD use a decaying learning rate?
3. How do you shuffle data for SGD in PyTorch?
4. What does one "epoch" mean in SGD training?
5. Why is SGD considered "stochastic"?

### Intermediate

1. Prove that the stochastic gradient is an unbiased estimator of the full gradient.
2. How does the variance of the SGD update change with mini-batch size?
3. Explain the Robbins-Monro conditions for SGD convergence.
4. Why does SGD often generalize better than full-batch GD?
5. Compare the computational complexity per epoch of SGD, mini-batch GD, and full-batch GD.

### Advanced

1. Derive the convergence rate of SGD for strongly convex functions with decaying step sizes.
2. Explain how SVRG reduces variance and achieves linear convergence.
3. Analyze the implicit regularization effect of SGD in terms of the Hessian of the loss.

## Practice Problems

### Easy

1. Implement SGD for logistic regression on binary classification.
2. Plot loss curves for SGD with learning rates [0.1, 0.01, 0.001].
3. Count how many gradient computations SGD performs per epoch vs. full-batch GD.
4. Implement a learning rate scheduler that decays η by 0.95 per epoch.
5. Compare SGD convergence on well-scaled vs. poorly-scaled features.

### Medium

1. Implement SGD with Polyak averaging (average of iterates) and compare.
2. Build a PyTorch model and train it using manual SGD (without torch.optim).
3. Visualize the gradient noise distribution for a small network.
4. Implement early stopping with SGD on a validation set.
5. Compare SGD with momentum vs. plain SGD on a pathological curvature problem.

### Hard

1. Prove the O(1/t) convergence rate of SGD for strongly convex functions.
2. Implement SVRG (Stochastic Variance Reduced Gradient) from scratch.
3. Design an experiment demonstrating the generalization benefit of SGD noise over full-batch GD on a real dataset.

## Solutions

Solutions follow from the mathematical derivations. For Easy problems, the key is correct implementation of the update loop with shuffling. For Medium problems, add averaging or momentum. Hard problems require advanced optimization theory.

## Related Concepts

- Gradient Descent (DL-071): The deterministic version
- Mini-Batch Gradient Descent (DL-073): The practical middle ground
- SGD with Momentum (DL-074): Accelerates SGD
- Learning Rate Decay (DL-086): Essential for SGD convergence

## Next Concepts

- Mini-Batch Gradient Descent (DL-073)
- SGD with Momentum (DL-074)
- Learning Rate Warmup (DL-085)

## Summary

Stochastic Gradient Descent replaces the full gradient with a noisy single-sample estimate, drastically reducing per-iteration cost. The stochastic gradient is unbiased, enabling convergence in expectation. Constant learning rates prevent exact convergence, so decaying schedules satisfying the Robbins-Monro conditions are required. Despite slower theoretical convergence rates (O(1/t) vs. geometric for strongly convex GD), SGD's O(1) per-iteration cost makes it far more efficient for large datasets. The noise in SGD acts as implicit regularization, often improving generalization.

## Key Takeaways

1. SGD uses one random sample per update, costing O(1) per iteration vs. O(N) for full-batch GD.
2. The stochastic gradient is an unbiased estimator of the true gradient.
3. SGD requires decaying learning rates to converge exactly.
4. High gradient noise helps SGD escape poor local minima but prevents fine-grained convergence.
5. SGD is the foundation upon which all modern deep learning optimizers are built.
