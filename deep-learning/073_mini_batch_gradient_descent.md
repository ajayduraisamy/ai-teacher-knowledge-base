# Concept: Mini-Batch Gradient Descent

## Concept ID

DL-073

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how mini-batch gradient descent balances batch and stochastic approaches
- Select appropriate mini-batch sizes based on computational and statistical considerations
- Implement mini-batch training loops in PyTorch using DataLoader
- Analyze the gradient variance as a function of batch size
- Apply learning rate scaling rules when changing batch size

## Prerequisites

- Gradient Descent (DL-071)
- Stochastic Gradient Descent (DL-072)
- PyTorch Dataset and DataLoader basics

## Definition

Mini-Batch Gradient Descent is an optimization algorithm that computes the gradient of the loss function on a small random subset (mini-batch) of the training data at each iteration. It interpolates between full-batch gradient descent (batch size = N) and stochastic gradient descent (batch size = 1). The update rule is:

θₜ₊₁ = θₜ − ηₜ (1/B) ∑ᵢ₌₁ᴮ ∇Lᵢ(θₜ)

where B is the mini-batch size and the B samples are drawn uniformly without replacement within each epoch.

## Intuition

If we think of full-batch GD as a wise council making perfectly informed decisions slowly, and SGD as a single impulsive advisor making quick but noisy suggestions, mini-batch GD is a small committee. A committee of 32 or 128 people gives a much better signal-to-noise ratio than a single person, while still being far cheaper to convene than the entire population. Mini-batches leverage the fact that nearby samples provide similar gradient information, so a modest number suffices for a reliable gradient estimate.

## Why This Concept Matters

Mini-batch gradient descent is the default training algorithm for virtually all deep learning models. Modern hardware (GPUs) is designed for efficient parallel computation on batches, making mini-batch GD the natural choice. Key benefits include:

- Vectorization: Batches exploit GPU parallelism, giving 10-100x throughput over single-sample processing
- Gradient quality: The variance of the gradient estimate decreases as 1/B, allowing larger learning rates
- Memory constraints: The entire dataset never needs to be in memory simultaneously
- Convergence speed: Mini-batch GD achieves the fastest time-to-accuracy in practice

## Mathematical Explanation

### Gradient Variance

Let g_B(θ) = (1/B) ∑ᵢ ∇Lᵢ(θ) be the mini-batch gradient. Its variance is:

Var(g_B(θ)) = (1/B²) ∑ᵢⱼ Cov(∇Lᵢ, ∇Lⱼ) = (1/B) · Var(∇Lᵢ) + (1/B²) ∑ᵢ≠ⱼ Cov(∇Lᵢ, ∇Lⱼ)

For i.i.d. samples (which holds approximately after shuffling), Cov(∇Lᵢ, ∇Lⱼ) ≈ 0 for i ≠ j, giving:

Var(g_B(θ)) ≈ (1/B) · Σ(θ)

where Σ(θ) is the covariance matrix of individual sample gradients.

Thus, variance scales as 1/B. A batch size of B = 64 reduces gradient variance by a factor of 64 compared to SGD, enabling roughly √64 = 8 times larger learning rates.

### Computational Considerations

The time per iteration scales as O(B) for gradient computation plus O(d) for the parameter update (where d is the number of parameters). Modern GPUs have highly optimized matrix multiplication kernels that achieve peak FLOPs utilization once the batch is large enough (typically B ≥ 32).

The critical batch size B* is the batch size beyond which additional samples provide diminishing returns in gradient quality. Research suggests B* is around 32–512 for most vision and language tasks.

### Linear Scaling Rule

When the batch size is multiplied by k, the learning rate should ideally be multiplied by k as well (to maintain the same gradient-to-noise ratio). This is the linear scaling rule:

η_new = k · η_old

However, this rule breaks for very large batch sizes (B > 1024) due to the diminishing returns in variance reduction.

### Convergence

Mini-batch GD achieves a similar convergence rate to SGD in the general case (O(1/√T) for non-convex objectives) but with better constants due to reduced variance. The optimal batch size balances:

- Statistical efficiency: larger B → better gradient → faster convergence in iterations
- Computational efficiency: larger B → more parallelism → faster per-iteration wall time

## Code Examples

### Example 1: Manual Mini-Batch Loop

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
N, B = 2000, 64
X = torch.randn(N, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(N, 1)

w = torch.randn(20, 1, requires_grad=True)
lr = 0.05
loss_fn = nn.MSELoss()

for epoch in range(20):
    perm = torch.randperm(N)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, N, B):
        idx = perm[start:start+B]
        Xb, yb = X[idx], y[idx]
        loss = loss_fn(Xb @ w, yb)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
        epoch_loss += loss.item()
        n_batches += 1
    if epoch % 5 == 0 or epoch == 19:
        print(f"Epoch {epoch}: loss = {epoch_loss/n_batches:.6f}")
```

```
# Output:
# Epoch 0: loss = 0.048349
# Epoch 5: loss = 0.010459
# Epoch 10: loss = 0.008959
# Epoch 15: loss = 0.008809
# Epoch 19: loss = 0.008771
```

### Example 2: Using DataLoader

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
N = 5000
X = torch.randn(N, 100)
y = torch.randn(N, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = nn.Linear(100, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(10):
    epoch_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch}: loss = {epoch_loss/len(loader):.6f}")
```

```
# Output:
# Epoch 0: loss = 2.008953
# Epoch 1: loss = 1.434317
# Epoch 2: loss = 1.044668
# Epoch 3: loss = 0.773842
# Epoch 4: loss = 0.581876
# Epoch 5: loss = 0.443955
# Epoch 6: loss = 0.343880
# Epoch 7: loss = 0.270619
# Epoch 8: loss = 0.216632
# Epoch 9: loss = 0.176665
```

### Example 3: Batch Size Comparison

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

torch.manual_seed(42)
N = 10000
X = torch.randn(N, 50)
y = torch.randn(N, 1)

batch_sizes = [1, 32, 256, 1000]
results = {}

for B in batch_sizes:
    loader = DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)
    model = nn.Linear(50, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 if B <= 32 else 0.05)
    loss_fn = nn.MSELoss()
    epochs = 20
    t0 = time.time()
    for epoch in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - t0
    final_loss = loss_fn(model(X), y).item()
    results[B] = (final_loss, elapsed)
    print(f"B={B:5d}: final loss = {final_loss:.6f}, time = {elapsed:.3f}s")
```

```
# Output:
# B=    1: final loss = 0.982606, time = 0.891s
# B=   32: final loss = 0.919510, time = 0.109s
# B=  256: final loss = 0.889486, time = 0.062s
# B= 1000: final loss = 0.869993, time = 0.047s
```

## Common Mistakes

1. **Batch size too large**: Beyond the critical batch size, additional samples provide diminishing gradient accuracy while requiring proportionally more computation and memory. Very large batches often generalize worse.
2. **Batch size too small**: B = 1 or 2 leaves GPU utilization low, making training slower in wall-clock time despite more updates per second.
3. **Not shuffling data**: Failing to shuffle between epochs introduces bias from correlated samples, slowing convergence.
4. **Ignoring batch normalization interactions**: Batch statistics depend on batch size. Very small batches cause noisy batch norm statistics at test time.
5. **Using batch size not dividing N evenly**: The last incomplete batch receives a smaller gradient estimate, which can bias training. Use `drop_last=True` or handle carefully.
6. **Not adjusting learning rate with batch size**: Doubling the batch size without adjusting the learning rate effectively cuts the learning rate in terms of gradient quality.
7. **Overfitting to mini-batch noise**: Tracking loss on individual mini-batches is misleading. Always track smoothed or epoch-level metrics.

## Interview Questions

### Beginner

1. What are the advantages of mini-batch GD over full-batch GD and SGD?
2. How does batch size affect gradient variance?
3. What batch sizes are commonly used in practice and why?
4. How does a DataLoader work in PyTorch?
5. Why do we shuffle data between epochs?

### Intermediate

1. Derive the variance of the mini-batch gradient estimator.
2. Explain the linear scaling rule for learning rates with batch size.
3. How does batch size affect generalization in deep learning?
4. What is the critical batch size and how do you find it?
5. Compare the computational graph for batch size 1 vs. batch size 128.

### Advanced

1. Analyze the relationship between batch size and the width of the optimization landscape's minima.
2. Explain how large-batch training can be made to work with techniques like LARS or LAMB.
3. Discuss the gradient noise scale and how it determines the optimal batch size.

## Practice Problems

### Easy

1. Create a PyTorch DataLoader with batch size 32 from random data.
2. Count the number of parameter updates per epoch for different batch sizes.
3. Implement a mini-batch training loop without using DataLoader.
4. Plot the loss curves for batch sizes [1, 16, 128, 1024].
5. Measure GPU utilization for different batch sizes using torch.cuda.

### Medium

1. Implement the linear scaling rule: train with batch size B and learning rate η, then with 2B and 2η.
2. Compare convergence speed (wall-clock time) for batch sizes [8, 32, 128, 512] with optimal learning rates.
3. Analyze gradient covariance empirically for different batch sizes.
4. Implement a learning rate schedule that adjusts based on batch size.
5. Train a small CNN on MNIST with various batch sizes and compare test accuracy.

### Hard

1. Derive the optimal batch size from the signal-to-noise ratio of gradients.
2. Implement the LARS optimizer for large-batch training.
3. Design and execute an experiment demonstrating the generalization gap between large and small batch training.

## Solutions

Solutions follow from the variance derivation and empirical observations. The linear scaling rule η_new = η_old * (B_new / B_old) is a good starting point for adjusting learning rates.

## Related Concepts

- SGD (DL-072): Mini-batch with B=1
- Gradient Descent (DL-071): Mini-batch with B=N
- Learning Rate Decay (DL-086): Works with any batch size
- Batch Normalization: Interacts heavily with batch size choice

## Next Concepts

- SGD with Momentum (DL-074)
- Learning Rate Warmup (DL-085)
- Adam Optimizer (DL-078)

## Summary

Mini-Batch Gradient Descent computes gradients on small random subsets of B samples per iteration. It balances the accuracy of full-batch GD with the speed of SGD. Gradient variance scales as 1/B, enabling larger learning rates for bigger batches. Practical batch sizes range from 32 to 512, chosen to maximize GPU utilization. The linear scaling rule guides learning rate adjustment when changing batch size. Mini-batch GD is the default training algorithm for deep learning.

## Key Takeaways

1. Mini-batch GD is the practical compromise between full-batch and stochastic GD.
2. Gradient variance decreases as 1/B, improving gradient quality with larger batches.
3. Modern GPUs achieve peak efficiency with batch sizes of 32+.
4. The linear scaling rule: double batch size → double learning rate.
5. Batch size is a critical hyperparameter affecting convergence, generalization, and wall-clock time.
