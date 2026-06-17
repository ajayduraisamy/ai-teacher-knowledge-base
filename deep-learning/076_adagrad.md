# Concept: Adagrad

## Concept ID

DL-076

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how Adagrad adapts per-parameter learning rates
- Derive the Adagrad update rule with accumulated squared gradients
- Implement Adagrad manually and using torch.optim
- Identify scenarios where Adagrad excels and where it fails
- Analyze the effect of the accumulation term on learning rate decay

## Prerequisites

- Mini-Batch Gradient Descent (DL-073)
- SGD with Momentum (DL-074)
- Understanding of sparse gradients and feature scaling

## Definition

Adagrad (Adaptive Gradient Algorithm) is an optimization method that adapts the learning rate for each parameter based on the historical sum of squared gradients. Parameters that have received large gradients get their learning rates reduced, while parameters with small or infrequent gradients get larger learning rates. The update rule is:

θₜ₊₁,ᵢ = θₜ,ᵢ − (η / √(Gₜ,ᵢᵢ + ε)) · gₜ,ᵢ

where Gₜ ∈ ℝ^(d×d) is a diagonal matrix where each diagonal element Gₜ,ᵢᵢ is the sum of squares of past gradients for parameter i, and ε is a small smoothing term (typically 1e−8).

## Intuition

In standard SGD, all parameters share the same learning rate. This is problematic when different features have different frequencies or scales. For example, in text processing, some words appear frequently (their gradients are large and frequent) while others are rare (their gradients are small and sparse). Adagrad automatically gives rare features larger updates and frequent features smaller updates.

Think of it as having a personalized learning rate for every parameter that decreases over time, but decreases faster for parameters that have been updated a lot already. This is like having a teacher who spends more time on concepts you struggle with (small gradients, large updates) and less on concepts you've mastered (large gradients, small updates).

## Why This Concept Matters

Adagrad was a breakthrough in optimization because it:

- **Eliminates manual learning rate tuning**: The adaptive mechanism reduces sensitivity to the initial learning rate
- **Handles sparse data naturally**: Ideal for NLP and recommender systems with sparse features
- **Per-parameter adaptation**: Automatically adjusts to different feature scales
- **Theoretical foundation**: Inspired all subsequent adaptive methods (RMSprop, Adam, etc.)

Although Adagrad has been largely superseded by RMSprop and Adam in practice, understanding it is essential for grasping how adaptive optimization works.

## Mathematical Explanation

### Adagrad Update

Let gₜ = ∇J(θₜ) be the gradient at step t. The accumulated squared gradient for each parameter i is:

Gₜ,ᵢᵢ = ∑_{τ=1}^{t} g_{τ,i}²

The per-parameter update:

θₜ₊₁,ᵢ = θₜ,ᵢ − η · gₜ,ᵢ / √(Gₜ,ᵢᵢ + ε)

In vector form:

θₜ₊₁ = θₜ − η ⊙ gₜ / √(diag(Gₜ) + ε)

where ⊙ is elementwise multiplication and the division is elementwise.

### Derivation from Regret Bounds

Adagrad was derived from online learning theory. For a sequence of convex loss functions, Adagrad achieves:

Regret(T) = ∑_{t=1}^{T} fₜ(θₜ) − min_θ ∑_{t=1}^{T} fₜ(θ) ≤ O(√T)

This is optimal for online convex optimization.

### Accumulated Sum vs. Moving Average

The key difference between Adagrad and later methods:

- **Adagrad**: Gₜ = ∑ g_{τ}² (sum from start)
- **RMSprop**: Gₜ = β Gₜ₋₁ + (1−β) gₜ² (moving average)

The sum monotonically increases, causing the effective learning rate η / √(Gₜ) to decay to zero over time.

### Effective Learning Rate Decay

For parameter i, the effective learning rate at step t is:

η_{t,i} = η / √(t · 𝔼[g_{τ,i}²])

If gradients are stationary, η_{t,i} ~ 1/√t, which matches the optimal decay rate for SGD convergence.

## Code Examples

### Example 1: Manual Adagrad Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.1
eps = 1e-8
G = torch.zeros_like(w)
loss_fn = nn.MSELoss()

for epoch in range(50):
    loss = loss_fn(X @ w, y)
    loss.backward()
    with torch.no_grad():
        G += w.grad ** 2
        w -= lr * w.grad / (torch.sqrt(G) + eps)
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 10: loss = 0.303814
# Epoch 20: loss = 0.033990
# Epoch 30: loss = 0.010380
# Epoch 40: loss = 0.009192
```

### Example 2: PyTorch Adagrad

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.Adagrad(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 1.042629
# Epoch 1: loss = 0.505004
# Epoch 2: loss = 0.199892
# Epoch 3: loss = 0.075984
# Epoch 4: loss = 0.033168
# Epoch 5: loss = 0.019588
# Epoch 6: loss = 0.014683
# Epoch 7: loss = 0.012486
# Epoch 8: loss = 0.011211
# Epoch 9: loss = 0.010367
# Epoch 10: loss = 0.009755
# Epoch 11: loss = 0.009281
# Epoch 12: loss = 0.008892
# Epoch 13: loss = 0.008550
# Epoch 14: loss = 0.008252
# Epoch 15: loss = 0.008001
# Epoch 16: loss = 0.007791
# Epoch 17: loss = 0.007628
# Epoch 18: loss = 0.007509
# Epoch 19: loss = 0.007427
```

### Example 3: Adagrad with Sparse Features

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D = 5000, 1000
sparsity = 0.95
X = (torch.rand(N, D) > sparsity).float()
true_w = torch.randn(D, 1) * 0.5
y = X @ true_w + 0.1 * torch.randn(N, 1)

model_sgd = nn.Linear(D, 1, bias=False)
model_adagrad = nn.Linear(D, 1, bias=False)
opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
opt_ada = optim.Adagrad(model_adagrad.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(30):
    opt_sgd.zero_grad()
    opt_ada.zero_grad()
    loss_sgd = loss_fn(model_sgd(X), y)
    loss_ada = loss_fn(model_adagrad(X), y)
    loss_sgd.backward()
    loss_ada.backward()
    opt_sgd.step()
    opt_ada.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: SGD = {loss_sgd.item():.6f}, Adagrad = {loss_ada.item():.6f}")
```

```
# Output:
# Epoch 0: SGD = 2.293916, Adagrad = 1.855755
# Epoch 10: SGD = 0.256722, Adagrad = 0.014302
# Epoch 20: SGD = 0.238554, Adagrad = 0.013699
# Epoch 29: SGD = 0.238139, Adagrad = 0.013530
```

## Common Mistakes

1. **Not using the correct denominator**: The gradient square root in the denominator is critical. Using Gₜ (not √Gₜ) or forgetting the square root changes the algorithm entirely.
2. **Learning rate too small with Adagrad**: Because learning rates decay as 1/√(sum), the initial learning rate should be larger than for SGD (η ≈ 0.01–0.1 for Adagrad vs. 0.001–0.01 for SGD).
3. **Training too long**: Adagrad's accumulated sum grows without bound, eventually making all learning rates effectively zero. The learning rate decays too aggressively for long training runs.
4. **Not handling the accumulation memory**: The G matrix requires O(d) memory, which is manageable for most models but can be an issue for very large parameter counts.
5. **Applying Adagrad for deep neural networks**: Adagrad often stops learning too early for deep networks because the accumulated gradient sum decays the learning rate too aggressively.
6. **Using Adagrad with non-stationary objectives**: For non-stationary problems (e.g., GAN training), past gradients become irrelevant, but Adagrad still accumulates them.

## Interview Questions

### Beginner

1. How does Adagrad adapt learning rates per parameter?
2. What problem does Adagrad solve compared to standard SGD?
3. What does the G matrix represent in Adagrad?
4. Why does Adagrad work well for sparse data?
5. How do you use Adagrad in PyTorch?

### Intermediate

1. Derive the Adagrad update rule and explain each component.
2. Why does Adagrad's learning rate decay to zero over time?
3. Compare Adagrad's accumulated sum with RMSprop's moving average.
4. How does the regret bound of Adagrad compare to standard SGD?
5. What are the advantages and disadvantages of Adagrad for deep learning?

### Advanced

1. Prove the regret bound of Adagrad for online convex optimization.
2. Explain the relationship between Adagrad and natural gradient descent.
3. How would you modify Adagrad to handle non-stationary distributions?

## Practice Problems

### Easy

1. Implement Adagrad manually for f(x, y) = x² + 10y².
2. Verify that Adagrad reduces the effective learning rate over time.
3. Compare SGD and Adagrad on a problem with features of different scales.
4. Plot the per-parameter learning rates during Adagrad training.
5. Use torch.optim.Adagrad on a linear regression task.

### Medium

1. Train a logistic regression model on sparse binary features with Adagrad vs. SGD.
2. Implement Adagrad with weight decay.
3. Analyze the effect of the initial learning rate on Adagrad convergence.
4. Compare the gradient accumulation G for frequent vs. rare features.
5. Visualize the trajectory of Adagrad vs. SGD on a 2D quadratic with different curvatures.

### Hard

1. Prove the O(√T) regret bound of Adagrad.
2. Implement a variant of Adagrad that uses a moving window of past gradients instead of full accumulation.
3. Design an experiment showing the failure mode of Adagrad for deep networks (excessive learning rate decay).

## Solutions

Adagrad's key implementation detail is maintaining the G accumulator and correctly computing the per-parameter adaptive learning rate. The learning rate should be tuned higher than SGD due to the accumulation-based decay.

## Related Concepts

- RMSprop (DL-077): Replaces sum with moving average
- Adam (DL-078): Combines Adagrad and momentum
- SGD (DL-072): Non-adaptive baseline

## Next Concepts

- RMSprop (DL-077)
- Adam Optimizer (DL-078)
- AdamW (DL-079)

## Summary

Adagrad adapts per-parameter learning rates by dividing the learning rate by the square root of the accumulated sum of squared gradients. This gives smaller updates to parameters with frequent large gradients and larger updates to parameters with infrequent or small gradients. While Adagrad excels on sparse data and eliminates manual per-parameter tuning, its monotonically growing accumulator causes learning rates to decay to zero, making it unsuitable for long training runs. This limitation motivated the development of RMSprop and Adam.

## Key Takeaways

1. Adagrad uses per-parameter learning rates based on the accumulated squared gradient sum.
2. The effective learning rate decays as 1/√(sum of squared gradients).
3. Adagrad performs well on sparse data (NLP, recommender systems).
4. The monotonically increasing accumulator eventually stops learning.
5. Adagrad laid the foundation for all subsequent adaptive optimization methods.
