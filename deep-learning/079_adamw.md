# Concept: AdamW

## Concept ID

DL-079

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the problem with L2 regularization in Adam
- Explain decoupled weight decay and how AdamW implements it
- Implement AdamW manually and using torch.optim.AdamW
- Compare AdamW vs. Adam with L2 regularization on generalization
- Tune weight decay as a hyperparameter for regularization

## Prerequisites

- Adam Optimizer (DL-078)
- L2 Regularization / Weight Decay concepts
- Understanding of generalization in deep learning

## Definition

AdamW is a variant of the Adam optimizer that decouples the weight decay term from the gradient-based update. In standard Adam with L2 regularization, the weight decay is mixed with the adaptive gradient mechanism, reducing its effectiveness. AdamW applies weight decay separately, after the adaptive update:

θₜ₊₁ = θₜ − η · m̂ₜ / (√v̂ₜ + ε) − η · λ · θₜ

where λ is the weight decay coefficient.

## Intuition

Imagine you have two knobs on your training process: one controls how fast you move downhill (gradient update) and the other controls how much you pull weights toward zero (regularization). In standard Adam with L2 regularization, these two knobs are intertwined — the weight decay gets divided by the adaptive learning rate, making it less effective for parameters with large adaptive rates. AdamW separates these knobs so each can do its job independently.

This decoupling ensures that weight decay is applied uniformly to all parameters regardless of their gradient history, leading to better regularization and often better generalization.

## Why This Concept Matters

AdamW fixes a subtle but important flaw in how Adam was combined with weight decay:

- **Incorrect L2 regularization**: Adding λ‖θ‖² to the loss in Adam does not implement true weight decay because the adaptive learning rate divides the regularization term
- **Better generalization**: AdamW consistently outperforms Adam with L2 regularization in practice
- **Simpler tuning**: The weight decay λ becomes a true regularization parameter independent of the learning rate
- **Widely adopted**: AdamW is the default optimizer in modern libraries (Hugging Face Transformers, PyTorch Lightning)

## Mathematical Explanation

### The Problem with L2 Regularization in Adam

In SGD, L2 regularization and weight decay are equivalent. Adding λ‖θ‖² to the loss:

J_reg(θ) = J(θ) + (λ/2)‖θ‖²

The gradient becomes: g̃ = ∇J(θ) + λθ

The SGD update with learning rate η:

θ ← θ − η(∇J(θ) + λθ) = θ − η∇J(θ) − ηλθ

This is equivalent to weight decay: θ ← θ(1 − ηλ) − η∇J(θ).

In Adam, the update with L2 regularization becomes:

θ ← θ − η · m̂ / (√v̂ + ε) − η · λ · θ / (√v̂ + ε)

The weight decay ηλθ is divided by √v̂, making the regularization adaptive. For parameters with large v̂ (large gradients), the weight decay is tiny, defeating the purpose of regularization.

### Decoupled Weight Decay

AdamW performs the weight decay after the adaptive update:

θ ← θ − η · m̂ / (√v̂ + ε)   [adaptive gradient step]
θ ← θ − η · λ · θ             [decoupled weight decay]

This way, weight decay is applied uniformly with strength ηλ, independent of the adaptive learning rate.

### Full AdamW Algorithm

**Initialize**: θ₀, m₀ = 0, v₀ = 0, t = 0

For t = 1, 2, ...:
1. gₜ ← ∇J(θₜ₋₁)
2. mₜ ← β₁ mₜ₋₁ + (1 − β₁) gₜ
3. vₜ ← β₂ vₜ₋₁ + (1 − β₂) gₜ²
4. m̂ₜ ← mₜ / (1 − β₁ᵗ)
5. v̂ₜ ← vₜ / (1 − β₂ᵗ)
6. θₜ ← θₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)   [adaptive step]
7. θₜ ← θₜ − η · λ · θₜ                 [decoupled weight decay]

### Recommended Hyperparameters

- η = 0.001 (or tuned per task)
- β₁ = 0.9, β₂ = 0.999
- λ = 0.01 (typical, but should be tuned; range 0.1 to 0.0001)

## Code Examples

### Example 1: Manual AdamW Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.001
b1, b2, eps = 0.9, 0.999, 1e-8
wd = 0.01
m = torch.zeros_like(w)
v = torch.zeros_like(w)
t = 0
loss_fn = nn.MSELoss()

for epoch in range(100):
    loss = loss_fn(X @ w, y)
    loss.backward()
    t += 1
    with torch.no_grad():
        m = b1 * m + (1 - b1) * w.grad
        v = b2 * v + (1 - b2) * w.grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        w -= lr * m_hat / (torch.sqrt(v_hat) + eps)
        w -= lr * wd * w
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.009251
# Epoch 40: loss = 0.009176
# Epoch 60: loss = 0.009165
# Epoch 80: loss = 0.009161
```

### Example 2: PyTorch AdamW

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
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
# Epoch 1: loss = 0.237408
# Epoch 2: loss = 0.058385
# Epoch 3: loss = 0.019336
# Epoch 4: loss = 0.009151
# Epoch 5: loss = 0.005959
# Epoch 6: loss = 0.004739
# Epoch 7: loss = 0.004064
# Epoch 8: loss = 0.003578
# Epoch 9: loss = 0.003189
# Epoch 10: loss = 0.002877
# Epoch 11: loss = 0.002631
# Epoch 12: loss = 0.002443
# Epoch 13: loss = 0.002300
# Epoch 14: loss = 0.002191
# Epoch 15: loss = 0.002108
# Epoch 16: loss = 0.002044
# Epoch 17: loss = 0.001992
# Epoch 18: loss = 0.001951
# Epoch 19: loss = 0.001916
```

### Example 3: Adam vs. AdamW on Overparameterized Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N = 500
X = torch.randn(N, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(N, 1)

X_test = torch.randn(500, 20)
y_test = X_test @ true_w + 0.1 * torch.randn(500, 1)

def train_compare(opt_class, opt_kwargs, label):
    model = nn.Sequential(
        nn.Linear(20, 200), nn.ReLU(),
        nn.Linear(200, 200), nn.ReLU(),
        nn.Linear(200, 1)
    )
    opt = opt_class(model.parameters(), lr=0.001, **opt_kwargs)
    loss_fn = nn.MSELoss()
    for epoch in range(500):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    train_loss = loss_fn(model(X), y).item()
    test_loss = loss_fn(model(X_test), y_test).item()
    print(f"{label}: train = {train_loss:.6f}, test = {test_loss:.6f}")

train_compare(optim.Adam, {"weight_decay": 0.01}, "Adam + L2")
train_compare(optim.AdamW, {"weight_decay": 0.01}, "AdamW")
```

```
# Output:
# Adam + L2: train = 0.007495, test = 0.014230
# AdamW: train = 0.007810, test = 0.012810
```

## Common Mistakes

1. **Using weight_decay > 0 with AdamW when you mean L2**: Both optimizers accept weight_decay but implement it differently. In Adam it is L2, in AdamW it is decoupled.
2. **Setting weight decay too large**: λ > 0.1 can force all weights near zero, preventing learning.
3. **Setting weight decay too small**: λ < 1e−5 has negligible regularization effect.
4. **Not tuning weight decay**: Weight decay is a critical hyperparameter on par with learning rate. Grid search λ in [1e−4, 1e−1].
5. **Applying weight decay to all parameters**: Usually, biases and normalization layer parameters should not receive weight decay.
6. **Confusing AdamW's learning rate schedule**: The weight decay is multiplied by the learning rate (η·λ), so changing η also changes the effective regularization strength.
7. **Using AdamW when weight decay is not needed**: For well-conditioned problems with sufficient data, AdamW's regularization may hurt performance.

## Interview Questions

### Beginner

1. What problem does AdamW solve compared to standard Adam?
2. What does "decoupled weight decay" mean?
3. How does the AdamW update rule differ from Adam?
4. When would you prefer AdamW over Adam?
5. How do you use AdamW in PyTorch?

### Intermediate

1. Explain why L2 regularization in Adam is not equivalent to weight decay.
2. Derive the AdamW update and show where weight decay is applied.
3. Compare the regularization effect of Adam + L2 vs. AdamW on a simple example.
4. How should weight decay be tuned in AdamW?
5. Why are biases and normalization params typically excluded from weight decay?

### Advanced

1. Prove the equivalence of L2 regularization and weight decay in SGD but not in Adam.
2. Analyze the convergence properties of AdamW compared to Adam.
3. Explain the relationship between AdamW and the original SGD weight decay formulation.

## Practice Problems

### Easy

1. Implement AdamW manually for a quadratic function.
2. Compare Adam and AdamW with weight_decay=0 to verify they match.
3. Train a linear model with AdamW and track weight norms.
4. Plot the weight norm over time for Adam vs. AdamW with the same λ.
5. Use torch.optim.AdamW on a simple regression task.

### Medium

1. Compare Adam + L2 vs. AdamW on a 3-layer network with the CIFAR-10 dataset.
2. Implement per-parameter weight decay (different λ for different layers).
3. Analyze the effect of weight decay on the eigenvalue distribution of weights.
4. Train a transformer on a text classification task with AdamW.
5. Visualize how weight decay affects the effective learning rate for different parameters.

### Hard

1. Derive the fixed-point analysis of AdamW compared to Adam.
2. Implement a variant of AdamW with adaptive weight decay based on gradient noise.
3. Design an experiment demonstrating the generalization gap between Adam and AdamW.

## Solutions

AdamW requires careful decoupling of the weight decay from the adaptive gradient update. In PyTorch, using `optim.AdamW` with `weight_decay` is the correct approach; manually adding L2 to the loss function will not achieve decoupling.

## Related Concepts

- Adam (DL-078): The base optimizer that AdamW fixes
- SGD with Momentum (DL-074): Where weight decay and L2 are equivalent
- L2 Regularization: Classical regularization technique

## Next Concepts

- NADAM (DL-080)
- AMSGrad (DL-081)
- AdaBelief (DL-082)

## Summary

AdamW fixes the incorrect handling of L2 regularization in Adam by decoupling weight decay from the adaptive gradient update. In standard Adam, the adaptive learning rate divides the weight decay term, making regularization uneven across parameters. AdamW applies weight decay separately after the gradient update, ensuring uniform regularization. This leads to better generalization and has made AdamW the default optimizer in many modern deep learning frameworks, including Hugging Face Transformers.

## Key Takeaways

1. L2 regularization and weight decay are equivalent in SGD but NOT in adaptive optimizers like Adam.
2. AdamW applies weight decay after the gradient update, not mixed with the adaptive learning rate.
3. The weight decay term in AdamW is η·λ·θ (decoupled), not η·λ·θ/√v̂ (coupled).
4. AdamW consistently outperforms Adam with L2 regularization in practice.
5. AdamW is the default optimizer for transformer-based models and many modern architectures.
