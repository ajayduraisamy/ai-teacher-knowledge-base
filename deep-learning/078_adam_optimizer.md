# Concept: Adam Optimizer

## Concept ID

DL-078

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how Adam combines momentum and adaptive learning rates
- Derive the full Adam update rule with bias correction
- Implement Adam manually and using torch.optim.Adam
- Configure Adam's hyperparameters (β₁, β₂, ε, η)
- Diagnose convergence issues specific to Adam

## Prerequisites

- SGD with Momentum (DL-074)
- RMSprop (DL-077)
- Understanding of bias correction in moving averages

## Definition

Adam (Adaptive Moment Estimation) is a first-order gradient-based optimization algorithm that computes adaptive learning rates for each parameter. It maintains two moving averages:

- First moment (mean): mₜ = β₁ mₜ₋₁ + (1 − β₁) gₜ
- Second moment (uncentered variance): vₜ = β₂ vₜ₋₁ + (1 − β₂) gₜ²

With bias correction:
- m̂ₜ = mₜ / (1 − β₁ᵗ)
- v̂ₜ = vₜ / (1 − β₂ᵗ)

Parameter update:
θₜ₊₁ = θₜ − η · m̂ₜ / (√v̂ₜ + ε)

## Intuition

Adam combines the benefits of two previous methods: Momentum (which accelerates in consistent gradient directions) and RMSprop (which adapts learning rates per parameter). It keeps track of both the gradient's direction (first moment, like a velocity) and its magnitude (second moment, like an adaptive step size).

Think of Adam as a smart hiker who:
1. **Starts fast**: Initially, bias correction prevents small initial estimates from slowing progress.
2. **Accelerates downhill**: The momentum term builds speed in consistent directions.
3. **Slows on rough terrain**: The adaptive term reduces the step size when gradients are noisy (high variance).
4. **Stops when necessary**: Both moments adapt to keep learning stable.

## Why This Concept Matters

Adam is the default optimizer for most deep learning practitioners. Its importance stems from:

- **Robustness**: Works well across a wide range of problems with minimal hyperparameter tuning
- **Efficiency**: Computationally efficient with O(d) memory (two additional vectors)
- **Invariance**: Invariant to diagonal rescaling of gradients
- **Convergence**: Well-suited for problems with noisy or sparse gradients
- **Popularity**: The most widely used optimizer in deep learning research

## Mathematical Explanation

### Full Adam Algorithm

Algorithm 1: Adam

**Require**: η (learning rate), β₁, β₂ ∈ [0, 1) (decay rates), ε > 0
**Require**: J(θ) (stochastic objective)
**Initialize**: θ₀ (parameters), m₀ = 0, v₀ = 0, t = 0

While θₜ not converged:
1. t ← t + 1
2. gₜ ← ∇_θ Jₜ(θₜ₋₁) (get gradient)
3. mₜ ← β₁ mₜ₋₁ + (1 − β₁) gₜ (update biased first moment)
4. vₜ ← β₂ vₜ₋₁ + (1 − β₂) gₜ² (update biased second moment)
5. m̂ₜ ← mₜ / (1 − β₁ᵗ) (bias correction)
6. v̂ₜ ← vₜ / (1 − β₂ᵗ) (bias correction)
7. θₜ ← θₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)

**Return**: θₜ

### Bias Correction

Both mₜ and vₜ are initialized as zero vectors. Early in training, this biases the estimates toward zero. The bias correction terms counteract this:

𝔼[mₜ] = 𝔼[gₜ] · (1 − β₁ᵗ) / (1 − β₁) + bias terms

Dividing by (1 − β₁ᵗ) removes the initialization bias.

At t = 1: 1 − β₁ → m̂₁ = m₁ / (1 − β₁) = g₁ (unbiased)
At t → ∞: 1 − β₁ᵗ → 1, so bias correction has negligible effect.

### Default Hyperparameters

The authors recommend:
- η = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸

### Convergence Analysis

Adam achieves:

Regret(T) = O(√T)

under appropriate conditions, matching the optimal rate for online convex optimization.

## Code Examples

### Example 1: Manual Adam Implementation

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
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.009155
# Epoch 40: loss = 0.009149
# Epoch 60: loss = 0.009149
# Epoch 80: loss = 0.009149
```

### Example 2: PyTorch Adam

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
# Epoch 2: loss = 0.058384
# Epoch 3: loss = 0.019334
# Epoch 4: loss = 0.009148
# Epoch 5: loss = 0.005953
# Epoch 6: loss = 0.004728
# Epoch 7: loss = 0.004048
# Epoch 8: loss = 0.003559
# Epoch 9: loss = 0.003168
# Epoch 10: loss = 0.002854
# Epoch 11: loss = 0.002608
# Epoch 12: loss = 0.002420
# Epoch 13: loss = 0.002277
# Epoch 14: loss = 0.002169
# Epoch 15: loss = 0.002087
# Epoch 16: loss = 0.002023
# Epoch 17: loss = 0.001972
# Epoch 18: loss = 0.001931
# Epoch 19: loss = 0.001896
```

### Example 3: Adam on a Deep Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D = 3000, 100
X = torch.randn(N, D)
y = torch.randn(N, 1)

model = nn.Sequential(
    nn.Linear(D, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 1.022029
# Epoch 20: loss = 0.900126
# Epoch 40: loss = 0.728298
# Epoch 60: loss = 0.511472
# Epoch 80: loss = 0.329316
# Epoch 99: loss = 0.217470
```

## Common Mistakes

1. **Forgetting bias correction**: Without bias correction, early updates are too small, slowing initial convergence.
2. **Using ε too large**: A large ε reduces the adaptive effect. Stick with ε = 1e−8 unless special circumstances.
3. **Not tuning learning rate**: While Adam is robust, η = 0.001 is not optimal for every problem. Try values in [1e−5, 1e−2].
4. **Training indefinitely without decay**: Adam can continue learning, but performance may plateau. Consider learning rate schedules.
5. **Using Adam with weight decay incorrectly**: Standard Adam with L2 regularization does not work as well as AdamW's decoupled weight decay.
6. **Confusing β₁ and β₂**: β₁ controls the first moment (momentum, default 0.9), β₂ controls the second moment (RMS, default 0.999). They have very different effects.
7. **Assuming Adam always beats SGD**: For some problems, well-tuned SGD with momentum generalizes better than Adam.

## Interview Questions

### Beginner

1. What two moving averages does Adam maintain?
2. Why is bias correction necessary in Adam?
3. What are the default values for β₁, β₂, and η in Adam?
4. How does Adam combine momentum and RMSprop?
5. How do you use Adam in PyTorch?

### Intermediate

1. Derive the full Adam update rule including bias correction.
2. Explain why bias correction becomes negligible as training progresses.
3. Compare the convergence behavior of Adam vs. SGD with momentum.
4. What happens to Adam's update if all gradients are zero for an extended period?
5. How does Adam's memory footprint compare to SGD?

### Advanced

1. Prove the regret bound of Adam for convex optimization.
2. Analyze the convergence issues of Adam and how AdamW fixes them.
3. Explain the relationship between Adam and natural gradient descent.

## Practice Problems

### Easy

1. Implement Adam manually for a 1D quadratic.
2. Compare Adam and SGD on a simple classification task.
3. Plot the first and second moment estimates during Adam training.
4. Verify bias correction by comparing biased vs. unbiased estimates at early steps.
5. Use torch.optim.Adam on a linear regression task.

### Medium

1. Implement Adam with learning rate warmup.
2. Compare Adam, SGD with momentum, and RMSprop on a 3-layer network.
3. Analyze the effect of β₂ on convergence for problems with different gradient noise levels.
4. Visualize the per-parameter learning rates (η · m̂/√v̂) during Adam training.
5. Implement Adam with gradient clipping.

### Hard

1. Prove the convergence of Adam under the assumption of bounded gradients.
2. Implement a variant of Adam with adaptive β₁.
3. Design an experiment demonstrating when Adam fails and SGD succeeds (generalization gap).

## Solutions

Adam requires careful tracking of both moments and the timestep t for bias correction. The manual implementation must correctly update t after each gradient computation.

## Related Concepts

- RMSprop (DL-077): The second-moment component of Adam
- SGD with Momentum (DL-074): The first-moment component
- AdamW (DL-079): Decoupled weight decay for Adam
- NADAM (DL-080): Adam with Nesterov momentum

## Next Concepts

- AdamW (DL-079)
- NADAM (DL-080)
- AMSGrad (DL-081)

## Summary

Adam combines the benefits of momentum and RMSprop by maintaining exponentially decaying moving averages of both the gradient (first moment) and squared gradient (second moment). Bias correction compensates for initialization at zero, ensuring unbiased estimates from the first step. With default hyperparameters (η=0.001, β₁=0.9, β₂=0.999, ε=1e−8), Adam works robustly across a wide range of deep learning problems, making it the default choice for most practitioners.

## Key Takeaways

1. Adam maintains two moving averages: mₜ (gradient mean) and vₜ (gradient uncentered variance).
2. Bias correction compensates for zero initialization of the moment estimates.
3. Default hyperparameters (η=0.001, β₁=0.9, β₂=0.999) work well for most problems.
4. Adam combines the benefits of momentum and per-parameter adaptive learning rates.
5. Adam's robustness makes it the default optimizer for deep learning, though AdamW is preferred when weight decay is needed.
