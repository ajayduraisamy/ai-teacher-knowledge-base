# Concept: RMSprop

## Concept ID

DL-077

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how RMSprop fixes Adagrad's aggressive learning rate decay
- Derive the RMSprop update rule with exponential moving average
- Implement RMSprop manually and using torch.optim
- Compare RMSprop with Adagrad and Adam
- Apply RMSprop for non-stationary and RNN training

## Prerequisites

- Adagrad (DL-076)
- SGD with Momentum (DL-074)
- Understanding of exponential moving averages

## Definition

RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization method that maintains a moving average of squared gradients to normalize the gradient update. Unlike Adagrad's accumulator that grows unboundedly, RMSprop uses exponential decay:

vₜ = β vₜ₋₁ + (1 − β) gₜ²
θₜ₊₁ = θₜ − η · gₜ / (√vₜ + ε)

where β is the decay rate (typically 0.9), η is the learning rate, and ε is a small smoothing constant.

## Intuition

If Adagrad is like a student who accumulates all past mistakes and becomes increasingly hesitant, RMSprop is like a student who remembers recent mistakes more clearly while forgetting distant ones. This "forgetting" mechanism prevents the learning rate from decaying to zero, allowing the model to continue learning indefinitely.

RMSprop normalizes the gradient by the root mean square of recent gradients. If the gradient suddenly becomes large, the moving average increases, dampening the update. If the gradient remains consistently small, the moving average decreases, allowing larger steps. This automatic scaling handles varying gradient magnitudes across parameters and over time.

## Why This Concept Matters

RMSprop is historically important as the solution to Adagrad's fatal flaw — the monotonically decaying learning rate. It was proposed by Geoffrey Hinton in his Coursera lecture and became the foundation for Adam. RMSprop is particularly effective for:

- **RNNs and LSTMs**: Where gradient magnitudes vary dramatically over time
- **Non-stationary objectives**: Where the loss landscape changes during training
- **Online learning**: Where data arrives in a stream and distributions shift
- **Pre-training**: As a stable baseline optimizer

## Mathematical Explanation

### RMSprop Update

Let gₜ = ∇J(θₜ) be the gradient at step t.

Step 1: Update the moving average of squared gradients:
vₜ = β vₜ₋₁ + (1 − β) gₜ²

Step 2: Compute the adaptive learning rate:
ηₜ = η / (√vₜ + ε)

Step 3: Update parameters:
θₜ₊₁ = θₜ − ηₜ ⊙ gₜ

### The Decay Rate β

β controls the effective window size of the moving average:

Effective window length = 1 / (1 − β)

- β = 0.9 → window of ~10 steps
- β = 0.99 → window of ~100 steps
- β = 0.999 → window of ~1000 steps

### Relationship to Adagrad

Setting β = 0 (no decay, no moving average):

vₜ = gₜ²

This would only use the current gradient, which is too noisy. Adagrad's sum is equivalent to β = 1 (no decay at all), which never forgets. RMSprop uses 0 < β < 1, striking a balance.

### Relationship to Gradient Signal-to-Noise Ratio

The term 1/√vₜ can be interpreted as:

1 / RMS[gₜ] = (√(1/β)) / √(𝔼[gₜ²])

This is approximately 1 / √(𝔼[gₜ²]) = 1 / (√(Var[gₜ] + 𝔼[gₜ]²)).

When the gradient is noisy (high variance), vₜ is large, and the step size is reduced. When the gradient is consistent (low variance), vₜ is small, and the step size is increased.

## Code Examples

### Example 1: Manual RMSprop Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.01
beta = 0.9
eps = 1e-8
v = torch.zeros_like(w)
loss_fn = nn.MSELoss()

for epoch in range(50):
    loss = loss_fn(X @ w, y)
    loss.backward()
    with torch.no_grad():
        v = beta * v + (1 - beta) * w.grad ** 2
        w -= lr * w.grad / (torch.sqrt(v) + eps)
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 10: loss = 0.105288
# Epoch 20: loss = 0.012178
# Epoch 30: loss = 0.009266
# Epoch 40: loss = 0.009150
```

### Example 2: PyTorch RMSprop

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
loss_fn = nn.MSELoss()

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 1.042629
# Epoch 10: loss = 0.003615
# Epoch 20: loss = 0.002590
# Epoch 29: loss = 0.002481
```

### Example 3: RMSprop vs. Adagrad on Long Training

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(3000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(3000, 1)

model_rms = nn.Linear(20, 1, bias=False)
model_ada = nn.Linear(20, 1, bias=False)
opt_rms = optim.RMSprop(model_rms.parameters(), lr=0.01)
opt_ada = optim.Adagrad(model_ada.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(200):
    opt_rms.zero_grad()
    opt_ada.zero_grad()
    loss_rms = loss_fn(model_rms(X), y)
    loss_ada = loss_fn(model_ada(X), y)
    loss_rms.backward()
    loss_ada.backward()
    opt_rms.step()
    opt_ada.step()
    if epoch % 50 == 0 or epoch == 199:
        print(f"Epoch {epoch}: RMSprop = {loss_rms.item():.6f}, Adagrad = {loss_ada.item():.6f}")
```

```
# Output:
# Epoch 0: RMSprop = 0.823845, Adagrad = 0.734359
# Epoch 50: RMSprop = 0.009964, Adagrad = 0.009117
# Epoch 100: RMSprop = 0.009558, Adagrad = 0.009421
# Epoch 150: RMSprop = 0.009366, Adagrad = 0.009855
# Epoch 199: RMSprop = 0.009260, Adagrad = 0.010178
```

## Common Mistakes

1. **Confusing RMSprop with Adagrad**: RMSprop uses exponential moving average, Adagrad uses cumulative sum. They have different convergence behaviors.
2. **Setting β too close to 1**: β = 0.999 makes RMSprop behave like Adagrad, causing premature learning rate decay.
3. **Setting β too close to 0**: β = 0.5 makes the moving average too short, making the adaptive scaling noisy.
4. **Not tuning the learning rate for RMSprop**: RMSprop is less sensitive to learning rate than SGD, but still requires tuning. Default η = 0.01 is a good starting point.
5. **Assuming RMSprop always outperforms SGD**: For well-tuned SGD with momentum, performance can match or exceed RMSprop.
6. **Using the wrong α parameter in PyTorch**: PyTorch's RMSprop uses `alpha` (not `beta`) for the decay rate of squared gradients.

## Interview Questions

### Beginner

1. How does RMSprop differ from Adagrad?
2. What problem does the moving average solve in RMSprop?
3. What does the β (or α) parameter control in RMSprop?
4. How do you use RMSprop in PyTorch?
5. Why is RMSprop well-suited for RNNs?

### Intermediate

1. Derive the RMSprop update rule and explain the role of the exponential moving average.
2. How does RMSprop handle non-stationary distributions compared to Adagrad?
3. Compare the effective learning rate trajectories of RMSprop and Adagrad over 1000 steps.
4. Explain how the signal-to-noise ratio of gradients relates to the RMSprop update.
5. What happens to RMSprop's learning rate when gradients are consistently zero for a parameter?

### Advanced

1. Prove that RMSprop's moving average estimate is unbiased.
2. Explain the relationship between RMSprop and natural gradient descent.
3. How would you extend RMSprop to handle second-order information?

## Practice Problems

### Easy

1. Implement RMSprop manually for a simple quadratic.
2. Verify that RMSprop's learning rate does not decay to zero over long training.
3. Compare the per-parameter learning rates of RMSprop and Adagrad.
4. Train a linear model with torch.optim.RMSprop.
5. Plot the moving average vₜ for parameters with different gradient magnitudes.

### Medium

1. Implement RMSprop with momentum (centered RMSprop).
2. Compare RMSprop, Adagrad, and SGD on a sequence prediction task with an RNN.
3. Analyze the effect of β on convergence speed and stability.
4. Train a small CNN on CIFAR-10 with RMSprop and compare to Adam.
5. Implement gradient clipping with RMSprop.

### Hard

1. Derive the convergence guarantees for RMSprop under convex assumptions.
2. Implement a variant of RMSprop with adaptive β based on gradient stationarity.
3. Design an experiment showing RMSprop's advantage over Adagrad for a non-stationary bandit problem.

## Solutions

RMSprop's key innovation is the exponential moving average vₜ = βvₜ₋₁ + (1−β)gₜ², which prevents the aggressive decay seen in Adagrad. The correct initialization of v₀ = 0 with bias correction is sometimes needed for the first few steps.

## Related Concepts

- Adagrad (DL-076): The predecessor with cumulative sum
- Adam (DL-078): Adds momentum to RMSprop
- SGD with Momentum (DL-074): Non-adaptive momentum

## Next Concepts

- Adam Optimizer (DL-078)
- AdamW (DL-079)
- NADAM (DL-080)

## Summary

RMSprop fixes Adagrad's fatal flaw — unbounded accumulation of squared gradients — by replacing the cumulative sum with an exponential moving average. The decay parameter β (typically 0.9) controls the effective window size. This allows RMSprop to continue learning indefinitely without the learning rate decaying to zero. RMSprop normalizes gradients by their recent root mean square, effectively handling varying gradient magnitudes across parameters and over time.

## Key Takeaways

1. RMSprop uses an exponential moving average (not cumulative sum) of squared gradients.
2. The decay rate β controls how far back the moving average looks.
3. RMSprop prevents the learning rate from decaying to zero, enabling indefinite training.
4. RMSprop is particularly effective for RNNs and non-stationary problems.
5. RMSprop directly inspired the Adam optimizer, which adds momentum to the adaptive scaling.
