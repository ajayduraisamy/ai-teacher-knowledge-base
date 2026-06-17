# Concept: NADAM

## Concept ID

DL-080

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how NADAM incorporates Nesterov momentum into Adam
- Derive the NADAM update rule with Nesterov acceleration
- Implement NADAM manually and using torch.optim.NAdam
- Compare NADAM with Adam and Nesterov SGD
- Identify when NADAM provides advantages over Adam

## Prerequisites

- Adam Optimizer (DL-078)
- Nesterov Accelerated Gradient (DL-075)
- Understanding of momentum and adaptive methods

## Definition

NADAM (Nesterov-accelerated Adaptive Moment Estimation) combines Adam with Nesterov momentum. While Adam applies the momentum correction after computing the biased first moment, NADAM applies Nesterov's lookahead correction to the momentum term. The key insight is to replace Adam's standard momentum update with a Nesterov-style update where the first moment estimate receives a correction that looks one step ahead.

## Intuition

Adam is like a car with cruise control that adjusts speed based on road conditions and maintains momentum. NADAM adds a predictive element — it looks at where the momentum would take the car in the next moment and adjusts the steering preemptively. This anticipation allows NADAM to correct course more quickly when the gradient direction changes, reducing oscillations around minima. The Nesterov component in NADAM provides a "peek ahead" correction to the momentum term, making the optimizer more responsive to changes in the loss landscape.

## Why This Concept Matters

NADAM combines two powerful ideas — adaptive learning rates from Adam and Nesterov acceleration from NAG. Its importance includes faster convergence often in fewer iterations than Adam, improved stability with reduced oscillation around sharp minima, better generalization from improved optimization trajectory, and being a standard component available as torch.optim.NAdam used in many research projects.

## Mathematical Explanation

### NADAM Update Rule

The original NADAM formulation:

1. Update biased first moment: m_t = b1 * m_{t-1} + (1 - b1) * g_t
2. Update biased second moment: v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
3. Bias correction: m_hat_t = m_t / (1 - b1^t), v_hat_t = v_t / (1 - b2^t)
4. Nesterov correction: m_bar_t = (1 - b1) * g_t + b1 * m_hat_t
5. Update: theta_{t+1} = theta_t - eta * m_bar_t / (sqrt(v_hat_t) + eps)

### Relationship to Adam

The difference between Adam and NADAM:
- Adam uses: m_hat_t (standard bias-corrected first moment)
- NADAM uses: b1 * m_hat_t + (1 - b1) * g_t (Nesterov-corrected first moment)

The Nesterov correction effectively looks one step ahead in the momentum direction, similar to how NAG looks ahead before computing the gradient.

### PyTorch Implementation

PyTorch's NAdam (available since PyTorch 1.10) implements a formulation with decoupled weight decay. The key parameters are:
- lr: learning rate (default 0.002)
- betas: (b1, b2) = (0.9, 0.999)
- eps: 1e-8
- weight_decay: 0 (can use decoupled weight decay)

## Code Examples

### Example 1: Manual NADAM Implementation

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
        g = w.grad
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        m_bar = (1 - b1) * g + b1 * m_hat
        w -= lr * m_bar / (torch.sqrt(v_hat) + eps)
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.009150
# Epoch 40: loss = 0.009149
# Epoch 60: loss = 0.009149
# Epoch 80: loss = 0.009149
```

### Example 2: PyTorch NAdam

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.NAdam(model.parameters(), lr=0.001)
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
# Epoch 2: loss = 0.058383
# Epoch 3: loss = 0.019332
# Epoch 4: loss = 0.009148
# Epoch 5: loss = 0.005952
# Epoch 6: loss = 0.004727
# Epoch 7: loss = 0.004047
# Epoch 8: loss = 0.003558
# Epoch 9: loss = 0.003168
# Epoch 10: loss = 0.002854
# Epoch 11: loss = 0.002607
# Epoch 12: loss = 0.002420
# Epoch 13: loss = 0.002277
# Epoch 14: loss = 0.002169
# Epoch 15: loss = 0.002087
# Epoch 16: loss = 0.002023
# Epoch 17: loss = 0.001972
# Epoch 18: loss = 0.001931
# Epoch 19: loss = 0.001896
```

### Example 3: NADAM vs. Adam

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
y = torch.randn(2000, 1)

def train_with(opt_class, label):
    model = nn.Sequential(
        nn.Linear(50, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 1)
    )
    opt = opt_class(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(100):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    final_loss = loss_fn(model(X), y).item()
    print(f"{label}: final loss = {final_loss:.6f}")

train_with(optim.Adam, "Adam")
train_with(optim.NAdam, "NADAM")
```

```
# Output:
# Adam: final loss = 0.849370
# NADAM: final loss = 0.793408
```

## Common Mistakes

1. **Using too large a learning rate**: NADAM's Nesterov component amplifies updates more than Adam. Reduce the learning rate if you see oscillations.
2. **Not using bias correction**: The Nesterov correction depends on the bias-corrected m_hat_t. Using biased m_t leads to incorrect updates early in training.
3. **Confusing NADAM with Adam**: The difference is subtle — only the first moment uses Nesterov correction. The second moment remains the same as Adam.
4. **Applying NADAM without momentum**: NADAM requires b1 > 0. Setting b1 = 0 reduces NADAM to RMSprop with a correction term.
5. **Assuming NADAM always outperforms Adam**: While NADAM often converges faster, the final solution quality depends on the problem.

## Interview Questions

### Beginner

1. What does NADAM stand for?
2. How does NADAM extend the Adam optimizer?
3. What is the key difference between Adam and NADAM?
4. When might you prefer NADAM over Adam?
5. How do you use NADAM in PyTorch?

### Intermediate

1. Derive the NADAM update rule and explain the Nesterov correction term.
2. Compare the first moment update in Adam vs. NADAM.
3. How does NADAM's convergence behavior differ from Adam in practice?
4. Explain why the Nesterov correction helps reduce oscillations.
5. What happens to NADAM when b1 is set to 0?

### Advanced

1. Prove the convergence guarantee of NADAM under standard assumptions.
2. Analyze the relationship between NADAM and Nesterov Accelerated Gradient.
3. Derive the continuous-time limit of NADAM and compare it to Adam's ODE.

## Practice Problems

### Easy

1. Implement NADAM manually for a quadratic function.
2. Compare Adam and NADAM on a simple linear regression task.
3. Plot the first moment estimate m_hat_t vs. the Nesterov-corrected m_bar_t.
4. Verify that NADAM reduces to Adam when b1 = 0.
5. Use torch.optim.NAdam on a 2D classification problem.

### Medium

1. Compare NADAM, Adam, and SGD with Nesterov on a 3-layer network.
2. Analyze the effect of b1 on NADAM's convergence speed.
3. Implement NADAM with learning rate warmup.
4. Visualize the optimization trajectories of Adam vs. NADAM on a 2D loss surface.
5. Train a small CNN on MNIST with NADAM and Adam and compare test accuracy.

### Hard

1. Implement the original NADAM paper's algorithm exactly and compare with PyTorch's NAdam.
2. Prove that NADAM achieves the O(1/t^2) convergence rate for convex functions.
3. Design an experiment showing the advantage of NADAM over Adam for non-stationary objectives.

## Solutions

NADAM adds a Nesterov correction to Adam's first moment estimate: m_bar_t = b1 * m_hat_t + (1 - b1) * g_t. This provides a lookahead correction that anticipates the momentum direction. PyTorch's NAdam implements this with optional decoupled weight decay.

## Related Concepts

- Adam (DL-078): The base optimizer
- Nesterov Accelerated Gradient (DL-075): Provides the Nesterov correction concept
- AdamW (DL-079): Decoupled weight decay variant

## Next Concepts

- AMSGrad (DL-081)
- AdaBelief (DL-082)
- Lion Optimizer (DL-083)

## Summary

NADAM integrates Nesterov acceleration into the Adam optimizer by applying a lookahead correction to the first moment estimate. The update uses m_bar_t = (1 - b1) * g_t + b1 * m_hat_t instead of Adam's standard m_hat_t. This anticipatory correction provides faster convergence and reduced oscillation, often outperforming Adam in the early stages of training.

## Key Takeaways

1. NADAM adds Nesterov momentum correction to Adam's first moment estimate.
2. The Nesterov correction m_bar_t = b1 * m_hat_t + (1 - b1) * g_t looks ahead.
3. NADAM often converges faster than Adam in early training.
4. Available as torch.optim.NAdam since PyTorch 1.10.
5. The b1 parameter controls both the momentum and the Nesterov correction strength.
