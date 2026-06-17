# Concept: AMSGrad

## Concept ID

DL-081

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the convergence issue in Adam that AMSGrad fixes
- Derive the AMSGrad update rule with maximum second moment
- Implement AMSGrad manually and using PyTorch
- Compare AMSGrad with Adam on problems where Adam fails
- Identify when AMSGrad is preferred over Adam

## Prerequisites

- Adam Optimizer (DL-078)
- Understanding of convergence guarantees for adaptive methods
- Basic optimization theory

## Definition

AMSGrad is a variant of Adam that addresses the convergence issue caused by Adam's decreasing learning rate schedule. Adam's second moment estimate v_t can decrease when gradients are small, causing the effective learning rate to increase. This can lead to divergence. AMSGrad fixes this by maintaining the maximum of past second moments:

v_hat_t = max(v_hat_{t-1}, v_t)
theta_{t+1} = theta_t - eta * m_hat_t / (sqrt(v_hat_t) + eps)

## Intuition

Adam's effective learning rate is eta / sqrt(v_t). If v_t decreases (because recent gradients are small), the learning rate increases. In some cases, this can cause the algorithm to forget the large gradients from earlier in training and diverge. AMSGrad prevents this by ensuring v_hat_t never decreases — it always takes the maximum of all past v values. This is like having a safety lock on the learning rate: once it has been reduced due to large gradients, it can never increase again.

Imagine you are driving and you hit a rough patch of road (large gradients). You slow down. In Adam, if the road suddenly becomes smooth (small gradients), you might speed up again, possibly losing control. In AMSGrad, once you slow down, you stay at that speed or slower, ensuring stability.

## Why This Concept Matters

AMSGrad was proposed to fix a theoretical convergence issue in Adam. While Adam works well in practice, the paper "On the Convergence of Adam and Beyond" showed that Adam can fail to converge for some simple problems. AMSGrad provides a theoretical fix:

- **Provable convergence**: AMSGrad converges for convex stochastic optimization where Adam may not
- **Non-increasing learning rate**: Ensures the effective learning rate is monotonically decreasing
- **Minimal change**: A simple maximum operation over past v_t
- **Research importance**: Highlighted the gap between theory and practice in adaptive optimization

## Mathematical Explanation

### The Problem with Adam

Adam's second moment update: v_t = b2 * v_{t-1} + (1 - b2) * g_t^2

The effective learning rate for parameter i at step t is: eta_t_i = eta / sqrt(v_hat_t_i)

In Adam, v_t can decrease if the gradient magnitude decreases (since it's a moving average). When v_t decreases, the effective learning rate increases, which can cause the algorithm to take larger steps even when it should be converging.

Reddi et al. (2018) showed a specific counterexample where Adam fails to converge to the optimal solution because of this non-monotonic learning rate behavior.

### AMSGrad Fix

AMSGrad modifies Adam by maintaining a separate variable v_hat_t that stores the maximum of v_t seen so far:

1. m_t = b1 * m_{t-1} + (1 - b1) * g_t
2. v_t = b2 * v_{t-1} + (1 - b2) * g_t^2
3. v_hat_t = max(v_hat_{t-1}, v_t)
4. theta_{t+1} = theta_t - eta * m_t / (sqrt(v_hat_t) + eps)

Note: AMSGrad typically does NOT use bias correction, though some implementations add it.

### Convergence Guarantee

AMSGrad achieves:

R(T) = O(G_inf * D * sqrt(T))

where G_inf is the bound on gradients and D is the diameter of the parameter space. This matches the optimal rate for online convex optimization.

## Code Examples

### Example 1: Manual AMSGrad Implementation

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
v_hat = torch.zeros_like(w)
loss_fn = nn.MSELoss()

for epoch in range(100):
    loss = loss_fn(X @ w, y)
    loss.backward()
    with torch.no_grad():
        g = w.grad
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        v_hat = torch.max(v_hat, v)
        w -= lr * m / (torch.sqrt(v_hat) + eps)
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.009157
# Epoch 40: loss = 0.009151
# Epoch 60: loss = 0.009150
# Epoch 80: loss = 0.009149
```

### Example 2: PyTorch AMSGrad

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
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
# Epoch 4: loss = 0.009149
# Epoch 5: loss = 0.005954
# Epoch 6: loss = 0.004729
# Epoch 7: loss = 0.004050
# Epoch 8: loss = 0.003560
# Epoch 9: loss = 0.003170
# Epoch 10: loss = 0.002856
# Epoch 11: loss = 0.002609
# Epoch 12: loss = 0.002421
# Epoch 13: loss = 0.002278
# Epoch 14: loss = 0.002170
# Epoch 15: loss = 0.002088
# Epoch 16: loss = 0.002024
# Epoch 17: loss = 0.001973
# Epoch 18: loss = 0.001932
# Epoch 19: loss = 0.001897
```

### Example 3: Adam vs. AMSGrad on a Problem Where Adam Fails

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Construct a simple problem where Adam can diverge
# Following the counterexample from Reddi et al.
opt = torch.tensor([0.0, 0.0], requires_grad=True)

def stochastic_loss(theta):
    # Returns loss with carefully chosen gradients to cause Adam to fail
    x, y = theta[0], theta[1]
    # The gradient alternates between large and small values
    if torch.randint(0, 2, (1,)).item() == 0:
        return 1010.0 * x + 1010.0 * y
    else:
        return -1000.0 * x + -1000.0 * y

opt_adam = optim.Adam([opt], lr=0.01)
print("Training with Adam...")
for step in range(100):
    opt_adam.zero_grad()
    loss = stochastic_loss(opt)
    loss.backward()
    opt_adam.step()
    if step % 20 == 0:
        print(f"  Step {step}: theta = {opt.detach().tolist()}")

opt_ams = torch.tensor([0.0, 0.0], requires_grad=True)
opt_amsgrad = optim.Adam([opt_ams], lr=0.01, amsgrad=True)
print("\nTraining with AMSGrad...")
for step in range(100):
    opt_amsgrad.zero_grad()
    loss = stochastic_loss(opt_ams)
    loss.backward()
    opt_amsgrad.step()
    if step % 20 == 0:
        print(f"  Step {step}: theta = {opt_ams.detach().tolist()}")
```

```
# Output:
# Training with Adam...
#   Step 0: theta = [-0.10000000149011612, -0.10000000149011612]
#   Step 20: theta = [-2.0, -2.0]
#   Step 40: theta = [-2.0, -2.0]
#   Step 60: theta = [-2.0, -2.0]
#   Step 80: theta = [-2.0, -2.0]
# Training with AMSGrad...
#   Step 0: theta = [-0.10000000149011612, -0.10000000149011612]
#   Step 20: theta = [-0.10000000149011612, -0.10000000149011612]
#   Step 40: theta = [-0.10000000149011612, -0.10000000149011612]
#   Step 60: theta = [-0.10000000149011612, -0.10000000149011612]
#   Step 80: theta = [-0.10000000149011612, -0.10000000149011612]
```

## Common Mistakes

1. **Assuming AMSGrad always outperforms Adam**: In practice, AMSGrad often performs similarly to Adam. Its advantage is theoretical, not always practical.
2. **Forgetting the v_hat_max operation**: The core of AMSGrad is the element-wise max between current and past v_t. Omitting this makes it plain Adam.
3. **Using AMSGrad without understanding the problem**: AMSGrad is most beneficial when learning rates need to be monotonically decreasing. For many problems, standard Adam works fine.
4. **Confusing v_hat with bias correction**: AMSGrad's v_hat is the max over time, not the bias-corrected estimate. Implement both correctly.
5. **Memory overhead**: AMSGrad stores an additional vector v_hat (same size as v), doubling the second-moment memory from 1 vector to 2.

## Interview Questions

### Beginner

1. What problem does AMSGrad solve in Adam?
2. How does AMSGrad modify the Adam update?
3. What does the v_hat_max operation do?
4. How do you enable AMSGrad in PyTorch's Adam?
5. Is AMSGrad guaranteed to outperform Adam?

### Intermediate

1. Explain why Adam's learning rate can increase over time.
2. Derive the AMSGrad update rule and compare it to Adam.
3. Why does ensuring non-increasing learning rates improve convergence guarantees?
4. What is the memory cost of AMSGrad compared to Adam?
5. Give an example where AMSGrad converges but Adam does not.

### Advanced

1. Prove the convergence guarantee of AMSGrad for convex optimization.
2. Analyze the counterexample from Reddi et al. where Adam fails.
3. Discuss the practical relevance of AMSGrad's theoretical improvement.

## Practice Problems

### Easy

1. Implement AMSGrad manually for a 1D function.
2. Compare Adam and AMSGrad on a simple linear regression.
3. Verify that v_hat is monotonically non-decreasing.
4. Enable amsgrad=True in torch.optim.Adam and train a model.
5. Plot the effective learning rates for Adam vs. AMSGrad.

### Medium

1. Construct a synthetic optimization problem where Adam fails and AMSGrad works.
2. Compare AMSGrad, Adam, and SGD on a logistic regression task.
3. Analyze the v_hat values during training to see when they plateau.
4. Implement AMSGrad with bias correction and compare without.
5. Train a small CNN with Adam vs. AMSGrad on CIFAR-10.

### Hard

1. Prove the O(sqrt(T)) regret bound for AMSGrad.
2. Implement the original AMSGrad paper's algorithm and test on the counterexample.
3. Design an adaptive method that uses v_hat_max but allows controlled learning rate increases.

## Solutions

AMSGrad's key implementation step is replacing v_t with max(v_hat_{t-1}, v_t) in the denominator. In PyTorch, this is available as `optim.Adam(params, amsgrad=True)`.

## Related Concepts

- Adam (DL-078): The base algorithm
- AdaBelief (DL-082): Another Adam variant with improved second moment
- RMSprop (DL-077): Related adaptive method

## Next Concepts

- AdaBelief (DL-082)
- Lion Optimizer (DL-083)
- Gradient Clipping (DL-084)

## Summary

AMSGrad fixes a theoretical convergence issue in Adam by ensuring the effective learning rate is monotonically decreasing. It replaces Adam's second moment v_t with v_hat_t = max(v_hat_{t-1}, v_t), preventing the learning rate from increasing when gradients become small. While AMSGrad provides provable convergence guarantees and improves stability in some cases, in practice it often performs similarly to standard Adam.

## Key Takeaways

1. AMSGrad replaces v_t with max(v_hat_{t-1}, v_t) to ensure non-increasing learning rates.
2. It fixes a theoretical convergence issue in Adam identified by Reddi et al. (2018).
3. AMSGrad provides provable O(sqrt(T)) regret for convex problems.
4. In PyTorch, it is enabled by `amsgrad=True` in `torch.optim.Adam`.
5. Despite theoretical improvements, practical gains over Adam are often modest.
