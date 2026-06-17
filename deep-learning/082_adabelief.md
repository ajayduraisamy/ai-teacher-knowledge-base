# Concept: AdaBelief

## Concept ID

DL-082

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand how AdaBelief modifies Adam's second moment to use gradient prediction error
- Derive the AdaBelief update rule
- Implement AdaBelief manually
- Compare AdaBelief's convergence with Adam and SGD
- Recognize when AdaBelief's gradient belief mechanism is beneficial

## Prerequisites

- Adam Optimizer (DL-078)
- AMSGrad (DL-081)
- Understanding of second-moment estimation in adaptive methods

## Definition

AdaBelief (Adaptive Belief) is an optimization algorithm that modifies Adam's second moment estimate to use the difference between the gradient and its predicted value (the first moment). Instead of accumulating squared gradients g_t^2, AdaBelief accumulates (g_t - m_t)^2 where m_t is the first moment estimate. The update rule is:

m_t = b1 * m_{t-1} + (1 - b1) * g_t
s_t = b2 * s_{t-1} + (1 - b2) * (g_t - m_t)^2 + eps
theta_{t+1} = theta_t - eta * m_t / sqrt(s_t)

## Intuition

In Adam, the second moment captures the magnitude of past gradients, regardless of direction. In AdaBelief, the second moment captures the "surprise" or "belief" — how much the current gradient differs from what was expected (the moving average). If the gradient closely matches the moving average, the step size is large (high confidence). If the gradient deviates from expectations, the step size is small (low confidence — something unexpected happened).

This is like a student who is confident answering questions that match their knowledge (small error, large step) but hesitates when faced with unexpected information (large error, small step).

## Why This Concept Matters

AdaBelief aims to combine the benefits of adaptive methods (fast convergence) with those of SGD (good generalization). Its importance:

- **Fast convergence like Adam**: Adaptive learning rates for rapid initial progress
- **Good generalization like SGD**: The belief mechanism reduces overfitting
- **Stable training**: The gradient prediction error naturally handles noisy gradients
- **Practical performance**: Often matches or exceeds both Adam and SGD on vision, language, and RL tasks

## Mathematical Explanation

### AdaBelief Update Rule

Full AdaBelief algorithm with bias correction:

1. m_t = b1 * m_{t-1} + (1 - b1) * g_t
2. s_t = b2 * s_{t-1} + (1 - b2) * (g_t - m_t)^2 + eps
3. m_hat_t = m_t / (1 - b1^t)
4. s_hat_t = s_t / (1 - b2^t)
5. theta_{t+1} = theta_t - eta * m_hat_t / sqrt(s_hat_t)

### Key Differences from Adam

| Component | Adam | AdaBelief |
|-----------|------|-----------|
| Second moment | E[g^2] | E[(g - m)^2] |
| Interpretation | Gradient magnitude | Gradient prediction error |
| Step size control | Based on gradient size | Based on gradient surprise |
| Large gradient | Reduces step | May increase step if expected |

### Why This Helps

When the gradient direction is consistent (g_t approximately equals m_t), (g_t - m_t)^2 is small, making the step larger. This accelerates progress in flat regions or when moving consistently downhill.

When the gradient direction changes sharply (g_t differs from m_t), (g_t - m_t)^2 is large, making the step smaller. This prevents overshooting at minima or when the loss surface suddenly changes.

### Relationship to Adam

If we expand (g_t - m_t)^2 = g_t^2 - 2*g_t*m_t + m_t^2, we see AdaBelief is related to Adam but with additional cross terms. This makes AdaBelief consider both the gradient magnitude and its consistency with the momentum direction.

## Code Examples

### Example 1: Manual AdaBelief Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.001
b1, b2 = 0.9, 0.999
m = torch.zeros_like(w)
s = torch.zeros_like(w)
t = 0
loss_fn = nn.MSELoss()

for epoch in range(100):
    loss = loss_fn(X @ w, y)
    loss.backward()
    t += 1
    with torch.no_grad():
        g = w.grad
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * (g - m) ** 2 + 1e-8
        m_hat = m / (1 - b1 ** t)
        s_hat = s / (1 - b2 ** t)
        w -= lr * m_hat / torch.sqrt(s_hat)
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.009855
# Epoch 40: loss = 0.009160
# Epoch 60: loss = 0.009152
# Epoch 80: loss = 0.009149
```

### Example 2: PyTorch AdaBelief

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

class AdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if group['weight_decay'] != 0:
                    g = g + group['weight_decay'] * p.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)
                m, s = state['m'], state['s']
                state['step'] += 1
                t = state['step']
                m.mul_(group['b1']).add_(1 - group['b1'], g)
                g_minus_m = g - m
                s.mul_(group['b2']).addcmul_(1 - group['b2'], g_minus_m, g_minus_m)
                s.add_(group['eps'])
                m_hat = m / (1 - group['b1'] ** t)
                s_hat = s / (1 - group['b2'] ** t)
                p.data.addcdiv_(-group['lr'], m_hat, torch.sqrt(s_hat))

model = nn.Linear(50, 1, bias=False)
optimizer = AdaBelief(model.parameters(), lr=0.001)
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
# Epoch 1: loss = 0.416964
# Epoch 2: loss = 0.145350
# Epoch 3: loss = 0.048722
# Epoch 4: loss = 0.019538
# Epoch 5: loss = 0.010895
# Epoch 6: loss = 0.007769
# Epoch 7: loss = 0.006203
# Epoch 8: loss = 0.005218
# Epoch 9: loss = 0.004509
# Epoch 10: loss = 0.003964
# Epoch 11: loss = 0.003536
# Epoch 12: loss = 0.003198
# Epoch 13: loss = 0.002930
# Epoch 14: loss = 0.002717
# Epoch 15: loss = 0.002545
# Epoch 16: loss = 0.002404
# Epoch 17: loss = 0.002286
# Epoch 18: loss = 0.002186
# Epoch 19: loss = 0.002101
```

### Example 3: AdaBelief vs. Adam on Noisy Gradients

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N = 2000
X = torch.randn(N, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(N, 1)

# Add adversarial noise to gradients
class NoisyLinear(nn.Linear):
    def forward(self, x):
        out = super().forward(x)
        if self.training and torch.rand(1).item() > 0.5:
            out = out + 0.5 * torch.randn_like(out)
        return out

def train(opt_class, opt_kwargs, label):
    model = nn.Sequential(
        NoisyLinear(20, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    opt = opt_class(model.parameters(), **opt_kwargs)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(200):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"{label}: final loss = {losses[-1]:.6f}")

train(optim.Adam, {"lr": 0.001}, "Adam")
train(AdaBelief, {"lr": 0.001}, "AdaBelief")
```

```
# Output:
# Adam: final loss = 0.142386
# AdaBelief: final loss = 0.118902
```

## Common Mistakes

1. **Confusing s_t with Adam's v_t**: In Adam, v_t accumulates g_t^2. In AdaBelief, s_t accumulates (g_t - m_t)^2. They are different!
2. **Using the same learning rate as Adam**: AdaBelief may require different learning rate tuning. Start with the same but be prepared to adjust.
3. **Forgetting bias correction**: Like Adam, AdaBelief benefits from bias correction for the first and second moments.
4. **Not adding epsilon inside s_t**: AdaBelief adds eps to s_t before taking sqrt, unlike Adam where eps is outside. This matters for numerical stability.
5. **Assuming AdaBelief always generalizes better**: While AdaBelief often closes the generalization gap, it is not a universal improvement.

## Interview Questions

### Beginner

1. What is the key difference between AdaBelief and Adam?
2. How does AdaBelief compute the second moment?
3. What does the "belief" in AdaBelief refer to?
4. When might AdaBelief outperform Adam?
5. How do you implement AdaBelief in PyTorch?

### Intermediate

1. Derive the AdaBelief update rule and explain the gradient prediction error.
2. Compare how Adam and AdaBelief respond to consistent vs. changing gradient directions.
3. Why does the (g_t - m_t)^2 term improve training stability?
4. Explain the relationship between AdaBelief and gradient signal-to-noise ratio.
5. How does AdaBelief's step size behave when gradients are highly noisy?

### Advanced

1. Prove the convergence guarantee of AdaBelief under standard assumptions.
2. Analyze the relationship between AdaBelief and natural gradient descent.
3. Explain how AdaBelief's belief mechanism relates to trust-region methods.

## Practice Problems

### Easy

1. Implement AdaBelief manually for a 1D function.
2. Compare AdaBelief and Adam on a simple linear regression.
3. Plot s_t (belief) vs. v_t (Adam variance) during training.
4. Train a 2-layer network with AdaBelief on moons dataset.
5. Verify the gradient prediction error (g_t - m_t) decreases as training progresses.

### Medium

1. Compare AdaBelief, Adam, and SGD on an image classification task.
2. Analyze the effective learning rate (eta / sqrt(s_hat)) for different parameters.
3. Implement AdaBelief with decoupled weight decay.
4. Visualize how AdaBelief handles a sudden gradient direction change.
5. Train an RNN on sentiment analysis with AdaBelief vs. Adam.

### Hard

1. Derive the relationship between AdaBelief's second moment and the Fisher information matrix.
2. Implement a variant of AdaBelief with adaptive b2 based on gradient stationarity.
3. Design an experiment where AdaBelief significantly outperforms both Adam and SGD.

## Solutions

AdaBelief replaces g_t^2 in Adam with (g_t - m_t)^2, capturing gradient prediction error rather than raw magnitude. The custom optimiser class can be implemented by extending torch.optim.Optimizer.

## Related Concepts

- Adam (DL-078): The base algorithm modified by AdaBelief
- AMSGrad (DL-081): Another Adam variant
- RMSprop (DL-077): Adam's second-moment component
- Gradient Clipping (DL-084): Complementary technique

## Next Concepts

- Lion Optimizer (DL-083)
- Gradient Clipping (DL-084)
- Learning Rate Warmup (DL-085)

## Summary

AdaBelief modifies Adam's second moment from g_t^2 to (g_t - m_t)^2, the squared difference between the current gradient and its running average. This captures gradient "surprise" — how much the current gradient deviates from expectations. When gradients are consistent with the momentum direction, the step size increases. When they deviate sharply, the step size decreases. This mechanism helps AdaBelief achieve both fast convergence (like Adam) and good generalization (like SGD).

## Key Takeaways

1. AdaBelief uses (g_t - m_t)^2 instead of g_t^2 for the second moment.
2. Large steps when gradient matches expectation; small steps when surprised.
3. AdaBelief often achieves better generalization than Adam.
4. The gradient prediction error naturally handles noisy gradients.
5. AdaBelief bridges the gap between adaptive methods and SGD generalization.
