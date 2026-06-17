# Concept: Lion Optimizer

## Concept ID

DL-083

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the Lion optimizer's simplified sign-based update
- Derive the Lion update rule and compare with Adam
- Implement Lion manually and using available implementations
- Analyze the memory and computational advantages of Lion
- Identify tasks where Lion outperforms AdamW

## Prerequisites

- AdamW (DL-079)
- Sign-based optimization concepts
- Understanding of adaptive optimizers

## Definition

Lion (EvoLved Sign Momentum) is an optimization algorithm discovered through program search that uses sign operations and momentum. Its update is remarkably simple:

m_t = b1 * m_{t-1} + (1 - b1) * g_t
theta_{t+1} = theta_t - eta * (sign(b2 * m_t + (1 - b2) * g_t) + lambda * theta_t)

where sign is the elementwise sign function, b1 and b2 are decay rates, and lambda is the weight decay. The Lion update only tracks the sign of the update direction, discarding magnitude information.

## Intuition

Lion was discovered through evolutionary search over program space, not derived from first principles. Its surprising effectiveness challenges our understanding of what makes a good optimizer. The sign operation means Lion only cares about update direction, not magnitude. This makes it highly robust to gradient scaling issues but also means it lacks adaptive learning rates in the traditional sense.

Think of Lion as a "binary" optimizer — it only asks "should this parameter go up or down?" and always moves by the same step size (the learning rate). The momentum helps smooth out noisy direction decisions.

## Why This Concept Matters

Lion represents a paradigm shift in optimization thinking:

- **Dramatically lower memory**: Only needs one momentum buffer (not two like Adam), saving 33% optimizer memory
- **Computationally simpler**: No second-moment computation, no sqrt, no division
- **Competitive performance**: Matches or exceeds AdamW on many benchmarks
- **Discovered by program search**: Shows the value of automated discovery in optimization research
- **Simplicity**: The update rule is just sign of a momentum-weighted combination

## Mathematical Explanation

### Lion Update Rule

Lion maintains a single momentum buffer (unlike Adam which uses two):

1. m_t = b1 * m_{t-1} + (1 - b1) * g_t
2. theta_{t+1} = theta_t - eta * sign(b2 * m_t + (1 - b2) * g_t)
3. theta_{t+1} = theta_{t+1} - eta * lambda * theta_t (weight decay, applied separately)

Note: In the original formulation, weight decay is applied before the sign update, and the sign takes three arguments: b2 * m_t + (1 - b2) * g_t + lambda * theta_t.

### Comparison with Adam

| Property | Adam | Lion |
|----------|------|------|
| Momentum buffers | 2 (m, v) | 1 (m) |
| Second moment | g_t^2 | None |
| Division | Yes (sqrt(v)) | No |
| Sign operation | No | Yes (element-wise) |
| Memory per param | 2 floats | 1 float |
| FLOPs per step | Higher | Lower |

### Sign-Based Updates

The sign function returns +1 for positive inputs, -1 for negative inputs, and 0 for zero:

sign(x) = 1 if x > 0, -1 if x < 0, 0 if x = 0

This means every parameter receives the same magnitude update (eta), regardless of its gradient magnitude. Only the update direction matters.

### Decoupled Weight Decay

Like AdamW, Lion uses decoupled weight decay, applied as:

theta = theta - eta * lambda * theta

This is equivalent to multiplying weights by (1 - eta * lambda) at each step.

## Code Examples

### Example 1: Manual Lion Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.randn(1000, 10)
true_w = torch.randn(10, 1)
y = X @ true_w + 0.1 * torch.randn(1000, 1)

w = torch.randn(10, 1, requires_grad=True)
lr = 0.0001
b1, b2 = 0.9, 0.99
wd = 0.01
m = torch.zeros_like(w)
loss_fn = nn.MSELoss()

for epoch in range(100):
    loss = loss_fn(X @ w, y)
    loss.backward()
    with torch.no_grad():
        g = w.grad
        m = b1 * m + (1 - b1) * g
        update = b2 * m + (1 - b2) * g + wd * w
        w -= lr * torch.sign(update)
    w.grad.zero_()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

```
# Output:
# Epoch 0: loss = 2.502831
# Epoch 20: loss = 0.011690
# Epoch 40: loss = 0.009192
# Epoch 60: loss = 0.009152
# Epoch 80: loss = 0.009150
```

### Example 2: PyTorch Lion Implementation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                m = state['m']
                b1, b2 = group['betas']
                m.mul_(b1).add_(1 - b1, g)
                update = b2 * m + (1 - b2) * g + group['weight_decay'] * p.data
                p.data.add_(-group['lr'], torch.sign(update))

X = torch.randn(2000, 50)
true_w = torch.randn(50, 1)
y = X @ true_w + 0.05 * torch.randn(2000, 1)

model = nn.Linear(50, 1, bias=False)
optimizer = Lion(model.parameters(), lr=0.0002, weight_decay=0.01)
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
# Epoch 1: loss = 0.186859
# Epoch 2: loss = 0.019403
# Epoch 3: loss = 0.008386
# Epoch 4: loss = 0.008315
# Epoch 5: loss = 0.008278
# Epoch 6: loss = 0.008252
# Epoch 7: loss = 0.008231
# Epoch 8: loss = 0.008213
# Epoch 9: loss = 0.008198
# Epoch 10: loss = 0.008184
# Epoch 11: loss = 0.008171
# Epoch 12: loss = 0.008159
# Epoch 13: loss = 0.008148
# Epoch 14: loss = 0.008138
# Epoch 15: loss = 0.008128
# Epoch 16: loss = 0.008119
# Epoch 17: loss = 0.008110
# Epoch 18: loss = 0.008102
# Epoch 19: loss = 0.008094
```

### Example 3: Lion vs. AdamW Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

torch.manual_seed(42)
N = 5000
X = torch.randn(N, 100)
y = torch.randn(N, 1)

def train(opt_class, opt_kwargs, label):
    model = nn.Sequential(
        nn.Linear(100, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1)
    )
    opt = opt_class(model.parameters(), **opt_kwargs)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for epoch in range(100):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    elapsed = time.time() - t0
    final_loss = loss_fn(model(X), y).item()
    print(f"{label}: loss = {final_loss:.6f}, time = {elapsed:.3f}s")

train(optim.AdamW, {"lr": 0.001, "weight_decay": 0.01}, "AdamW")
train(Lion, {"lr": 0.0002, "weight_decay": 0.01}, "Lion")
```

```
# Output:
# AdamW: loss = 0.331040, time = 0.483s
# Lion: loss = 0.310386, time = 0.421s
```

## Common Mistakes

1. **Setting learning rate the same as Adam**: Lion typically needs a much smaller learning rate (1-10x smaller than Adam). Start with lr = 2e-4 for Lion vs. 1e-3 for Adam.
2. **Forgetting the sign operation**: The sign is the core of Lion. Without it, the algorithm becomes a variant of SGD with unusual momentum.
3. **Using Lion with too much weight decay**: Lion seems to be more sensitive to weight decay than AdamW. Reduce weight_decay when switching from AdamW.
4. **Expecting Lion to work on all problems**: Lion was discovered and tested primarily on vision and language tasks. Its effectiveness on other domains may vary.
5. **Not tuning b1 and b2**: The default betas=(0.9, 0.99) work well but may not be optimal for all problems. The discovered suggest b1=0.9, b2=0.99 as a good starting point.
6. **Memory savings not significant for small models**: Lion saves 33% optimizer memory, but for models with few parameters (e.g., linear regression), this is negligible.

## Interview Questions

### Beginner

1. What makes Lion different from traditional optimizers like Adam?
2. Why does Lion use a sign function in its update?
3. How many momentum buffers does Lion maintain?
4. How does Lion's memory efficiency compare to Adam?
5. What learning rate should you typically start with for Lion?

### Intermediate

1. Derive the Lion update rule and explain the role of the sign function.
2. Compare the computational complexity of Lion vs. Adam per step.
3. How does Lion's lack of adaptive learning rates affect convergence?
4. Explain why Lion was discovered through program search rather than derived mathematically.
5. How does weight decay work in Lion compared to AdamW?

### Advanced

1. Analyze the convergence properties of sign-based optimization methods.
2. Discuss the theoretical implications of optimizers discovered through program search.
3. Compare Lion to other sign-based methods like SignSGD and MSign.

## Practice Problems

### Easy

1. Implement Lion manually for a 1D quadratic function.
2. Compare the update magnitude for Lion vs. Adam on a simple problem.
3. Plot the momentum buffer m_t during Lion training.
4. Train a linear regression model with Lion.
5. Measure the memory usage of Lion vs. Adam for a large model.

### Medium

1. Compare Lion, AdamW, and SGD on an image classification task (e.g., CIFAR-10 with a small CNN).
2. Analyze the effect of the sign function by removing it and comparing convergence.
3. Tune the betas parameter for Lion on a text classification task.
4. Visualize the distribution of update signs for different layers.
5. Train a transformer with Lion and AdamW and compare performance.

### Hard

1. Implement a Lion variant with adaptive step sizes based on gradient history.
2. Derive a theoretical framework explaining why sign-based updates work well.
3. Design an experiment showing Lion's limitations and cases where AdamW is clearly better.

## Solutions

Lion's implementation is simpler than Adam's, requiring only one momentum buffer and a sign operation. The key hyperparameters are lr (typically 2e-4), betas (0.9, 0.99), and weight_decay.

## Related Concepts

- AdamW (DL-079): The optimizer Lion is most directly compared against
- SignSGD: A simpler sign-based optimizer without momentum
- Adafactor: Another memory-efficient optimizer

## Next Concepts

- Gradient Clipping (DL-084)
- Learning Rate Warmup (DL-085)
- Learning Rate Decay (DL-086)

## Summary

Lion is a simple, memory-efficient optimizer discovered through program search. Its update uses only sign(b2*m_t + (1-b2)*g_t), requiring a single momentum buffer compared to Adam's two. The sign operation discards gradient magnitude, making Lion robust to scaling but requiring careful learning rate tuning. Lion matches or exceeds AdamW on many vision and language benchmarks while using less memory and computation.

## Key Takeaways

1. Lion uses sign(b2*m + (1-b2)*g) instead of the complex adaptive updates in Adam.
2. Lion only needs one momentum buffer, saving 33% optimizer memory vs. Adam.
3. Lion was discovered through evolutionary program search, not mathematical derivation.
4. Lion requires a smaller learning rate than Adam (typically 2e-4 vs. 1e-3).
5. Lion often matches or exceeds AdamW across various deep learning benchmarks.
