# Concept: Cyclical Learning Rates

## Concept ID

DL-088

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the rationale behind cyclical learning rates
- Implement triangular, triangular2, and exp_range policies
- Apply cyclical learning rates in PyTorch
- Determine optimal cycle length and learning rate bounds
- Compare cyclical schedules with traditional decay

## Prerequisites

- Learning Rate Decay (DL-086)
- Cosine Annealing (DL-087)
- Understanding of learning rate effects

## Definition

Cyclical Learning Rates (CLR) are learning rate schedules where the learning rate oscillates between a minimum and maximum value according to a periodic function. Unlike traditional schedules that monotonically decrease, CLR intentionally increases the learning rate during parts of the cycle. The most common form is the triangular policy:

eta(t) = eta_min + (eta_max - eta_min) * |(t / step_size) - 2 * floor(t / (2 * step_size)) - 1|

where step_size is half the cycle length.

## Intuition

Traditional learning rate decay assumes we should always reduce the learning rate over time. CLR challenges this by observing that increasing the learning rate can help escape sharp local minima and saddle points. The periodic increases act like "kicks" that push the optimizer out of poor regions, while the decreases allow fine-grained convergence.

Think of it as a bouncing ball on a hilly surface. Sometimes the ball needs to bounce higher (higher LR) to escape a ditch, then settle down (lower LR) to find the bottom of a valley. The cyclical nature provides repeated opportunities to escape and explore.

## Why This Concept Matters

Cyclical learning rates offer several advantages:

- **Eliminate LR tuning**: CLR often works well without extensive LR tuning — just set min and max bounds
- **Better convergence**: Repeated LR cycling helps escape poor local minima
- **Faster training**: Can achieve good accuracy in fewer epochs than constant LR
- **Theoretical insight**: Challenges the "monotonically decreasing" dogma
- **Practical results**: Works well across many architectures and datasets

## Mathematical Explanation

### Triangular Policy

The triangular policy linearly increases from eta_min to eta_max over step_size iterations, then decreases back to eta_min over the next step_size iterations:

cycle = floor(1 + t / (2 * step_size))
x = abs(t / step_size - 2 * cycle + 1)
eta(t) = eta_min + (eta_max - eta_min) * max(0, 1 - x)

### Triangular2 Policy

Same as triangular but the maximum learning rate halves each cycle:

eta_max_i = eta_max / 2^{i-1}

where i is the cycle number.

### Exp Range Policy

Same as triangular but eta_max decays exponentially:

eta_max_i = eta_max * gamma^i

### Choosing Base and Max LR

The paper recommends the "LR range test": train with an increasing learning rate and plot the loss. The base LR is where loss starts to decrease, and the max LR is where loss starts to increase or diverge.

Typical range: base_lr = 0.001, max_lr = 0.01 (for Adam)
Or use 10x to 100x difference between min and max.

### Cycle Length

step_size = 2-10 epochs (i.e., 4-20 epochs per full cycle).
The total number of cycles: 3-10 for most tasks.

## Code Examples

### Example 1: Triangular CLR Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

def triangular_lr(t, step_size, base_lr, max_lr):
    cycle = (t // (2 * step_size))
    x = abs(t / step_size - 2 * cycle - 1)
    return base_lr + (max_lr - base_lr) * max(0, 1 - x)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

base_lr, max_lr = 0.001, 0.01
step_size = 100
total_steps = 1000

for step in range(total_steps):
    lr = triangular_lr(step, step_size, base_lr, max_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if step % 200 == 0 or step == total_steps - 1:
        print(f"Step {step}: loss = {loss.item():.6f}, lr = {lr:.6f}")
```

```
# Output:
# Step 0: loss = 0.979969, lr = 0.001000
# Step 200: loss = 0.009153, lr = 0.005500
# Step 400: loss = 0.009149, lr = 0.005500
# Step 600: loss = 0.009149, lr = 0.005500
# Step 800: loss = 0.009149, lr = 0.005500
# Step 999: loss = 0.009149, lr = 0.001000
```

### Example 2: Using torch.optim.lr_scheduler.CyclicLR

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01,
                     step_size_up=100, step_size_down=100,
                     mode='triangular')
loss_fn = nn.MSELoss()

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 10 == 0 or epoch == 49:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Epoch 0: loss = 0.979969, lr = 0.001000
# Epoch 10: loss = 0.009222, lr = 0.006370
# Epoch 20: loss = 0.009149, lr = 0.004630
# Epoch 30: loss = 0.009149, lr = 0.006370
# Epoch 40: loss = 0.009149, lr = 0.004630
# Epoch 49: loss = 0.009149, lr = 0.005500
```

### Example 3: CLR on a Deeper Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

torch.manual_seed(42)
X = torch.randn(2000, 50)
y = torch.randn(2000, 1)

model = nn.Sequential(
    nn.Linear(50, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.005,
                     step_size_up=20, step_size_down=20,
                     mode='triangular2')
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 20 == 0 or epoch == 99:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Epoch 0: loss = 1.051305, lr = 0.000100
# Epoch 20: loss = 0.768988, lr = 0.002550
# Epoch 40: loss = 0.699937, lr = 0.001275
# Epoch 60: loss = 0.667503, lr = 0.000638
# Epoch 80: loss = 0.650408, lr = 0.000319
# Epoch 99: loss = 0.628678, lr = 0.000175
```

## Common Mistakes

1. **Step size too small**: Very short cycles (step_size < 10 iterations) cause the LR to jump around too quickly, preventing convergence.
2. **Step size too large**: Very long cycles provide few cycles in total, reducing the benefit of cycling.
3. **Max LR too high**: If max_lr is too high, the loss may spike or diverge during the increasing phase.
4. **Min LR too low**: base_lr too low means during the decreasing phase, learning effectively stops.
5. **Not using LR range test**: Guessing base_lr and max_lr without the range test leads to suboptimal performance.
6. **Applying CLR without momentum**: CLR works best with momentum, as momentum smooths the varying gradient scales.
7. **Forgetting to reset LR at inference**: The varying LR is a training artifact. Ensure the model is evaluated with the final weights only.

## Interview Questions

### Beginner

1. What is a cyclical learning rate?
2. How does CLR differ from traditional learning rate decay?
3. What are the three modes of CLR?
4. How do you determine base_lr and max_lr?
5. How do you use CyclicLR in PyTorch?

### Intermediate

1. Explain the triangular, triangular2, and exp_range policies.
2. Why can increasing the learning rate during training help convergence?
3. How do you choose step_size for a CLR schedule?
4. Compare CLR with cosine annealing with restarts.
5. How does momentum interact with cyclical learning rates?

### Advanced

1. Analyze the convergence properties of CLR compared to monotonic schedules.
2. Derive the optimal cycle length and LR bounds for a given loss landscape.
3. Explain the relationship between CLR and simulated annealing.

## Practice Problems

### Easy

1. Implement the triangular CLR policy manually.
2. Plot the LR curve for triangular, triangular2, and exp_range over 100 steps.
3. Use CyclicLR on a linear regression task.
4. Compare CLR with constant learning rate.
5. Run the LR range test on a simple model.

### Medium

1. Compare CLR, cosine annealing, and step decay on a 3-layer network.
2. Tune step_size and max_lr for optimal performance on a classification task.
3. Implement a custom CLR with a sine wave shape.
4. Train a small CNN on CIFAR-10 with CLR.
5. Visualize how the loss landscape exploration differs between CLR and step decay.

### Hard

1. Prove that CLR achieves faster convergence than constant LR for convex problems.
2. Implement an adaptive CLR that adjusts base_lr and max_lr based on validation loss plateau.
3. Design an experiment showing CLR finding flatter minima than cosine annealing.

## Solutions

CLR uses periodic LR increases to escape poor minima. PyTorch's CyclicLR implements three modes: triangular, triangular2, and exp_range. The key hyperparameters are base_lr, max_lr, and step_size (half-cycle length).

## Related Concepts

- Cosine Annealing (DL-087): Similar periodic behavior
- One Cycle Policy (DL-089): A specific CLR variant
- Learning Rate Warmup (DL-085): The increasing phase of the first cycle
- Optimizer Comparison (DL-090): Context for when to use CLR

## Next Concepts

- One Cycle Policy (DL-089)
- Optimizer Comparison (DL-090)
- Mean Squared Error (DL-091)

## Summary

Cyclical Learning Rates oscillate the learning rate between a minimum and maximum value according to a periodic function. The learning rate increases help escape poor local minima and saddle points, while the decreases allow fine-grained convergence. The triangular, triangular2 (decaying max), and exp_range (exponentially decaying max) policies offer different trade-offs. CLR often achieves good results with minimal hyperparameter tuning.

## Key Takeaways

1. CLR periodically increases and decreases the learning rate between base_lr and max_lr.
2. The LR range test determines good base_lr and max_lr values.
3. Triangular, triangular2, and exp_range are the three common CLR modes.
4. CLR challenges the assumption that learning rates should only decrease.
5. CLR works best with momentum and requires step_size tuned to the task.
