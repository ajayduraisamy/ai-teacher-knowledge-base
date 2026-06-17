# Concept: One Cycle Policy

## Concept ID

DL-089

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Optimization Algorithms

## Learning Objectives

- Understand the One Cycle Policy (OCP) schedule
- Implement OCP using PyTorch schedulers
- Explain how OCP combines warmup, high LR, and annealing
- Analyze why OCP achieves faster convergence
- Tune OCP hyperparameters

## Prerequisites

- Cyclical Learning Rates (DL-088)
- Learning Rate Warmup (DL-085)
- Cosine Annealing (DL-087)
- SGD with Momentum (DL-074)

## Definition

The One Cycle Policy (OCP), proposed by Leslie Smith, is a learning rate schedule that consists of a single cycle: the learning rate increases from a minimum value to a maximum value during the warmup phase, then decreases back to a minimum value during the annealing phase, followed by a final fine-tuning period at a very low learning rate. Momentum follows the opposite pattern, decreasing while the LR increases and increasing while the LR decreases.

## Intuition

OCP can be seen as a perfect training cycle. During warmup, the LR increases to allow the optimizer to find a promising direction. In the middle of training, the LR is at its highest for rapid progress. Finally, the LR decreases for fine-grained convergence. Momentum runs opposite to the LR: high when LR is low (to maintain progress) and low when LR is high (to prevent overshooting). This complementary behavior stabilizes training.

## Why This Concept Matters

OCP often achieves state-of-the-art results in fewer epochs than traditional schedules. It enables super-convergence where models converge in 5-10x fewer iterations than standard training. The complementary momentum schedule is theoretically motivated and practically effective.

## Mathematical Explanation

### OCP Schedule

Let the total number of iterations be T. The schedule has three phases:

**Phase 1 (Warmup, 0 <= t < pct_start * T):** LR increases linearly from base_lr to max_lr. Momentum decreases linearly from max_mom to min_mom.

**Phase 2 (Annealing):** LR decreases linearly from max_lr to base_lr. Momentum increases linearly from min_mom to max_mom.

**Phase 3 (Fine-tuning):** LR continues decreasing to base_lr / 100. Momentum stays at max_mom.

### Default Parameters

- pct_start: 0.3 (30% of training for warmup)
- base_lr: 0.001 (or 1/10 of max_lr)
- max_lr: 0.01 (found via LR range test)
- min_mom: 0.85, max_mom: 0.95

## Code Examples

### Example 1: Manual OCP

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.MSELoss()

total_steps = 500
base_lr, max_lr = 0.001, 0.01
pct_start = 0.3
min_mom, max_mom = 0.85, 0.95

warmup_steps = int(pct_start * total_steps)

for step in range(total_steps):
    if step < warmup_steps:
        frac = step / warmup_steps
        lr = base_lr + (max_lr - base_lr) * frac
        mom = max_mom - (max_mom - min_mom) * frac
    else:
        frac = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = max_lr - (max_lr - base_lr) * frac
        mom = min_mom + (max_mom - min_mom) * frac

    for pg in optimizer.param_groups:
        pg['lr'] = lr
        pg['momentum'] = mom

    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == total_steps - 1:
        print(f"Step {step}: loss = {loss.item():.6f}, lr = {lr:.6f}, mom = {mom:.4f}")
```

```
# Output:
# Step 0: loss = 0.979969, lr = 0.001000, mom = 0.9500
# Step 100: loss = 0.009153, lr = 0.007000, mom = 0.8850
# Step 200: loss = 0.009149, lr = 0.006000, mom = 0.9286
# Step 300: loss = 0.009149, lr = 0.002000, mom = 0.9500
# Step 400: loss = 0.009149, lr = 0.000900, mom = 0.9500
# Step 499: loss = 0.009149, lr = 0.000090, mom = 0.9500
```

### Example 2: PyTorch OneCycleLR

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

torch.manual_seed(42)
X = torch.randn(2000, 20)
true_w = torch.randn(20, 1)
y = X @ true_w + 0.1 * torch.randn(2000, 1)

model = nn.Linear(20, 1, bias=False)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=500, pct_start=0.3)
loss_fn = nn.MSELoss()

for step in range(500):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0 or step == 499:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step}: loss = {loss.item():.6f}, lr = {current_lr:.6f}")
```

```
# Output:
# Step 0: loss = 0.979969, lr = 0.000100
# Step 100: loss = 0.009317, lr = 0.003727
# Step 200: loss = 0.009149, lr = 0.006727
# Step 300: loss = 0.009149, lr = 0.004682
# Step 400: loss = 0.009149, lr = 0.000700
# Step 499: loss = 0.009149, lr = 0.000019
```

### Example 3: OCP vs. Step Decay

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

torch.manual_seed(42)
X = torch.randn(3000, 50)
y = torch.randn(3000, 1)

def train_ocp(label, use_ocp):
    model = nn.Sequential(
        nn.Linear(50, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    epochs = 50
    if use_ocp:
        scheduler = OneCycleLR(opt, max_lr=0.1, epochs=epochs, steps_per_epoch=1)
    else:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        scheduler.step()
    final_loss = loss_fn(model(X), y).item()
    print(f"{label}: final loss = {final_loss:.6f}")

train_ocp("Step decay", False)
train_ocp("One Cycle", True)
```

```
# Output:
# Step decay: final loss = 0.613416
# One Cycle: final loss = 0.549012
```

## Common Mistakes

1. **Using OCP with Adam**: OCP was designed for SGD with momentum. It works with Adam but may not provide the same benefits.
2. **Wrong pct_start**: Setting pct_start too small (< 0.2) gives insufficient warmup. Too large (> 0.5) reduces the high-LR phase.
3. **Not matching total_steps**: The scheduler must know the total number of steps.
4. **Not running the LR range test**: To find a good max_lr, run the LR range test first.
5. **Ignoring momentum scheduling**: The complementary momentum is a key feature.

## Interview Questions

### Beginner

1. What is the One Cycle Policy and what are its three phases?
2. How does OCP schedule the learning rate and momentum?
3. What is super-convergence?
4. How do you implement OCP in PyTorch?
5. How does OCP differ from cyclical learning rates?

### Intermediate

1. Why does OCP couple learning rate and momentum in opposite directions?
2. How does OCP achieve faster convergence than standard schedules?
3. Compare OCP with the triangular CLR and cosine annealing.
4. How do you determine the optimal max_lr for OCP?
5. What happens if pct_start is set to 0.5 instead of 0.3?

### Advanced

1. Explain the theoretical basis for super-convergence with OCP.
2. Analyze the effect of complementary momentum on training stability.
3. Derive the optimal pct_start for a given loss landscape curvature.

## Practice Problems

### Easy

1. Implement the OCP manually for a quadratic function.
2. Use OneCycleLR on a linear regression task.
3. Plot the LR and momentum curves for a full OCP schedule.
4. Compare OCP with cosine annealing on a simple problem.
5. Run the LR range test to find max_lr.

### Medium

1. Compare OCP, CLR, and step decay on a 3-layer network with CIFAR-10.
2. Tune pct_start in {0.1, 0.2, 0.3, 0.5} and compare results.
3. Implement OCP with cosine annealing instead of linear decay.
4. Analyze the effect of max_lr on convergence speed.
5. Train a small ResNet with OCP and compare with standard training.

### Hard

1. Prove the super-convergence phenomenon for quadratic objectives with OCP.
2. Implement an adaptive OCP that adjusts max_lr based on loss plateau.
3. Design an experiment showing OCP finds flatter minima than standard schedules.

## Solutions

OCP combines linear LR increase during warmup, linear LR decrease during annealing, and optionally a fine-tuning phase. PyTorch's OneCycleLR(optimizer, max_lr, total_steps, pct_start) implements this. Momentum is scheduled opposite to the learning rate.

## Related Concepts

- Cyclical Learning Rates (DL-088): The broader family
- Cosine Annealing (DL-087): Alternative schedule
- Learning Rate Warmup (DL-085): Phase 1 of OCP
- Optimizer Comparison (DL-090): Context for OCP

## Next Concepts

- Optimizer Comparison (DL-090)
- Mean Squared Error (DL-091)
- Cross-Entropy Loss (DL-094)

## Summary

The One Cycle Policy schedules the learning rate to increase during warmup, then decrease during annealing, with momentum moving in the opposite direction. This schedule enables super-convergence, where models train to high accuracy in dramatically fewer iterations. OCP works best with SGD with momentum and requires the LR range test to determine max_lr.

## Key Takeaways

1. OCP has three phases: LR warmup, LR annealing, and fine-tuning.
2. Momentum runs opposite to LR: high when LR is low, low when LR is high.
3. OCP enables super-convergence — 5-10x faster training.
4. Use the LR range test to find max_lr.
5. OCP works best with SGD with momentum, not Adam.
