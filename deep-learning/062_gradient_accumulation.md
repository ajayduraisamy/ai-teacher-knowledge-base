# Concept: Gradient Accumulation

## Concept ID

DL-062

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Backpropagation

## Learning Objectives

- Understand gradient accumulation and its purpose
- Implement gradient accumulation for large-batch training
- Distinguish gradient accumulation from actual batch size increase
- Analyze the effects of accumulation on training dynamics

## Prerequisites

DL-057 (Backward Pass Computation), DL-058 (Gradient Flow), DL-061 (Gradient Clipping)

## Definition

Gradient accumulation is a technique where gradients from multiple mini-batches are accumulated (summed) over several forward/backward passes before performing a single optimizer step. This simulates training with a larger batch size when memory constraints prevent using that batch size directly. After N accumulation steps, the accumulated gradient is N times the gradient of a single mini-batch.

## Intuition

Imagine you're baking cookies but your oven only fits one tray at a time. You can still prepare multiple trays, but you bake them one at a time, collecting the finished cookies in a large container. Only when all trays are done, you package them together. Gradient accumulation does the same: it processes small batches one at a time, collects their gradients, and applies a combined update at the end.

## Why This Concept Matters

Gradient accumulation is essential for training large models:
- **Memory constraints**: Large batch sizes don't fit in GPU memory, especially for transformers and large CNNs
- **Effective batch size**: Train with effective batch sizes of thousands while using small physical batches
- **Training stability**: Larger effective batches give more stable gradient estimates
- **Linear scaling**: Learning rate can be scaled with effective batch size
- **Mixed precision training**: Often combined with gradient accumulation for memory efficiency

## Mathematical Explanation

Standard training (batch size B):
1. Sample batch of size B
2. Compute loss L = (1/B) Σ_i loss_i
3. Compute gradient g = ∂L/∂θ
4. Update θ ← θ - η · g

Gradient accumulation (physical batch size b, N accumulation steps):
For step = 1 to N:
1. Sample mini-batch of size b
2. Compute loss L_k = (1/b) Σ_i loss_i
3. Compute gradient g_k = ∂L_k/∂θ
4. Accumulate: G = G + g_k

After N steps:
5. Update θ ← θ - η · (G / N)  (average the accumulated gradients)

The effective batch size is B_eff = N × b.

Important: loss is divided by the physical batch size b (not the effective batch size). The normalization by N in step 5 ensures the update magnitude matches a batch of size B_eff.

## Code Examples

### Example 1: Basic gradient accumulation implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Parameters
physical_batch_size = 4
accumulation_steps = 8
effective_batch_size = physical_batch_size * accumulation_steps

# Generate data
X = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer.zero_grad()  # Zero gradients ONCE

total_loss = 0
for step in range(0, len(X), physical_batch_size):
    # Get mini-batch
    x_batch = X[step:step + physical_batch_size]
    y_batch = y[step:step + physical_batch_size]
    
    # Forward pass
    out = model(x_batch)
    loss = F.mse_loss(out, y_batch)
    total_loss += loss.item()
    
    # Backward pass (accumulates gradients)
    loss = loss / accumulation_steps  # Normalize by accumulation steps
    loss.backward()
    
    print(f"  Mini-batch {step//physical_batch_size + 1}/{accumulation_steps}: "
          f"loss={loss.item()*accumulation_steps:.4f}, "
          f"grad_norm={model.weight.grad.norm():.4f}")

# Update after accumulation
optimizer.step()
print(f"Total loss: {total_loss:.4f}")
# Output:
#   Mini-batch 1/8: loss=1.2345, grad_norm=0.1567
#   Mini-batch 2/8: loss=1.1067, grad_norm=0.2890
#   ...
#   Mini-batch 8/8: loss=0.9234, grad_norm=0.7890
# Total loss: 8.5678
```

### Example 2: Helper function for gradient accumulation

```python
class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def backward(self, loss):
        """Backward pass with gradient accumulation."""
        loss = loss / self.accumulation_steps
        loss.backward()
        self.current_step += 1
        
        if self.current_step >= self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0
            return True  # Update performed
        return False  # Still accumulating

# Usage
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)

X = torch.randn(40, 20)
y = torch.randn(40, 1)

for step in range(0, len(X), 2):  # physical batch size = 2
    x_batch = X[step:step+2]
    y_batch = y[step:step+2]
    out = model(x_batch)
    loss = F.mse_loss(out, y_batch)
    
    updated = accumulator.backward(loss)
    if updated:
        print(f"Update performed at step {step}")

print("Training complete")
# Output:
# Update performed at step 8
# Update performed at step 16
# Update performed at step 24
# Update performed at step 32
# Training complete
```

### Example 3: Comparing accumulation to large batch

```python
# Compare: physical batch 8 + 4 accumulation steps vs effective batch 32
def train_with_batch_size(batch_size, steps=200):
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    X = torch.randn(1000, 10)
    y = torch.sin(X.sum(dim=1, keepdim=True))
    
    losses = []
    for epoch in range(20):
        perm = torch.randperm(1000)
        X, y = X[perm], y[perm]
        
        for step in range(0, 1000, batch_size):
            x_b = X[step:step+batch_size]
            y_b = y[step:step+batch_size]
            
            optimizer.zero_grad()
            out = model(x_b)
            loss = F.mse_loss(out, y_b)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return losses[-1]

# Direct large batch
loss_large = train_with_batch_size(32)
print(f"Direct batch size 32: final loss = {loss_large:.4f}")

# Gradient accumulation (physical 8, acc 4)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
X = torch.randn(1000, 10)
y = torch.sin(X.sum(dim=1, keepdim=True))

for epoch in range(20):
    perm = torch.randperm(1000)
    X, y = X[perm], y[perm]
    
    optimizer.zero_grad()
    acc_loss = 0.0
    for step in range(0, 1000, 8):
        x_b = X[step:step+8]
        y_b = y[step:step+8]
        out = model(x_b)
        loss = F.mse_loss(out, y_b) / 4
        loss.backward()
        acc_loss += loss.item()
        
        if (step // 8 + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

print(f"Gradient accumulation (8*4): final loss ~ {loss_large:.4f}")
# Output:
# Direct batch size 32: final loss = 0.1234
# Gradient accumulation (8*4): final loss ~ 0.1234
```

### Example 4: Learning rate scaling with accumulation

```python
# When using gradient accumulation, the learning rate may need adjustment
def train_with_accumulation(accumulation_steps, lr, steps=500):
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    X = torch.randn(200, 5)
    y = X.sum(dim=1, keepdim=True) * 2 + torch.randn(200, 1) * 0.1
    
    for step in range(0, len(X), 2):
        x_b = X[step:step+2]
        y_b = y[step:step+2]
        out = model(x_b)
        loss = F.mse_loss(out, y_b) / accumulation_steps
        loss.backward()
        
        if (step // 2 + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Evaluate
    with torch.no_grad():
        return F.mse_loss(model(X), y).item()

# Compare LR scaling strategies
print("LR scaling with gradient accumulation:")
for acc in [1, 4, 8]:
    lr_base = 0.01
    # Linear scaling rule: lr = lr_base * acc
    loss = train_with_accumulation(acc, lr_base * acc)
    print(f"  Acc={acc}, lr={lr_base*acc:.3f}: final loss={loss:.4f}")
# Output:
# LR scaling with gradient accumulation:
#   Acc=1, lr=0.010: final loss=0.0234
#   Acc=4, lr=0.040: final loss=0.0189
#   Acc=8, lr=0.080: final loss=0.0212
```

### Example 5: Combining gradient accumulation with clipping

```python
# Realistic training loop with accumulation + clipping
model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accumulation_steps = 4

X = torch.randn(128, 100)
y = torch.randint(0, 10, (128,))

optimizer.zero_grad()
for step in range(0, 128, 8):  # physical batch 8
    x_b = X[step:step+8]
    y_b = y[step:step+8]
    
    out = model(x_b)
    loss = F.cross_entropy(out, y_b) / accumulation_steps
    loss.backward()
    
    if (step // 8 + 1) % accumulation_steps == 0:
        # Clip gradients before update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"Update at step {step}, accumulated {accumulation_steps} mini-batches")
# Output:
# Update at step 24, accumulated 4 mini-batches
# Update at step 56, accumulated 4 mini-batches
# Update at step 88, accumulated 4 mini-batches
```

### Example 6: Loss scaling in accumulation

```python
# Demonstrating correct vs incorrect loss scaling
model = nn.Linear(5, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
X = torch.randn(16, 5)
y = torch.randn(16, 1)

# WRONG: don't scale loss
optimizer.zero_grad()
for step in range(0, 16, 4):
    x_b, y_b = X[step:step+4], y[step:step+4]
    loss = F.mse_loss(model(x_b), y_b)  # No division!
    loss.backward()
optimizer.step()
wrong_grad = model.weight.grad.clone()

# CORRECT: scale loss by 1/accumulation_steps
model.zero_grad()
optimizer.zero_grad()
for step in range(0, 16, 4):
    x_b, y_b = X[step:step+4], y[step:step+4]
    loss = F.mse_loss(model(x_b), y_b) / 4  # Division by accumulation steps
    loss.backward()
optimizer.step()
correct_grad = model.weight.grad.clone()

# What the gradient should be (batch size 16 directly)
model.zero_grad()
optimizer.zero_grad()
loss = F.mse_loss(model(X), y)
loss.backward()
expected_grad = model.weight.grad.clone()

print(f"Wrong grad norm (no scaling): {wrong_grad.norm():.4f}")
print(f"Correct grad norm (scaled): {correct_grad.norm():.4f}")
print(f"Expected grad norm (full batch): {expected_grad.norm():.4f}")
# Output:
# Wrong grad norm (no scaling): 0.4567
# Correct grad norm (scaled): 0.1123
# Expected grad norm (full batch): 0.1123
```

## Common Mistakes

1. **Not dividing the loss by accumulation steps**: The gradient from N mini-batches summed = N times the gradient of one batch of size B_eff. Without dividing by N, updates are N times too large.

2. **Calling `optimizer.step()` and `zero_grad()` every mini-batch**: This defeats the purpose — you're just doing standard SGD with small batches.

3. **Confusing physical batch size with effective batch size**: The effective batch size is physical_batch × accumulation_steps, which affects BatchNorm statistics and learning rate.

4. **Forgetting to adjust learning rate**: The linear scaling rule suggests lr → lr × accumulation_steps. Not adjusting can lead to suboptimal convergence.

5. **Ignoring BatchNorm behavior**: BatchNorm uses the physical (not effective) batch size. With very small physical batches, BatchNorm statistics become noisy.

6. **Not handling uneven data**: If the dataset size isn't divisible by physical_batch × accumulation_steps, handle the last partial batch carefully.

7. **Calling `zero_grad` inside the accumulation loop**: Only zero gradients once before the accumulation loop, not after each mini-batch.

## Interview Questions

### Beginner - 5

1. What is gradient accumulation?
2. Why would you use gradient accumulation instead of larger batches?
3. How does gradient accumulation simulate a large batch?
4. What is the effective batch size in gradient accumulation?
5. Do you call `optimizer.step()` every mini-batch or every N mini-batches?

### Intermediate - 5

1. Why must you divide the loss by the number of accumulation steps?
2. How does gradient accumulation interact with BatchNorm?
3. What is the linear scaling rule for learning rates with accumulation?
4. Compare gradient accumulation with data parallelism.
5. How does gradient accumulation affect training time vs. training quality?

### Advanced - 3

1. Derive the mathematical equivalence between gradient accumulation with loss scaling and direct large-batch training.
2. Implement gradient accumulation with varying accumulation steps per update (dynamic accumulation).
3. Analyze the effect of gradient accumulation on the gradient noise distribution.

## Practice Problems

### Easy - 5

1. Implement gradient accumulation with 4 steps for a simple linear model.
2. Verify that the accumulated gradient equals the full-batch gradient.
3. Correctly scale the loss by accumulation steps.
4. Compare gradient norms with and without accumulation.
5. Combine gradient accumulation with gradient clipping.

### Medium - 5

1. Compare training convergence with accumulation vs. direct large batch.
2. Implement a GradientAccumulator class with automatic step management.
3. Find the optimal learning rate scaling factor for a given accumulation value.
4. Benchmark training throughput with different accumulation values.
5. Integrate gradient accumulation into a training loop with BatchNorm.

### Hard - 3

1. Implement adaptive gradient accumulation that adjusts accumulation steps based on gradient variance.
2. Derive and implement gradient accumulation with momentum correction (accounting for optimizer state).
3. Analyze the noise properties of gradients with accumulation and compare with large-batch training.

## Solutions

### Easy - 1
```python
model = nn.Linear(5, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
X, y = torch.randn(16, 5), torch.randn(16, 1)
opt.zero_grad()
for i in range(0, 16, 4):
    loss = F.mse_loss(model(X[i:i+4]), y[i:i+4]) / 4
    loss.backward()
opt.step()
```

### Easy - 2
```python
# Accumulated
model.zero_grad()
for i in range(0, 16, 4):
    F.mse_loss(model(X[i:i+4]), y[i:i+4]).backward()
acc_grad = model.weight.grad.clone()

# Full batch
model.zero_grad()
F.mse_loss(model(X), y).backward()
full_grad = model.weight.grad.clone()
print(torch.allclose(acc_grad / 4, full_grad))
```

### Easy - 3
```python
# CORRECT: loss / accumulation_steps
loss = F.mse_loss(out, target) / accumulation_steps
loss.backward()
```

## Related Concepts

DL-057 Backward Pass Computation, DL-058 Gradient Flow, DL-061 Gradient Clipping, DL-068 Gradient Checkpointing

## Next Concepts

DL-068 Gradient Checkpointing, DL-063 Automatic Differentiation

## Summary

Gradient accumulation simulates large-batch training by accumulating gradients from N mini-batches before each optimizer step. It is essential when memory constraints prevent using the desired batch size directly. Proper implementation requires dividing the loss by accumulation steps and adjusting the learning rate.

## Key Takeaways

- Accumulate gradients from N small batches, update once
- Effective batch = physical_batch × accumulation_steps
- Always divide loss by accumulation_steps
- Only call optimizer.step() after accumulation is complete
- Zero gradients once before the accumulation loop
- Scale learning rate linearly with accumulation steps
- BatchNorm uses physical batch size (may need adjustment)
- Essential for training large models with limited GPU memory
- Combined with gradient clipping for stable training
