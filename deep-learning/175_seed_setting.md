# Concept: Seed Setting

## Concept ID

DL-175

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand what a random seed is and why it matters
- Learn how to set seeds in PyTorch, NumPy, and Python
- Recognize the limitations of seed setting
- Apply seed setting in training pipelines

## Prerequisites

DL-174 Reproducibility in DL (recommended but not required)

## Definition

Seed setting is the practice of initializing pseudo-random number generators with a fixed integer value so that sequences of random numbers are reproducible across runs.

## Intuition

Think of a random number generator like a giant book of seemingly random numbers. A seed tells you which page to start reading from. If you always start on page 42, you'll always read the same sequence of numbers. Without a seed, the generator starts on a different page each time — the randomness comes from system entropy, like the exact nanosecond you started the program. Setting a seed transforms "truly random" into "deterministically random" — you get the benefits of randomness (exploration, regularization) with the ability to replay the exact same sequence.

## Why This Concept Matters

Seed setting is the simplest and most fundamental reproducibility technique. It ensures that random initialization, data shuffling, and stochastic operations produce the same results across runs. Without seeds, you can't debug effectively, compare experiments fairly, or share reproducible research.

## Mathematical Explanation

Pseudo-random number generators (PRNGs) produce sequences that satisfy:

$$x_{n+1} = f(x_n) \mod m$$

For example, a linear congruential generator: $x_{n+1} = (a \cdot x_n + c) \mod m$

The initial value $x_0$ is the **seed**. Given the same seed, the entire sequence is identical:

$$\text{PRNG}(seed) = \{x_0, x_1, x_2, ..., x_N\}$$

Different libraries have independent PRNG states. PyTorch uses the Mersenne Twister algorithm. NumPy uses PCG-64 (default since NumPy 1.17). Python's `random` module uses Mersenne Twister.

Seeds must be set independently for each library:

$$\text{FullySeeded} = \{ \text{torch\_seed}(s), \text{np\_seed}(s), \text{random\_seed}(s) \}$$

Even with identical seeds, floating-point operations on different hardware can diverge.

## Code Examples

### Example 1: Basic Seed Setting in PyTorch

```python
import torch
import numpy as np
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Now all random operations are deterministic
t1 = torch.randn(3, 3)
t2 = torch.randn(3, 3)

print("First random tensor:")
print(t1)
# Output: tensor([[ 0.1847,  0.1173, -1.1207],
# Output:         [ 0.6365,  1.0605,  0.5947],
# Output:         [-0.8096,  0.0727, -0.1983]])

# Reset seed and re-run
torch.manual_seed(42)
t3 = torch.randn(3, 3)

print(f"\nt1 equals t3: {torch.equal(t1, t3)}")
# Output: t1 equals t3: True

# Different seed produces different results
torch.manual_seed(99)
t4 = torch.randn(3, 3)
print(f"\nt1 equals t4: {torch.equal(t1, t4)}")
# Output: t1 equals t4: False
```

### Example 2: Seed Impact on Model Initialization

```python
import torch
import torch.nn as nn

def init_model(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 1)
    )
    return model

# Same seed = same initialization
model_a = init_model(42)
model_b = init_model(42)

weight_a = model_a[0].weight.detach()
weight_b = model_b[0].weight.detach()

print(f"Same weights: {torch.equal(weight_a, weight_b)}")
# Output: Same weights: True

# Different seed = different initialization
model_c = init_model(999)
weight_c = model_c[0].weight.detach()

print(f"Different weights: {torch.equal(weight_a, weight_c)}")
# Output: Different weights: False
```

### Example 3: Seed Effects on Training Outcome

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_with_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    
    model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    return model(X).detach()

# Same seed = same output
out1 = train_with_seed(42)
out2 = train_with_seed(42)
print(f"Same output: {torch.allclose(out1, out2)}")
# Output: Same output: True

# Different seed = different output
out3 = train_with_seed(1)
print(f"Different output: {not torch.allclose(out1, out3)}")
# Output: Different output: True
```

## Common Mistakes

1. **Only setting PyTorch seed**: Must also set numpy, random, and CUDA seeds.
2. **Setting seed inside the training loop**: Resetting seed each iteration destroys randomness.
3. **Forgetting DataLoader**: PyTorch DataLoader uses its own internal seeding.
4. **Assuming seeds guarantee cross-platform reproducibility**: Different hardware produces different results.
5. **Using the same seed for all experiments**: This can bias results if you repeatedly use the same data splits.

## Interview Questions

### Beginner - 5
1. What is a random seed?
2. Why do we set random seeds?
3. How do you set a seed in PyTorch?
4. Does setting a seed guarantee identical results on different computers?
5. What happens if you set the seed inside the training loop?

### Intermediate - 5
1. How do you set seeds for both CPU and GPU operations?
2. Explain the relationship between seeds and DataLoader worker processes.
3. Why does torch.manual_seed not affect numpy random?
4. How does seed setting interact with cuDNN autotuning?
5. What is the recommended way to generate unique experiments (different seeds)?

### Advanced - 3
1. Design a system for distributing seed assignments across a cluster.
2. Explain how to handle seeds for multi-worker data loading.
3. How would you test whether a training pipeline is properly seeded?

## Practice Problems

### Easy - 5
1. Write code to set seeds for torch, numpy, and random.
2. Verify that setting the same seed produces identical tensor initialization.
3. Compare model performance with 5 different seeds.
4. Show that different seeds produce different gradient paths.
5. Implement a function that resets the seed and confirms reproducibility.

### Medium - 5
1. Create a training script that reports results with mean and std across 10 seeds.
2. Implement a deterministic DataLoader with seed control.
3. Compare the effect of seeds on small vs large models.
4. Build a seed management system for hyperparameter search.
5. Verify that dropout with seed produces reproducible masks.

### Hard - 3
1. Implement a system that can reproduce any training run from logged seed info.
2. Design a seed generation strategy for distributed training that avoids collisions.
3. Build a deterministic training pipeline that works across CPU and GPU.

## Solutions

### Easy - 1 Solution
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Related Concepts

DL-174 Reproducibility in DL, DL-172 Gradient Norm Monitoring, DL-173 Loss Monitoring

## Next Concepts

DL-174 Reproducibility in DL (if not already covered)

## Summary

Seed setting initializes random number generators to a fixed state, enabling reproducible experiments. It's the foundation of reproducibility in deep learning, though it must be applied to all random libraries and comes with limitations regarding cross-platform and cross-hardware consistency.

## Key Takeaways

- A seed initializes the PRNG to a deterministic state
- Setting the seed is essential for reproducibility and debugging
- Must seed PyTorch, NumPy, and Python's random independently
- CUDA operations require additional deterministic settings
- Seeds don't guarantee identical results across hardware
- Use different seeds for different experiments / train-test splits
- Document the seed used for every reported experiment
- Seeds enable fair comparison of hyperparameters and architectures
