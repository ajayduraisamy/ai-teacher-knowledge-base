# Concept: Reproducibility in Deep Learning

## Concept ID

DL-174

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand why reproducibility is challenging in deep learning
- Identify sources of non-determinism in deep learning pipelines
- Implement reproducibility measures in PyTorch
- Design reproducible experimental workflows

## Prerequisites

DL-175 Seed Setting, DL-171 Learning Rate Selection, DL-101 Gradient Descent

## Definition

Reproducibility in deep learning refers to the ability to obtain identical results when running the same experiment multiple times, or for different researchers to obtain consistent results using the same methodology.

## Intuition

Imagine trying to bake a cake, but the oven's temperature varies each time, the mixing time depends on how fast you stir, and the ingredients interact differently depending on the humidity. That's deep learning without reproducibility. Every random initialization, every GPU operation, every data shuffle introduces variation. Reproducibility means controlling all these factors so that two runs of the same experiment produce the same trained model. It's fundamental to scientific validity and engineering reliability.

## Why This Concept Matters

Without reproducibility, you cannot trust your own experimental results, compare methods fairly, or deploy reliable systems. It's the foundation of scientific progress in deep learning. Non-reproducible results have led to embarrassing retractions and wasted resources across the field.

## Mathematical Explanation

Sources of non-determinism in deep learning:

1. **Weight initialization**: Parameters randomly sampled from distributions. Without fixed seeds, each run starts from a different point in the loss landscape.

2. **Data shuffling**: Stochasticity in data order affects gradient estimates. Without controlling shuffle order, gradient trajectories diverge.

3. **GPU parallelism**: CUDA operations like `atomicAdd` and cuDNN convolution algorithms are inherently non-deterministic. The same operation can produce slightly different floating point results depending on thread scheduling.

4. **Dropout / Data augmentation**: Random masks and transformations introduce stochasticity.

5. **Multi-GPU training**: Gradient averaging order across GPUs varies.

6. **Floating-point non-associativity**: $a + (b + c) \neq (a + b) + c$ in floating-point arithmetic.

The reproducibility equation: For a training run $R$ with seed $S$, dataset $D$, model architecture $M$, and hyperparameters $H$:

$$\text{Deterministic}(R) \iff \text{Control}(S, D, M, H, \text{RNG}, \text{GPU})$$

Breaking any deterministic link introduces irreproducibility.

## Code Examples

### Example 1: Basic Seed Setting

```python
import torch
import torch.nn as nn
import numpy as np
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Makes convolution operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
model1 = nn.Linear(10, 1)
w1 = model1.weight.clone()

set_seed(42)
model2 = nn.Linear(10, 1)
w2 = model2.weight.clone()

print(f"Weights identical: {torch.equal(w1, w2)}")
# Output: Weights identical: True

# Without seed, they differ
model3 = nn.Linear(10, 1)
model4 = nn.Linear(10, 1)
print(f"Weights identical without seed: {torch.equal(model3.weight, model4.weight)}")
# Output: Weights identical without seed: False
```

### Example 2: Reproducible Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import os

def reproducible_training(seed=42):
    # Full reproducibility setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Use generator for deterministic data loading
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=g)
    
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(10):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return losses

# Run twice
losses1 = reproducible_training(42)
losses2 = reproducible_training(42)

print(f"Same number of steps: {len(losses1) == len(losses2)}")
# Output: Same number of steps: True

print(f"All losses identical: {all(abs(a-b) < 1e-10 for a,b in zip(losses1, losses2))}")
# Output: All losses identical: True
```

### Example 3: CUDA Non-Determinism Issue

```python
import torch
import torch.backends.cudnn as cudnn

torch.manual_seed(42)

# Default CUDA behavior (non-deterministic)
x = torch.randn(16, 3, 32, 32).cuda()
conv = nn.Conv2d(3, 16, 3).cuda()

# Benchmark mode can change algorithm between runs
print(f"Default deterministic: {torch.backends.cudnn.deterministic}")
# Output: Default deterministic: False

print(f"Default benchmark: {torch.backends.cudnn.benchmark}")
# Output: Default benchmark: True

# Enable deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Now convolution results are deterministic
out1 = conv(x)
out2 = conv(x)
print(f"Convolution identical: {torch.equal(out1, out2)}")
# Output: Convolution identical: True

# Warning: deterministic mode is slower
# Trade-off: reproducibility vs speed
```

## Common Mistakes

1. **Only setting PyTorch seed**: Must also set numpy, random, CUDA seeds, and disable cuDNN benchmark.
2. **Forgetting DataLoader shuffling**: DataLoader's shuffle is seeded independently — use `generator` argument.
3. **Assuming determinism across hardware**: Different GPU models produce different numerical results.
4. **Ignoring augmentation randomness**: Data augmentation operations need their own seed management.
5. **Not saving initial model state**: Without saving the initialized model, you can't reproduce the starting point.

## Interview Questions

### Beginner - 5
1. What is reproducibility in deep learning?
2. Why is reproducibility difficult in deep learning?
3. What is a random seed?
4. Why might two identical training runs produce different results?
5. What is the simplest way to improve reproducibility?

### Intermediate - 5
1. Explain the role of cuDNN deterministic mode.
2. How does DataLoader shuffling affect reproducibility?
3. Why does batch normalization behave differently between train and eval modes?
4. How do you handle reproducibility with multi-GPU training?
5. What is the trade-off between reproducibility and performance?

### Advanced - 3
1. Design a complete reproducible experiment framework for a research paper.
2. Explain floating point non-determinism in GPU reductions.
3. How would you verify that your training pipeline is deterministic?

## Practice Problems

### Easy - 5
1. Write a function that sets all seeds deterministically.
2. Verify that two models initialized with the same seed have identical weights.
3. Compare training results with and without reproducibility settings.
4. Create a deterministic DataLoader.
5. Test that torch.allclose passes for two runs with same seed.

### Medium - 5
1. Implement a reproducibility test suite for a training pipeline.
2. Investigate how different seeds affect final model accuracy.
3. Compare run times with and without deterministic CUDA.
4. Build a logging system that records all seed information.
5. Implement reproducible data augmentation.

### Hard - 3
1. Design a multi-GPU training setup that produces exactly the same results regardless of GPU count.
2. Create a tool that verifies the reproducibility of any PyTorch training script.
3. Implement a distributed training system that maintains determinism across machines.

## Solutions

### Easy - 1 Solution
```python
def set_full_reproducibility(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

## Related Concepts

DL-175 Seed Setting, DL-173 Loss Monitoring, DL-171 Learning Rate Selection

## Next Concepts

DL-175 Seed Setting

## Summary

Reproducibility is essential for scientific rigor in deep learning. It requires controlling all sources of randomness: seeds, DataLoader shuffling, CUDA operations, and augmentation. While deterministic mode can slow training, it's critical for debugging, comparing methods, and ensuring reliable results.

## Key Takeaways

- Deep learning involves many sources of non-determinism
- Seed setting alone is insufficient — must also control DataLoader and CUDA operations
- cuDNN deterministic mode ensures reproducibility at the cost of speed
- Reproducibility across different hardware is fundamentally limited
- Document all seed and environment information for every experiment
- Use deterministic mode for debugging; benchmark mode for production training
- Multi-GPU reproducibility adds additional complexity
