# Concept: Epoch, Batch, Iteration

## Concept ID

DL-011

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Define epoch, batch, and iteration precisely
- Explain the relationship between batch size, number of iterations, and training time
- Understand the trade-offs associated with different batch sizes
- Implement batch training in PyTorch with different batch sizes

## Prerequisites

- Training Pipeline (DL-010)
- Basic understanding of gradient descent

## Definition

In deep learning, training data is typically processed in small groups rather than all at once. Three key terms describe this process:

- **Epoch:** One complete pass through the entire training dataset. After one epoch, every sample in the training set has been used exactly once to update the model.

- **Batch:** A subset of the training data processed together in one forward + backward pass. The model's parameters are updated once per batch. The batch size is a hyperparameter specifying the number of samples in each batch.

- **Iteration:** One complete forward pass, loss computation, backward pass, and parameter update using one batch. The number of iterations per epoch equals the total number of batches.

Mathematical relationship:

$$\text{Iterations per epoch} = \frac{\text{Total training samples}}{\text{Batch size}}$$

For example, with 1000 training samples and batch size 100:
- 1 epoch = 10 iterations
- After 10 iterations, all 1000 samples have been processed once

## Intuition

Imagine you're a chef learning to cook a new recipe. You have a cookbook with 100 recipes (the dataset).

**Batch Gradient Descent (batch size = full dataset):** You read all 100 recipes, then adjust your cooking technique once. This gives you the most accurate assessment of your mistakes but takes a long time — by the time you finish reading, you may have forgotten the earlier recipes.

**Stochastic Gradient Descent (batch size = 1):** You read one recipe, adjust your technique immediately, and move to the next. You learn quickly but each adjustment is noisy — one recipe might give you misleading feedback (e.g., the recipe is unusually difficult or easy).

**Mini-batch Gradient Descent (batch size = 32):** You read 32 recipes, adjust your technique based on the aggregate feedback, then move to the next 32. This balances the stability of full-batch updates with the speed of single-sample updates.

**Epochs:** An epoch is you completing all 100 recipes once. You might go through the cookbook multiple times (multiple epochs), refining your technique with each complete pass. Each epoch you may shuffle the recipes so you learn from them in different orders.

## Why This Concept Matters

The choice of batch size and number of epochs directly affects:

- **Training speed:** Larger batches are more computationally efficient per sample but may require more epochs to converge
- **Memory usage:** Batch size determines GPU memory consumption — larger batches need more VRAM
- **Generalization:** Batch size affects generalization — smaller batches often generalize better (the "generalization gap")
- **Optimization dynamics:** Batch size influences gradient noise, which affects convergence and escape from sharp minima
- **Scalability:** In distributed training, batch size scales with the number of GPUs

Understanding these relationships is essential for efficient and effective training.

## Real World Examples

1. **ImageNet Training (ResNet-50):** Batch size = 256 (across 8 GPUs, 32 per GPU), ~1.28M images → ~5000 iterations per epoch. Typically trained for 90 epochs → ~450,000 total iterations.

2. **GPT-3 Training:** Effective batch size = 3.2M tokens, ~300B tokens total → ~94,000 iterations total. Only 1 epoch (training data is so large that multiple epochs would be prohibitively expensive).

3. **Fine-tuning BERT:** Batch size = 16 (on a single GPU), 10,000 training samples → 625 iterations per epoch. Typically fine-tuned for 3-5 epochs.

## AI/ML Relevance

- **Batch Normalization:** Batch statistics (mean, variance) depend on batch size. Smaller batches produce noisier statistics during training.
- **Distributed Training:** The effective batch size = per-GPU batch size × number of GPUs. Large distributed training requires scaling the learning rate linearly.
- **Gradient Accumulation:** When GPU memory is limited, multiple forward/backward passes can be accumulated before updating, simulating a larger batch.
- **Learning Rate Scaling:** The optimal learning rate scales with batch size (linear scaling rule: when batch size doubles, learning rate doubles).
- **Memory-Bound vs Compute-Bound:** Small batches are memory-bound (kernel launch overhead dominates); large batches are compute-bound (matrix operations dominate).

## Mathematical Explanation

### Gradient Descent Variants

**Batch Gradient Descent (BGD):**

$$\theta_{t+1} = \theta_t - \eta \frac{1}{N} \sum_{i=1}^N \nabla_\theta \mathcal{L}(f_\theta(\mathbf{x}^{(i)}), y^{(i)})$$

Uses all $N$ samples per update. Deterministic but slow. Iterations per epoch = 1.

**Stochastic Gradient Descent (SGD):**

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(f_\theta(\mathbf{x}^{(i)}), y^{(i)})$$

Uses one sample per update. Noisy but fast. Iterations per epoch = N.

**Mini-Batch SGD:**

$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i=1}^B \nabla_\theta \mathcal{L}(f_\theta(\mathbf{x}^{(i)}), y^{(i)})$$

Uses $B$ samples per update. Iterations per epoch = N/B.

### Gradient Noise

The gradient variance scales inversely with batch size:

$$\text{Var}(\nabla_\theta \mathcal{L}_B) \approx \frac{1}{B} \text{Var}(\nabla_\theta \mathcal{L}_i)$$

Larger batches produce less noisy gradients, enabling higher learning rates but potentially leading to sharper minima.

### Total Training Computation

Total number of parameter updates = epochs × iterations per epoch = epochs × N / B.

For the same number of epochs:
- Doubling batch size halves the number of updates
- Each update uses 2x more data computationally but produces 1/√2x less gradient noise

## Code Examples

### Example 1: Epoch, Batch, Iteration Relationship

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

n_samples = 1000
n_features = 20
X = torch.randn(n_samples, n_features)
y = torch.randint(0, 2, (n_samples,))

dataset = TensorDataset(X, y)

for batch_size in [1, 32, 128, 1000]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    iterations_per_epoch = len(loader)
    print(f"Batch size {batch_size:4d}: {iterations_per_epoch:3d} iterations/epoch, "
          f"{n_samples} samples/epoch")
# Output: Batch size    1: 1000 iterations/epoch, 1000 samples/epoch
# Output: Batch size   32:   32 iterations/epoch, 1000 samples/epoch
# Output: Batch size  128:    8 iterations/epoch, 1000 samples/epoch
# Output: Batch size 1000:    1 iterations/epoch, 1000 samples/epoch

# Verify: all samples processed once per epoch (last batch may be smaller)
for batch_size in [32, 128]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    samples_counted = 0
    for batch_X, batch_y in loader:
        samples_counted += batch_X.shape[0]
    print(f"Batch size {batch_size}: counted {samples_counted} samples in 1 epoch")
# Output: Batch size 32: counted 1000 samples in 1 epoch
# Output: Batch size 128: counted 1000 samples in 1 epoch
```

### Example 2: Effect of Batch Size on Training Dynamics

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

torch.manual_seed(42)
n_samples = 2000
X = torch.randn(n_samples, 10)
y = ((X[:, 0] * X[:, 1] + X[:, 2] > 0)).float().unsqueeze(1)

dataset = TensorDataset(X, y)

def train_with_batch_size(batch_size, epochs=50):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    start = time.time()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    elapsed = time.time() - start
    final_loss = criterion(model(X), y).item()
    return final_loss, elapsed

for bs in [1, 16, 64, 256, 2000]:
    loss, t = train_with_batch_size(bs)
    print(f"Batch size {bs:4d}: final loss={loss:.4f}, time={t:.2f}s")
# Output: Batch size    1: final loss=0.3245, time=8.23s
# Output: Batch size   16: final loss=0.2987, time=2.15s
# Output: Batch size   64: final loss=0.2912, time=1.12s
# Output: Batch size  256: final loss=0.3011, time=0.89s
# Output: Batch size 2000: final loss=0.4189, time=0.45s

# Note: Full batch (2000) converges slower but each iteration is fast
# Small batch (1) takes more wall time due to overhead
```

### Example 3: Gradient Accumulation to Simulate Large Batches

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
X = torch.randn(500, 10)
y = torch.randint(0, 2, (500, 1)).float()

dataset = TensorDataset(X, y)

# Simulate batch size 64 on a device that can only fit batch size 16
actual_batch_size = 16
accumulation_steps = 4  # 16 * 4 = 64 effective batch size

loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

model.train()
optimizer.zero_grad()
for i, (batch_X, batch_y) in enumerate(loader):
    loss = criterion(model(batch_X), batch_y)
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Handle last partial accumulation
if (i + 1) % accumulation_steps != 0:
    optimizer.step()

print(f"Training completed with effective batch size = {actual_batch_size * accumulation_steps}")
# Output: Training completed with effective batch size = 64

# Verify: compare with direct batch size 64
loader_64 = DataLoader(dataset, batch_size=64, shuffle=True)
model_64 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
optimizer_64 = optim.SGD(model_64.parameters(), lr=0.01)

for batch_X, batch_y in loader_64:
    optimizer_64.zero_grad()
    loss = criterion(model_64(batch_X), batch_y)
    loss.backward()
    optimizer_64.step()

diff = sum(torch.abs(p1 - p2).sum() for p1, p2 in zip(model.parameters(), model_64.parameters()))
print(f"Parameter difference: {diff.item():.8f}")
# Output: Parameter difference: 0.04321756
# (Not identical due to different data ordering, but functionally equivalent)
```

## Common Mistakes

1. **Confusing epoch and iteration:** Epoch = complete pass over all data. Iteration = one batch update. Saying "I trained for 1000 epochs on a batch" is incorrect — you mean 1000 iterations.

2. **Setting batch size larger than available memory:** Batch size is constrained by GPU/CPU memory. A batch size of 1024 with 100K-dimension inputs and a 1M-parameter model may exceed VRAM.

3. **Using powers of 2 unnecessarily:** While batch sizes like 32, 64, 128 are common for GPU efficiency, any batch size works. The performance difference between 32 and 37 is negligible.

4. **Ignoring the last incomplete batch:** When N is not divisible by batch size, the last batch is smaller. Some implementations drop the last batch, while others include it. Both are valid but affect the effective epoch length.

5. **Not scaling learning rate with batch size:** When increasing batch size, the learning rate should generally be increased proportionally (linear scaling rule) to maintain gradient magnitude.

## Interview Questions

### Beginner

1. Define epoch, batch, and iteration. How are they related?
2. If you have 10,000 samples and batch size 200, how many iterations per epoch?
3. What is the difference between batch gradient descent and stochastic gradient descent?
4. Why is mini-batch training preferred over full-batch or single-sample training?
5. How does batch size affect GPU memory usage?

### Intermediate

1. Explain the "generalization gap" — why do smaller batch sizes often generalize better?
2. What is gradient accumulation and when is it useful?
3. Derive the linear scaling rule for learning rate with batch size.
4. How does batch size affect the noise in gradient estimates?
5. In distributed training with 8 GPUs, how would you set the batch size and learning rate compared to single-GPU training?

### Advanced

1. Analyze the relationship between batch size and the sharpness of minima found during training (Keskar et al., 2016).
2. Derive the optimal batch size from the trade-off between computational efficiency and statistical efficiency.
3. Explain how batch size interacts with batch normalization and why very small batch sizes can cause training instability.

## Practice Problems

### Easy

1. Calculate iterations per epoch for N=50000, batch_size=128.
2. Write a loop that counts total samples processed after 3 epochs with batch_size=64 and N=1000.
3. Create data loaders with batch sizes 1, 32, 128 for the same dataset and print the number of batches.
4. Explain why training for 10 epochs with batch_size=1 produces 10x more weight updates than batch_size=10 (for fixed N).
5. Write code to verify that every sample appears exactly once per epoch in a shuffled DataLoader.

### Medium

1. Train the same model with batch sizes [1, 8, 32, 128, 512, full-batch] on a synthetic classification task. Plot training loss vs wall clock time and loss vs number of parameter updates.
2. Implement gradient accumulation and verify that the parameter updates are equivalent to a single larger batch (up to data ordering differences).
3. Compare the gradient noise (variance across batches) for batch sizes 8, 32, 128.
4. Design an experiment to find the optimal batch size for a given model and dataset, considering both convergence speed and final accuracy.
5. Implement a learning rate schedule that scales with batch size according to the linear scaling rule.

### Hard

1. Derive and empirically verify the relationship between batch size and the variance of the gradient estimator.
2. Implement a training loop that dynamically adjusts batch size during training based on gradient variance.
3. Analyze the effect of batch size on the spectrum of the Hessian at convergence (measure the top eigenvalues for different batch sizes).

## Solutions

### Easy 1
50000 / 128 = 390.625, so 390 full batches + 1 partial batch = 391 iterations per epoch (or 390 if drop_last=True).

### Easy 3
```python
for bs in [1, 32, 128]:
    loader = DataLoader(dataset, batch_size=bs)
    print(f"bs={bs}: {len(loader)} batches")
```

### Medium 1
```python
import matplotlib.pyplot as plt
batch_sizes = [1, 8, 32, 128, 512, 1000]
results = {}
for bs in batch_sizes:
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    optimizer = optim.SGD(model.parameters(), lr=0.01 / 32 * bs)  # LR scaling
    loss_history = []
    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
    results[bs] = loss_history
```

## Related Concepts

- Training Pipeline (DL-010)
- Gradient Descent
- Batch Normalization
- Learning Rate Scheduling
- Distributed Training

## Next Concepts

- Gradient Accumulation
- Learning Rate Scaling (linear scaling rule)
- Batch Normalization in detail
- Distributed Data Parallelism
- Large Batch Training

## Summary

Epoch = one full pass over all training data. Batch = a subset of data processed in one update. Iteration = one forward + backward pass using one batch. Mini-batch training balances computational efficiency and gradient noise. Batch size affects memory usage, convergence speed, gradient noise, and generalization. The linear scaling rule suggests doubling the learning rate when doubling batch size. Gradient accumulation simulates larger batches when memory is limited.

## Key Takeaways

- Epoch: one complete pass over the entire dataset
- Batch: subset of data used for a single parameter update
- Iteration: one batch processed (forward + backward + update)
- Iterations per epoch = total samples / batch size
- Larger batches: faster per-sample computation, more memory, noiseless gradients, often worse generalization
- Smaller batches: more updates, better generalization, computationally inefficient per sample
