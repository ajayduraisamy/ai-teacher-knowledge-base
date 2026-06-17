# Concept: Batch Size Selection

## Concept ID

DL-170

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand how batch size affects training dynamics
- Select appropriate batch sizes for different scenarios
- Analyze the tradeoffs between small and large batches
- Implement batch size tuning strategies
- Understand the relationship between batch size and learning rate

## Prerequisites

- Mini-batch gradient descent (DL-073)
- SGD with momentum (DL-074)
- Learning rate schedules
- Training loop (DL-156)

## Definition

Batch size is the number of training samples processed before the model parameters are updated. It is a critical hyperparameter that affects training speed, model quality, generalization, and memory usage. Batch sizes range from 1 (stochastic gradient descent) to the full dataset (batch gradient descent), with mini-batch sizes (8-512) being most common. The batch size determines the number of samples used to estimate the gradient: a larger batch gives a more accurate gradient estimate but uses more memory and may lead to poorer generalization. The choice of batch size is intimately connected to the learning rate: the linear scaling rule states that when doubling batch size, the learning rate should also be doubled to maintain similar training dynamics.

## Intuition

Think of batch size as the number of opinions you gather before making a decision. Asking one person (batch size 1) gives a noisy, unreliable opinion but you can make decisions quickly. Asking everyone (full batch) gives the most accurate opinion but takes a long time and uses many resources. A mini-batch asks a small group - enough to be reasonably confident but fast enough to iterate quickly. The key insight is that gradient estimates from batches of 32-256 are usually good enough because the gradient direction from a small random sample is statistically similar to the true gradient. Larger batches give diminishing returns in gradient accuracy while requiring quadratically more memory. Additionally, the noise from smaller batches can help escape sharp local minima, leading to better generalization.

## Why This Concept Matters

Batch size selection is one of the most impactful hyperparameter decisions. It directly determines: (1) how much GPU memory you need, (2) how fast training progresses (samples per second), (3) the quality of the final model (generalization), (4) the stability of training, (5) the choice of learning rate and schedule. Incorrect batch size selection leads to wasted resources (too large), unstable training (too small), or poor final performance. Understanding batch size effects is essential for scaling training across multiple GPUs, achieving state-of-the-art results, and efficient experiment iteration.

## Mathematical Explanation

The gradient computed on a mini-batch B is: g_B = (1/|B|) * sum_{i in B} grad L(f(x_i, w), y_i). This is an unbiased estimate of the true gradient: E[g_B] = g_true. The variance of the gradient estimate scales as Var(g_B) ~ sigma^2 / |B| where sigma^2 is the per-sample gradient variance. The signal-to-noise ratio of the gradient is SNR = ||g_true|| / (sigma / sqrt(|B|)). Larger batches give higher SNR but with diminishing returns: doubling batch size from 32 to 64 increases SNR by sqrt(2) ~ 1.41x, not 2x. The learning rate scaling rule: when multiplying batch size by k, multiply learning rate by k to keep the expected update magnitude constant. However, there is a critical batch size beyond which larger batches are not beneficial: B_crit ~ 1 / (epsilon * ||g||^2) where epsilon is the target accuracy and g is the gradient. Beyond this point, larger batches waste computation.

## Code Examples

### Example 1: Batch Size Effect on Training Dynamics

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def train_with_batch_size(batch_size, num_epochs=30):
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(2000, 20)
    y = torch.sum(X[:, :5]**2, dim=1, keepdim=True) + 0.1 * torch.randn(2000, 1)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Linear scaling rule: lr proportional to batch_size
    base_lr = 0.001
    lr = base_lr * (batch_size / 32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    epoch_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
    
    elapsed = time.time() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_loss = loss_fn(model(X), y).item()
    
    return {
        'batch_size': batch_size,
        'final_loss': round(train_loss, 5),
        'epochs_trained': num_epochs,
        'time_seconds': round(elapsed, 2),
        'batches_per_epoch': len(dataloader),
        'samples_per_second': round(2000 * num_epochs / elapsed, 1)
    }

print("Batch Size Effect on Training")
print("=" * 80)
print(f"{'Batch Size':12s} {'Final Loss':12s} {'Time (s)':10s} {'Samples/s':12s} {'Batches/Epoch':15s}")
print("-" * 80)

for bs in [1, 4, 16, 64, 256, 1024]:
    result = train_with_batch_size(bs, num_epochs=30)
    print(f"{result['batch_size']:<12d} {result['final_loss']:<12.5f} {result['time_seconds']:<10.2f} "
          f"{result['samples_per_second']:<12.1f} {result['batches_per_epoch']:<15d}")
```

### Example 2: Batch Size and Generalization

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def compare_batch_size_generalization():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a synthetic classification task
    X_train = torch.randn(1000, 10)
    y_train = (torch.sum(X_train**2, dim=1) > 1.5).float().unsqueeze(1)
    
    X_val = torch.randn(500, 10)
    y_val = (torch.sum(X_val**2, dim=1) > 1.5).float().unsqueeze(1)
    
    results = []
    
    for batch_size in [1, 8, 32, 128, 512]:
        model = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        lr = 0.001 * (batch_size / 32)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(30):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                train_loss = loss_fn(model(X_train), y_train).item()
                val_loss = loss_fn(model(X_val), y_val).item()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        train_acc = ((model(X_train) > 0.5).float() == y_train).float().mean().item()
        val_acc = ((model(X_val) > 0.5).float() == y_val).float().mean().item()
        
        results.append({
            'batch_size': batch_size,
            'train_loss': round(train_losses[-1], 4),
            'val_loss': round(val_losses[-1], 4),
            'train_acc': round(train_acc, 4),
            'val_acc': round(val_acc, 4),
            'gap': round(val_losses[-1] - train_losses[-1], 4)
        })
    
    print("Batch Size vs Generalization")
    print("=" * 80)
    print(f"{'Batch Size':12s} {'Train Loss':12s} {'Val Loss':12s} {'Train Acc':10s} "
          f"{'Val Acc':10s} {'Gap':10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['batch_size']:<12d} {r['train_loss']:<12.4f} {r['val_loss']:<12.4f} "
              f"{r['train_acc']:<10.4f} {r['val_acc']:<10.4f} {r['gap']:<+10.4f}")
    
    print("\nKey observation: Small batches often generalize better due to gradient noise")
    print("Large batches converge faster per epoch but may find sharper minima")

compare_batch_size_generalization()
```

### Example 3: Adaptive Batch Size Selection

```python
import numpy as np

def suggest_batch_size(dataset_size, model_size_mb=None, gpu_memory_mb=None):
    """Suggest a batch size based on constraints and best practices."""
    
    suggestions = []
    
    # Rule 1: Batch size should divide dataset size
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        if dataset_size % bs == 0:
            suggestions.append(bs)
    
    if not suggestions:
        # Find nearest divisors
        for bs in [32, 64, 128, 256, 512]:
            suggestions.append(bs)
    
    # Rule 2: Memory constraint
    if model_size_mb is not None and gpu_memory_mb is not None:
        max_bs = int(gpu_memory_mb / (model_size_mb * 3))  # 3x for activations
        suggestions = [bs for bs in suggestions if bs <= max_bs]
        print(f"GPU memory constraint: max batch size = {max_bs}")
    
    # Rule 3: Power of 2 recommended for hardware efficiency
    power_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    efficient = [bs for bs in suggestions if bs in power_of_2]
    
    # Rule 4: Recommended range
    recommended = [bs for bs in efficient if 16 <= bs <= 512]
    
    if not recommended:
        recommended = efficient[:3] if efficient else suggestions[:3]
    
    # Rule 5: Learning rate suggestion based on batch size
    print("\nBatch Size Suggestions:")
    print("=" * 60)
    print(f"Dataset size: {dataset_size}")
    if model_size_mb:
        print(f"Model size: {model_size_mb} MB")
    if gpu_memory_mb:
        print(f"GPU memory: {gpu_memory_mb} MB")
    print()
    
    for bs in recommended[:5]:
        suggested_lr = 0.001 * (bs / 32)
        batches_per_epoch = dataset_size / bs
        print(f"Batch size {bs:4d}: lr ~ {suggested_lr:.5f}, {batches_per_epoch:.0f} batches/epoch")
    
    best_bs = recommended[len(recommended) // 2] if recommended else 32
    return {
        'suggested_batch_size': best_bs,
        'suggested_lr': 0.001 * (best_bs / 32),
        'candidates': recommended[:5]
    }

# Example usage
print("=== Scenario 1: Small dataset (1000 samples) ===")
result = suggest_batch_size(1000, model_size_mb=50, gpu_memory_mb=4000)
print(f"\nRecommended: batch_size={result['suggested_batch_size']}, lr={result['suggested_lr']:.5f}")

print("\n=== Scenario 2: Large dataset (50000 samples) ===")
result = suggest_batch_size(50000, model_size_mb=200, gpu_memory_mb=8000)
print(f"\nRecommended: batch_size={result['suggested_batch_size']}, lr={result['suggested_lr']:.5f}")

print("\n=== Scenario 3: Massive dataset (1.2M samples, multi-GPU) ===")
result = suggest_batch_size(1200000, model_size_mb=350, gpu_memory_mb=16000)
print(f"\nRecommended: batch_size={result['suggested_batch_size']}, lr={result['suggested_lr']:.5f}")
```

## Common Mistakes

1. **Using batch size that is not a power of 2**: GPUs are optimized for power-of-2 tensor dimensions. Use 8, 16, 32, 64, 128, 256, 512 for optimal performance.
2. **Not scaling learning rate with batch size**: When you double batch size, the gradient variance halves. Without doubling the learning rate, updates become too small.
3. **Ignoring memory constraints**: Batch size must fit in GPU memory with room for activations (not just parameters). Use gradient accumulation for larger effective batch sizes.
4. **Assuming larger batches always give better results**: Large batches often converge to sharper minima with poorer generalization. Small batches inject noise that helps escape sharp minima.
5. **Using the same batch size for every layer**: Batch normalization statistics depend on batch size. Very small batches (<8) give poor batch norm statistics.

## Interview Questions

### Beginner

1. What is batch size?
2. How does batch size affect training speed?
3. What is the difference between SGD, mini-batch, and full-batch?
4. Why is batch size typically a power of 2?
5. How does batch size affect memory usage?

### Intermediate

1. Explain the linear scaling rule for learning rate and batch size.
2. How does batch size affect generalization?
3. What is gradient accumulation and when would you use it?
4. How do you choose batch size for multi-GPU training?
5. What happens if batch size is too small (e.g., 1 or 2)?

### Advanced

1. Derive the critical batch size in terms of gradient signal-to-noise ratio.
2. Explain the relationship between batch size and the sharpness of minima.
3. Design an adaptive batch size schedule that changes during training.

## Practice Problems

### Easy

1. Compute the number of batches per epoch for given dataset and batch size.
2. Calculate the learning rate for batch size=128 given base lr=0.001 at batch size=32.
3. Determine if a batch size fits in GPU memory given model parameters.
4. List three tradeoffs between small and large batch sizes.
5. Compute the effective batch size when using gradient accumulation with 4 steps.

### Medium

1. Implement gradient accumulation for training with effective batch size 256 on a GPU that can only fit batch size 32.
2. Create a batch size sweep experiment and analyze the effect on convergence.
3. Implement a learning rate schedule that scales with batch size.
4. Build a memory calculator that determines max batch size for a given model.
5. Analyze the relationship between batch size and validation accuracy.

### Hard

1. Implement the Smith et al. batch size selection method using the noisy gradient approximation.
2. Design an adaptive batch size scheduler that increases batch size during training.
3. Implement a distributed training setup that uses different batch sizes per GPU.

## Solutions

### Easy Solutions

1. `num_batches = ceil(dataset_size / batch_size)`
2. `lr = 0.001 * (128 / 32) = 0.004`
3. Estimate: total_params * 4 bytes * 3 (params, grads, optim states) + batch_size * activation_memory
4. Small batch: noisy gradients, better generalization, slower per epoch. Large batch: accurate gradients, worse generalization, faster per epoch.
5. `effective_bs = batch_size * accumulation_steps = 32 * 4 = 128`

## Related Concepts

- Mini-batch gradient descent (DL-073)
- SGD with momentum (DL-074)
- Learning rate schedules
- Gradient accumulation

## Next Concepts

- Learning Rate Warmup (DL-085)
- Learning Rate Decay (DL-086)
- One Cycle Policy (DL-089)

## Summary

Batch size controls the number of samples used per gradient update. It affects training speed, memory usage, gradient accuracy, and generalization. Smaller batches often generalize better but train slower per epoch. Larger batches are more computationally efficient but may find sharper minima. The learning rate should scale linearly with batch size.

## Key Takeaways

- Batch size is a critical hyperparameter
- Use powers of 2 for GPU efficiency
- Scale learning rate linearly with batch size
- Small batches often generalize better
- Large batches converge faster per sample
- Gradient noise from small batches aids exploration
- Consider GPU memory constraints
- Use gradient accumulation for large effective batches
- Match batch size to dataset size and model capacity
- Always validate batch size choice empirically
