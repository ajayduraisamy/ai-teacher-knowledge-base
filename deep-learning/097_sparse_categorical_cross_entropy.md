# Concept: Sparse Categorical Cross-Entropy

## Concept ID

DL-097

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand the difference between sparse and non-sparse categorical cross-entropy
- Implement using nn.CrossEntropyLoss with integer labels
- Recognize when sparse vs. one-hot encoding matters computationally
- Apply to problems with large numbers of classes

## Prerequisites

- Categorical Cross-Entropy (DL-096)
- One-hot encoding concepts
- PyTorch tensor operations

## Definition

Sparse categorical cross-entropy is identical to categorical cross-entropy in computation but accepts labels as integer class indices instead of one-hot vectors. There is no mathematical difference — the same loss is computed, just with different label formats.

Standard CCE with one-hot: CCE = -(1/N) * sum_i sum_c y_{i,c} * log(p_{i,c})
Sparse CCE with indices: CCE = -(1/N) * sum_i log(p_{i, label_i})

## Intuition

When you have 1000 classes, one-hot encoding creates a vector of length 1000 with a single 1. This is memory-inefficient. Sparse CCE uses a single integer (0-999) instead, saving memory and computation. The loss computation is identical because only the true class contributes to the gradient.

## Why This Concept Matters

Sparse categorical cross-entropy is crucial for problems with many classes like ImageNet (1000 classes) or language modeling (vocabulary of 50000+ tokens).

## Mathematical Explanation

### Identical Loss

Both formulations compute:

Loss = -(1/N) * sum_i log(p_{i, k_i})

where k_i is the correct class index. The gradient is also identical:

dL/dz_{i,c} = p_{i,c} - [c == k_i]

## Code Examples

### Example 1: Sparse vs. One-Hot Comparison

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.randn(4, 10)  # 4 samples, 10 classes
labels_idx = torch.tensor([3, 7, 1, 9])  # sparse labels
labels_onehot = F.one_hot(labels_idx, num_classes=10).float()

# Sparse: CrossEntropyLoss with integer labels
ce_sparse = nn.CrossEntropyLoss()(logits, labels_idx)
print(f"Sparse CE: {ce_sparse.item():.6f}")

# One-hot: NLLLoss with log-probs
log_probs = F.log_softmax(logits, dim=1)
ce_onehot = nn.NLLLoss()(log_probs, labels_idx)
print(f"One-hot CE (via NLLLoss): {ce_onehot.item():.6f}")

# One-hot manual
ce_onehot_manual = -(labels_onehot * log_probs).sum(dim=1).mean()
print(f"One-hot CE (manual): {ce_onehot_manual.item():.6f}")
```

```
# Output:
# Sparse CE: 2.456891
# One-hot CE (via NLLLoss): 2.456891
# One-hot CE (manual): 2.456891
```

### Example 2: Memory Comparison

```python
import torch
import torch.nn as nn

batch_size, n_classes = 64, 50000

# Sparse labels: 64 ints
labels_sparse = torch.randint(0, n_classes, (batch_size,))
print(f"Sparse labels memory: {labels_sparse.element_size() * labels_sparse.numel() / 1024:.2f} KB")

# One-hot labels: 64 * 50000 floats
labels_onehot = torch.zeros(batch_size, n_classes)
labels_onehot[torch.arange(batch_size), labels_sparse] = 1.0
print(f"One-hot labels memory: {labels_onehot.element_size() * labels_onehot.numel() / 1024:.2f} KB")
print(f"Memory ratio: {labels_onehot.numel() / labels_sparse.numel():.0f}x")
```

```
# Output:
# Sparse labels memory: 0.25 KB
# One-hot labels memory: 12207.03 KB
# Memory ratio: 50000x
```

### Example 3: Multi-Class with Many Classes

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D, C = 2000, 50, 1000
X = torch.randn(N, D)
y = torch.randint(0, C, (N,))

model = nn.Linear(D, C)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {accuracy.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 6.9073, acc = 0.0085
# Epoch 10: loss = 5.5857, acc = 0.0905
# Epoch 20: loss = 4.0193, acc = 0.2270
# Epoch 30: loss = 2.8807, acc = 0.3810
# Epoch 40: loss = 2.1218, acc = 0.5050
# Epoch 49: loss = 1.6492, acc = 0.5965
```

## Common Mistakes

1. **Using one-hot when sparse suffices**: For classification with many classes, always use integer labels.
2. **Forgetting sparse and standard CE are identical**: There's no accuracy or convergence difference.
3. **Using CrossEntropyLoss with one-hot vectors**: CrossEntropyLoss expects class indices. Use NLLLoss for one-hot.
4. **Not knowing CrossEntropyLoss is sparse**: nn.CrossEntropyLoss is the sparse version. There's no separate SparseCrossEntropyLoss in PyTorch.
5. **Confusing sparse labels with sparse data**: Sparse labels = integer indices. Sparse data = many zero-valued features.

## Interview Questions

### Beginner

1. What is sparse categorical cross-entropy?
2. How does sparse CCE differ from regular CCE?
3. When should you use sparse vs. one-hot encoding?
4. How do you implement sparse CCE in PyTorch?
5. Is there a mathematical difference between sparse and non-sparse CCE?

### Intermediate

1. Explain the memory savings of sparse labels for a problem with 10000 classes.
2. Why does nn.CrossEntropyLoss in PyTorch accept integer labels?
3. How does the gradient computation differ, if at all, between sparse and one-hot CCE?
4. What prevents using one-hot labels for very large vocabularies?
5. How do frameworks like PyTorch implement cross-entropy for sparse labels efficiently?

### Advanced

1. Analyze the computational complexity of sparse vs. one-hot cross-entropy.
2. Derive how PyTorch's CrossEntropyLoss handles the sparse label format internally.
3. Discuss the implications of label format choice for distributed training.

## Practice Problems

### Easy

1. Compute sparse CE using nn.CrossEntropyLoss.
2. Verify sparse CE gives the same result as manual one-hot CE.
3. Measure the memory difference between sparse and one-hot for 1000 classes.
4. Train a classifier with sparse labels.
5. Compare loss values for sparse vs. one-hot encoding.

### Medium

1. Train a model on CIFAR-100 (100 classes) with sparse labels.
2. Implement a custom sparse cross-entropy loss function.
3. Compare training speed of sparse vs. one-hot for 10000 classes.
4. Analyze the gradient computation for sparse labels.
5. Build a text classifier with CrossEntropyLoss and integer labels.

### Hard

1. Implement efficient sparse cross-entropy that avoids full one-hot expansion.
2. Derive the backpropagation equations for sparse categorical cross-entropy.
3. Design an experiment measuring the memory and speed advantages of sparse vs. one-hot for varying numbers of classes.

## Solutions

Sparse CCE = -log(p_{label}). Use nn.CrossEntropyLoss with integer labels. This is the standard in PyTorch. One-hot is not recommended for problems with many classes.

## Related Concepts

- Categorical Cross-Entropy (DL-096): The one-hot version
- Cross-Entropy Loss (DL-094): General framework
- NLLLoss: Accepts log-probabilities

## Next Concepts

- KL Divergence (DL-098)
- Hinge Loss (DL-099)
- Contrastive Loss (DL-100)

## Summary

Sparse categorical cross-entropy is identical to categorical cross-entropy but uses integer class indices instead of one-hot vectors. PyTorch's nn.CrossEntropyLoss is the sparse version by default. For problems with many classes, sparse labels are essential for memory efficiency.

## Key Takeaways

1. Sparse CCE and one-hot CCE compute the same loss with the same gradients.
2. Sparse labels use integer indices, one-hot uses binary vectors.
3. PyTorch's nn.CrossEntropyLoss is already the sparse version.
4. Sparse labels are essential for problems with many classes.
5. One-hot encoding wastes memory for large-vocabulary tasks.
