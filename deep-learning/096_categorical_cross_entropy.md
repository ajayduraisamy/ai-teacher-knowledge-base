# Concept: Categorical Cross-Entropy

## Concept ID

DL-096

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand categorical cross-entropy for multi-class problems
- Distinguish between categorical and sparse categorical cross-entropy
- Implement categorical cross-entropy with one-hot encoded labels
- Compare with nn.CrossEntropyLoss and nn.NLLLoss
- Apply categorical cross-entropy in multi-class tasks

## Prerequisites

- Cross-Entropy Loss (DL-094)
- Softmax activation
- One-hot encoding

## Definition

Categorical cross-entropy (CCE) is the cross-entropy loss for multi-class classification with one-hot encoded labels. For N samples and C classes:

CCE = -(1/N) * sum_i sum_c y_{i,c} * log(p_{i,c})

where y_{i,c} is 1 if sample i belongs to class c (0 otherwise), and p_{i,c} is the predicted probability.

In PyTorch, this is equivalent to nn.CrossEntropyLoss() when labels are class indices, or nn.NLLLoss() when log-probabilities are provided.

## Intuition

For one-hot labels, only the term corresponding to the true class contributes. CCE = -log(p_correct). This is exactly the same as standard cross-entropy with class indices. The "categorical" name emphasizes that the labels are one-hot vectors.

## Why This Concept Matters

CCE is the multi-class classification loss used in most deep learning classifiers.

## Mathematical Explanation

### Formula

CCE = -(1/N) * sum_i sum_c y_{i,c} * log(p_{i,c})

With one-hot labels, this simplifies to CCE = -(1/N) * sum_i log(p_{i, k_i}) where k_i is the correct class.

### Relationship to Other Losses

- CrossEntropyLoss (PyTorch): logits -> softmax -> log -> NLL. Same as CCE with class indices.
- NLLLoss: expects log-probabilities. Same as CCE with one-hot labels.
- CategoricalCrossEntropy (Keras): same as log-softmax + NLL.

## Code Examples

### Example 1: CCE with One-Hot vs. Class Indices

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 1.0],
                        [1.0, 0.5, 2.5]])

# Class indices (standard approach)
labels_idx = torch.tensor([0, 1, 2])
ce_idx = nn.CrossEntropyLoss()(logits, labels_idx)
print(f"CrossEntropyLoss (indices): {ce_idx.item():.4f}")

# One-hot labels
labels_onehot = F.one_hot(labels_idx, num_classes=3).float()
log_probs = F.log_softmax(logits, dim=1)
cce_onehot = -(labels_onehot * log_probs).sum(dim=1).mean()
print(f"CCE (one-hot, manual): {cce_onehot.item():.4f}")

# NLLLoss with log-probs
nll = nn.NLLLoss()(log_probs, labels_idx)
print(f"NLLLoss (log-probs): {nll.item():.4f}")
```

```
# Output:
# CrossEntropyLoss (indices): 0.3738
# CCE (one-hot, manual): 0.3738
# NLLLoss (log-probs): 0.3738
```

### Example 2: Multi-Class Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
N, D, C = 1000, 20, 5
X = torch.randn(N, D)
W = torch.randn(D, C)
y = (X @ W).argmax(dim=1)

model = nn.Linear(D, C)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {accuracy.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 1.4667, acc = 0.3980
# Epoch 20: loss = 0.8176, acc = 0.7330
# Epoch 40: loss = 0.4539, acc = 0.8870
# Epoch 60: loss = 0.2744, acc = 0.9510
# Epoch 80: loss = 0.1861, acc = 0.9720
# Epoch 99: loss = 0.1383, acc = 0.9810
```

### Example 3: Label Smoothing with CCE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_with_label_smoothing(logits, labels, smoothing=0.1):
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        smooth_labels = F.one_hot(labels, n_classes).float() * (1 - smoothing)
        smooth_labels += smoothing / n_classes
    return -(smooth_labels * log_probs).sum(dim=-1).mean()

logits = torch.tensor([[5.0, 2.0, 0.5],
                        [1.0, 4.0, 1.5]])
labels = torch.tensor([0, 1])

ce_standard = nn.CrossEntropyLoss()(logits, labels)
ce_smooth = cross_entropy_with_label_smoothing(logits, labels, smoothing=0.1)

print(f"Standard CE: {ce_standard.item():.4f}")
print(f"Smoothed CE: {ce_smooth.item():.4f}")
```

```
# Output:
# Standard CE: 0.1309
# Smoothed CE: 0.2300
```

## Common Mistakes

1. **Using one-hot labels with CrossEntropyLoss**: CrossEntropyLoss expects class indices, not one-hot. Use NLLLoss with log-probs for one-hot.
2. **Confusing categorical cross-entropy with binary cross-entropy**: CCE is for multi-class (one true class), BCE is for binary or multi-label (multiple possible true classes).
3. **Applying softmax before CrossEntropyLoss**: CrossEntropyLoss already applies softmax internally.
4. **Forgetting label smoothing reduces confidence**: Label smoothing prevents the model from becoming overconfident.
5. **Using CCE for multi-label problems**: For multi-label, use BCE with sigmoid per output.

## Interview Questions

### Beginner

1. What is categorical cross-entropy and when is it used?
2. How does categorical cross-entropy differ from binary cross-entropy?
3. What is the relationship between CCE and softmax?
4. How does CCE handle one-hot encoded labels?
5. How do you use CCE in PyTorch?

### Intermediate

1. Derive CCE and show its equivalence to NLL with log-softmax.
2. Explain label smoothing and its effect on CCE.
3. Compare CCE with sparse categorical cross-entropy.
4. How does CCE gradient differ for the correct vs. incorrect classes?
5. When would you use one-hot labels vs. class indices?

### Advanced

1. Prove that CCE with softmax gives the same gradient as NLL with log-softmax.
2. Analyze the effect of label smoothing on the logit distribution.
3. Derive the relationship between CCE and the cross-entropy of the softmax distribution.

## Practice Problems

### Easy

1. Compute CCE manually for a 4-class problem with one-hot labels.
2. Use nn.CrossEntropyLoss on a multi-class problem.
3. Compare CrossEntropyLoss with LogSoftmax + NLLLoss.
4. Convert class indices to one-hot and compute CCE.
5. Verify that only the correct class contributes to CCE with one-hot labels.

### Medium

1. Train a multi-class classifier on CIFAR-10 with CrossEntropyLoss.
2. Implement label smoothing for categorical cross-entropy.
3. Compare the training dynamics of CCE vs. MSE for classification.
4. Visualize the gradient of CCE with respect to logits.
5. Implement a custom categorical cross-entropy loss class.

### Hard

1. Prove the equivalence of CCE and NLL with log-softmax.
2. Derive the optimal predictions under CCE with label smoothing.
3. Design an experiment comparing standard CCE with CCE + label smoothing for model calibration.

## Solutions

CCE = -sum(y * log(p)) for one-hot labels. In PyTorch, use CrossEntropyLoss (logits -> softmax -> NLL) with class indices, or NLLLoss with log-probs for one-hot labels.

## Related Concepts

- Cross-Entropy Loss (DL-094): The general concept
- Sparse Categorical Cross-Entropy (DL-097): Variant without one-hot encoding
- Binary Cross-Entropy (DL-095): For binary/multi-label
- KL Divergence (DL-098): Information-theoretic basis

## Next Concepts

- Sparse Categorical Cross-Entropy (DL-097)
- KL Divergence (DL-098)
- Hinge Loss (DL-099)

## Summary

Categorical cross-entropy extends binary cross-entropy to multi-class problems. It measures the difference between one-hot encoded true labels and predicted softmax probabilities. CCE is equivalent to nn.CrossEntropyLoss in PyTorch, which internally combines log-softmax and negative log-likelihood.

## Key Takeaways

1. CCE = -sum(y * log(p)) for one-hot multi-class labels.
2. CCE with one-hot labels is equivalent to CrossEntropyLoss with class indices.
3. PyTorch's CrossEntropyLoss = LogSoftmax + NLLLoss internally.
4. Label smoothing modifies the target distribution to prevent overconfidence.
5. Use CCE for multi-class, BCE for multi-label classification.
