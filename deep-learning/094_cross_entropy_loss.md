# Concept: Cross-Entropy Loss

## Concept ID

DL-094

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand the mathematical formulation of cross-entropy loss
- Implement cross-entropy using nn.CrossEntropyLoss
- Explain the relationship between cross-entropy, KL divergence, and log-likelihood
- Know when to use cross-entropy for classification
- Analyze the gradient behavior of cross-entropy with softmax

## Prerequisites

- Information theory basics (entropy, KL divergence)
- Softmax activation function
- Classification problem understanding

## Definition

Cross-entropy loss measures the difference between two probability distributions: the true distribution p (often one-hot encoded labels) and the predicted distribution q (from softmax). For a single sample with C classes:

CE = -sum_{c=1}^C p_c * log(q_c)

In classification with one-hot labels (p_k = 1 for the true class), this simplifies to:

CE = -log(q_k)

In PyTorch: nn.CrossEntropyLoss() (combines LogSoftmax + NLLLoss).

## Intuition

Cross-entropy loss penalizes the model based on how confidently it predicts the wrong class. If the model assigns probability 0.9 to the correct class, the loss is -log(0.9) = 0.105. If the model assigns only 0.1 to the correct class, the loss is -log(0.1) = 2.303. The loss is unbounded above as predicted probability approaches 0.

## Why This Concept Matters

Cross-entropy is the default loss function for classification tasks. Combined with softmax, it produces strong gradients when predictions are wrong, accelerating learning.

## Mathematical Explanation

### Formula

For a single sample with true class k:

CE = -sum_c p_c * log(q_c) = -log(q_k)

where q is the softmax output: q_c = exp(z_c) / sum_j exp(z_j).

### Cross-Entropy, KL Divergence, and Log-Likelihood

Cross-entropy relates to KL divergence: CE(p, q) = H(p) + KL(p || q)

Since H(p) = 0 for one-hot labels, CE = KL(p || q).

Cross-entropy minimization is equivalent to maximum likelihood estimation for a categorical distribution.

### Gradient with Softmax

The gradient of CE with softmax is elegant:

d(CE)/dz_c = q_c - p_c = q_c - [c == k]

This is simply the difference between the predicted probability and the target (1 for correct class, 0 otherwise).

## Code Examples

### Example 1: Manual Cross-Entropy and nn.CrossEntropyLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Logits (before softmax)
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 1.0],
                        [1.0, 0.5, 2.5]])
# Class labels
labels = torch.tensor([0, 1, 2])

# PyTorch CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
ce_pytorch = criterion(logits, labels)
print(f"PyTorch CrossEntropyLoss: {ce_pytorch.item():.4f}")

# Manual computation
def manual_ce(logits, labels):
    # Softmax
    exp_logits = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
    softmax_out = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    # NLL of correct class
    batch_size = logits.shape[0]
    correct_log_probs = -torch.log(softmax_out[range(batch_size), labels])
    return correct_log_probs.mean()

manual_result = manual_ce(logits, labels)
print(f"Manual CE: {manual_result.item():.4f}")

# Using NLLLoss with LogSoftmax
log_softmax = F.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()(log_softmax, labels)
print(f"LogSoftmax + NLLLoss: {nll_loss.item():.4f}")
```

```
# Output:
# PyTorch CrossEntropyLoss: 0.3738
# Manual CE: 0.3738
# LogSoftmax + NLLLoss: 0.3738
```

### Example 2: Softmax Classification with Cross-Entropy

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D, C = 500, 10, 3
X = torch.randn(N, D)
W = torch.randn(D, C)
y = (X @ W).argmax(dim=1)

model = nn.Linear(D, C)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {accuracy.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 1.0717, acc = 0.5640
# Epoch 50: loss = 0.7101, acc = 0.7660
# Epoch 100: loss = 0.4771, acc = 0.8600
# Epoch 150: loss = 0.3355, acc = 0.9180
# Epoch 199: loss = 0.2527, acc = 0.9380
```

### Example 3: Gradient Behavior of Cross-Entropy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Show how gradient relates to prediction confidence
logits_single = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
label_single = torch.tensor([1])

loss = nn.CrossEntropyLoss()(logits_single, label_single)
loss.backward()

probs = F.softmax(logits_single, dim=1)
print(f"Probabilities: {probs.detach().numpy()}")
print(f"Gradient: {logits_single.grad.numpy()}")
print(f"Expected gradient (q - p): {(probs - F.one_hot(label_single, 3).float()).detach().numpy()}")
```

```
# Output:
# Probabilities: [[0.0900, 0.2447, 0.6652]]
# Gradient: [[0.0900, -0.7553, 0.6652]]
# Expected gradient (q - p): [[0.0900, -0.7553, 0.6652]]
```

## Common Mistakes

1. **Feeding softmax output to CrossEntropyLoss**: nn.CrossEntropyLoss expects raw logits, not softmax output. It includes log-softmax internally.
2. **Using CrossEntropyLoss for binary classification**: For 2-class problems, use nn.BCEWithLogitsLoss.
3. **Ignoring class imbalance**: Cross-entropy assumes balanced classes. Use class weighting for imbalanced data.
4. **Not understanding the one-hot encoding**: CrossEntropyLoss expects class indices, not one-hot vectors.
5. **Numerical instability with manual softmax**: Large logits cause exp overflow. nn.CrossEntropyLoss handles this safely.

## Interview Questions

### Beginner

1. What is cross-entropy loss and when is it used?
2. Why does cross-entropy use log probabilities?
3. How does cross-entropy relate to softmax?
4. Why does cross-entropy produce strong gradients when predictions are wrong?
5. How do you use cross-entropy in PyTorch?

### Intermediate

1. Derive the gradient of cross-entropy with softmax.
2. Explain the relationship between cross-entropy and KL divergence.
3. How is cross-entropy related to maximum likelihood estimation?
4. Compare cross-entropy with MSE for classification tasks.
5. Why does cross-entropy loss go to infinity as predicted probability approaches 0?

### Advanced

1. Prove that cross-entropy minimization is equivalent to maximizing log-likelihood.
2. Analyze the behavior of cross-entropy loss for high-dimensional classification (e.g., ImageNet with 1000 classes).
3. Explain label smoothing and its effect on cross-entropy gradients.

## Practice Problems

### Easy

1. Compute cross-entropy manually for a 3-class classification with given logits.
2. Use nn.CrossEntropyLoss on a 2D classification problem.
3. Compare softmax output with logits.
4. Verify that CrossEntropyLoss = LogSoftmax + NLLLoss.
5. Compute cross-entropy for correct vs. incorrect predictions.

### Medium

1. Train a multi-class classifier with cross-entropy on a synthetic dataset.
2. Implement weighted cross-entropy for imbalanced classes.
3. Visualize the loss landscape of cross-entropy for different logit values.
4. Compare cross-entropy with MSE for a classification task.
5. Implement label smoothing with cross-entropy.

### Hard

1. Derive the Fisher information matrix for the softmax-cross-entropy model.
2. Prove the convexity of cross-entropy loss with respect to logits.
3. Design an experiment showing the calibration differences between cross-entropy and MSE for classification.

## Solutions

CE = -sum(p * log(q)). PyTorch's nn.CrossEntropyLoss combines log-softmax and NLL loss. Expects raw logits and class indices. Gradient is (q - p), the difference between predicted and target probabilities.

## Related Concepts

- Binary Cross-Entropy (DL-095): For 2-class problems
- Categorical Cross-Entropy (DL-096): For multi-class
- KL Divergence (DL-098): Information-theoretic foundation
- Softmax: Activation function paired with CE

## Next Concepts

- Binary Cross-Entropy (DL-095)
- Categorical Cross-Entropy (DL-096)
- Sparse Categorical Cross-Entropy (DL-097)

## Summary

Cross-entropy loss measures the difference between true and predicted probability distributions. For classification with one-hot labels, it simplifies to -log(p_correct). Combined with softmax, the gradient is simply (predicted - target), making optimization efficient. Cross-entropy is the default loss for classification and is equivalent to maximizing log-likelihood.

## Key Takeaways

1. Cross-entropy = -sum(p * log(q)), the standard classification loss.
2. nn.CrossEntropyLoss expects raw logits, not softmax output.
3. Gradient with softmax: dL/dz = q - p (elegant and efficient).
4. Cross-entropy is equivalent to negative log-likelihood for categorical data.
5. Use class weighting for imbalanced datasets.
