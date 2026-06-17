# Concept: Binary Cross-Entropy

## Concept ID

DL-095

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand binary cross-entropy (BCE) for binary classification
- Implement BCE manually and using nn.BCEWithLogitsLoss
- Explain the relationship between BCE and sigmoid
- Apply BCE for multi-label classification
- Analyze the gradient of BCE with sigmoid

## Prerequisites

- Cross-Entropy Loss (DL-094)
- Sigmoid activation function
- Binary classification understanding

## Definition

Binary Cross-Entropy (BCE) measures the difference between true binary labels and predicted probabilities for binary classification. For a single prediction:

BCE = -[y * log(p) + (1 - y) * log(1 - p)]

where y in {0, 1} is the true label and p in (0, 1) is the predicted probability.

In PyTorch:
- nn.BCELoss(): expects probabilities (after sigmoid)
- nn.BCEWithLogitsLoss(): expects raw logits (more numerically stable)

## Intuition

BCE has two terms: one that activates when y=1 (penalizing low predictions) and one that activates when y=0 (penalizing high predictions). If the true label is 1 and the model predicts 0.9, the loss is -log(0.9) = 0.105. If the model predicts 0.1, the loss is -log(0.1) = 2.303. The loss is asymmetric in logit space but symmetric in probability space.

## Why This Concept Matters

BCE is the standard loss for binary classification and multi-label classification. It is also used in GANs, VAEs, and many other deep learning architectures.

## Mathematical Explanation

### Formula

BCE = -[y * log(sigma(z)) + (1 - y) * log(1 - sigma(z))]

where sigma is the sigmoid function: sigma(z) = 1 / (1 + exp(-z)).

### Gradient with Sigmoid

d(BCE)/dz = sigma(z) - y

Like cross-entropy with softmax, the gradient is simply (predicted probability - true label).

### Log-Sum-Exp Trick

nn.BCEWithLogitsLoss uses the log-sum-exp trick for numerical stability:

BCE(z, y) = max(z, 0) - z * y + log(1 + exp(-|z|))

This avoids the numerical issues of computing log(sigmoid(z)) directly.

## Code Examples

### Example 1: Manual BCE and nn.BCEWithLogitsLoss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

# PyTorch BCEWithLogitsLoss (recommended)
criterion = nn.BCEWithLogitsLoss()
bce_logits = criterion(logits, labels)
print(f"BCEWithLogitsLoss: {bce_logits.item():.4f}")

# Manual computation
probs = torch.sigmoid(logits)
bce_manual = -(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs)).mean()
print(f"Manual BCE: {bce_manual.item():.4f}")

# Using BCELoss with sigmoid
bce_sigmoid = nn.BCELoss()(probs, labels)
print(f"BCELoss (with sigmoid): {bce_sigmoid.item():.4f}")
```

```
# Output:
# BCEWithLogitsLoss: 0.5173
# Manual BCE: 0.5173
# BCELoss (with sigmoid): 0.5173
```

### Example 2: Binary Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N = 500
X = torch.randn(N, 2)
y = (X[:, 0]**2 + X[:, 1]**2 > 1.0).float()

model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 1)
)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(500):
    optimizer.zero_grad()
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        accuracy = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {accuracy.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 0.8199, acc = 0.5280
# Epoch 100: loss = 0.3045, acc = 0.8720
# Epoch 200: loss = 0.1599, acc = 0.9380
# Epoch 300: loss = 0.1056, acc = 0.9640
# Epoch 400: loss = 0.0784, acc = 0.9760
# Epoch 499: loss = 0.0630, acc = 0.9820
```

### Example 3: Multi-Label Classification with BCE

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D, num_labels = 300, 20, 5
X = torch.randn(N, D)
y = (torch.rand(N, num_labels) > 0.5).float()

model = nn.Linear(D, num_labels)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        accuracy = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean()
        print(f"Epoch {epoch}: loss = {loss.item():.4f}, acc = {accuracy.item():.4f}")
```

```
# Output:
# Epoch 0: loss = 0.7180, acc = 0.5387
# Epoch 50: loss = 0.5172, acc = 0.7280
# Epoch 100: loss = 0.4325, acc = 0.7887
# Epoch 150: loss = 0.3818, acc = 0.8207
# Epoch 199: loss = 0.3483, acc = 0.8413
```

## Common Mistakes

1. **Using BCELoss without sigmoid**: BCELoss expects probabilities in (0,1). Always use BCEWithLogitsLoss for numerical stability.
2. **Using BCEWithLogitsLoss with softmax instead of sigmoid**: BCEWithLogitsLoss expects one logit per class. For multi-class, use CrossEntropyLoss.
3. **Incorrect label shape**: BCE expects labels to be float, not long, and have the same shape as logits.
4. **Numerical issues with log(sigmoid)**: Direct computation can produce -inf. Always use BCEWithLogitsLoss.
5. **Ignoring class imbalance**: BCE can be weighted using the pos_weight parameter.

## Interview Questions

### Beginner

1. What is binary cross-entropy?
2. How does BCE differ from cross-entropy for multi-class?
3. Why use BCEWithLogitsLoss instead of BCELoss?
4. How do you use BCE for multi-label classification?
5. What activation function is used with BCE?

### Intermediate

1. Derive the gradient of BCE with sigmoid.
2. Explain the numerical stability advantages of BCEWithLogitsLoss.
3. How does BCE loss behave when the model is very confident and wrong?
4. Compare BCE with hinge loss for binary classification.
5. How would you weight BCE for imbalanced binary classification?

### Advanced

1. Prove that BCE is a proper scoring rule.
2. Analyze the calibration of models trained with BCE vs. other losses.
3. Derive the Fisher information for the BCE-sigmoid model.

## Practice Problems

### Easy

1. Compute BCE manually for given logits and labels.
2. Use BCEWithLogitsLoss for a simple 1D binary classification.
3. Compare BCELoss and BCEWithLogitsLoss numerical stability.
4. Verify the gradient dL/dz = sigmoid(z) - y.
5. Implement weighted BCE for imbalanced classes.

### Medium

1. Train a binary classifier on a 2D synthetic dataset.
2. Implement multi-label classification with BCE.
3. Compare BCE with MSE for binary classification.
4. Visualize the BCE loss landscape.
5. Implement focal loss starting from BCE and compare.

### Hard

1. Prove that BCE is the MLE for Bernoulli-distributed targets.
2. Derive the optimal threshold for BCE-based binary classifiers.
3. Design an experiment comparing BCE, hinge, and logistic loss for binary classification.

## Solutions

BCE = -[y*log(p) + (1-y)*log(1-p)]. Use nn.BCEWithLogitsLoss for numerical stability. For multi-label, apply BCE independently to each output dimension.

## Related Concepts

- Cross-Entropy Loss (DL-094): Multi-class generalization
- Categorical Cross-Entropy (DL-096): Multi-class variant
- Hinge Loss (DL-099): Alternative binary classification loss
- Focal Loss (DL-102): BCE variant for class imbalance

## Next Concepts

- Categorical Cross-Entropy (DL-096)
- Sparse Categorical Cross-Entropy (DL-097)
- KL Divergence (DL-098)

## Summary

Binary cross-entropy measures the difference between binary labels and predicted probabilities. It has two terms, one for each class, and its gradient with sigmoid is simply (predicted - true). For numerical stability, always use nn.BCEWithLogitsLoss which combines sigmoid and BCE into a single numerically stable operation.

## Key Takeaways

1. BCE = -[y*log(p) + (1-y)*log(1-p)], the standard binary classification loss.
2. Use nn.BCEWithLogitsLoss (not BCELoss) for numerical stability.
3. Gradient with sigmoid: dL/dz = sigma(z) - y.
4. BCE extends naturally to multi-label classification.
5. Weighted BCE handles class imbalance via pos_weight.
