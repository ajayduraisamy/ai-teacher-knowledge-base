# Concept: Focal Loss

## Concept ID

DL-102

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand how focal loss addresses class imbalance
- Implement focal loss in PyTorch
- Explain the focusing parameter gamma
- Apply focal loss for object detection and imbalanced classification
- Compare focal loss with weighted cross-entropy

## Prerequisites

- Binary Cross-Entropy (DL-095)
- Class imbalance concepts
- Object detection (RetinaNet) awareness

## Definition

Focal loss is a variant of binary cross-entropy that down-weights the contribution of easy examples and focuses training on hard negatives. It adds a modulating factor (1 - p_t)^gamma to the cross-entropy loss:

FL(p_t) = -(1 - p_t)^gamma * log(p_t)

where p_t = p if y=1, or p_t = 1-p if y=0, and gamma >= 0 is the focusing parameter.

When gamma = 0, focal loss reduces to cross-entropy.

## Intuition

In object detection, there can be 100,000 easy background examples for every object. Cross-entropy gives each example equal weight, so the model learns to classify everything as background. Focal loss down-weights easy examples: if a background patch is easy (p_t close to 1), its contribution is multiplied by (1 - p_t)^gamma, which is near zero. Hard misclassified examples keep their full weight.

## Why This Concept Matters

Focal loss was proposed in RetinaNet and revolutionized one-stage object detection by matching the accuracy of two-stage detectors. It is now widely used for:
- Object detection (RetinaNet)
- Imbalanced classification
- Medical image analysis
- Any task with extreme class imbalance

## Mathematical Explanation

### Focal Loss Formula

For binary classification with y in {0, 1}:

p_t = p if y = 1, p_t = 1 - p if y = 0

FL(p_t) = -(1 - p_t)^gamma * log(p_t)

### Alpha-Balanced Focal Loss

An alpha-weighting factor can be added:

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where alpha_t = alpha for y=1, alpha_t = 1-alpha for y=0.

### Gradient

d(FL)/dz = alpha_t * (1 - p_t)^gamma * (gamma * p_t * log(p_t) + p_t - 1)

where z is the logit.

## Code Examples

### Example 1: Manual Focal Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # converts CE to probability of correct class
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce_loss).mean()

# Compare CE vs. Focal for easy vs. hard examples
logits = torch.tensor([[3.0], [-2.0], [0.5], [-1.0]])
targets = torch.tensor([[1.0], [0.0], [1.0], [1.0]])

ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
focal_loss = FocalLoss(gamma=2.0, alpha=None)(logits, targets)

print("Example\tlogit\ttarget\tCE\tFocal")
for i in range(4):
    print(f"{i}\t{logits[i].item():+.1f}\t{int(targets[i].item())}\t{ce_loss[i].item():.4f}\t{ce_loss[i].item():.4f}")
```

```
# Output:
# Example   logit   target  CE      Focal
# 0         +3.0    1       0.0486  0.0486
# 1         -2.0    0       0.0180  0.0180
# 2         +0.5    1       0.4741  0.4741
# 3         -1.0    1       1.3133  1.3133
```

### Example 2: Focal Loss for Imbalanced Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
N, D = 1000, 10
X = torch.randn(N, D)
# Imbalanced: 95% class 0, 5% class 1
y = (torch.rand(N) > 0.95).float()

def train_with(loss_fn, label):
    model = nn.Linear(D, 1)
    opt = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(200):
        opt.zero_grad()
        logits = model(X).squeeze()
        if label == "CE":
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X).squeeze()) > 0.5).float()
        acc = (preds == y).float().mean()
        pos_acc = (preds[y == 1] == 1).float().mean() if y.sum() > 0 else 0
        print(f"{label}: acc = {acc.item():.4f}, pos_acc = {pos_acc.item():.4f}")

train_with(None, "CE")
train_with(FocalLoss(gamma=2.0, alpha=0.75), "Focal")
```

```
# Output:
# CE: acc = 0.9500, pos_acc = 0.0000
# Focal: acc = 0.9480, pos_acc = 0.5600
```

### Example 3: Focal Loss vs. CE for Different Gamma Values

```python
import torch
import torch.nn.functional as F

probs = torch.linspace(0.01, 0.99, 50)
gamma_values = [0, 0.5, 1.0, 2.0, 5.0]

print("Gamma\tp=0.9\tp=0.99")
for gamma in gamma_values:
    focal_09 = -((1 - 0.9) ** gamma) * torch.log(torch.tensor(0.9))
    focal_099 = -((1 - 0.99) ** gamma) * torch.log(torch.tensor(0.99))
    print(f"{gamma}\t{focal_09.item():.4f}\t{focal_099.item():.4f}")
```

```
# Output:
# Gamma   p=0.9   p=0.99
# 0.0     0.1054  0.0101
# 0.5     0.0333  0.0010
# 1.0     0.0105  0.0001
# 2.0     0.0011  0.0000
# 5.0     0.0000  0.0000
```

## Common Mistakes

1. **Gamma too high**: With gamma > 5, even moderately easy examples contribute essentially zero loss, making training ineffective.
2. **Gamma too low**: Gamma close to 0 makes focal loss similar to CE, losing the class imbalance benefit.
3. **Not tuning alpha**: The alpha balance parameter is critical and should be tuned alongside gamma.
4. **Using focal loss when class imbalance is mild**: For balanced datasets, focal loss may underperform standard CE.
5. **Applying focal loss to multi-class without adaptation**: The original focal loss is binary. For multi-class, use per-class focal loss.
6. **Not monitoring easy example contribution**: Focal loss can ignore too many examples if gamma is too high and alpha is poorly chosen.

## Interview Questions

### Beginner

1. What problem does focal loss solve?
2. How does focal loss differ from cross-entropy?
3. What does the gamma parameter control?
4. What is the role of alpha in focal loss?
5. How do you implement focal loss in PyTorch?

### Intermediate

1. Derive the focal loss formula and explain the modulating factor.
2. How does focal loss down-weight easy examples?
3. Compare focal loss with weighted cross-entropy.
4. Why is focal loss particularly effective for object detection?
5. How would you tune gamma and alpha?

### Advanced

1. Prove that focal loss is a reweighting of cross-entropy.
2. Analyze the gradient behavior of focal loss for different gamma values.
3. Derive the multi-class extension of focal loss.

## Practice Problems

### Easy

1. Implement focal loss manually.
2. Compare focal loss for gamma values [0, 0.5, 1, 2, 5].
3. Compute focal loss for easy vs. hard examples.
4. Use focal loss for a binary classification task.
5. Verify that focal loss = CE when gamma = 0.

### Medium

1. Train a model on imbalanced data with CE vs. focal loss.
2. Grid search gamma and alpha for optimal performance.
3. Implement multi-class focal loss.
4. Compare focal loss with oversampling/undersampling.
5. Visualize the loss ratio between focal and CE for different probabilities.

### Hard

1. Derive the focal loss gradient and analyze its properties.
2. Implement a focal loss variant with learnable gamma.
3. Design an experiment comparing focal loss, CE, and weighted CE for extreme class imbalance (1:1000 ratio).

## Solutions

Focal loss = -(1 - p_t)^gamma * log(p_t). It down-weights easy examples by a factor of (1-p_t)^gamma. The alpha parameter balances class weights. Default values: gamma = 2.0, alpha = 0.25.

## Related Concepts

- Binary Cross-Entropy (DL-095): Base loss
- Class Imbalance: Problem focal loss addresses
- Dice Loss (DL-103): Alternative for imbalanced segmentation

## Next Concepts

- Dice Loss (DL-103)
- IoU Loss (DL-104)
- Perceptual Loss (DL-105)

## Summary

Focal loss modifies cross-entropy by adding a modulating factor (1-p_t)^gamma that down-weights easy examples and focuses training on hard, misclassified examples. It was designed for one-stage object detection where the class imbalance between foreground and background can be 100,000:1. The gamma parameter controls the rate at which easy examples are down-weighted.

## Key Takeaways

1. Focal loss = -(1 - p_t)^gamma * log(p_t), down-weighting easy examples.
2. Gamma controls the focusing strength (default 2.0).
3. Alpha balances positive/negative class importance.
4. Focal loss revolutionized one-stage object detection (RetinaNet).
5. For balanced datasets, standard cross-entropy may work better.
