# Concept: Multilabel Image Classification

## Concept ID

DL-226

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand multilabel classification and its applications
- Implement multilabel classification in PyTorch
- Choose appropriate loss functions and evaluation metrics
- Handle label correlations and class imbalance

## Prerequisites

DL-225 Classification Head, DL-105 Loss Functions

## Definition

Multilabel image classification assigns multiple labels to each image simultaneously, where each label is binary (present/absent), requiring the model to predict a set of relevant labels rather than a single class.

## Intuition

An image might contain a cat, a dog, and a ball — all simultaneously. Multilabel classification must detect all of them. Unlike multiclass (one label per image) where labels compete via softmax, multilabel uses independent binary classifiers per label, typically with sigmoid activation. Each label is treated as a separate "is this present?" question, and multiple answers can be "yes."

## Why This Concept Matters

Multilabel classification reflects real-world complexity — most images contain multiple objects, scenes, and attributes. Applications include tag prediction, attribute recognition, scene understanding, and medical image diagnosis (where multiple conditions may coexist).

## Mathematical Explanation

**Problem formulation**:
For input $x$, predict $y \in \{0,1\}^C$ where $C$ is the number of labels.

**Binary cross-entropy loss** (per-label):
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} [y_{i,c} \log(\hat{y}_{i,c}) + (1-y_{i,c}) \log(1-\hat{y}_{i,c})]$$

where $\hat{y}_{i,c} = \sigma(z_c)$ (sigmoid output).

**Evaluation metrics**:
- **Precision@k**: Fraction of top-k predictions that are correct
- **Recall@k**: Fraction of ground-truth labels in top-k predictions
- **mAP (mean Average Precision)**: Average precision across labels
- **F1 score**: Harmonic mean of precision and recall

## Code Examples

### Example 1: Multilabel Model

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class MultilabelCNN(nn.Module):
    def __init__(self, num_labels=20):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_labels)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)  # Raw logits (no sigmoid here)

# Create model and data
model = MultilabelCNN(num_labels=20)
x = torch.randn(8, 3, 64, 64)
y = (torch.rand(8, 20) > 0.7).float()  # Multiple labels per image

# Forward pass with sigmoid for probabilities
logits = model(x)
probabilities = torch.sigmoid(logits)

print(f"Input: {x.shape}")
print(f"Logits: {logits.shape}")
print(f"Probabilities: {probabilities.shape}")
print(f"Ground-truth labels per image: {y.sum(dim=1).tolist()}")
print(f"Predicted labels per image: {(probabilities > 0.5).sum(dim=1).tolist()}")
```

### Example 2: Multilabel Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
N = 1000
num_labels = 20
X = torch.randn(N, 3, 64, 64)
y = (torch.rand(N, num_labels) > 0.85).float()

train_X, val_X = X[:800], X[800:]
train_y, val_y = y[:800], y[800:]

train_loader = DataLoader(TensorDataset(train_X, train_y), 
                          batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)

# Model
model = MultilabelCNN(num_labels=num_labels).to(device)

# Binary cross-entropy loss (for multilabel)
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            predictions = (outputs > 0.5).float()
            
            # Per-label accuracy
            correct += (predictions == labels).float().sum().item()
            total += labels.numel()
    
    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, "
          f"Label Acc={acc:.2f}%")
```

### Example 3: Multilabel Metrics

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

def multilabel_metrics(predictions, targets, k=3):
    """Compute multilabel classification metrics."""
    # Precision@k and Recall@k
    _, topk_idx = predictions.topk(k, dim=1)
    targets_expanded = targets.gather(1, topk_idx)
    
    precision_at_k = targets_expanded.sum(dim=1).float().mean().item() / k
    recall_at_k = targets_expanded.sum(dim=1).float().mean().item() / \
                  targets.sum(dim=1).float().clamp(min=1).mean().item()
    
    # F1 score (per-label, then macro)
    pred_bin = (predictions > 0.5).float()
    tp = (pred_bin * targets).sum(dim=0)
    fp = (pred_bin * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_bin) * targets).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        f'Precision@{k}': precision_at_k,
        f'Recall@{k}': recall_at_k,
        'Per-label F1': f1.tolist(),
        'Macro F1': f1.mean().item(),
    }

# Example
logits = torch.randn(32, 20)
targets = (torch.rand(32, 20) > 0.8).float()

metrics = multilabel_metrics(logits, targets, k=5)
for name, value in metrics.items():
    if isinstance(value, list):
        print(f"{name}: {[f'{v:.3f}' for v in value[:5]]}...")
    else:
        print(f"{name}: {value:.4f}")
```

## Common Mistakes

1. **Using softmax instead of sigmoid**: Softmax forces competition between labels; sigmoid allows multiple labels.
2. **Using CrossEntropyLoss instead of BCEWithLogitsLoss**: CE is for single-label; BCE is for multilabel.
3. **Ignoring label imbalance**: Some labels are rare; use class weighting or focal loss.
4. **Evaluating with accuracy only**: For multilabel, precision@k, recall@k, and mAP are more informative.
5. **Using 0.5 threshold blindly**: Optimal threshold varies per label; use label-specific thresholds.

## Interview Questions

### Beginner - 5
1. What is multilabel classification?
2. How does multilabel differ from multiclass classification?
3. What activation function is used in multilabel output?
4. What loss function is used?
5. What is an example application of multilabel classification?

### Intermediate - 5
1. Explain BCEWithLogitsLoss and its advantages.
2. How do you evaluate multilabel models?
3. What is mean Average Precision (mAP)?
4. How do you handle label correlations?
5. Compare multilabel with multi-class and multi-output classification.

### Advanced - 3
1. Design a model that captures label dependencies (e.g., with a CRF).
2. Implement a class-aware threshold selection strategy.
3. Analyze the impact of label imbalance and propose solutions.

## Practice Problems

### Easy - 5
1. Change a multiclass model to multilabel.
2. Implement BCEWithLogitsLoss training loop.
3. Compute precision@3 for multilabel predictions.
4. Convert single-label dataset to multilabel format.
5. Implement sigmoid output thresholding.

### Medium - 5
1. Implement weighted BCE for class imbalance.
2. Train a multilabel model on COCO or similar dataset.
3. Implement macro-F1 and micro-F1 evaluation.
4. Add label correlation modeling (pairwise loss).
5. Implement focal loss for multilabel.

### Hard - 3
1. Design a graph-based label dependency model.
2. Implement a learned threshold per label.
3. Analyze the label co-occurrence patterns and model them.

## Solutions

### Easy - 1 Solution
```python
# Replace nn.Linear(num_classes) with nn.Linear(num_labels)
# Replace nn.CrossEntropyLoss with nn.BCEWithLogitsLoss
model = MultilabelCNN(num_labels=20)  # sigmoid applied in loss
criterion = nn.BCEWithLogitsLoss()
```

## Related Concepts

DL-105 Loss Functions, DL-225 Classification Head, DL-227 Hierarchical Classification

## Next Concepts

DL-227 Hierarchical Classification

## Summary

Multilabel classification assigns multiple labels per image using independent sigmoid outputs and binary cross-entropy loss. It's essential for real-world applications where multiple objects, attributes, or conditions coexist. Evaluation requires specialized metrics like mAP and precision@k.

## Key Takeaways

- Multilabel: multiple labels can be active simultaneously
- Sigmoid (not softmax) for per-label binary prediction
- BCEWithLogitsLoss = sigmoid + binary cross-entropy
- Evaluation: precision@k, recall@k, mAP, macro-F1
- Label imbalance is a significant challenge
- Threshold can be tuned per label (not fixed at 0.5)
- Label correlations can improve predictions
- Applications: tags, attributes, scene labels, medical diagnosis
- More challenging than single-label classification
- Modern approaches use transformer-based models for better label correlation capture
