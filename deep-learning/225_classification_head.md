# Concept: Classification Head

## Concept ID

DL-225

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the role of the classification head
- Implement different classification head designs
- Replace classification heads for transfer learning
- Analyze the impact of head design on model performance

## Prerequisites

DL-191 Global Average Pooling, DL-222 Training Image Classifiers, DL-224 Feature Extraction

## Definition

The classification head is the final part of a classification network that transforms the learned feature representation into class predictions, typically consisting of global pooling followed by one or more fully connected layers with a softmax or sigmoid output.

## Intuition

The feature extractor (backbone) is like a team of experts analyzing the image from different angles, producing a rich set of observations. The classification head is the decision-maker who looks at all the expert reports and makes a final ruling: "this is a cat." The head must summarize the spatial information (pooling), combine the features (linear layers), and produce calibrated probabilities (softmax). The design of the head — how many layers, how much dropout, what activation — affects both accuracy and computational cost.

## Why This Concept Matters

The classification head is where the final decision is made. Proper head design is crucial for transfer learning (replacing the head), multi-task learning (multiple heads), and achieving good calibration. Many modern architectures use a simple GAP + Linear head, but more complex heads can improve performance.

## Mathematical Explanation

**Standard classification head**:
$$p(y=c|x) = \text{softmax}(W \cdot \text{GAP}(F) + b)$$

**Multi-layer head**:
$$h_1 = \sigma(W_1 \cdot \text{GAP}(F) + b_1)$$
$$h_2 = \sigma(W_2 \cdot h_1 + b_2)$$
$$p(y=c|x) = \text{softmax}(W_3 \cdot h_2 + b_3)$$

**Parameter count** for head with GAP feature dimension $d$ and $C$ classes:
- Linear head: $d \cdot C + C$
- 2-layer head with hidden size $h$: $d \cdot h + h + h \cdot C + C$

**Head design considerations**:
- Dropout rate (typically 0.2-0.5)
- Hidden layer size
- Number of layers
- Weight initialization

## Code Examples

### Example 1: Different Classification Head Designs

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Different classification head styles
class LinearHead(nn.Module):
    """Simple linear classifier."""
    def __init__(self, in_features, num_classes, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class MLPHead(nn.Module):
    """Multi-layer perceptron classifier head."""
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)

class BottleneckHead(nn.Module):
    """Bottleneck head with dimension reduction."""
    def __init__(self, in_features, num_classes, bottleneck_dim=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, num_classes)
    
    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        x = self.bn(self.bottleneck(x))
        return self.fc(x)

# Test all heads
features = torch.randn(8, 512, 7, 7)

linear_head = LinearHead(512, 10)
mlp_head = MLPHead(512, 10)
bottleneck_head = BottleneckHead(512, 10)

for name, head in [('Linear', linear_head), ('MLP', mlp_head), 
                   ('Bottleneck', bottleneck_head)]:
    out = head(features)
    params = sum(p.numel() for p in head.parameters())
    print(f"{name:15s}: {out.shape}, {params:,} params")
```

### Example 2: Replacing Classification Head

```python
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(42)

# Different model backbones and their head dimensions
backbones = {
    'resnet18': models.resnet18(pretrained=False),
    'resnet50': models.resnet50(pretrained=False),
    'mobilenet_v2': models.mobilenet_v2(pretrained=False),
    'efficientnet_b0': models.efficientnet_b0(pretrained=False),
}

num_classes = 10

for name, model in backbones.items():
    # Find the final classifier layer
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name:20s}: head in_features={in_features:5d}, "
          f"total params={total_params/1e6:.2f}M")
```

### Example 3: Impact of Head Design on Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Compare simple vs complex head training
feature_dim = 512
N = 2000

X = torch.randn(N, feature_dim)
y = torch.randint(0, 5, (N,))
train_X, val_X = X[:1500], X[1500:]
train_y, val_y = y[:1500], y[1500:]

train_loader = DataLoader(TensorDataset(train_X, train_y), 
                          batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)

heads = {
    'Linear (no dropout)': nn.Sequential(nn.Linear(feature_dim, 5)),
    'Linear (dropout)': nn.Sequential(nn.Dropout(0.3), nn.Linear(feature_dim, 5)),
    'MLP': nn.Sequential(
        nn.Dropout(0.3), nn.Linear(feature_dim, 256), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(256, 5),
    ),
}

for name, head in heads.items():
    head = head.to(device)
    optimizer = optim.Adam(head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(30):
        head.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(head(bx), by)
            loss.backward()
            optimizer.step()
    
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            _, pred = head(bx).max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
    
    params = sum(p.numel() for p in head.parameters())
    print(f"{name:25s}: Val Acc={100.*correct/total:.2f}%, Params={params}")
```

## Common Mistakes

1. **Overly complex head for small datasets**: Complex heads overfit when data is limited.
2. **No dropout in the head**: The classifier head is prone to overfitting; dropout helps.
3. **Incorrect pooling**: Forgetting GAP before the linear layer leads to shape mismatches.
4. **Wrong bias handling**: The final FC layer should have bias for softmax.
5. **Using softmax in the head definition**: nn.CrossEntropyLoss expects raw logits, not softmax outputs.

## Interview Questions

### Beginner - 5
1. What is the classification head?
2. What is the simplest classification head?
3. Why is GAP used before the classifier?
4. What activation is used at the output?
5. How do you modify the head for transfer learning?

### Intermediate - 5
1. Compare linear vs MLP classifier heads.
2. How does dropout in the head prevent overfitting?
3. Why do modern architectures use simple heads?
4. What is the role of the bottleneck in the head?
5. How does the head design affect model calibration?

### Advanced - 3
1. Design a multi-task head for classification + regression.
2. Implement a learnable pooling layer in the head.
3. Analyze the gradient flow through different head designs.

## Practice Problems

### Easy - 5
1. Replace ResNet-50's classifier for 5 classes.
2. Add dropout to a classification head.
3. Compare GAP vs max pooling in the head.
4. Count parameters in a 2-layer head.
5. Remove the classification head to use as feature extractor.

### Medium - 5
1. Implement a classification head with learnable temperature scaling.
2. Compare training with different head complexities.
3. Add a normalization layer (LayerNorm, BatchNorm) to the head.
4. Implement a weight-normalized classification head.
5. Build a multi-head architecture for multiple classification tasks.

### Hard - 3
1. Design a classification head with uncertainty estimation.
2. Implement a cosine-similarity-based classifier head.
3. Analyze the feature geometry before and after the classification head.

## Solutions

### Easy - 1 Solution
```python
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
```

## Related Concepts

DL-191 Global Average Pooling, DL-224 Feature Extraction, DL-223 Finetuning

## Next Concepts

DL-226 Multilabel Image Classification

## Summary

The classification head transforms learned feature representations into class predictions. Modern heads are typically simple (GAP + Linear + Softmax), but more complex designs can improve performance for specific tasks. Proper head design is critical for effective transfer learning.

## Key Takeaways

- Classification head = pooling + fully connected layers + output activation
- GAP produces fixed-size features regardless of input size
- Simple linear heads work well with large backbones
- MLP heads add capacity but risk overfitting
- Dropout in the head prevents overfitting
- No softmax in the head (CrossEntropyLoss expects logits)
- Head replacement is the primary mechanism for transfer learning
- Head design affects model calibration and uncertainty
- Modern trend: simpler heads (GAP + single linear layer)
- Feature dimension varies by backbone (ResNet-18: 512, ResNet-50: 2048)
