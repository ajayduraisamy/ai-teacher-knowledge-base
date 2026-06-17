# Concept: Finetuning Image Classifiers

## Concept ID

DL-223

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the transfer learning and finetuning paradigm
- Implement finetuning of pretrained models in PyTorch
- Analyze different finetuning strategies
- Choose appropriate finetuning approaches for different scenarios

## Prerequisites

DL-222 Training Image Classifiers, DL-200 ResNet, DL-217 Image Classification Pipeline

## Definition

Finetuning is the process of taking a pretrained model (typically trained on ImageNet) and adapting it to a new task by continuing training on the target dataset, usually with a modified classifier head and carefully tuned learning rates.

## Intuition

Think of finetuning like hiring a chef who already knows basic techniques (knife skills, sauce-making) and teaching them your restaurant's specific recipes. The chef doesn't need to learn cooking from scratch — they just need to adapt their existing skills. Similarly, a model pretrained on ImageNet has already learned general visual features (edges, textures, shapes, object parts). Finetuning adapts these general features to your specific task, requiring far less data and compute than training from scratch.

## Why This Concept Matters

Finetuning is the standard practice for practically all vision applications. It enables high accuracy even with limited data, dramatically reduces training time, and provides strong baselines. Understanding when and how to finetune is one of the most important practical skills in deep learning.

## Mathematical Explanation

**Transfer learning setup**:
- Source domain (ImageNet): $D_S = \{(x^S_i, y^S_i)\}$, $C_S = 1000$ classes
- Target domain (custom): $D_T = \{(x^T_i, y^T_i)\}$, $C_T$ classes (usually $C_T \ll C_S$)

**Model adaptation**:
$$f_{T}(x) = h_T \circ g_S(x)$$

Where $g_S$ is the frozen feature extractor from source task, and $h_T$ is the new classifier head.

**Finetuning strategies**:
1. **Feature extraction**: Freeze $g_S$, train only $h_T$
2. **Full finetuning**: Train all parameters with small LR
3. **Discriminative finetuning**: Different LRs for different layers
4. **Gradual unfreezing**: Progressively unfreeze layers from top to bottom

## Code Examples

### Example 1: Feature Extraction

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
model = models.resnet18(pretrained=False)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # 5 target classes

# Only the new FC layer is trainable
model = model.to(device)

# Verify which parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,} "
      f"({100*trainable_params/total_params:.2f}%)")

# Training (only FC layer updated)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Synthetic data
X = torch.randn(200, 3, 224, 224)
y = torch.randint(0, 5, (200,))
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

model.train()
for epoch in range(5):
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

### Example 2: Full Finetuning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
model = models.resnet50(pretrained=False)

# Replace classifier
model.fc = nn.Linear(2048, 10)

# All parameters are trainable (with small LR)
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

# Differential learning rates
# Feature extractor: small LR, classifier: normal LR
params = [
    {'params': model.conv1.parameters(), 'lr': 0.00001},
    {'params': model.bn1.parameters(), 'lr': 0.00001},
    {'params': model.layer1.parameters(), 'lr': 0.0001},
    {'params': model.layer2.parameters(), 'lr': 0.0001},
    {'params': model.layer3.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.001},
    {'params': model.fc.parameters(), 'lr': 0.01},
]

optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-4)

print("Finetuning with differential learning rates:")
for group in optimizer.param_groups:
    print(f"  LR={group['lr']:.5f}, params={sum(p.numel() for p in group['params']):,}")

# Training would proceed as usual
criterion = nn.CrossEntropyLoss()
print("\nModel ready for finetuning")
```

### Example 3: Gradual Unfreezing

```python
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 5)
model = model.to(device)

# Phase 1: Train only the new classifier
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)

print("Phase 1: Training classifier head only")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# After Phase 1 completes, unfreeze layer4
def unfreeze_stage(model, stage_name, requires_grad=True):
    for name, param in model.named_parameters():
        if stage_name in name:
            param.requires_grad = requires_grad
            print(f"  Unfrozen: {name}")

print("\nPhase 2: Unfreezing layer4")
unfreeze_stage(model, 'layer4')
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("\nPhase 3: Unfreezing all layers")
for param in model.parameters():
    param.requires_grad = True
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

## Common Mistakes

1. **Using too high learning rate for pretrained layers**: Can destroy learned features.
2. **Not freezing batch norm layers during finetuning with small batches**: BN statistics are noisy with small batches.
3. **Training all layers from the start**: Better to start with classifier-only training, then gradually unfreeze.
4. **Forgetting to set model.eval() for frozen BatchNorm layers**: Frozen BN needs to use running statistics.
5. **Using different image preprocessing than what the pretrained model expects**.

## Interview Questions

### Beginner - 5
1. What is finetuning?
2. How does finetuning differ from training from scratch?
3. What is feature extraction?
4. Why use pretrained models?
5. What layers do you typically replace in finetuning?

### Intermediate - 5
1. Explain the different finetuning strategies.
2. How do you choose learning rates for finetuning?
3. What is catastrophic forgetting in finetuning?
4. How does dataset size affect the finetuning strategy?
5. Compare finetuning with and without freezing.

### Advanced - 3
1. Design a finetuning strategy for a very small dataset (100 images).
2. Analyze the feature hierarchy and decide which layers to freeze.
3. Implement a layer-wise finetuning schedule based on gradient statistics.

## Practice Problems

### Easy - 5
1. Load a pretrained ResNet and replace the classifier.
2. Freeze feature extractor and train only the FC layer.
3. Fine-tune all layers with a small learning rate.
4. Add a new classification head with 2 hidden layers.
5. Compare feature extraction vs full finetuning on a small dataset.

### Medium - 5
1. Implement gradual unfreezing over 3 stages.
2. Compare differential learning rates vs uniform LR.
3. Fine-tune a pretrained model on a dataset very different from ImageNet (e.g., medical images).
4. Implement finetuning with mixed precision.
5. Add a custom bottleneck layer before the classifier.

### Hard - 3
1. Design a layer-wise finetuning strategy based on Fisher information.
2. Implement a meta-learning finetuning approach.
3. Analyze the feature drift during finetuning using CKA similarity.

## Solutions

### Easy - 1 Solution
```python
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 5)
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
```

## Related Concepts

DL-224 Feature Extraction with CNNs, DL-225 Classification Head, DL-222 Training Image Classifiers

## Next Concepts

DL-224 Feature Extraction with CNNs, DL-225 Classification Head

## Summary

Finetuning adapts pretrained models to new tasks by continuing training on target data. It's the standard practice for vision applications, enabling high accuracy with limited data. Key strategies range from simple feature extraction (freeze all) to full finetuning with differential learning rates.

## Key Takeaways

- Finetuning transfers general features from pretrained models to new tasks
- Feature extraction: freeze backbone, train new classifier head
- Full finetuning: train everything with small LR
- Differential learning rates: smaller LR for early layers, larger for classifier
- Gradual unfreezing: progressively train more layers
- Critical: match preprocessing to what the pretrained model expects
- Catastrophic forgetting: too much adaptation destroys useful features
- Pretrained models save time, data, and compute
- Standard practice for almost all real-world vision applications
- BatchNorm must be handled carefully during finetuning
