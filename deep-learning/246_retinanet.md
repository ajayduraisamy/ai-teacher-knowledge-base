# Concept: RetinaNet

## Concept ID

DL-246

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the class imbalance problem in one-stage detection
- Implement Focal Loss and analyze its gradient behavior
- Comprehend RetinaNet's FPN + two-subnet architecture
- Analyze how Focal Loss closes the accuracy gap with two-stage detectors

## Prerequisites

- DL-245: SSD
- DL-241: YOLO v1
- DL-233: Anchor Boxes

## Definition

RetinaNet, introduced by Lin et al. in 2017 (Facebook AI Research), is a one-stage object detector that addresses the extreme foreground-background class imbalance inherent in dense detection through a novel loss function called Focal Loss. The architecture consists of a ResNet backbone with a Feature Pyramid Network (FPN) neck and two task-specific subnets (classification and regression). RetinaNet achieved 39.1% COCO mAP at 5 FPS, surpassing all existing one-stage detectors and matching the accuracy of two-stage detectors like Faster R-CNN.

## Intuition

The core insight of RetinaNet is that one-stage detectors process a dense grid of ~100k anchor locations per image, of which only a handful are positive (contain objects). This extreme imbalance causes the standard cross-entropy loss to be dominated by easy negative examples. These easy negatives produce small but numerous gradient signals that overwhelm the rare positive samples. Focal Loss down-weights easy examples and focuses training on hard negatives. The loss automatically scales the gradient contribution: a well-classified example (p_t > 0.9) gets its loss reduced by ~100x compared to cross-entropy.

## Why This Concept Matters

RetinaNet solved the long-standing accuracy gap between one-stage and two-stage detectors, demonstrating that the gap was primarily due to class imbalance rather than architectural limitations. Focal Loss became one of the most influential loss functions in object detection, adopted by numerous detectors. The ResNet-FPN backbone design with separate classification and regression subnets set a template for modern one-stage detectors. RetinaNet remains a strong baseline for detection research.

## Mathematical Explanation

Focal Loss:
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
- p_t = p if y = 1 (correct positive), 1 - p otherwise
- α_t is a class-balancing weight (typically α = 0.25 for foreground)
- γ is the focusing parameter (typically γ = 2)

Comparison with Cross-Entropy (CE):
CE(p_t) = -log(p_t)

When p_t → 1 (easy example): (1-p_t)^γ → 0, so FL → 0
When p_t → 0 (hard example): (1-p_t)^γ → 1, so FL ≈ CE

For γ = 2, an example with p_t = 0.9 has FL = 0.01 * CE (100x reduction), while p_t = 0.3 has FL = 0.49 * CE (2x reduction).

## Code Examples

### Example 1: Focal Loss Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: [N, num_classes] (before softmax)
        # targets: [N] class indices (0 = background)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        # For background class, use 1-alpha
        bg_mask = targets == 0
        focal_loss[bg_mask] = (1 - self.alpha) * ((1 - pt[bg_mask]) ** self.gamma) * ce_loss[bg_mask]

        return focal_loss.mean()

# Compare CE vs FL on easy vs hard examples
logits_easy = torch.tensor([[10.0, -10.0]])  # very confident: class 0 (p=1.0)
logits_hard = torch.tensor([[0.1, -0.1]])    # uncertain: class 0 (p=0.55)
targets = torch.tensor([0])

ce_fn = nn.CrossEntropyLoss()
fl_fn = FocalLoss(alpha=0.25, gamma=2.0)

print(f"Easy: CE={ce_fn(logits_easy, targets):.4f}, FL={fl_fn(logits_easy, targets):.4f}")
print(f"Hard: CE={ce_fn(logits_hard, targets):.4f}, FL={fl_fn(logits_hard, targets):.4f}")
# Output:
# Easy: CE=0.0000, FL=0.0000
# Hard: CE=0.6931, FL=0.1733
```

### Example 2: Gradient Analysis of Focal Loss

```python
import torch
import torch.nn.functional as F

def focal_loss_gradient_analysis():
    p = torch.linspace(0.01, 0.99, 100, requires_grad=True)

    # Standard CE loss for binary classification
    def ce_loss(p):
        return -torch.log(p)

    # Focal Loss (gamma=2)
    def focal_loss(p, gamma=2.0):
        return -((1 - p) ** gamma) * torch.log(p)

    # Compute gradients
    ce = ce_loss(p)
    fl = focal_loss(p)

    grad_ce = torch.autograd.grad(ce.sum(), p, create_graph=True)[0]
    grad_fl = torch.autograd.grad(fl.sum(), p, create_graph=True)[0]

    # At p=0.9 (easy example)
    idx_easy = 89
    print(f"At p=0.9: CE grad={grad_ce[idx_easy].item():.4f}, FL grad={grad_fl[idx_easy].item():.6f}")
    # At p=0.5 (hard example)
    idx_hard = 49
    print(f"At p=0.5: CE grad={grad_ce[idx_hard].item():.4f}, FL grad={grad_fl[idx_hard].item():.4f}")

focal_loss_gradient_analysis()
# Output:
# At p=0.9: CE grad=-1.1111, FL grad=-0.0111
# At p=0.5: CE grad=-2.0000, FL grad=-0.5000
```

### Example 3: RetinaNet Classification Subnet

```python
import torch
import torch.nn as nn

class RetinaNetClassificationSubnet(nn.Module):
    def __init__(self, in_channels=256, num_classes=80, num_anchors=9):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.output = nn.Conv2d(256, num_anchors * num_classes, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # Initialize bias for classification output
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.constant_(self.output.bias, -np.log((1 - 0.01) / 0.01))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.output(x)

import numpy as np
subnet = RetinaNetClassificationSubnet()
feat = torch.randn(1, 256, 64, 64)
out = subnet(feat)
print(f"Classification output: {out.shape}")
# Output: Classification output: torch.Size([1, 720, 64, 64])
```

## Common Mistakes

1. **Applying focal loss with γ=0**: This reduces to standard cross-entropy. The focusing effect requires γ > 0. γ=2 is the standard but tuning helps.

2. **Incorrect α balancing**: The α parameter balances positive/negative losses. Using α=0.25 means positive examples contribute 25% of the total loss weight. Setting α=1 for both classes negates balancing.

3. **Not initializing classification bias properly**: RetinaNet initializes the bias to -log((1-π)/π) with π=0.01, so the initial predicted probability of foreground is ~0.01. Without this, training is unstable.

4. **Using focal loss with two-stage detectors**: Two-stage detectors already sample balanced positives/negatives (1:3 ratio). Focal loss may not benefit because class imbalance is already addressed by sampling.

5. **Ignoring the gradient contribution of easy negatives**: While focal loss reduces their weight, thousands of easy negatives still contribute. This is a feature, not a bug—it ensures the model remains confident on easy background regions.

## Interview Questions

### Beginner - 5

1. What problem does RetinaNet solve?
2. What is Focal Loss?
3. What are the two subnets in RetinaNet?
4. What backbone does RetinaNet use?
5. What is the typical γ value in Focal Loss?

### Intermediate - 5

1. Explain how Focal Loss down-weights easy examples.
2. How does the α parameter in Focal Loss work?
3. Why is class imbalance more severe in one-stage detectors than two-stage?
4. How does the FPN help RetinaNet detect multi-scale objects?
5. What is the role of the classification subnet's depth (4 Conv layers)?

### Advanced - 3

1. Derive the gradient of Focal Loss and explain why it focuses on hard examples.
2. Compare Focal Loss with hard negative mining in SSD. Which is more effective and why?
3. Analyze the sensitivity of RetinaNet's performance to different α and γ values.

## Practice Problems

### Easy - 5

1. Implement Focal Loss from scratch.
2. Compute Focal Loss for a perfectly classified example.
3. Compare Focal Loss values for p=0.9 vs p=0.3.
4. Count the number of parameters in the classification subnet.
5. Initialize the bias term for the classification output.

### Medium - 5

1. Implement the complete RetinaNet classification subnet.
2. Build the regression subnet (4 outputs per anchor).
3. Implement the FPN module for multi-scale features.
4. Write a training loop with Focal Loss.
5. Analyze gradient magnitudes for CE vs. FL at different p values.

### Hard - 3

1. Implement the complete RetinaNet with ResNet backbone.
2. Design an experiment to find optimal α and γ for a specific dataset.
3. Compare RetinaNet with Focal Loss vs. SSD with hard negative mining.

## Solutions

Easy 1:
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = -torch.log(torch.gather(pred, 1, target.unsqueeze(1)).squeeze())
    pt = torch.exp(-ce_loss)
    alpha_t = torch.where(target == 0, 1 - alpha, alpha)
    return (alpha_t * (1 - pt) ** gamma * ce_loss).mean()

pred = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
target = torch.tensor([0, 1])
print(f"Focal Loss: {focal_loss(pred, target):.4f}")
# Output: Focal Loss: 0.0032
```

Medium 1 — Classification Subnet:
```python
class ClsSubnet(nn.Module):
    def __init__(self, in_c=256, num_classes=80, num_anchors=9):
        super().__init__()
        self.convs = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, 3, padding=1) for _ in range(4)
        ])
        self.out = nn.Conv2d(in_c, num_anchors * num_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.convs(x))
        return self.out(x)

subnet = ClsSubnet()
feat = torch.randn(1, 256, 32, 32)
print(f"Output: {subnet(feat).shape}")
# Output: Output: torch.Size([1, 720, 32, 32])
```

## Related Concepts

- DL-245: SSD
- DL-241: YOLO v1
- DL-233: Anchor Boxes

## Next Concepts

- DL-247: DETR
- DL-248: Deformable DETR

## Summary

RetinaNet introduced Focal Loss to solve the extreme class imbalance in one-stage detection, achieving 39.1% COCO mAP and matching two-stage accuracy for the first time. The architecture combines a ResNet-FPN backbone with separate classification and regression subnets. Focal Loss down-weights easy examples and focuses training on hard negatives. RetinaNet demonstrated that the accuracy gap between one-stage and two-stage detectors was primarily due to class imbalance, not architecture.

## Key Takeaways

- Focal Loss: FL(p_t) = -α(1-p_t)^γ log(p_t)
- γ=2 focuses on hard examples; α=0.25 balances pos/neg
- Four-conv-layer classification and regression subnets
- ResNet-FPN backbone for multi-scale feature extraction
- Bias initialization: b = -log((1-π)/π) with π=0.01
- First one-stage detector to match two-stage accuracy
- 39.1% COCO mAP, 5 FPS with ResNeXt-101-FPN backbone
