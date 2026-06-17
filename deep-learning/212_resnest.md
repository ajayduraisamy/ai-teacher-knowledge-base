# Concept: ResNeSt

## Concept ID

DL-212

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the Split-Attention block in ResNeSt
- Implement ResNeSt in PyTorch
- Analyze how cardinality and attention combine
- Compare ResNeSt with ResNet and ResNeXt

## Prerequisites

DL-200 ResNet, DL-201 ResNeXt, DL-204 SE-Net

## Definition

ResNeSt (Residual Neural Network with Split-Attention) incorporates a Split-Attention mechanism that divides the feature maps into groups (cardinal groups), each of which is further split into subgroups (radix groups), and applies attention within each cardinal group.

## Intuition

ResNeSt combines the best of ResNeXt (cardinality/grouped convs) and SE-Net (channel attention) in a novel way. The feature map is divided into K cardinal groups, and each cardinal group is further split into R radix groups. Within each cardinal group, the radix groups compete for attention — the network learns which radix features are most important and amplifies them. This creates a "soft" version of group selection, where each cardinal group outputs a weighted combination of its radix features.

## Why This Concept Matters

ResNeSt achieved state-of-the-art results across many vision tasks (classification, detection, segmentation) when released. It demonstrated that combining cardinality and attention in this specific way outperforms either alone. The split-attention mechanism has become a useful building block.

## Mathematical Explanation

**Split-Attention block**:
1. Split input into K cardinal groups (each with C/K channels)
2. For each cardinal group, split into R radix groups
3. Apply transformations (1x1 -> 3x3 -> 1x1) to each radix group
4. Fuse within each cardinal group via element-wise sum
5. Apply global pooling and attention to compute radix weights
6. Weighted combination of radix features within each cardinal group
7. Concatenate all cardinal groups and apply skip connection

**Radix**: The number of splits within each cardinal group. Higher radix = more fine-grained attention.

## Code Examples

### Example 1: Split-Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class SplitAttention(nn.Module):
    """Split-Attention block (2D version)."""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, cardinality=1, radix=2):
        super().__init__()
        self.cardinality = cardinality
        self.radix = radix
        self.cardinal_channels = in_channels // cardinality
        
        # Transformations per cardinal group (shared across radix)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Depthwise conv per cardinal group
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size,
                              stride=stride, padding=kernel_size//2,
                              groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Radix attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_conv = nn.Conv2d(in_channels, in_channels * radix, 1)
        self.attn_bn = nn.BatchNorm2d(in_channels * radix)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, 
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        B, C, H, W = x.shape
        
        # Common transformations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Reshape for cardinality and radix
        x = x.view(B, self.cardinality, self.radix, 
                  C // (self.cardinality * self.radix), H, W)
        
        # Sum within each cardinal group for attention
        x_card = x.sum(dim=2)  # (B, card, channels_per_card, H, W)
        
        # Global context for attention
        attn = self.avg_pool(x_card)  # (B, card, channels_per_card, 1, 1)
        attn = attn.view(B, self.cardinality, -1)  # (B, card, ch_per_card)
        attn = torch.softmax(attn, dim=-1).view(
            B, self.cardinality, 1, -1, 1, 1)
        
        # Apply attention to radix groups and sum
        x = (x * attn).sum(dim=2)  # weighted sum over radix
        x = x.view(B, C, H, W)
        
        x = self.bn3(self.conv3(x))
        x = F.relu(x + identity)
        return x

x = torch.randn(1, 64, 32, 32)
block = SplitAttention(64, 128, cardinality=4, radix=2)
out = block(x)
print(f"Split-Attention block: {x.shape} -> {out.shape}")
```

### Example 2: ResNeSt Stage

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ResNeStStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, 
                 cardinality=4, radix=2, stride=1):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(SplitAttention(
                in_channels if i == 0 else out_channels,
                out_channels,
                stride=stride if i == 0 else 1,
                cardinality=cardinality,
                radix=radix,
            ))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)

# Build a small ResNeSt
class ResNeSt(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResNeStStage(64, 64, 2, cardinality=4, radix=2, stride=1)
        self.layer2 = ResNeStStage(64, 128, 2, cardinality=4, radix=2, stride=2)
        self.layer3 = ResNeStStage(128, 256, 2, cardinality=4, radix=2, stride=2)
        self.layer4 = ResNeStStage(256, 512, 2, cardinality=4, radix=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)

model = ResNeSt(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"ResNeSt: {out.shape}, {total_params/1e6:.2f}M params")
```

### Example 3: Radix Ablation

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare different radix configurations
x = torch.randn(1, 128, 28, 28)

for radix in [1, 2, 4]:
    block = SplitAttention(128, 128, cardinality=8, radix=radix)
    out = block(x)
    params = sum(p.numel() for p in block.parameters())
    print(f"Radix={radix}: {out.shape}, {params:,} params")
```

## Common Mistakes

1. **Confusing cardinality and radix**: Cardinality = number of parallel groups; radix = splits within each group.
2. **Not using softmax properly**: Attention weights should sum to 1 within each cardinal group.
3. **Wrong reshape for radix**: Need to carefully manage dimensions for cardinality, radix, and channels.
4. **Forgetting to normalize attention**: Without proper normalization, attention can collapse.
5. **Using too many radix groups**: Radix > 2 gives diminishing returns.

## Interview Questions

### Beginner - 5
1. What is ResNeSt?
2. What is split-attention?
3. What is cardinality in ResNeSt?
4. What is radix?
5. How does ResNeSt improve upon ResNeXt?

### Intermediate - 5
1. Explain the split-attention mechanism in detail.
2. How does radix cardinality differ from standard cardinality?
3. Compare ResNeSt with SE-ResNet.
4. What are the computational costs of adding radix groups?
5. How does ResNeSt perform on detection and segmentation tasks?

### Advanced - 3
1. Design a variant with learnable radix per stage.
2. Analyze the attention patterns learned by the radix mechanism.
3. Compare ResNeSt with Transformer-based attention mechanisms.

## Practice Problems

### Easy - 5
1. Count cardinal groups in a ResNeSt-50.
2. Compute channels per cardinal group for 1024 channels, cardinality=32.
3. Load ResNeSt from timm library.
4. Replace classifier for transfer learning.
5. Compare params: ResNet vs ResNeXt vs ResNeSt.

### Medium - 5
1. Implement a ResNeSt-50 variant.
2. Train ResNeSt on CIFAR-100.
3. Visualize split-attention weights.
4. Ablate different radix values.
5. Compare ResNeSt with SENet on detection tasks.

### Hard - 3
1. Design a 3D split-attention block.
2. Implement a differentiable radix search.
3. Analyze the theoretical capacity increase from split-attention.

## Solutions

### Easy - 1 Solution
```python
# In a ResNeSt-50, cardinality is typically set per stage
# Common: layer1=1, layer2=2, layer3=4, layer4=4
print("ResNeSt typically uses cardinality=4 for deep stages")
```

## Related Concepts

DL-200 ResNet, DL-201 ResNeXt, DL-204 SE-Net

## Next Concepts

DL-213 ConvNeXt

## Summary

ResNeSt introduces split-attention blocks that combine cardinality (ResNeXt) with channel attention (SE-Net) by splitting groups into radix subgroups and applying soft attention. It achieves state-of-the-art results across classification, detection, and segmentation.

## Key Takeaways

- Split-attention: cardinal groups split into radix subgroups
- Soft attention within each cardinal group selects informative radix features
- Combines ResNeXt cardinality with SE-style attention
- State-of-the-art on ImageNet, COCO, ADE20K at time of release
- Radix = 2 is typically sufficient
- Cardinality and radix are orthogonal: can tune both
- More params than ResNet but better accuracy
- Used as backbone in many vision tasks
- Attention operates on groups, not individual channels
- Bridges grouped convs and attention mechanisms
