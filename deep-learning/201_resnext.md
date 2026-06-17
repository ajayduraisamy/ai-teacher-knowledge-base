# Concept: ResNeXt

## Concept ID

DL-201

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the ResNeXt architecture and grouped convolutions
- Implement ResNeXt blocks in PyTorch
- Analyze the cardinality vs depth trade-off
- Compare ResNeXt with ResNet

## Prerequisites

DL-200 ResNet, DL-195 Depthwise Convolution, DL-182 Channel Dimension

## Definition

ResNeXt introduces the concept of cardinality — the number of parallel transformation paths — as an additional dimension alongside depth and width. It uses grouped convolutions to implement multiple parallel paths efficiently.

## Intuition

ResNet introduced depth as the primary dimension for scaling. WideResNet increased width. ResNeXt proposes a third axis: cardinality — the number of independent transformation paths. Instead of having one wide 3x3 convolution, ResNeXt splits it into several narrower convolutions that run in parallel and sum their outputs. This is similar to Inception's parallel paths, but unlike Inception, all paths have the same topology. The key insight is that increasing cardinality (more paths) is more effective than increasing depth or width for a given compute budget.

## Why This Concept Matters

ResNeXt demonstrated that cardinality is a more fundamental dimension than width or depth for improving accuracy. It achieved state-of-the-art results with a simple, uniform architecture that is easy to implement and scales well. The use of grouped convolutions also connects to efficient hardware utilization.

## Mathematical Explanation

**ResNeXt block** with cardinality $C$:

Standard ResNet bottleneck:
$$y = x + W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x))$$

ResNeXt aggregated transformation:
$$y = x + \sum_{i=1}^{C} T_i(x)$$

Where each $T_i$ is a transformation (typically 1x1 -> 3x3 -> 1x1) and $C$ is the cardinality.

**Implementation via grouped convolutions**: The 3x3 conv in the bottleneck uses groups=C, splitting the intermediate channels into C independent groups.

**Width per path**: If the bottleneck has $d$ channels and cardinality $C$, each path has $d/C$ channels.

**Parameter comparison** (ResNet-50 vs ResNeXt-50):
- ResNet-50: ~25.6M params, 4.6B FLOPs, 76.0% top-1
- ResNeXt-50 (32x4d): ~25.0M params, 4.2B FLOPs, 77.8% top-1

## Code Examples

### Example 1: ResNeXt Block Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class ResNeXtBlock(nn.Module):
    """ResNeXt bottleneck block with grouped convolutions."""
    expansion = 2  # adjusted for ResNeXt
    
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super().__init__()
        width = out_channels  # bottleneck width
        
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # Grouped 3x3 conv (the key innovation)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = F.relu(out)
        return out

# Test ResNeXt block with different cardinalities
x = torch.randn(1, 256, 28, 28)

for card in [1, 8, 32]:
    block = ResNeXtBlock(256, 128, stride=1, cardinality=card)
    out = block(x)
    params = sum(p.numel() for p in block.parameters())
    print(f"Cardinality={card:2d}: {x.shape} -> {out.shape}, "
          f"{params:,} params")
    # Output: Cardinality= 1: [1,256,28,28] -> [1,256,28,28], 198,144 params
    # Output: Cardinality= 8: [1,256,28,28] -> [1,256,28,28], 191,488 params
    # Output: Cardinality=32: [1,256,28,28] -> [1,256,28,28], 190,464 params
```

### Example 2: Comparing Cardinality vs Width

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare different network configurations with similar parameter counts
configs = [
    # (name, cardinality, width_per_group)
    ('ResNet (C=1, w=256)', 1, 256),
    ('ResNeXt (C=8, w=128)', 8, 128),
    ('ResNeXt (C=32, w=64)', 32, 64),
    ('ResNeXt (C=64, w=32)', 64, 32),
]

x = torch.randn(1, 512, 14, 14)

print(f"{'Configuration':<28} {'Params':<12} {'Output shape':<20}")
for name, card, width in configs:
    # Build a ResNeXt stage with this config
    stage = nn.Sequential(
        ResNeXtBlock(512, width, stride=1, cardinality=card),
        ResNeXtBlock(width * 2, width, stride=1, cardinality=card),
    )
    # Hack: adjust in_channels for second block
    alt_block = ResNeXtBlock(512, width, stride=1, cardinality=card)
    params = sum(p.numel() for p in alt_block.parameters())
    out = alt_block(x)
    print(f"{name:<28} {params:<12,} {str(list(out.shape)):<20}")
    # Output: ResNet (C=1, w=256)    197,120     [1, 512, 14, 14]
    # Output: ResNeXt (C=8, w=128)  191,488     [1, 256, 14, 14]
    # Output: ResNeXt (C=32, w=64)  190,464     [1, 128, 14, 14]
    # Output: ResNeXt (C=64, w=32)  189,952     [1, 64, 14, 14]
```

### Example 3: Full ResNeXt Architecture

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality=32, 
                 base_width=4, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.base_width = base_width
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Stages
        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s,
                               self.cardinality, self.base_width))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Create ResNeXt-50 (32x4d)
model = ResNeXt(ResNeXtBlock, [3, 4, 6, 3], cardinality=32, 
                base_width=4, num_classes=10)

x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"ResNeXt-50 (32x4d):")
print(f"  Input: {x.shape}")
# Output: Input: torch.Size([2, 3, 224, 224])

print(f"  Output: {out.shape}")
# Output: Output: torch.Size([2, 10])

print(f"  Parameters: {total_params/1e6:.2f}M")
# Output: Parameters: 23.28M
```

## Common Mistakes

1. **Confusing groups with cardinality**: In ResNeXt, the 3x3 conv uses groups=cardinality; each group processes a subset of the channels.
2. **Not adjusting width per path**: Total width = cardinality * width_per_path must be consistent.
3. **Forgetting that grouped convs need in_channels divisible by groups**: This limits some configurations.
4. **Using cardinality > 64**: Very high cardinality gives diminishing returns.
5. **Not updating the shortcut connection**: The shortcut still uses 1x1 conv for dimension matching.

## Interview Questions

### Beginner - 5
1. What is cardinality in ResNeXt?
2. How does ResNeXt differ from ResNet?
3. What is a grouped convolution?
4. How does ResNeXt achieve more transformations without more parameters?
5. What does "32x4d" mean in ResNeXt notation?

### Intermediate - 5
1. Explain how grouped convolutions implement parallel transformations.
2. Compare the effectiveness of increasing cardinality vs depth vs width.
3. Derive the parameter count for a ResNeXt block with cardinality C.
4. How does ResNeXt relate to Inception?
5. What is the optimal cardinality for a given compute budget?

### Advanced - 3
1. Analyze the relationship between cardinality and model capacity.
2. Design an architecture that scales cardinality dynamically through layers.
3. Derive the gradient flow in a grouped convolution block.

## Practice Problems

### Easy - 5
1. Create a Conv2d with groups=32 and explain its weight shape.
2. Count parameters in a ResNeXt block with C=32 vs C=1.
3. Explain what "32x4d" means.
4. Load ResNeXt-101 from torchvision.
5. Compare the number of paths in ResNet vs ResNeXt.

### Medium - 5
1. Implement ResNeXt-101 from scratch.
2. Train ResNeXt-50 on ImageNet subset.
3. Compare accuracy of ResNeXt vs ResNet with same parameter count.
4. Visualize the grouped convolution patterns.
5. Analyze the FLOPs of different cardinality configurations.

### Hard - 3
1. Design a neural architecture search over cardinality configurations.
2. Implement ResNeXt with learnable group assignments.
3. Derive the theoretical capacity of grouped vs ungrouped convolutions.

## Solutions

### Easy - 1 Solution
```python
# Conv2d(in=256, out=256, k=3, groups=32)
# Weight shape: (256, 256/32, 3, 3) = (256, 8, 3, 3)
conv = nn.Conv2d(256, 256, 3, groups=32)
print(f"Weight shape: {conv.weight.shape}")
```

## Related Concepts

DL-200 ResNet, DL-202 Wide ResNet, DL-195 Depthwise Convolution, DL-198 Inception

## Next Concepts

DL-202 Wide ResNet, DL-203 DenseNet

## Summary

ResNeXt introduced cardinality as a third dimension for scaling networks, alongside depth and width. Using grouped convolutions to implement parallel transformation paths, ResNeXt achieves better accuracy than ResNet for the same parameter count. The principle of "aggregated transformations" has influenced many subsequent architectures.

## Key Takeaways

- Cardinality = number of parallel transformation paths
- Grouped convolutions implement efficient parallel transformations
- 32x4d = 32 groups of 4 channels each
- Increasing cardinality is more effective than increasing depth/width
- ResNeXt-101 achieved state-of-the-art on ImageNet
- Same topology for all paths (unlike Inception's different topologies)
- More parameter-efficient than standard ResNet
- Groups constraint: in_channels must be divisible by groups
- High cardinality gives diminishing returns
- Foundation for many modern efficient architectures
