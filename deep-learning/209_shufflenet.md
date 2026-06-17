# Concept: ShuffleNet

## Concept ID

DL-209

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand channel shuffle operation for group convolutions
- Implement ShuffleNet units in PyTorch
- Analyze how channel shuffle improves information flow
- Compare ShuffleNet with other efficient architectures

## Prerequisites

DL-195 Depthwise Convolution, DL-182 Channel Dimension, DL-201 ResNeXt

## Definition

ShuffleNet is an efficient CNN architecture that uses pointwise group convolutions and channel shuffle operations to reduce computation while maintaining accuracy, designed specifically for mobile devices with limited compute.

## Intuition

Group convolutions are efficient but have a problem: each group processes only a subset of channels, so information doesn't flow between groups. This limits representational power. ShuffleNet's key insight is the channel shuffle operation: after each group convolution, the output channels are permuted so that in the next group convolution, each group receives channels from all previous groups. This restores cross-group communication at almost zero cost (just a tensor reshape/permute).

## Why This Concept Matters

ShuffleNet demonstrated that group convolutions + channel shuffling can match or exceed the accuracy of depthwise separable convolutions at lower computational cost. Its channel shuffle operation is a simple but effective technique for improving group convolution performance.

## Mathematical Explanation

**ShuffleNet unit**:
- Pointwise group conv (1x1, groups=g)
- Channel shuffle
- Depthwise 3x3 conv
- Pointwise group conv (1x1, groups=g)
- Residual connection (if stride=1)

**Channel shuffle**: Given a tensor of shape (B, C, H, W), reshape to (B, g, C/g, H, W), transpose to (B, C/g, g, H, W), reshape back to (B, C, H, W). This interleaves channels across groups.

**Computational cost** for a ShuffleNet unit with groups=g:
C = (C_in * C_out / g) * 2 (two pointwise group convs) + K^2 * C_out (depthwise)

## Code Examples

### Example 1: Channel Shuffle

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

def channel_shuffle(x, groups):
    """Channel shuffle operation."""
    B, C, H, W = x.shape
    assert C % groups == 0
    
    # Reshape -> transpose -> reshape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)

# Demonstrate channel shuffle
x = torch.arange(24).reshape(1, 8, 1, 3).float()
print("Original (groups=2):")
print(x.squeeze())
# Output: [0, 1, 2, 3, 4, 5, 6, 7]

shuffled = channel_shuffle(x, groups=2)
print("\nShuffled (groups=2):")
print(shuffled.squeeze())
# Output: [0, 4, 1, 5, 2, 6, 3, 7]
# Group 0 channels: 0,1,2,3 | Group 1 channels: 4,5,6,7
# After shuffle: each group gets channels from both groups

# Verify: original groups: [0,1,2,3], [4,5,6,7]
# After shuffle: [0,4,1,5], [2,6,3,7] — mixed!
```

### Example 2: ShuffleNet Unit

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class ShuffleNetUnit(nn.Module):
    """ShuffleNet basic unit."""
    def __init__(self, in_channels, out_channels, stride, groups=3):
        super().__init__()
        self.stride = stride
        hidden_channels = out_channels // 4  # bottleneck
        
        # Pointwise group conv 1
        self.gconv1 = nn.Conv2d(in_channels, hidden_channels, 1,
                               groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Depthwise conv
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3,
                               stride=stride, padding=1,
                               groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Pointwise group conv 2
        self.gconv2 = nn.Conv2d(hidden_channels, out_channels, 1,
                               groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.groups = groups
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.gconv1(x)))
        
        # Channel shuffle
        out = channel_shuffle(out, self.groups)
        
        out = F.relu(self.bn2(self.dwconv(out)))
        out = self.bn3(self.gconv2(out))
        
        if hasattr(self, 'shortcut'):
            identity = self.shortcut(identity)
        
        out = F.relu(out + identity)
        return out

x = torch.randn(1, 24, 28, 28)
unit = ShuffleNetUnit(24, 120, stride=2, groups=3)
out = unit(x)
print(f"ShuffleNet unit: {x.shape} -> {out.shape}")
```

### Example 3: Complete ShuffleNet

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=3):
        super().__init__()
        
        # Stage configs: (out_channels, num_blocks, stride)
        # Different groups affect channel counts
        if groups == 1:
            stage_outs = [144, 288, 576]
        elif groups == 2:
            stage_outs = [200, 400, 800]
        elif groups == 3:
            stage_outs = [240, 480, 960]
        else:
            stage_outs = [272, 544, 1088]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        # Stages
        self.stage2 = self._make_stage(24, stage_outs[0], 4, groups, stride=2)
        self.stage3 = self._make_stage(stage_outs[0], stage_outs[1], 8, groups, stride=2)
        self.stage4 = self._make_stage(stage_outs[1], stage_outs[2], 4, groups, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_outs[2], num_classes)
    
    def _make_stage(self, in_c, out_c, num_blocks, groups, stride):
        layers = []
        layers.append(ShuffleNetUnit(in_c, out_c, stride, groups))
        for _ in range(1, num_blocks):
            layers.append(ShuffleNetUnit(out_c, out_c, 1, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

x = torch.randn(2, 3, 224, 224)
for g in [1, 2, 3]:
    model = ShuffleNet(num_classes=10, groups=g)
    out = model(x)
    p = sum(p.numel() for p in model.parameters())
    print(f"ShuffleNet (g={g}): {out.shape}, {p/1e6:.2f}M params")
```

## Common Mistakes

1. **Forgetting the channel shuffle**: Without shuffle, group convolutions don't communicate.
2. **Wrong groups parameter**: The groups parameter affects both convs and must be consistent.
3. **Not using pointwise group convs**: Both pointwise convs should use groups.
4. **Using depthwise conv without enough channels**: Depthwise conv needs sufficient channels to be expressive.
5. **Confusing ShuffleNet with ShuffleNet v2**: v2 uses different design principles (efficiency guidelines).

## Interview Questions

### Beginner - 5
1. What is channel shuffle?
2. Why is channel shuffle needed in group convolutions?
3. How many groups does ShuffleNet typically use?
4. What is the main building block of ShuffleNet?
5. How does ShuffleNet compare to MobileNet?

### Intermediate - 5
1. Derive the computational cost of a ShuffleNet unit.
2. Explain how channel shuffle enables group conv communication.
3. Compare ShuffleNet groups: more groups = better?
4. How does ShuffleNet's design differ from ResNeXt?
5. What are the advantages of pointwise group convs?

### Advanced - 3
1. Design an optimal group assignment strategy for ShuffleNet.
2. Analyze the gradient flow through shuffled channels.
3. Implement a learnable permutation for channel grouping.

## Practice Problems

### Easy - 5
1. Implement channel shuffle with groups=4.
2. Count params in a ShuffleNet unit with g=3.
3. Load ShuffleNet from torchvision.
4. Compare channel order before/after shuffle.
5. Count groups parameter for different ShuffleNet variants.

### Medium - 5
1. Implement ShuffleNet v1 from scratch.
2. Train ShuffleNet on CIFAR-10.
3. Compare accuracy with different group numbers.
4. Visualize channel grouping patterns.
5. Benchmark ShuffleNet vs MobileNet speed.

### Hard - 3
1. Implement ShuffleNet v2's efficiency guidelines.
2. Design a learnable channel grouping mechanism.
3. Analyze the information bottleneck in grouped convolutions.

## Solutions

### Easy - 1 Solution
```python
x = torch.randn(2, 16, 8, 8)
shuffled = channel_shuffle(x, groups=4)
print(f"Shuffled shape: {shuffled.shape}")
```

## Related Concepts

DL-201 ResNeXt, DL-195 Depthwise Convolution, DL-206 MobileNet, DL-182 Channel Dimension

## Next Concepts

DL-210 SqueezeNet

## Summary

ShuffleNet uses pointwise group convolutions combined with channel shuffle to build efficient networks for mobile devices. The channel shuffle operation restores cross-group information flow at near-zero cost, enabling competitive accuracy with fewer FLOPs than depthwise separable approaches.

## Key Takeaways

- Channel shuffle enables communication between groups in group convs
- Reshape -> Transpose -> Reshape: O(1) operation
- Pointwise group convs reduce computation by factor of groups
- Depthwise conv in the middle of each unit
- Groups = 1, 2, 3, 4 or 8 depending on variant
- Comparable accuracy to MobileNet with lower latency
- Channel shuffle is a general technique for any grouped architecture
- ShuffleNet v2 later improved on this with efficiency guidelines
- Groups parameter controls the accuracy-efficiency trade-off
- Skip connections used when stride=1 and channels match
