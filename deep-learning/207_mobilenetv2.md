# Concept: MobileNetV2

## Concept ID

DL-207

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand MobileNetV2's inverted residual with linear bottleneck
- Implement MobileNetV2 in PyTorch
- Analyze the differences from MobileNet v1
- Compare MobileNetV2 with prior efficient architectures

## Prerequisites

DL-206 MobileNet, DL-195 Depthwise Convolution, DL-200 ResNet

## Definition

MobileNetV2 introduces the inverted residual structure with linear bottlenecks, where the input/output of each block are low-dimensional compressed representations (bottleneck) while the internal expansion operates at higher dimensions, with depthwise convolutions in the expanded space.

## Intuition

MobileNet v1 processed each channel independently then mixed them, but all at the same dimension. MobileNetV2 inverts the residual design: instead of wide -> narrow -> wide (like ResNet), it does narrow -> wide -> narrow (inverted). The narrow parts are the "bottlenecks" that compress information, and the wide part is where the depthwise conv operates. This is better because depthwise convolutions work best on high-dimensional tensors. The linear bottleneck (no ReLU after the second 1x1) preserves information that would otherwise be lost by ReLU's zeroing of negative values.

## Why This Concept Matters

MobileNetV2 significantly improved accuracy over v1 (by about 3% on ImageNet) with similar computational cost. Its inverted residual design has been adopted by many subsequent architectures (including EfficientNet). The linear bottleneck concept provided insights about information preservation in efficient networks.

## Mathematical Explanation

**Inverted residual block**:
- Input: C_in channels (low-dim bottleneck)
- Expand: 1x1 conv to C_in * t channels (t = 6, expansion factor)
- Depthwise: 3x3 conv on expanded channels
- Project: 1x1 conv back to C_out channels (linear — no ReLU after)

**Parameter count**: For expansion factor t:
Params = C_in * (t*C_in) * 1 + t*C_in * K^2 + t*C_in * C_out
       = t*C_in^2 + K^2*t*C_in + t*C_in*C_out

**Memory efficiency**: Memory footprint is dominated by the intermediate expanded tensor.

## Code Examples

### Example 1: Inverted Residual Block

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class InvertedResidualBlock(nn.Module):
    """MobileNetV2 inverted residual with linear bottleneck."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion (1x1) — only if expand_ratio > 1
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(),
            )
        
        # Depthwise (3x3)
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
        )
        
        # Projection (1x1, linear — no activation)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand'):
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + identity
        return x

x = torch.randn(1, 24, 56, 56)
block = InvertedResidualBlock(24, 32, stride=2, expand_ratio=6)
out = block(x)
print(f"Inverted residual: {x.shape} -> {out.shape}")
```

### Example 2: Complete MobileNetV2

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        def make_layers(arch):
            layers = []
            in_channels = int(32 * width_mult)
            
            # First standard conv
            layers.append(nn.Sequential(
                nn.Conv2d(3, in_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(),
            ))
            
            # Inverted residual blocks
            for t, c, n, s in arch:
                c = int(c * width_mult)
                for i in range(n):
                    stride = s if i == 0 else 1
                    layers.append(InvertedResidualBlock(
                        in_channels, c, stride, expand_ratio=t
                    ))
                    in_channels = c
            
            self.features = nn.Sequential(*layers)
            
            # Last conv (1x1)
            last_channels = int(1280 * width_mult)
            self.features.add_module('last_conv', nn.Sequential(
                nn.Conv2d(in_channels, last_channels, 1, bias=False),
                nn.BatchNorm2d(last_channels),
                nn.ReLU6(),
            ))
            
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(last_channels, num_classes),
            )
        
        # MobileNetV2 architecture: (t, c, n, s)
        arch = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        make_layers(arch)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = MobileNetV2(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"MobileNetV2: {out.shape}, {total_params/1e6:.2f}M params")
```

### Example 3: Comparison with MobileNet v1

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare block designs
class MobileNetV1Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1,
                           groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        return self.relu(self.bn2(self.pw(self.relu(self.bn1(self.dw(x))))))

x = torch.randn(1, 32, 28, 28)
v1_block = MobileNetV1Block(32, 64, stride=2)
v2_block = InvertedResidualBlock(32, 64, stride=2, expand_ratio=6)

out_v1 = v1_block(x)
out_v2 = v2_block(x)

params_v1 = sum(p.numel() for p in v1_block.parameters())
params_v2 = sum(p.numel() for p in v2_block.parameters())

print(f"V1 block: {out_v1.shape}, {params_v1:,} params")
print(f"V2 block: {out_v2.shape}, {params_v2:,} params")
print(f"V2 improvement: more params but ~3% better ImageNet accuracy")
```

## Common Mistakes

1. **Adding ReLU after projection**: The linear bottleneck (no activation after 1x1 project) is critical for preserving information.
2. **Using residual without checking dimensions**: Skip connections only possible when stride=1 and channels match.
3. **Wrong expansion factor**: MobileNetV2 uses t=6 for most blocks (not t=1).
4. **Not using ReLU6**: ReLU6 is important for quantization-friendliness.
5. **Ignoring the expansion when t=1**: The first block (t=1) has no expansion conv.

## Interview Questions

### Beginner - 5
1. What is an inverted residual?
2. What is a linear bottleneck?
3. How does MobileNetV2 differ from MobileNet v1?
4. What expansion factor is used in MobileNetV2?
5. Why use linear bottleneck?

### Intermediate - 5
1. Derive the parameter count for an inverted residual block.
2. Explain why the inverted design is better than ResNet's bottleneck.
3. Why does removing ReLU from the projection improve accuracy?
4. Compare MobileNetV2 vs v1 in terms of accuracy and efficiency.
5. How does the expansion factor affect model performance?

### Advanced - 3
1. Analyze the information loss from ReLU in bottlenecks.
2. Design a variant with learnable expansion factors.
3. Explain the gradient flow advantages of the inverted residual.

## Practice Problems

### Easy - 5
1. Count params in an inverted residual with 32->64, t=6.
2. Load MobileNetV2 from torchvision.
3. Replace classifier for 100 classes.
4. Compute expand ratio for 32->192 channels.
5. Identify blocks with stride 2.

### Medium - 5
1. Implement MobileNetV2 from scratch.
2. Train MobileNetV2 on CIFAR-100.
3. Compare accuracy vs FLOPs for v1 vs v2.
4. Visualize the linear bottleneck effect.
5. Implement MobileNetV2 with different width multipliers.

### Hard - 3
1. Design a neural architecture search that optimizes expansion factors.
2. Implement MobileNetV2 with quantized inference.
3. Analyze the feature space geometry of linear bottlenecks.

## Solutions

### Easy - 1 Solution
```python
in_c, out_c, t, k = 32, 64, 6, 3
hidden = in_c * t
expand_params = in_c * hidden * 1  # 1x1
dw_params = hidden * k * k  # depthwise
project_params = hidden * out_c * 1  # 1x1
total = expand_params + dw_params + project_params
print(f"Params: {total}")
```

## Related Concepts

DL-206 MobileNet, DL-208 MobileNetV3, DL-195 Depthwise Convolution, DL-200 ResNet

## Next Concepts

DL-208 MobileNetV3

## Summary

MobileNetV2 improves upon v1 with inverted residual blocks and linear bottlenecks. The narrow-input-wide-output design allows depthwise convolutions to operate efficiently in high-dimensional space while keeping the information flow compressed. This achieves significantly better accuracy than v1 at similar computational cost.

## Key Takeaways

- Inverted residual: narrow -> wide (expand) -> narrow (project)
- Linear bottleneck: no ReLU after projection preserves information
- Expansion factor t=6 for most blocks
- Skip connections when stride=1 and channels match
- ~3% better ImageNet accuracy than MobileNet v1
- Adopted as backbone in many downstream tasks
- ReLU6 for quantization compatibility
- Foundation for MobileNetV3 and EfficientNet
- Parameter count varies with expansion and width multiplier
- Memory-efficient: large tensor only during intermediate processing
