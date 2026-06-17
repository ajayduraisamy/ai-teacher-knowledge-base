# Concept: MobileNet

## Concept ID

DL-206

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the MobileNet architecture for mobile/edge devices
- Implement depthwise separable convolutions in MobileNet
- Analyze the width multiplier and resolution multiplier
- Compare MobileNet with standard convolutions

## Prerequisites

DL-195 Depthwise Convolution, DL-193 2D Convolution, DL-182 Channel Dimension

## Definition

MobileNet is a family of efficient CNN architectures designed for mobile and embedded vision applications, using depthwise separable convolutions to dramatically reduce computation and model size while maintaining reasonable accuracy.

## Intuition

Standard convolutions are computationally expensive because they simultaneously map spatial patterns and channel patterns. MobileNet decouples these: first, each channel is processed independently with a lightweight spatial filter (depthwise conv), then a 1x1 conv mixes the channels (pointwise conv). This separation reduces computation by 8-9x with minimal accuracy loss. Additionally, MobileNet introduces width and resolution multipliers that let you trade off accuracy for speed/size, making it adaptable to various hardware constraints.

## Why This Concept Matters

MobileNet made deep learning practical on mobile devices, IoT, and edge computing. Its depthwise separable convolution design became the foundation for most efficient architectures. MobileNet v1 demonstrated that you don't need heavy computation for good accuracy.

## Mathematical Explanation

**Standard convolution cost**:
C_std = K^2 * C_in * C_out * H_out * W_out

**Depthwise separable convolution cost**:
C_dw = K^2 * C_in * H_out * W_out (depthwise)
C_pw = C_in * C_out * H_out * W_out (pointwise)
C_sep = C_dw + C_pw

**Reduction ratio**:
C_sep / C_std = 1/C_out + 1/K^2

For K=3, C_out >= 64: reduction ~ 1/9 + 1/64 = 0.127 (7.9x).

**Width multiplier alpha**: Scales all channel counts by alpha in [0, 1].
C_out' = alpha * C_out

**Resolution multiplier rho**: Scales input resolution.
Input = 224 * rho

## Code Examples

### Example 1: Depthwise Separable Layer

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Depthwise: 3x3 conv per channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, 
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

x = torch.randn(1, 32, 56, 56)
ds_conv = DepthwiseSeparableConv(32, 64, stride=1)
out = ds_conv(x)
print(f"Depthwise separable: {x.shape} -> {out.shape}")
```

### Example 2: Complete MobileNet v1

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class MobileNetV1(nn.Module):
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
            
            # Depthwise separable convs
            for c_out, stride in arch:
                c_out = int(c_out * width_mult)
                layers.append(DepthwiseSeparableConv(
                    in_channels, c_out, stride
                ))
                in_channels = c_out
            
            self.features = nn.Sequential(*layers)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_channels, num_classes)
        
        # MobileNet v1 architecture
        arch = [
            (64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
            (512, 2), (512, 1), (512, 1), (512, 1), (512, 1),
            (512, 1), (1024, 2), (1024, 1),
        ]
        make_layers(arch)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = MobileNetV1(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"MobileNet v1: {out.shape}, {total_params/1e6:.2f}M params")
```

### Example 3: Width and Resolution Multipliers

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare models with different width multipliers
configs = [(0.5, 224), (0.75, 224), (1.0, 224), (1.0, 192)]

print(f"{'Width':<8} {'Resolution':<12} {'Params':<12} {'Output':<10}")
for w, r in configs:
    model_i = MobileNetV1(num_classes=10, width_mult=w)
    p = sum(p.numel() for p in model_i.parameters())
    x_i = torch.randn(1, 3, r, r)
    out_i = model_i(x_i)
    print(f"{w:<8.2f} {r:<12} {p/1e6:<12.2f}M {str(list(out_i.shape)):<10}")
```

## Common Mistakes

1. **Forgetting ReLU6**: MobileNet uses ReLU6 (clipped at 6) for lower precision compatibility.
2. **Not using batch norm after every conv**: BN is critical for MobileNet's training stability.
3. **Applying width multiplier too aggressively**: Very small alpha loses too much accuracy.
4. **Using standard convs everywhere**: The whole point is depthwise separable — use it as much as possible.
5. **Ignoring the first standard conv**: The first layer is a standard conv (few input channels).

## Interview Questions

### Beginner - 5
1. What is MobileNet?
2. How does depthwise separable convolution work?
3. What is the width multiplier?
4. What is the resolution multiplier?
5. Why is MobileNet efficient?

### Intermediate - 5
1. Derive the computational cost reduction of depthwise separable convs.
2. Explain ReLU6 and why it's used in MobileNet.
3. Compare MobileNet v1 with standard CNN efficiency.
4. How does the width multiplier affect accuracy and speed?
5. What are the trade-offs in choosing resolution multiplier?

### Advanced - 3
1. Design a custom MobileNet variant for a specific latency budget.
2. Analyze the gradient flow in depthwise separable convolutions.
3. Compare MobileNet's design philosophy with EfficientNet.

## Practice Problems

### Easy - 5
1. Count parameters in a depthwise separable layer with 64->128.
2. Load MobileNet v1 from torchvision.
3. Compute FLOPs for standard vs separable conv.
4. Create a model with width_mult=0.5.
5. Replace classifier for 5 classes.

### Medium - 5
1. Implement MobileNet v1 from scratch.
2. Train MobileNet on CIFAR-10 with different width multipliers.
3. Benchmark inference speed on CPU vs GPU.
4. Visualize the effect of width multiplier on feature maps.
5. Implement MobileNet with different depth multipliers.

### Hard - 3
1. Design a hardware-aware MobileNet variant.
2. Implement quantization-aware training for MobileNet.
3. Analyze the accuracy-efficiency Pareto frontier for MobileNet variants.

## Solutions

### Easy - 1 Solution
```python
# Depthwise: K^2 * C_in + C_in bias = 9*64 + 64 = 640
# Pointwise: C_in * C_out + C_out = 64*128 + 128 = 8320
# Total: 8960
print(f"Depthwise separable params: {9*64 + 64 + 64*128 + 128}")
```

## Related Concepts

DL-195 Depthwise Convolution, DL-207 MobileNetV2, DL-208 MobileNetV3, DL-205 EfficientNet

## Next Concepts

DL-207 MobileNetV2, DL-208 MobileNetV3

## Summary

MobileNet v1 introduced depthwise separable convolutions for efficient mobile vision, achieving comparable accuracy to standard models with 8-9x fewer operations. The width and resolution multipliers provide flexible accuracy-efficiency trade-offs, making deep learning practical for resource-constrained environments.

## Key Takeaways

- Depthwise separable conv = Depthwise (spatial) + Pointwise (channel mixing)
- 8-9x fewer operations than standard conv for typical configurations
- ReLU6 activation for low-precision compatibility
- Width multiplier alpha scales all channels
- Resolution multiplier rho scales input size
- 13 depthwise separable layers in MobileNet v1
- First layer is standard conv (3->32 channels)
- Foundation for MobileNetV2/V3 and efficient architecture design
- Enables real-time mobile inference
- BN + ReLU6 after every conv layer
