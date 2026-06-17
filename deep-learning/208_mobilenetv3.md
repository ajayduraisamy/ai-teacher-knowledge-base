# Concept: MobileNetV3

## Concept ID

DL-208

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand MobileNetV3's improvements with NAS, Swish, and SE
- Implement MobileNetV3 blocks in PyTorch
- Analyze the hardware-aware architecture search
- Compare MobileNetV3 with earlier versions

## Prerequisites

DL-207 MobileNetV2, DL-204 SE-Net, DL-215 Architecture Search

## Definition

MobileNetV3 combines hardware-aware neural architecture search (NAS), the Swish activation function, and squeeze-and-excitation attention to produce state-of-the-art mobile models with improved accuracy and latency.

## Intuition

MobileNetV3 took the best ideas from MobileNetV2 (inverted residuals) and added two key improvements: hardware-aware NAS to find optimal block configurations, and the Swish activation (replacing ReLU6) which provides better gradient flow. SE blocks were also integrated into the architecture. The result is a model family that achieves the best accuracy for a given latency on mobile hardware.

## Why This Concept Matters

MobileNetV3 represents the state of the art in mobile-efficient architectures before transformers. It demonstrates the power of combining algorithmic innovations (Swish, SE) with automated architecture search. The model is widely used in production mobile applications.

## Mathematical Explanation

**MobileNetV3 building block** = Inverted residual + SE + Swish:
- 1x1 expand (Swish)
- 3x3 depthwise (Swish)
- SE attention
- 1x1 project (linear, no activation)
- Residual connection (if stride=1 and channels match)

**Swish activation**: 
swish(x) = x * sigmoid(x)

**H-Swish** (hard version for latency):
h-swish(x) = x * ReLU6(x+3) / 6

**NetAdapt**: Used to fine-tune the NAS-found architecture layer-by-layer for latency constraints.

**MobileNetV3-Large vs Small**: Two variants for high and low resource scenarios.

## Code Examples

### Example 1: MobileNetV3 Block

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

def hard_swish(x, inplace=False):
    return x * torch.relu6(x + 3) / 6

def hard_sigmoid(x, inplace=False):
    return torch.relu6(x + 3) / 6

class MobileNetV3Block(nn.Module):
    """MobileNetV3 inverted residual with SE and h-swish."""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, use_se, use_hs):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Determine activation
        self.act = hard_swish if use_hs else torch.relu
        
        # Expansion
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        
        # SE
        if use_se:
            reduced = max(1, hidden_dim // 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, reduced, 1),
                nn.Conv2d(reduced, hidden_dim, 1),
            )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand'):
            x = self.act(self.expand(x))
        x = self.act(self.depthwise(x))
        
        if hasattr(self, 'se'):
            se_weights = hard_sigmoid(self.se(x))
            x = x * se_weights
        
        x = self.project(x)
        
        if self.use_residual:
            x = x + identity
        return x

x = torch.randn(1, 16, 112, 112)
block = MobileNetV3Block(16, 24, kernel_size=3, stride=2, 
                         expand_ratio=4, use_se=False, use_hs=False)
out = block(x)
print(f"MobileNetV3 block: {x.shape} -> {out.shape}")
```

### Example 2: MobileNetV3-Large

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, mode='large'):
        super().__init__()
        
        if mode == 'large':
            arch = [
                # (kernel, exp, out, use_se, use_hs, stride)
                (3, 1, 16, False, False, 1),
                (3, 4, 24, False, False, 2),
                (3, 3, 24, False, False, 1),
                (5, 3, 40, True, False, 2),
                (5, 3, 40, True, False, 1),
                (5, 3, 40, True, False, 1),
                (3, 6, 80, False, True, 2),
                (3, 2.5, 80, False, True, 1),
                (3, 2.3, 80, False, True, 1),
                (3, 2.3, 80, False, True, 1),
                (3, 6, 112, True, True, 1),
                (3, 6, 112, True, True, 1),
                (5, 6, 160, True, True, 2),
                (5, 6, 160, True, True, 1),
                (5, 6, 160, True, True, 1),
            ]
            last_channels = 960
            final_channels = 1280
        else:  # small
            arch = [
                (3, 1, 16, True, False, 2),
                (3, 4.5, 24, False, False, 2),
                (3, 3.67, 24, False, False, 1),
                (5, 4, 40, True, True, 2),
                (5, 6, 40, True, True, 1),
                (5, 6, 40, True, True, 1),
                (5, 3, 48, True, True, 1),
                (5, 3, 48, True, True, 1),
                (5, 6, 96, True, True, 2),
                (5, 6, 96, True, True, 1),
                (5, 6, 96, True, True, 1),
            ]
            last_channels = 576
            final_channels = 1024
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        
        # Blocks
        in_channels = 16
        self.blocks = nn.Sequential()
        for k, exp, out, use_se, use_hs, s in arch:
            self.blocks.add_module(f'block_{len(self.blocks)}', 
                MobileNetV3Block(in_channels, out, k, s, 
                                int(exp), use_se, use_hs))
            in_channels = out
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, final_channels),
            hard_swish,
            nn.Dropout(0.2),
            nn.Linear(final_channels, num_classes),
        )
    
    def forward(self, x):
        x = hard_swish(self.stem(x))
        x = self.blocks(x)
        x = hard_swish(self.final_conv(x))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.classifier(x)

model_large = MobileNetV3(num_classes=10, mode='large')
model_small = MobileNetV3(num_classes=10, mode='small')

x = torch.randn(2, 3, 224, 224)
out_large = model_large(x)
out_small = model_small(x)

params_large = sum(p.numel() for p in model_large.parameters())
params_small = sum(p.numel() for p in model_small.parameters())

print(f"MobileNetV3-Large: {out_large.shape}, {params_large/1e6:.2f}M params")
print(f"MobileNetV3-Small: {out_small.shape}, {params_small/1e6:.2f}M params")
```

### Example 3: Activation Comparison

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Compare activations
x = torch.linspace(-5, 5, 100)

relu6 = torch.relu6(x)
swish = x * torch.sigmoid(x)
h_swish = x * torch.relu6(x + 3) / 6

print(f"Activation comparison at key points:")
test_points = [-5, -3, 0, 3, 5]
for pt in test_points:
    idx = (x - pt).abs().argmin()
    print(f"  x={pt:2d}: ReLU6={relu6[idx]:.3f}, "
          f"Swish={swish[idx]:.3f}, H-Swish={h_swish[idx]:.3f}")
```

## Common Mistakes

1. **Using wrong activation**: Not all layers use h-swish; some use ReLU for latency reasons.
2. **Forgetting SE**: MobileNetV3 heavily relies on SE for accuracy gains.
3. **Wrong SE reduction ratio**: MobileNetV3 uses hidden_dim // 4, not a fixed ratio.
4. **Ignoring the hard sigmoid for SE**: SE uses hard sigmoid (not regular sigmoid) for speed.
5. **Not applying activation after the stem**: The initial conv uses h-swish as well.

## Interview Questions

### Beginner - 5
1. What is MobileNetV3?
2. What is Swish / h-swish?
3. How does MobileNetV3 differ from V2?
4. What role does NAS play in MobileNetV3?
5. What are the two variants of MobileNetV3?

### Intermediate - 5
1. Explain the hardware-aware NAS used in MobileNetV3.
2. Why is h-swish faster than swish on mobile hardware?
3. How does SE improve MobileNetV3?
4. Compare MobileNetV3 with EfficientNet-B0.
5. What is NetAdapt and how is it used?

### Advanced - 3
1. Design a hardware-aware search space for mobile architectures.
2. Analyze the latency-accuracy Pareto frontier for MobileNetV3.
3. Implement platform-specific optimizations for MobileNetV3 inference.

## Practice Problems

### Easy - 5
1. Load MobileNetV3 from torchvision.
2. Count SE blocks in MobileNetV3-Large.
3. Identify layers using h-swish vs ReLU.
4. Replace classifier for transfer learning.
5. Compare params: V3-Large vs V3-Small.

### Medium - 5
1. Implement MobileNetV3-Small from scratch.
2. Benchmark MobileNetV3 on CPU vs GPU.
3. Visualize SE attention weights.
4. Compare accuracy across MobileNet generations.
5. Implement h-swish and compare with swish.

### Hard - 3
1. Reproduce the NetAdapt fine-tuning algorithm.
2. Design a platform-specific optimized MobileNetV3.
3. Analyze the NAS-discovered architecture patterns.

## Solutions

### Easy - 1 Solution
```python
model = models.mobilenet_v3_large(pretrained=False)
total = sum(p.numel() for p in model.parameters())
print(f"MobileNetV3-Large: {total/1e6:.2f}M params")
```

## Related Concepts

DL-207 MobileNetV2, DL-204 SE-Net, DL-215 Architecture Search, DL-206 MobileNet

## Next Concepts

DL-209 ShuffleNet, DL-210 SqueezeNet

## Summary

MobileNetV3 combines hardware-aware NAS, Swish/h-swish activation, and SE attention to produce state-of-the-art mobile-efficient models. It set new accuracy-latency records on mobile hardware and represents the culmination of the MobileNet design philosophy.

## Key Takeaways

- Hardware-aware NAS optimizes for target platform latency
- Swish/h-swish activation improves gradient flow
- SE blocks integrated throughout the architecture
- MobileNetV3-Large: ~5.4M params, 75.2% top-1 ImageNet
- MobileNetV3-Small: ~2.5M params, 67.5% top-1
- Hard versions (h-swish, h-sigmoid) for mobile efficiency
- NetAdapt fine-tunes layer configurations for latency
- Inverted residual + SE + Swish = core building block
- Represents peak of mobile CNN design before transformers
- Widely deployed in production mobile applications
