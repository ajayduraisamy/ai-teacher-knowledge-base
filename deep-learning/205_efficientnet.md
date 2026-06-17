# Concept: EfficientNet

## Concept ID

DL-205

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand compound scaling for neural networks
- Implement EfficientNet in PyTorch
- Analyze the depth, width, and resolution scaling trade-offs
- Compare EfficientNet with traditional architectures

## Prerequisites

DL-200 ResNet, DL-204 SE-Net, DL-195 Depthwise Convolution, DL-206 MobileNet

## Definition

EfficientNet is a family of CNN architectures that achieve state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPs than previous architectures, using neural architecture search (NAS) and a compound scaling method that jointly scales depth, width, and resolution.

## Intuition

Previous approaches scaled models along a single dimension: depth (ResNet), width (WideResNet), or resolution (bigger images). The key insight of EfficientNet is that these dimensions are not independent. If you increase the image resolution, you also need a deeper network with wider layers to process the additional information. EfficientNet's compound scaling uses a coefficient phi to simultaneously scale all three dimensions with formulas d = alpha^phi, w = beta^phi, r = gamma^phi where alpha, beta, gamma are constants determined by a small grid search. This produces a family of models (B0 through B7) with smoothly increasing accuracy and cost.

## Why This Concept Matters

EfficientNet achieved state-of-the-art accuracy (84.4% top-1 on ImageNet) with 8.4x fewer parameters than GPipe and 5.6x fewer than SENet. It demonstrated that compound scaling is far more effective than single-dimension scaling. The architecture search used to design EfficientNet-B0 also showed the power of NAS for finding optimal building blocks.

## Mathematical Explanation

Compound scaling: Scale depth (d), width (w), and resolution (r) with coefficient phi:
d = alpha^phi, w = beta^phi, r = gamma^phi
alpha * beta^2 * gamma^2 approx 2

The beta^2 and gamma^2 terms account for the quadratic FLOPs scaling of width and resolution.

EfficientNet-B0 baseline (NAS-found architecture):
- Stem: Conv3x3, 32 channels
- MBConv blocks (Mobile Inverted Bottleneck) with SE attention
- Stages: repeated MBConv blocks with increasing channels
- Final: Conv1x1, Pool, FC

EfficientNet-B7 (largest variant): 66M parameters, 37B FLOPs, 84.4% top-1 accuracy.

## Code Examples

### Example 1: MBConv Block (EfficientNet Building Block)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv with SE."""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, expand_ratio=6, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                                 stride=stride, padding=kernel_size//2,
                                 groups=hidden_dim, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_dim)
        
        reduced_dim = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU()
    
    def forward(self, x):
        identity = x
        if hasattr(self, 'expand_conv'):
            x = self.swish(self.expand_bn(self.expand_conv(x)))
        x = self.swish(self.dw_bn(self.dw_conv(x)))
        se_weights = self.se(x)
        x = x * se_weights
        x = self.project_bn(self.project_conv(x))
        if self.use_residual:
            x = x + identity
        return x

x = torch.randn(1, 32, 112, 112)
mbconv = MBConvBlock(32, 32, kernel_size=3, stride=1, expand_ratio=6)
out = mbconv(x)
params = sum(p.numel() for p in mbconv.parameters())
print(f"MBConv: {x.shape} -> {out.shape}, {params:,} params")
```

### Example 2: Compound Scaling

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

def compound_scale(base_params, phi):
    """Apply compound scaling to get scaled model parameters."""
    alpha = 1.2
    beta = 1.1
    gamma = 1.15
    
    depth_scale = alpha ** phi
    width_scale = beta ** phi
    resolution_scale = gamma ** phi
    
    return {
        'depth': depth_scale,
        'width': width_scale,
        'resolution': int(224 * resolution_scale),
        'flops_scale': depth_scale * (width_scale ** 2) * (resolution_scale ** 2)
    }

print(f"{'Model':<15} {'phi':<5} {'Depth':<10} {'Width':<10} {'Resolution':<12} {'FLOPs scale':<12}")
for phi in range(8):
    scales = compound_scale(1.0, phi)
    print(f"{f'EfficientNet-B{phi}':<15} {phi:<5} "
          f"{scales['depth']:<10.2f} {scales['width']:<10.2f} "
          f"{scales['resolution']:<12} {scales['flops_scale']:<12.1f}")
```

### Example 3: Using torchvision's EfficientNet

```python
import torch
import torch.nn as nn
import torchvision.models as models

effnet = models.efficientnet_b0(pretrained=False)
print("Loaded EfficientNet-B0 from torchvision")

x = torch.randn(4, 3, 224, 224)
out = effnet(x)
total_params = sum(p.numel() for p in effnet.parameters())

# Modify for transfer learning
num_classes = 10
effnet.classifier[1] = nn.Linear(1280, num_classes)

print(f"Input: {x.shape}")
print(f"Output: {out.shape if hasattr(effnet(x), 'shape') else str([4, num_classes])}")
print(f"Total parameters: {total_params/1e6:.2f}M")
```

## Common Mistakes

1. **Scaling only one dimension**: Compound scaling is key — scaling depth alone without width/resolution is suboptimal.
2. **Not using Swish activation**: Swish (SiLU) is important for MBConv performance.
3. **Wrong SE ratio**: Using se_ratio=0.25 is standard for EfficientNet.
4. **Ignoring the resolution scaling**: Larger images need appropriately scaled networks.
5. **Using expand_ratio=1 everywhere**: Different stages use different expansion ratios.

## Interview Questions

### Beginner - 5
1. What is compound scaling?
2. How does EfficientNet differ from ResNet?
3. What is MBConv?
4. How many EfficientNet variants exist?
5. What accuracy does EfficientNet-B7 achieve?

### Intermediate - 5
1. Derive the compound scaling formula.
2. Explain how NAS designed the EfficientNet-B0 baseline.
3. Compare EfficientNet with ResNet parameter efficiency.
4. What is the role of SE blocks in EfficientNet?
5. How does EfficientNet achieve such high efficiency?

### Advanced - 3
1. Design a compound scaling strategy for a custom compute budget.
2. Analyze the trade-offs between scaling depth, width, and resolution.
3. Explain the theoretical justification for compound scaling.

## Practice Problems

### Easy - 5
1. Load EfficientNet-B3 from torchvision.
2. Count MBConv blocks in EfficientNet-B0.
3. Compute scaled resolution for B5.
4. Replace classifier for 5 classes.
5. Compare params: B0 vs B4.

### Medium - 5
1. Implement compound scaling for a custom architecture.
2. Train EfficientNet-B0 on CIFAR-100.
3. Compare B0 through B4 accuracy vs FLOPs.
4. Visualize SE attention weights in MBConv.
5. Ablate different scaling dimensions.

### Hard - 3
1. Reproduce the compound scaling grid search.
2. Design a NAS that jointly optimizes architecture and scaling.
3. Derive the optimal alpha, beta, gamma for a given task.

## Solutions

### Easy - 1 Solution
```python
model = models.efficientnet_b3(pretrained=False)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
```

## Related Concepts

DL-206 MobileNet, DL-204 SE-Net, DL-200 ResNet, DL-215 Architecture Search

## Next Concepts

DL-206 MobileNet, DL-207 MobileNetV2

## Summary

EfficientNet uses neural architecture search and compound scaling to achieve state-of-the-art accuracy with exceptional efficiency. The compound scaling method jointly scales depth, width, and resolution, producing a family of models from B0 to B7 that offer smooth accuracy-efficiency trade-offs.

## Key Takeaways

- Compound scaling: depth, width, resolution scaled jointly
- NAS-designed MBConv blocks with SE attention
- EfficientNet-B7: 84.4% top-1, 66M params (vs 145M ResNet-152)
- Swish activation, SE attention, depthwise convs
- Compound scaling coefficient phi controls all dimensions
- B0 baseline: found via neural architecture search
- B0-B7 family offers smooth accuracy-compute trade-offs
- Order of magnitude more efficient than previous architectures
- Established compound scaling as a key design principle
- Widely used as a backbone for vision tasks
