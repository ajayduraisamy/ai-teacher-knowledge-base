# Concept: ConvNeXt

## Concept ID

DL-213

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the ConvNeXt modernization of CNNs
- Implement ConvNeXt blocks in PyTorch
- Analyze how modern design principles improve CNNs
- Compare ConvNeXt with Vision Transformers

## Prerequisites

DL-200 ResNet, DL-204 SE-Net, DL-195 Depthwise Convolution, DL-132 Batch Normalization

## Definition

ConvNeXt is a pure CNN architecture that modernizes standard ConvNets (like ResNet) with design choices inspired by Vision Transformers (ViT), achieving competitive performance with transformers while maintaining CNN simplicity and efficiency.

## Intuition

Vision Transformers (ViT, Swin) achieved state-of-the-art results by using self-attention instead of convolutions. ConvNeXt asked: what if we systematically update a standard ResNet with modern design choices borrowed from transformers? The result is a modernized CNN that matches or exceeds transformer performance. Key changes include: using patchify stem (like ViT), larger kernel sizes (7x7), GELU activation, LayerNorm instead of BatchNorm, and inverted bottleneck designs.

## Why This Concept Matters

ConvNeXt demonstrated that CNNs can still compete with transformers when properly modernized. It provides a bridge between the CNN and transformer worlds, showing that the core ideas from transformers (large receptive fields, better normalization, modern activations) can be retrofitted into CNNs.

## Mathematical Explanation

**Modernization steps from ResNet to ConvNeXt**:

1. **Training recipe**: AdamW, data augmentation, stochastic depth, cosine LR
2. **Patchify stem**: 4x4 conv (stride 4) instead of 7x7 conv + maxpool
3. **Stage ratio**: (3, 3, 9, 3) instead of (3, 4, 6, 3)
4. **Inverted bottleneck**: hidden_dim = 4 * in_dim (like MobileNetV2)
5. **Large kernel**: 7x7 depthwise conv (moved to early position)
6. **LayerNorm**: Replace BatchNorm with LayerNorm
7. **GELU**: Replace ReLU with GELU activation
8. **Fewer activations**: Only one GELU per block (not two)

## Code Examples

### Example 1: ConvNeXt Block

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with inverted bottleneck and large kernel."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        
        # Depthwise conv (large kernel, early position)
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # LayerNorm (after depthwise, before MLP)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted bottleneck MLP
        hidden_dim = dim * 4
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # Layer scaling (learnable per-channel scaling)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init > 0 else None
    
    def forward(self, x):
        identity = x
        
        # Depthwise conv (spatial processing)
        x = self.dwconv(x)
        
        # Permute to (B, H*W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # LayerNorm + MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Skip connection
        x = identity + x
        return x

x = torch.randn(1, 96, 56, 56)
block = ConvNeXtBlock(96)
out = block(x)
print(f"ConvNeXt block: {x.shape} -> {out.shape}")
```

### Example 2: Complete ConvNeXt

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768]):
        super().__init__()
        
        # Patchify stem (4x4 conv, stride 4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        input_dim = dims[0]
        
        for i in range(len(depths)):
            stage = nn.Sequential()
            
            # Downsampling layer (except first stage)
            if i > 0:
                stage.add_module('downsample', nn.Sequential(
                    nn.LayerNorm(input_dim, eps=1e-6),
                    nn.Conv2d(input_dim, dims[i], 2, stride=2),
                ))
            
            # ConvNeXt blocks
            for j in range(depths[i]):
                stage.add_module(f'block_{j}', ConvNeXtBlock(dims[i]))
            
            self.stages.append(stage)
            input_dim = dims[i]
        
        # Classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.norm(x)
        return self.head(x)

# Create ConvNeXt-T (tiny)
model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"ConvNeXt-T: {out.shape}, {total_params/1e6:.2f}M params")
```

### Example 3: Comparison with ResNet Block

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class ResNetBlock(nn.Module):
    """Standard ResNet bottleneck block for comparison."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim*4, 1)
        self.bn1 = nn.BatchNorm2d(dim*4)
        self.conv2 = nn.Conv2d(dim*4, dim*4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim*4)
        self.conv3 = nn.Conv2d(dim*4, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + identity)

x = torch.randn(1, 96, 28, 28)

resnet_block = ResNetBlock(96)
convnext_block = ConvNeXtBlock(96)

r_out = resnet_block(x)
c_out = convnext_block(x)

r_params = sum(p.numel() for p in resnet_block.parameters())
c_params = sum(p.numel() for p in convnext_block.parameters())

print(f"ResNet block: {r_out.shape}, {r_params:,} params")
print(f"ConvNeXt block: {c_out.shape}, {c_params:,} params")
```

## Common Mistakes

1. **Forgetting the permute for LayerNorm**: ConvNeXt uses LayerNorm in (B, H, W, C) format — needs permute.
2. **Using BatchNorm instead of LayerNorm**: LayerNorm is a key modernization.
3. **Wrong kernel position**: Depthwise conv comes first, then MLP (inverted bottleneck).
4. **Not using layer scaling**: Learnable gamma per channel is important.
5. **Using small kernel (3x3)**: ConvNeXt uses 7x7 depthwise convs.

## Interview Questions

### Beginner - 5
1. What is ConvNeXt?
2. How does ConvNeXt differ from ResNet?
3. What is a patchify stem?
4. What activation does ConvNeXt use?
5. What kernel size does ConvNeXt use?

### Intermediate - 5
1. List the key modernizations from ResNet to ConvNeXt.
2. Why does ConvNeXt use LayerNorm instead of BatchNorm?
3. Explain the inverted bottleneck design in ConvNeXt.
4. How does ConvNeXt compare to Vision Transformers?
5. What is the role of depthwise convs in ConvNeXt?

### Advanced - 3
1. Analyze why moving the depthwise conv to the front helps.
2. Design a ConvNeXt variant for video.
3. Compare the effective receptive field of ConvNeXt vs ResNet.

## Practice Problems

### Easy - 5
1. Load ConvNeXt from torchvision.
2. Count depthwise convs in ConvNeXt-T.
3. Replace classifier for transfer learning.
4. Compute FLOPs for a ConvNeXt block.
5. Compare stem designs: ResNet vs ConvNeXt.

### Medium - 5
1. Implement ConvNeXt-S from scratch.
2. Train ConvNeXt on ImageNet-1K.
3. Compare ConvNeXt vs Swin Transformer accuracy.
4. Visualize ConvNeXt feature maps.
5. Ablate different modernization steps.

### Hard - 3
1. Design a ConvNeXt variant with variable kernel sizes.
2. Implement ConvNeXt with stochastic depth.
3. Analyze the depthwise conv kernel patterns learned by ConvNeXt.

## Solutions

### Easy - 1 Solution
```python
model = models.convnext_tiny(pretrained=False)
total = sum(p.numel() for p in model.parameters())
print(f"ConvNeXt-T: {total/1e6:.2f}M params")
```

## Related Concepts

DL-200 ResNet, DL-195 Depthwise Convolution, DL-204 SE-Net, Vision Transformers

## Next Concepts

DL-214 RegNet, DL-215 Architecture Search

## Summary

ConvNeXt modernizes standard CNNs with design choices inspired by Vision Transformers, including patchify stem, large kernel depthwise convs, LayerNorm, GELU activation, and inverted bottlenecks. It achieves transformer-competitive performance while maintaining CNN efficiency and simplicity.

## Key Takeaways

- Systematic modernization of ResNet with ViT-inspired design
- Patchify stem (4x4, stride 4) replaces 7x7+maxpool
- Large kernel (7x7) depthwise convs moved to early position
- LayerNorm replaces BatchNorm throughout
- GELU activation replaces ReLU
- Inverted bottleneck (4x expansion)
- Stage ratio (3, 3, 9, 3) for better accuracy
- Layer scaling (learnable gamma per channel)
- Matches Swin Transformer accuracy with CNN efficiency
- Demonstrates CNNs can still compete with transformers
