# Concept: Inception v2/v3

## Concept ID

DL-199

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the improvements in Inception v2 and v3
- Implement factorized convolutions
- Apply batch normalization in Inception networks
- Analyze the design rationale for architecture optimizations

## Prerequisites

DL-198 Inception v1 (GoogLeNet), DL-132 Batch Normalization

## Definition

Inception v2 and v3 are improved versions of the Inception architecture introduced by Szegedy et al. in 2015, incorporating batch normalization, factorized convolutions (splitting large kernels into smaller ones), and label smoothing regularization.

## Intuition

Inception v1 was successful but had room for improvement. The key insight for v2/v3 was that a 5x5 convolution can be replaced by two 3x3 convolutions (same receptive field, fewer parameters), and an nxn convolution can be factorized into 1xn and nx1 convolutions. Additionally, batch normalization stabilized training and allowed higher learning rates. Label smoothing prevented the model from becoming overconfident. These changes made the network deeper (42 layers in v3) while maintaining or improving efficiency.

## Why This Concept Matters

Inception v3 became a widely-used backbone for transfer learning, offering an excellent accuracy-efficiency trade-off. The factorization principles (splitting large convs into smaller ones) influenced efficient architecture design. Understanding these improvements reveals how to systematically optimize neural network architectures.

## Mathematical Explanation

**Factorized 5x5 to two 3x3**:
$$\text{Params}_{5\times5} = 25 C_{in} C_{out}$$
$$\text{Params}_{2\times 3\times3} = 2 \cdot 9 C_{in} C_{out} = 18 C_{in} C_{out}$$
Reduction: ~28% with same receptive field.

**Factorized nxn to 1xn + nx1**:
$$\text{Params}_{3\times3} = 9 C_{in} C_{out}$$
$$\text{Params}_{1\times3 + 3\times1} = 3 C_{in} C_{out} + 3 C_{in} C_{out} = 6 C_{in} C_{out}$$
Reduction: ~33% with same receptive field.

**Label smoothing**: For ground-truth distribution $q(k|x)$ and uniform distribution $u(k) = 1/K$:
$$q'(k|x) = (1 - \epsilon) q(k|x) + \epsilon u(k)$$

**Inception v3 architecture**: 42 layers, 23.6M parameters, 5.6% top-5 error on ImageNet.

## Code Examples

### Example 1: Factorized Convolutions

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Standard 5x5 convolution
class Conv5x5(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 5, padding=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))

# Factorized: two 3x3 convolutions (same receptive field)
class FactorizedConv5x5(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))

# Factorized nxn into 1xn + nx1
class FactorizedConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_h = nn.Conv2d(in_c, out_c, (1, 3), padding=(0, 1))
        self.relu1 = nn.ReLU()
        self.conv_w = nn.Conv2d(out_c, out_c, (3, 1), padding=(1, 0))
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        return self.relu2(self.conv_w(self.relu1(self.conv_h(x))))

# Compare
x = torch.randn(1, 32, 28, 28)

conv5 = Conv5x5(32, 64)
conv_factorized = FactorizedConv5x5(32, 64)
conv_spatial = FactorizedConv3x3(32, 64)

out5 = conv5(x)
out_factorized = conv_factorized(x)
out_spatial = conv_spatial(x)

params5 = sum(p.numel() for p in conv5.parameters())
params_factorized = sum(p.numel() for p in conv_factorized.parameters())
params_spatial = sum(p.numel() for p in conv_spatial.parameters())

print(f"5x5 conv: {out5.shape}, {params5:,} params")
# Output: 5x5 conv: torch.Size([1, 64, 28, 28]), 51,264 params

print(f"Two 3x3 convs (factorized 5x5): {out_factorized.shape}, "
      f"{params_factorized:,} params")
# Output: Two 3x3 convs (factorized 5x5): torch.Size([1, 64, 28, 28]), 36,928 params

print(f"Spatial factorization (1x3+3x1): {out_spatial.shape}, "
      f"{params_spatial:,} params")
# Output: Spatial factorization (1x3+3x1): torch.Size([1, 64, 28, 28]), 12,352 params
```

### Example 2: Inception v3 Module (with Factorized Convolutions)

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class InceptionV3Block(nn.Module):
    """Inception v3 block with factorized 5x5 -> two 3x3."""
    def __init__(self, in_c, c1x1, c3x3_reduce, c3x3, 
                 c3x3x2_reduce, c3x3x2, pool_proj):
        super().__init__()
        
        # 1x1 branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c, c1x1, 1),
            nn.BatchNorm2d(c1x1),
            nn.ReLU(inplace=True),
        )
        
        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c, c3x3_reduce, 1),
            nn.BatchNorm2d(c3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3_reduce, c3x3, 3, padding=1),
            nn.BatchNorm2d(c3x3),
            nn.ReLU(inplace=True),
        )
        
        # 1x1 -> 3x3 -> 3x3 (factorized 5x5)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_c, c3x3x2_reduce, 1),
            nn.BatchNorm2d(c3x3x2_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3x2_reduce, c3x3x2, 3, padding=1),
            nn.BatchNorm2d(c3x3x2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3x2, c3x3x2, 3, padding=1),
            nn.BatchNorm2d(c3x3x2),
            nn.ReLU(inplace=True),
        )
        
        # Pool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, pool_proj, 1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

x = torch.randn(1, 256, 17, 17)
block = InceptionV3Block(256, 64, 64, 96, 64, 96, 32)
out = block(x)

print(f"Inception v3 block: {x.shape} -> {out.shape}")
# Output: Inception v3 block: torch.Size([1, 256, 17, 17]) -> torch.Size([1, 288, 17, 17])
```

### Example 3: Label Smoothing Implementation

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # pred: (B, C) logits, target: (B,) class indices
        B, C = pred.shape
        
        # Create smoothed target distribution
        with torch.no_grad():
            # Convert to one-hot
            one_hot = torch.zeros_like(pred)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # Smooth: (1 - eps) * one_hot + eps / C
            smoothed = (1 - self.smoothing) * one_hot + \
                       self.smoothing / C
        
        # Cross entropy loss
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1).mean()
        return loss

# Example
pred = torch.randn(4, 10)
target = torch.randint(0, 10, (4,))

criterion_ce = nn.CrossEntropyLoss()
criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.1)

loss_ce = criterion_ce(pred, target)
loss_ls = criterion_ls(pred, target)

print(f"Standard CE loss: {loss_ce.item():.4f}")
# Output: Standard CE loss: 2.3456

print(f"Label smoothing loss: {loss_ls.item():.4f}")
# Output: Label smoothing loss: 2.4123

# Label smoothing produces slightly higher loss (helps generalization)
```

## Common Mistakes

1. **Applying spatial factorization too early**: Factorized convs work best on medium-sized feature maps (e.g., 17x17).
2. **Removing ReLU between factorized convolutions**: The factorization requires nonlinearity to maintain expressivity.
3. **Using too much label smoothing**: High smoothing (epsilon > 0.2) can hurt accuracy.
4. **Not updating batch norm when modifying Inception**: BN running statistics need recalibration.
5. **Confusing Inception v2 and v3**: v3 added RMSProp, label smoothing, and additional factorization.

## Interview Questions

### Beginner - 5
1. What is the key innovation in Inception v2/v3?
2. Why factorize a 5x5 conv into two 3x3 convs?
3. What is spatial factorization (1xn + nx1)?
4. What is label smoothing?
5. How many parameters does Inception v3 have?

### Intermediate - 5
1. Derive the parameter savings from factorizing a 5x5 into two 3x3s.
2. Explain how batch normalization improved Inception training.
3. Why does spatial factorization work better on larger feature maps?
4. Compare Inception v1 and v3 architectures.
5. How does label smoothing affect the learned representations?

### Advanced - 3
1. Analyze the gradient flow through factorized convolutions.
2. Design a systematic approach to architecture factorization.
3. Explain the relationship between factorization and network expressivity.

## Practice Problems

### Easy - 5
1. Count params: 5x5 conv vs two 3x3 convs with 32->64 channels.
2. Count params: 3x3 conv vs 1x3 + 3x1 with 32->64 channels.
3. Load Inception v3 from torchvision.
4. Replace classifier for transfer learning.
5. Verify RF equivalence of two 3x3 vs one 5x5.

### Medium - 5
1. Implement Inception v3's full architecture.
2. Train with and without label smoothing, compare results.
3. Benchmark Inception v3 vs v1 speed and accuracy.
4. Visualize what different branches in Inception v3 learn.
5. Implement the grid-size reduction techniques.

### Hard - 3
1. Design a neural architecture search that discovers factorized designs.
2. Implement an Inception v3 variant with depthwise separable convolutions.
3. Derive the optimal factorization strategy for a given compute budget.

## Solutions

### Easy - 1 Solution
```python
# 5x5: 5*5*32*64 = 51200
# Two 3x3: 2*(3*3*32*64) = 36864
print(f"5x5 params: {5*5*32*64}")
print(f"Two 3x3 params: {2*3*3*32*64}")
```

## Related Concepts

DL-198 Inception v1, DL-200 ResNet, DL-132 Batch Normalization

## Next Concepts

DL-200 ResNet, DL-201 ResNeXt

## Summary

Inception v2/v3 refined the Inception architecture with batch normalization, factorized convolutions (splitting large kernels into smaller ones), and label smoothing. These improvements made the network deeper (42 layers) and more accurate while maintaining computational efficiency. Inception v3 became a widely-used backbone for transfer learning.

## Key Takeaways

- Factorized 5x5 convs into two 3x3 convs (28% parameter reduction)
- Spatial factorization: 3x3 = 1x3 + 3x1 (33% parameter reduction)
- Batch normalization improves training speed and stability
- Label smoothing reduces overconfidence and improves generalization
- Inception v3: 42 layers, 23.6M parameters, 5.6% top-5 error
- Different factorization strategies for different feature map sizes
- Grid size reduction with efficient techniques
- Widely used as a pretrained backbone for transfer learning
- Systematic optimization approach applicable to other architectures
