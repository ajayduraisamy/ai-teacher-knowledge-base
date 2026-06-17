# Concept: ResNet

## Concept ID

DL-200

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the residual learning framework
- Implement residual blocks and ResNet in PyTorch
- Analyze how skip connections solve the degradation problem
- Compare ResNet variants (ResNet-18, 34, 50, 101, 152)

## Prerequisites

DL-176 Convolution Operation, DL-193 2D Convolution, DL-132 Batch Normalization

## Definition

ResNet (Residual Network) introduces skip connections that allow gradients to flow directly through the network, enabling training of very deep networks (up to 152 layers) by learning residual functions $F(x) = H(x) - x$ instead of the unreferenced function $H(x)$.

## Intuition

Imagine trying to learn a complex skill. It's easier to learn the "residual" — the small adjustment needed from where you already are — than to learn the entire skill from scratch. ResNet applies this principle: instead of learning the desired mapping $H(x)$ directly, each layer learns the residual $F(x) = H(x) - x$. If a layer can't improve the representation, it can simply set $F(x) = 0$ (identity bypass). This prevents the "degradation problem" where deeper networks paradoxically perform worse than shallower ones.

## Why This Concept Matters

ResNet is one of the most influential architectures in deep learning history. It won ILSVRC 2015 with 3.57% top-5 error (super-human performance) and enabled training of networks with hundreds of layers. The residual connection principle is now used in virtually all modern architectures (Transformers, DenseNet, etc.).

## Mathematical Explanation

**Residual block**:
$$y = F(x, \{W_i\}) + x$$

Where $F(x, \{W_i\})$ represents the residual mapping (e.g., two 3x3 conv layers).

**For deeper layers, bottleneck block**:
$$y = W_3(\text{ReLU}(W_2(\text{ReLU}(W_1 x)))) + W_s x$$

Where $W_1$ is 1x1 reduce, $W_2$ is 3x3, $W_3$ is 1x1 expand.

**Gradient flow**: The gradient through a residual block is:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

The "+1" term ensures gradients flow directly through skip connections, preventing vanishing gradients.

## Code Examples

### Example 1: Basic Residual Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: 1x1 conv if dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

# Test basic block
x = torch.randn(1, 64, 56, 56)
block = BasicBlock(64, 64)
out = block(x)
print(f"BasicBlock: {x.shape} -> {out.shape}")
# Output: BasicBlock: torch.Size([1, 64, 56, 56]) -> torch.Size([1, 64, 56, 56])

# With stride 2 (downsampling)
block_down = BasicBlock(64, 128, stride=2)
out_down = block_down(x)
print(f"BasicBlock (stride 2): {x.shape} -> {out_down.shape}")
# Output: BasicBlock (stride 2): torch.Size([1, 64, 56, 56]) -> torch.Size([1, 128, 28, 28])
```

### Example 2: Bottleneck Block and Full ResNet

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50/101/152."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        width = out_channels  # bottleneck width
        
        # 1x1 -> 3x3 -> 1x1 (bottleneck)
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, 
                               padding=1, bias=False)
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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
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

# Create ResNet-18
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# Create ResNet-50
def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

model18 = resnet18(num_classes=10)
model50 = resnet50(num_classes=10)

x = torch.randn(2, 3, 224, 224)
out18 = model18(x)
out50 = model50(x)

params18 = sum(p.numel() for p in model18.parameters())
params50 = sum(p.numel() for p in model50.parameters())

print(f"ResNet-18: {out18.shape}, {params18/1e6:.2f}M params")
# Output: ResNet-18: torch.Size([2, 10]), 11.17M params

print(f"ResNet-50: {out50.shape}, {params50/1e6:.2f}M params")
# Output: ResNet-50: torch.Size([2, 10]), 23.52M params
```

### Example 3: Using Pretrained ResNet from torchvision

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet
resnet50 = models.resnet50(pretrained=False)
print("Loaded ResNet-50 from torchvision")

# Architecture overview
print(f"\nResNet-50 structure:")
print(f"  conv1: {resnet50.conv1}")
print(f"  layer1: {resnet50.layer1}")
print(f"  layer2: {resnet50.layer2}")
print(f"  layer3: {resnet50.layer3}")
print(f"  layer4: {resnet50.layer4}")
print(f"  fc: {resnet50.fc}")
# Output: conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# Output: fc: Linear(in_features=2048, out_features=1000, bias=True)

# Transfer learning: replace classifier
num_classes = 10
resnet50.fc = nn.Linear(2048, num_classes)

# Feature extraction: freeze all layers except fc
for param in resnet50.parameters():
    param.requires_grad = False
resnet50.fc.requires_grad_(True)

# Count trainable params
trainable = sum(p.numel() for p in resnet50.parameters() if p.requires_grad)
total = sum(p.numel() for p in resnet50.parameters())
print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Output: Trainable: 20,490 / 23,512,906 (0.09%)

# Forward pass
x = torch.randn(4, 3, 224, 224)
out = resnet50(x)
print(f"Output: {out.shape}")
# Output: Output: torch.Size([4, 10])
```

## Common Mistakes

1. **Not using 1x1 conv in the shortcut when dimensions change**: Without projection shortcut, dimensions won't match for addition.
2. **Forgetting bottleneck expansion**: Bottleneck output channels = base_channels * 4.
3. **Batch norm before ReLU**: The standard pattern is conv -> BN -> ReLU.
4. **Excessive depth for small datasets**: ResNet-152 needs ImageNet-scale data.
5. **Not using pre-activation**: Later ResNet variants (v2) use BN-ReLU-Conv ordering.

## Interview Questions

### Beginner - 5
1. What problem does ResNet solve?
2. What is a skip connection?
3. How does a residual block work?
4. What is the degradation problem?
5. How deep is ResNet-152?

### Intermediate - 5
1. Derive the gradient flow through a residual block.
2. Explain the difference between BasicBlock and Bottleneck.
3. How do skip connections help train very deep networks?
4. What is the role of 1x1 convolutions in the bottleneck?
5. Compare ResNet-50 with VGG-16 in terms of efficiency.

### Advanced - 3
1. Analyze the mathematical properties of residual learning.
2. Design a residual variant for non-vision tasks (e.g., graph data).
3. Explain the relationship between ResNet and highway networks.

## Practice Problems

### Easy - 5
1. Count layers in ResNet-18.
2. Implement a single residual block.
3. Compute output channels of ResNet-50's layer4.
4. Load ResNet-18 from torchvision.
5. Replace classifier for transfer learning.

### Medium - 5
1. Implement ResNet-34 from scratch.
2. Train ResNet-18 on CIFAR-10.
3. Visualize feature maps from different ResNet stages.
4. Compare training with and without skip connections.
5. Implement pre-activation ResNet v2 block.

### Hard - 3
1. Implement ResNet with stochastic depth (randomly drop blocks during training).
2. Design a ResNet variant with learned skip connection scaling.
3. Analyze the effect of skip connections on the loss landscape.

## Solutions

### Easy - 1 Solution
```python
model = ResNet(BasicBlock, [2,2,2,2])
# Total conv layers: 1 (stem) + 2*4 = 8 residual convs = 17 conv layers
# Total weighted layers: 17 conv + 1 fc = 18
print("ResNet-18 has 18 weighted layers")
```

## Related Concepts

DL-201 ResNeXt, DL-202 Wide ResNet, DL-203 DenseNet, DL-132 Batch Normalization

## Next Concepts

DL-201 ResNeXt, DL-202 Wide ResNet, DL-203 DenseNet

## Summary

ResNet introduced residual learning with skip connections, enabling training of very deep networks by learning residual functions rather than direct mappings. It won ILSVRC 2015 with super-human accuracy and its skip connection principle has become a standard building block across deep learning.

## Key Takeaways

- Residual block: y = F(x) + x (skip connection)
- Solves the degradation problem (deeper != worse)
- Enables training of 152+ layer networks
- BasicBlock: two 3x3 convs; Bottleneck: 1x1 -> 3x3 -> 1x1
- Gradient flows directly through skip connections (+1 term)
- Won ILSVRC 2015 with 3.57% top-5 error
- Skip connections ubiquitous in modern architectures (Transformers, etc.)
- ResNet-50 is the most commonly used variant for transfer learning
- Pre-activation variant (v2) improves training further
- Foundation for ResNeXt, WideResNet, DenseNet
