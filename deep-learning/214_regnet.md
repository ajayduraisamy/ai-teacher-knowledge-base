# Concept: RegNet

## Concept ID

DL-214

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the RegNet design paradigm
- Implement RegNet stages in PyTorch
- Analyze the network design space and quantization
- Compare RegNet with manually designed architectures

## Prerequisites

DL-200 ResNet, DL-215 Architecture Search, DL-182 Channel Dimension

## Definition

RegNet (Regular Network) is a family of CNN architectures derived from systematic exploration of the network design space, revealing simple design principles that produce high-performing networks without complex manual tuning or expensive NAS.

## Intuition

Instead of manually designing architecture or running expensive NAS, the RegNet authors systematically explored the design space of network configurations (depth, width, groups, bottleneck ratio) and found that optimal networks follow simple, quantized linear patterns. Specifically, the optimal channel counts across stages follow a linear relationship: stage i's channels = a + b * i for some constants a and b. This means the best architectures are not random but follow regular patterns. The "Reg" in RegNet stands for "Regular" — the designs are regular and predictable.

## Why This Concept Matters

RegNet provides a principled approach to architecture design, replacing intuition-driven design with systematic exploration. It showed that high-performing architectures can be derived from simple design principles, challenging the notion that complex NAS is necessary. RegNets are competitive with EfficientNets across a range of compute budgets.

## Mathematical Explanation

**Design space**:
- Depth per stage: d_i
- Width per stage: w_i
- Bottleneck ratio: b_i (b_i = 1 for basic block, 4 for bottleneck)
- Group width: g_i (for grouped convs)

**Linear parameterization of widths**:
w_i = a + b * i for i = 0, ..., s-1

Where s is the number of stages, and a, b are constants.

**Quantization**: Channel counts are rounded to multiples of 8 for hardware efficiency.

**AnyNet -> RegNet**: Start with any configuration (AnyNet), constrain to regular patterns, find Pareto-optimal models, observe that all Pareto-optimal models follow the linear width pattern.

## Code Examples

### Example 1: Linear Width Parameterization

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

def generate_regnet_stages(depth=20, width_mult=1.0, groups=16):
    """Generate RegNet stage configurations using linear width."""
    n_stages = 4
    total_depth = depth
    
    # Depth per stage (roughly equal)
    depths = [total_depth // n_stages] * n_stages
    # Distribute remainder
    for i in range(total_depth % n_stages):
        depths[i] += 1
    
    # Linear width: w_i = a + b * i
    # For simplicity, use predefined widths
    widths = [32, 64, 128, 256]
    widths = [int(w * width_mult) for w in widths]
    widths = [int(round(w / 8) * 8) for w in widths]  # quantize to 8
    
    return depths, widths

class RegNetBlock(nn.Module):
    """Simple RegNet block with grouped conv."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Grouped conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, 
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + identity)

# Test RegNet block
x = torch.randn(1, 32, 56, 56)
block = RegNetBlock(32, 64, stride=2, groups=4)
out = block(x)
print(f"RegNet block: {x.shape} -> {out.shape}")
```

### Example 2: Simple RegNet Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class RegNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # RegNet parameters (from design space exploration)
        depths = [2, 3, 4, 2]        # blocks per stage
        widths = [32, 64, 128, 256]  # channels per stage
        groups = [4, 8, 16, 32]      # groups per stage
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = widths[0]
        
        for i in range(len(depths)):
            stage = nn.Sequential()
            # First block may downsample
            stage.add_module(f'block_0', RegNetBlock(
                in_channels, widths[i], 
                stride=2 if i > 0 else 1,
                groups=groups[i]
            ))
            # Remaining blocks (no downsampling)
            for j in range(1, depths[i]):
                stage.add_module(f'block_{j}', RegNetBlock(
                    widths[i], widths[i],
                    stride=1, groups=groups[i]
                ))
            in_channels = widths[i]
            self.stages.append(stage)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = RegNet(num_classes=10)
x = torch.randn(2, 3, 224, 224)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"RegNet: {out.shape}, {total_params/1e6:.2f}M params")
```

### Example 3: Design Space Exploration

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Explore linear width parameterization
def compute_widths(a, b, n_stages=4):
    """Compute stage widths using linear formula."""
    widths = [int(a + b * i) for i in range(n_stages)]
    return [int(round(w / 8) * 8) for w in widths]

print(f"{'a':<6} {'b':<6} {'Widths':<30}")
for a, b in [(32, 16), (24, 24), (48, 0), (16, 32)]:
    widths = compute_widths(a, b)
    print(f"{a:<6} {b:<6} {str(widths):<30}")

# Compare RegNet with ResNet width patterns
resnet_widths = [64, 128, 256, 512]
regnet_widths = compute_widths(32, 32)

print(f"\nResNet widths: {resnet_widths}")
print(f"RegNet-style widths: {regnet_widths}")
```

## Common Mistakes

1. **Ignoring width quantization**: Channel counts should be multiples of 8 for hardware efficiency.
2. **Not using group widths**: RegNet benefits from grouped convolutions with appropriate group widths.
3. **Arbitrary depth distribution**: Depth should be distributed based on design space findings.
4. **Using uniform stage designs**: Different stages should have different widths and groups.
5. **Confusing RegNet with AnyNet**: RegNet constrains the design space; AnyNet is the unconstrained version.

## Interview Questions

### Beginner - 5
1. What is RegNet?
2. What does "Reg" in RegNet stand for?
3. How does RegNet differ from manually designed architectures?
4. What is the design space?
5. What is width quantization?

### Intermediate - 5
1. Explain the linear width parameterization.
2. How was the RegNet design space explored?
3. Compare RegNet with EfficientNet.
4. What are the key design parameters in RegNet?
5. Why are simple regular patterns optimal?

### Advanced - 3
1. Derive the optimal width parameterization for a given FLOP budget.
2. Design a systematic approach to architecture design space exploration.
3. Compare RegNet's methodology with neural architecture search.

## Practice Problems

### Easy - 5
1. Compute quantized widths for a=48, b=24.
2. Load RegNet from torchvision.
3. Count stages in RegNet.
4. Count groups in different RegNet stages.
5. Compare RegNet vs ResNet parameter counts.

### Medium - 5
1. Implement a RegNet from design space parameters.
2. Train RegNet on CIFAR-100.
3. Explore the AnyNet design space.
4. Visualize the linear width pattern.
5. Compare RegNet with NAS-found architectures.

### Hard - 3
1. Reproduce the RegNet design space exploration.
2. Design a RegNet variant optimized for mobile devices.
3. Analyze the theoretical justification for regular network design.

## Solutions

### Easy - 1 Solution
```python
a, b = 48, 24
widths = [a + b*i for i in range(4)]
widths = [int(round(w/8)*8) for w in widths]
print(f"Widths: {widths}")
```

## Related Concepts

DL-215 Architecture Search, DL-200 ResNet, DL-205 EfficientNet, DL-201 ResNeXt

## Next Concepts

DL-215 Architecture Search

## Summary

RegNet systematically explores the network design space to discover simple design principles, particularly that optimal architectures follow a linear width parameterization. RegNets offer strong performance across compute budgets without requiring expensive NAS or manual tuning.

## Key Takeaways

- Design space exploration reveals simple regular patterns
- Linear width: w_i = a + b * i per stage
- Widths quantized to multiples of 8 for efficiency
- Group width (g) is an important design parameter
- AnyNet -> RegNet: constrain to regularity for better results
- Competitive with EfficientNet across compute budgets
- Bottleneck ratio and group width follow predictable patterns
- Stage depth distribution is important
- Provides principled alternative to NAS and manual design
- RegNet parameters (depth, width, groups) follow simple formulas
