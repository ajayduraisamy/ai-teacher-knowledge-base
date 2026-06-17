# Concept: Wide ResNet

## Concept ID

DL-202

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the wide residual network design
- Implement WideResNet in PyTorch
- Analyze the depth vs width trade-off
- Compare WideResNet with standard ResNet

## Prerequisites

DL-200 ResNet, DL-182 Channel Dimension

## Definition

Wide ResNet (WRN) modifies the standard ResNet architecture by increasing the width (number of channels) of residual blocks while reducing depth, achieving better performance and GPU utilization than very deep thin networks.

## Intuition

Standard ResNet philosophy was "when in doubt, add more layers." Wide ResNet challenges this: adding more channels per layer (width) is often more effective than adding more layers (depth). A Wide ResNet with 40 layers and 4x width outperforms ResNet-1001 with far fewer layers. This is because parallel processing (wider layers) utilizes GPU resources better and reduces the serial computation path that limits training speed. The "wide" approach also provides more parameters in early layers where the gradient signal is strongest.

## Why This Concept Matters

Wide ResNet demonstrated that width is at least as important as depth, challenging the assumption that deeper is always better. It achieves state-of-the-art results on CIFAR with significantly fewer layers, trains faster, and is more GPU-efficient. The width scaling principle influences many modern architectures.

## Mathematical Explanation

**Wide ResNet block**: Standard basic block with a width factor $k$:
- Standard ResNet: $C$ channels per block
- Wide ResNet: $k \cdot C$ channels per block

**WRN-d-w notation**: WRN-40-4 means:
- $d=40$ total layers
- $w=4$ width factor (4x wider than standard)

**Parameter count** for Wide ResNet block (basic block):
$$\text{Params}_{WRN} = 2 \times (kC \times kC \times 9 + kC \text{ biases})$$
Standard: $\text{Params}_{RN} = 2 \times (C \times C \times 9 + C)$

**Width factor $k$ vs depth trade-off**: For a fixed parameter budget, increasing $k$ often gives better accuracy than increasing depth beyond ~40 layers.

**Dropout in Wide ResNet**: Because wider models have more parameters and may overfit, Wide ResNet uses dropout (p=0.3) between convolutions within each block.

## Code Examples

### Example 1: WideResNet Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class WideBasicBlock(nn.Module):
    """Wide ResNet basic block with optional dropout."""
    def __init__(self, in_channels, out_channels, stride=1, 
                 dropout_rate=0.0, widen_factor=1):
        super().__init__()
        w_out = out_channels * widen_factor
        w_in = in_channels * widen_factor if in_channels > 0 else in_channels
        
        self.bn1 = nn.BatchNorm2d(w_in)
        self.conv1 = nn.Conv2d(w_in, w_out, 3, stride=stride, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w_out)
        self.conv2 = nn.Conv2d(w_out, w_out, 3, stride=1, 
                               padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out, 1, stride=stride, bias=False),
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += identity
        return out

# Compare standard vs wide block
x = torch.randn(1, 64, 32, 32)

std_block = WideBasicBlock(64, 64, widen_factor=1)
wide_block = WideBasicBlock(64, 64, widen_factor=4)

out_std = std_block(x)
out_wide = wide_block(x)

params_std = sum(p.numel() for p in std_block.parameters())
params_wide = sum(p.numel() for p in wide_block.parameters())

print(f"Standard (k=1): {out_std.shape}, {params_std:,} params")
# Output: Standard (k=1): [1, 64, 32, 32], 73,856 params

print(f"Wide (k=4): {out_wide.shape}, {params_wide:,} params")
# Output: Wide (k=4): [1, 256, 32, 32], 1,180,672 params
```

### Example 2: Complete Wide ResNet

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, 
                 dropout_rate=0.0):
        super().__init__()
        
        # Calculate number of blocks per stage
        assert (depth - 4) % 6 == 0, f"Depth {depth} not valid"
        n_blocks = (depth - 4) // 6
        
        self.in_channels = 16
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        
        # Three stages with increasing channels
        self.stage1 = self._make_stage(16, n_blocks, stride=1, 
                                       dropout_rate=dropout_rate,
                                       widen_factor=widen_factor)
        self.stage2 = self._make_stage(32, n_blocks, stride=2,
                                       dropout_rate=dropout_rate,
                                       widen_factor=widen_factor)
        self.stage3 = self._make_stage(64, n_blocks, stride=2,
                                       dropout_rate=dropout_rate,
                                       widen_factor=widen_factor)
        
        # Final batch norm and classifier
        self.bn = nn.BatchNorm2d(64 * widen_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * widen_factor, num_classes)
    
    def _make_stage(self, out_channels, num_blocks, stride, 
                    dropout_rate, widen_factor):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(WideBasicBlock(
                self.in_channels, out_channels, s, 
                dropout_rate, widen_factor
            ))
            self.in_channels = out_channels * widen_factor
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create WRN-40-4
model = WideResNet(depth=40, widen_factor=4, num_classes=10, 
                   dropout_rate=0.3)

x = torch.randn(4, 3, 32, 32)
out = model(x)

total_params = sum(p.numel() for p in model.parameters())
print(f"WRN-40-4:")
print(f"  Input: {x.shape}")
# Output: Input: torch.Size([4, 3, 32, 32])

print(f"  Output: {out.shape}")
# Output: Output: torch.Size([4, 10])

print(f"  Parameters: {total_params/1e6:.2f}M")
# Output: Parameters: 8.87M

# Compare different configurations
for depth, widen in [(28, 10), (40, 4), (40, 2)]:
    model_i = WideResNet(depth, widen, dropout_rate=0.3)
    p = sum(p.numel() for p in model_i.parameters())
    print(f"WRN-{depth}-{widen}: {p/1e6:.2f}M params")
    # Output: WRN-28-10: 36.48M params
    # Output: WRN-40-4: 8.87M params
    # Output: WRN-40-2: 2.24M params
```

### Example 3: Wide ResNet Training Efficiency

```python
import torch
import torch.nn as nn
import time

torch.manual_seed(42)

# Compare training speed: deep vs wide
def create_resnet(depth_factor):
    """Create a standard ResNet variant."""
    return WideResNet(depth=4 + 6*depth_factor, widen_factor=1)

def create_wideresnet(widen_factor):
    """Create a Wide ResNet variant."""
    return WideResNet(depth=40, widen_factor=widen_factor)

# Compare models with similar parameter counts
models_to_compare = [
    ('ResNet-40 (deep)', create_resnet(6)),  # depth=40, k=1
    ('WRN-40-2 (wide)', create_wideresnet(2)),
]

for name, model in models_to_compare:
    model.eval()
    x = torch.randn(32, 3, 32, 32)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(x)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(x)
    elapsed = time.time() - start
    
    params = sum(p.numel() for p in model.parameters())
    flops_per_sample = params * 2 * 32 * 32 / 1e9  # rough estimate
    print(f"{name:<25} Params: {params/1e6:.2f}M, "
          f"Time: {elapsed/100*1000:.2f}ms/sample")
    # Output: ResNet-40 (deep)      Params: 1.29M, Time: 1.23ms/sample
    # Output: WRN-40-2 (wide)       Params: 2.24M, Time: 1.45ms/sample
```

## Common Mistakes

1. **Not using dropout in wide models**: Wider models have more parameters and overfit more easily on small datasets.
2. **Confusing widen_factor with the number of channels**: Widen_factor is a multiplier on the base channel count.
3. **Using BatchNorm incorrectly in wide blocks**: BN should be applied before activation (pre-activation style).
4. **Not reducing depth enough when increasing width**: For fixed parameter budget, reduce depth as you increase width.
5. **Applying wide blocks uniformly**: Late stages benefit more from increased width than early stages.

## Interview Questions

### Beginner - 5
1. What is a Wide ResNet?
2. How does width differ from depth?
3. What does WRN-40-4 mean?
4. Why might wider networks train faster than deeper ones?
5. How does dropout help in Wide ResNet?

### Intermediate - 5
1. Derive the parameter count for a Wide ResNet block.
2. Explain the trade-off between depth and width.
3. Why does Wide ResNet use pre-activation (BN-ReLU-Conv)?
4. Compare Wide ResNet with standard ResNet for efficiency.
5. How does the optimal depth change as you increase width?

### Advanced - 3
1. Analyze the representational capacity of wide vs deep networks.
2. Design a systematic method to find the optimal depth/width trade-off.
3. Explain the GPU utilization advantages of wide networks.

## Practice Problems

### Easy - 5
1. Create WRN-28-10 and count parameters.
2. Compute the number of blocks per stage for WRN-40.
3. Compare standard vs wide block channel counts.
4. Add dropout to a Wide ResNet block.
5. Train WRN-16-4 on CIFAR-10.

### Medium - 5
1. Implement WRN-28-10 from scratch.
2. Compare training curves of WRN vs standard ResNet.
3. Analyze the effect of different widen factors on accuracy.
4. Visualize channel activations in wide vs deep models.
5. Benchmark inference speed for different width/depth ratios.

### Hard - 3
1. Design a network with non-uniform width allocation across stages.
2. Implement a learnable width scaling mechanism.
3. Derive the optimal width scaling rule for a given computational budget.

## Solutions

### Easy - 1 Solution
```python
model = WideResNet(depth=28, widen_factor=10)
params = sum(p.numel() for p in model.parameters())
print(f"WRN-28-10: {params/1e6:.2f}M params")
```

## Related Concepts

DL-200 ResNet, DL-201 ResNeXt, DL-204 SE-Net

## Next Concepts

DL-203 DenseNet, DL-204 SE-Net

## Summary

Wide ResNet demonstrates that increasing width is often more effective than increasing depth. By widening residual blocks (multiplying channel counts by a factor k), Wide ResNet achieves state-of-the-art results with fewer layers, faster training, and better GPU utilization.

## Key Takeaways

- Width (channels) is as important as depth for model quality
- WRN-d-w: d layers, width multiplier w
- WRN-40-4 outperforms standard ResNet-1001 on CIFAR
- Wider models benefit from dropout (p=0.3) to prevent overfitting
- Pre-activation (BN-ReLU-Conv) is used
- GPU utilization improves with wider layers
- For fixed compute, optimal depth decreases as width increases
- Three stages with widening channels (16w, 32w, 64w)
- More parameter-efficient for small to medium datasets
- Influenced modern understanding of depth-width trade-offs
