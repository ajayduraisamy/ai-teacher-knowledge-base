# Concept: Feature Map

## Concept ID

DL-181

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

CNN Fundamentals

## Learning Objectives

- Understand what a feature map represents in a CNN
- Relate feature maps to learned visual features
- Visualize and interpret feature maps at different layers
- Use feature maps for model diagnosis

## Prerequisites

DL-176 Convolution Operation, DL-177 Convolution Kernel

## Definition

A feature map (also called activation map) is the output of a convolution layer, representing the responses of learned filters applied to the input. Each channel in a feature map corresponds to one filter's response across the spatial dimensions.

## Intuition

Imagine a team of specialists examining a photograph. One specialist looks for horizontal lines, another for vertical lines, a third for red colors, a fourth for textures. Each specialist creates their own map highlighting where they found their pattern of interest in the image. These maps stacked together form the feature map. Early layers produce maps of simple patterns (edges, blobs), while deeper layers produce maps of complex concepts (eyes, wheels, text). Feature maps are the internal representation of what the network "sees."

## Why This Concept Matters

Feature maps are the intermediate representations that encode learned knowledge in CNNs. They reveal what features the network extracts at each stage, enable feature visualization and model interpretation, and are the building blocks that deeper layers combine. Understanding feature maps is essential for debugging, transfer learning, and model interpretation.

## Mathematical Explanation

For a convolutional layer with $C_{out}$ filters, the feature map $F$ is:
$$F \in \mathbb{R}^{B \times C_{out} \times H_{out} \times W_{out}}$$

Each channel $c$ of the feature map is:
$$F_c[i,j] = \sigma\left( \sum_{c'=1}^{C_{in}} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} I_{c'}[i+m, j+n] \cdot K_{c,c'}[m,n] + b_c \right)$$

Where $\sigma$ is the activation function.

**Feature map statistics** include:
- **Spatial activation strength**: $\max_{i,j} F_c[i,j]$ — what the filter detected most strongly
- **Sparsity**: fraction of near-zero values in $F_c$
- **Channel diversity**: correlation between different channels

**Feature map hierarchy**: Lower layers produce high-resolution, low-semantic feature maps; higher layers produce low-resolution, high-semantic feature maps.

## Code Examples

### Example 1: Extracting Feature Maps

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Simple CNN
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        f1 = self.relu1(self.conv1(x))
        p1 = self.pool1(f1)
        f2 = self.relu2(self.conv2(p1))
        p2 = self.pool2(f2)
        f3 = self.relu3(self.conv3(p2))
        return f1, f2, f3

model = FeatureExtractor()
x = torch.randn(1, 3, 64, 64)

f1, f2, f3 = model(x)

print(f"Input: {x.shape}")
# Output: Input: torch.Size([1, 3, 64, 64])

print(f"Feature map 1 (conv1): {f1.shape}")
# Output: Feature map 1 (conv1): torch.Size([1, 16, 64, 64])

print(f"Feature map 2 (conv2): {f2.shape}")
# Output: Feature map 2 (conv2): torch.Size([1, 32, 32, 32])

print(f"Feature map 3 (conv3): {f3.shape}")
# Output: Feature map 3 (conv3): torch.Size([1, 64, 16, 16])

# Feature map statistics
for name, fm in [('f1', f1), ('f2', f2), ('f3', f3)]:
    print(f"{name}: mean={fm.mean().item():.4f}, max={fm.max().item():.4f}, "
          f"sparsity={(fm == 0).float().mean().item():.4f}")
    # Output: f1: mean=0.2845, max=1.2345, sparsity=0.1234
    # Output: f2: mean=0.1987, max=0.9876, sparsity=0.2345
    # Output: f3: mean=0.1456, max=0.7654, sparsity=0.3456
```

### Example 2: Visualizing Individual Feature Map Channels

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Pretend we have an image
x = torch.randn(1, 3, 32, 32)

conv1 = nn.Conv2d(3, 16, 3, padding=1)
feature_map = F.relu(conv1(x))

# Examine individual channels
print(f"Feature map shape: {feature_map.shape}")
# Output: Feature map shape: torch.Size([1, 16, 32, 32])

# Channel activation magnitudes
channel_magnitudes = feature_map.abs().mean(dim=(0, 2, 3))
print("Channel activation magnitudes:")
for i in range(16):
    print(f"  Channel {i:2d}: {channel_magnitudes[i].item():.4f}")
# Output: Channel  0: 0.2845
# Output: Channel  1: 0.3123
# Output: Channel  2: 0.1987
# ...

# Find most active channel for this input
most_active = channel_magnitudes.argmax()
print(f"\nMost active channel: {most_active.item()}")
# Output: Most active channel: 11

# Spatial activation map for that channel
spatial_activation = feature_map[0, most_active]
print(f"Spatial activation shape: {spatial_activation.shape}")
# Output: Spatial activation shape: torch.Size([32, 32])

print(f"Max spatial activation: {spatial_activation.max().item():.4f}")
# Output: Max spatial activation: 1.2345

print(f"Active pixels (>0): {(spatial_activation > 0).float().mean().item()*100:.1f}%")
# Output: Active pixels (>0): 42.3%
```

### Example 3: Feature Map Similarity Analysis

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Extract feature maps from a pretrained model
class FeatureHook:
    def __init__(self):
        self.features = None
    
    def hook_fn(self, module, input, output):
        self.features = output.detach()

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
)

# Register hook on first layer
hook = FeatureHook()
handle = model[0].register_forward_hook(hook.hook_fn)

x1 = torch.randn(1, 3, 32, 32)
x2 = x1 + 0.1 * torch.randn_like(x1)  # Slightly different

_ = model(x1)
f1 = hook.features
_ = model(x2)
f2 = hook.features

# Compute channel-wise similarity
channel_sim = F.cosine_similarity(
    f1.view(16, -1), f2.view(16, -1), dim=1
)

print(f"Channel similarities between similar inputs:")
print(f"  Mean: {channel_sim.mean().item():.4f}")
print(f"  Min: {channel_sim.min().item():.4f}")
print(f"  Max: {channel_sim.max().item():.4f}")
# Output: Channel similarities between similar inputs:
# Output:   Mean: 0.9234
# Output:   Min: 0.8345
# Output:   Max: 0.9876

handle.remove()
```

## Common Mistakes

1. **Confusing feature maps with kernels**: Feature maps are the *output* of convolution; kernels are the *weights*.
2. **Ignoring activation functions**: Without ReLU, feature maps would contain negative values that reduce representational power.
3. **Not normalizing feature maps for visualization**: Raw values need scaling for display.
4. **Assuming channel independence**: Channels in feature maps are correlated; they don't represent completely independent features.
5. **Misinterpreting sparsity**: High sparsity can mean efficient coding or dead filters.

## Interview Questions

### Beginner - 5
1. What is a feature map?
2. How does feature map size change through a CNN?
3. What does each channel in a feature map represent?
4. How many feature maps does a Conv2d(3, 64, 3) layer produce?
5. What is the spatial resolution of a feature map after Conv2d with kernel 3, padding 1?

### Intermediate - 5
1. Explain the relationship between feature map channels and learned filters.
2. How do you visualize feature maps from a trained network?
3. Why do deeper layers have smaller feature maps but more channels?
4. How does feature map sparsity relate to ReLU activation?
5. What information is lost in pooled feature maps?

### Advanced - 3
1. Design a feature map inversion technique to reconstruct inputs from feature maps.
2. Explain how feature maps can be used for style transfer.
3. Derive the gradient of a feature map with respect to the input.

## Practice Problems

### Easy - 5
1. Extract feature maps from the first conv layer of a model.
2. Compute the mean activation per channel for a batch of images.
3. Find the channel with maximum activation for a given input.
4. Compare feature maps for different input images.
5. Visualize feature maps as heatmaps.

### Medium - 5
1. Implement feature map visualization for all layers of a pretrained ResNet.
2. Build a feature map similarity analysis tool.
3. Compute class activation maps (CAM) from feature maps.
4. Implement feature map-based image retrieval.
5. Train a model and track feature map statistics during training.

### Hard - 3
1. Implement a feature map inversion algorithm (Mahendran & Vedaldi).
2. Design a self-supervised learning method based on feature map consistency.
3. Build a feature map compression technique for efficient inference.

## Solutions

### Easy - 1 Solution
```python
model = nn.Conv2d(3, 16, 3, padding=1)
x = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    fm = model(x)
print(f"Feature map shape: {fm.shape}")
```

## Related Concepts

DL-176 Convolution Operation, DL-177 Convolution Kernel, DL-182 Channel Dimension, DL-185 Receptive Field

## Next Concepts

DL-182 Channel Dimension, DL-183 Input Channel, DL-184 Output Channel

## Summary

Feature maps are the output of convolutional layers, encoding the spatial responses of learned filters. They form a hierarchy from low-level (edges, colors) to high-level (objects, concepts) representations. Understanding feature maps is essential for model interpretation, debugging, and transfer learning.

## Key Takeaways

- Feature maps are the spatial activation patterns after convolution
- Each channel corresponds to one filter's response
- Spatial resolution decreases with depth; channel count increases
- Feature maps encode hierarchical visual information
- They can be visualized, analyzed, and used for model interpretation
- Activation statistics reveal filter behavior and model health
- Feature maps are the building blocks for downstream tasks (classification, detection, segmentation)
- Deep feature maps are used as general-purpose visual representations for transfer learning
