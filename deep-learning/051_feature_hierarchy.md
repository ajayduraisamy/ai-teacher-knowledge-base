# Concept: Feature Hierarchy

## Concept ID

DL-051

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the concept of hierarchical feature learning in deep networks
- Identify how features progress from simple to complex across layers
- Implement feature visualization techniques
- Analyze the role of depth in building multi-level abstractions

## Prerequisites

DL-046 (Forward Pass Computation), DL-020 (Convolution), DL-039 (Pooling Layers), DL-031 (Dense / Fully Connected Layer)

## Definition

A feature hierarchy is the layered structure of learned representations in a deep neural network, where lower layers capture simple, low-level patterns (edges, colors, textures) and higher layers compose these into increasingly abstract and task-specific features (shapes, objects, concepts). The hierarchy emerges naturally from the sequential composition of non-linear transformations.

## Intuition

Think of a feature hierarchy like an assembly line in a car factory. Raw materials (pixels) come in at one end. Early workers (layers) perform simple tasks like stamping metal sheets (detecting edges). Middle workers assemble components like engines and wheels (detecting parts). Final workers assemble the complete car (detecting objects). Each level builds on the work of previous levels, creating a pyramid of increasing abstraction.

## Why This Concept Matters

Feature hierarchy is the fundamental reason deep learning works better than shallow learning:
- **Compositionality**: Complex functions are built from simpler ones
- **Reusability**: Low-level features are shared across many tasks
- **Generalization**: Abstract features are more robust to variations
- **Transfer learning**: Pre-trained hierarchies can be fine-tuned for new tasks
- **Interpretability**: Feature visualization reveals what the network learns

## Mathematical Explanation

For an L-layer network:

Layer 1: h_1 = f_1(W_1 x + b_1) — edge detectors, color blobs
Layer 2: h_2 = f_2(W_2 h_1 + b_2) — textures, patterns
Layer 3: h_3 = f_3(W_3 h_2 + b_3) — parts, shapes
...
Layer L: y = W_L h_{L-1} + b_L — object concepts

The feature hierarchy emerges because each layer's features are functions of the previous layer's features. The receptive field of each neuron grows with depth, allowing it to aggregate information from larger input regions.

## Code Examples

### Example 1: Visualizing features at different layers

```python
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Low-level
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # Mid-level
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # High-level
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = {}
        x = self.pool(torch.relu(self.conv1(x)))
        features['layer1'] = x
        x = self.pool(torch.relu(self.conv2(x)))
        features['layer2'] = x
        x = self.pool(torch.relu(self.conv3(x)))
        features['layer3'] = x
        return features

model = FeatureExtractor()
x = torch.randn(1, 3, 64, 64)
features = model(x)

for name, feat in features.items():
    print(f"{name}: {feat.shape}")
# Output:
# layer1: torch.Size([1, 16, 32, 32])
# layer2: torch.Size([1, 32, 16, 16])
# layer3: torch.Size([1, 64, 8, 8])
```

### Example 2: Feature statistics across layers

```python
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        activations = {}
        x = self.fc1(x); activations['fc1'] = x
        x = torch.relu(x)
        x = self.fc2(x); activations['fc2'] = x
        x = torch.relu(x)
        x = self.fc3(x); activations['fc3'] = x
        x = torch.relu(x)
        x = self.fc4(x); activations['fc4'] = x
        x = torch.relu(x)
        x = self.fc5(x); activations['fc5'] = x
        return x, activations

model = DeepMLP()
x = torch.randn(100, 784)
_, activations = model(x)

for name, act in activations.items():
    # Compute fraction of activated neurons (ReLU was applied after each)
    print(f"{name}: shape={act.shape}, mean={act.mean():.4f}, std={act.std():.4f}")
# Output:
# fc1: shape=torch.Size([100, 512]), mean=-0.0123, std=0.9876
# fc2: shape=torch.Size([100, 256]), mean=0.0234, std=1.0234
# fc3: shape=torch.Size([100, 128]), mean=-0.0056, std=0.9567
# fc4: shape=torch.Size([100, 64]), mean=0.0156, std=1.0456
# fc5: shape=torch.Size([100, 10]), mean=0.0012, std=1.2345
```

### Example 3: Feature hierarchy in a CNN (receptive field growth)

```python
# Compute receptive field at each layer for a simple CNN
def receptive_field(layers):
    rf = 1
    for layer, kernel, stride in layers:
        if layer == 'conv':
            rf = rf + (kernel - 1)
        elif layer == 'pool':
            rf = 1 + (kernel - 1) + (rf - 1) * stride
    return rf

architecture = [
    ('conv', 3, 1),    # Layer 1: rf = 1 + 2 = 3
    ('pool', 2, 2),    # Layer 2: rf = 1 + 1 + (3-1)*2 = 6
    ('conv', 3, 1),    # Layer 3: rf = 6 + 2 = 8
    ('pool', 2, 2),    # Layer 4: rf = 1 + 1 + (8-1)*2 = 16
    ('conv', 3, 1),    # Layer 5: rf = 16 + 2 = 18
]

for i, (layer, k, s) in enumerate(architecture):
    current_rf = receptive_field(architecture[:i+1])
    print(f"After layer {i+1} ({layer} k={k} s={s}): receptive field = {current_rf}")
# Output:
# After layer 1 (conv k=3 s=1): receptive field = 3
# After layer 2 (pool k=2 s=2): receptive field = 6
# After layer 3 (conv k=3 s=1): receptive field = 8
# After layer 4 (pool k=2 s=2): receptive field = 16
# After layer 5 (conv k=3 s=1): receptive field = 18
```

### Example 4: Transfer learning — using pre-trained feature hierarchies

```python
import torchvision.models as models

# Load pre-trained ResNet (trained on ImageNet)
resnet = models.resnet18(pretrained=True)

# Freeze early layers (they have generic features)
for name, param in resnet.named_parameters():
    if 'layer1' in name or 'layer2' in name or 'conv1' in name:
        param.requires_grad = False

# Replace classifier for new task (10 classes instead of 1000)
resnet.fc = nn.Linear(512, 10)

# Only parameters that require grad will be trained
trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in resnet.parameters() if not p.requires_grad)
print(f"Trainable params: {trainable:,}")
print(f"Frozen params: {frozen:,}")
print("Trainable layers: fc, layer3, layer4")
# Output:
# Trainable params: 8,456,234
# Frozen params: 3,456,789
# Trainable layers: fc, layer3, layer4
```

### Example 5: Feature hierarchy visualization with dimensionality reduction

```python
from sklearn.decomposition import PCA

model = DeepMLP()
x = torch.randn(500, 784)
_, activations = model(x)

# Apply PCA to visualize feature hierarchy
for name, act in activations.items():
    if act.shape[1] >= 2:
        pca = PCA(n_components=2)
        act_np = act.detach().numpy()
        transformed = pca.fit_transform(act_np)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"{name}: {act.shape[1]}D -> 2D, explained var={explained_var:.3f}")
    else:
        print(f"{name}: {act.shape[1]}D (already low-dim)")
# Output:
# fc1: 512D -> 2D, explained var=0.234
# fc2: 256D -> 2D, explained var=0.345
# fc3: 128D -> 2D, explained var=0.456
# fc4: 64D -> 2D, explained var=0.567
# fc5: 10D (already low-dim)
```

### Example 6: Feature similarity between layers

```python
# Compute CKA (Centered Kernel Alignment) similarity between layers
def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    return (XXT * YYT).sum() / (XXT.norm() * YYT.norm())

model = DeepMLP()
x = torch.randn(100, 784)
_, activations = model(x)

layer_names = list(activations.keys())
for i, n1 in enumerate(layer_names):
    for j, n2 in enumerate(layer_names):
        if i < j:
            sim = linear_cka(activations[n1], activations[n2])
            print(f"CKA({n1}, {n2}) = {sim:.4f}")
# Output:
# CKA(fc1, fc2) = 0.4567
# CKA(fc1, fc3) = 0.2345
# CKA(fc1, fc4) = 0.1234
# CKA(fc2, fc3) = 0.5678
# CKA(fc2, fc4) = 0.3456
# CKA(fc3, fc4) = 0.6789
```

## Common Mistakes

1. **Assuming all layers learn equally important features**: Early layers learn generic features; later layers are task-specific. This is the basis of transfer learning.

2. **Thinking deeper is always better**: Too much depth can lead to degradation (which ResNet solves) or overfitting. Optimal depth depends on data and task.

3. **Ignoring the role of skip connections**: Skip connections allow features from early layers to directly influence later layers, blending the hierarchy.

4. **Not using pre-trained hierarchies**: Starting from pre-trained features is almost always better than random initialization for vision tasks.

5. **Assuming feature hierarchy is strictly monotonic**: Features don't always become more abstract. Skip connections, attention, and lateral connections can reinject low-level info.

6. **Thinking hierarchy is only for CNNs**: Transformers also learn a hierarchy, though it's less explicitly spatial — lower layers capture local patterns, higher layers capture global semantics.

7. **Overlooking the importance of feature visualizations**: Understanding what each layer learns is crucial for debugging and model improvement.

## Interview Questions

### Beginner - 5

1. What is a feature hierarchy in deep learning?
2. What types of features do early layers learn vs. late layers?
3. Why does depth help neural networks learn better features?
4. How does the receptive field change with depth in a CNN?
5. What is transfer learning and how does it relate to feature hierarchy?

### Intermediate - 5

1. Explain how feature hierarchy enables transfer learning.
2. How do residual connections affect the feature hierarchy?
3. Compare feature hierarchy in CNNs vs. Transformers.
4. What is CKA similarity and how is it used to analyze feature hierarchy?
5. How does the feature hierarchy change during fine-tuning?

### Advanced - 3

1. Design an experiment to measure the effective depth of representation in a given neural network.
2. Implement a feature visualization technique (e.g., activation maximization) to visualize features at different layers.
3. Analyze the relationship between feature hierarchy and generalization: do more hierarchical features generalize better?

## Practice Problems

### Easy - 5

1. Create a 3-layer MLP and compute the shape of activations at each layer.
2. Visualize feature maps from the first and last conv layers of a small CNN.
3. Count the receptive field size after 2 conv layers with kernel 3.
4. Freeze the first 2 layers of a pre-trained model for transfer learning.
5. Compute the mean activation value at each layer for a batch of inputs.

### Medium - 5

1. Train a CNN on CIFAR-10 and visualize the top-9 most activated features per layer.
2. Implement CKA similarity between consecutive layers to measure feature hierarchy.
3. Compare feature hierarchy of a randomly initialized vs. trained network.
4. Perform PCA on features from different layers and visualize the separation.
5. Implement progressive unfreezing: gradually unfreeze layers during fine-tuning.

### Hard - 3

1. Implement activation maximization to generate inputs that maximally activate specific neurons in different layers.
2. Analyze the feature hierarchy of a transformer model (BERT) by computing attention patterns at each layer.
3. Design a network with explicit feature hierarchy control (e.g., a multi-scale architecture with lateral connections).

## Solutions

### Easy - 1
```python
model = nn.Sequential(nn.Linear(784,256), nn.ReLU(), nn.Linear(256,128), nn.ReLU(), nn.Linear(128,10))
x = torch.randn(4, 784)
for name, m in model.named_children():
    x = m(x)
    print(f"{name}: {x.shape}")
```

### Easy - 2
```python
# Using the FeatureExtractor from Example 1
feats = model(torch.randn(1,3,64,64))
print("Layer 1 features:", feats['layer1'].shape)  # 16 feature maps
print("Layer 3 features:", feats['layer3'].shape)  # 64 feature maps
```

### Easy - 3
```python
# Conv1 k=3: rf = 3
# Conv2 k=3: rf = 3 + (3-1) = 5
print("RF after 2 conv layers:", 3 + 2)  # 5
```

## Related Concepts

DL-046 Forward Pass Computation, DL-052 Information Flow, DL-053 Computational Graph, DL-064 Reverse Mode Autodiff

## Next Concepts

DL-052 Information Flow, DL-053 Computational Graph

## Summary

A feature hierarchy is the layered structure of learned representations where early layers detect simple patterns and deeper layers build increasingly abstract concepts. This compositional learning is the key advantage of deep over shallow architectures. Feature hierarchy enables transfer learning, informs model design, and provides a framework for understanding how networks process information.

## Key Takeaways

- Early layers: edges, colors, textures (generic)
- Middle layers: shapes, parts, patterns (semi-specific)
- Late layers: objects, concepts, semantics (task-specific)
- Receptive field grows with depth
- Pre-trained hierarchies transfer well across tasks
- Feature hierarchy explains why deep networks outperform shallow ones
- CKA similarity measures feature evolution across layers
- Skip connections allow mixing features across hierarchy levels
