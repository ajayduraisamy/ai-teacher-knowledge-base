# Concept: Normalization for Images

## Concept ID

DL-220

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the purpose and mathematics of image normalization
- Implement normalization in PyTorch
- Compute dataset mean and standard deviation
- Analyze how normalization affects training dynamics

## Prerequisites

DL-219 Preprocessing for Images, DL-217 Image Classification Pipeline

## Definition

Image normalization is the process of standardizing pixel values to have a fixed mean and variance, typically zero mean and unit variance per channel, which stabilizes training by ensuring consistent input statistics across the dataset.

## Intuition

Neural networks learn best when inputs have similar statistical properties. Natural images have pixel values ranging from 0-255, with uneven distributions (e.g., the green channel tends to be brighter in natural scenes). Without normalization, some input values are systematically larger, causing certain weights to receive larger updates. Normalization centers the data (subtracting mean) and scales it to unit variance, so every channel has the same dynamic range. This is like measuring everyone's height in standard deviations from the mean rather than in absolute inches.

## Why This Concept Matters

Normalization is one of the simplest yet most important preprocessing steps. It accelerates convergence, improves stability, and is required for many pretrained models. Using wrong normalization statistics can significantly degrade performance.

## Mathematical Explanation

**Per-channel normalization**:
$$x_{c,i,j}' = \frac{x_{c,i,j} - \mu_c}{\sigma_c}$$

Where $\mu_c$ and $\sigma_c$ are the mean and standard deviation of channel $c$ over the entire training dataset.

**Computing dataset statistics**:
$$\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{n,c,i,j}$$
$$\sigma_c = \sqrt{\frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{i=1}^{H} \sum_{j=1}^{W} (x_{n,c,i,j} - \mu_c)^2}$$

**Common normalization values**:

ImageNet (RGB, in [0,1] after ToTensor):
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

CIFAR-10 (RGB):
- Mean: [0.4914, 0.4822, 0.4465]
- Std: [0.2470, 0.2435, 0.2616]

**Min-max normalization** (alternative):
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

## Code Examples

### Example 1: Computing Dataset Statistics

```python
import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

# Compute mean and std of CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

# Compute mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
total_pixels = 0

for image, _ in dataset:
    # image shape: (3, H, W)
    mean += image.sum(dim=(1, 2))
    total_pixels += image.shape[1] * image.shape[2]

mean /= total_pixels

# Compute std
for image, _ in dataset:
    std += ((image - mean[:, None, None]) ** 2).sum(dim=(1, 2))

std = torch.sqrt(std / total_pixels)

print(f"Computed CIFAR-10 statistics:")
print(f"  Mean: {mean.tolist()}")
print(f"  Std: {std.tolist()}")
```

### Example 2: Effect of Normalization

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Create data with different scales
N = 500
X_unscaled = torch.randn(N, 3, 32, 32) * torch.tensor([100, 50, 20]).view(1, 3, 1, 1)
X_unscaled += torch.tensor([50, 100, 150]).view(1, 3, 1, 1)

# Normalized version
mean = X_unscaled.mean(dim=(0, 2, 3))
std = X_unscaled.std(dim=(0, 2, 3))
X_normalized = (X_unscaled - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

y = torch.randint(0, 10, (N,))

print(f"Unscaled - Mean: {X_unscaled.mean(dim=(0,2,3)).tolist()}")
print(f"Unscaled - Std: {X_unscaled.std(dim=(0,2,3)).tolist()}")
print(f"Normalized - Mean: {X_normalized.mean(dim=(0,2,3)).tolist():.4f}")
print(f"Normalized - Std: {X_normalized.std(dim=(0,2,3)).tolist():.4f}")

# Train both
def train_model(X, y, name):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(50):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    
    return losses

losses_unscaled = train_model(X_unscaled, y, "Unscaled")
losses_norm = train_model(X_normalized, y, "Normalized")

print(f"\nFinal loss (unscaled): {losses_unscaled[-1]:.4f}")
print(f"Final loss (normalized): {losses_norm[-1]:.4f}")
```

### Example 3: Per-Channel Normalization Visualization

```python
import torch
import numpy as np

torch.manual_seed(42)

# Demonstrate per-channel normalization
random_image = torch.rand(3, 32, 32) * 255
print(f"Raw image - R: mean={random_image[0].mean():.1f}, "
      f"G: mean={random_image[1].mean():.1f}, "
      f"B: mean={random_image[2].mean():.1f}")

# Normalize with ImageNet stats
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Simulate 0-255 to 0-1 conversion
img_01 = random_image / 255.0
normalized = (img_01 - mean) / std

print(f"\nAfter normalization:")
print(f"  R: mean={normalized[0].mean():.3f}, std={normalized[0].std():.3f}")
print(f"  G: mean={normalized[1].mean():.3f}, std={normalized[1].std():.3f}")
print(f"  B: mean={normalized[2].mean():.3f}, std={normalized[2].std():.3f}")

# Demonstrate inversion
reconstructed = normalized * std + mean
reconstructed = reconstructed * 255
print(f"\nReconstruction error: {(reconstructed - random_image).abs().max():.6f}")
```

## Common Mistakes

1. **Wrong normalization values**: Using CIFAR norms on ImageNet or vice versa.
2. **Normalizing before ToTensor**: ToTensor already scales to [0,1]; normalizing raw [0,255] values with ImageNet norms is wrong.
3. **Computing statistics incorrectly**: Must use training set only, not validation/test.
4. **Channel order mismatch**: Some libraries use BGR instead of RGB.
5. **Forgetting to normalize during inference**: Models won't work correctly without same normalization.

## Interview Questions

### Beginner - 5
1. Why do we normalize images?
2. What are the ImageNet normalization values?
3. How do you compute dataset mean and std?
4. What is per-channel normalization?
5. What happens if you use wrong normalization values?

### Intermediate - 5
1. Derive the formula for dataset standard deviation in one pass.
2. How does normalization affect gradient updates?
3. Compare image normalization with batch normalization.
4. Why does normalization accelerate convergence?
5. How do you handle normalization for grayscale images?

### Advanced - 3
1. Design an adaptive normalization strategy.
2. Analyze the effect of normalization on the loss landscape.
3. Compare standard normalization with whitening (PCA-based).

## Practice Problems

### Easy - 5
1. Compute mean and std of a small image dataset.
2. Apply ImageNet normalization to a batch of images.
3. Reconstruct original values from normalized ones.
4. Compare normalized vs unnormalized training curves.
5. Implement min-max normalization.

### Medium - 5
1. Compute dataset statistics efficiently with DataLoader.
2. Compare training with different normalization strategies.
3. Implement instance normalization for per-image normalization.
4. Analyze how normalization affects weight gradients.
5. Implement normalization that adapts to batch statistics.

### Hard - 3
1. Implement channel-wise whitening (decorrelation).
2. Design a learnable normalization layer.
3. Analyze the spectral properties of normalized vs unnormalized features.

## Solutions

### Easy - 1 Solution
```python
def compute_stats(dataloader):
    mean = 0.0
    std = 0.0
    n = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(dim=2).sum(dim=0)
        n += batch_samples
    mean /= n
    return mean
```

## Related Concepts

DL-219 Preprocessing for Images, DL-132 Batch Normalization, DL-221 ImageDataGenerator

## Next Concepts

DL-221 ImageDataGenerator

## Summary

Image normalization standardizes pixel values to have consistent statistical properties (typically zero mean, unit variance per channel). This accelerates training, improves stability, and is required when using pretrained models. Proper normalization is a simple but critical preprocessing step.

## Key Takeaways

- Normalization: (x - mean) / std, per channel
- Standard ImageNet values: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Must compute statistics only from training data
- ToTensor scales [0,255] to [0,1]; normalize after ToTensor
- Proper normalization accelerates convergence significantly
- Wrong normalization degrades model performance
- Different datasets have different optimal normalization values
- Normalization is applied identically at train and test time
- Enables consistent gradient scales across channels
- Foundation for effective training of deep networks
