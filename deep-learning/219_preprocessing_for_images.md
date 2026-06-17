# Concept: Preprocessing for Images

## Concept ID

DL-219

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand common image preprocessing techniques
- Implement preprocessing pipelines in PyTorch
- Analyze how preprocessing affects model performance
- Design appropriate preprocessing for different tasks

## Prerequisites

DL-217 Image Classification Pipeline, DL-220 Normalization for Images

## Definition

Image preprocessing is the set of transformations applied to raw images before they are fed into a neural network, including resizing, cropping, color space conversion, normalization, and data augmentation.

## Intuition

Neural networks don't see the world the way humans do. A raw image is just a grid of pixel values (0-255) to the computer. Preprocessing transforms these raw values into a format the network can learn from effectively: resizing to a consistent size, normalizing pixel values to a standard range, and augmenting to create more training examples. It's like preparing ingredients before cooking — the quality of the preprocessing directly affects the quality of the final result.

## Why This Concept Matters

Proper preprocessing is essential for model performance. Using the wrong preprocessing can reduce accuracy by 10-20%. Models trained with one preprocessing pipeline may fail when deployed with different preprocessing. Understanding preprocessing is crucial for both training and deployment.

## Mathematical Explanation

**Common preprocessing operations**:

1. **Resizing**: Images are resized to a fixed size (e.g., 256x256) using interpolation.
   - Bilinear: weighted average of 4 nearest pixels
   - Bicubic: weighted average of 16 nearest pixels

2. **Center crop**: Crop the center region (e.g., 224x224 from 256x256).

3. **Random crop**: Crop a random region during training for augmentation.

4. **Random horizontal flip**: Flip image horizontally with 50% probability.

5. **Color jitter**: Randomly adjust brightness, contrast, saturation, hue.

6. **To tensor**: Convert PIL image (HWC, 0-255 uint8) to tensor (CHW, 0.0-1.0 float32).

7. **Normalize**: $x' = (x - \mu) / \sigma$ where $\mu$ and $\sigma$ are dataset statistics.

## Code Examples

### Example 1: Basic Preprocessing Pipeline

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

torch.manual_seed(42)

# Standard ImageNet preprocessing for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Standard ImageNet preprocessing for inference
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Test on a random image
random_pil = Image.fromarray(
    (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
)

train_img = train_transform(random_pil)
val_img = val_transform(random_pil)

print(f"After train transform: {train_img.shape}, "
      f"mean={train_img.mean():.3f}")
# Output: After train transform: torch.Size([3, 224, 224]), mean=... (varies)

print(f"After val transform: {val_img.shape}, "
      f"mean={val_img.mean():.3f}")
# Output: After val transform: torch.Size([3, 224, 224]), mean=... (varies)
```

### Example 2: Custom Preprocessing

```python
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

torch.manual_seed(42)

class CustomAugmentation:
    """Custom preprocessing with controllable augmentation."""
    def __init__(self, size=224, train=True):
        self.size = size
        self.train = train
    
    def __call__(self, img):
        # Resize
        if self.train:
            # Random resize with scale variation
            scale = random.uniform(0.08, 1.0)
            new_size = int(self.size * scale ** 0.5)
            img = TF.resize(img, new_size)
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.size, self.size))
            img = TF.crop(img, i, j, h, w)
            # Random flip
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.resize(img, self.size + 32)
            img = TF.center_crop(img, self.size)
        
        # To tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        return img

# Test
random_pil = Image.fromarray(
    (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
)

train_transform = CustomAugmentation(size=224, train=True)
val_transform = CustomAugmentation(size=224, train=False)

train_out = train_transform(random_pil)
val_out = val_transform(random_pil)

print(f"Custom train: {train_out.shape}")
print(f"Custom val: {val_out.shape}")
```

### Example 3: Visualizing Preprocessing Effects

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

torch.manual_seed(42)

# Create test image with pattern
img_array = np.zeros((256, 256, 3), dtype=np.uint8)
img_array[64:192, 64:192] = [255, 0, 0]  # red square
img_array[96:160, 96:160] = [0, 255, 0]  # green inner square
test_image = Image.fromarray(img_array)

# Apply different transforms
transforms_to_try = [
    ("Original", transforms.Resize(224)),
    ("Center crop", transforms.Compose([transforms.Resize(256), 
                                        transforms.CenterCrop(224)])),
    ("Random crop", transforms.Compose([transforms.Resize(256), 
                                        transforms.RandomCrop(224)])),
    ("Horizontal flip", transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=1.0),
    ])),
    ("Color jitter", transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5,
                              saturation=0.5, hue=0.3),
    ])),
]

for name, t in transforms_to_try:
    result = t(test_image)
    # Convert tensor back to numpy for info
    if isinstance(result, torch.Tensor):
        arr = result.permute(1, 2, 0).numpy()
    else:
        arr = np.array(result)
    print(f"{name:20s} -> {arr.shape}, range=[{arr.min():.0f}, {arr.max():.0f}]")
```

## Common Mistakes

1. **Using training transforms in evaluation**: Random crop/flip should not be used during inference.
2. **Wrong normalization statistics**: Must match what the model was trained with.
3. **Incorrect resize order**: For ImageNet: resize to 256, then center crop to 224.
4. **Using ToTensor twice**: ToTensor also scales to [0,1]; don't divide by 255 separately.
5. **Different interpolation methods**: Some pretrained models expect specific interpolation.

## Interview Questions

### Beginner - 5
1. Why do we resize images before feeding them to a CNN?
2. What does ToTensor() do?
3. What is the purpose of normalization?
4. Why use different preprocessing for train vs test?
5. What is the standard ImageNet preprocessing?

### Intermediate - 5
1. Explain random resized crop and why it's effective.
2. How does color jitter affect model robustness?
3. What are the trade-offs of larger input sizes?
4. How do you handle non-square images?
5. What preprocessing is needed for deployment on mobile?

### Advanced - 3
1. Design an automatic preprocessing search system.
2. Analyze how different interpolation methods affect model accuracy.
3. Implement differentiable preprocessing for end-to-end training.

## Practice Problems

### Easy - 5
1. Create transforms that resize to 224x224.
2. Add random horizontal flip to a pipeline.
3. Normalize images with custom mean and std.
4. Convert a PIL image to a PyTorch tensor.
5. Apply center crop to 224 from 256.

### Medium - 5
1. Build a custom augmentation pipeline with random rotation and shear.
2. Compare model accuracy with and without color jitter.
3. Implement the exact ImageNet preprocessing pipeline.
4. Add random erasing / Cutout augmentation.
5. Visualize the effect of different augmentations.

### Hard - 3
1. Implement AutoAugment policy search.
2. Design a preprocessing strategy for videos.
3. Implement a preprocessing pipeline that handles various input sizes dynamically.

## Solutions

### Easy - 1 Solution
```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## Related Concepts

DL-220 Normalization for Images, DL-221 ImageDataGenerator, DL-217 Image Classification Pipeline

## Next Concepts

DL-220 Normalization for Images

## Summary

Image preprocessing transforms raw images into a format suitable for neural network training and inference. Proper preprocessing, including resizing, normalization, and augmentation, is critical for achieving good model performance and generalization.

## Key Takeaways

- Training: random crops, flips, color jitter (augmentation)
- Validation: resize to larger, center crop (deterministic)
- Standard ImageNet: Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize()
- Normalization uses per-channel mean and std from training set
- ToTensor converts (HWC, 0-255 uint8) to (CHW, 0.0-1.0 float32)
- Different models may expect different preprocessing
- Preprocessing affects both accuracy and generalization
- Consistent preprocessing between training and deployment is critical
- Augmentation effectively increases dataset size
- Advanced methods: AutoAugment, RandAugment, CutMix, MixUp
