# Concept: Semantic Segmentation

## Concept ID

DL-251

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

- Understand the semantic segmentation task and its formulation as pixel-wise classification
- Implement a basic FCN for semantic segmentation
- Comprehend evaluation metrics including mIoU and pixel accuracy
- Identify the challenges specific to semantic segmentation

## Prerequisites

- DL-201: Convolutional Neural Networks
- DL-221: Image Classification
- Basic understanding of upsampling and downsampling

## Definition

Semantic Segmentation is the task of assigning a class label to every pixel in an image. Unlike object detection which outputs bounding boxes, semantic segmentation produces a dense prediction map where each pixel is classified into a predefined category (e.g., road, building, person, sky). The output is typically a 2D array of class indices with the same spatial dimensions as the input image. Semantic segmentation does not distinguish between instances of the same class (all "car" pixels are treated identically regardless of which car they belong to).

## Intuition

Think of semantic segmentation as coloring an image by object category: every pixel of the sky gets blue, every pixel of a person gets red, etc. This requires the model to understand both global context (what objects are present) and local details (precise object boundaries). Unlike classification which looks at the whole image, or detection which looks at rectangular regions, segmentation requires per-pixel reasoning. This demands an architecture that preserves spatial resolution while having sufficient receptive field to understand context.

## Why This Concept Matters

Semantic segmentation is critical for applications requiring fine-grained scene understanding: autonomous driving (delineating road, vehicles, pedestrians), medical image analysis (segmenting organs, tumors, cells), satellite imagery analysis (land cover classification), and augmented reality (understanding scene geometry). Advances in segmentation directly impact these application domains. The fully convolutional architecture pioneered for segmentation also influenced other dense prediction tasks including depth estimation and optical flow.

## Mathematical Explanation

Semantic segmentation outputs a probability distribution over C classes for each pixel:
P(y_ij = c | X) for each pixel (i, j) and class c ∈ {1, ..., C}

The loss function is typically pixel-wise cross-entropy:
L = -1/(H*W) * Σ_{i,j} Σ_c y_ij^c * log(p_ij^c)

where y_ij^c is 1 if pixel (i,j) belongs to class c, 0 otherwise, and p_ij^c is the predicted probability.

Evaluation metrics:
- Pixel Accuracy: (correct pixels) / (total pixels)
- Mean IoU (mIoU): (1/C) * Σ_k IoU_k
  where IoU_k = TP_k / (TP_k + FP_k + FN_k)
- Frequency-weighted IoU: weights each class by its frequency

## Code Examples

### Example 1: Simple FCN Segmentation Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.classifier = nn.Conv2d(32, num_classes, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x = self.pool(x1)
        x2 = self.relu(self.conv2(x))
        x = self.pool(x2)
        x3 = self.relu(self.conv3(x))
        x = self.pool(x3)
        x4 = self.relu(self.conv4(x))
        x = self.pool(x4)

        # Decoder
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.relu(self.upconv3(x))
        x = self.relu(self.upconv4(x))

        return self.classifier(x)

model = SimpleSegmentationModel(num_classes=21)
dummy = torch.randn(1, 3, 256, 256)
output = model(dummy)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([1, 21, 256, 256])
```

### Example 2: Computing mIoU

```python
import torch

def compute_miou(pred, target, num_classes=21):
    # pred: [N, H, W] class indices
    # target: [N, H, W] class indices
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))
    # Mean over non-nan classes
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    return sum(valid_ious) / len(valid_ious)

# Simulate predictions and ground truth
pred = torch.randint(0, 21, (2, 256, 256))
target = torch.randint(0, 21, (2, 256, 256))
miou = compute_miou(pred, target)
print(f"Mean IoU: {miou:.4f}")
# Output: Mean IoU: 0.0512 (example)

# Pixel accuracy
correct = (pred == target).sum().item()
total = pred.numel()
pixel_acc = correct / total
print(f"Pixel Accuracy: {pixel_acc:.4f}")
# Output: Pixel Accuracy: 0.0504 (example)
```

### Example 3: Training Loop Snippet

```python
import torch
import torch.nn.functional as F

def train_epoch(model, dataloader, optimizer, device, num_classes=21):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)  # [N, H, W] with class indices

        # Forward
        logits = model(images)  # [N, num_classes, H, W]

        # Loss: pixel-wise cross-entropy
        loss = F.cross_entropy(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, num_classes=21):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            total_iou += compute_miou(pred, targets, num_classes)
    return total_iou / len(dataloader)

print("Training and evaluation functions defined")
# Output: Training and evaluation functions defined
```

## Common Mistakes

1. **Up-sampling directly to full resolution**: Using a single transpose convolution to go from low-res to full-res creates checkerboard artifacts. Use progressive upsampling with skip connections.

2. **Ignoring class imbalance**: In natural images, some classes (e.g., road) dominate others (e.g., traffic light). Use weighted cross-entropy or Dice loss to handle imbalance.

3. **Using same loss weight for boundary pixels**: Boundary pixels between classes are harder to classify. Some methods use boundary-aware loss weighting.

4. **Assuming fixed input size**: Most segmentation models require fixed-size inputs due to fully connected layers or fixed upsampling ratios. Use fully convolutional designs for arbitrary input sizes.

5. **Evaluating mIoU without ignoring void/unlabeled pixels**: Many datasets have a void/unlabeled class (label 255). These must be excluded from evaluation to avoid misleading results.

## Interview Questions

### Beginner - 5

1. What is semantic segmentation?
2. How is semantic segmentation different from object detection?
3. What is pixel-wise cross-entropy loss?
4. What is mIoU and how is it computed?
5. What is the output shape of a segmentation model?

### Intermediate - 5

1. Explain the encoder-decoder architecture for segmentation.
2. What is the role of skip connections in segmentation?
3. How does class imbalance affect segmentation training?
4. What is the difference between upsampling and transpose convolution?
5. How do you evaluate a segmentation model?

### Advanced - 3

1. Compare semantic segmentation with instance segmentation and panoptic segmentation.
2. How do receptive field size and spatial resolution trade off in segmentation?
3. Design a loss function that emphasizes boundary regions.

## Practice Problems

### Easy - 5

1. Implement pixel-wise cross-entropy loss.
2. Compute pixel accuracy between two segmentation maps.
3. Implement a function to visualize a segmentation mask as a color image.
4. Create a simple FCN with 4 encoder and 4 decoder layers.
5. Count the number of parameters in a segmentation model.

### Medium - 5

1. Implement mIoU with void class handling.
2. Build a weighted cross-entropy loss for class imbalance.
3. Implement skip connections between encoder and decoder.
4. Write an evaluation loop that computes per-class IoU.
5. Implement data augmentation for segmentation (random crop, flip, color jitter).

### Hard - 3

1. Implement Dice loss and combine with cross-entropy.
2. Build a multiscale segmentation model with feature pyramid.
3. Implement boundary-aware loss for improved edge quality.

## Solutions

Easy 1:
```python
def pixel_ce_loss(pred, target, num_classes=21):
    # pred: [N, C, H, W], target: [N, H, W]
    return F.cross_entropy(pred, target)

pred = torch.randn(2, 21, 256, 256)
target = torch.randint(0, 21, (2, 256, 256))
loss = pixel_ce_loss(pred, target)
print(f"Pixel CE loss: {loss.item():.4f}")
# Output: Pixel CE loss: 3.0440 (example)
```

Medium 1 — mIoU with void:
```python
def miou_with_void(pred, target, num_classes=20, void_label=255):
    pred = pred.clone()
    target = target.clone()
    valid = target != void_label
    pred = pred[valid]
    target = target[valid]
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return sum(ious) / len(ious) if ious else 0.0

pred = torch.randint(0, 20, (1000,))
target = torch.randint(0, 20, (1000,))
target[:100] = 255  # void pixels
print(f"mIoU with void: {miou_with_void(pred, target):.4f}")
# Output: mIoU with void: 0.0571 (example)
```

## Related Concepts

- DL-252: Instance Segmentation
- DL-253: Panoptic Segmentation
- DL-254: FCN
- DL-255: U-Net

## Next Concepts

- DL-252: Instance Segmentation
- DL-253: Panoptic Segmentation
- DL-254: FCN

## Summary

Semantic segmentation assigns a class label to every pixel, producing dense scene understanding. Encoder-decoder architectures downsample for context and upsample for detail, with skip connections preserving spatial information. Evaluation uses mIoU and pixel accuracy. Challenges include class imbalance, boundary precision, and resolution preservation. Segmentation is foundational for autonomous driving, medical imaging, and scene understanding applications.

## Key Takeaways

- Pixel-wise classification: every pixel gets a class label
- Encoder-decoder architecture downsamples then upsamples
- Skip connections preserve spatial details
- mIoU is the standard evaluation metric
- Class imbalance requires weighted loss functions
- Transpose convolution for learned upsampling
- Fully convolutional design accepts arbitrary input sizes
- Foundation for instance and panoptic segmentation
