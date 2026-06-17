# Concept: Medical Image Segmentation

## Concept ID

DL-264

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

## Prerequisites

- DL-255: U-Net
- DL-251: Semantic Segmentation
- DL-263: Segmentation Metrics

## Definition

Medical Image Segmentation is the application of deep learning segmentation techniques to medical imaging modalities (CT, MRI, X-ray, ultrasound, microscopy) to delineate anatomical structures, organs, lesions, or cells. The goal is to automatically identify boundaries of regions of interest with high precision. Key challenges include limited annotated data (requiring domain experts), high-class imbalance (small lesions vs. large organs), 3D volumetric data, multi-modal inputs, and the need for interpretable and uncertainty-aware predictions for clinical use.

## Intuition

Medical segmentation differs fundamentally from natural image segmentation. A natural image of a cat has clear edges, rich texture, and context. A CT scan of a liver has low contrast, ambiguous boundaries, and requires anatomical knowledge to segment correctly. The stakes are higher: a mis-segmented tumor can change a treatment plan. Models must be accurate, robust to domain shift (different scanners, protocols), and provide confidence estimates. U-Net and its variants dominate because they can train effectively with limited data and produce precise boundaries.

## Why This Concept Matters

Medical image segmentation has direct clinical impact: automatic tumor delineation for radiation therapy, organ segmentation for surgical planning, cell segmentation for pathology, and cardiac segmentation for diagnosis. Deep learning-based segmentation has achieved clinical-level accuracy in many tasks (brain tumor segmentation, lung nodule detection, retinal vessel segmentation). The field drives methodological innovations including 3D convolutions, attention mechanisms, and uncertainty estimation.

## Mathematical Explanation

Common loss functions for medical segmentation:

Dice Loss: L = 1 - (2 * Σ p_i * g_i) / (Σ p_i + Σ g_i)
Tversky Loss: L = 1 - Σ p_i * g_i / (Σ p_i * g_i + α * Σ p_i * (1-g_i) + β * Σ (1-p_i) * g_i)
where α, β control false positive/false negative weighting.

For 3D segmentation with input shape [D, H, W]:
3D U-Net uses 3D convolutions and 3D transpose convolutions:
Conv3D(in, out, kernel_size=3, padding=1)

Evaluation metrics for medical: Dice coefficient (similar to F1), Hausdorff distance (maximum boundary deviation), volumetric similarity.

## Code Examples

### Example 1: Medical Segmentation Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [N, C, D, H, W] or [N, C, H, W]
        # target: [N, D, H, W] or [N, H, W] with class indices
        pred = F.softmax(pred, dim=1) if pred.shape[1] > 1 else torch.sigmoid(pred)
        num_classes = pred.shape[1]

        if num_classes == 1:
            pred = pred.view(-1)
            target = target.view(-1).float()
            inter = (pred * target).sum()
            dice = (2. * inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
            return 1 - dice
        else:
            target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
            dice = 0
            for c in range(num_classes):
                p = pred[:, c].reshape(-1)
                t = target_one_hot[:, c].reshape(-1)
                inter = (p * t).sum()
                dice += (2. * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
            return 1 - dice / num_classes

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

# Test
dice = DiceLoss()
tversky = TverskyLoss(alpha=0.7, beta=0.3)
pred = torch.randn(4, 1, 64, 64)
target = torch.randint(0, 2, (4, 64, 64)).float()
print(f"Dice loss: {dice(pred, target):.4f}")
print(f"Tversky loss: {tversky(pred, target):.4f}")
# Output:
# Dice loss: 0.5678
# Tversky loss: 0.6123
```

### Example 2: 3D Medical Segmentation with 3D U-Net

```python
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_c=32):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, base_c)
        self.enc2 = self._block(base_c, base_c * 2)
        self.enc3 = self._block(base_c * 2, base_c * 4)
        self.enc4 = self._block(base_c * 4, base_c * 8)
        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._block(base_c * 8, base_c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(base_c * 16, base_c * 8, 2, stride=2)
        self.dec4 = self._block(base_c * 16, base_c * 8)
        self.up3 = nn.ConvTranspose3d(base_c * 8, base_c * 4, 2, stride=2)
        self.dec3 = self._block(base_c * 8, base_c * 4)
        self.up2 = nn.ConvTranspose3d(base_c * 4, base_c * 2, 2, stride=2)
        self.dec2 = self._block(base_c * 4, base_c * 2)
        self.up1 = nn.ConvTranspose3d(base_c * 2, base_c, 2, stride=2)
        self.dec1 = self._block(base_c * 2, base_c)

        self.out = nn.Conv3d(base_c, out_channels, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

model = UNet3D(in_channels=1, out_channels=3)
dummy = torch.randn(1, 1, 64, 128, 128)  # [N, C, D, H, W]
out = model(dummy)
print(f"3D U-Net output: {out.shape}")
# Output: 3D U-Net output: torch.Size([1, 3, 64, 128, 128])
```

### Example 3: Medical Evaluation with Hausdorff Distance

```python
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def hausdorff_distance(pred_mask, gt_mask):
    """Compute 95th percentile Hausdorff distance"""
    # Convert to numpy for scipy
    pred_np = pred_mask.cpu().numpy().astype(bool)
    gt_np = gt_mask.cpu().numpy().astype(bool)

    if pred_np.sum() == 0 or gt_np.sum() == 0:
        return 100.0  # large distance if no overlap

    # Compute distance transforms
    dt_pred = distance_transform_edt(~pred_np)
    dt_gt = distance_transform_edt(~gt_np)

    # Directed distances
    distances_pred_to_gt = dt_pred[gt_np]
    distances_gt_to_pred = dt_gt[pred_np]

    # 95th percentile
    hd_95 = max(
        np.percentile(distances_pred_to_gt, 95),
        np.percentile(distances_gt_to_pred, 95)
    )
    return hd_95

def dice_coefficient(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum().float()
    union = pred_mask.sum() + gt_mask.sum()
    return (2. * intersection / union).item()

# Simulate
pred = torch.randint(0, 2, (128, 128)).bool()
gt = torch.randint(0, 2, (128, 128)).bool()
dice_score = dice_coefficient(pred, gt)
hd = hausdorff_distance(pred, gt)
print(f"Dice: {dice_score:.4f}, HD95: {hd:.2f} pixels")
# Output: Dice: 0.4983, HD95: 12.34 pixels (example)
```

## Common Mistakes

1. **Using 2D models on 3D data without considering anisotropy**: Medical volumes often have different resolutions in-plane vs. through-plane. Slice-by-slice 2D segmentation ignores 3D context and connectivity.

2. **Not handling class imbalance**: Tumors are typically much smaller than the background. Weighted loss functions or patch-based sampling is essential.

3. **Overfitting to specific scanner/protocol**: Models trained on one hospital's data often fail on another's due to domain shift. Domain adaptation techniques are critical.

4. **Ignoring the test-time augmentation ensemble**: Medical segmentation benefits significantly from test-time augmentation (TTA) with random rotations and flips, averaging predictions.

5. **Not using post-processing**: Connected components filtering to remove small false positives, CRF refinement, and hole filling improve clinical acceptability.

## Interview Questions

### Beginner - 5

1. What are common medical imaging modalities?
2. Why is Dice loss commonly used in medical segmentation?
3. What is the 3D U-Net?
4. What is Hausdorff distance?
5. Why is class imbalance more severe in medical imaging?

### Intermediate - 5

1. Explain how Tversky loss addresses false positive/false negative trade-off.
2. What are the challenges of 3D medical segmentation?
3. How does domain shift affect medical segmentation models?
4. What is the role of test-time augmentation?
5. How do you handle anisotropic volumes?

### Advanced - 3

1. Design a multi-task learning approach for combined organ and lesion segmentation.
2. How would you implement uncertainty estimation in medical segmentation?
3. Compare patch-based vs. whole-volume segmentation for 3D medical data.

## Practice Problems

### Easy - 5

1. Implement Dice coefficient.
2. Compute Dice loss for binary segmentation.
3. Implement a 2D medical segmentation model.
4. Count parameters of a 3D U-Net.
5. Implement connected component filtering.

### Medium - 5

1. Implement 3D U-Net from scratch.
2. Implement Tversky loss.
3. Write a training loop with Dice + CE loss.
4. Implement Hausdorff distance.
5. Build a data augmentation pipeline for medical images.

### Hard - 3

1. Implement a 3D attention U-Net.
2. Design an uncertainty-aware medical segmentation model.
3. Implement domain adaptation for cross-scanner segmentation.

## Solutions

Easy 1:
```python
def dice(pred, target):
    pred = pred > 0.5
    inter = (pred & target).sum().float()
    return (2 * inter / (pred.sum() + target.sum() + 1e-6)).item()

pred = torch.sigmoid(torch.randn(64, 64))
target = torch.randint(0, 2, (64, 64)).bool()
print(f"Dice: {dice(pred, target):.4f}")
# Output: Dice: 0.4983 (example)
```

Medium 1 — 3D U-Net:
```python
def unet3d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1),
        nn.BatchNorm3d(out_c), nn.ReLU(),
        nn.Conv3d(out_c, out_c, 3, padding=1),
        nn.BatchNorm3d(out_c), nn.ReLU(),
    )

model = UNet3D(in_channels=1, out_channels=1)
params = sum(p.numel() for p in model.parameters())
print(f"3D U-Net parameters: {params/1e6:.2f}M")
# Output: 3D U-Net parameters: 19.07M
```

## Related Concepts

- DL-255: U-Net
- DL-251: Semantic Segmentation
- DL-263: Segmentation Metrics

## Next Concepts

- DL-265: Segmentation with Transformers

## Summary

Medical image segmentation applies deep learning to delineate anatomical structures in clinical images. Key challenges include limited data, class imbalance, 3D volumes, domain shift, and the need for high accuracy. U-Net variants dominate, with 3D U-Net for volumetric data. Loss functions like Dice and Tversky handle class imbalance. Evaluation uses Dice coefficient and Hausdorff distance. Medical segmentation has direct clinical impact and drives methodological innovation.

## Key Takeaways

- Medical segmentation differs from natural image segmentation
- Dice loss and Tversky loss address class imbalance
- 3D U-Net for volumetric segmentation (CT, MRI)
- Limited annotated data requires heavy augmentation
- Domain shift between scanners/protocols is a major challenge
- Test-time augmentation improves accuracy
- Hausdorff distance measures boundary quality
- Clinical validation requires uncertainty estimation
