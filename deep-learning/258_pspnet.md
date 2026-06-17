# Concept: PSPNet (Pyramid Scene Parsing Network)

## Concept ID

DL-258

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Segmentation

## Learning Objectives

## Prerequisites

- DL-251: Semantic Segmentation
- DL-254: FCN
- Understanding of spatial pooling

## Definition

PSPNet (Pyramid Scene Parsing Network), introduced by Zhao et al. in 2017, is a semantic segmentation architecture that incorporates global contextual information through a Pyramid Pooling Module (PPM). The PPM applies adaptive average pooling at four different scales, followed by 1×1 convolutions and upsampling, to aggregate context from sub-regions of varying sizes. This multi-scale pooling captures information from the whole image down to small regions, enabling the model to understand scene-level context. PSPNet achieved 85.4% mIoU on PASCAL VOC 2012 and won the ImageNet Scene Parsing Challenge 2016.

## Intuition

Standard FCNs struggle with context: a boat on water can be confused with a car on road if the model only looks at local pixels. PSPNet addresses this by explicitly pooling features at multiple scales to capture global scene context. The pyramid pooling module pools the feature map into grids of sizes 1×1 (global), 2×2, 3×3, and 6×6. The 1×1 pooling gives a scene-level descriptor (e.g., "this is a city scene"), the 6×6 pooling captures finer details (e.g., "this region has cobblestone texture"). All scales are combined to inform the final pixel classification.

## Why This Concept Matters

PSPNet demonstrated the critical importance of global context for semantic segmentation, especially for scene understanding tasks with complex inter-object relationships. The pyramid pooling module became a standard component in segmentation architectures, inspiring similar designs in DeepLab's ASPP and other models. PSPNet's insight that different sub-regions provide different levels of context (from global scene to local texture) is fundamental to scene parsing.

## Mathematical Explanation

Pyramid Pooling Module (PPM):
Given input feature map F with spatial size H×W, the PPM applies:

1. Adaptive average pooling to K×K bins: F_k = Pool_k(F) for k ∈ {1, 2, 3, 6}
2. 1×1 convolution to reduce channels to 1/N of original (N=4 levels)
3. Upsample back to H×W via bilinear interpolation
4. Concatenate all upsampled features with the original feature map

Output: Concat(F_1, F_2, F_3, F_6, F) with channels C + C/N × 4

The PSPNet backbone is typically a dilated ResNet with output stride 8.

Loss: Auxiliary loss on the res4b22 feature (before final PPM) for better gradient flow, weighted at 0.4 of the main loss.

## Code Examples

### Example 1: Pyramid Pooling Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels=2048, out_channels=512, bin_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.bin_sizes = bin_sizes
        inter_channels = in_channels // len(bin_sizes)

        for size in bin_sizes:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
            ))

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels * len(bin_sizes), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        pooled = []
        for branch in self.branches:
            pooled.append(F.interpolate(branch(x), size=size, mode='bilinear', align_corners=False))
        x = torch.cat([x] + pooled, dim=1)
        return self.fuse(x)

ppm = PyramidPoolingModule(in_channels=2048, out_channels=512)
dummy = torch.randn(1, 2048, 32, 32)
out = ppm(dummy)
print(f"PPM output: {out.shape}")
# Output: PPM output: torch.Size([1, 512, 32, 32])
```

### Example 2: PSPNet Architecture

```python
import torch
import torch.nn as nn

class PSPNet(nn.Module):
    def __init__(self, num_classes=21, backbone_c=2048):
        super().__init__()
        # Simplified backbone (replaces ResNet)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(1024, backbone_c, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(backbone_c, backbone_c, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(backbone_c, backbone_c, 3, padding=8, dilation=8), nn.ReLU(),
        )

        self.ppm = PyramidPoolingModule(in_channels=backbone_c, out_channels=512)

        self.cls = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        # Auxiliary loss branch
        self.aux_cls = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone(x)
        x = self.ppm(x)
        x = self.cls(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

model = PSPNet(num_classes=21)
dummy = torch.randn(1, 3, 512, 512)
out = model(dummy)
print(f"PSPNet output: {out.shape}")
# Output: PSPNet output: torch.Size([1, 21, 512, 512])
```

### Example 3: PSPNet Auxiliary Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSPNetWithAux(nn.Module):
    def __init__(self, num_classes=21, aux_weight=0.4):
        super().__init__()
        self.aux_weight = aux_weight
        # Same as PSPNet but with auxiliary branch
        
    def forward_loss(self, x, target):
        """Compute main + auxiliary loss"""
        main_out, aux_out = self(x)  # returns both
        main_loss = F.cross_entropy(main_out, target)
        aux_loss = F.cross_entropy(aux_out, target)
        return main_loss + self.aux_weight * aux_loss

# Simulate auxiliary loss computation
def compute_psp_loss(main_logits, aux_logits, target, aux_weight=0.4):
    main_loss = F.cross_entropy(main_logits, target)
    aux_loss = F.cross_entropy(aux_logits, target)
    total_loss = main_loss + aux_weight * aux_loss
    return total_loss, main_loss.item(), aux_loss.item()

main_logits = torch.randn(2, 21, 128, 128)
aux_logits = torch.randn(2, 21, 128, 128)
target = torch.randint(0, 21, (2, 128, 128))
total, main_l, aux_l = compute_psp_loss(main_logits, aux_logits, target)
print(f"Total: {total:.4f}, Main: {main_l:.4f}, Aux: {aux_l:.4f}")
# Output: Total: 5.1234, Main: 3.4678, Aux: 4.1390 (example)
```

## Common Mistakes

1. **Using the same channel reduction for all pyramid levels**: The standard PPM divides input channels by the number of pyramid levels. Very small bin sizes (1×1) may not benefit from large channel capacities.

2. **Forgetting to upsample after pooling**: Pooling reduces spatial dimensions. All pyramid features must be upsampled back to the original feature map size before concatenation.

3. **Auxiliary loss branch implementation**: The auxiliary loss is applied to an intermediate layer (res4b22 in ResNet), not the final features. Ensure the auxiliary branch operates on the correct feature level.

4. **Fixed bin sizes across datasets**: The optimal bin sizes depend on the dataset's spatial statistics. For datasets with very different resolutions, adjust bin sizes accordingly.

5. **Over-reliance on global context**: While PPM captures global context, very small objects may be overwhelmed by the dominant global signal. Use PPM in combination with high-resolution skip connections.

## Interview Questions

### Beginner - 5

1. What does PSPNet stand for?
2. What is the main innovation of PSPNet?
3. How many pyramid levels does the PPM use?
4. What bin sizes does the standard PPM use?
5. What is the purpose of the auxiliary loss?

### Intermediate - 5

1. Explain how the Pyramid Pooling Module works.
2. How does multi-scale pooling improve segmentation?
3. What is the role of the auxiliary loss in PSPNet?
4. How does PSPNet compare with DeepLab's ASPP?
5. How does dilated ResNet backbone help PSPNet?

### Advanced - 3

1. Analyze the information captured at different pyramid scales in PPM.
2. Compare PSPNet's global context aggregation with DeepLab's ASPP—what are the trade-offs?
3. How would you adapt the PPM for real-time segmentation?

## Practice Problems

### Easy - 5

1. Implement adaptive average pooling to (1,1) output.
2. Compute output channels of PPM after concatenation.
3. Upsample a 4x4 feature map to 32x32 using bilinear interpolation.
4. Count the number of pyramid branches in standard PPM.
5. Implement a 1x1 convolution for channel reduction.

### Medium - 5

1. Implement the full Pyramid Pooling Module.
2. Build the PSPNet architecture.
3. Implement the auxiliary loss branch.
4. Write a training loop with main + aux loss.
5. Compare PPM vs. global average pooling.

### Hard - 3

1. Implement PSPNet with ResNet backbone and dilated convolutions.
2. Design an adaptive PPM that learns bin sizes.
3. Implement a multi-scale PSPNet with feature pyramid.

## Solutions

Easy 1:
```python
x = torch.randn(1, 512, 32, 32)
pooled = F.adaptive_avg_pool2d(x, (1, 1))
print(f"Global pooling output: {pooled.shape}")
# Output: Global pooling output: torch.Size([1, 512, 1, 1])
```

Medium 1 — PPM:
```python
class PPM(nn.Module):
    def __init__(self, in_c=2048, out_c=512, bins=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList()
        for b in bins:
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_c, out_c // len(bins), 1),
                nn.BatchNorm2d(out_c // len(bins)),
                nn.ReLU()
            ))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_c + out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        H, W = x.shape[2:]
        feat = [F.interpolate(b(x), (H, W), mode='bilinear', align_corners=False) for b in self.branches]
        return self.fuse(torch.cat([x] + feat, 1))

ppm = PPM(2048, 512)
print(f"PPM: {ppm(torch.randn(1, 2048, 32, 32)).shape}")
# Output: PPM: torch.Size([1, 512, 32, 32])
```

## Related Concepts

- DL-257: DeepLab
- DL-251: Semantic Segmentation
- DL-254: FCN

## Next Concepts

- DL-259: SegNet
- DL-263: Segmentation Metrics

## Summary

PSPNet introduced the Pyramid Pooling Module to aggregate global context at multiple scales for semantic segmentation. By pooling features into grids of varying granularity, the model captures information from whole-scene to local texture. The auxiliary loss on intermediate features improves gradient flow and convergence. PSPNet demonstrated that global scene understanding is critical for accurate pixel classification, especially in complex scene parsing tasks.

## Key Takeaways

- Pyramid Pooling Module captures context at 4 scales (1, 2, 3, 6)
- Multi-scale pooling aggregated via concatenation and fusion
- Dilated ResNet backbone maintains resolution (output stride 8)
- Auxiliary loss on intermediate features at weight 0.4
- 85.4% mIoU on PASCAL VOC 2012
- Global scene context prevents confusion between similar local patterns
- PPM inspired by spatial pyramid matching in traditional computer vision
- Fuse global and local features for optimal segmentation
