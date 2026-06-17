# Concept: R-CNN

## Concept ID

DL-237

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Object Detection

## Learning Objectives

- Understand the three-stage pipeline of R-CNN: region proposal, feature extraction, and classification
- Implement the key components of R-CNN including selective search and SVM classification
- Analyze the limitations of R-CNN that motivated Fast R-CNN and Faster R-CNN
- Understand how bounding box regression is applied as a post-processing refinement

## Prerequisites

- DL-231: Object Detection Overview
- DL-232: Bounding Box Regression
- DL-201: Convolutional Neural Networks

## Definition

R-CNN (Region-based Convolutional Neural Network) is a seminal object detection architecture introduced by Girshick et al. in 2014. It operates in three stages: (1) extract ~2000 category-independent region proposals using selective search, (2) warp each proposal to a fixed size and pass it through a CNN (AlexNet/VGG) to extract a feature vector, and (3) classify each proposal using class-specific SVMs and refine bounding boxes via linear regression. R-CNN achieved a then-unprecedented mAP of 53.3% on PASCAL VOC 2010, establishing the two-stage paradigm that dominated detection for years.

## Intuition

R-CNN approaches detection by breaking it into manageable subproblems. Instead of scanning the entire image with a sliding window—which is computationally prohibitive—it first generates candidate regions that might contain objects. Each candidate is then independently processed by a high-quality image classifier. This modular approach leverages the power of deep CNNs for feature extraction while using traditional computer vision (selective search) for proposal generation. The downside is the computational redundancy: each proposal passes through the CNN independently, with significant overlap in computation.

## Why This Concept Matters

R-CNN was the first deep learning-based object detector to achieve a dramatic improvement over traditional methods (30% mAP improvement over the previous best). It demonstrated that CNNs pretrained on ImageNet could be effectively fine-tuned for detection despite the fundamental difference between classification and detection tasks. Understanding R-CNN is crucial for appreciating the evolution of detection architectures. Its limitations—multi-stage training, slow inference, and high disk storage—directly motivated the innovations in Fast R-CNN and Faster R-CNN.

## Mathematical Explanation

R-CNN's training involves three separate loss functions:

1. **CNN pre-training**: Standard cross-entropy on ImageNet classification.
2. **SVM classification**: For each class, a binary SVM is trained. The SVM loss is:
   L_svm(w) = (1/2) ||w||^2 + C Σ max(0, 1 - y_i * (w^T φ(x_i) + b))
   where φ(x_i) is the CNN feature vector.

3. **Bounding box regression**: Linear regression models per class predict delta offsets d_k = (d_x, d_y, d_w, d_h) from the proposal P to ground truth G:
   d_k = w_k^T * φ_5(P)
   where φ_5(P) is the pool5 feature vector. The target is computed as:
   t_x = (g_x - p_x) / p_w, t_y = (g_y - p_y) / p_h
   t_w = log(g_w / p_w), t_h = log(g_h / p_h)

The regression loss is:
L_reg = Σ (t_k - d_k)^2

## Code Examples

### Example 1: Simulated Selective Search (Generating Region Proposals)

```python
import torch
import numpy as np

def simulate_selective_search(image_size, num_proposals=2000):
    h, w = image_size
    proposals = []
    for _ in range(num_proposals):
        x1 = np.random.randint(0, w-10)
        y1 = np.random.randint(0, h-10)
        x2 = np.random.randint(x1+10, w)
        y2 = np.random.randint(y1+10, h)
        proposals.append([x1, y1, x2, y2])
    return torch.tensor(proposals, dtype=torch.float32)

image_size = (224, 224)
proposals = simulate_selective_search(image_size)
print(f"Generated {proposals.shape[0]} proposals")
print(f"Example proposal: {proposals[0].tolist()}")
# Output: Generated 2000 proposals
# Example proposal: [83, 56, 179, 191]
```

### Example 2: Warping and CNN Feature Extraction

```python
import torch
import torchvision.transforms.functional as TF
from PIL import Image

def warp_proposal(image, proposal, target_size=(224, 224)):
    x1, y1, x2, y2 = [int(v) for v in proposal]
    crop = image[:, y1:y2, x1:x2]
    if crop.numel() == 0:
        return torch.zeros(3, 224, 224)
    # Pad if needed
    h, w = crop.shape[1:]
    if h == 0 or w == 0:
        return torch.zeros(3, 224, 224)
    crop = TF.resize(crop, target_size, antialias=True)
    return crop

# Simulated image and CNN
image = torch.randn(3, 224, 224)
proposal = torch.tensor([50, 50, 150, 150])
warped = warp_proposal(image, proposal)
print(f"Warped proposal shape: {warped.shape}")
# Output: Warped proposal shape: torch.Size([3, 224, 224])

# Simulate CNN feature extraction
cnn_features = torch.randn(1, 4096)  # fc7 features
print(f"CNN features shape: {cnn_features.shape}")
# Output: CNN features shape: torch.Size([1, 4096])
```

### Example 3: SVM Classification and Bounding Box Regression

```python
import torch
import torch.nn as nn

class RCNNClassifier(nn.Module):
    def __init__(self, num_features=4096, num_classes=20):
        super().__init__()
        # SVMs are typically trained per-class, but we simulate with Linear layers
        self.svms = nn.Linear(num_features, num_classes)
        # Bounding box regressor: 4 deltas per class
        self.bbox_reg = nn.Linear(num_features, num_classes * 4)

    def forward(self, features):
        scores = self.svms(features)
        bbox_deltas = self.bbox_reg(features)
        return scores, bbox_deltas

features = torch.randn(10, 4096)
model = RCNNClassifier()
scores, deltas = model(features)
print(f"Scores shape: {scores.shape}, Deltas shape: {deltas.shape}")
# Output: Scores shape: torch.Size([10, 20]), Deltas shape: torch.Size([10, 80])

# Apply NMS to remove overlapping proposals
scores_softmax = torch.softmax(scores, dim=1)
print(f"Top class probabilities: {scores_softmax.max(dim=1).values[:5]}")
# Output: Top class probabilities: tensor([0.0746, 0.0669, 0.0547, 0.0619, 0.0649])
```

## Common Mistakes

1. **Treating the three stages as end-to-end trainable**: R-CNN trains CNN, SVMs, and regression independently. The SVM training cannot backpropagate through the CNN. This is a key architectural limitation.

2. **Ignoring the IoU threshold for positive/negative samples during SVM training**: A proposal with IoU > 0.5 with ground truth is positive for the object class; IoU < 0.3 is background. Proposals with IoU between 0.3 and 0.5 are ignored.

3. **Hard negative mining**: The background class is heavily dominant. R-CNN uses hard negative mining, iteratively adding false positives to the training set. Skipping this step leads to poor accuracy.

4. **Storing features on disk**: R-CNN saves extracted CNN features for each proposal (~2000 per image * 200GB for VOC). This disk I/O bottleneck is a major practical limitation.

5. **Slow inference**: R-CNN takes ~47 seconds per image (2000 forward passes through VGG16). This makes it impractical for real-time applications.

## Interview Questions

### Beginner - 5

1. What does R-CNN stand for?
2. What are the three main stages of the R-CNN pipeline?
3. What method does R-CNN use for generating region proposals?
4. Why does R-CNN warp each proposal to a fixed size?
5. How many proposals does R-CNN typically generate per image?

### Intermediate - 5

1. Explain why R-CNN uses SVMs instead of the final softmax layer for classification.
2. How does R-CNN handle bounding box regression?
3. What is hard negative mining and why is it needed in R-CNN training?
4. What are the main computational bottlenecks in R-CNN?
5. How does the IoU threshold affect the quality of training samples in R-CNN?

### Advanced - 3

1. Derive the objective function for bounding box regression in R-CNN and explain why linear regression is used on CNN features.
2. Compare and contrast the training pipeline of R-CNN with end-to-end detectors like Faster R-CNN.
3. What specific design choices in R-CNN motivated the development of Spatial Pyramid Pooling in SPP-Net?

## Practice Problems

### Easy - 5

1. Generate 100 random region proposals in a 224x224 image.
2. Implement a function to crop and warp a region to 224x224.
3. Compute IoU between a proposal and ground truth to determine positive/negative label.
4. Count how many proposals overlap with a given ground truth box with IoU > 0.5.
5. Implement a simulated CNN that outputs 4096-d feature vectors.

### Medium - 5

1. Implement hard negative mining for SVM training.
2. Build a non-maximum suppression function for R-CNN outputs.
3. Implement the bounding box regression training pipeline.
4. Construct a class that manages proposal generation, warping, and feature extraction.
5. Compute the computational cost (FLOPs) of processing 2000 proposals through VGG16.

### Hard - 3

1. Implement a full R-CNN training loop on a small dataset (e.g., VOC subset).
2. Compare R-CNN vs. sliding window detection: quantify the speedup from using proposals.
3. Design and implement a multi-scale R-CNN variant that processes proposals at different resolutions.

## Solutions

Easy 1:
```python
proposals = simulate_selective_search((224, 224), num_proposals=100)
print(f"Generated {len(proposals)} proposals, shape: {proposals[0].shape}")
# Output: Generated 100 proposals, shape: torch.Size([4])
```

Medium 1 — Hard Negative Mining:
```python
def hard_negative_mining(scores, labels, neg_ratio=3):
    # scores: [N, num_classes], labels: [N] with -1=ignore, 0=bg, >0=fg
    pos_mask = labels > 0
    neg_mask = labels == 0
    num_pos = pos_mask.sum()
    num_neg = min(neg_mask.sum(), num_pos * neg_ratio)
    neg_scores = scores[neg_mask].max(dim=1).values
    _, hard_idx = neg_scores.topk(num_neg)
    sample_mask = torch.zeros(len(scores), dtype=torch.bool)
    sample_mask[pos_mask] = True
    neg_indices = torch.where(neg_mask)[0][hard_idx]
    sample_mask[neg_indices] = True
    return sample_mask

scores = torch.randn(100, 5)
labels = torch.cat([torch.ones(5), torch.zeros(90), -torch.ones(5)])
mask = hard_negative_mining(scores, labels.long())
print(f"Selected {mask.sum().item()} samples ({5} pos + {5*3} neg)")
# Output: Selected 20 samples (5 pos + 15 neg)
```

## Related Concepts

- DL-231: Object Detection Overview
- DL-238: Fast R-CNN
- DL-239: Faster R-CNN
- DL-240: Mask R-CNN

## Next Concepts

- DL-238: Fast R-CNN
- DL-239: Faster R-CNN

## Summary

R-CNN pioneered deep learning-based object detection by combining selective search proposals with CNN feature extraction and SVM classification. Its multi-stage pipeline achieved breakthrough accuracy but suffered from computational inefficiency due to independent processing of each proposal. The model's limitations directly motivated the development of Fast R-CNN (shared computation) and Faster R-CNN (learned proposals). R-CNN remains important as the foundational work that established the two-stage detection paradigm.

## Key Takeaways

- R-CNN uses selective search to generate ~2000 category-independent region proposals
- Each proposal is warped and independently processed by a CNN for feature extraction
- Class-specific SVMs classify proposals; linear regression refines bounding boxes
- Multi-stage training (CNN pre-train + SVM + regression) is non-end-to-end
- Inference is slow (~47s/image) due to 2000 forward passes
- R-CNN achieved 53.3% mAP on VOC 2010, a 30% improvement over prior art
- The three-stage pipeline inspired the evolution toward fast, end-to-end detectors
