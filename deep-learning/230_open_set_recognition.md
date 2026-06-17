# Concept: Open Set Recognition

## Concept ID

DL-230

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand the open set recognition problem
- Implement open set classifiers with rejection
- Analyze the open space risk
- Compare open set methods with closed set approaches

## Prerequisites

DL-229 Zero-shot Image Classification, DL-226 Multilabel Image Classification

## Definition

Open set recognition (OSR) is the problem of classifying known classes while simultaneously detecting unknown/novel classes that were not present during training, addressing the fundamental limitation of closed-set classifiers that must classify everything into known categories.

## Intuition

Traditional classifiers are "closed set" — they assume every test sample belongs to one of the training classes. But in the real world, you encounter new things constantly. A self-driving car trained on cars, pedestrians, and bikes might encounter a deer on the road — it should recognize "I don't know what this is" rather than confidently misclassifying it. Open set recognition adds a "reject" or "unknown" option to the classifier, detecting when inputs don't belong to any known class.

## Why This Concept Matters

Closed-set classifiers are dangerously overconfident on unknown inputs. Open set recognition is critical for safety-critical applications (autonomous driving, medical diagnosis, security) where encountering novel categories is inevitable and misclassification could be catastrophic.

## Mathematical Explanation

**Open space risk**: The risk incurred by labeling a point far from known training data as a known class.
$$R_O(f) = \frac{\int_O f(x) dx}{\int_{S_O} f(x) dx}$$

Where $O$ is the open space (region far from known classes) and $S_O$ is the total space.

**Open set recognition objective**: Minimize both empirical risk on known classes and open space risk:
$$\min_f R_K(f) + \lambda R_O(f)$$

**OpenMax**: Replace softmax with OpenMax, which recalibrates probabilities and adds a background class:
$$\hat{P}(y=j|x) = \frac{e^{\mu_j(x)/T}}{\sum e^{\mu_k(x)/T} + e^{\text{unknown}(x)/T}}$$

**Entropy-based detection**: Unknown samples tend to have higher prediction entropy:
$$H(p) = -\sum_c p_c \log p_c$$

## Code Examples

### Example 1: Open Set Detection with Entropy

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class OpenSetClassifier(nn.Module):
    def __init__(self, num_known_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_known_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        return logits, probs, features

def detect_unknown(model, x, threshold=0.5):
    """Detect unknown samples using max softmax probability."""
    with torch.no_grad():
        logits, probs, features = model(x)
        max_probs, predictions = probs.max(dim=1)
        
        # Low max-probability = likely unknown
        is_known = max_probs > threshold
        return is_known, predictions, max_probs

# Test
model = OpenSetClassifier(num_known_classes=10)
model.eval()

# Known class samples
known = torch.randn(8, 3, 32, 32)
# Unknown class samples (out of distribution)
unknown = torch.randn(4, 3, 32, 32) * 3 + 10  # shifted distribution

# Classify known
is_known_k, preds_k, probs_k = detect_unknown(model, known, threshold=0.3)
# Classify unknown
is_known_u, preds_u, probs_u = detect_unknown(model, unknown, threshold=0.3)

print(f"Known samples detected as known: {is_known_k.sum()}/{len(known)}")
print(f"Known probs: {probs_k.tolist()}")
print(f"Unknown samples detected as unknown: {(~is_known_u).sum()}/{len(unknown)}")
print(f"Unknown probs: {probs_u.tolist()}")
```

### Example 2: OpenMax Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class OpenMax(nn.Module):
    """OpenMax layer for open set recognition."""
    def __init__(self, num_classes, alpha=10, threshold=0.7):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.threshold = threshold
    
    def fit_activation_vector(self, features, labels):
        """Fit class means and distances from training data."""
        self.class_means = {}
        self.distances = {}
        
        for c in range(self.num_classes):
            class_feats = features[labels == c]
            if len(class_feats) > 0:
                mean = class_feats.mean(dim=0)
                self.class_means[c] = mean
                dists = torch.cdist(class_feats, mean.unsqueeze(0)).squeeze()
                self.distances[c] = dists
    
    def forward(self, logits, features):
        """Compute OpenMax probabilities with unknown class."""
        probs = F.softmax(logits, dim=1)
        
        # Estimate Weibull or use simple distance-based rescaling
        openmax_probs = []
        for i in range(logits.size(0)):
            feat = features[i]
            
            # Compute distance to nearest class mean
            min_dist = float('inf')
            for c in range(self.num_classes):
                if c in self.class_means:
                    dist = torch.dist(feat, self.class_means[c]).item()
                    min_dist = min(min_dist, dist)
            
            # Rescale based on distance
            unknown_score = min(1.0, min_dist / self.alpha)
            
            # Adjust probabilities
            adjusted = probs[i] * (1 - unknown_score)
            adjusted = torch.cat([adjusted, torch.tensor([unknown_score])])
            openmax_probs.append(adjusted)
        
        return torch.stack(openmax_probs)

# Demo OpenMax
num_classes = 5
model = OpenSetClassifier(num_known_classes=num_classes)
openmax = OpenMax(num_classes)

# Simulate fitting
train_features = torch.randn(500, 128)
train_labels = torch.randint(0, num_classes, (500,))
openmax.fit_activation_vector(train_features, train_labels)

# Test
test_logits = torch.randn(10, num_classes)
test_features = torch.randn(10, 128)
openmax_probs = openmax(test_logits, test_features)

print(f"OpenMax output: {openmax_probs.shape}")
print(f"Standard softmax: {test_logits.shape[-1]} classes")
print(f"OpenMax: {openmax_probs.shape[-1]} classes (includes unknown)")
print(f"Unknown probabilities: {openmax_probs[:, -1].tolist()}")
```

### Example 3: Open Set Evaluation

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

def open_set_evaluation(model, known_loader, unknown_loader, threshold=0.5):
    """Evaluate open set recognition performance."""
    model.eval()
    
    # Metrics
    known_correct = 0
    known_total = 0
    unknown_detected = 0
    unknown_total = 0
    
    with torch.no_grad():
        # Known class performance
        for images, labels in known_loader:
            logits, probs, _ = model(images)
            max_probs, predictions = probs.max(dim=1)
            
            # Correctly classified known
            known_correct += ((predictions == labels) & (max_probs > threshold)).sum().item()
            known_total += labels.size(0)
        
        # Unknown detection
        for images, _ in unknown_loader:
            logits, probs, _ = model(images)
            max_probs, _ = probs.max(dim=1)
            
            # Detected as unknown
            unknown_detected += (max_probs <= threshold).sum().item()
            unknown_total += images.size(0)
    
    # Metrics
    known_acc = known_correct / known_total if known_total > 0 else 0
    unknown_detection_rate = unknown_detected / unknown_total if unknown_total > 0 else 0
    
    return {
        'known_accuracy': known_acc,
        'unknown_detection_rate': unknown_detection_rate,
        'open_set_f1': 2 * known_acc * unknown_detection_rate / 
                       (known_acc + unknown_detection_rate + 1e-8),
    }

# Simulate evaluation
class SimpleLoader:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __iter__(self):
        yield self.data, self.labels

model = OpenSetClassifier(10)
known_data = torch.randn(50, 3, 32, 32)
known_labels = torch.randint(0, 10, (50,))
unknown_data = torch.randn(20, 3, 32, 32)  # different distribution

known_loader = SimpleLoader(known_data, known_labels)
unknown_loader = SimpleLoader(unknown_data, torch.zeros(20))

metrics = open_set_evaluation(model, known_loader, unknown_loader, threshold=0.3)
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

## Common Mistakes

1. **Assuming softmax probabilities are calibrated**: Standard softmax is overconfident; calibration is needed.
2. **Threshold too high or low**: High = miss known, Low = miss unknown; needs tuning on validation set.
3. **Not evaluating on true unknowns**: Test with genuinely novel classes (not just noise).
4. **Ignoring the open space risk**: Far-from-training points should be rejected.
5. **Using closed-set accuracy as only metric**: Open set requires evaluating both closed-set accuracy and unknown detection.

## Interview Questions

### Beginner - 5
1. What is open set recognition?
2. How does it differ from closed-set classification?
3. What is open space risk?
4. Why is open set recognition important?
5. What is the reject option?

### Intermediate - 5
1. Explain OpenMax and how it works.
2. How do you evaluate open set classifiers?
3. Compare open set with zero-shot learning.
4. What is the relationship between open set and out-of-distribution detection?
5. How do you choose the rejection threshold?

### Advanced - 3
1. Design an open set recognition method using generative models.
2. Analyze the theoretical properties of different OSR approaches.
3. Implement a method that incrementally adds unknown classes when detected.

## Practice Problems

### Easy - 5
1. Implement max-softmax unknown detection.
2. Compute entropy-based outlier score.
3. Add rejection threshold to a classifier.
4. Evaluate open set accuracy.
5. Count unknown detection rate.

### Medium - 5
1. Implement OpenMax from scratch.
2. Train a classifier with open set loss (e.g., Center Loss).
3. Compare threshold-based vs distance-based methods.
4. Implement a baseline using temperature scaling.
5. Evaluate on a realistic open set benchmark (e.g., TinyImageNet).

### Hard - 3
1. Design a generative open set method (e.g., GAN-based).
2. Implement a method that learns a calibrated open set score.
3. Analyze the trade-off between known accuracy and unknown detection.

## Solutions

### Easy - 1 Solution
```python
def is_unknown(probs, threshold=0.5):
    max_prob = probs.max().item()
    return max_prob < threshold
```

## Related Concepts

DL-229 Zero-shot Image Classification, DL-226 Multilabel Image Classification, DL-228 Few-shot Image Classification

## Next Concepts

None (end of sequence)

## Summary

Open set recognition enables classifiers to detect and reject samples from unknown classes, addressing the fundamental limitation of closed-set classifiers. Methods range from simple thresholding of softmax scores to sophisticated approaches like OpenMax and generative models.

## Key Takeaways

- Closed-set: must classify into training classes; Open-set: can reject unknowns
- Open space risk: risk of labeling distant points as known classes
- Max softmax probability is a simple baseline for unknown detection
- OpenMax adds an unknown class by recalibrating probabilities
- Entropy-based detection: unknowns have higher prediction entropy
- Evaluation: closed-set accuracy + unknown detection rate
- Critical for safety-critical applications
- Related to out-of-distribution (OOD) detection
- Calibration is essential for reliable open-set decisions
- Frontiers: incremental learning, generative open-set, open-world recognition
