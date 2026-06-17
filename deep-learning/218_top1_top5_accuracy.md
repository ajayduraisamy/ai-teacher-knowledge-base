# Concept: Top-1 and Top-5 Accuracy

## Concept ID

DL-218

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Image Classification

## Learning Objectives

- Understand top-1 and top-5 accuracy metrics
- Implement both metrics in PyTorch
- Analyze why top-5 is used for large-scale classification
- Compare models using these metrics

## Prerequisites

DL-217 Image Classification Pipeline

## Definition

Top-1 accuracy measures the proportion of test samples where the model's highest-confidence prediction matches the ground truth label. Top-5 accuracy measures whether the ground truth label appears among the model's five highest-confidence predictions.

## Intuition

When you ask "what's in this image?", sometimes there's ambiguity — is that a wolf or a husky? A goldfish or a koi? Top-5 accuracy acknowledges this ambiguity: the model gets credit if the correct answer is in its top 5 guesses. This is especially important for ImageNet with its fine-grained classes (200 dog breeds). Top-5 was the primary metric in the ILSVRC challenge and is commonly reported alongside top-1.

## Why This Concept Matters

Top-5 accuracy was the official metric for ImageNet challenges and continues to be widely reported. It provides information about how close the model's predictions are to being correct, and is valuable when you want to know if the model is recognizing broadly correct categories.

## Mathematical Explanation

**Top-1 accuracy**:
$$\text{Top-1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i = \text{argmax}_k \hat{y}_{i,k}]$$

**Top-5 accuracy**:
$$\text{Top-5} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{Top5}(\hat{y}_i)]$$

Where $\text{Top5}(\hat{y}_i)$ is the set of indices with the 5 highest predicted probabilities.

**Relationship**: Top-5 > Top-1 (always). The gap indicates how often the correct label is close to being predicted.

## Code Examples

### Example 1: Computing Top-1 and Top-5

```python
import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    # Get top-k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    # Compare with target
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    
    return results

# Example usage
output = torch.tensor([
    [0.1, 0.1, 0.7, 0.05, 0.05],   # Highest: class 2
    [0.3, 0.2, 0.1, 0.2, 0.2],      # Highest: class 0
    [0.05, 0.8, 0.05, 0.05, 0.05],  # Highest: class 1
])
target = torch.tensor([2, 0, 1])

top1, top5 = accuracy(output, target, topk=(1, 3))
print(f"Top-1: {top1.item():.2f}%")   # 3/3 = 100%
print(f"Top-3: {top5.item():.2f}%")   # 3/3 = 100%
```

### Example 2: Realistic ImageNet Evaluation

```python
import torch
import torch.nn as nn
import torchvision.models as models

torch.manual_seed(42)

# Simulate ImageNet evaluation
model = models.resnet18(pretrained=False)
model.eval()

# Batch of 256 images (1000 classes)
batch_size = 256
num_classes = 1000
outputs = torch.randn(batch_size, num_classes)
targets = torch.randint(0, num_classes, (batch_size,))

def topk_accuracy(output, target, k=1):
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:k].reshape(-1).float().sum(0).item() / output.size(0)

top1_acc = topk_accuracy(outputs, targets, k=1)
top5_acc = topk_accuracy(outputs, targets, k=5)

print(f"Batch size: {batch_size}")
print(f"Top-1 accuracy (random): {top1_acc*100:.2f}%")
print(f"Top-5 accuracy (random): {top5_acc*100:.2f}%")

# Expected random performance
print(f"\nExpected random performance:")
print(f"  Top-1: {100/1000:.2f}%")
print(f"  Top-5: {500/1000:.2f}%")
```

### Example 3: Accuracy vs Confidence Analysis

```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Analyze the gap between top-1 and top-5
def accuracy_gap_analysis(outputs, targets):
    probs = F.softmax(outputs, dim=1)
    top1_preds = probs.argmax(dim=1)
    top5_vals, top5_preds = probs.topk(5, dim=1)
    
    top1_correct = top1_preds.eq(targets)
    top5_correct = targets.unsqueeze(1).eq(top5_preds).any(dim=1)
    
    # Gap: how many are top-5 correct but not top-1
    gap_cases = top5_correct & ~top1_correct
    gap_confidence = top5_vals[gap_cases][:, 0] - top5_vals[gap_cases][:, 1]
    
    return {
        'top1': top1_correct.float().mean().item(),
        'top5': top5_correct.float().mean().item(),
        'gap': (top5_correct.float().mean() - top1_correct.float().mean()).item(),
        'gap_confidence_mean': gap_confidence.mean().item() if gap_cases.any() else 0,
    }

# Generate predictions where top-5 benefit matters
outputs = torch.randn(100, 100)
targets = torch.randint(0, 100, (100,))

stats = accuracy_gap_analysis(outputs, targets)
print(f"Top-1: {stats['top1']*100:.2f}%")
print(f"Top-5: {stats['top5']*100:.2f}%")
print(f"Gap: {stats['gap']*100:.2f}%")
```

## Common Mistakes

1. **Confusing top-5 with multiple labels**: Top-5 is about the model's top predictions, not multiple ground-truth labels.
2. **Using argmax for top-5**: Argmax gives only the top-1; use torch.topk for top-k.
3. **Not normalizing output size**: Accuracy must be divided by batch size.
4. **Computing accuracy on the wrong dimension**: Ensure you're comparing along the class dimension.
5. **Reporting only top-1 or only top-5**: Both are informative and should be reported together.

## Interview Questions

### Beginner - 5
1. What is top-1 accuracy?
2. What is top-5 accuracy?
3. Why is top-5 used on ImageNet?
4. Is top-5 always higher than top-1?
5. What is the expected random top-1 accuracy on ImageNet?

### Intermediate - 5
1. Write code to compute top-5 accuracy in PyTorch.
2. Explain why the gap between top-1 and top-5 is informative.
3. How does the number of classes affect top-5 accuracy?
4. What is the expected random top-5 accuracy on ImageNet?
5. How can you improve top-1 accuracy without changing model architecture?

### Advanced - 3
1. Design a metric that smoothly interpolates between top-1 and top-5.
2. Analyze the trade-offs between top-k and calibration error.
3. Design an evaluation protocol for fine-grained classification where top-5 is misleading.

## Practice Problems

### Easy - 5
1. Compute top-1 and top-5 for 10 samples with 5 classes.
2. Implement the accuracy function using torch.topk.
3. Calculate expected random top-1 for 100 classes.
4. Compare top-1 vs top-5 for a model's predictions.
5. Plot accuracy vs k for top-k.

### Medium - 5
1. Add top-5 tracking to a training loop.
2. Compare ResNet-18 and ResNet-50 using top-1 and top-5.
3. Visualize examples where top-1 is wrong but top-5 is correct.
4. Compute weighted top-5 accuracy (more credit for closer classes).
5. Implement class-wise top-1 and top-5 analysis.

### Hard - 3
1. Design a metric that considers the hierarchy of classes.
2. Implement a confidence-aware accuracy metric.
3. Analyze the overconfidence in top-1 predictions vs top-5.

## Solutions

### Easy - 1 Solution
```python
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, 1)
    target_expanded = target.unsqueeze(1).expand_as(pred)
    correct = pred.eq(target_expanded).sum().item()
    return correct / output.size(0)
```

## Related Concepts

DL-216 ImageNet Dataset, DL-217 Image Classification Pipeline, DL-225 Classification Head

## Next Concepts

DL-219 Preprocessing for Images

## Summary

Top-1 and top-5 accuracy are standard metrics for multi-class classification. Top-1 measures exact match; top-5 measures whether the correct label is among the top 5 predictions. Top-5 is particularly important for large-scale classification with fine-grained categories.

## Key Takeaways

- Top-1: highest prediction matches ground truth
- Top-5: ground truth in top 5 predictions
- Top-5 >= Top-1 (always); gap indicates near-miss predictions
- Official ILSVRC metric was top-5
- Random top-1 on ImageNet: 0.1%; random top-5: 0.5%
- Use torch.topk() for efficient top-k computation
- Top-5 useful when fine-grained distinctions are hard
- Both metrics should be reported together
- Top-5 saturates more quickly than top-1 as models improve
- Modern models achieve >99% top-5 on ImageNet
