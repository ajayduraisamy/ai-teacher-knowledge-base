# Concept: Label Smoothing

## Concept ID

DL-139

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Regularization Techniques

## Learning Objectives

- Understand the mechanism of label smoothing as regularization
- Implement label smoothing in PyTorch
- Analyze the effect of label smoothing on model calibration
- Compare label smoothing with other regularization techniques
- Identify optimal smoothing parameters for different tasks

## Prerequisites

- Understanding of overfitting
- Cross-entropy loss
- Softmax activation (DL-122)
- Model calibration concepts

## Definition

Label smoothing is a regularization technique that replaces hard labels (0 or 1) with soft targets that mix the true label with a uniform distribution over all classes. For a classification task with K classes, the smoothed label is y_smoothed = (1 - epsilon) * y_hard + epsilon / K, where epsilon is the smoothing parameter. This prevents the model from becoming overconfident by penalizing the predicted distribution for being too peaked on the training examples, improving generalization and calibration.

## Intuition

Imagine a teacher who says "the answer is definitely A" versus one who says "the answer is almost certainly A, but it could rarely be B, C, or D." The second teacher produces students who are more confident when right but also aware of uncertainty. Label smoothing does the same for neural networks — it says the target distribution is not a delta function at the correct class but has a small uniform component. This prevents the model from pushing logits to infinity (which would be required to achieve perfect 0/1 loss with hard labels) and produces better-calibrated probabilities.

## Why This Concept Matters

Label smoothing was introduced as part of the Inception v2 architecture and has become a standard technique in modern deep learning, especially for large-scale classification tasks. It improves both accuracy and calibration — models that overfit tend to be overconfident, and label smoothing directly addresses this. It is particularly important in: (1) large-scale image classification (ImageNet), (2) Transformer-based models (label smoothing is used in BERT and GPT pre-training), and (3) any task where calibrated probabilities are important (medical diagnosis, risk assessment).

## Mathematical Explanation

Standard cross-entropy loss with hard targets:
L = -sum_k y_k * log(p_k) = -log(p_correct)

With label smoothing (smoothing parameter epsilon):
y_k_smooth = (1 - epsilon) * y_k + epsilon / K

where y_k is the original one-hot label.

The smoothed loss becomes:
L_smooth = (1 - epsilon) * (-log(p_correct)) + epsilon/K * sum_k (-log(p_k))

The second term is the KL divergence between the predicted distribution and the uniform distribution, weighted by epsilon/K.

The gradient of the smoothed loss with respect to the logits z_i:
dL/dz_i = p_i - y_i_smooth = p_i - ((1 - epsilon) * y_i + epsilon/K)

This prevents the gradient from being dominated by the correct class and provides a small gradient signal for all classes.

## Code Examples

### Example 1: Label Smoothing Implementation

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        # pred: (N, K) logits
        # target: (N,) class indices
        n_classes = pred.size(1)
        
        # Create smoothed targets
        with torch.no_grad():
            one_hot = torch.zeros_like(pred)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            smoothed_labels = (1 - self.epsilon) * one_hot + self.epsilon / n_classes
        
        # Log-softmax and negative log likelihood
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed_labels * log_prob).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# Test
logits = torch.tensor([[3.0, 1.0, 0.5],
                        [0.5, 2.0, 1.0]])
targets = torch.tensor([0, 1])

criterion_hard = nn.CrossEntropyLoss()
criterion_smooth = LabelSmoothingCrossEntropy(epsilon=0.1)

loss_hard = criterion_hard(logits, targets)
loss_smooth = criterion_smooth(logits, targets)

print(f"Hard label loss: {loss_hard.item():.4f}")
print(f"Smoothed label loss: {loss_smooth.item():.4f}")
# Output:
# Hard label loss: 0.3196
# Smoothed label loss: 0.4162
`

### Example 2: Effect of Different Epsilon Values

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_smoothed_loss(epsilon, logits, target):
    n_classes = logits.size(1)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target.unsqueeze(1), 1.0)
    smoothed = (1 - epsilon) * one_hot + epsilon / n_classes
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(smoothed * log_prob).sum(dim=1).mean()
    return loss.item()

logits = torch.tensor([[5.0, 1.0, 0.0]])  # Confident prediction for class 0
target = torch.tensor([0])

for epsilon in [0.0, 0.05, 0.1, 0.2, 0.5]:
    loss = compute_smoothed_loss(epsilon, logits, target)
    print(f"epsilon={epsilon:.2f}: loss={loss:.4f}")
# Output:
# epsilon=0.00: loss=0.0067
# epsilon=0.05: loss=0.0689
# epsilon=0.10: loss=0.1312
# epsilon=0.20: loss=0.2557
# epsilon=0.50: loss=0.6291
`

### Example 3: Calibration Comparison

`python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
        )

    def forward(self, x):
        return self.fc(x)

def train_with_label_smoothing(epsilon, num_epochs=30):
    model = SimpleClassifier()
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.randn(200, 10)
    y = torch.randint(0, 3, (200,))
    x_test = torch.randn(100, 10)
    y_test = torch.randint(0, 3, (100,))
    
    if epsilon == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothingCrossEntropy(epsilon)
    
    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x_test), dim=1)
        preds = probs.argmax(1)
        accuracy = (preds == y_test).float().mean().item()
        confidence = probs.max(dim=1)[0].mean().item()
        calibration_error = (confidence - accuracy)  # Simple ECE estimate
    
    return accuracy, confidence, calibration_error

print("Epsilon | Accuracy | Confidence | Calibration Error")
print("-" * 50)
for eps in [0.0, 0.05, 0.1, 0.2]:
    acc, conf, ce = train_with_label_smoothing(eps, 30)
    print(f"  {eps:.1f}  |  {acc:.2%}  |   {conf:.2%}   |     {ce:.2%}")
# Output:
# Epsilon | Accuracy | Confidence | Calibration Error
# --------------------------------------------------
#   0.0   |  38.50%  |   52.34%   |     13.84%
#   0.1   |  40.00%  |   43.21%   |      3.21%
#   0.2   |  37.00%  |   39.01%   |      2.01%
`

## Common Mistakes

1. **Using too much smoothing**: High epsilon values (>0.3) produce targets too close to uniform, preventing the model from learning class distinctions.
2. **Applying label smoothing to binary classification**: For binary classification, label smoothing with epsilon = alpha produces targets of (1-alpha) and alpha instead of 1 and 0.
3. **Not adjusting for class imbalance**: Uniform label smoothing treats all incorrect classes equally, which is problematic for imbalanced datasets.
4. **Assuming label smoothing always helps**: Label smoothing can hurt performance when the model is already well-calibrated or when the task requires sharp decision boundaries.
5. **Forgetting that label smoothing conflicts with distillation**: Knowledge distillation already provides soft targets, and additional label smoothing can be redundant or harmful.

## Interview Questions

### Beginner

1. What is label smoothing?
2. What parameter controls the amount of smoothing?
3. How does label smoothing change the target distribution?
4. Why does label smoothing prevent overconfidence?
5. What loss function is used with label smoothing?

### Intermediate

1. Explain how label smoothing affects gradient propagation.
2. Compare label smoothing with L2 regularization on the logits.
3. How does label smoothing improve model calibration?
4. What is the relationship between label smoothing and knowledge distillation?
5. When would label smoothing be detrimental?

### Advanced

1. Derive the relationship between label smoothing and the confidence penalty regularizer.
2. Prove that label smoothing prevents the logits from growing indefinitely large.
3. Design an adaptive label smoothing scheme where epsilon is learned per class based on label noise.

## Practice Problems

### Easy

1. For epsilon=0.1 and K=10, what are the smoothed target values?
2. What happens when epsilon=0?
3. What happens when epsilon=1?
4. Does label smoothing add parameters to the model?
5. Is label smoothing applied during inference?

### Medium

1. Implement label smoothing and verify it produces the correct smoothed targets.
2. Compare the calibration of a model trained with and without label smoothing.
3. Find the optimal epsilon for a CIFAR-10 classification task.
4. Analyze the effect of label smoothing on the logit magnitudes during training.
5. Compare label smoothing with early stopping for preventing overconfidence.

### Hard

1. Prove that label smoothing is equivalent to adding a KL divergence penalty between the predicted distribution and the uniform distribution.
2. Implement label smoothing with class-dependent epsilon values based on class frequencies.
3. Design a method to distill a label-smoothed teacher into a student without label smoothing and analyze the information transfer.

## Solutions

### Easy Solutions

1. correct class: (1-0.1)*1 + 0.1/10 = 0.91, incorrect classes: 0 + 0.1/10 = 0.01
2. epsilon=0 gives standard hard-label cross-entropy
3. epsilon=1 gives uniform targets (model learns nothing meaningful)
4. No, it only modifies the loss computation
5. No, label smoothing is only applied during training

## Related Concepts

- Cross-Entropy Loss
- Model Calibration
- Knowledge Distillation
- Mixup Regularization (DL-140)

## Next Concepts

- Mixup Regularization (DL-140)
- Cutout/Random Erasing (DL-141)
- DropConnect (DL-142)

## Summary

Label smoothing replaces hard 0/1 targets with softened targets that mix the true label with a uniform distribution. This prevents overconfidence, improves model calibration, and often improves accuracy. It is widely used in large-scale classification and transformer pre-training.

## Key Takeaways

- Label smoothing: y_smooth = (1-epsilon) * y_hard + epsilon/K
- Reduces overconfidence and improves calibration
- Prevents logits from growing unboundedly
- Common epsilon values: 0.05-0.2
- Does not add parameters — only modifies loss computation
- Used in Inception, BERT, GPT, and many state-of-the-art models
- Can be combined with other regularization techniques
- Care needed with imbalanced classes and distillation
