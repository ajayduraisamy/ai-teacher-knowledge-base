# Concept: Softmax Activation

## Concept ID

DL-122

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Activation Functions

## Learning Objectives

- Understand the mathematical formulation of the softmax function
- Implement softmax in PyTorch and apply to multi-class classification
- Analyze the temperature parameter and its effect on output distribution
- Identify numerical stability issues and their solutions
- Distinguish softmax from sigmoid for multi-class vs multi-label tasks

## Prerequisites

- Sigmoid activation (DL-111)
- Multi-class classification concepts
- Understanding of probability distributions
- Cross-entropy loss familiarity

## Definition

The softmax function converts a vector of real-valued logits into a probability distribution over K classes. For an input vector z in R^K, softmax(z)_i = e^(z_i) / sum_j e^(z_j). The outputs sum to 1 and are strictly positive, making them interpretable as class probabilities. Softmax is a generalization of the sigmoid function to multiple classes: for K = 2, softmax produces equivalent results to sigmoid (with the two classes being complementary). The function is invariant to additive constants: adding the same value to all logits produces the same probabilities.

## Intuition

Imagine you are at an auction for rare items, and you are deciding which item to bid on. Each item has a value in your mind (logit). You want to choose probabilistically — the most valuable item gets the highest probability, but you still sometimes choose others. Softmax turns these raw values into a soft auction where the highest value gets the highest probability, but the probabilities smoothly transition based on the values. If one item is far more valuable than others, its probability dominates. If values are similar, the probabilities are spread more evenly. The temperature parameter controls how soft this auction is — high temperature makes it uniform, low temperature makes it more deterministic.

## Why This Concept Matters

Softmax is arguably the most important output activation for neural networks — it is the standard for multi-class classification, language modeling, and any task requiring a probability distribution over categories. Every image classifier (ResNet, EfficientNet, ViT), language model (GPT, BERT), and speech recognition system uses softmax (or its efficient approximations) in its output layer. Understanding softmax's properties — normalization, temperature scaling, numerical stability — is essential for building any classification or sequence generation model.

## Mathematical Explanation

Softmax is defined as:

softmax(z_i) = e^(z_i) / sum_j e^(z_j) for i = 1, ..., K

Properties:
- Outputs form a probability distribution: 0 < softmax(z_i) < 1, sum_i softmax(z_i) = 1
- Invariant to translation: softmax(z + c) = softmax(z) for any constant c
- The derivative involves the Kronecker delta:
  d/dz_j softmax(z_i) = softmax(z_i) * (delta_ij - softmax(z_j))

The temperature parameter T modifies the function:
softmax(z_i, T) = e^(z_i / T) / sum_j e^(z_j / T)

- T -> 0: hardmax (one-hot distribution)
- T = 1: standard softmax
- T -> infinity: uniform distribution

Numerical stability trick: subtract max(z) before exponentiation to avoid overflow.
softmax_stable(z_i) = e^(z_i - max(z)) / sum_j e^(z_j - max(z))

## Code Examples

### Example 1: Basic Softmax

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.tensor([[1.0, 2.0, 3.0],
                       [-1.0, 0.0, 1.0]])

softmax = nn.Softmax(dim=1)
probs_module = softmax(logits)
probs_func = F.softmax(logits, dim=1)

print("Logits:\n", logits)
print("Probabilities:\n", probs_module)
print("Sum of probabilities:", probs_module.sum(dim=1))
# Output:
# Logits:
#  tensor([[ 1.,  2.,  3.],
#          [-1.,  0.,  1.]])
# Probabilities:
#  tensor([[0.0900, 0.2447, 0.6652],
#          [0.0900, 0.2447, 0.6652]])
# Sum of probabilities: tensor([1., 1.])
`

### Example 2: Temperature Scaling

`python
import torch
import torch.nn.functional as F

def softmax_with_temp(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

logits = torch.tensor([[1.0, 2.0, 3.0]])
temps = [0.5, 1.0, 2.0, 10.0]
for T in temps:
    probs = softmax_with_temp(logits, T)
    print(f"T={T:5.1f}: {probs[0].tolist()}")
# Output:
# T= 0.5: [0.0159, 0.1173, 0.8668]
# T= 1.0: [0.0900, 0.2447, 0.6652]
# T= 2.0: [0.1863, 0.3072, 0.5065]
# T=10.0: [0.3006, 0.3322, 0.3672]
`

### Example 3: CrossEntropyLoss with Softmax

`python
import torch
import torch.nn as nn

logits = torch.tensor([[1.0, 2.0, 3.0],
                        [3.0, 2.0, 1.0]])
targets = torch.tensor([2, 0])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print("CrossEntropyLoss:", loss.item())

softmax_vals = torch.softmax(logits, dim=1)
pred_classes = torch.argmax(softmax_vals, dim=1)
print("Predicted classes:", pred_classes)
print("Accuracy:", (pred_classes == targets).float().mean().item())
# Output:
# CrossEntropyLoss: 0.4076
# Predicted classes: tensor([2, 0])
# Accuracy: 1.0
`

## Common Mistakes

1. **Using softmax on the wrong dimension**: Softmax normalizes along a single dimension. For batched data, use dim=1 (features) for classification logits.
2. **Not accounting for numerical stability**: e^(large numbers) can overflow. PyTorch's implementation handles this internally, but custom implementations must subtract max.
3. **Using softmax for multi-label classification**: Softmax assumes mutually exclusive classes. Use sigmoid (one vs all) for multi-label problems.
4. **Interpreting softmax outputs as calibrated probabilities**: Softmax outputs need temperature scaling for calibration. They are not naturally calibrated.
5. **Applying softmax to logits before CrossEntropyLoss**: CrossEntropyLoss in PyTorch already applies softmax internally. Doing it manually would apply log-softmax twice.

## Interview Questions

### Beginner

1. What does softmax do to a vector of logits?
2. What is the sum of softmax outputs for a single input?
3. How is softmax different from sigmoid?
4. What is the range of each softmax output?
5. Why is the max subtraction trick used in softmax implementation?

### Intermediate

1. Explain the effect of temperature on softmax outputs.
2. Derive the derivative of softmax with respect to its input.
3. Why is CrossEntropyLoss preferred over manually computing softmax plus NLL?
4. How does softmax handle logit vectors of varying scales?
5. What problem arises when using softmax for large vocabulary size?

### Advanced

1. Derive and explain the Jacobian of the softmax function and its properties.
2. Prove that softmax is invariant to additive constants: softmax(z + c) = softmax(z).
3. Design and implement a hierarchical softmax for efficient language modeling with large vocabularies.

## Practice Problems

### Easy

1. Compute softmax for logits [1, 1, 1].
2. Compute softmax for logits [10, 0, 0].
3. What is the derivative of softmax_i with respect to z_i?
4. What happens to softmax if one logit is much larger than others?
5. For K = 2, show that softmax is equivalent to sigmoid.

### Medium

1. Implement numerically stable softmax from scratch in PyTorch.
2. Train a multi-class classifier on MNIST (10 classes) and analyze output probabilities.
3. Implement temperature scaling for model calibration and evaluate on ECE.
4. Compare softmax with sigmoid for a 3-class problem.
5. Compute the gradient of the cross-entropy loss with respect to the logits.

### Hard

1. Implement a noise-contrastive estimation (NCE) loss as an alternative to softmax for large vocabularies.
2. Prove that the cross-entropy loss plus softmax gradient is (p - y), where p is the softmax output and y is the one-hot target.
3. Design a sparsemax activation (softmax with sparsity) and compare its properties to softmax.

## Solutions

### Easy Solutions

1. softmax([1,1,1]) = [e/3e, e/3e, e/3e] = [1/3, 1/3, 1/3]
2. softmax([10,0,0]) is approximately [1.0, 0.0, 0.0] (e^10 dominates)
3. dp_i/dz_i = p_i * (1 - p_i)
4. The softmax becomes approximately one-hot for the dominant class
5. For K=2: p1 = e^z1/(e^z1+e^z2) = 1/(1+e^-(z1-z2)) = sigmoid(z1-z2), p2 = 1-p1

## Related Concepts

- Sigmoid Activation (DL-111)
- Cross-Entropy Loss
- Temperature Scaling
- Log-Sum-Exp Trick

## Next Concepts

- Hard Sigmoid (DL-123)
- Hard Swish (DL-124)
- Maxout Activation (DL-125)

## Summary

Softmax converts a vector of logits into a probability distribution over K classes. It is the standard output activation for multi-class classification, language modeling, and any task requiring normalized class probabilities. The temperature parameter controls the sharpness of the distribution, and numerical stability is achieved by subtracting the maximum logit before exponentiation.

## Key Takeaways

- softmax(z)_i = e^z_i / sum_j e^z_j, outputs sum to 1
- Standard output activation for multi-class classification
- Generalization of sigmoid to K > 2 classes
- Invariant to additive constant shifts in logits
- Temperature parameter controls distribution sharpness
- CrossEntropyLoss combines log-softmax and NLL loss for numerical stability
- Not naturally calibrated — requires temperature scaling for reliable probabilities
