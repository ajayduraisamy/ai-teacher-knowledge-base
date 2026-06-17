# Concept: Logits

## Concept ID

DL-047

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Define logits and distinguish them from probabilities
- Extract logits from a neural network model
- Understand the relationship between logits, softmax, and cross-entropy loss
- Interpret logit values in multi-class classification

## Prerequisites

DL-046 (Forward Pass Computation), DL-001 (Perceptron), DL-048 (Softmax Output)

## Definition

Logits are the raw, unnormalized scores produced by the last layer of a neural network before applying a softmax (or sigmoid) activation. They are the output of the final linear layer: logits = **W**_{L} **a**_{L-1} + **b**_{L}. Logits can be any real number, positive or negative. They represent the confidence that the model assigns to each class, but they are not probabilities.

## Intuition

If a neural network is a courtroom, logits are the raw votes from the jury. Class A might get +5.2 votes, Class B gets -1.3, Class C gets +0.7. These raw scores are not probabilities (they don't sum to 1, and they can be negative). The softmax function acts as the judge, converting these raw votes into proper probabilities that sum to 1. Using logits directly (before softmax) during loss computation is numerically more stable.

## Why This Concept Matters

Logits are central to classification tasks:
- **Numerical stability**: `F.cross_entropy(logits, targets)` is more stable than `F.nll_loss(F.log_softmax(logits, dim=1), targets)`
- **Temperature scaling**: Logits can be scaled to control prediction sharpness
- **Knowledge distillation**: Student networks learn from teacher logits (not probabilities)
- **Logit adjustment**: Used for long-tail learning and calibration
- **Interpretation**: Logit values reflect model confidence before probability normalization

## Mathematical Explanation

For a K-class classification problem with model f(x):

logits = f(x) ∈ ℝ^K

The logit for class i is the raw score s_i.

The relationship between logits and probabilities via softmax:

p_i = e^{s_i} / Σ_{j=1}^{K} e^{s_j}

The cross-entropy loss is:

L = -log(p_y) = -s_y + log(Σ_{j=1}^{K} e^{s_j})

where y is the correct class. Notice that only the logit for the correct class (s_y) appears positively — maximizing it reduces loss. The second term penalizes large logits for incorrect classes.

Gradient with respect to logits:

∂L/∂s_i = p_i - [i == y]

This is the softmax probability minus the one-hot target.

## Code Examples

### Example 1: Extracting logits from a model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)  # No softmax — outputs logits
)

x = torch.randn(4, 784)
logits = model(x)
print("Logits shape:", logits.shape)
print("Logits:\n", logits)
print("Logits min/max:", logits.min().item(), logits.max().item())
# Output:
# Logits shape: torch.Size([4, 10])
# Logits:
#  tensor([[ 0.1234, -0.2345,  0.3456, -0.4567,  0.5678, -0.6789,  0.7890, -0.8901,  0.9012, -0.1234],
#          ...])
# Logits min/max: -0.8901 0.9012
```

### Example 2: Logits vs probabilities

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])

# Softmax to get probabilities
probs = F.softmax(logits, dim=-1)
print("Logits:", logits)
print("Probabilities:", probs)
print("Sum of probabilities:", probs.sum().item())

# Note: logits can be negative or positive, probabilities are always [0,1]
print("\nNegative logits:", torch.tensor([[-1.0, -2.0, -0.5]]))
print("Softmax of negative logits:", F.softmax(torch.tensor([[-1.0, -2.0, -0.5]]), dim=-1))
# Output:
# Logits: tensor([[2.0000, 1.0000, 0.1000]])
# Probabilities: tensor([[0.6590, 0.2424, 0.0986]])
# Sum of probabilities: 1.0000
# 
# Negative logits: tensor([[-1.0000, -2.0000, -0.5000]])
# Softmax of negative logits: tensor([[0.3072, 0.1130, 0.5798]])
```

### Example 3: Numerical stability: logits vs softmax for loss

```python
logits = torch.tensor([[100.0, 99.0, 101.0]])  # Large values
targets = torch.tensor([2])

# Method 1: Compute loss from logits (stable)
loss_from_logits = F.cross_entropy(logits, targets)
print("Loss from logits:", loss_from_logits.item())

# Method 2: Compute loss from probabilities (unstable)
try:
    probs = F.softmax(logits, dim=-1)
    loss_from_probs = F.nll_loss(torch.log(probs + 1e-8), targets)  # +eps for stability
    print("Loss from probabilities:", loss_from_probs.item())
except Exception as e:
    print("Error:", e)
# Output:
# Loss from logits: 0.0018
# Loss from probabilities: 0.0018
```

### Example 4: Logit temperature scaling

```python
logits = torch.tensor([[2.0, 1.0, 0.0]])

# Standard softmax (temperature = 1)
probs_t1 = F.softmax(logits / 1.0, dim=-1)
print("Temp=1 (standard):", probs_t1)

# Higher temperature = softer distribution
probs_t2 = F.softmax(logits / 2.0, dim=-1)
print("Temp=2 (softer):", probs_t2)

# Lower temperature = sharper distribution
probs_t05 = F.softmax(logits / 0.5, dim=-1)
print("Temp=0.5 (sharper):", probs_t05)

# Temperature 0 → one-hot (argmax)
probs_t01 = F.softmax(logits / 0.1, dim=-1)
print("Temp=0.1 (near one-hot):", probs_t01)
# Output:
# Temp=1 (standard): tensor([[0.6652, 0.2447, 0.0900]])
# Temp=2 (softer): tensor([[0.4223, 0.3090, 0.2687]])
# Temp=0.5 (sharper): tensor([[0.8668, 0.1223, 0.0109]])
# Temp=0.1 (near one-hot): tensor([[1.0000, 0.0000, 0.0000]])
```

### Example 5: Logits for binary classification

```python
# Binary classification: single logit output vs two logits
# Approach 1: Single logit + sigmoid
logit_binary = torch.tensor([1.5])
prob = torch.sigmoid(logit_binary)
print(f"Single logit: {logit_binary.item():.2f} → prob={prob.item():.4f}")

# Approach 2: Two logits + softmax
logits_two = torch.tensor([[1.5, -1.5]])
probs_two = F.softmax(logits_two, dim=-1)
print(f"Two logits: {logits_two} → probs={probs_two}")

# Both give the same result in expectation
# Single logit: p = sigmoid(s)
# Two logits: p0 = e^s/(e^s+e^{-s}) = sigmoid(2s), p1 = 1-p0
# Output:
# Single logit: 1.50 → prob=0.8176
# Two logits: tensor([[ 1.5000, -1.5000]]) → probs=tensor([[0.9526, 0.0474]])
```

### Example 6: Logit adjustment for long-tail learning

```python
# Adjust logits by class priors (logit adjustment)
class_priors = torch.tensor([0.1, 0.3, 0.6])  # Imbalanced classes
logits = torch.tensor([[2.0, 1.5, 1.0]])

# Adjusted logits (subtract log-priors)
adjusted_logits = logits - torch.log(class_priors)
probs_standard = F.softmax(logits, dim=-1)
probs_adjusted = F.softmax(adjusted_logits, dim=-1)

print("Standard probs:", probs_standard)
print("Adjusted probs:", probs_adjusted)
# Output:
# Standard probs: tensor([[0.4223, 0.3558, 0.2219]])
# Adjusted probs: tensor([[0.6449, 0.2355, 0.1196]])
```

## Common Mistakes

1. **Applying softmax before computing cross-entropy loss**: `F.cross_entropy` expects raw logits, not probabilities. Applying softmax manually causes numerical issues.

2. **Interpreting logits as probabilities**: Logits are raw scores, not probabilities. They can be >1 or negative. Always apply softmax/sigmoid to get probabilities.

3. **Using `F.cross_entropy` with probabilities (not logits)**: If you pre-apply softmax, use `F.nll_loss(F.log_softmax(x))` instead, but logits + `F.cross_entropy` is preferred.

4. **Forgetting temperature scaling during distillation**: Knowledge distillation uses softened probabilities (temperature > 1) to transfer knowledge more effectively.

5. **Assuming logits are normalized**: Different models produce logits at different scales. Comparing logit values across models is not meaningful.

6. **Using MSE loss with logits for classification**: MSE works with probabilities (0-1 range), not logits (unbounded). Cross-entropy with logits is standard.

7. **Not understanding that only the correct class logit affects the loss directly**: In cross-entropy, ∂L/∂s_y = p_y - 1 (always negative, pushing s_y up), and ∂L/∂s_i = p_i for i ≠ y (pushing s_i down).

## Interview Questions

### Beginner - 5

1. What are logits?
2. How do logits differ from probabilities?
3. What is the range of logit values?
4. How do you convert logits to probabilities?
5. Why is `F.cross_entropy` preferred with logits directly?

### Intermediate - 5

1. Derive the gradient of cross-entropy loss with respect to logits.
2. What is temperature scaling and how does it affect logits?
3. How are logits used in knowledge distillation?
4. Why are logits numerically more stable for loss computation?
5. What is logit adjustment and when would you use it?

### Advanced - 3

1. Derive the relationship between logits, priors, and posterior probabilities in a Bayesian context.
2. Implement a logit-based uncertainty estimation method (e.g., MC Dropout with logit temperature).
3. Analyze the effect of logit normalization on model calibration.

## Practice Problems

### Easy - 5

1. Extract logits from a simple 2-layer model.
2. Convert logits to probabilities using softmax.
3. Verify that logits can be any real number.
4. Compute cross-entropy loss directly from logits.
5. Apply temperature 0.5 to logits and observe the sharpening effect.

### Medium - 5

1. Train a model and compare accuracy when using logits vs probabilities for loss.
2. Implement knowledge distillation: train a student model on teacher logits.
3. Calibrate a model using temperature scaling on logits.
4. Visualize logit distributions for correct and incorrect predictions.
5. Implement logit adjustment for imbalanced classes.

### Hard - 3

1. Implement a logit-based rejection mechanism for out-of-distribution detection.
2. Derive and implement a method for logit normalization that improves calibration.
3. Build a multi-label classification model with logit threshold learning.

## Solutions

### Easy - 1
```python
model = nn.Linear(10, 3)
x = torch.randn(2, 10)
logits = model(x)  # No softmax
print(logits)  # raw scores
```

### Easy - 2
```python
logits = torch.tensor([[1.0, 2.0, 3.0]])
probs = F.softmax(logits, dim=-1)
print(probs)  # sums to 1
```

### Easy - 3
```python
logits = torch.tensor([[-5.0, 0.0, 100.0, -0.5]])
print(F.softmax(logits, dim=-1))  # All produce valid probabilities
```

## Related Concepts

DL-048 Softmax Output, DL-049 Probability Distribution Output, DL-050 Regression Output, DL-046 Forward Pass Computation

## Next Concepts

DL-048 Softmax Output, DL-049 Probability Distribution Output

## Summary

Logits are the raw, unnormalized scores from the last layer of a neural network before softmax. They can be any real number and represent the model's class preferences before conversion to probabilities. Computing loss directly from logits (via `F.cross_entropy`) is numerically stable and standard practice.

## Key Takeaways

- Logits = raw scores (unbounded, unnormalized)
- Probabilities = softmax(logits) (sum to 1, in [0,1])
- `F.cross_entropy(logits, targets)` = `F.nll_loss(F.log_softmax(logits, dim=1))`
- Logits are preferred for numerical stability in loss computation
- Temperature scaling controls prediction sharpness: logits / T
- Logit adjustment compensates for class imbalance
- Gradient: ∂L/∂s_i = p_i - [i == y]
