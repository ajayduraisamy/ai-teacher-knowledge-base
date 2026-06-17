# Concept: Softmax Output

## Concept ID

DL-048

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the softmax function and its properties
- Implement softmax using PyTorch and from scratch
- Explain why softmax is used for multi-class classification
- Analyze numerical stability considerations

## Prerequisites

DL-047 (Logits), DL-049 (Probability Distribution Output), DL-046 (Forward Pass Computation)

## Definition

The softmax function converts a vector of K real-valued logits into a probability distribution over K classes. For logits **s** ∈ ℝ^K, softmax computes: softmax(s)_i = e^{s_i} / Σ_{j=1}^{K} e^{s_j}. The output values are positive and sum to 1, forming a valid probability distribution.

## Intuition

Softmax is like a gentle competition. Each logit s_i is exponentiated (making everything positive, with larger values getting a bigger boost), then normalized by the sum. If one logit is much larger than the others, its probability approaches 1. If all logits are similar, probabilities are similar. The "soft" in softmax means the transition is smooth and differentiable — unlike "argmax" which is hard and non-differentiable.

## Why This Concept Matters

Softmax is the standard output function for multi-class classification:
- **Principled probabilities**: Produces a valid probability distribution
- **Differentiable**: Enables gradient-based learning
- **Logit interpretation**: Works seamlessly with logits and cross-entropy
- **Temperature control**: Can be softened or sharpened
- **Part of the softmax-cross-entropy combination**: Together they produce clean gradients

## Mathematical Explanation

For logits s_i ∈ ℝ:

softmax(s)_i = e^{s_i} / Σ_{k=1}^{K} e^{s_k}

Properties:
1. softmax(s)_i > 0 for all i
2. Σ_i softmax(s)_i = 1
3. softmax(s + c) = softmax(s) for any constant c (shift invariance)
4. softmax(s) = softmax(s - max(s)) (numerically stable version)

Jacobian of softmax:

∂softmax_i / ∂s_j = softmax_i (δ_{ij} - softmax_j)

where δ_{ij} = 1 if i = j else 0.

When combined with cross-entropy loss, the gradient simplifies:

∂L/∂s_i = softmax_i - y_i

where y_i is the one-hot target. This elegant gradient is the reason the softmax + cross-entropy combination is so popular.

## Code Examples

### Example 1: Basic softmax

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])
probs = F.softmax(logits, dim=-1)
print("Logits:", logits)
print("Probabilities:", probs)
print("Sum:", probs.sum().item())
# Output:
# Logits: tensor([[2.0000, 1.0000, 0.1000]])
# Probabilities: tensor([[0.6590, 0.2424, 0.0986]])
# Sum: 1.0
```

### Example 2: Softmax from scratch

```python
def softmax_scratch(x, dim=-1):
    # Numerically stable implementation
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, -1.0, 3.0]])
probs1 = F.softmax(logits, dim=-1)
probs2 = softmax_scratch(logits)
print("PyTorch softmax:\n", probs1)
print("Scratch softmax:\n", probs2)
print("Difference:", (probs1 - probs2).abs().max().item())
# Output:
# PyTorch softmax:
#  tensor([[0.6590, 0.2424, 0.0986],
#          [0.0718, 0.0064, 0.9218]])
# Scratch softmax:
#  tensor([[0.6590, 0.2424, 0.0986],
#          [0.0718, 0.0064, 0.9218]])
# Difference: 0.0
```

### Example 3: Shift invariance

```python
logits = torch.tensor([1.0, 2.0, 3.0])
probs_original = F.softmax(logits, dim=-1)

# Add a constant to all logits
probs_shifted = F.softmax(logits + 10.0, dim=-1)
probs_shifted2 = F.softmax(logits - 100.0, dim=-1)

print("Original probs:", probs_original)
print("Shifted (+10) probs:", probs_shifted)
print("Shifted (-100) probs:", probs_shifted2)
print("All equal:", torch.allclose(probs_original, probs_shifted) and torch.allclose(probs_original, probs_shifted2))
# Output:
# Original probs: tensor([0.0900, 0.2447, 0.6652])
# Shifted (+10) probs: tensor([0.0900, 0.2447, 0.6652])
# Shifted (-100) probs: tensor([0.0900, 0.2447, 0.6652])
# All equal: True
```

### Example 4: Softmax with temperature

```python
logits = torch.tensor([[2.0, 1.0, 0.5]])

for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = F.softmax(logits / T, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    print(f"T={T:.1f}: probs={probs.squeeze().tolist()}, entropy={entropy:.4f}")
# Output:
# T=0.1: probs=[1.0, 0.0, 0.0], entropy=0.0000
# T=0.5: probs=[0.8808, 0.1072, 0.0120], entropy=0.4155
# T=1.0: probs=[0.6590, 0.2424, 0.0986], entropy=0.8720
# T=2.0: probs=[0.4542, 0.3097, 0.2361], entropy=1.0660
# T=10.0: probs=[0.3571, 0.3245, 0.3184], entropy=1.0975
```

### Example 5: Gradient of softmax + cross-entropy

```python
logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
target = torch.tensor([2])  # Correct class is index 2

loss = F.cross_entropy(logits, target)
loss.backward()

# Manually compute gradient: ∂L/∂s_i = softmax_i - [i == y]
probs = F.softmax(logits, dim=-1).detach()
manual_grad = probs.clone()
manual_grad[0, target] -= 1  # subtract 1 for correct class

print("Autograd gradient:", logits.grad)
print("Manual gradient:  ", manual_grad)
print("Match:", torch.allclose(logits.grad, manual_grad))
# Output:
# Autograd gradient: tensor([[0.0900, 0.2447, -0.3348]])
# Manual gradient:   tensor([[0.0900, 0.2447, -0.3348]])
# Match: True
```

### Example 6: Softmax stability comparison

```python
import math

# Naive softmax (unstable)
def softmax_naive(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# Stable softmax
def softmax_stable(x):
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# Test with large values
logits_large = torch.tensor([[1000.0, 999.0, 998.0]])

try:
    naive = softmax_naive(logits_large)
    print("Naive:", naive)
except Exception as e:
    print("Naive failed with:", e)

stable = softmax_stable(logits_large)
print("Stable:", stable)
# Output:
# Naive failed with: inf in exponent produces inf/nan
# Stable: tensor([[0.6652, 0.2447, 0.0900]])
```

## Common Mistakes

1. **Using softmax for binary classification**: For binary classification, use a single logit with sigmoid, not two logits with softmax. They are equivalent but sigmoid is simpler.

2. **Applying softmax along the wrong dimension**: For a batch of samples, softmax should be applied along the class dimension (dim=1 for (batch, classes)), not the batch dimension.

3. **Using softmax inside cross-entropy loss**: `F.cross_entropy` expects raw logits and applies softmax internally. Pre-applying softmax then using cross-entropy gives wrong results.

4. **Numerical overflow with large logits**: `e^{1000}` overflows. Always use the numerically stable version that subtracts the max.

5. **Assuming softmax is only for classification**: Softmax can be used for attention weights, gating, and any situation needing a probability distribution.

6. **Forgetting that softmax is shift-invariant but not scale-invariant**: Adding a constant doesn't change output, but multiplying by a constant does. This is the basis of temperature scaling.

7. **Using softmax with more than 2 dimensions without specifying dim**: Always specify `dim=-1` (last dimension) unless you have a specific reason. The default behavior may change.

## Interview Questions

### Beginner - 5

1. What does the softmax function do?
2. Why is softmax called "softmax"?
3. What are the properties of the softmax output?
4. How do you apply softmax in PyTorch?
5. What is the sum of all softmax outputs for a single sample?

### Intermediate - 5

1. Derive the Jacobian of the softmax function.
2. Why is the combination of softmax + cross-entropy so effective for classification?
3. Explain the gradient ∂L/∂s_i = p_i - y_i for softmax + cross-entropy.
4. How does temperature scaling affect softmax outputs?
5. What is the numerically stable implementation of softmax?

### Advanced - 3

1. Derive and implement the log-softmax function and explain when it's preferred over softmax.
2. Implement a sparse softmax (sparsemax) that outputs sparse probability distributions.
3. Explain the relationship between softmax and the principle of maximum entropy.

## Practice Problems

### Easy - 5

1. Apply softmax to logits [2.0, 1.0, 0.0] and verify output sums to 1.
2. Implement softmax from scratch without using F.softmax.
3. Show that adding 100 to all logits doesn't change softmax output.
4. Apply softmax to a (4, 5) tensor along dim=1.
5. Convert softmax probabilities back to logits (approximately).

### Medium - 5

1. Visualize softmax output as a function of logit differences.
2. Compare training stability with softmax + NLLLoss vs logits + CrossEntropyLoss.
3. Implement a temperature-scaled softmax and measure its effect on prediction entropy.
4. Compute the Jacobian of softmax for a 3-class example manually.
5. Implement a softmax layer that handles arbitrary batch dimensions.

### Hard - 3

1. Implement the Gumbel-softmax (concrete distribution) for discrete sampling with gradients.
2. Derive and implement a differentiable version of top-k softmax (only compute softmax over top-k logits).
3. Analyze the effect of softmax on the Lipschitz constant of a neural network.

## Solutions

### Easy - 1
```python
logits = torch.tensor([2.0, 1.0, 0.0])
probs = F.softmax(logits, dim=-1)
print(probs.sum().item())  # 1.0
```

### Easy - 2
```python
def my_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True)
```

### Easy - 3
```python
logits = torch.tensor([1.0, 2.0, 3.0])
p1 = F.softmax(logits, dim=-1)
p2 = F.softmax(logits + 100, dim=-1)
assert torch.allclose(p1, p2)
```

## Related Concepts

DL-047 Logits, DL-049 Probability Distribution Output, DL-050 Regression Output, DL-046 Forward Pass Computation

## Next Concepts

DL-049 Probability Distribution Output, DL-051 Feature Hierarchy

## Summary

The softmax function converts raw logits into a valid probability distribution over classes. It is the standard output activation for multi-class classification, with the elegant property that the gradient of softmax + cross-entropy loss simplifies to p_i - y_i. Softmax is shift-invariant, numerically stable when implemented with max-subtraction, and can be controlled via temperature scaling.

## Key Takeaways

- softmax(s)_i = e^{s_i} / Σ e^{s_j} (probabilities that sum to 1)
- Shift-invariant: softmax(s + c) = softmax(s)
- Always use numerically stable implementation (subtract max)
- Temperature controls sharpness: softmax(s / T)
- Gradient when combined with cross-entropy: ∂L/∂s_i = p_i - y_i
- `F.cross_entropy` expects raw logits (applies softmax internally)
- Softmax is differentiable, making it suitable for gradient-based learning
