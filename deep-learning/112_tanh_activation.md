# Concept: Tanh Activation

## Concept ID

DL-112

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Activation Functions

## Learning Objectives

- Understand the mathematical formulation of the hyperbolic tangent activation
- Compare tanh with sigmoid in terms of output range and gradient properties
- Implement tanh activation in PyTorch for hidden layers
- Analyze the zero-centered property and its impact on training
- Identify scenarios where tanh outperforms sigmoid

## Prerequisites

- Sigmoid activation (DL-111)
- Basic understanding of gradient-based optimization
- Familiarity with activation function properties
- Calculus fundamentals

## Definition

The hyperbolic tangent (tanh) activation function is a scaled and shifted version of the sigmoid that outputs values in the range (-1, 1). It is defined as tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)). Unlike sigmoid, tanh is zero-centered, meaning its output is symmetric around 0. This property helps mitigate the problem of biased gradients during training. Tanh is continuous, differentiable, and monotonic, with a steeper slope near zero compared to sigmoid, which often leads to faster convergence in practice.

## Intuition

Think of tanh as a volume knob that can both increase (positive) and decrease (negative) the signal strength. While sigmoid only adds — it pushes values toward 1 — tanh can both amplify and dampen, naturally centering the data around zero. This zero-centered property is like having an equalizer that balances the signal before passing it to the next layer. In deep networks, tanh often helps because the activations oscillate around zero, allowing the next layer to receive both excitatory and inhibitory signals, which tends to produce more stable gradients than sigmoid.

## Why This Concept Matters

Tanh was the preferred activation before the rise of ReLU, especially in recurrent neural networks (RNNs) where its zero-centered property helps with gradient flow. While it still suffers from saturation at extreme values (tanh(3) ≈ 0.995, tanh(-3) ≈ -0.995), the gradient near zero is steeper than sigmoid, which means the network learns faster during early training. In modern practice, tanh is still used in RNN cells (LSTM, GRU), certain autoencoders, and any architecture where negative activations carry useful information. Understanding tanh is crucial for diagnosing gradient saturation in sequence models.

## Mathematical Explanation

The tanh function is defined as:

tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1)

Key properties:
- Output range: (-1, 1)
- tanh(0) = 0
- Odd function: tanh(-x) = -tanh(x)
- Derivative: tanh'(x) = 1 - tanh²(x) = sech²(x) = 4 / (e^x + e^(-x))²

The maximum derivative is 1 at x = 0, compared to sigmoid's maximum of 0.25. This means tanh can propagate larger gradients, reducing the vanishing gradient problem somewhat. However, the derivative still approaches 0 as |x| → ∞.

Relationship to sigmoid: tanh(x) = 2σ(2x) - 1, where σ is the sigmoid function.

The zero-centered property means that the expected value of tanh outputs over symmetric inputs is 0, which helps condition the optimization problem for the next layer.

## Code Examples

### Example 1: Basic Tanh Implementation

```python
import torch
import torch.nn as nn

x = torch.linspace(-10, 10, 5)
tanh = nn.Tanh()
y = tanh(x)

print("Input:", x)
print("Output:", y)
# Output:
# Input: tensor([-10.,  -5.,   0.,   5.,  10.])
# Output: tensor([-1.0000, -0.9999,  0.0000,  0.9999,  1.0000])
```

### Example 2: Tanh in a Feed-Forward Network

```python
import torch
import torch.nn as nn

class TanhNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = TanhNetwork()
sample = torch.randn(8, 20)
output = model(sample)
print("Output shape:", output.shape)
print("Output mean:", output.mean().item())
print("Output std:", output.std().item())
# Output:
# Output shape: torch.Size([8, 10])
# Output mean: 0.0123
# Output std: 0.8912
```

### Example 3: Gradient Comparison with Sigmoid

```python
import torch

x = torch.linspace(-5, 5, 100, requires_grad=True)

y_sigmoid = torch.sigmoid(x)
loss_sig = y_sigmoid.sum()
loss_sig.backward()
sig_grad = x.grad.clone()

x.grad.zero_()

y_tanh = torch.tanh(x)
loss_tanh = y_tanh.sum()
loss_tanh.backward()
tanh_grad = x.grad.clone()

print("Sigmoid max gradient:", sig_grad.max().item())
print("Tanh max gradient:", tanh_grad.max().item())
print("Gradient at x=3 - Sigmoid:", sig_grad[x.detach() == 3].item())
print("Gradient at x=3 - Tanh:", tanh_grad[x.detach() == 3].item())
# Output:
# Sigmoid max gradient: 0.25
# Tanh max gradient: 1.0
# Gradient at x=3 - Sigmoid: 0.0452
# Gradient at x=3 - Tanh: 0.0099
```

## Common Mistakes

1. **Using tanh for output layer probabilities**: Tanh outputs range from -1 to 1, not 0 to 1. Using it directly as probabilities is incorrect unless rescaled.
2. **Assuming tanh eliminates vanishing gradients entirely**: While better than sigmoid, tanh still saturates at large |x|, and the gradient vanishes to zero.
3. **Not normalizing inputs to the tanh active region**: Tanh saturates around |x| > 3. Inputs should be normalized (e.g., layer norm, batch norm) to stay in the linear regime.
4. **Confusing tanh with sigmoid scaling**: Remember that tanh(x) = 2σ(2x) - 1, so it's not just a simple rescaling of the output — the input is also scaled.
5. **Using tanh in very deep networks without skip connections**: Despite better gradient properties, very deep tanh networks still suffer from gradient attenuation.

## Interview Questions

### Beginner

1. What is the output range of the tanh function?
2. How does tanh differ from sigmoid in terms of zero-centering?
3. Write the formula for tanh.
4. What is tanh(0)?
5. Is tanh symmetric about the origin?

### Intermediate

1. Derive the derivative of tanh in terms of tanh itself.
2. Explain why zero-centered activations help with optimization.
3. Compare tanh and sigmoid: when would you prefer one over the other?
4. How does the tanh derivative at x=0 compare with sigmoid's derivative at x=0?
5. Why is tanh still used in RNNs despite the vanishing gradient problem?

### Advanced

1. Prove that tanh(x) = 2σ(2x) - 1 and discuss the implications for gradient scale.
2. Design a variant of tanh with a learnable scale parameter and analyze its gradient flow properties.
3. Explain the relationship between tanh saturation and the exploding/vanishing gradient problem in LSTMs.

## Practice Problems

### Easy

1. Compute tanh(0), tanh(1), tanh(-1) approximately.
2. Show that tanh(x) ∈ (-1, 1) for all real x.
3. Compute the derivative of tanh at x=0.
4. If tanh(x) = 0.5, estimate x.
5. Plot tanh and its derivative on the same axes.

### Medium

1. Prove that tanh is an odd function and sigmoid is neither odd nor even.
2. Implement a numerically stable tanh without using torch.tanh.
3. Train a 5-layer network on MNIST with tanh and sigmoid activations and compare convergence curves.
4. Derive the relationship between tanh and sigmoid and explain the factor of 2.
5. Analyze the gradient norm at each layer of a 10-layer tanh network.

### Hard

1. Implement a custom tanh with a learnable slope and bias, and test on a regression problem.
2. Derive the second derivative of tanh and analyze its curvature properties near zero.
3. Propose a method to prevent tanh saturation in a 50-layer network without using batch normalization.

## Solutions

### Easy Solutions

1. tanh(0) = 0, tanh(1) = (e - 1/e)/(e + 1/e) ≈ 0.7616, tanh(-1) ≈ -0.7616
2. As x → ∞, e^x dominates, so tanh → e^x/e^x = 1. As x → -∞, e^(-x) dominates, so tanh → -e^x/e^x = -1. For finite x, the numerator is always smaller in magnitude than the denominator.
3. tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
4. ln((1+0.5)/(1-0.5))/2 = ln(3)/2 ≈ 0.5493
5. The derivative peaks at x=0 with value 1 and decays symmetrically to 0 at both ends.

## Related Concepts

- Sigmoid Activation (DL-111)
- ReLU Activation (DL-113)
- LSTM and GRU Cells
- Vanishing Gradient Problem

## Next Concepts

- ReLU Activation (DL-113)
- Leaky ReLU (DL-114)
- ELU Activation (DL-116)

## Summary

Tanh is a zero-centered activation function that maps inputs to (-1, 1). It offers steeper gradients than sigmoid near zero and symmetric output distribution, which improves conditioning for optimization. However, it still saturates at extreme values, limiting its depth before the vanishing gradient problem appears. Tanh remains relevant in RNN architectures and situations where negative activations provide meaningful signal.

## Key Takeaways

- Tanh maps ℝ → (-1, 1) and is zero-centered, unlike sigmoid
- Derivative peaks at 1 (vs 0.25 for sigmoid), enabling stronger gradient flow
- The function saturates for |x| > 3, limiting depth without normalization
- Tanh is preferred over sigmoid for hidden layers when using saturating activations
- Used extensively in LSTM and GRU gates
- Relationship: tanh(x) = 2σ(2x) - 1
