# Concept: Parametric ReLU

## Concept ID

DL-115

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Activation Functions

## Learning Objectives

- Understand the formulation of PReLU with learnable negative slope
- Implement PReLU in PyTorch with channel-wise and layer-wise parameterization
- Analyze the training dynamics of learnable activation slopes
- Compare PReLU performance vs Leaky ReLU on benchmark tasks
- Diagnose when learned α converges to useful values vs collapsing to zero

## Prerequisites

- Leaky ReLU (DL-114)
- ReLU activation (DL-113)
- Understanding of backpropagation through activation functions
- Experience with hyperparameter optimization

## Definition

Parametric ReLU (PReLU) is an activation function that generalizes Leaky ReLU by making the negative slope α a learnable parameter. It is defined identically to Leaky ReLU — f(x) = max(αx, x) — but α is optimized jointly with the network weights via backpropagation. PReLU introduces a very small number of additional parameters (one per channel or one per layer) while allowing the network to learn the optimal negative slope for its specific task and data distribution.

## Intuition

Think of PReLU as having a learnable "leakiness" knob for each channel or layer. Instead of manually tuning the negative slope α, the network itself determines how much information should pass through negative activations. For some layers, the optimal α might be close to 0 (nearly standard ReLU), while for others it might be larger (more leaky). The gradient signal from the loss function naturally guides α to its optimal value. This is particularly powerful in convolutional networks where different feature channels may benefit from different negative slopes — one channel might encode edge detectors where negative information is useful, while another might benefit from hard sparsity.

## Why This Concept Matters

PReLU represents a significant philosophical shift in activation function design — instead of hand-engineering the function, let the network learn it. This approach, introduced by He et al. (2015), was part of the ResNet breakthrough and consistently improved accuracy over ReLU and Leaky ReLU on ImageNet. PReLU also paved the way for other learnable activations (Swish, Mish) and the broader trend of differentiable architecture search. The key insight is that the activation function is just another layer parameter that can be optimized, and the marginal computational cost is negligible.

## Mathematical Explanation

PReLU is defined as:

f(y_i) = y_i if y_i > 0, else α_i * y_i

where y_i is the input at channel i, and α_i is the learnable slope for that channel.

Two parameterization variants:
- **Channel-wise**: One α per input channel (most common)
- **Layer-wise**: Shared α across all channels (fewer parameters)

The gradient with respect to α is:
∂f / ∂α = 0 if x > 0, else x (gradient flows only through negative inputs)

The gradient with respect to the input x is:
∂f / ∂x = 1 if x > 0, else α

This means α receives gradient updates only from negative activations, proportional to the magnitude of those activations.

The update rule for α is:
Δα = η * Σ(∂L / ∂f * ∂f / ∂α) = η * Σ(∂L / ∂f * x) for x ≤ 0

No weight decay is typically applied to α to prevent it from being forced to 0.

## Code Examples

### Example 1: Basic PReLU Usage

```python
import torch
import torch.nn as nn

x = torch.randn(4, 64, 7, 7)  # batch=4, channels=64, spatial=7x7
prelu = nn.PReLU(num_parameters=64)  # one alpha per channel
y = prelu(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Alpha values:", prelu.weight.shape)
print("Initial alpha:", prelu.weight.data[:5].tolist())
# Output:
# Input shape: torch.Size([4, 64, 7, 7])
# Output shape: torch.Size([4, 64, 7, 7])
# Alpha values: torch.Size([64])
# Initial alpha: [0.25, 0.25, 0.25, 0.25, 0.25]
```

### Example 2: Training with PReLU

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.prelu1 = nn.PReLU(256)
        self.fc2 = nn.Linear(256, 128)
        self.prelu2 = nn.PReLU(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        x = self.fc3(x)
        return x

model = PReLUModel()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

sample = torch.randn(16, 784)
target = torch.randint(0, 10, (16,))

optimizer.zero_grad()
output = model(sample)
loss = nn.CrossEntropyLoss()(output, target)
loss.backward()
optimizer.step()

print("Alpha values after one step (first 5):", 
      model.prelu1.weight.data[:5].tolist())
# Output:
# Alpha values after one step (first 5):
# [0.2498, 0.2501, 0.2499, 0.2502, 0.2500]
```

### Example 3: Channel-wise vs Layer-wise PReLU

```python
import torch
import torch.nn as nn

# Channel-wise PReLU
prelu_channel = nn.PReLU(num_parameters=64)

# Layer-wise PReLU (shared alpha)
prelu_layer = nn.PReLU(num_parameters=1)

x = torch.randn(2, 64, 8, 8)

y_channel = prelu_channel(x)
y_layer = prelu_layer(x)

print("Channel-wise PReLU params:", prelu_channel.weight.numel())
print("Layer-wise PReLU params:", prelu_layer.weight.numel())
print("Channel alphas (first 5):", prelu_channel.weight.data[:5].view(-1).tolist())
print("Layer alpha:", prelu_layer.weight.data.item())
# Output:
# Channel-wise PReLU params: 64
# Layer-wise PReLU params: 1
# Channel alphas (first 5): [0.25, 0.25, 0.25, 0.25, 0.25]
# Layer alpha: 0.25
```

## Common Mistakes

1. **Applying weight decay to α**: Weight decay pushes α toward 0, which would make PReLU behave like standard ReLU. α should typically be excluded from weight decay.
2. **Using too many α parameters**: Channel-wise in very deep networks adds many parameters. Layer-wise is often sufficient.
3. **Initializing α too large**: Values > 1 can make the activation function non-monotonic and harm training.
4. **Not monitoring α convergence**: α that converges to exactly 0 or very large values indicates optimization problems.
5. **Using PReLU in very small networks**: The additional expressivity may not be justified by the parameter cost in tiny models.

## Interview Questions

### Beginner

1. How does PReLU differ from Leaky ReLU?
2. What is the default initial value for α in PyTorch's PReLU?
3. Is α in PReLU learned or fixed?
4. How many parameters does PReLU add per channel?
5. Can α become negative during training?

### Intermediate

1. Derive the gradient update rule for α in PReLU.
2. Explain why weight decay should not be applied to α.
3. Compare channel-wise vs layer-wise PReLU: when would you use each?
4. How does PReLU affect the number of dead neurons compared to ReLU?
5. What happens if α converges to 0 during training?

### Advanced

1. Prove that PReLU networks are universal approximators and analyze how the learnable slope affects the approximation rate.
2. Design a regularization scheme for α that prevents it from collapsing to extreme values.
3. Analyze the gradient covariance of PReLU compared to ReLU and Leaky ReLU in deep networks initialized with He initialization.

## Practice Problems

### Easy

1. How many additional parameters does channel-wise PReLU add to a Conv2d(3, 64, 3) layer?
2. What is the gradient of PReLU at x = -2 when α = 0.5?
3. What is the default initial α in PyTorch's nn.PReLU?
4. Is PReLU monotonic for all α values?
5. How does PReLU's computational cost compare to ReLU?

### Medium

1. Implement PReLU from scratch and verify gradients match PyTorch's implementation.
2. Train a ResNet-18 on CIFAR-10 with ReLU and PReLU and compare accuracy.
3. Analyze the learned α values in each layer of a trained PReLU network. Which layers tend to have larger α?
4. Design an experiment to determine whether α should be shared across layers or per-channel.
5. Implement PReLU with L1 regularization on α and analyze the sparsity of learned alphas.

### Hard

1. Derive the optimal α for a single PReLU neuron under Gaussian input distribution and MSE loss.
2. Implement a "learned activation" where each neuron has its own α but with a shared sparsity prior, and analyze the training dynamics.
3. Prove convergence bounds for gradient descent on α in a two-layer PReLU network.

## Solutions

### Easy Solutions

1. 64 additional parameters (one per output channel)
2. ∂f/∂x = α = 0.5, ∂f/∂α = x (for x ≤ 0) = -2
3. 0.25
4. Yes, as long as α ≥ 0. If α < 0, the function is no longer monotonic.
5. Negligible difference — one extra learnable parameter update per channel.

## Related Concepts

- Leaky ReLU (DL-114)
- ReLU Activation (DL-113)
- Learnable Activation Functions
- Swish/SiLU (DL-119)

## Next Concepts

- ELU Activation (DL-116)
- SELU Activation (DL-117)
- GELU Activation (DL-118)

## Summary

Parametric ReLU (PReLU) extends Leaky ReLU by making the negative slope a learnable parameter optimized via backpropagation. It introduces minimal additional parameters (one per channel or layer) while allowing the network to adapt its activation function to the data. PReLU consistently improves accuracy over fixed-slope variants and was a key component in the ResNet breakthrough.

## Key Takeaways

- PReLU learns the negative slope α via gradient descent during training
- Two variants: channel-wise (one α per channel) and layer-wise (shared α)
- Initialized to 0.25 by default; no weight decay should be applied to α
- Gradient for α only flows through negative activations
- Minimal parameter overhead with measurable accuracy improvements
- Represents the paradigm shift toward learnable activation functions
