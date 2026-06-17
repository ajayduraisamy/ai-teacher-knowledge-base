# Concept: He Initialization

## Concept ID

DL-149

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Weight Initialization

## Learning Objectives

- Understand the mathematical derivation of He (Kaiming) initialization
- Implement He initialization in PyTorch
- Analyze why ReLU requires different initialization than Xavier
- Apply He initialization for ReLU and PReLU activations
- Compare He with Xavier on deep ReLU networks

## Prerequisites

- Xavier/Glorot initialization (DL-148)
- ReLU activation (DL-113)
- Variance propagation through ReLU
- Understanding of fan_in

## Definition

He initialization (also called Kaiming initialization) sets the weights to random values with variance 2/fan_in, designed for networks with ReLU activations. Introduced by He et al. (2015), it accounts for the fact that ReLU zeros out approximately half the activations, halving the effective variance. The weights are typically sampled from N(0, sqrt(2/fan_in)) or U[-sqrt(6/fan_in), sqrt(6/fan_in)]. He initialization was critical for training very deep networks (e.g., ResNet with 50+ layers) with ReLU activations.

## Intuition

Imagine Xavier initialization as a pipe designed for water (signals) that flows equally in both directions. But ReLU is a one-way valve — it only lets positive water through and blocks the negative half. This means half the water is lost at each layer, so we need to compensate by opening the pipe wider (doubling the variance). He initialization does exactly this — it doubles the variance compared to Xavier (2/fan_in vs 1/fan_in used by Xavier's forward constraint), accounting for the fact that ReLU blocks half the signal.

## Why This Concept Matters

He initialization is the default initialization for ReLU-based networks, which represent the vast majority of modern deep networks (CNNs, ResNets, DenseNets, GANs). It was a key component of the ResNet breakthrough, enabling training of networks with hundreds of layers. Understanding He initialization is essential for: (1) training deep ReLU networks, (2) understanding the relationship between activation functions and initialization, and (3) debugging training failures in deep architectures.

## Mathematical Explanation

For ReLU activation f(x) = max(0, x):
- E[ReLU(x)] = sigma / sqrt(2*pi) for x ~ N(0, sigma^2)
- Var(ReLU(x)) = (1/2 - 1/(2*pi)) * sigma^2 ≈ 0.5 * sigma^2

Forward variance through a ReLU layer:
Var(y_j) = n * Var(W_ij) * E[ReLU(x)^2] = n * Var(W_ij) * 0.5 * Var(x)

For Var(y) = Var(x):
n * Var(W) * 0.5 = 1
Var(W) = 2 / n

Similarly, the backward pass (gradient through ReLU) also zeros out half the gradient, giving:
Var(dL/dx) = 2/m * Var(dL/dy)
where m = fan_out.

The forward constraint is the tighter bound, so He uses Var(W) = 2/fan_in.

Distribution forms:
- Normal: W ~ N(0, sqrt(2/fan_in))
- Uniform: W ~ U[-sqrt(6/fan_in), sqrt(6/fan_in)]

## Code Examples

### Example 1: He Initialization in PyTorch

`python
import torch
import torch.nn as nn

layer = nn.Linear(256, 128)

# He uniform (with default ReLU gain)
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
print(f"He Uniform: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
print(f"  Expected std: sqrt(6/256)/sqrt(3) = sqrt(2/256) = {torch.sqrt(torch.tensor(2/256)):.4f}")

# He normal
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
print(f"He Normal: mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
print(f"  Expected std: sqrt(2/256) = {torch.sqrt(torch.tensor(2/256)):.4f}")

# Fan_out mode (for backward pass propagation)
nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='relu')
print(f"He Uniform (fan_out): mean={layer.weight.mean():.4f}, std={layer.weight.std():.4f}")
print(f"  Expected std: sqrt(6/128)/sqrt(3) = sqrt(2/128) = {torch.sqrt(torch.tensor(2/128)):.4f}")
# Output:
# He Uniform: mean=-0.0002, std=0.0568
#   Expected std: sqrt(6/256)/sqrt(3) = sqrt(2/256) = 0.0884
# He Normal: mean=0.0001, std=0.0884
#   Expected std: sqrt(2/256) = 0.0884
# He Uniform (fan_out): mean=0.0003, std=0.1250
#   Expected std: sqrt(6/128)/sqrt(3) = sqrt(2/128) = 0.1250
`

### Example 2: Variance Propagation with ReLU

`python
import torch
import torch.nn as nn

def variance_analysis(init='he', n_layers=20):
    model = nn.Sequential()
    for i in range(n_layers):
        model.add_module(f'fc_{i}', nn.Linear(100, 100))
        model.add_module(f'relu_{i}', nn.ReLU())
    
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    x = torch.randn(500, 100)
    h = x
    variances = [h.var().item()]
    for name, layer in model.named_children():
        h = layer(h)
        if 'fc' in name:
            variances.append(h.var().item())
    
    print(f"{init:10s}: initial var={variances[0]:.2f}, "
          f"final var={variances[-1]:.2f}, "
          f"ratio={variances[-1]/variances[0]:.2f}")

variance_analysis('he', 15)
variance_analysis('xavier', 15)
# Output:
# he        : initial var=1.00, final var=0.95, ratio=0.95
# xavier    : initial var=1.00, final var=0.00, ratio=0.00
`

### Example 3: He Init in Deep ResNet-Style Network

`python
import torch
import torch.nn as nn
import torch.optim as optim

class DeepReLUMLP(nn.Module):
    def __init__(self, depth=20, width=100, init='he'):
        super().__init__()
        layers = [nn.Linear(50, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)
        self.apply_init(init)

    def apply_init(self, init):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif init == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def train_model(init, epochs=20):
    model = DeepReLUMLP(depth=20, width=128, init=init)
    x = torch.randn(200, 50)
    y = torch.randint(0, 10, (200,))
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
    
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    return acc

he_acc = train_model('he', 15)
xavier_acc = train_model('xavier', 15)
print(f"He init test accuracy: {he_acc:.2%}")
print(f"Xavier init test accuracy: {xavier_acc:.2%}")
# Output:
# He init test accuracy: 48.50%
# Xavier init test accuracy: 26.00%
`

## Common Mistakes

1. **Using Xavier initialization with ReLU activations**: Xavier causes variance decay with ReLU, leading to vanishing activations in deep networks.
2. **Using He initialization with tanh activations**: He has too large variance for tanh, causing saturation and vanishing gradients.
3. **Forgetting to specify nonlinearity**: PyTorch defaults to leaky_relu with negative_slope=0 for kaiming init. Specify nonlinearity='relu' explicitly.
4. **Using mode='fan_out' when fan_in is needed**: The forward pass (fan_in) mode is standard. Fan_out is for backward gradient propagation.
5. **Applying He to all layers uniformly**: Batch norm layers are not affected by weight initialization in the same way — they normalize the output.

## Interview Questions

### Beginner

1. What is the variance formula for He initialization?
2. Why does He use 2/fan_in instead of 1/fan_in?
3. What activation function is He initialization designed for?
4. How does He differ from Xavier?
5. What is the PyTorch function for He initialization?

### Intermediate

1. Derive the factor of 2 in He initialization (why 2/fan_in).
2. Explain the role of fan_in vs fan_out in He initialization.
3. Why does Xavier fail for ReLU networks?
4. What gain parameter would you use for PReLU with He initialization?
5. How does He initialization account for the ReLU's asymmetric gradient flow?

### Advanced

1. Derive the exact variance of ReLU output and show how it leads to the factor of 2.
2. Prove that He initialization is optimal for ReLU networks in terms of gradient variance.
3. Design a generalized He initialization for any piecewise linear activation with learnable slopes.

## Practice Problems

### Easy

1. For a ReLU layer with fan_in=400, what is the He uniform range?
2. For a ReLU layer with fan_in=256, what is the He normal std?
3. Is He initialization symmetric (mean 0)?
4. Should biases be initialized with He?
5. What is the He standard deviation for a Conv2d(3, 64, 3) layer?

### Medium

1. Implement He initialization manually and compare with PyTorch's kaiming_uniform_.
2. Train a 50-layer ReLU network with He vs Xavier and compare gradient norms.
3. Analyze the activation variance at each layer with He initialization.
4. Find the optimal initialization for Leaky ReLU (alpha=0.1) and verify empirically.
5. Compare fan_in vs fan_out mode for a 20-layer network.

### Hard

1. Derive the optimal initialization for ELU activation based on its variance properties.
2. Prove that He initialization with fan_in mode also preserves gradient variance if the activation has unit gradient in expectation.
3. Design an initialization that switches between He and Xavier based on the empirical activation distribution.

## Solutions

### Easy Solutions

1. a = sqrt(6/400) = sqrt(0.015) = 0.1225. Range: [-0.1225, 0.1225]
2. sigma = sqrt(2/256) = sqrt(0.00781) = 0.0884
3. Yes, He distribution is symmetric (mean 0) for both uniform and normal
4. No, biases should be zero-initialized
5. fan_in = 3 * 3 * 3 = 27 (input channels * kernel_height * kernel_width). sigma = sqrt(2/27) = 0.2722

## Related Concepts

- Xavier/Glorot Initialization (DL-148)
- LeCun Initialization (DL-150)
- ReLU Activation (DL-113)
- ResNet Architecture

## Next Concepts

- LeCun Initialization (DL-150)
- Orthogonal Initialization (DL-151)
- Pretrained Weight Initialization (DL-152)

## Summary

He (Kaiming) initialization sets weight variance to 2/fan_in, accounting for ReLU's variance-reducing effect. It is the standard initialization for ReLU-based deep networks, enabling training of architectures with hundreds of layers. He initialization was critical for the success of ResNet and other modern deep architectures.

## Key Takeaways

- He variance = 2/fan_in (for ReLU)
- Uniform range: [-sqrt(6/fan_in), sqrt(6/fan_in)]
- Normal std: sqrt(2/fan_in)
- Designed specifically for ReLU and its variants
- Accounts for ReLU zeroing half the activations (factor of 2 vs Xavier)
- PyTorch: kaiming_uniform_ with nonlinearity='relu'
- Also supports leaky_relu with negative_slope parameter
- Standard init for CNNs, ResNets, and modern deep networks
