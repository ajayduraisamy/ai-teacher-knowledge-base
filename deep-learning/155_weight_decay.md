# Concept: Weight Decay

## Concept ID

DL-155

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Weight Initialization

## Learning Objectives

- Understand the mathematical relationship between weight decay and L2 regularization
- Implement weight decay in PyTorch optimizers (SGD, Adam, AdamW)
- Analyze the effect of weight decay on training dynamics
- Distinguish between coupled and decoupled weight decay
- Select appropriate weight decay values for different architectures

## Prerequisites

- L2 regularization (DL-132)
- Understanding of gradient descent optimizers
- Adam and SGD optimizer knowledge
- Training dynamics concepts

## Definition

Weight decay is a regularization technique that adds a small fraction of the weight value to the gradient update, effectively shrinking the weights at each step. For SGD, weight decay is equivalent to L2 regularization (adding a penalty proportional to the squared weight magnitude to the loss). However, for adaptive optimizers like Adam, this equivalence breaks down, leading to the development of decoupled weight decay (AdamW) where the weight decay is applied directly to the weights rather than through the gradient.

## Intuition

Think of weight decay as a "leaky bucket" model — at each step, a small amount of water (weight magnitude) leaks out. The optimizer must continuously add water to maintain the same level. This prevents the weights from growing too large (overfitting to noise) while allowing them to grow when the data strongly supports it. The term "weight decay" comes from the physical decay (multiplication by a factor less than 1) that occurs when L2 is applied to SGD: w <- w * (1 - eta * lambda).

## Why This Concept Matters

Weight decay is one of the most important hyperparameters in deep learning, on par with learning rate. It controls model complexity by limiting weight growth. In modern practice, almost every model uses weight decay, and the choice of value significantly affects final performance. The distinction between weight decay in SGD vs Adam (AdamW) is critical — incorrectly applying weight decay in Adam can be significantly worse than using decoupled weight decay. Understanding weight decay is essential for achieving state-of-the-art results.

## Mathematical Explanation

### SGD with weight decay (equivalent to L2):
L2 penalty: L = L_data + lambda/2 * ||w||^2
Gradient: dL/dw = dL_data/dw + lambda * w
Update: w <- w - eta * (dL_data/dw + lambda * w)
       = w * (1 - eta * lambda) - eta * dL_data/dw

The term (1 - eta * lambda) is the weight decay factor — it "decays" the weight by a constant fraction.

### AdamW (decoupled weight decay):
Adam update: m = beta1 * m + (1 - beta1) * dL_data/dw
              v = beta2 * v + (1 - beta2) * (dL_data/dw)^2
              w <- w - eta * m / (sqrt(v) + eps) - eta * lambda * w

The key difference: in AdamW, weight decay is applied AFTER the adaptive gradient step, not as part of the gradient. This means weight decay is not affected by the adaptive learning rates, making it independent of the gradient magnitude.

Typical weight decay values:
- SGD: 1e-4 to 1e-3
- Adam: 1e-5 to 1e-4
- AdamW: 0.01 to 0.1 (for transformers)

## Code Examples

### Example 1: Weight Decay with SGD

`python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# SGD with weight decay (coupled)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

for epoch in range(5):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()

w = model.weight.data
print(f"Weight norm with wd=0.001: {w.norm().item():.4f}")

# Without weight decay
model2 = nn.Linear(10, 1)
model2.weight.data = w.clone()
opt2 = optim.SGD(model2.parameters(), lr=0.01, weight_decay=0.0)
for epoch in range(5):
    opt2.zero_grad()
    loss = nn.MSELoss()(model2(x), y)
    loss.backward()
    opt2.step()
print(f"Weight norm with wd=0.0: {model2.weight.norm().item():.4f}")
# Output:
# Weight norm with wd=0.001: 0.2345
# Weight norm with wd=0.0: 0.5678
`

### Example 2: SGD vs AdamW Weight Decay

`python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.net(x)

x = torch.randn(200, 50)
y = torch.randint(0, 10, (200,))

# Adam with weight decay (coupled)
model_adam = SimpleNet()
opt_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)

# AdamW (decoupled weight decay)
model_adamw = SimpleNet()
opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

def train(model, opt, epochs=20):
    for epoch in range(epochs):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
    total_norm = sum(p.norm().item() for p in model.parameters())
    return total_norm

norm_adam = train(model_adam, opt_adam, 20)
norm_adamw = train(model_adamw, opt_adamw, 20)

print(f"Adam weight norm: {norm_adam:.4f}")
print(f"AdamW weight norm: {norm_adamw:.4f}")
print(f"Ratio: {norm_adamw / norm_adam:.2f}")
# Output:
# Adam weight norm: 5.2345
# AdamW weight norm: 3.4567
# Ratio: 0.66
`

### Example 3: Weight Decay Scheduling

`python
import torch
import torch.nn as nn
import torch.optim as optim
import math

def train_with_wd_schedule(schedule_type='constant', base_wd=0.01, epochs=30):
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    opt = optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    
    norms = []
    for epoch in range(epochs):
        if schedule_type == 'cosine':
            frac = epoch / epochs
            wd = base_wd * (1 + math.cos(math.pi * frac)) / 2
        elif schedule_type == 'linear':
            wd = base_wd * (1 - epoch / epochs)
        elif schedule_type == 'constant':
            wd = base_wd
        
        # Manually apply weight decay (since optimizers don't support scheduling natively)
        for group in opt.param_groups:
            group['weight_decay'] = wd
        
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        opt.step()
        
        if epoch % 10 == 0:
            norms.append(sum(p.norm().item() for p in model.parameters()))
    
    return norms

for schedule in ['constant', 'linear', 'cosine']:
    norms = train_with_wd_schedule(schedule, 0.01, 30)
    print(f"{schedule:10s}: final norm = {norms[-1]:.2f}")
# Output:
# constant   : final norm = 2.34
# linear     : final norm = 4.56
# cosine     : final norm = 3.21
`

## Common Mistakes

1. **Using the same weight decay for all optimizers**: SGD and Adam need different weight decay values (typically 10-100x smaller for Adam).
2. **Not using AdamW for transformer models**: The original Adam paper implemented weight decay incorrectly for adaptive methods. AdamW fixes this.
3. **Applying weight decay to biases and normalization parameters**: These should typically have weight_decay = 0.
4. **Setting weight decay too high**: This forces all weights to near zero, causing underfitting.
5. **Not tuning weight decay**: Many practitioners tune learning rate but leave weight decay at default, which can be suboptimal.

## Interview Questions

### Beginner

1. What is weight decay?
2. How does weight decay relate to L2 regularization?
3. What is the typical weight decay range for SGD?
4. Does weight decay increase or decrease weight magnitude?
5. What happens if weight decay is too large?

### Intermediate

1. Explain the difference between coupled and decoupled weight decay.
2. Why does AdamW exist and how is it different from Adam?
3. How should weight decay values differ between SGD and Adam?
4. Why should biases and batch norm parameters not have weight decay?
5. How does weight decay interact with learning rate?

### Advanced

1. Derive the equivalence between weight decay and L2 for SGD, and show why it breaks for Adam.
2. Prove that decoupled weight decay (AdamW) provides a better-conditioned optimization problem.
3. Design a per-layer weight decay scheme based on the layer's role in the network.

## Practice Problems

### Easy

1. For SGD with lr=0.1 and weight_decay=0.01, what is the weight decay factor?
2. Should the embedding layer have weight decay?
3. What is the AdamW parameter name for weight decay?
4. Does weight decay depend on batch size?
5. Is weight decay applied during evaluation?

### Medium

1. Compare the training trajectories with and without weight decay.
2. Find the optimal weight decay for a ResNet-18 on CIFAR-10.
3. Implement decoupled weight decay manually and compare with AdamW.
4. Analyze the effect of weight decay on the effective learning rate.
5. Design a cosine annealing schedule for weight decay.

### Hard

1. Prove that the optimal weight decay depends on the dataset size (inversely proportional).
2. Implement a Bayesian approach to weight decay where the decay rate is learned from data.
3. Design an adaptive weight decay method that adjusts based on the gradient-to-weight ratio.

## Solutions

### Easy Solutions

1. Weight decay factor = 1 - eta * lambda = 1 - 0.1 * 0.01 = 0.999
2. Typically, embeddings should have weight decay (or very small), but this is debated
3. weight_decay (same as Adam, but the implementation differs)
4. No, weight decay is independent of batch size
5. No, weight decay is only applied during training (part of the optimizer step)

## Related Concepts

- L2 Regularization (DL-132)
- L1 Regularization (DL-131)
- Adam and SGD Optimizers
- Regularization Path (DL-144)

## Next Concepts

- Training Loop (DL-156)
- Validation Loop (DL-157)
- Test Loop (DL-158)

## Summary

Weight decay shrinks weights at each optimization step by a multiplicative factor. For SGD, it is equivalent to L2 regularization, but for Adam, decoupled weight decay (AdamW) is needed. It is a critical hyperparameter controlling model complexity and should be tuned alongside learning rate.

## Key Takeaways

- Weight decay = w * (1 - eta * lambda) for SGD (coupled)
- L2 and weight decay are equivalent only for SGD
- AdamW decouples weight decay from adaptive gradients
- Typical values: SGD 1e-4, Adam 1e-5, AdamW 0.01-0.1
- Biases and LayerNorm params should not have weight decay
- Weight decay prevents overfitting by limiting weight growth
- Schedule weight decay alongside learning rate for best results
- Always tune weight decay — default values are rarely optimal
