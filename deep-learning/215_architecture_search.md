# Concept: Neural Architecture Search

## Concept ID

DL-215

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Advanced CNN Architectures

## Learning Objectives

- Understand the concept and types of neural architecture search
- Implement basic NAS approaches in PyTorch
- Analyze the computational cost and trade-offs of NAS
- Compare different NAS strategies

## Prerequisites

DL-200 ResNet, DL-205 EfficientNet, DL-214 RegNet

## Definition

Neural Architecture Search (NAS) is the process of automating the design of neural network architectures, treating architecture design as an optimization problem where the goal is to find the architecture that maximizes performance on a given task.

## Intuition

Designing neural network architectures by hand requires expertise, intuition, and extensive trial and error. NAS asks: can we automate this? Like a robot architect that tries millions of floor plans to find the best one. Early NAS approaches used reinforcement learning or evolutionary algorithms to search over architectures, but these required thousands of GPU-days. Modern approaches use weight-sharing (DARTS, ENAS) to reduce cost, where all architectures share weights in a super-network. This reduces search cost from thousands of GPU-days to a few GPU-days.

## Why This Concept Matters

NAS discovered architectures (NASNet, EfficientNet, AmoebaNet) that achieve state-of-the-art results with better efficiency than manually designed ones. Understanding NAS provides insight into how the best architectures are created and opens the door to task-specific architecture design.

## Mathematical Explanation

**NAS pipeline**:
1. Define search space (set of possible architectures)
2. Define search strategy (how to explore the space)
3. Define performance estimation strategy (how to evaluate architectures)

**Search spaces**:
- **Macro search**: Full architecture topology
- **Micro search**: Cell-based (repeatable building blocks)
- **Layer-wise search**: Per-layer configurations

**Search strategies**:
- **Reinforcement learning**: Controller RNN samples architectures, trained with REINFORCE
- **Evolutionary**: Population of architectures, mutate and select
- **Gradient-based (DARTS)**: Continuous relaxation, learn architecture weights via gradient descent

**DARTS formulation**: Continuous relaxation of discrete choices:
$$\bar{o}^{(i,j)} = \sum_{o \in O} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in O} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

Where $\alpha$ are learnable architecture parameters.

## Code Examples

### Example 1: Simple Random Search (Baseline)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Define a simple search space
OPERATIONS = {
    'conv3': lambda C: nn.Conv2d(C, C, 3, padding=1),
    'conv5': lambda C: nn.Conv2d(C, C, 5, padding=2),
    'conv3_d2': lambda C: nn.Conv2d(C, C, 3, padding=2, dilation=2),
    'sep3': lambda C: nn.Conv2d(C, C, 3, padding=1, groups=C),
}

def random_architecture(num_ops=4):
    """Sample a random architecture."""
    ops = []
    for _ in range(num_ops):
        ops.append(torch.randint(0, len(OPERATIONS), (1,)).item())
    return ops

# Evaluate random architectures
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_loss = float('inf')
best_arch = None

for trial in range(50):
    arch = random_architecture(6)
    
    # Build model
    model = nn.Sequential()
    in_channels = 16
    list(OPERATIONS.values())[0](16)  # first conv (input)
    model.add_module('stem', nn.Conv2d(3, 16, 3, padding=1))
    
    for i, op_idx in enumerate(arch):
        op = list(OPERATIONS.values())[op_idx]
        model.add_module(f'conv_{i}', op(16))
        model.add_module(f'bn_{i}', nn.BatchNorm2d(16))
        model.add_module(f'relu_{i}', nn.ReLU())
    
    model.add_module('pool', nn.AdaptiveAvgPool2d(1))
    model.add_module('fc', nn.Flatten())
    
    # Quick evaluation (toy task)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randn(8, 10)
    loss = F.mse_loss(model(x).mean(dim=1, keepdim=True).expand(-1, 10), y)
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_arch = arch

print(f"Best architecture (random search): {best_arch}")
print(f"Best loss: {best_loss:.4f}")
```

### Example 2: DARTS-like Search Space

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MixedOp(nn.Module):
    """Mixed operation (continuous relaxation in DARTS)."""
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Identity() if i == 0 else nn.Conv2d(channels, channels, k, 
                padding=k//2) if k == 1 else nn.Conv2d(channels, channels, k,
                padding=k//2, groups=channels)
            for i, k in enumerate([1, 3, 5])
        ])
        # Architecture weights (learned)
        self.alpha = nn.Parameter(torch.zeros(3))
    
    def forward(self, x):
        # Weighted sum of operations (softmax over alphas)
        weights = F.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class NASCell(nn.Module):
    """A cell with learnable architecture."""
    def __init__(self, channels):
        super().__init__()
        self.op1 = MixedOp(channels)
        self.op2 = MixedOp(channels)
        self.op3 = MixedOp(channels)
    
    def forward(self, x):
        h1 = self.op1(x)
        h2 = self.op2(h1)
        h3 = self.op3(h2)
        return h3

class NASNet(nn.Module):
    """Simple network with NAS cells."""
    def __init__(self, num_cells=4, channels=32, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.cells = nn.ModuleList([
            NASCell(channels) for _ in range(num_cells)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))

model = NASNet(num_cells=4, num_classes=10)
x = torch.randn(4, 3, 32, 32)
out = model(x)

# Architecture weights
for name, param in model.named_parameters():
    if 'alpha' in name:
        print(f"{name}: {F.softmax(param, dim=0).detach().numpy()}")
```

### Example 3: Architecture Parameter Analysis

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Analyze how architecture weights evolve
model = NASNet(num_cells=2, num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Simulate training
x = torch.randn(16, 3, 32, 32)
y = torch.randint(0, 10, (16,))

for step in range(20):
    optimizer.zero_grad()
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        alphas = []
        for name, param in model.named_parameters():
            if 'alpha' in name:
                alphas.extend(F.softmax(param, dim=0).detach().numpy().tolist())
        print(f"Step {step:2d}: Architecture weights: "
              f"{['{:.3f}'.format(a) for a in alphas]}")
```

## Common Mistakes

1. **Insufficient search cost budget**: Full NAS requires thousands of GPU-hours.
2. **Search space too constrained**: Limits the potential of discovered architectures.
3. **Search space too large**: Infeasible to explore effectively.
4. **Poor performance estimation**: Training each architecture to convergence is too expensive; surrogate metrics needed.
5. **Transferability gap**: Architecture found on proxy task may not transfer to the target task.

## Interview Questions

### Beginner - 5
1. What is Neural Architecture Search?
2. What are the three components of NAS?
3. What is a search space?
4. What is a search strategy?
5. Why is NAS computationally expensive?

### Intermediate - 5
1. Compare reinforcement learning, evolutionary, and gradient-based NAS.
2. Explain weight sharing in NAS.
3. How does DARTS work?
4. What is the proxy task in NAS?
5. Compare NASNet with manually designed architectures.

### Advanced - 3
1. Design a NAS approach optimized for a specific hardware platform.
2. Analyze the search space requirements for finding novel architectures.
3. Compare one-shot NAS with conventional NAS approaches.

## Practice Problems

### Easy - 5
1. Define a search space with 3 operations.
2. Implement random architecture search.
3. Count possible architectures in a simple search space.
4. Load a NAS-found architecture from torchvision.
5. Compare NASNet with ResNet parameters.

### Medium - 5
1. Implement a simple evolutionary NAS.
2. Implement DARTS on CIFAR-10.
3. Compare convergence of different search strategies.
4. Visualize the architecture weights from DARTS.
5. Analyze the discretization process in DARTS.

### Hard - 3
1. Implement a full NAS pipeline with proxy task.
2. Design a multi-objective NAS (accuracy + latency).
3. Implement a weight-sharing NAS super-network.

## Solutions

### Easy - 1 Solution
```python
search_space = {
    'kernel_size': [3, 5, 7],
    'channels': [32, 64, 128],
    'depth': [1, 2, 3],
    'activation': ['relu', 'gelu', 'swish'],
}
print(f"Search space size: {3*3*3*3}")
```

## Related Concepts

DL-205 EfficientNet, DL-214 RegNet, DL-213 ConvNeXt, DL-206 MobileNet

## Next Concepts

DL-216 ImageNet Dataset

## Summary

Neural Architecture Search automates the design of neural network architectures, discovering high-performing models that often surpass manually designed ones. Modern NAS approaches use weight sharing for efficiency, with DARTS being a popular gradient-based method.

## Key Takeaways

- NAS automates architecture design optimization
- Three components: search space, search strategy, performance estimation
- Early NAS (RNN controller): 2000+ GPU-days
- Modern NAS (DARTS, weight-sharing): <10 GPU-days
- Discovered architectures: NASNet, EfficientNet, AmoebaNet
- Search space design is critical to NAS success
- Proxy tasks (small dataset, few epochs) reduce cost
- Transferability gap remains a challenge
- NAS is increasingly used for task-specific architecture design
- Combined with hardware-aware optimization for deployment
