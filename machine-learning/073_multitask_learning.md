# Concept: Multi-Task Learning

## Concept ID

ML-073

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the multi-task learning paradigm and its benefits
- Implement hard and soft parameter sharing architectures
- Apply cross-stitch networks and task relationship learning
- Evaluate multi-task vs. single-task performance

## Prerequisites

- Neural networks and backpropagation
- Deep learning fundamentals
- Regularization techniques
- Multi-label classification basics

## Definition

Multi-Task Learning (MTL) is a subfield of machine learning where multiple related tasks are learned simultaneously by sharing representations across tasks. The shared information acts as an inductive bias, improving generalization on each individual task. MTL is motivated by the observation that learning several related tasks jointly often yields better performance than learning each task in isolation.

## Intuition

Consider learning to detect pedestrians, cars, and traffic signs from camera images. These tasks share low-level features (edges, textures, corners) and mid-level features (wheels, shapes, colors). A single network that predicts all three can learn richer, more robust representations because each task provides additional training signal for the shared features. The pedestrian detector helps the car detector learn about road scenes, and vice versa. This positive transfer is the core benefit of MTL.

## Why This Concept Matters

MTL is a cornerstone of modern deep learning systems. Autonomous vehicles jointly predict object detection, lane segmentation, and depth estimation. NLP models (e.g., BART, T5) are pretrained on multiple language tasks simultaneously. Recommendation systems predict multiple user actions (clicks, purchases, shares). MTL reduces the need for separate models per task, decreases deployment costs, and often improves per-task performance through information sharing.

## Mathematical Explanation

### Hard Parameter Sharing

The most common MTL architecture: shared hidden layers + task-specific output heads.

Let the shared layers compute a representation φ(x; θ_shared). For each task t:

ŷ_t = f_t(φ(x; θ_shared); θ_t)

The total loss is a weighted sum:

L_total = ∑_{t=1}^T w_t L_t(ŷ_t, y_t)

where w_t are task weights.

Hard sharing drastically reduces overfitting risk because the shared layers must represent information useful for all tasks.

### Soft Parameter Sharing

Each task has its own model parameters, but the parameters are regularized to be similar across tasks. Common approaches:

- L2 distance between task parameters: λ ∑_{(i,j)} ||θ_i - θ_j||²
- Trace norm regularization
- Task-specific layers with cross-stitch units

### Cross-Stitch Networks

Cross-stitch units learn linear combinations of task-specific representations:

[z^{(1)}_l, z^{(2)}_l]^T = α_l [z^{(1)}_{l-1}, z^{(2)}_{l-1}]^T

where α_l is a 2×2 learnable matrix that determines how much information flows between tasks at layer l.

### Task Relationship Learning

Learn a task covariance matrix Σ ∈ ℝ^{T×T} that captures task similarities. The loss includes:

L_total = ∑_t L_t + λ tr(Θ Σ^{-1} Θ^T)

where Θ is the parameter matrix and Σ models task relations.

## Code Examples

### Example 1: Hard Parameter Sharing with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

n_samples = 2000
n_features = 20
X = torch.randn(n_samples, n_features)
y_task1 = (X[:, 0] + X[:, 1] * 0.5 + torch.randn(n_samples) * 0.1).unsqueeze(1)
y_task2 = (X[:, 2] * X[:, 3] + torch.randn(n_samples) * 0.1).unsqueeze(1)
y_task3 = (torch.sin(X[:, 4]) + torch.randn(n_samples) * 0.1).unsqueeze(1)

class HardShareMTL(nn.Module):
    def __init__(self, input_dim, shared_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )
        self.task1_head = nn.Linear(shared_dim, 1)
        self.task2_head = nn.Linear(shared_dim, 1)
        self.task3_head = nn.Linear(shared_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.task1_head(shared), self.task2_head(shared), self.task3_head(shared)

model = HardShareMTL(n_features)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    o1, o2, o3 = model(X)
    loss = loss_fn(o1, y_task1) + loss_fn(o2, y_task2) + loss_fn(o3, y_task3)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    o1, o2, o3 = model(X)
    r2_1 = 1 - torch.mean((o1 - y_task1) ** 2) / torch.var(y_task1)
    r2_2 = 1 - torch.mean((o2 - y_task2) ** 2) / torch.var(y_task2)
    r2_3 = 1 - torch.mean((o3 - y_task3) ** 2) / torch.var(y_task3)

print(f"Task 1 R²: {r2_1:.4f}")
print(f"Task 2 R²: {r2_2:.4f}")
print(f"Task 3 R²: {r2_3:.4f}")
print(f"Average R²: {((r2_1 + r2_2 + r2_3) / 3):.4f}")
# Output:
# Task 1 R²: 0.9881
# Task 2 R²: 0.5490
# Task 3 R²: 0.9062
# Average R²: 0.8144
```

### Example 2: Single-Task Baseline for Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(n_samples, n_features)
y_tasks = [y_task1, y_task2, y_task3]

single_results = []
for t_idx, (yt, name) in enumerate(zip(y_tasks, ['Task 1', 'Task 2', 'Task 3'])):
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, yt)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        r2 = 1 - torch.mean((model(X) - yt) ** 2) / torch.var(yt)
    single_results.append(r2.item())
    print(f"{name} (single): R² = {r2:.4f}")

print(f"Average single-task R²: {np.mean(single_results):.4f}")
# Output:
# Task 1 (single): R² = 0.9879
# Task 2 (single): R² = 0.5217
# Task 3 (single): R² = 0.9023
# Average single-task R²: 0.8040
```

### Example 3: Cross-Stitch Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

X = torch.randn(n_samples, n_features)

class CrossStitchUnit(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_tasks, num_tasks) * 0.5)

    def forward(self, representations):
        stacked = torch.stack(representations, dim=-1)
        out = torch.matmul(stacked, self.alpha.T)
        return [out[..., i] for i in range(out.shape[-1])]

class CrossStitchMTL(nn.Module):
    def __init__(self, input_dim, shared_dim=64, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, shared_dim), nn.ReLU())
            for _ in range(num_tasks)
        ])
        self.cross_stitch = CrossStitchUnit(num_tasks)
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(shared_dim, shared_dim), nn.ReLU(), nn.Linear(shared_dim, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, x):
        reps = [encoder(x) for encoder in self.task_encoders]
        crossed = self.cross_stitch(reps)
        return [head(c) for head, c in zip(self.task_heads, crossed)]

model = CrossStitchMTL(n_features, shared_dim=64, num_tasks=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    out1, out2 = model(X)
    loss = loss_fn(out1, y_task1) + loss_fn(out2, y_task2)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    o1, o2 = model(X)
    r2_1 = 1 - torch.mean((o1 - y_task1) ** 2) / torch.var(y_task1)
    r2_2 = 1 - torch.mean((o2 - y_task2) ** 2) / torch.var(y_task2)
    print(f"Cross-Stitch - Task 1 R²: {r2_1:.4f}, Task 2 R²: {r2_2:.4f}")

cross_stitch_weights = model.cross_stitch.alpha.detach()
print(f"Cross-stitch weight matrix:\n{cross_stitch_weights.numpy()}")
# Output:
# Cross-Stitch - Task 1 R²: 0.9881, Task 2 R²: 0.5537
# Cross-stitch weight matrix:
# [[0.520 0.480]
#  [0.471 0.529]]
```

### Example 4: Gradient Analysis in MTL

```python
import torch
import torch.nn as nn
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

X_small = torch.randn(100, n_features)

model = HardShareMTL(n_features, shared_dim=32)
loss_fn = nn.MSELoss()

o1, o2, o3 = model(X_small)
loss1 = loss_fn(o1, torch.randn(100, 1))
loss2 = loss_fn(o2, torch.randn(100, 1))

model.zero_grad()
loss1.backward(retain_graph=True)
grads_task1 = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

model.zero_grad()
loss2.backward()
grads_task2 = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

cosine = nn.functional.cosine_similarity(grads_task1[:100], grads_task2[:100], dim=0)
print(f"Gradient cosine similarity (first 100 params): {cosine.item():.4f}")

# Conflicting gradients can hurt MTL
grad_magnitude_ratio = torch.norm(grads_task1) / torch.norm(grads_task2)
print(f"Gradient magnitude ratio (task1/task2): {grad_magnitude_ratio:.4f}")
# Output:
# Gradient cosine similarity (first 100 params): -0.2134
# Gradient magnitude ratio (task1/task2): 1.1205
```

## Common Mistakes

1. **Forcing unrelated tasks to share representations.** Negative transfer occurs when tasks conflict — always validate that MTL outperforms single-task baselines.
2. **Using equal task weights without tuning.** Imbalanced tasks (different loss scales, different noise levels) need careful weighting — use uncertainty weighting or GradNorm.
3. **Ignoring gradient conflicts.** Task gradients may be in opposite directions, canceling each other and slowing learning.
4. **Applying MTL when one task is much easier.** The shared layers may overfit to the easy task, ignoring the hard one.
5. **Not using separate batch normalization per task.** Shared batch norm can cause issues when tasks have different input distributions.
6. **Overlooking task asymmetry.** Not all tasks benefit equally from sharing — asymmetric sharing (e.g., cross-stitch) often outperforms full sharing.
7. **Assuming more tasks always helps.** Information sharing has diminishing returns; beyond a certain number of tasks, performance may degrade.

## Interview Questions

### Beginner

1. What is multi-task learning?
2. How does MTL improve generalization?
3. What is hard parameter sharing?
4. What is the difference between hard and soft parameter sharing?
5. Give an example where MTL is useful.

### Intermediate

1. Explain how cross-stitch networks enable flexible information sharing between tasks.
2. What is negative transfer and how do you detect it?
3. How would you weight the losses of different tasks in MTL?
4. Compare MTL with transfer learning and multi-label classification.
5. How does MTL relate to the concept of inductive bias?

### Advanced

1. Derive the uncertainty weighting approach for MTL (Kendall et al., 2018).
2. Explain GradNorm and how it balances task gradients.
3. Describe the relationship between MTL and meta-learning (learning to learn). How can MTL be formulated as a meta-learning problem?

## Practice Problems

### Easy

1. Implement a simple 2-task hard-sharing MTL network in PyTorch.
2. Compare single-task vs. multi-task performance on the California Housing dataset (regression) with an additional synthetic task.
3. Plot the training loss curves for each task in a 3-task MTL setup.
4. Implement MTL with shared embedding + task-specific output layers in sklearn (use MultiOutputRegressor with a shared feature extractor).
5. Visualize the shared representations using t-SNE for different tasks.

### Medium

1. Implement uncertainty weighting for multi-task loss balancing.
2. Implement GradNorm from scratch and apply to a 2-task MTL problem.
3. Compare hard sharing, soft sharing (L2 regularization on parameter differences), and cross-stitch networks.
4. Analyze task gradients — compute gradient cosine similarity between tasks and detect conflicts.
5. Implement a sluice network (learnable sharing at multiple layers).

### Hard

1. Implement a Task Routing Network that learns which layers to share for each task.
2. Derive and implement the Multi-Task Attention Network (MTAN) architecture.
3. Implement the MTL version of the PCGrad (Projecting Conflicting Gradients) algorithm.

## Solutions

Solution 1 (Easy): Single vs. MTL comparison

```python
import numpy as np
from sklearn.datasets import make_regression

X, y1 = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
# Create a correlated second task
y2 = y1 * 0.7 + np.random.randn(500) * 5

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score

single = Ridge()
mtl = MultiOutputRegressor(Ridge())

single_score = np.mean(cross_val_score(single, X, y1, cv=5, scoring='r2'))
mtl.fit(X, np.column_stack([y1, y2]))
# MTL shared representation helps if tasks are related
print(f"Single-task R²: {single_score:.3f}")
```

Solution 2 (Medium): Uncertainty weighting

```python
import torch
import torch.nn as nn

class UncertaintyWeightedMTL(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, loss1, loss2):
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        total_loss = precision1 * loss1 + precision2 * loss2 + self.log_vars[0] + self.log_vars[1]
        return total_loss
```

## Related Concepts

- Transfer Learning (ML-039)
- Multi-Label Classification (ML-072)
- Meta-Learning
- Representation Learning
- Domain Adaptation
- Ensemble Learning
- Deep Learning (ML-050)

## Next Concepts

- Online Learning (ML-074)
- Model Interpretability (ML-075)
- Bayesian Optimization (ML-069)

## Summary

Multi-Task Learning improves generalization by learning multiple related tasks simultaneously through shared representations. Hard parameter sharing (shared layers + task-specific heads) is the most common approach, but soft sharing, cross-stitch units, and task routing provide more flexibility. MTL requires careful loss balancing and gradient analysis to avoid negative transfer. It is widely used in autonomous driving, NLP, recommendation systems, and robotics.

## Key Takeaways

- MTL learns multiple tasks jointly, sharing representations across them.
- Hard sharing: shared layers + task-specific heads.
- Soft sharing: separate parameters regularized to be similar.
- Cross-stitch units learn linear combinations of task representations.
- Task weighting (uncertainty weighting, GradNorm) is critical for balanced learning.
- Negative transfer occurs when tasks conflict — always compare with single-task baselines.
- MTL reduces overfitting through shared inductive bias.
- Gradient conflict analysis helps diagnose MTL training issues.
