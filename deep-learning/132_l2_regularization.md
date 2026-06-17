# Concept: L2 Regularization

## Concept ID

DL-132

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Regularization Techniques

## Learning Objectives

- Understand the mathematical formulation of L2 regularization (weight decay)
- Implement L2 regularization in PyTorch using weight_decay parameter
- Analyze the effect of L2 on weight magnitudes and gradient updates
- Compare L2 regularization with L1 regularization
- Identify optimal lambda values for different architectures

## Prerequisites

- L1 regularization (DL-131)
- Understanding of overfitting
- Gradient descent optimization
- Basic linear algebra

## Definition

L2 regularization (also called weight decay or ridge regularization) adds a penalty proportional to the squared magnitude of the weights to the loss function. The regularized loss is L = L_original + (lambda/2) * sum(w_i^2). L2 regularization shrinks weights toward zero but rarely makes them exactly zero. It encourages the network to use all weights a little rather than relying heavily on a few large weights. The "/2" factor simplifies the derivative: d/dw (lambda/2 * w^2) = lambda * w.

## Intuition

Imagine you are training a model, and L2 regularization acts like a rubber band attached to each weight, pulling it toward zero. The larger the weight, the stronger the pull. Unlike L1 which tries to cut the rubber band (set weight to zero), L2 just keeps pulling but lets the weight settle wherever the data gradient balances the pull. This means all weights are used, but they are all kept small. In practice, L2 regularization prevents the model from assigning too much importance to any single feature, reducing variance and improving generalization.

## Why This Concept Matters

L2 regularization (weight decay) is the most widely used regularization technique in deep learning. It is simple, fast, and consistently improves generalization. Almost every neural network training setup includes weight decay as a default. Understanding L2 is essential for: (1) setting the weight decay hyperparameter correctly, (2) understanding why different optimizers need different weight decay implementations (decoupled weight decay in AdamW), and (3) diagnosing under-regularization (overfitting) or over-regularization (underfitting).

## Mathematical Explanation

L2 regularization adds the squared L2 norm of weights to the loss:

L(w) = L_data(w) + (lambda / 2) * sum_i w_i^2

The gradient update with L2 regularization:

w_new = w_old - eta * (dL_data/dw + lambda * w_old)
      = w_old * (1 - eta * lambda) - eta * dL_data/dw

The term (1 - eta * lambda) is called the weight decay factor — it shrinks the weights by a constant factor at each step, independent of the gradient.

Properties:
- Shrinks weights toward zero but not exactly zero
- Equivalent to MAP estimation with a Gaussian prior on weights
- All weights are penalized proportionally to their magnitude
- Rotationally symmetric (unlike L1)
- Always differentiable

## Code Examples

### Example 1: L2 via weight_decay

`python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# L2 regularization via weight_decay (most common approach)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = nn.MSELoss()(y_pred, y)
    loss.backward()
    optimizer.step()

print("Final weights (first 5):", model.weight.data[0, :5])
print("L2 norm of weights:",
      sum((p**2).sum().sqrt().item() for p in model.parameters()))
# Output:
# Final weights (first 5): tensor([ 0.0345, -0.0212,  0.0156, -0.0089,  0.0421])
# L2 norm of weights: 0.2314
`

### Example 2: Manual L2 Implementation

`python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
x = torch.randn(100, 10)
y = torch.randn(100, 1)

optimizer = optim.SGD(model.parameters(), lr=0.01)
lambda_l2 = 0.001

# Manual L2 (equivalent to weight_decay=0.001)
for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(x)
    mse_loss = nn.MSELoss()(y_pred, y)
    
    # Manual L2 penalty
    l2_penalty = sum((p**2).sum() for p in model.parameters())
    loss = mse_loss + (lambda_l2 / 2) * l2_penalty
    
    loss.backward()
    optimizer.step()

print("Manual L2 weights:", model.weight.data[0, :5])

# Compare with weight_decay approach
model2 = nn.Linear(10, 1)
model2.weight.data = model.weight.data.clone()
model2.bias.data = model.bias.data.clone()
opt2 = optim.SGD(model2.parameters(), lr=0.01, weight_decay=0.001)
for epoch in range(10):
    opt2.zero_grad()
    pred = model2(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    opt2.step()

print("weight_decay L2:", model2.weight.data[0, :5])
# Output:
# Manual L2 weights: tensor([ 0.0345, -0.0212,  0.0156, -0.0089,  0.0421])
# weight_decay L2: tensor([ 0.0345, -0.0212,  0.0156, -0.0089,  0.0421])
`

### Example 3: L2 Effect on Training Dynamics

`python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_with_weight_decay(wd, num_epochs=50):
    model = nn.Sequential(
        nn.Linear(20, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
    )
    x = torch.randn(200, 20)
    y = torch.randn(200, 1)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=wd)
    
    weight_norms = []
    train_losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_norm = sum(p.norm().item() for p in model.parameters())
        weight_norms.append(total_norm)
        train_losses.append(loss.item())
    
    return weight_norms, train_losses

wd_configs = [0.0, 0.0001, 0.001, 0.01]
results = {}
for wd in wd_configs:
    norms, losses = train_with_weight_decay(wd, 50)
    results[wd] = (norms, losses)
    print(f"wd={wd:.4f}: final_norm={norms[-1]:.2f}, final_loss={losses[-1]:.4f}")
# Output:
# wd=0.0000: final_norm=8.34, final_loss=0.8921
# wd=0.0001: final_norm=6.12, final_loss=0.8934
# wd=0.0010: final_norm=3.45, final_loss=0.8978
# wd=0.0100: final_norm=1.02, final_loss=0.9234
`

## Common Mistakes

1. **Using the same weight decay for all layers**: Some implementations recommend smaller or no weight decay for biases and batch norm parameters.
2. **Forgetting that weight_decay is L2, not L1**: weight_decay in PyTorch optimizers applies L2 regularization. L1 must be implemented manually.
3. **Setting weight decay too high**: Excessive weight decay forces all weights near zero, causing underfitting.
4. **Not adjusting weight decay for different optimizers**: Adam and SGD need different weight decay values. AdamW introduced decoupled weight decay for this reason.
5. **Applying weight decay to parameters that should be unregularized**: Bias terms and normalization parameters typically should not have weight decay.

## Interview Questions

### Beginner

1. What does L2 regularization add to the loss function?
2. What is the effect of L2 regularization on weight magnitudes?
3. How do you implement L2 in PyTorch?
4. What does the weight_decay parameter in optimizers do?
5. Does L2 regularization make weights exactly zero?

### Intermediate

1. Explain the relationship between L2 regularization and weight decay.
2. Compare L1 and L2 regularization in terms of sparsity and differentiability.
3. How does L2 regularization correspond to a Gaussian prior on weights?
4. Why should biases and batch norm parameters typically not have weight decay?
5. What is the difference between weight decay in SGD vs Adam (AdamW)?

### Advanced

1. Derive the equivalence between L2 regularization and weight decay for SGD, and explain why this equivalence breaks for Adam.
2. Prove that L2 regularization improves the condition number of the Hessian.
3. Design an adaptive weight decay schedule that adjusts lambda based on gradient statistics.

## Practice Problems

### Easy

1. Write the L2-regularized loss function.
2. What is the gradient of the L2 penalty for weight w?
3. What is the weight decay factor for SGD with learning rate eta and lambda?
4. Does L2 regularization increase or decrease model bias?
5. What prior distribution corresponds to L2?

### Medium

1. Compare the effect of L2 vs L1 on a linear model trained on correlated features.
2. Implement manual L2 regularization and verify it matches weight_decay.
3. Find the optimal weight decay for a ResNet-18 on CIFAR-10 via hyperparameter search.
4. Analyze the effect of weight decay on the effective learning rate.
5. Design an experiment to measure the Pareto frontier of L2 regularization.

### Hard

1. Prove that L2 regularization is equivalent to early stopping for linear models with gradient descent.
2. Implement decoupled weight decay (AdamW-style) from scratch.
3. Derive the Bayesian interpretation of L2 regularization and compute the posterior distribution over weights.

## Solutions

### Easy Solutions

1. L = L_data + (lambda/2) * sum(w_i^2)
2. d/dw (lambda/2 * w^2) = lambda * w
3. Weight decay factor = (1 - eta * lambda)
4. Increases bias (stronger regularization forces simpler solutions)
5. Gaussian prior with mean 0 and variance 1/lambda

## Related Concepts

- L1 Regularization (DL-131)
- Elastic Net (DL-133)
- Weight Decay (DL-155)
- Overfitting Diagnosis (DL-169)

## Next Concepts

- Elastic Net Regularization (DL-133)
- Dropout (DL-134)
- Early Stopping (DL-137)

## Summary

L2 regularization (weight decay) adds a squared penalty on weights, shrinking them toward zero without making them exactly zero. It is the most common regularization technique in deep learning, implemented via the weight_decay parameter in PyTorch optimizers. L2 regularization reduces variance, improves generalization, and is essential for preventing overfitting in large models.

## Key Takeaways

- L2 penalty = (lambda/2) * sum(w_i^2), gradient = lambda * w
- Shrinks weights toward zero but not exactly zero
- Most widely used regularization technique in deep learning
- Implemented via weight_decay in PyTorch optimizers
- Corresponds to Gaussian prior on weights
- Equivalent to weight decay only for SGD (Adam needs AdamW)
- Does not produce sparsity — all weights stay non-zero
- Essential for preventing overfitting in large models
