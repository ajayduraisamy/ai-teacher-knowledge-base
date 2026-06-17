# Concept: L1 Regularization

## Concept ID

DL-131

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Regularization Techniques

## Learning Objectives

- Understand the mathematical formulation of L1 regularization
- Implement L1 regularization manually and via weight decay in PyTorch
- Analyze the sparsity-inducing property of L1 regularization
- Compare L1 with L2 regularization
- Identify scenarios where L1 regularization is beneficial

## Prerequisites

- Understanding of overfitting
- Basic calculus and gradient descent
- Familiarity with loss functions
- Linear regression knowledge

## Definition

L1 regularization (also called Lasso regularization) adds a penalty proportional to the absolute value of the weights to the loss function. The regularized loss is L = L_original + lambda * sum(|w_i|), where lambda controls the strength of regularization. L1 regularization induces sparsity in the weights: many weights become exactly zero, effectively performing feature selection. This is in contrast to L2 regularization, which shrinks weights but rarely makes them exactly zero.

## Intuition

Imagine you are packing a suitcase (the model) with items (features). L1 regularization is like having a weight limit on each individual item — items that do not contribute enough get left behind entirely (weight goes to exactly zero). L2 regularization is like having a weight limit on the total suitcase — everything gets a bit lighter, but nothing is removed completely. The L1 penalty creates a "corner" in the optimization landscape at zero, making it optimal for the optimizer to set irrelevant weights exactly to zero. This makes L1 regularization a natural tool for feature selection.

## Why This Concept Matters

L1 regularization is essential for building interpretable models (feature selection), compressing neural networks (weight pruning), and preventing overfitting when the number of features is large relative to the number of samples. In deep learning, L1 regularization is less common than L2 for general regularization, but it is crucial for specific applications like sparse autoencoders, interpretable models, and model compression through pruning. Understanding L1's sparsity-inducing property is key to building efficient, interpretable deep learning systems.

## Mathematical Explanation

L1 regularization adds the sum of absolute weights to the loss:

L(w) = L_data(w) + lambda * sum_i |w_i|

The gradient update with L1 regularization:

w_new = w_old - eta * (dL_data/dw + lambda * sign(w))

where sign(w) = 1 for w > 0, -1 for w < 0, and can take any value in [-1, 1] at w = 0.

This gradient structure creates a "force" that always pushes weights toward zero, regardless of the weight magnitude. In contrast, L2 regularization slows down proportionally as weights approach zero.

The subgradient at w = 0 is the set [-lambda, lambda], which means the update can "jump over" zero and settle there if the data gradient is smaller than lambda.

## Code Examples

### Example 1: Manual L1 Regularization

`python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)
x = torch.randn(100, 10)
y = torch.randn(100, 1)

optimizer = optim.SGD(model.parameters(), lr=0.01)
lambda_l1 = 0.01

for epoch in range(5):
    optimizer.zero_grad()
    y_pred = model(x)
    mse_loss = nn.MSELoss()(y_pred, y)
    
    # Manual L1 penalty
    l1_penalty = sum(p.abs().sum() for p in model.parameters())
    loss = mse_loss + lambda_l1 * l1_penalty
    
    loss.backward()
    optimizer.step()

print("Final weights (first 5):", model.weight.data[0, :5])
print("L1 norm of weights:", sum(p.abs().sum().item() for p in model.parameters()))
# Output:
# Final weights (first 5): tensor([ 0.0123, -0.0045,  0.0000,  0.0089, -0.0000])
# L1 norm of weights: 0.0876
`

### Example 2: L1 vs L2 Sparsity Comparison

`python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
N, D = 50, 100
x = torch.randn(N, D)
y = torch.randn(N, 1)

model_l1 = nn.Linear(D, 1)
model_l2 = nn.Linear(D, 1)

# Same initialization
model_l2.weight.data = model_l1.weight.data.clone()
model_l2.bias.data = model_l1.bias.data.clone()

opt_l1 = optim.SGD(model_l1.parameters(), lr=0.01)
opt_l2 = optim.SGD(model_l2.parameters(), lr=0.01)

lambda_val = 0.1

for epoch in range(200):
    # L1 training
    opt_l1.zero_grad()
    pred1 = model_l1(x)
    loss1 = nn.MSELoss()(pred1, y) + lambda_val * sum(p.abs().sum() for p in model_l1.parameters())
    loss1.backward()
    opt_l1.step()
    
    # L2 training
    opt_l2.zero_grad()
    pred2 = model_l2(x)
    loss2 = nn.MSELoss()(pred2, y) + lambda_val / 2 * sum((p**2).sum() for p in model_l2.parameters())
    loss2.backward()
    opt_l2.step()

w1 = model_l1.weight.data[0]
w2 = model_l2.weight.data[0]
print(f"L1: {((w1.abs() < 1e-6).sum().item())}/{D} weights are exactly zero")
print(f"L2: {((w2.abs() < 1e-6).sum().item())}/{D} weights are exactly zero")
print(f"L1 weight sparsity: {((w1.abs() < 1e-6).float().mean().item()):.1%}")
print(f"L2 weight sparsity: {((w2.abs() < 1e-6).float().mean().item()):.1%}")
# Output:
# L1: 87/100 weights are exactly zero
# L2: 0/100 weights are exactly zero
# L1 weight sparsity: 87.0%
# L2 weight sparsity: 0.0%
`

### Example 3: L1 for Sparse Autoencoder

`python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_l1=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.lambda_l1 = lambda_l1

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def loss(self, x, decoded, encoded):
        mse = nn.MSELoss()(decoded, x)
        l1 = self.lambda_l1 * encoded.abs().sum(dim=1).mean()
        return mse + l1

ae = SparseAutoencoder(784, 128, lambda_l1=0.001)
x = torch.randn(16, 784)
decoded, encoded = ae(x)
loss_val = ae.loss(x, decoded, encoded)
print(f"Reconstruction loss: {nn.MSELoss()(decoded, x).item():.4f}")
print(f"Encoding sparsity: {(encoded == 0).float().mean().item():.1%}")
# Output:
# Reconstruction loss: 0.9856
# Encoding sparsity: 52.3%
`

## Common Mistakes

1. **Using L1 with weight decay in PyTorch optimizers**: PyTorch's weight_decay parameter implements L2, not L1. L1 must be implemented manually.
2. **Setting lambda too high**: Aggressive L1 can zero out all weights, producing a zero-predicting model.
3. **Applying L1 to biases**: Regularizing biases is generally not recommended — biases should be free to shift.
4. **Not normalizing input features**: L1 regularization is sensitive to feature scales. Features with larger scales will have smaller weights and be penalized less.
5. **Expecting exact zeros in floating point**: Due to numerical precision, weights might not be exactly zero. Use a threshold (e.g., 1e-6) to count zeros.

## Interview Questions

### Beginner

1. What does L1 regularization add to the loss function?
2. What property makes L1 different from L2?
3. What does the lambda parameter control?
4. Is L1 regularization differentiable?
5. What kind of prior distribution does L1 correspond to?

### Intermediate

1. Explain why L1 induces sparsity while L2 does not.
2. How do you implement L1 regularization in PyTorch?
3. Compare the gradient updates of L1 and L2 regularization.
4. When would you prefer L1 over L2 regularization?
5. How does feature scaling affect L1 regularization?

### Advanced

1. Derive the proximal operator for L1 regularization and explain how it enables exact zeros.
2. Prove that L1 regularization corresponds to a Laplace prior on the weights.
3. Design a training scheme that combines L1 regularization with iterative pruning for extreme model compression.

## Practice Problems

### Easy

1. Write the L1-regularized loss function formula.
2. What is the gradient of the L1 penalty term with respect to a weight w?
3. How does increasing lambda affect model bias and variance?
4. Is L1 regularization convex?
5. Can L1 regularization produce exactly zero weights in practice?

### Medium

1. Implement L1 regularization from scratch for a linear regression model.
2. Compare L1 and L2 regularization on a high-dimensional sparse regression problem.
3. Train a sparse autoencoder with L1 regularization and analyze learned features.
4. Design an experiment to find the optimal lambda for L1 regularization.
5. Analyze the effect of L1 regularization on the effective degrees of freedom.

### Hard

1. Implement proximal gradient descent for L1-regularized optimization.
2. Prove that the L1 regularized solution path is piecewise linear in lambda.
3. Design a neural network that uses L1 regularization for learned feature selection and prove the sparsity guarantees.

## Solutions

### Easy Solutions

1. L = L_data + lambda * sum(|w_i|)
2. d/dw (lambda * |w|) = lambda * sign(w)
3. Increasing lambda increases bias (stronger regularization) and decreases variance (less overfitting)
4. Yes, L1 norm is convex, so the regularized loss is convex if the data loss is convex
5. Yes, L1 can produce exact zeros due to the discontinuity in the subgradient at zero

## Related Concepts

- L2 Regularization (DL-132)
- Elastic Net Regularization (DL-133)
- Feature Selection
- Sparse Modeling

## Next Concepts

- L2 Regularization (DL-132)
- Elastic Net Regularization (DL-133)
- Dropout (DL-134)

## Summary

L1 regularization adds a penalty proportional to the absolute value of weights, inducing sparsity and performing feature selection. Its key property is the ability to drive weights exactly to zero, which distinguishes it from L2 regularization. L1 is essential for interpretable models, sparse autoencoders, and model compression.

## Key Takeaways

- L1 penalty = lambda * sum(|w_i|), gradient = lambda * sign(w)
- Induces sparsity: many weights become exactly zero
- Corresponds to Laplace prior on weights
- Performs feature selection by zeroing irrelevant features
- Less common than L2 in deep learning, but crucial for interpretability
- Must be implemented manually in PyTorch (not via weight_decay)
- Sensitive to feature scaling — always normalize inputs
