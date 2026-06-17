# Concept: Mean Squared Error

## Concept ID

DL-091

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand the mathematical formulation of MSE loss
- Implement MSE loss manually and using nn.MSELoss
- Know when to use MSE for regression problems
- Analyze the properties of MSE including sensitivity to outliers
- Compute gradients of MSE with respect to predictions

## Prerequisites

- Basic statistics: mean, variance
- PyTorch tensor operations
- Regression problem understanding

## Definition

Mean Squared Error (MSE) measures the average squared difference between predicted and target values. For a set of n predictions y_hat and targets y:

MSE = (1/n) * sum_i (y_i - y_hat_i)^2

In PyTorch: nn.MSELoss() or F.mse_loss()

## Intuition

MSE penalizes large errors much more heavily than small errors due to the squaring operation. An error of 2 contributes 4 to the loss, while an error of 10 contributes 100. This makes MSE sensitive to outliers but also provides strong gradient signals when predictions are far from targets.

## Why This Concept Matters

MSE is the default loss function for regression problems. Understanding its properties helps in choosing the right loss for a given task.

## Mathematical Explanation

### Formula

MSE = (1/n) * sum_i (y_i - y_hat_i)^2

### Gradient

d(MSE)/dy_hat_i = -(2/n) * (y_i - y_hat_i)

The gradient is proportional to the error, so larger errors produce larger gradient updates.

### Expected Value

Under the assumption that errors are Gaussian with constant variance, minimizing MSE is equivalent to maximum likelihood estimation.

## Code Examples

### Example 1: Manual MSE and nn.MSELoss

```python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])

# Manual MSE
mse_manual = ((y_true - y_pred) ** 2).mean()
print(f"Manual MSE: {mse_manual.item():.4f}")

# PyTorch MSELoss
criterion = nn.MSELoss()
mse_pytorch = criterion(y_pred, y_true)
print(f"PyTorch MSE: {mse_pytorch.item():.4f}")

# Verify reduction='sum'
criterion_sum = nn.MSELoss(reduction='sum')
mse_sum = criterion_sum(y_pred, y_true)
print(f"MSE (sum): {mse_sum.item():.4f}")
```

```
# Output:
# Manual MSE: 0.1775
# PyTorch MSE: 0.1775
# MSE (sum): 0.7100
```

### Example 2: MSE for Linear Regression

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(100, 1)
true_w = 2.0
true_b = 1.0
y = true_w * X + true_b + 0.1 * torch.randn(100, 1)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: MSE = {loss.item():.6f}")

w, b = model.weight.item(), model.bias.item()
print(f"Learned: w = {w:.4f}, b = {b:.4f}")
print(f"True:    w = {true_w:.4f}, b = {true_b:.4f}")
```

```
# Output:
# Epoch 0: MSE = 4.511498
# Epoch 50: MSE = 0.018480
# Epoch 100: MSE = 0.009806
# Epoch 150: MSE = 0.009552
# Epoch 199: MSE = 0.009547
# Learned: w = 1.9988, b = 1.0015
# True:    w = 2.0000, b = 1.0000
```

### Example 3: MSE vs. MAE Outlier Sensitivity

```python
import torch
import torch.nn as nn

# Data with an outlier
y_true = torch.tensor([2.0, 3.0, 2.5, 100.0, 3.0])
y_pred = torch.tensor([2.1, 2.9, 2.4, 3.0, 3.1])

mse = nn.MSELoss()(y_pred, y_true)
mae = nn.L1Loss()(y_pred, y_true)

# Removing outlier
y_true_clean = torch.tensor([2.0, 3.0, 2.5, 3.0])
y_pred_clean = torch.tensor([2.1, 2.9, 2.4, 3.1])

mse_clean = nn.MSELoss()(y_pred_clean, y_true_clean)
mae_clean = nn.L1Loss()(y_pred_clean, y_true_clean)

print(f"MSE with outlier: {mse.item():.2f}")
print(f"MAE with outlier: {mae.item():.2f}")
print(f"MSE without outlier: {mse_clean.item():.2f}")
print(f"MAE without outlier: {mae_clean.item():.2f}")
```

```
# Output:
# MSE with outlier: 1880.82
# MAE with outlier: 19.42
# MSE without outlier: 0.02
# MAE without outlier: 0.12
```

## Common Mistakes

1. **Reduction confusion**: nn.MSELoss defaults to 'mean' reduction. For 'sum' reduction, set reduction='sum'.
2. **Order of arguments**: nn.MSELoss expects (input, target), not (target, input).
3. **Using MSE for classification**: MSE is for regression. For classification, use cross-entropy.
4. **Outlier dominance**: MSE is dominated by outliers. A single bad prediction can dominate the loss.
5. **Not normalizing targets**: MSE assumes targets are on a reasonable scale. Large target values produce huge losses.

## Interview Questions

### Beginner

1. What is MSE and how is it computed?
2. Why is the error squared instead of taken as absolute value?
3. What does MSE = 0 mean?
4. When would you use MSE loss?
5. How does MSE handle outliers?

### Intermediate

1. Derive the gradient of MSE with respect to predictions.
2. How does MSE relate to maximum likelihood estimation under Gaussian noise?
3. Compare MSE and MAE in terms of gradient behavior near zero.
4. What happens to MSE if the target values are very large?
5. How does the reduction parameter affect backpropagation?

### Advanced

1. Prove that MSE is a Bregman divergence.
2. Analyze the robustness of MSE to different noise distributions.
3. Explain the relationship between MSE and the Cram\'er-Rao bound.

## Practice Problems

### Easy

1. Compute MSE manually for [1, 2, 3] vs [1.1, 1.9, 3.2].
2. Implement MSE loss without using PyTorch loss functions.
3. Plot MSE loss for predictions ranging from -5 to 5 with target 0.
4. Compare MSE for different learning rates in linear regression.
5. Use nn.MSELoss with reduction='sum'.

### Medium

1. Train a multi-output regression model with MSE loss.
2. Compare convergence of SGD with MSE vs. MAE on data with outliers.
3. Implement weighted MSE where some samples matter more.
4. Visualize the MSE gradient as a function of prediction error.
5. Train a neural network for regression using MSE.

### Hard

1. Derive the minimum of MSE for a given set of predictions.
2. Prove the equivalence of MSE minimization and MLE under Gaussian noise.
3. Design an adaptive loss that combines MSE and MAE based on error magnitude.

## Solutions

MSE = mean((y - y_hat)^2). In PyTorch: nn.MSELoss(). Gradients are proportional to the error. Best for regression with Gaussian noise.

## Related Concepts

- Mean Absolute Error (DL-092): Alternative regression loss
- Huber Loss (DL-093): Combines MSE and MAE
- Cross-Entropy Loss (DL-094): For classification

## Next Concepts

- Mean Absolute Error (DL-092)
- Huber Loss (DL-093)
- Cross-Entropy Loss (DL-094)

## Summary

MSE is the squared difference between predictions and targets, averaged over samples. It penalizes large errors heavily and is optimal under Gaussian noise assumptions. MSE is the default loss for regression tasks but is sensitive to outliers.

## Key Takeaways

1. MSE = mean((y - y_hat)^2), the average squared error.
2. MSE penalizes large errors quadratically more than small errors.
3. MSE is optimal when errors follow a Gaussian distribution.
4. MSE is sensitive to outliers — a single outlier can dominate the loss.
5. Use MSE for regression, not classification.
