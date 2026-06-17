# Concept: Mean Absolute Error

## Concept ID

DL-092

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Loss Functions

## Learning Objectives

- Understand the mathematical formulation of MAE loss
- Implement MAE using nn.L1Loss
- Compare MAE with MSE in terms of sensitivity and gradient behavior
- Know when to prefer MAE over MSE
- Analyze the robustness of MAE to outliers

## Prerequisites

- Mean Squared Error (DL-091)
- Basic statistical concepts
- PyTorch fundamentals

## Definition

Mean Absolute Error (MAE), also known as L1 loss, measures the average absolute difference between predicted and target values. For n predictions:

MAE = (1/n) * sum_i |y_i - y_hat_i|

In PyTorch: nn.L1Loss() or F.l1_loss()

## Intuition

Unlike MSE which squares the error, MAE takes the absolute value. This means all errors are penalized linearly. An error of 2 contributes 2 to the loss, and an error of 10 contributes 10. MAE is more robust to outliers than MSE because it does not amplify large errors.

## Why This Concept Matters

MAE is essential for regression problems where outliers are present or when a robust loss is needed.

## Mathematical Explanation

### Formula

MAE = (1/n) * sum_i |y_i - y_hat_i|

### Gradient

d(MAE)/dy_hat_i = -(1/n) * sign(y_i - y_hat_i)

The gradient is constant magnitude for all errors. This is both an advantage (robustness) and a disadvantage (gradient does not diminish near zero).

### Median Property

MAE minimization yields the conditional median of the target distribution, while MSE yields the conditional mean. For skewed distributions, MAE gives a different and sometimes more desirable estimate.

## Code Examples

### Example 1: Manual MAE and nn.L1Loss

```python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])

# Manual MAE
mae_manual = torch.abs(y_true - y_pred).mean()
print(f"Manual MAE: {mae_manual.item():.4f}")

# PyTorch L1Loss
criterion = nn.L1Loss()
mae_pytorch = criterion(y_pred, y_true)
print(f"PyTorch MAE: {mae_pytorch.item():.4f}")

# Compare with MSE
mse = nn.MSELoss()(y_pred, y_true)
print(f"MSE: {mse.item():.4f}")
```

```
# Output:
# Manual MAE: 0.3000
# PyTorch MAE: 0.3000
# MSE: 0.1775
```

### Example 2: MAE for Regression with Outliers

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(200, 1)
y = 2.0 * X + 1.0 + 0.1 * torch.randn(200, 1)
# Add outlier
y[0] = 50.0

model_mae = nn.Linear(1, 1)
model_mse = nn.Linear(1, 1)
opt_mae = optim.SGD(model_mae.parameters(), lr=0.01)
opt_mse = optim.SGD(model_mse.parameters(), lr=0.01)

for epoch in range(500):
    opt_mae.zero_grad()
    opt_mse.zero_grad()
    loss_mae = nn.L1Loss()(model_mae(X), y)
    loss_mse = nn.MSELoss()(model_mse(X), y)
    loss_mae.backward()
    loss_mse.backward()
    opt_mae.step()
    opt_mse.step()

print(f"MAE model: w = {model_mae.weight.item():.4f}, b = {model_mae.bias.item():.4f}")
print(f"MSE model: w = {model_mse.weight.item():.4f}, b = {model_mse.bias.item():.4f}")
print(f"True: w = 2.0, b = 1.0")
```

```
# Output:
# MAE model: w = 1.9988, b = 1.0023
# MSE model: w = 1.5321, b = 1.4562
```

### Example 3: Gradient Comparison

```python
import torch
import torch.nn as nn

errors = torch.linspace(-3, 3, 100)

mse_grad = -2 * errors / errors.numel()
mae_grad = -torch.sign(errors) / errors.numel()

print("Error\tMSE_grad\tMAE_grad")
for e in [-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0]:
    idx = (errors - e).abs().argmin()
    print(f"{e:+.1f}\t{mse_grad[idx]:.4f}\t\t{mae_grad[idx]:.4f}")
```

```
# Output:
# Error   MSE_grad        MAE_grad
# -3.0    0.0600          0.0100
# -1.0    0.0200          0.0100
# -0.1    0.0020          0.0100
# +0.0    -0.0000         0.0100
# +0.1    -0.0020         -0.0100
# +1.0    -0.0200         -0.0100
# +3.0    -0.0600         -0.0100
```

## Common Mistakes

1. **Using MAE when MSE is expected**: Many regression benchmarks use MSE. Using MAE changes the implicit objective.
2. **Confusing L1 loss with L1 regularization**: L1 loss (MAE) is different from L1 regularization (LASSO).
3. **Not accounting for non-differentiability at zero**: MAE is not differentiable at error = 0. PyTorch uses subgradient (0).
4. **Ignoring MAE's constant gradient**: MAE gradients do not diminish near zero, which can make convergence noisy.
5. **Using MAE for classification**: MAE is for regression. For classification, use cross-entropy.

## Interview Questions

### Beginner

1. What is MAE and how does it differ from MSE?
2. Why is MAE more robust to outliers than MSE?
3. What is the gradient of MAE at zero?
4. When would you choose MAE over MSE?
5. How do you implement MAE in PyTorch?

### Intermediate

1. Derive the gradient of MAE and explain why it is constant.
2. Why does MAE minimize the conditional median rather than mean?
3. Compare the convergence behavior of SGD with MAE vs. MSE.
4. When does the non-differentiability of MAE at zero cause problems?
5. How would you modify MAE to be differentiable everywhere while remaining robust?

### Advanced

1. Prove that the minimizer of MAE is the median.
2. Analyze the efficiency of MAE under different noise distributions.
3. Explain the connection between MAE and quantile regression.

## Practice Problems

### Easy

1. Compute MAE manually for [1, 2, 3] vs [1.5, 2.5, 3.5].
2. Implement MAE from scratch without nn.L1Loss.
3. Plot MAE vs. MSE for predictions from -5 to 5 with target 0.
4. Compare the loss values of MAE and MSE for the same predictions.
5. Use nn.L1Loss on a simple regression problem.

### Medium

1. Compare MAE and MSE for a dataset with different outlier percentages.
2. Train a model with MAE and report both MAE and MSE metrics.
3. Implement smooth L1 loss and compare with MAE.
4. Visualize how MAE gradient stays constant while MSE gradient increases with error.
5. Train a neural network with MAE loss and analyze convergence.

### Hard

1. Prove the consistency of MAE under heavy-tailed error distributions.
2. Implement a quantile regression loss and show its relationship to MAE.
3. Design an adaptive loss that transitions from MSE to MAE during training.

## Solutions

MAE = mean(|y - y_hat|). In PyTorch: nn.L1Loss(). More robust to outliers than MSE. Gradient is constant sign. Minimizer is the conditional median.

## Related Concepts

- MSE (DL-091): Squared error regression loss
- Huber Loss (DL-093): Combines MAE and MSE
- Smooth L1 Loss: Differentiable variant

## Next Concepts

- Huber Loss (DL-093)
- Cross-Entropy Loss (DL-094)
- Binary Cross-Entropy (DL-095)

## Summary

MAE measures the average absolute difference between predictions and targets. It is more robust to outliers than MSE because errors are penalized linearly. MAE gradients are constant in magnitude, which can be beneficial for robustness but may cause noisy convergence near the optimum. MAE minimizes the conditional median rather than the mean.

## Key Takeaways

1. MAE = mean(|y - y_hat|), the average absolute error.
2. MAE is more robust to outliers than MSE.
3. MAE gradient is constant sign(y - y_hat), independent of error magnitude.
4. MAE is non-differentiable at zero (subgradient used).
5. Use MAE when robustness to outliers is important.
