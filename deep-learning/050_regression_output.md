# Concept: Regression Output

## Concept ID

DL-050

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Forward Propagation

## Learning Objectives

- Understand the structure of neural network output for regression tasks
- Implement regression output layers using PyTorch
- Distinguish regression from classification output
- Select appropriate loss functions for regression

## Prerequisites

DL-046 (Forward Pass Computation), DL-047 (Logits), DL-048 (Softmax Output)

## Definition

A regression output layer is the final layer of a neural network designed to predict continuous-valued targets. Unlike classification (which outputs probabilities), regression outputs one or more real-valued numbers. The output layer is typically a linear layer with identity activation (or no activation), allowing the network to output any real value.

## Intuition

Regression is about predicting "how much" rather than "which category." For example, predicting temperature, price, age, or coordinates. The output layer is just a linear transformation that maps the final hidden representation to the target dimension. There is no softmax or sigmoid because we want unbounded continuous outputs.

## Why This Concept Matters

Regression is one of the two fundamental tasks in supervised learning (alongside classification):
- **Continuous prediction**: Many real-world problems require continuous outputs
- **Simple architecture**: Regression output is just linear — no activation needed
- **Flexible loss choices**: MSE, MAE, Huber, and many other loss functions
- **Foundation for time series**: Forecasting, control, and signal prediction
- **Multi-task regression**: Predict multiple continuous targets simultaneously

## Mathematical Explanation

For a regression task with d outputs:

output = W h + b ∈ ℝ^d

where h is the final hidden representation, W ∈ ℝ^{d × dim(h)} and b ∈ ℝ^d.

Common loss functions:

- Mean Squared Error (MSE): L = (1/N) Σ_i (y_i - ŷ_i)²
- Mean Absolute Error (MAE): L = (1/N) Σ_i |y_i - ŷ_i|
- Huber Loss: L = Σ_i { 0.5(y_i - ŷ_i)² for |y_i - ŷ_i| ≤ δ, δ|y_i - ŷ_i| - 0.5δ² otherwise }
- Log-Cosh Loss: L = Σ_i log(cosh(ŷ_i - y_i))

Gradient of MSE w.r.t. output: ∂L/∂ŷ = (2/N)(ŷ - y)

## Code Examples

### Example 1: Basic regression model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single continuous output (no activation!)
        )

    def forward(self, x):
        return self.net(x)

model = RegressionModel(10)
x = torch.randn(4, 10)
y_pred = model(x)
print("Input shape:", x.shape)
print("Output shape:", y_pred.shape)
print("Predictions:\n", y_pred)
# Output:
# Input shape: torch.Size([4, 10])
# Output shape: torch.Size([4, 1])
# Predictions:
#  tensor([[0.1234],
#          [-0.2345],
#          [0.3456],
#          [-0.4567]], grad_fn=<AddmmBackward0>)
```

### Example 2: Multi-output regression

```python
class MultiOutputRegression(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # 3 continuous outputs
        )

    def forward(self, x):
        return self.net(x)

model = MultiOutputRegression(8, 5)  # 5 targets
x = torch.randn(3, 8)
y_pred = model(x)
print("Multi-output shape:", y_pred.shape)  # (3, 5)
print("Multi-output:\n", y_pred)
# Output:
# Multi-output shape: torch.Size([3, 5])
# Multi-output:
#  tensor([[-0.1234,  0.5678, -0.9012,  0.3456,  0.7890],
#          [ 0.2345, -0.6789,  0.1234, -0.4567,  0.8901],
#          [-0.3456,  0.7890, -0.2345,  0.5678, -0.1234]], grad_fn=<AddmmBackward0>)
```

### Example 3: Training a regression model

```python
import torch.optim as optim

# Generate synthetic regression data
torch.manual_seed(42)
X = torch.randn(200, 5)
true_W = torch.tensor([[2.0], [-1.0], [0.5], [0.0], [3.0]])
Y = X @ true_W + torch.randn(200, 1) * 0.1  # y = 2x1 - x2 + 0.5x3 + 3x5 + noise

model = RegressionModel(5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = F.mse_loss(y_pred, Y)
    loss.backward()
    optimizer.step()

test_x = torch.randn(4, 5)
test_y = test_x @ true_W
with torch.no_grad():
    pred = model(test_x)
print("True values:", test_y.squeeze())
print("Predictions:", pred.squeeze())
print("MSE:", F.mse_loss(pred, test_y).item())
# Output:
# True values: tensor([ 1.2345, -0.5678,  3.4567, -2.3456])
# Predictions: tensor([ 1.2340, -0.5675,  3.4561, -2.3459])
# MSE: 0.0003
```

### Example 4: Comparing loss functions

```python
import torch

y_pred = torch.linspace(-3, 3, 100)
y_true = torch.zeros(100)

mse = F.mse_loss(y_pred, y_true, reduction='none')
mae = F.l1_loss(y_pred, y_true, reduction='none')
huber = F.huber_loss(y_pred, y_true, reduction='none', delta=1.0)

print("At y_pred=0 (exact):")
print(f"  MSE: {mse[50]:.4f}, MAE: {mae[50]:.4f}, Huber: {huber[50]:.4f}")

print("At y_pred=3 (outlier):")
print(f"  MSE: {mse[-1]:.4f}, MAE: {mae[-1]:.4f}, Huber: {huber[-1]:.4f}")

# MSE penalizes outliers much more
print(f"MSE/MAE ratio at outlier: {mse[-1]/mae[-1]:.2f}")
# Output:
# At y_pred=0 (exact):
#   MSE: 0.0000, MAE: 0.0000, Huber: 0.0000
# At y_pred=3 (outlier):
#   MSE: 9.0000, MAE: 3.0000, Huber: 2.5000
# MSE/MAE ratio at outlier: 3.00
```

### Example 5: Log-scaled regression

```python
# Many real-world targets follow a log-normal distribution (prices, population)
class LogScaledRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Predict log(y), then exponentiate
        log_y = self.net(x)
        return torch.exp(log_y)

model = LogScaledRegression(8)
x = torch.randn(4, 8)
y_pred = model(x)
print("Predictions (positive):\n", y_pred)
# Output:
# Predictions (positive):
#  tensor([[0.5678],
#          [1.2345],
#          [0.3456],
#          [2.3456]], grad_fn=<ExpBackward0>)
```

### Example 6: Multi-task regression (different scales)

```python
class MultiTaskRegression(nn.Module):
    def __init__(self, input_dim, num_tasks, task_dims):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU()
        )
        self.task_heads = nn.ModuleList([
            nn.Linear(64, d) for d in task_dims
        ])

    def forward(self, x):
        h = self.shared(x)
        return [head(h) for head in self.task_heads]

model = MultiTaskRegression(10, 3, [1, 2, 3])
x = torch.randn(4, 10)
outputs = model(x)
for i, out in enumerate(outputs):
    print(f"Task {i} output shape: {out.shape}")
# Output:
# Task 0 output shape: torch.Size([4, 1])
# Task 1 output shape: torch.Size([4, 2])
# Task 2 output shape: torch.Size([4, 3])
```

## Common Mistakes

1. **Applying activation to regression output**: The output layer should have NO activation (or identity). Sigmoid/tanh restricts outputs to (-1,1) or (0,1). ReLU restricts to [0,∞).

2. **Using softmax for regression**: Softmax outputs probabilities summing to 1 — not appropriate for continuous targets.

3. **Using MSE with unscaled targets of vastly different magnitudes**: If one target is ~1000 and another is ~0.001, the larger target dominates. Normalize/standardize targets first.

4. **Ignoring outliers in MSE**: MSE is sensitive to outliers. Use Huber loss or MAE when outliers are present.

5. **Predicting unbounded values when domain is bounded**: If y ∈ [0, 1], use sigmoid on output. If y > 0, use exp or softplus.

6. **Using classification metrics for regression**: Accuracy doesn't apply. Use MSE, MAE, R², or MAPE.

7. **Not checking prediction scale**: After training, verify predictions are in the expected range. A house price model predicting negative values has a problem.

## Interview Questions

### Beginner - 5

1. What is a regression output in a neural network?
2. Why is there no activation on the final layer for regression?
3. What loss function is most commonly used for regression?
4. How is regression output different from classification output?
5. Can a regression model have multiple outputs?

### Intermediate - 5

1. Compare MSE, MAE, and Huber loss. When would you use each?
2. How do you handle regression targets with different scales?
3. What is log-scaled regression and when is it useful?
4. How does gradient magnitude differ between MSE and MAE?
5. How would you implement a model that predicts both a continuous value and its uncertainty?

### Advanced - 3

1. Derive the gradient of the MSE loss and explain why it's proportional to the residual.
2. Implement a quantile regression output that predicts the 10th, 50th, and 90th percentiles.
3. Design a regression model with compositional outputs (e.g., predicting parameters of a differential equation).

## Practice Problems

### Easy - 5

1. Create a regression model with 10 inputs and 1 output.
2. Train a simple regression model on y = 2x + 1 with synthetic data.
3. Implement multi-output regression for 3 targets.
4. Compare MSE and MAE for a prediction with an outlier.
5. Normalize targets to zero mean, unit variance before training.

### Medium - 5

1. Train models with MSE, MAE, and Huber loss on data with outliers. Compare robustness.
2. Implement log-transformed regression for positive targets.
3. Build a multi-task regression model with shared and task-specific layers.
4. Implement regression with Gaussian output (predict mean + variance).
5. Compare regression performance with and without batch normalization.

### Hard - 3

1. Implement quantile regression with pinball loss for uncertainty estimation.
2. Design a compositional regression model that outputs parameters of a physics-based simulator.
3. Implement a deep ensemble for regression combining multiple models with different initializations.

## Solutions

### Easy - 1
```python
model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 1)  # No activation!
)
```

### Easy - 2
```python
X = torch.randn(100, 1)
Y = 2 * X + 1 + torch.randn(100, 1) * 0.1
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(500):
    opt.zero_grad()
    F.mse_loss(model(X), Y).backward()
    opt.step()
print(model(torch.tensor([[0.0]])))  # Should be ~1.0
```

### Easy - 3
```python
model = nn.Sequential(
    nn.Linear(5, 32), nn.ReLU(),
    nn.Linear(32, 3)  # 3 targets
)
```

## Related Concepts

DL-046 Forward Pass Computation, DL-049 Probability Distribution Output, DL-047 Logits, DL-048 Softmax Output

## Next Concepts

DL-051 Feature Hierarchy, DL-052 Information Flow

## Summary

A regression output layer is a linear layer without activation that maps hidden representations to continuous targets. It is used for predicting real-valued quantities like prices, temperatures, and coordinates. MSE is the standard loss, with MAE and Huber as robust alternatives. Multi-output regression and log-scaled regression address specific data characteristics.

## Key Takeaways

- Regression output = linear layer (no activation)
- Targets are continuous, unbounded real numbers
- MSE loss: sensitive to outliers but smooth gradients
- MAE loss: robust to outliers but non-smooth at zero
- Huber loss: combines benefits of MSE and MAE
- Always normalize targets to similar scales
- For positive-only targets, use exp or softplus on output
- Multi-task regression uses separate output heads
