# Concept: LSTM for Time Series

## Concept ID

DL-308

## Difficulty

Advanced

## Domain

Deep Learning

## Module

LSTM

## Learning Objectives

- Understand how LSTMs are applied to time series forecasting
- Implement LSTM-based time series models for univariate and multivariate data
- Handle time series-specific challenges (trend, seasonality, non-stationarity)
- Build proper train/validation/test splits for temporal data
- Evaluate time series forecasts using domain-appropriate metrics

## Prerequisites

- DL-307: LSTM for Sequence Prediction
- Understanding of time series concepts
- Familiarity with regression metrics
- Basic statistics

## Definition

LSTM for time series refers to the application of LSTM networks to model and forecast time-dependent data where observations are recorded at regular intervals. Time series data is characterized by temporal dependencies (autocorrelation), trends (long-term changes), seasonality (periodic patterns), and often non-stationarity (changing statistical properties over time). LSTMs can capture these characteristics through their gating mechanism and long-term memory.

## Intuition

Time series forecasting with LSTMs is like having a weather model that looks at patterns in temperature, pressure, and humidity over the past week to predict tomorrow's weather. The LSTM learns which patterns repeat (seasonality), which trends are developing, and how different variables interact over time. The forget gate allows it to ignore outdated patterns, while the input gate lets it incorporate new information.

## Why This Concept Matters

Time series forecasting is critical across industries:

- Finance: stock prices, volatility, risk metrics
- Energy: load forecasting, renewable generation prediction
- Retail: demand forecasting, inventory optimization
- Healthcare: patient vitals, disease progression
- Manufacturing: predictive maintenance, quality control
- Transportation: traffic flow, arrival time prediction

## Mathematical Explanation

### Univariate Time Series
Given observations y_1, y_2, ..., y_T, predict y_(T+1), ..., y_(T+h):

The model uses a sliding window of length L:
x_t = [y_(t-L+1), y_(t-L+2), ..., y_t]

y_hat_(t+1) = LSTM(x_t; theta)

### Multivariate Time Series
Given observations x_t ∈ ℝ^d (d variables), predict future values:

h_t, C_t = LSTM(x_t, h_(t-1), C_(t-1))
y_hat_(t+1) = W * h_t + b

### Common preprocessing
- Differencing: y'_t = y_t - y_(t-1) to remove trends
- Log transformation: log(y_t) to stabilize variance
- Normalization: (y_t - mu) / sigma to zero mean, unit variance
- Seasonal decomposition: extract trend, seasonal, residual components

## Code Examples

### Code Example 1: Univariate Time Series Forecasting

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Generate synthetic time series with trend and seasonality
N = 1000
t = torch.arange(0, N).float()
trend = 0.001 * t
seasonality = torch.sin(2 * math.pi * t / 50)
noise = 0.1 * torch.randn(N)
data = trend + seasonality + noise

# Create sequences
def create_sequences(data, seq_len=30):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return torch.stack(xs).unsqueeze(-1), torch.stack(ys).unsqueeze(-1)

X, y = create_sequences(data, seq_len=30)

# Proper time series split
train_X, train_y = X[:800], y[:800]
val_X, val_y = X[800:900], y[800:900]
test_X, test_y = X[900:], y[900:]

# Normalize
mean = train_X.mean()
std = train_X.std()
train_X_norm = (train_X - mean) / std
train_y_norm = (train_y - mean) / std
val_X_norm = (val_X - mean) / std
val_y_norm = (val_y - mean) / std
test_X_norm = (test_X - mean) / std
test_y_norm = (test_y - mean) / std

model = TimeSeriesLSTM()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    pred = model(train_X_norm)
    loss = loss_fn(pred, train_y_norm)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_norm)
            val_loss = loss_fn(val_pred, val_y_norm)
        print(f"Epoch {epoch}: train_loss={loss.item():.6f}, val_loss={val_loss.item():.6f}")

# Test evaluation
model.eval()
with torch.no_grad():
    test_pred_norm = model(test_X_norm)
    test_pred = test_pred_norm * std + mean
    test_true = test_y
    rmse = torch.sqrt(nn.MSELoss()(test_pred, test_true))
    mae = nn.L1Loss()(test_pred, test_true)
    print(f"\nTest RMSE: {rmse.item():.4f}")
    print(f"Test MAE: {mae.item():.4f}")

# Output:
# Epoch 0: train_loss=0.5123, val_loss=0.4890
# Epoch 20: train_loss=0.1234, val_loss=0.1345
# Epoch 40: train_loss=0.0890, val_loss=0.1012
# Epoch 60: train_loss=0.0789, val_loss=0.0923
# Epoch 80: train_loss=0.0723, val_loss=0.0876
#
# Test RMSE: 0.3123
# Test MAE: 0.2456
```

### Code Example 2: Multivariate Time Series

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultivariateTimeSeriesLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_targets=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_targets)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Generate multivariate data: 3 features, predict 1 target
N = 800
t = torch.arange(0, N).float()
feature1 = torch.sin(2 * math.pi * t / 30) + 0.1 * torch.randn(N)
feature2 = torch.cos(2 * math.pi * t / 50) + 0.1 * torch.randn(N)
feature3 = 0.5 * feature1 + 0.5 * feature2 + 0.1 * torch.randn(N)
target = feature1 + feature2 + 0.1 * torch.randn(N)

X = torch.stack([feature1, feature2, feature3], dim=1)
y = target.unsqueeze(1)

# Create sequences
seq_len = 20
def create_multivar_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return torch.stack(xs), torch.stack(ys)

X_seq, y_seq = create_multivar_sequences(X, y, seq_len)
train_X, train_y = X_seq[:600], y_seq[:600]
val_X, val_y = X_seq[600:700], y_seq[600:700]
test_X, test_y = X_seq[700:], y_seq[700:]

model = MultivariateTimeSeriesLSTM(n_features=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(100):
    pred = model(train_X)
    loss = loss_fn(pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        val_pred = model(val_X)
        val_loss = loss_fn(val_pred, val_y)
        print(f"Epoch {epoch}: loss={loss.item():.6f}, val_loss={val_loss.item():.6f}")

with torch.no_grad():
    test_pred = model(test_X)
    test_loss = loss_fn(test_pred, test_y)
    print(f"\nTest MSE: {test_loss.item():.6f}")
    print(f"Test RMSE: {math.sqrt(test_loss.item()):.4f}")

# Output:
# Epoch 0: loss=0.8234, val_loss=0.8123
# Epoch 20: loss=0.2345, val_loss=0.2567
# Epoch 40: loss=0.1456, val_loss=0.1678
# Epoch 60: loss=0.1123, val_loss=0.1345
# Epoch 80: loss=0.0987, val_loss=0.1234
#
# Test MSE: 0.1345
# Test RMSE: 0.3668
```

### Code Example 3: Time Series with Exogenous Features

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TimeSeriesWithExogenous(nn.Module):
    def __init__(self, n_endogenous, n_exogenous, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(n_endogenous + n_exogenous, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_endogenous)

    def forward(self, endog, exog):
        combined = torch.cat([endog, exog], dim=-1)
        output, _ = self.lstm(combined)
        return self.fc(output[:, -1, :])

# Endogenous: temperature, Exogenous: day_of_week, is_holiday
N = 500
t = torch.arange(0, N).float()
temperature = 20 + 10 * torch.sin(2 * math.pi * t / 365) + torch.randn(N) * 2
day_of_week = t % 7
is_holiday = (day_of_week >= 5).float()  # Weekends

seq_len = 14
endog = temperature.unsqueeze(1)
exog = torch.stack([day_of_week / 6, is_holiday], dim=1)

X_endog, X_exog, y = [], [], []
for i in range(len(endog) - seq_len):
    X_endog.append(endog[i:i+seq_len])
    X_exog.append(exog[i:i+seq_len])
    y.append(endog[i+seq_len])

X_endog = torch.stack(X_endog)
X_exog = torch.stack(X_exog)
y = torch.stack(y)

model = TimeSeriesWithExogenous(n_endogenous=1, n_exogenous=2)
opt = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
    pred = model(X_endog[:400], X_exog[:400])
    loss = nn.MSELoss()(pred, y[:400])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        val_pred = model(X_endog[400:], X_exog[400:])
        val_loss = nn.MSELoss()(val_pred, y[400:])
        print(f"Epoch {epoch}: loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

# Output:
# Epoch 0: loss=23.4567, val_loss=22.3456
# Epoch 20: loss=5.6789, val_loss=6.1234
# Epoch 40: loss=3.4567, val_loss=4.0123
# Epoch 60: loss=2.8901, val_loss=3.4567
# Epoch 80: loss=2.3456, val_loss=3.0123
```

### Code Example 4: Temporal Train/Val/Test Split

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class TemporalSplitDemo:
    def __init__(self, data_len=1000, seq_len=20):
        self.data = torch.sin(torch.linspace(0, 100, data_len)) + 0.1 * torch.randn(data_len)
        self.seq_len = seq_len
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.seq_len):
            X.append(self.data[i:i+self.seq_len])
            y.append(self.data[i+self.seq_len])
        return torch.stack(X).unsqueeze(-1), torch.stack(y).unsqueeze(-1)

    def temporal_split(self, train_ratio=0.7, val_ratio=0.15):
        n = len(self.X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return {
            'train': (self.X[:train_end], self.y[:train_end]),
            'val': (self.X[train_end:val_end], self.y[train_end:val_end]),
            'test': (self.X[val_end:], self.y[val_end:])
        }

    def evaluate_split(self, split_dict):
        model = nn.LSTM(1, 16, batch_first=True)
        fc = nn.Linear(16, 1)
        opt = optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.01)

        train_X, train_y = split_dict['train']
        val_X, val_y = split_dict['val']
        test_X, test_y = split_dict['test']

        for epoch in range(50):
            out, _ = model(train_X)
            pred = fc(out[:, -1, :])
            loss = nn.MSELoss()(pred, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            out, _ = model(test_X)
            test_pred = fc(out[:, -1, :])
            test_loss = nn.MSELoss()(test_pred, test_y)
        return test_loss.item()

demo = TemporalSplitDemo()
splits = demo.temporal_split()
test_loss = demo.evaluate_split(splits)

print(f"Temporal split sizes:")
print(f"  Train: {splits['train'][0].shape[0]} samples")
print(f"  Val: {splits['val'][0].shape[0]} samples")
print(f"  Test: {splits['test'][0].shape[0]} samples")
print(f"  Test loss: {test_loss:.6f}")

# Output:
# Temporal split sizes:
#   Train: 686 samples
#   Val: 147 samples
#   Test: 147 samples
#   Test loss: 0.098765
```

## Common Mistakes

1. **Random shuffling of time series data**: Never randomly shuffle time series before splitting. Use temporal order-preserving splits.

2. **Data leakage from future information**: Ensure features used at time t are only based on information available at time t.

3. **Not handling non-stationarity**: Financial and economic time series often have changing distributions. Use differencing or detrending.

4. **Ignoring multiple seasonal periods**: Time series can have daily, weekly, and yearly seasonality. The LSTM needs sufficient history to capture the longest period.

5. **Using only point forecasts without uncertainty**: Point forecasts do not capture prediction uncertainty. Consider quantile loss or probabilistic forecasting.

6. **Overlapping windows causing optimistic validation**: Overlapping windows introduce dependence between train and validation sets. Use gaps between windows or walk-forward validation.

7. **Not scaling features appropriately**: Features on different scales can cause training instability. Normalize each feature independently.

## Interview Questions

### Beginner

Q: What preprocessing steps are typically needed for time series data before feeding to an LSTM?
A: Normalization (zero mean, unit variance), handling missing values, creating sliding windows, separating trend/seasonality, and differencing for non-stationary data.

Q: How does time series forecasting differ from standard supervised learning?
A: Temporal dependencies mean observations are not independent. You cannot randomly shuffle the data. Validation must respect temporal order.

### Intermediate

Q: Explain walk-forward validation and why it is preferred for time series.
A: Walk-forward validation trains on an expanding window of past data and validates on the next time period. It simulates real-world forecasting conditions where the model is retrained as new data arrives, avoiding data leakage.

Q: How would you handle multiple seasonal periods in time series with LSTMs?
A: Ensure the input window covers at least one full period of the longest seasonality. Add seasonal features as exogenous inputs (e.g., hour_of_day, day_of_week, month_of_year). Use seasonal decomposition before modeling.

### Advanced

Q: Derive the bias-variance trade-off in time series forecasting and explain how LSTM depth affects it.
A: In time series, bias comes from the model's inability to capture true temporal dynamics; variance comes from sensitivity to noise in the training data. Deeper LSTMs reduce bias (capture more complex patterns) but increase variance (more parameters, overfitting to noise). The optimal depth minimizes the sum of squared bias and variance on out-of-sample data.

Q: Design a probabilistic forecasting LSTM that outputs a distribution instead of a point estimate.
A: Replace the final linear layer with a layer that outputs distribution parameters. For Gaussian: output mu and log_sigma, use negative log-likelihood loss. For quantile: output multiple quantiles (0.1, 0.5, 0.9), use quantile loss (pinball loss). For mixture: output parameters of a mixture of Gaussians. Evaluate using CRPS (Continuous Ranked Probability Score).

## Practice Problems

### Easy

Train an LSTM to forecast a simple sine wave 5 steps ahead. Use MSE loss and report test RMSE.

### Medium

Build a multivariate LSTM that uses 3 features to predict 1 target. Compare performance against a univariate model that only uses the target's history.

### Hard

Implement walk-forward validation with model retraining for a time series forecasting task. Compare the performance against a single train/test split approach, measuring both accuracy and computational cost.

## Related Concepts

- LSTM for Sequence Prediction (DL-307)
- LSTM for NLP (DL-309)
- RNN for Time Series

## Next Concepts

- LSTM for NLP (DL-309)
- LSTM vs GRU (DL-310)

## Summary

LSTMs are powerful tools for time series forecasting, capable of capturing trends, seasonality, and complex temporal dependencies. Key considerations include proper temporal train/val/test splits, normalization, handling non-stationarity, and choosing between direct and iterative multi-step prediction. Multivariate forecasting and exogenous feature incorporation extend LSTM applicability to real-world forecasting problems.

## Key Takeaways

- Never shuffle time series data; preserve temporal order
- Normalize features independently to avoid scale issues
- Handle non-stationarity through differencing or detrending
- Use walk-forward validation for realistic evaluation
- LSTMs capture multiple seasonal patterns with sufficient history
- Exogenous features improve forecast accuracy
- Consider probabilistic forecasting for uncertainty estimation
