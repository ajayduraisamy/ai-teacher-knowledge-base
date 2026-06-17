# Concept: Time Series Forecasting

## Concept ID

ML-092

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Explain stationarity and apply the Augmented Dickey-Fuller test
- Identify ARIMA and SARIMA model orders using ACF/PACF plots
- Implement exponential smoothing (Holt-Winters) for trend and seasonality
- Build LSTM/GRU models for sequence prediction
- Use TimeSeriesSplit for robust time series cross-validation
- Forecast with Facebook Prophet for business time series

## Prerequisites

- Basic probability and statistics (mean, variance, autocorrelation)
- Python with pandas, numpy, and matplotlib
- Linear regression fundamentals
- Basic neural network concepts (for the LSTM section)

## Definition

Time series forecasting is the task of predicting future values based on historically observed temporal patterns. Unlike standard supervised learning, time series data has a temporal ordering that must be preserved during training, validation, and testing. Key components include trend (long-term direction), seasonality (regular periodic patterns), cyclical components (irregular long-term patterns), and residual noise.

## Intuition

Imagine recording the temperature every day for five years. You notice it is colder in January and hotter in July (seasonality), generally rising over decades due to climate change (trend), with some unusually hot or cold days (noise). A good time series model captures the repeating yearly pattern and the gradual upward trend so it can forecast next summer's temperatures more accurately than a naive average.

## Why This Concept Matters

Time series forecasting is critical across industries: retail demand forecasting (inventory optimization), energy load prediction (grid management), financial market prediction (risk management), weather forecasting, healthcare (epidemic spread prediction), and cloud resource auto-scaling. Poor forecasts lead to stockouts, wasted capacity, or financial losses. According to McKinsey, improved forecasting can reduce inventory costs by 10-30%.

## Mathematical Explanation

### Stationarity

A time series is strictly stationary if its joint distribution is invariant to time shifts. Weak stationarity (practical definition) requires:

1. Constant mean: E[Y_t] = mu for all t
2. Constant variance: Var(Y_t) = sigma^2 for all t
3. Autocovariance depends only on lag: Cov(Y_t, Y_{t-k}) = gamma(k)

### Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis that a unit root is present (non-stationary). The test regression:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

H0: gamma = 0 (non-stationary). If the ADF statistic is more negative than the critical value, reject H0.

### ARIMA(p,d,q)

AR: Autoregressive — uses past values: y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p}
I: Integrated — differencing to achieve stationarity (d times)
MA: Moving Average — uses past forecast errors: y_t = c + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}

Combined ARIMA(p,d,q):

$$\Delta^d y_t = c + \sum_{i=1}^{p} \phi_i \Delta^d y_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$$

Where Delta^d is differencing applied d times.

### SARIMA(p,d,q)(P,D,Q,m)

Adds seasonal components with period m:

$$\Phi_P(B^m) \phi_p(B) (1-B)^d (1-B^m)^D y_t = \Theta_Q(B^m) \theta_q(B) \varepsilon_t$$

Where B is the backshift operator (B y_t = y_{t-1}).

### Exponential Smoothing (Holt-Winters)

Level: L_t = alpha * Y_t + (1 - alpha)(L_{t-1} + T_{t-1})
Trend: T_t = beta * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
Seasonality: S_t = gamma * (Y_t - L_t) + (1 - gamma) * S_{t-m}
Forecast: F_{t+h} = L_t + h * T_t + S_{t+h-m}

### LSTM for Forecasting

An LSTM cell has forget gate f_t, input gate i_t, output gate o_t, and cell state C_t:

f_t = sigma(W_f * [h_{t-1}, x_t] + b_f)
i_t = sigma(W_i * [h_{t-1}, x_t] + b_i)
C_t = f_t * C_{t-1} + i_t * tanh(W_C * [h_{t-1}, x_t] + b_C)
o_t = sigma(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)

## Code Examples

### Example 1: Stationarity Check and Differencing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate non-stationary series (random walk + trend)
np.random.seed(42)
n = 200
t = np.arange(n)
trend = 0.02 * t
random_walk = np.cumsum(np.random.randn(n) * 0.5)
y = 10 + trend + random_walk

# ADF test on original series
result = adfuller(y)
print(f"ADF Statistic (original): {result[0]:.4f}")
print(f"p-value (original): {result[1]:.4f}")
print("Critical values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("=> Series is stationary (reject H0)")
else:
    print("=> Series is NON-stationary")
# Output:
# ADF Statistic (original): -1.8227
# p-value (original): 0.3690
# Critical values:
#   1%: -3.4648
#   5%: -2.8755
#   10%: -2.5743
# => Series is NON-stationary

# Apply first differencing
y_diff = np.diff(y)
result_diff = adfuller(y_diff)
print(f"\nADF Statistic (differenced): {result_diff[0]:.4f}")
print(f"p-value (differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("=> Differenced series IS stationary")
# Output:
# ADF Statistic (differenced): -13.5280
# p-value (differenced): 0.0000
# => Differenced series IS stationary

# ACF and PACF plots to determine p and q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(y_diff, lags=20, ax=ax1)
plot_pacf(y_diff, lags=20, ax=ax2)
plt.tight_layout()
plt.show()
```

### Example 2: ARIMA and SARIMA with statsmodels

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate seasonal data
np.random.seed(42)
n = 200
t = np.arange(n)
seasonal = 5 * np.sin(2 * np.pi * t / 12)  # yearly seasonality (monthly data)
trend = 0.05 * t
noise = np.random.randn(n) * 0.5
y = 20 + trend + seasonal + noise

# Train/test split (time order preserved!)
train = y[:160]
test = y[160:]

# Fit ARIMA(1,1,1)
model_arima = ARIMA(train, order=(1, 1, 1))
result_arima = model_arima.fit()
print(result_arima.summary())

# Forecast
pred_arima = result_arima.forecast(steps=len(test))

# Fit SARIMA(1,1,1)(1,1,1,12)
model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result_sarima = model_sarima.fit(disp=False)

# Forecast
pred_sarima = result_sarima.forecast(steps=len(test))

# Compare
mae_arima = mean_absolute_error(test, pred_arima)
mae_sarima = mean_absolute_error(test, pred_sarima)
rmse_arima = np.sqrt(mean_squared_error(test, pred_arima))
rmse_sarima = np.sqrt(mean_squared_error(test, pred_sarima))

print(f"ARIMA - MAE: {mae_arima:.3f}, RMSE: {rmse_arima:.3f}")
print(f"SARIMA - MAE: {mae_sarima:.3f}, RMSE: {rmse_sarima:.3f}")
# Output:
# ARIMA - MAE: 2.198, RMSE: 2.639
# SARIMA - MAE: 1.196, RMSE: 1.557
```

### Example 3: LSTM for Time Series Forecasting

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU

# Generate sine wave data
np.random.seed(42)
n = 500
t = np.linspace(0, 50, n)
y = np.sin(t) + np.random.randn(n) * 0.1

# Prepare sequences
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Create sequences
seq_length = 20
X, y_seq = create_sequences(data_scaled, seq_length)

# Train/test split (temporal!)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Reshape for LSTM: (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=False, input_shape=(seq_length, 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=0)

# Predict
y_pred = model.predict(X_test, verbose=0)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(((y_test_inv - y_pred_inv) ** 2).mean())
print(f"LSTM - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
# Output: LSTM - MAE: 0.0640, RMSE: 0.0839

# Compare with naive forecast (last value)
naive_pred = X_test[:, -1, 0]
naive_pred_inv = scaler.inverse_transform(naive_pred.reshape(-1, 1))
mae_naive = mean_absolute_error(y_test_inv, naive_pred_inv)
print(f"Naive (last value) - MAE: {mae_naive:.4f}")
# Output: Naive (last value) - MAE: 0.1269
```

## Common Mistakes

1. **Not using time-based cross-validation**: Using KFold or ShuffleSplit randomly shuffles time, leaking future information into the training set. Always use TimeSeriesSplit or expanding window validation.

2. **Failing to check stationarity**: ARIMA models require stationary data. Without the ADF test and appropriate differencing, forecasts diverge or produce spurious results.

3. **Over-differencing**: Applying differencing when the series is already stationary introduces unnecessary autocorrelation (moving average roots near unity). Use the ADF test to confirm you need d=1.

4. **Ignoring seasonality**: Many real-world series have weekly, monthly, or yearly cycles. A plain ARIMA will miss these patterns. Always check ACF for significant lags at seasonal periods.

5. **Using error metrics that treat all errors equally**: Time series errors compound. A small error at step 1 propagates. Use multi-step metrics like MAPE, sMAPE, or MASE, and evaluate on the full forecast horizon, not just one-step-ahead.

6. **Optimizing hyperparameters on the test set**: Tuning p, d, q or LSTM architecture using test set performance overfits to that specific forecast window. Use a validation set or time series cross-validation.

7. **Scaling data incorrectly for LSTMs**: Fit the scaler only on training data, then transform test data using the training scaler. Transforming test data independently causes data leakage.

8. **Predicting too far into the future**: Forecast uncertainty grows with horizon. Always report prediction intervals, not just point forecasts.

## Interview Questions

### Beginner

1. What is stationarity and why does it matter for time series forecasting?
2. Explain the components of an ARIMA(p,d,q) model.
3. What is the difference between white noise and a random walk?
4. How do ACF and PACF plots help determine AR and MA orders?
5. What is the difference between trend and seasonality?

### Intermediate

1. Explain the Box-Jenkins methodology for ARIMA model selection.
2. How would you handle multiple seasonalities (e.g., daily + weekly patterns) in a time series?
3. Compare exponential smoothing (Holt-Winters) with ARIMA. When would you use each?
4. How does TimeSeriesSplit differ from KFold and why is it necessary?
5. Explain how an LSTM avoids the vanishing gradient problem compared to vanilla RNNs.

### Advanced

1. Design a hybrid model that combines Prophet with an LSTM for demand forecasting. How would you ensemble them?
2. How would you implement probabilistic forecasting (predicting the full distribution, not just the mean) using deep learning?
3. Explain the concept of cointegration and how it can be used for pairs trading in finance.

## Practice Problems

### Easy

1. Generate a random walk and apply the ADF test. How many times do you need to difference it?
2. Plot the ACF and PACF for an AR(2) process with phi = [0.6, -0.3]. What pattern do you see?
3. Compute the MAE and RMSE for forecasts [10, 12, 13] and actuals [9, 11, 15].
4. Apply first differencing to the series [5, 7, 6, 8, 10, 9] and check stationarity.
5. Using statsmodels, fit an ARIMA(0,1,0) model (random walk) to a series and forecast 5 steps ahead.

### Medium

1. Use auto_arima from pmdarima to find the optimal (p,d,q) for the AirPassengers dataset and compare with manual selection.
2. Implement Holt-Winters exponential smoothing with additive seasonality from scratch.
3. Build an LSTM model to forecast the next 30 days of electricity demand and compare with SARIMA.
4. Use Prophet to forecast daily website traffic with weekly and yearly seasonality and holiday effects.
5. Implement TimeSeriesSplit with 5 folds and evaluate an ARIMA model on each fold, reporting the mean and std of RMSE.

### Hard

1. Implement a sequence-to-sequence LSTM model for multi-step forecasting (predict 24 hours ahead using 72 hours of history).
2. Build a Bayesian structural time series model using PyMC for forecasting with uncertainty quantification.
3. Implement a Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting with static and time-varying features.

## Solutions

### Easy 1 — Random walk differencing
```python
import numpy as np
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)
rw = np.cumsum(np.random.randn(200))

def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: ADF={result[0]:.3f}, p={result[1]:.4f}, " +
          f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'}")

check_stationarity(rw, "Original")
check_stationarity(np.diff(rw), "After 1 diff")
# Output:
# Original: ADF=-1.123, p=0.7057, Non-stationary
# After 1 diff: ADF=-13.928, p=0.0000, Stationary
```

### Easy 4 — Differencing
```python
series = np.array([5, 7, 6, 8, 10, 9])
diff1 = np.diff(series)
print(f"Original: {series}")
print(f"1st diff: {diff1}")
# Output:
# Original: [ 5  7  6  8 10  9]
# 1st diff: [ 2 -1  2  2 -1]
```

## Related Concepts

- Feature Engineering — ML-070
- Overfitting and Regularization — ML-067
- RNNs and LSTMs — ML-086
- Cross-Validation — ML-066

## Next Concepts

- Fraud Detection — ML-093
- NLP with ML — ML-094
- Recommender Systems — ML-091

## Summary

Time series forecasting predicts future values from historical temporal patterns. ARIMA/SARIMA models capture autocorrelation and seasonality through differencing and autoregressive/moving average terms. Exponential smoothing provides an intuitive alternative with level, trend, and seasonal components. LSTM/GRU models excel at capturing complex nonlinear dependencies in long sequences. Proper evaluation requires time-based cross-validation (TimeSeriesSplit) and error metrics that account for the forecast horizon. Stationarity checking via the ADF test is a critical prerequisite for classical models.

## Key Takeaways

- Always maintain temporal ordering in train/test splits
- ADF test determines if differencing is needed (d in ARIMA)
- ACF and PACF plots guide p and q selection
- SARIMA extends ARIMA with seasonal components
- LSTMs handle long-range dependencies via gated cell states
- TimeSeriesSplit prevents future leakage in cross-validation
- Forecast uncertainty grows with horizon — always report intervals
- Simple models (exponential smoothing) often beat complex ones on clean seasonal data
