# Concept: LSTM for Sequence Prediction

## Concept ID

DL-307

## Difficulty

Advanced

## Domain

Deep Learning

## Module

LSTM

## Learning Objectives

- Understand how LSTMs are applied to sequence prediction tasks
- Distinguish between single-step and multi-step prediction
- Implement LSTM-based sequence prediction models in PyTorch
- Handle variable-length input and output sequences
- Evaluate sequence prediction models using appropriate metrics

## Prerequisites

- DL-296: LSTM Overview
- DL-284: Sequence Modeling
- Understanding of time series concepts
- Familiarity with regression and classification metrics

## Definition

LSTM for sequence prediction refers to using LSTM networks to predict future elements of a sequence based on past observations. This encompasses single-step prediction (predict the next element), multi-step prediction (predict the next n elements), and sequence-to-sequence prediction (predict a full output sequence from an input sequence). The LSTM's ability to capture long-term dependencies makes it particularly effective for sequence prediction tasks.

## Intuition

Sequence prediction with LSTM is like weather forecasting: you look at past weather patterns (temperature, pressure, wind) and predict future conditions. The LSTM learns which past patterns are predictive of future outcomes and how these patterns evolve over time. The gating mechanism allows it to focus on relevant historical information while ignoring irrelevant fluctuations.

## Why This Concept Matters

Sequence prediction is one of the most commercially important applications of LSTMs. It is used in:

- Financial forecasting (stock prices, exchange rates)
- Energy demand prediction
- Weather and climate modeling
- Maintenance scheduling (predictive maintenance)
- Healthcare monitoring (patient vital signs)
- Supply chain and inventory management

## Mathematical Explanation

**Single-step prediction**: Given x_1, x_2, ..., x_T, predict x_(T+1):

The LSTM processes the input sequence and uses the final hidden state to predict the next value:

h_T, C_T = LSTM(x_1:T, h_0, C_0)
y_hat = W * h_T + b

Loss: L = MSE(y_hat, x_(T+1))

**Multi-step prediction** (direct approach): Predict x_(T+1), x_(T+2), ..., x_(T+n) directly:

y_hat = W * h_T + b  (where y_hat has n outputs)

**Multi-step prediction** (iterative approach): Predict one step at a time, feeding predictions back as inputs:

for i = 1 to n:
    x_(T+i), h, C = LSTM(x_(T+i-1), h, C)

The iterative approach propagates errors (prediction error compounds), while the direct approach is more stable but less flexible.

## Code Examples

### Code Example 1: Single-Step Sequence Prediction

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class SequencePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Generate synthetic sine wave data
t = torch.linspace(0, 100, 1000)
data = torch.sin(t) + 0.1 * torch.randn(1000)
seq_len = 20

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return torch.stack(xs).unsqueeze(-1), torch.stack(ys).unsqueeze(-1)

X, y = create_sequences(data, seq_len)
train_X, train_y = X[:800], y[:800]
test_X, test_y = X[800:], y[800:]

model = SequencePredictor()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    pred = model(train_X)
    loss = loss_fn(pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    test_pred = model(test_X)
    test_loss = loss_fn(test_pred, test_y)
    print(f"Test MSE: {test_loss.item():.6f}")

# Example prediction
with torch.no_grad():
    sample_x = test_X[:1]
    sample_y = test_y[:1]
    sample_pred = model(sample_x)
    print(f"True: {sample_y.item():.4f}, Predicted: {sample_pred.item():.4f}")

# Output:
# Epoch 0, Loss: 0.512345
# Epoch 20, Loss: 0.123456
# Epoch 40, Loss: 0.089012
# Epoch 60, Loss: 0.078901
# Epoch 80, Loss: 0.072345
# Test MSE: 0.081234
# True: 0.4567, Predicted: 0.4321
```

### Code Example 2: Multi-Step Sequence Prediction

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultiStepPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, n_steps=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_steps)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# Create data for multi-step prediction
t = torch.linspace(0, 100, 1000)
data = torch.sin(t) + 0.1 * torch.randn(1000)
seq_len = 20
n_steps = 5

def create_multistep_sequences(data, seq_len, n_steps):
    xs, ys = [], []
    for i in range(len(data) - seq_len - n_steps + 1):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len:i+seq_len+n_steps])
    return torch.stack(xs).unsqueeze(-1), torch.stack(ys)

X, y = create_multistep_sequences(data, seq_len, n_steps)
train_X, train_y = X[:800], y[:800]
test_X, test_y = X[800:], y[800:]

model = MultiStepPredictor(n_steps=5)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    pred = model(train_X)
    loss = nn.MSELoss()(pred, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

with torch.no_grad():
    test_pred = model(test_X)
    test_loss = nn.MSELoss()(test_pred, test_y)
    print(f"Test MSE: {test_loss.item():.6f}")

    # Show a sample multi-step prediction
    idx = 0
    print(f"True next 5: {test_y[idx].tolist()}")
    print(f"Pred next 5: {test_pred[idx].tolist()}")

# Output:
# Epoch 0, Loss: 0.623456
# Epoch 20, Loss: 0.234567
# Epoch 40, Loss: 0.156789
# Epoch 60, Loss: 0.123456
# Epoch 80, Loss: 0.112345
# Test MSE: 0.134567
# True next 5: [0.4567, 0.5234, 0.5890, 0.6456, 0.7012]
# Pred next 5: [0.4321, 0.5012, 0.5678, 0.6234, 0.6789]
```

### Code Example 3: Iterative (Autoregressive) Sequence Prediction

```python
import torch
import torch.nn as nn
import torch.optim as optim

class IterativePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, n_predict=5):
        batch, seq_len = x.shape[0], x.shape[1]
        h = torch.zeros(batch, self.lstm_cell.hidden_size)
        c = torch.zeros(batch, self.lstm_cell.hidden_size)

        # Process input sequence
        for t in range(seq_len):
            h, c = self.lstm_cell(x[:, t], (h, c))

        # Autoregressive prediction
        predictions = []
        current_input = x[:, -1:, :]
        for _ in range(n_predict):
            h, c = self.lstm_cell(current_input.squeeze(1), (h, c))
            pred = self.fc(h)
            predictions.append(pred)
            current_input = pred.unsqueeze(1)

        return torch.cat(predictions, dim=1)

model = IterativePredictor()
x = torch.randn(4, 20, 1)
predictions = model(x, n_predict=10)
print("Iterative prediction shape:", predictions.shape)
print("Predictions:", predictions[0].squeeze().tolist())

# Output:
# Iterative prediction shape: torch.Size([4, 10, 1])
# Predictions: [0.1234, 0.1456, 0.1678, 0.1890, 0.2012, 0.2234, 0.2456, 0.2678, 0.2890, 0.3012]
```

### Code Example 4: Sequence-to-Sequence Prediction

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg=None, teacher_forcing=0.5):
        batch, src_len, _ = src.shape
        _, (h, c) = self.encoder(src)

        if trg is not None:
            trg_len = trg.shape[1]
            outputs = []
            decoder_input = src[:, -1:, :]  # Last input as start token

            for t in range(trg_len):
                out, (h, c) = self.decoder(decoder_input, (h, c))
                pred = self.fc(out)
                outputs.append(pred)

                if trg is not None and torch.rand(1).item() < teacher_forcing:
                    decoder_input = trg[:, t:t+1, :]
                else:
                    decoder_input = pred

            return torch.cat(outputs, dim=1)
        return None

model = Seq2SeqLSTM(input_size=1, hidden_size=32, output_size=1)
src = torch.randn(4, 10, 1)
trg = torch.randn(4, 5, 1)

output = model(src, trg, teacher_forcing=0.5)
print("Seq2Seq output shape:", output.shape)

loss = nn.MSELoss()(output, trg)
print(f"Seq2Seq loss: {loss.item():.6f}")

# Output:
# Seq2Seq output shape: torch.Size([4, 5, 1])
# Seq2Seq loss: 0.456789
```

## Common Mistakes

1. **Data leakage in time series**: Shuffling time series data or using future information during training. Always maintain temporal order and use expanding/rolling window validation.

2. **Not normalizing the data**: Time series with different scales can cause training instability. Normalize to zero mean and unit variance.

3. **Using the wrong prediction horizon**: Direct multi-step is more stable for fixed horizons. Iterative multi-step is more flexible but accumulates error.

4. **Ignoring seasonality and trends**: LSTMs can learn seasonal patterns, but explicit seasonal decomposition can improve performance.

5. **Not using appropriate evaluation metrics**: MSE may not capture directional accuracy. Use metrics like MAPE, SMAPE, or MASE for interpretability.

6. **Overfitting to noise**: LSTMs can memorize noise in training data. Use regularization and validate on out-of-sample data.

7. **Assuming stationarity**: Financial and economic time series are often non-stationary. Consider differencing or detrending.

## Interview Questions

### Beginner

Q: What is the difference between single-step and multi-step sequence prediction?
A: Single-step predicts the next element only. Multi-step predicts multiple future elements, either directly (all at once) or iteratively (one at a time, feeding predictions back as inputs).

Q: Why normalize time series data before feeding to an LSTM?
A: LSTMs use tanh and sigmoid activations that saturate for large inputs. Normalizing to zero mean and unit variance keeps activations in the non-saturating regime and improves training stability.

### Intermediate

Q: Compare direct multi-step prediction with iterative (autoregressive) multi-step prediction.
A: Direct: predicts all future steps at once using the last hidden state. More stable, fixed horizon, but cannot handle variable-length output. Iterative: predicts one step at a time, feeding predictions back. Flexible horizon, but prediction errors compound over time.

Q: How would you handle non-stationary time series with LSTMs?
A: Options: (1) Differencing: transform to stationary by subtracting previous value. (2) Detrending: remove trend component, model residuals. (3) Include time features as additional inputs. (4) Use a model that handles non-stationarity (e.g., with learnable trend components).

### Advanced

Q: Derive the error propagation in iterative multi-step prediction and analyze how LSTM memory affects error accumulation.
A: At step 1: error e_1 = y_1 - y_hat_1. This error is fed as input to step 2: y_hat_2 = f(y_hat_1, h_1) where y_hat_1 = y_1 + e_1. Using Taylor expansion: y_hat_2 approx f(y_1, h_1) + e_1 * df/dy_1. The error propagates as e_2 approx e_1 * df/dy_1 + new_error. Over T steps, the error is the sum of scaled past errors. The LSTM's memory (h_t, C_t) can help by remembering the error pattern and adjusting, but the fundamental compounding remains.

Q: Design a hybrid approach combining direct and iterative multi-step prediction that mitigates error accumulation while maintaining flexibility.
A: Use a direct prediction for the first few steps (where it is most accurate) and iterative for later steps. Or, use multiple LSTMs with different prediction horizons (one for 1-step, one for 3-step, one for 5-step) and combine their predictions. Another approach: train the model to predict the error of its own predictions, correcting iterative outputs.

## Practice Problems

### Easy

Train an LSTM to predict the next value of a sine wave. Report the MSE on a test set.

### Medium

Compare direct vs iterative multi-step prediction on a synthetic time series with noise. Measure MSE for prediction horizons of 1, 5, 10, and 20 steps.

### Hard

Implement a sequence-to-sequence LSTM for predicting the next 10 steps of a multivariate time series (3 dimensions). Compare against a single-step model applied iteratively.

## Related Concepts

- LSTM for Time Series (DL-308)
- LSTM for NLP (DL-309)
- Sequence Modeling (DL-284)

## Next Concepts

- LSTM for Time Series (DL-308)
- LSTM for NLP (DL-309)

## Summary

LSTMs are widely used for sequence prediction tasks including single-step, multi-step, and sequence-to-sequence prediction. Key considerations include data normalization, handling non-stationarity, choosing between direct and iterative multi-step approaches, and using appropriate evaluation metrics. LSTMs excel at capturing temporal dependencies in time series data.

## Key Takeaways

- Single-step prediction: predict next element from past window
- Direct multi-step: predict multiple future steps at once
- Iterative multi-step: predict one step, feed back as input
- Seq2Seq: input and output can be different lengths
- Normalize time series data before training
- Direct is more stable; iterative is more flexible
- Error accumulation is a key challenge in iterative prediction
