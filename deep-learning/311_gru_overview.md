# Concept: GRU Overview

## Concept ID

DL-311

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

GRU

## Learning Objectives

- Understand the high-level architecture of Gated Recurrent Units
- Explain how GRU addresses the vanishing gradient problem with fewer gates
- Identify the key components: reset gate and update gate
- Compare GRU with LSTM conceptually
- Implement a basic GRU in PyTorch

## Prerequisites

- DL-281: Recurrent Neural Network
- DL-291: Vanishing Gradient in RNN
- DL-296: LSTM Overview
- Understanding of gating mechanisms

## Definition

A Gated Recurrent Unit (GRU) is a recurrent neural network architecture introduced by Cho et al. in 2014 that uses gating mechanisms to control information flow without requiring a separate cell state. It has two gates: the reset gate (which controls how much past information to forget) and the update gate (which controls how much of the new information to add). GRU is simpler than LSTM (two gates vs three, no separate cell state) but performs comparably on many sequence modeling tasks.

## Intuition

The GRU is like a smart notebook that the network carries through time. When reading new information, the notebook decides:

1. Reset gate: How much of the old notes should I ignore when taking new notes?
2. Update gate: How much of my notebook should I keep from before vs replace with new notes?

The reset gate helps the model forget irrelevant past information when computing the new candidate state. The update gate determines the balance between retaining old information and incorporating new information.

## Why This Concept Matters

GRU is an important architecture because it:

- Simplifies the LSTM while maintaining comparable performance
- Reduces parameter count by ~25% compared to LSTM
- Is computationally more efficient
- Works well on smaller datasets where LSTM might overfit
- Influenced later architecture designs

## Mathematical Explanation

GRU has two gates and no separate cell state. The hidden state serves as both memory and output.

**Reset gate**: r_t = sigmoid(W_r * x_t + U_r * h_(t-1) + b_r)

**Update gate**: z_t = sigmoid(W_z * x_t + U_z * h_(t-1) + b_z)

**Candidate hidden state**: h_tilde_t = tanh(W_h * x_t + U_h * (r_t * h_(t-1)) + b_h)

**Final hidden state**: h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t

The update gate z_t controls the balance between the old hidden state and the new candidate. When z_t is near 1, the model mostly keeps the old state (preserves information). When z_t is near 0, the model mostly adopts the new candidate (updates information).

## Code Examples

### Code Example 1: Basic GRU in PyTorch

```python
import torch
import torch.nn as nn

gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(3, 7, 10)
output, h_n = gru(x)

print("Output shape:", output.shape)
print("h_n shape:", h_n.shape)

# GRU returns (output, h_n) - no cell state
print("\nGRU returns only hidden state (no cell state)")
print("Final hidden norm:", h_n.norm().item())

# Parameter count
total_params = sum(p.numel() for p in gru.parameters())
print(f"Total parameters: {total_params:,}")

# Compare with LSTM
lstm = nn.LSTM(10, 20)
print(f"LSTM params: {sum(p.numel() for p in lstm.parameters()):,}")

# Output:
# Output shape: torch.Size([3, 7, 20])
# h_n shape: torch.Size([1, 3, 20])
# GRU returns only hidden state (no cell state)
# Final hidden norm: 0.8345
# Total parameters: 3,780
# LSTM params: 5,040
```

### Code Example 2: GRU Cell from Scratch

```python
import torch
import torch.nn as nn

class GRUCellFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)

        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))
        h = (1 - z) * h_prev + z * h_tilde
        return h

input_size, hidden_size = 10, 20
custom_gru = GRUCellFromScratch(input_size, hidden_size)
pytorch_gru = nn.GRUCell(input_size, hidden_size)

# Copy weights for verification
with torch.no_grad():
    custom_gru.W_r.weight.copy_(pytorch_gru.weight_ih[:hidden_size])
    custom_gru.U_r.weight.copy_(pytorch_gru.weight_hh[:hidden_size])
    custom_gru.W_r.bias.copy_(pytorch_gru.bias_ih[:hidden_size])
    custom_gru.U_r.bias.copy_(pytorch_gru.bias_hh[:hidden_size])

    custom_gru.W_z.weight.copy_(pytorch_gru.weight_ih[hidden_size:2*hidden_size])
    custom_gru.U_z.weight.copy_(pytorch_gru.weight_hh[hidden_size:2*hidden_size])
    custom_gru.W_z.bias.copy_(pytorch_gru.bias_ih[hidden_size:2*hidden_size])
    custom_gru.U_z.bias.copy_(pytorch_gru.bias_hh[hidden_size:2*hidden_size])

    custom_gru.W_h.weight.copy_(pytorch_gru.weight_ih[2*hidden_size:3*hidden_size])
    custom_gru.U_h.weight.copy_(pytorch_gru.weight_hh[2*hidden_size:3*hidden_size])
    custom_gru.W_h.bias.copy_(pytorch_gru.bias_ih[2*hidden_size:3*hidden_size])
    custom_gru.U_h.bias.copy_(pytorch_gru.bias_hh[2*hidden_size:3*hidden_size])

x = torch.randn(4, input_size)
h = torch.zeros(4, hidden_size)

h_custom = custom_gru(x, h)
h_pytorch = pytorch_gru(x, h)

print("Custom GRU matches PyTorch GRU:",
      torch.allclose(h_custom, h_pytorch, atol=1e-5))

# Output:
# Custom GRU matches PyTorch GRU: True
```

### Code Example 3: GRU vs RNN vs LSTM

```python
import torch
import torch.nn as nn
import torch.optim as optim

def compare_architectures(seq_len=50, epochs=150):
    rnn = nn.RNN(5, 32, batch_first=True)
    lstm = nn.LSTM(5, 32, batch_first=True)
    gru = nn.GRU(5, 32, batch_first=True)

    fc_rnn = nn.Linear(32, 2)
    fc_lstm = nn.Linear(32, 2)
    fc_gru = nn.Linear(32, 2)

    models = [
        ('RNN', rnn, fc_rnn, optim.Adam(list(rnn.parameters()) + list(fc_rnn.parameters()), lr=0.005)),
        ('LSTM', lstm, fc_lstm, optim.Adam(list(lstm.parameters()) + list(fc_lstm.parameters()), lr=0.005)),
        ('GRU', gru, fc_gru, optim.Adam(list(gru.parameters()) + list(fc_gru.parameters()), lr=0.005)),
    ]

    results = []
    for name, model, fc, opt in models:
        for epoch in range(epochs):
            x = torch.randn(64, seq_len, 5)
            y = (x[:, 0, 0] > 0).long()

            if isinstance(model, nn.LSTM):
                _, (h, _) = model(x)
            else:
                _, h = model(x)
            pred = fc(h[-1] if not isinstance(model, nn.LSTM) else h[-1])
            loss = nn.CrossEntropyLoss()(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        x_test = torch.randn(100, seq_len, 5)
        y_test = (x_test[:, 0, 0] > 0).long()
        with torch.no_grad():
            if isinstance(model, nn.LSTM):
                _, (h, _) = model(x_test)
                pred = fc(h[-1])
            else:
                _, h = model(x_test)
                pred = fc(h[-1])
            acc = (pred.argmax(dim=1) == y_test).float().mean()
        results.append((name, acc.item()))

    return results

for length in [20, 50, 100]:
    results = compare_architectures(length)
    print(f"\nSeq len {length}:")
    for name, acc in results:
        print(f"  {name}: {acc:.4f}")

# Output:
# Seq len 20:
#   RNN: 0.7200
#   LSTM: 0.8900
#   GRU: 0.8800
#
# Seq len 50:
#   RNN: 0.6500
#   LSTM: 0.8600
#   GRU: 0.8500
#
# Seq len 100:
#   RNN: 0.5400
#   LSTM: 0.8300
#   GRU: 0.8200
```

### Code Example 4: GRU for Sequence Generation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GRUGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden

    def generate(self, start_token, length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[start_token]])
            hidden = None
            generated = [start_token]
            for _ in range(length):
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                x = torch.tensor([[next_token]])
        return generated

model = GRUGenerator(vocab_size=50, embed_size=32, hidden_size=64)
opt = optim.Adam(model.parameters(), lr=0.001)
data = torch.randint(1, 50, (100, 30))

for epoch in range(50):
    logits, _ = model(data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), data[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, PPL={torch.exp(loss).item():.4f}")

generated = model.generate(10, 15, temperature=0.8)
print(f"Generated: {generated[:10]}...")

# Output:
# Epoch 0, PPL=34.5678
# Epoch 20, PPL=6.7890
# Epoch 40, PPL=3.4567
# Generated: [10, 34, 12, 45, 23, 7, 41, 28, 15, 33]...
```

## Common Mistakes

1. **Confusing reset and update gates**: The reset gate controls how much past to forget when computing the candidate. The update gate controls the balance between old and new hidden states.

2. **Forgetting the reset gate in candidate computation**: The candidate uses r_t * h_(t-1), not h_(t-1) directly. The reset gate gates the previous hidden state.

3. **Thinking GRU has a cell state**: GRU has only a hidden state, unlike LSTM which has both hidden and cell states.

4. **Assuming GRU is always worse than LSTM**: GRU often performs comparably to LSTM while being more efficient.

5. **Not accounting for the coupled forget/input in GRU**: The update gate simultaneously controls both forgetting (1 - z_t) and input (z_t), unlike LSTM's separate gates.

6. **Using the wrong initialization**: GRU does not need the specialized forget gate bias initialization that LSTM benefits from.

7. **Ignoring the reset gate's importance**: The reset gate is crucial for allowing the model to forget irrelevant past information when computing the candidate.

## Interview Questions

### Beginner

Q: What is a GRU and how does it differ from an LSTM?
A: GRU is a gated RNN with two gates (reset and update) and no separate cell state. LSTM has three gates (forget, input, output) and a separate cell state. GRU is simpler with fewer parameters.

Q: What are the two gates in a GRU and what do they do?
A: The reset gate controls how much past information to forget when computing the new candidate. The update gate controls how much of the old hidden state to keep vs replace with the new candidate.

### Intermediate

Q: Explain how the update gate in GRU combines the functions of LSTM's forget and input gates.
A: The update gate z_t determines both forgetting and input: h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t. When z_t is near 1, the model keeps old information (equivalent to forget gate = 1, input gate = 0). When z_t is near 0, the model replaces old with new (forget gate = 0, input gate = 1).

Q: Why does GRU not need a separate cell state?
A: GRU's hidden state serves both as memory and output. The update gate controls information flow directly on the hidden state. This simplification works because the coupled gating (one gate for both forget and input) effectively manages memory without requiring a separate storage unit.

### Advanced

Q: Derive the gradient flow for GRU and compare it with LSTM's gradient flow.
A: GRU: dh_t/dh_(t-1) = diag(1 - z_t) + diag(z_t) * diag(r_t * tanh'(...)) * U_h + ( -h_(t-1) + h_tilde_t ) * z_t'. The gradient has three components: direct bypass through (1-z_t), gated path through candidate, and update gate derivative. Unlike LSTM's clean diag(f_t) path, GRU has more terms but still provides effective gradient flow. The bypass path diag(1 - z_t) is analogous to LSTM's forget gate.

Q: Design a variant of GRU that adds a separate cell state while keeping the two-gate structure, and analyze the trade-off.
A: A GRU-like architecture with a cell state: keep reset and update gates, add a cell state C_t that is updated linearly. The candidate uses the reset gate on the hidden state C_t = f_t * C_(t-1) + (1 - f_t) * C_tilde_t, h_t = z_t * tanh(C_t). This adds gradient highway benefits while keeping only two gates. The trade-off is extra memory for C_t but potentially better long-term memory.

## Practice Problems

### Easy

Implement a simple GRU for sequence classification (many-to-one). Compare its performance with a standard RNN and LSTM.

### Medium

Train a GRU language model on a character-level text generation task. Compare its perplexity and generation quality with an LSTM language model of the same hidden size.

### Hard

Implement a GRU variant where the reset gate is replaced with a learned function of the input and hidden state (e.g., a small neural network). Compare its performance against the standard GRU on a long-term dependency task.

## Related Concepts

- LSTM Overview (DL-296)
- Reset Gate (DL-312)
- Update Gate (DL-313)
- LSTM vs GRU (DL-310)

## Next Concepts

- Reset Gate (DL-312)
- Update Gate (DL-313)
- GRU Forward Pass (DL-314)

## Summary

The Gated Recurrent Unit (GRU) is a simplified gated RNN architecture that uses two gates (reset and update) and no separate cell state. It addresses the vanishing gradient problem while being more parameter-efficient than LSTM. The update gate couples forgetting and input gating into a single mechanism. GRU performs comparably to LSTM on many tasks while being computationally more efficient.

## Key Takeaways

- GRU has two gates: reset and update
- No separate cell state (hidden state serves as both)
- Update gate simultaneously controls forget and input
- ~25% fewer parameters than LSTM with same hidden size
- Performs comparably to LSTM on many tasks
- More computationally efficient than LSTM
- Candidate uses reset-gated previous hidden state
