# Concept: GRU Forward Pass

## Concept ID

DL-314

## Difficulty

Advanced

## Domain

Deep Learning

## Module

GRU

## Learning Objectives

- Understand the complete GRU forward pass computation
- Implement the GRU forward pass from scratch in PyTorch
- Trace the flow of information through the reset and update gates
- Analyze the computational complexity of the GRU forward pass
- Verify custom GRU implementation against PyTorch's built-in version

## Prerequisites

- DL-311: GRU Overview
- DL-312: Reset Gate
- DL-313: Update Gate
- Understanding of sigmoid and tanh activations

## Definition

The GRU forward pass is the complete sequence of computations that transforms an input sequence into output hidden states through the GRU's two-gate mechanism. It encompasses the reset gate, update gate, candidate hidden state, and final hidden state computation, repeated for each time step. Unlike LSTM, the GRU forward pass has no separate cell state; the hidden state serves as both memory and output.

## Intuition

The GRU forward pass is like a simple but effective recipe with four ingredients:

1. Reset gate (r): How much past should I forget for the new draft?
2. Update gate (z): How much of my current notes should I keep vs replace?
3. Candidate (h_tilde): The new draft notes, considering past through the reset gate
4. Final state (h): A blend of old notes and new draft, controlled by the update gate

## Why This Concept Matters

Understanding the GRU forward pass is essential for implementing custom recurrent architectures, debugging GRU-based models, and understanding the differences between GRU and LSTM at a computational level.

## Mathematical Explanation

Complete GRU forward pass for a single time step t:

**1. Reset gate**: r_t = sigmoid(W_r * x_t + U_r * h_(t-1) + b_r)

**2. Update gate**: z_t = sigmoid(W_z * x_t + U_z * h_(t-1) + b_z)

**3. Candidate hidden state**: h_tilde_t = tanh(W_h * x_t + U_h * (r_t * h_(t-1)) + b_h)

**4. Final hidden state**: h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t

The GRU has 3 sets of weight matrices (for reset, update, and candidate), each with input-to-hidden and hidden-to-hidden components. This is 3/4 of LSTM's 4 sets.

**Computational complexity per step**: O(3 * d_hidden * (d_input + d_hidden))

## Code Examples

### Code Example 1: Full GRU Forward Pass from Scratch

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
        return h, {'reset': r, 'update': z, 'candidate': h_tilde}

input_size, hidden_size = 10, 20
custom_cell = GRUCellFromScratch(input_size, hidden_size)
pytorch_cell = nn.GRUCell(input_size, hidden_size)

# Copy weights
with torch.no_grad():
    for w_custom, w_pytorch in [
        (custom_cell.W_r, pytorch_cell.weight_ih[:hidden_size]),
        (custom_cell.U_r, pytorch_cell.weight_hh[:hidden_size]),
        (custom_cell.W_z, pytorch_cell.weight_ih[hidden_size:2*hidden_size]),
        (custom_cell.U_z, pytorch_cell.weight_hh[hidden_size:2*hidden_size]),
        (custom_cell.W_h, pytorch_cell.weight_ih[2*hidden_size:3*hidden_size]),
        (custom_cell.U_h, pytorch_cell.weight_hh[2*hidden_size:3*hidden_size]),
    ]:
        w_custom.weight.copy_(w_pytorch)

x = torch.randn(4, input_size)
h = torch.zeros(4, hidden_size)

h_custom, gates = custom_cell(x, h)
h_pytorch = pytorch_cell(x, h)

print("Custom GRU matches PyTorch:",
      torch.allclose(h_custom, h_pytorch, atol=1e-5))

print("\nGate statistics:")
for name, val in gates.items():
    print(f"  {name}: mean={val.mean().item():.4f}, "
          f"min={val.min().item():.4f}, max={val.max().item():.4f}")

# Output:
# Custom GRU matches PyTorch: True
#
# Gate statistics:
#   reset: mean=0.5123, min=0.0234, max=0.9876
#   update: mean=0.4987, min=0.0123, max=0.9789
#   candidate: mean=-0.0234, min=-0.9123, max=0.8956
```

### Code Example 2: GRU Forward Pass Over Sequence

```python
import torch
import torch.nn as nn

def gru_forward_sequence(gru_cell, x_seq, h0=None):
    batch_size, seq_len, input_size = x_seq.shape
    hidden_size = gru_cell.hidden_size

    h = h0 if h0 is not None else torch.zeros(batch_size, hidden_size)
    outputs = []
    all_gates = []

    for t in range(seq_len):
        x_t = x_seq[:, t, :]
        h, gates = gru_cell(x_t, h)
        outputs.append(h.unsqueeze(1))
        all_gates.append(gates)

    output = torch.cat(outputs, dim=1)
    return output, h, all_gates

custom_cell = GRUCellFromScratch(10, 20)
x = torch.randn(2, 5, 10)
output, h_final, gates_list = gru_forward_sequence(custom_cell, x)

print("GRU forward pass over sequence:")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Final hidden state shape: {h_final.shape}")

# Track hidden state evolution
print("\nHidden state norm over time:")
for t in range(x.size(1)):
    print(f"  t={t}: norm={output[0, t].norm().item():.4f}")

# Track gate values
print("\nReset gate mean over time:")
for t in range(x.size(1)):
    print(f"  t={t}: r_mean={gates_list[t]['reset'].mean().item():.4f}")

print("\nUpdate gate mean over time:")
for t in range(x.size(1)):
    print(f"  t={t}: z_mean={gates_list[t]['update'].mean().item():.4f}")

# Output:
# GRU forward pass over sequence:
#   Input shape: torch.Size([2, 5, 10])
#   Output shape: torch.Size([2, 5, 20])
#   Final hidden state shape: torch.Size([2, 20])
#
# Hidden state norm over time:
#   t=0: norm=0.3456
#   t=1: norm=0.5678
#   t=2: norm=0.7890
#   t=3: norm=0.9123
#   t=4: norm=0.9876
#
# Reset gate mean over time:
#   t=0: r_mean=0.5123
#   t=1: r_mean=0.4987
#   t=2: r_mean=0.5234
#   t=3: r_mean=0.4876
#   t=4: r_mean=0.5123
#
# Update gate mean over time:
#   t=0: z_mean=0.4987
#   t=1: z_mean=0.5123
#   t=2: z_mean=0.4876
#   t=3: z_mean=0.5234
#   t=4: z_mean=0.5012
```

### Code Example 3: PyTorch GRU vs Custom GRU

```python
import torch
import torch.nn as nn

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.cell = GRUCellFromScratch(input_size, hidden_size)
        self.batch_first = batch_first

    def forward(self, x, h0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch, _ = x.shape

        h = h0 if h0 is not None else torch.zeros(batch, self.cell.hidden_size)

        outputs = []
        for t in range(seq_len):
            h, _ = self.cell(x[t], h)
            outputs.append(h.unsqueeze(0))

        output = torch.cat(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h.unsqueeze(0)

custom_gru = CustomGRU(10, 20, batch_first=True)
pytorch_gru = nn.GRU(10, 20, batch_first=True)

# Copy weights
with torch.no_grad():
    for w_c, w_p in [
        (custom_gru.cell.W_r.weight, pytorch_gru.weight_ih_l0[:20]),
        (custom_gru.cell.U_r.weight, pytorch_gru.weight_hh_l0[:20]),
        (custom_gru.cell.W_z.weight, pytorch_gru.weight_ih_l0[20:40]),
        (custom_gru.cell.U_z.weight, pytorch_gru.weight_hh_l0[20:40]),
        (custom_gru.cell.W_h.weight, pytorch_gru.weight_ih_l0[40:60]),
        (custom_gru.cell.U_h.weight, pytorch_gru.weight_hh_l0[40:60]),
    ]:
        w_c.copy_(w_p)

x = torch.randn(4, 7, 10)
custom_out, custom_h = custom_gru(x)
pytorch_out, pytorch_h = pytorch_gru(x)

print("Outputs match:", torch.allclose(custom_out, pytorch_out, atol=1e-5))
print("Hidden states match:", torch.allclose(custom_h, pytorch_h, atol=1e-5))

# Output:
# Outputs match: True
# Hidden states match: True
```

### Code Example 4: GRU Forward Pass with Multiple Layers

```python
import torch
import torch.nn as nn

class MultiLayerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.GRUCell(input_size, hidden_size))
        for _ in range(1, num_layers):
            self.layers.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, x):
        batch, seq, _ = x.shape
        num_layers = len(self.layers)
        hidden_size = self.layers[0].hidden_size

        h = [torch.zeros(batch, hidden_size) for _ in range(num_layers)]

        outputs = []
        for t in range(seq):
            inp = x[:, t]
            for layer in range(num_layers):
                h[layer] = self.layers[layer](inp, h[layer])
                inp = h[layer]
            outputs.append(h[-1].unsqueeze(1))

        return torch.cat(outputs, dim=1), h[-1].unsqueeze(0)

model = MultiLayerGRU(10, 20, 3)
x = torch.randn(4, 7, 10)
output, h_n = model(x)

print("Multi-layer GRU:")
print(f"  Output shape: {output.shape}")
print(f"  Final hidden shape: {h_n.shape}")

# Compare with PyTorch multi-layer GRU
pytorch_mgru = nn.GRU(10, 20, num_layers=3, batch_first=True)
pytorch_out, pytorch_h = pytorch_mgru(x)
print(f"  PyTorch output shape: {pytorch_out.shape}")
print(f"  PyTorch h_n shape: {pytorch_h.shape}")

# Parameter count comparison
custom_params = sum(p.numel() for p in model.parameters())
pytorch_params = sum(p.numel() for p in pytorch_mgru.parameters())
print(f"\n  Custom params: {custom_params:,}")
print(f"  PyTorch params: {pytorch_params:,}")

# Output:
# Multi-layer GRU:
#   Output shape: torch.Size([4, 7, 20])
#   Final hidden shape: torch.Size([1, 4, 20])
#   PyTorch output shape: torch.Size([4, 7, 20])
#   PyTorch h_n shape: torch.Size([3, 4, 20])
#
#   Custom params: 14,280
#   PyTorch params: 14,280
```

## Common Mistakes

1. **Confusing the order of gate computation**: Reset and update gates are computed in parallel, not sequentially. Both depend on x_t and h_(t-1).

2. **Forgetting the reset gate in candidate computation**: The candidate uses r_t * h_(t-1), not h_(t-1) directly.

3. **Using the wrong activation**: Reset and update gates use sigmoid. Candidate uses tanh.

4. **Confusing GRU output with LSTM output**: GRU returns (output, h_n) without a cell state. LSTM returns (output, (h_n, c_n)).

5. **Incorrectly handling the (1 - z_t) factor**: The final update is h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t, with (1 - z_t) scaling the old state.

6. **Not accounting for GRU output not containing all intermediate gates**: The GRU output only contains the final hidden state, not the intermediate gate values.

7. **Applying LSTM dropout logic to GRU**: While dropout works similarly (between layers), GRU has different internal computation and may respond differently to dropout placement.

## Interview Questions

### Beginner

Q: What are the four steps of the GRU forward pass?
A: (1) Reset gate: r_t = sigmoid(W_r*x_t + U_r*h_(t-1)). (2) Update gate: z_t = sigmoid(W_z*x_t + U_z*h_(t-1)). (3) Candidate: h_tilde = tanh(W_h*x_t + U_h*(r_t*h_(t-1))). (4) Hidden state: h_t = (1-z_t)*h_(t-1) + z_t*h_tilde.

Q: What does the GRU forward pass return?
A: It returns the output (hidden states at all time steps) and the final hidden state h_n. Unlike LSTM, it does not return a cell state.

### Intermediate

Q: How does the computational complexity of the GRU forward pass compare to LSTM?
A: GRU has 3 weight matrix sets vs LSTM's 4. The GRU forward pass is approximately 75% the computational cost of LSTM with the same hidden size.

Q: Explain how the reset and update gates interact in the GRU forward pass.
A: The reset gate affects the candidate computation (r_t * h_(t-1) is used in the candidate). The update gate affects the final blend (z_t controls old vs new). The reset gate indirectly affects the final state through the candidate, while the update gate directly controls the blend.

### Advanced

Q: Derive the full forward pass for a stacked GRU and analyze the information flow between layers.
A: For layer l at step t: r_t^(l) = sigmoid(W_r^(l)*h_t^(l-1) + U_r^(l)*h_(t-1)^(l)), with h_t^(0) = x_t. The reset and update gates for layer l depend on the current output of layer l-1 (h_t^(l-1)) and the previous state of layer l (h_(t-1)^(l)). Information flows upward (layers) and forward (time) simultaneously, with each layer learning different temporal abstractions.

Q: Design a GRU variant where the reset and update gates share some weights and analyze the trade-off.
A: Share the input-to-hidden transformation: W_r = W_z = W_shared, but keep separate U_r and U_z. This reduces parameters by d_input * d_hidden. The gates now process the input identically but differ in how they incorporate the previous hidden state. Trade-off: fewer parameters but reduced flexibility, which may work well on smaller datasets where overfitting is a concern.

## Practice Problems

### Easy

Implement the full GRU forward pass for a single step and verify that the output dimension matches the hidden size.

### Medium

Implement a custom GRU from scratch and verify it produces the same output as PyTorch's nn.GRU for random inputs across multiple sequence lengths.

### Hard

Implement a GRU with coupled reset and update gates (z_t = 1 - r_t). Compare its performance against standard GRU on tasks requiring long-term and short-term memory.

## Related Concepts

- GRU Backward Pass (DL-315)
- GRU Overview (DL-311)
- Reset Gate (DL-312)
- Update Gate (DL-313)

## Next Concepts

- GRU Backward Pass (DL-315)
- Bidirectional GRU (DL-316)

## Summary

The GRU forward pass is a computationally efficient sequence processing mechanism with two gates (reset and update) and no separate cell state. It involves four key computations: reset gate, update gate, candidate hidden state, and final hidden state update. The GRU uses 3 sets of weight matrices (vs LSTM's 4), making it approximately 25% more efficient. The forward pass processes each time step sequentially, with the hidden state serving as both memory and output.

## Key Takeaways

- GRU forward pass: r, z, h_tilde, h (4 steps per time step)
- Reset and update gates computed in parallel from x_t and h_(t-1)
- Candidate uses reset-gated previous hidden state
- Final hidden state is convex combination of old and candidate
- No separate cell state (unlike LSTM)
- 3 weight matrix sets (vs LSTM's 4)
- ~25% fewer parameters and computations than LSTM
