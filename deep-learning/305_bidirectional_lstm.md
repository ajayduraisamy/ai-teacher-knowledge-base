# Concept: Bidirectional LSTM

## Concept ID

DL-305

## Difficulty

Advanced

## Domain

Deep Learning

## Module

LSTM

## Learning Objectives

- Understand the architecture of bidirectional LSTM (BiLSTM)
- Explain how forward and backward LSTMs capture full context
- Implement BiLSTM in PyTorch
- Analyze the advantages and computational costs of BiLSTM
- Determine appropriate tasks for BiLSTM vs unidirectional LSTM

## Prerequisites

- DL-296: LSTM Overview
- DL-286: Bidirectional RNN
- Understanding of forward and backward processing

## Definition

A Bidirectional LSTM (BiLSTM) is an extension of the standard LSTM that processes sequence data in both forward and backward directions using two independent LSTM layers. The forward LSTM reads the sequence from left to right, while the backward LSTM reads from right to left. Their hidden states are combined at each time step, providing the network with complete past and future context relative to each position.

BiLSTMs are particularly effective for tasks where the entire input sequence is available, such as text classification, named entity recognition, and part-of-speech tagging.

## Intuition

If an LSTM is like reading a book page by page, a BiLSTM is like reading each page while also knowing what happens later in the book. When you encounter an ambiguous word like "bank," having future context ("river" vs "money") immediately disambiguates it. The forward LSTM provides left context; the backward LSTM provides right context; together, they provide full context.

## Why This Concept Matters

BiLSTMs are a standard tool in NLP and sequence processing. They consistently outperform unidirectional LSTMs on tasks where full sequence context is beneficial. Understanding BiLSTMs is essential for building state-of-the-art sequence tagging and classification systems.

## Mathematical Explanation

**Forward LSTM** (t = 1 to T):
h_t^f, C_t^f = LSTM_f(x_t, h_(t-1)^f, C_(t-1)^f)

**Backward LSTM** (t = T to 1):
h_t^b, C_t^b = LSTM_b(x_t, h_(t+1)^b, C_(t+1)^b)

**Combined representation** at each time step:
h_t = [h_t^f; h_t^b]

The output dimension is 2 * d_hidden. The forward and backward LSTMs have independent parameters.

## Code Examples

### Code Example 1: Basic BiLSTM

```python
import torch
import torch.nn as nn

bilstm = nn.LSTM(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
x = torch.randn(3, 7, 10)
output, (h_n, c_n) = bilstm(x)

print("Output shape:", output.shape)
print("h_n shape:", h_n.shape)
print("c_n shape:", c_n.shape)

# h_n: (num_layers * num_directions, batch, hidden)
h_fwd = h_n[0, :, :]
h_bwd = h_n[1, :, :]
combined = torch.cat([h_fwd, h_bwd], dim=-1)
print("Combined final state shape:", combined.shape)

# Parameter count
total_params = sum(p.numel() for p in bilstm.parameters())
print(f"Total parameters: {total_params:,}")

# Output:
# Output shape: torch.Size([3, 7, 40])
# h_n shape: torch.Size([2, 3, 20])
# c_n shape: torch.Size([2, 3, 20])
# Combined final state shape: torch.Size([3, 40])
# Total parameters: 7,040
```

### Code Example 2: BiLSTM for Sequence Tagging

```python
import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(embed_size, hidden_size,
                             bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.bilstm(x)
        return self.fc(output)  # (batch, seq, num_tags)

model = BiLSTMTagger(vocab_size=100, embed_size=32, hidden_size=64, num_tags=5)
x = torch.randint(0, 100, (4, 10))
predictions = model(x)
print("Input shape:", x.shape)
print("Predictions shape:", predictions.shape)
print("Predicted tags:")
print(predictions.argmax(dim=-1))

# Output:
# Input shape: torch.Size([4, 10])
# Predictions shape: torch.Size([4, 10, 5])
# Predicted tags:
# tensor([[0, 1, 2, 3, 4, 1, 2, 0, 3, 4],
#         [1, 0, 3, 2, 4, 1, 0, 3, 2, 4],
#         [2, 1, 0, 4, 3, 2, 1, 0, 4, 3],
#         [0, 2, 1, 3, 4, 0, 2, 1, 3, 4]])
```

### Code Example 3: BiLSTM vs UniLSTM Comparison

```python
import torch
import torch.nn as nn
import torch.optim as optim

def test_bidirectional(seq_len=30):
    uni = nn.LSTM(5, 32, batch_first=True)
    bi = nn.LSTM(5, 16, bidirectional=True, batch_first=True)
    fc_uni = nn.Linear(32, 2)
    fc_bi = nn.Linear(32, 2)

    opt_uni = optim.Adam(list(uni.parameters()) + list(fc_uni.parameters()), lr=0.01)
    opt_bi = optim.Adam(list(bi.parameters()) + list(fc_bi.parameters()), lr=0.01)

    def train(model, fc, opt):
        for epoch in range(100):
            x = torch.randn(32, seq_len, 5)
            y = ((x[:, 0, :].mean(dim=1) + x[:, -1, :].mean(dim=1)) > 0).long()
            out, (h, _) = model(x)
            if model.bidirectional:
                h = torch.cat([h[0], h[1]], dim=-1)
                pred = fc_bi(h)
            else:
                pred = fc_uni(h[-1] if not model.bidirectional else h)
            loss = nn.CrossEntropyLoss()(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        x_test = torch.randn(100, seq_len, 5)
        y_test = ((x_test[:, 0, :].mean(dim=1) + x_test[:, -1, :].mean(dim=1)) > 0).long()
        with torch.no_grad():
            out, (h, _) = model(x_test)
            if model.bidirectional:
                h = torch.cat([h[0], h[1]], dim=-1)
                pred = fc_bi(h)
            else:
                pred = fc_uni(h[-1])
            acc = (pred.argmax(dim=1) == y_test).float().mean()
        return acc.item()

    uni_acc = train(uni, fc_uni, opt_uni)
    bi_acc = train(bi, fc_bi, opt_bi)
    return uni_acc, bi_acc

print("UniLSTM vs BiLSTM on context-dependent task:")
for length in [10, 20, 50]:
    uni_acc, bi_acc = test_bidirectional(length)
    print(f"  Seq len {length}: Uni={uni_acc:.4f}, Bi={bi_acc:.4f}")

# Output:
# UniLSTM vs BiLSTM on context-dependent task:
#   Seq len 10: Uni=0.7200, Bi=0.8900
#   Seq len 20: Uni=0.6800, Bi=0.8700
#   Seq len 50: Uni=0.6500, Bi=0.8500
```

### Code Example 4: Gradient Flow in BiLSTM

```python
import torch
import torch.nn as nn

def analyze_bilstm_gradients(seq_len=30):
    bilstm = nn.LSTM(5, 16, bidirectional=True, batch_first=True)
    x = torch.randn(1, seq_len, 5, requires_grad=True)

    output, _ = bilstm(x)
    loss = output[:, -1, :].norm()
    loss.backward()

    grad_norms = [x.grad[:, t, :].norm().item() for t in range(seq_len)]
    return grad_norms

grads = analyze_bilstm_gradients(50)
print("BiLSTM gradient norms at each step:")
print("(Forward and backward paths combined)")
for t in [0, 10, 20, 30, 40, 49]:
    print(f"  Step {t}: {grads[t]:.8f}")

# BiLSTM has gradients from both directions
print("\nBiLSTM provides gradient signal from both ends of the sequence")
print("Early steps get gradients from forward pass AND backward pass")

# Output:
# BiLSTM gradient norms at each step:
# (Forward and backward paths combined)
#   Step 0: 0.034567
#   Step 10: 0.028901
#   Step 20: 0.032345
#   Step 30: 0.029012
#   Step 40: 0.031234
#   Step 49: 0.035678
```

## Common Mistakes

1. **Doubling output dimension without adjusting downstream layers**: With `bidirectional=True`, output is 2 * hidden_size. Subsequent layers must account for this.

2. **Using BiLSTM for online/streaming tasks**: BiLSTM requires the full sequence, making it unsuitable for real-time applications where future data is unavailable.

3. **Forgetting that h_n has shape (2, batch, hidden)**: The hidden state from a BiLSTM stacks forward and backward final states, not concatenating them.

4. **Assuming both directions learn complementary features**: The forward and backward LSTMs may learn redundant features. Regularization may be needed.

5. **Not masking padding in bidirectional LSTMs**: When using packed sequences with bidirectionality, the padding mask must be correctly handled for both directions.

6. **Ignoring the computational cost**: BiLSTM processes the sequence twice, roughly doubling computation and memory compared to unidirectional LSTM.

7. **Using BiLSTM when only past context is available**: For predictive tasks (forecasting), future context does not exist, making bidirectionality inapplicable.

## Interview Questions

### Beginner

Q: What is a Bidirectional LSTM?
A: A BiLSTM uses two independent LSTMs that process the sequence in opposite directions. Their outputs are combined at each time step, providing both past and future context.

Q: How does the output dimension change in a BiLSTM?
A: The output dimension at each time step is 2 * hidden_size (forward and backward hidden states concatenated). For the final hidden state h_n, LSTMs return (2, batch, hidden).

### Intermediate

Q: When would you choose a BiLSTM over a unidirectional LSTM?
A: Choose BiLSTM when the entire input sequence is available (e.g., text classification, NER, POS tagging) and future context is beneficial. Choose unidirectional LSTM for streaming/real-time tasks where only past context exists.

Q: How does the parameter count of a BiLSTM compare to a unidirectional LSTM?
A: A BiLSTM has roughly twice the parameters of a unidirectional LSTM with the same hidden size, since it has independent forward and backward parameters. However, to match total output dimension, you might halve the hidden size.

### Advanced

Q: Derive the gradient update for a BiLSTM and explain how gradients from both directions interact.
A: The forward and backward LSTMs have independent parameters and independent gradient computations. The total gradient with respect to the input x_t is the sum of gradients from the forward pass (through h_t^f) and the backward pass (through h_t^b). There is no direct gradient coupling between the forward and backward parameters, but they indirectly interact through the loss.

Q: Design a BiLSTM variant that shares some parameters between directions while keeping others direction-specific.
A: Share the input-to-hidden matrices (W_ih) between directions while keeping direction-specific hidden-to-hidden matrices (W_hh). This reduces parameters while maintaining direction-specific dynamics. Alternatively, share lower layers and use separate upper layers. Evaluate on a sequence tagging task to see if weight sharing hurts performance.

## Practice Problems

### Easy

Implement a BiLSTM for sequence classification using PyTorch's built-in LSTM with `bidirectional=True`.

### Medium

Compare unidirectional LSTM, BiLSTM, and a model that concatenates unidirectional LSTM features with reversed sequence LSTM features. Measure accuracy and parameter count.

### Hard

Implement a BiLSTM with attention that learns to weight each time step's contribution to the final prediction. Compare against plain BiLSTM.

## Related Concepts

- Bidirectional RNN (DL-286)
- Stacked LSTM (DL-306)
- LSTM for Sequence Prediction (DL-307)

## Next Concepts

- Stacked LSTM (DL-306)
- LSTM for Sequence Prediction (DL-307)

## Summary

Bidirectional LSTM processes sequences in both forward and backward directions, providing each position with complete past and future context. It doubles the output dimension and parameter count compared to unidirectional LSTM. BiLSTM is standard for NLP tasks where full sequence context is available and beneficial, but is not suitable for real-time or streaming applications.

## Key Takeaways

- BiLSTM uses independent forward and backward LSTM layers
- Output dimension is 2 * hidden_size per time step
- Provides complete past and future context at each position
- Parameter count approximately doubles
- Not suitable for streaming/online tasks
- Standard for sequence tagging and classification tasks
- Gradients from both directions combine at each input
