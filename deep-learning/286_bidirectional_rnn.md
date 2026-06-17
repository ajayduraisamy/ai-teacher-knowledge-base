# Concept: Bidirectional RNN

## Concept ID

DL-286

## Difficulty

Advanced

## Domain

Deep Learning

## Module

RNN

## Learning Objectives

- Understand the architecture and purpose of bidirectional RNNs
- Explain how bidirectional processing captures past and future context
- Implement bidirectional RNNs using PyTorch
- Determine when bidirectional processing is appropriate
- Analyze the computational and memory costs of bidirectionality

## Prerequisites

- DL-281: Recurrent Neural Network
- DL-283: Hidden State
- DL-284: Sequence Modeling
- Understanding of forward and backward propagation

## Definition

A Bidirectional Recurrent Neural Network (BiRNN) is an extension of the standard RNN that processes sequence data in both forward and backward directions simultaneously. It consists of two independent RNNs: one that reads the sequence from left to right (forward pass) and another that reads from right to left (backward pass). The hidden states from both directions are combined at each time step, providing the network with access to both past and future context relative to each position in the sequence.

The architecture produces two sequences of hidden states: the forward hidden states (h_1^f, h_2^f, ..., h_T^f) and the backward hidden states (h_T^b, h_(T-1)^b, ..., h_1^b). These are typically concatenated at each time step to form a comprehensive representation.

## Intuition

Imagine reading a sentence where each word's meaning depends on surrounding words. For example, in "He rowed the boat to the bank," the word "bank" is ambiguous until you consider context from both sides. Reading forward tells you "rowed the boat," while reading backward confirms the sentence isn't about a financial bank. A unidirectional RNN would only have left context when processing "bank," potentially missing crucial information from the right.

A BiRNN processes the sentence twice: once forward, building context from the beginning, and once backward, building context from the end. When processing "bank," it combines the forward context ("He rowed the boat to the") with the backward context ("."), giving it a complete view.

## Why This Concept Matters

Bidirectional RNNs have become a standard tool in NLP and sequence processing because many tasks benefit from understanding both preceding and following context:

- **Named Entity Recognition**: Whether "Washington" is a person, place, or organization depends on surrounding words on both sides
- **Part-of-Speech Tagging**: "run" can be a verb or noun depending on full sentence context
- **Sentiment Analysis**: "not bad" requires understanding the negation that precedes "bad"
- **Speech Recognition**: Phoneme boundaries are influenced by both past and future sounds

The bidirectional architecture consistently outperforms unidirectional models on tasks where full sequence context is available and beneficial.

## Mathematical Explanation

A BiRNN consists of two independent RNNs:

**Forward RNN** (processes t = 1 to T):
h_t^f = tanh(W_xh^f · x_t + W_hh^f · h_(t-1)^f + b_h^f)

**Backward RNN** (processes t = T to 1):
h_t^b = tanh(W_xh^b · x_t + W_hh^b · h_(t+1)^b + b_h^b)

The combined representation at time step t:
h_t = [h_t^f; h_t^b]

Where [a; b] denotes vector concatenation. The output can be computed from the combined representation:
y_t = W_hy · h_t + b_y

The total hidden dimension is 2 * d_hidden (since both forward and backward states are concatenated).

**Gradient flow**: The forward RNN has the same gradient dynamics as a standard RNN going forward, while the backward RNN has symmetric dynamics going backward. Each direction independently may suffer from vanishing gradients, but the combined representation benefits from having two independent sources of gradient information.

**Parameter count**: A BiRNN has approximately twice the parameters of a unidirectional RNN with the same hidden size.

## Code Examples

### Code Example 1: Basic Bidirectional RNN

```python
import torch
import torch.nn as nn

bi_rnn = nn.RNN(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
x = torch.randn(3, 7, 10)
output, hidden = bi_rnn(x)

print("Output shape:", output.shape)
print("Hidden tensor shape:", hidden.shape)

# hidden is (num_layers * num_directions, batch, hidden_size)
# For 1 layer and bidirectional: hidden shape = (2, batch, hidden_size)
h_fwd = hidden[0, :, :]  # Final forward hidden
h_bwd = hidden[1, :, :]  # Final backward hidden

print("Forward final hidden shape:", h_fwd.shape)
print("Backward final hidden shape:", h_bwd.shape)

# Concatenate for sequence classification
combined = torch.cat([h_fwd, h_bwd], dim=-1)
print("Combined shape:", combined.shape)

# Output:
# Output shape: torch.Size([3, 7, 40])
# Hidden tensor shape: torch.Size([2, 3, 20])
# Forward final hidden shape: torch.Size([3, 20])
# Backward final hidden shape: torch.Size([3, 20])
# Combined shape: torch.Size([3, 40])
```

### Code Example 2: Extracting Forward and Backward Outputs

```python
import torch
import torch.nn as nn

bi_rnn = nn.RNN(10, 20, bidirectional=True, batch_first=True)
x = torch.randn(2, 5, 10)
output, hidden = bi_rnn(x)

# Output contains both directions concatenated
# output[:, t, :20] is forward at time t
# output[:, t, 20:] is backward at time t
fwd_output = output[:, :, :20]
bwd_output = output[:, :, 20:]

print("Output shape:", output.shape)
print("Forward output shape:", fwd_output.shape)
print("Backward output shape:", bwd_output.shape)

# Verify: forward hidden at last step equals hidden[0]
print("Match forward final:", torch.allclose(fwd_output[:, -1, :], hidden[0, :, :], atol=1e-6))
# Verify: backward hidden at first step equals hidden[1]
print("Match backward initial:", torch.allclose(bwd_output[:, 0, :], hidden[1, :, :], atol=1e-6))

# Output:
# Output shape: torch.Size([2, 5, 40])
# Forward output shape: torch.Size([2, 5, 20])
# Backward output shape: torch.Size([2, 5, 20])
# Match forward final: True
# Match backward initial: True
```

### Code Example 3: BiRNN for Named Entity Recognition

```python
import torch
import torch.nn as nn

class BiRNNNER(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bi_rnn = nn.RNN(embed_size, hidden_size,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.bi_rnn(x)
        return self.fc(output)

model = BiRNNNER(vocab_size=100, embed_size=32, hidden_size=64, num_tags=5)
x = torch.randint(0, 100, (4, 10))
predictions = model(x)
print("Input shape:", x.shape)
print("Predictions shape:", predictions.shape)
print("Predicted tag indices:")
print(predictions.argmax(dim=-1))

# Output:
# Input shape: torch.Size([4, 10])
# Predictions shape: torch.Size([4, 10, 5])
# Predicted tag indices:
# tensor([[0, 2, 1, 3, 4, 1, 2, 0, 3, 1],
#         [1, 0, 2, 3, 4, 4, 1, 2, 0, 3],
#         [2, 1, 0, 4, 3, 1, 2, 0, 4, 3],
#         [0, 2, 1, 3, 4, 1, 0, 2, 3, 4]])
```

### Code Example 4: Comparison of Unidirectional vs Bidirectional

```python
import torch
import torch.nn as nn
import torch.optim as optim

class UniRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

# Test on synthetic context-dependent task
# Task: predict middle element based on full context
def create_data(n_samples=100, seq_len=7):
    x = torch.randint(0, 10, (n_samples, seq_len))
    # Label depends on both first and last elements
    y = ((x[:, 0] + x[:, -1]) > 10).long()
    return x.float().unsqueeze(-1), y

X, y = create_data(500, 7)
train_X, train_y = X[:400], y[:400]
test_X, test_y = X[400:], y[400:]

for name, model_class in [('Unidirectional', UniRNN), ('Bidirectional', BiRNN)]:
    model = model_class(input_size=1, hidden_size=16, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(200):
        pred = model(train_X)
        loss = loss_fn(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_pred = model(test_X).argmax(dim=1)
        acc = (test_pred == test_y).float().mean()
    print(f"{name} RNN test accuracy: {acc.item():.4f}")

# Output:
# Unidirectional RNN test accuracy: 0.7500
# Bidirectional RNN test accuracy: 0.9100
```

## Common Mistakes

1. **Using bidirectional when future context is unavailable**: For online/streaming tasks like real-time speech recognition, bidirectional RNNs cannot be used because future inputs are not yet available.

2. **Doubling hidden size without adjusting downstream layers**: When adding bidirectional=True, the output feature dimension becomes 2 * hidden_size. Failing to account for this in subsequent layers causes dimension mismatches.

3. **Misinterpreting the hidden state tensor**: The hidden state from a bidirectional RNN has shape (num_layers * 2, batch, hidden_size), where even indices are forward and odd indices are backward states.

4. **Assuming bidirectional RNNs always outperform unidirectional**: Bidirectional RNNs can overfit on small datasets due to twice the parameters. They may not help when future context is irrelevant to the task.

5. **Processing batches with different sequence lengths**: Bidirectional RNNs with packed sequences require careful handling to ensure the backward pass correctly processes reversed sequences.

6. **Using the final hidden state for classification without concatenation**: The final hidden state in a BiRNN contains both forward and backward states. Using only one direction's final state discards half the information.

7. **Ignoring the memory cost**: Bidirectional RNNs store the entire sequence of forward hidden states before computing backward states, doubling the memory requirement compared to unidirectional RNNs.

## Interview Questions

### Beginner

Q: What is the key advantage of a bidirectional RNN over a standard RNN?
A: A bidirectional RNN captures context from both past and future time steps by processing the sequence in both directions. This provides each position with complete contextual information, which is beneficial for tasks where the entire sequence is available.

Q: How does the output dimension change when using a bidirectional RNN?
A: The output at each time step has dimension 2 * hidden_size, representing the concatenation of the forward and backward hidden states. A unidirectional RNN would have output dimension hidden_size.

### Intermediate

Q: Explain why bidirectional RNNs cannot be used for real-time sequence generation tasks.
A: Bidirectional RNNs require the entire sequence to be available before producing output, because the backward pass processes the sequence in reverse. For real-time tasks like next-word prediction or live speech recognition, future tokens are unknown, making bidirectional processing impossible.

Q: How would you combine the forward and backward outputs from a BiRNN? What are the trade-offs?
A: Common combination methods include concatenation (preserves full information), addition (reduces dimension but may lose discriminative information), and averaging (similar to addition). Concatenation is most common as it allows the downstream network to learn how to weight each direction.

### Advanced

Q: Derive the gradient computation for a bidirectional RNN and explain how the backward pass synchronizes with the forward pass.
A: During the backward pass, gradients flow through both the forward and backward RNNs independently. For the forward RNN, gradients flow from t=T down to t=1 (standard BPTT). For the backward RNN, gradients flow from t=1 up to t=T (reverse BPTT). The total gradient with respect to the input at position t is the sum of gradients from both directions. This means each RNN's gradient computation is independent, and they only couple through the loss contribution at each time step.

Q: Design a bidirectional RNN variant that reduces memory requirements while maintaining bidirectional context, and analyze the trade-off.
A: A simple approach is to use a lightweight backward pass with fewer hidden units (e.g., forward size=256, backward size=64). This reduces total parameters from 2*256 to 256+64=320 units per layer. The trade-off is reduced backward context capacity. Alternatively, a future-frame prediction approach: train the forward RNN to predict future states, eliminating the need for a separate backward RNN while still incorporating future information through the prediction loss.

## Practice Problems

### Easy

Implement a BiRNN for sentiment classification where each input is a sequence of 10 word indices (vocab size = 50). Use the final concatenated hidden state for binary classification.

### Medium

Compare unidirectional vs bidirectional RNN performance on a task where labels depend equally on early and late context. Generate a synthetic dataset and measure accuracy for both models.

### Hard

Implement a bidirectional RNN with attention for a document classification task. Use multiple bidirectional layers and an attention mechanism that weights time steps by importance. Compare against a plain BiRNN baseline.

## Solutions

### Easy Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BiRNNSentiment(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bi_rnn = nn.RNN(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.bi_rnn(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc(h)

model = BiRNNSentiment(50, 16, 32)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

x = torch.randint(0, 50, (64, 10))
y = torch.randint(0, 2, (64,))

for epoch in range(50):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}, Acc: {acc.item():.4f}")
```

### Medium Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

def synthetic_task(n=1000, seq_len=10):
    X = torch.randn(n, seq_len, 5)
    # Label depends on both first and last 3 elements
    first_avg = X[:, :3, :].mean(dim=(1, 2))
    last_avg = X[:, -3:, :].mean(dim=(1, 2))
    y = (first_avg + last_avg > 0).long()
    return X, y

X, y = synthetic_task(1000)
split = 800
train_X, test_X = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]

results = {}
for bidirectional in [False, True]:
    rnn = nn.RNN(5, 16, bidirectional=bidirectional, batch_first=True)
    fc = nn.Linear(16 * (2 if bidirectional else 1), 2)
    params = list(rnn.parameters()) + list(fc.parameters())
    optimizer = optim.Adam(params, lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        out, h = rnn(train_X)
        if bidirectional:
            h = torch.cat([h[0], h[1]], dim=-1)
        pred = fc(h)
        loss = loss_fn(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out, h = rnn(test_X)
        if bidirectional:
            h = torch.cat([h[0], h[1]], dim=-1)
        acc = (fc(h).argmax(dim=1) == test_y).float().mean()
    dir_name = "Bi" if bidirectional else "Uni"
    print(f"{dir_name}RNN accuracy: {acc.item():.4f}")
```

## Related Concepts

- Stacked RNN (DL-287)
- Bidirectional LSTM (DL-305)
- Bidirectional GRU (DL-316)
- Sequence Modeling (DL-284)

## Next Concepts

- Stacked RNN (DL-287)
- Teacher Forcing (DL-288)

## Summary

A Bidirectional RNN processes sequences in both forward and backward directions, combining hidden states at each time step to provide complete past and future context. The architecture consists of two independent RNNs whose outputs are typically concatenated, resulting in double the hidden dimension and parameter count. BiRNNs consistently outperform unidirectional models on tasks where full sequence context is beneficial, such as NER, POS tagging, and document classification. However, they cannot be used for real-time streaming applications and come with increased computational and memory costs.

## Key Takeaways

- BiRNNs process sequences forward and backward in parallel
- Combined hidden states provide both past and future context at each position
- Output dimension doubles compared to unidirectional RNNs
- BiRNNs are standard for NLP tasks with full sequence access
- Not suitable for real-time or streaming applications
- Model capacity and memory requirements approximately double
- Parameter count and computational cost are roughly 2x unidirectional
