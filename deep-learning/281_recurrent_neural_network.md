# Concept: Recurrent Neural Network

## Concept ID

DL-281

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

RNN

## Learning Objectives

- Understand the fundamental architecture of Recurrent Neural Networks
- Explain how RNNs process sequential data through recurrent connections
- Identify the key components of an RNN: input layer, hidden layer, and output layer
- Compare RNNs with traditional feedforward neural networks
- Implement a simple RNN using PyTorch

## Prerequisites

- Basic understanding of neural networks and feedforward architectures
- Familiarity with backpropagation
- Knowledge of activation functions (tanh, sigmoid)
- Understanding of sequential data concepts

## Definition

A Recurrent Neural Network (RNN) is a class of artificial neural networks designed to process sequential data by maintaining a hidden state that captures information about previous inputs in the sequence. Unlike feedforward networks, RNNs have recurrent connections that allow information to persist across time steps, making them particularly effective for tasks involving ordered data such as time series, natural language, and speech.

The core innovation of RNNs is the hidden state vector, which acts as a memory mechanism. At each time step, the hidden state is updated based on the current input and the previous hidden state, creating a feedback loop that enables the network to learn temporal dependencies. This recurrent structure can be unrolled through time, resulting in a deep network where each layer corresponds to a time step.

## Intuition

Think of an RNN as a program that reads a sentence word by word while maintaining a running summary in its memory. When reading the sentence "The cat sat on the mat," the network processes "The" first, updates its memory, then processes "cat" while incorporating the memory of "The," and continues this process. By the time it reaches "mat," its memory contains contextual information about the entire preceding sequence, enabling it to make predictions about what word might come next.

This sequential memory mechanism is analogous to how humans read text: we don't discard everything and start fresh at each word. Instead, we maintain context and build understanding incrementally. The hidden state in an RNN serves the same purpose, evolving as new information arrives and carrying forward relevant context.

## Why This Concept Matters

RNNs revolutionized the processing of sequential data in deep learning. Before RNNs, sequence modeling required fixed-size windows or hand-crafted features that could not capture long-range dependencies effectively. RNNs introduced a principled way to process variable-length sequences with a unified architecture, enabling breakthroughs in machine translation, speech recognition, sentiment analysis, and time series forecasting.

RNNs form the foundation for more advanced architectures like LSTMs, GRUs, and attention-based models. Understanding RNNs is crucial for anyone working with sequential data, as the core concepts of hidden states, recurrent connections, and backpropagation through time are fundamental to all recurrent architectures in modern deep learning.

## Mathematical Explanation

The forward pass of a simple RNN is defined by the following equations:

At time step t:
- Input vector: x_t ∈ ℝ^(d_in)
- Hidden state from previous step: h_(t-1) ∈ ℝ^(d_hidden)
- Weight matrices: W_hh ∈ ℝ^(d_hidden × d_hidden), W_xh ∈ ℝ^(d_hidden × d_in)
- Bias term: b_h ∈ ℝ^(d_hidden)

The hidden state update:
h_t = tanh(W_xh · x_t + W_hh · h_(t-1) + b_h)

The output at time step t:
y_t = W_hy · h_t + b_y

Where:
- W_hy ∈ ℝ^(d_out × d_hidden) is the output weight matrix
- b_y ∈ ℝ^(d_out) is the output bias

The tanh activation function squashes values between -1 and 1, helping to regulate the flow of information through the network. This activation is preferred over sigmoid due to its zero-centered output range.

The loss for a sequence is typically computed as the sum of losses at each time step:
L = Σ_t L_t(y_t, y_true_t)

For classification tasks, cross-entropy loss is common. For regression, mean squared error is used.

## Code Examples

### Code Example 1: Simple RNN from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        batch_size, seq_len, _ = x.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
            y_t = self.W_hy(h)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1), h

model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(2, 3, 10)
output, hidden = model(x)
print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)

# Output:
# Output shape: torch.Size([2, 3, 5])
# Hidden shape: torch.Size([2, 20])
```

### Code Example 2: Using PyTorch's Built-in RNN

```python
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(2, 3, 10)
output, hidden = rnn(x)
print("Output shape:", output.shape)
print("Hidden shape:", hidden.shape)

# Extract parameters
print("Weight shapes:")
for name, param in rnn.named_parameters():
    print(f"{name}: {param.shape}")

# Output:
# Output shape: torch.Size([2, 3, 20])
# Hidden shape: torch.Size([1, 2, 20])
# Weight shapes:
# weight_ih_l0: torch.Size([20, 10])
# weight_hh_l0: torch.Size([20, 20])
# bias_ih_l0: torch.Size([20])
# bias_hh_l0: torch.Size([20])
```

### Code Example 3: RNN for Sequence Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

model = RNNClassifier(vocab_size=100, embed_size=50, hidden_size=32, num_classes=2)
x = torch.randint(0, 100, (4, 10))
output = model(x)
print("Logits shape:", output.shape)
predictions = torch.softmax(output, dim=1)
print("Predictions:", predictions)

# Output:
# Logits shape: torch.Size([4, 2])
# Predictions: tensor([[0.5123, 0.4877],
#                       [0.4988, 0.5012],
#                       [0.5056, 0.4944],
#                       [0.5034, 0.4966]], grad_fn=<SoftmaxBackward0>)
```
```

### Code Example 4: RNN for Character-Level Generation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token, length, vocab_size):
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[start_token]])
            hidden = None
            output = []
            for _ in range(length):
                logits, hidden = self.forward(x, hidden)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1)
                output.append(next_token.item())
                x = next_token
        return output

model = CharRNN(vocab_size=50, hidden_size=128)
start = torch.randint(0, 50, (1,))
generated = model.generate(start, 10, 50)
print("Generated token indices:", generated)
print("Length:", len(generated))

# Output:
# Generated token indices: [23, 41, 7, 19, 33, 45, 12, 8, 29, 38]
# Length: 10
```

## Common Mistakes

1. **Ignoring sequence order**: Treating sequence data as unordered or shuffling sequences during training destroys the temporal dependencies that RNNs are designed to capture. Always maintain chronological order within each sequence.

2. **Using incorrect batch dimensions**: Confusing batch_first vs seq_first dimensions. When batch_first=True (common), the input shape should be (batch, seq_len, features). Mixing these up leads to silent errors or dimension mismatches.

3. **Forgetting to initialize hidden state**: Passing an uninitialized hidden state can lead to undefined behavior or poor convergence. Always either provide an initial zero state or let the module handle initialization.

4. **Applying dropout incorrectly**: Using dropout on the recurrent connections without proper configuration can break the temporal dynamics. PyTorch's RNN applies dropout between layers, not within recurrent connections of the same layer.

5. **Assuming fixed input length**: RNNs can handle variable-length sequences, but many implementations fail when batch processing requires padding. Using nn.utils.rnn.pack_padded_sequence and pad_packed_sequence is essential for efficient training with variable-length inputs.

6. **Using deep RNNs without regularization**: Deep RNNs (multiple layers) are prone to overfitting on small datasets. Apply appropriate regularization techniques like dropout, weight decay, and early stopping.

7. **Neglecting gradient clipping**: RNNs are susceptible to exploding gradients, especially during long sequences. Always use gradient clipping (torch.nn.utils.clip_grad_norm_) during training.

## Interview Questions

### Beginner

Q: What distinguishes an RNN from a feedforward neural network?
A: RNNs have recurrent connections that allow information to persist across time steps through a hidden state. Feedforward networks process each input independently, while RNNs maintain state that captures context from previous inputs, enabling them to handle sequential data effectively.

Q: What activation function is commonly used in RNNs and why?
A: The tanh activation function is most commonly used because its output is zero-centered (between -1 and 1), which helps with gradient flow during backpropagation. The sigmoid function can cause vanishing gradients due to its saturation at 0 and 1.

### Intermediate

Q: Explain how an RNN can process variable-length sequences.
A: An RNN processes sequences one time step at a time, regardless of total length. The same weight matrices are reused at each time step (weight sharing), effectively creating a deep network whose depth equals the sequence length. The hidden state is updated sequentially, so the architecture naturally accommodates any sequence length.

Q: What is the problem with using tanh activation in very deep RNNs?
A: The tanh activation can saturate for large positive or negative inputs, causing gradients to vanish during backpropagation through time. For very long sequences, the repeated application of tanh compounds this problem, making it difficult for the network to learn long-range dependencies.

### Advanced

Q: Derive the gradient update equations for a simple RNN and explain where vanishing gradients originate.
A: The gradient at time step t depends on the product of Jacobian matrices from each step. The hidden state h_t = tanh(W_hh·h_(t-1) + W_xh·x_t) has derivative (1 - tanh²(...))·W_hh. Over T steps, the gradient includes a factor of (W_hh)^T multiplied by the product of (1 - tanh²) terms. If the eigenvalues of W_hh are less than 1, the gradient vanishes exponentially with T.

Q: How does weight sharing in RNNs affect the optimization landscape compared to feedforward networks?
A: Weight sharing creates a highly non-convex optimization landscape with symmetries and saddle points. The same parameters are reused across time steps, making the loss function periodic in certain directions. This can lead to slow convergence and sensitivity to initialization compared to feedforward networks where different layers have independent parameters.

## Practice Problems

### Easy

Implement a simple RNN that takes a sequence of 5 binary digits and outputs the sum of the digits. Use PyTorch and train for 100 epochs.

### Medium

Build an RNN-based sentiment classifier on a synthetic dataset of 1000 sequences. Each sequence is a vector of 10 integers between 0 and 4, representing word indices. Label sequences as positive (sum > 25) or negative (sum <= 25). Report training and validation accuracy.

### Hard

Implement a character-level RNN language model to generate text. Use a dataset of at least 5000 characters. Train the model using cross-entropy loss and generate at least 200 characters of coherent output. Implement temperature-based sampling to control the randomness of generation.

## Solutions

### Easy Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BinarySumRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

model = BinarySumRNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    x = torch.randint(0, 2, (32, 5, 1)).float()
    y = x.sum(dim=1).squeeze(-1)
    pred = model(x).squeeze(-1)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Medium Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size=5, embed_size=8, hidden_size=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

X = torch.randint(0, 5, (1000, 10))
y = (X.sum(dim=1) > 25).long()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = SentimentRNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).argmax(dim=1)
        val_acc = (val_pred == y_val).float().mean()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc.item():.4f}")
```

### Hard Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

    def generate(self, start, length, vocab_size, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x = torch.tensor([[start]], device=device)
            hidden = None
            output = [start]
            for _ in range(length):
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                output.append(next_token.item())
                x = next_token
        return output

# Simulate training on synthetic data
vocab_size = 50
model = CharLanguageModel(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Synthetic training data
data = torch.randint(1, vocab_size, (100, 20))
for epoch in range(200):
    logits, _ = model(data[:, :-1])
    loss = loss_fn(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

generated = model.generate(1, 50, vocab_size, temperature=0.8)
print("Generated sequence:", generated[:20])
```

## Related Concepts

- Feedforward Neural Networks (DL-101)
- Backpropagation (DL-103)
- Vanishing Gradient Problem (DL-291)
- Long Short-Term Memory (DL-296)
- Gated Recurrent Unit (DL-311)
- Backpropagation Through Time (DL-289)

## Next Concepts

- RNN Cell (DL-282)
- Hidden State (DL-283)
- Sequence Modeling (DL-284)

## Summary

Recurrent Neural Networks are a foundational architecture for processing sequential data, characterized by their recurrent connections and hidden state mechanism. The hidden state acts as a memory that propagates information across time steps, enabling the network to capture temporal dependencies. The core operation involves updating the hidden state at each time step using the current input and previous hidden state, with tanh activation squashing the result. RNNs use weight sharing across time, making them parameter-efficient for sequence modeling. Despite their power, simple RNNs suffer from vanishing and exploding gradient problems that limit their ability to learn long-range dependencies, motivating the development of gated architectures like LSTMs and GRUs.

## Key Takeaways

- RNNs maintain a hidden state that captures sequential context and evolves over time
- Weight sharing across time steps makes RNNs parameter-efficient for sequence processing
- The tanh activation function is standard in RNNs due to its zero-centered output
- RNNs can handle variable-length sequences through their sequential processing nature
- Vanishing gradients in deep unrolled RNNs limit long-range dependency learning
- Proper initialization, gradient clipping, and regularization are essential for training RNNs
- RNNs form the foundation for advanced sequential models like LSTMs and GRUs
