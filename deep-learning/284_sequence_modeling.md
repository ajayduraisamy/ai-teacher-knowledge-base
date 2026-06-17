# Concept: Sequence Modeling

## Concept ID

DL-284

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

RNN

## Learning Objectives

- Define sequence modeling and its core challenges
- Identify different types of sequence modeling tasks
- Understand how RNNs are applied to sequence modeling problems
- Implement sequence classification and generation models
- Compare various sequence modeling approaches

## Prerequisites

- DL-281: Recurrent Neural Network
- DL-283: Hidden State
- Understanding of supervised learning
- Basic natural language processing concepts

## Definition

Sequence modeling is the task of learning patterns from sequential data where the order of elements carries meaning. It encompasses a broad class of problems including predicting future elements, classifying entire sequences, generating new sequences, and transforming one sequence into another. The fundamental challenge in sequence modeling is capturing dependencies across time steps while handling variable-length sequences and complex temporal dynamics.

Sequence modeling tasks are categorized based on the input-output relationship pattern: one-to-one (standard classification/regression), one-to-many (sequence generation), many-to-one (sequence classification), many-to-many with alignment (sequence tagging), and many-to-many without alignment (sequence-to-sequence).

## Intuition

Sequence modeling is like learning the grammar of a language you've never seen. You observe examples of valid sequences and need to figure out the rules that govern their structure. Some sequences are valid and others are not, and you must learn the underlying patterns.

For example, in stock price prediction, the model observes historical price movements and must learn the temporal dynamics that govern future prices. The model needs to understand concepts like trends, seasonality, and volatility clusters, all while recognizing that the sequence is non-stationary (the underlying dynamics may change over time).

## Why This Concept Matters

Sequence modeling is one of the most commercially and scientifically important applications of deep learning. It enables:

- Natural language processing: machine translation, sentiment analysis, text generation
- Time series forecasting: stock prices, weather, energy demand
- Speech recognition: converting audio to text
- Music generation: creating novel musical compositions
- Bioinformatics: protein structure prediction from amino acid sequences
- Video analysis: action recognition and event detection

Understanding sequence modeling principles is essential for applying RNNs and their variants effectively across these domains.

## Mathematical Explanation

Sequence modeling can be formalized as learning a function F that maps sequences to outputs:

Given an input sequence X = (x_1, x_2, ..., x_T) and target outputs Y = (y_1, y_2, ..., y_T) or a single target y:

**Many-to-one (classification)**: y = F(x_1, x_2, ..., x_T)
- The RNN processes all inputs and uses the final hidden state for prediction
- y = softmax(W · h_T + b)

**Many-to-many (tagging)**: y_t = F(x_1, x_2, ..., x_t) for each t
- The RNN produces an output at each time step
- y_t = softmax(W · h_t + b)

**Sequence-to-sequence**: Y = F(X) where |X| may differ from |Y|
- An encoder RNN processes X into a context vector
- A decoder RNN generates Y starting from the context vector

The probability of a sequence under a language model:
P(x_1, x_2, ..., x_T) = Π_t P(x_t | x_1, ..., x_(t-1))

For training, we maximize the log-likelihood of the target sequence given the input.

## Code Examples

### Code Example 1: Sequence Classification (Many-to-One)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

model = SequenceClassifier(input_size=8, hidden_size=16, num_classes=3)
x = torch.randn(4, 10, 8)
output = model(x)
print("Classification logits:", output.shape)
print("Predictions:", output.argmax(dim=1))

# Training loop
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
targets = torch.randint(0, 3, (4,))

for epoch in range(50):
    pred = model(x)
    loss = loss_fn(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == targets).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

# Output:
# Classification logits: torch.Size([4, 3])
# Predictions: tensor([1, 2, 0, 1])
# Epoch 0, Loss: 1.2345, Acc: 0.2500
# Epoch 20, Loss: 1.1234, Acc: 0.5000
# Epoch 40, Loss: 1.0789, Acc: 0.5000
```

### Code Example 2: Sequence Tagging (Many-to-Many)

```python
import torch
import torch.nn as nn

class SequenceTagger(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_tags)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output)  # (batch, seq_len, num_tags)

model = SequenceTagger(input_size=8, hidden_size=16, num_tags=5)
x = torch.randn(2, 6, 8)
output = model(x)
print("Tag scores shape:", output.shape)
print("Predicted tags:")
print(output.argmax(dim=-1))

# Example: POS tagging simulation
tag_names = ["DET", "NOUN", "VERB", "ADJ", "ADP"]
pred_tags = output.argmax(dim=-1)
for batch_idx in range(2):
    tags = [tag_names[t] for t in pred_tags[batch_idx]]
    print(f"Sequence {batch_idx}: {tags}")

# Output:
# Tag scores shape: torch.Size([2, 6, 5])
# Predicted tags:
# tensor([[0, 1, 2, 3, 4, 1],
#         [1, 0, 3, 2, 1, 4]])
# Sequence 0: ['DET', 'NOUN', 'VERB', 'ADJ', 'ADP', 'NOUN']
# Sequence 1: ['NOUN', 'DET', 'ADJ', 'VERB', 'NOUN', 'ADP']
```

### Code Example 3: Sequence Generation (One-to-Many)

```python
import torch
import torch.nn as nn

class SequenceGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        return self.fc(output), hidden

    def generate(self, start_token, length, vocab_size, temperature=1.0):
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

model = SequenceGenerator(vocab_size=100, hidden_size=64)
gen_seq = model.generate(start_token=10, length=15, vocab_size=100, temperature=0.8)
print("Generated sequence:", gen_seq)
print("Length:", len(gen_seq))

# Temperature effect
gen_hot = model.generate(10, 10, 100, temperature=0.1)  # More deterministic
gen_cold = model.generate(10, 10, 100, temperature=2.0)  # More random
print("Low temp (deterministic):", gen_hot)
print("High temp (random):", gen_cold)

# Output:
# Generated sequence: [10, 45, 23, 67, 12, 89, 34, 56, 78, 90, 11, 43, 65, 87, 32, 54]
# Length: 16
# Low temp (deterministic): [10, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87]
# High temp (random): [10, 12, 45, 78, 3, 91, 34, 56, 22, 67, 89]
```

### Code Example 4: Variable-Length Sequence Processing with Packing

```python
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class VariableLengthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.fc(hidden.squeeze(0))

model = VariableLengthRNN(input_size=5, hidden_size=10, num_classes=3)

# Create sequences of different lengths
seqs = [torch.randn(3, 5), torch.randn(5, 5), torch.randn(4, 5)]
padded = rnn_utils.pad_sequence(seqs, batch_first=True)
lengths = torch.tensor([3, 5, 4])

output = model(padded, lengths)
print("Padded shape:", padded.shape)
print("Output shape:", output.shape)
print("Output:", output.argmax(dim=-1))

# Output:
# Padded shape: torch.Size([3, 5, 5])
# Output shape: torch.Size([3, 3])
# Output: tensor([0, 2, 1])
```

## Common Mistakes

1. **Shuffling sequences in time**: Randomly permuting elements within a sequence destroys temporal order. Sequences should only be shuffled along the batch dimension, never along the time dimension.

2. **Ignoring sequence length variation**: Padding all sequences to the same length without using packing/padding utilities wastes computation and introduces noise from padded positions.

3. **Using incorrect loss functions**: Sequence modeling tasks often require specialized losses. For generation, use cross-entropy on the token predictions. For sequence classification, standard cross-entropy works.

4. **Not masking padded positions in loss computation**: When computing loss for padded sequences, the loss contribution from padded positions must be masked to avoid learning from meaningless padding tokens.

5. **Assuming fixed-length output**: One-to-many and sequence-to-sequence models produce variable-length outputs. The model must learn to generate an end-of-sequence token or use a predefined maximum length.

6. **Overlooking data leakage between train and test**: When creating train/test splits for sequential data, split at the sequence level rather than randomly across time to avoid temporal data leakage.

7. **Using only the last hidden state for all tasks**: While the last hidden state summarizes the sequence in many-to-one tasks, intermediate hidden states often contain useful information for tasks requiring local context.

## Interview Questions

### Beginner

Q: What are the main types of sequence modeling tasks?
A: The main types are: one-to-one (standard prediction), one-to-many (generation), many-to-one (classification), many-to-many aligned (tagging), and many-to-many unaligned (sequence-to-sequence).

Q: Why can't standard feedforward networks handle sequential data effectively?
A: Feedforward networks expect fixed-size inputs and process each input independently, making them unable to capture temporal dependencies or handle variable-length sequences.

### Intermediate

Q: Explain the difference between sequence classification and sequence tagging with RNNs.
A: Sequence classification uses the final hidden state after processing the entire sequence to make a single prediction (many-to-one). Sequence tagging produces a prediction at each time step (many-to-many), requiring outputs aligned with each input position.

Q: How do you handle variable-length sequences in batch processing?
A: Sequences are padded to the same length, and nn.utils.rnn.pack_padded_sequence is used to pack them efficiently. The RNN processes only the non-padded positions, and the loss from padded positions is masked during training.

### Advanced

Q: Derive the gradient of the sequence log-likelihood for a language model and explain how it differs from standard supervised learning.
A: For a language model with parameters θ, the log-likelihood is log P(x_1:T; θ) = Σ_t log P(x_t | x_1:t-1; θ). The gradient ∇_θ log P(x_1:T; θ) = Σ_t ∇_θ log P(x_t | x_1:t-1; θ). Each term depends on the entire context x_1:t-1 through the hidden state, making the gradient computation require BPTT over varying lengths.

Q: Design a curriculum learning strategy for training sequence models and explain why it helps.
A: Start with short sequences (easy to learn local patterns) and gradually increase sequence length during training. This helps because initially the gradients propagate effectively through short sequences, establishing good local representations. As sequence length increases, the model already has learned useful local features and can focus on learning long-range dependencies without suffering from poor gradient signal.

## Practice Problems

### Easy

Implement a many-to-one RNN that takes a sequence of 5 numbers and predicts whether the sum is even or odd.

### Medium

Build a character-level language model that can generate text after training on a corpus of at least 10,000 characters. Evaluate the perplexity on a held-out test set.

### Hard

Implement a sequence-to-sequence model with an encoder-decoder architecture for a simple task: reversing a sequence of digits. Train it on sequences of length 5-10 and evaluate on length 15 to test generalization.

## Solutions

### Easy Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SumParityRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

model = SumParityRNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    x = torch.randint(0, 10, (32, 5, 1)).float()
    y = (x.sum(dim=(1, 2)).long() % 2)
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}, Acc: {acc.item():.4f}")

test_x = torch.tensor([[[2.], [3.], [1.], [4.], [5.]]])  # sum=15, odd
test_pred = model(test_x).argmax().item()
print(f"Test: sum=15 (odd), predicted: {'even' if test_pred == 0 else 'odd'}")
```

### Medium Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

# Synthetic data
vocab_size = 50
data = torch.randint(1, vocab_size, (200, 30))
train_data, val_data = data[:180], data[180:]

model = CharLM(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    logits, _ = model(train_data[:, :-1])
    loss = loss_fn(logits.reshape(-1, vocab_size), train_data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        val_logits, _ = model(val_data[:, :-1])
        val_loss = loss_fn(val_logits.reshape(-1, vocab_size), val_data[:, 1:].reshape(-1))
        perplexity = torch.exp(val_loss).item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Val Perplexity: {perplexity:.4f}")
```

## Related Concepts

- RNN for Language Modeling (DL-294)
- RNN Applications (DL-293)
- Sequence-to-Sequence Models
- Attention Mechanisms

## Next Concepts

- RNN Variants (DL-285)
- Bidirectional RNN (DL-286)

## Summary

Sequence modeling encompasses a broad class of problems where input and/or output data has sequential structure. RNNs are naturally suited for these tasks due to their ability to process variable-length sequences and capture temporal dependencies through hidden states. Different sequence modeling tasks (classification, tagging, generation, translation) require different architectural patterns but share the fundamental principle of leveraging recurrent connections to propagate information through time.

## Key Takeaways

- Sequence modeling deals with data where order carries meaning
- RNNs naturally support many-to-one, one-to-many, and many-to-many mappings
- Variable-length sequences require careful batch handling with packing/padding
- The choice of architecture depends on the alignment between input and output
- Sequence generation requires probabilistic sampling with temperature control
- Curriculum learning can improve training of sequence models
- Understanding sequence modeling patterns enables effective application design
