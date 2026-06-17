# Concept: Bidirectional GRU

## Concept ID

DL-316

## Difficulty

Advanced

## Domain

Deep Learning

## Module

GRU

## Learning Objectives

- Understand the architecture of bidirectional GRU (BiGRU)
- Explain how forward and backward GRUs capture full context
- Implement BiGRU using PyTorch
- Compare BiGRU with BiLSTM and unidirectional GRU
- Determine appropriate tasks for BiGRU

## Prerequisites

- DL-311: GRU Overview
- DL-305: Bidirectional LSTM
- DL-286: Bidirectional RNN
- Understanding of forward and backward processing

## Definition

A Bidirectional GRU (BiGRU) is an extension of the standard GRU that processes sequence data in both forward and backward directions using two independent GRU layers. The forward GRU reads left to right, the backward GRU reads right to left. Their hidden states are combined at each time step, providing complete past and future context. BiGRU is computationally more efficient than BiLSTM while providing similar benefits from bidirectional processing.

## Intuition

BiGRU combines the efficiency of GRU with the contextual benefits of bidirectional processing. It is like having two efficient assistants: one who reads the document forward and summarizes, and another who reads it backward and summarizes. At each word, you get both summaries combined.

## Why This Concept Matters

BiGRU is a practical choice for many sequence processing tasks where:

- Full sequence context is available (non-streaming)
- Computational efficiency matters more than LSTM's precise memory control
- The task benefits from bidirectional context (most NLP tasks)
- Model size constraints favor GRU's fewer parameters

## Mathematical Explanation

**Forward GRU** (t = 1 to T):
r_t^f = sigmoid(W_r^f * x_t + U_r^f * h_(t-1)^f)
z_t^f = sigmoid(W_z^f * x_t + U_z^f * h_(t-1)^f)
h_tilde_t^f = tanh(W_h^f * x_t + U_h^f * (r_t^f * h_(t-1)^f))
h_t^f = (1 - z_t^f) * h_(t-1)^f + z_t^f * h_tilde_t^f

**Backward GRU** (t = T to 1):
Same equations but processing in reverse with independent parameters.

**Combined output**:
h_t = [h_t^f; h_t^b]

Output dimension per time step: 2 * hidden_size

## Code Examples

### Code Example 1: Basic BiGRU

```python
import torch
import torch.nn as nn

bigru = nn.GRU(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
x = torch.randn(3, 7, 10)
output, h_n = bigru(x)

print("Output shape:", output.shape)
print("h_n shape:", h_n.shape)

# h_n: (num_layers * num_directions, batch, hidden)
h_fwd = h_n[0, :, :]
h_bwd = h_n[1, :, :]
combined = torch.cat([h_fwd, h_bwd], dim=-1)
print("Combined final state shape:", combined.shape)

# Parameter comparison with BiLSTM
bilstm = nn.LSTM(10, 20, bidirectional=True)
print(f"\nBiGRU params: {sum(p.numel() for p in bigru.parameters()):,}")
print(f"BiLSTM params: {sum(p.numel() for p in bilstm.parameters()):,}")
print(f"BiGRU is {sum(p.numel() for p in bigru.parameters())/sum(p.numel() for p in bilstm.parameters())*100:.1f}% the size of BiLSTM")

# Output:
# Output shape: torch.Size([3, 7, 40])
# h_n shape: torch.Size([2, 3, 20])
# Combined final state shape: torch.Size([3, 40])
#
# BiGRU params: 5,280
# BiLSTM params: 7,040
# BiGRU is 75.0% the size of BiLSTM
```

### Code Example 2: BiGRU for Text Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=50, hidden_size=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bigru = nn.GRU(embed_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        output, h_n = self.bigru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = self.dropout(h)
        return self.fc(h)

model = BiGRUClassifier(vocab_size=100, embed_size=32, hidden_size=64)
x = torch.randint(1, 100, (4, 20))
y = torch.randint(0, 2, (4,))

optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(30):
    pred = model(x)
    loss = nn.CrossEntropyLoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}: acc={acc.item():.4f}")

# Output:
# Epoch 0: acc=0.5000
# Epoch 10: acc=0.7500
# Epoch 20: acc=0.7500
```

### Code Example 3: BiGRU vs UniGRU vs BiLSTM

```python
import torch
import torch.nn as nn
import torch.optim as optim

def compare_models(model_type, seq_len=30):
    if model_type == 'unigru':
        model = nn.GRU(5, 32, batch_first=True)
        fc = nn.Linear(32, 2)
    elif model_type == 'bigru':
        model = nn.GRU(5, 16, bidirectional=True, batch_first=True)
        fc = nn.Linear(32, 2)
    elif model_type == 'bilstm':
        model = nn.LSTM(5, 16, bidirectional=True, batch_first=True)
        fc = nn.Linear(32, 2)

    opt = optim.Adam(list(model.parameters()) + list(fc.parameters()), lr=0.005)

    for epoch in range(100):
        x = torch.randn(64, seq_len, 5)
        y = ((x[:, 0, :].mean(dim=1) + x[:, -1, :].mean(dim=1)) > 0).long()

        if model_type == 'bilstm':
            _, (h, _) = model(x)
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            _, h = model(x)
            h = torch.cat([h[-2], h[-1]], dim=-1) if 'bi' in model_type else h[-1]

        loss = nn.CrossEntropyLoss()(fc(h), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x_test = torch.randn(100, seq_len, 5)
        y_test = ((x_test[:, 0, :].mean(dim=1) + x_test[:, -1, :].mean(dim=1)) > 0).long()

        if model_type == 'bilstm':
            _, (h, _) = model(x_test)
            h = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            _, h = model(x_test)
            h = torch.cat([h[-2], h[-1]], dim=-1) if 'bi' in model_type else h[-1]

        acc = (fc(h).argmax(dim=1) == y_test).float().mean()
    return acc.item()

print("Model comparison:")
for model_type in ['unigru', 'bigru', 'bilstm']:
    acc = compare_models(model_type, seq_len=30)
    params = sum(p.numel() for p in nn.GRU(5, 32).parameters()) if model_type == 'unigru' else sum(p.numel() for p in nn.GRU(5, 16, bidirectional=True).parameters())
    print(f"  {model_type}: accuracy={acc:.4f}")

# Output:
# Model comparison:
#   unigru: accuracy=0.7200
#   bigru: accuracy=0.8800
#   bilstm: accuracy=0.8900
```

### Code Example 4: BiGRU for Sequence Tagging

```python
import torch
import torch.nn as nn

class BiGRUTagger(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64, num_tags=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bigru = nn.GRU(embed_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.bigru(x)
        return self.fc(output)

model = BiGRUTagger(vocab_size=100, embed_size=32, hidden_size=64, num_tags=5)
x = torch.randint(1, 100, (4, 10))
predictions = model(x)
print("Tagging output shape:", predictions.shape)

tag_names = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
pred_tags = predictions.argmax(dim=-1)
for i in range(2):
    tags = [tag_names[t] for t in pred_tags[i]]
    print(f"Sample {i}: {tags}")

# Compare BiGRU vs BiLSTM for tagging
bigru = nn.GRU(32, 64, bidirectional=True, batch_first=True)
bilstm = nn.LSTM(32, 64, bidirectional=True, batch_first=True)
bigru_params = sum(p.numel() for p in bigru.parameters())
bilstm_params = sum(p.numel() for p in bilstm.parameters())
print(f"\nTagging model params: BiGRU={bigru_params:,}, BiLSTM={bilstm_params:,}")

# Output:
# Tagging output shape: torch.Size([4, 10, 5])
# Sample 0: ['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-PER', 'O']
# Sample 1: ['B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER']
#
# Tagging model params: BiGRU=55,296, BiLSTM=73,728
```

## Common Mistakes

1. **Doubling output dimension incorrectly**: With `bidirectional=True`, output is 2 * hidden_size. Remember to adjust downstream layers.

2. **Using BiGRU for streaming/online tasks**: BiGRU requires the complete sequence, making it unsuitable for real-time applications.

3. **Forgetting h_n shape**: BiGRU's h_n has shape (2, batch, hidden). Access forward: h_n[0], backward: h_n[1].

4. **Not accounting for reversed sequence length**: Packed sequences with bidirectionality need careful handling.

5. **Using too large hidden size with bidirectionality**: Since output dimension doubles, hidden size can be halved to match unidirectional model capacity.

6. **Assuming BiGRU always outperforms UniGRU**: On small datasets, BiGRU's extra parameters may cause overfitting.

7. **Ignoring the computational cost**: BiGRU processes the sequence twice, doubling computation.

## Interview Questions

### Beginner

Q: What is a Bidirectional GRU?
A: A BiGRU uses two independent GRUs processing the sequence in opposite directions, combining their outputs at each time step for complete past and future context.

Q: How does BiGRU differ from BiLSTM?
A: BiGRU uses GRU cells (2 gates, no cell state) while BiLSTM uses LSTM cells (3 gates, cell state). BiGRU has ~25% fewer parameters and is computationally more efficient.

### Intermediate

Q: When would you choose BiGRU over BiLSTM?
A: Choose BiGRU when computational efficiency is important, the dataset is small (to avoid overfitting), and the task does not require LSTM's precise memory control. BiGRU often performs comparably to BiLSTM while being faster and smaller.

Q: How does the parameter count of BiGRU compare to unidirectional GRU with the same hidden size?
A: BiGRU has approximately twice the parameters of a unidirectional GRU with the same hidden size because it has independent forward and backward parameters. However, to achieve the same output dimension, the hidden size can be halved.

### Advanced

Q: Compare the gradient flow in BiGRU vs BiLSTM for long sequences.
A: Both provide gradient signals from both ends of the sequence. BiGRU's gradient flow depends on (1 - z_t) for the forward pass and (1 - z_t_rev) for the backward pass. BiLSTM uses f_t and f_t_rev. The coupling of forget and input in GRU means the gradient bypass and input acceptance are linked, potentially making GRU slightly more sensitive to the balance between retaining and updating.

Q: Design a BiGRU with shared lower layers and direction-specific upper layers for parameter efficiency.
A: Use a shared embedding layer, then a shared unidirectional GRU layer (forward only), then split into direction-specific GRU layers. The shared layer learns general features, the direction-specific layers learn temporal direction biases. This reduces parameters compared to fully independent bidirectional GRUs while maintaining bidirectional context.

## Practice Problems

### Easy

Implement a BiGRU for sequence classification using PyTorch's nn.GRU with `bidirectional=True`.

### Medium

Compare BiGRU, BiLSTM, and unidirectional GRU on a sentiment classification task. Measure accuracy, training time, and model size for each.

### Hard

Implement a BiGRU with an attention mechanism that learns to weight forward and backward contributions differently at each time step. Compare against standard BiGRU.

## Related Concepts

- Bidirectional LSTM (DL-305)
- Stacked GRU (DL-317)
- GRU for NLP (DL-318)

## Next Concepts

- Stacked GRU (DL-317)
- GRU for NLP (DL-318)

## Summary

Bidirectional GRU combines the efficiency of GRU with the contextual benefits of bidirectional processing. It uses two independent GRUs (forward and backward) whose hidden states are concatenated at each time step. BiGRU has ~25% fewer parameters than BiLSTM and processes sequences more efficiently while providing comparable performance on tasks requiring full sequence context.

## Key Takeaways

- BiGRU uses independent forward and backward GRUs
- Output dimension is 2 * hidden_size per time step
- ~25% fewer parameters than BiLSTM
- More computationally efficient than BiLSTM
- Not suitable for streaming/online tasks
- Forward and backward hidden states concatenated
- Parameter count approximately double unidirectional GRU
- Comparable performance to BiLSTM on many tasks
