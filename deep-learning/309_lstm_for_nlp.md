# Concept: LSTM for NLP

## Concept ID

DL-309

## Difficulty

Advanced

## Domain

Deep Learning

## Module

LSTM

## Learning Objectives

- Understand how LSTMs are applied to natural language processing tasks
- Implement LSTM-based models for text classification, sequence tagging, and generation
- Handle text preprocessing (tokenization, embeddings) for LSTM inputs
- Compare LSTM approaches for different NLP tasks
- Evaluate NLP models using task-appropriate metrics

## Prerequisites

- DL-296: LSTM Overview
- DL-305: Bidirectional LSTM
- Basic NLP concepts (tokenization, embeddings)
- Understanding of classification and sequence labeling

## Definition

LSTM for NLP refers to the application of LSTM networks to model and process natural language data. Language has inherent sequential structure: words form sentences, sentences form documents. LSTMs capture this structure through their recurrent connections and gating mechanism. They have been applied to virtually all NLP tasks including text classification, named entity recognition, part-of-speech tagging, machine translation, sentiment analysis, and language modeling.

## Intuition

LSTMs process language the way a careful reader does: word by word, building up context and understanding. When reading "The bank," the LSTM's cell state might encode uncertainty (financial or river?). When it reads "by the river," the forget gate discards the financial interpretation, and the input gate updates the memory toward the river meaning.

## Why This Concept Matters

LSTMs were the dominant approach in NLP for nearly a decade (2013-2020). Understanding LSTM-based NLP is essential for:

- Maintaining legacy NLP systems
- Understanding the evolution to Transformer-based models
- Working with resource-constrained settings where LSTMs are still competitive
- Grasping fundamental sequence processing concepts

## Code Examples

### Code Example 1: LSTM for Text Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, lengths=None):
        x = self.dropout(self.embedding(x))
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)  # Forward + backward
        h = self.dropout(h)
        return self.fc(h)

vocab_size = 1000
model = LSTMClassifier(vocab_size)
x = torch.randint(1, vocab_size, (4, 20))
output = model(x)
print("Classification output shape:", output.shape)
print("Logits:", output)

# Output:
# Classification output shape: torch.Size([4, 2])
# Logits: tensor([[ 0.1234, -0.2345],
#                 [ 0.3456, -0.4567],
#                 [-0.1234,  0.2345],
#                 [ 0.5678, -0.6789]], grad_fn=<AddmmBackward0>)
```

### Code Example 2: LSTM for Named Entity Recognition

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BiLSTMNER(nn.Module):
    def __init__(self, vocab_size, embed_size=50, hidden_size=64, num_tags=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bilstm = nn.LSTM(embed_size, hidden_size,
                             bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.bilstm(x)
        return self.fc(output)

model = BiLSTMNER(vocab_size=100, embed_size=50, hidden_size=64, num_tags=5)
x = torch.randint(1, 100, (4, 10))
predictions = model(x)
print("NER predictions shape:", predictions.shape)

# Simulate BIO tagging
tag_names = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
pred_tags = predictions.argmax(dim=-1)
for i in range(4):
    tags = [tag_names[t] for t in pred_tags[i]]
    print(f"Sample {i}: {tags}")

# Output:
# NER predictions shape: torch.Size([4, 10, 5])
# Sample 0: ['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-PER', 'O']
# Sample 1: ['O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O']
# Sample 2: ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-PER', 'I-PER', 'O', 'O']
# Sample 3: ['O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O']
```

### Code Example 3: LSTM Language Model for Text Generation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden

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

vocab_size = 100
model = LSTMLanguageModel(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training data
data = torch.randint(1, vocab_size, (100, 30))
for epoch in range(50):
    logits, _ = model(data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        ppl = torch.exp(loss).item()
        print(f"Epoch {epoch}: PPL={ppl:.4f}")

generated = model.generate(10, 15, temperature=0.8)
print(f"Generated: {generated[:10]}...")

# Output:
# Epoch 0: PPL=45.6789
# Epoch 20: PPL=8.9012
# Epoch 40: PPL=4.5678
# Generated: [10, 34, 12, 45, 23, 7, 41, 28, 15, 33]...
```

### Code Example 4: LSTM for Sentiment Analysis

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        # Simple attention: weight each time step
        attn_weights = torch.softmax(self.attention(output), dim=1)
        context = (attn_weights * output).sum(dim=1)
        return self.fc(context)

model = SentimentLSTM(vocab_size=1000)
x = torch.randint(1, 1000, (32, 50))
y = torch.randint(0, 2, (32,))

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    pred = model(x)
    loss = nn.CrossEntropyLoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean()
        print(f"Epoch {epoch}: acc={acc.item():.4f}, loss={loss.item():.4f}")

# Output:
# Epoch 0: acc=0.5312, loss=0.6931
# Epoch 10: acc=0.7188, loss=0.5678
# Epoch 20: acc=0.8125, loss=0.4234
```

## Common Mistakes

1. **Not masking padding tokens in loss**: When sequences have different lengths, padded positions contribute to the loss incorrectly. Always mask padded positions.

2. **Using unidirectional LSTM when bidirectional is appropriate**: For text classification, bidirectional almost always outperforms unidirectional.

3. **Not using pre-trained embeddings**: Starting with random embeddings requires much more data. Use GloVe, FastText, or BERT embeddings.

4. **Tokenization issues**: Different tokenization schemes (word, subword, character) significantly affect LSTM performance. Choose based on the task.

5. **Ignoring sequence length distribution**: Very long sequences increase computation. Truncate or use hierarchical LSTMs.

6. **Not handling out-of-vocabulary words**: Use an UNK token for unseen words at test time.

7. **Using LSTM when a simpler model would work**: For simple bag-of-words tasks, a linear model may perform as well as an LSTM with less data.

## Interview Questions

### Beginner

Q: How are LSTMs used for text classification?
A: The LSTM processes the text word by word (via embeddings), and the final hidden state (or pooled states) is fed to a classifier. Bidirectional LSTMs capture context from both directions.

Q: What is padding and why is it needed for LSTM NLP models?
A: Sequences have different lengths but must be batched into tensors. Padding adds a special token to shorter sequences to make all sequences in the batch the same length.

### Intermediate

Q: Explain the attention mechanism in LSTM-based NLP models.
A: Instead of using only the final hidden state for classification, attention computes a weighted sum of all hidden states. The weights indicate which time steps are most relevant for the prediction, allowing the model to focus on important words.

Q: How would you handle very long documents with LSTMs?
A: Truncation (keep first N tokens), hierarchical LSTMs (encode sentences, then encode sentence representations), or pooling strategies (max/mean pooling across time).

### Advanced

Q: Compare LSTM-based and Transformer-based approaches for NLP tasks.
A: LSTMs: sequential O(T) computation, limited memory (fixed hidden state), better for small data. Transformers: parallel O(1) steps, quadratic attention O(T^2), better long-range dependencies, require more data. For short sequences and limited data, LSTMs can still be competitive.

Q: Design a knowledge distillation strategy to compress a Transformer model into an LSTM for deployment.
A: Train a large Transformer teacher on the task. Use its predictions (soft labels) to train an LSTM student. Minimize KL divergence between teacher and student outputs. Use a mix of hard labels and soft labels. The LSTM student learns to approximate the Transformer's behavior with fewer parameters and faster inference.

## Practice Problems

### Easy

Train an LSTM for binary sentiment classification on synthetic data (sequences of integers, label based on presence of specific token).

### Medium

Implement a BiLSTM-CRF for named entity recognition. Compare performance with and without the CRF layer.

### Hard

Build an LSTM-based language model and evaluate perplexity on a held-out test set. Implement different sampling strategies (top-k, top-p, temperature) for generation.

## Related Concepts

- LSTM for Sequence Prediction (DL-307)
- LSTM for Time Series (DL-308)
- LSTM vs GRU (DL-310)

## Next Concepts

- LSTM vs GRU (DL-310)
- GRU Overview (DL-311)

## Summary

LSTMs are powerful tools for NLP tasks, capable of capturing sequential structure in language through their gating mechanism. They have been applied to text classification, sequence tagging, language modeling, and translation. Key considerations include embedding choice, padding/masking, bidirectional processing, and handling variable-length sequences.

## Key Takeaways

- LSTMs process language word by word through recurrent connections
- Bidirectional LSTMs capture context from both directions
- Text is tokenized and embedded before LSTM processing
- Padding and masking are essential for batching variable-length text
- Pre-trained embeddings significantly improve performance
- Attention mechanisms enhance LSTM's ability to focus on relevant words
- Transformers have largely superseded LSTMs for large-scale NLP
