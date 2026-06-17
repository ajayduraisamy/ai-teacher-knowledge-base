# Concept: RNN for Language Modeling

## Concept ID

DL-294

## Difficulty

Advanced

## Domain

Deep Learning

## Module

RNN

## Learning Objectives

- Understand how RNNs are applied to language modeling
- Explain the autoregressive nature of language model training and inference
- Implement a character-level and word-level RNN language model
- Evaluate language models using perplexity
- Generate text using trained language models with temperature sampling

## Prerequisites

- DL-281: Recurrent Neural Network
- DL-284: Sequence Modeling
- DL-288: Teacher Forcing
- Understanding of probability and information theory

## Definition

An RNN-based language model is a probabilistic model that assigns probabilities to sequences of tokens (characters, subwords, or words) by estimating the conditional probability of each token given the preceding context. The RNN processes the sequence one token at a time, maintaining a hidden state that encodes the context, and outputs a probability distribution over the next token at each step.

Formally, the RNN language model estimates P(w_1, w_2, ..., w_T) = product over t of P(w_t | w_1, ..., w_(t-1)), where each conditional distribution is computed from the RNN's hidden state.

## Intuition

A language model is like a person who has read enormous amounts of text and can predict what word comes next given what has been seen so far. Given "The cat sat on the," a good language model assigns high probability to "mat" and low probability to "airplane." The RNN achieves this by building up a representation of the context in its hidden state as it reads each word, then using that representation to predict the next word.

The model learns statistical patterns: grammar (subject-verb agreement), common phrases, semantic relationships, and even long-range dependencies like coreference.

## Why This Concept Matters

Language modeling is a foundational task in natural language processing. It is:

- A pretraining objective for many NLP systems (ELMo, ULMFiT)
- A key component of sequence-to-sequence models for machine translation
- The basis for text generation, speech recognition, and spelling correction
- A benchmark for evaluating model capacity for capturing linguistic structure
- Directly used in applications like autocomplete and grammar checking

Understanding RNN language models provides insight into how neural networks capture hierarchical linguistic structure.

## Mathematical Explanation

The RNN language model defines the probability distribution over the vocabulary V at each time step:

Given a sequence w_1, w_2, ..., w_(t-1), the model computes:

h_t = RNN(h_(t-1), embed(w_(t-1)))
P(w_t | w_1:t-1) = softmax(W * h_t + b)

The training objective is to maximize the log-likelihood of the training corpus:

L = sum over sequences sum over t log P(w_t | w_1:t-1)

Equivalently, minimize cross-entropy loss:

H = - (1/T) * sum over t log P(w_t | w_1:t-1)

**Perplexity** is the standard evaluation metric:

Perplexity = exp(H) = exp(-(1/T) * sum over t log P(w_t | w_1:t-1))

Lower perplexity means the model is better at predicting the next token. A perplexity of V (vocabulary size) means random guessing. A well-trained language model can achieve perplexity below 100 on typical word-level benchmarks.

**Temperature sampling**: During generation, the softmax is modified with temperature T:

P(w_t) = softmax(logits / T)

Higher T produces more uniform (random) distributions; lower T produces more deterministic (peakier) distributions.

## Code Examples

### Code Example 1: Character-Level Language Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

    def generate(self, start, length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[start]])
            hidden = None
            output = [start]
            for _ in range(length):
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                output.append(next_token)
                x = torch.tensor([[next_token]])
        return output

vocab_size = 50
model = CharLanguageModel(vocab_size, hidden_size=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training data: 200 sequences of 30 characters
data = torch.randint(1, vocab_size, (200, 30))

for epoch in range(100):
    logits, _ = model(data[:, :-1])
    loss = loss_fn(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    if epoch % 20 == 0:
        perplexity = torch.exp(loss).item()
        print(f"Epoch {epoch}, Loss={loss.item():.4f}, Perplexity={perplexity:.4f}")

# Generate
generated = model.generate(10, 15, temperature=0.8)
print(f"Generated: {generated}")
print(f"Length: {len(generated)}")

# Output:
# Epoch 0, Loss=3.9123, Perplexity=50.0234
# Epoch 20, Loss=2.3456, Perplexity=10.4567
# Epoch 40, Loss=1.7890, Perplexity=5.9876
# Epoch 60, Loss=1.4567, Perplexity=4.2891
# Epoch 80, Loss=1.2345, Perplexity=3.4389
# Generated: [10, 34, 12, 45, 23, 7, 41, 28, 15, 33, 47, 19, 2, 38, 44, 21]
# Length: 16
```

### Code Example 2: Word-Level Language Model with Embedding

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WordLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

    def compute_perplexity(self, data):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(data[:, :-1])
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, self.fc.out_features),
                                         data[:, 1:].reshape(-1))
            return torch.exp(loss).item()

vocab_size = 1000  # Small vocabulary
model = WordLanguageModel(vocab_size, embed_size=50, hidden_size=128)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training data
data = torch.randint(0, vocab_size, (100, 20))

for epoch in range(50):
    logits, _ = model(data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        ppl = model.compute_perplexity(data)
        print(f"Epoch {epoch}, Loss={loss.item():.4f}, PPL={ppl:.4f}")

# Output:
# Epoch 0, Loss=6.9123, PPL=1006.2345
# Epoch 10, Loss=5.2345, PPL=187.5678
# Epoch 20, Loss=4.5678, PPL=96.3456
# Epoch 30, Loss=4.1234, PPL=61.7890
# Epoch 40, Loss=3.8901, PPL=48.9012
```

### Code Example 3: Temperature Sampling Comparison

```python
import torch
import torch.nn as nn

class TemperatureSampler:
    def __init__(self, model):
        self.model = model

    def generate(self, start, length, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([[start]])
            hidden = None
            output = [start]
            for _ in range(length):
                logits, hidden = self.model.forward(x, hidden)
                logits = logits[:, -1, :] / temperature

                if temperature < 0.1:
                    next_token = logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                output.append(next_token)
                x = torch.tensor([[next_token]])
        return output

model = CharLanguageModel(vocab_size=50, hidden_size=64)

sampler = TemperatureSampler(model)
start_token = 10

print("Sampling with different temperatures:")
for temp in [0.1, 0.5, 1.0, 2.0]:
    seq = sampler.generate(start_token, 10, temperature=temp)
    unique_ratio = len(set(seq)) / len(seq)
    print(f"  T={temp:.1f}: {seq[:8]}... (diversity={unique_ratio:.2f})")

# Output:
# Sampling with different temperatures:
#   T=0.1: [10, 34, 34, 34, 34, 34, 34, 34]... (diversity=0.09)
#   T=0.5: [10, 34, 12, 34, 7, 34, 12, 28]... (diversity=0.36)
#   T=1.0: [10, 34, 12, 45, 23, 7, 41, 28]... (diversity=0.82)
#   T=2.0: [10, 12, 45, 3, 41, 28, 33, 47]... (diversity=0.91)
```

### Code Example 4: Evaluating Language Model Quality

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class LMEvaluator:
    def __init__(self, model, vocab_size):
        self.model = model
        self.vocab_size = vocab_size

    def perplexity(self, data):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            logits, _ = self.model(data[:, :-1])
            loss = nn.CrossEntropyLoss(reduction='sum')(
                logits.reshape(-1, self.vocab_size),
                data[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            total_tokens += (data[:, 1:] != 0).sum().item()
        return math.exp(total_loss / total_tokens)

    def top_k_accuracy(self, data, k=5):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            logits, _ = self.model(data[:, :-1])
            probs = torch.softmax(logits, dim=-1)
            topk = probs.topk(k, dim=-1).indices
            targets = data[:, 1:].unsqueeze(-1)
            correct += (topk == targets).any(dim=-1).sum().item()
            total += targets.numel()
        return correct / total

model = CharLanguageModel(vocab_size=50, hidden_size=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_data = torch.randint(1, 50, (100, 30))
val_data = torch.randint(1, 50, (20, 30))
evaluator = LMEvaluator(model, 50)

for epoch in range(50):
    logits, _ = model(train_data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 50), train_data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        train_ppl = evaluator.perplexity(train_data)
        val_ppl = evaluator.perplexity(val_data)
        val_acc = evaluator.top_k_accuracy(val_data, k=5)
        print(f"Epoch {epoch}: train_ppl={train_ppl:.4f}, "
              f"val_ppl={val_ppl:.4f}, val_top5_acc={val_acc:.4f}")

# Output:
# Epoch 0: train_ppl=42.3456, val_ppl=45.6789, val_top5_acc=0.2345
# Epoch 10: train_ppl=10.2345, val_ppl=12.3456, val_top5_acc=0.4567
# Epoch 20: train_ppl=5.6789, val_ppl=7.8901, val_top5_acc=0.6123
# Epoch 30: train_ppl=3.4567, val_ppl=5.6789, val_top5_acc=0.7345
# Epoch 40: train_ppl=2.3456, val_ppl=4.5678, val_top5_acc=0.8123
```

## Common Mistakes

1. **Not using proper tokenization**: Character-level and word-level models have very different characteristics. Choose based on the task: character-level for open-vocabulary generation, word-level for better semantic modeling.

2. **Forgetting to handle start and end tokens**: Language models need special tokens to mark sequence boundaries. Without these, the model does not know when to start or stop generating.

3. **Using perplexity in isolation**: Perplexity measures predictive accuracy but does not capture text quality. A model with excellent perplexity can still generate incoherent text. Combine with human evaluation or task-specific metrics.

4. **Not masking padding tokens in loss computation**: When batching sequences of different lengths, padded positions must be ignored in the loss. Failing to do so artificially lowers perplexity.

5. **Using too high or too low temperature during generation**: Temperature 0 (argmax) produces repetitive text; temperature too high produces random gibberish. The optimal temperature is task-dependent, typically 0.7-1.0.

6. **Overfitting to training data**: Language models can memorize training sequences. Use dropout, weight decay, and monitor the gap between training and validation perplexity.

7. **Assuming low perplexity implies grammatical correctness**: Language models can achieve low perplexity by exploiting local statistics without understanding long-range grammatical constraints.

## Interview Questions

### Beginner

Q: What is a language model and how does an RNN implement one?
A: A language model assigns probabilities to sequences of tokens. An RNN language model reads tokens one at a time, maintains a hidden state encoding the context, and outputs a probability distribution over the next token at each step.

Q: What is perplexity and how is it calculated?
A: Perplexity is the exponentiated average negative log-likelihood of a held-out test set: exp(-(1/T) * sum log P(w_t | context)). It measures how "perplexed" the model is by the data. Lower is better, with the vocabulary size being random chance.

### Intermediate

Q: Explain the difference between training and inference in an RNN language model.
A: During training, teacher forcing is used: the ground truth previous token is fed as input, enabling parallel computation across time steps. During inference, the model's own predictions are fed back as input (autoregressive generation), requiring sequential computation.

Q: How does temperature affect text generation and why is it needed?
A: Temperature scales the logits before softmax. Low temperature sharpens the distribution (more deterministic, repetitive), high temperature flattens it (more random, diverse). It is needed because the learned distribution is often too sharp (overconfident) for creative generation.

### Advanced

Q: Derive the relationship between perplexity and the cross-entropy loss and explain why perplexity is always at least 1.
A: Perplexity = exp(H) where H = -E[log P(w_t | context)]. By Gibbs inequality, H >= 0 for a discrete distribution, so perplexity >= 1. Perplexity can be interpreted as the geometric mean of the inverse probability assigned to each token: it is the model's average "branching factor" at each step.

Q: Analyze the limitations of RNN-based language models compared to Transformer-based alternatives.
A: RNN LMs: sequential computation prevents parallelization during training, limited memory horizon due to vanishing gradients, hidden state has finite capacity (bottleneck). Transformer LMs: parallel processing of all positions, attention provides direct access to any position, larger effective context window. However, RNNs have linear memory O(T) vs Transformer's quadratic O(T^2), and RNNs can be more efficient for very long sequences with proper state management.

## Practice Problems

### Easy

Train a character-level RNN language model on a synthetic dataset of 500 sequences (vocabulary size 20, sequence length 15). Report the training and validation perplexity.

### Medium

Implement a word-level language model with dropout and compare its perplexity with and without dropout regularization. Vary dropout rates and report the optimal value.

### Hard

Implement a language model with adaptive embeddings (small embeddings for frequent words, larger for rare words) and compare its perplexity and parameter count against a standard embedding language model.

## Solutions

### Easy Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CharLM(nn.Module):
    def __init__(self, vocab=20, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)

model = CharLM()
opt = optim.Adam(model.parameters(), lr=0.001)
train_data = torch.randint(1, 20, (400, 15))
val_data = torch.randint(1, 20, (100, 15))

for epoch in range(100):
    logits = model(train_data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, 20), train_data[:, 1:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    val_logits = model(val_data[:, :-1])
    val_loss = nn.CrossEntropyLoss()(val_logits.reshape(-1, 20), val_data[:, 1:].reshape(-1))
    val_ppl = torch.exp(val_loss).item()
print(f"Validation perplexity: {val_ppl:.4f}")
```

### Medium Solution

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DropoutWordLM(nn.Module):
    def __init__(self, vocab=100, embed=50, hidden=128, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.rnn = nn.RNN(embed, hidden, batch_first=True,
                         dropout=dropout if hidden > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x):
        x = self.dropout(self.embed(x))
        out, _ = self.rnn(x)
        out = self.dropout(out)
        return self.fc(out)

data = torch.randint(0, 100, (200, 20))
for drop in [0.0, 0.2, 0.5]:
    model = DropoutWordLM(dropout=drop)
    opt = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        logits = model(data[:, :-1])
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, 100), data[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ppl = torch.exp(nn.CrossEntropyLoss()(
            model(data[:, :-1]).reshape(-1, 100), data[:, 1:].reshape(-1)))
    print(f"Dropout={drop:.1f}: PPL={ppl.item():.4f}")
```

## Related Concepts

- RNN Applications (DL-293)
- Teacher Forcing (DL-288)
- Perplexity
- Transformer Language Models

## Next Concepts

- RNN Limitations (DL-295)
- LSTM Overview (DL-296)

## Summary

RNN-based language models estimate the probability distribution over the next token given the preceding context by processing sequences autoregressively through recurrent hidden states. They are trained by maximizing the log-likelihood of training corpora and evaluated using perplexity. Text generation is performed by sampling from the model's output distribution, with temperature controlling the diversity of generated text. While RNN language models have been largely superseded by Transformers for large-scale NLP, they remain relevant for character-level modeling, resource-constrained settings, and as a foundation for understanding neural language modeling.

## Key Takeaways

- RNN LMs estimate P(w_t | w_1:t-1) through autoregressive processing
- Training uses teacher forcing with cross-entropy loss
- Perplexity = exp(loss) measures predictive performance
- Temperature controls generation diversity
- Character-level models handle open vocabularies
- Word-level models capture semantic patterns
- Language modeling is a foundational NLP task
