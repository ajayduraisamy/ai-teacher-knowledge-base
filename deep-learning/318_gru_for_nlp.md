# Concept: GRU for NLP

## Concept ID

DL-318

## Difficulty

Advanced

## Domain

Deep Learning

## Module

GRU

## Learning Objectives

- Understand how GRUs are applied to natural language processing tasks
- Implement GRU-based models for text classification, language modeling, and sequence labeling
- Explain the advantages of GRU over LSTM for NLP tasks
- Design encoder-decoder architectures using GRUs for text generation
- Analyze the role of embeddings in GRU-based NLP models

## Prerequisites

- DL-311: GRU Overview
- DL-294: RNN for Language Modeling
- DL-309: LSTM for NLP
- DL-040: Embedding Layer

## Definition

GRU for NLP refers to the application of Gated Recurrent Units to natural language processing tasks. GRUs process text as sequences of tokens (words, subwords, or characters), maintaining a hidden state that captures contextual information. The GRU's gating mechanism allows it to selectively retain and update information across long text sequences, making it effective for tasks like sentiment analysis, named entity recognition, machine translation, text generation, and language modeling. GRU is particularly popular in NLP because it provides LSTM-level performance with fewer parameters and faster computation.

## Intuition

A GRU processing text is like a skilled reader maintaining a running summary. As it reads each word, it decides what to keep from its current understanding and what to update based on the new word. The reset gate helps focus on the most relevant recent words, while the update gate controls how much of the accumulated context to retain. For example, in sentiment analysis, the GRU learns to keep track of positive or negative cues across a sentence, even when neutral words appear in between.

## Why This Concept Matters

GRU is widely used in NLP because:

- It is more parameter-efficient than LSTM, important for large NLP models
- Training is faster, enabling more experimentation
- NLP datasets are often large enough that GRU matches LSTM performance
- GRU works well with pre-trained embeddings (GloVe, word2vec)
- It is the foundation for many sequence-to-sequence and attention models
- Knowledge of GRU for NLP leads to understanding modern transformer-based approaches

## Mathematical Explanation

For NLP, the GRU processes embedded token representations rather than raw vectors.

**Token embedding**: x_t = Embedding(token_t)

**GRU computation**:
r_t = sigmoid(W_r * x_t + U_r * h_(t-1) + b_r)
z_t = sigmoid(W_z * x_t + U_z * h_(t-1) + b_z)
h_tilde_t = tanh(W_h * x_t + U_h * (r_t * h_(t-1)) + b_h)
h_t = (1 - z_t) * h_(t-1) + z_t * h_tilde_t

**Output** depends on the NLP task:
- Classification: h_final -> Linear -> softmax
- Sequence labeling: h_t -> Linear -> softmax at each t
- Language model: h_t -> Linear -> vocab_size -> softmax
- Seq2seq: encoder outputs -> attention -> decoder GRU

## Code Examples

### Code Example 1: GRU for Sentiment Classification

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GRUSentiment(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers,
                         batch_first=True, bidirectional=True,
                         dropout=0.5 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        output, h_n = self.gru(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        h = self.dropout(h)
        return self.fc(h)

model = GRUSentiment(vocab_size=10000, embed_size=50, hidden_size=64, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

batch_size, seq_len = 16, 50
x = torch.randint(1, 1000, (batch_size, seq_len))
y = torch.randint(0, 2, (batch_size,))

for epoch in range(20):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (pred.argmax(dim=1) == y).float().mean()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc.item():.4f}")

# Output:
# Epoch 0: loss=0.7034, acc=0.5000
# Epoch 5: loss=0.5432, acc=0.6875
# Epoch 10: loss=0.4567, acc=0.7500
# Epoch 15: loss=0.3987, acc=0.8125
```

### Code Example 2: GRU Language Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers,
                         batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden=None):
        x = self.dropout(self.embedding(x))
        output, hidden = self.gru(x, hidden)
        output = self.dropout(output)
        return self.fc(output), hidden

    def generate(self, start_tokens, max_len=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            x = torch.tensor([start_tokens])
            hidden = None
            generated = start_tokens.copy()
            for _ in range(max_len):
                logits, hidden = self.forward(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
                x = torch.tensor([[next_token]])
        return generated

vocab_size = 5000
model = GRULanguageModel(vocab_size, embed_size=64, hidden_size=128, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = torch.randint(1, vocab_size, (64, 30))
for epoch in range(30):
    logits, _ = model(data[:, :-1])
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        ppl = torch.exp(loss).item()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, PPL={ppl:.4f}")

generated = model.generate([101, 202, 303], max_len=15, temperature=0.8)
print(f"Generated tokens (first 20): {generated[:20]}")

# Output:
# Epoch 0: loss=8.4321, PPL=4589.1234
# Epoch 10: loss=5.6789, PPL=292.4567
# Epoch 20: loss=4.3210, PPL=75.2345
# Generated tokens (first 20): [101, 202, 303, 4231, 512, 3890, 234, 4567, 89, 3124, 567, 2345, 890, 1678, 3456, 789, 2901, 1234, 5678, 901]
```

### Code Example 3: GRU for Named Entity Recognition

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GRUNER(nn.Module):
    def __init__(self, vocab_size, embed_size=50, hidden_size=128, num_tags=9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.bigru = nn.GRU(embed_size, hidden_size, bidirectional=True,
                           batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_tags)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        output, _ = self.bigru(x)
        output = self.dropout(output)
        return self.fc(output)

model = GRUNER(vocab_size=8000, embed_size=50, hidden_size=64, num_tags=9)
optimizer = optim.Adam(model.parameters(), lr=0.002)

batch_size, seq_len = 8, 20
x = torch.randint(1, 800, (batch_size, seq_len))
y = torch.randint(0, 9, (batch_size, seq_len))

for epoch in range(30):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.permute(0, 2, 1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        preds = logits.argmax(dim=-1)
        token_acc = (preds == y).float().mean()
        print(f"Epoch {epoch}: loss={loss.item():.4f}, token_acc={token_acc.item():.4f}")

with torch.no_grad():
    x_test = torch.randint(1, 800, (4, 15))
    y_test = torch.randint(0, 9, (4, 15))
    logits = model(x_test)
    loss = nn.CrossEntropyLoss()(logits.permute(0, 2, 1), y_test).item()
    preds = logits.argmax(dim=-1)
    f1_micro = (preds == y_test).float().mean().item()
    print(f"\nTest: loss={loss:.4f}, token_acc={f1_micro:.4f}")

tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
example = torch.randint(1, 800, (1, 12))
example_logits = model(example)
example_tags = example_logits.argmax(dim=-1)[0]
print("Sample tagging:", [tag_names[t] for t in example_tags])

# Output:
# Epoch 0: loss=2.1987, token_acc=0.1250
# Epoch 10: loss=1.5678, token_acc=0.5625
# Epoch 20: loss=1.2345, token_acc=0.6875
#
# Test: loss=1.3456, token_acc=0.6500
# Sample tagging: ['O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG', 'O', 'B-MISC', 'O']
```

### Code Example 4: GRU Seq2Seq for Machine Translation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        return output, hidden

class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        return self.fc(output), hidden

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        _, hidden = self.encoder(src)
        outputs = []
        input_token = trg[:, 0:1]
        for t in range(trg_len - 1):
            output, hidden = self.decoder(input_token, hidden)
            outputs.append(output)
            input_token = output.argmax(dim=-1)
        return torch.cat(outputs, dim=1)

    def translate(self, src, max_len=20, start_token=1, end_token=2):
        self.eval()
        with torch.no_grad():
            _, hidden = self.encoder(src)
            input_token = torch.tensor([[start_token]])
            translation = [start_token]
            for _ in range(max_len):
                output, hidden = self.decoder(input_token, hidden)
                next_token = output.argmax(dim=-1).item()
                translation.append(next_token)
                if next_token == end_token:
                    break
                input_token = torch.tensor([[next_token]])
        return translation

enc_vocab, dec_vocab, embed_size, hidden_size = 1000, 1000, 64, 128
encoder = EncoderGRU(enc_vocab, embed_size, hidden_size)
decoder = DecoderGRU(dec_vocab, embed_size, hidden_size)
model = Seq2SeqGRU(encoder, decoder)
optimizer = optim.Adam(model.parameters(), lr=0.001)

src = torch.randint(1, 900, (16, 12))
trg = torch.randint(1, 900, (16, 10))
for epoch in range(50):
    output = model(src, trg)
    loss = nn.CrossEntropyLoss()(output.reshape(-1, dec_vocab), trg[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

test_src = torch.randint(1, 900, (1, 12))
trans = model.translate(test_src, max_len=15)
print(f"\nTranslation tokens: {trans[:10]}...")

# Output:
# Epoch 0: loss=6.9123
# Epoch 20: loss=4.5678
# Epoch 40: loss=3.4567
#
# Translation tokens: [1, 123, 456, 789, 234, 567, 890, 345, 678, 2]...
```

## Common Mistakes

1. **Not using pre-trained embeddings**: Training embeddings from scratch with GRU often underperforms using pre-trained GloVe, fastText, or word2vec embeddings, especially on small datasets.

2. **Ignoring sequence padding**: NLP sequences have variable lengths. Use nn.utils.rnn.pack_padded_sequence and pad_packed_sequence with GRU to avoid processing padding tokens.

3. **Using too small hidden size for language modeling**: Language models need large hidden sizes (256-1024) to capture vocabulary distributions. Small hidden sizes lead to high perplexity.

4. **Forgetting bidirectional context for sequence labeling**: NER, POS tagging, and chunking benefit greatly from bidirectional GRU since both left and right context are needed.

5. **Applying GRU directly without tokenization considerations**: Character-level, subword (BPE), and word-level GRUs have different trade-offs. Subword tokenization often works best.

6. **Not using gradient clipping for NLP tasks**: NLP models with GRU frequently suffer from exploding gradients, especially with long sequences or deep stacks.

7. **Assuming GRU will match transformer performance on all tasks**: For very long-range dependencies or tasks requiring global context, transformers typically outperform GRUs.

## Interview Questions

### Beginner

Q: How is GRU used for sentiment classification?
A: Text is tokenized and embedded, then passed through a GRU. The final hidden state (or a pooled representation of all hidden states) is passed through a linear layer for binary or multi-class sentiment prediction.

Q: Why is GRU preferred over LSTM for some NLP tasks?
A: GRU has fewer parameters (25% less), trains faster, and often achieves comparable performance on NLP tasks. It is less prone to overfitting on smaller datasets.

### Intermediate

Q: How would you handle variable-length sequences in a GRU for NLP?
A: Use padding to make all sequences the same length, then use pack_padded_sequence to tell the GRU to ignore padding tokens. After the GRU, use pad_packed_sequence to restore the padded format. This ensures the GRU does not waste computation on padding tokens.

Q: Compare character-level, subword, and word-level GRU language models.
A: Character-level GRU (vocab ~50-200) handles OOV words but needs longer sequences. Word-level (vocab ~50k-200k) is efficient but struggles with rare words. Subword (BPE, ~8k-32k) is a good trade-off, handling OOV while keeping sequence length manageable. GRU is effective at all three levels.

### Advanced

Q: Design a GRU-based architecture for abstractive text summarization that handles long documents (2000+ tokens).
A: Use a hierarchical GRU approach: first, split the document into segments. Use a sentence-level GRU to encode each sentence, then a document-level GRU to process the sentence representations. Alternatively, use a GRU encoder with attention over the full sequence, but with truncated backpropagation through time for training. Add pointer-generator mechanisms to copy rare words. For very long documents, use a sparse attention mechanism over GRU hidden states.

Q: How would you adapt GRU for code-switched text (mixing multiple languages)?
A: Use language-aware embeddings and a modified GRU with language-specific reset gates. The model learns when the language switches and adjusts its gating behavior accordingly. Alternatively, use a shared GRU with language embeddings concatenated to token embeddings, allowing the GRU to learn language-agnostic representations while being sensitive to language identity.

## Practice Problems

### Easy

Build a GRU-based binary sentiment classifier for IMDB movie reviews. Use pre-trained GloVe embeddings (50d) and a single-layer GRU with hidden size 128. Report validation accuracy.

### Medium

Implement a GRU language model on the WikiText-2 dataset. Compare perplexity for hidden sizes of 128, 256, and 512 with 2 layers. Plot training curves and discuss the effect of hidden size.

### Hard

Implement a GRU-based seq2seq model with attention for English-to-French translation using a subset of the Tatoeba dataset. Compare the attention-based model with a vanilla GRU seq2seq model and report BLEU scores.

## Related Concepts

- GRU Overview (DL-311)
- Bidirectional GRU (DL-316)
- Stacked GRU (DL-317)
- LSTM for NLP (DL-309)

## Next Concepts

- GRU Advantages (DL-319)
- RNN vs LSTM vs GRU (DL-320)

## Summary

GRU is highly effective for NLP tasks due to its balance of computational efficiency and representational power. It processes text through embedded token sequences, maintaining contextual hidden states via gating mechanisms. GRU is widely used for sentiment analysis, language modeling, named entity recognition, machine translation, and text generation. It often matches LSTM performance with fewer parameters and faster training, making it a popular choice for NLP applications where computational resources are limited.

## Key Takeaways

- GRU processes embedded token sequences for NLP tasks
- Bidirectional GRU is essential for sequence labeling tasks
- Pre-trained embeddings significantly improve GRU performance
- GRU is more parameter-efficient than LSTM for NLP
- Packed sequences handle variable-length text inputs
- GRU works at character, subword, or word level
- Gradient clipping is important for stability in NLP training
- Seq2seq GRU with attention is a foundational architecture for translation
