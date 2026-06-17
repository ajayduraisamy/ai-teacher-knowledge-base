# Concept: Sequence to Sequence Architecture

## Concept ID

DL-321

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the fundamental architecture of sequence-to-sequence models and their role in mapping variable-length input sequences to variable-length output sequences.
- Differentiate between the encoder and decoder components and their respective responsibilities in the seq2seq pipeline.
- Analyze how seq2seq models handle sequential data of arbitrary length through recurrent connections and hidden state propagation.
- Evaluate the strengths and limitations of seq2seq architectures in real-world applications like machine translation, text summarization, and speech recognition.
- Implement a basic seq2seq model in PyTorch and interpret its behavior on sample sequence transformation tasks.

## Prerequisites

- Proficiency in recurrent neural networks (RNNs), including vanishing gradient problems and backpropagation through time.
- Understanding of word embeddings and tokenization techniques for natural language processing.
- Familiarity with PyTorch tensor operations, including `nn.RNN`, `nn.LSTM`, and `nn.GRU` modules.
- Basic knowledge of language modeling and next-token prediction as a foundation for decoding strategies.

## Definition

A Sequence-to-Sequence (Seq2Seq) architecture is a deep learning framework designed to transform an input sequence of arbitrary length into an output sequence of arbitrary length. First popularized by Sutskever, Vinyals, and Le in 2014 for machine translation, the architecture consists of two recurrent neural networks: an encoder that reads the input sequence and compresses it into a fixed-length context vector, and a decoder that generates the output sequence token by token conditioned on this context vector. The key innovation of seq2seq is its ability to handle variable-length inputs and outputs without requiring aligned data, making it widely applicable to tasks such as translation, summarization, dialogue generation, and image captioning.

## Intuition

Imagine you are tasked with translating an English sentence into French. As a human, you first read the entire English sentence to understand its full meaning before you start speaking the French translation. This is exactly what the seq2seq architecture does. The encoder acts like the reading phase: it processes each word of the input sequentially, updating its internal memory (hidden state) along the way. By the time it finishes reading the last word, it has formed an internal representation of the entire sentence's meaning encapsulated in a context vector. The decoder then acts as the speaking phase: starting from a special start-of-sequence token, it uses the context vector to generate the first French word, then feeds that word back as input to generate the next word, and continues until it produces an end-of-sequence token. This two-phase process allows the model to internalize the full input before beginning to generate output, which is essential for tasks where the relationship between input and output tokens is not monotonic or one-to-one.

## Why This Concept Matters

Sequence-to-sequence architectures represent a paradigm shift in how neural networks handle sequential data. Before seq2seq models, most neural sequence processing approaches required inputs and outputs to have fixed sizes or strict alignment. Seq2seq removed these constraints, enabling end-to-end training for complex sequence transduction tasks. This breakthrough directly enabled modern machine translation systems like Google Translate's Neural Machine Translation (NMT), which replaced decades of statistical machine translation research with a single unified neural model. Beyond translation, seq2seq laid the foundation for attention mechanisms, transformer architectures, and virtually every modern language model. Understanding seq2seq is essential because it introduces the core encoder-decoder abstraction that underpins all modern generative sequence models, from BART and T5 to GPT-style architectures. Mastering this concept provides the conceptual bridge between simple recurrent networks and the sophisticated attention-based models that dominate contemporary NLP.

## Mathematical Explanation

Let an input sequence be denoted as \( X = (x_1, x_2, \ldots, x_T) \) where each \( x_t \) is a token from a vocabulary at timestep \( t \). Similarly, the output sequence is \( Y = (y_1, y_2, \ldots, y_{T'}) \). Note that \( T \) and \( T' \) may differ.

### Encoder

The encoder is an RNN that reads each input token sequentially. Let \( h_t^{\text{enc}} \) be the encoder's hidden state at timestep \( t \). For a simple RNN encoder:

\[
h_t^{\text{enc}} = f(W_{xh} x_t + W_{hh} h_{t-1}^{\text{enc}} + b_h)
\]

where \( f \) is a non-linear activation function (typically tanh or ReLU), \( W_{xh} \) is the input-to-hidden weight matrix, \( W_{hh} \) is the hidden-to-hidden weight matrix, and \( b_h \) is the bias term. For LSTM encoders, we maintain both a hidden state \( h_t \) and a cell state \( c_t \).

After processing the entire input sequence, the encoder's final hidden state \( h_T^{\text{enc}} \) (and cell state for LSTM) is used to initialize the decoder. This final state serves as the context vector \( c \):

\[
c = h_T^{\text{enc}}
\]

### Decoder

The decoder is another RNN that generates the output sequence one token at a time. It is initialized with the context vector from the encoder:

\[
h_0^{\text{dec}} = c
\]

For each output timestep \( t' \), the decoder computes the next hidden state conditioned on the previous target token \( y_{t'-1} \) and the previous hidden state \( h_{t'-1}^{\text{dec}} \):

\[
h_{t'}^{\text{dec}} = g(W_{yh} y_{t'-1} + W_{hh} h_{t'-1}^{\text{dec}} + b_h)
\]

The output distribution over the target vocabulary at timestep \( t' \) is then:

\[
P(y_{t'} \mid y_{<t'}, X) = \text{softmax}(W_{hy} h_{t'}^{\text{dec}} + b_y)
\]

The model is trained to maximize the conditional probability of the target sequence given the input sequence:

\[
P(Y \mid X) = \prod_{t'=1}^{T'} P(y_{t'} \mid y_{<t'}, X)
\]

### Training Objective

The training loss is the negative log-likelihood of the target sequence:

\[
\mathcal{L} = -\sum_{t'=1}^{T'} \log P(y_{t'} \mid y_{<t'}, X)
\]

During training, teacher forcing is used where the ground truth token \( y_{t'-1} \) is fed as input to the decoder at step \( t' \), rather than the model's own prediction from the previous step.

## Code Examples

### Example 1: Basic Seq2Seq with Simple RNN

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return hidden

class DecoderRNN(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# Example usage
INPUT_DIM = 100
OUTPUT_DIM = 100
EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 1
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = EncoderRNN(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = DecoderRNN(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

src_tensor = torch.randint(0, 100, (4, 7))
trg_tensor = torch.randint(0, 100, (4, 5))
output = model(src_tensor, trg_tensor, 1.0)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([4, 5, 100])
```

### Example 2: LSTM-based Seq2Seq

```python
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

INPUT_DIM = 100
OUTPUT_DIM = 100
EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 2
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = LSTMEncoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = LSTMDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = LSTMSeq2Seq(enc, dec, DEVICE).to(DEVICE)

src = torch.randint(0, 100, (2, 10))
trg = torch.randint(0, 100, (2, 8))
out = model(src, trg, 0.8)
print(f"LSTM Seq2Seq output shape: {out.shape}")
# Output: LSTM Seq2Seq output shape: torch.Size([2, 8, 100])
```

### Example 3: Seq2Seq Inference (Greedy Decoding)

```python
import torch
import torch.nn.functional as F

def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        trg_indices = [torch.full((batch_size,), sos_idx, dtype=torch.long).to(device)]
        for _ in range(max_len):
            input_token = trg_indices[-1]
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            probs = F.softmax(output, dim=-1)
            next_token = probs.argmax(1)
            trg_indices.append(next_token)
            if (next_token == eos_idx).all():
                break
        trg_indices = torch.stack(trg_indices, dim=1)
    return trg_indices

# Usage
model = LSTMSeq2Seq(
    LSTMEncoder(100, 32, 64, 2, 0.2),
    LSTMDecoder(100, 32, 64, 2, 0.2),
    DEVICE
).to(DEVICE)

src = torch.randint(0, 100, (1, 7))
decoded = greedy_decode(model, src, max_len=20, sos_idx=1, eos_idx=2, device=DEVICE)
print(f"Decoded sequence shape: {decoded.shape}")
print(f"Decoded tokens: {decoded.tolist()}")
# Output: Decoded sequence shape: torch.Size([1, 5])
# Output: Decoded tokens: [[1, 23, 67, 45, 2]]
```

## Common Mistakes

1. **Ignoring the vanishing gradient problem**: Using simple RNNs for the encoder or decoder without addressing the vanishing gradient problem through LSTM or GRU cells leads to poor long-range dependency modeling. Gradients during backpropagation through time diminish exponentially, making it impossible to learn relationships between distant tokens.

2. **Improper initialization of decoder hidden state**: Passing the encoder's final hidden state directly to the decoder without considering different layer counts or dimensionality mismatches. If the encoder has 2 layers and the decoder expects 2 layers, the shapes must match exactly, or truncation/padding errors occur silently.

3. **Forgetting to handle variable-length sequences**: Batches with sequences of different lengths require careful padding and masking. Without proper masking of padding tokens in the loss computation, the model learns to predict padding tokens as valid output, degrading performance.

4. **Using teacher forcing at inference time**: Teacher forcing is a training-only technique. During inference, the model must use its own predictions as input for the next timestep. Using ground truth during inference (cheating) creates a mismatch between training and deployment called exposure bias.

5. **Training a deep seq2seq without skip connections or gradient clipping**: Stacking multiple RNN layers increases representational capacity but also exacerbates gradient instability. Without gradient clipping (typically max norm of 1.0 or 5.0), training often diverges due to exploding gradients.

## Interview Questions

### Beginner

Q: What are the two main components of a Seq2Seq model?

A: The two main components are the encoder and the decoder. The encoder processes the input sequence and compresses it into a context vector, while the decoder uses this context vector to generate the output sequence token by token.

### Intermediate

Q: Why is teacher forcing used during Seq2Seq training, and what problem does it address?

A: Teacher forcing addresses the problem of slow convergence and unstable training in early stages. By feeding the ground truth token as input to the decoder at each timestep (rather than the model's own prediction), teacher forcing provides a strong learning signal that helps the decoder learn the conditional distribution more efficiently. However, it creates exposure bias because at inference time, the model must rely on its own potentially incorrect predictions, which compounds over time.

### Advanced

Q: Explain the concept of exposure bias in Seq2Seq models and describe at least two strategies to mitigate it.

A: Exposure bias refers to the mismatch between training and inference conditions in autoregressive models like Seq2Seq. During training, teacher forcing provides the ground truth token at each step, but during inference, the model uses its own (potentially erroneous) previous predictions. Errors compound, leading to distribution shift. Mitigation strategies include: (1) Scheduled sampling, where teacher forcing probability decays over training time, gradually exposing the model to its own errors; (2) Professor forcing, which uses adversarial training to make the hidden state distributions of the free-running and teacher-forced modes indistinguishable; (3) Reinforcement learning approaches like Minimum Risk Training (MRT) that directly optimize sequence-level evaluation metrics.

## Practice Problems

### Easy

Implement a simple Seq2Seq model with GRU cells instead of LSTM or vanilla RNN. Test it on a toy copy task where the model must output an identical sequence to the input. Use sequences of length 5 with a vocabulary size of 20.

### Medium

Modify the Seq2Seq model to support bidirectional encoder. The hidden state of a bidirectional RNN consists of a forward and backward pass. Explain how to combine these two states to initialize the decoder and implement the modification in PyTorch.

### Hard

Implement scheduled sampling for Seq2Seq training. Create a training loop where the teacher forcing ratio decays linearly from 1.0 to 0.0 over the course of training. Compare the performance against constant teacher forcing on a simple English-to-French translation dataset.

## Solutions

### Easy Solution

```python
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.gru(embedded)
        return hidden

class GRUDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden
```

### Medium Solution

```python
class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)
        hidden = self.fc_hidden(hidden).unsqueeze(0)
        cell = self.fc_cell(cell).unsqueeze(0)
        return hidden, cell
```

## Related Concepts

- Recurrent Neural Networks (RNN, LSTM, GRU)
- Word Embeddings (Word2Vec, GloVe, FastText)
- Teacher Forcing and Scheduled Sampling
- Backpropagation Through Time (BPTT)
- Neural Machine Translation (NMT)

## Next Concepts

- DL-322: Encoder-Decoder Framework
- DL-323: Context Vector
- DL-324: Encoder RNN
- DL-335: Seq2Seq with Attention

## Summary

The Sequence-to-Sequence architecture is a foundational deep learning framework for mapping variable-length input sequences to variable-length output sequences. It consists of an encoder RNN that compresses the input into a context vector and a decoder RNN that generates the output token by token. The architecture revolutionized machine translation and natural language generation by enabling end-to-end training for sequence transduction tasks. While simple RNN-based implementations suffer from vanishing gradients and fixed-context bottlenecks, the encoder-decoder abstraction established the paradigm that later evolved into attention mechanisms and transformer models. Understanding seq2seq is critical for grasping the evolution of modern deep learning architectures for sequential data.

## Key Takeaways

- Seq2Seq models map variable-length input sequences to variable-length output sequences using an encoder-decoder framework.
- The encoder compresses the entire input sequence into a fixed-dimensional context vector, which initializes the decoder.
- The decoder generates the output autoregressively, producing one token at a time conditioned on the context and previously generated tokens.
- Teacher forcing during training accelerates convergence but introduces exposure bias during inference.
- The seq2seq architecture, while superseded by transformers for many tasks, remains conceptually foundational for understanding modern generative sequence models.
