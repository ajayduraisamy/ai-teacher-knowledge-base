# Concept: Encoder RNN

## Concept ID

DL-324

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the role of the encoder RNN in compressing an input sequence into a fixed-dimensional hidden representation.
- Differentiate between unidirectional and bidirectional encoder RNNs and explain the trade-offs of each.
- Analyze the impact of RNN cell type (vanilla RNN, LSTM, GRU) on encoder performance and gradient flow.
- Implement encoder RNNs with multiple layers, dropout, and bidirectional processing in PyTorch.
- Evaluate encoder design choices for different sequence transduction tasks.

## Prerequisites

- Solid understanding of recurrent neural networks and their mathematical formulation.
- Familiarity with backpropagation through time (BPTT) and the vanishing/exploding gradient problem.
- Experience with PyTorch's RNN, LSTM, and GRU modules.
- Knowledge of the seq2seq architecture and the encoder-decoder framework.

## Definition

The Encoder RNN is the component of a sequence-to-sequence model responsible for reading and encoding an input sequence into a sequence of hidden representations. It processes the input tokens one by one, updating its internal hidden state at each timestep to incorporate information from the current token along with the accumulated context from previous tokens. Formally, given an input sequence \( X = (x_1, x_2, \ldots, x_T) \), the encoder RNN computes a sequence of hidden states \( (h_1, h_2, \ldots, h_T) \) where each \( h_t \in \mathbb{R}^H \). These hidden states capture progressively more context as the sequence is processed. The encoder can be unidirectional (processing left to right), bidirectional (processing both left to right and right to left), or even multi-directional in more complex architectures. The final hidden state (or the full sequence of hidden states when attention is used) is passed to the decoder to condition output generation.

## Intuition

Think of the encoder RNN as a meticulous reader who reads a sentence word by word, maintaining a running summary in their memory. As they read each new word, they update their summary to incorporate the new information. By the time they reach the end of the sentence, their summary contains the gist of the entire sentence. If the reader reads only forward (unidirectional), their understanding of each word is influenced only by words that came before. But if they read forward and then backward (bidirectional), their understanding of each word benefits from both left and right context, just as a human reader might skim the whole sentence before settling on the meaning of an ambiguous word. The choice of memory mechanism matters too: a simple reader (vanilla RNN) forgets details from the beginning of long sentences, while a reader with a notebook (LSTM) can write down important details and retrieve them later. Stacking multiple readers (multi-layer RNN) allows the model to build hierarchical abstractions, with lower layers capturing local patterns and higher layers capturing global sentence structure.

## Why This Concept Matters

The encoder RNN is the foundation of representation learning in sequence-to-sequence models. Its design choices directly determine what information is preserved from the input and how it is structured. A well-designed encoder captures hierarchical structure, long-range dependencies, and bidirectional context that are essential for accurate sequence transduction. The encoder's output — whether a single context vector or a full sequence of hidden states — is the only source of information about the input that the decoder ever receives. Therefore, the encoder must be powerful enough to extract and represent all task-relevant information from the input, while being efficient enough to train. Understanding encoder RNN design is also the gateway to more advanced concepts: bidirectional processing informs masked language models, multi-layer architectures relate to deep transformers, and the encoder's role in producing token-level representations is the basis for encoder-decoder attention.

## Mathematical Explanation

### Unidirectional Encoder

Let \( x_t \in \mathbb{R}^V \) be a one-hot encoded input token at timestep \( t \), or more commonly, \( x_t \in \mathbb{R}^E \) be an embedding vector. For a simple RNN encoder:

\[
h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})
\]

For an LSTM encoder:

\[
i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
\]
\[
f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
\]
\[
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
\]
\[
o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
\]
\[
c_t = f_t \odot c_{t-1} + i_t \odot g_t
\]
\[
h_t = o_t \odot \tanh(c_t)
\]

For a GRU encoder:

\[
r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})
\]
\[
z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})
\]
\[
n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))
\]
\[
h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
\]

### Bidirectional Encoder

A bidirectional RNN processes the sequence in both directions using two independent RNNs. The forward RNN reads from \( x_1 \) to \( x_T \), producing \( (\overrightarrow{h}_1, \ldots, \overrightarrow{h}_T) \). The backward RNN reads from \( x_T \) to \( x_1 \), producing \( (\overleftarrow{h}_1, \ldots, \overleftarrow{h}_T) \). The final hidden state at each position is typically a concatenation:

\[
h_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t] \in \mathbb{R}^{2H}
\]

### Multi-Layer Encoder

A multi-layer (stacked) RNN stacks multiple RNN layers on top of each other. The hidden state of layer \( l \) at timestep \( t \) becomes the input to layer \( l+1 \) at the same timestep:

\[
h_t^{(1)} = \text{RNN}_1(x_t, h_{t-1}^{(1)})
\]
\[
h_t^{(2)} = \text{RNN}_2(h_t^{(1)}, h_{t-1}^{(2)})
\]
\[
\vdots
\]
\[
h_t^{(L)} = \text{RNN}_L(h_t^{(L-1)}, h_{t-1}^{(L)})
\]

## Code Examples

### Example 1: Single-Layer LSTM Encoder

```python
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

encoder = LSTMEncoder(vocab_size=100, emb_dim=32, hid_dim=64)
src = torch.randint(0, 100, (2, 7))
outputs, hidden, cell = encoder(src)
print(f"Encoder outputs shape: {outputs.shape}")
print(f"Hidden state shape: {hidden.shape}")
print(f"Cell state shape: {cell.shape}")
# Output: Encoder outputs shape: torch.Size([2, 7, 64])
# Output: Hidden state shape: torch.Size([1, 2, 64])
# Output: Cell state shape: torch.Size([1, 2, 64])
```

### Example 2: Bidirectional GRU Encoder

```python
import torch
import torch.nn as nn

class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.fc_hidden(hidden).unsqueeze(0)
        return outputs, hidden

encoder = BiGRUEncoder(vocab_size=100, emb_dim=32, hid_dim=64, n_layers=2)
src = torch.randint(0, 100, (2, 7))
outputs, hidden = encoder(src)
print(f"BiGRU outputs shape: {outputs.shape}")
print(f"Projected hidden shape: {hidden.shape}")
# Output: BiGRU outputs shape: torch.Size([2, 7, 128])
# Output: Projected hidden shape: torch.Size([1, 2, 64])
```

### Example 3: Deep Multi-Layer Encoder with Dropout

```python
import torch
import torch.nn as nn

class DeepEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

encoder = DeepEncoder(
    vocab_size=100, emb_dim=32,
    hid_dim=128, n_layers=4, dropout=0.3
)
src = torch.randint(0, 100, (2, 10))
outputs, hidden, cell = encoder(src)
print(f"Deep encoder outputs: {outputs.shape}")
print(f"Deep encoder hidden layers: {hidden.shape}")
# Output: Deep encoder outputs: torch.Size([2, 10, 256])
# Output: Deep encoder hidden layers: torch.Size([8, 2, 128])
```

## Common Mistakes

1. **Misinterpreting bidirectional hidden dimensions**: A bidirectional RNN with hidden dimension \( H \) produces outputs of dimension \( 2H \) at each timestep. The decoder must account for this when receiving the encoder's output, either by projecting down or by using a decoder with matching input dimension.

2. **Forgetting that multi-layer RNNs need dropout between layers**: PyTorch's `dropout` parameter in RNN/LSTM/GRU only applies dropout between layers when `num_layers > 1`. It does NOT apply dropout to the input or output. Additional `nn.Dropout` modules are needed for input/output regularization.

3. **Using the wrong hidden state index for context**: For bidirectional RNNs, `hidden[-2, :, :]` is the final forward hidden state and `hidden[-1, :, :]` is the final backward hidden state. Beginners often mix up the indexing, especially when using multi-layer bidirectional RNNs where the hidden tensor shape is `(num_layers * num_directions, batch, hid_dim)`.

4. **Ignoring sequence length when packing**: When batches contain sequences of varying lengths, the encoder should use `nn.utils.rnn.pack_padded_sequence` and `pad_packed_sequence`. Without packing, the encoder wastes computation on padding tokens and may learn spurious correlations.

5. **Overly deep encoders without proper regularization**: Stacking many RNN layers increases the risk of overfitting and gradient instability. Without sufficient dropout, gradient clipping, and possibly residual connections, deep encoders often perform worse than shallower alternatives.

## Interview Questions

### Beginner

Q: What is the purpose of the encoder RNN in a seq2seq model?

A: The encoder RNN processes the input sequence token by token, updating its hidden state to accumulate information about the entire sequence. It produces a set of hidden representations (one per token) that summarize the input, which are then used by the decoder to generate the output.

### Intermediate

Q: What are the advantages of a bidirectional encoder over a unidirectional one? When might you still prefer unidirectional?

A: A bidirectional encoder captures context from both left and right of each token, providing a richer representation, especially for tasks where full context is needed (e.g., translation, sentiment analysis). However, bidirectional encoders double the computation and memory cost, cannot be used for real-time streaming applications (where future tokens are unavailable), and are not suitable for language modeling where the goal is to predict the next token given only past tokens.

### Advanced

Q: How does the choice between LSTM and GRU for the encoder affect model performance and training dynamics? Explain the mathematical and empirical differences.

A: LSTMs have a separate cell state and three gates (input, forget, output), giving them more parameters and the ability to maintain a long-term memory that is decoupled from the hidden state. GRUs have two gates (reset and update) and merge the cell and hidden states, resulting in fewer parameters. Mathematically, the LSTM's forget gate allows it to reset its memory entirely, while the GRU's update gate interpolates between old and new states. Empirically, LSTMs often perform slightly better on tasks requiring very long-range dependencies, while GRUs train faster and require less data to generalize. For most seq2seq translation tasks, the differences are negligible with proper tuning, and GRUs are often preferred for their computational efficiency.

## Practice Problems

### Easy

Implement a single-layer unidirectional GRU encoder that takes sequences of token indices and returns the full sequence of encoder outputs and the final hidden state. Test it with a batch of 3 sequences of length 8, vocabulary size 50, embedding dimension 16, and hidden dimension 32.

### Medium

Implement a 2-layer bidirectional LSTM encoder. Include a linear projection layer that maps the concatenated forward/backward final hidden states to the decoder's hidden dimension. Explain why this projection is necessary.

### Hard

Implement an encoder that uses variable-length sequence handling with `pack_padded_sequence` and `pad_packed_sequence`. Create a DataLoader that provides sequences of varying lengths and measure the speed difference between packed and un-packed encoding.

## Solutions

### Easy Solution

```python
class SimpleGRUEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

encoder = SimpleGRUEncoder(50, 16, 32)
src = torch.randint(0, 50, (3, 8))
outputs, hidden = encoder(src)
print(f"Outputs: {outputs.shape}, Hidden: {hidden.shape}")
# Output: Outputs: torch.Size([3, 8, 32]), Hidden: torch.Size([1, 3, 32])
```

## Related Concepts

- Recurrent Neural Networks (RNN, LSTM, GRU)
- Bidirectional RNNs
- Stacked/Deep RNNs
- Backpropagation Through Time
- Gradient Clipping

## Next Concepts

- DL-325: Decoder RNN
- DL-326: Seq2Seq Training
- DL-335: Seq2Seq with Attention

## Summary

The encoder RNN is the input-processing component of sequence-to-sequence models. It reads the input token by token, generating a sequence of hidden states that capture increasingly rich contextual information. Key design choices include cell type (vanilla RNN, LSTM, GRU), directionality (unidirectional vs. bidirectional), and depth (single vs. multi-layer). Bidirectional encoders provide richer representations by capturing context from both directions but double computational cost. Multi-layer encoders build hierarchical representations but require careful regularization. The encoder's output—either the final hidden state or the full sequence of hidden states—serves as the information source for the decoder and is critical to overall seq2seq performance.

## Key Takeaways

- The encoder RNN compresses an input sequence into hidden representations by processing tokens sequentially.
- Bidirectional encoding captures context from both past and future tokens, improving representation quality.
- LSTM and GRU cells address the vanishing gradient problem better than vanilla RNNs.
- Multi-layer encoders build hierarchical representations but need dropout and gradient clipping.
- Proper handling of variable-length sequences via packing is essential for efficiency and correctness.
