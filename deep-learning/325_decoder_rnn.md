# Concept: Decoder RNN

## Concept ID

DL-325

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the role and functioning of the decoder RNN in generating output sequences from a context representation.
- Differentiate between training (teacher forcing) and inference (autoregressive) modes of the decoder.
- Analyze the impact of decoder initialization strategies using the context vector.
- Implement decoder RNNs with various cell types and output projection layers in PyTorch.
- Recognize the autoregressive nature of the decoder and its implications for error propagation.

## Prerequisites

- Understanding of the encoder RNN and its role in producing context representations.
- Familiarity with the seq2seq architecture and the encoder-decoder framework.
- Experience with PyTorch's RNN, LSTM, and GRU modules and their input/output conventions.
- Knowledge of softmax classification and cross-entropy loss.

## Definition

The Decoder RNN is the component of a sequence-to-sequence model responsible for generating the output sequence token by token, conditioned on the context vector produced by the encoder. It is an autoregressive model: at each timestep, it takes the previously generated token as input, updates its hidden state, and produces a probability distribution over the output vocabulary from which the next token is sampled or selected. Formally, let the context vector from the encoder be \( c \). The decoder generates an output sequence \( Y = (y_1, y_2, \ldots, y_{T'}) \) by computing:

\[
s_0 = f_{\text{init}}(c)
\]
\[
s_{t'} = \text{RNN}_{\text{dec}}(y_{t'-1}, s_{t'-1})
\]
\[
P(y_{t'} \mid y_{<t'}, X) = \text{softmax}(W_o s_{t'} + b_o)
\]

where \( s_{t'} \) is the decoder's hidden state at timestep \( t' \), \( y_0 \) is a special start-of-sequence token (SOS), and generation stops when an end-of-sequence token (EOS) is produced or a maximum length is reached. During training, teacher forcing replaces the model's previous prediction with the ground truth token, while during inference, the model's own prediction is fed back as input.

## Intuition

Imagine you are writing a story based on a single prompt sentence (the context vector). You begin by writing the first word ("The"), then based on that word and the prompt, you write the next word ("quick"), and so on. Each word you write depends on the prompt and all the words you have written so far. This is exactly how the decoder RNN works: it starts with a special start token, uses the context vector to set its initial state of mind, and generates one word at a time, feeding each newly generated word back as input for the next step. The decoder must maintain an internal state that tracks what has been generated so far to ensure coherence — for example, it should not generate "the cat" and later "they" (plural) without a transition. During training, a kind teacher (teacher forcing) occasionally corrects the decoder by providing the right next word even when the decoder made a mistake, helping it learn faster. During testing, the decoder is on its own and must recover from its own mistakes, which can lead to cascading errors if it goes off-track early.

## Why This Concept Matters

The decoder RNN is the generative engine of the seq2seq model. Its design determines the quality, diversity, and fluency of the generated output. Understanding the decoder is essential because it introduces the autoregressive generation paradigm that underlies all modern language models, from GPT to LLaMA. The decoder's training dynamics — particularly the exposure bias introduced by teacher forcing — are a central challenge in sequence generation that motivates techniques like scheduled sampling, beam search, and reinforcement learning. The decoder also establishes the pattern for cross-attention (in later seq2seq with attention models) where the decoder queries encoder representations. Mastering the decoder RNN provides insight into how neural networks generate structured output sequences and why generation is fundamentally more difficult than classification.

## Mathematical Explanation

### Decoder Initialization

The decoder's initial hidden state \( s_0 \) is computed from the encoder's final hidden state (context vector). For a simple RNN decoder:

\[
s_0 = c
\]

For LSTM decoders, both the hidden state \( s_0 \) and cell state \( c_0^{\text{dec}} \) must be initialized. When the encoder is bidirectional, the context has dimension \( 2H \) while the decoder expects dimension \( H \), so a linear projection is required:

\[
s_0 = \tanh(W_s [\overrightarrow{h}_T ; \overleftarrow{h}_1] + b_s)
\]
\[
c_0^{\text{dec}} = \tanh(W_c [\overrightarrow{h}_T ; \overleftarrow{h}_1] + b_c)
\]

### Autoregressive Generation

At each timestep \( t' \), the decoder takes the previous token \( y_{t'-1} \) (embedded as \( e_{t'-1} \)) and computes:

\[
s_{t'} = f_{\text{dec}}(e_{t'-1}, s_{t'-1})
\]

where \( f_{\text{dec}} \) is the RNN cell function. The output distribution is:

\[
o_{t'} = W_o s_{t'} + b_o
\]
\[
P(y_{t'} = v \mid y_{<t'}, X) = \frac{\exp(o_{t'}[v])}{\sum_{k=1}^{V} \exp(o_{t'}[k])}
\]

where \( V \) is the vocabulary size.

### Teacher Forcing

During training, the input to the decoder at step \( t' \) is the ground truth token \( y_{t'-1}^* \) rather than the model's prediction:

\[
s_{t'} = f_{\text{dec}}(\text{embed}(y_{t'-1}^*), s_{t'-1})
\]

The loss is computed at each timestep:

\[
\mathcal{L}_{t'} = -\log P(y_{t'}^* \mid y_{<t'}^*, X)
\]

### Inference (Free-Running)

During inference, the model feeds its own prediction back:

\[
\hat{y}_{t'-1} = \arg\max_v P(y_{t'-1} = v \mid y_{<t'-1}, X)
\]
\[
s_{t'} = f_{\text{dec}}(\text{embed}(\hat{y}_{t'-1}), s_{t'-1})
\]

## Code Examples

### Example 1: Basic GRU Decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

decoder = GRUDecoder(output_dim=100, emb_dim=32, hid_dim=64)
hidden = torch.randn(1, 2, 64)
trg = torch.randint(0, 100, (2,))
pred, new_hidden = decoder(trg, hidden)
print(f"Prediction shape: {pred.shape}")
print(f"New hidden shape: {new_hidden.shape}")
# Output: Prediction shape: torch.Size([2, 100])
# Output: New hidden shape: torch.Size([1, 2, 64])
```

### Example 2: LSTM Decoder with Cell State

```python
import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

decoder = LSTMDecoder(output_dim=100, emb_dim=32, hid_dim=64)
hidden = torch.randn(1, 2, 64)
cell = torch.randn(1, 2, 64)
trg = torch.randint(0, 100, (2,))
pred, new_hidden, new_cell = decoder(trg, hidden, cell)
print(f"LSTM Prediction: {pred.shape}")
print(f"New hidden: {new_hidden.shape}, New cell: {new_cell.shape}")
# Output: LSTM Prediction: torch.Size([2, 100])
# Output: New hidden: torch.Size([1, 2, 64]), New cell: torch.Size([1, 2, 64])
```

### Example 3: Decoder with Projected Initialization from Bidirectional Encoder

```python
import torch
import torch.nn as nn

class ProjectedDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, enc_hid_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, hid_dim)

    def initialize_hidden(self, encoder_final_forward, encoder_final_backward):
        combined = torch.cat((encoder_final_forward, encoder_final_backward), dim=1)
        return torch.tanh(self.fc_hidden(combined)).unsqueeze(0)

    def forward(self, trg, hidden):
        trg = trg.unsqueeze(1)
        embedded = self.embedding(trg)
        output, hidden = self.gru(embedded, hidden)
        return self.fc_out(output.squeeze(1)), hidden

enc_forward = torch.randn(2, 64)
enc_backward = torch.randn(2, 64)
decoder = ProjectedDecoder(output_dim=100, emb_dim=32, hid_dim=64, enc_hid_dim=64)
hidden = decoder.initialize_hidden(enc_forward, enc_backward)
trg = torch.randint(0, 100, (2,))
pred, new_hidden = decoder(trg, hidden)
print(f"Projected decoder prediction: {pred.shape}")
# Output: Projected decoder prediction: torch.Size([2, 100])
```

## Common Mistakes

1. **Feeding the full sequence at once during inference**: During inference, the decoder must generate tokens one at a time in a loop. Feeding the entire sequence at once (as in teacher forcing) is not possible because the target sequence is unknown. Implementations must explicitly loop over timesteps.

2. **Not unsqueezing the token dimension**: The RNN expects input of shape `(batch, seq_len, emb_dim)`. When feeding a single token (seq_len=1), the input must be `unsqueeze(1)` to add the sequence dimension. Forgetting this causes shape mismatches.

3. **Mixing training and inference modes**: Using teacher forcing during inference (by feeding ground truth) gives unrealistically good performance. Conversely, using free-running during early training leads to slow convergence and instability. These modes must be carefully separated.

4. **Ignoring the start and end tokens**: The decoder must receive a SOS token as the first input to begin generation, and stop when EOS is produced or max length is reached. Omitting proper SOS/EOS handling leads to infinite loops or malformed sequences.

5. **Mismatched cell state initialization for LSTM decoders**: When the encoder is an LSTM, both hidden and cell states must be properly projected and passed to the decoder. Initializing only the hidden state and leaving the cell state as zeros can significantly harm performance.

## Interview Questions

### Beginner

Q: What is the difference between teacher forcing and free-running in a decoder?

A: In teacher forcing, the decoder receives the ground truth token as input at each timestep during training. In free-running (used during inference), the decoder receives its own previously generated token as input. Teacher forcing speeds up training but creates a mismatch with inference conditions.

### Intermediate

Q: Explain the exposure bias problem in decoder RNNs and why it is harmful.

A: Exposure bias is the mismatch between training (teacher forcing) and inference (free-running) conditions. During training, the decoder always receives correct previous tokens, so it never learns to recover from its own mistakes. During inference, an early error cascades because the decoder conditions on incorrect tokens it has never seen during training. This causes distribution shift and degrades output quality, especially for longer sequences.

### Advanced

Q: Describe at least three strategies to address exposure bias in decoder RNNs, including their strengths and weaknesses.

A: (1) Scheduled sampling: gradually anneal the teacher forcing ratio from 1.0 to 0.0 during training, exposing the model to its own errors. Weakness: it changes the training distribution and can cause instability. (2) Sequence-level objectives: train with reinforcement learning (e.g., policy gradient) using metrics like BLEU or ROUGE as rewards. Weakness: high variance gradients and expensive sampling. (3) Professor forcing: use a discriminative adversarial network to distinguish between teacher-forced and free-running hidden states, training the decoder to produce indistinguishable states. Weakness: complex two-player training dynamics. (4) DAGGER (Dataset Aggregation): collect training data from the model's own rollouts and retrain. Weakness: computationally expensive.

## Practice Problems

### Easy

Implement a GRU decoder that generates a sequence of length 10 given a context vector. Start with a SOS token and generate tokens one by one. Use greedy decoding (always pick the token with highest probability).

### Medium

Implement scheduled sampling for decoder training. The teacher forcing ratio should follow a linear decay schedule from 1.0 to 0.2 over 10 epochs. Compare the validation loss with constant teacher forcing at 1.0.

### Hard

Implement a decoder that uses a beam search of width 3 during inference. The decoder must maintain multiple hypotheses simultaneously and select the best one at the end. Include length normalization in the scoring.

## Solutions

### Easy Solution

```python
def greedy_generate(decoder, context, max_len, sos_idx, eos_idx):
    batch_size = context.shape[1]
    hidden = context
    input_token = torch.full((batch_size,), sos_idx, dtype=torch.long)
    generated = [input_token]
    for _ in range(max_len):
        pred, hidden = decoder(input_token, hidden)
        next_token = pred.argmax(dim=1)
        generated.append(next_token)
        input_token = next_token
        if (next_token == eos_idx).all():
            break
    return torch.stack(generated, dim=1)
```

## Related Concepts

- Autoregressive Models
- Teacher Forcing
- Exposure Bias
- Scheduled Sampling
- Greedy Decoding and Beam Search

## Next Concepts

- DL-326: Seq2Seq Training
- DL-327: Greedy Decoding
- DL-328: Beam Search

## Summary

The decoder RNN is the autoregressive generative component of sequence-to-sequence models. It generates output tokens one at a time, conditioned on the encoder's context vector and its own previously generated tokens. During training, teacher forcing provides ground truth inputs for faster convergence, but this introduces exposure bias that must be addressed for robust inference. The decoder's initialization from the encoder's context vector, its cell type, and its output projection are critical design choices. Understanding the decoder's autoregressive nature and the training-inference mismatch is essential for building effective sequence generation systems.

## Key Takeaways

- The decoder RNN generates output sequences autoregressively, one token at a time.
- Teacher forcing uses ground truth tokens during training; free-running uses model predictions during inference.
- Exposure bias arises from the mismatch between teacher-forced training and free-running inference.
- Proper initialization from the encoder context vector is critical, especially with bidirectional or LSTM encoders.
- SOS and EOS tokens must be handled correctly in both training and inference loops.
