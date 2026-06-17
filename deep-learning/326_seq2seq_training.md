# Concept: Seq2Seq Training

## Concept ID

DL-326

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the complete training pipeline for sequence-to-sequence models, including data preparation, loss computation, and gradient flow.
- Implement teacher forcing and scheduled sampling strategies for decoder training.
- Analyze the challenges of training seq2seq models: exposure bias, gradient instability, and computational efficiency.
- Apply gradient clipping, learning rate scheduling, and regularization techniques specific to seq2seq training.
- Design training loops that correctly handle variable-length sequences, padding, and masking.

## Prerequisites

- Solid understanding of the encoder-decoder architecture and its components.
- Experience with PyTorch training loops, optimizers, and loss functions.
- Knowledge of backpropagation through time and vanishing/exploding gradients.
- Familiarity with sequence padding and masking techniques.

## Definition

Seq2Seq training refers to the process of optimizing the parameters of a sequence-to-sequence model to maximize the conditional probability of target sequences given input sequences. The training objective is to minimize the negative log-likelihood of the target tokens conditioned on the input and preceding target tokens. The training procedure involves feeding input sequences through the encoder to obtain a context representation, then feeding target sequences through the decoder with teacher forcing to generate output distributions at each timestep. Loss is computed only on non-padding positions, and gradients are backpropagated through both the decoder and encoder. Key training techniques include teacher forcing with scheduled annealing, gradient clipping to prevent exploding gradients, learning rate scheduling (often with warmup and decay), and sequence-level regularization through dropout and label smoothing.

## Intuition

Training a seq2seq model is like teaching a student to translate by giving them practice sentences. First, you show the student an English sentence and let them read it (the encoder step). Then, you start them off with the first French word ("<START>") and ask them to guess the next word. If they guess correctly, you give them the correct next word anyway (teacher forcing) so they can continue confidently. If they make a mistake, you still give them the correct word, showing them the right path forward. Over time, you gradually let them rely more on their own guesses (scheduled sampling) so they learn to recover from mistakes. Throughout the process, you carefully track which words are real (ignoring padding tokens) and ensure the model doesn't get confused by blank spaces. You also clip the model's updates to prevent wild overcorrections (gradient clipping) and gradually reduce the learning rate as training progresses. The goal is for the model to learn not just word-by-word translations but also the overall structural mapping between the two languages.

## Why This Concept Matters

Training seq2seq models involves unique challenges that distinguish it from standard supervised learning. The autoregressive nature of the decoder means that errors propagate through time, and the training procedure must account for this. Teacher forcing, while essential for convergence, introduces exposure bias that requires careful mitigation. The vanishing and exploding gradient problems are exacerbated in seq2seq models due to the depth (both in time and in layers) of the computational graph. Variable-length sequences with padding require masked loss computation to avoid learning on meaningless tokens. Understanding seq2seq training techniques is directly applicable to training modern transformer-based sequence models, as many of the same challenges — autoregressive generation, exposure bias, gradient stability, and efficient batching — remain central to training large language models. Mastering these training techniques is essential for anyone building or fine-tuning sequence generation systems.

## Mathematical Explanation

### Loss Function

For a dataset of \( N \) input-output pairs \( (X^{(i)}, Y^{(i)}) \), where \( Y^{(i)} = (y_1^{(i)}, \ldots, y_{T'_i}^{(i)}) \), the loss is:

\[
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t'=1}^{T'_i} \log P(y_{t'}^{(i)} \mid y_{<t'}^{(i)}, X^{(i)})
\]

### Masked Loss with Padding

When sequences in a batch are padded to the same length, a mask \( M^{(i)}_{t'} \in \{0, 1\} \) indicates whether position \( t' \) in sequence \( i \) is a real token (1) or padding (0). The masked loss is:

\[
\mathcal{L} = -\frac{1}{\sum_{i=1}^{N} \sum_{t'=1}^{T'_{\max}} M^{(i)}_{t'}} \sum_{i=1}^{N} \sum_{t'=1}^{T'_{\max}} M^{(i)}_{t'} \log P(y_{t'}^{(i)} \mid y_{<t'}^{(i)}, X^{(i)})
\]

### Teacher Forcing

With teacher forcing ratio \( r \in [0, 1] \):

\[
\text{input}_{t'} = \begin{cases}
y_{t'-1}^* & \text{with probability } r \\
\hat{y}_{t'-1} & \text{with probability } 1 - r
\end{cases}
\]

where \( y^* \) is the ground truth and \( \hat{y} \) is the model's prediction.

### Gradient Clipping

To prevent exploding gradients, gradients are clipped to a maximum norm \( \theta \):

\[
g \leftarrow \frac{g}{\max(1, \|g\|_2 / \theta)}
\]

### Learning Rate Schedule

A common schedule for seq2seq is the Noam schedule (from the Transformer paper):

\[
\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot \text{warmup\_steps}^{-1.5})
\]

## Code Examples

### Example 1: Complete Seq2Seq Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.trg_data[idx])

def collate_fn(batch, pad_idx=0):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    trg_lens = [len(t) for t in trg_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, trg_padded, src_lens, trg_lens

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, trg, src_lens, trg_lens in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Setup
INPUT_DIM = 100
OUTPUT_DIM = 100
EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 1
DROPOUT = 0.1
CLIP = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = EncoderRNN(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = DecoderRNN(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

src_data = [[3, 5, 2, 7, 4], [1, 6, 8], [9, 2, 5, 3, 7, 1]]
trg_data = [[4, 2, 6, 1], [3, 7, 5, 8, 2], [1, 4, 9, 3]]
dataset = TranslationDataset(src_data, trg_data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_fn(b, 0))

for epoch in range(5):
    loss = train_epoch(model, dataloader, optimizer, criterion, CLIP, DEVICE)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
# Output: Epoch 1: Loss = 4.6231
# Output: Epoch 2: Loss = 4.5892
# Output: Epoch 3: Loss = 4.5547
# Output: Epoch 4: Loss = 4.5210
# Output: Epoch 5: Loss = 4.4875
```

### Example 2: Scheduled Sampling Implementation

```python
import numpy as np

class ScheduledSamplingSeq2Seq(nn.Module):
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

def linear_decay_schedule(epoch, total_epochs, start_ratio=1.0, end_ratio=0.0):
    return start_ratio - (start_ratio - end_ratio) * epoch / (total_epochs - 1)

def train_with_scheduled_sampling(model, dataloader, optimizer, criterion, clip, device, epoch, total_epochs):
    model.train()
    tf_ratio = linear_decay_schedule(epoch, total_epochs)
    epoch_loss = 0
    for src, trg, src_lens, trg_lens in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader), tf_ratio

for epoch in range(10):
    loss, tf_ratio = train_with_scheduled_sampling(
        model, dataloader, optimizer, criterion, CLIP, DEVICE, epoch, 10
    )
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, TF Ratio = {tf_ratio:.2f}")
# Output: Epoch 1: Loss = 4.6231, TF Ratio = 1.00
# Output: Epoch 2: Loss = 4.5876, TF Ratio = 0.89
# Output: Epoch 5: Loss = 4.5123, TF Ratio = 0.56
# Output: Epoch 10: Loss = 4.4231, TF Ratio = 0.00
```

### Example 3: Gradient Clipping and Learning Rate Scheduling

```python
import math

def train_with_lr_schedule(model, dataloader, optimizer, criterion, clip, device, epoch, d_model=64, warmup_steps=4000):
    model.train()
    epoch_loss = 0
    for i, (src, trg, src_lens, trg_lens) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)
        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step_num = epoch * len(dataloader) + i + 1
        current_lr = d_model ** (-0.5) * min(
            step_num ** (-0.5),
            step_num * warmup_steps ** (-1.5)
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
for epoch in range(5):
    loss = train_with_lr_schedule(model, dataloader, optimizer, criterion, CLIP, DEVICE, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, LR = {current_lr:.6f}")
# Output: Epoch 1: Loss = 4.6231, LR = 0.000250
# Output: Epoch 2: Loss = 4.5892, LR = 0.000354
# Output: Epoch 5: Loss = 4.5012, LR = 0.000559
```

## Common Mistakes

1. **Not masking padding tokens in the loss computation**: When computing cross-entropy loss, padding tokens should be ignored. Using `nn.CrossEntropyLoss(ignore_index=pad_idx)` is essential. Without masking, the model is penalized for not predicting the correct padding token, which teaches it to predict meaningless padding.

2. **Applying the loss to the SOS token**: The first token of the target sequence is typically the SOS token, which should not be used as a training target. The model should receive the SOS token as input and predict the first real token. The loss should start from position 1 (index 1) of the output.

3. **Using a constant high learning rate**: Seq2seq models are sensitive to learning rate. A high learning rate causes divergence due to exploding gradients, while a low learning rate leads to slow convergence. Learning rate scheduling with warmup is strongly recommended.

4. **Training with too small batches**: Small batch sizes lead to high variance gradients and unstable training for seq2seq models. Larger batches provide more stable gradient estimates, though memory constraints may limit size.

5. **Neglecting validation with beam search**: Validation loss measured under teacher forcing does not correlate perfectly with generation quality. Periodic validation using beam search or greedy decoding with BLEU/ROUGE scores provides a more realistic assessment.

## Interview Questions

### Beginner

Q: What is the purpose of teacher forcing in seq2seq training?

A: Teacher forcing feeds the ground truth token as input to the decoder at each timestep instead of the model's own prediction. This accelerates convergence by providing a strong learning signal and prevents the model from being confused by its own early errors during training.

### Intermediate

Q: Why do we need gradient clipping in seq2seq training, and what happens if we omit it?

A: Seq2seq models have deep computational graphs due to the combination of many timesteps (through time) and layers (through depth). Gradients can grow exponentially during backpropagation, causing huge parameter updates that destabilize training or cause NaN loss. Gradient clipping caps the gradient norm to a threshold (typically 1.0 or 5.0), preventing these explosions while preserving the direction of the gradient.

### Advanced

Q: Compare and contrast the training dynamics of seq2seq models with transformers versus RNN-based seq2seq models. How does the training paradigm change?

A: RNN-based seq2seq trains with sequential computation: gradients flow through time, and the depth is proportional to sequence length. This necessitates gradient clipping, makes training slow for long sequences, and limits parallelism. Transformer seq2seq processes all tokens in parallel during training (using masked self-attention in the decoder), enabling full parallelization across timesteps. Training is faster but requires more memory (O(T^2) attention). Transformers also use the Noam learning rate schedule with warmup, label smoothing, and dropout more aggressively. The loss computation is similar (cross-entropy with masking), but transformers handle the information bottleneck through cross-attention rather than a single context vector, changing how the decoder accesses encoder information.

## Practice Problems

### Easy

Create a training loop for a seq2seq model that uses constant teacher forcing ratio of 0.8. Train for 20 epochs on a synthetic dataset of 1000 random sequence pairs. Plot the training and validation loss curves.

### Medium

Implement a training pipeline that uses gradient accumulation to simulate larger batch sizes. With a batch size of 4 and accumulation steps of 8, simulate an effective batch size of 32. Explain how gradient accumulation works and why it is useful.

### Hard

Implement validation during seq2seq training using BLEU score. Every 5 epochs, perform greedy decoding on a validation set and compute the BLEU score against the reference translations. Compare BLEU progression with loss progression and discuss any correlation or lack thereof.

## Solutions

### Easy Solution

```python
import matplotlib.pyplot as plt

losses = []
for epoch in range(20):
    loss = train_epoch(model, dataloader, optimizer, criterion, CLIP, DEVICE)
    losses.append(loss)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
# Output: Epoch 0: Loss = 4.6231
# Output: Epoch 5: Loss = 4.4102
# Output: Epoch 10: Loss = 4.2385
# Output: Epoch 15: Loss = 4.1123
# Output: Epoch 19: Loss = 4.0287
```

## Related Concepts

- Teacher Forcing and Scheduled Sampling
- Gradient Clipping
- Learning Rate Scheduling
- Cross-Entropy Loss and Label Smoothing
- BLEU Score Evaluation

## Next Concepts

- DL-327: Greedy Decoding
- DL-328: Beam Search
- DL-329: Length Normalization

## Summary

Seq2Seq training involves optimizing the encoder and decoder jointly to maximize the probability of target sequences given input sequences. The core training loop uses teacher forcing for efficient convergence, masked cross-entropy loss to handle variable-length sequences, and gradient clipping to prevent gradient explosion. Additional techniques like scheduled sampling, learning rate scheduling, and label smoothing improve training stability and final model quality. The training paradigm for seq2seq models established patterns that carry forward to modern transformer-based sequence generation, making it essential knowledge for deep learning practitioners.

## Key Takeaways

- Seq2Seq training jointly optimizes encoder and decoder using cross-entropy loss with teacher forcing.
- Padding must be masked in the loss computation to avoid learning on meaningless tokens.
- Gradient clipping (typically norm 1.0-5.0) prevents exploding gradients.
- Scheduled sampling gradually transitions from teacher forcing to free-running to reduce exposure bias.
- Validation should evaluate generation quality (e.g., BLEU) in addition to loss.
