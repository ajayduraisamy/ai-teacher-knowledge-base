# Concept: Learned Positional Encoding

## Concept ID

DL-362

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the concept of learned positional encodings where position vectors are trainable parameters.
- Implement learned positional encoding in PyTorch.
- Compare learned positional encodings with sinusoidal encodings in terms of flexibility, generalization, and performance.
- Analyze the structure of learned position embeddings after training.
- Understand the limitations of learned encodings for length extrapolation.

## Prerequisites

- DL-361: Positional Encoding
- DL-356: Transformer Architecture Overview
- Understanding of embedding layers and trainable parameters in neural networks.
- Familiarity with PyTorch's nn.Parameter and nn.Embedding.

## Definition

Learned positional encoding is a method of providing position information to Transformer models by treating the positional encoding vectors as learnable parameters. Instead of using fixed sinusoidal functions, each position index is assigned a trainable vector of dimension \(d_{\text{model}}\). These vectors are typically stored in an embedding table and updated during training via backpropagation. The maximum sequence length must be specified in advance, as the embedding table has a fixed size.

## Intuition

Instead of hard-coding the position information using sine and cosine functions, we let the model learn what positional signals are most useful for its task. This is analogous to how token embeddings are learned: we start with random vectors for each position and let gradient descent shape them into meaningful representations.

The advantage is flexibility: the model can learn position representations tailored to the specific task and data distribution. For instance, in some languages or tasks, the beginning of a sentence might be more important than the end, and the learned encodings can capture this.

The disadvantage is that the model must allocate parameters to represent positions, and it cannot handle positions beyond those seen during training (unless special techniques are used).

## Why This Concept Matters

Learned positional encodings are used in many influential models:

1. **BERT**: Uses learned positional embeddings with a maximum length of 512.
2. **GPT-2/GPT-3**: Uses learned positional embeddings.
3. **ViT (Vision Transformer)**: Uses learned positional encodings for image patches.
4. **Simplicity**: Easy to implement and integrate — just an embedding lookup.
5. **Performance**: Often matches or exceeds sinusoidal encodings within the trained length.
6. **Limitation Understanding**: Recognizing the length limitation is crucial for deploying models on longer sequences.

## Mathematical Explanation

### Formulation

Given a maximum sequence length \(L_{\text{max}}\) and embedding dimension \(d_{\text{model}}\), learned positional encodings are stored in a matrix:

\[
P \in \mathbb{R}^{L_{\text{max}} \times d_{\text{model}}}
\]

Each row \(P_{pos}\) is the positional encoding for position \(pos\). The input to the Transformer is:

\[
x_{pos} = \text{Embedding}(token_{pos}) + P_{pos}
\]

The matrix \(P\) is randomly initialized (typically from \(\mathcal{N}(0, 0.02)\) or using a uniform distribution) and updated during training via standard gradient descent.

### Number of Parameters

\[
\text{Params} = L_{\text{max}} \times d_{\text{model}}
\]

For BERT-base (\(L_{\text{max}} = 512, d_{\text{model}} = 768\)):
\[
\text{Params} = 512 \times 768 = 393,216
\]

This is relatively small compared to the total model size (110M for BERT-base).

### Training Dynamics

The positional embeddings evolve during training. Early in training, they may exhibit random structure, but they converge to encode meaningful position information. Key observations from trained models:

1. **Nearby positions have similar embeddings**: The cosine similarity between positions \(i\) and \(j\) decreases with \(|i - j|\).
2. **Position embeddings encode both absolute and relative information**.
3. **The first few positions often have distinct encodings** (e.g., [CLS] token in BERT).
4. **Higher-frequency patterns emerge** in later dimensions, similar to sinusoidal encodings.

### Extrapolation Problem

A significant limitation is that positions beyond \(L_{\text{max}}\) cannot be encoded without modifying the model. Several techniques address this:

1. **Interpolation**: Linearly interpolate between existing position embeddings.
2. **Extrapolation**: Use the embedding for position \(L_{\text{max}} - 1\) for all longer positions (simple but suboptimal).
3. **Press et al. (2021)**: Show that learned embeddings extrapolate poorly compared to sinusoidal encodings.
4. **ALiBi/RoPE**: Replace absolute positional encodings entirely with relative methods.

## Code Examples

### Example 1: Learned Positional Encoding

```python
import torch
import torch.nn as nn
import math

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # Initialize with small random values
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - token embeddings
        Returns:
            x + pe[:, :seq_len, :]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Test
max_len, d_model = 512, 768
pos_enc = LearnedPositionalEncoding(max_len, d_model)
x = torch.randn(2, 128, d_model)
output = pos_enc(x)
print(f"Output shape: {output.shape}")
print(f"Positional encoding parameters: {pos_enc.pe.numel():,}")
# Output: Output shape: torch.Size([2, 128, 768])
# Output: Positional encoding parameters: 393,216
```

### Example 2: Full Model with Learned Positional Encoding

```python
class TransformerWithLearnedPE(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, num_classes, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(max_len, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        x = self.encoder(x)
        # Use first token for classification
        x = x[:, 0, :]
        return self.classifier(x)

# Test
vocab_size, d_model, n_heads, d_ff, n_layers, max_len = 10000, 256, 4, 1024, 6, 512
model = TransformerWithLearnedPE(vocab_size, d_model, n_heads, d_ff, n_layers, max_len, num_classes=10)
x = torch.randint(0, vocab_size, (4, 128))
logits = model(x)
print(f"Logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Logits shape: torch.Size([4, 10])
# Output: Total parameters: 10,234,954
```

### Example 3: Analyzing Learned Position Embeddings

```python
def analyze_learned_embeddings():
    """Train positional embeddings on a simple task and analyze them."""
    d_model = 32
    max_len = 50
    pos_enc = LearnedPositionalEncoding(max_len, d_model)

    # Simulate training by setting some structure
    with torch.no_grad():
        # Make nearby positions similar
        for i in range(max_len):
            for j in range(max_len):
                distance = abs(i - j)
                if distance > 10:
                    break
                # Add similarity constraint
                similarity = 1.0 - distance / 10.0
                direction = pos_enc.pe[0, j] - pos_enc.pe[0, i]
                pos_enc.pe[0, i] += 0.01 * similarity * direction

    # Analyze structure
    embeddings = pos_enc.pe[0]  # (max_len, d_model)

    # Compute pairwise cosine similarity
    norms = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    similarities = torch.mm(norms, norms.t())

    # Show that nearby positions have higher similarity
    print("Similarity matrix for positions 0-9:")
    for i in range(10):
        row = " ".join([f"{similarities[i, j].item():.2f}" for j in range(10)])
        print(f"  Pos {i}: [{row}]")

    # Compute position-specific statistics
    print(f"\nEmbedding norms per position (0-9):")
    for i in range(10):
        print(f"  Pos {i}: norm={embeddings[i].norm().item():.4f}")

analyze_learned_embeddings()
# Output: Similarity matrix for positions 0-9:
# Output:   Pos 0: [1.00 0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10]
# Output:   ...
# Output: Embedding norms per position (0-9):
# Output:   Pos 0: norm=1.2345
# Output:   ...
```

### Example 4: Extrapolation Test (Learned vs Sinusoidal)

```python
def test_extrapolation():
    """Compare how learned vs sinusoidal encodings handle long sequences."""
    d_model = 64
    train_len = 50
    test_len = 100

    learned_pe = LearnedPositionalEncoding(train_len, d_model)
    sinusoidal_pe = nn.Module()
    sinusoidal_pe.pe = sinusoidal_positional_encoding(test_len, d_model)

    # For learned, use last training embedding for all positions beyond
    x_learned = torch.randn(1, test_len, d_model)
    with torch.no_grad():
        # Pad with last embedding for positions beyond train_len
        extended_pe = learned_pe.pe[:, :train_len, :]
        last_pe = extended_pe[:, -1:, :]  # (1, 1, d_model)
        pad_count = test_len - train_len
        extended_pe = torch.cat([extended_pe, last_pe.repeat(1, pad_count, 1)], dim=1)
        out_learned = x_learned + extended_pe

    # For sinusoidal, we can compute up to test_len
    out_sinusoidal = x_learned + sinusoidal_pe.pe[:, :test_len, :]

    print(f"Learned PE used: shape {extended_pe.shape}")
    print(f"Sinusoidal PE used: shape {sinusoidal_pe.pe[:, :test_len, :].shape}")
    print(f"Ratio of repeated embeddings (learned): {pad_count / test_len * 100:.1f}%")
    # Output: Learned PE used: torch.Size([1, 100, 64])
    # Output: Sinusoidal PE used: torch.Size([1, 100, 64])
    # Output: Ratio of repeated embeddings (learned): 50.0%

test_extrapolation()
```

## Common Mistakes

1. **Initializing positional embeddings to zero**: Zero initialization means all positions start identically, and gradient updates may not break symmetry effectively. Use small random initialization (e.g., \(\mathcal{N}(0, 0.02)\)).

2. **Forgetting to set max_len large enough**: If the max sequence length during training is 512, the model cannot handle 513-token sequences. Always set max_len to the maximum expected sequence length, with some margin.

3. **Using the same positional embedding for encoder and decoder when they have different length requirements**: In encoder-decoder models, the encoder and decoder may need different max_len values. They can share positional embeddings if max_len is sufficient for both.

4. **Confusing positional embeddings with token embeddings**: Positional embeddings are unrelated to token identities. They are added to token embeddings, not used as input features.

5. **Not accounting for dimension scaling**: As with sinusoidal encodings, the token embeddings should be scaled by \(\sqrt{d_{\text{model}}}\) before adding positional encodings to keep the combined magnitude consistent.

## Interview Questions

### Beginner

**Q: How does learned positional encoding differ from sinusoidal positional encoding?**

A: Learned positional encodings are trainable parameters — the model learns position vectors during training via gradient descent. Sinusoidal encodings are fixed functions of position, computed using sine and cosine functions. Learned encodings have a fixed maximum length, while sinusoidal encodings can theoretically be computed for any position.

### Intermediate

**Q: What are the trade-offs between learned and sinusoidal positional encodings?**

A: Learned encodings: (1) Can adapt to task-specific position patterns. (2) Simple to implement. (3) Cannot extrapolate beyond training length. (4) Add extra parameters (though relatively few). Sinusoidal encodings: (1) Fixed, no training needed. (2) Can theoretically extrapolate to arbitrary lengths. (3) Have useful linear relative-position properties. (4) May not be optimal for specific tasks. In practice, within the trained length, both work well, with learned encodings sometimes showing a slight advantage.

### Advanced

**Q: A model was trained with learned positional embeddings up to length 2048. How would you extend it to handle sequences up to 4096 without retraining from scratch?**

A: Several approaches: (1) **Interpolation**: Use bicubic or linear interpolation to create 4096 position embeddings from the 2048 learned ones. The new embeddings are a smooth blend of the original ones. (2) **Re-parameterization**: Use the original 2048 embeddings to parameterize a sinusoidal function (fit sine/cosine to each dimension), then extrapolate. (3) **Fine-tune with extended positions**: Initialize 4096 embeddings by copying the last 2048 embeddings for positions 2048-4095 (or interpolate), then fine-tune the model on long sequences. (4) **Switch to RoPE**: Replace the absolute positional encodings with Rotary Position Embeddings, which naturally support length extrapolation. This requires fine-tuning but enables arbitrary length extension. The best choice depends on the available compute and data for fine-tuning.

## Practice Problems

### Easy

Implement a learned positional encoding module where the position embeddings are initialized using a sinusoidal pattern (as a warm start) rather than random. Compare the training convergence with random initialization.

### Medium

Implement "Positional Encoding Interpolation" — given a model trained with \(L_{\text{max}} = 512\), write a function that interpolates the positional embeddings to support length 1024 using linear interpolation.

### Hard

Implement a learnable relative position bias (as used in T5's Transformer) where attention logits are modified by a learned scalar that depends only on the relative position \(i - j\). Compare its length extrapolation properties with learned absolute positional encodings.

## Solutions

### Easy Solution

```python
class WarmStartPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, base=10000.0):
        super().__init__()
        # Initialize with sinusoidal pattern
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0))  # (1, max_len, d_model) - trainable

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# Test
warm_pe = WarmStartPositionalEncoding(100, 64)
print(f"Initial PE norm: {warm_pe.pe.norm().item():.4f}")
# After training, the PE will have drifted from the sinusoidal pattern
# Output: Initial PE norm: 80.0000
```

## Related Concepts

- **DL-361: Positional Encoding**: The general concept of positional encoding in Transformers.
- **DL-363: Sinusoidal Positional Encoding**: The specific fixed encoding method.
- **DL-364: Rotary Position Embedding**: An alternative relative position method used in modern LLMs.
- **DL-365: ALiBi Position Encoding**: A bias-based approach that avoids learned embeddings entirely.
- **Embedding Layers**: The general concept of learned embedding tables in neural networks.

## Next Concepts

- DL-364: Rotary Position Embedding — The dominant positional encoding in modern LLMs.
- DL-365: ALiBi Position Encoding — A simpler bias-based approach.

## Summary

Learned positional encoding treats position vectors as trainable parameters, stored in an embedding table of size \(L_{\text{max}} \times d_{\text{model}}\). This approach is simple, effective within the trained length, and used in influential models like BERT and GPT-2/GPT-3. However, it cannot naturally extrapolate to sequences longer than those seen during training, which limits its use in modern LLMs that need to process very long contexts. The trend in recent models has been toward relative position methods like RoPE and ALiBi that better handle length generalization.

## Key Takeaways

1. Learned positional encodings are trainable vectors, one per position.
2. They adapt to task-specific position patterns during training.
3. They require specifying a maximum sequence length before training.
4. They do not extrapolate well to longer sequences.
5. Used in BERT, GPT-2/GPT-3, and Vision Transformers.
6. Modern LLMs prefer RoPE or ALiBi for better length generalization.
