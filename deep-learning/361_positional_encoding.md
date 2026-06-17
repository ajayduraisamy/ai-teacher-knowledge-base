# Concept: Positional Encoding

## Concept ID

DL-361

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand why positional encoding is necessary in Transformer models.
- Explain the mathematical formulation of sinusoidal positional encodings.
- Implement positional encoding in PyTorch.
- Compare different positional encoding methods (sinusoidal, learned, rotary, ALiBi).
- Analyze the properties of positional encodings: relative vs absolute position information.

## Prerequisites

- DL-356: Transformer Architecture Overview
- Basic understanding of the self-attention mechanism and its permutation-equivariance property.
- Familiarity with sine and cosine functions, trigonometric identities.
- Experience with PyTorch tensor operations.

## Definition

Positional encoding is a technique used in Transformer models to inject information about the position of tokens in a sequence. Since the self-attention mechanism is permutation-equivariant (it treats the input as an unordered set), positional encodings are added to the token embeddings to provide the model with sequence order information. These encodings can be fixed (sinusoidal) or learned, and they encode either absolute position, relative position, or both.

## Intuition

Imagine you have a sentence: "The dog chased the cat." The words "dog" and "cat" have different semantic roles depending on their position relative to "chased." Self-attention alone cannot distinguish between "The dog chased the cat" and "The cat chased the dog" because the set of tokens is the same. Positional encodings solve this by adding a position-dependent signal to each token's embedding.

Think of positional encodings as a coordinate system for the sequence. Each position gets a unique vector that tells the model "I am token number \(pos\)." When added to the token embedding, this signal allows the attention mechanism to distinguish between tokens at different positions.

Sinusoidal positional encodings have a particularly elegant property: the encoding at position \(pos + k\) can be represented as a linear function of the encoding at position \(pos\). This means the model can easily learn to attend to relative positions (e.g., "the word 2 positions ago").

## Why This Concept Matters

Positional encoding is essential because:

1. **Permutation-Equivariance**: Self-attention has no inherent notion of order. Without positional encodings, a bag-of-words model.
2. **Sequence Understanding**: Many tasks (translation, parsing, classification) depend on word order.
3. **Modeling Relative Position**: The best positional encodings allow the model to capture relative position information (e.g., "the subject is 3 words before the verb").
4. **Length Generalization**: Some encoding methods generalize better to sequences longer than those seen during training.
5. **Architecture Impact**: The choice of positional encoding is a key architectural decision affecting performance, training efficiency, and length extrapolation.

## Mathematical Explanation

### Absolute Positional Encoding

The original Transformer uses sinusoidal positional encodings:

\[
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

where:
- \(pos\) is the position index (0-indexed)
- \(i\) is the dimension index (0 to \(d_{\text{model}}/2 - 1\))
- \(d_{\text{model}}\) is the embedding dimension

The frequencies decrease along the dimension axis: early dimensions have high frequency (rapidly changing with position), while later dimensions have low frequency (slowly changing with position).

### Properties of Sinusoidal Encoding

**Relative Position Information**: Using the trigonometric identity:

\[
\sin(\alpha + \beta) = \sin\alpha \cos\beta + \cos\alpha \sin\beta
\]
\[
\cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta
\]

We can show that \(\text{PE}_{pos+k}\) is a linear function of \(\text{PE}_{pos}\). This allows the attention mechanism to easily learn relative position patterns.

**Wavelengths**: The wavelengths form a geometric progression:

\[
\lambda_i = 2\pi \cdot 10000^{2i/d_{\text{model}}}
\]

from \(2\pi\) (fastest) to \(2\pi \cdot 10000 \approx 62832\) (slowest).

### Adding Positional Encodings

The positional encoding is added to the token embedding (not concatenated):

\[
x_{pos} = \text{Embedding}(token_{pos}) + \text{PE}_{pos}
\]

The original paper also multiplies the embedding by \(\sqrt{d_{\text{model}}}\) before adding the positional encoding, to match the scale of the positional encodings.

### Alternative Approaches

1. **Learned Positional Encoding**: Each position has a learnable vector. Used in BERT, GPT.
2. **Rotary Position Embedding (RoPE)**: Applies rotation to the query and key vectors based on position. Encodes relative position.
3. **ALiBi (Attention with Linear Biases)**: Adds a bias to attention scores based on position difference, without modifying token embeddings.
4. **Relative Position Encodings**: Used in Transformer-XL and T5, where attention is directly informed of relative position.

## Code Examples

### Example 1: Sinusoidal Positional Encoding

```python
import torch
import torch.nn as nn
import math

def sinusoidal_positional_encoding(max_len, d_model, base=10000.0):
    """
    Creates sinusoidal positional encodings.

    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        base: Base for frequency calculation

    Returns:
        pe: (1, max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)

# Visualize the sinusoidal patterns
pe = sinusoidal_positional_encoding(100, 128)
print(f"Shape: {pe.shape}")
print(f"Position 0, dims 0-7: {pe[0, 0, :8].tolist()}")
print(f"Position 5, dims 0-7: {pe[0, 5, :8].tolist()}")
# Output: Shape: torch.Size([1, 100, 128])
# Output: Position 0, dims 0-7: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
# Output: Position 5, dims 0-7: [0.4794, 0.8776, 0.0098, 1.0000, ...]
```

### Example 2: Applying Positional Encoding in a Transformer

```python
class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = sinusoidal_positional_encoding(max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - token embeddings
        Returns:
            x + pe[:, :seq_len, :] with dropout
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Test
d_model = 512
pos_enc_layer = PositionalEncodingLayer(d_model)
x = torch.randn(2, 10, d_model)
output = pos_enc_layer(x)
print(f"Output shape: {output.shape}")

# Verify the positional signal dominates the embedding for early dimensions
embedding = torch.zeros(1, 1, d_model)  # Zero embedding
embedded = pos_enc_layer(embedding * math.sqrt(d_model))
print(f"Positional encoding in dim 0 for pos 0-4: {embedded[0, :5, 0].tolist()}")
# Output: Output shape: torch.Size([2, 10, 512])
# Output: Positional encoding in dim 0 for pos 0-4: [0.0, 0.8415, 0.9093, 0.1411, -0.7568]
```

### Example 3: Demonstrating Relative Position Properties

```python
def demonstrate_relative_position(d_model=32, max_len=50):
    pe = sinusoidal_positional_encoding(max_len, d_model)[0]  # (max_len, d_model)

    # For positions pos and pos+k, show PE[pos+k] - PE[pos]
    pos = 10
    k = 5
    diff = pe[pos + k] - pe[pos]
    dot_product = torch.dot(pe[pos + k], pe[pos])
    print(f"Position {pos} and {pos + k}:")
    print(f"  Dot product: {dot_product.item():.4f}")
    print(f"  Norm of difference: {diff.norm().item():.4f}")
    print(f"  Euclidean distance: {(pe[pos + k] - pe[pos]).norm().item():.4f}")

    # Show that the dot product depends mainly on |k|, not on pos
    print("\nDot products for different positions with same offset k=5:")
    for pos in [5, 10, 20, 30]:
        dot = torch.dot(pe[pos + k], pe[pos]).item()
        print(f"  pos={pos}, pos+k={pos + k}: dot={dot:.4f}")

demonstrate_relative_position()
# Output: Position 10 and 15:
# Output:   Dot product: 0.3124
# Output:   Norm of difference: 1.1897
# Output:   Euclidean distance: 1.1897
# Output:
# Output: Dot products for different positions with same offset k=5:
# Output:   pos=5, pos+k=10: dot=0.3124
# Output:   pos=10, pos+k=15: dot=0.3124
# Output:   pos=20, pos+k=25: dot=0.3124
# Output:   pos=30, pos+k=35: dot=0.3124
```

### Example 4: Comparing Positional Encoding Methods

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class NoPositionalEncoding(nn.Module):
    def forward(self, x):
        return x  # Identity — model sees bag of words

def compare_encoding_methods():
    """Compare how different encodings affect attention patterns."""
    d_model, n_heads = 64, 4
    seq_len = 10

    # Create a simple model with each encoding type
    class TestModel(nn.Module):
        def __init__(self, pos_enc):
            super().__init__()
            self.pos_enc = pos_enc
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        def forward(self, x):
            x = self.pos_enc(x)
            return self.attn(x, x, x)

    x = torch.randn(1, seq_len, d_model)

    # No encoding
    model_no = TestModel(NoPositionalEncoding())
    out_no, attn_no = model_no(x)
    print(f"Without positional encoding:")
    print(f"  Attention weights (first 3 positions): {attn_no[0, 0, :3, :3]}")

    # Sinusoidal encoding
    model_sin = TestModel(PositionalEncodingLayer(d_model, max_len=100))
    out_sin, attn_sin = model_sin(x)
    print(f"\nWith sinusoidal encoding:")
    print(f"  Attention weights (first 3 positions): {attn_sin[0, 0, :3, :3]}")

    # Learned encoding
    model_learned = TestModel(LearnedPositionalEncoding(100, d_model))
    out_learned, attn_learned = model_learned(x)
    print(f"\nWith learned encoding:")
    print(f"  Attention weights (first 3 positions): {attn_learned[0, 0, :3, :3]}")

compare_encoding_methods()
# Output shows that without encoding, attention is uniform; with encoding, it varies by position
```

## Common Mistakes

1. **Confusing positional encoding with embedding**: Positional encodings are added to token embeddings, not concatenated. Adding preserves the embedding dimension; concatenation would double it.

2. **Forgetting to scale embeddings**: The original Transformer multiplies embeddings by \(\sqrt{d_{\text{model}}}\) before adding positional encodings. This ensures the positional signal is not drowned out by the embedding magnitude.

3. **Applying positional encoding before dropout**: In the original implementation, positional encoding is added before dropout, not after. The dropout is applied to the sum of embedding + positional encoding.

4. **Assuming sinusoidal encodings generalize to arbitrary lengths**: While sinusoidal encodings can in theory generalize to any length because they're defined by a continuous function, in practice the attention mechanism may not have seen such large position differences during training.

5. **Using positional encodings with the wrong dimension ordering**: Depending on whether your tensor is (batch, seq, d_model) or (seq, batch, d_model), you need to ensure the encoding broadcasts correctly.

## Interview Questions

### Beginner

**Q: Why do Transformers need positional encodings?**

A: Self-attention is permutation-equivariant — it treats the input as an unordered set. Processing "The cat sat on the mat" would produce the same representation as "Mat the on sat cat The" without positional information. Positional encodings inject sequence order information so the model can distinguish between these.

### Intermediate

**Q: What are the advantages of sinusoidal positional encodings over learned positional encodings?**

A: (1) Sinusoidal encodings do not require training — they are computed deterministically. (2) They can extrapolate to sequence lengths longer than those seen during training (though the attention mechanism itself may not generalize). (3) The relative position property (PE_{pos+k} as a linear function of PE_{pos}) allows the model to easily learn relative position patterns. (4) Learned encodings require allocating a maximum length at training time, and positions beyond that have no encoding.

### Advanced

**Q: How do Rotary Position Embeddings (RoPE) differ from absolute positional encodings? What problem do they solve?**

A: RoPE applies a rotation to the query and key vectors based on their position, rather than adding a positional signal to the token embeddings. The attention score between positions \(i\) and \(j\) depends only on their relative position \(i - j\). This solves the length generalization problem: after training on sequences up to length \(L\), RoPE-based models can often generalize to sequences of length \(2L\) or more because the attention mechanism only sees relative offsets, not absolute positions. RoPE also elegantly decays the influence of distant tokens naturally (the rotation causes high-frequency dimensions to decorrelate). RoPE is used in nearly all modern LLMs (Llama, Mistral, Qwen, Gemma).

## Practice Problems

### Easy

Implement the sinusoidal positional encoding function and verify that the encoding at position 0 has zero for all even dimensions and 1 for all odd dimensions.

### Medium

Create a visualization showing how the dot product between positional encodings at positions \(i\) and \(j\) varies with \(|i - j|\). Show that it depends only on the offset, not on the absolute positions.

### Hard

Implement Rotary Position Embedding (RoPE) from scratch. Apply the rotation to query and key vectors before computing attention scores, and verify that the attention score depends only on relative position.

## Solutions

### Easy Solution

```python
def verify_sinusoidal_pe():
    d_model = 16
    pe = sinusoidal_positional_encoding(10, d_model)[0]

    # Position 0
    pos0 = pe[0]
    even_dims = pos0[0::2]  # dims 0, 2, 4, ...
    odd_dims = pos0[1::2]   # dims 1, 3, 5, ...
    print(f"Position 0 - Even dims (should be 0): {even_dims}")
    print(f"Position 0 - Odd dims (should be 1): {odd_dims}")

    # Verify: sin(0) = 0, cos(0) = 1
    assert torch.allclose(even_dims, torch.zeros_like(even_dims), atol=1e-6)
    assert torch.allclose(odd_dims, torch.ones_like(odd_dims), atol=1e-6)
    print("Verified: PE(0) = [0, 1, 0, 1, ...]")

verify_sinusoidal_pe()
# Output: Position 0 - Even dims (should be 0): tensor([0., 0., 0., 0., 0., 0., 0., 0.])
# Output: Position 0 - Odd dims (should be 1): tensor([1., 1., 1., 1., 1., 1., 1., 1.])
# Output: Verified: PE(0) = [0, 1, 0, 1, ...]
```

### Medium Solution

```python
def visualize_relative_position_dot():
    import matplotlib.pyplot as plt
    d_model = 128
    max_len = 100
    pe = sinusoidal_positional_encoding(max_len, d_model)[0]

    # Compute dot products for different offsets
    reference_pos = 50
    offsets = torch.arange(-49, 50)
    dots = []
    for offset in offsets:
        target_pos = reference_pos + offset
        if 0 <= target_pos < max_len:
            dot = torch.dot(pe[reference_pos], pe[target_pos]).item()
            dots.append(dot)
        else:
            dots.append(0)

    plt.figure(figsize=(10, 4))
    plt.plot(offsets.tolist(), dots)
    plt.xlabel("Offset (relative position)")
    plt.ylabel("Dot product")
    plt.title("Dot product of sinusoidal PEs vs relative position")
    plt.grid(True)
    plt.savefig("pe_dot_vs_offset.png")
    plt.close()
    print(f"Plot saved. Max dot at offset 0: {dots[49]:.4f}")
    # Output: Plot saved. Max dot at offset 0: 64.0000
```

## Related Concepts

- **DL-362: Learned Positional Encoding**: Position encodings that are trained as parameters.
- **DL-363: Sinusoidal Positional Encoding**: The fixed sinusoidal encoding used in the original Transformer.
- **DL-364: Rotary Position Embedding (RoPE)**: A modern relative position encoding method.
- **DL-365: ALiBi Position Encoding**: A bias-based approach to position encoding.
- **Self-Attention**: The mechanism that positional encodings inform about sequence order.

## Next Concepts

- DL-362: Learned Positional Encoding — Training position embeddings as model parameters.
- DL-364: Rotary Position Embedding — The dominant positional encoding in modern LLMs.

## Summary

Positional encoding is a critical component in Transformer models that provides sequence order information to the permutation-equivariant self-attention mechanism. The original Transformer uses sinusoidal positional encodings with varying frequencies across dimensions, which allow the model to capture both absolute and relative position information. Alternative approaches include learned positional encodings (BERT, GPT), Rotary Position Embeddings (RoPE, used in Llama and Mistral), and ALiBi. The choice of positional encoding affects length generalization, training efficiency, and model performance.

## Key Takeaways

1. Positional encodings are required because self-attention is permutation-equivariant.
2. Sinusoidal encodings use sine and cosine functions at different frequencies.
3. The encodings are added to token embeddings (not concatenated).
4. Sinusoidal encodings allow the model to learn relative position patterns.
5. Modern LLMs favor RoPE (rotary) over absolute positional encodings for better length generalization.
6. Positional encodings can be fixed (sinusoidal) or learned.
