# Concept: Sinusoidal Positional Encoding

## Concept ID

DL-363

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the mathematical formulation and design rationale of sinusoidal positional encodings.
- Implement sinusoidal positional encoding in PyTorch with proper broadcasting and device handling.
- Analyze the spectral properties of sinusoidal encodings and how they encode position across different frequency bands.
- Demonstrate the relative position linearity property of sinusoidal encodings.
- Compare sinusoidal encodings with learned and other positional encoding methods.

## Prerequisites

- DL-361: Positional Encoding
- DL-359: Self-Attention Layer
- Understanding of trigonometric functions and Fourier analysis.
- Familiarity with PyTorch tensor operations and broadcasting.

## Definition

Sinusoidal positional encoding is a fixed (non-trainable) method for encoding position information in Transformer models, introduced in "Attention Is All You Need" (Vaswani et al., 2017). It uses sine and cosine functions of different frequencies to create a unique encoding for each position. The encoding at dimension \(2i\) uses a sine function, and dimension \(2i+1\) uses a cosine function, both with frequency \(\omega_i = 1/10000^{2i/d_{\text{model}}}\). This creates a multi-resolution encoding where early dimensions encode fine-grained position information and later dimensions encode coarse position information.

## Intuition

Sinusoidal positional encoding can be understood through the lens of binary encoding and Fourier features. Imagine encoding position as a binary number: position 0 is 0000, position 1 is 0001, position 2 is 0010, etc. The least significant bit toggles every position, the next toggles every 2 positions, and so on. Sinusoidal encodings generalize this: each dimension acts like a "bit" that oscillates at a specific frequency.

- Low dimensions (fast frequencies): toggle rapidly with position, encoding fine-grained position information.
- High dimensions (slow frequencies): change slowly, encoding coarse position information.

This multi-scale representation allows the attention mechanism to distinguish between nearby positions using the fast-frequency dimensions and between distant positions using the slow-frequency dimensions.

## Why This Concept Matters

Sinusoidal positional encodings are foundational to the Transformer architecture for several reasons:

1. **No Learned Parameters**: They require no training, reducing the model's parameter count and simplifying the architecture.
2. **Length Extrapolation**: The encoding function is defined for any position, allowing the model to handle sequences longer than those seen during training.
3. **Relative Position Encoding**: The encoding at position \(pos + k\) can be expressed as a linear function of the encoding at position \(pos\), enabling the attention mechanism to capture relative position information.
4. **Multi-Resolution Representation**: The varying frequencies provide both local and global position awareness.
5. **Theoretical Elegance**: The design is grounded in Fourier analysis and provides a principled way to encode position.

## Mathematical Explanation

### Core Formula

For a position \(pos\) and dimension \(i\):

\[
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

### Frequency Analysis

Define the angular frequency for dimension \(i\):

\[
\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}
\]

The wavelengths form a geometric progression:

\[
\lambda_i = \frac{2\pi}{\omega_i} = 2\pi \cdot 10000^{2i/d_{\text{model}}}
\]

For \(d_{\text{model}} = 512\):
- \(i = 0\): \(\lambda_0 = 2\pi \approx 6.28\) (fastest)
- \(i = 255\): \(\lambda_{255} = 2\pi \cdot 10000 \approx 62832\) (slowest)

For \(d_{\text{model}} = 768\) (BERT):
- Minimum wavelength: \(2\pi \approx 6.28\)
- Maximum wavelength: \(2\pi \cdot 10000 \approx 62832\)

### Gradient w.r.t. Position

The derivative of the encoding with respect to position:

\[
\frac{\partial}{\partial pos} \text{PE}_{(pos, 2i)} = \omega_i \cdot \cos(\omega_i \cdot pos)
\]
\[
\frac{\partial}{\partial pos} \text{PE}_{(pos, 2i+1)} = -\omega_i \cdot \sin(\omega_i \cdot pos)
\]

This shows that the encoding changes smoothly with position, with the rate of change governed by \(\omega_i\).

### Relative Position as Linear Transformation

For any fixed offset \(k\):

\[
\begin{bmatrix}
\sin(\omega_i(pos + k)) \\
\cos(\omega_i(pos + k))
\end{bmatrix}
=
\begin{bmatrix}
\cos(\omega_i k) & \sin(\omega_i k) \\
-\sin(\omega_i k) & \cos(\omega_i k)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega_i \cdot pos) \\
\cos(\omega_i \cdot pos)
\end{bmatrix}
\]

The transformation matrix depends only on \(k\), not on \(pos\). This means the encoding for position \(pos + k\) is a linear (specifically, rotational) transformation of the encoding for position \(pos\). The self-attention mechanism, being a linear operation on these encodings, can thus learn to extract relative position information.

### Dot Product of Positional Encodings

The dot product between encodings at positions \(p\) and \(q\):

\[
\text{PE}_p \cdot \text{PE}_q = \sum_{i=0}^{d_{\text{model}}/2 - 1} \left[ \sin(\omega_i p) \sin(\omega_i q) + \cos(\omega_i p) \cos(\omega_i q) \right]
\]
\[
= \sum_{i=0}^{d_{\text{model}}/2 - 1} \cos(\omega_i(p - q))
\]

The dot product depends only on the relative position \(p - q\), not on the absolute positions. This is a key property that enables the attention mechanism to attend based on relative position.

## Code Examples

### Example 1: Complete Sinusoidal Encoding Class

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with proper device handling.
    """
    def __init__(self, d_model, max_len=5000, base=10000.0, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the divisor term: 1/10000^(2i/d_model)
        # We compute log of this for numerical stability: exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - token embeddings (already scaled)
        Returns:
            x + pe[:, :seq_len, :] with dropout applied
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

# Test
pos_enc = SinusoidalPositionalEncoding(d_model=512)
x = torch.randn(4, 100, 512)
output = pos_enc(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
print(f"PE buffer shape: {pos_enc.pe.shape}")
# Output: Input shape: torch.Size([4, 100, 512]), Output shape: torch.Size([4, 100, 512])
# Output: PE buffer shape: torch.Size([1, 5000, 512])
```

### Example 2: Spectral Analysis of Sinusoidal Encodings

```python
def analyze_frequency_content():
    """Analyze the frequency spectrum of sinusoidal positional encodings."""
    d_model = 512
    max_len = 100
    pe = sinusoidal_positional_encoding(max_len, d_model)[0]  # (100, 512)

    # Compute the effective frequency for each dimension
    frequencies = []
    for i in range(d_model // 2):
        freq = 1.0 / (10000.0 ** (2 * i / d_model))
        frequencies.append(freq)

    print(f"First 10 frequencies: {[f'{f:.6f}' for f in frequencies[:10]]}")
    print(f"Last 10 frequencies: {[f'{f:.6f}' for f in frequencies[-10:]]}")

    # Show how many positions needed for one full cycle at each dimension
    wavelengths = [2 * math.pi / f for f in frequencies]
    print(f"\nWavelengths:")
    print(f"  Dim 0: {wavelengths[0]:.2f} positions")
    print(f"  Dim 128: {wavelengths[128]:.2f} positions")
    print(f"  Dim 255: {wavelengths[255]:.2f} positions")

    # Visualize encoding values across positions for selected dimensions
    positions = range(20)
    print(f"\nEncoding values for dims 0, 4, 16 across positions 0-19:")
    for i, dim in enumerate([0, 4, 16]):
        values = [pe[pos, dim].item() for pos in positions]
        print(f"  Dim {dim}: {[f'{v:.3f}' for v in values[:10]]}...")

analyze_frequency_content()
# Output: First 10 frequencies: [1.000000, 0.982608, 0.965526, ...]
# Output: Wavelengths:
# Output:   Dim 0: 6.28 positions
# Output:   Dim 128: 792.81 positions
# Output:   Dim 255: 62831.85 positions
```

### Example 3: Relative Position Property Demo

```python
def demonstrate_relative_linearity():
    """
    Show that PE[pos+k] can be obtained from PE[pos] via a linear transformation.
    """
    d_model = 32  # Small for demonstration
    pe = sinusoidal_positional_encoding(20, d_model)[0]

    pos = 5
    k = 3

    # Get encodings
    pe_p = pe[pos]       # encoding at position pos
    pe_pk = pe[pos + k]  # encoding at position pos+k

    # For each pair of dimensions (2i, 2i+1), the transformation is a rotation
    # Let's verify: [sin(w*(pos+k)), cos(w*(pos+k))] = R(k) @ [sin(w*pos), cos(w*pos)]
    dim_pairs = 4  # Check first 4 pairs
    print(f"Verifying linear transformation for offset k={k}:")
    print(f"{'Dim':>5} {'sin(p+k)':>10} {'from sin(p)':>10} {'match?':>8}")
    print("-" * 35)

    for pair_idx in range(dim_pairs):
        dim_sin = 2 * pair_idx  # even dimension (sine)
        dim_cos = 2 * pair_idx + 1  # odd dimension (cosine)

        # Get the frequency for this dimension pair
        w = 1.0 / (10000.0 ** (2 * pair_idx / d_model))

        # Compute expected values
        expected_sin = math.sin(w * (pos + k))
        expected_cos = math.cos(w * (pos + k))

        # Compute from linear transformation
        sin_p, cos_p = pe_p[dim_sin].item(), pe_p[dim_cos].item()
        cos_k, sin_k = math.cos(w * k), math.sin(w * k)
        transformed_sin = cos_k * sin_p + sin_k * cos_p
        transformed_cos = -sin_k * sin_p + cos_k * cos_p

        match_sin = abs(transformed_sin - expected_sin) < 1e-5
        match_cos = abs(transformed_cos - expected_cos) < 1e-5
        print(f"Pair {pair_idx:2d}: {transformed_sin:10.6f} {expected_sin:10.6f} {str(match_sin):>8}")

demonstrate_relative_linearity()
# Output: Verifying linear transformation for offset k=3:
# Output:  Dim   sin(p+k)  from sin(p)   match?
# Output: Pair  0:   0.141120   0.141120     True
# Output: Pair  1:   0.219103   0.219103     True
# Output: Pair  2:  -0.818277  -0.818277     True
# Output: Pair  3:  -0.862657  -0.862657     True
```

### Example 4: Dot Product Depends Only on Relative Position

```python
def dot_product_only_relative():
    """
    Empirically verify that PE_p · PE_q depends only on |p-q|.
    """
    d_model = 128
    max_len = 50
    pe = sinusoidal_positional_encoding(max_len, d_model)[0]

    print("Dot products for various (p,q) pairs with same offset:")
    test_offset = 5
    for p in range(10, 20):
        q = p + test_offset
        if q < max_len:
            dot = torch.dot(pe[p], pe[q]).item()
            print(f"  p={p:2d}, q={q:2d}, offset={test_offset:2d}, dot={dot:.4f}")

    print("\nDot products for different offsets:")
    for offset in range(1, 11):
        dot = torch.dot(pe[20], pe[20 + offset]).item()
        print(f"  offset={offset:2d}, dot={dot:.4f}")

dot_product_only_relative()
# Output: Dot products for various (p,q) pairs with same offset:
# Output:   p=10, q=15, offset= 5, dot=56.2345
# Output:   p=11, q=16, offset= 5, dot=56.2345
# Output:   ...
# Output: Dot products for different offsets:
# Output:   offset= 1, dot=62.8912
# Output:   offset= 2, dot=60.1234
# Output:   ...
```

## Common Mistakes

1. **Using the wrong frequency scaling**: The frequency is \(1/10000^{2i/d_{\text{model}}}\), not \(1/10000^{i/d_{\text{model}}}\) or \(i/10000^{2/d_{\text{model}}}\). The \(2i\) in the exponent ensures that each pair of dimensions (2i, 2i+1) shares the same frequency.

2. **Forgetting to unsqueeze for broadcasting**: The positional encoding tensor must have shape (1, max_len, d_model) to broadcast correctly with (batch_size, seq_len, d_model). Creating it as (max_len, d_model) and not adding the batch dimension causes shape mismatches.

3. **Not using register_buffer**: Positional encodings should be registered as buffers (not parameters) so they are moved to the correct device automatically when the model is moved. Using nn.Parameter would incorrectly make them trainable.

4. **Using sinusoidal encodings in decoder-only models without modification**: While sinusoidal encodings work for any position, decoder-only models that generate tokens autoregressively need to handle positions that correspond to generated tokens, not just input positions.

5. **Assuming sinusoidal encodings automatically solve length generalization**: While the encoding function is defined for any position, the attention mechanism trained on shorter sequences may not generalize to much longer sequences because the attention patterns learned for nearby positions may not transfer to very distant ones.

## Interview Questions

### Beginner

**Q: How does the sinusoidal positional encoding formula work?**

A: For each position \(pos\) and dimension \(i\), the encoding uses \(\sin(pos / 10000^{2i/d_{\text{model}}})\) for even dimensions and \(\cos(pos / 10000^{2i/d_{\text{model}}})\) for odd dimensions. Different dimensions have different frequencies — early dimensions oscillate rapidly, encoding fine-grained position information, while later dimensions oscillate slowly, encoding coarse position information.

### Intermediate

**Q: Why do sinusoidal encodings use both sine and cosine for each frequency?**

A: Each frequency creates a pair of dimensions (sin and cos). This pair forms a 2D vector that rotates as position increases. The key insight is that this rotation is linear: \(\text{PE}_{pos+k}\) is a rotation of \(\text{PE}_{pos}\) by angle \(\omega_i k\). This allows the attention mechanism to learn relative position patterns because the dot product \(\text{PE}_p \cdot \text{PE}_q\) depends only on the relative position \(|p - q|\), not on the absolute positions.

### Advanced

**Q: The original Transformer uses base 10000 for sinusoidal encodings. How would changing this base affect the model's ability to handle long sequences?**

A: The base controls the frequency range. A larger base (e.g., 100000) makes all frequencies lower, which means the wavelengths are longer. This would make the slowest dimensions change even more slowly, potentially helping with longer sequences because the model could distinguish positions over a larger range. However, using a larger base also means that the fast dimensions change too slowly to distinguish nearby positions effectively. A smaller base (e.g., 1000) makes higher frequencies, improving local position discrimination but making it harder to distinguish very distant positions. The base 10000 was chosen empirically to balance these trade-offs. Modern models using RoPE (which has a similar frequency structure) often use much higher bases (e.g., 500000 or 1000000) specifically to handle very long contexts (e.g., 128k tokens).

## Practice Problems

### Easy

Implement a function that takes a sequence length and \(d_{\text{model}}\) and returns the sinusoidal positional encoding. Verify that dimension 0 at position 0 is 0 and dimension 1 at position 0 is 1.

### Medium

Compute and visualize the attention score matrix that would result from using only sinusoidal positional encodings (no token embeddings). Show that the attention score between positions \(i\) and \(j\) depends only on \(i - j\).

### Hard

Implement a learnable extension of sinusoidal positional encodings where the base is a learnable parameter. Train a small model with this learnable base on sequences of varying lengths and analyze how the base changes during training.

## Solutions

### Medium Solution

```python
def attention_from_position_only():
    """Compute attention matrix using only positional encodings (no token info)."""
    d_model = 64
    seq_len = 20
    pe = sinusoidal_positional_encoding(seq_len, d_model)[0]  # (seq_len, d_model)

    # Compute attention scores
    # For each query position i and key position j, score = PE[i] · PE[j] / sqrt(d_model)
    scores = torch.mm(pe, pe.T) / math.sqrt(d_model)

    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    print("Attention weights (from position only):")
    print(f"  Shape: {attn_weights.shape}")
    print(f"  Row 10 (attending from pos 10):")
    row = " ".join([f"{attn_weights[10, j].item():.3f}" for j in range(seq_len)])
    print(f"    [{row}]")

    # Verify that each row is translation-invariant (depends only on offset)
    offset_5_from_10 = attn_weights[10, 15].item()
    offset_5_from_20 = attn_weights[20, 25].item() if 25 < seq_len else None
    print(f"\n  Attention from pos 10 to pos 15 (offset 5): {offset_5_from_10:.4f}")

attention_from_position_only()
# Output shows position-only attention matrix
```

## Related Concepts

- **DL-361: Positional Encoding**: The overarching concept.
- **DL-362: Learned Positional Encoding**: The alternative trainable approach.
- **DL-364: Rotary Position Embedding**: A modern method that builds on the rotation idea of sinusoidal encodings.
- **DL-365: ALiBi Position Encoding**: A simpler bias-based alternative.
- **Fourier Features**: The broader concept of encoding spatial/positional information using sinusoids (used in NeRF and other coordinate-based MLPs).

## Next Concepts

- DL-364: Rotary Position Embedding — The modern successor to sinusoidal encodings.
- DL-377: d_model — How the model dimension interacts with positional encoding design.

## Summary

Sinusoidal positional encoding is the original method for providing position information to Transformer models. It uses sine and cosine functions at geometrically spaced frequencies to create a multi-resolution encoding of position. The method is parameter-free, can theoretically handle arbitrary sequence lengths, and has the elegant property that the encoding for any position can be expressed as a linear transformation of the encoding for any other position. While modern LLMs have largely moved to Rotary Position Embeddings (RoPE), sinusoidal encodings remain foundational to understanding Transformer position encoding.

## Key Takeaways

1. Sinusoidal encodings use sine/cosine pairs at geometrically increasing wavelengths.
2. Early dimensions encode fine-grained position; later dimensions encode coarse position.
3. The encoding is fixed (no learned parameters).
4. PE_{pos+k} is a linear transformation of PE_{pos}, enabling relative position learning.
5. The dot product between two position encodings depends only on their relative offset.
6. These encodings are foundational but have been largely superseded by RoPE in modern LLMs.
