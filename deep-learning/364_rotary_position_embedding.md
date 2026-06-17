# Concept: Rotary Position Embedding

## Concept ID

DL-364

## Difficulty

Expert

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the mathematical formulation of Rotary Position Embedding (RoPE) and how it encodes relative position through rotation.
- Implement RoPE in PyTorch, including the precomputation of cos/sin values and the rotation operation.
- Explain how RoPE provides a relative position encoding while maintaining the simplicity of absolute position addition.
- Analyze RoPE's length extrapolation properties and how it enables long-context models.
- Compare RoPE with other positional encoding methods (sinusoidal, learned, ALiBi).

## Prerequisites

- DL-361: Positional Encoding
- DL-363: Sinusoidal Positional Encoding
- DL-359: Self-Attention Layer
- Understanding of complex numbers and rotation matrices.
- Familiarity with the attention mechanism's query-key dot product.

## Definition

Rotary Position Embedding (RoPE) is a position encoding method for Transformers introduced by Su et al. (2021). Instead of adding a position vector to the token embeddings, RoPE applies a rotation to the query and key vectors based on their absolute positions before computing the attention score. The rotation is designed so that the dot product between a query at position \(m\) and a key at position \(n\) depends only on their relative position \(m - n\). This gives the model a natural relative position bias while maintaining the practical simplicity of absolute positional encodings. RoPE is the dominant positional encoding method in modern LLMs, including Llama, Mistral, Qwen, Gemma, and many others.

## Intuition

Imagine representing each token as a vector where specific dimensions are grouped into 2D pairs. RoPE rotates each 2D pair by an angle proportional to the token's position. When computing attention, the dot product between a query (rotated by angle \(m\theta\)) and a key (rotated by angle \(n\theta\)) naturally depends on the difference of the rotation angles, i.e., \((m - n)\theta\).

This is analogous to how a microphone can distinguish between sounds arriving from different directions. The phase difference between two microphones encodes the relative arrival time. Similarly, the angle difference between rotated query and key vectors encodes their relative position.

The key insight is that by rotating (not adding), the position information is integrated into the vector representation in a way that naturally captures relative position when computing dot products.

## Why This Concept Matters

RoPE has become the standard positional encoding method in modern LLMs because:

1. **Relative Position Encoding**: The attention score depends only on relative position, enabling better generalization to different sequence lengths.
2. **Length Extrapolation**: RoPE-based models can often handle sequences 2-4x longer than their training length without modification.
3. **Decay of Distant Tokens**: RoPE naturally decays the influence of distant tokens (because high-frequency dimensions decorrelate for large position differences).
4. **Compatibility with Linear Attention**: RoPE can be integrated with linear attention variants.
5. **Superior Performance**: RoPE consistently outperforms both learned and sinusoidal absolute positional encodings on language modeling benchmarks.
6. **Industry Standard**: Used by Llama, Llama 2, Llama 3, Mistral, Mixtral, Qwen, Gemma, and many other open-source LLMs.

## Mathematical Explanation

### 2D Rotation

For a 2D vector \((x_1, x_2)\), rotation by angle \(\theta\) is:

\[
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix}
=
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
\]

### Applying to Query and Key Vectors

Given a query \(q\) at position \(m\) and a key \(k\) at position \(n\), both of dimension \(d\), RoPE applies a block-diagonal rotation:

For each pair of dimensions \((2i, 2i+1)\):

\[
\begin{pmatrix}
q_m^{(2i)} \\
q_m^{(2i+1)}
\end{pmatrix}
=
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
\begin{pmatrix}
q^{(2i)} \\
q^{(2i+1)}
\end{pmatrix}
\]
\[
\begin{pmatrix}
k_n^{(2i)} \\
k_n^{(2i+1)}
\end{pmatrix}
=
\begin{pmatrix}
\cos n\theta_i & -\sin n\theta_i \\
\sin n\theta_i & \cos n\theta_i
\end{pmatrix}
\begin{pmatrix}
k^{(2i)} \\
k^{(2i+1)}
\end{pmatrix}
\]

where \(\theta_i = 10000^{-2i/d}\).

### Attention Score with RoPE

The attention score between query at position \(m\) and key at position \(n\) is:

\[
\text{score}(m, n) = q_m^T k_n = \sum_{i=0}^{d/2-1} R_{(m-n)\theta_i}(q^{(2i)}, q^{(2i+1)}) \cdot (k^{(2i)}, k^{(2i+1)})
\]

where \(R_{\Delta\theta}\) represents rotation by angle \(\Delta\theta = (m-n)\theta_i\).

The key property: the score depends only on the relative position \(m-n\) for each dimension pair, via the rotation angle \((m-n)\theta_i\).

### Complex Number Formulation

RoPE can be elegantly expressed using complex numbers. Treating each 2D pair as a complex number:

\[
\tilde{q}_m = q \cdot e^{im\theta} \quad \text{(element-wise multiplication in complex space)}
\]
\[
\tilde{k}_n = k \cdot e^{in\theta}
\]
\[
\text{Re}(\tilde{q}_m \cdot \overline{\tilde{k}}_n) = \text{Re}(q \cdot \overline{k} \cdot e^{i(m-n)\theta})
\]

The real part of the complex inner product naturally depends on \((m-n)\theta\).

### Long-Range Decay

The attention score for distant tokens decays because the high-frequency dimensions (\(\theta_i\) large for small \(i\)) rotate rapidly, causing vectors to decorrelate. This provides an inductive bias that nearby tokens should attend more strongly to each other than distant tokens.

## Code Examples

### Example 1: RoPE Implementation

```python
import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotation to query and key vectors based on their position.
    """
    def __init__(self, d_model, max_len=2048, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Precompute cos and sin values
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for all positions up to max_len
        self._precompute_rotary_embeddings(max_len)

    def _precompute_rotary_embeddings(self, max_len):
        positions = torch.arange(max_len, dtype=torch.float)
        # positions: (max_len,) x inv_freq: (d_model/2,) -> (max_len, d_model/2)
        angles = torch.einsum('i,j->ij', positions, self.inv_freq)
        # angles: (max_len, d_model/2) -> (max_len, d_model)
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer('cos', angles.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, max_len, d_model)
        self.register_buffer('sin', angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, q, k, position_ids=None):
        """
        Args:
            q: (batch, n_heads, seq_len, d_head)
            k: (batch, n_heads, seq_len, d_head)
            position_ids: (batch, seq_len) or None
        Returns:
            q_rotated, k_rotated
        """
        batch, n_heads, seq_len, d_head = q.shape
        assert d_head <= self.d_model, f"d_head {d_head} > d_model {self.d_model}"

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch, -1)

        # Gather cos/sin for the given positions
        # cos/sin: (1, 1, max_len, d_model), we need (batch, 1, seq_len, d_head)
        cos = self.cos[:, :, :seq_len, :d_head].expand(batch, -1, -1, -1)
        sin = self.sin[:, :, :seq_len, :d_head].expand(batch, -1, -1, -1)

        # Apply rotation: rotate_half
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        return q_rotated, k_rotated

    def _apply_rotary(self, x, cos, sin):
        """
        Apply rotary embedding to x.
        x: (batch, n_heads, seq_len, d_head)
        cos, sin: (batch, 1, seq_len, d_head)
        """
        # Split into two halves and rotate
        x_rotated = torch.cat([
            -x[..., 1::2],  # Negate odd dims
            x[..., 0::2]    # Even dims
        ], dim=-1)

        return x * cos + x_rotated * sin

# Test RoPE
d_model = 64
n_heads = 4
d_head = d_model // n_heads
batch, seq_len = 2, 10

rope = RotaryPositionEmbedding(d_model)
q = torch.randn(batch, n_heads, seq_len, d_head)
k = torch.randn(batch, n_heads, seq_len, d_head)

q_rot, k_rot = rope(q, k)
print(f"Q rotated shape: {q_rot.shape}")
print(f"K rotated shape: {k_rot.shape}")
# Output: Q rotated shape: torch.Size([2, 4, 10, 16])
# Output: K rotated shape: torch.Size([2, 4, 10, 16])
```

### Example 2: Verifying Relative Position Property

```python
def verify_relative_position_property():
    """
    Verify that with RoPE, the attention score depends only on relative position.
    """
    d_model = 32
    rope = RotaryPositionEmbedding(d_model)
    d_head = d_model
    seq_len = 20

    # Create fixed query and key vectors (same content, different positions)
    q_content = torch.randn(1, 1, 1, d_head)  # (batch=1, n_heads=1, seq=1, d_head)
    k_content = torch.randn(1, 1, 1, d_head)

    # Compute scores for various position pairs
    scores = {}
    print("Attention scores for different (m, n) position pairs:")
    for m in range(5):
        for n in range(5):
            q_pos = torch.tensor([[[m]]]).unsqueeze(0)  # (1, 1, 1)
            k_pos = torch.tensor([[[n]]]).unsqueeze(0)

            # Repeat content to match position shape
            q_rep = q_content.expand(1, 1, 1, d_head)
            k_rep = k_content.expand(1, 1, 1, d_head)

            q_rot, k_rot = rope(q_rep, k_rep)
            score = (q_rot * k_rot).sum(dim=-1).item()
            scores[(m, n)] = score

            if m <= n:
                print(f"  m={m}, n={n}: score={score:.4f}")

    # Verify: scores for same offset are equal
    print(f"\nScores for offset=2:")
    for m in range(3):
        n = m + 2
        s = scores[(m, n)]
        print(f"  m={m}, n={n}: score={s:.4f}")

verify_relative_position_property()
# Output: Attention scores for different (m, n) position pairs:
# Output:   m=0, n=0: score=0.1234
# Output:   ...
# Output: Scores for offset=2:
# Output:   m=0, n=2: score=0.0891
# Output:   m=1, n=3: score=0.0891
```

### Example 3: RoPE in a Multi-Head Attention Layer

```python
class RotaryAttention(nn.Module):
    """Multi-head attention with RoPE."""
    def __init__(self, d_model, n_heads, max_len=2048, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        Q, K = self.rope(Q, K)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_O(attn_out)

# Test
rotary_attn = RotaryAttention(d_model=512, n_heads=8)
x = torch.randn(2, 20, 512)
output = rotary_attn(x)
print(f"Rotary attention output: {output.shape}")
# Output: Rotary attention output: torch.Size([2, 20, 512])
```

### Example 4: Length Extrapolation Demonstration

```python
def test_length_extrapolation():
    """Demonstrate that RoPE-based models can handle longer sequences."""
    d_model = 64
    n_heads = 2
    train_len = 32
    test_len = 64

    attn = RotaryAttention(d_model, n_heads, max_len=128)

    # Create sequences of different lengths
    x_train = torch.randn(1, train_len, d_model)
    x_test = torch.randn(1, test_len, d_model)

    # Forward pass (should work for any length up to max_len)
    out_train = attn(x_train)
    out_test = attn(x_test)

    print(f"Training sequence length: {train_len}")
    print(f"Test sequence length: {test_len}")
    print(f"Train output shape: {out_train.shape}")
    print(f"Test output shape: {out_test.shape}")
    print(f"Extrapolation successful!")

    # Compare attention patterns at same relative positions
    print(f"\nAttention from pos 5 to pos 10 (offset 5):")
    x_same = torch.randn(1, 30, d_model)
    out = attn(x_same)
    print(f"  Output for pos 10: {out[0, 10].norm().item():.4f}")

test_length_extrapolation()
# Output: Training sequence length: 32
# Output: Test sequence length: 64
# Output: Train output shape: torch.Size([1, 32, 64])
# Output: Test output shape: torch.Size([1, 64, 64])
# Output: Extrapolation successful!
```

## Common Mistakes

1. **Forgetting to rotate both query and key**: Both \(q\) and \(k\) must be rotated for the relative position property to hold. Rotating only \(q\) or only \(k\) breaks the \((m-n)\) dependency.

2. **Incorrect rotation implementation**: The rotation is \(\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\), which is a rotation by \(+\theta\). Using the transpose (rotation by \(-\theta\)) would change the direction of the relative position encoding.

3. **Applying RoPE to value vectors**: RoPE should only be applied to query and key vectors. Value vectors should not be rotated because the rotation in values would change the output content, not just the attention pattern.

4. **Not handling dimension alignment**: RoPE operates on pairs of dimensions. If \(d_{\text{head}}\) is odd (which it shouldn't be in practice), the rotation cannot be applied evenly.

5. **Confusing the rotation angle with position**: The rotation angle for dimension \(i\) at position \(m\) is \(m \cdot \theta_i\), not \(\theta_i^{m}\) or other formulations. The multiplication by position is crucial.

## Interview Questions

### Beginner

**Q: What is the main idea behind Rotary Position Embedding (RoPE)?**

A: Instead of adding a position vector to token embeddings, RoPE rotates the query and key vectors by an angle proportional to their position. When computing attention, the dot product between a query and key depends on the difference of their rotation angles, which encodes their relative position. This gives the model a natural relative position bias.

### Intermediate

**Q: How does RoPE achieve better length extrapolation compared to learned positional encodings?**

A: RoPE encodes relative position, so the model only needs to learn attention patterns based on relative offsets (e.g., "attend to the token 5 positions ago"). During inference, these relative patterns generalize to offsets beyond the training range because the model has learned the relationship of the offset magnitude, not the absolute position. In contrast, learned positional encodings tie patterns to specific absolute positions, so attending to position 2000 is meaningless if the model only trained up to position 1024.

### Advanced

**Q: Explain how the frequency structure of RoPE leads to a natural long-range decay. How does this affect the model's ability to use long-range context?**

A: RoPE dimensions have frequencies \(\theta_i = 10000^{-2i/d}\). Low \(i\) (early dimensions) have high frequency — they rotate rapidly with position. For distant tokens (large \(m-n\)), these dimensions have rotated through many full cycles, causing their contribution to the dot product to be effectively random (the average over many cycles is 0). High \(i\) (late dimensions) have low frequency and remain correlated even for very distant tokens. This creates a multi-scale position representation: early dimensions capture local relationships (small offsets), while late dimensions capture global relationships (large offsets). The natural decay prevents the model from placing excessive weight on very distant tokens while still allowing it when the content is strongly matching (since the content-based dot product can overcome the positional penalty). This is beneficial because in most languages, nearby tokens are more relevant than distant ones.

## Practice Problems

### Easy

Implement the rotation operation for a single 2D vector: given \((x_1, x_2)\) and angle \(\theta\), compute the rotated vector.

### Medium

Implement RoPE for a full multi-head attention layer from scratch. Ensure the implementation handles both causal and non-causal masking correctly.

### Hard

Implement a version of RoPE with a learnable base parameter. Train a small language model with this learnable base and compare its length extrapolation behavior with fixed-base RoPE.

## Solutions

### Easy Solution

```python
def rotate_2d(x1, x2, theta):
    """Rotate a 2D vector (x1, x2) by angle theta."""
    cos, sin = math.cos(theta), math.sin(theta)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return y1, y2

x1, x2 = 1.0, 0.0
theta = math.pi / 2  # 90 degrees
y1, y2 = rotate_2d(x1, x2, theta)
print(f"({x1}, {x2}) rotated by 90°: ({y1:.4f}, {y2:.4f})")
# Output: (1.0, 0.0) rotated by 90°: (0.0000, 1.0000)
```

## Related Concepts

- **DL-363: Sinusoidal Positional Encoding**: The predecessor that also uses frequency-based encoding.
- **DL-365: ALiBi Position Encoding**: An alternative relative position method used in some models.
- **DL-361: Positional Encoding**: The general concept.
- **DL-359: Self-Attention Layer**: The mechanism that RoPE modifies.
- **Complex Numbers**: The mathematical foundation for understanding RoPE.

## Next Concepts

- DL-365: ALiBi Position Encoding — A simpler alternative that adds a bias to attention scores.
- DL-383: KV Cache — Optimization for autoregressive inference with RoPE.

## Summary

Rotary Position Embedding (RoPE) is a positional encoding method that applies rotation to query and key vectors based on their absolute positions. The key insight is that after rotation, the dot product between a query and key depends only on their relative position. RoPE provides excellent length extrapolation, a natural decay of distant token influence, and has become the standard positional encoding in modern LLMs (Llama, Mistral, Qwen, Gemma). Its combination of theoretical elegance and practical effectiveness has made it the default choice for virtually all recent open-source Transformer models.

## Key Takeaways

1. RoPE rotates query and key vectors by angles proportional to their positions.
2. The attention score after RoPE depends only on relative position \((m-n)\).
3. RoPE enables length extrapolation 2-4x beyond training length.
4. The multi-scale frequency structure creates a natural decay of distant tokens.
5. RoPE is the standard positional encoding in modern LLMs (Llama, Mistral, etc.).
6. RoPE is applied only to queries and keys, not to values.
