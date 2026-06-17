# Concept: Multi-Head Attention Splitting

## Concept ID

DL-372

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand how the input tensor is split across multiple attention heads in multi-head attention.
- Implement the splitting operation in PyTorch using view, transpose, and reshape operations.
- Explain the relationship between d_model, n_heads, and d_head in the splitting process.
- Analyze the computational and memory implications of different splitting strategies.
- Understand how the splitting enables parallel computation across heads.

## Prerequisites

- DL-371: Attention Head
- DL-359: Self-Attention Layer
- Understanding of tensor reshaping and transposition in PyTorch.
- Familiarity with matrix multiplication and linear projections.

## Definition

Multi-head attention splitting is the operation of dividing the projected query, key, and value tensors into multiple "heads" by reshaping and transposing the tensor dimensions. Given an input tensor of shape \((batch, seq\_len, d_{model})\), the splitting operation projects it to \((batch, seq\_len, d_{model})\) via linear layers, then reshapes to \((batch, seq\_len, n_{heads}, d_{head})\) and transposes to \((batch, n_{heads}, seq\_len, d_{head})\). This creates separate "viewpoints" for each head, enabling parallel attention computation across heads.

## Intuition

Think of multi-head attention as having multiple experts examine the same input. The splitting operation is like giving each expert a different lens (projection) to view the input and then arranging them side-by-side so they can all work simultaneously.

The key insight is that the splitting is purely a tensor manipulation — no information is lost or duplicated. The same total number of features (\(d_{\text{model}}\)) is simply partitioned into \(h\) groups of size \(d_{\text{model}}/h\). Each head processes its group independently.

The splitting operation is analogous to the "head" operation in a multi-threaded program: the input is "fanned out" to multiple parallel workers, each working on a portion of the data.

## Why This Concept Matters

Understanding multi-head attention splitting is important because:

1. **Implementation**: Correct splitting is essential for implementing multi-head attention from scratch.
2. **Efficiency**: The splitting determines how computations are parallelized on GPUs.
3. **Architecture Design**: The relationship between \(d_{\text{model}}\), \(n_{\text{heads}}\), and \(d_{\text{head}}\) affects model capacity.
4. **Memory Layout**: Different splitting/transposition approaches affect memory access patterns.
5. **Debugging**: Incorrect splitting is a common source of bugs in custom Transformer implementations.

## Mathematical Explanation

### Linear Projection

First, the input is projected to query, key, and value spaces:

\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]

where \(X \in \mathbb{R}^{b \times n \times d_{\text{model}}}\) and \(W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\).

### Splitting into Heads

The projected tensors have shape \((b, n, d_{\text{model}})\). To split into \(h\) heads of dimension \(d_k = d_{\text{model}} / h\):

**Step 1: Reshape**

\[
Q_{\text{reshaped}} = Q.\text{view}(b, n, h, d_k)
\]

This creates a new dimension for heads. The last dimension \(d_{\text{model}}\) is split into \((h, d_k)\).

**Step 2: Transpose**

\[
Q_{\text{final}} = Q_{\text{reshaped}}.\text{transpose}(1, 2)
\]

This moves the heads dimension to position 1, resulting in shape \((b, h, n, d_k)\).

Now, each head's queries are in a contiguous block: \(Q_{\text{final}}[:, h, :, :]\) contains the queries for head \(h\).

### Why Transpose?

The transpose is necessary because the attention computation for each head is:

\[
\text{head}_h(Q, K, V) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h
\]

where \(Q_h, K_h, V_h \in \mathbb{R}^{b \times n \times d_k}\). Having the heads dimension first (after batch) allows parallel computation across all heads with a single batched matrix multiplication.

### Tensor Shape Evolution

| Operation | Shape |
|-----------|-------|
| Input | \((b, n, d_{\text{model}})\) |
| After linear projection | \((b, n, d_{\text{model}})\) |
| After reshape | \((b, n, h, d_k)\) |
| After transpose | \((b, h, n, d_k)\) |
| After attention | \((b, h, n, d_k)\) |
| After transpose | \((b, n, h, d_k)\) |
| After reshape | \((b, n, d_{\text{model}})\) |

## Code Examples

### Example 1: Manual Splitting of Multi-Head Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionSplitting(nn.Module):
    """
    Multi-head attention with explicit splitting and concatenation.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Split the last dimension of x into (n_heads, d_head).
        Input:  (batch, seq_len, d_model)
        Output: (batch, n_heads, seq_len, d_head)
        """
        batch, seq_len, _ = x.shape
        # Step 1: Reshape to (batch, seq_len, n_heads, d_head)
        x = x.view(batch, seq_len, self.n_heads, self.d_head)
        # Step 2: Transpose to (batch, n_heads, seq_len, d_head)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        """
        Reverse of split_heads.
        Input:  (batch, n_heads, seq_len, d_head)
        Output: (batch, seq_len, d_model)
        """
        batch, _, seq_len, _ = x.shape
        # Step 1: Transpose to (batch, seq_len, n_heads, d_head)
        x = x.transpose(1, 2)
        # Step 2: Reshape to (batch, seq_len, d_model)
        x = x.contiguous().view(batch, seq_len, self.d_model)
        return x

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split into heads
        Q = self.split_heads(Q)  # (batch, n_heads, seq_len, d_head)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention (batched across heads)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_head)

        # Combine heads
        attn_out = self.combine_heads(attn_out)  # (batch, seq_len, d_model)

        # Output projection
        return self.W_O(attn_out), attn_weights

# Test splitting
mha = MultiHeadAttentionSplitting(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output, attn_weights = mha(x)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"n_heads={mha.n_heads}, d_head={mha.d_head}")
# Output: Output shape: torch.Size([2, 10, 512])
# Output: Attention weights shape: torch.Size([2, 8, 10, 10])
# Output: n_heads=8, d_head=64
```

### Example 2: Visualizing the Splitting Operation

```python
def visualize_splitting():
    """Show the tensor shapes at each stage of splitting."""
    d_model, n_heads = 32, 4
    d_head = d_model // n_heads

    x = torch.randn(2, 6, d_model)  # (batch=2, seq=6, d_model=32)

    # Linear projection (no splitting, just for demo)
    W_Q = nn.Linear(d_model, d_model)
    Q = W_Q(x)
    print(f"After projection: {Q.shape}")

    # Reshape to (batch, seq, n_heads, d_head)
    Q_reshaped = Q.view(2, 6, n_heads, d_head)
    print(f"After reshape:    {Q_reshaped.shape}")
    print(f"  Head 0 values:  {Q_reshaped[0, 0, 0, :5]}...")  # First 5 values of head 0 at pos 0
    print(f"  Head 1 values:  {Q_reshaped[0, 0, 1, :5]}...")  # First 5 values of head 1 at pos 0

    # Transpose to (batch, n_heads, seq, d_head)
    Q_transposed = Q_reshaped.transpose(1, 2)
    print(f"\nAfter transpose: {Q_transposed.shape}")
    print(f"  Head 0 (all pos): {Q_transposed[0, 0, :, :5]}")  # All positions for head 0

    # Verify that the values are the same (just rearranged)
    assert torch.allclose(Q_reshaped[0, 0, 0, :], Q_transposed[0, 0, 0, :])
    print(f"\nValues preserved through split: ✓")

visualize_splitting()
# Output: After projection: torch.Size([2, 6, 32])
# Output: After reshape:    torch.Size([2, 6, 4, 8])
# Output:   Head 0 values:  tensor([...])...
# Output:   Head 1 values:  tensor([...])...
# Output: After transpose: torch.Size([2, 4, 6, 8])
# Output:   Head 0 (all pos): tensor([...])
# Output: Values preserved through split: ✓
```

### Example 3: Checking Contiguity After Transpose

```python
def check_memory_layout():
    """Show why contiguous() is needed after transpose."""
    d_model, n_heads = 32, 4
    d_head = d_model // n_heads

    x = torch.randn(2, 6, d_model)
    W = nn.Linear(d_model, d_model)
    Q = W(x)

    # After reshape
    Q_reshaped = Q.view(2, 6, n_heads, d_head)
    print(f"Reshaped contiguous: {Q_reshaped.is_contiguous()}")

    # After transpose
    Q_transposed = Q_reshaped.transpose(1, 2)
    print(f"Transposed contiguous: {Q_transposed.is_contiguous()}")

    # After transpose + contiguous
    Q_contiguous = Q_transposed.contiguous()
    print(f"Transposed + contiguous: {Q_contiguous.is_contiguous()}")

    # Memory layout matters for view operations
    try:
        Q_transposed.view(2, 6, d_model)  # Will fail
        print("view() after transpose: OK")
    except RuntimeError as e:
        print(f"view() after transpose: FAILED - {e}")

    # Use reshape instead (it handles contiguity automatically)
    Q_reshaped2 = Q_transposed.reshape(2, 6, d_model)  # Works
    print(f"reshape() after transpose: OK")

check_memory_layout()
# Output: Reshaped contiguous: True
# Output: Transposed contiguous: False
# Output: Transposed + contiguous: True
# Output: view() after transpose: FAILED - view size is not compatible with input tensor's size...
# Output: reshape() after transpose: OK
```

### Example 4: Computational Efficiency of Batched Heads

```python
def benchmark_splitting_methods():
    """Compare manual splitting with native nn.MultiheadAttention."""
    import time

    d_model, n_heads = 512, 8
    batch, seq_len = 4, 128

    manual_mha = MultiHeadAttentionSplitting(d_model, n_heads)
    native_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    x = torch.randn(batch, seq_len, d_model)

    # Warmup
    for _ in range(10):
        manual_mha(x)
        native_mha(x, x, x)

    # Benchmark manual
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        manual_mha(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    manual_time = (time.time() - start) / 100

    # Benchmark native
    start = time.time()
    for _ in range(100):
        native_mha(x, x, x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    native_time = (time.time() - start) / 100

    print(f"Manual MHA:   {manual_time*1000:.3f} ms")
    print(f"Native MHA:   {native_time*1000:.3f} ms")
    print(f"Ratio:        {manual_time/native_time:.2f}x")

# Uncomment to run
# benchmark_splitting_methods()
```

## Common Mistakes

1. **Forgetting contiguous() after transpose**: After transposing a tensor, calling `view()` will raise an error. Use `reshape()` or call `contiguous().view()`. Modern PyTorch handles this in many cases, but it's a common source of bugs.

2. **Incorrect reshape dimensions**: When reshaping from (batch, seq, d_model) to (batch, seq, n_heads, d_head), ensure that `d_model == n_heads * d_head`. A mismatch will cause a runtime error.

3. **Transposing the wrong dimensions**: The transpose should exchange the seq_len and n_heads dimensions (dimensions 1 and 2), not the batch dimension. The correct call is `transpose(1, 2)` when batch is dimension 0.

4. **Splitting Q, K, V before projection**: The splitting should happen after the linear projections, not before. Applying separate projections per head would be inefficient.

5. **Using split_heads for values differently from keys/queries**: All three — Q, K, V — should be split identically. In grouped-query attention, keys and values may have fewer heads, but they are still split consistently.

## Interview Questions

### Beginner

**Q: How is the input tensor split across multiple attention heads?**

A: After linear projection to (batch, seq_len, d_model), the tensor is reshaped to (batch, seq_len, n_heads, d_head) by splitting the d_model dimension into n_heads groups of size d_head. Then it is transposed to (batch, n_heads, seq_len, d_head) so that each head's data is contiguous for batched matrix multiplication.

### Intermediate

**Q: Why is the transpose operation needed after reshaping for multi-head attention?**

A: The transpose is needed so that each head's queries, keys, and values are in contiguous memory locations. After the transpose, the tensor shape is (batch, n_heads, seq_len, d_head). This allows the attention computation to be performed as a single batched matrix multiplication across all heads, which is much more efficient on GPUs than looping over heads individually.

### Advanced

**Q: Compare the approach of splitting heads via reshape+transpose with splitting via separate linear projections per head (one per head). What are the trade-offs?**

A: Using a single large linear projection followed by reshape+transpose is more efficient because: (1) It uses a single matrix multiplication (large and efficient on GPU) instead of h smaller ones. (2) It requires fewer parameters — \(d_{\text{model}} \times d_{\text{model}}\) total vs. \(h \times (d_{\text{model}} \times d_{\text{head}})\) per head (same total but less overhead). (3) The memory access pattern is more cache-friendly. However, separate projections per head would allow each head to have a different dimension (useful for variable-head architectures), would be more modular for pruning (removing individual heads), and would avoid the need for transpose/contiguous operations. In practice, the single-projection approach is always preferred for efficiency.

## Practice Problems

### Easy

Implement the split_heads and combine_heads functions and verify that applying split_heads followed by combine_heads returns the original tensor (up to floating point precision).

### Medium

Implement multi-head attention without using reshape or transpose. Instead, use a for loop over heads with separate linear projections (one per head). Compare the output with the efficient batched version.

### Hard

Implement "variable head dimensions" where different heads can have different d_head values. The attention for heads with larger d_head should capture more fine-grained patterns.

## Solutions

### Easy Solution

```python
def test_split_combine():
    d_model, n_heads = 32, 4
    d_head = d_model // n_heads
    mha = MultiHeadAttentionSplitting(d_model, n_heads)

    x = torch.randn(2, 6, d_model)

    # Project and split
    Q = mha.W_Q(x)
    Q_split = mha.split_heads(Q)
    Q_combined = mha.combine_heads(Q_split)

    print(f"Original shape:  {Q.shape}")
    print(f"Split shape:     {Q_split.shape}")
    print(f"Combined shape:  {Q_combined.shape}")
    print(f"Original == Combined: {torch.allclose(Q, Q_combined)}")
    # Output: Original shape:  torch.Size([2, 6, 32])
    # Output: Split shape:     torch.Size([2, 4, 6, 8])
    # Output: Combined shape:  torch.Size([2, 6, 32])
    # Output: Original == Combined: True
```

## Related Concepts

- **DL-371: Attention Head**: The individual head concept.
- **DL-373: Attention Head Concatenation**: The reverse operation (combining heads).
- **DL-376: Transformer Dimensionality**: How dimensions relate across components.
- **DL-379: n_heads**: The hyperparameter.
- **DL-377: d_model**: The base dimension that gets split.

## Next Concepts

- DL-373: Attention Head Concatenation — How heads are combined after attention.
- DL-374: FFN Expansion Factor — A different type of dimension transformation.

## Summary

Multi-head attention splitting is the tensor manipulation operation that divides the projected query, key, and value tensors into multiple heads for parallel attention computation. It involves reshaping the (batch, seq_len, d_model) tensor to (batch, seq_len, n_heads, d_head) and transposing to (batch, n_heads, seq_len, d_head). The reverse operation (combine_heads) performs the inverse transformation after attention. This splitting is essential for the efficient parallel computation of multi-head attention.

## Key Takeaways

1. Splitting partitions the d_model dimension into n_heads groups of d_head each.
2. The reshape splits d_model; the transpose makes heads the second dimension.
3. Batched attention across all heads is more efficient than per-head loops.
4. contiguous() is often needed after transpose for view() to work.
5. The combine_heads operation reverses the split for output projection.
6. Correct splitting is essential for implementing multi-head attention.
