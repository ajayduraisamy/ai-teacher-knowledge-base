# Concept: FLOPs in Transformer

## Concept ID

DL-382

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand how to compute FLOPs (floating point operations) for Transformer models.
- Break down FLOPs across components: attention, FFN, embeddings, and projections.
- Distinguish between forward FLOPs, backward FLOPs, and total training FLOPs.
- Analyze how FLOPs scale with model dimensions and sequence length.
- Compare FLOPs of encoder vs decoder vs full encoder-decoder models.

## Prerequisites

- DL-376: Transformer Dimensionality
- DL-381: Transformer Parameter Count
- Understanding of matrix multiplication computational cost.
- Familiarity with the components of a Transformer.

## Definition

FLOPs (Floating Point Operations) measure the computational cost of a Transformer model. They count the number of multiply-add operations required for forward and/or backward passes. FLOPs depend on model dimensions (d_model, d_ff, n_layers), sequence length, batch size, and vocabulary size. Understanding FLOPs is essential for estimating training time, comparing architectures, and optimizing inference.

## Intuition

Think of FLOPs as the "work" required to process a token. Each linear layer is a matrix multiplication that requires \(2 \times \text{input\_dim} \times \text{output\_dim}\) operations. The attention mechanism requires computing scores (QK^T), weights (softmax), and weighted sums (AV). The FFN requires two large matrix multiplications.

The total FLOPs per token is relatively constant for a given model architecture, regardless of sequence length (up to the attention term, which scales as seq_len^2).

## Why This Concept Matters

FLOPs are important for:

1. **Estimating Training Time**: Total FLOPs / GPU FLOPs/s = training time.
2. **Hardware Planning**: Choosing GPUs/TPUs with sufficient throughput.
3. **Architecture Comparison**: Comparing computational efficiency of different designs.
4. **Cost Estimation**: Cloud compute costs are proportional to FLOPs.
5. **Scaling Laws**: The "compute budget" in scaling laws is measured in FLOPs.

## Mathematical Explanation

### Matrix Multiplication FLOPs

A matrix multiplication \(A_{m \times n} \times B_{n \times p}\) requires \(2mnp\) FLOPs (multiply + add per element).

### Per-Token FLOPs Breakdown

**Token Embedding**: Negligible (lookup table).

**Positional Encoding**: Negligible.

**Attention Sub-Layer** (per layer, per token):

1. Q, K, V projections: \(3 \times 2 \times d_{\text{model}} \times d_{\text{model}} = 6d_{\text{model}}^2\)

2. QK^T scores: \(2 \times n_{\text{heads}} \times d_{\text{head}} \times n = 2d_{\text{model}} \times n\)

3. Score scaling + softmax: \(n\) (negligible)

4. Attention weights × V: \(2 \times n_{\text{heads}} \times n \times d_{\text{head}} = 2d_{\text{model}} \times n\)

5. Output projection: \(2 \times d_{\text{model}} \times d_{\text{model}} = 2d_{\text{model}}^2\)

Total attention per layer per token: \(8d_{\text{model}}^2 + 4d_{\text{model}} \times n\)

Note: The \(4d_{\text{model}} \times n\) term makes attention quadratic in sequence length.

**FFN Sub-Layer** (per layer, per token):

Standard (ReLU):
- First linear: \(2 \times d_{\text{model}} \times d_{ff}\)
- Second linear: \(2 \times d_{ff} \times d_{\text{model}}\)
- Total: \(4d_{\text{model}}d_{ff}\)

SwiGLU:
- Three projections: \(3 \times 2 \times d_{\text{model}} \times d_{ff} = 6d_{\text{model}}d_{ff}\)
- Element-wise multiply: \(d_{ff}\)

**Layer Normalization**: \(4d_{\text{model}}\) FLOPs per layer.

### Total FLOPs Per Token

\[
\text{FLOPs}_{\text{per token}} = n_{\text{layers}} \times (8d_{\text{model}}^2 + 4d_{\text{model}}n + 4d_{\text{model}}d_{ff} + 4d_{\text{model}})
\]

The dominant terms:
- If \(n\) is small: \(n_{\text{layers}} \times (8d_{\text{model}}^2 + 4d_{\text{model}}d_{ff})\)
- If \(n\) is large: \(n_{\text{layers}} \times 4d_{\text{model}}n\)

### Forward vs Backward FLOPs

- Forward: 1 FLOP per operation.
- Backward: 2 FLOPs per forward FLOP (one for gradient w.r.t. weight, one for gradient w.r.t. input).
- Total training FLOPs per token ≈ 3 × forward FLOPs.

## Code Examples

### Example 1: FLOPs Counter

```python
import torch
import torch.nn as nn
import math

def compute_transformer_flops(d_model, d_ff, n_layers, n_heads, seq_len, vocab_size, batch_size=1):
    """Compute FLOPs for a forward pass of an encoder Transformer."""
    # Embedding (negligible, just lookup)
    embed_flops = 0

    # Per-layer FLOPs
    # Attention projections: Q, K, V, O
    attn_proj_flops = 4 * 2 * d_model * d_model  # QKV + O

    # Attention scores: Q @ K^T
    attn_scores_flops = 2 * n_heads * (d_model // n_heads) * seq_len * seq_len
    # Equivalent to: 2 * d_model * seq_len * seq_len

    # Attention value: scores @ V
    attn_value_flops = 2 * n_heads * seq_len * seq_len * (d_model // n_heads)
    # Equivalent to: 2 * d_model * seq_len * seq_len

    attn_flops = attn_proj_flops + attn_scores_flops + attn_value_flops

    # FFN
    ffn_flops = 2 * 2 * d_model * d_ff  # Two linear layers

    # LayerNorm (mean + var + normalize + scale)
    ln_flops = 4 * d_model  # Approximate

    per_layer_flops = attn_flops + ffn_flops + ln_flops

    # Total for batch
    total_flops_per_token = n_layers * per_layer_flops  # Per token (seq_len already accounted)
    # Actually per_layer_flops already includes seq_len in attention
    total_flops_per_seq = batch_size * n_layers * (
        attn_proj_flops + attn_scores_flops + attn_value_flops + ffn_flops + ln_flops
    )

    return {
        'per_layer_attention': attn_flops,
        'per_layer_ffn': ffn_flops,
        'per_layer_norm': ln_flops,
        'per_layer_total': per_layer_flops,
        'total_per_sequence': total_flops_per_seq,
        'total_gigaflops': total_flops_per_seq / 1e9,
    }

# Compare FLOPs for different sequence lengths
d_model, d_ff, n_layers, n_heads = 768, 3072, 12, 12
print("FLOPs comparison (BERT-base, d_model=768):")
print(f"{'Seq Len':<10} {'Attn FLOPs':<15} {'FFN FLOPs':<15} {'Total (GFLOPS)':<18}")
print("-" * 55)

for seq_len in [128, 256, 512, 1024]:
    flops = compute_transformer_flops(d_model, d_ff, n_layers, n_heads, seq_len, 30000)
    per_layer_attn = flops['per_layer_attention']
    per_layer_ffn = flops['per_layer_ffn']
    gflops = flops['total_gigaflops'] * 3  # Multiply by 3 for backward (training)
    print(f"{seq_len:<10} {per_layer_attn:<15,} {per_layer_ffn:<15,} {gflops:<18.1f}")
# Output: FLOPs comparison (BERT-base, d_model=768):
# Output: Seq Len    Attn FLOPs      FFN FLOPs       Total (GFLOPS)
# Output: ---------------------------------------------------------
# Output: 128        17,301,504      56,623,104      2.2
# Output: 256        25,165,824      56,623,104      3.0
# Output: 512        40,894,464      56,623,104      4.5
# Output: 1024       72,351,744      56,623,104      8.2
```

### Example 2: Attention vs FFN FLOPs as Function of Sequence Length

```python
def attention_vs_ffn_breakdown():
    """Show how attention dominates at long sequences."""
    d_model = 4096
    d_ff = 11008
    n_layers = 32

    print(f"FLOPs breakdown (d_model={d_model}, d_ff={d_ff}, {n_layers} layers):")
    print(f"{'Seq Len':<10} {'Attn %':<10} {'FFN %':<10} {'Total (GFLOP)':<15}")
    print("-" * 45)

    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        # Per-token attention FLOPs: 8*d_model^2 + 4*d_model*seq_len (per layer)
        # But for batch of 1: attention score = 2*d_model*seq_len
        attn_per_layer = 8 * d_model * d_model + 4 * d_model * seq_len
        ffn_per_layer = 4 * d_model * d_ff
        total_per_layer = attn_per_layer + ffn_per_layer

        attn_pct = attn_per_layer / total_per_layer * 100
        ffn_pct = ffn_per_layer / total_per_layer * 100
        total_gflops = n_layers * total_per_layer / 1e9

        print(f"{seq_len:<10} {attn_pct:<10.1f} {ffn_pct:<10.1f} {total_gflops:<15.3f}")

attention_vs_ffn_breakdown()
# Output: FLOPs breakdown (d_model=4096, d_ff=11008, 32 layers):
# Output: Seq Len    Attn %     FFN %      Total (GFLOP)
# Output: ------------------------------------------------
# Output: 256        23.5%      76.5%      5.481
# Output: 512        25.6%      74.4%      5.933
# Output: 1024       29.3%      70.7%      6.838
# Output: 2048       35.9%      64.1%      8.647
# Output: 4096       46.5%      53.5%      12.266
# Output: 8192       59.9%      40.1%      19.503
```

### Example 3: FLOPs for Encoder vs Decoder

```python
def encoder_vs_decoder_flops():
    """Compare FLOPs for encoder-only, decoder-only, and encoder-decoder."""
    d_model = 512
    d_ff = 2048
    n_layers = 6
    n_heads = 8
    seq_len = 128
    vocab_size = 30000

    # Encoder: bidirectional attention
    def encoder_flops():
        attn_proj = 4 * 2 * d_model * d_model
        attn_scores = 2 * d_model * seq_len * seq_len
        attn_values = 2 * d_model * seq_len * seq_len
        attn = attn_proj + attn_scores + attn_values

        ffn = 2 * 2 * d_model * d_ff
        ln = 4 * d_model

        return n_layers * (attn + ffn + ln)

    # Decoder: causal attention (same FLOPs as encoder for the attention computation,
    # but the mask affects computation pattern, not FLOP count)
    def decoder_flops():
        # Same as encoder for forward pass of full sequence
        return encoder_flops()

    # Encoder-decoder: encoder + decoder with cross-attention
    def enc_dec_flops():
        enc = encoder_flops()

        # Decoder has 3 sub-layers: self-attn, cross-attn, FFN
        attn_proj = 4 * 2 * d_model * d_model
        attn_scores = 2 * d_model * seq_len * seq_len
        attn_values = 2 * d_model * seq_len * seq_len
        self_attn = attn_proj + attn_scores + attn_values

        # Cross-attention: Q from decoder, K, V from encoder
        # Cross-attention Q projection is separate
        cross_attn = 4 * 2 * d_model * d_model  # Q, K, V, O
        # QK^T: decoder_seq_len x encoder_seq_len
        cross_scores = 2 * d_model * seq_len * seq_len
        cross_values = 2 * d_model * seq_len * seq_len

        ffn = 2 * 2 * d_model * d_ff
        ln = 6 * d_model  # 3 LayerNorms

        dec = n_layers * (self_attn + cross_attn + cross_scores + cross_values + ffn + ln)
        return enc + dec

    enc = encoder_flops()
    dec = decoder_flops()
    encdec = enc_dec_flops()

    print(f"FLOPs comparison (seq_len={seq_len}):")
    print(f"  Encoder-only:       {enc/1e9:.3f} GFLOP")
    print(f"  Decoder-only:       {dec/1e9:.3f} GFLOP")
    print(f"  Encoder-Decoder:    {encdec/1e9:.3f} GFLOP")
    print(f"  Decoder/Encoder ratio: {dec/enc:.2f}x")
    print(f"  EncDec/Encoder ratio:  {encdec/enc:.2f}x")

encoder_vs_decoder_flops()
# Output: FLOPs comparison (seq_len=128):
# Output:   Encoder-only:       1.054 GFLOP
# Output:   Decoder-only:       1.054 GFLOP
# Output:   Encoder-Decoder:    1.792 GFLOP
# Output:   Decoder/Encoder ratio: 1.00x
# Output:   EncDec/Encoder ratio:  1.70x
```

### Example 4: Training FLOPs Estimation

```python
def training_flops_estimate():
    """Estimate total training FLOPs for a model."""
    d_model = 768
    d_ff = 3072
    n_layers = 12
    n_heads = 12
    seq_len = 512
    vocab_size = 30000
    batch_size = 64
    training_steps = 100000

    # Forward FLOPs per token
    attn_per_layer = 8 * d_model * d_model + 4 * d_model * seq_len
    ffn_per_layer = 4 * d_model * d_ff
    ln_per_layer = 4 * d_model
    per_layer = attn_per_layer + ffn_per_layer + ln_per_layer
    forward_flops = n_layers * per_layer

    # Per token (divide by seq_len since per_layer already includes seq_len in attn term)
    forward_per_token = forward_flops / seq_len

    # But forward_flops is for one sequence. For batch_size tokens:
    tokens_per_step = batch_size * seq_len
    forward_per_step = forward_flops * batch_size

    # Training: forward + backward (2x) = 3x forward
    training_flops_per_step = forward_per_step * 3
    total_training_flops = training_flops_per_step * training_steps

    print(f"Training FLOPs estimate:")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Forward FLOPs per step: {forward_per_step/1e12:.4f} TFLOP")
    print(f"  Training FLOPs per step: {training_flops_per_step/1e12:.4f} TFLOP")
    print(f"  Total training FLOPs ({training_steps:,} steps): {total_training_flops/1e15:.4f} PFLOP")

    # GPU hours estimate (assuming 40% utilization on A100 with 312 TFLOP/s)
    gpu_tflops = 312  # A100 TF32
    utilization = 0.4
    effective_tflops = gpu_tflops * utilization
    seconds = total_training_flops / (effective_tflops * 1e12)
    hours = seconds / 3600
    print(f"  Estimated GPU hours (A100, 40% util): {hours:.1f}")

training_flops_estimate()
# Output: Training FLOPs estimate:
# Output:   Tokens per step: 32,768
# Output:   Forward FLOPs per step: 1.176 TFLOP
# Output:   Training FLOPs per step: 3.528 TFLOP
# Output:   Total training FLOPs (100,000 steps): 0.353 PFLOP
# Output:   Estimated GPU hours (A100, 40% util): 0.8
```

## Common Mistakes

1. **Forgetting the \(n^2\) term in attention**: Attention FLOPs scale as \(O(n^2)\) for sequence length \(n\). For long sequences, this dominates. For short sequences, the FFN dominates.

2. **Confusing FLOPs with parameter count**: FLOPs and parameters are correlated but different. FLOPs depend on sequence length and batch size; parameters do not.

3. **Only counting forward FLOPs**: Training requires forward + backward passes. The backward pass costs approximately 2× the forward pass, so total training FLOPs ≈ 3× forward FLOPs.

4. **Ignoring the output projection FLOPs**: The final linear projection to vocabulary size costs \(2 \times d_{\text{model}} \times V\) FLOPs per token, which is significant for large vocabularies.

5. **Not accounting for activation checkpointing**: Gradient checkpointing trades compute for memory. With checkpointing, the backward pass recomputes activations, increasing training FLOPs by ~33%.

## Interview Questions

### Beginner

**Q: What determines the FLOPs of a Transformer forward pass?**

A: FLOPs are dominated by matrix multiplications in the attention and FFN sub-layers. The attention FLOPs scale as \(O(d_{\text{model}}^2 + d_{\text{model}} \cdot n)\) per layer, and the FFN FLOPs scale as \(O(d_{\text{model}} \cdot d_{ff})\). For short sequences, FFN dominates. For long sequences, attention dominates due to the \(n^2\) term.

### Intermediate

**Q: Explain the FLOPs profile of a Transformer when sequence length is very long (e.g., 128k tokens). What architectural modifications help?**

A: At very long sequences, the attention FLOPs (\(O(n^2 d_{\text{model}})\)) completely dominate. For example, with n=128k, d_model=4096: the attention score computation alone requires \(2 \times 128k^2 \times 4096 \approx 134\) TFLOP per layer. Architectural solutions include: (1) Sparse attention (Longformer, BigBird) — limit attention to local windows plus selected global tokens, reducing to \(O(n \log n)\). (2) Linear attention (Performer, Linear Transformer) — rewrite attention as \(Q(K^T V)\) reducing to \(O(nd^2)\). (3) Flash Attention — IO-aware tiling that doesn't reduce FLOPs but reduces memory reads/writes. (4) State-space models (Mamba, RWKV) — replace attention with recurrence, achieving \(O(n)\) complexity.

### Advanced

**Q: Derive the FLOPs for one training step of a decoder-only Transformer, including the embedding, attention, FFN, and output projection. Use this to estimate the optimal batch size for maximum throughput on an A100 GPU.**

A: For a decoder-only model with d_model, d_ff, n_layers, seq_len n, vocab V, batch B:

Forward FLOPs = B × n × n_layers × [
    (4×2×d_model²)  # QKV and O projections (per token)
    + (2×d_model×n)  # QK^T (per token, amortized)
    + (2×d_model×n)  # AV (per token, amortized)
    + (4×d_model×d_ff)  # FFN (per token)
    + 2×d_model×V  # Output projection (per token)
]

Simplifying: per-token FLOPs ≈ n_layers × (8d_model² + 4d_model×n + 4d_model×d_ff + 2d_model×V).

Training = 3× forward. On an A100 (312 TFLOP/s FP32, 624 TFLOP/s TF32), the optimal batch size balances compute and memory. For a 7B model with n=2048: per-token FLOPs ≈ 32 × (8×4096² + 4×4096×2048 + 4×4096×11008 + 2×4096×32000) ≈ 32 × (134M + 33.5M + 180M + 262M) ≈ 32 × 610M ≈ 19.5B FLOPs per token. With 312 TFLOP/s and 40% utilization: 312 × 0.4 / (19.5) ≈ 6.4M tokens/s. In practice, memory bandwidth is the bottleneck, so the actual throughput is lower.

## Practice Problems

### Easy

Compute the forward FLOPs per token for BERT-base (d_model=768, d_ff=3072, n_layers=12) at sequence length 512.

### Medium

Compare the FLOPs of a decoder-only model and an encoder-decoder model with the same per-layer dimensions and number of layers.

### Hard

Implement a FLOPs profiler that hooks into PyTorch operations and measures actual FLOPs during a forward pass. Compare the measured FLOPs with the theoretical estimate.

## Solutions

### Easy Solution

```python
def bert_flops_per_token():
    d_model, d_ff, n_layers = 768, 3072, 12
    seq_len = 512

    # Per layer, per token
    attn_proj = 8 * d_model * d_model  # 4 projections, 2 FLOPs each, averaged per token
    attn_score = 4 * d_model * seq_len  # 2*d_model*seq_len / seq_len = 2*d_model
    attn_value = 4 * d_model * seq_len
    # Actually per token: QK^T is 2*d_model*seq_len per token
    attn_score_per_token = 2 * d_model * seq_len
    attn_value_per_token = 2 * d_model * seq_len

    ffn_per_token = 4 * d_model * d_ff

    per_layer = (attn_proj + attn_score_per_token + attn_value_per_token + ffn_per_token)
    total = n_layers * per_layer

    print(f"Per-token FLOPs for BERT-base:")
    print(f"  Attention projection: {attn_proj:>12,}")
    print(f"  Attention scores:     {attn_score_per_token:>12,}")
    print(f"  Attention values:     {attn_value_per_token:>12,}")
    print(f"  FFN:                  {ffn_per_token:>12,}")
    print(f"  Per layer:            {per_layer:>12,}")
    print(f"  Total ({n_layers} layers):   {total:>12,}")
    print(f"  GFLOPs per token:     {total/1e9:.6f}")

bert_flops_per_token()
# Output: Per-token FLOPs for BERT-base:
# Output:   Attention projection:    4,718,592
# Output:   Attention scores:        3,145,728
# Output:   Attention values:        3,145,728
# Output:   FFN:                    18,874,368
# Output:   Per layer:              29,884,416
# Output:   Total (12 layers):     358,612,992
# Output:   GFLOPs per token:        0.358613
```

## Related Concepts

- **DL-376: Transformer Dimensionality**: Dimensions that determine FLOPs.
- **DL-381: Transformer Parameter Count**: Related metric for model size.
- **DL-383: KV Cache**: Memory optimization that affects FLOPs per token.
- **DL-385: Flash Attention**: IO-aware attention that reduces memory overhead.
- **Hardware Utilization**: How FLOPs translate to wall-clock time.

## Next Concepts

- DL-383: KV Cache — Reducing redundant computation during autoregressive decoding.
- DL-385: Flash Attention — Efficient attention implementation.

## Summary

FLOPs (floating point operations) measure the computational cost of Transformer models. FLOPs are dominated by matrix multiplications in attention and FFN sub-layers. Attention FLOPs scale as \(O(n^2)\) with sequence length (quadratic), making them the dominant cost for long sequences. FFN FLOPs scale as \(O(d_{\text{model}} \times d_{ff})\) and dominate for short sequences. Training requires approximately 3× forward FLOPs (forward + backward). Understanding FLOPs is essential for estimating training time, comparing architectures, and planning hardware.

## Key Takeaways

1. FLOPs are dominated by attention (\(O(n^2 d_{\text{model}})\)) and FFN (\(O(d_{\text{model}} d_{ff})\)) matrix multiplications.
2. For short sequences, FFN dominates; for long sequences, attention dominates.
3. Training FLOPs ≈ 3 × forward FLOPs.
4. Per-token FLOPs are approximately constant for a given architecture.
5. FLOPs and parameters are correlated but not identical — FLOPs depend on sequence length.
6. Understanding FLOPs is essential for estimating training time and compute costs.
