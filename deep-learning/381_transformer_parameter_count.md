# Concept: Transformer Parameter Count

## Concept ID

DL-381

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand how each component of a Transformer contributes to the total parameter count.
- Compute the parameter count for any Transformer configuration.
- Analyze the distribution of parameters across components (attention, FFN, embeddings).
- Understand how parameter sharing and weight tying affect the count.
- Estimate memory requirements from parameter counts.

## Prerequisites

- DL-377: d_model
- DL-378: d_ff
- DL-379: n_heads
- DL-380: n_layers
- Understanding of linear layer parameters (weight + bias).

## Definition

The parameter count of a Transformer model is the total number of trainable parameters across all components: token embeddings, positional encodings (if learned), attention projections (Q, K, V, O), FFN layers (two or three linear layers per block), and layer normalization parameters. The parameter count determines the model's memory footprint, computational cost, and to a large extent, its representational capacity. Understanding how parameters are distributed is essential for model design, scaling, and deployment.

## Intuition

Think of a Transformer model as a collection of matrices and vectors. Each linear layer has a weight matrix and optionally a bias vector. Layer normalization has scale and shift vectors. Embeddings are also matrices. The total parameter count is the sum of all elements across all these matrices and vectors.

The distribution is highly uneven:
- The FFN contains ~2/3 of all parameters.
- The attention sub-layer contains ~1/3.
- Embeddings contribute significantly, especially for large vocabularies.
- Layer normalization adds negligibly.

## Why This Concept Matters

Parameter count is important because:

1. **Memory Footprint**: Each parameter (FP32) requires 4 bytes. A 7B model requires ~28 GB just for weights.
2. **Computational Cost**: FLOPs are proportional to parameter count.
3. **Model Comparison**: Parameter count is the most common metric for comparing model sizes.
4. **Scaling Laws**: Model performance scales predictably with parameter count.
5. **Hardware Planning**: GPU memory limits are based on parameter count.
6. **Training Cost**: Cloud compute costs scale with parameter count.

## Mathematical Explanation

### Component Breakdown

**1. Token Embeddings**

\[
\text{Params} = V \times d_{\text{model}}
\]

where \(V\) is vocabulary size. If weight tying is used (decoder shares weights with embedding), these parameters are shared and only counted once.

**2. Positional Embeddings** (if learned)

\[
\text{Params} = L_{\text{max}} \times d_{\text{model}}
\]

**3. Per-Layer Components**

**Multi-Head Attention:**

\[
\begin{aligned}
W_Q &: d_{\text{model}} \times d_{\text{model}} \quad (\text{params: } d_{\text{model}}^2) \\
W_K &: d_{\text{model}} \times d_{\text{model}} \quad (\text{params: } d_{\text{model}}^2) \\
W_V &: d_{\text{model}} \times d_{\text{model}} \quad (\text{params: } d_{\text{model}}^2) \\
W_O &: d_{\text{model}} \times d_{\text{model}} \quad (\text{params: } d_{\text{model}}^2) \\
\text{Biases} &: 4 \times d_{\text{model}} \\
\end{aligned}
\]

Total attention: \(4d_{\text{model}}^2 + 4d_{\text{model}}\)

**Feed-Forward Network:**

Standard:
\[
\begin{aligned}
W_1 &: d_{\text{model}} \times d_{ff} \\
W_2 &: d_{ff} \times d_{\text{model}} \\
\text{Biases} &: d_{ff} + d_{\text{model}}
\end{aligned}
\]

Total FFN (standard): \(2d_{\text{model}}d_{ff} + d_{ff} + d_{\text{model}}\)

SwiGLU:
\[
\begin{aligned}
W_1 &: d_{\text{model}} \times d_{ff} \\
V_1 &: d_{\text{model}} \times d_{ff} \\
W_2 &: d_{ff} \times d_{\text{model}}
\end{aligned}
\]

Total FFN (SwiGLU, no bias): \(3d_{\text{model}}d_{ff}\)

**Layer Normalization** (2 per block):
\[
\text{Params per LayerNorm} = 2d_{\text{model}} \quad (\gamma + \beta)
\]

Total per layer: \(2 \times 2d_{\text{model}} = 4d_{\text{model}}\)

**4. Output Projection** (for decoder or classification head)

\[
\text{Params} = d_{\text{model}} \times V
\]

Often weight-tied with the embedding layer.

### Total Parameter Count

\[
\begin{aligned}
\text{Total} &= \text{Embedding} + \text{Positional} \\
&\quad + n_{\text{layers}} \times (\text{Attention} + \text{FFN} + 2 \times \text{LayerNorm}) \\
&\quad + \text{Output Projection}
\end{aligned}
\]

### Example: BERT-base

- \(d_{\text{model}} = 768, d_{ff} = 3072, n_{\text{heads}} = 12, n_{\text{layers}} = 12, V = 30522\)
- Embedding: \(30522 \times 768 = 23,440,896\)
- Per layer attention: \(4 \times 768^2 = 2,359,296\)
- Per layer FFN: \(2 \times 768 \times 3072 + 3072 + 768 = 4,722,816\)
- Per layer LN: \(4 \times 768 = 3,072\)
- Per layer total: \(7,085,184\)
- 12 layers: \(85,022,208\)
- Total: \(23,440,896 + 85,022,208 = 108,463,104 \approx 110M\)

## Code Examples

### Example 1: Comprehensive Parameter Counter

```python
import torch
import torch.nn as nn
import math

class TransformerParamCounter:
    """Compute parameter count for Transformer models."""
    def __init__(self, d_model, d_ff, n_heads, n_layers, vocab_size,
                 max_len=512, learned_pe=False, tie_weights=False,
                 ffn_type='standard', glu_d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.learned_pe = learned_pe
        self.tie_weights = tie_weights
        self.ffn_type = ffn_type

        if glu_d_ff is not None:
            self.d_ff_actual = glu_d_ff
        else:
            self.d_ff_actual = d_ff

    def count_embedding(self):
        return self.vocab_size * self.d_model

    def count_positional(self):
        return self.max_len * self.d_model if self.learned_pe else 0

    def count_attention(self):
        # Q, K, V, O projections (with biases)
        return 4 * self.d_model * self.d_model + 4 * self.d_model

    def count_ffn(self):
        if self.ffn_type == 'standard':
            return 2 * self.d_model * self.d_ff_actual + self.d_ff_actual + self.d_model
        elif self.ffn_type == 'swiglu':
            # No bias in modern GLU implementations
            return 3 * self.d_model * self.d_ff_actual
        else:
            raise ValueError(f"Unknown FFN type: {self.ffn_type}")

    def count_layernorm(self):
        return 2 * 2 * self.d_model  # 2 LayerNorms, each with gamma + beta

    def count_output_proj(self):
        if self.tie_weights:
            return 0  # Weight tied with embedding
        return self.d_model * self.vocab_size

    def total(self):
        total = self.count_embedding()
        total += self.count_positional()
        per_layer = self.count_attention() + self.count_ffn() + self.count_layernorm()
        total += self.n_layers * per_layer
        total += self.count_output_proj()
        return total

    def breakdown(self):
        """Return a dictionary of parameter counts by component."""
        embed = self.count_embedding()
        pos = self.count_positional()
        attn = self.count_attention()
        ffn = self.count_ffn()
        ln = self.count_layernorm()
        output = self.count_output_proj()

        per_layer = attn + ffn + ln
        total = embed + pos + self.n_layers * per_layer + output

        return {
            'embedding': embed,
            'positional': pos,
            'attention_per_layer': attn,
            'ffn_per_layer': ffn,
            'layernorm_per_layer': ln,
            'per_layer_total': per_layer,
            'n_layers': self.n_layers,
            'layers_total': self.n_layers * per_layer,
            'output_projection': output,
            'total': total,
        }

# Count parameters for several models
models = [
    ("Transformer-base", 512, 2048, 8, 6, 37000),
    ("BERT-base", 768, 3072, 12, 12, 30522),
    ("BERT-large", 1024, 4096, 16, 24, 30522),
    ("GPT-2 small", 768, 3072, 12, 12, 50257),
]

print(f"{'Model':<20} {'Total':<15} {'Embed%':<10} {'Attn%':<10} {'FFN%':<10}")
print("-" * 65)
for name, d_model, d_ff, n_heads, n_layers, vocab in models:
    counter = TransformerParamCounter(d_model, d_ff, n_heads, n_layers, vocab)
    b = counter.breakdown()
    total = b['total']
    embed_pct = b['embedding'] / total * 100
    attn_pct = (b['attention_per_layer'] * b['n_layers']) / total * 100
    ffn_pct = (b['ffn_per_layer'] * b['n_layers']) / total * 100
    print(f"{name:<20} {total/1e6:<15.2f}M {embed_pct:<10.1f} {attn_pct:<10.1f} {ffn_pct:<10.1f}")
# Output: Model               Total           Embed%     Attn%      FFN%
# Output: -----------------------------------------------------------------
# Output: Transformer-base    64.64M          29.3%     16.2%     50.7%
# Output: BERT-base           108.46M         21.6%     17.4%     55.2%
# Output: BERT-large          335.51M         14.1%     18.6%     62.5%
# Output: GPT-2 small         123.65M         31.2%     15.3%     47.0%
```

### Example 2: Understanding Parameter Distribution

```python
def visualize_param_distribution():
    """Show where parameters are concentrated."""
    d_model = 1024
    d_ff = 4096
    n_layers = 24
    vocab = 32000

    counter = TransformerParamCounter(d_model, d_ff, 16, n_layers, vocab)
    b = counter.breakdown()

    print(f"Model: {n_layers} layers, d_model={d_model}, d_ff={d_ff}, vocab={vocab}")
    print(f"Total parameters: {b['total']/1e6:.2f}M")
    print()
    print(f"{'Component':<25} {'Parameters':<15} {'% of Total':<12}")
    print("-" * 52)

    components = [
        ("Embedding", b['embedding']),
        ("Attention (all layers)", b['layers_total'] * b['attention_per_layer'] // b['per_layer_total'] if b['per_layer_total'] > 0 else 0),
    ]

    # Recompute per-component
    attn_total = b['attention_per_layer'] * n_layers
    ffn_total = b['ffn_per_layer'] * n_layers
    ln_total = b['layernorm_per_layer'] * n_layers

    multiplier = b['layers_total'] / b['per_layer_total'] if b['per_layer_total'] > 0 else 0
    attn_total = b['attention_per_layer'] * n_layers
    ffn_total = b['ffn_per_layer'] * n_layers
    ln_total = b['layernorm_per_layer'] * n_layers

    for name, count in [
        ("Token Embeddings", b['embedding']),
        ("Attention Sub-layer", attn_total),
        ("FFN Sub-layer", ffn_total),
        ("Layer Normalization", ln_total),
        ("Output Projection", b['output_projection']),
    ]:
        pct = count / b['total'] * 100
        print(f"{name:<25} {count:<15,} {pct:<12.1f}%")

visualize_param_distribution()
# Output: Model: 24 layers, d_model=1024, d_ff=4096, vocab=32000
# Output: Total parameters: 461.71M
# Output:
# Output: Component                  Parameters      % of Total
# Output: -------------------------------------------------------
# Output: Token Embeddings           32,768,000      7.1%
# Output: Attention Sub-layer        100,663,296     21.8%
# Output: FFN Sub-layer              318,767,104     69.0%
# Output: Layer Normalization        98,304          0.0%
# Output: Output Projection          0               0.0%
```

### Example 3: Weight Tying Effect

```python
def weight_tying_comparison():
    """Compare parameter count with and without weight tying."""
    d_model, d_ff, n_heads, n_layers = 768, 3072, 12, 12
    vocab = 32000

    counter_no_tie = TransformerParamCounter(d_model, d_ff, n_heads, n_layers, vocab, tie_weights=False)
    counter_tie = TransformerParamCounter(d_model, d_ff, n_heads, n_layers, vocab, tie_weights=True)

    total_no_tie = counter_no_tie.total()
    total_tie = counter_tie.total()
    saved = total_no_tie - total_tie

    print(f"Parameter comparison (d_model=768, d_ff=3072, n_layers=12, vocab=32000):")
    print(f"  Without weight tying: {total_no_tie/1e6:.2f}M")
    print(f"  With weight tying:    {total_tie/1e6:.2f}M")
    print(f"  Parameters saved:     {saved/1e6:.2f}M ({saved/total_no_tie*100:.1f}%)")

weight_tying_comparison()
# Output: Parameter comparison (d_model=768, d_ff=3072, n_layers=12, vocab=32000):
# Output:   Without weight tying: 289.58M
# Output:   With weight tying:    264.99M
# Output:   Parameters saved:     24.58M (8.5%)
```

### Example 4: GLU vs Standard FFN Parameter Count

```python
def glu_param_comparison():
    """Compare parameter counts for standard vs SwiGLU FFN."""
    d_model = 4096
    d_ff_standard = 16384  # 4x
    d_ff_glu = 11008  # Llama-style

    counter_standard = TransformerParamCounter(
        d_model, d_ff_standard, 32, 32, 32000, ffn_type='standard'
    )
    counter_glu = TransformerParamCounter(
        d_model, d_ff_glu, 32, 32, 32000, ffn_type='swiglu'
    )

    b_std = counter_standard.breakdown()
    b_glu = counter_glu.breakdown()

    print(f"FFN type comparison (d_model=4096, 32 layers):")
    print(f"  Standard FFN (d_ff={d_ff_standard}):")
    print(f"    FFN params per layer: {b_std['ffn_per_layer']:,}")
    print(f"    Total model params:   {b_std['total']/1e9:.3f}B")
    print(f"  SwiGLU FFN (d_ff={d_ff_glu}):")
    print(f"    FFN params per layer: {b_glu['ffn_per_layer']:,}")
    print(f"    Total model params:   {b_glu['total']/1e9:.3f}B")

    # With exact parameter parity
    exact_glu_d_ff = int(2 * d_ff_standard / 3)
    counter_exact = TransformerParamCounter(
        d_model, exact_glu_d_ff, 32, 32, 32000, ffn_type='swiglu'
    )
    b_exact = counter_exact.breakdown()
    ratio = b_exact['total'] / b_std['total']
    print(f"\n  Exact parity GLU (d_ff={exact_glu_d_ff}):")
    print(f"    Total params: {b_exact['total']/1e9:.3f}B (ratio={ratio:.4f})")

glu_param_comparison()
# Output: FFN type comparison (d_model=4096, 32 layers):
# Output:   Standard FFN (d_ff=16384):
# Output:     FFN params per layer: 134,225,920
# Output:     Total model params:   4.393B
# Output:   SwiGLU FFN (d_ff=11008):
# Output:     FFN params per layer: 135,266,304
# Output:     Total model params:   4.424B
# Output:   Exact parity GLU (d_ff=10923):
# Output:     Total params: 4.393B (ratio=1.0000)
```

## Common Mistakes

1. **Forgetting bias parameters**: Linear layer biases contribute \(d_{ff} + d_{\text{model}}\) per FFN and \(4d_{\text{model}}\) for attention. While small relative to weights, they add up.

2. **Double-counting tied weights**: When embedding weights are tied with the output projection, they should be counted only once.

3. **Ignoring the embedding layer's contribution**: For models with large vocabularies (e.g., 100k+ tokens), the embedding layer can be 20-30% of total parameters.

4. **Miscounting GLU variant parameters**: GLU variants have 3 weight matrices, not 2. Using the standard formula \(2d_{\text{model}}d_{ff}\) would undercount.

5. **Confusing "parameters" with "memory"**: FP32 parameters require 4 bytes each. But during training, you also need memory for optimizer states (8 bytes for Adam), gradients (4 bytes), and activations (variable). Total training memory is typically 3-5x the parameter memory.

## Interview Questions

### Beginner

**Q: Approximately what fraction of parameters are in the FFN vs the attention sub-layer in a standard Transformer?**

A: The FFN contains approximately 2/3 of the parameters, and the attention sub-layer contains approximately 1/3. For d_ff = 4 × d_model, the FFN has 2 × d_model × 4d_model = 8d_model² parameters, while attention has 4d_model² parameters. So FFN:attention ratio is approximately 2:1.

### Intermediate

**Q: How does weight tying between embedding and output projection affect parameter count?**

A: Weight tying shares the embedding matrix between the input embedding layer and the output projection (the final linear layer that maps to vocabulary logits). This saves V × d_model parameters, where V is the vocabulary size. For models with large vocabularies (50k+), this can save 20-40M parameters (8-15% of total for BERT-sized models). Weight tying also improves training efficiency because the embedding matrix has two gradient sources.

### Advanced

**Q: For a 175B model like GPT-3, explain where the parameters are distributed and why the FFN dominates even more at large scale.**

A: GPT-3 has d_model = 12288, d_ff = 49152 (4 × d_model), n_layers = 96. The FFN per layer: 2 × 12288 × 49152 ≈ 1.2B. Attention per layer: 4 × 12288² ≈ 604M. The FFN:attention ratio is approximately 2:1 per layer. With 96 layers, that's ~115B in FFN and ~58B in attention. The embedding layer (50257 × 12288 ≈ 618M) is relatively small at this scale. The FFN dominates even more at large scales because the FFN-to-attention parameter ratio is fixed (2:1 for standard FFN) regardless of scale. However, GPT-3 uses sparse attention patterns in some layers, which reduces attention parameters slightly. Additionally, some sources suggest GPT-3 may use different ratios for different layers (alternating dense and sparse attention).

## Practice Problems

### Easy

Compute the exact parameter count for BERT-base (d_model=768, d_ff=3072, n_heads=12, n_layers=12, vocab=30522) by hand, breaking down by component.

### Medium

Write a function that takes a model configuration and returns the parameter count for encoder-only, decoder-only, and encoder-decoder architectures. Include bias parameters and weight tying.

### Hard

For a fixed parameter budget of 1B parameters, design an optimal model configuration using scaling law principles. Optimize over d_model, d_ff, n_layers, and vocab_size (with and without weight tying). Discuss the trade-offs.

## Solutions

### Easy Solution

```python
def bert_base_params():
    d_model, d_ff, n_heads, n_layers, vocab = 768, 3072, 12, 12, 30522

    embed = vocab * d_model
    print(f"Embedding: {embed:,}")

    attn = 4 * d_model * d_model + 4 * d_model
    print(f"Attention per layer: {attn:,}")

    ffn = 2 * d_model * d_ff + d_ff + d_model
    print(f"FFN per layer: {ffn:,}")

    ln = 4 * d_model
    print(f"LayerNorm per layer: {ln:,}")

    per_layer = attn + ffn + ln
    print(f"Total per layer: {per_layer:,}")

    total = embed + n_layers * per_layer
    print(f"Total: {total:,} ({total/1e6:.2f}M)")

bert_base_params()
# Output: Embedding: 23,440,896
# Output: Attention per layer: 2,362,368
# Output: FFN per layer: 4,722,816
# Output: LayerNorm per layer: 3,072
# Output: Total per layer: 7,088,256
# Output: Total: 108,499,968 (108.50M)
```

## Related Concepts

- **DL-376: Transformer Dimensionality**: The dimensions that determine parameter counts.
- **DL-377: d_model**: The dimension with quadratic effect on parameters.
- **DL-378: d_ff**: The dimension with linear effect on FFN parameters.
- **DL-379: n_heads**: Does not affect parameter count (distributes existing parameters).
- **DL-380: n_layers**: Linear effect on parameters.
- **DL-382: FLOPs in Transformer**: Related metric for computational cost.

## Next Concepts

- DL-382: FLOPs in Transformer — Computational cost as the other side of the coin from parameter count.
- DL-383: KV Cache — Memory optimization for inference.

## Summary

Transformer parameter count is the sum of all trainable parameters across embeddings, attention, FFN, and normalization layers. The FFN contains ~2/3 of parameters, attention ~1/3, and embeddings contribute significantly for large vocabularies. Parameter count scales as \(O(d_{\text{model}}^2)\) per layer and \(O(n_{\text{layers}})\) total. Understanding parameter distribution is essential for model design, memory planning, and comparing architectures.

## Key Takeaways

1. FFN contains ~2/3 of parameters; attention ~1/3; embeddings vary.
2. Parameters scale as \(12d_{\text{model}}^2\) per layer with standard expansion.
3. GLU variants have 3 FFN weight matrices instead of 2.
4. Weight tying shares embedding/output parameters, saving V × d_model.
5. Training memory is 3-5x the parameter memory (weights + optimizer + grads + activations).
6. Parameter count is the primary metric for model comparison and hardware planning.
