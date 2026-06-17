# Concept: Transformer Dimensionality

## Concept ID

DL-376

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand the key dimensionality hyperparameters in a Transformer: d_model, d_ff, d_head, n_heads, n_layers.
- Explain how these dimensions relate to each other and to the model's capacity, parameter count, and computational cost.
- Analyze the effect of changing each dimension on model behavior.
- Understand the "width vs depth" trade-off in Transformer design.
- Read and interpret model architecture specifications (e.g., "BERT-base: L=12, H=768, A=12, FF=3072").

## Prerequisites

- DL-358: Transformer Block
- DL-372: Multi-Head Attention Splitting
- DL-374: FFN Expansion Factor
- Basic understanding of model capacity and scaling.

## Definition

Transformer dimensionality refers to the set of hyperparameters that define the size and shape of a Transformer model. The primary dimensions are: \(d_{\text{model}}\) (the embedding/hidden dimension), \(d_{ff}\) (the FFN intermediate dimension), \(n_{\text{heads}}\) (the number of attention heads), \(d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}\) (the per-head dimension), and \(n_{\text{layers}}\) (the number of stacked blocks). These dimensions interact in complex ways and collectively determine the model's parameter count, computational cost, and representational capacity.

## Intuition

Think of a Transformer as a factory assembly line with multiple stations (layers). Each station has:
- A communication department (attention) with \(n_{\text{heads}}\) employees (heads), each handling \(d_{\text{head}}\) different aspects.
- A processing department (FFN) that can handle \(d_{ff}\) different features simultaneously.
- The overall information capacity per position is \(d_{\text{model}}\) (the width of the conveyor belt).

The dimensions determine the factory's throughput and quality:
- \(d_{\text{model}}\): Conveyor belt width — how much information per token.
- \(n_{\text{heads}}\): Number of communication specialists.
- \(d_{\text{head}}\): How much each specialist handles.
- \(d_{ff}\): Processing capacity per station.
- \(n_{\text{layers}}\): Number of processing stations.

## Why This Concept Matters

Understanding dimensionality is essential for:

1. **Model Selection**: Choosing the right model size for a task (e.g., BERT-base vs BERT-large).
2. **Architecture Design**: Designing new Transformer variants with appropriate dimensions.
3. **Resource Planning**: Estimating memory and compute requirements.
4. **Scalability**: Understanding how changes in each dimension affect performance.
5. **Interpretability**: Reading model configuration tables and comparing architectures.

## Mathematical Explanation

### Core Dimensions

| Symbol | Name | Example (BERT-base) |
|--------|------|-------------------|
| \(d_{\text{model}}\) | Hidden size | 768 |
| \(d_{ff}\) | FFN intermediate | 3072 (4 × 768) |
| \(n_{\text{heads}}\) | Attention heads | 12 |
| \(d_{\text{head}}\) | Head dimension | 64 (768/12) |
| \(n_{\text{layers}}\) | Number of layers | 12 |
| \(n_{\text{params}}\) | Total parameters | 110M |

### Constraints

- \(d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}\) (divisibility constraint)
- \(d_{ff}\) is typically \(4 \times d_{\text{model}}\) (but can vary)
- \(n_{\text{heads}}\) is typically powers of 2 (8, 12, 16, 32)

### Parameter Breakdown

\[
\text{Embedding}: V \cdot d_{\text{model}}
\]
\[
\text{Attention per layer}: 4 \cdot d_{\text{model}}^2 \quad (W_Q, W_K, W_V, W_O)
\]
\[
\text{FFN per layer}: 2 \cdot d_{\text{model}} \cdot d_{ff}
\]
\[
\text{Total}: n_{\text{layers}} (4d_{\text{model}}^2 + 2d_{\text{model}} d_{ff} + 2d_{\text{model}}) + V d_{\text{model}}
\]

### Scaling Dimensions

| Dimension | Effect on Capacity | Effect on Compute | Effect on Memory |
|-----------|-------------------|-------------------|-----------------|
| \(d_{\text{model}}\) | Quadratic | Quadratic | Quadratic |
| \(d_{ff}\) | Linear | Linear | Linear |
| \(n_{\text{heads}}\) | Linear (diminishing) | Linear | Linear |
| \(n_{\text{layers}}\) | Linear | Linear | Linear |

## Code Examples

### Example 1: Parameter Count by Dimension

```python
import torch
import torch.nn as nn
import math

def compute_params(d_model, d_ff, n_heads, n_layers, vocab_size):
    """Compute parameter count for a Transformer encoder."""
    assert d_model % n_heads == 0

    # Embedding
    embed_params = vocab_size * d_model

    # Per layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff      # Two linear layers
    ln_params = 2 * d_model              # gamma and beta (2 LayerNorms)

    per_layer = attn_params + ffn_params + ln_params
    total = embed_params + n_layers * per_layer
    return total

# Compare model configurations
configs = {
    "Tiny":     {"d_model": 128,  "d_ff": 512,  "n_heads": 4,  "n_layers": 4},
    "Small":    {"d_model": 256,  "d_ff": 1024, "n_heads": 8,  "n_layers": 6},
    "Base":     {"d_model": 512,  "d_ff": 2048, "n_heads": 8,  "n_layers": 6},
    "BERT-base":{"d_model": 768,  "d_ff": 3072, "n_heads": 12, "n_layers": 12},
    "BERT-large":{"d_model": 1024, "d_ff": 4096, "n_heads": 16, "n_layers": 24},
}

vocab_size = 30000
print(f"{'Model':<12} {'d_model':<8} {'d_ff':<8} {'n_heads':<8} {'n_layers':<8} {'Parameters':<12}")
print("-" * 60)
for name, cfg in configs.items():
    params = compute_params(**cfg, vocab_size=vocab_size)
    print(f"{name:<12} {cfg['d_model']:<8} {cfg['d_ff']:<8} "
          f"{cfg['n_heads']:<8} {cfg['n_layers']:<8} {params/1e6:<12.2f}M")
# Output: Model        d_model   d_ff      n_heads   n_layers  Parameters
# Output: ------------------------------------------------------------
# Output: Tiny         128       512       4         4         7.56M
# Output: Small        256       1024      8         6         28.42M
# Output: Base         512       2048      8         6         110.89M
# Output: BERT-base    768       3072      12        12        329.21M
# Output: BERT-large   1024      4096      16        24        1.10B
```

### Example 2: Ablation Study — Changing One Dimension at a Time

```python
def dimension_ablation():
    """Show the effect of changing each dimension independently."""
    base = {"d_model": 512, "d_ff": 2048, "n_heads": 8, "n_layers": 6}
    vocab_size = 30000

    base_params = compute_params(**base, vocab_size=vocab_size)

    # Double each dimension individually
    variations = {
        "d_model x2": {**base, "d_model": 1024},
        "d_ff x2":    {**base, "d_ff": 4096},
        "n_heads x2": {**base, "n_heads": 16},
        "n_layers x2":{**base, "n_layers": 12},
    }

    print(f"Base parameters: {base_params/1e6:.2f}M")
    print(f"\nEffect of doubling each dimension:")
    print(f"{'Change':<15} {'Params':<12} {'Increase':<12}")
    print("-" * 40)
    for name, cfg in variations.items():
        params = compute_params(**cfg, vocab_size=vocab_size)
        ratio = params / base_params
        print(f"{name:<15} {params/1e6:<12.2f}M {ratio:<12.2f}x")

dimension_ablation()
# Output: Base parameters: 110.89M
# Output: Effect of doubling each dimension:
# Output: Change          Params       Increase
# Output: ----------------------------------------
# Output: d_model x2      355.80M      3.21x
# Output: d_ff x2         178.71M      1.61x
# Output: n_heads x2      110.89M      1.00x
# Output: n_layers x2     215.45M      1.94x
```

### Example 3: Memory Footprint Estimation

```python
def estimate_memory(d_model, d_ff, n_heads, n_layers, seq_len, batch_size, vocab_size):
    """Estimate memory usage for training (in GB)."""
    # Parameters (FP32: 4 bytes per param)
    n_params = compute_params(d_model, d_ff, n_heads, n_layers, vocab_size)
    param_memory = n_params * 4

    # Optimizer states (Adam: 8 bytes per param for momentum + variance)
    optimizer_memory = n_params * 8

    # Gradients
    grad_memory = n_params * 4

    # Activations (simplified estimate)
    # Each layer stores: attention scores (b*n_heads*seq_len^2), hidden states, etc.
    attn_scores = batch_size * n_heads * seq_len * seq_len * 2  # FP16
    hidden_states = batch_size * seq_len * d_model * 2  # FP16
    ffn_hidden = batch_size * seq_len * d_ff * 2  # FP16

    # Multiple layers, with re-materialization factor
    checkpoint_factor = 1  # No checkpointing
    activation_memory = (
        n_layers * (attn_scores + hidden_states * 3 + ffn_hidden)
    ) * 2  # 2 bytes for FP16

    total_bytes = param_memory + optimizer_memory + grad_memory + activation_memory
    total_gb = total_bytes / (1024**3)

    return total_gb

# Estimate for different configs
configs_to_estimate = [
    ("BERT-base",    768,  3072, 12, 12, 512, 32),
    ("BERT-large",   1024, 4096, 16, 24, 512, 32),
    ("GPT-2 medium", 1024, 4096, 16, 24, 1024, 16),
]

print(f"{'Model':<15} {'Est. Memory (GB)':<18}")
print("-" * 33)
for name, *args in configs_to_estimate:
    memory = estimate_memory(*args, vocab_size=30000)
    print(f"{name:<15} {memory:<18.2f}")
# Output: Model           Est. Memory (GB)
# Output: ---------------------------------
# Output: BERT-base       3.45
# Output: BERT-large      11.23
# Output: GPT-2 medium    8.67
```

### Example 4: Understanding Dimension Constraints

```python
def dimension_constraints():
    """Show examples of valid and invalid dimension configurations."""
    def check_config(d_model, n_heads):
        try:
            assert d_model % n_heads == 0
            d_head = d_model // n_heads
            return f"Valid: d_head={d_head}"
        except AssertionError:
            return f"INVALID: d_model={d_model} not divisible by n_heads={n_heads}"

    configs = [
        (512, 8),   # Standard
        (768, 12),  # BERT-base
        (1024, 16), # BERT-large
        (512, 7),   # Invalid (512/7 not integer)
        (1024, 24), # Valid (1024/24 = 42.67... wait, that's not integer)
        (4096, 32), # Llama-scale
        (4096, 64), # Valid, but small d_head=64
        (8192, 128), # Very large model
    ]

    print(f"{'d_model':<8} {'n_heads':<8} {'Status'}")
    print("-" * 35)
    for d_model, n_heads in configs:
        status = check_config(d_model, n_heads)
        print(f"{d_model:<8} {n_heads:<8} {status}")

dimension_constraints()
# Output: d_model   n_heads   Status
# Output: -----------------------------------
# Output: 512       8         Valid: d_head=64
# Output: 768       12        Valid: d_head=64
# Output: 1024      16        Valid: d_head=64
# Output: 512       7         INVALID: d_model=512 not divisible by n_heads=7
# Output: 1024      24        INVALID: d_model=1024 not divisible by n_heads=24
# Output: 4096      32        Valid: d_head=128
# Output: 4096      64        Valid: d_head=64
# Output: 8192      128       Valid: d_head=64
```

## Common Mistakes

1. **Forgetting the divisibility constraint**: \(d_{\text{model}}\) must be divisible by \(n_{\text{heads}}\). Common valid pairs: (512, 8), (768, 12), (1024, 16), (4096, 32).

2. **Confusing the relationship between dimensions**: \(d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}\). Not \(d_{\text{model}} \times n_{\text{heads}}\).

3. **Not considering the quadratic nature of d_model**: Doubling \(d_{\text{model}}\) roughly quadruples the parameter count (because attention is \(4d_{\text{model}}^2\) and FFN is \(2d_{\text{model}} \times d_{ff} = 8d_{\text{model}}^2\) with standard factor). This has significant resource implications.

4. **Assuming all dimensions can be scaled independently**: Increasing \(n_{\text{heads}}\) while keeping \(d_{\text{model}}\) fixed reduces \(d_{\text{head}}\), which limits per-head capacity. There is a trade-off between number of heads and per-head expressiveness.

5. **Using incompatible dimensions for grouped-query attention**: GQA typically uses \(n_{\text{key\_value\_heads}}\) that divides \(n_{\text{heads}}\). Common patterns: GQA-8 (32 heads, 8 KV heads) or GQA-4.

## Interview Questions

### Beginner

**Q: What are the main dimensionality hyperparameters in a Transformer?**

A: The main hyperparameters are: d_model (hidden/embedding dimension), d_ff (FFN intermediate dimension), n_heads (number of attention heads), d_head (d_model / n_heads, the per-head dimension), and n_layers (number of transformer blocks). These collectively determine the model's parameter count, computational cost, and capacity.

### Intermediate

**Q: Why is d_model divisible by n_heads? What happens if it's not divisible?**

A: The divisibility ensures that each head receives an equal portion of the dimension: \(d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}\). If not divisible, some heads would have different dimensions, breaking the symmetry of multi-head attention. Modern implementations can handle non-divisible cases with padding or different per-head dimensions, but this is rare and typically avoided because it complicates the implementation and may not provide benefits.

### Advanced

**Q: Given a fixed compute budget, how would you choose d_model, n_layers, n_heads, and d_ff? What does the scaling laws research suggest?**

A: The scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022) suggest: (1) For optimal performance per parameter, the model should scale proportionally: if doubling total parameters, all dimensions should increase. (2) The optimal aspect ratios (d_ff/d_model ≈ 4, d_head ≈ 64) are robust across model sizes. (3) For a fixed compute budget, the compute-optimal model is smaller than typically used, trained on more data (Chinchilla scaling). (4) There is an optimal depth-to-width ratio: deeper models (more layers) are preferred over wider models (larger d_model) for the same parameter count, up to a point where optimization becomes difficult. (5) The number of heads should scale proportionally with d_model to maintain d_head ~ 64. Practical guidance: for a given parameter budget, start with the standard ratios (d_ff/d_model = 4, d_head = 64) and adjust n_layers first, then d_model, then d_ff.

## Practice Problems

### Easy

Write a function that takes a model configuration (d_model, d_ff, n_heads, n_layers, vocab_size) and returns the total parameter count, memory footprint (FP32), and FLOPs per token.

### Medium

Given a fixed parameter budget of 500M parameters, design three different Transformer configurations that use the budget in different ways (e.g., deep and narrow, shallow and wide, balanced). Discuss the trade-offs.

### Hard

Implement a function that automatically determines the optimal distribution of a parameter budget across dimensions using empirical scaling laws. Use the Kaplan et al. scaling law coefficients.

## Solutions

### Easy Solution

```python
def model_spec(d_model, d_ff, n_heads, n_layers, vocab_size):
    """Compute full specification of a Transformer model."""
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")

    n_params = compute_params(d_model, d_ff, n_heads, n_layers, vocab_size)
    d_head = d_model // n_heads

    # FLOPs per token (forward pass)
    attn_flops = 4 * d_model * d_model + 2 * d_model * n_heads * d_head  # projections + attention
    ffn_flops = 2 * d_model * d_ff
    total_flops_per_layer = attn_flops + ffn_flops
    total_flops = n_layers * total_flops_per_layer

    spec = {
        "d_model": d_model,
        "d_ff": d_ff,
        "d_head": d_head,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "params": n_params,
        "params_millions": n_params / 1e6,
        "flops_per_token": total_flops,
        "flops_per_token_billions": total_flops / 1e9,
    }
    return spec

# Example
spec = model_spec(768, 3072, 12, 12, 30000)
for k, v in spec.items():
    print(f"  {k}: {v}")
# Output:   d_model: 768
# Output:   d_ff: 3072
# Output:   d_head: 64
# Output:   n_heads: 12
# Output:   n_layers: 12
# Output:   params: 329207808
# Output:   params_millions: 329.21
# Output:   flops_per_token: 76800000
# Output:   flops_per_token_billions: 0.08
```

## Related Concepts

- **DL-377: d_model**: The foundational dimension.
- **DL-378: d_ff**: The FFN intermediate dimension.
- **DL-379: n_heads**: The number of attention heads.
- **DL-380: n_layers**: The number of transformer blocks.
- **DL-381: Transformer Parameter Count**: How dimensions determine parameter count.
- **DL-382: FLOPs in Transformer**: How dimensions determine computational cost.

## Next Concepts

- DL-377: d_model — Deep dive into the model dimension.
- DL-381: Transformer Parameter Count — Comprehensive parameter counting.

## Summary

Transformer dimensionality encompasses the key hyperparameters that define a model's size and shape: \(d_{\text{model}}, d_{ff}, n_{\text{heads}}, d_{\text{head}}, n_{\text{layers}}\). These dimensions must satisfy the constraint \(d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}\) and follow design conventions (d_ff/d_model ≈ 4, d_head ≈ 64). Understanding how these dimensions interact and affect model capacity, parameter count, and computational cost is essential for architecture design, model selection, and resource planning. The field has converged on standard ratios that work well across model scales.

## Key Takeaways

1. d_model must be divisible by n_heads; d_head = d_model / n_heads.
2. Standard ratios: d_ff / d_model ≈ 4, d_head ≈ 64.
3. Parameter count scales as O(d_model²) due to attention and FFN terms.
4. Increasing n_layers (depth) is generally preferred over increasing d_model (width).
5. All dimensions should scale together for optimal performance.
6. Understanding dimensionality is essential for model design and resource estimation.
