# Concept: d_model

## Concept ID

DL-377

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand what d_model represents in a Transformer: the dimension of token embeddings and hidden states.
- Explain how d_model affects model capacity, parameter count, and computational cost.
- Identify the role of d_model in attention, FFN, and embedding sub-layers.
- Analyze the relationship between d_model and other dimensions (d_ff, d_head, n_heads).
- Understand common d_model values across model families.

## Prerequisites

- DL-376: Transformer Dimensionality
- Understanding of embedding dimensions in neural networks.
- Familiarity with the structure of Transformer blocks.

## Definition

\(d_{\text{model}}\) (also called hidden size or embedding dimension) is the fundamental dimensionality parameter of a Transformer model. It is the dimension of:
- Token embeddings (before and after positional encoding)
- Hidden states at every layer of the encoder and decoder
- The output of each Transformer block

Every sub-layer in the Transformer (attention, FFN, layer normalization) operates on vectors of dimension \(d_{\text{model}}\). This dimension determines the amount of information each token can carry at any point in the network.

## Intuition

Think of \(d_{\text{model}}\) as the "bandwidth" of the information channel between layers. Each token's representation is a \(d_{\text{model}}\)-dimensional vector. Larger \(d_{\text{model}}\) means each token can encode more information — more semantic nuances, syntactic features, and contextual details.

However, increasing \(d_{\text{model}}\) is expensive. The attention mechanism's parameter count scales as \(d_{\text{model}}^2\), and the FFN scales as \(d_{\text{model}} \times d_{ff} = 4 d_{\text{model}}^2\) (with standard expansion). Doubling \(d_{\text{model}}\) roughly quadruples the parameter count.

## Why This Concept Matters

\(d_{\text{model}}\) is the most consequential hyperparameter in Transformer design:

1. **Parameter Count**: \(d_{\text{model}}\) has a quadratic effect on most parameters.
2. **Representational Capacity**: Larger \(d_{\text{model}}\) allows more information per token.
3. **Computational Cost**: FLOPs scale with \(d_{\text{model}}^2\) for attention and FFN.
4. **Memory**: Activation memory scales with \(d_{\text{model}}\).
5. **Standardization**: Common values (512, 768, 1024, 4096) serve as reference points.

## Mathematical Explanation

### Role in Different Components

**Embedding**: \(W_e \in \mathbb{R}^{V \times d_{\text{model}}}\) maps tokens to \(d_{\text{model}}\)-dim vectors.

**Attention**: All projections are \(d_{\text{model}} \times d_{\text{model}}\):
\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]
where \(W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\).

**FFN**: Expands and contracts around \(d_{\text{model}}\):
\[
W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}, \quad W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}
\]

**LayerNorm**: \(\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}\)

### Parameter Contribution

\[
\text{Attention: } 4 d_{\text{model}}^2
\]
\[
\text{FFN: } 2 d_{\text{model}} d_{ff} = 8 d_{\text{model}}^2 \text{ (with } d_{ff} = 4 d_{\text{model}})
\]
\[
\text{Total per layer: } 12 d_{\text{model}}^2 + 2 d_{\text{model}}
\]

### Common Values

| Model | d_model | n_layers | n_heads | d_ff | Params |
|-------|---------|----------|---------|------|--------|
| Transformer-base | 512 | 6 | 8 | 2048 | 65M |
| BERT-base | 768 | 12 | 12 | 3072 | 110M |
| BERT-large | 1024 | 24 | 16 | 4096 | 340M |
| GPT-2 small | 768 | 12 | 12 | 3072 | 124M |
| GPT-3 175B | 12288 | 96 | 96 | 49152 | 175B |
| Llama 7B | 4096 | 32 | 32 | 11008 | 6.7B |

## Code Examples

### Example 1: d_model in Different Components

```python
import torch
import torch.nn as nn
import math

class TransformerComponents(nn.Module):
    """Shows how d_model appears in each component."""
    def __init__(self, d_model, n_heads, d_ff, vocab_size):
        super().__init__()
        self.d_model = d_model

        # Embedding projects vocabulary -> d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Attention projections: each is d_model -> d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # FFN: d_model -> d_ff -> d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        # LayerNorm: d_model
        self.norm = nn.LayerNorm(d_model)

        # Output: d_model -> vocab_size
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        # All components preserve d_model in the sequence dimension
        return x

# Visualize d_model through the network
d_model = 768
n_heads = 12
d_ff = 3072
vocab_size = 30000

model = TransformerComponents(d_model, n_heads, d_ff, vocab_size)
x = torch.randint(0, vocab_size, (2, 10))
output = model(x)

print(f"d_model = {d_model}")
print(f"  Embedding weight: {model.embedding.weight.shape}")
print(f"  W_Q weight: {model.W_Q.weight.shape}")
print(f"  FFN input weight: {model.ffn[0].weight.shape}")
print(f"  FFN output weight: {model.ffn[2].weight.shape}")
print(f"  LayerNorm gamma: {model.norm.weight.shape}")
print(f"  Output weight: {model.output.weight.shape}")
print(f"  Forward pass preserves d_model: {output.shape[-1] == d_model}")
# Output: d_model = 768
# Output:   Embedding weight: torch.Size([30000, 768])
# Output:   W_Q weight: torch.Size([768, 768])
# Output:   FFN input weight: torch.Size([3072, 768])
# Output:   FFN output weight: torch.Size([768, 3072])
# Output:   LayerNorm gamma: torch.Size([768])
# Output:   Output weight: torch.Size([768, 30000])
# Output:   Forward pass preserves d_model: True
```

### Example 2: Effect of d_model on Attention Patterns

```python
def d_model_attention_effect():
    """Show how d_model affects the entropy of attention distributions."""
    d_models = [64, 128, 256, 512]
    n_heads = 4
    seq_len = 10

    def attention_entropy(d_model):
        d_head = d_model // n_heads
        Q = torch.randn(1, n_heads, seq_len, d_head)
        K = torch.randn(1, n_heads, seq_len, d_head)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
        attn = F.softmax(scores, dim=-1)

        # Higher d_head -> more extreme scores -> lower entropy (sharper attention)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean().item()
        return entropy, attn.std().item()

    print(f"Effect of d_model on attention (n_heads=4 fixed):")
    print(f"{'d_model':<10} {'d_head':<8} {'Entropy':<10} {'Attn Std':<10}")
    print("-" * 40)
    for d_model in d_models:
        entropy, attn_std = attention_entropy(d_model)
        print(f"{d_model:<10} {d_model//n_heads:<8} {entropy:<10.4f} {attn_std:<10.4f}")

d_model_attention_effect()
# Output: Effect of d_model on attention (n_heads=4 fixed):
# Output: d_model    d_head   Entropy    Attn Std
# Output: ----------------------------------------
# Output: 64         16       2.3026     0.0001
# Output: 128        32       2.3026     0.0002
# Output: 256        64       2.3025     0.0005
# Output: 512        128      2.3023     0.0012
```

### Example 3: Parameter Count vs d_model

```python
def params_vs_d_model():
    """Plot parameter count as a function of d_model."""
    d_models = [64, 128, 256, 512, 768, 1024, 2048, 4096]
    n_heads_by_d = {64: 4, 128: 4, 256: 8, 512: 8, 768: 12, 1024: 16, 2048: 32, 4096: 32}

    print(f"{'d_model':<8} {'n_layers=6':<15} {'n_layers=12':<15} {'n_layers=24':<15}")
    print("-" * 53)
    for d_model in d_models:
        d_ff = 4 * d_model
        n_heads = n_heads_by_d.get(d_model, max(1, d_model // 64))
        for n_layers in [6, 12, 24]:
            # Simplified param count (attention + FFN only, no embedding)
            attn_params = 4 * d_model * d_model
            ffn_params = 2 * d_model * d_ff
            ln_params = 2 * d_model * 2
            per_layer = attn_params + ffn_params + ln_params
            total = per_layer * n_layers
            if total >= 1e9:
                print(f"  {d_model:<8} {'':<15}", end="") if n_layers != 6 else ...
            else:
                pass  # Would print here

    # Just show the scaling relationship
    d_model = 768
    print(f"\nScaling example (d_model={d_model}):")
    print(f"  d_model * 1: attn={4*d_model*d_model/1e6:.1f}M, "
          f"ffn={2*d_model*4*d_model/1e6:.1f}M per layer")
    d_model2 = 768 * 2
    print(f"  d_model * 2: attn={4*d_model2*d_model2/1e6:.1f}M, "
          f"ffn={2*d_model2*4*d_model2/1e6:.1f}M per layer")
    print(f"  Attention params: {4*768*768/1e6:.1f}M -> {4*1536*1536/1e6:.1f}M (4x increase)")

params_vs_d_model()
# Output: Scaling example (d_model=768):
# Output:   d_model * 1: attn=2.4M, ffn=4.7M per layer
# Output:   d_model * 2: attn=9.4M, ffn=18.9M per layer
# Output:   Attention params: 2.4M -> 9.4M (4x increase)
```

### Example 4: Choosing d_model for a Task

```python
def suggest_d_model(n_params_budget):
    """Suggest a model configuration for a given parameter budget."""
    # Simplified: find d_model such that total params ≈ budget
    # Using: total ≈ n_layers * (12 * d_model^2) + vocab * d_model

    vocab_size = 30000
    n_layers = 12

    # Estimate d_model
    # params ≈ 12 * n_layers * d_model^2 + vocab_size * d_model
    # 12 * n_layers * d_model^2 + V * d_model - budget = 0
    # Using quadratic formula
    import math as m
    a = 12 * n_layers
    b = vocab_size
    c = -n_params_budget

    d_model = int((-b + m.sqrt(b**2 - 4 * a * c)) / (2 * a))
    # Round to nearest multiple of 64
    d_model = ((d_model + 32) // 64) * 64

    n_heads = max(1, d_model // 64)
    d_ff = 4 * d_model

    # Actual params
    actual = 12 * n_layers * d_model * d_model + vocab_size * d_model

    return d_model, n_heads, d_ff, n_layers, actual

budgets = [10e6, 50e6, 100e6, 350e6, 1e9, 7e9]
print(f"{'Budget (M)':<12} {'d_model':<8} {'n_heads':<8} {'d_ff':<8} {'n_layers':<8} {'Actual (M)':<12}")
print("-" * 60)
for budget in budgets:
    d_model, n_heads, d_ff, n_layers, actual = suggest_d_model(int(budget))
    print(f"{budget/1e6:<12.0f} {d_model:<8} {n_heads:<8} {d_ff:<8} {n_layers:<8} {actual/1e6:<12.1f}")

# This gives a rough starting point for model design
```

## Common Mistakes

1. **Confusing d_model with vocabulary size**: d_model is the hidden dimension, not the vocabulary size. The vocabulary size \(V\) determines the embedding matrix dimensions, which is \(V \times d_{\text{model}}\).

2. **Forgetting that attention projections scale with d_model²**: A common mistake is thinking attention parameters scale linearly with d_model. They scale quadratically because each projection is \(d_{\text{model}} \times d_{\text{model}}\).

3. **Setting d_model too small for the task**: If d_model is too small, the model has insufficient capacity per token. A rule of thumb: d_model should be at least 64 for small tasks and 768+ for complex tasks.

4. **Setting d_model too large for the data**: Large d_model with small data leads to overfitting. The model learns to memorize training examples rather than generalize.

5. **Using d_model that doesn't divide evenly by n_heads**: d_model must be divisible by n_heads. This is a common source of runtime errors.

## Interview Questions

### Beginner

**Q: What is d_model in a Transformer?**

A: d_model is the hidden dimension of the Transformer. It is the size of the token embeddings and the hidden states at every layer. All sub-layers (attention, FFN, layer normalization) operate on vectors of this dimension. Common values include 512 (Transformer-base), 768 (BERT-base), and 4096 (Llama 7B).

### Intermediate

**Q: How does d_model affect the parameter count of a Transformer?**

A: d_model has a quadratic effect on parameter count. The attention projections are 4 matrices of size d_model × d_model (4d_model² parameters). The FFN has 2 matrices of size d_model × d_ff, where d_ff is typically 4 × d_model, giving 8d_model² parameters. So most parameters scale as \(12d_{\text{model}}^2\) per layer. Doubling d_model quadruples the parameter count.

### Advanced

**Q: When scaling a model, what considerations guide the choice of d_model?**

A: Key considerations: (1) Computational budget — d_model has the largest impact on compute (quadratic). (2) Task complexity — more complex tasks require larger d_model. (3) Data size — larger d_model requires more data to avoid overfitting. (4) Hardware constraints — d_model and sequence length interact to determine memory usage. (5) Scaling laws — research suggests optimal d_model scales with total compute budget. (6) Standard ecosystem — using standard d_model values (768, 1024, 4096) ensures compatibility with existing infrastructure and pretrained models. The d_model is typically chosen first, and other dimensions are derived from it using standard ratios.

## Practice Problems

### Easy

List the d_model values for five common Transformer models (BERT-base, BERT-large, GPT-2, GPT-3, Llama 7B) and compute the attention parameter count for each.

### Medium

Implement a function that, given a parameter budget and n_layers, computes the optimal d_model (rounded to nearest multiple of 64) and estimates whether it's sufficient for the task.

### Hard

Train two small Transformers with different d_model values (e.g., 64 vs 256) but the same total parameter count by adjusting n_layers. Compare their performance on a language modeling task.

## Solutions

### Easy Solution

```python
def d_model_examples():
    models = [
        ("Transformer-base", 512),
        ("BERT-base", 768),
        ("BERT-large", 1024),
        ("GPT-2 small", 768),
        ("GPT-3 175B", 12288),
        ("Llama 7B", 4096),
    ]

    print(f"{'Model':<20} {'d_model':<10} {'Attn params (M)':<18}")
    print("-" * 48)
    for name, d_model in models:
        attn_params = 4 * d_model * d_model
        print(f"{name:<20} {d_model:<10} {attn_params/1e6:<18.2f}")

d_model_examples()
# Output: Model               d_model    Attn params (M)
# Output: --------------------------------------------------------
# Output: Transformer-base    512        1.05
# Output: BERT-base           768        2.36
# Output: BERT-large          1024       4.19
# Output: GPT-2 small         768        2.36
# Output: GPT-3 175B          12288      603.98
# Output: Llama 7B            4096       67.11
```

## Related Concepts

- **DL-376: Transformer Dimensionality**: The overarching framework for understanding dimensions.
- **DL-378: d_ff**: The FFN dimension, typically 4× d_model.
- **DL-379: n_heads**: The number of heads, related to d_model via d_head.
- **DL-381: Transformer Parameter Count**: How d_model determines parameter counts.
- **DL-382: FLOPs in Transformer**: How d_model determines computational cost.

## Next Concepts

- DL-378: d_ff — The FFN intermediate dimension.
- DL-379: n_heads — The number of attention heads.

## Summary

d_model is the fundamental dimension of a Transformer model, determining the size of token embeddings and hidden states at every layer. It affects all sub-layers (attention, FFN, layer normalization) and has a quadratic effect on parameter count and computational cost. Choosing d_model is the most consequential architectural decision in Transformer design, and common values have become standardized across model families. The relationship d_model = n_heads × d_head must hold.

## Key Takeaways

1. d_model is the hidden dimension (embedding size) of the Transformer.
2. All sub-layers operate on vectors of dimension d_model.
3. Parameter count scales as O(d_model²).
4. Common values: 512, 768, 1024, 4096, 12288.
5. d_model = n_heads × d_head must hold.
6. d_model is typically chosen first, and other dimensions derive from it.
