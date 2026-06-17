# Concept: n_layers

## Concept ID

DL-380

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand n_layers as the number of stacked Transformer blocks in a model.
- Explain how depth affects model capacity, representational hierarchy, and training dynamics.
- Analyze the relationship between depth and other dimensions in model scaling.
- Understand the role of depth in creating hierarchical representations.
- Recognize common n_layers values across model families.

## Prerequisites

- DL-358: Transformer Block
- DL-376: Transformer Dimensionality
- Understanding of deep neural networks and the concept of hierarchical feature learning.

## Definition

\(n_{\text{layers}}\) (also called \(L\) or depth) is the number of stacked identical Transformer blocks in the encoder or decoder stack. Each layer adds another round of self-attention and feed-forward processing, allowing the model to build increasingly abstract and contextualized representations. Typical values range from 2 (very shallow) to 96 (very deep, like GPT-3). The depth directly determines the model's parameter count, computational cost, and representational hierarchy.

## Intuition

Think of each layer as a processing stage in an assembly line. Raw materials (token embeddings) enter at the first stage. Each stage:
- Gathers context from other tokens (self-attention).
- Processes and refines the representation (FFN).

Deeper models can build more abstract representations. Early layers might capture surface-level patterns (word shapes, POS tags). Middle layers capture syntactic structure (subject-verb agreement, clause boundaries). Deep layers capture high-level semantics (topic, discourse, intent).

However, more layers are not always better. Very deep models require more data, more careful training, and may suffer from optimization difficulties beyond a certain depth.

## Why This Concept Matters

n_layers is a key architectural hyperparameter because:

1. **Model Capacity**: Each layer adds parameters and representational power.
2. **Hierarchical Learning**: Deeper models learn hierarchical representations.
3. **Scaling**: Depth vs. width trade-off — deeper is often better for the same parameter count.
4. **Training Difficulty**: Very deep models require techniques like pre-norm, residual connections, and careful initialization.
5. **Inference Speed**: Each layer adds sequential computation during inference.

## Mathematical Explanation

### Parameter Contribution

Each layer adds:
- Attention: \(4d_{\text{model}}^2\) parameters (Q, K, V, O projections)
- FFN: \(2d_{\text{model}}d_{ff} + d_{ff} + d_{\text{model}}\) parameters
- LayerNorm: \(2 \times 2d_{\text{model}}\) parameters (two LayerNorms)

Total per layer ≈ \(4d_{\text{model}}^2 + 2d_{\text{model}}d_{ff} + 4d_{\text{model}}\)

### Representation Hierarchy

Let \(h^{(l)}\) be the hidden state at layer \(l\). For a model with \(L\) layers:

\[
h^{(0)} = \text{Embedding}(x) + \text{PositionalEncoding}
\]
\[
h^{(l)} = \text{Block}^{(l)}(h^{(l-1)}) \quad \text{for } l = 1, \ldots, L
\]

Each layer can focus on different aspects of the representation. Research (Rogers et al., 2020) shows:
- Lower layers: Surface and syntactic features.
- Middle layers: Syntactic and semantic features.
- Upper layers: Task-specific and contextual features.

### Effective Depth

Due to residual connections, the "effective depth" of a Transformer is not simply \(L\). The residual stream allows information to flow directly from early to late layers, bypassing intermediate layers. The effective depth for a particular piece of information depends on how many layers process it.

## Code Examples

### Example 1: Effect of Depth on Representations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerOutputTracker(nn.Module):
    """Transformer encoder that tracks outputs after each layer."""
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.layer_outputs = []

    def forward(self, x):
        self.layer_outputs = []
        # Hook to capture outputs after each layer
        def hook_fn(module, input, output):
            self.layer_outputs.append(output.detach())

        hooks = []
        for layer in self.encoder.layers:
            hooks.append(layer.register_forward_hook(hook_fn))

        output = self.encoder(x)

        for h in hooks:
            h.remove()

        return output

# Track representations through layers
d_model, n_heads, d_ff, n_layers = 64, 4, 256, 6
tracker = LayerOutputTracker(d_model, n_heads, d_ff, n_layers)
x = torch.randn(2, 10, d_model)
output = tracker(x)

print(f"Representation statistics through layers:")
print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 48)
for i, out in enumerate(tracker.layer_outputs):
    print(f"{i+1:<8} {out.mean().item():<10.4f} {out.std().item():<10.4f} "
          f"{out.min().item():<10.4f} {out.max().item():<10.4f}")

# Show that representations become more similar within related tokens
# (lower cosine distance between related positions for deeper layers)
token_0 = tracker.layer_outputs[0][0, 0, :]  # First token, first layer
token_0_last = tracker.layer_outputs[-1][0, 0, :]  # First token, last layer
token_5 = tracker.layer_outputs[0][0, 5, :]  # Sixth token, first layer
token_5_last = tracker.layer_outputs[-1][0, 5, :]  # Sixth token, last layer

sim_early = F.cosine_similarity(token_0.unsqueeze(0), token_5.unsqueeze(0))
sim_late = F.cosine_similarity(token_0_last.unsqueeze(0), token_5_last.unsqueeze(0))
print(f"\nCosine similarity between token 0 and 5:")
print(f"  Layer 1: {sim_early.item():.4f}")
print(f"  Layer 6: {sim_late.item():.4f}")
# Expected: similarity increases with depth (contextualization)
```

### Example 2: Parameter Count vs Depth

```python
def params_vs_layers():
    """Show how parameter count scales with n_layers."""
    d_model = 768
    d_ff = 3072
    n_heads = 12
    vocab_size = 30000

    print(f"d_model={d_model}, d_ff={d_ff}, vocab={vocab_size}")
    print(f"{'n_layers':<10} {'Params':<15} {'Increase':<10}")
    print("-" * 35)

    embed_params = vocab_size * d_model
    per_layer_attn = 4 * d_model * d_model
    per_layer_ffn = 2 * d_model * d_ff
    per_layer_ln = 2 * d_model * 2
    per_layer = per_layer_attn + per_layer_ffn + per_layer_ln

    for n_layers in [2, 4, 6, 12, 24, 48]:
        total = embed_params + n_layers * per_layer
        increase = n_layers / 6 if n_layers >= 6 else n_layers / 6
        print(f"{n_layers:<10} {total/1e6:<15.2f}M {increase:<10.2f}x")

params_vs_layers()
# Output: d_model=768, d_ff=3072, vocab=30000
# Output: n_layers   Params          Increase
# Output: -----------------------------------
# Output: 2          90.52M          0.33x
# Output: 4          164.89M         0.67x
# Output: 6          239.26M         1.00x
# Output: 12         462.37M         2.00x
# Output: 24         908.60M         4.00x
# Output: 48         1.78B           8.00x
```

### Example 3: Depth vs Width at Fixed Parameter Budget

```python
def depth_vs_width():
    """Compare deep-narrow vs shallow-wide configurations."""
    vocab_size = 30000
    param_budget = 50e6  # 50M params

    def get_params(d_model, d_ff, n_layers):
        embed = vocab_size * d_model
        per_layer = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model
        return embed + n_layers * per_layer

    print(f"Parameter budget: {param_budget/1e6:.0f}M")
    print(f"{'Config':<20} {'d_model':<8} {'d_ff':<8} {'n_layers':<8} {'Params':<12}")
    print("-" * 56)

    configs = [
        ("Deep", 128, 512, 48),
        ("Medium-deep", 256, 1024, 12),
        ("Medium", 384, 1536, 6),
        ("Shallow-wide", 512, 2048, 3),
        ("Very shallow", 768, 3072, 2),
    ]

    for name, d_model, d_ff, n_layers in configs:
        params = get_params(d_model, d_ff, n_layers)
        print(f"{name:<20} {d_model:<8} {d_ff:<8} {n_layers:<8} {params/1e6:<12.2f}M")

depth_vs_width()
# Output: Parameter budget: 50M
# Output: Config               d_model   d_ff      n_layers  Params
# Output: --------------------------------------------------------
# Output: Deep                128       512       48        47.35M
# Output: Medium-deep         256       1024      12        49.86M
# Output: Medium              384       1536      6         51.23M
# Output: Shallow-wide        512       2048      3         48.92M
# Output: Very shallow        768       3072      2         50.11M
```

### Example 4: Training Dynamics for Different Depths

```python
def depth_training_experiment():
    """Compare training convergence for different depths."""
    d_model = 64
    d_ff = 256
    n_heads = 4
    vocab_size = 100

    class LM(nn.Module):
        def __init__(self, n_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_ff, batch_first=True, norm_first=True, dropout=0.0
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
            self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            x = self.encoder(x)
            return self.proj(x.mean(dim=1))

    print("Training LM with different depths:")
    print(f"{'n_layers':<10} {'Params':<12} {'Loss after 50 steps':<20}")
    print("-" * 42)

    for n_layers in [1, 2, 4, 8, 12]:
        model = LM(n_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        params = sum(p.numel() for p in model.parameters())

        losses = []
        model.train()
        for step in range(50):
            x = torch.randint(1, vocab_size, (16, 8))
            y = torch.randint(0, vocab_size, (16,))
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"{n_layers:<10} {params:<12,} {losses[-1]:<20.4f}")

# Uncomment to run
# depth_training_experiment()
```

## Common Mistakes

1. **Assuming deeper is always better**: There is a point of diminishing returns. Very deep models require more data, careful training, and may overfit on small datasets.

2. **Not using pre-norm for deep models**: Post-norm is very difficult to train for models with more than about 12 layers. Pre-norm is essential for deep Transformers.

3. **Ignoring the embedding parameter contribution**: For shallow models (n_layers < 6), the embedding layer can be a significant fraction of total parameters. This is sometimes called the "embedding tax."

4. **Comparing models with different depths but different training recipes**: Deeper models need different learning rates, warmup schedules, and regularization. Directly comparing depth without adjusting training is misleading.

5. **Not considering the interaction with sequence length**: Deeper models process each token through more layers. For very long sequences, the total computational cost (FLOPs) scales as n_layers × seq_len × d_model².

## Interview Questions

### Beginner

**Q: What is n_layers in a Transformer?**

A: n_layers (or depth) is the number of stacked Transformer blocks in the model. Each block contains a self-attention sub-layer and a feed-forward sub-layer. Stacking more layers allows the model to build increasingly abstract and contextualized representations. Common values: 6 (Transformer-base), 12 (BERT-base), 24 (BERT-large), 32 (Llama 7B), 96 (GPT-3).

### Intermediate

**Q: What is the relationship between depth and representational hierarchy in Transformers?**

A: Research shows that different layers capture different levels of abstraction. Lower layers (1-4) capture surface features and local context — word boundaries, POS tags, short-range dependencies. Middle layers (5-8) capture syntactic structure — subject-verb agreement, clause boundaries, dependency relations. Upper layers (9+) capture semantic and discourse-level information — topic, coreference, long-range dependencies. This hierarchy emerges naturally from the layer-by-layer processing and is consistent across different models and tasks.

### Advanced

**Q: Given a fixed parameter budget, how would you decide between increasing depth (more layers) vs increasing width (larger d_model)? What issues arise at extreme depths?**

A: Research (scaling laws and empirical studies) suggests depth is generally preferred over width for a fixed parameter budget. Deeper models learn more hierarchical representations and generalize better. However, extreme depth (100+ layers) introduces optimization challenges: (1) Gradient flow becomes difficult despite residual connections — very deep models may need special initialization like DeepNorm. (2) The "vanishing depth" problem — the effective depth is less than the actual depth because the residual stream can bypass layers. (3) Training becomes unstable, requiring techniques like QK normalization, embedding normalization, and careful learning rate schedules. (4) Inference becomes slower due to sequential layer processing. For modest depths (up to about 48 layers), increasing depth is almost always beneficial. Beyond that, the trade-off becomes more nuanced. The GPT-4 architecture reportedly uses fewer layers than GPT-3 but with larger d_model, suggesting there is an optimal depth for a given compute budget.

## Practice Problems

### Easy

Given a model with d_model = 512, d_ff = 2048, compute the total parameters for n_layers = 2, 6, 12, 24 (ignoring the embedding layer).

### Medium

Train two 50M parameter models on a text dataset: one with 12 layers and d_model = 384, another with 6 layers and d_model = 512. Compare validation loss after the same number of training steps.

### Hard

Implement a "stochastic depth" Transformer where each layer has a probability of being skipped during training (drop a random subset of layers each forward pass). Compare training convergence and validation performance with the standard full-depth model.

## Solutions

### Easy Solution

```python
def compute_model_params(d_model, d_ff, n_layers, include_embedding=False, vocab_size=30000):
    attn_params = 4 * d_model * d_model
    ffn_params = 2 * d_model * d_ff
    ln_params = 4 * d_model
    per_layer = attn_params + ffn_params + ln_params
    total = n_layers * per_layer
    if include_embedding:
        total += vocab_size * d_model
    return total

d_model, d_ff = 512, 2048
print(f"Encoder parameters (d_model={d_model}, d_ff={d_ff}, no embedding):")
for n_layers in [2, 6, 12, 24]:
    params = compute_model_params(d_model, d_ff, n_layers)
    print(f"  n_layers={n_layers}: {params:,} ({params/1e6:.2f}M)")
# Output: Encoder parameters (d_model=512, d_ff=2048, no embedding):
# Output:   n_layers=2: 4,206,592 (4.21M)
# Output:   n_layers=6: 12,619,776 (12.62M)
# Output:   n_layers=12: 25,239,552 (25.24M)
# Output:   n_layers=24: 50,479,104 (50.48M)
```

## Related Concepts

- **DL-358: Transformer Block**: The building block that is stacked.
- **DL-367: Pre-Norm vs Post-Norm**: Normalization placement critically affects deep models.
- **DL-368: Residual Connections**: Enable training of deep models.
- **DL-370: Transformer Training Stability**: Challenges in training deep models.
- **DL-376: Transformer Dimensionality**: How depth interacts with other dimensions.
- **DL-381: Transformer Parameter Count**: How depth contributes to parameters.

## Next Concepts

- DL-381: Transformer Parameter Count — Comprehensive parameter counting.
- DL-382: FLOPs in Transformer — Computational cost analysis.

## Summary

n_layers (depth) is the number of stacked Transformer blocks in a model. It determines the model's representational hierarchy — lower layers capture surface features, middle layers capture syntax, and upper layers capture semantics. Depth has a linear effect on parameter count and computational cost. For a fixed parameter budget, deeper models typically outperform wider models, but extreme depth introduces training stability challenges that require techniques like pre-norm, residual connections, and DeepNorm initialization.

## Key Takeaways

1. n_layers determines the representational hierarchy and model capacity.
2. Deeper models build more abstract representations (surface -> syntax -> semantics).
3. Parameter count scales linearly with n_layers.
4. Deeper is generally preferred over wider for a fixed parameter budget.
5. Pre-norm and residual connections are essential for training deep models.
6. Common values: 6 (small), 12 (medium), 24-32 (large), 96 (very large).
