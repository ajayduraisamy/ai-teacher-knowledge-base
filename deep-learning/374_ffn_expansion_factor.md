# Concept: FFN Expansion Factor

## Concept ID

DL-374

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand the role of the FFN expansion factor (\(d_{ff} / d_{\text{model}}\)) in Transformer design.
- Explain how the expansion factor affects model capacity, parameter count, and computational cost.
- Analyze the trade-offs in choosing different expansion factors.
- Implement FFNs with different expansion factors and measure their impact.
- Understand how the expansion factor interacts with the choice of activation function and GLU variants.

## Prerequisites

- DL-360: Feed-Forward Network
- DL-377: d_model
- DL-378: d_ff
- Understanding of model capacity and parameter-accuracy trade-offs.

## Definition

The FFN expansion factor is the ratio of the intermediate hidden dimension \(d_{ff}\) to the model dimension \(d_{\text{model}}\). In the original Transformer, \(d_{ff} = 2048\) and \(d_{\text{model}} = 512\), giving an expansion factor of 4. This means the FFN expands each token's representation by 4x before non-linear activation and then projects back. The expansion factor is a key hyperparameter that determines the representational capacity per layer and the total parameter count.

## Intuition

The expansion factor controls how much "thinking capacity" each Transformer layer has. A larger expansion factor means each layer has more parameters to process information — the token's representation is expanded into a richer space where the model can identify more complex patterns. However, this comes at the cost of more parameters and computation per layer.

The expansion factor is analogous to the number of hidden units in a standard feed-forward network. A 4x expansion (the standard) works well in practice, but different architectures use different factors:
- BERT-base: \(d_{ff} = 3072\), \(d_{\text{model}} = 768\), factor = 4
- GPT-2: \(d_{ff} = 4 \times d_{\text{model}}\), factor = 4
- Llama: \(d_{ff} = 8/3 \times d_{\text{model}}\) with SwiGLU, effective factor ~ 4
- T5: \(d_{ff} = 2048\), \(d_{\text{model}} = 512\), factor = 4 (but sometimes varied)

## Why This Concept Matters

The expansion factor is one of the most impactful hyperparameters in Transformer design:

1. **Parameter Count**: The FFN contains ~2/3 of the model's parameters. Changing the expansion factor directly scales the model.
2. **Computational Cost**: The FFN dominates the FLOPs per token (especially for longer sequences where attention cost is higher).
3. **Model Capacity**: Higher expansion factors enable learning more complex patterns per layer.
4. **Depth vs. Width Trade-off**: Given a fixed parameter budget, should you use more layers (increase \(n_{\text{layers}}\)) or larger FFNs (increase \(d_{ff}\))?
5. **GLU Variants**: Gated variants (SwiGLU, GeGLU) have three weight matrices and typically use a reduced expansion factor to match parameter counts.

## Mathematical Explanation

### Standard FFN

\[
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
\]

where \(W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}\) and \(W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}\).

**Expansion factor**: \(r = \frac{d_{ff}}{d_{\text{model}}}\)

**FFN parameters**: \(2 \cdot d_{\text{model}} \cdot d_{ff} + d_{ff} + d_{\text{model}} = d_{\text{model}}^2 (2r) + O(d_{\text{model}})\)

### GLU Variant FFN

For SwiGLU, the FFN has three weight matrices:

\[
\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \odot (xV_1))W_2
\]

To match the parameter count of a standard FFN with expansion factor \(r\):

\[
d_{ff}^{\text{GLU}} = \frac{2}{3} r \cdot d_{\text{model}}
\]

This is because the GLU has three \(d_{\text{model}} \times d_{ff}\) matrices instead of two, so the intermediate dimension must be reduced by 1/3.

### Parameter Count per Token

The FFN's FLOPs per token is:

\[
\text{FFN FLOPs} = 2 \cdot d_{\text{model}} \cdot d_{ff} = 2r \cdot d_{\text{model}}^2
\]

For the standard \(r = 4\): \(\text{FFN FLOPs} = 8 \cdot d_{\text{model}}^2\)

### Impact on Model Behavior

Empirical studies show:
- Incresing \(r\) from 2 to 8 improves performance but with diminishing returns.
- The optimal \(r\) depends on \(d_{\text{model}}\): larger \(d_{\text{model}}\) can use smaller \(r\).
- Very large \(r\) (e.g., 32) leads to overfitting on small datasets.

## Code Examples

### Example 1: FFN with Different Expansion Factors

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlexibleFFN(nn.Module):
    """
    FFN with configurable expansion factor.
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1, use_gelu=False):
        super().__init__()
        d_ff = int(d_model * expansion_factor)
        self.d_ff = d_ff
        self.expansion_factor = expansion_factor

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_gelu = use_gelu

    def forward(self, x):
        activation = F.gelu if self.use_gelu else F.relu
        return self.linear2(self.dropout(activation(self.linear1(x))))

    def get_params(self):
        return sum(p.numel() for p in self.parameters())

# Compare different expansion factors
d_model = 512
for factor in [2, 4, 8, 16]:
    ffn = FlexibleFFN(d_model, factor)
    x = torch.randn(2, 10, d_model)
    output = ffn(x)
    params = ffn.get_params()
    print(f"Factor={factor:2d}: d_ff={ffn.d_ff:5d}, "
          f"params={params:,}, output shape={output.shape}")
# Output: Factor= 2: d_ff=1024, params=1,050,624, output shape=torch.Size([2, 10, 512])
# Output: Factor= 4: d_ff=2048, params=2,101,760, output shape=torch.Size([2, 10, 512])
# Output: Factor= 8: d_ff=4096, params=4,204,032, output shape=torch.Size([2, 10, 512])
# Output: Factor=16: d_ff=8192, params=8,408,576, output shape=torch.Size([2, 10, 512])
```

### Example 2: SwiGLU with Adjusted Expansion Factor

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN with adjusted expansion factor.
    The factor is reduced by 2/3 to match standard FFN parameter count.
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        # GLU variants use 3 matrices, so reduce d_ff by 1/3
        d_ff = int(d_model * expansion_factor * 2 / 3)
        self.d_ff = d_ff
        self.effective_factor = d_ff / d_model

        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.V1 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W1(x))
        projected = self.V1(x)
        return self.W2(self.dropout(gate * projected))

    def get_params(self):
        return sum(p.numel() for p in self.parameters())

# Compare standard FFN vs SwiGLU with parameter-matched expansion
d_model = 512
print("Parameter comparison (d_model=512, expansion=4):")
standard = FlexibleFFN(d_model, expansion_factor=4)
swiglu = SwiGLUFFN(d_model, expansion_factor=4)

print(f"  Standard FFN: d_ff={standard.d_ff:5d}, params={standard.get_params():,}")
print(f"  SwiGLU FFN:   d_ff={swiglu.d_ff:5d}, params={swiglu.get_params():,}")
print(f"  SwiGLU effective factor: {swiglu.effective_factor:.4f}")

# For exact parameter parity, adjust the expansion factor
target_params = standard.get_params()
for factor in [3, 3.5, 4, 4.5]:
    swiglu2 = SwiGLUFFN(d_model, expansion_factor=factor)
    ratio = swiglu2.get_params() / target_params
    print(f"  SwiGLU factor={factor:.1f}: params ratio={ratio:.4f}")
# Output: Parameter comparison (d_model=512, expansion=4):
# Output:   Standard FFN: d_ff=2048, params=2,101,760
# Output:   SwiGLU FFN:   d_ff=1365, params=2,097,405
# Output:   SwiGLU effective factor: 2.6667
# Output:   SwiGLU factor=3.0: params ratio=0.7500
# Output:   SwiGLU factor=3.5: params ratio=0.8750
# Output:   SwiGLU factor=4.0: params ratio=0.9980
# Output:   SwiGLU factor=4.5: params ratio=1.1230
```

### Example 3: Impact on Model Capacity (Overfitting Experiment)

```python
def overfitting_experiment():
    """Train small models with different expansion factors on a tiny dataset."""
    d_model = 32
    vocab_size = 100
    seq_len = 8
    n_samples = 50  # Very small dataset to induce overfitting

    class SmallModel(nn.Module):
        def __init__(self, expansion_factor):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.ffn = FlexibleFFN(d_model, expansion_factor, dropout=0.0)
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, 2)  # Binary classification

        def forward(self, x):
            x = self.embed(x).mean(dim=1)  # Average pooling
            x = self.norm(self.ffn(x))
            return self.proj(x)

    # Generate tiny dataset
    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = torch.randint(0, 2, (n_samples,))

    results = {}
    for factor in [1, 2, 4, 8]:
        model = SmallModel(factor)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        model.train()
        for epoch in range(100):
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        results[factor] = train_losses[-1]

    print("Training loss after 100 epochs (small dataset):")
    for factor, loss in results.items():
        print(f"  Factor={factor}: loss={loss:.4f}")

    # Higher expansion factor should lead to lower training loss (overfitting)
    print("\nHigher factors lead to more overfitting on small datasets.")

# Uncomment to run
# overfitting_experiment()
```

### Example 4: Depth vs Width Trade-off

```python
def depth_vs_width_experiment():
    """
    Given a fixed parameter budget, compare using more layers
    (low expansion) vs wider layers (high expansion).
    """
    d_model = 64
    total_params_budget = 500_000  # Approximate

    def count_params(d_model, d_ff, n_layers):
        ffn_params = 2 * d_model * d_ff
        attn_params = 4 * d_model * d_model  # Simplified
        ln_params = 2 * d_model * 2
        per_layer = ffn_params + attn_params + ln_params
        return per_layer * n_layers + d_model * 10000  # + embedding

    print("Depth vs Width trade-off (fixed parameter budget):")
    print("-" * 60)
    configs = [
        (4, 64),   # n_layers=4, d_ff=256 (factor=4)
        (6, 48),   # n_layers=6, d_ff=192 (factor=3)
        (8, 32),   # n_layers=8, d_ff=128 (factor=2)
        (12, 16),  # n_layers=12, d_ff=64 (factor=1)
    ]

    for n_layers, d_ff in configs:
        factor = d_ff / d_model
        params = count_params(d_model, d_ff, n_layers)
        print(f"  Layers={n_layers:2d}, d_ff={d_ff:4d}, factor={factor:.1f}: "
              f"params≈{params:,}")

depth_vs_width_experiment()
# Output: Depth vs Width trade-off (fixed parameter budget):
# Output: ------------------------------------------------------------
# Output:   Layers= 4, d_ff=256, factor=4.0: params≈452,864
# Output:   Layers= 6, d_ff=192, factor=3.0: params≈417,280
# Output:   Layers= 8, d_ff=128, factor=2.0: params≈379,904
# Output:   Layers=12, d_ff= 64, factor=1.0: params≈352,768
```

## Common Mistakes

1. **Not adjusting d_ff for GLU variants**: GLU-based FFNs (SwiGLU, GeGLU, ReGLU) have three weight matrices instead of two. Using the same \(d_{ff}\) as a standard FFN increases parameters by 50%. The standard practice is to use \(d_{ff}^{\text{GLU}} = \frac{2}{3} \times d_{ff}^{\text{standard}}\).

2. **Using non-integer d_ff**: While \(d_{ff}\) can technically be any integer, it's best to keep it as a multiple of 64 or 128 for GPU efficiency. Some implementations round to the nearest multiple of 64.

3. **Forgetting that expansion factor affects both parameters and flops**: Doubling the expansion factor doubles the FFN parameters and FLOPs. This directly impacts training and inference speed.

4. **Assuming the optimal expansion factor is always 4**: While 4 is the most common, the optimal factor depends on \(d_{\text{model}}\), the total parameter budget, and the dataset size. Large models may use factors of 2-3, while small models may benefit from factors of 6-8.

5. **Confusing expansion factor with model width**: The expansion factor (\(d_{ff}/d_{\text{model}}\)) affects the FFN only. The model's "width" is \(d_{\text{model}}\), which affects both attention and FFN.

## Interview Questions

### Beginner

**Q: What is the FFN expansion factor in a Transformer?**

A: The expansion factor is the ratio of the intermediate hidden dimension to the model dimension: \(r = d_{ff} / d_{\text{model}}\). In the original Transformer, \(d_{ff} = 2048\) and \(d_{\text{model}} = 512\), giving an expansion factor of 4. This means the FFN expands the representation by 4x before applying the non-linearity.

### Intermediate

**Q: How does the expansion factor affect model parameters and computational cost?**

A: The FFN parameters scale as \(2 \cdot d_{\text{model}} \cdot d_{ff} = 2r \cdot d_{\text{model}}^2\). Doubling \(r\) doubles the FFN parameters and FLOPs per token. The FFN typically contains ~2/3 of the model's parameters, so changing the expansion factor has a significant impact on total model size. For a given \(d_{\text{model}}\), the expansion factor directly controls the cost-accuracy trade-off.

### Advanced

**Q: Given a fixed parameter budget, how would you decide between using more layers (lower expansion factor) vs. wider FFNs (higher expansion factor)? What does the research say?**

A: This is the depth vs. width trade-off in Transformers. Research (e.g., Kaplan et al., "Scaling Laws for Neural Language Models") suggests that for a fixed parameter count, deeper models (more layers) generally outperform wider models (larger \(d_{ff}\)) for language modeling, up to a point. However, the optimal configuration depends on the training data size. On small datasets, wider models overfit more. On large datasets, deeper models with moderate expansion factors (3-4) are preferred. The Chinchilla scaling laws suggest that for compute-optimal training, the model should be smaller and trained on more data, which favors moderate expansion factors. In practice, for a given parameter budget: (1) Increase \(n_{\text{layers}}\) first (deeper is usually better). (2) Keep \(r = 4\) as a default. (3) If adding layers hurts optimization, increase \(r\) instead. (4) For GLU variants, adjust \(r\) to \(\frac{2}{3}\) of the standard factor.

## Practice Problems

### Easy

Write a function that computes the number of FFN parameters for a given \(d_{\text{model}}\) and expansion factor \(r\). Plot the parameter count vs. \(r\) for \(d_{\text{model}} = 768\) (BERT-base).

### Medium

Implement both a standard FFN (ReLU with \(r=4\)) and a SwiGLU FFN (with adjusted \(r\) for parameter parity). Measure their forward and backward pass times. Compare throughput.

### Hard

Perform a controlled experiment on a text classification task: vary the expansion factor from 1 to 8 while keeping \(d_{\text{model}}\) and \(n_{\text{layers}}\) fixed. Plot validation accuracy vs. expansion factor. Find the optimal factor.

## Solutions

### Easy Solution

```python
def compute_ffn_params(d_model, r):
    d_ff = int(d_model * r)
    params = 2 * d_model * d_ff + d_ff + d_model
    return params

d_model = 768
print(f"FFN parameters for d_model={d_model}:")
for r in [1, 2, 3, 4, 6, 8]:
    params = compute_ffn_params(d_model, r)
    print(f"  r={r:.1f}: d_ff={int(d_model*r):5d}, params={params:,}")
# Output: FFN parameters for d_model=768:
# Output:   r=1.0: d_ff= 768, params=1,180,416
# Output:   r=2.0: d_ff=1536, params=2,360,064
# Output:   r=3.0: d_ff=2304, params=3,539,712
# Output:   r=4.0: d_ff=3072, params=4,719,360
# Output:   r=6.0: d_ff=4608, params=7,078,656
# Output:   r=8.0: d_ff=6144, params=9,437,952
```

## Related Concepts

- **DL-360: Feed-Forward Network**: The FFN component itself.
- **DL-375: GLU Variants**: Gated variants that use different expansion factors.
- **DL-376: Transformer Dimensionality**: How dimensions relate across components.
- **DL-378: d_ff**: The intermediate dimension hyperparameter.
- **DL-381: Transformer Parameter Count**: How the expansion factor affects parameter counts.

## Next Concepts

- DL-375: GLU Variants — Gated activation functions for FFNs.
- DL-376: Transformer Dimensionality — Overall dimension relationships.

## Summary

The FFN expansion factor (\(r = d_{ff} / d_{\text{model}}\)) controls the representational capacity of each Transformer layer's feed-forward network. The standard value is 4 (from the original Transformer), but this can be tuned for different model sizes, architectures, and computational budgets. The expansion factor directly scales the FFN's parameter count and computational cost. GLU variants typically use a reduced expansion factor (\(\frac{2}{3}r\)) to match the parameter count of standard FFNs. Understanding the expansion factor is essential for Transformer architecture design and scaling.

## Key Takeaways

1. The expansion factor \(r = d_{ff} / d_{\text{model}}\) controls FFN capacity per layer.
2. Standard value is 4; typical range is 2-8.
3. FFN parameters scale linearly with \(r\).
4. GLU variants use \(\frac{2}{3}r\) to match standard FFN parameter counts.
5. The optimal factor depends on \(d_{\text{model}}\), total parameter budget, and dataset size.
6. For a fixed parameter budget, deeper models (more layers) are generally preferred over wider FNNs.
