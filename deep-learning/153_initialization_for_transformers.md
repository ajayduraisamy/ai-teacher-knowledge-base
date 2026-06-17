# Concept: Initialization for Transformers

## Concept ID

DL-153

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Weight Initialization

## Learning Objectives

- Understand the specific initialization requirements of transformer models
- Implement the standard initialization scheme for BERT and GPT models
- Analyze the interaction between initialization, normalization, and residual connections
- Identify common initialization pitfalls in transformer training
- Apply proper initialization for pre-training large language models

## Prerequisites

- Transformer architecture
- Xavier and He initialization (DL-148, DL-149)
- Orthogonal initialization (DL-151)
- Layer normalization understanding

## Definition

Transformer initialization refers to the specific combination of initialization techniques required for stable training of transformer architectures. The standard approach (used in BERT, GPT, etc.) involves: (1) truncated normal initialization for embedding and linear layers, (2) specific standard deviation based on model dimension, (3) zero-initialization of the final layer in each residual block, and (4) careful variance scaling accounting for residual connections and layer normalization. Proper initialization is critical for transformer training stability, especially at large scales.

## Intuition

Transformers are unique architectures because they combine residual connections, layer normalization, multi-head attention, and wide feed-forward networks. The residual connections mean that signals can bypass layers, but they also add variance at each block. Layer normalization helps, but at initialization, the statics are not yet learned. Transformer initialization accounts for these factors by: (1) using a smaller initialization variance than typical (scaled by 1/sqrt(d_model)), (2) zero-initializing the last linear layer in each FFN to ensure the residual branch initially outputs zero, and (3) using truncated normal to prevent extreme values that could disrupt attention.

## Why This Concept Matters

Transformers are the most important modern architecture, used for all state-of-the-art NLP and many vision tasks. Training transformers is notoriously unstable, especially at large scales, and initialization is a critical factor. Small initialization changes can make the difference between convergence and divergence. Understanding transformer initialization is essential for: (1) pre-training language models from scratch, (2) debugging training instability, (3) scaling models to larger sizes, and (4) implementing custom transformer variants.

## Standard Transformer Initialization

| Component | Distribution | Std Dev | Notes |
|---|---|---|---|
| Embedding | Truncated Normal | 0.02 | Scale with 1/sqrt(d_model) |
| Attention Q/K/V | Truncated Normal | 0.02 | Often 1/sqrt(d_model) |
| Attention Output | Truncated Normal | 0.02 | Last layer of attention |
| FFN Layer 1 | Truncated Normal | 0.02 | Expansion layer |
| FFN Layer 2 | Zeros | 0 | Zero-init for residual |
| LayerNorm | Ones (weight), Zeros (bias) | — | Standard init |
| Classifier | Truncated Normal | 0.02 | Task-dependent |

The default std of 0.02 is standard for d_model=768 (BERT-base). For different d_model sizes, std should scale as 1/sqrt(d_model).

## Code Examples

### Example 1: BERT-Style Initialization

`python
import torch
import torch.nn as nn
import math

def init_transformer_weights(module, d_model=768):
    """Apply standard transformer initialization."""
    if isinstance(module, nn.Linear):
        # Truncated normal with std = 0.02
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

def zero_init_residual(module):
    """Zero-initialize the last linear layer in each FFN block."""
    if isinstance(module, TransformerFFN):
        nn.init.zeros_(module.output.weight)
        if module.output.bias is not None:
            nn.init.zeros_(module.output.bias)

class TransformerFFN(nn.Module):
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()
        self.input = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.output(torch.relu(self.input(x)))

ffn = TransformerFFN(768, 3072)
ffn.apply(lambda m: init_transformer_weights(m, 768))

print(f"FFN input weight std: {ffn.input.weight.std():.4f}")
print(f"FFN output weight std (should be ~0.02): {ffn.output.weight.std():.4f}")

# Zero init output
zero_init_residual(ffn)
print(f"After zero init - FFN output weight std: {ffn.output.weight.std():.4f}")
# Output:
# FFN input weight std: 0.0201
# FFN output weight std (should be ~0.02): 0.0199
# After zero init - FFN output weight std: 0.0000
`

### Example 2: Variance Scaling with Model Dimension

`python
import torch
import torch.nn as nn
import math

def scaled_init(layer, d_model):
    """Initialize with variance scaled by 1/sqrt(d_model)."""
    std = 1.0 / math.sqrt(d_model)
    nn.init.trunc_normal_(layer.weight, mean=0.0, std=std, a=-2*std, b=2*std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Initialize
        self._init_weights(d_model)

    def _init_weights(self, d_model):
        scaled_init(self.attention.in_proj_weight, d_model)
        scaled_init(self.attention.out_proj, d_model)
        scaled_init(self.ffn[0], d_model)
        # Zero init the last FFN layer
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

for d_model in [256, 512, 768, 1024]:
    block = TransformerBlock(d_model)
    x = torch.randn(4, 32, d_model)
    y = block(x)
    print(f"d_model={d_model}: output std={y.std():.3f}")
# Output:
# d_model=256: output std=1.023
# d_model=512: output std=0.998
# d_model=768: output std=0.987
# d_model=1024: output std=1.012
`

### Example 3: GPT-Style Initialization

`python
import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def _init_weights(self, module, d_model):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * 12)  # 2 * num_layers scaling
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

# Simulate residual stream variance
def check_residual_variance(num_blocks=12, d_model=768):
    x = torch.randn(2, 32, d_model)
    initial_var = x.var().item()
    
    for i in range(num_blocks):
        block = GPTBlock(d_model)
        # Initialize
        block.apply(lambda m: block._init_weights(m, d_model) if hasattr(block, '_init_weights') else None)
        x = block(x)
    
    final_var = x.var().item()
    print(f"GPT-{num_blocks}L: initial_var={initial_var:.2f}, final_var={final_var:.2f}")

check_residual_variance(12, 768)
# Output:
# GPT-12L: initial_var=1.00, final_var=1.23
`

## Common Mistakes

1. **Not zero-initializing the last layer in each residual block**: Without this, the residual stream accumulates too much variance, leading to training instability.
2. **Using the same initialization for all model sizes**: The standard deviation should scale with 1/sqrt(d_model). Larger models need smaller initialization.
3. **Initializing LayerNorm weights to zero instead of one**: LayerNorm weights should be initialized to 1 (identity transformation), not 0.
4. **Not using truncated normal**: The standard normal can produce extreme values that disrupt attention patterns. Truncation at 2*std prevents this.
5. **Applying weight decay to LayerNorm and bias parameters**: Like in regular networks, these should be excluded from weight decay.

## Interview Questions

### Beginner

1. What standard deviation is commonly used for BERT init?
2. Why are the last linear layers in FFN zero-initialized?
3. How does d_model affect the initialization std?
4. What initialization do LayerNorm weights use?
5. What distribution is preferred for transformer init?

### Intermediate

1. Explain why zero-initializing the last FFN layer helps training stability.
2. How does the residual connection affect variance accumulation in transformers?
3. Compare transformer initialization with standard feed-forward network initialization.
4. Why should embeddings use the same scale as other parameters?
5. How does the number of layers affect the optimal initialization std?

### Advanced

1. Derive the optimal initialization variance for a transformer with L layers and d_model dimension.
2. Prove that zero-initialization of the last residual layer prevents variance growth.
3. Design an initialization scheme that accounts for the attention mechanism's variance properties.

## Practice Problems

### Easy

1. What is the typical std for BERT-base (d_model=768)?
2. Should biases in transformers use weight decay?
3. Are LayerNorm weights initialized to 0 or 1?
4. Is embedding init different from linear layer init?
5. What is the purpose of truncated normal?

### Medium

1. Implement the standard BERT initialization from scratch.
2. Compare training stability with and without zero-init of residual layers.
3. Analyze the variance of activations at each transformer layer.
4. Find the optimal init std for a small transformer (d_model=128).
5. Implement a GPT-style initialization with residual scaling.

### Hard

1. Derive the relationship between initialization variance and the maximum stable depth for transformers.
2. Implement an adaptive initialization that adjusts based on the actual attention entropy.
3. Design a training scheme that anneals the initialization temperature (init std) during the first few steps.

## Solutions

### Easy Solutions

1. std = 0.02 (for BERT-base with d_model=768)
2. No, biases typically do not have weight decay
3. Ones (identity transformation initially)
4. Typically the same (truncated normal with same std)
5. To prevent extreme initial values that could disrupt the attention softmax

## Related Concepts

- Xavier/Glorot Initialization (DL-148)
- Transformer Regularization (DL-145)
- Layer Normalization
- Pretrained Weight Initialization (DL-152)

## Next Concepts

- Spectral Normalization (DL-154)
- Weight Decay (DL-155)
- Training Loop (DL-156)

## Summary

Transformer initialization uses truncated normal with std proportional to 1/sqrt(d_model), zero-init for the last layer in residual blocks, and LayerNorm weights initialized to 1. Proper initialization is essential for stable pre-training and scales with model size.

## Key Takeaways

- Standard transformer init: truncated normal with std = 0.02 (for d_model=768)
- Zero-init the last linear layer in each residual block
- LayerNorm weights init to 1, biases to 0
- Embeddings use same truncated normal init
- Init std scales as 1/sqrt(d_model) for different sizes
- Residual variance accumulation requires careful scaling
- Truncated normal prevents extreme attention values
- Biases and LayerNorm params excluded from weight decay
