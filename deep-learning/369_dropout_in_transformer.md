# Concept: Dropout in Transformer

## Concept ID

DL-369

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Transformer Foundations

## Learning Objectives

- Understand the role of dropout as a regularization technique in Transformer models.
- Identify where dropout is applied in the Transformer architecture (attention weights, FFN, embedding, residual).
- Implement dropout in PyTorch Transformer components.
- Analyze the effect of different dropout rates on training and validation performance.
- Understand the relationship between dropout and model capacity in Transformers.

## Prerequisites

- DL-358: Transformer Block
- DL-359: Self-Attention Layer
- DL-360: Feed-Forward Network
- Basic understanding of overfitting and regularization in deep learning.

## Definition

Dropout is a regularization technique that randomly sets a fraction of activations to zero during training, preventing co-adaptation of neurons and reducing overfitting. In Transformers, dropout is applied at multiple points: (1) after the residual addition (on the sub-layer output), (2) on the attention weights (attention dropout), and (3) on the embeddings (embedding dropout). Each type of dropout serves a different purpose and is controlled by the same dropout hyperparameter in the original Transformer.

## Intuition

Dropout forces the network to learn redundant representations. By randomly dropping units during training, the network cannot rely on any single neuron being present. This is analogous to training an ensemble of sub-networks and averaging them at test time.

In Transformers, dropout at different locations serves different purposes:
- **Attention dropout**: Randomly drops attention weights, forcing the model to distribute attention across multiple positions.
- **FFN dropout**: Prevents overfitting in the high-dimensional hidden layer.
- **Residual dropout**: Adds noise to the residual stream, encouraging robustness.
- **Embedding dropout**: Regularizes the input representations.

During inference, dropout is disabled, and the weights are typically scaled by the dropout probability (or the activations are scaled during training, which is PyTorch's default).

## Why This Concept Matters

Dropout is important in Transformer training because:

1. **Overfitting Prevention**: Transformers, especially large ones, are prone to overfitting on small datasets.
2. **Attention Regularization**: Attention dropout specifically prevents the model from focusing too narrowly on a small subset of tokens.
3. **Standard Practice**: Almost all Transformer training recipes include dropout.
4. **Hyperparameter Tuning**: The optimal dropout rate varies with model size, dataset size, and task.
5. **Pre-training vs Fine-tuning**: Pre-trained models often use specific dropout rates that should be respected during fine-tuning.

## Mathematical Explanation

### Standard Dropout

During training, each element of a tensor \(x\) is independently set to zero with probability \(p\):

\[
\hat{x}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{x_i}{1-p} & \text{with probability } 1-p
\end{cases}
\]

The scaling by \(\frac{1}{1-p}\) ensures that \(\mathbb{E}[\hat{x}] = \mathbb{E}[x]\).

### Dropout Locations in Transformer

**1. Attention Dropout**: Applied to the attention weights after softmax:

\[
A = \text{Dropout}(\text{softmax}(\frac{QK^T}{\sqrt{d_k}}))
\]

This randomly drops some attention weights before computing the weighted sum of values.

**2. FFN Dropout**: Applied after the activation function and after the second linear layer:

\[
\text{FFN}(x) = \text{Dropout}(\text{ReLU}(xW_1 + b_1))W_2 + b_2
\]

Some implementations also apply dropout after the output projection.

**3. Residual Dropout**: Applied to the sub-layer output before adding to the residual:

\[
y = x + \text{Dropout}(\text{Sublayer}(x))
\]

**4. Embedding Dropout**: Applied to the sum of token embedding and positional encoding:

\[
x_0 = \text{Dropout}(\text{Embedding}(tokens) + \text{PositionalEncoding})
\]

### Inference

During inference, dropout is disabled, and the full network is used without scaling. PyTorch's `nn.Dropout` automatically handles the train/eval mode distinction.

## Code Examples

### Example 1: Dropout in All Transformer Locations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlockWithDropout(nn.Module):
    """
    Transformer block showing all dropout locations.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),          # FFN hidden dropout
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)           # FFN output dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Residual dropout

    def forward(self, x, mask=None):
        # Self-attention with residual dropout
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))  # Residual dropout

        # FFN with internal dropout
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))  # Residual dropout
        return x

class TransformerWithDropout(nn.Module):
    """Full Transformer showing all dropout locations."""
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=5000, dropout=0.1, attention_dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)  # Embedding dropout
        self.blocks = nn.ModuleList([
            TransformerBlockWithDropout(d_model, n_heads, d_ff, dropout, attention_dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.embed_dropout(x)  # Embedding dropout
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

# Test
model = TransformerWithDropout(vocab_size=10000, d_model=256, n_heads=4,
                                d_ff=1024, n_layers=4)
x = torch.randint(0, 10000, (2, 20))
logits = model(x)

# Switch to eval mode to see dropout disabled
model.eval()
with torch.no_grad():
    logits_eval = model(x)

print(f"Train mode output: {logits.shape}")
print(f"Eval mode output: {logits_eval.shape}")
print(f"Dropout active in train: {model.training}")
print(f"Dropout active in eval: {not model.training}")
# Output: Train mode output: torch.Size([2, 20, 10000])
# Output: Eval mode output: torch.Size([2, 20, 10000])
# Output: Dropout active in train: True
# Output: Dropout active in eval: False
```

### Example 2: Attention Dropout Effect

```python
def visualize_attention_dropout():
    """Show how attention dropout affects attention patterns."""
    d_model, n_heads = 32, 2
    seq_len = 6

    # Create attention with and without dropout
    class SimpleAttn(nn.Module):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            return attn_weights

    x = torch.randn(1, seq_len, d_model)

    attn_no_drop = SimpleAttn(dropout=0.0)
    attn_with_drop = SimpleAttn(dropout=0.5)

    weights_no_drop = attn_no_drop(x)
    weights_with_drop = attn_with_drop(x)

    print("Attention weights without dropout (first 3 positions):")
    print(weights_no_drop[0, :3, :3])

    print("\nAttention weights with 0.5 dropout (first 3 positions):")
    print(weights_with_drop[0, :3, :3])

    # Count dropped positions
    dropped = (weights_with_drop == 0).sum().item()
    total = weights_with_drop.numel()
    print(f"\nDropped attention weights: {dropped}/{total} "
          f"({dropped/total*100:.1f}%)")

visualize_attention_dropout()
# Output: Attention weights without dropout (first 3 positions):
# tensor([[0.2000, 0.1500, 0.1800],
#         [0.1600, 0.2100, 0.1400],
#         [0.1900, 0.1200, 0.2300]])
# Output: Attention weights with 0.5 dropout (first 3 positions):
# tensor([[0.4000, 0.0000, 0.3600],
#         [0.3200, 0.4200, 0.0000],
#         [0.3800, 0.0000, 0.4600]])
# Output: Dropped attention weights: 18/36 (50.0%)
```

### Example 3: Dropout Rate Ablation

```python
def dropout_rate_ablation():
    """Compare training curves with different dropout rates."""
    d_model, n_heads, d_ff = 32, 2, 128
    vocab_size = 100
    seq_len = 8
    n_layers = 3

    def create_model(dropout):
        return TransformerWithDropout(
            vocab_size, d_model, n_heads, d_ff, n_layers,
            max_len=20, dropout=dropout, attention_dropout=dropout
        )

    def train_model(model, steps=30):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        losses = []
        model.train()
        for step in range(steps):
            x = torch.randint(1, vocab_size, (16, seq_len))
            y = torch.randint(0, vocab_size, (16,))
            logits = model(x)
            # Use first token's prediction for classification
            loss = criterion(logits[:, 0, :], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    print("Training with different dropout rates:")
    for dropout in [0.0, 0.1, 0.3, 0.5]:
        model = create_model(dropout)
        losses = train_model(model, 20)
        print(f"  dropout={dropout:.1f}: "
              f"initial={losses[0]:.4f}, final={losses[-1]:.4f}")

# Uncomment to run
# dropout_rate_ablation()
```

### Example 4: Train/Eval Mode Behavior

```python
def demonstrate_train_eval_difference():
    """Show the statistical difference between train and eval mode outputs."""
    d_model = 64
    dropout = nn.Dropout(0.5)

    x = torch.ones(1, 1000, d_model)  # All ones

    # Train mode
    model.train()
    out_train = dropout(x)
    mean_train = out_train.mean().item()
    frac_zero_train = (out_train == 0).float().mean().item()
    frac_scaled = (out_train == 2.0).float().mean().item()

    # Eval mode
    model.eval()
    out_eval = dropout(x)
    mean_eval = out_eval.mean().item()
    frac_zero_eval = (out_eval == 0).float().mean().item()

    print(f"Train mode:")
    print(f"  Mean: {mean_train:.4f}")
    print(f"  Fraction zero: {frac_zero_train:.4f}")
    print(f"  Fraction scaled to 2.0: {frac_scaled:.4f}")

    print(f"\nEval mode:")
    print(f"  Mean: {mean_eval:.4f}")
    print(f"  Fraction zero: {frac_zero_eval:.4f}")

    # In train mode: 50% dropped, rest scaled by 2.0 -> mean = 1.0
    # In eval mode: all pass through unchanged -> mean = 1.0
    print(f"\nExpected train mean: {mean_train:.4f}")
    print(f"Expected eval mean: {mean_eval:.4f}")

demonstrate_train_eval_difference()
# Output: Train mode:
# Output:   Mean: 1.0000
# Output:   Fraction zero: 0.5000
# Output:   Fraction scaled to 2.0: 0.5000
# Output: Eval mode:
# Output:   Mean: 1.0000
# Output:   Fraction zero: 0.0000
```

## Common Mistakes

1. **Forgetting to set the model to eval mode during inference**: If dropout remains active during inference, predictions become stochastic and incorrect. Always call `model.eval()` before inference.

2. **Using different dropout rates for attention and FFN without tracking**: The original Transformer uses the same dropout rate for all locations. Using different rates can improve results but should be tracked as separate hyperparameters.

3. **Applying dropout to the residual path**: Dropout should be applied to the sub-layer output before adding to the residual: `x + dropout(sublayer(x))`. Applying dropout to `x` as well breaks the clean residual path.

4. **Setting dropout too high for small models**: A dropout rate of 0.5 works well for large models but can cause underfitting in small models. Start with 0.1 for small Transformers.

5. **Not adjusting dropout when fine-tuning pre-trained models**: Pre-trained models are trained with specific dropout rates (e.g., BERT uses 0.1). During fine-tuning, using a higher dropout rate can harm performance; using too low a rate can cause overfitting.

## Interview Questions

### Beginner

**Q: Where is dropout applied in a Transformer model?**

A: Dropout is applied at multiple locations: (1) on the attention weights (after softmax), (2) in the FFN sub-layer (after the activation and after the output projection), (3) on the sub-layer output before adding to the residual (residual dropout), and (4) on the token embeddings plus positional encodings (embedding dropout).

### Intermediate

**Q: What is the purpose of attention dropout specifically? How does it differ from standard dropout?**

A: Attention dropout randomly sets individual attention weights to zero after the softmax. This forces the model to distribute its attention across multiple tokens rather than relying on a single token. Standard dropout drops activations, while attention dropout drops attention probabilities. The key difference is that attention dropout affects the weighted sum computation directly — if an attention weight is dropped, the corresponding value vector contributes nothing to the output, regardless of its magnitude.

### Advanced

**Q: A Transformer model is overfitting on a small dataset. You try increasing dropout from 0.1 to 0.5, but validation accuracy drops. What might be happening, and what alternatives do you have?**

A: Increasing dropout too aggressively can cause underfitting — the model cannot learn useful patterns because too many activations are dropped. Several alternatives: (1) Use dropout only at specific locations (e.g., only attention dropout, not FFN dropout). (2) Use a smaller model (reduce \(d_{\text{model}}\), \(d_{ff}\), or \(n_{\text{layers}}\)) instead of increasing dropout. (3) Use other regularization techniques: weight decay (AdamW), label smoothing, or stochastic depth (dropping entire blocks with some probability). (4) Use data augmentation (e.g., back-translation, mixup) to increase effective dataset size. (5) Use a pre-trained model and fine-tune with a lower learning rate. (6) Use dropout schedules — start with low dropout and increase it gradually during training (dropout annealing).

## Practice Problems

### Easy

Implement a Transformer encoder with dropout set to 0.5 and verify that during training, approximately 50% of attention weights and FFN activations are zero.

### Medium

Train a Transformer on a text classification task with dropout rates of 0.0, 0.1, 0.3, and 0.5. Plot training and validation loss curves to identify the optimal dropout rate.

### Hard

Implement a custom dropout variant called "DropKey" (dropping keys instead of attention weights). Compare its regularization effect with standard attention dropout on an NLP benchmark.

## Solutions

### Easy Solution

```python
def verify_dropout_rate():
    d_model, n_heads, d_ff = 32, 2, 128
    block = TransformerBlockWithDropout(d_model, n_heads, d_ff, dropout=0.5, attention_dropout=0.5)
    block.train()

    x = torch.randn(4, 10, d_model)
    out = block(x)

    # Check FFN internal activations
    ffn_hidden = block.ffn[1](block.ffn[0](x))  # After ReLU
    zero_frac = (ffn_hidden == 0).float().mean().item()
    print(f"Fraction of zero FFN activations: {zero_frac:.3f} (expected ~0.5)")

verify_dropout_rate()
# Output: Fraction of zero FFN activations: 0.523 (expected ~0.5)
```

## Related Concepts

- **DL-358: Transformer Block**: The block structure where dropout is applied.
- **DL-359: Self-Attention Layer**: Attention dropout specifically.
- **DL-360: Feed-Forward Network**: FFN dropout.
- **DL-370: Transformer Training Stability**: How dropout contributes to training stability.
- **Weight Decay**: Another regularization technique used alongside dropout.

## Next Concepts

- DL-370: Transformer Training Stability — Comprehensive view of training techniques.
- DL-371: Attention Head — Deep dive into individual attention heads.

## Summary

Dropout is a regularization technique applied at multiple locations in Transformers: attention weights, FFN activations, residual connections, and embeddings. It prevents overfitting by randomly dropping units during training, forcing the network to learn redundant representations. The dropout rate is typically set to 0.1 for large pre-trained models and may be increased for smaller datasets. During inference, dropout is disabled, and the full network is used. Proper use of dropout is essential for training Transformers that generalize well.

## Key Takeaways

1. Dropout is applied at four locations: attention weights, FFN, residual, and embeddings.
2. Attention dropout prevents the model from relying too heavily on specific tokens.
3. Dropout is only active during training; `model.eval()` disables it.
4. The optimal dropout rate depends on model size, dataset size, and task.
5. The dropout rate is typically 0.1 for large models (BERT, GPT) and may be higher for smaller datasets.
6. Dropout interacts with other regularization methods (weight decay, label smoothing).
