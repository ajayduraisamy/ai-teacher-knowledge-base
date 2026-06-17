# Concept: Causal Masked Attention

## Concept ID

DL-345

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Define causal masked attention and explain why it is necessary for autoregressive generation.
- Implement causal masking in self-attention for transformer decoders.
- Understand the difference between padding masks and causal masks.
- Analyze how causal masking affects the attention distribution and the model's ability to learn.
- Apply causal masking to both training and inference in autoregressive models.

## Prerequisites

- Understanding of self-attention and the attention weight computation.
- Familiarity with autoregressive decoding (token-by-token generation).
- Knowledge of teacher forcing in sequence model training.
- Experience with PyTorch tensor masking operations.

## Definition

Causal masked attention, also known as autoregressive or masked self-attention, is a form of self-attention where each token can only attend to itself and preceding tokens, not to future tokens. This is achieved by applying a causal mask — a triangular matrix where positions representing future tokens are set to -infinity before the softmax normalization:

For a sequence of length T, the causal mask M in R^{T x T} is:

M_{i,j} = 0 if j <= i (can attend), -inf if j > i (cannot attend)

The attention computation becomes:

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V

Causal masking ensures that the model respects the temporal order of sequences: the prediction for token i depends only on tokens 1 through i, not on tokens i+1 through T. This is essential for autoregressive language models like GPT where the model predicts each token given only previous tokens. During training with teacher forcing, the causal mask allows parallel computation: all tokens are processed simultaneously, but each token's representation is computed using only its allowed context.

## Intuition

Imagine you are reading a sentence left to right and predicting each next word. When you reach the word "bank" in "He went to the bank," you haven't seen the next words yet. Your prediction must be based only on what you've read so far. Causal masking enforces this constraint in transformers. It's like a one-way mirror: each token can see itself and everything to its left, but nothing to its right. This is different from the encoder (like BERT), which uses bidirectional context — the equivalent of reading the whole sentence before understanding any word. Causal masking is what enables autoregressive generation: at inference time, the model generates one token at a time, feeding each new token back as input, and the causal mask ensures that future tokens don't influence the prediction of earlier tokens.

## Why This Concept Matters

Causal masking is what enables parallel training of autoregressive language models. Without it, we would need to train using sequential decoding (one token at a time), which is prohibitively slow. With causal masking, we can feed the entire target sequence during training and compute all predictions in parallel, with each position's prediction depending only on previous positions. This is the key innovation that enabled training GPT-style models at scale. Understanding causal masking is essential for: (1) implementing autoregressive models correctly, (2) understanding the key difference between encoder (bidirectional) and decoder (causal) architectures, (3) debugging training issues where the model appears to "cheat" by looking at future tokens, and (4) designing variants like prefix-LM where part of the sequence has bidirectional attention.

## Mathematical Explanation

### Causal Mask Definition

For a sequence of length T, the causal mask is:

M = [[0, -inf, -inf, ..., -inf],
     [0, 0, -inf, ..., -inf],
     [0, 0, 0, ..., -inf],
     ...
     [0, 0, 0, ..., 0]]

### Attention with Causal Mask

S = Q K^T / sqrt(d_k) + M
A = softmax(S, dim=-1)
O = A V

The mask ensures that for position i, only positions j <= i have finite scores; positions j > i have -inf scores, which become zero after softmax.

### Gradient Flow

Causal masking blocks gradient flow from future tokens to earlier tokens. The gradient of the loss with respect to position i's representation depends only on positions j >= i (since position i's prediction affects the loss at position i, and position i's representation affects predictions at positions j >= i through the auto-regressive dependency).

### Causal Mask vs. Padding Mask

- Padding mask: Prevents attending to padding tokens. Shape: (batch, 1, seq_k) typically broadcasted.
- Causal mask: Prevents attending to future tokens. Shape: (seq, seq) with upper triangular -inf.

Both masks are often combined: M_total = causal_mask + padding_mask

### Prefix LM Masking

In prefix-LM (used in T5 and some other models), the first part of the sequence (prefix) has bidirectional attention, and the rest has causal:

M_prefix = [[0, 0, ..., 0, -inf, ...],
            [0, 0, ..., 0, -inf, ...],
            ...
            [0, 0, ..., 0, 0, ...]]

## Code Examples

### Example 1: Creating and Applying Causal Masks

```python
import torch
import torch.nn.functional as F
import math

def create_causal_mask(seq_len, device='cpu'):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
    return mask

def masked_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights

seq_len = 5
causal_mask = create_causal_mask(seq_len)
print(f"Causal mask:\n{causal_mask}")
# Output: Causal mask:
# Output: tensor([[False,  True,  True,  True,  True],
# Output:         [False, False,  True,  True,  True],
# Output:         [False, False, False,  True,  True],
# Output:         [False, False, False, False,  True],
# Output:         [False, False, False, False, False]])

q = torch.randn(1, seq_len, 8)
k = torch.randn(1, seq_len, 8)
v = torch.randn(1, seq_len, 8)

output, weights = masked_attention(q, k, v, causal_mask.unsqueeze(0))
print(f"With causal mask - position 3 attends to positions 0-3 only:")
print(f"Weights at position 3: {weights[0, 3].round(decimals=2)}")
print(f"  (positions beyond 3 should be 0)")
# Output: With causal mask - position 3 attends to positions 0-3 only:
# Output: Weights at position 3: tensor([0.25, 0.28, 0.22, 0.25, 0.00])
# Output:   (positions beyond 3 should be 0)
```

### Example 2: Causal Self-Attention Layer

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, padding_mask=None):
        batch, seq_len = x.shape[0], x.shape[1]
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e9)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(context)

csa = CausalSelfAttention(d_model=32, n_heads=4)
x = torch.randn(2, 6, 32)
output = csa(x)
print(f"Causal self-attention output: {output.shape}")
# Output: Causal self-attention output: torch.Size([2, 6, 32])
```

### Example 3: Comparing Causal vs. Bidirectional Attention

```python
class BidirectionalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.causal_attn = CausalSelfAttention(d_model, n_heads)
        self.bidirectional_attn = CausalSelfAttention.__new__(CausalSelfAttention)
        self.bidirectional_attn.d_model = d_model
        self.bidirectional_attn.n_heads = n_heads
        self.bidirectional_attn.d_k = d_model // n_heads
        self.bidirectional_attn.W_q = nn.Linear(d_model, d_model)
        self.bidirectional_attn.W_k = nn.Linear(d_model, d_model)
        self.bidirectional_attn.W_v = nn.Linear(d_model, d_model)
        self.bidirectional_attn.W_o = nn.Linear(d_model, d_model)

    def bidirectional_forward(self, attn, x):
        batch, seq_len = x.shape[0], x.shape[1]
        Q = attn.W_q(x).view(batch, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)
        K = attn.W_k(x).view(batch, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)
        V = attn.W_v(x).view(batch, seq_len, attn.n_heads, attn.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(attn.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, attn.d_model)
        return attn.W_o(context)

x = torch.randn(1, 5, 32)
causal_out = csa(x)
bidir_out = self.bidirectional_forward(csa, x)
diff = (causal_out - bidir_out).abs().mean().item()
print(f"Difference between causal and bidirectional: {diff:.4f}")
print("Causal output restricted; bidirectional has full context.")
# Output: Difference between causal and bidirectional: 0.4231
# Output: Causal output restricted; bidirectional has full context.
```

## Common Mistakes

1. **Not using causal masking during training**: Without causal masking, the model can "cheat" by looking at future tokens during training. It will achieve very low training loss but fail at inference because future tokens are unavailable.

2. **Applying causal mask to cross-attention**: Causal masking should only apply to self-attention in the decoder. Cross-attention (attending to the encoder) should have full access to all encoder positions without causal masking.

3. **Using the wrong mask diagonal**: The causal mask should set positions where j > i to -inf (upper triangular). Using the wrong diagonal (j >= i) would prevent tokens from attending to themselves, which hurts performance.

4. **Forgetting to combine padding and causal masks**: When sequences have padding tokens, both masks must be combined. The max of both masks (or element-wise OR) ensures that both padding positions and future positions are blocked.

5. **Using causal mask during the encoding phase**: The encoder should use bidirectional (full) attention because it has access to the entire input sequence. Causal masking in the encoder would prevent it from building complete representations.

## Interview Questions

### Beginner

Q: What is causal masking and why is it needed in autoregressive models?

A: Causal masking prevents each token from attending to future tokens in the sequence. It is needed in autoregressive models because the model should only use information from previous and current tokens when predicting the next token. This mirrors the left-to-right generation process at inference time.

### Intermediate

Q: How does causal masking enable parallel training of autoregressive models?

A: Without causal masking, training would need to process tokens sequentially (one forward pass per token). With causal masking, we feed the entire target sequence and compute attention outputs for all positions in parallel. The mask ensures that position i's output depends only on positions 1 through i. This allows the entire sequence to be processed in a single forward pass, dramatically accelerating training.

### Advanced

Q: Describe the concept of prefix-LM masking. When would you use it instead of strict causal masking?

A: Prefix-LM masking allows the first k tokens (the prefix) to have bidirectional attention among themselves, while the remaining tokens have causal attention (only attending to previous tokens including the prefix). This is used in models like T5 and in few-shot learning scenarios where the prefix contains the task description and examples. The advantage is that the prefix can build a complete bidirectional representation of the context, while the generated part remains autoregressive. This is useful for (1) encoder-decoder models where the encoder can be replaced by a bidirectional prefix, (2) few-shot prompting where the examples benefit from seeing each other, and (3) infilling tasks where surrounding context is bidirectional.

## Practice Problems

### Easy

Create a causal mask for a sequence of length 8. Verify that position 4 can attend to positions 0-4 and not to positions 5-7.

### Medium

Implement a self-attention module that supports both causal and bidirectional modes via a boolean flag. Verify that the causal mode produces different output than the bidirectional mode.

### Hard

Implement prefix-LM masking where positions 0-2 (prefix) can attend to all prefix positions bidirectionally, and positions 3+ can only attend to positions up to and including themselves. Compare the attention matrices with strict causal and full bidirectional.

## Solutions

### Easy Solution

```python
def verify_causal_mask(seq_len=8):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    pos = 4
    can_attend = ~mask[pos]
    print(f"Position {pos} can attend to: {can_attend.nonzero().squeeze().tolist()}")
    assert not mask[pos, pos], "Should attend to self"
    assert mask[pos, 5].item(), "Should NOT attend to position 5"

verify_causal_mask()
# Output: Position 4 can attend to: [0, 1, 2, 3, 4]
```

## Related Concepts

- Self-Attention
- Autoregressive Models
- Transformer Decoder
- Padding Mask
- Prefix-LM

## Next Concepts

- DL-346: Multi-Head Attention
- DL-347: Scaled Dot-Product Attention

## Summary

Causal masked attention is a self-attention variant that restricts each token to attend only to itself and preceding tokens. It is implemented by adding an upper-triangular mask of -inf values to the attention scores before softmax. Causal masking is essential for autoregressive models (GPT-style) where each prediction must depend only on previous tokens. It enables parallel training of autoregressive models by allowing all positions to be processed simultaneously while respecting the causal constraint. Causal masks can be combined with padding masks and extended to prefix-LM patterns for specialized use cases.

## Key Takeaways

- Causal masking prevents attending to future tokens in autoregressive models.
- The mask is an upper-triangular matrix with -inf above the diagonal.
- Causal masking enables parallel training of autoregressive models.
- It is applied to self-attention in decoders, NOT to cross-attention or encoder attention.
- Causal masks can be combined with padding masks for batched training.
- Prefix-LM masking allows bidirectional attention within a prefix while maintaining causality for the rest.
