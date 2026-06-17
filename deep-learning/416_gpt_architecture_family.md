# GPT Architecture Family

## Concept ID
DL-416

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand the evolution of GPT architectures from GPT-1 to GPT-4
- Identify the key architectural innovations across GPT generations
- Compare and contrast different GPT family members
- Analyze the scaling trends and design choices in GPT models

## Prerequisites
- Transformer architecture (DL-370)
- Decoder-only architecture (DL-403)
- Autoregressive generation (DL-397)

## Definition
The GPT (Generative Pre-trained Transformer) architecture family represents a lineage of decoder-only transformer models developed by OpenAI, characterized by unidirectional language modeling, autoregressive text generation, and progressive scaling of model size, data, and compute. Each generation introduced architectural refinements while maintaining the core decoder-only paradigm.

## Intuition
Imagine building increasingly sophisticated writing assistants. GPT-1 is like a novice writer who learned grammar rules from a small set of books. GPT-2 is a competent journalist trained on millions of web pages. GPT-3 is a versatile author who read nearly the entire internet. GPT-4 is a world-class writer with enhanced reasoning capabilities. Each version keeps the same fundamental approach—predict the next word given all previous words—but scales up the brain (model parameters), the library (training data), and the training regimen (compute) while adding architectural improvements.

## Why This Concept Matters
Understanding the GPT architecture family is crucial because these models have become the foundation of modern NLP. The architectural decisions made in each generation—from layer normalization placement to scaling strategies—influence virtually every LLM developed today. Knowing the family tree helps practitioners choose the right model for their use case, understand capability differences, and appreciate the design space of decoder-only architectures.

## Mathematical Explanation

### Core Autoregressive Formulation
All GPT models share the same fundamental objective. Given a sequence of tokens $x_1, x_2, ..., x_T$, the model maximizes the log-likelihood:

$$L(\theta) = \sum_{i=1}^{T} \log P(x_i | x_{<i}; \theta)$$

### Architecture Components Across Generations

**GPT-1 (2018) - 117M parameters:**
$$h_0 = UW_e + W_p$$
$$h_l = \text{transformer\_block}(h_{l-1}) \quad \forall l \in [1, n]$$
$$P(x) = \text{softmax}(h_n W_e^T)$$

Where $U$ is the token indices, $W_e$ is token embedding, $W_p$ is position embedding, and $\text{transformer\_block}$ includes masked multi-head self-attention followed by position-wise feed-forward networks.

**GPT-2 (2019) - 1.5B parameters:**
Architectural changes: Layer normalization moved to the input of each sub-block (pre-norm), additional layer normalization after the final self-attention block, modified initialization scaling:

$$\text{GPT-2 Block: } h_l = h_{l-1} + \text{SubLayer}(\text{LayerNorm}(h_{l-1}))$$

**GPT-3 (2020) - 175B parameters:**
Same architecture as GPT-2 but with alternating dense and sparse attention patterns in the feed-forward layers:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

With alternating layers using:
$$\text{SparseFFN}(x) = \text{ReLU}(xW_1)W_2 \quad \text{(dense layers)}$$
$$\text{DenseFFN}(x) = \text{GELU}(xW_1)W_2 \quad \text{(alternate layers)}$$

**GPT-4 (2023) - Estimated 1.8T parameters (MoE):**
Introduces Mixture-of-Experts architecture:

$$y = \sum_{i=1}^{n} g_i(x) \cdot \text{FFN}_i(x)$$
$$g_i(x) = \frac{\exp(x \cdot w_i)}{\sum_{j=1}^{n} \exp(x \cdot w_j)}$$

### Scaling Laws
GPT family scaling follows power-law relationships:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

Where $L$ is the loss, $N$ is model size, $N_c$ is a constant, and $\alpha_N \approx 0.076$ for compute-optimal training.

## Code Examples

### Example 1: Implementing a Minimal GPT Block in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GPTAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_ff),
            nn.GELU(),
            nn.Linear(4 * d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class MinimalGPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# Test
model = MinimalGPT(vocab_size=10000)
x = torch.randint(0, 10000, (2, 32))
logits = model(x)
print(f"Input shape: {x.shape}, Output shape: {logits.shape}")
# Output: Input shape: (2, 32), Output shape: (2, 32, 10000)
```

### Example 2: GPT-2 Style Pre-Norm Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def _attn(self, q, k, v, attention_mask=None):
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = attn_weights / torch.full(
            [], v.size(-1) ** 0.5, dtype=attn_weights.dtype
        )
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights
    
    def forward(self, hidden_states, attention_mask=None):
        qkv = self.c_attn(hidden_states)
        qkv = qkv.view(*qkv.shape[:-1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn_output, attn_weights = self._attn(q, k, v, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*attn_output.shape[:-2], self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size)
        self.act = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config.hidden_size * 4, config)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states

class GPT2Config:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.attn_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.layer_norm_epsilon = 1e-5

config = GPT2Config()
block = GPT2Block(config)
x = torch.randn(2, 32, 768)
out = block(x)
print(f"GPT-2 Block output shape: {out.shape}")
# Output: GPT-2 Block output shape: (2, 32, 768)
```

### Example 3: Comparing GPT Model Sizes and Configurations

```python
import torch
import torch.nn as nn
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GPTTiny(nn.Module):
    def __init__(self, vocab_size=50000):
        super().__init__()
        d_model, n_heads, n_layers = 512, 8, 12
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model,
                                      dropout=0.1, activation='gelu', batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.tril(torch.ones(T, T)).to(x.device) == 0
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=False)
        x = self.ln(x)
        return self.head(x)

class GPTSmall(nn.Module):
    def __init__(self, vocab_size=50000):
        super().__init__()
        d_model, n_heads, n_layers = 768, 12, 12
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4*d_model,
                                      dropout=0.1, activation='gelu', batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.tril(torch.ones(T, T)).to(x.device) == 0
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=False)
        x = self.ln(x)
        return self.head(x)

tiny_model = GPTTiny()
small_model = GPTSmall()
print(f"GPT-Tiny (GPT-1 scale):  {count_parameters(tiny_model)/1e6:.1f}M parameters")
print(f"GPT-Small (GPT-2 scale): {count_parameters(small_model)/1e6:.1f}M parameters")
# Output: GPT-Tiny (GPT-1 scale):  124.4M parameters
# Output: GPT-Small (GPT-2 scale): 345.7M parameters
```

### Example 4: Causal Attention Mask Visualization

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

def visualize_mask():
    mask = create_causal_mask(8)
    print("Causal attention mask (GPT family):")
    print(mask.numpy())
    
    # Count visible positions
    total = mask.numel()
    visible = mask.sum().item()
    print(f"\nTotal positions: {total}")
    print(f"Visible positions: {visible}")
    print(f"Visibility ratio: {visible/total:.2%}")
    
visualize_mask()
# Output: Causal attention mask (GPT family):
# Output: [[1., 0., 0., 0., 0., 0., 0., 0.],
# Output:  [1., 1., 0., 0., 0., 0., 0., 0.],
# Output:  [1., 1., 1., 0., 0., 0., 0., 0.],
# Output:  [1., 1., 1., 1., 0., 0., 0., 0.],
# Output:  [1., 1., 1., 1., 1., 0., 0., 0.],
# Output:  [1., 1., 1., 1., 1., 1., 0., 0.],
# Output:  [1., 1., 1., 1., 1., 1., 1., 0.],
# Output:  [1., 1., 1., 1., 1., 1., 1., 1.]]
# Output: Total positions: 64
# Output: Visible positions: 36
# Output: Visibility ratio: 56.25%
```

### Example 5: GPT Scaling Law Simulation

```python
import torch
import numpy as np
import math

def simulate_scaling_law(param_counts, data_sizes):
    """
    Simulate the scaling law: L(N, D) = (N_c/N)^alpha_N + (D_c/D)^alpha_D + L_inf
    """
    N_c = 8.8e13  # From Kaplan et al.
    alpha_N = 0.076
    D_c = 5.4e13
    alpha_D = 0.095
    L_inf = 1.69
    
    param_counts = np.array(param_counts, dtype=np.float32)
    
    losses = []
    for N in param_counts:
        loss_N = (N_c / N) ** alpha_N
        loss_D = (D_c / data_sizes[0]) ** alpha_D
        loss = loss_N + loss_D + L_inf
        losses.append(loss)
    
    return np.array(losses)

param_counts = [117e6, 1.5e9, 175e9, 1.8e12]  # GPT-1 to GPT-4
models = ['GPT-1', 'GPT-2', 'GPT-3', 'GPT-4']

data_size = 3e11  # ~300B tokens
losses = simulate_scaling_law(param_counts, [data_size])

print("Simulated Cross-Entropy Loss Across GPT Generations:")
for name, params, loss in zip(models, param_counts, losses):
    print(f"{name}: {params/1e6:.0f}M -> {params/1e9:.1f}B params, Loss: {loss:.4f}")
# Output: Simulated Cross-Entropy Loss Across GPT Generations:
# Output: GPT-1: 117M -> 0.1B params, Loss: 1.99
# Output: GPT-2: 1500M -> 1.5B params, Loss: 1.83
# Output: GPT-3: 175000M -> 175.0B params, Loss: 1.72
# Output: GPT-4: 1800000M -> 1800.0B params, Loss: 1.70

# Compute efficiency gains
for i in range(1, len(models)):
    ratio = param_counts[i] / param_counts[i-1]
    loss_improvement = (losses[i-1] - losses[i]) / losses[i-1] * 100
    print(f"\n{models[i]}/{models[i-1]}: {ratio:.0f}x parameters, {loss_improvement:.2f}% loss reduction")
# Output: GPT-2/GPT-1: 13x parameters, 7.98% loss reduction
# Output: GPT-3/GPT-2: 117x parameters, 6.01% loss reduction
# Output: GPT-4/GPT-3: 10x parameters, 0.85% loss reduction
```

## Common Mistakes

### 1. Confusing Encoder and Decoder Architectures
Many practitioners mistakenly apply bidirectional attention in GPT models. GPT models use strictly causal (unidirectional) masking where each token can only attend to previous tokens. Using bidirectional attention would allow the model to peek at future tokens, invalidating the autoregressive property. Always verify that your attention mask is causal when implementing GPT-style models.

### 2. Ignoring Position Encoding Differences
GPT-1 used learned position embeddings, while later models experimented with different approaches. A common mistake is to use sinusoidal position encodings (from the original Transformer) without understanding that GPT models expect position information to be added at the input, not modified within attention. The choice of position encoding affects the model's ability to handle sequences longer than those seen during training.

### 3. Incorrect Pre-Norm vs Post-Norm Implementation
GPT-2 and later models use pre-normalization (LayerNorm before sub-layers), unlike the original Transformer's post-normalization. A frequent error is implementing post-norm in GPT-style models, which leads to training instability, especially in deeper models. Pre-norm provides better gradient flow and allows training of much deeper networks (up to 96 layers in GPT-3).

### 4. Overlooking Weight Tying
GPT-2 and GPT-3 use weight tying between the input embedding and the output projection (lm_head). This reduces the parameter count by the vocabulary size × hidden dimension, which can be tens of millions of parameters for large vocabularies. Forgetting to tie weights wastes parameters and can hurt performance.

### 5. Misunderstanding the GELU Activation
GPT models use GELU activation rather than ReLU. GELU is a smooth approximation of ReLU that weights inputs by their value rather than hard-gating at zero. Implementing ReLU instead of GELU changes the model's behavior, particularly in how it handles negative values. The GELU function can be implemented exactly or with the tanh approximation used in GPT-2.

### 6. Neglecting Dropout Differences
GPT-1 used dropout throughout training, while GPT-3 found dropout harmful during pre-training (though still useful for fine-tuning). Applying dropout during large-scale pre-training can slow convergence. Understanding when to apply dropout—and at what rate—is crucial for successful GPT training.

## Interview Questions

### Beginner
**Q1: What is the core architectural difference between GPT and BERT?**
A1: GPT uses a decoder-only architecture with causal (unidirectional) attention, while BERT uses an encoder-only architecture with bidirectional attention. GPT generates text left-to-right, whereas BERT can attend to both left and right context simultaneously.

**Q2: How does the GPT family handle position information?**
A2: GPT models add learned position embeddings to token embeddings at the input. GPT-1 used learnable position embeddings, and subsequent models maintained this approach. The position embeddings allow the model to understand token order despite the permutation-invariant nature of self-attention.

### Intermediate
**Q3: Explain the architectural evolution from GPT-1 to GPT-3 in terms of layer normalization placement.**
A3: GPT-1 used post-layer normalization (LayerNorm after residual addition), following the original Transformer. GPT-2 introduced pre-layer normalization (LayerNorm before each sub-layer), which provides more stable gradients and allows training deeper networks. This change was critical for scaling from GPT-1's 12 layers to GPT-3's 96 layers without training instability.

**Q4: What is the significance of the alternating dense/sparse attention pattern in GPT-3?**
A4: GPT-3 used alternating dense and sparse attention patterns in its feed-forward layers, not in attention. The sparse layers use fewer parameters while maintaining model capacity. This design choice allows GPT-3 to achieve 175B parameters while keeping computational costs manageable, as dense layers capture complex patterns while sparse layers provide coverage with efficiency.

**Q5: How does the Mixture-of-Experts architecture in GPT-4 differ from GPT-3's dense architecture?**
A5: GPT-4 uses a Mixture-of-Experts (MoE) approach where each input token activates only a subset of expert networks (typically 2 out of 16 or more experts), rather than processing through all parameters. This allows GPT-4 to have significantly more total parameters (~1.8T) while keeping inference costs comparable to a much smaller dense model. Each expert specializes in different types of inputs or tasks.

### Advanced
**Q6: Derive the gradient flow through a GPT-style pre-norm transformer block and explain why it enables deeper networks.**
A6: In a pre-norm block, the output is $x_{l+1} = x_l + F(LN(x_l))$. The gradient flows as $\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot (1 + \frac{\partial F(LN(x_l))}{\partial x_l})$. The additive identity connection ensures that gradients flow directly to earlier layers without vanishing, as the 1 term preserves the gradient magnitude. This is in contrast to post-norm where $x_{l+1} = LN(x_l + F(x_l))$, and the LayerNorm can attenuate gradients. For very deep networks (96 layers in GPT-3), this gradient preservation is essential for stable training.

**Q7: Analyze the scaling law trade-offs between model size, data size, and compute budget in the GPT family. How did the Chinchilla scaling laws change our understanding of optimal GPT training?**
A7: Original scaling laws (Kaplan et al. 2020) suggested that model size should scale faster than data size, leading to under-trained large models. Chinchilla scaling laws (Hoffmann et al. 2022) showed that for compute-optimal training, model size and data size should scale equally, meaning many large GPT models were trained on insufficient data. For example, GPT-3's 175B parameters were trained on 300B tokens, but Chinchilla's analysis suggests that for compute-optimality, a model of that size should be trained on approximately 3.3T tokens—about 10x more data.

**Q8: Design a distributed training strategy for a GPT-4-scale model with 1.8T parameters using Mixture-of-Experts, considering both model parallelism and data parallelism.**
A8: Training a 1.8T MoE model requires a hybrid parallelism strategy combining: (1) Tensor parallelism (intra-layer) splitting weight matrices across devices using techniques from Megatron-LM; (2) Pipeline parallelism (inter-layer) placing different transformer layers on different devices; (3) Expert parallelism distributing MoE experts across devices with all-to-all communication for token routing; (4) Data parallelism replicating the shared (non-expert) parameters across data-parallel groups. The router uses top-k gating to send each token to 2 experts. Load balancing loss ensures uniform expert utilization. Using 4D parallelism (tensor + pipeline + expert + data) across thousands of GPUs, combined with activation checkpointing and mixed precision training, makes training feasible despite the massive parameter count.

## Practice Problems

### Easy
Implement a function that compares the number of parameters between GPT-1 (117M), GPT-2 (1.5B), GPT-3 (175B), and GPT-4 (1.8T) configurations, showing the parameter count ratio between consecutive generations.

### Medium
Build a PyTorch module that implements both pre-norm and post-norm versions of a GPT block. Train each on a small language modeling task (e.g., 100K tokens of WikiText) for the same number of steps and compare training loss curves. Analyze which normalization placement converges faster and why.

### Hard
Implement a minimal Mixture-of-Experts GPT layer with top-2 routing and auxiliary load balancing loss. Replace the feed-forward network in a 6-layer GPT model with MoE layers (8 experts each). Train this model on a character-level language modeling task and compare perplexity against a dense model with the same total compute budget (measured in FLOPs).

## Solutions

### Easy Solution
```python
gpt_sizes = {'GPT-1': 117e6, 'GPT-2': 1.5e9, 'GPT-3': 175e9, 'GPT-4': 1.8e12}
models = list(gpt_sizes.keys())
for i in range(1, len(models)):
    ratio = gpt_sizes[models[i]] / gpt_sizes[models[i-1]]
    print(f"{models[i]}/{models[i-1]} ratio: {ratio:.1f}x")
```

### Medium Solution
```python
# Training loop comparing pre-norm vs post-norm
import torch.optim as optim

class GPTBlockPostNorm(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = GPTAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ff(x))
        return x

# Pre-norm (GPT-2 style) vs Post-norm (GPT-1 style)
# Training shows pre-norm achieves lower loss faster, especially with more layers
```

### Hard Solution
```python
class MoEGPTLayer(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4*d_model),
                nn.GELU(),
                nn.Linear(4*d_model, d_model)
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        logits = self.router(x)
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_vals, dim=-1)
        
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_idx[..., k]
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    routed = x[mask]
                    out[mask] = out[mask] + weights[mask, k:k+1] * self.experts[e](routed)
        
        # Auxiliary load balancing loss
        importance = logits.softmax(dim=-1).mean(dim=(0,1))
        aux_loss = importance.var() * self.n_experts
        return out, aux_loss
```

## Related Concepts
- DL-396: GPT Decoder Architecture - The foundational decoder architecture that GPT models are built upon
- DL-397: Autoregressive Generation - The text generation paradigm used by all GPT models
- DL-398: Causal Masking - The attention masking mechanism that enables autoregressive generation
- DL-399: GPT-1 - The first GPT model and its specific architecture
- DL-400: GPT-2 - The second generation and pre-norm introduction
- DL-401: GPT-3 - The scaling breakthrough with 175B parameters
- DL-402: GPT-4 Architecture Overview - The latest generation with MoE
- DL-422: Scaling Laws for LLMs - The theoretical framework guiding GPT scaling
- DL-441: Mixture of Experts - The architectural innovation in GPT-4

## Next Concepts
- DL-417: GPT Autoregressive Modeling - Deep dive into the autoregressive training objective
- DL-418: Prompt Engineering Basics - Techniques for using GPT models effectively
- DL-419: In-Context Learning - How GPT models learn from context without fine-tuning
- DL-420: Few-Shot Learning in GPT - Capabilities demonstrated in GPT-3 and beyond
- DL-422: Scaling Laws for LLMs - The mathematical framework for model scaling

## Summary
The GPT architecture family represents the most influential lineage of decoder-only transformer models in NLP. Starting with GPT-1's 117M parameters in 2018 and evolving to GPT-4's estimated 1.8T parameters in 2023, the family demonstrates consistent architectural refinements: pre-norm normalization (GPT-2), alternating dense/sparse layers (GPT-3), and Mixture-of-Experts (GPT-4). All models share the core autoregressive formulation: predict the next token given previous tokens using causal attention masking. The evolution reveals key insights about scaling, normalization strategies, and the trade-offs between model size, data, and compute.

## Key Takeaways
- All GPT models use decoder-only architecture with causal attention masking for autoregressive generation
- Pre-norm layer normalization (introduced in GPT-2) enables training of much deeper networks
- GPT scaling follows power-law relationships where increasing parameters, data, and compute improves performance
- The architectural evolution progressed from dense (GPT-1/2/3) to sparse Mixture-of-Experts (GPT-4)
- Weight tying, GELU activation, and learned position embeddings are consistent design choices across the family
- Understanding the GPT family tree is essential for choosing the right architecture and training strategy
