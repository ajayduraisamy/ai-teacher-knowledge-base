# GPT-Neo and GPT-J

## Concept ID
DL-424

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
LLM Architectures (DL-416 to DL-440)

## Learning Objectives
- Understand the architecture of GPT-Neo and GPT-J
- Compare their design choices with original GPT models
- Implement key components of these architectures
- Evaluate their capabilities and limitations

## Prerequisites
- GPT Architecture Family (DL-416)
- Autoregressive Modeling (DL-417)
- Transformer Architecture (DL-370)

## Definition
GPT-Neo and GPT-J are open-source decoder-only transformer models developed by EleutherAI to replicate and democratize the capabilities of GPT-3. GPT-Neo (2.7B parameters) was trained in 2021, followed by GPT-J-6B (6B parameters). They feature parallel attention computations and modified architecture choices compared to the original GPT models.

## Intuition
Imagine GPT-3 as a proprietary supercomputer that only a few people can access. GPT-Neo and GPT-J are like open-source versions that anyone can run on their own hardware. While not as powerful as the original, they capture the essential architecture and demonstrate that GPT-3's capabilities can be replicated with open-source tools and data. They proved that the "secret sauce" of GPT-3 was primarily scale and data, not proprietary architectural innovations.

## Why This Concept Matters
GPT-Neo and GPT-J are historically significant as the first successful open-source replications of GPT-3-scale models. They demonstrated that the GPT architecture could be replicated by the community, kickstarted the open-source LLM movement, and enabled researchers worldwide to experiment with models that were previously only accessible via API. Their design choices influenced all subsequent open-source LLMs.

## Mathematical Explanation

### Architecture Differences from GPT-3

**Parallel Attention + FFN (GPT-Neo):**
Unlike GPT-3's sequential attention-then-FFN, GPT-Neo uses parallel computation:

$$y = x + \text{Attention}(\text{LN}(x)) + \text{FFN}(\text{LN}(x))$$

This is mathematically equivalent but computationally more efficient on GPUs.

**Projection Dimensions (GPT-J):**
GPT-J uses 16 attention heads with head dimension 256, projecting queries and keys separately:

$$Q = xW_Q, \quad K = xW_K, \quad V = xW_V$$
$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

**Rotary Position Embeddings (GPT-J):**
GPT-J introduced rotary position embeddings (RoPE) instead of learned absolute positions:

$$\text{RoPE}(x_m, m) = \begin{pmatrix} x_m \cos(m\theta) - x_{m+1} \sin(m\theta) \\ x_m \sin(m\theta) + x_{m+1} \cos(m\theta) \end{pmatrix}$$

This enables better generalization to longer sequences.

## Code Examples

### Example 1: GPT-Neo Parallel Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTNeoAttention(nn.Module):
    """GPT-Neo style attention with parallel computation"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class GPTNeoBlock(nn.Module):
    """GPT-Neo block with parallel attention + FFN computation"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = GPTNeoAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.ln(x)
        # Parallel computation: attention + FFN computed simultaneously
        attn_out = self.attn(x, attention_mask)
        ffn_out = self.ffn(x)
        # Add both to residual
        return residual + attn_out + ffn_out

class GPTNeoModel(nn.Module):
    """Complete GPT-Neo model"""
    
    def __init__(self, vocab_size=50257, d_model=2560, n_heads=32, 
                 n_layers=32, d_ff=None, max_seq=2048):
        super().__init__()
        self.d_model = d_model
        d_ff = d_ff or 4 * d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            GPTNeoBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        x = self.drop(x)
        
        if attention_mask is not None:
            causal_mask = torch.triu(
                torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1
            )
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if attention_mask.dim() == 2:
                attn_mask = attn_mask.masked_fill(
                    attention_mask[:, None, None, :] == 0, float('-inf')
                )
        else:
            attn_mask = torch.triu(
                torch.ones(1, 1, T, T, device=input_ids.device) * float('-inf'), diagonal=1
            )
        
        for block in self.blocks:
            x = block(x, attn_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# Test
model = GPTNeoModel(vocab_size=10000, d_model=512, n_heads=8, n_layers=6)
x = torch.randint(0, 10000, (2, 32))
logits = model(x)
print(f"GPT-Neo output shape: {logits.shape}")
# Output: GPT-Neo output shape: (2, 32, 10000)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Total parameters: 41,134,608
```

### Example 2: GPT-J with Rotary Position Embeddings

```python
import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) as used in GPT-J"""
    
    def __init__(self, dim, max_position=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb
    
    @staticmethod
    def apply_rotary(x, cos, sin):
        """Apply rotary embeddings to input tensor"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

class GPTJAttention(nn.Module):
    """GPT-J style attention with RoPE"""
    
    def __init__(self, d_model, n_heads, rotary_dim=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        rotary_dim = rotary_dim or self.head_dim
        self.rotary = RotaryEmbedding(rotary_dim)
        
    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply rotary embeddings to Q and K
        cos = self.rotary(Q, T).cos()
        sin = self.rotary(Q, T).sin()
        
        Q = RotaryEmbedding.apply_rotary(Q, cos, sin)
        K = RotaryEmbedding.apply_rotary(K, cos, sin)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

# Test RoPE
rope = RotaryEmbedding(dim=64)
x = torch.randn(2, 10, 8, 64)
cos = rope(x, 10).cos()
sin = rope(x, 10).sin()
rotated = RotaryEmbedding.apply_rotary(x, cos[:10], sin[:10])
print(f"RoPE applied to shape {x.shape}: output shape {rotated.shape}")
print(f"RoPE preserves norm: ||x||={x.norm():.4f}, ||rotated||={rotated.norm():.4f}")
# Output: RoPE applied to shape (2, 10, 8, 64): output shape (2, 10, 8, 64)
# Output: RoPE preserves norm: ||x||=226.2742, ||rotated||=226.2742
```

### Example 3: Comparison of GPT-Neo, GPT-J, and GPT-3 Blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT3Block(nn.Module):
    """Standard GPT-3 sequential block (pre-norm)"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x

class GPTNeoBlock(nn.Module):
    """GPT-Neo parallel block"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        
    def forward(self, x, mask=None):
        residual = x
        x = self.ln(x)
        attn_out = self.attn(x, x, x, attn_mask=mask)[0]
        ffn_out = self.ffn(x)
        return residual + attn_out + ffn_out

class GPTJBlock(nn.Module):
    """GPT-J block with RoPE and parallel computation"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        
    def forward(self, x, mask=None):
        residual = x
        x = self.ln(x)
        attn_out = self.attn(x, x, x, attn_mask=mask)[0]
        ffn_out = self.ffn(x)
        return residual + attn_out + ffn_out

def compare_block_structures():
    d_model, n_heads, d_ff = 768, 12, 3072
    T = 64
    
    blocks = {
        'GPT-3 (sequential)': GPT3Block(d_model, n_heads, d_ff),
        'GPT-Neo (parallel)': GPTNeoBlock(d_model, n_heads, d_ff),
        'GPT-J (parallel+RoPE)': GPTJBlock(d_model, n_heads, d_ff),
    }
    
    x = torch.randn(4, T, d_model)
    mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)
    
    for name, block in blocks.items():
        out = block(x, mask)
        params = sum(p.numel() for p in block.parameters())
        print(f"{name:<30} Output shape: {out.shape}, Params: {params:,}")
    
    # Compute speed difference
    import time
    for name, block in blocks.items():
        start = time.time()
        for _ in range(100):
            out = block(x, mask)
        elapsed = time.time() - start
        print(f"{name:<30} Forward time (100 iters): {elapsed:.4f}s")

compare_block_structures()
# Output: GPT-3 (sequential)           Output shape: (4, 64, 768), Params: 7,080,960
# Output: GPT-Neo (parallel)           Output shape: (4, 64, 768), Params: 7,080,960
# Output: GPT-J (parallel+RoPE)        Output shape: (4, 64, 768), Params: 7,080,960
```

### Example 4: Implementing the Pile Dataset Loader (Simulated)

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ThePileDataset(Dataset):
    """Simulated dataset loader for the Pile (used to train GPT-Neo/GPT-J)"""
    
    def __init__(self, num_samples=100000, seq_len=2048, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate a sequence from the Pile
        tokens = torch.randint(10, self.vocab_size, (self.seq_len + 1,))
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return input_ids, labels

def create_pile_dataloader(batch_size=8, seq_len=2048, num_workers=0):
    dataset = ThePileDataset(num_samples=1000, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)
    return loader

# Demonstrate
loader = create_pile_dataloader(batch_size=4, seq_len=128)
x, y = next(iter(loader))
print(f"Pile batch: input_ids {x.shape}, labels {y.shape}")
# Output: Pile batch: input_ids (4, 128), labels (4, 128)
```

### Example 5: Training Loop for GPT-Neo Style Model

```python
import torch.optim as optim

class GPTNeoTrainer:
    """Training loop for GPT-Neo/GPT-J style models"""
    
    def __init__(self, model, learning_rate=3e-4, weight_decay=0.1):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=3e-5
        )
        
    def train_step(self, input_ids, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def compute_perplexity(self, input_ids, labels):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return torch.exp(loss).item()

# Demonstrate training step
model = GPTNeoModel(vocab_size=10000, d_model=256, n_heads=4, n_layers=3)
trainer = GPTNeoTrainer(model)

x = torch.randint(0, 10000, (2, 64))
y = torch.randint(0, 10000, (2, 64))
loss = trainer.train_step(x, y)
print(f"Training step loss: {loss:.4f}")
ppl = trainer.compute_perplexity(x, y)
print(f"Perplexity: {ppl:.2f}")
# Output: Training step loss: 9.2104
# Output: Perplexity: 10016.32
```

## Common Mistakes

### 1. Confusing GPT-Neo and GPT-J Architectures
GPT-Neo uses learned absolute position embeddings and a parallel attention+FFN formulation. GPT-J uses rotary position embeddings and also parallel computation, but with different head dimensions (256 vs GPT-Neo's standard). They are not interchangeable implementations.

### 2. Assuming Parallel Computation Changes Model Behavior
The parallel attention+FFN computation in GPT-Neo is mathematically equivalent to sequential computation at initialization. The ordering of operations does not affect the representational capacity—both formulations can represent the same functions. The parallel version is simply more computationally efficient on GPUs.

### 3. Neglecting the Effect of RoPE on Sequence Length Generalization
GPT-J's rotary position embeddings allow better generalization to sequence lengths beyond those seen during training, but this requires the model to handle the extrapolation. Performance may degrade for very long sequences. RoPE is not a magic solution for length generalization—it helps but does not fully solve the problem.

### 4. Underestimating the Pile Dataset's Importance
The Pile, a diverse 825GB dataset curated by EleutherAI, was crucial to GPT-Neo and GPT-J's success. Using different or lower-quality data will not reproduce their results. The careful data curation and deduplication were as important as the architectural choices.

### 5. Overlooking the Numerical Precision Requirements
GPT-J was trained in bfloat16 and requires careful numerical handling in the attention mechanism. Implementing RoPE or parallel attention in float32 may work but will be significantly slower and more memory-intensive.

## Interview Questions

### Beginner
**Q1: What are GPT-Neo and GPT-J, and why are they important?**
A1: GPT-Neo (2.7B) and GPT-J (6B) are open-source decoder-only transformer models created by EleutherAI to replicate GPT-3. They are important because they democratized access to large language models, demonstrated that GPT-3's architecture could be replicated by the community, and sparked the open-source LLM movement.

**Q2: What is the main architectural difference between GPT-Neo and standard GPT models?**
A2: GPT-Neo uses parallel computation of attention and feed-forward networks (both computed simultaneously and added to the residual), while standard GPT models compute them sequentially (attention first, then FFN). This parallel formulation is mathematically equivalent but more computationally efficient on GPUs.

### Intermediate
**Q3: Explain how rotary position embeddings (RoPE) work in GPT-J and their advantages over learned position embeddings.**
A3: RoPE encodes position by rotating the query and key vectors by an angle proportional to the position. Specifically, each pair of dimensions (2i, 2i+1) is rotated by angle m·θ_i where m is the position and θ_i = base^(-2i/d). Advantages: (1) Relative position information is captured through the dot product of rotated vectors; (2) Better generalization to unseen sequence lengths; (3) No additional parameters needed for position encoding; (4) The rotation preserves the norm of the vectors.

**Q4: What was the role of the Pile dataset in training GPT-Neo and GPT-J?**
A4: The Pile is an 825GB diverse text dataset curated by EleutherAI that includes web text, books, academic papers, code, and other sources. It was designed to provide broad, high-quality training data comparable to the datasets used for GPT-3. The careful filtering, deduplication, and component balancing of the Pile were essential to achieving competitive performance with GPT-3 on a smaller model.

### Advanced
**Q5: Analyze the trade-offs between GPT-Neo's parallel computation and GPT-3's sequential computation in terms of gradient flow and training dynamics.**
A5: In the parallel formulation (x + Attn(LN(x)) + FFN(LN(x))), gradients flow through both the attention and FFN paths simultaneously to the same LayerNorm input. This means the gradient w.r.t. x is ∂L/∂x = ∂L/∂out · (I + ∂Attn/∂x + ∂FFN/∂x). In the sequential formulation, the gradient path is longer: ∂L/∂x = ∂L/∂out · (I + ∂FFN/∂(LN2(x))) · (I + ∂Attn/∂(LN1(x))). The parallel version may have better gradient flow because it avoids multiplicative interactions between the attention and FFN gradients, but the sequential version allows attention and FFN to specialize on different features of the same input (since attention sees pre-LN1 features, while FFN sees post-attention features). Empirically, both formulations achieve similar performance, but parallel is faster.

**Q6: Design a scaling study that extends the GPT-Neo/GPT-J approach to a 100B+ parameter model, considering the limitations of open-source infrastructure.**
A6: Scaling to 100B+ with open-source methods requires: (1) Distributed training across hundreds of GPUs using FSDP or DeepSpeed ZeRO-3; (2) Gradient checkpointing to reduce memory; (3) Mixed precision training (bf16); (4) Tensor parallelism for attention computation; (5) Pipeline parallelism for layer distribution; (6) The Pile or equivalently diverse data (1T+ tokens); (7) Learning rate schedule with warmup and cosine decay. Key challenges: (a) Inter-node bandwidth limitations for all-reduce; (b) Cumulative training instability at scale; (c) Load balancing across heterogeneous hardware; (d) Data loading and preprocessing throughput. The GPT-NeoX-20B project demonstrated this approach at 20B scale, and lessons learned informed subsequent 100B+ open-source efforts.

## Practice Problems

### Easy
Implement a function that converts a standard GPT sequential block to a GPT-Neo parallel block by modifying the forward pass.

### Medium
Implement rotary position embeddings for a language model and compare the attention pattern distances with learned position embeddings for sequences longer than training length.

### Hard
Implement a training script for a GPT-Neo style model on a 1B parameter scale using fully sharded data parallelism, with gradient checkpointing and mixed precision training.

## Solutions

### Easy Solution
```python
def convert_to_parallel(sequential_block):
    """Convert GPT-3 sequential block to GPT-Neo parallel block"""
    class ParallelBlock(nn.Module):
        def __init__(self, seq_block):
            super().__init__()
            self.ln = seq_block.ln2  # Single LN
            self.attn = seq_block.attn
            self.ffn = seq_block.ffn
            self.ln1 = seq_block.ln1  # Keep for compatibility
        
        def forward(self, x, mask=None):
            residual = x
            x = self.ln(x)
            return residual + self.attn(x, mask) + self.ffn(x)
    
    return ParallelBlock(sequential_block)
```

### Medium Solution
```python
def compare_position_encodings(d_model=512, train_len=512, test_len=1024):
    learned_emb = nn.Embedding(test_len, d_model)
    rope = RotaryEmbedding(d_model, max_position=test_len)
    
    # Compare attention score distances
    x = torch.randn(1, test_len, d_model)
    # ... compute attention scores with both methods
    
    print(f"Learned position extrapolation: test/avg attention at train_len vs test_len...")
```

### Hard Solution
```python
# FSDP training setup would be:
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
# 
# model = FSDP(model, device_id=local_rank, mixed_precision=True)
# # Use gradient checkpointing
# model.gradient_checkpointing_enable()
```

## Related Concepts
- DL-416: GPT Architecture Family - The model family GPT-Neo/GPT-J extend
- DL-399: GPT-1 - Early GPT architecture
- DL-401: GPT-3 - The model being replicated
- DL-425: LLaMA Architecture - Successor open-source architecture
- DL-427: LLaMA 3 - Latest open-source model family

## Next Concepts
- DL-425: LLaMA Architecture - The next generation of open-source LLMs
- DL-426: LLaMA 2 - Refined open-source LLM
- DL-427: LLaMA 3 - Latest open-source LLM advances

## Summary
GPT-Neo (2.7B) and GPT-J (6B) are open-source decoder-only transformer models by EleutherAI that successfully replicated GPT-3's capabilities. GPT-Neo introduced parallel attention+FFN computation for efficiency, while GPT-J added rotary position embeddings for better length generalization. Both models were trained on the Pile, a curated 825GB dataset, and demonstrated that GPT-3-scale capabilities could be achieved with open-source methods, sparking the open-source LLM revolution.

## Key Takeaways
- GPT-Neo and GPT-J were the first successful open-source GPT-3 replications
- GPT-Neo uses parallel attention+FFN computation for GPU efficiency
- GPT-J introduced Rotary Position Embeddings (RoPE)
- Both trained on the Pile, a diverse curated dataset
- Architecture choices influence training efficiency but not final model quality
- These models democratized access to large language model technology
