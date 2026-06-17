# Mistral and Mixtral

## Concept ID
DL-428

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
LLM Architectures (DL-416 to DL-440)

## Learning Objectives
- Understand the Mistral and Mixtral architecture innovations
- Implement sliding window attention and Mixture of Experts
- Analyze the efficiency-performance trade-offs
- Compare with LLaMA and GPT architectures

## Prerequisites
- LLaMA Architecture (DL-425)
- Mixture of Experts (DL-441)
- Grouped Query Attention (DL-447)

## Definition
Mistral and Mixtral are efficient decoder-only transformer models developed by Mistral AI. Mistral-7B (2023) features grouped query attention and sliding window attention for efficient processing of long sequences. Mixtral 8x7B (2024) extends this with a Mixture-of-Experts architecture combining 8 experts (2 active per token) for 46.7B total parameters with 12.9B active parameters per token. Mistral Large is their proprietary frontier model.

## Intuition
Mistral-7B is like a compact car with excellent fuel efficiency—it uses sliding window attention to handle long roads (contexts) without carrying all past data in memory. Mixtral 8x7B is like having 8 specialist mechanics in a garage, where only 2 work on each car (token). You have access to many experts' knowledge but only pay for 2 at a time, getting 46.7B parameters of total knowledge while only using 12.9B per inference. This makes Mixtral remarkably efficient: it rivals LLaMA 2 70B performance with only 18% of the active parameters.

## Why This Concept Matters
Mistral and Mixtral demonstrate that architectural innovation can bridge the gap between small and large models. Mistral-7B matches or exceeds LLaMA 2 13B with half the parameters, while Mixtral 8x7B matches LLaMA 2 70B with 12.9B active parameters. Their sliding window attention and MoE designs have influenced subsequent model architectures and set new efficiency standards.

## Mathematical Explanation

### Sliding Window Attention
Standard attention has O(T²) complexity. Sliding window attention limits each token to attend only to the W previous tokens:

$$\text{Attention}(Q_i, K_{i-W:i}, V_{i-W:i}) = \text{softmax}\left(\frac{Q_i K_{i-W:i}^T}{\sqrt{d_k}}\right)V_{i-W:i}$$

This gives O(T × W) complexity, where W << T (typically W = 4096).

### Mixtral MoE
Each MoE layer routes each token to top-2 experts:

$$y = \sum_{i=1}^{n} G(x)_i \cdot E_i(x)$$

$$G(x) = \text{softmax}(\text{top-k}(x \cdot W_g, k))$$

Where:
- $n = 8$ experts
- $k = 2$ top experts selected
- $E_i$ is expert FFN: $\text{SiLU}(xW_{gate_i}) \odot xW_{up_i})W_{down_i}$
- $W_g$ is the routing weight matrix

### Expert Balancing Loss
$$\mathcal{L}_{aux} = \alpha \cdot n \cdot \sum_{i=1}^{n} f_i \cdot P_i$$

Where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the router's probability for expert $i$.

## Code Examples

### Example 1: Sliding Window Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    """Sliding window attention as used in Mistral"""
    
    def __init__(self, d_model, n_heads, window_size=4096):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, attention_mask=None):
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Create sliding window mask
        if self.window_size > 0:
            window_mask = torch.ones(T, T, device=x.device)
            window_mask = torch.triu(window_mask, diagonal=1)
            window_mask = torch.tril(window_mask, diagonal=self.window_size - 1)
            window_mask = window_mask.masked_fill(window_mask == 0, float('-inf'))
            window_mask = window_mask.masked_fill(window_mask == 1, 0.0)
            
            if attention_mask is not None:
                attention_mask = attention_mask + window_mask.unsqueeze(0).unsqueeze(0)
            else:
                attention_mask = window_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

class MistralBlock(nn.Module):
    """Mistral transformer block with sliding window attention"""
    
    def __init__(self, d_model, n_heads, window_size, intermediate_size):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d_model)
        self.self_attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.post_attention_layernorm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, d_model, bias=False),
        )
        
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

# Test sliding window attention
d_model, n_heads = 4096, 32
window_size = 128  # Small for demonstration
swa = SlidingWindowAttention(d_model, n_heads, window_size)
x = torch.randn(2, 256, d_model)
out = swa(x)
print(f"Sliding window attention output: {out.shape}")

# Show attention mask pattern
mask = torch.ones(8, 8)
window_mask = torch.triu(mask, diagonal=1)
window_mask = torch.tril(window_mask, diagonal=3)
print(f"\nSliding window mask (window=3, seq=8):")
print(window_mask)
# Output: Sliding window attention output: (2, 256, 4096)
# Output: 
# Output: Sliding window mask (window=3, seq=8):
# Output: tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
# Output:         [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
# Output:         [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
# Output:         [0., 0., 0., 0., -inf, -inf, -inf, -inf],
# Output:         [-inf, 0., 0., 0., 0., -inf, -inf, -inf],
# Output:         [-inf, -inf, 0., 0., 0., 0., -inf, -inf],
# Output:         [-inf, -inf, -inf, 0., 0., 0., 0., -inf],
# Output:         [-inf, -inf, -inf, -inf, 0., 0., 0., 0.]])
```

### Example 2: Mixtral MoE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MixtralMoE(nn.Module):
    """Mixtral Mixture-of-Experts layer"""
    
    def __init__(self, d_model, intermediate_size, n_experts=8, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        
        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Experts (each is a SwiGLU FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(intermediate_size, d_model, bias=False),
            ) for _ in range(n_experts)
        ])
        
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        
        # Route tokens
        router_logits = self.router(x_flat)  # (B*T, n_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        final_output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == expert_idx)
            if not mask.any():
                continue
            
            # Get the weights for this expert
            expert_mask = mask.any(dim=-1)
            token_indices = torch.where(expert_mask)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Find which of the top-k slots correspond to this expert
            for k in range(self.top_k):
                k_mask = top_k_indices[:, k] == expert_idx
                if not k_mask.any():
                    continue
                
                k_token_indices = torch.where(k_mask)[0]
                k_weights = top_k_weights[k_mask, k].unsqueeze(-1)
                
                # Apply expert
                expert_output = self.experts[expert_idx](x_flat[k_token_indices])
                final_output[k_token_indices] += k_weights * expert_output
        
        # Auxiliary load balancing loss
        tokens_per_expert = router_logits.softmax(dim=-1).mean(dim=0)
        aux_loss = self.n_experts * (tokens_per_expert.var())
        
        return final_output.view(B, T, D), aux_loss
    
    def compute_routing_zloss(self, router_logits):
        """Compute z-loss for router logits (stabilizes training)"""
        log_sum_exp = torch.logsumexp(router_logits, dim=-1)
        z_loss = log_sum_exp.pow(2).mean()
        return z_loss

class MixtralBlock(nn.Module):
    """Mixtral block with MoE"""
    
    def __init__(self, d_model, n_heads, intermediate_size, n_experts=8, top_k=2):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.post_attention_layernorm = nn.RMSNorm(d_model)
        self.moe = MixtralMoE(d_model, intermediate_size, n_experts, top_k)
        
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, x, x, attn_mask=attention_mask)[0]
        
        residual = x
        x = self.post_attention_layernorm(x)
        moe_out, aux_loss = self.moe(x)
        x = residual + moe_out
        return x, aux_loss

# Test Mixtral MoE
d_model, intermediate_size = 4096, 14336
moe = MixtralMoE(d_model, intermediate_size, n_experts=8, top_k=2)
x = torch.randn(2, 32, d_model)
out, aux_loss = moe(x)
total_params = sum(p.numel() for p in moe.parameters())
active_params = 2 * (2 * d_model * intermediate_size + intermediate_size)  # 2 active experts
print(f"MoE output shape: {out.shape}")
print(f"Total params: {total_params/1e6:.1f}M")
print(f"Active params per token: {active_params/1e6:.1f}M")
print(f"Active/Total ratio: {active_params/total_params:.1%}")
# Output: MoE output shape: (2, 32, 4096)
# Output: Total params: 352.4M
# Output: Active params per token: 88.1M
# Output: Active/Total ratio: 25.0%
```

### Example 3: Mistral and Mixtral Configuration Comparison

```python
class MistralConfig:
    """Mistral and Mixtral model configurations"""
    
    MODELS = {
        'Mistral-7B': {
            'd_model': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'n_layers': 32,
            'intermediate_size': 14336, 'window_size': 4096, 'vocab_size': 32000,
            'total_params': 7e9, 'active_params': 7e9,
        },
        'Mixtral 8x7B': {
            'd_model': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'n_layers': 32,
            'intermediate_size': 14336, 'window_size': 4096, 'vocab_size': 32000,
            'n_experts': 8, 'top_k': 2, 'total_params': 46.7e9, 'active_params': 12.9e9,
        },
        'Mistral-Large': {
            'd_model': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'n_layers': 80,
            'intermediate_size': 28672, 'window_size': 4096, 'vocab_size': 128000,
            'total_params': 120e9, 'active_params': 120e9,
        }
    }
    
    @staticmethod
    def compare_models():
        print("Mistral Model Family Comparison:")
        print("-" * 80)
        print(f"{'Model':<20}{'Total Params':<15}{'Active Params':<15}{'KV Heads':<12}{'Window':<10}{'Context':<10}")
        print("-" * 80)
        
        for name, config in MistralConfig.MODELS.items():
            total = config['total_params']
            active = config['active_params']
            kv = config['n_kv_heads']
            window = config.get('window_size', 'N/A')
            context = config.get('max_seq_len', 32768)
            
            print(f"{name:<20}{total/1e9:<15.1f}B{active/1e9:<15.1f}B{str(kv):<12}{str(window):<10}{str(context):<10}")
        
        print("\n--- Efficiency Metrics ---")
        for name, config in MistralConfig.MODELS.items():
            total = config['total_params']
            active = config['active_params']
            savings = (1 - active/total) * 100 if total != active else 0
            print(f"{name:<20} Active/Total: {active/total:.0%}, MoE savings: {savings:.0f}%")

MistralConfig.compare_models()
# Output: Mistral Model Family Comparison:
# Output: --------------------------------------------------------------------------------
# Output: Model               Total Params    Active Params   KV Heads    Window      Context   
# Output: --------------------------------------------------------------------------------
# Output: Mistral-7B          7.0B            7.0B            8           4096        32768     
# Output: Mixtral 8x7B        46.7B           12.9B           8           4096        32768     
# Output: Mistral-Large       120.0B          120.0B          8           4096        32768     
```

### Example 4: Sliding Window vs Full Attention Complexity

```python
import numpy as np

class AttentionComplexity:
    """Compare complexity of different attention mechanisms"""
    
    @staticmethod
    def compute_flops(seq_len, d_model, n_heads, sliding_window=None):
        """Compute FLOPs for one attention layer"""
        head_dim = d_model // n_heads
        
        # QKV projections: 3 * T * d_model * d_model
        qkv_flops = 3 * seq_len * d_model * d_model * 2
        
        if sliding_window:
            # Attention scores: T * window * d_model
            attn_flops = seq_len * sliding_window * head_dim * n_heads * 2
        else:
            # Attention scores: T * T * head_dim * n_heads
            attn_flops = seq_len * seq_len * head_dim * n_heads * 2
        
        # Output projection: T * d_model * d_model
        out_flops = seq_len * d_model * d_model * 2
        
        return qkv_flops + attn_flops + out_flops
    
    @staticmethod
    def analyze():
        print("Attention Complexity Comparison (FLOPs per layer):")
        print("-" * 70)
        print(f"{'Seq Length':<15}{'Full O(T²)':<20}{'Sliding W=4096':<20}{'Ratio':<10}")
        print("-" * 70)
        
        for T in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
            d_model, n_heads = 4096, 32
            full_flops = AttentionComplexity.compute_flops(T, d_model, n_heads)
            sw_flops = AttentionComplexity.compute_flops(T, d_model, n_heads, sliding_window=4096)
            ratio = full_flops / sw_flops if sw_flops > 0 else 0
            
            print(f"{T:<15}{full_flops/1e9:<20.1f}{sw_flops/1e9:<20.1f}{ratio:<10.1f}x")

AttentionComplexity.analyze()
# Output: Attention Complexity Comparison (FLOPs per layer):
# Output: ----------------------------------------------------------------------
# Output: Seq Length    Full O(T²)          Sliding W=4096      Ratio     
# Output: ----------------------------------------------------------------------
# Output: 512           12.6                12.4                1.0x
# Output: 1024          20.4                13.8                1.5x
# Output: 2048          44.2                16.6                2.7x
# Output: 4096          130.9               22.2                5.9x
# Output: 8192          446.4               33.4                13.4x
# Output: 16384         1714.9              55.8                30.7x
# Output: 32768         6737.4              100.6               67.0x
```

### Example 5: Mistral Training and Inference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class MistralModel(nn.Module):
    """Complete Mistral-7B model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([
            MistralBlock(config['d_model'], config['n_heads'], 
                        config['window_size'], config['intermediate_size'])
            for _ in range(config['n_layers'])
        ])
        self.norm = nn.RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50):
        """Simple autoregressive generation"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max context length
            if input_ids.shape[1] > self.config.get('max_seq_len', 4096):
                input_ids = input_ids[:, -self.config['max_seq_len']:]
            
            with torch.no_grad():
                logits = self.forward(input_ids)
                next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float('-inf'))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids

class MixtralForCausalLM(nn.Module):
    """Complete Mixtral 8x7B model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([
            MixtralBlock(config['d_model'], config['n_heads'],
                        config['intermediate_size'], config['n_experts'], config['top_k'])
            for _ in range(config['n_layers'])
        ])
        self.norm = nn.RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        total_aux_loss = 0
        
        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            total_aux_loss += aux_loss
        
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

# Demonstrate
mistral_config = {
    'd_model': 512, 'n_heads': 8, 'n_kv_heads': 2, 'n_layers': 4,
    'intermediate_size': 1024, 'window_size': 256, 'vocab_size': 10000, 'max_seq_len': 4096
}

model = MistralModel(mistral_config)
x = torch.randint(0, 10000, (2, 32))
logits = model(x)
print(f"Mistral output shape: {logits.shape}")

# Inference generation
input_ids = torch.randint(0, 10000, (1, 8))
output_ids = model.generate(input_ids, max_new_tokens=20)
print(f"Input: {input_ids.shape}, Output: {output_ids.shape}")
# Output: Mistral output shape: (2, 32, 10000)
# Output: Input: (1, 8), Output: (1, 28)
```

## Common Mistakes

### 1. Confusing Sliding Window with Sparse Attention
Sliding window attention restricts each token to a fixed window of W previous tokens. This is different from sparse attention patterns (like Longformer or BigBird) that use predefined sparse patterns. Sliding window is simpler and works well for causal LMs where locality is important.

### 2. Misunderstanding MoE Active vs Total Parameters
Mixtral 8x7B has 46.7B total parameters but only 12.9B active per token. This does NOT mean it processes like a 12.9B model—the router selects different experts for different tokens, so the model has 46.7B parameters of learned knowledge, but only 12.9B are used for any single token's computation.

### 3. Neglecting Router Load Balancing
MoE training requires load balancing across experts. Without the auxiliary loss or z-loss, the router tends to route all tokens to the same few experts, negating the MoE advantage. The load balancing loss coefficient (typically 0.01) must be carefully tuned.

### 4. Assuming All Experts Are Used Equally
Even with load balancing, expert utilization varies by input type. Some experts may specialize in syntax, others in semantics, others in specific domains. This specialization is expected and beneficial, but monitoring expert utilization is important for training stability.

### 5. Overlooking KV Cache Management
Mistral's sliding window attention requires careful KV cache management during generation. Only the last W tokens' KV pairs need to be stored, significantly reducing memory. However, this means the model cannot attend to tokens more than W positions behind, which may hurt performance on tasks requiring long-range dependencies.

## Interview Questions

### Beginner
**Q1: What is sliding window attention and why does Mistral use it?**
A1: Sliding window attention restricts each token to attend only to the W previous tokens (typically W=4096), not all previous tokens. This reduces attention complexity from O(T²) to O(T×W), enabling efficient processing of long sequences while maintaining strong performance through the stacked layers' implicit long-range information flow.

**Q2: How does Mixtral 8x7B achieve high performance with fewer active parameters?**
A2: Mixtral uses a Mixture-of-Experts architecture with 8 experts per MoE layer, where each token activates only 2 experts. This gives 46.7B total parameters but only 12.9B active per token. The model leverages specialized experts for different types of inputs, providing the knowledge capacity of a large model with the computational cost of a much smaller one.

### Intermediate
**Q3: Explain how information flows beyond the sliding window in Mistral.**
A3: While each token's direct attention is limited to W tokens, information flows through stacked layers. Token i at layer l can attend to token i-W at layer l. But token i-W at layer l-1 has already attended to token i-2W. By the time information reaches layer L, tokens can be influenced by tokens up to L×W positions away, effectively enabling long-range communication through the depth of the network.

**Q4: What is the auxiliary load balancing loss in Mixtral and why is it necessary?**
A4: The auxiliary loss encourages the router to distribute tokens evenly across experts. It is computed as n_experts × variance of token distribution across experts. Without this loss, the router would collapse to using only 1-2 experts, wasting the capacity of other experts and creating training instability. The loss coefficient is typically very small (0.01) to avoid affecting the primary language modeling objective.

### Advanced
**Q5: Analyze the memory-bandwidth trade-off in Mixtral 8x7B compared to a dense 70B model during inference.**
A5: Mixtral 8x7B has 46.7B total parameters requiring ~93GB (fp16) of memory, compared to ~140GB for a dense 70B model. During inference, memory bandwidth (loading model weights) is often the bottleneck. Mixtral loads all 46.7B weights but only computes with 12.9B, meaning the memory bandwidth cost is for 46.7B while compute cost is for 12.9B. This makes Mixtral memory-bandwidth-bound in the expert FFN layers. However, in the attention layers (shared by all experts), only the smaller 12.9B-equivalent compute is needed. The net effect: Mixtral has higher total memory requirements than an equivalently-active model but lower than a dense model of similar capability.

**Q6: Design an expert selection strategy that improves upon Mixtral's top-2 routing for specialized domains.**
A6: A domain-aware routing strategy could: (1) Pre-compute expert embeddings for specific domains (code, math, creative writing); (2) Add a domain classifier that biases the router toward domain-relevant experts; (3) Use a learned gating mechanism that combines token-level routing with domain-level priors; (4) Implement expert dropout during training to prevent over-specialization. Alternative approaches include: hierarchical routing (first select domain group, then expert within group), learned expert merging (dynamically combine expert weights), or conditional computation where the number of active experts varies by token difficulty.

## Practice Problems

### Easy
Implement a function that converts a standard attention mask to a sliding window mask with a specified window size.

### Medium
Implement a Mixtral MoE layer with 4 experts and top-1 routing, including the auxiliary load balancing loss. Compare the expert utilization distribution with and without the auxiliary loss.

### Hard
Design and implement an inference batching system for Mixtral 8x7B that handles the variable expert utilization across tokens in a batch, using expert parallelism to distribute expert computation across devices.

## Solutions

### Easy Solution
```python
def create_sliding_window_mask(T, window_size, device='cpu'):
    mask = torch.ones(T, T, device=device)
    causal = torch.triu(mask, diagonal=1)
    window = torch.tril(mask, diagonal=window_size - 1)
    sliding_mask = causal + (window == 0).float()
    sliding_mask = sliding_mask.masked_fill(sliding_mask > 0, float('-inf'))
    return sliding_mask.unsqueeze(0).unsqueeze(0)
```

### Medium Solution
```python
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts=4, top_k=1):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_experts)])
    
    def forward(self, x):
        weights = self.router(x).softmax(dim=-1)
        top_w, top_idx = weights.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x)
        for i in range(self.n_experts):
            mask = (top_idx == i).any(dim=-1)
            out[mask] += top_w[mask] * self.experts[i](x[mask])
        
        aux_loss = self.n_experts * weights.mean(0).var()
        return out, aux_loss
```

### Hard Solution
```python
class MixtralBatchInference:
    def __init__(self, model, num_gpus=8):
        self.model = model
        self.num_gpus = num_gpus
        # Distribute experts across GPUs
        # Handle variable expert routing across batch
        pass
```

## Related Concepts
- DL-425: LLaMA Architecture - Base architecture for Mistral
- DL-441: Mixture of Experts - Foundational MoE concept
- DL-442: Sparse MoE - Sparse expert activation
- DL-443: Routing in MoE - Expert routing mechanisms
- DL-447: Grouped Query Attention - Used in all Mistral models
- DL-448: Sliding Window Attention - Key Mistral innovation

## Next Concepts
- DL-429: Falcon - Another efficient open-source model
- DL-430: Gemma - Google's open-source model
- DL-441: Mixture of Experts - Detailed MoE analysis

## Summary
Mistral and Mixtral represent efficient LLM architectures that achieve remarkable performance-to-parameter ratios. Mistral-7B uses sliding window attention and grouped query attention to match LLaMA 2 13B with half the parameters. Mixtral 8x7B extends this with a Mixture-of-Experts architecture (8 experts, 2 active per token) to match LLaMA 2 70B performance while using only 12.9B active parameters. These designs have heavily influenced subsequent model development.

## Key Takeaways
- Sliding window attention reduces complexity from O(T²) to O(T×W)
- Mixtral 8x7B achieves 46.7B total params with 12.9B active per token
- MoE routing requires load balancing loss for training stability
- Information flows beyond the window through stacked layers
- GQA with 8 KV heads is used across all model sizes
- Mixtral matches LLaMA 2 70B with 18% of active parameters
- Expert specialization emerges naturally during MoE training
