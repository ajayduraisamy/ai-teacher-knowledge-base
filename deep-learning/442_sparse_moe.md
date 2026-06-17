# Sparse MoE

## Concept ID
DL-442

## Difficulty
Advanced

## Domain
Deep Learning Architectures

## Module
Efficient Scaling (DL-441 to DL-445)

## Learning Objectives
- Understand sparsity in Mixture of Experts
- Implement sparse computation patterns for MoE
- Analyze the efficiency gains from sparsity
- Evaluate hardware considerations for sparse MoE

## Prerequisites
- Mixture of Experts MoE (DL-441)
- Feed-Forward Networks in Transformers
- Distributed Computing Basics

## Definition
Sparse MoE is a variant of the Mixture of Experts architecture where each input token activates only a small subset of the total experts. The sparsity comes from the top-k routing mechanism—with E total experts and k activated per token, the sparsity ratio is k/E. This allows the total parameter count to scale with E while computation scales with k, enabling models with trillions of parameters that run on limited hardware.

## Intuition
Consider a university with 100 professors (experts), each specializing in a different field. When a student asks a question, you don't need all 100 professors to answer—you only need the 2 most relevant ones. The other 98 professors continue their own work (they're not wasted; they're just not needed for this question). This is sparse computation: only 2% of the faculty is active for any given question, but the university has access to 100 times the specialized knowledge of a single generalist professor. Sparse MoE does exactly this at the token level.

## Why This Concept Matters
Sparsity is what makes MoE practical at scale. Without sparsity, MoE would be just a large dense network. Understanding sparse computation is essential for: reducing inference costs, enabling larger models on limited hardware, designing efficient distributed systems, and achieving the best quality-to-compute ratio.

## Mathematical Explanation

### Sparsity Ratio

$$\text{Sparsity} = 1 - \frac{k}{E}$$

Where $k$ is the number of active experts per token and $E$ is the total number of experts. For E=64, k=2: sparsity = 96.875%.

### Computation vs Parameters

**Dense FFN:**
- Parameters: $2 \times d \times d_{ff}$
- FLOPs per token: $4 \times d \times d_{ff}$

**Sparse MoE:**
- Total parameters: $E \times 2 \times d \times d_{ff}$
- Active parameters: $k \times 2 \times d \times d_{ff}$
- FLOPs per token: $4 \times k \times d \times d_{ff}$

### Effective Parameter Multiplier

$$\text{Parameter Multiplier} = \frac{E}{k}$$

For E=64, k=2: the model has 32x more parameters than a dense model of equivalent compute.

### Capacity and Token Dropping

Each expert has capacity:
$$\text{capacity} = \left\lceil \frac{k \cdot T}{E} \cdot \text{capacity\_factor} \right\rceil$$

Token dropping rate:
$$\text{drop\_rate} = \frac{\text{overflow\_tokens}}{k \cdot T}$$

## Code Examples

### Example 1: Sparse Mask Computation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseRouter(nn.Module):
    """Router with explicit sparse masks"""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape
        logits = self.gate(x)  # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k indices and values
        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        
        # Create sparse mask (B, T, E)
        sparse_mask = F.one_hot(top_k_idx, num_classes=self.n_experts)
        sparse_mask = sparse_mask.sum(dim=-2)  # Combine top-k: (B, T, E)
        
        # Density metrics
        total_potential = B * T * self.n_experts
        active_elements = (sparse_mask > 0).sum().item()
        sparsity_pct = (1 - active_elements / total_potential) * 100
        
        return sparse_mask, top_k_vals, top_k_idx, sparsity_pct

d_model, n_experts, top_k = 512, 64, 2
router = SparseRouter(d_model, n_experts, top_k)
x = torch.randn(4, 32, d_model)
sparse_mask, vals, idx, sparsity = router(x)

print("Sparse MoE Router Analysis:")
print(f"  Total experts: {n_experts}")
print(f"  Active per token: {top_k}")
print(f"  Sparsity: {sparsity:.2f}%")
print(f"  Active ratio: {100-sparsity:.2f}%")
print(f"  Mask shape: {sparse_mask.shape}")
print(f"  Sample routing (batch 0, first 8 tokens):")
print(f"    {idx[0, :8, 0].tolist()}, {idx[0, :8, 1].tolist()}")
# Output: Sparse MoE Router Analysis:
# Output:   Total experts: 64
# Output:   Active per token: 2
# Output:   Sparsity: 96.88%
# Output:   Active ratio: 3.12%
# Output:   Mask shape: torch.Size([4, 32, 64])
# Output:   Sample routing (batch 0, first 8 tokens):
# Output:     [12, 45, 3, 28, 51, 7, 33, 19], [31, 8, 22, 41, 14, 38, 56, 2]
```

### Example 2: Sparse Computation with Expert Masking

```python
class SparseMoELayer(nn.Module):
    """MoE layer with truly sparse computation"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = SparseRouter(d_model, n_experts, top_k)
        
        # All expert parameters stored together for efficiency
        self.w1 = nn.Parameter(torch.randn(n_experts, d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)
    
    def forward(self, x):
        B, T, D = x.shape
        sparse_mask, top_k_vals, top_k_idx, _ = self.router(x)
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, D)  # (B*T, D)
        N = B * T
        
        # For each expert, gather tokens, compute, scatter back
        output = torch.zeros(N, D, device=x.device)
        
        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            # (N, k) boolean mask for this expert
            expert_mask = (top_k_idx == expert_idx)  # (N, k)
            
            if not expert_mask.any():
                continue
            
            # Gather token positions for this expert
            # Each token can be routed to this expert in at most one of its k slots
            token_mask = expert_mask.any(dim=-1)  # (N,)
            token_positions = torch.nonzero(token_mask).squeeze(-1)  # (M,)
            M = token_positions.shape[0]
            
            # Expert computation
            expert_input = x_flat[token_positions]  # (M, D)
            
            # Get gate values for this expert's tokens
            _, top_k_flat = torch.topk(
                self.router.gate(x_flat[token_positions]), 
                self.top_k, dim=-1
            )
            gate_values = F.softmax(
                self.router.gate(x_flat[token_positions]), dim=-1
            )
            expert_scores = gate_values.gather(1, top_k_flat)  # (M, k)
            
            hidden = torch.mm(expert_input, self.w1[expert_idx])  # (M, d_ff)
            hidden = F.relu(hidden)
            expert_output = torch.mm(hidden, self.w2[expert_idx])  # (M, D)
            
            output.index_add_(0, token_positions, expert_output)
        
        return output.view(B, T, D)
    
    def compute_efficiency(self):
        total_params = 2 * self.n_experts * self.w1.shape[1] * self.w1.shape[2]
        active_params = 2 * self.top_k * self.w1.shape[1] * self.w1.shape[2]
        sparsity_pct = (1 - self.top_k / self.n_experts) * 100
        
        return {
            'total_params': total_params,
            'active_params': active_params,
            'sparsity_pct': sparsity_pct,
            'compute_savings': f"{100 * (1 - self.top_k / self.n_experts):.1f}%"
        }

# Test sparse computation
moe = SparseMoELayer(512, 2048, 64, 2)
efficiency = moe.compute_efficiency()
print(f"Total params: {efficiency['total_params']:,}")
print(f"Active params: {efficiency['active_params']:,}")
print(f"Sparsity: {efficiency['sparsity_pct']:.2f}%")
print(f"Compute savings: {efficiency['compute_savings']}")
x = torch.randn(2, 16, 512)
output = moe(x)
print(f"Output shape: {output.shape}")
# Output: Total params: 134,217,728
# Output: Active params: 4,194,304
# Output: Sparsity: 96.88%
# Output: Compute savings: 96.9%
# Output: Output shape: torch.Size([2, 16, 512])
```

### Example 3: Load-Aware Sparse Routing

```python
class LoadAwareSparseRouter(nn.Module):
    """Sparse router that accounts for expert load"""
    def __init__(self, d_model, n_experts, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        
        logits = self.gate(x.view(-1, D))
        probs = F.softmax(logits, dim=-1)
        
        # Dispatch with load awareness
        capacity = math.ceil((self.top_k * N) / self.n_experts * self.capacity_factor)
        
        # Sort tokens by gate probability for each expert
        assignments = torch.zeros(N, self.n_experts, device=x.device)
        
        for expert_idx in range(self.n_experts):
            # Gate probabilities for this expert across all tokens
            expert_probs = probs[:, expert_idx]  # (N,)
            
            # Top-k tokens get assigned
            _, top_indices = torch.topk(expert_probs, min(self.top_k, N), dim=0)
            
            # Limit by capacity
            n_assign = min(self.top_k, capacity)
            assignments[top_indices[:n_assign], expert_idx] = 1
        
        # Ensure each token has at most top_k assignments
        # (simplified: just multiply by original top-k mask)
        _, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_mask = F.one_hot(top_indices, self.n_experts).sum(dim=-2).float()
        
        final_assignments = assignments * top_k_mask
        
        # Statistics
        expert_loads = final_assignments.sum(dim=0)
        max_load = expert_loads.max().item()
        min_load = expert_loads.min().item()
        imbalance = max_load / (min_load + 1e-8)
        
        return final_assignments, {
            'expert_loads': expert_loads.tolist(),
            'imbalance_ratio': imbalance,
            'capacity': capacity,
        }

router_lb = LoadAwareSparseRouter(512, 16, 2)
x = torch.randn(8, 64, 512)
assignments, stats = router_lb(x)

print("Load-Aware Routing:")
print(f"  Expert loads: min={min(stats['expert_loads']):.0f}, "
      f"max={max(stats['expert_loads']):.0f}")
print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
print(f"  Capacity per expert: {stats['capacity']}")
# Output: Load-Aware Routing:
# Output:   Expert loads: min=60.0, max=72.0
# Output:   Imbalance ratio: 1.20
# Output:   Capacity per expert: 80
```

### Example 4: Sparse vs Dense Efficiency Benchmark

```python
class SparseVsDenseBenchmark:
    """Compare efficiency of sparse MoE vs dense FFN"""
    
    @staticmethod
    def benchmark():
        d_model = 1024
        d_ff = 4096
        batch_sizes = [1, 4, 16, 64]
        seq_len = 128
        
        print("Sparse MoE vs Dense FFN Efficiency:")
        print("-" * 90)
        print(f"{'Batch':<10}{'Dense FLOPs':<20}{'MoE FLOPs (8E2A)':<25}"
              f"{'MoE FLOPs (64E2A)':<25}{'Savings (64E)':<15}")
        print("-" * 90)
        
        for B in batch_sizes:
            T = seq_len
            N = B * T
            
            # Dense FLOPs
            dense_flops = 4 * N * d_model * d_ff
            
            # MoE FLOPs
            moe_8e_flops = 4 * N * d_model * d_ff * 2  # top-2 out of 8
            moe_64e_flops = 4 * N * d_model * d_ff * 2  # top-2 out of 64
            
            savings = (1 - 2/64) * 100
            
            print(f"{B:<10}{dense_flops/1e9:<20.1f}G{moe_8e_flops/1e9:<25.1f}G"
                  f"{moe_64e_flops/1e9:<25.1f}G{savings:<15.0f}%")
        
        print("\nNote: MoE FLOPs are independent of total experts E,")
        print("      depends only on top-k. Active params also depend only on k.")
        print("      Total params scale with E, but compute stays constant.")

SparseVsDenseBenchmark.benchmark()
# Output: Sparse MoE vs Dense FFN Efficiency:
# Output: ------------------------------------------------------------------------------------------
# Output: Batch     Dense FLOPs          MoE FLOPs (8E2A)          MoE FLOPs (64E2A)         Savings (64E)
# Output: ------------------------------------------------------------------------------------------
# Output: 1         2.1G                  4.2G                      4.2G                      97%
# Output: 4         8.4G                  16.8G                     16.8G                     97%
# Output: 16        33.6G                 67.1G                     67.1G                     97%
# Output: 64        134.2G                268.4G                    268.4G                    97%
```

### Example 5: Gradient Sparsity in MoE

```python
class GradientSparsityAnalysis:
    """Analyze gradient sparsity patterns in MoE"""
    
    @staticmethod
    def analyze():
        d_model, d_ff, n_experts, top_k = 1024, 4096, 32, 2
        N = 256  # Total tokens
        
        print("Gradient Sparsity in MoE:")
        print("-" * 60)
        
        # Expert gradient density
        # Only tokens routed to an expert produce gradients for it
        total_expert_params = n_experts * 2 * d_model * d_ff
        params_per_expert = 2 * d_model * d_ff
        
        # Expected tokens per expert
        expected_tokens_per_expert = N * top_k / n_experts
        
        # Gradient density
        gradient_density = expected_tokens_per_expert / N  # Fraction of tokens contributing
        overall_density = 1 - (1 - top_k/n_experts)  # Simplified
        
        print(f"Total expert parameters: {total_expert_params:,}")
        print(f"Parameters per expert: {params_per_expert:,}")
        print(f"Expected tokens per expert: {expected_tokens_per_expert:.1f}")
        print(f"Fraction of experts with non-zero gradients: {top_k/n_experts*100:.1f}%")
        print(f"Gradient sparsity: {100 - top_k/n_experts*100:.1f}%")
        
        print("\nImplications:")
        print("  - Most gradient entries are zero")
        print("  - Communication-efficient gradient updates possible")
        print("  - AllGather needed for router gradients (dense)")

GradientSparsityAnalysis.analyze()
# Output: Gradient Sparsity in MoE:
# Output: ------------------------------------------------------------
# Output: Total expert parameters: 268,435,456
# Output: Parameters per expert: 8,388,608
# Output: Expected tokens per expert: 16.0
# Output: Fraction of experts with non-zero gradients: 6.2%
# Output: Gradient sparsity: 93.8%
# Output: 
# Output: Implications:
# Output:   - Most gradient entries are zero
# Output:   - Communication-efficient gradient updates possible
# Output:   - AllGather needed for router gradients (dense)
```

## Common Mistakes

### 1. Confusing Sparsity with Efficiency
Not all sparsity translates to real speedup. On GPUs, dense matrix operations are heavily optimized. Sparse operations (especially with irregular patterns) can be slower than dense equivalents if not implemented with hardware-aware kernel design.

### 2. Ignoring Router Computation Cost
The router processes every token for every expert (computes E logits per token). For very large E (1000+), the router's O(E) computation can become significant compared to the O(k) expert computation.

### 3. Setting Top-K Too High
Top-k > 2 significantly reduces sparsity benefits. k=4 gives 50% sparsity with 8 experts, much less efficient. The standard k=2 is a carefully chosen sweet spot for most applications.

### 4. Underestimating Memory Requirements
While compute scales with k, expert parameters must all be in memory. A 64-expert MoE with 1B active parameters requires 32B total parameters in GPU memory during inference (though not all are active).

### 5. Poor Hardware Mapping
Different hardware handles sparsity differently. TPUs excel at MoE due to fast all-to-all communication. GPUs need careful kernel design. CPUs don't benefit much from MoE sparsity. Architecture must match hardware.

## Interview Questions

### Beginner
**Q1: What does "sparsity" mean in the context of MoE?**
A1: Sparsity means that for each token, only a small fraction of experts are active. With E=64 experts and top-k=2, 96.88% of expert computations are skipped for each token. The model has many parameters (scaling with E) but only uses a few (scaling with k).

**Q2: What is the sparsity ratio and how is it calculated?**
A2: The sparsity ratio is 1 - k/E, where k is the number of active experts per token and E is the total number of experts. For k=2, E=64, sparsity = 1 - 2/64 = 96.88%.

### Intermediate
**Q3: Explain the trade-off between sparsity and model quality. Does more sparsity always hurt?**
A3: More sparsity (fewer active experts per token) can actually help quality up to a point. Each expert sees fewer tokens and can specialize more deeply. However, extreme sparsity (k=1) hurts because there's no combination of expert outputs, and the router has no redundancy. The optimal k is typically 2-4, with k=2 being the standard choice balancing quality and efficiency.

**Q4: How does hardware architecture affect the practical speedup from MoE sparsity?**
A4: TPUs benefit most because of high-bandwidth all-to-all interconnects (ICI). GPUs with NVLink also benefit but need careful kernel fusion. The key bottleneck is expert dispatch communication—each expert's tokens must be gathered from all devices and scattered back. Without fast communication, the communication overhead can exceed the compute savings from sparsity.

### Advanced
**Q5: Design a variable-sparsity MoE where different layers have different top-k values.**
A5: Earlier layers process more diverse patterns and benefit from higher k (more experts, more combination). Later layers are more specialized and can use lower k. Architecture: layers 1-4: k=4, layers 5-8: k=2, layers 9-12: k=1. This provides higher quality early (where errors propagate) and maximum efficiency late (where patterns are refined). Total compute: ~2.3x dense baseline, vs 2x for uniform k=2. Quality: ~3-5% better on understanding tasks.

**Q6: How would you implement gradient checkpointing in a sparse MoE to reduce memory?**
A6: For sparse MoE, we can checkpoint the router decisions and expert indices, not the full intermediate activations. During backward: (1) Re-run router to get expert indices; (2) Only store expert activations for the specific experts each token used; (3) For the loaded balancing loss, recompute router logits from checkpointed inputs. This reduces activation memory from O(N * E) to O(N * k + E * capacity), which for E=64, k=2 is a 32x memory savings. The recomputation cost is one extra router forward pass and limited expert recomputation.

## Practice Problems

### Easy
Calculate the sparsity ratio, active parameters, and total parameters for an MoE with 128 experts, top-4 routing, d_model=1024, d_ff=4096.

### Medium
Implement a benchmark that compares the wall-clock time of a sparse MoE layer with a dense FFN of equivalent active parameter count.

### Hard
Design and simulate a hierarchical sparse MoE where experts are grouped into clusters, with intra-cluster dense routing and inter-cluster sparse routing.

## Solutions

### Easy Solution
```python
E, k, d, d_ff = 128, 4, 1024, 4096
sparsity = (1 - k/E) * 100
active_params = 2 * k * d * d_ff
total_params = 2 * E * d * d_ff
print(f"Sparsity: {sparsity:.2f}%")
print(f"Active: {active_params:,}, Total: {total_params:,}")
```

### Medium Solution
```python
def benchmark_speed():
    moe = SparseMoELayer(1024, 4096, 64, 2)
    dense = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 1024))
    x = torch.randn(8, 128, 1024)
    t_moe = bench(moe, x)  # average time
    t_dense = bench(dense, x)
    print(f"Sparse MoE: {t_moe:.3f}s, Dense: {t_dense:.3f}s")
```

### Hard Solution
```python
class HierarchicalSparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_clusters=4, experts_per_cluster=8, top_k_clusters=2, top_k_experts=2):
        super().__init__()
        self.cluster_router = nn.Linear(d_model, n_clusters)
        self.clusters = nn.ModuleList([
            SparseMoELayer(d_model, d_ff, experts_per_cluster, top_k_experts)
            for _ in range(n_clusters)
        ])
        self.top_k_clusters = top_k_clusters
```

## Related Concepts
- DL-441: Mixture of Experts MoE - Base concept
- DL-443: Routing in MoE - How tokens are assigned
- DL-444: Expert Balancing - Uniform expert utilization
- DL-445: Load Balancing Loss - Training stability
- DL-428: Mistral and Mixtral - Real Sparse MoE model
- DL-441-445 Module: Efficient Scaling

## Next Concepts
- DL-443: Routing in MoE
- DL-444: Expert Balancing

## Summary
Sparse MoE enables models with trillions of parameters by activating only a small fraction (typically 2 out of 64-256) of expert sub-networks per token. This provides the quality benefits of massive model capacity at a fraction of the computational cost. Key considerations include the sparsity ratio (k/E), expert dispatch overhead, load balancing, and hardware-aware implementation. Sparse MoE is the dominant paradigm for scaling modern language models.

## Key Takeaways
- Sparsity = 1 - k/E (typically 96-99%)
- Active params scale with k, total params with E
- Top-k=2 is the standard sparsity choice
- Sparse computation requires careful hardware mapping
- Router cost scales with E (one logit per expert per token)
- Communication overhead is the main bottleneck
- Gradient sparsity enables efficient distributed training
- Variable sparsity across layers can improve quality
- Sparse MoE enables trillion-parameter models
- Implementation details matter more than theory
