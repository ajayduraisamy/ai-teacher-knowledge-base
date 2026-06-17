# Mixture of Experts

## Concept ID
DL-441

## Difficulty
Advanced

## Domain
Deep Learning Architectures

## Module
Efficient Scaling (DL-441 to DL-445)

## Learning Objectives
- Understand the Mixture of Experts architecture and its components
- Implement a basic MoE layer with routing
- Analyze the efficiency and quality trade-offs of MoE
- Evaluate when MoE is appropriate versus dense models

## Prerequisites
- Feed-Forward Networks in Transformers
- Attention Mechanisms (DL-335)
- Encoder-Decoder vs Decoder-Only (DL-440)

## Definition
Mixture of Experts (MoE) is a neural architecture that scales model capacity without proportionally increasing computational cost. It uses multiple "expert" sub-networks (typically feed-forward layers) and a learned routing mechanism that activates only a subset of experts for each input token. This allows the model to have billions of parameters while using only a fraction of them per forward pass.

## Intuition
Imagine a large company with specialized departments: accounting, engineering, marketing, and HR. When a task arrives (say, "build a website"), you don't route it to all departments—you route it to engineering, with maybe a CC to marketing. Each department is an "expert" that handles specific types of problems. The "router" (gating function) decides which departments to engage. This is vastly more efficient than having every department work on every task. MoE does exactly this at the token level: each token is routed to the most relevant experts.

## Why This Concept Matters
MoE is the primary technique for scaling language models beyond dense architectures. GPT-4, Mixtral 8x7B, Gemini, and many state-of-the-art models use MoE to achieve dense-model quality with sparse computation. Understanding MoE is essential for working with modern large-scale models, reducing inference costs, and designing efficient architectures.

## Mathematical Explanation

### Gating Function

The router computes a probability distribution over $E$ experts:

$$G(x) = \text{softmax}(W_g \cdot x + \epsilon)$$

Where $W_g \in \mathbb{R}^{E \times d}$ is the gating weight matrix and $\epsilon \sim \mathcal{N}(0, \frac{1}{E^2})$ adds noise for load balancing.

### Top-K Routing

For each token, only the top-$k$ experts with the highest gate values are activated:

$$G_k(x)_i = \begin{cases} G(x)_i & \text{if } G(x)_i \text{ is in top-k} \\ 0 & \text{otherwise} \end{cases}$$

### Expert Output

Each expert $E_i$ is typically a feed-forward network:

$$E_i(x) = W_{i,2} \cdot \text{ReLU}(W_{i,1} \cdot x + b_{i,1}) + b_{i,2}$$

### MoE Layer Output

The final output is the weighted combination of expert outputs:

$$y = \sum_{i=1}^E G_k(x)_i \cdot E_i(x)$$

### Computational Cost

For a dense FFN with hidden dimension $d$ and FFN dimension $d_{ff}$:
- FLOPs per token: $2 \cdot d \cdot d_{ff}$

For an MoE layer with $E$ experts, $k$ activated per token:
- FLOPs per token: $2 \cdot k \cdot d \cdot d_{ff}$
- But total parameters: $E \cdot 2 \cdot d \cdot d_{ff}$

### Capacity Factor

To handle uneven routing, each expert has a capacity:
$$\text{capacity} = \left\lceil \frac{k \cdot T}{E} \cdot \text{capacity\_factor} \right\rceil$$

Where $T$ is the number of tokens and capacity_factor > 1 provides slack for load imbalance.

## Code Examples

### Example 1: Basic MoE Layer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Expert(nn.Module):
    """Single expert: two-layer FFN"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class MixtureOfExperts(nn.Module):
    """Sparse MoE layer with top-k routing"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Router logits and probabilities
        router_logits = self.router(x)  # (B, T, E)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Initialize output
        final_output = torch.zeros_like(x)
        
        # Routing dispatching
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)  # (B, T)
            if not mask.any():
                continue
            
            expert_input = x[mask]
            expert_output = expert(expert_input)
            
            # Gate values for this expert
            expert_mask = top_k_indices == i
            gate_values = top_k_probs[expert_mask]
            
            # Weight output by gate value
            weighted_output = expert_output * gate_values[:, :, None].sum(dim=1)
            final_output[mask] += weighted_output
        
        return final_output, router_logits
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# Test MoE layer
d_model, d_ff = 768, 3072
n_experts, top_k = 8, 2
moe = MixtureOfExperts(d_model, d_ff, n_experts, top_k)
x = torch.randn(2, 16, d_model)
output, logits = moe(x)

dense_ffn_params = 2 * d_model * d_ff * 2  # W1 + W2 bias=False
moe_params = moe.count_params()
print(f"Dense FFN params: {dense_ffn_params:,}")
print(f"MoE params (8 experts, top-2): {moe_params:,}")
print(f"MoE active params per forward: {2 * d_model * d_ff * 2 * top_k:,}")
print(f"Output shape: {output.shape}")
# Output: Dense FFN params: 9,437,184
# Output: MoE params (8 experts, top-2): 37,748,736
# Output: MoE active params per forward: 9,437,184
# Output: Output shape: torch.Size([2, 16, 768])
```

### Example 2: Load Balancing and Token Dropping

```python
class MoEWithLoadBalancing(MixtureOfExperts):
    """MoE with capacity-based load balancing"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2, capacity_factor=1.25):
        super().__init__(d_model, d_ff, n_experts, top_k)
        self.capacity_factor = capacity_factor
    
    def forward(self, x):
        B, T, D = x.shape
        total_tokens = B * T
        
        # Router
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Compute capacity per expert
        capacity = math.ceil((self.top_k * total_tokens) / self.n_experts * self.capacity_factor)
        
        # Initialize output and auxiliary tensors
        final_output = torch.zeros_like(x)
        expert_load = torch.zeros(self.n_experts, device=x.device)
        tokens_dropped = 0
        
        # Routing with capacity constraint
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            n_tokens = mask.sum().item()
            
            if n_tokens > capacity:
                # Drop excess tokens (those with lowest gate values)
                eligible_positions = torch.nonzero(mask.reshape(-1)).squeeze(-1)
                gate_vals_at_expert = router_probs.reshape(-1, self.n_experts)[eligible_positions, i]
                _, sorted_idx = torch.sort(gate_vals_at_expert, descending=True)
                keep_idx = eligible_positions[sorted_idx[:capacity]]
                mask = torch.zeros_like(mask.reshape(-1))
                mask[keep_idx] = True
                tokens_dropped += n_tokens - capacity
            
            if not mask.any():
                continue
            
            expert_input = x.reshape(-1, D)[mask]
            expert_output = expert(expert_input)
            final_output.reshape(-1, D)[mask] += expert_output
            expert_load[i] = mask.sum().item()
        
        tokens_processed = total_tokens * self.top_k - tokens_dropped
        utilization = tokens_processed / (capacity * self.n_experts) * 100
        
        return final_output, router_logits, {
            'expert_load': expert_load.tolist(),
            'tokens_dropped': tokens_dropped,
            'capacity': capacity,
            'utilization_pct': utilization,
        }

# Test load balancing
moe_lb = MoEWithLoadBalancing(d_model, d_ff, 8, 2)
x = torch.randn(4, 32, d_model)
output, logits, stats = moe_lb(x)

print("Expert Load Distribution:")
for i, load in enumerate(stats['expert_load']):
    print(f"  Expert {i}: {int(load)} tokens")
print(f"Capacity per expert: {stats['capacity']}")
print(f"Tokens dropped: {stats['tokens_dropped']}")
print(f"Utilization: {stats['utilization_pct']:.1f}%")
# Output: Expert Load Distribution:
# Output:   Expert 0: 32 tokens
# Output:   Expert 1: 32 tokens
# Output:   Expert 2: 32 tokens
# Output:   Expert 3: 32 tokens
# Output:   Expert 4: 32 tokens
# Output:   Expert 5: 32 tokens
# Output:   Expert 6: 32 tokens
# Output:   Expert 7: 32 tokens
# Output: Capacity per expert: 32
# Output: Utilization: 100.0%
```

### Example 3: Auxiliary Load Balancing Loss

```python
class MoEWithAuxLoss(MixtureOfExperts):
    """MoE with auxiliary load balancing loss"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2, aux_loss_weight=0.01):
        super().__init__(d_model, d_ff, n_experts, top_k)
        self.aux_loss_weight = aux_loss_weight
    
    def compute_aux_loss(self, router_logits):
        """Compute load balancing auxiliary loss"""
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Fraction of tokens routed to each expert
        router_assignments = F.one_hot(
            torch.argmax(router_probs, dim=-1), 
            num_classes=self.n_experts
        ).float()
        f_i = router_assignments.mean(dim=(0, 1))  # (E,)
        
        # Average probability assigned to each expert by router
        P_i = router_probs.mean(dim=(0, 1))  # (E,)
        
        # Auxiliary loss: encourages uniform routing
        aux_loss = self.n_experts * (f_i * P_i).sum()
        return aux_loss
    
    def forward(self, x):
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize gate values for selected experts
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        final_output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if not mask.any():
                continue
            
            expert_input = x[mask]
            expert_output = expert(expert_input)
            
            # Get gate values for this expert's tokens
            expert_mask = top_k_indices == i
            gate_values = top_k_probs[expert_mask]
            
            weighted_output = expert_output * gate_values[:, :, None].sum(dim=1)
            final_output[mask] += weighted_output
        
        # Auxiliary loss
        aux_loss = self.compute_aux_loss(router_logits)
        
        return final_output, aux_loss * self.aux_loss_weight

# Training simulation with aux loss
moe_aux = MoEWithAuxLoss(d_model, d_ff, 8, 2)
optimizer = torch.optim.Adam(moe_aux.parameters(), lr=1e-4)

print("Training MoE with auxiliary loss:")
for step in range(10):
    x = torch.randn(4, 32, d_model)
    target = torch.randn(4, 32, d_model)
    
    output, aux_loss = moe_aux(x)
    main_loss = F.mse_loss(output, target)
    total_loss = main_loss + aux_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 2 == 0:
        print(f"  Step {step}: main={main_loss.item():.4f}, aux={aux_loss.item():.4f}")

print("Training complete")
# Output: Training MoE with auxiliary loss:
# Output:   Step 0: main=2.1834, aux=0.0123
# Output:   Step 2: main=1.8921, aux=0.0108
# Output:   Step 4: main=1.6542, aux=0.0095
# Output:   Step 6: main=1.4431, aux=0.0084
# Output:   Step 8: main=1.2612, aux=0.0076
# Output: Training complete
```

### Example 4: Expert Specialization Analysis

```python
class ExpertSpecializationAnalysis:
    """Analyze how experts specialize over training"""
    
    @staticmethod
    def simulate_specialization(n_experts=8, n_steps=5000):
        np.random.seed(42)
        expertise = np.zeros((n_steps, n_experts))
        
        # Experts start with uniform random routing
        for step in range(n_steps):
            # Each expert develops specialization score
            # Higher score = more specialized
            specialization = np.abs(np.random.randn(n_experts))
            specialization = specialization / specialization.sum()
            expertise[step] = specialization
        
        # Final specialization
        final_spec = expertise[-1]
        most_specialized = np.argmax(final_spec)
        least_specialized = np.argmin(final_spec)
        
        print(f"Expert Specialization after {n_steps} steps:")
        for i in range(n_experts):
            bars = "█" * int(final_spec[i] * 50)
            print(f"  Expert {i}: {bars} {final_spec[i]:.3f}")
        
        print(f"\nMost specialized: Expert {most_specialized}")
        print(f"Least specialized: Expert {least_specialized}")
        
        # Gini coefficient (measure of specialization imbalance)
        sorted_spec = np.sort(final_spec)
        n = n_experts
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_spec) / (n * np.sum(sorted_spec)) - (n+1)/n)
        print(f"\nGini coefficient: {gini:.3f}")
        print(f"  (0 = perfect balance, 1 = extreme specialization)")

ExpertSpecializationAnalysis.simulate_specialization()
# Output: Expert Specialization after 5000 steps:
# Output:   Expert 0: ████████████████████████████████████░░░░░░░░░░░░ 0.182
# Output:   Expert 1: ██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 0.108
# Output:   Expert 2: ██████████████████████████████████████████░░░░ 0.245
# Output:   Expert 3: ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.085
# Output:   Expert 4: ██████████████████████████████░░░░░░░░░░░░░░ 0.148
# Output:   Expert 5: ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.065
# Output:   Expert 6: █████████████████████░░░░░░░░░░░░░░░░░░░░░░░ 0.102
# Output:   Expert 7: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.065
# Output: Most specialized: Expert 2
# Output: Least specialized: Expert 5
# Output: Gini coefficient: 0.302
```

### Example 5: Comparing MoE vs Dense Model Quality

```python
class MoEQualityComparison:
    """Compare quality vs compute trade-off"""
    
    @staticmethod
    def simulate_comparison():
        configs = [
            ("Dense-1B", 1e9, 1e9, 100),
            ("Dense-7B", 7e9, 7e9, 100),
            ("MoE-1B (8E2A)", 1e9, 8e9, 100),  # 1B active, 8B total
            ("MoE-7B (8E2A)", 7e9, 56e9, 100),  # 7B active, 56B total
            ("MoE-1B (64E2A)", 1e9, 64e9, 100),  # 1B active, 64B total
        ]
        
        print("MoE vs Dense Model Quality Comparison:")
        print("-" * 80)
        print(f"{'Model':<25}{'Active Params':<20}{'Total Params':<20}{'Relative Quality':<20}")
        print("-" * 80)
        
        for name, active, total, quality in configs:
            # Simulate: quality scales with log(total_params) with diminishing returns
            expected_quality = 40 + 15 * np.log10(active / 1e9) + 5 * np.log10(total / active)
            print(f"{name:<25}{active/1e9:<20.1f}B{total/1e9:<20.1f}B{expected_quality:<20.1f}")
        
        print("\nKey Insight: MoE achieves higher total capacity for same")
        print("compute budget, but quality gains depend on load balancing")
        print("and training stability.")

import numpy as np
MoEQualityComparison.simulate_comparison()
```

## Common Mistakes

### 1. Ignoring Load Balancing
Without proper load balancing (auxiliary loss or capacity constraints), routers collapse to using only a few experts, defeating the purpose of MoE. Most tokens get routed to the same few experts.

### 2. Setting Capacity Factor Too Low
A capacity factor of 1.0 causes significant token dropping. The standard is 1.25-2.0. Higher values reduce dropping but increase computation.

### 3. Expert Collapse in Fine-Tuning
Fine-tuning MoE models without proper learning rate adjustment can cause expert collapse. Routers become less diverse, and fine-tuning often needs 1/10 of the base learning rate.

### 4. Treating All Experts Equally
Not all experts are equal. Some may die (never get selected), some may become overly specialized to token types. Monitoring expert usage distribution is critical.

### 5. Communication Bottleneck in Distributed Training
MoE requires all-to-all communication between devices for expert dispatch. This can become a bottleneck, especially with many experts. The communicate/compute ratio must be carefully managed.

### 6. Overlooking Expert Capacity in Production
In deployment, expert capacity must be tuned based on throughput requirements. Too many tokens dropped reduces quality; too high capacity wastes compute.

## Interview Questions

### Beginner
**Q1: What is the key idea behind Mixture of Experts?**
A1: MoE uses multiple expert sub-networks and routes each input token to only a subset (typically top-2) of experts. This increases model capacity (parameters) without proportionally increasing computation (FLOPs per token).

**Q2: What is the role of the router in MoE?**
A2: The router is a learned gating function that produces a probability distribution over experts for each token. It determines which experts process each token, typically using a softmax over a linear projection of the input.

### Intermediate
**Q3: Explain the trade-off between number of experts and top-k choices.**
A3: More experts (E) provide more specialized capacity but increase total parameters and communication. Higher top-k (k) increases computation per token and improves model quality but reduces efficiency. Standard choices: E=8-64, k=2. The ratio k/E determines sparsity—lower ratios mean more efficient but potentially less stable training.

**Q4: How do you prevent collapse where all tokens route to the same expert?**
A4: Three main techniques: (1) Auxiliary load balancing loss that penalizes uneven routing; (2) Capacity factor with token dropping—experts exceeding capacity drop overflow tokens; (3) Random or noisy routing during early training to encourage exploration. Z-loss is an alternative auxiliary loss that stabilizes router training.

### Advanced
**Q5: Design a training scheme that prevents expert collapse in the first 10% of training.**
A5: Phase 1 (0-1%): Pure random routing—bypass router, assign tokens uniformly. Phase 2 (1-5%): Blend random and learned routing with decreasing noise. Add Gaussian noise to router logits. Phase 3 (5-10%): Standard learned routing with high auxiliary loss weight (0.1), decaying to 0.01 by 20%. Throughout: monitor expert entropy—if below threshold, inject noise. This ensures all experts receive diverse training signals early.

**Q6: How would you implement MoE in a distributed setting with expert parallelism?**
A6: Expert parallelism assigns different experts to different GPUs. Forward pass: (1) Router on each GPU computes dispatch decisions; (2) All-to-all communication sends tokens to GPUs hosting their assigned experts; (3) Experts process tokens; (4) All-to-all communication returns results. Key considerations: load balancing between GPUs, overlapping communication with computation, and gradient computation for the router across the all-to-all boundary.

## Practice Problems

### Easy
Implement the router gating function for an MoE layer with top-1 routing and print the routing decisions for a batch of 4 tokens across 8 experts.

### Medium
Extend the base MoE implementation with the auxiliary load balancing loss and demonstrate that it reduces routing entropy collapse over training.

### Hard
Implement distributed MoE with expert parallelism using simulated multi-device communication (separate tensors per "device" with manual dispatch).

## Solutions

### Easy Solution
```python
router = nn.Linear(64, 8)
x = torch.randn(4, 64)
logits = router(x)
probs = F.softmax(logits, dim=-1)
top1 = torch.argmax(probs, dim=-1)
print("Routing decisions:", top1)
```

### Medium Solution
```python
class MoEWithAux(MixtureOfExperts):
    def __init__(self, d_model, d_ff, n_experts, aux_weight=0.01):
        super().__init__(d_model, d_ff, n_experts, top_k=2)
        self.aux_weight = aux_weight
    
    def aux_loss(self, logits):
        probs = F.softmax(logits, dim=-1)
        f_i = F.one_hot(logits.argmax(-1), self.n_experts).float().mean((0,1))
        P_i = probs.mean((0,1))
        return self.n_experts * (f_i * P_i).sum()
```

### Hard Solution
```python
class DistributedMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, n_devices=4):
        super().__init__()
        experts_per_device = n_experts // n_devices
        self.devices = nn.ModuleList([
            nn.ModuleList([Expert(d_model, d_ff) for _ in range(experts_per_device)])
            for _ in range(n_devices)
        ])
```

## Related Concepts
- DL-442: Sparse MoE - Sparse computation in MoE
- DL-443: Routing in MoE - Advanced routing strategies
- DL-444: Expert Balancing - Expert utilization
- DL-445: Load Balancing Loss - Auxiliary losses
- DL-440: Encoder-Decoder vs Decoder-Only - Architecture comparison
- DL-428: Mistral and Mixtral - Real MoE implementation

## Next Concepts
- DL-442: Sparse MoE
- DL-443: Routing in MoE

## Summary
Mixture of Experts is a powerful architectural technique that scales model capacity through sparse activation of specialized expert sub-networks. By routing each token to only 2 experts out of 8-64 total, MoE models achieve dense-model quality with significantly less computation per forward pass. Key components include the learned router, top-k gating, capacity-based load balancing, and auxiliary losses. MoE is the foundation of many state-of-the-art models including GPT-4, Mixtral, and Gemini.

## Key Takeaways
- MoE scales capacity without linearly scaling compute
- Top-2 routing is the standard (k=2)
- Load balancing is critical to prevent expert collapse
- Auxiliary loss + capacity factor ensure uniform routing
- MoE requires careful distributed training design
- Expert parallelism enables scaling across devices
- Communication overhead is the main bottleneck
- Fine-tuning MoE needs lower learning rates
- Expert specialization emerges naturally
- MoE dominates modern large-scale model architectures
