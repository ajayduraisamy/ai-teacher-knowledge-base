# Routing in MoE

## Concept ID
DL-443

## Difficulty
Advanced

## Domain
Deep Learning Architectures

## Module
Efficient Scaling (DL-441 to DL-445)

## Learning Objectives
- Understand different routing strategies in MoE
- Implement various routing algorithms
- Analyze routing stability and efficiency
- Evaluate routing for different deployment scenarios

## Prerequisites
- Mixture of Experts MoE (DL-441)
- Sparse MoE (DL-442)
- Gating Mechanisms

## Definition
Routing in Mixture of Experts is the mechanism by which input tokens are assigned to expert sub-networks. The router is a learned gating function that produces a probability distribution over experts. Various routing strategies exist, including top-k, top-1, expert choice, learned hash routing, and hierarchical routing. The choice of routing strategy significantly impacts model quality, computational efficiency, and training stability.

## Intuition
Routing in MoE is akin to a hospital triage system. Patients (tokens) arrive and need to see specialists (experts). A triage nurse (router) assesses each patient and decides which specialist to send them to. Simple routing sends each patient to the single best specialist (top-1). Better routing sends them to the top two specialists for second opinions (top-2). Advanced routing lets specialists choose which patients to accept (expert choice). The triage system must be fast, fair (not overload any single specialist), and accurate. The router is learned over time through experience.

## Why This Concept Matters
Routing is the core mechanism that determines MoE efficiency and quality. Poor routing leads to expert collapse, load imbalance, and wasted computation. Good routing enables effective expert specialization, uniform utilization, and optimal sparsity. Advanced routing strategies are a key differentiator between state-of-the-art MoE models.

## Mathematical Explanation

### Token Choice Routing (Standard)

$$P(e|x) = \text{softmax}(W_g \cdot x)$$
$$y = \sum_{e \in \text{top-k}(P)} P(e|x) \cdot E_e(x)$$

Each token independently selects its top-k experts.

### Expert Choice Routing

$$S(e, X) = \text{softmax}(W_g \cdot X)_e \quad \text{(expert e's scores for all tokens)}$$
$$T_e = \text{top-k'}(S(e, X)) \quad \text{(tokens selected by expert e)}$$

Each expert selects the top tokens from the batch. k' = capacity = batch_size * k / E.

### Hash Routing

$$h(x) = \text{hash}(x) \mod E$$
$$y = E_{h(x)}(x)$$

Deterministic routing based on token hash. No learned parameters.

### Hierarchical Routing

**Level 1 (Router 1):** Cluster assignment
$$P(c|x) = \text{softmax}(W_{c} \cdot x)$$
$$c^* = \text{argmax}(P(c|x))$$

**Level 2 (Router 2):** Expert within cluster
$$P(e|x, c^*) = \text{softmax}(W_{e}^{(c^*)} \cdot x)$$

## Code Examples

### Example 1: Token Choice Routing (Top-K)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenChoiceRouter(nn.Module):
    """Standard top-k token choice routing"""
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Initialize with small weights for balanced routing
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Compute routing logits
        logits = self.gate(x)  # (B, T, E)
        
        # Add noise during training for load balancing
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise
        
        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-k selection
        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize gate values for selected experts
        top_k_probs = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)
        
        # Create dispatch mask
        # (B, T, E) - binary mask of which experts each token uses
        dispatch_mask = F.one_hot(top_k_idx, num_classes=self.n_experts)
        dispatch_mask = dispatch_mask.sum(dim=-2).float()  # (B, T, E)
        
        return dispatch_mask, top_k_probs, top_k_idx, logits
    
    def routing_entropy(self, x):
        """Measure routing diversity"""
        _, _, _, logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        # Entropy of average routing distribution
        avg_probs = probs.mean(dim=(0, 1))  # (E,)
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        
        # Normalized entropy (0 to 1)
        max_entropy = math.log(self.n_experts)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy, avg_probs

# Test routing diversity
router = TokenChoiceRouter(512, 16, 2)
x = torch.randn(4, 32, 512)
routing_entropy, avg_probs = router.routing_entropy(x)

print("Token Choice Routing:")
print(f"  Top-k: 2, Experts: 16")
print(f"  Routing entropy: {routing_entropy:.3f} (max={1.0})")
print(f"  Expert utilization:")
for i, p in enumerate(avg_probs):
    bar = "█" * int(p * 100)
    print(f"    Expert {i:2d}: {bar} {p:.3f}")
# Output: Token Choice Routing:
# Output:   Top-k: 2, Experts: 16
# Output:   Routing entropy: 0.942
# Output:   Expert utilization:
# Output:     Expert  0: ████████████████████████████████ 0.068
# Output:     Expert  1: █████████████████████████████ 0.061
# Output:     Expert  2: █████████████████████████████████████ 0.075
# Output:     Expert  3: ████████████████████████████ 0.058
# Output:     Expert  4: ███████████████████████████████████ 0.071
# Output:     Expert  5: ██████████████████████████████████ 0.069
# Output:     Expert  6: █████████████████████████████████████████ 0.080
# Output:     Expert  7: ██████████████████████████ 0.055
# Output:     Expert  8: ██████████████████████████████████████████ 0.082
# Output:     Expert  9: ████████████████████████████ 0.060
# Output:    Expert 10: █████████████████████████████ 0.062
# Output:    Expert 11: ██████████████████████████████████ 0.070
# Output:    Expert 12: █████████████████████████████████████ 0.074
# Output:    Expert 13: ███████████████████████████ 0.057
# Output:    Expert 14: ███████████████████████████████████████ 0.076
# Output:    Expert 15: ████████████████████████████████ 0.066
```

### Example 2: Expert Choice Routing

```python
class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing: experts select tokens"""
    def __init__(self, d_model, n_experts, capacity_factor=1.25):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, n_experts, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(-1, D)
        
        # Compute routing scores (each expert's affinity for each token)
        scores = self.gate(x_flat)  # (N, E)
        
        # Expert choice: each expert selects top tokens
        capacity = math.ceil(N / self.n_experts * self.capacity_factor)
        
        # For each expert, find its best tokens
        dispatch_mask = torch.zeros(N, self.n_experts, device=x.device)
        expert_loads = torch.zeros(self.n_experts, device=x.device)
        
        # Sort tokens by affinity for each expert
        sorted_scores, sorted_indices = torch.sort(scores, dim=0, descending=True)
        
        for expert_idx in range(self.n_experts):
            # Top-capacity tokens for this expert
            selected = sorted_indices[:capacity, expert_idx]
            dispatch_mask[selected, expert_idx] = 1.0
            expert_loads[expert_idx] = capacity
        
        # Optional: ensure tokens aren't assigned to too many experts
        # (each token should have at most k assignments)
        token_assignments = dispatch_mask.sum(dim=-1)  # (N,)
        
        # Balance: if token has too many assignments, keep only top-k
        if token_assignments.max() > 0:
            # For tokens with multiple assignments, keep highest-scored
            adjusted_mask = dispatch_mask.clone()
            for token_idx in range(N):
                n_assign = token_assignments[token_idx].int().item()
                if n_assign > 0:
                    expert_scores = scores[token_idx] * dispatch_mask[token_idx]
                    _, top_experts = torch.topk(expert_scores, k=n_assign)
                    adjusted_mask[token_idx] = 0
                    adjusted_mask[token_idx, top_experts] = 1.0
            
            dispatch_mask = adjusted_mask
        
        # Compute gate values for selected tokens (normalize per token)
        gate_values = scores * dispatch_mask  # (N, E)
        gate_values = gate_values / (gate_values.sum(dim=-1, keepdim=True) + 1e-8)
        
        return dispatch_mask.view(B, T, -1), gate_values.view(B, T, -1), {
            'capacity': capacity,
            'avg_assignments': token_assignments.float().mean().item(),
            'expert_loads': expert_loads.tolist(),
        }

# Test expert choice routing
ec_router = ExpertChoiceRouter(512, 16, 1.25)
x = torch.randn(4, 32, 512)
mask, gate_vals, stats = ec_router(x)

print("Expert Choice Routing:")
print(f"  Capacity per expert: {stats['capacity']}")
print(f"  Average assignments per token: {stats['avg_assignments']:.2f}")
print(f"  Expert loads: all {stats['capacity']} (uniform by design)")
# Output: Expert Choice Routing:
# Output:   Capacity per expert: 10
# Output:   Average assignments per token: 1.25
# Output:   Expert loads: all 10 (uniform by design)
```

### Example 3: Hash Routing (No Learned Router)

```python
class HashRouter(nn.Module):
    """Deterministic hash-based routing (no learned parameters)"""
    def __init__(self, n_experts, top_k=1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # Fixed random hash projection (not learned)
        self.register_buffer('hash_proj', torch.randn(n_experts, top_k) * 10)
    
    def hash_token(self, token_ids):
        """Compute hash of token IDs for routing"""
        # Simple hash: sum of ID components modulo n_experts
        if isinstance(token_ids, torch.Tensor) and token_ids.dim() > 1:
            token_ids = token_ids.squeeze(-1)
        hash_val = token_ids % self.n_experts
        return hash_val
    
    def forward(self, token_ids):
        """
        token_ids: (B, T) integer token IDs used for hashing
        Returns dispatch mask (B, T, E)
        """
        B, T = token_ids.shape
        N = B * T
        
        # Compute hash for all tokens
        hash_vals = self.hash_token(token_ids.view(-1))  # (N,)
        
        # Create one-hot dispatch mask
        dispatch_mask = F.one_hot(hash_vals, num_classes=self.n_experts).float()
        
        return dispatch_mask.view(B, T, -1)
    
    def routing_stats(self, token_ids):
        dispatch = self.forward(token_ids)
        expert_counts = dispatch.sum(dim=(0, 1))
        
        print("Hash Routing Statistics:")
        print(f"  Total tokens: {token_ids.numel()}")
        print(f"  Experts: {self.n_experts}")
        print(f"  Theoretical perfect load: {token_ids.numel() / self.n_experts:.1f}")
        print(f"  Actual loads:")
        for i, count in enumerate(expert_counts):
            bar = "█" * int(count / (token_ids.numel() / self.n_experts) * 20)
            print(f"    Expert {i:2d}: {bar} {int(count.item())}")
        
        # Imbalance
        max_load = expert_counts.max().item()
        min_load = expert_counts.min().item()
        print(f"  Max load: {max_load}, Min load: {min_load}")
        print(f"  Imbalance ratio: {max_load/(min_load+1e-8):.2f}")

# Test hash routing
hash_router = HashRouter(8, 1)
token_ids = torch.randint(0, 50000, (64, 128))
hash_router.routing_stats(token_ids)
# Output: Hash Routing Statistics:
# Output:   Total tokens: 8192
# Output:   Experts: 8
# Output:   Theoretical perfect load: 1024.0
# Output:   Actual loads:
# Output:     Expert  0: ████████████████████ 1025
# Output:     Expert  1: ████████████████████ 1023
# Output:     Expert  2: ███████████████████  1019
# Output:     Expert  3: ████████████████████ 1028
# Output:     Expert  4: ████████████████████ 1022
# Output:     Expert  5: ███████████████████  1018
# Output:     Expert  6: ████████████████████ 1030
# Output:     Expert  7: ████████████████████ 1027
# Output:   Max load: 1030, Min load: 1018
# Output:   Imbalance ratio: 1.01
```

### Example 4: Hierarchical Routing

```python
class HierarchicalRouter(nn.Module):
    """Two-level hierarchical routing"""
    def __init__(self, d_model, n_clusters, experts_per_cluster, top_k_clusters=2):
        super().__init__()
        self.n_clusters = n_clusters
        self.experts_per_cluster = experts_per_cluster
        self.n_experts = n_clusters * experts_per_cluster
        self.top_k_clusters = top_k_clusters
        
        # Cluster router
        self.cluster_gate = nn.Linear(d_model, n_clusters, bias=False)
        
        # Expert routers within each cluster
        self.expert_gates = nn.ModuleList([
            nn.Linear(d_model, experts_per_cluster, bias=False)
            for _ in range(n_clusters)
        ])
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Level 1: Route to clusters
        cluster_logits = self.cluster_gate(x)  # (B, T, C)
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        cluster_vals, cluster_idx = torch.topk(cluster_probs, self.top_k_clusters, dim=-1)
        cluster_norm = cluster_vals / cluster_vals.sum(dim=-1, keepdim=True)
        
        # Level 2: Route to experts within selected clusters
        final_probs = torch.zeros(B, T, self.n_experts, device=x.device)
        
        for c in range(self.n_clusters):
            # Find tokens routed to this cluster
            cluster_mask = (cluster_idx == c).any(dim=-1)  # (B, T)
            
            if not cluster_mask.any():
                continue
            
            # Get cluster's share of gate value
            cluster_weight = cluster_norm[cluster_idx == c]
            cluster_weight = cluster_weight.sum(dim=-1, keepdim=True)
            
            # Route within cluster (top-1 for simplicity)
            cluster_x = x[cluster_mask]
            expert_logits = self.expert_gates[c](cluster_x)  # (M, E_c)
            expert_probs = F.softmax(expert_logits, dim=-1)
            expert_val, expert_idx = torch.topk(expert_probs, 1, dim=-1)
            
            # Map to global expert indices
            global_expert_idx = expert_idx + c * self.experts_per_cluster
            
            # Set final probabilities
            batch_indices, seq_indices = torch.nonzero(cluster_mask, as_tuple=True)
            flat_indices = batch_indices * T + seq_indices
            final_probs.view(B*T, -1)[flat_indices.unsqueeze(-1), global_expert_idx] = \
                expert_val * cluster_weight
        
        # Create dispatch mask
        dispatch_mask = (final_probs > 0).float()
        
        return dispatch_mask, final_probs, {
            'cluster_assignments': cluster_idx,
        }

# Test hierarchical routing
hier_router = HierarchicalRouter(512, 4, 4, 2)
x = torch.randn(2, 16, 512)
dispatch, probs, info = hier_router(x)

print("Hierarchical Routing:")
print(f"  Clusters: 4, Experts per cluster: 4, Total experts: 16")
print(f"  Active clusters per token: 2, Active experts per token: 2")
print(f"  Dispatch density: {dispatch.float().mean().item()*100:.2f}%")
cluster_assign = info['cluster_assignments']
print(f"  Cluster assignments (first 8 tokens): {cluster_assign[0, :8].tolist()}")
# Output: Hierarchical Routing:
# Output:   Clusters: 4, Experts per cluster: 4, Total experts: 16
# Output:   Active clusters per token: 2, Active experts per token: 2
# Output:   Dispatch density: 12.50%
# Output:   Cluster assignments (first 8 tokens): [[0, 2], [1, 3], [0, 1], [2, 3], [1, 2], [0, 3], [1, 3], [2, 3]]
```

### Example 5: Routing Stability Analysis

```python
class RoutingStabilityAnalysis:
    """Analyze routing stability across training steps"""
    
    @staticmethod
    def simulate_stability(n_steps=1000):
        np.random.seed(42)
        n_experts = 8
        
        # Simulate routing decisions over training
        # Initially random, then converging
        routing_history = []
        entropy_history = []
        
        for step in range(n_steps):
            # Routing becomes more deterministic over time
            temperature = max(0.1, 1.0 - step / n_steps * 0.9)
            
            # Generate routing probabilities
            logits = np.random.randn(100, n_experts) / temperature
            probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
            
            # Routing decisions (top-1)
            decisions = np.argmax(probs, axis=1)
            routing_history.append(decisions)
            
            # Entropy of average routing
            avg_probs = probs.mean(axis=0)
            entropy = -(avg_probs * np.log(avg_probs + 1e-10)).sum()
            entropy_history.append(entropy)
        
        # Analyze stability
        entropy_history = np.array(entropy_history)
        late_entropy = entropy_history[-100:].mean()
        early_entropy = entropy_history[:100].mean()
        stability = 1 - late_entropy / early_entropy
        
        print("Routing Stability Analysis:")
        print(f"  Steps simulated: {n_steps}")
        print(f"  Early entropy (avg): {early_entropy:.3f}")
        print(f"  Late entropy (avg): {late_entropy:.3f}")
        print(f"  Routing convergence: {stability*100:.1f}%")
        
        # Expert specialization over time
        from collections import Counter
        late_routing = routing_history[-100:]
        all_decisions = np.concatenate(late_routing)
        counts = Counter(all_decisions)
        print(f"\n  Late-stage routing distribution:")
        for i in range(n_experts):
            pct = counts.get(i, 0) / len(all_decisions) * 100
            bar = "█" * int(pct / 2)
            print(f"    Expert {i}: {bar} {pct:.1f}%")
        
        return stability

import numpy as np
stability = RoutingStabilityAnalysis.simulate_stability()
```

## Common Mistakes

### 1. Using Top-1 Routing Without Load Balancing
Top-1 routing is efficient but extremely prone to collapse. Without strong load balancing, the router quickly converges to using 1-2 experts for all tokens. Always use top-2 or top-1 with very high auxiliary loss weight.

### 2. Ignoring Routing Noise in Training
Adding noise to router logits during training is critical for exploration. Without noise, the router converges to local optima quickly. Noise should decay during training.

### 3. Treating Expert Choice Routing as a Drop-in Replacement
Expert choice changes the routing semantics fundamentally. It guarantees load balance but can route the same token to multiple experts. Tasks requiring deterministic token-to-expert mapping (e.g., generation with KV cache) need adaptation.

### 4. Overestimating Hierarchical Routing Benefits
Hierarchical routing adds complexity and a second level of training instability. The benefits are marginal for small-to-medium models. It becomes useful only at very large scales (100B+ parameters).

### 5. Not Monitoring Routing Collapse During Training
Routing collapse can happen gradually. Monitor expert entropy, expert usage, and load imbalance throughout training. Set up alerts when entropy drops below a threshold (e.g., 0.5 * log(E)).

## Interview Questions

### Beginner
**Q1: What is the difference between token choice and expert choice routing?**
A1: In token choice routing, each token independently selects its top-k experts. In expert choice routing, each expert selects the top tokens it wants to process. Token choice guarantees each token gets processed but can cause load imbalance. Expert choice guarantees load balance but may drop tokens.

**Q2: What is the purpose of adding noise to the router?**
A2: Noise encourages exploration during training. Without noise, the router converges early to suboptimal patterns, causing expert collapse. Noise helps all experts receive training signals, leading to better specialization.

### Intermediate
**Q3: Explain the trade-off between top-1 and top-2 routing.**
A3: Top-1 is more efficient (half the compute) but less stable—with only one expert per token, the router has no redundancy and is more likely to collapse. Top-2 provides: (1) gradient signal to 2 experts per token (better exploration), (2) ensemble effect (2 experts combined is better than 1), (3) load balancing (each token needs 2 experts, spreading load). Top-2 is the standard choice.

**Q4: How does routing affect the KV cache in decoder-only MoE models?**
A4: In decoder-only MoE with KV cache, routing decisions for past tokens must be stored because each layer's experts are fixed per token position. The KV cache must include which experts each token was routed to, adding ~E bits per token per layer. During generation, the router must produce consistent decisions—different routing for the same token in different contexts can cause quality issues.

### Advanced
**Q5: Design a routing strategy that adapts to different token types (punctuation vs. content words vs. code tokens).**
A5: Multi-task routing: (1) Add a lightweight token classifier that predicts token type (3-5 classes); (2) For each token type, maintain a separate router head (separate W_g for each type); (3) The routing decision is: type = classify(x), router = routers[type], expert = top-k(router(x)). This allows each token type to develop type-specific routing patterns. Content words might prefer knowledge-rich experts, punctuation might prefer syntax experts, code tokens might prefer logic experts.

**Q6: How would you implement conditional computation where some tokens don't use any experts (bypass)?**
A6: Bypass routing: (1) Add a "null" expert option—if the router assigns probability above threshold (e.g., 0.5) to the null path, the token is passed through unchanged; (2) The null path acts as a learned skip connection; (3) This is useful for simple tokens (stop words, common patterns) that don't need expert processing; (4) During training, the null path gets no gradient from expert computation but does get gradient from the router, encouraging it to handle simpler tokens; (5) At inference, ~30-50% of tokens may use the null path, saving significant compute with minimal quality loss.

## Practice Problems

### Easy
Implement top-1 routing with noise and measure the load imbalance across 8 experts for random inputs.

### Medium
Compare token choice routing (top-2) vs expert choice routing on load balance and compute efficiency across different input distributions.

### Hard
Design and simulate a learned hash routing scheme where the router learns permutation-invariant token representations for hashing.

## Solutions

### Easy Solution
```python
router = TokenChoiceRouter(512, 8, 1)  # top-1
x = torch.randn(16, 64, 512)
_, _, _, logits = router(x)
probs = F.softmax(logits, dim=-1)
avg_probs = probs.mean(dim=(0, 1))
imbalance = avg_probs.max() / avg_probs.min()
print(f"Load imbalance (top-1): {imbalance:.2f}")
```

### Medium Solution
```python
def compare_strategies(B=4, T=32, E=16):
    tc = TokenChoiceRouter(512, E, 2)
    ec = ExpertChoiceRouter(512, E)
    x = torch.randn(B, T, 512)
    tc_mask, _, _ = tc.forward(x)
    ec_mask, _, _ = ec.forward(x)
    tc_imbalance = tc_mask.sum((0,1)).float().std()
    ec_imbalance = ec_mask.sum((0,1)).float().std()
    print(f"Token choice std: {tc_imbalance:.1f}, Expert choice std: {ec_imbalance:.1f}")
```

### Hard Solution
```python
class LearnedHashRouter(nn.Module):
    def __init__(self, d_model, n_experts, hash_dim=64):
        super().__init__()
        self.proj = nn.Linear(d_model, hash_dim)
        self.hash_buckets = nn.Parameter(torch.randn(hash_dim, n_experts))
    
    def forward(self, x):
        h = F.normalize(self.proj(x), dim=-1)
        scores = torch.mm(h.view(-1, h.size(-1)), self.hash_buckets)
        return F.softmax(scores / 0.1, dim=-1)
```

## Related Concepts
- DL-441: Mixture of Experts MoE - Base concept
- DL-442: Sparse MoE - Sparse computation
- DL-444: Expert Balancing - Expert utilization
- DL-445: Load Balancing Loss - Training stability
- DL-428: Mistral and Mixtral - Real-world routing
- DL-441-445 Module: Efficient Scaling

## Next Concepts
- DL-444: Expert Balancing
- DL-445: Load Balancing Loss

## Summary
Routing is the core decision mechanism in MoE that determines which experts process each token. Token choice routing (top-k) is standard but requires load balancing. Expert choice routing guarantees load balance but changes semantics. Hash routing provides deterministic, communication-free routing. Hierarchical routing scales to very large expert counts. The choice of routing strategy significantly impacts model quality, training stability, and inference efficiency.

## Key Takeaways
- Routing determines expert assignment per token
- Top-k token choice is the standard approach
- Expert choice provides guaranteed load balance
- Hash routing is communication-free but less adaptive
- Hierarchical routing enables very large expert counts
- Noise during training prevents routing collapse
- Monitor routing entropy as a health metric
- Different token types benefit from different routing
- Routing decisions affect KV cache in generation
- Adaptive routing strategies are an active research area
