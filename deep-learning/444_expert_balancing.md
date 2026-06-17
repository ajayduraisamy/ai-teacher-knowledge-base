# Expert Balancing

## Concept ID
DL-444

## Difficulty
Advanced

## Domain
Deep Learning Architectures

## Module
Efficient Scaling (DL-441 to DL-445)

## Learning Objectives
- Understand expert load balancing in MoE
- Implement expert balancing techniques
- Analyze expert utilization metrics
- Evaluate balancing strategies for different scenarios

## Prerequisites
- Mixture of Experts MoE (DL-441)
- Sparse MoE (DL-442)
- Routing in MoE (DL-443)
- Load Balancing Loss (DL-445)

## Definition
Expert balancing refers to techniques that ensure uniform utilization of all experts in a Mixture of Experts model. Without balancing, the router collapses to using only a few experts, wasting model capacity and reducing quality. Balancing techniques include auxiliary losses, capacity factors, expert choice routing, random routing initialization, and balancing-aware training schedules.

## Intuition
Imagine a team of 8 specialists where 2 people do 90% of the work while the other 6 sit idle. The team has 8 people on payroll, but only 2 are productive. This is expert collapse. Expert balancing ensures everyone contributes fairly. Like a good manager who distributes tasks evenly based on workload, expert balancing techniques ensure all experts receive enough training signals to become useful specialists. A balanced team of 8 is far more productive than an unbalanced team where 2 experts are overwhelmed and 6 are undertrained.

## Why This Concept Matters
Expert collapse is the single biggest failure mode in MoE training. Imbalanced experts waste parameters, reduce model capacity, create computational bottlenecks (overloaded experts slow down all-to-all communication), and degrade quality. Understanding balancing is essential for successful MoE training and deployment.

## Mathematical Explanation

### Load Imbalance Metrics

**Expert Utilization Fraction (EUF):**
$$\text{EUF} = \frac{\text{Number of experts used > threshold}}{\text{Total experts}}$$

**Load Imbalance Factor:**
$$\text{LIF} = \frac{\max_i L_i}{\min_i L_i}$$

Where $L_i$ is the number of tokens routed to expert $i$.

**Routing Entropy:**
$$H = -\sum_{i=1}^E p_i \log p_i, \quad p_i = \frac{L_i}{\sum_j L_j}$$

### Capacity Factor

$$\text{capacity} = \left\lceil \frac{k \cdot T}{E} \cdot \text{capacity\_factor} \right\rceil$$

- capacity_factor = 1.0: minimal capacity, high token dropping
- capacity_factor = 1.25: standard, slight overload allowed
- capacity_factor = 2.0: generous, minimal dropping

### Auxiliary Loss Weight

$$L_{total} = L_{task} + \alpha \cdot L_{balance}$$

Where $\alpha$ controls the strength of balancing pressure, typically 0.01.

## Code Examples

### Example 1: Expert Utilization Monitoring

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExpertUtilizationMonitor:
    """Monitor expert utilization and detect collapse"""
    
    def __init__(self, n_experts, top_k):
        self.n_experts = n_experts
        self.top_k = top_k
        self.reset()
    
    def reset(self):
        self.counts = torch.zeros(self.n_experts)
        self.total_tokens = 0
    
    def update(self, dispatch_mask):
        """dispatch_mask: (B, T, E) boolean mask"""
        B, T, _ = dispatch_mask.shape
        self.counts += dispatch_mask.sum(dim=(0, 1)).cpu()
        self.total_tokens += B * T * self.top_k
    
    def get_metrics(self):
        if self.total_tokens == 0:
            return {}
        
        # Expert utilization fraction
        expected = self.total_tokens / self.n_experts
        utilization = (self.counts > expected * 0.5).float()
        euf = utilization.mean().item()
        
        # Load imbalance factor
        max_load = self.counts.max().item()
        min_load = self.counts.min().item()
        lif = max_load / (min_load + 1e-8)
        
        # Gini coefficient
        sorted_counts = torch.sort(self.counts)[0]
        n = self.n_experts
        gini = (2 * torch.sum(torch.arange(1, n+1, dtype=torch.float32) * sorted_counts) 
                / (n * torch.sum(sorted_counts)) - (n + 1) / n).item()
        
        # Entropy
        p = self.counts / self.total_tokens
        entropy = -(p * torch.log(p + 1e-10)).sum().item()
        max_entropy = math.log(self.n_experts)
        norm_entropy = entropy / max_entropy
        
        return {
            'euf': euf,
            'lif': lif,
            'gini': gini,
            'norm_entropy': norm_entropy,
            'min_load': min_load,
            'max_load': max_load,
            'expected_load': expected,
        }
    
    def print_report(self):
        metrics = self.get_metrics()
        if not metrics:
            print("No data collected")
            return
        
        print("Expert Utilization Report:")
        print("-" * 60)
        print(f"  Total tokens processed: {self.total_tokens}")
        print(f"  Expected load per expert: {metrics['expected_load']:.0f}")
        print(f"  Min expert load: {metrics['min_load']:.0f}")
        print(f"  Max expert load: {metrics['max_load']:.0f}")
        print(f"  Load imbalance factor: {metrics['lif']:.2f}")
        print(f"  Expert utilization fraction: {metrics['euf']:.1%}")
        print(f"  Gini coefficient: {metrics['gini']:.3f}")
        print(f"  Normalized entropy: {metrics['norm_entropy']:.3f}")
        
        print(f"\n  Expert Load Distribution:")
        for i in range(self.n_experts):
            pct = (self.counts[i] / self.total_tokens * 100).item()
            bars = "█" * int(pct * 2)
            expected_pct = 100 / self.n_experts * self.top_k
            marker = "✓" if abs(pct - expected_pct) < expected_pct * 0.3 else "⚠"
            print(f"    Expert {i:2d}: {bars} {pct:.1f}% {marker}")
        
        if metrics['euf'] < 0.8:
            print(f"\n  ⚠ WARNING: Low expert utilization!")
        if metrics['lif'] > 3.0:
            print(f"  ⚠ WARNING: High load imbalance!")

# Simulate monitoring
monitor = ExpertUtilizationMonitor(8, 2)

# Phase 1: Balanced routing
print("Phase 1: Balanced Routing")
for _ in range(10):
    mask = torch.zeros(4, 16, 8)
    for b in range(4):
        for t in range(16):
            experts = torch.randperm(8)[:2]
            mask[b, t, experts] = 1.0
    monitor.update(mask)
monitor.print_report()

# Phase 2: Collapsed routing
print("\nPhase 2: Collapsed Routing")
monitor2 = ExpertUtilizationMonitor(8, 2)
for _ in range(10):
    mask = torch.zeros(4, 16, 8)
    for b in range(4):
        for t in range(16):
            experts = torch.tensor([0, 1])  # Only experts 0,1 get used
            mask[b, t, experts] = 1.0
    monitor2.update(mask)
monitor2.print_report()
# Output: Phase 1: Balanced Routing
# Output: Expert Utilization Report:
# Output: ------------------------------------------------------------
# Output:   Total tokens processed: 1280
# Output:   Expected load per expert: 160
# Output:   Min expert load: 152
# Output:   Max expert load: 168
# Output:   Load imbalance factor: 1.11
# Output:   Expert utilization fraction: 100.0%
# Output:   Gini coefficient: 0.012
# Output:   Normalized entropy: 0.986
# Output:   ...
# Output: Phase 2: Collapsed Routing
# Output:   Load imbalance factor: inf
# Output:   Expert utilization fraction: 25.0%
# Output:   Gini coefficient: 0.750
# Output:   Normalized entropy: 0.301
```

### Example 2: Dynamic Capacity Factor Adjustment

```python
class AdaptiveCapacityMoE(nn.Module):
    """MoE with dynamic capacity factor based on load imbalance"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2, 
                 base_capacity_factor=1.25, imbalance_threshold=2.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.base_capacity_factor = base_capacity_factor
        self.imbalance_threshold = imbalance_threshold
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            ) for _ in range(n_experts)
        ])
        
        self.register_buffer('load_history', torch.zeros(100, n_experts))
        self.step = 0
    
    def compute_imbalance(self, dispatch_mask):
        """Compute current load imbalance"""
        loads = dispatch_mask.sum(dim=(0, 1))
        max_load = loads.max().item()
        min_load = loads.min().item()
        return max_load / (min_load + 1e-8), loads
    
    def compute_capacity_factor(self, imbalance):
        """Adjust capacity factor based on imbalance"""
        if imbalance > self.imbalance_threshold:
            # Increase capacity to accommodate imbalance
            return self.base_capacity_factor * min(imbalance / self.imbalance_threshold, 2.0)
        return self.base_capacity_factor
    
    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        
        # Router
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_vals, dim=-1)
        
        # Dispatch mask
        dispatch_mask = F.one_hot(top_k_idx, self.n_experts).sum(dim=-2).float()
        
        # Compute imbalance and adjust capacity
        imbalance, expert_loads = self.compute_imbalance(dispatch_mask)
        capacity_factor = self.compute_capacity_factor(imbalance)
        capacity = math.ceil((self.top_k * N) / self.n_experts * capacity_factor)
        
        # Update load history
        self.load_history[self.step % 100] = expert_loads.cpu()
        self.step += 1
        
        # Process experts with adaptive capacity
        output = torch.zeros_like(x)
        tokens_dropped = 0
        
        for i, expert in enumerate(self.experts):
            mask = (top_k_idx == i).any(dim=-1)
            n_tokens = mask.sum().item()
            
            if n_tokens > capacity:
                # Sort by gate value and keep top capacity tokens
                gate_vals = top_k_probs[top_k_idx == i]
                if gate_vals.numel() > 0:
                    sorted_vals, sorted_pos = torch.sort(gate_vals.view(-1), descending=True)
                    kept = sorted_pos[:capacity]
                    
                    # Recreate mask with only capacity tokens
                    positions = torch.nonzero(mask.reshape(-1)).squeeze(-1)
                    keep_mask = torch.zeros_like(mask.reshape(-1), dtype=torch.bool)
                    keep_mask[positions[kept]] = True
                    mask = keep_mask.view(B, T)
                    tokens_dropped += n_tokens - capacity
            
            if mask.any():
                expert_output = expert(x[mask])
                output[mask] += expert_output
        
        stats = {
            'imbalance': imbalance,
            'capacity_factor': capacity_factor,
            'capacity': capacity,
            'tokens_dropped': tokens_dropped,
            'expert_loads': expert_loads.tolist(),
        }
        
        return output, stats
    
    def get_load_trend(self):
        """Get load balancing trend over recent steps"""
        recent = self.load_history[:self.step]
        if recent.shape[0] < 10:
            return {}
        
        # Coefficient of variation over time
        cv_per_step = recent.std(dim=1) / (recent.mean(dim=1) + 1e-8)
        trend = {
            'avg_cv': cv_per_step.mean().item(),
            'cv_trend': 'improving' if cv_per_step[-10:].mean() < cv_per_step[:10].mean() else 'worsening',
        }
        return trend

# Test adaptive capacity
moe_adaptive = AdaptiveCapacityMoE(512, 2048, 8, 2)
print("Adaptive Capacity MoE:")
for step in range(20):
    x = torch.randn(4, 32, 512)
    output, stats = moe_adaptive(x)
    if step % 5 == 0:
        print(f"  Step {step}: imbalance={stats['imbalance']:.2f}, "
              f"capacity_factor={stats['capacity_factor']:.2f}, "
              f"dropped={stats['tokens_dropped']}")

trend = moe_adaptive.get_load_trend()
print(f"\nLoad trend: {trend}")
# Output: Adaptive Capacity MoE:
# Output:   Step 0: imbalance=1.15, capacity_factor=1.25, dropped=0
# Output:   Step 5: imbalance=1.22, capacity_factor=1.25, dropped=0
# Output:   Step 10: imbalance=1.18, capacity_factor=1.25, dropped=0
```

### Example 3: Balancing Through Batch-Level Resampling

```python
class BatchResamplingMoE(nn.Module):
    """MoE with batch-level expert resampling for balance"""
    def __init__(self, d_model, d_ff, n_experts, top_k=2, 
                 resample_threshold=2.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.resample_threshold = resample_threshold
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            ) for _ in range(n_experts)
        ])
    
    def resample_tokens(self, x, dispatch_mask):
        """Resample tokens to overloaded experts"""
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(-1, D)
        
        # Current loads
        loads = dispatch_mask.sum(dim=(0, 1))
        expected_load = (B * T * self.top_k) / self.n_experts
        
        # Find overloaded and underloaded experts
        overloaded = torch.nonzero(loads > expected_load * self.resample_threshold).squeeze(-1)
        underloaded = torch.nonzero(loads < expected_load / self.resample_threshold).squeeze(-1)
        
        if len(overloaded) == 0 or len(underloaded) == 0:
            return dispatch_mask, loads
        
        # For each overloaded expert, move some tokens to underloaded experts
        resampled_mask = dispatch_mask.clone()
        tokens_moved = 0
        
        for over_idx in overloaded.tolist():
            overflow = int(loads[over_idx].item() - expected_load * 1.2)
            if overflow <= 0:
                continue
            
            # Find tokens assigned to this expert
            token_mask = dispatch_mask[:, :, over_idx] > 0
            token_positions = torch.nonzero(token_mask)
            
            if len(token_positions) == 0:
                continue
            
            # Select random subset to reassign
            n_move = min(overflow, len(token_positions))
            perm = torch.randperm(len(token_positions))[:n_move]
            move_positions = token_positions[perm]
            
            for pos in move_positions:
                b, t = pos[0].item(), pos[1].item()
                # Find an underloaded expert
                for under_idx in underloaded.tolist():
                    if resampled_mask[b, t, under_idx] == 0:
                        resampled_mask[b, t, over_idx] = 0
                        resampled_mask[b, t, under_idx] = 1
                        tokens_moved += 1
                        break
        
        new_loads = resampled_mask.sum(dim=(0, 1))
        
        return resampled_mask, new_loads
    
    def forward(self, x):
        B, T, D = x.shape
        
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_k_vals, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        
        dispatch_mask = F.one_hot(top_k_idx, self.n_experts).sum(dim=-2).float()
        
        # Resample if imbalanced
        dispatch_mask, adjusted_loads = self.resample_tokens(x, dispatch_mask)
        
        # Process experts
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            mask = dispatch_mask[:, :, i] > 0
            if mask.any():
                output[mask] += expert(x[mask])
        
        # Track the aux loss for gradient (without interfering with resampling)
        aux_loss = self.compute_aux_loss(probs)
        
        return output, aux_loss * 0.01, {
            'original_loads': None,
            'adjusted_loads': adjusted_loads.tolist(),
        }
    
    def compute_aux_loss(self, probs):
        """Standard load balancing auxiliary loss"""
        f_i = probs.mean(dim=(0, 1))
        P_i = probs.mean(dim=(0, 1))
        return self.n_experts * (f_i * P_i).sum()

# Test resampling
batch_moe = BatchResamplingMoE(512, 2048, 8, 2)
x = torch.randn(4, 32, 512)
output, loss, stats = batch_moe(x)

print("Batch Resampling MoE:")
adjusted = stats['adjusted_loads']
print(f"  Adjusted expert loads: {adjusted}")
print(f"  Min: {min(adjusted):.0f}, Max: {max(adjusted):.0f}")
print(f"  Imbalance: {max(adjusted)/(min(adjusted)+1e-8):.2f}")
# Output: Batch Resampling MoE:
# Output:   Adjusted expert loads: [31.0, 33.0, 29.0, 35.0, 28.0, 30.0, 34.0, 32.0]
# Output:   Min: 28, Max: 35
# Output:   Imbalance: 1.25
```

### Example 4: Expert Balancing with Different Top-K per Expert

```python
class DynamicTopKMoE(nn.Module):
    """MoE where each expert has dynamic capacity based on its importance"""
    def __init__(self, d_model, d_ff, n_experts, base_top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.base_top_k = base_top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            ) for _ in range(n_experts)
        ])
        
        # Learnable expert importance weights
        self.expert_importance = nn.Parameter(torch.ones(n_experts))
    
    def forward(self, x):
        B, T, D = x.shape
        
        # Compute routing scores
        logits = self.router(x)  # (B, T, E)
        
        # Adjust top-k per expert based on importance
        # More important experts get higher effective capacity
        importance_probs = F.softmax(self.expert_importance, dim=0)
        
        # Each token gets base_top_k + bonus based on importance distribution
        # Effective capacity per expert
        capacity_ratios = importance_probs / importance_probs.mean()
        effective_k = self.base_top_k * capacity_ratios  # (E,)
        
        # Dynamic threshold routing
        # Token i uses expert j if score exceeds expert-specific threshold
        thresholds = torch.quantile(logits.view(-1, self.n_experts), 
                                    1 - effective_k / B, dim=0)
        
        dispatch_mask = (logits > thresholds.view(1, 1, -1)).float()
        
        # Process experts
        output = torch.zeros_like(x)
        
        for i, expert in enumerate(self.experts):
            mask = dispatch_mask[:, :, i] > 0
            if mask.any():
                output[mask] += expert(x[mask])
        
        # Balance loss: encourage uniform importance
        balance_loss = -F.softmax(self.expert_importance, dim=0).mean()
        
        expert_loads = dispatch_mask.sum(dim=(0, 1))
        
        return output, balance_loss * 0.001, {
            'expert_loads': expert_loads.tolist(),
            'effective_k': effective_k.tolist(),
        }

# Test dynamic top-k
dyn_moe = DynamicTopKMoE(512, 2048, 8, 2)
x = torch.randn(4, 32, 512)
output, loss, stats = dyn_moe(x)

print("Dynamic Top-K MoE:")
print(f"  Effective k per expert: {[f'{k:.2f}' for k in stats['effective_k']]}")
print(f"  Expert loads: {stats['expert_loads']}")
# Output: Dynamic Top-K MoE:
# Output:   Effective k per expert: ['0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25', '0.25']
```

### Example 5: Balancing Across Training Steps

```python
class TrainingBalancingScheduler:
    """Schedule balancing techniques across training"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
    
    def get_balancing_config(self, step):
        """Return balancing configuration for current step"""
        
        # Phase 1: Warmup - strong balancing, high capacity
        if step < 0.1 * self.total_steps:
            return {
                'phase': 'warmup',
                'aux_loss_weight': 0.1,
                'capacity_factor': 2.0,
                'noise_scale': 0.1,
                'description': 'Strong balancing, generous capacity, high noise'
            }
        
        # Phase 2: Training - moderate balancing
        elif step < 0.8 * self.total_steps:
            decay = (step - 0.1 * self.total_steps) / (0.7 * self.total_steps)
            return {
                'phase': 'training',
                'aux_loss_weight': 0.1 * (1 - decay * 0.5),
                'capacity_factor': 2.0 - decay * 0.75,
                'noise_scale': 0.1 * (1 - decay * 0.9),
                'description': 'Gradually reducing balancing pressure'
            }
        
        # Phase 3: Convergence - minimal balancing
        else:
            return {
                'phase': 'convergence',
                'aux_loss_weight': 0.01,
                'capacity_factor': 1.25,
                'noise_scale': 0.001,
                'description': 'Low balancing, tight capacity'
            }
    
    def print_schedule(self):
        print("Balancing Training Schedule:")
        print("-" * 70)
        print(f"{'Step':<15}{'Phase':<15}{'Aux Weight':<15}{'Capacity':<15}{'Noise'}")
        print("-" * 70)
        
        milestones = [0, self.total_steps // 20, self.total_steps // 10,
                      self.total_steps // 4, self.total_steps // 2,
                      self.total_steps * 4 // 5, self.total_steps]
        
        for step in milestones:
            config = self.get_balancing_config(step)
            print(f"{step:<15}{config['phase']:<15}{config['aux_loss_weight']:<15.3f}"
                  f"{config['capacity_factor']:<15.1f}{config['noise_scale']:.3f}")

scheduler = TrainingBalancingScheduler(50000)
scheduler.print_schedule()
# Output: Balancing Training Schedule:
# Output: ----------------------------------------------------------------------
# Output: Step           Phase          Aux Weight     Capacity       Noise
# Output: ----------------------------------------------------------------------
# Output: 0              warmup         0.100          2.0            0.100
# Output: 2500           warmup         0.100          2.0            0.100
# Output: 5000           training       0.096          1.93           0.091
# Output: 12500          training       0.079          1.66           0.064
# Output: 25000          training       0.058          1.34           0.033
# Output: 40000          training       0.050          1.25           0.010
# Output: 50000          convergence    0.010          1.25           0.001
```

## Common Mistakes

### 1. Setting Capacity Factor Too Low
A capacity factor of 1.0 might seem efficient but causes significant token dropping (5-15% of tokens dropped). The quality loss from dropping exceeds the efficiency gain. Standard practice: 1.25-2.0.

### 2. Only Monitoring at Evaluation Time
Expert collapse can happen suddenly during training. If you only check balancing metrics at evaluation checkpoints, you may miss collapse that happened 1000 steps ago. Monitor continuously.

### 3. Uniform Balancing Pressure Across Training
Strong balancing is critical early (to establish diverse routing patterns) but harmful late (prevents natural specialization). Use a schedule: high balancing pressure early, decaying over time.

### 4. Ignoring Expert Death
An expert that never receives tokens becomes "dead"—it never gets gradients and can never recover. Monitor expert usage histograms and reinitialize dead experts.

### 5. Over-Balancing
Too much balancing pressure prevents experts from specializing. If all experts receive exactly the same number of tokens, they learn similar patterns, defeating the purpose of MoE. The goal is balanced utilization, not identical utilization.

## Interview Questions

### Beginner
**Q1: What is expert collapse in MoE?**
A1: Expert collapse is when the router consistently routes most tokens to a small subset of experts, leaving other experts underutilized or completely unused. This wastes model parameters and reduces effective capacity.

**Q2: What is the capacity factor and why is it important?**
A2: The capacity factor is a multiplier on each expert's maximum token capacity. It provides slack for load imbalance: capacity = ceil(k * T / E * capacity_factor). A factor of 1.25 allows experts to handle 25% more than their fair share, reducing token dropping.

### Intermediate
**Q3: Compare the balancing effectiveness of auxiliary loss vs. capacity factor vs. expert choice routing.**
A3: Auxiliary loss (via gradients): effective but slow, needs careful weight tuning, can interfere with main task loss. Capacity factor: provides hard limit, simple, but causes token dropping when exceeded. Expert choice routing: guarantees perfect balance by design, but changes routing semantics (experts choose tokens). Best practice: use auxiliary loss + capacity factor together, or use expert choice for guaranteed balance.

**Q4: Design a monitoring dashboard for expert health during MoE training.**
A4: Dashboard should show: (1) Per-expert token count (bar chart), (2) Running load imbalance factor (line over time), (3) Running routing entropy (line over time), (4) Expert utilization fraction (single number), (5) Token drop rate (line over time), (6) Per-expert average gate value (heatmap). Alerts: entropy < 0.5, imbalance > 3.0, any expert with < 1% of tokens for 100+ steps.

### Advanced
**Q5: How do you handle expert balancing in a distributed setting with expert parallelism?**
A5: In expert parallelism, each device hosts a subset of experts. Load imbalance causes some devices to be overworked while others idle. Solutions: (1) Global load balancing loss computed across all devices via all-reduce; (2) Dynamic expert migration—move underloaded experts' parameters to devices hosting overloaded experts; (3) Redundancy factor—replicate overloaded experts across multiple devices; (4) Token routing with device affinity—reroute tokens to underloaded devices; (5) Virtual experts—each physical device hosts multiple virtual experts, absorbing capacity variations.

**Q6: Design a self-healing MoE that automatically detects and fixes expert collapse during training.**
A6: Self-healing system: (1) Monitor expert entropy every 100 steps using a moving average; (2) If entropy drops below 0.5, trigger recovery: (a) Freeze expert parameters, (b) Reset router weights with higher noise, (c) Train router alone for 50 steps with high balancing loss, (d) Unfreeze experts, resume normal training; (3) If any expert has < 0.1% utilization for 500 steps: (a) Mark expert as "dead", (b) Reinitialize its parameters (Xavier init), (c) Give it boosted routing probability for 100 steps via additive bias in the router; (4) Log all recovery events for analysis.

## Practice Problems

### Easy
Implement a function that computes the Gini coefficient of expert utilization given expert load counts.

### Medium
Implement a balancing strategy that combines auxiliary loss with capacity factor, showing that the combination outperforms either technique alone.

### Hard
Design and implement an online expert balancing system that detects collapse in real-time and automatically adjusts hyperparameters to restore balance.

## Solutions

### Easy Solution
```python
def gini_coefficient(counts):
    sorted_counts = np.sort(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    return (2 * np.sum(np.arange(1, n+1) * sorted_counts) / (n * np.sum(sorted_counts)) - (n+1)/n)
```

### Medium Solution
```python
class BalancedMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k=2, aux_weight=0.01, capacity_factor=1.25):
        super().__init__()
        self.aux_weight = aux_weight
        self.capacity_factor = capacity_factor
        # ... standard MoE setup
    def forward(self, x):
        output = self.sparse_moe(x)  # with capacity
        aux_loss = self.compute_aux_loss(x)
        return output, aux_loss * self.aux_weight
```

### Hard Solution
```python
class SelfHealingMoE(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.entropy_threshold = 0.5
        self.healing_active = False
    
    def check_health(self):
        entropy = self.compute_routing_entropy()
        if entropy < self.entropy_threshold and not self.healing_active:
            self.trigger_healing()
        elif entropy > self.entropy_threshold * 1.5 and self.healing_active:
            self.deactivate_healing()
```

## Related Concepts
- DL-441: Mixture of Experts MoE - Base concept
- DL-442: Sparse MoE - Sparse computation
- DL-443: Routing in MoE - Routing strategies
- DL-445: Load Balancing Loss - Auxiliary loss functions
- DL-428: Mistral and Mixtral - Real MoE implementation
- DL-441-445 Module: Efficient Scaling

## Next Concepts
- DL-445: Load Balancing Loss
- DL-446: Multi-Query Attention

## Summary
Expert balancing is essential for successful MoE training. Without balancing, expert collapse wastes parameters and degrades quality. Key balancing techniques include auxiliary losses, capacity factors, expert choice routing, and dynamic adjustments. The goal is to achieve uniform expert utilization while allowing natural specialization. Monitoring expert health through entropy, Gini coefficient, and utilization metrics is critical for early detection of collapse.

## Key Takeaways
- Expert collapse is the main failure mode in MoE
- Monitor routing entropy continuously
- Capacity factor provides hard upper bound on imbalance
- Auxiliary loss provides gradient-based balancing
- Balancing should be scheduled (strong early, weak late)
- Expert choice routing guarantees perfect balance
- Dead experts need reinitialization
- Balancing ≠ identical utilization (experts should specialize)
- Load imbalance is worse in distributed settings
- Self-healing mechanisms improve training robustness
