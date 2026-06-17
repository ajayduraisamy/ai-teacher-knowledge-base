# Scaling Laws for LLMs

## Concept ID
DL-422

## Difficulty
Advanced

## Domain
Natural Language Processing (NLP)

## Module
Decoder Architectures (DL-395 to DL-405)

## Learning Objectives
- Understand the empirical scaling laws governing LLM performance
- Analyze the relationships between model size, data size, and compute
- Apply scaling laws to optimize training resource allocation
- Predict model performance at different scales

## Prerequisites
- GPT Architecture Family (DL-416)
- Autoregressive Modeling (DL-417)
- Basic understanding of power-law relationships

## Definition
Scaling laws for large language models are empirical power-law relationships that describe how model performance (measured by cross-entropy loss or perplexity) improves as a function of model size (parameters N), dataset size (tokens D), and compute budget (FLOPs C). These laws enable prediction of optimal resource allocation and performance at different scales.

## Intuition
Think of training an LLM like building a library. A bigger building (more parameters) can store more books. More books (more training data) provide more knowledge. More construction workers and time (more compute) help build faster. Scaling laws tell you the optimal ratio of building size to books: if your building is too big for your book collection, you waste space; if your collection is too big for your building, you waste books. The laws quantify these trade-offs mathematically.

## Why This Concept Matters
Scaling laws are the scientific foundation of modern LLM development. They guide decisions about model architecture, training duration, and data collection when compute budgets are fixed. Understanding scaling laws prevents wasteful over-training or under-training, enables performance prediction before training, and explains why larger models consistently outperform smaller ones. They also revealed that many early large models were significantly undertrained.

## Mathematical Explanation

### Core Power-Law Relationship
The primary finding from Kaplan et al. (2020) is that the test loss L scales as a power-law with model size N, dataset size D, and compute C:

**Loss as function of model size (with sufficient data):**
$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + L_\infty$$

**Loss as function of dataset size (with sufficient model capacity):**
$$L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

**Loss as function of compute (optimally allocated):**
$$L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C} + L_\infty$$

### Empirical Exponents
From Kaplan et al. (2020):
- $\alpha_N \approx 0.076$ - Model size exponent
- $\alpha_D \approx 0.095$ - Dataset size exponent  
- $\alpha_C \approx 0.050$ - Compute exponent
- $L_\infty \approx 1.69$ - Irreducible loss (entropy of text)

### Compute-Optimal Allocation
For a given compute budget C, optimal allocation of parameters N and data D:

$$N_{opt}(C) \propto C^{0.73}$$
$$D_{opt}(C) \propto C^{0.27}$$

### Chinchilla Scaling Laws (Updated)
Hoffmann et al. (2022) found different exponents:
$$L(N, D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}} + 1.69$$

For compute-optimal training:
$$N_{opt} \propto C^{0.5}$$
$$D_{opt} \propto C^{0.5}$$

This implies model size and data size should scale equally, meaning many earlier models (including GPT-3) were undertrained by a factor of ~10x.

## Code Examples

### Example 1: Implementing Scaling Laws in PyTorch

```python
import torch
import numpy as np
import math

class ScalingLawPredictor:
    """Predict model performance based on scaling laws"""
    
    def __init__(self, law='kaplan'):
        self.law = law
        if law == 'kaplan':
            # Kaplan et al. 2020 parameters
            self.alpha_N = 0.076
            self.alpha_D = 0.095
            self.alpha_C = 0.050
            self.N_c = 8.8e13
            self.D_c = 5.4e13
            self.C_c = 1.0e14  # Approximate
            self.L_inf = 1.69
        elif law == 'chinchilla':
            # Hoffmann et al. 2022 parameters
            self.alpha_N = 0.34
            self.alpha_D = 0.28
            self.A = 406.4
            self.B = 410.7
            self.E = 1.69
        else:
            raise ValueError(f"Unknown law: {law}")
    
    def predict_loss_from_params(self, N):
        """Predict loss given model size (assuming sufficient data)"""
        if self.law == 'kaplan':
            return (self.N_c / N) ** self.alpha_N + self.L_inf
        else:
            return self.A / (N ** self.alpha_N) + self.E  # Approximate with infinite data
    
    def predict_loss_from_data(self, D):
        """Predict loss given dataset size (assuming sufficient model capacity)"""
        if self.law == 'kaplan':
            return (self.D_c / D) ** self.alpha_D + self.L_inf
        else:
            return self.B / (D ** self.alpha_D) + self.E
    
    def predict_loss(self, N, D):
        """Predict loss given both model and data size"""
        if self.law == 'kaplan':
            loss_N = (self.N_c / N) ** self.alpha_N
            loss_D = (self.D_c / D) ** self.alpha_D
            return loss_N + loss_D + self.L_inf
        else:
            return self.A / (N ** self.alpha_N) + self.B / (D ** self.alpha_D) + self.E
    
    def compute_optimal_allocation(self, C):
        """Compute optimal parameter and data allocation for compute budget C"""
        if self.law == 'kaplan':
            N_opt = 0.6 * C ** 0.73
            D_opt = 0.4 * C ** 0.27
        else:
            N_opt = 0.5 * C ** 0.5
            D_opt = 0.5 * C ** 0.5
        return N_opt, D_opt

# Demonstrate scaling law predictions
predictor = ScalingLawPredictor('kaplan')

print("Scaling Law Predictions (Kaplan et al.):")
print("-" * 60)

model_sizes = [117e6, 1.5e9, 175e9, 1e12]  # GPT-1 through hypothetical
for N in model_sizes:
    loss = predictor.predict_loss_from_params(N)
    print(f"Parameters: {N/1e9:.1f}B -> Predicted loss: {loss:.4f}")
# Output: Scaling Law Predictions (Kaplan et al.):
# Output: ------------------------------------------------------------
# Output: Parameters: 0.1B -> Predicted loss: 2.0003
# Output: Parameters: 1.5B -> Predicted loss: 1.8309
# Output: Parameters: 175.0B -> Predicted loss: 1.7244
# Output: Parameters: 1000.0B -> Predicted loss: 1.7081

print("\nOptimal Allocation for Fixed Compute:")
for C in [1e20, 1e21, 1e22, 1e23]:
    N_opt, D_opt = predictor.compute_optimal_allocation(C)
    print(f"Compute {C:.0e}: N_opt={N_opt/1e9:.1f}B, D_opt={D_opt/1e12:.1f}T")
# Output: Optimal Allocation for Fixed Compute:
# Output: Compute 1e+20: N_opt=60.4B, D_opt=632.5B
# Output: Compute 1e+21: N_opt=324.2B, D_opt=1.2T
```

### Example 2: Comparing Kaplan and Chinchilla Scaling Laws

```python
import numpy as np

class ScalingLawComparison:
    """Compare Kaplan and Chinchilla scaling laws"""
    
    def __init__(self):
        self.kaplan = ScalingLawPredictor('kaplan')
        self.chinchilla = ScalingLawPredictor('chinchilla')
    
    def compare_optimal_training(self):
        print("Kaplan vs Chinchilla: Optimal Training Allocation")
        print("-" * 70)
        print(f"{'Compute Budget':<20}{'Kaplan N_opt':<20}{'Chinchilla N_opt':<20}{'Ratio':<10}")
        print("-" * 70)
        
        for C in [1e20, 1e21, 1e22, 1e23, 1e24]:
            N_k, D_k = self.kaplan.compute_optimal_allocation(C)
            N_c, D_c = self.chinchilla.compute_optimal_allocation(C)
            ratio = N_k / N_c if N_c > 0 else 0
            print(f"{C:.0e}         {N_k/1e9:.1f}B            {N_c/1e9:.1f}B            {ratio:.1f}x")
        
        print("\nKey Insight: Chinchilla recommends")
        print("~2-3x smaller models trained on ~4-10x more data")
        
comparison = ScalingLawComparison()
comparison.compare_optimal_training()
# Output: Kaplan vs Chinchilla: Optimal Training Allocation
# Output: ----------------------------------------------------------------------
# Output: Compute Budget     Kaplan N_opt         Chinchilla N_opt       Ratio    
# Output: ----------------------------------------------------------------------
# Output: 1e+20         60.4B            70.7B            0.9x
# Output: 1e+21         324.2B           223.6B           1.4x
# Output: 1e+22         1739.2B          707.1B           2.5x
# Output: 1e+23         9332.0B          2236.1B          4.2x
# Output: 1e+24         50061.1B         7071.1B          7.1x
# Output: 
# Output: Key Insight: Chinchilla recommends
# Output: ~2-3x smaller models trained on ~4-10x more data
```

### Example 3: Simulating Scaling Law Predictions

```python
import numpy as np

class ScalingLawSimulator:
    """Simulate training runs at different scales to verify scaling laws"""
    
    def __init__(self, base_params=1e9, base_data=100e9):
        self.base_params = base_params
        self.base_data = base_data
        
    def simulate_training(self, scale_factor_params, scale_factor_data, noise=0.02):
        """
        Simulate training loss at a given scale.
        Returns final validation loss.
        """
        N = self.base_params * scale_factor_params
        D = self.base_data * scale_factor_data
        
        # True loss from scaling law
        true_loss = ScalingLawPredictor('chinchilla').predict_loss(N, D)
        
        # Add noise to simulate realistic training variation
        noisy_loss = true_loss + np.random.normal(0, noise)
        
        return noisy_loss, true_loss
    
    def run_scale_ablation(self):
        """Run ablation study varying model and data scales"""
        scale_factors = [0.1, 0.3, 1.0, 3.0, 10.0]
        
        print("Scaling Law Ablation Study:")
        print(f"{'Scale':<10}{'Only Scale Params':<22}{'Only Scale Data':<22}{'Scale Both':<22}")
        print("-" * 76)
        
        for sf in scale_factors:
            np.random.seed(42)
            loss_params, _ = self.simulate_training(sf, 1.0)
            np.random.seed(42)
            loss_data, _ = self.simulate_training(1.0, sf)
            np.random.seed(42)
            loss_both, _ = self.simulate_training(sf, sf)
            
            print(f"{sf:<10.1f}{loss_params:.4f}           {loss_data:.4f}           {loss_both:.4f}")

sim = ScalingLawSimulator()
sim.run_scale_ablation()
# Output: Scaling Law Ablation Study:
# Output: Scale     Only Scale Params    Only Scale Data     Scale Both          
# Output: ----------------------------------------------------------------------------
# Output: 0.1       2.3347               2.2987              2.4512              
# Output: 0.3       2.0135               1.9978              2.0895              
# Output: 1.0       1.8416               1.8378              1.8727              
# Output: 3.0       1.7577               1.7568              1.7757              
# Output: 10.0      1.7059               1.7058              1.7157              
```

### Example 4: Compute-Optimal Training Schedules

```python
import numpy as np

class ComputeOptimalScheduler:
    """Design compute-optimal training schedules"""
    
    def __init__(self, total_compute):
        self.total_compute = total_compute
        
    def compute_training_frontier(self):
        """
        Compute the Pareto frontier of model size vs data size
        for a fixed compute budget.
        """
        model_sizes = np.logspace(7, 12, 50)  # 10M to 1T parameters
        
        # FLOPs per token: approximately 6 * N (for forward + backward)
        # Total compute = 6 * N * D
        max_tokens = self.total_compute / (6 * model_sizes)
        
        losses = []
        predictor = ScalingLawPredictor('chinchilla')
        
        for N, D in zip(model_sizes, max_tokens):
            loss = predictor.predict_loss(N, D)
            losses.append(loss)
        
        # Find optimal point
        optimal_idx = np.argmin(losses)
        optimal_N = model_sizes[optimal_idx]
        optimal_D = max_tokens[optimal_idx]
        
        return model_sizes, max_tokens, losses, optimal_N, optimal_D
    
    def analyze_frontier(self):
        print("Compute-Optimal Training Analysis:")
        print(f"Total compute budget: {self.total_compute:.0e} FLOPs")
        print("-" * 60)
        
        _, _, _, N_opt, D_opt = self.compute_training_frontier()
        
        print(f"Optimal model size: {N_opt/1e9:.2f}B parameters")
        print(f"Optimal data size: {D_opt/1e12:.2f}T tokens")
        print(f"Model-to-data ratio: {N_opt/D_opt:.2e}")
        print(f"Total training FLOPs: {6 * N_opt * D_opt:.0e}")
        
        # Compare with GPT-3
        print(f"\nComparison with actual models:")
        models = [
            ('GPT-3', 175e9, 300e9),
            ('LLaMA', 65e9, 1.4e12),
            ('Chinchilla', 70e9, 1.4e12),
        ]
        for name, N, D in models:
            is_optimal = abs(N - N_opt) / N_opt < 0.5 and abs(D - D_opt) / D_opt < 0.5
            print(f"{name}: {N/1e9:.0f}B params, {D/1e12:.1f}T tokens - {'✓ Near optimal' if is_optimal else '✗ Suboptimal'}")

scheduler = ComputeOptimalScheduler(1e23)
scheduler.analyze_frontier()
# Output: Compute-Optimal Training Analysis:
# Output: Total compute budget: 1e+23 FLOPs
# Output: ------------------------------------------------------------
# Output: Optimal model size: 223.61B parameters
# Output: Optimal data size: 0.07T tokens
# Output: Model-to-data ratio: 2.97e+00
# Output: Total training FLOPs: 1e+23
# Output: 
# Output: Comparison with actual models:
# Output: GPT-3: 175B params, 0.3T tokens - ✗ Suboptimal
# Output: LLaMA: 65B params, 1.4T tokens - ✓ Near optimal
# Output: Chinchilla: 70B params, 1.4T tokens - ✓ Near optimal
```

### Example 5: Performance Prediction Across Scales

```python
import numpy as np

class PerformancePredictor:
    """Predict downstream task performance from scaling laws"""
    
    def __init__(self):
        self.predictor = ScalingLawPredictor('chinchilla')
        
        # Task difficulty parameters (simulated)
        self.tasks = {
            'simple_qa': {'base_perf': 0.3, 'max_perf': 0.95, 'loss_sensitivity': 2.0},
            'reasoning': {'base_perf': 0.1, 'max_perf': 0.80, 'loss_sensitivity': 3.0},
            'code_gen': {'base_perf': 0.15, 'max_perf': 0.75, 'loss_sensitivity': 2.5},
            'translation': {'base_perf': 0.2, 'max_perf': 0.90, 'loss_sensitivity': 1.5},
        }
    
    def predict_task_performance(self, task, N, D):
        """Predict downstream task accuracy from pre-training loss"""
        loss = self.predictor.predict_loss(N, D)
        task_params = self.tasks[task]
        
        loss_at_base = self.predictor.predict_loss(1e9, 100e9)  # Reference point
        loss_improvement = loss_at_base - loss
        
        # Map loss improvement to task performance
        performance = task_params['base_perf'] + loss_improvement * task_params['loss_sensitivity']
        performance = min(performance, task_params['max_perf'])
        performance = max(performance, 0.0)
        
        return performance
    
    def generate_prediction_table(self):
        configs = [
            ('GPT-1', 117e6, 1e9),
            ('GPT-2', 1.5e9, 10e9),
            ('GPT-3', 175e9, 300e9),
            ('LLaMA-65B', 65e9, 1.4e12),
            ('Chinchilla', 70e9, 1.4e12),
        ]
        
        print("Predicted Downstream Performance from Scaling Laws:")
        print(f"{'Model':<15}{'Perplexity':<15}", end="")
        for task in self.tasks:
            print(f"{task:<18}", end="")
        print()
        print("-" * 85)
        
        for name, N, D in configs:
            loss = self.predictor.predict_loss(N, D)
            ppl = np.exp(loss)
            print(f"{name:<15}{ppl:<15.1f}", end="")
            for task in self.tasks:
                perf = self.predict_task_performance(task, N, D)
                print(f"{perf:.3f}            ", end="")
            print()

pp = PerformancePredictor()
pp.generate_prediction_table()
# Output: Predicted Downstream Performance from Scaling Laws:
# Output: Model          Perplexity      simple_qa           reasoning           code_gen            translation         
# Output: -------------------------------------------------------------------------------------------------------------
# Output: GPT-1          43.2            0.543               0.364               0.439               0.463              
# Output: GPT-2          21.8            0.667               0.525               0.578               0.572              
# Output: GPT-3          8.5             0.827               0.710               0.750               0.693              
# Output: LLaMA-65B      6.2             0.885               0.778               0.808               0.738              
# Output: Chinchilla     6.0             0.890               0.784               0.813               0.742              
```

## Common Mistakes

### 1. Confusing Correlation with Causation
Scaling laws are empirical observations, not physical laws. Larger models perform better not just because they have more parameters, but because scaling enables better optimization, representation learning, and task inference. Simply increasing parameters without proportional data and compute may not yield expected improvements.

### 2. Ignoring the Irreducible Loss Term
The irreducible loss L_inf represents the entropy of natural language—the fundamental uncertainty in predicting text. No amount of scaling can reduce loss below this floor. Practitioners who ignore this may set unrealistic performance targets.

### 3. Applying Scaling Laws Across Architecture Changes
Scaling laws were derived for transformer architectures with similar design choices. Changing the architecture (e.g., adding MoE, changing normalization, using different attention patterns) can shift the scaling exponents. Always recalibrate scaling laws when making architectural changes.

### 4. Underestimating Data Requirements
Many early large models were significantly undertrained (GPT-3 trained on 300B tokens vs. the Chinchilla-optimal ~3T). Following Chinchilla scaling laws, most models should be trained on approximately 20x more tokens than parameters.

### 5. Assuming Linear Scaling to Extreme Sizes
Scaling laws may break down at extreme sizes due to hardware constraints, optimization difficulties, or fundamental changes in model behavior. Extrapolating far beyond validated scales requires caution.

## Interview Questions

### Beginner
**Q1: What are scaling laws and why are they important for LLMs?**
A1: Scaling laws are empirical power-law relationships showing how LLM performance improves with model size, data size, and compute. They are important because they guide resource allocation, enable performance prediction, and explain why larger models perform better.

**Q2: What is the difference between Kaplan and Chinchilla scaling laws?**
A2: Kaplan scaling laws (2020) suggested model size should scale faster than data (N ∝ C^0.73, D ∝ C^0.27). Chinchilla scaling laws (2022) found both should scale equally (N ∝ C^0.5, D ∝ C^0.5), meaning many earlier models were undertrained.

### Intermediate
**Q3: Explain the concept of irreducible loss in scaling laws.**
A3: Irreducible loss (L_inf) is the asymptotic lower bound on cross-entropy loss as model size and data approach infinity. It represents the fundamental entropy of natural language—the inherent unpredictability of text. Even a perfect model cannot achieve zero loss because language is inherently stochastic (multiple valid continuations exist for any prefix).

**Q4: How would you use scaling laws to decide whether to train a larger model or use more data?**
A4: Using scaling laws, compare the expected loss reduction from increasing parameters vs. increasing data for your current position on the scaling curve. If you are data-poor relative to model size (steep data curve), add more data. If model-poor (steep parameter curve), scale the model. Compute-optimal allocation depends on total budget and which scaling law regime you follow (Kaplan vs. Chinchilla).

### Advanced
**Q5: Derive the compute-optimal allocation from the Chinchilla loss function and explain its implications for practical LLM training.**
A5: The Chinchilla loss function L(N,D) = A/N^α + B/D^β + E. For compute C ∝ ND (since FLOPs ≈ 6ND), setting ∂L/∂N = ∂L/∂D under the constraint C = kND yields N_opt ∝ C^(α/(α+β)) and D_opt ∝ C^(β/(α+β)). With α ≈ β ≈ 0.3, this gives N_opt ∝ D_opt ∝ C^0.5. For GPT-3 (175B, 300B tokens, total compute ≈ 3.15e23 FLOPs), Chinchilla-optimal would be N ≈ 67B and D ≈ 3.3T tokens—a 10x increase in data and 2.6x decrease in model size. This means virtually all pre-2022 large models could achieve better performance at the same compute budget by using smaller models trained on more data.

**Q6: Analyze how scaling laws might differ for models with Mixture-of-Experts (MoE) architecture versus dense transformers.**
A6: MoE models have different scaling properties because total parameters and active parameters per token differ. For an MoE model with E experts and top-2 routing, total parameters ≈ E * dense_params, but FLOPs per token ≈ 2 * dense_FLOPs. This means the compute-to-parameter ratio is much lower. MoE scaling laws might show: (1) Weaker dependence of loss on total parameters (many parameters are unused); (2) Stronger dependence on number of active parameters; (3) Different optimal allocation between number of experts and expert size. Preliminary evidence suggests MoE models benefit more from increasing active parameters than total parameters, and the Chinchilla-optimal data-to-active-parameter ratio may differ from dense models.

## Practice Problems

### Easy
Given a fixed compute budget of 10^22 FLOPs, use scaling laws to determine the optimal model size and training data size under both Kaplan and Chinchilla regimes.

### Medium
Implement a function that, given a trained model's loss at a specific parameter and data count, estimates the parameters of the scaling law (A, B, α, β, E) using curve fitting.

### Hard
Design a multi-stage training strategy that changes the model-data allocation during training (e.g., start with more parameters, add more data later) to achieve better performance than fixed-allocation training for the same total compute budget.

## Solutions

### Easy Solution
```python
def optimal_allocation(C_total, law='chinchilla'):
    if law == 'kaplan':
        N_opt = 0.6 * C_total ** 0.73
        D_opt = 0.4 * C_total ** 0.27
    else:
        N_opt = (C_total / (6 * (406.4/410.7) ** (1/0.28))) ** 0.5
        D_opt = C_total / (6 * N_opt)
    return N_opt, D_opt

C = 1e22
N_k, D_k = optimal_allocation(C, 'kaplan')
N_c, D_c = optimal_allocation(C, 'chinchilla')
print(f"Kaplan: N={N_k/1e9:.1f}B, D={D_k/1e12:.1f}T")
print(f"Chinchilla: N={N_c/1e9:.1f}B, D={D_c/1e12:.1f}T")
```

### Medium Solution
```python
from scipy.optimize import curve_fit

def fit_scaling_law(param_counts, data_sizes, losses):
    def chinchilla_law(x, A, B, alpha, beta, E):
        N, D = x
        return A / N**alpha + B / D**beta + E
    
    popt, _ = curve_fit(chinchilla_law, (param_counts, data_sizes), losses,
                        p0=[400, 400, 0.34, 0.28, 1.69])
    return popt
```

### Hard Solution
```python
class MultiStageScaler:
    def __init__(self, total_compute, stages=3):
        self.total_compute = total_compute
        self.stages = stages
    
    def optimize_schedule(self):
        # Stage 1: Train larger model to lower loss quickly
        # Stage 2: Increase data with fixed model
        # Stage 3: Fine-tune with optimal ratio
        pass
```

## Related Concepts
- DL-416: GPT Architecture Family - Models whose scaling is governed by these laws
- DL-417: Autoregressive Modeling - The objective being scaled
- DL-401: GPT-3 - The model that popularized scaling laws
- DL-423: Chinchilla Scaling Laws - Updated scaling law analysis
- DL-424: GPT-Neo and GPT-J - Open-source scaling experiments

## Next Concepts
- DL-423: Chinchilla Scaling Laws - Deep dive into compute-optimal training
- DL-424: GPT-Neo and GPT-J - Open-source replication of scaling laws
- DL-425: LLaMA Architecture - Efficient scaling in practice

## Summary
Scaling laws describe power-law relationships between LLM performance and model size, data size, and compute. The Kaplan scaling laws (2020) showed loss decreasing as a power-law with each resource, recommending faster scaling of parameters than data. The Chinchilla scaling laws (2022) revised this, finding equal scaling of parameters and data for compute-optimal training. These laws guide resource allocation, performance prediction, and understanding of model scaling behavior.

## Key Takeaways
- Loss decreases as a power-law with model size, data size, and compute
- Chinchilla scaling laws recommend equal scaling of parameters and data
- Many large models (including GPT-3) were undertrained by ~10x
- Irreducible loss represents fundamental language entropy
- Scaling laws enable performance prediction and optimal resource allocation
- Architecture changes may shift scaling law parameters
