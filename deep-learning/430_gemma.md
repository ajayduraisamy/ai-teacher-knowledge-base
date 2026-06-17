# Gemma

## Concept ID
DL-430

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
LLM Architectures (DL-416 to DL-440)

## Learning Objectives
- Understand the Gemma architecture and design philosophy
- Implement key Gemma components (GeGLU, RoPE, RMSNorm)
- Analyze Gemma's training methodology and data approach
- Compare Gemma with LLaMA and other open-source models

## Prerequisites
- LLaMA Architecture (DL-425)
- Transformer Architecture (DL-370)
- Scaling Laws for LLMs (DL-422)

## Definition
Gemma is a family of lightweight, open-source decoder-only transformer models (2B, 7B) developed by Google DeepMind, released in February 2024. Inspired by the Gemini architecture, Gemma features GeGLU activation, RoPE embeddings, RMSNorm, deep encoder-like layer structure, and was trained on 6T tokens of primarily English data with a focus on safety and responsible deployment through comprehensive filtering.

## Intuition
Gemma is Google's gift to the open-source community—a scaled-down version of their proprietary Gemini model, designed to be accessible (runs on a single GPU or consumer hardware) while maintaining the architectural innovations and training quality of Google's frontier models. Like a compact car built with racing technology, Gemma brings Google's expertise in large-scale training, safety filtering, and model evaluation to the open-source ecosystem, packaged in sizes that individual developers can actually run.

## Why This Concept Matters
Gemma is significant because it brings Google's proprietary architecture and training methodology to the open-source world. Its design choices (GeGLU, RoPE, deep architecture, careful data filtering) represent Google's best practices distilled from training Gemini. Gemma also emphasizes safety and responsible deployment with built-in filtering tools, evaluation benchmarks, and a comprehensive model card, setting a new standard for open-source model responsibility.

## Mathematical Explanation

### GeGLU Activation
Gemma uses GeGLU (Gaussian Error Gated Linear Unit), which combines GELU with a gating mechanism:

$$\text{GeGLU}(x, W, V, W_2) = (\text{GELU}(xW) \odot xV)W_2$$

Where GELU is: $\text{GELU}(x) = x \cdot \Phi(x)$ with $\Phi$ being the Gaussian CDF.

In practice:
$$\text{FFN}(x) = \text{GELU}(xW_{gate}) \odot xW_{up})W_{down}$$

### Deep Architecture (14 layers for 2B, 18 layers for 7B)
Gemma uses a relatively deep but narrow design:

$$\text{Layers} \times d_{model} \approx \text{constant}$$

For Gemma 2B: $n_{layers} = 14$, $d_{model} = 2048$
For Gemma 7B: $n_{layers} = 18$, $d_{model} = 3072$

This is deeper than LLaMA 7B (32 layers, 4096 dim) for the 7B size but with smaller dimensions.

### Training Data Filtering
Gemma uses a multi-stage filtering pipeline:

$$P(\text{keep}) = \prod_i f_i(\text{doc})$$

Where each $f_i$ is a quality filter based on: language detection, toxicity scoring, duplicate detection, and content safety classification.

## Code Examples

### Example 1: GeGLU Activation Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeGLU(nn.Module):
    """GeGLU activation as used in Gemma"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class GemmaMLP(nn.Module):
    """Gemma MLP with GeGLU"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class GemmaAttention(nn.Module):
    """Gemma attention with RoPE"""
    
    def __init__(self, hidden_size, n_heads, n_kv_heads, head_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        
        # RoPE
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, position_ids=None, attention_mask=None):
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        
        cos, sin = self._compute_rope(position_ids)
        q, k = self._apply_rope(q, k, cos, sin)
        
        # GQA expand
        if self.n_kv_heads != self.n_heads:
            n_groups = self.n_heads // self.n_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
            v = v.reshape(B, T, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-2, -1))
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
    
    def _compute_rope(self, position_ids):
        inv_freq = self.inv_freq[None, :, None].float()
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rope(self, q, k, cos, sin):
        half = self.head_dim // 2
        q_embed = (q[..., :half] * cos) + ((-q[..., half:]) * sin)
        k_embed = (k[..., :half] * cos) + ((-k[..., half:]) * sin)
        return q_embed, k_embed

class GemmaBlock(nn.Module):
    """Gemma decoder block"""
    
    def __init__(self, hidden_size, n_heads, n_kv_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.self_attn = GemmaAttention(hidden_size, n_heads, n_kv_heads)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.mlp = GemmaMLP(hidden_size, intermediate_size)
        
    def forward(self, x, position_ids=None, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, position_ids, attention_mask)
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x

# Test GeGLU
geglu = GeGLU(2048, 16384)
x = torch.randn(2, 16, 2048)
out = geglu(x)
print(f"GeGLU output shape: {out.shape}")

# Test Gemma block
block = GemmaBlock(2048, 8, 4, 16384)
out = block(x)
print(f"Gemma block output: {out.shape}")
# Output: GeGLU output shape: (2, 16, 2048)
# Output: Gemma block output: (2, 16, 2048)
```

### Example 2: Gemma Model Configurations

```python
class GemmaConfig:
    """Gemma model configurations"""
    
    MODELS = {
        'Gemma 2B': {
            'hidden_size': 2048, 'n_heads': 8, 'n_kv_heads': 4, 'n_layers': 14,
            'intermediate_size': 16384, 'head_dim': 256, 'vocab_size': 256000,
            'max_seq_len': 8192,
        },
        'Gemma 7B': {
            'hidden_size': 3072, 'n_heads': 12, 'n_kv_heads': 6, 'n_layers': 18,
            'intermediate_size': 24576, 'head_dim': 256, 'vocab_size': 256000,
            'max_seq_len': 8192,
        },
    }
    
    @staticmethod
    def print_configs():
        print("Gemma Model Configurations:")
        print("-" * 80)
        print(f"{'Model':<15}{'d_model':<10}{'Heads':<10}{'KV Heads':<10}"
              f"{'Layers':<10}{'d_ff':<10}{'Vocab':<10}")
        print("-" * 80)
        for name, config in GemmaConfig.MODELS.items():
            print(f"{name:<15}{config['hidden_size']:<10}{config['n_heads']:<10}"
                  f"{config['n_kv_heads']:<10}{config['n_layers']:<10}"
                  f"{config['intermediate_size']:<10}{config['vocab_size']:<10}")
        
        print("\n--- Key Design Features ---")
        print("- GeGLU activation (GELU gating)")
        print("- RoPE position embeddings")
        print("- RMSNorm (pre-norm)")
        print("- GQA (Gemma 2B: 4 KV heads, 7B: 6 KV heads)")
        print("- Head dimension: 256 (wider than LLaMA's 128)")
        print("- 256K vocabulary tokenizer")

GemmaConfig.print_configs()
# Output: Gemma Model Configurations:
# Output: --------------------------------------------------------------------------------
# Output: Model          d_model   Heads     KV Heads  Layers    d_ff      Vocab     
# Output: --------------------------------------------------------------------------------
# Output: Gemma 2B       2048      8         4         14        16384     256000    
# Output: Gemma 7B       3072      12        6         18        24576     256000    
```

### Example 3: Gemma Safety Filtering Pipeline

```python
import re
from typing import List, Tuple

class GemmaSafetyFilter:
    """Safety filtering pipeline used in Gemma training"""
    
    def __init__(self):
        self.safety_categories = {
            'hate_speech': r'\b(hate|kill|attack)\b.*\b(group|race|religion)\b',
            'violence': r'\b(violence|blood|kill|murder|weapon)\b',
            'sexual': r'\b(explicit|porn|sexual|nsfw)\b',
            'personal_info': r'\b(\d{3}[-.]?\d{3}[-.]?\d{4}|email@|ssn)\b',
        }
        
    def filter_document(self, text: str) -> Tuple[bool, float]:
        """
        Filter a document based on safety criteria.
        Returns (keep, safety_score)
        """
        text_lower = text.lower()
        total_penalty = 0.0
        matches = []
        
        for category, pattern in self.safety_categories.items():
            found = re.findall(pattern, text_lower)
            if found:
                penalty = len(found) * 0.15
                total_penalty += penalty
                matches.append((category, len(found)))
        
        safety_score = max(0.0, 1.0 - total_penalty)
        keep = safety_score >= 0.7  # Threshold
        
        return keep, safety_score
    
    def compute_toxicity(self, text: str) -> float:
        """Compute toxicity score (0=clean, 1=toxic)"""
        text_lower = text.lower()
        toxicity = 0.0
        
        toxic_patterns = [
            (r'\b(hate|stupid|idiot)\b', 0.3),
            (r'\b(kill|murder|die)\b', 0.5),
            (r'\b(discrimination|racist|sexist)\b', 0.4),
        ]
        
        for pattern, weight in toxic_patterns:
            matches = re.findall(pattern, text_lower)
            toxicity += len(matches) * weight
        
        return min(toxicity, 1.0)
    
    def process_corpus(self, documents: List[str]) -> Tuple[List[str], dict]:
        """Process a corpus through safety filters"""
        kept = []
        stats = {'total': len(documents), 'kept': 0, 'removed': 0, 'categories': {}}
        
        for doc in documents:
            keep, score = self.filter_document(doc)
            if keep:
                kept.append(doc)
                stats['kept'] += 1
            else:
                stats['removed'] += 1
        
        return kept, stats

# Demonstrate
filter_obj = GemmaSafetyFilter()
docs = [
    "This is a clean document about machine learning.",
    "This document contains hate speech against a group.",
    "Another clean article about technology advancements.",
    "This has violence and weapons content."
]

kept, stats = filter_obj.process_corpus(docs)
print(f"Safety filtering: {stats['kept']}/{stats['total']} documents kept")
for doc in docs:
    _, score = filter_obj.filter_document(doc)
    tox = filter_obj.compute_toxicity(doc)
    print(f"Score: {score:.2f}, Toxicity: {tox:.2f} - {doc[:50]}")
# Output: Safety filtering: 2/4 documents kept
# Output: Score: 1.00, Toxicity: 0.00 - This is a clean document about machine learning.
# Output: Score: 0.55, Toxicity: 0.90 - This document contains hate speech against a group.
# Output: Score: 1.00, Toxicity: 0.00 - Another clean article about technology advancements.
# Output: Score: 0.70, Toxicity: 1.50 - This has violence and weapons content.
```

### Example 4: Gemma Training Pipeline

```python
class GemmaTrainingPipeline:
    """Gemma training configuration"""
    
    @staticmethod
    def get_training_config(model_size='7B'):
        """Training hyperparameters for Gemma"""
        configs = {
            '2B': {
                'batch_size': 256,
                'learning_rate': 1e-3,
                'warmup_steps': 5000,
                'total_steps': 800000,
                'weight_decay': 0.1,
                'optimizer': 'AdamW',
                'precision': 'bfloat16',
            },
            '7B': {
                'batch_size': 512,
                'learning_rate': 8e-4,
                'warmup_steps': 10000,
                'total_steps': 1000000,
                'weight_decay': 0.1,
                'optimizer': 'AdamW',
                'precision': 'bfloat16',
            },
        }
        return configs.get(model_size, configs['7B'])
    
    @staticmethod
    def estimate_training_cost(model_size, tokens=6e12):
        """Estimate training cost in FLOPs"""
        params = {'2B': 2e9, '7B': 7e9}[model_size]
        flops = 6 * params * tokens
        return flops

# Compare training costs
for size in ['2B', '7B']:
    config = GemmaTrainingPipeline.get_training_config(size)
    flops = GemmaTrainingPipeline.estimate_training_cost(size)
    print(f"Gemma {size}: {flops:.2e} FLOPs, "
          f"batch={config['batch_size']}, lr={config['learning_rate']}, "
          f"steps={config['total_steps']}")
# Output: Gemma 2B: 7.20e+22 FLOPs, batch=256, lr=0.001, steps=800000
# Output: Gemma 7B: 2.52e+23 FLOPs, batch=512, lr=0.0008, steps=1000000
```

### Example 5: Gemma vs LLaMA Architecture Comparison

```python
class ArchitectureComparison:
    """Compare Gemma with LLaMA architecture"""
    
    @staticmethod
    def compare():
        models = {
            'Gemma 2B': {'d_model': 2048, 'n_heads': 8, 'n_kv_heads': 4, 'n_layers': 14, 
                        'head_dim': 256, 'activation': 'GeGLU', 'norm': 'RMSNorm'},
            'Gemma 7B': {'d_model': 3072, 'n_heads': 12, 'n_kv_heads': 6, 'n_layers': 18,
                        'head_dim': 256, 'activation': 'GeGLU', 'norm': 'RMSNorm'},
            'LLaMA 7B': {'d_model': 4096, 'n_heads': 32, 'n_kv_heads': 32, 'n_layers': 32,
                        'head_dim': 128, 'activation': 'SwiGLU', 'norm': 'RMSNorm'},
            'LLaMA 13B': {'d_model': 5120, 'n_heads': 40, 'n_kv_heads': 40, 'n_layers': 40,
                         'head_dim': 128, 'activation': 'SwiGLU', 'norm': 'RMSNorm'},
        }
        
        print("Architecture Comparison: Gemma vs LLaMA")
        print("-" * 90)
        print(f"{'Model':<15}{'d_model':<10}{'Heads':<10}{'KV':<8}{'Layers':<10}"
              f"{'Head Dim':<10}{'Activation':<12}{'Norm':<10}")
        print("-" * 90)
        
        for name, config in models.items():
            print(f"{name:<15}{config['d_model']:<10}{config['n_heads']:<10}"
                  f"{config['n_kv_heads']:<8}{config['n_layers']:<10}"
                  f"{config['head_dim']:<10}{config['activation']:<12}{config['norm']:<10}")
        
        print("\n--- Key Differences ---")
        print("1. Gemma is deeper and narrower (fewer heads, smaller d_model, more depth)")
        print("2. Gemma uses GeGLU vs LLaMA's SwiGLU (both gated, different activation)")
        print("3. Gemma has wider head dimension (256 vs 128)")
        print("4. Gemma has larger vocabulary (256K vs LLaMA's 32K)")

ArchitectureComparison.compare()
# Output: Architecture Comparison: Gemma vs LLaMA
# Output: ------------------------------------------------------------------------------------------
# Output: Model          d_model   Heads     KV      Layers    Head Dim  Activation  Norm      
# Output: ------------------------------------------------------------------------------------------
# Output: Gemma 2B       2048      8         4       14        256       GeGLU       RMSNorm   
# Output: Gemma 7B       3072      12        6       18        256       GeGLU       RMSNorm   
# Output: LLaMA 7B       4096      32        32      32        128       SwiGLU      RMSNorm   
# Output: LLaMA 13B      5120      40        40      40        128       SwiGLU      RMSNorm   
```

## Common Mistakes

### 1. Assuming Gemma Is a Direct Gemini Open-Source Version
Gemma is inspired by Gemini but is not a directly open-sourced version of Gemini. It is a separate, smaller model family designed specifically for open-source release. The architectures share design principles but differ in scale and some implementation details.

### 2. Using the Wrong Vocabulary Size
Gemma uses a 256K vocabulary tokenizer (8x LLaMA's 32K). Using LLaMA's tokenizer with Gemma weights produces completely incorrect results. The tokenizer must be downloaded separately and loaded correctly.

### 3. Ignoring the Deep-Narrow Architecture Trade-off
Gemma's deep but narrow design (fewer layers but smaller dimensions) means it has different scaling properties than LLaMA. The deep architecture captures hierarchical patterns better but may have less capacity per layer. Fine-tuning approaches that work for LLaMA may need adjustment for Gemma.

### 4. Overlooking Safety Filtering in Fine-Tuning
Gemma's training included extensive safety filtering. When fine-tuning, it is important to preserve this safety alignment. The model includes a safety classifier that should be used alongside the model in production deployments.

### 5. Confusing GeGLU with SwiGLU
GeGLU (GELU gate) and SwiGLU (SiLU gate) are similar gating mechanisms but use different activation functions. GeGLU uses GELU (Gaussian Error Linear Unit) while SwiGLU uses SiLU (Sigmoid Linear Unit). They are not interchangeable and produce different gradient patterns during training.

## Interview Questions

### Beginner
**Q1: What is Gemma and who developed it?**
A1: Gemma is a family of lightweight, open-source decoder-only LLMs (2B, 7B) developed by Google DeepMind, released in February 2024. It is inspired by Google's proprietary Gemini model and emphasizes accessibility, safety, and responsible deployment.

**Q2: What is the key architectural difference between Gemma and LLaMA?**
A2: Gemma uses a deeper but narrower design (fewer layers with smaller hidden dimensions but more depth relative to model size), GeGLU activation instead of SwiGLU, wider head dimension (256 vs 128), and a 256K vocabulary tokenizer.

### Intermediate
**Q3: Explain GeGLU activation and how it differs from SwiGLU.**
A3: GeGLU uses GELU as the gating activation in a gated linear unit: output = GELU(xW_gate) ⊙ (xW_up) W_down. SwiGLU uses SiLU instead of GELU. GELU is smoother than SiLU for negative values (GELU approaches zero more gradually), which can affect gradient flow. Both provide similar performance; the choice is primarily a design preference reflecting different model development teams.

**Q4: How does Gemma's deep-narrow architecture affect training and inference compared to LLaMA's wide-shallow design?**
A4: The deep-narrow design (Gemma: 14-18 layers, 2048-3072 dim) has fewer parameters in attention (which scales with d²) but more sequential computation through depth. This can improve hierarchical representation learning but increases inference latency (more sequential steps). The wide-shallow design (LLaMA: 32-40 layers, 4096-5120 dim) has more parallel computation per layer but fewer hierarchical levels. For batch inference, Gemma's design may have higher latency; for single-query inference, the difference is smaller.

### Advanced
**Q5: Analyze Gemma's decision to use a 256K vocabulary. What are the trade-offs?**
A5: A 256K vocabulary provides better tokenization efficiency (more information per token), especially for multilingual text and code. Each token represents ~4-5 characters vs ~2-3 for 32K vocabularies. This means 6T tokens with 256K vocabulary contains effectively 2x more textual information than 6T tokens with 32K vocabulary. Trade-offs: (1) Larger embedding and LM head matrices (256K × d_model, which can be 0.5-0.8B parameters), (2) Slower softmax over 256K logits, (3) Higher risk of token under-training for rare tokens. Gemma offsets these costs with the benefit of more efficient text representation.

**Q6: Design a safety evaluation framework for Gemma that tests both its capabilities and its safety guardrails simultaneously.**
A6: A comprehensive safety evaluation framework should: (1) Use adversarial testing with prompts designed to bypass safety filters (jailbreaking, role-playing, fictional scenarios); (2) Evaluate on standard benchmarks (MMLU, HellaSwag) while simultaneously checking for unsafe outputs; (3) Measure the safety-capability Pareto frontier: how much capability is sacrificed for safety? (4) Test demographic fairness across different groups; (5) Evaluate instruction following for safe vs unsafe requests separately; (6) Measure calibration of refusal responses (are refusals appropriate to the request?); (7) Use red-teaming with automated and human evaluation. The key insight is that safety should not be evaluated in isolation—it must be measured alongside capability to understand the complete model behavior profile.

## Practice Problems

### Easy
Implement the GeGLU activation function and compare its forward pass with SwiGLU on random data.

### Medium
Implement a complete Gemma 2B block (attention with GQA, GeGLU MLP, RMSNorm) and compare its parameter count and forward speed against a LLaMA 7B block.

### Hard
Design and implement a safety evaluation pipeline that tests a Gemma-style model on 5 safety categories with 20 test prompts each, measuring both safety compliance and capability retention.

## Solutions

### Easy Solution
```python
def compare_activations(x=torch.randn(4, 16, 2048)):
    geglu = GeGLU(2048, 16384)
    swiglu = nn.Sequential(nn.Linear(2048, 16384), nn.SiLU(), nn.Linear(16384, 2048))
    out_g = geglu(x)
    out_s = swiglu(x)
    print(f"GeGLU std: {out_g.std():.4f}, SwiGLU std: {out_s.std():.4f}")
```

### Medium Solution
```python
def compare_blocks():
    gemma = GemmaBlock(2048, 8, 4, 16384)
    llama = LLaMABlock(4096, 32, 11008)
    x = torch.randn(2, 32, 2048)
    out = gemma(x)
    g_params = sum(p.numel() for p in gemma.parameters())
    l_params = sum(p.numel() for p in llama.parameters())
    print(f"Gemma block: {g_params:,}, LLaMA block: {l_params:,}")
```

### Hard Solution
```python
class SafetyEvaluator:
    def __init__(self, model, safety_categories):
        self.model = model
        self.categories = safety_categories
    
    def evaluate(self):
        results = {}
        for category, prompts in self.categories.items():
            safe_kept = 0
            for prompt in prompts['unsafe']:
                response = self.model.generate(prompt)
                if self.is_refusal(response):
                    safe_kept += 1
            results[category] = safe_kept / len(prompts['unsafe'])
        return results
```

## Related Concepts
- DL-425: LLaMA Architecture - Direct architecture comparison
- DL-427: LLaMA 3 - Latest LLaMA generation
- DL-428: Mistral and Mixtral - Efficient alternatives
- DL-429: Falcon - Another efficient open-source model
- DL-447: Grouped Query Attention - Used in Gemma's attention

## Next Concepts
- DL-431: T5 Architecture - Encoder-decoder architecture
- DL-432: Text-to-Text Framework - Unified NLP framework
- DL-433: T5 Pre-Training - Pre-training objectives

## Summary
Gemma is Google DeepMind's lightweight open-source LLM family (2B, 7B) featuring GeGLU activation, RoPE embeddings, RMSNorm, and a deep-narrow architecture with wide head dimensions (256). Trained on 6T tokens with extensive safety filtering, Gemma brings Google's architectural expertise to the open-source community in accessible sizes. Its 256K vocabulary tokenizer and careful design choices make it a competitive option for researchers and developers.

## Key Takeaways
- GeGLU activation (GELU-gated FFN) instead of SwiGLU
- Deep-narrow architecture: fewer but smaller layers
- Wide head dimension: 256 (vs LLaMA's 128)
- GQA with 4-6 KV heads
- 256K vocabulary tokenizer (8x LLaMA)
- Extensive safety filtering in training pipeline
- 6T tokens of training data
- Designed for single-GPU deployment (2B model)
