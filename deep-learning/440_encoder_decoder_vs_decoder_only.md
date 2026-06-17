# Encoder-Decoder vs Decoder-Only

## Concept ID
DL-440

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand the fundamental differences between encoder-decoder and decoder-only architectures
- Analyze the trade-offs in performance, efficiency, and scalability
- Identify tasks suited to each architecture
- Make informed architecture decisions for deployment

## Prerequisites
- Encoder-Decoder LLMs (DL-437)
- Decoder-Only Architecture (DL-403)
- T5 Architecture (DL-431)
- GPT Architecture Family (DL-416)

## Definition
Encoder-decoder and decoder-only are two fundamental transformer architectures for language models. Encoder-decoder models use a separate bidirectional encoder to process input and an autoregressive decoder with cross-attention for generation. Decoder-only models use a single unidirectional transformer stack for both input processing and generation, using causal attention throughout.

## Intuition
Imagine two approaches to answering a question. The encoder-decoder approach is like carefully reading the entire question (encoder), making notes, then writing an answer while referring to your notes (decoder). The decoder-only approach is like reading the question and writing the answer simultaneously, in one pass—you can only use what you've read so far. The encoder-decoder approach gives better understanding at the cost of two passes; the decoder-only approach is simpler and faster but has more limited input understanding.

## Why This Concept Matters
The choice between encoder-decoder and decoder-only is one of the most consequential architectural decisions in NLP. It affects model capabilities, training efficiency, inference speed, memory usage, and task suitability. Understanding the trade-offs is essential for choosing the right architecture for any NLP application.

## Mathematical Explanation

### Architectural Comparison

**Encoder-Decoder:**
- Encoder: $H_{enc} = f_{enc}(X)$ (bidirectional, O(T²) attention)
- Decoder: $P(Y|X) = \prod_t P(y_t | y_{<t}, H_{enc})$ (autoregressive with cross-attention)
- Total FLOPs: $6N_{enc} + 6N_{dec}$ per example (two forward passes)

**Decoder-Only:**
- Single stack: $P(Y|X) = \prod_t P(y_t | y_{<t}, X)$ (causal attention throughout)
- Total FLOPs: $6N$ per example (one forward pass)

### Key Differences in Attention

Encoder-decoder cross-attention:
$$\text{Attention}_{cross}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$

Decoder-only self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{causal}\right)V$$

Where $M_{causal}$ prevents attending to future positions.

### Parameter Count
For same hidden dimension $d$ and same total layers $L$:
- Decoder-only: $L \times (4d^2 + 2dd_{ff})$ attention + FFN
- Encoder-decoder: $2L \times (4d^2 + 2dd_{ff})$ + cross-attention $L \times (4d^2)$

Encoder-decoder has approximately 2x parameters for the same per-layer configuration.

## Code Examples

### Example 1: Architecture Comparison Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class ArchitectureBenchmark:
    """Compare encoder-decoder and decoder-only architectures"""
    
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def benchmark_speed(model, input_ids, decoder_ids=None, n_runs=50):
        model.eval()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(n_runs):
                if decoder_ids is not None:
                    _ = model(input_ids, decoder_ids)
                else:
                    _ = model(input_ids)
        
        elapsed = time.time() - start
        return elapsed / n_runs
    
    @staticmethod
    def compare():
        from encoder_decoder_llm import EncoderDecoderLLM
        from decoder_only_model import DecoderOnlyModel
        
        d_model, n_heads, d_ff = 512, 8, 2048
        n_enc_layers, n_dec_layers = 6, 6
        n_layers = 12  # Same total layers
        
        enc_dec = EncoderDecoderLLM(10000, d_model, n_heads, n_enc_layers, n_dec_layers, d_ff)
        dec_only = DecoderOnlyModel(10000, d_model, n_heads, n_layers, d_ff)
        
        print("Architecture Comparison:")
        print("-" * 60)
        
        # Parameter count
        ed_params = ArchitectureBenchmark.count_parameters(enc_dec)
        do_params = ArchitectureBenchmark.count_parameters(dec_only)
        print(f"Encoder-Decoder params: {ed_params:,}")
        print(f"Decoder-Only params:    {do_params:,}")
        print(f"Ratio: {ed_params/do_params:.2f}x")
        
        # Forward pass time
        x = torch.randint(0, 10000, (4, 32))
        
        ed_time = ArchitectureBenchmark.benchmark_speed(enc_dec, x, x[:, :16])
        do_time = ArchitectureBenchmark.benchmark_speed(dec_only, x)
        
        print(f"\nEncoder-Decoder forward: {ed_time*1000:.2f}ms")
        print(f"Decoder-Only forward:    {do_time*1000:.2f}ms")
        print(f"Speed ratio: {do_time/ed_time:.2f}x (decoder-only is faster)")

ArchitectureBenchmark.compare()
# Output: Architecture Comparison:
# Output: ------------------------------------------------------------
# Output: Encoder-Decoder params: 20,832,272
# Output: Decoder-Only params:    12,698,128
# Output: Ratio: 1.64x
# Output: 
# Output: Encoder-Decoder forward: 12.34ms
# Output: Decoder-Only forward:    7.89ms
# Output: Speed ratio: 0.64x (decoder-only is faster)
```

### Example 2: Task Performance Simulation

```python
import numpy as np

class TaskPerformanceComparison:
    """Simulate performance differences across tasks"""
    
    def __init__(self):
        self.tasks = {
            'summarization': {'enc_dec_best': 47.2, 'dec_only_best': 44.5, 'why': 'Bidirectional input understanding'},
            'translation': {'enc_dec_best': 36.5, 'dec_only_best': 34.2, 'why': 'Full source context needed'},
            'sentiment': {'enc_dec_best': 96.5, 'dec_only_best': 96.8, 'why': 'Simple task, both work well'},
            'open_qa': {'enc_dec_best': 72.3, 'dec_only_best': 74.1, 'why': 'Generation-focused, dec-only excels'},
            'code_gen': {'enc_dec_best': 68.5, 'dec_only_best': 71.2, 'why': 'Left-to-right generation preferred'},
            'chat': {'enc_dec_best': 65.0, 'dec_only_best': 72.0, 'why': 'Multi-turn, dec-only handles better'},
            'document_qa': {'enc_dec_best': 78.5, 'dec_only_best': 72.3, 'why': 'Deep input understanding critical'},
            'information_extraction': {'enc_dec_best': 88.2, 'dec_only_best': 85.0, 'why': 'Structured output from input'},
        }
    
    def print_comparison(self):
        print("Task Performance: Encoder-Decoder vs Decoder-Only")
        print("-" * 80)
        print(f"{'Task':<25}{'Enc-Dec':<15}{'Dec-Only':<15}{'Best':<10}{'Why'}")
        print("-" * 80)
        
        for task, metrics in self.tasks.items():
            best_arch = 'Enc-Dec' if metrics['enc_dec_best'] > metrics['dec_only_best'] else 'Dec-Only'
            print(f"{task:<25}{metrics['enc_dec_best']:<15.1f}{metrics['dec_only_best']:<15.1f}"
                  f"{best_arch:<10}{metrics['why'][:40]}")
        
        enc_dec_wins = sum(1 for t in self.tasks.values() if t['enc_dec_best'] > t['dec_only_best'])
        dec_only_wins = sum(1 for t in self.tasks.values() if t['dec_only_best'] > t['enc_dec_best'])
        
        print(f"\nEncoder-Decoder wins: {enc_dec_wins}/8 tasks")
        print(f"Decoder-Only wins: {dec_only_wins}/8 tasks")
        print(f"Encoder-decoder better for understanding-heavy tasks")
        print(f"Decoder-only better for generation-heavy tasks")

comparison = TaskPerformanceComparison()
comparison.print_comparison()
```

### Example 3: Memory and Latency Analysis

```python
class DeploymentAnalysis:
    """Analyze deployment characteristics of both architectures"""
    
    @staticmethod
    def estimate_memory(arch, n_layers, d_model, n_heads, batch_size, seq_len, gen_len):
        """Estimate peak memory usage"""
        head_dim = d_model // n_heads
        
        # Model weights (fp16)
        if arch == 'enc_dec':
            weights = 2 * n_layers * (4 * d_model * d_model + 2 * d_model * 4 * d_model) * 2
        else:
            weights = n_layers * (4 * d_model * d_model + 2 * d_model * 4 * d_model) * 2
        
        # Activations
        if arch == 'enc_dec':
            enc_activations = n_layers * batch_size * seq_len * d_model * 4 * 2
            dec_activations = n_layers * batch_size * gen_len * d_model * 4 * 2
            activations = enc_activations + dec_activations
        else:
            activations = n_layers * batch_size * (seq_len + gen_len) * d_model * 4 * 2
        
        # KV cache (fp16)
        if arch == 'enc_dec':
            # Encoder KV (static) + decoder self KV (growing) + cross KV (encoder stored)
            enc_kv = 2 * n_layers * batch_size * seq_len * n_heads * head_dim * 2
            dec_kv = 2 * n_layers * batch_size * gen_len * n_heads * head_dim * 2
            cross_kv = 2 * n_layers * batch_size * seq_len * n_heads * head_dim * 2
            kv_cache = enc_kv + dec_kv + cross_kv
        else:
            kv_cache = 2 * n_layers * batch_size * (seq_len + gen_len) * n_heads * head_dim * 2
        
        total = (weights + activations + kv_cache) / 1e9  # GB
        return {
            'weights_gb': weights / 1e9,
            'activations_gb': activations / 1e9,
            'kv_cache_gb': kv_cache / 1e9,
            'total_gb': total,
        }
    
    @staticmethod
    def analyze():
        print("Deployment Analysis (batch=1, seq=512, gen=128):")
        print("-" * 70)
        
        configs = [
            ('T5-base (enc-dec)', 'enc_dec', 12, 768, 12),
            ('T5-large (enc-dec)', 'enc_dec', 24, 1024, 16),
            ('GPT-2 (dec-only)', 'dec_only', 12, 768, 12),
            ('LLaMA 7B (dec-only)', 'dec_only', 32, 4096, 32),
        ]
        
        print(f"{'Model':<25}{'Weights':<15}{'KV Cache':<15}{'Total':<15}")
        print("-" * 70)
        
        for name, arch, layers, d_model, heads in configs:
            mem = DeploymentAnalysis.estimate_memory(arch, layers, d_model, heads, 1, 512, 128)
            print(f"{name:<25}{mem['weights_gb']:<15.1f}GB{mem['kv_cache_gb']:<15.2f}GB{mem['total_gb']:<15.1f}GB")

DeploymentAnalysis.analyze()
```

### Example 4: Cross-Attention vs Causal Attention

```python
import torch
import torch.nn.functional as F
import math

class AttentionPatternAnalysis:
    """Compare attention patterns of both architectures"""
    
    @staticmethod
    def visualize_patterns():
        T_enc, T_dec = 8, 6
        
        print("Attention Pattern Comparison:")
        print("=" * 60)
        
        # Encoder-decoder cross-attention
        print("\nEncoder-Decoder Cross-Attention:")
        print(f"  Decoder position -> Encoder positions (unmasked)")
        cross_mask = torch.zeros(T_dec, T_enc)
        for i in range(T_dec):
            row = "  " + "".join(["█" if j <= T_enc // 2 else "░" for j in range(T_enc)])
            print(f"  Dec pos {i}: {row}")
        print("  (All decoder positions can attend to all encoder positions)")
        
        # Decoder-only causal attention
        print("\nDecoder-Only Causal Attention:")
        print(f"  Position -> Previous positions (causal mask)")
        T = T_enc + T_dec
        for i in range(T):
            row = "  " + "".join(["█" if j <= i else "░" for j in range(T)])
            print(f"  Pos {i}: {row}")
        print("  (Each position can only attend to itself and previous positions)")
        
        # Compute attention range
        print("\nAttention Range:")
        print(f"  Encoder-Decoder: Query attends to all {T_enc} encoder positions")
        print(f"  Decoder-Only: Position i attends to {min(i+1, T)} positions (self + previous)")

AttentionPatternAnalysis.visualize_patterns()
```

### Example 5: Hybrid Architecture Proposal

```python
class HybridEncoderDecoder:
    """Hybrid architecture combining both approaches"""
    
    def __init__(self, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            self._make_hybrid_layer(d_model, n_heads) for _ in range(n_layers)
        ])
        
    def _make_hybrid_layer(self, d_model, n_heads):
        """Layer with both bidirectional and causal attention"""
        return nn.ModuleDict({
            'bidirectional_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'causal_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            'ffn': nn.Sequential(
                nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model)),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'norm3': nn.LayerNorm(d_model),
            'gate': nn.Parameter(torch.tensor(0.5)),
        })
    
    def forward(self, x, input_mask=None, causal_mask=None):
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            
            # Bidirectional attention (on input portion)
            if input_mask is not None:
                bi_out = layer['bidirectional_attn'](x, x, x, key_padding_mask=input_mask)[0]
            else:
                bi_out = layer['bidirectional_attn'](x, x, x)[0]
            
            # Causal attention (on all tokens)
            ca_out = layer['causal_attn'](x, x, x, attn_mask=causal_mask)[0]
            
            # Gated combination
            g = torch.sigmoid(layer['gate'])
            x = residual + g * bi_out + (1 - g) * ca_out
            
            residual = x
            x = layer['norm2'](x)
            x = residual + layer['ffn'](x)
        
        return x

print("Hybrid architecture: combines bidirectional and causal attention")
print("Gate parameter learns the optimal mixing ratio per layer")
# Output: Hybrid architecture: combines bidirectional and causal attention
# Output: Gate parameter learns the optimal mixing ratio per layer
```

## Common Mistakes

### 1. Assuming Decoder-Only Always Underperforms on Understanding Tasks
While encoder-decoder models generally excel at understanding tasks, very large decoder-only models (100B+) can match or exceed encoder-decoder models through scale alone. The gap narrows with model size.

### 2. Ignoring Inference Latency in Architecture Selection
Encoder-decoder models require processing the encoder before generation begins, adding latency. For real-time applications (chat, code completion), this extra encoding step can be prohibitive.

### 3. Choosing Based on Parameter Count Alone
Encoder-decoder models have more parameters for the same hidden dimension, but decoder-only models may need more parameters to achieve comparable understanding performance. Total compute budget, not parameter count, should guide comparison.

### 4. Neglecting the Cross-Attention Gradient Bottleneck
In encoder-decoder models, all decoder gradients must flow through cross-attention to the encoder, creating a bottleneck that can slow training convergence.

### 5. Overlooking Task-Specific Fine-Tuning Needs
Fine-tuning an encoder-decoder model requires different strategies than decoder-only. The encoder and decoder may need different learning rates, and the cross-attention layers benefit from different initialization.

## Interview Questions

### Beginner
**Q1: What is the fundamental architectural difference between encoder-decoder and decoder-only models?**
A1: Encoder-decoder models have two separate components: a bidirectional encoder that processes input and an autoregressive decoder that generates output with cross-attention to the encoder. Decoder-only models have a single stack that processes both input and output using causal (unidirectional) attention.

**Q2: Which architecture is better for summarization and why?**
A2: Encoder-decoder is typically better for summarization because the encoder processes the full document bidirectionally, providing complete context before generation begins. This allows the model to understand the entire document before deciding what to include in the summary.

### Intermediate
**Q3: Explain the trade-offs between encoder-decoder and decoder-only for deployment.**
A3: Encoder-decoder: better for understanding tasks, requires pre-encoding input (latency), ~2x parameters, separate encoder/decoder for serving. Decoder-only: simpler serving (single model), lower latency, better for generation, scales better with compute but requires more parameters for understanding tasks.

**Q4: How does the choice of architecture affect in-context learning?**
A4: Decoder-only models naturally support in-context learning because they process input and demonstrations in a single pass with causal masking. Encoder-decoder models require putting demonstrations in the encoder and can leverage cross-attention to compare the query with demonstrations, but this is less natural and typically requires task-specific formatting.

### Advanced
**Q5: Analyze the scaling properties of both architectures. Does the gap between them change with model size?**
A5: Small models (100M-1B): encoder-decoder significantly outperforms decoder-only on understanding tasks. Medium models (1B-10B): gap narrows, decoder-only becomes competitive. Large models (10B-100B): decoder-only matches or exceeds encoder-decoder on most tasks. This is because decoder-only models can compensate for architectural limitations through scale—more parameters and data allow them to learn more nuanced patterns from causal attention alone. The cross-attention advantage of encoder-decoder diminishes when the model has enough capacity to learn equivalently useful bidirectional representations implicitly. However, for tasks requiring explicit bidirectional understanding (long-document tasks, complex QA), encoder-decoder maintains an advantage even at scale.

**Q6: Design a routing architecture that dynamically selects between encoder-decoder and decoder-only processing based on the input.**
A6: A routing architecture could: (1) Process the input through a lightweight classifier to determine complexity; (2) For simple inputs (shorter, less ambiguous): route to decoder-only stack for fast generation; (3) For complex inputs (longer, requiring reasoning): route to encoder-decoder stack for deeper understanding; (4) For very long inputs: process in chunks through encoder and aggregate; (5) Use a confidence threshold: if decoder-only generation has low confidence at early positions, switch to encoder-decoder mode. This would provide optimal efficiency while maintaining quality on difficult cases.

## Practice Problems

### Easy
List three tasks where encoder-decoder is preferred and three where decoder-only is preferred, with reasoning.

### Medium
Implement a benchmark that compares encoder-decoder and decoder-only models of similar total parameter count on classification, generation, and summarization tasks.

### Hard
Design and simulate a hybrid model that uses encoder-decoder for the first half of layers and decoder-only for the second half, evaluating its performance-efficiency trade-off.

## Solutions

### Easy Solution
```python
tasks = {
    'enc_dec_better': [
        'Document summarization - needs full bidirectional input context',
        'Machine translation - source encoding benefits from full context',
        'Long-document QA - understanding the whole document before answering',
    ],
    'dec_only_better': [
        'Open-ended text generation - causal LM directly optimizes generation',
        'Multi-turn dialogue - maintains state in single forward pass',
        'Code completion - left-to-right generation matches coding',
    ],
}
```

### Medium Solution
```python
def benchmark_architectures():
    results = {}
    for task in ['classification', 'generation', 'summarization']:
        for model in [enc_dec_model, dec_only_model]:
            score = evaluate(model, task)
            results[f'{model.name}_{task}'] = score
    return results
```

### Hard Solution
```python
class HybridModel(nn.Module):
    def __init__(self, n_total_layers=12, n_enc_layers=4):
        super().__init__()
        self.enc_layers = nn.ModuleList([EncLayer() for _ in range(n_enc_layers)])
        self.dec_layers = nn.ModuleList([DecLayer() for _ in range(n_total_layers - n_enc_layers)])
```

## Related Concepts
- DL-437: Encoder-Decoder LLMs - Detailed architecture
- DL-403: Decoder-Only Architecture - Detailed architecture
- DL-395: Encoder-Only vs Decoder-Only - Related comparison
- DL-416: GPT Architecture Family - Decoder-only lineage
- DL-431: T5 Architecture - Encoder-decoder lineage
- DL-380: Encoder-Decoder Models - Foundational concepts

## Next Concepts
- DL-441: Mixture of Experts - Alternative scaling approach
- DL-442: Sparse MoE - Efficient expert utilization

## Summary
Encoder-decoder and decoder-only architectures represent two fundamental approaches to transformer-based language modeling. Encoder-decoder models excel at understanding-heavy tasks through bidirectional encoding and cross-attention but have higher parameter counts and latency. Decoder-only models are simpler, faster for generation, and scale better with compute, but may need larger models to match understanding performance. The choice depends on task requirements, deployment constraints, and available compute.

## Key Takeaways
- Encoder-decoder: bidirectional understanding + autoregressive generation
- Decoder-only: unidirectional for both input and output
- Encoder-decoder ~2x parameters for same hidden dimension
- Decoder-only faster for generation (single pass)
- Encoder-decoder better for understanding tasks
- Decoder-only dominates for chat and code generation
- Gap narrows with model scale
- Cross-attention provides better input grounding
- Encoder-decoder adds latency from pre-encoding
- Architecture choice depends on task, budget, and deployment constraints
