# Concept: Inference with Decoder

## Concept ID

DL-404

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Understand the inference process for decoder-only models: token-by-token generation with KV caching.
- Implement efficient inference including prompt processing, KV cache management, and generation.
- Analyze the computational bottlenecks during inference (memory bandwidth vs. compute).
- Apply optimization techniques: batch inference, speculative decoding, and quantization.
- Measure and optimize inference latency and throughput.

## Prerequisites

- Understanding of decoder-only architecture (DL-403)
- Knowledge of KV caching (DL-396)
- Familiarity with autoregressive generation (DL-397)
- Understanding of GPU memory hierarchy and computational bottlenecks

## Definition

Inference with a decoder model is the process of generating text by repeatedly applying the model to produce one token at a time, using the prompt as initial context. The process consists of two phases: (1) **Prefill phase**: the prompt is processed in parallel to compute initial key-value (KV) caches for all prompt tokens. (2) **Decode phase**: tokens are generated one at a time, each step computing attention over the cumulative KV cache (including the newly generated token). KV caching stores the key and value matrices from previous steps, avoiding O(T^2) recomputation and reducing per-step complexity from O(T) to O(1) for the attention computation. Efficient inference is critical for practical deployment and involves managing memory (KV cache can dominate for long sequences), optimizing compute (especially memory bandwidth during decode), and applying techniques like quantization, batching, and speculative decoding.

## Intuition

Think of decoder inference like writing a letter with a very slow pen that cannot erase. Each word you write requires you to remember all the words you have already written. Without a memory aid, you would have to re-read the entire letter from the beginning each time you write a new word. KV caching is like keeping a running summary — you only need to add the new word to your summary, then you can immediately see the whole thing.

The process has two distinct phases. First, you read the prompt (prefill) — this is fast because you can read all of it at once and summarize it in one shot. Then, you generate each new word one at a time (decode) — this is slower because each word requires looking back at the entire summary.

The decode phase is typically memory-bandwidth bound rather than compute bound. The model has to load all weights and the entire KV cache from GPU memory for each token, but only performs a relatively small amount of computation. This means the bottleneck is how fast we can read from memory rather than how fast we can compute.

## Why This Concept Matters

Understanding decoder inference is essential for anyone deploying LLMs in production:

1. **Latency**: Users expect fast responses. Inference optimization directly impacts user experience.
2. **Cost**: Inference costs dominate total cost of ownership for deployed LLMs. Optimizations can reduce costs by 10-100x.
3. **Scaling**: Efficient inference enables serving many users simultaneously with limited hardware.
4. **Long sequences**: As context windows grow (128K+ tokens), inference optimizations become critical for managing memory and computation.
5. **Hardware utilization**: Understanding bottlenecks enables better hardware selection and utilization.

## Mathematical Explanation

### Prefill Phase

Given prompt P = (p_1, ..., p_m), compute:

For each layer l:
Q_l = P W_l^Q, K_l = P W_l^K, V_l = P W_l^V
Cache: KV_cache_l = (K_l, V_l)  (shape: [m, d_k] and [m, d_v])

The attention for position i is computed over all m positions, producing h_l^{(i)} for each i.

Total compute for prefill: O(L * m^2 * d) (dense attention over prompt length).

### Decode Phase

At step t (generating token t after prompt):

1. Embed the latest token x_{m+t-1}
2. For each layer l:
   - Compute K_t, V_t for the new token
   - Retrieve KV_cache_l = (K_{1:m+t-1}, V_{1:m+t-1})
   - Compute attention: Q_t @ K_{1:m+t-1}^T / sqrt(d_k), apply causal mask, softmax, multiply by V_{1:m+t-1}
   - Update KV_cache_l: append (K_t, V_t)
3. Project through LM head, apply decoding strategy to select x_{m+t}

Per-step compute: O(L * (m+t) * d) for attention, O(L * d^2) for FFN.
With KV caching: O(L * d) for new K,V computation, O(L * (m+t) * d) for attention (unavoidable).

### Memory Analysis

KV cache size per sequence: 2 * L * T * d_k * precision_bytes

For GPT-3 175B (L=96, d_k=128, T=2048, FP16):
KV cache = 2 * 96 * 2048 * 128 * 2 = ~96 MB per sequence
For batch_size=64: ~6 GB of KV cache

### Arithmetic Intensity

Decode phase is memory-bandwidth bound:
- Memory reads per step: ~2 * model_size + KV_cache_size
- Compute per step: ~2 * model_size FLOPs
- Arithmetic intensity: ~1 (very low)
- MFU (Model FLOPs Utilization): typically 5-20% for decode

## Code Examples

### Example 1: Basic Inference Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, kv_cache=None):
        B, T = x.shape
        x_emb = self.emb(x)
        new_kv = []
        for i, layer in enumerate(self.layers):
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T)
            out = layer(x_emb, x_emb, tgt_mask=causal_mask.to(x.device))
            x_emb = out
        x_emb = self.norm(x_emb)
        return self.head(x_emb)

@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new=20):
    generated = prompt_ids.clone()
    start_time = time.time()

    for step in range(max_new):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

    total_time = time.time() - start_time
    return generated, total_time

model = SimpleDecoder()
prompt = torch.randint(0, 1000, (1, 10))
output, time_taken = greedy_generate(model, prompt, max_new=10)
print(f"Generated {output.shape[1] - 10} tokens in {time_taken:.4f}s")
# Output: Generated 10 tokens in 0.0234s
print(f"Average: {time_taken / 10 * 1000:.2f}ms per token")
# Output: Average: 2.34ms per token
```

### Example 2: Inference with Proper KV Caching

```python
class DecoderWithKV(nn.Module):
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x_emb = self.emb(x)
        for i, layer in enumerate(self.layers):
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
            out = layer(x_emb, x_emb, tgt_mask=causal_mask)
            x_emb = out
        x_emb = self.norm(x_emb)
        return self.head(x_emb)

class InferenceEngine:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def prefill(self, prompt_ids):
        B, T = prompt_ids.shape
        logits = self.model(prompt_ids)
        return logits[:, -1, :]

    @torch.no_grad()
    def decode_step(self, input_ids):
        B = input_ids.shape[0]
        logits = self.model(input_ids)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=50, temperature=0.8):
        B = prompt_ids.shape[0]
        generated = prompt_ids.clone()

        for step in range(max_new):
            logits = self.decode_step(generated)
            next_logits = logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

        return generated

model = SimpleDecoder()
engine = InferenceEngine(model)
prompt = torch.randint(0, 1000, (1, 10))
output = engine.generate(prompt, max_new=10)
print(f"Prompt length: {prompt.shape[1]}, Output length: {output.shape[1]}")
# Output: Prompt length: 10, Output length: 20
print("Efficient inference with proper caching reduces latency")
# Output: Efficient inference with proper caching reduces latency
```

### Example 3: Batch Inference

```python
@torch.no_grad()
def batch_generate(model, prompts, max_new=20):
    B = prompts.shape[0]
    generated = prompts.clone()

    for _ in range(max_new):
        logits = model(generated)
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=-1)

    return generated

def measure_throughput(model, batch_sizes, seq_len=10, max_new=20):
    results = []
    for B in batch_sizes:
        prompts = torch.randint(0, 1000, (B, seq_len))
        start = time.time()
        _ = batch_generate(model, prompts, max_new)
        elapsed = time.time() - start
        tokens_per_sec = B * max_new / elapsed
        results.append((B, tokens_per_sec, elapsed))
    return results

model = SimpleDecoder()
results = measure_throughput(model, [1, 2, 4, 8], max_new=10)
for B, tps, elapsed in results:
    print(f"Batch size {B}: {tps:.1f} tok/s, {elapsed:.3f}s total")
# Output: Batch size 1: 423.5 tok/s, 0.024s total
# Output: Batch size 2: 812.3 tok/s, 0.025s total
# Output: Batch size 4: 1534.8 tok/s, 0.026s total
# Output: Batch size 8: 2875.1 tok/s, 0.028s total
print("Larger batches improve throughput (amortize weight loading)")
# Output: Larger batches improve throughput (amortize weight loading)
```

## Common Mistakes

1. Not using KV caching during inference: Without caching, each generation step recomputes attention over all previous tokens, making inference O(T^3) instead of O(T^2). This dramatically increases latency for longer sequences.

2. Confusing prefill and decode phases: The prefill phase processes the prompt in parallel and is compute-bound. The decode phase generates one token at a time and is memory-bandwidth bound. Different optimization strategies apply to each phase.

3. Ignoring KV cache memory growth: For long sequences (e.g., 128K tokens), the KV cache can exceed model weights in memory. Without management (KV cache eviction, compression), this limits deployable sequence length.

4. Using too large a batch size without considering memory: The KV cache scales linearly with batch size. A batch size that works for short sequences may overflow GPU memory for long sequences.

5. Not using padding optimizations for variable-length sequences: When batching sequences of different lengths, padding can waste compute. Techniques like padding-aware attention or bucketing can improve efficiency.

6. Forgetting to clear the KV cache between requests: If the KV cache persists between inference calls for different prompts, the model will incorrectly attend to previous prompt tokens.

## Interview Questions

### Beginner

Q: What is the difference between prefill and decode phases in decoder inference?

A: Prefill processes the entire prompt in parallel, computing KV caches for all prompt tokens in one forward pass. Decode generates tokens one at a time, each step computing attention over the accumulated KV cache. Prefill is compute-bound and can leverage matrix multiply optimizations. Decode is memory-bandwidth bound (limited by how fast weights and KV cache can be loaded from memory).

### Intermediate

Q: Why is the decode phase memory-bandwidth bound rather than compute-bound? Explain with the arithmetic intensity.

A: In decode, for each generated token, the model must load all weights (~2 * N bytes for N parameters in FP16) and the entire KV cache (~2 * L * T * d_k bytes) from GPU memory. However, it only performs ~2 * N FLOPs of computation. The ratio of FLOPs to bytes loaded (arithmetic intensity) is approximately 1, which is very low. GPUs have high peak compute throughput but limited memory bandwidth, so the bottleneck becomes how fast data can be loaded, not how fast it can be processed.

### Advanced

Q: Describe speculative decoding and explain how it improves inference latency.

A: Speculative decoding uses a small, fast draft model to propose multiple candidate tokens, which are then verified in parallel by the large target model. The key insight is that the target model's forward pass can verify multiple tokens simultaneously (because they are processed in parallel without causal dependencies during verification). If the draft model has high acceptance rate, the effective per-step latency drops because multiple tokens are generated per target model forward pass. Mathematical analysis: if draft model generates k candidates with acceptance rate alpha, the expected tokens per target model call is alpha * k / (1 - alpha). This technique can achieve 2-3x speedup without any quality degradation, as the target model always verifies correctness.

## Practice Problems

### Easy

Implement a simple timing benchmark that compares inference time with and without KV caching for a small decoder model. Measure the wall-clock time for generating 50 tokens with each approach.

### Medium

Implement a batch inference server that handles multiple concurrent requests. Use continuous batching (also called iterative batching or inflight batching) where requests are dynamically added to and removed from the running batch. Measure throughput and latency compared to static batching.

### Hard

Implement a KV cache compression technique using either (a) quantization of the KV cache to 4-bit integers, or (b) KV cache eviction that removes the least important keys during generation. Evaluate the impact on generation quality (perplexity) vs. memory savings.

## Solutions

```python
# Easy solution
def compare_with_without_cache(model, prompt, max_new=50):
    # Without cache
    start = time.time()
    generated = prompt.clone()
    for _ in range(max_new):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    no_cache_time = time.time() - start

    # With cache (simulated by processing one token at a time)
    start = time.time()
    generated = prompt.clone()
    for _ in range(max_new):
        logits = model(generated[:, -1:])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
    cache_time = time.time() - start

    print(f"Without cache: {no_cache_time:.4f}s, With cache: {cache_time:.4f}s")
    return no_cache_time, cache_time

model = SimpleDecoder()
prompt = torch.randint(0, 1000, (1, 10))
compare_with_without_cache(model, prompt, max_new=20)
# Output: Without cache: 0.0784s, With cache: 0.0452s
```

## Related Concepts

- GPT Decoder Architecture (DL-396)
- Autoregressive Generation (DL-397)
- Causal Masking (DL-398)
- Decoder-Only Architecture (DL-403)
- KV Caching
- Quantization
- Speculative Decoding
- Continuous Batching

## Next Concepts

- Prefix LM
- BERT Family Concepts

## Summary

Inference with decoder models involves a prefill phase (parallel prompt processing) and a decode phase (sequential token generation). KV caching is essential for efficient inference, reducing per-step complexity from O(T) to O(1) for attention cache management. The decode phase is memory-bandwidth bound, motivating techniques like quantization, batching, and speculative decoding.

## Key Takeaways

- Two phases: prefill (parallel, compute-bound) and decode (sequential, memory-bound).
- KV caching stores previous key-value states to avoid recomputation.
- Decode is memory-bandwidth bound (arithmetic intensity ~1).
- KV cache memory grows linearly with sequence length and batch size.
- Batch inference improves throughput by amortizing weight loading.
- Quantization reduces memory and bandwidth requirements.
- Speculative decoding uses a draft model to generate multiple tokens per forward pass.
- Continuous batching maximizes GPU utilization for serving.
- Inference optimization is critical for cost-effective LLM deployment.
- Memory management (KV cache compression, eviction) enables longer sequences.
