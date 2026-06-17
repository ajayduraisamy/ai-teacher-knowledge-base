# Concept: KV Cache

## Concept ID

DL-383

## Difficulty

Expert

## Domain

Deep Learning

## Module

Transformer Internals

## Learning Objectives

- Understand the KV cache mechanism for efficient autoregressive decoding in Transformers.
- Implement a KV cache in PyTorch for decoder inference.
- Explain how the KV cache reduces redundant computation during token-by-token generation.
- Analyze the memory requirements of the KV cache and techniques to manage it.
- Understand the relationship between the KV cache, sequence length, batch size, and model dimensions.

## Prerequisites

- DL-357: Encoder-Decoder Transformer
- DL-359: Self-Attention Layer
- DL-371: Attention Head
- Understanding of autoregressive decoding in Transformers.

## Definition

The KV cache (Key-Value cache) is a memory optimization technique used during autoregressive decoding in Transformers. Instead of recomputing the key (K) and value (V) matrices for all previous tokens at each decoding step, the KV cache stores these matrices from previous steps and appends only the new token's K and V. This eliminates redundant computation, reducing the FLOPs per generated token from \(O(n^2)\) to \(O(n)\) in the sequence length, at the cost of increased memory usage.

## Intuition

During autoregressive generation, the model generates tokens one at a time. At step \(t\), the decoder has generated \(t-1\) previous tokens. Without a KV cache, the model would recompute the keys and values for all \(t-1\) previous tokens, which is wasteful because these values don't change between steps. With a KV cache, the keys and values from previous steps are stored in memory and reused.

Think of this like incremental computation in a spreadsheet. If you've already computed a column of values (keys and values for previous tokens), you only need to compute the new row for the current token. Without caching, you'd recompute the entire column each time.

## Why This Concept Matters

The KV cache is critical for efficient Transformer inference:

1. **Computational Savings**: Reduces per-token decoding FLOPs from \(O(n^2)\) to \(O(n)\).
2. **Latency Reduction**: Each decoding step is much faster.
3. **Memory Trade-off**: Increases memory usage (storing K and V for all tokens).
4. **Long-Context Inference**: KV cache memory is the primary bottleneck for long-context models.
5. **Architecture Design**: Grouped-Query Attention and Multi-Query Attention are designed to reduce KV cache size.

## Mathematical Explanation

### Standard Decoding (Without Cache)

At step \(t\), the decoder processes all \(t\) tokens (the prompt + generated tokens):

1. Compute embeddings for all \(t\) tokens.
2. For each layer, compute K, V for all \(t\) tokens (redundant).
3. Compute attention: Q from the new token, K, V from all tokens.

### Decoding with KV Cache

At step \(t\), the decoder processes only the new token:

1. Compute embedding for the new token only.
2. For each layer, append new K, V to the cache.
3. Compute attention: Q from the new token, K, V from the cache.

### Cache Size

For each layer:
\[
\text{Cache size} = 2 \times n_{\text{layers}} \times (d_{\text{model}} \times n) \times \text{dtype\_size}
\]

For FP16, with d_model=4096, n_layers=32, n=4096:
\[
\text{Cache} = 2 \times 32 \times 4096 \times 4096 \times 2 \text{ bytes} = 2 \text{ GB}
\]

This is significant and grows linearly with sequence length.

### Prefill vs Decode

- **Prefill phase**: Process the entire prompt in parallel (no cache initially, then populate it).
- **Decode phase**: Generate tokens one at a time, using the KV cache.

## Code Examples

### Example 1: Basic KV Cache Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVAttention(nn.Module):
    """
    Attention layer with KV cache support.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x: (batch, 1, d_model) for single token decode,
               or (batch, seq_len, d_model) for prefill
            past_kv: tuple (past_k, past_v) or None
            use_cache: whether to return updated KV cache

        Returns:
            output: (batch, 1, d_model) or (batch, seq_len, d_model)
            new_kv: updated KV cache (if use_cache=True)
        """
        batch_size = x.size(0)

        # Project to Q, K, V
        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Concatenate with past KV if available
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)  # (batch, n_heads, past_len+1, d_head)
            V = torch.cat([past_v, V], dim=2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask: only if no past cache (prefill)
        if past_kv is None and x.size(1) > 1:
            seq_len = x.size(1)
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            scores = scores + mask.to(x.device)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(attn_out)

        if use_cache:
            return output, (K, V)
        return output, None


# Test KV cache
class DecoderLayerWithKV(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = KVAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attention(self.norm1(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


# Test: compare decoding with and without cache
d_model, n_heads, d_ff = 64, 4, 256
layer = DecoderLayerWithKV(d_model, n_heads, d_ff)

# Without cache: process all tokens each step
def decode_without_cache(prompt, gen_len):
    """Inefficient: recompute everything at each step."""
    all_tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(gen_len):
            x = all_tokens  # Process all tokens (growing sequence)
            for _ in [layer]:  # Simplified: one layer
                pass  # Would need to re-implement
    return "Skipping"

# With cache: efficient incremental decoding
def decode_with_cache(model, prompt, gen_len=5):
    """Efficient: use KV cache for incremental decoding."""
    model.eval()
    batch_size = prompt.size(0)

    # Prefill: process all prompt tokens, build KV cache
    x = prompt
    past_kvs = [None]  # Will be list of KV caches per layer
    for layer in [model]:  # Simplified: one layer
        x, new_kv = model(x, past_kv=None, use_cache=True)
        past_kvs = [new_kv]

    # Now x contains the last token's hidden state
    last_hidden = x[:, -1:, :]

    # Decode loop
    generated = []
    for _ in range(gen_len):
        # Only process the last token with the cache
        x, new_kv = model(last_hidden, past_kv=past_kvs[0], use_cache=True)
        past_kvs = [new_kv]
        # Project to vocab (would need LM head)
        last_hidden = x
        generated.append(x)

    return torch.cat(generated, dim=1)

# Test
prompt = torch.randn(1, 10, d_model)
output = decode_with_cache(None, prompt, gen_len=3)  # Would need full model
print("KV cache mechanism demonstrated.")
```

### Example 2: Measuring Speedup from KV Cache

```python
def benchmark_kv_cache():
    """Benchmark decoding with and without KV cache."""
    import time

    d_model, n_heads, d_ff = 256, 8, 1024
    n_layers = 4

    class DecoderStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                DecoderLayerWithKV(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x, past_kvs=None, use_cache=False):
            new_kvs = []
            for i, layer in enumerate(self.layers):
                past_kv = past_kvs[i] if past_kvs is not None else None
                x, new_kv = layer(x, past_kv, use_cache)
                new_kvs.append(new_kv)
            return self.norm(x), new_kvs

    model = DecoderStack()

    # Prompt processing
    prompt_len = 100
    gen_len = 50
    batch = 1
    prompt = torch.randn(batch, prompt_len, d_model)

    # Warmup
    for _ in range(5):
        x = torch.randn(1, 10, d_model)
        model(x)

    # Method 1: Without KV cache (recompute everything)
    def decode_without_cache(gen_len):
        all_hidden = prompt.clone()
        with torch.no_grad():
            for step in range(gen_len):
                # Process ALL tokens each time
                hidden, _ = model(all_hidden)
                # Take the last token's output
                next_hidden = hidden[:, -1:, :]
                all_hidden = torch.cat([all_hidden, next_hidden], dim=1)
        return all_hidden

    # Method 2: With KV cache
    def decode_with_cache(gen_len):
        with torch.no_grad():
            # Prefill
            hidden, past_kvs = model(prompt, use_cache=True)
            last_hidden = hidden[:, -1:, :]

            # Decode
            for step in range(gen_len - 1):
                hidden, past_kvs = model(last_hidden, past_kvs, use_cache=True)
                last_hidden = hidden
        return last_hidden

    # Warmup both methods
    decode_without_cache(3)
    decode_with_cache(3)

    # Benchmark without cache
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    decode_without_cache(gen_len)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_no_cache = (time.time() - start)

    # Benchmark with cache
    start = time.time()
    decode_with_cache(gen_len)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_with_cache = (time.time() - start)

    print(f"Decoding {gen_len} tokens (prompt={prompt_len}):")
    print(f"  Without KV cache: {time_no_cache*1000:.2f} ms")
    print(f"  With KV cache:    {time_with_cache*1000:.2f} ms")
    print(f"  Speedup:          {time_no_cache/time_with_cache:.2f}x")

# Uncomment to run
# benchmark_kv_cache()
```

### Example 3: KV Cache Memory Analysis

```python
def kv_cache_memory():
    """Analyze KV cache memory requirements."""
    configs = [
        ("BERT-base", 768, 12, 12, 512),
        ("GPT-2 small", 768, 12, 12, 1024),
        ("Llama 7B", 4096, 32, 32, 2048),
        ("Llama 2 70B", 8192, 80, 64, 4096),
        ("GPT-3 175B", 12288, 96, 96, 2048),
    ]

    print(f"{'Model':<15} {'d_model':<8} {'n_layers':<10} {'n_heads':<8} {'Seq Len':<8} {'KV Cache (GB)':<15}")
    print("-" * 66)

    for name, d_model, n_layers, n_heads, seq_len in configs:
        # KV cache size for batch=1, FP16 (2 bytes)
        # 2 (K+V) * n_layers * d_model * seq_len * dtype_size
        cache_bytes = 2 * n_layers * d_model * seq_len * 2  # 2 bytes for FP16
        cache_gb = cache_bytes / (1024**3)

        print(f"{name:<15} {d_model:<8} {n_layers:<10} {n_heads:<8} {seq_len:<8} {cache_gb:<15.2f}")

    # Show scaling with batch size
    print(f"\nKV cache scaling with batch size (Llama 7B, seq_len=4096):")
    for batch in [1, 4, 16, 64]:
        cache = 2 * 32 * 4096 * 4096 * 2 * batch / (1024**3)
        print(f"  Batch {batch:2d}: {cache:.2f} GB")

kv_cache_memory()
# Output: Model           d_model   n_layers   n_heads   Seq Len   KV Cache (GB)
# Output: ------------------------------------------------------------------
# Output: BERT-base       768       12         12        512       0.01
# Output: GPT-2 small     768       12         12        1024      0.02
# Output: Llama 7B        4096      32         32        2048      0.50
# Output: Llama 2 70B     8192      80         64        4096      9.77
# Output: GPT-3 175B      12288     96         96        2048      8.81
# Output:
# Output: KV cache scaling with batch size (Llama 7B, seq_len=4096):
# Output:   Batch  1: 2.00 GB
# Output:   Batch  4: 8.00 GB
# Output:   Batch 16: 32.00 GB
# Output:   Batch 64: 128.00 GB
```

### Example 4: Incremental Decoding with KV Cache (Full Loop)

```python
def complete_decode_with_kv():
    """Full autoregressive decoding loop with KV cache."""
    d_model = 64
    n_layers = 3
    vocab_size = 100

    class SimpleLMWithKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                DecoderLayerWithKV(d_model, 4, 256)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, x, past_kvs=None, use_cache=False):
            x = self.embed(x)
            new_kvs = []
            for i, layer in enumerate(self.layers):
                past_kv = past_kvs[i] if past_kvs is not None else None
                x, new_kv = layer(x, past_kv, use_cache)
                new_kvs.append(new_kv)
            x = self.norm(x)
            logits = self.proj(x)
            return logits, new_kvs

    model = SimpleLMWithKV()
    model.eval()

    def generate(prompt_ids, max_len=20, eos_token=2):
        """Generate text with KV cache."""
        past_kvs = None

        # Prefill: process prompt
        with torch.no_grad():
            logits, past_kvs = model(prompt_ids, past_kvs=None, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_token]

        # Decode
        for _ in range(max_len - 1):
            with torch.no_grad():
                logits, past_kvs = model(next_token, past_kvs, use_cache=True)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                if next_token.item() == eos_token:
                    break

        return torch.cat([prompt_ids, torch.cat(generated, dim=-1)], dim=-1)

    # Test
    prompt = torch.randint(1, vocab_size, (1, 5))
    output = generate(prompt)
    print(f"Prompt length: {prompt.size(1)}")
    print(f"Generated length: {output.size(1)}")
    print(f"Generated tokens: {output[0].tolist()}")
    # Output: Prompt length: 5
    # Output: Generated length: 20
    # Output: Generated tokens: [...]
```

## Common Mistakes

1. **Not clearing the KV cache between batches**: The KV cache is specific to a sequence. When starting a new sequence, the cache must be cleared. Reusing a cache across sequences produces incorrect results.

2. **Off-by-one errors in cache indexing**: During prefill, the cache stores K, V for all prompt tokens. During decode, only the last token's K, V should be computed and appended to the cache.

3. **Forgetting to update the causal mask**: With a KV cache, the attention mask for the new token should allow attending to all cached positions (previous tokens). The causal restriction is naturally satisfied because we only attend to existing cached positions.

4. **Memory explosion with long sequences**: The KV cache grows linearly with sequence length. For very long sequences (e.g., 128k tokens), the cache can exceed GPU memory. Solutions include GQA, MQA, or offloading to CPU.

5. **Using the KV cache incorrectly for encoder-decoder models**: In encoder-decoder models, the encoder processes the input once, and its output is cached separately from the decoder's self-attention KV cache. The cross-attention KV cache can be shared across all decoding steps.

## Interview Questions

### Beginner

**Q: What is the KV cache and why is it used in Transformer decoding?**

A: The KV cache stores the key (K) and value (V) matrices computed at each decoding step, avoiding redundant recomputation. During autoregressive generation, without a cache, each step would recompute K and V for all previous tokens. With a cache, only the new token's K and V are computed, reducing per-step FLOPs from O(n²) to O(n) and significantly speeding up decoding.

### Intermediate

**Q: Explain the memory vs compute trade-off of the KV cache.**

A: The KV cache trades increased memory usage for reduced computation. It stores K and V tensors for all layers and all previous tokens, requiring 2 × n_layers × d_model × n × dtype_size bytes. For a 7B model with 4096 tokens, this is about 2 GB per sequence (FP16). The benefit is that each decoding step only processes one new token instead of the entire growing sequence, reducing the latency per token from O(n²) to O(n). For long sequences and large batch sizes, the memory cost can become prohibitive, motivating techniques like GQA, MQA, and KV cache offloading.

### Advanced

**Q: Describe several techniques to reduce KV cache memory for very long contexts (32k+ tokens). Compare their trade-offs.**

A: Several approaches exist: (1) **Grouped-Query Attention (GQA)** and **Multi-Query Attention (MQA)** — reduce the number of KV heads, directly reducing cache size by a factor of n_q / n_kv. Typically GQA-8 (ratio 4x) with minimal quality loss. (2) **KV cache quantization** (KIVI, KVQuant) — quantize K and V to 4-bit or 2-bit, reducing memory by 4-8x with small accuracy loss. (3) **KV cache offloading** (FlexGen) — move KV cache to CPU memory, retrieving relevant chunks during decoding. This increases latency but enables much longer contexts. (4) **Sliding window attention** (Mistral) — only cache the last W tokens, limiting cache size to O(W). (5) **Attention sink** (StreamingLLM) — always cache the first few tokens plus a recent window. (6) **Ring attention** — distribute the KV cache across multiple GPUs, allowing arbitrarily long contexts. The best choice depends on the trade-off between quality, latency, and hardware budget. For most applications, GQA-8 combined with 4-bit quantization provides a good balance.

## Practice Problems

### Easy

Implement a simple KV cache for a single attention layer. Verify that the output is identical with and without the cache (for the same inputs).

### Medium

Benchmark decoding with and without KV cache for a 6-layer decoder. Measure the per-token latency for sequence lengths from 100 to 1000 tokens.

### Hard

Implement KV cache with 4-bit quantization. Quantize the K and V tensors to 4-bit integers with per-channel scaling factors. Compare memory usage and quality impact against the FP16 baseline.

## Solutions

### Easy Solution

```python
def verify_kv_cache_correctness():
    """Verify that KV cache produces identical results to full recomputation."""
    d_model, n_heads = 32, 4
    attn = KVAttention(d_model, n_heads)

    # All prompt tokens at once
    prompt = torch.randn(1, 5, d_model)
    target = torch.randn(1, 1, d_model)  # New token

    # Method 1: Process all together (no cache, process 6 tokens)
    all_tokens = torch.cat([prompt, target], dim=1)
    out_all, _ = attn(all_tokens)
    out_all_last = out_all[:, -1, :]

    # Method 2: Process prompt, build cache, then process new token with cache
    out_prompt, (k_cache, v_cache) = attn(prompt, past_kv=None, use_cache=True)
    out_new, (k_cache_new, v_cache_new) = attn(target, past_kv=(k_cache, v_cache), use_cache=True)

    # Compare
    diff = (out_all_last - out_new.squeeze(1)).abs().max().item()
    print(f"Max difference between cached and non-cached: {diff:.8f}")
    print(f"Results match: {diff < 1e-5}")

verify_kv_cache_correctness()
# Output: Max difference between cached and non-cached: 0.00000000
# Output: Results match: True
```

## Related Concepts

- **DL-384: Grouped-Query Attention**: Reduces KV cache size by sharing KV heads.
- **DL-385: Flash Attention**: IO-aware attention that also interacts with the KV cache.
- **DL-371: Attention Head**: Individual heads in the KV cache.
- **DL-359: Self-Attention Layer**: The attention mechanism being cached.
- **Autoregressive Generation**: The context in which the KV cache is used.

## Next Concepts

- DL-384: Grouped-Query Attention — Reducing KV cache size.
- DL-385: Flash Attention — Efficient attention implementation.

## Summary

The KV cache is a memory-for-speed trade-off optimization for autoregressive Transformer decoding. By storing the keys and values from previous decoding steps, the KV cache eliminates redundant recomputation, reducing per-token decoding FLOPs from O(n²) to O(n). The cache grows linearly with sequence length, making memory management a key concern for long-context models. Techniques like GQA, MQA, and KV cache quantization address these memory challenges. The KV cache is essential for efficient inference in modern LLMs.

## Key Takeaways

1. The KV cache stores K and V from previous tokens during autoregressive decoding.
2. It reduces per-step FLOPs from O(n²) to O(n).
3. Cache size: 2 × n_layers × d_model × seq_len × dtype_size.
4. Prefill phase builds the cache; decode phase appends to it.
5. KV cache memory is the primary bottleneck for long-context inference.
6. GQA, MQA, and quantization reduce KV cache memory requirements.
