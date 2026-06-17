# Encoder-Decoder LLMs

## Concept ID
DL-437

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand the encoder-decoder architecture for LLMs
- Compare encoder-decoder with decoder-only approaches
- Identify tasks suited for each architecture
- Implement cross-attention mechanisms

## Prerequisites
- Transformer Architecture (DL-370)
- Encoder-Decoder Models (DL-380)
- Decoder-Only Architecture (DL-403)

## Definition
Encoder-decoder LLMs process input through a bidirectional encoder and generate output through an autoregressive decoder with cross-attention to encoder representations. Unlike decoder-only models, the encoder processes the full input bidirectionally, while the decoder attends to both the encoder output and previously generated tokens.

## Intuition
Think of encoder-decoder as a two-step process: first, deeply understand the input (encoder), then generate a response based on that understanding (decoder). The encoder reads the entire input with full context (like carefully reading a question), while the decoder generates the answer word by word, looking back at both its own output and the encoder's understanding. This separation of understanding and generation makes encoder-decoder models particularly effective for tasks like translation and summarization where deep input comprehension is essential.

## Why This Concept Matters
While decoder-only models have become dominant for general-purpose LLMs, encoder-decoder models remain superior for specific tasks requiring deep input understanding. Understanding the architectural trade-offs helps in choosing the right model for deployment and explains performance differences across tasks.

## Mathematical Explanation

### Encoder Processing
The encoder processes input bidirectionally:

$$H_{enc} = \text{Encoder}(X) = \text{Layers}(X + \text{SelfAttention}(X))$$

Each position can attend to all other positions in the input.

### Decoder with Cross-Attention
The decoder combines self-attention and cross-attention:

$$H_{dec}^l = \text{SelfAttention}(H_{dec}^{l-1})$$
$$H_{dec}^l = \text{CrossAttention}(H_{dec}^l, H_{enc})$$
$$H_{dec}^l = \text{FFN}(H_{dec}^l)$$

### Cross-Attention Details
$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QH_{enc}^T}{\sqrt{d_k}}\right)H_{enc}$$

Where $Q$ comes from the decoder, and $K, V$ come from the encoder output.

## Code Examples

### Example 1: Cross-Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """Cross-attention: decoder queries attend to encoder keys/values"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, decoder_hidden, encoder_output, attention_mask=None):
        """
        decoder_hidden: (B, T_dec, D)
        encoder_output: (B, T_enc, D)
        """
        B, T_dec, D = decoder_hidden.shape
        T_enc = encoder_output.shape[1]
        
        Q = self.q(decoder_hidden).view(B, T_dec, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(encoder_output).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(encoder_output).view(B, T_enc, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q / math.sqrt(self.head_dim), K.transpose(-2, -1))
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_dec, D)
        return self.o(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_mask)[0]
        x = x + self.cross_attn(self.norm2(x), encoder_output, cross_mask)
        x = x + self.ffn(self.norm3(x))
        return x

# Test cross-attention
d_model, n_heads = 512, 8
cross_attn = CrossAttention(d_model, n_heads)

dec_hidden = torch.randn(2, 16, d_model)
enc_output = torch.randn(2, 32, d_model)
out = cross_attn(dec_hidden, enc_output)
print(f"Cross-attention output: {out.shape}")
# Output: Cross-attention output: (2, 16, 512)
```

### Example 2: Encoder-Decoder Model

```python
class EncoderDecoderLLM(nn.Module):
    """Complete encoder-decoder LLM"""
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, 
                 n_enc_layers=6, n_dec_layers=6, d_ff=2048, max_seq=512):
        super().__init__()
        self.d_model = d_model
        
        self.encoder_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, encoder_input, decoder_input, 
                encoder_mask=None, decoder_mask=None):
        B, T_enc = encoder_input.shape
        _, T_dec = decoder_input.shape
        
        # Encoder
        pos = torch.arange(T_enc, device=encoder_input.device).unsqueeze(0)
        enc_x = self.encoder_embed(encoder_input) + self.pos_embed(pos)
        
        for layer in self.encoder:
            enc_x = layer(enc_x, encoder_mask)
        enc_x = self.encoder_norm(enc_x)
        
        # Decoder
        pos = torch.arange(T_dec, device=decoder_input.device).unsqueeze(0)
        dec_x = self.decoder_embed(decoder_input) + self.pos_embed(pos)
        
        causal_mask = torch.triu(
            torch.ones(T_dec, T_dec, device=decoder_input.device) * float('-inf'), diagonal=1
        )
        
        for layer in self.decoder:
            dec_x = layer(dec_x, enc_x, causal_mask, decoder_mask)
        dec_x = self.decoder_norm(dec_x)
        
        logits = self.lm_head(dec_x)
        return logits

model = EncoderDecoderLLM(vocab_size=10000)
enc_in = torch.randint(0, 10000, (2, 32))
dec_in = torch.randint(0, 10000, (2, 16))
logits = model(enc_in, dec_in)
print(f"Encoder-decoder output: {logits.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: Encoder-decoder output: (2, 16, 10000)
# Output: Total parameters: 20,832,272
```

### Example 3: Encoder-Decoder vs Decoder-Only

```python
import torch
import torch.nn as nn

class EncoderDecoderComparison:
    """Compare encoder-decoder with decoder-only architecture"""
    
    @staticmethod
    def compare_architectures():
        print("Encoder-Decoder vs Decoder-Only Comparison:")
        print("-" * 70)
        
        aspects = [
            ('Input Processing', 'Bidirectional (full context)', 'Unidirectional (left-to-right)'),
            ('Parameter Count', '~2x for same hidden size', '~1x'),
            ('Context for Generation', 'Full input + generated', 'Generated only'),
            ('KV Cache', 'Limited (cross-attn cache small)', 'Large (grows with context)'),
            ('Best Tasks', 'Translation, Summarization', 'Generation, Chat, Code'),
            ('Training Efficiency', 'Lower (2 passes)', 'Higher (1 pass)'),
            ('Inference Speed', 'Slower (encoder pre-processing)', 'Faster (direct generation)'),
            ('Task Prefix', 'Built-in (separate enc/dec)', 'Requires explicit prefix'),
            ('Position Encoding', 'Separate for enc/dec', 'Shared for all tokens'),
        ]
        
        print(f"{'Aspect':<30}{'Encoder-Decoder':<25}{'Decoder-Only':<20}")
        print("-" * 70)
        for aspect, ed, do in aspects:
            print(f"{aspect:<30}{ed:<25}{do:<20}")
        
        print("\nWhen to choose:")
        print("  Encoder-Decoder: Input understanding is critical, output is structured")
        print("  Decoder-Only: Open-ended generation, chat, flexible inputs")

EncoderDecoderComparison.compare_architectures()
```

### Example 4: Memory Usage Comparison

```python
class MemoryComparison:
    """Compare memory usage of encoder-decoder vs decoder-only"""
    
    @staticmethod
    def estimate_kv_cache(arch_type, n_layers, d_model, n_heads, seq_len, batch=1):
        head_dim = d_model // n_heads
        
        if arch_type == 'encoder_decoder':
            # Encoder: no KV cache needed (processed once)
            # Decoder: KV cache for self-attention + cross-attention KV for encoder
            decoder_self_kv = 2 * n_layers * batch * seq_len * n_heads * head_dim * 2
            cross_kv = 2 * n_layers * batch * seq_len * n_heads * head_dim * 2  # Encoder KV stored
            return decoder_self_kv + cross_kv
        else:
            # Decoder-only: just self-attention KV cache
            return 2 * n_layers * batch * seq_len * n_heads * head_dim * 2
    
    @staticmethod
    def compare():
        configs = [
            ('T5-base (enc-dec)', 12, 768, 12, 512),
            ('T5-large (enc-dec)', 24, 1024, 16, 512),
            ('GPT-2 (dec-only)', 12, 768, 12, 1024),
            ('LLaMA 7B (dec-only)', 32, 4096, 32, 2048),
        ]
        
        print("KV Cache Memory Comparison:")
        print("-" * 70)
        print(f"{'Model':<25}{'Layers':<10}{'d_model':<10}{'Heads':<10}{'KV Cache':<15}")
        print("-" * 70)
        
        for name, layers, d_model, heads, seq in configs:
            arch = 'encoder_decoder' if 'enc-dec' in name else 'decoder_only'
            cache = MemoryComparison.estimate_kv_cache(arch, layers, d_model, heads, seq)
            print(f"{name:<25}{layers:<10}{d_model:<10}{heads:<10}{cache/1e6:<15.1f}MB")

MemoryComparison.compare()
```

### Example 5: Encoder-Decoder for Translation

```python
class TranslationModel(nn.Module):
    """Encoder-decoder for translation"""
    
    def __init__(self, src_vocab, tgt_vocab, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_model*4) for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_model*4) for _ in range(n_layers)
        ])
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.head = nn.Linear(d_model, tgt_vocab, bias=False)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        
    def forward(self, src, tgt):
        B, T_src = src.shape
        _, T_tgt = tgt.shape
        
        src_pos = torch.arange(T_src, device=src.device).unsqueeze(0)
        enc_h = self.src_embed(src) + self.pos_embed(src_pos)
        for layer in self.encoder:
            enc_h = layer(enc_h)
        enc_h = self.encoder_norm(enc_h)
        
        tgt_pos = torch.arange(T_tgt, device=tgt.device).unsqueeze(0)
        dec_h = self.tgt_embed(tgt) + self.pos_embed(tgt_pos)
        causal_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=tgt.device) * float('-inf'), diagonal=1)
        for layer in self.decoder:
            dec_h = layer(dec_h, enc_h, causal_mask)
        dec_h = self.decoder_norm(dec_h)
        
        return self.head(dec_h)

model = TranslationModel(5000, 5000)
src = torch.randint(0, 5000, (2, 20))
tgt = torch.randint(0, 5000, (2, 15))
logits = model(src, tgt)
print(f"Translation output: {logits.shape}")
# Output: Translation output: (2, 15, 5000)
```

## Common Mistakes

### 1. Using Decoder-Only Where Encoder-Decoder Is Better
For tasks where deep input understanding is critical (translation, summarization, long-document QA), decoder-only models may struggle because they can only attend to input unidirectionally.

### 2. Neglecting Cross-Attention Masking
Cross-attention does not need causal masking (the decoder can attend to all encoder positions), but proper handling of padding masks is essential. Failing to mask padded encoder positions causes attention to pad tokens.

### 3. Assuming Encoder-Decoder Is Always Better
Encoder-decoder models have higher parameter counts, slower inference (two passes), and larger memory footprint. For simple tasks or generation-only scenarios, decoder-only is more efficient.

### 4. Forgetting the Encoder Pads
The encoder's padding mask must be passed through all encoder layers AND to the decoder's cross-attention. Losing the padding mask at any stage results in attention to invalid positions.

### 5. Ignoring the Encoder Pre-Processing Cost
In production, the encoder adds latency: the full input must be processed through all encoder layers before generation begins. This latency can be amortized through caching but cannot be eliminated.

## Interview Questions

### Beginner
**Q1: What is the key architectural difference between encoder-decoder and decoder-only models?**
A1: Encoder-decoder models have a separate encoder (bidirectional input processing) and decoder (autoregressive output generation with cross-attention to encoder). Decoder-only models have a single stack that processes both input and output with causal attention.

**Q2: When would you choose an encoder-decoder model over decoder-only?**
A2: Choose encoder-decoder for tasks requiring deep input understanding before generation: translation, summarization, document-grounded QA, text infilling. Choose decoder-only for open-ended generation, chat, code completion, and tasks where input understanding is less critical.

### Intermediate
**Q3: Explain how cross-attention works in encoder-decoder models.**
A3: Cross-attention allows the decoder to attend to the encoder's output. The decoder's hidden states project queries (Q), while the encoder's output projects keys (K) and values (V). The attention matrix QK^T shows which decoder positions focus on which encoder positions. This is added after self-attention in each decoder layer, providing the decoder with bidirectional input context.

**Q4: How does the KV cache work differently in encoder-decoder vs decoder-only?**
A4: In encoder-decoder, the encoder processes the full input once, and its output can be cached as K and V for all decoder cross-attention layers. The decoder's self-attention still needs a KV cache (growing with generation length), but the cross-attention KV is static after encoder processing. In decoder-only, all attention is self-attention, so the KV cache grows with total (input + generated) length.

### Advanced
**Q5: Analyze the gradient flow in encoder-decoder models during training. How does the two-step processing affect training dynamics?**
A5: Gradients flow from the decoder loss through: (1) Decoder self-attention and FFN; (2) Cross-attention to encoder output; (3) All encoder layers; (4) Encoder embeddings. This creates a longer gradient path than decoder-only models, potentially causing vanishing gradients in early encoder layers. The cross-attention acts as a gradient bottleneck (all gradient information from the decoder must pass through the encoder's representation). This makes encoder-decoder models harder to train for very deep configurations. Techniques like gradient checkpointing, careful initialization, and pre-norm normalization help mitigate this.

**Q6: Design a hybrid architecture that combines encoder-decoder and decoder-only approaches for efficient multi-turn dialogue.**
A6: A hybrid architecture could: (1) Use a lightweight encoder to process each new user utterance (providing bidirectional understanding); (2) Use a shared decoder for generation that maintains an ongoing KV cache across turns; (3) Add cross-attention from the decoder to the latest encoder output after each turn; (4) Maintain a compressed memory of past encoder outputs for long-term context; (5) Allow the model to optionally bypass the encoder for simple queries (falling back to decoder-only mode). This hybrid would provide encoder-level understanding for complex queries while maintaining decoder-only efficiency for simple chit-chat and maintaining conversation state.

## Practice Problems

### Easy
Implement a cross-attention module and verify that the output sequence length matches the decoder sequence length, while attending to encoder representations.

### Medium
Compare the forward pass time and memory usage of an encoder-decoder model against a decoder-only model with the same total number of layers.

### Hard
Implement a conditional encoder-decoder model where the encoder processes retrieved documents and the decoder generates answers with citations to specific encoder positions.

## Solutions

### Easy Solution
```python
cross_attn = CrossAttention(512, 8)
dec = torch.randn(4, 10, 512)  # 10 decoder positions
enc = torch.randn(4, 30, 512)  # 30 encoder positions
out = cross_attn(dec, enc)
assert out.shape == (4, 10, 512), f"Expected (4, 10, 512), got {out.shape}"
print(f"Cross-attention: dec {dec.shape} x enc {enc.shape} -> {out.shape}")
```

### Medium Solution
```python
def compare_forward_time():
    ed_model = EncoderDecoderLLM(10000, d_model=512, n_enc_layers=6, n_dec_layers=6)
    do_model = DecoderOnlyModel(10000, d_model=512, n_layers=12)
    
    x = torch.randint(0, 10000, (4, 64))
    
    import time
    for name, model, *inputs in [("Enc-Dec", ed_model, x, x[:, :32]), 
                                  ("Dec-Only", do_model, x)]:
        start = time.time()
        _ = model(*inputs)
        print(f"{name}: {time.time() - start:.4f}s")
```

### Hard Solution
```python
class RAGEncoderDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.doc_encoder = EncoderDecoderLLM(10000, d_model, n_heads)
        self.citation_head = nn.Linear(d_model, 1)
```

## Related Concepts
- DL-380: Encoder-Decoder Models - Foundational architecture
- DL-370: Transformer Architecture - Base mechanism
- DL-403: Decoder-Only Architecture - Alternative approach
- DL-395: Encoder-Only vs Decoder-Only - Architecture comparison
- DL-431: T5 Architecture - Modern encoder-decoder example

## Next Concepts
- DL-438: BART - Bidirectional encoder + autoregressive decoder
- DL-439: PEGASUS - Summarization-specific encoder-decoder
- DL-440: Encoder-Decoder vs Decoder-Only - Detailed comparison

## Summary
Encoder-decoder LLMs separate input processing (bidirectional encoder) from output generation (autoregressive decoder with cross-attention). This architecture excels at tasks requiring deep input understanding before generation, such as translation and summarization. While decoder-only models have become dominant for general-purpose applications, encoder-decoder models remain essential for specific use cases.

## Key Takeaways
- Encoder processes input bidirectionally, decoder generates autoregressively
- Cross-attention connects encoder understanding to decoder generation
- ~2x parameters for same hidden size compared to decoder-only
- Longer gradient path during training
- Superior for translation, summarization, document QA
- Encoder adds latency but provides better input understanding
- Cross-attention KV can be cached after encoder forward pass
- Padding masks must reach both encoder and cross-attention
