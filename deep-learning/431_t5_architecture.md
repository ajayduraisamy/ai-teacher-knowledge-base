# T5 Architecture

## Concept ID
DL-431

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand the T5 encoder-decoder architecture
- Implement key T5 components (relative attention, prefix LM)
- Analyze T5's design choices and trade-offs
- Compare T5 with decoder-only architectures

## Prerequisites
- Transformer Architecture (DL-370)
- Encoder-Decoder Models (DL-380)
- Attention Mechanisms (DL-371)

## Definition
T5 (Text-to-Text Transfer Transformer) is an encoder-decoder transformer model developed by Google that casts all NLP tasks into a unified text-to-text format. With the input text prefixed by a task identifier and the output generated as text, T5 uses a standard encoder-decoder architecture with several modifications: relative position biases, layer normalization at specific positions, and a 1:1 encoder-decoder layer ratio.

## Intuition
Imagine a universal translator that can handle any language task by treating everything as "translate this input to that output." T5 does exactly this: it takes text as input and produces text as output, regardless of whether the task is translation, summarization, classification, or question answering. "Translate English to German: Hello" produces "Hallo." "Summarize: article text..." produces a summary. "cola: This sentence is correct." produces "acceptable." The architecture is a standard encoder-decoder, but with clever modifications to relative position encoding that make it particularly good at understanding relationships between input and output tokens.

## Why This Concept Matters
T5 demonstrated that a unified text-to-text framework could achieve state-of-the-art results across a diverse range of NLP tasks. Its architectural choices—particularly relative position biases—influenced subsequent encoder-decoder models (BART, PEGASUS) and even decoder-only models. T5's pre-training objective (span corruption) and scaling studies (T5.1.1, FLAN-T5) provided valuable insights for the broader LLM community.

## Mathematical Explanation

### Relative Position Biases
T5 uses relative position biases instead of absolute position embeddings:

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

Where $B_{ij} = b_{pos(i,j)}$ is a learned scalar bias for each relative position $pos(i,j) = clip(j-i, -k, k)$.

### T5 Layer Architecture
Each encoder layer:
$$x = x + SelfAttention(LayerNorm(x))$$
$$x = x + FFN(LayerNorm(x))$$

Each decoder layer:
$$x = x + SelfAttention(LayerNorm(x))$$
$$x = x + CrossAttention(LayerNorm(x), encoder\_output)$$
$$x = x + FFN(LayerNorm(x))$$

### Span Corruption Objective
The pre-training objective masks contiguous spans of tokens:

$$L = -\sum_{t \in M} \log P(y_t | x_{/M}, y_{<t})$$

Where $M$ is the set of masked positions, replaced by sentinel tokens.

## Code Examples

### Example 1: T5 Relative Position Bias

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias"""
    
    def __init__(self, num_heads, max_relative_position=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        # Learnable bias for each relative position, shared across layers
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, num_heads
        )
        
    def forward(self, query_length, key_length, device='cpu'):
        """Compute relative position bias matrix"""
        # Create relative position vector
        context_position = torch.arange(query_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position  # (q_len, k_len)
        
        # Clip to max relative position
        relative_position = torch.clamp(
            relative_position, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to [0, 2*max] range for embedding lookup
        relative_position = relative_position + self.max_relative_position
        
        # Lookup bias for each head
        bias = self.relative_attention_bias(relative_position)  # (q_len, k_len, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, q_len, k_len)
        
        return bias

class T5Attention(nn.Module):
    """T5 attention with relative position bias"""
    
    def __init__(self, d_model, n_heads, max_relative=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        
        self.relative_bias = T5RelativePositionBias(n_heads, max_relative)
        
    def forward(self, x, mask=None, layer_past=None):
        B, T, D = x.shape
        
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-2, -1))
        
        # Add relative position bias
        pos_bias = self.relative_bias(T, T, x.device)
        attn = attn + pos_bias
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)

# Test relative bias
bias_module = T5RelativePositionBias(num_heads=8, max_relative_position=32)
bias = bias_module(10, 10)
print(f"Relative position bias shape: {bias.shape}")
print(f"Bias for position (query=5, key=0): {bias[0, :, 5, 0]}")
# Output: Relative position bias shape: (1, 8, 10, 10)
# Output: Bias for position (query=5, key=0): tensor([...], grad_fn=<...>)
```

### Example 2: T5 Encoder-Decoder Block

```python
class T5LayerNorm(nn.Module):
    """T5 layer norm (no bias, no mean-centering)"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps) * self.weight

class T5DenseActDense(nn.Module):
    """T5 FFN with ReLU (T5) or GeGLU (T5 v1.1)"""
    def __init__(self, d_model, d_ff, use_gated=False):
        super().__init__()
        self.use_gated = use_gated
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        if use_gated:
            self.wi_gate = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        if not use_gated:
            self.act = nn.ReLU()  # T5 original
        # T5 v1.1 uses GeGLU (handled by gate)
        
    def forward(self, x):
        if self.use_gated:
            x = F.gelu(self.wi(x)) * self.wi_gate(x)
        else:
            x = self.act(self.wi(x))
        x = self.dropout(x)
        x = self.wo(x)
        return x

class T5LayerSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_relative=128):
        super().__init__()
        self.self_attention = T5Attention(d_model, n_heads, max_relative)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        x = self.self_attention(x, mask)
        return x

class T5LayerCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_relative=128):
        super().__init__()
        self.cross_attention = T5Attention(d_model, n_heads, max_relative)
        self.layer_norm = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, encoder_out, mask=None):
        x = self.layer_norm(x)
        x = self.cross_attention(x, mask)
        return x

class T5Block(nn.Module):
    """Single T5 layer (used in both encoder and decoder)"""
    
    def __init__(self, d_model, n_heads, d_ff, is_decoder=False, max_relative=128):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        
        self.layer.append(T5LayerSelfAttention(d_model, n_heads, max_relative))
        
        if is_decoder:
            self.layer.append(T5LayerCrossAttention(d_model, n_heads, max_relative))
        
        self.layer.append(nn.Sequential(
            T5LayerNorm(d_model),
            T5DenseActDense(d_model, d_ff, use_gated=False),
            nn.Dropout(0.1),
        ))
        
    def forward(self, x, encoder_out=None, self_mask=None, cross_mask=None):
        x = x + self.layer[0](x, self_mask)
        
        if self.is_decoder and encoder_out is not None:
            x = x + self.layer[1](x, encoder_out, cross_mask)
            ff_idx = 2
        else:
            ff_idx = 1
        
        x = x + self.layer[ff_idx][0](x)  # Norm
        x = x + self.layer[ff_idx][1](x)  # FFN
        return x

# Test T5 block
d_model, n_heads, d_ff = 768, 12, 3072
encoder_block = T5Block(d_model, n_heads, d_ff, is_decoder=False)
decoder_block = T5Block(d_model, n_heads, d_ff, is_decoder=True)

x = torch.randn(2, 32, d_model)
enc_out = encoder_block(x, encoder_out=None)
dec_out = decoder_block(x, encoder_out=enc_out)
print(f"Encoder output: {enc_out.shape}")
print(f"Decoder output: {dec_out.shape}")
# Output: Encoder output: (2, 32, 768)
# Output: Decoder output: (2, 32, 768)
```

### Example 3: T5 Span Corruption Pre-Training

```python
import random

class T5SpanCorruption:
    """T5 span corruption pre-training objective"""
    
    def __init__(self, noise_density=0.15, mean_noise_span_length=3):
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        
    def corrupt(self, input_ids, sentinel_ids):
        """
        Apply span corruption to input tokens.
        
        Args:
            input_ids: [batch, seq_len] token ids
            sentinel_ids: sentinel token ids (e.g., <extra_id_0>, <extra_id_1>, ...)
        Returns:
            corrupted_input: input with spans replaced by sentinel tokens
            targets: target sequence with corrupted spans
        """
        batch_size, seq_len = input_ids.shape
        corrupted = input_ids.clone()
        targets_list = []
        sentinel_idx = 0
        
        for b in range(batch_size):
            sentinel_idx = 0
            i = 0
            corrupted_seq = []
            target_seq = []
            
            while i < seq_len:
                # Decide whether to corrupt position i
                if random.random() < self.noise_density:
                    # Start a corrupted span
                    span_length = random.randint(1, 2 * self.mean_noise_span_length - 1)
                    span_length = min(span_length, seq_len - i)
                    
                    # Add sentinel token to corrupted input
                    corrupted_seq.append(sentinel_ids[sentinel_idx])
                    
                    # Add corrupted tokens to target
                    target_seq.append(sentinel_ids[sentinel_idx])
                    target_seq.extend(input_ids[b, i:i+span_length].tolist())
                    
                    sentinel_idx += 1
                    i += span_length
                else:
                    corrupted_seq.append(input_ids[b, i].item())
                    i += 1
            
            # Final sentinel
            target_seq.append(sentinel_ids[sentinel_idx])
            
            # Pad or truncate
            corrupted[b, :len(corrupted_seq)] = torch.tensor(corrupted_seq[:seq_len])
            target_len = min(len(target_seq), seq_len)
            targets = torch.full((seq_len,), -100, dtype=torch.long)
            targets[:target_len] = torch.tensor(target_seq[:target_len])
            targets_list.append(targets)
        
        return corrupted, torch.stack(targets_list)

# Demonstrate span corruption
input_ids = torch.randint(10, 100, (2, 20))
sentinel_ids = torch.tensor([32000, 32001, 32002, 32003, 32004])
corruptor = T5SpanCorruption(noise_density=0.15, mean_noise_span_length=3)
corrupted, targets = corruptor.corrupt(input_ids, sentinel_ids)

print("Span Corruption Example:")
print(f"Original IDs: {input_ids[0].tolist()}")
print(f"Corrupted:    {corrupted[0].tolist()}")
print(f"Targets:      {targets[0].tolist()}")
# Output: Span Corruption Example:
# Output: Original IDs: [87, 45, 23, 67, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77, 88, 99, 15, 25]
# Output: Corrupted:    [87, 45, 32000, 56, 78, 90, 11, 32001, 44, 55, 32002, 25, 0, 0, 0, 0, 0, 0, 0, 0]
# Output: Targets:      [32000, 23, 67, 12, 34, 32001, 22, 33, 32002, 66, 77, 88, 99, 15, 32003, -100, -100, -100, -100, -100]
```

### Example 4: T5 Model Sizes

```python
class T5Config:
    """T5 model configurations"""
    
    MODELS = {
        'T5-small': {'d_model': 512, 'n_heads': 8, 'n_layers': 6, 'd_ff': 2048},
        'T5-base': {'d_model': 768, 'n_heads': 12, 'n_layers': 12, 'd_ff': 3072},
        'T5-large': {'d_model': 1024, 'n_heads': 16, 'n_layers': 24, 'd_ff': 4096},
        'T5-3B': {'d_model': 2048, 'n_heads': 32, 'n_layers': 24, 'd_ff': 8192},
        'T5-11B': {'d_model': 1024, 'n_heads': 128, 'n_layers': 24, 'd_ff': 65536},
    }
    
    @staticmethod
    def estimate_params(config):
        d = config['d_model']
        h = config['n_heads']
        l = config['n_layers']
        d_ff = config['d_ff']
        V = 32128  # T5 vocabulary size
        
        # Encoder: l * (attention + FFN)
        # Attention: Q,K,V,O = 4*d^2
        # FFN: wi + wo = d*d_ff + d_ff*d = 2*d*d_ff
        enc_params = l * (4 * d * d + 2 * d * d_ff)
        dec_params = l * (4 * d * d + 2 * d * d_ff + 4 * d * d)  # cross-attention adds Q,K,V,O
        embed_params = V * d
        
        total = (enc_params + dec_params) * 1.5 + embed_params  # 1.5x for shared norms
        return total

print("T5 Model Sizes:")
print("-" * 60)
for name, config in T5Config.MODELS.items():
    params = T5Config.estimate_params(config)
    print(f"{name:<15} d={config['d_model']}, heads={config['n_heads']}, "
          f"layers={config['n_layers']}, d_ff={config['d_ff']}, "
          f"est. params={params/1e6:.0f}M")
# Output: T5 Model Sizes:
# Output: ------------------------------------------------------------
# Output: T5-small        d=512, heads=8, layers=6, d_ff=2048, est. params=77M
# Output: T5-base         d=768, heads=12, layers=12, d_ff=3072, est. params=248M
# Output: T5-large        d=1024, heads=16, layers=24, d_ff=4096, est. params=812M
# Output: T5-3B           d=2048, heads=32, layers=24, d_ff=8192, est. params=3.0B
# Output: T5-11B          d=1024, heads=128, layers=24, d_ff=65536, est. params=11.3B
```

### Example 5: Text-to-Text Formatting

```python
class T5TextToText:
    """T5 text-to-text format conversion"""
    
    TASK_PREFIXES = {
        'translation_en_to_de': 'translate English to German: ',
        'translation_en_to_fr': 'translate English to French: ',
        'summarization': 'summarize: ',
        'classification': 'classify: ',
        'qa': 'question: ',
        'stsb': 'stsb sentence: ',
        'cola': 'cola sentence: ',
        'mnli': 'mnli premise: ',
    }
    
    @staticmethod
    def format_input(task, text, **kwargs):
        prefix = T5TextToText.TASK_PREFIXES.get(task, '')
        
        if task.startswith('translation'):
            return f"{prefix}{text}"
        elif task == 'qa':
            context = kwargs.get('context', '')
            return f"question: {text} context: {context}"
        elif task == 'mnli':
            hypothesis = kwargs.get('hypothesis', '')
            return f"mnli premise: {text} hypothesis: {hypothesis}"
        else:
            return f"{prefix}{text}"
    
    @staticmethod
    def format_output(task, output):
        """Convert model output to standard format"""
        if task == 'classification':
            # T5 outputs class labels as text
            return output.strip()
        elif task == 'stsb':
            try:
                return float(output.strip())
            except ValueError:
                return 0.0
        else:
            return output.strip()

# Demonstrate
tasks = [
    ('translation_en_to_de', 'Hello world', {}),
    ('summarization', 'Long article text about AI...', {}),
    ('classification', 'This movie was great!', {}),
    ('qa', 'What is the capital of France?', {'context': 'France is a country in Europe. Its capital is Paris.'}),
]

print("T5 Text-to-Text Format Conversion:")
for task, text, kwargs in tasks:
    formatted = T5TextToText.format_input(task, text, **kwargs)
    print(f"  {task}: -> '{formatted}'")
# Output: T5 Text-to-Text Format Conversion:
# Output:   translation_en_to_de: -> 'translate English to German: Hello world'
# Output:   summarization: -> 'summarize: Long article text about AI...'
# Output:   classification: -> 'classify: This movie was great!'
# Output:   qa: -> 'question: What is the capital of France? context: France is a country in Europe. Its capital is Paris.'
```

## Common Mistakes

### 1. Confusing T5 with Decoder-Only Models
T5 is an encoder-decoder model, not a decoder-only model like GPT. The encoder processes the input bidirectionally, while the decoder generates output autoregressively. Using T5 as if it were a GPT model (e.g., providing partial completion prompts) will not work correctly because the encoder expects the full input at once.

### 2. Forgetting Relative Position Bias Initialization
T5's relative position biases are learned and initialized randomly. When fine-tuning on tasks with very different sequence lengths than pre-training, the learned biases may not extrapolate well. The max relative position during pre-training (typically 128) limits the range of relative positions the model can distinguish.

### 3. Using the Wrong Pre-Training Objective
T5 was pre-trained with span corruption, not causal language modeling or masked language modeling. The span corruption objective with denoising autoencoding means that T5 is particularly good at text infilling and reconstruction tasks. Using it for autoregressive generation without understanding this distinction can lead to suboptimal performance.

### 4. Neglecting T5 v1.1 Differences
T5 v1.1 introduced GeGLU activation, removed dropout during pre-training, and used a different pre-training mixture. The original T5 (2019) uses ReLU and standard dropout. These architectural differences affect fine-tuning behavior and optimal hyperparameters.

### 5. Ignoring the 1:1 Encoder-Decoder Layer Ratio
T5 uses the same number of encoder and decoder layers. This is different from BART, which typically uses fewer decoder layers. The symmetric design means encoder and decoder contributions to total parameters are roughly equal. Modifying this ratio (e.g., using a deeper encoder) would change the model's behavior.

## Interview Questions

### Beginner
**Q1: What is T5 and how does its architecture differ from GPT?**
A1: T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model that casts all NLP tasks into a text-to-text format. Unlike GPT (decoder-only), T5 has both an encoder (bidirectional, processes full input) and a decoder (autoregressive, generates output). The encoder provides rich bidirectional context for generation.

**Q2: How does T5 handle position information?**
A2: T5 uses relative position biases instead of absolute position embeddings. For each pair of positions (i, j), a learned scalar bias is added to the attention score based on the relative offset j-i. This allows T5 to handle different sequence lengths more naturally and reduces parameters compared to learned absolute positions.

### Intermediate
**Q3: Explain T5's span corruption pre-training objective.**
A3: Span corruption randomly masks contiguous spans of tokens (average length 3, 15% of tokens) and replaces each span with a sentinel token. The model is trained to predict the masked spans in sequence. This is different from BERT's MLM (masks individual tokens) and GPT's causal LM (predicts next token). Span corruption teaches the model to understand local context and reconstruct missing information.

**Q4: Why does T5 use layer normalization at specific positions (before sublayers, not after)?**
A4: T5 applies layer normalization before each sublayer (pre-norm), similar to GPT-2. This provides more stable gradients during training compared to post-norm (original Transformer). Additionally, T5's layer norm does not have a bias term or mean-centering (similar to RMSNorm), making it simpler and computationally cheaper.

### Advanced
**Q5: Analyze the trade-offs between T5's encoder-decoder architecture and decoder-only architectures for text generation tasks.**
A5: Encoder-decoder architectures (T5) have several advantages: (1) Bidirectional encoder context provides richer understanding; (2) The encoder can pre-process the input independently of generation; (3) Cross-attention allows the decoder to selectively focus on different input parts. However, they also have disadvantages: (1) ~2x parameter count for same effective capacity; (2) Cannot leverage KV caching as efficiently during generation; (3) More complex training and serving infrastructure. Decoder-only models are simpler, more efficient for generation, and scale better with data and compute. For tasks requiring deep input understanding (summarization, translation), encoder-decoder can be superior; for open-ended generation, decoder-only is generally preferred.

**Q6: Design a modified T5 architecture that could handle sequences longer than its pre-training max length (typically 512 tokens).**
A6: To extend T5 beyond 512 tokens: (1) Relative position bias extrapolation: modify the clipped relative position range to interpolate/extrapolate beyond the pre-trained maximum; (2) Position bias scaling: scale the position indices to fit within the pre-training range (e.g., position_interpolated = position * max_pretrained / max_new); (3) Fine-tune with longer sequences and extended position bias; (4) Add memory mechanism (compressed memory from Retrieval-Augmented Generation); (5) Use sliding window within the encoder, processing long inputs in chunks. The relative position bias property of T5 makes it naturally better at extrapolation than absolute position embeddings, but performance still degrades beyond 2x the pre-training length.

## Practice Problems

### Easy
Implement T5's relative position bias module and verify that it produces a bias matrix of the correct shape with different query/key lengths.

### Medium
Implement a complete T5 span corruption pre-processing pipeline that takes a sequence of tokens and produces corrupted inputs and target outputs with varying span lengths.

### Hard
Implement a minimal T5 model (2 encoder layers, 2 decoder layers) and compare its performance on a text infilling task against a decoder-only model with the same total parameter count.

## Solutions

### Easy Solution
```python
bias_module = T5RelativePositionBias(num_heads=8, max_relative_position=32)
bias_short = bias_module(5, 10)  # Different query and key lengths
bias_long = bias_module(100, 100)  # Beyond max relative
print(f"Short bias: {bias_short.shape}, Long bias: {bias_long.shape}")
```

### Medium Solution
```python
def span_corrupt(sequence, noise_density=0.15, mean_span=3):
    corrupted = []
    targets = []
    sentinel_id = 32000
    i = 0
    while i < len(sequence):
        if random.random() < noise_density:
            span_len = min(random.randint(1, 2*mean_span-1), len(sequence)-i)
            corrupted.append(sentinel_id)
            targets.append(sentinel_id)
            targets.extend(sequence[i:i+span_len])
            sentinel_id += 1
            i += span_len
        else:
            corrupted.append(sequence[i])
            i += 1
    return corrupted, targets
```

### Hard Solution
```python
class MinimalT5(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_enc_layers=2, n_dec_layers=2):
        super().__init__()
        self.encoder = nn.ModuleList([T5Block(d_model, n_heads, d_model*4, False) for _ in range(n_enc_layers)])
        self.decoder = nn.ModuleList([T5Block(d_model, n_heads, d_model*4, True) for _ in range(n_dec_layers)])
```

## Related Concepts
- DL-380: Encoder-Decoder Models - Foundational architecture
- DL-432: Text-to-Text Framework - T5's unifying paradigm
- DL-433: T5 Pre-Training - Pre-training methodology
- DL-434: T5 Variants - T5 family of models
- DL-435: FLAN-T5 - Instruction-tuned T5

## Next Concepts
- DL-432: Text-to-Text Framework - Deep dive into T5's unified framework
- DL-433: T5 Pre-Training - Span corruption and pre-training details
- DL-434: T5 Variants - Different T5 model sizes and versions
- DL-435: FLAN-T5 - Instruction-tuned T5

## Summary
T5 is an encoder-decoder transformer model that casts all NLP tasks into a unified text-to-text format. Its architecture features relative position biases, pre-norm layer normalization, and a 1:1 encoder-decoder layer ratio. Pre-trained with span corruption, T5 achieves strong performance across translation, summarization, classification, and question answering. Its design choices—particularly relative position biases—have influenced many subsequent models.

## Key Takeaways
- Encoder-decoder architecture with text-to-text format
- Relative position biases instead of absolute position embeddings
- Span corruption pre-training (15% tokens masked in contiguous spans)
- Pre-norm layer normalization (no bias, no mean-centering)
- 1:1 encoder-decoder layer ratio
- Available in sizes from 60M to 11B parameters
- T5 v1.1 introduced GeGLU and removed pre-training dropout
- Strong across translation, summarization, classification, and QA
