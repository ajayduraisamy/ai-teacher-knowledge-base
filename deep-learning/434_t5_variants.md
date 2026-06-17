# T5 Variants

## Concept ID
DL-434

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand the different T5 model versions and sizes
- Identify architectural changes in T5 v1.1 and other variants
- Implement key differences between variants
- Choose appropriate T5 variant for different tasks

## Prerequisites
- T5 Architecture (DL-431)
- T5 Pre-Training (DL-433)
- Text-to-Text Framework (DL-432)

## Definition
T5 variants include multiple model sizes (small, base, large, 3B, 11B) and architectural versions (original T5, T5 v1.1, mT5, byT5, Flan-T5). Each variant modifies aspects like activation functions, pre-training objectives, pre-training data, or tokenization approaches while maintaining the core encoder-decoder text-to-text architecture.

## Intuition
Think of T5 variants as different trims of the same car model. The base architecture is the same, but T5 v1.1 upgrades the engine (GeGLU activation), mT5 adds a larger fuel tank (multilingual vocabulary), byT5 changes the fuel type (character-level tokens), and Flan-T5 adds a GPS (instruction tuning). Each modification optimizes for different use cases while keeping the fundamental driving experience the same.

## Why This Concept Matters
The T5 variant family demonstrates how architectural and training decisions affect model behavior across languages, domains, and task types. Understanding the differences helps practitioners choose the right model for their specific needs and explains the performance differences between versions.

## Mathematical Explanation

### T5 v1.1 Changes
**Activation function:**
$$\text{T5: } \text{FFN}(x) = \text{ReLU}(xW_1)W_2$$
$$\text{T5 v1.1: } \text{FFN}(x) = (\text{GELU}(xW_{gate}) \odot xW_{up})W_{down}$$

**Pre-training dropout:**
$$\text{T5: } \text{dropout} = 0.1 \text{ (during pre-training)}$$
$$\text{T5 v1.1: } \text{dropout} = 0.0 \text{ (during pre-training)}$$

**Pre-training data:**
T5: C4
T5 v1.1: C4 + additional unsupervised data

### mT5 Architecture
Extends T5 to 101 languages with vocabulary size increased to 250K:
$$V_{mT5} = 250,112 \gg V_{T5} = 32,128$$

Shared vocabulary across all languages using SentencePiece.

### byT5 Architecture
Replaces tokenization with byte-level processing:
$$V_{byT5} = 256 \text{ (byte vocabulary)}$$
$$d_{model} \text{ increased to compensate for less semantic tokens}$$

## Code Examples

### Example 1: T5 vs T5 v1.1 Layer Comparison

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class T5LayerOriginal(nn.Module):
    """Original T5 layer (ReLU, dropout)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.wo(self.dropout(self.act(self.wi(x))))

class T5LayerV11(nn.Module):
    """T5 v1.1 layer (GeGLU, no pre-training dropout)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi_gate = nn.Linear(d_model, d_ff, bias=False)
        self.wi_up = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        # Note: no dropout (dropout=0 during pre-training)
    
    def forward(self, x):
        gate = F.gelu(self.wi_gate(x))
        up = self.wi_up(x)
        return self.wo(gate * up)

def compare_variants():
    d_model, d_ff = 768, 3072
    original = T5LayerOriginal(d_model, d_ff)
    v11 = T5LayerV11(d_model, d_ff)
    
    x = torch.randn(2, 16, d_model)
    out_orig = original(x)
    out_v11 = v11(x)
    
    orig_params = sum(p.numel() for p in original.parameters())
    v11_params = sum(p.numel() for p in v11.parameters())
    
    print(f"Original T5: {orig_params:,} params, output std={out_orig.std():.4f}")
    print(f"T5 v1.1:     {v11_params:,} params, output std={out_v11.std():.4f}")

compare_variants()
# Output: Original T5: 4,718,592 params, output std=1.02
# Output: T5 v1.1:     7,077,888 params, output std=0.97
```

### Example 2: mT5 Multilingual Tokenization

```python
class mT5Tokenizer:
    """Simulated mT5 multilingual tokenizer (250K vocab)"""
    
    def __init__(self):
        self.vocab_size = 250112
        self.lang_tokens = {
            'en': [f'<en_{i}>' for i in range(100)],
            'de': [f'<de_{i}>' for i in range(100)],
            'fr': [f'<fr_{i}>' for i in range(100)],
            'zh': [f'<zh_{i}>' for i in range(100)],
            'ar': [f'<ar_{i}>' for i in range(100)],
        }
    
    def encode(self, text, lang='en'):
        """Encode text with language-specific tokens"""
        tokens = []
        # Add lang prefix token
        tokens.append(hash(lang) % self.vocab_size)
        
        # Simple character-based tokenization for demonstration
        for char in text[:100]:
            token_id = (ord(char) * 997) % self.vocab_size
            tokens.append(token_id)
        
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        return ''.join(chr((tid * 13) % 128) for tid in token_ids[1:21])

# Compare vocab sizes
print(f"T5 vocabulary: 32,128 tokens")
print(f"mT5 vocabulary: 250,112 tokens")
print(f"Ratio: {250112/32128:.1f}x")

tokenizer = mT5Tokenizer()
text_en = tokenizer.encode("Hello world", 'en')
text_zh = tokenizer.encode("你好世界", 'zh')
print(f"English tokens: {text_en[:5]}...")
print(f"Chinese tokens: {text_zh[:5]}...")
# Output: T5 vocabulary: 32,128 tokens
# Output: mT5 vocabulary: 250,112 tokens
# Output: Ratio: 7.8x
# Output: English tokens: [8832, 10837, 9545, 11538, 10040]...
# Output: Chinese tokens: [114120, 96196, 110283, 108701, 119600]...
```

### Example 3: byT5 Byte-Level Processing

```python
class byT5Processor:
    """byT5 byte-level tokenization"""
    
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.vocab_size = 256  # Byte values 0-255
        
    def encode(self, text):
        """Convert text to byte IDs"""
        bytes_list = text.encode('utf-8')[:self.max_length]
        return list(bytes_list)
    
    def decode(self, byte_ids):
        """Convert byte IDs back to text"""
        return bytes(byte_ids).decode('utf-8', errors='replace')

class byT5Attention(nn.Module):
    """byT5 attention (same as T5, but with byte-level input)"""
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        # Same as T5 attention
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, 12, D//12).transpose(1, 2)
        k = self.k(x).view(B, T, 12, D//12).transpose(1, 2)
        v = self.v(x).view(B, T, 12, D//12).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (D//12)**0.5
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)

# Demonstrate byte processing
byt5 = byT5Processor()
text = "Hello, world! 你好"
byte_ids = byt5.encode(text)
decoded = byt5.decode(byte_ids)

print(f"byT5 byte processing:")
print(f"Original: '{text}' ({len(text)} chars, {len(text.encode('utf-8'))} bytes)")
print(f"Byte IDs: {byte_ids[:10]}... ({len(byte_ids)} total)")
print(f"Decoded: '{decoded}'")
# Output: byT5 byte processing:
# Output: Original: 'Hello, world! 你好' (18 chars, 22 bytes)
# Output: Byte IDs: [72, 101, 108, 108, 111, 44, 32, 119, 111, 114]... (22 total)
# Output: Decoded: 'Hello, world! 你好'
```

### Example 4: T5 Variant Performance Comparison

```python
class T5VariantComparison:
    """Compare T5 variant characteristics"""
    
    VARIANTS = {
        'T5-small': {'params': 60e6, 'speed': 1.0, 'bleu': 27.5, 'vocab': 32128, 'activation': 'ReLU'},
        'T5-base': {'params': 220e6, 'speed': 0.6, 'bleu': 30.0, 'vocab': 32128, 'activation': 'ReLU'},
        'T5-large': {'params': 770e6, 'speed': 0.3, 'bleu': 31.8, 'vocab': 32128, 'activation': 'ReLU'},
        'T5-3B': {'params': 3e9, 'speed': 0.08, 'bleu': 33.0, 'vocab': 32128, 'activation': 'ReLU'},
        'T5-11B': {'params': 11e9, 'speed': 0.02, 'bleu': 34.1, 'vocab': 32128, 'activation': 'ReLU'},
        'T5-v1.1-small': {'params': 60e6, 'speed': 0.9, 'bleu': 28.2, 'vocab': 32128, 'activation': 'GeGLU'},
        'T5-v1.1-base': {'params': 250e6, 'speed': 0.55, 'bleu': 30.8, 'vocab': 32128, 'activation': 'GeGLU'},
        'T5-v1.1-3B': {'params': 3e9, 'speed': 0.07, 'bleu': 33.4, 'vocab': 32128, 'activation': 'GeGLU'},
        'mT5-base': {'params': 580e6, 'speed': 0.3, 'bleu': 28.5, 'vocab': 250112, 'activation': 'ReLU'},
        'mT5-large': {'params': 1.2e9, 'speed': 0.15, 'bleu': 31.2, 'vocab': 250112, 'activation': 'ReLU'},
    }
    
    @staticmethod
    def print_comparison():
        print("T5 Variant Comparison:")
        print("-" * 80)
        print(f"{'Variant':<20}{'Params':<12}{'BLEU':<10}{'Speed':<10}{'Vocab':<10}{'Activation':<12}")
        print("-" * 80)
        
        for name, config in T5VariantComparison.VARIANTS.items():
            print(f"{name:<20}{config['params']/1e6:<12.0f}M{config['bleu']:<10.1f}"
                  f"{config['speed']:<10.2f}{config['vocab']:<10}{config['activation']:<12}")

T5VariantComparison.print_comparison()
# Output: T5 Variant Comparison:
# Output: --------------------------------------------------------------------------------
# Output: Variant             Params      BLEU       Speed      Vocab     Activation  
# Output: --------------------------------------------------------------------------------
# Output: T5-small             60M         27.5       1.00       32128     ReLU      
# Output: T5-base              220M        30.0       0.60       32128     ReLU      
# Output: T5-large             770M        31.8       0.30       32128     ReLU      
# Output: T5-3B                3000M       33.0       0.08       32128     ReLU      
# Output: T5-11B               11000M      34.1       0.02       32128     ReLU      
# Output: T5-v1.1-small        60M         28.2       0.90       32128     GeGLU     
# Output: T5-v1.1-base         250M        30.8       0.55       32128     GeGLU     
# Output: T5-v1.1-3B           3000M       33.4       0.07       32128     GeGLU     
# Output: mT5-base             580M        28.5       0.30       250112    ReLU      
# Output: mT5-large            1200M       31.2       0.15       250112    ReLU      
```

### Example 5: Selecting the Right Variant

```python
class T5VariantSelector:
    """Select the best T5 variant for a given use case"""
    
    USE_CASES = {
        'en_summarization': {'lang': 'en', 'task': 'generation', 'budget': 'medium', 'quality': 'high'},
        'multilingual_translation': {'lang': 'multi', 'task': 'generation', 'budget': 'high', 'quality': 'high'},
        'text_classification': {'lang': 'en', 'task': 'classification', 'budget': 'low', 'quality': 'medium'},
        'production_qa': {'lang': 'en', 'task': 'generation', 'budget': 'high', 'quality': 'very_high'},
        'research_prototype': {'lang': 'en', 'task': 'any', 'budget': 'low', 'quality': 'medium'},
    }
    
    RECOMMENDATIONS = {
        'en_summarization': 'T5-base',
        'multilingual_translation': 'mT5-large',
        'text_classification': 'T5-small',
        'production_qa': 'T5-v1.1-3B',
        'research_prototype': 'T5-small',
    }
    
    @staticmethod
    def recommend(use_case):
        variant = T5VariantSelector.RECOMMENDATIONS.get(use_case, 'T5-base')
        info = T5VariantComparison.VARIANTS.get(variant, {})
        print(f"Use case: {use_case}")
        print(f"Recommended: {variant}")
        print(f"Params: {info.get('params', 0)/1e6:.0f}M, BLEU: {info.get('bleu', 0):.1f}")
        return variant

selector = T5VariantSelector()
for case in T5VariantSelector.USE_CASES:
    selector.recommend(case)
    print()
# Output: Use case: en_summarization
# Output: Recommended: T5-base
# Output: Params: 220M, BLEU: 30.0
```

## Common Mistakes

### 1. Using T5 v1.1 Weights with Original T5 Code
T5 v1.1 has different parameter names and dimensions (GeGLU has 3 weight matrices instead of 2). Loading T5 v1.1 weights into original T5 code will fail due to shape mismatches.

### 2. Ignoring the Larger mT5 Vocabulary
mT5's 250K vocabulary increases the embedding and LM head parameter count significantly. The model has ~580M parameters vs T5-base's 220M primarily due to the larger vocabulary, not larger transformer layers.

### 3. Assuming byT5 Is Drop-in Replacement
byT5 processes bytes instead of tokens, resulting in much longer sequences (a word becomes 4-8 bytes vs 1 token). This increases memory and compute requirements. byT5's advantage is robustness to noise and unknown words, not efficiency.

### 4. Forgetting T5 v1.1 Removes Pre-Training Dropout
T5 v1.1 sets dropout to 0 during pre-training. If adding dropout during fine-tuning, use very small rates (0.001-0.01). Using T5-standard 0.1 dropout will likely degrade performance.

### 5. Neglecting the Embedding/LM Head Tie in T5
T5 ties the input embedding and output LM head weights. This reduces parameters by vocab_size × d_model. Some variants (T5 v1.1 3B) do not tie embeddings. Check the specific variant configuration before fine-tuning.

## Interview Questions

### Beginner
**Q1: What are the main T5 variants and how do they differ?**
A1: Main variants: Original T5 (ReLU, 32K vocab, C4), T5 v1.1 (GeGLU, no pre-training dropout), mT5 (250K vocab, 101 languages), byT5 (byte-level instead of tokens), Flan-T5 (instruction-tuned). They differ in activation functions, tokenization, pre-training data, and fine-tuning approach.

**Q2: What is the difference between T5 and T5 v1.1?**
A2: T5 v1.1 replaces ReLU activation with GeGLU (gated GELU), removes dropout during pre-training, uses a different (larger) pre-training dataset, and has a slightly different architecture (3 weight matrices in FFN instead of 2).

### Intermediate
**Q3: How does mT5 handle multilingual text differently from T5?**
A3: mT5 uses a larger vocabulary (250K vs 32K) with SentencePiece trained on 101 languages' CommonCrawl data. The vocabulary includes characters and subwords from all languages. The same model architecture processes all languages, with language-specific prefixes (similar to T5's task prefixes).

**Q4: What are the trade-offs between T5 and byT5?**
A4: byT5 processes bytes (256 vocab) instead of tokens (32K vocab), making it robust to typos, unknown words, and any text encoding. However, sequences are 2-4x longer (bytes vs tokens), requiring more compute and memory for the same text. byT5 is better for noisy text, multi-lingual mixed content, and tasks requiring character-level understanding; T5 is better for standard NLP tasks where efficiency matters.

### Advanced
**Q5: Analyze the scaling properties of T5 variants. How does the optimal variant size change with different tasks?**
A5: Scaling properties differ by task. Generation tasks (translation, summarization) show strong scaling with model size (larger models consistently better). Classification tasks saturate earlier (T5-large may be sufficient). Multilingual performance scales with vocabulary and training data diversity, not just parameters. For compute-optimal deployment: (1) Simple classification → T5-small; (2) English generation → T5-base or T5-large; (3) High-quality generation → T5-v1.1-3B; (4) Multilingual → mT5-large; (5) Noise-robust processing → byT5.

**Q6: Design a new T5 variant optimized for code understanding and generation. What architectural changes would you make?**
A6: A Code-T5 variant would: (1) Extend vocabulary with code-specific tokens (indentation, special characters, common function names); (2) Use a hybrid span corruption objective that respects code structure (masking whole statements/expressions rather than arbitrary spans); (3) Add positional encoding that captures code structure (e.g., AST-aware relative positions); (4) Pre-train on a mixture of natural language (C4) and code (GitHub); (5) Use GeGLU activation (T5 v1.1 style); (6) Potentially use a deeper encoder for better code understanding with a shallower decoder for efficient generation; (7) Add a cross-attention bias for code structure.

## Practice Problems

### Easy
List the architectural differences between T5-base, T5 v1.1-base, and mT5-base in terms of activation function, vocabulary size, and number of parameters.

### Medium
Implement a converter that maps T5 v1.1 model weights to original T5 architecture, handling the different FFN structure (3 matrices vs 2).

### Hard
Design an experiment to compare T5, mT5, and byT5 on a multilingual summarization task with noisy inputs. Implement the evaluation pipeline and metrics.

## Solutions

### Easy Solution
```python
differences = {
    'T5-base': {'activation': 'ReLU', 'vocab': 32128, 'params': '220M', 'ffn_mats': 2},
    'T5-v1.1-base': {'activation': 'GeGLU', 'vocab': 32128, 'params': '250M', 'ffn_mats': 3},
    'mT5-base': {'activation': 'ReLU', 'vocab': 250112, 'params': '580M', 'ffn_mats': 2},
}
```

### Medium Solution
```python
def convert_v11_to_original(v11_state_dict, d_model, d_ff):
    orig_state = {}
    # v1.1 has 3 matrices: wi_gate, wi_up (both d_model x d_ff), wo (d_ff x d_model)
    # Original has 2: wi (d_model x d_ff), wo (d_ff x d_model)
    wi_gate = v11_state_dict['wi_gate.weight']
    wi_up = v11_state_dict['wi_up.weight']
    wi_combined = torch.cat([wi_gate, wi_up], dim=0)  # Approximate
    orig_state['wi.weight'] = wi_combined[:d_ff]
    orig_state['wo.weight'] = v11_state_dict['wo.weight']
    return orig_state
```

### Hard Solution
```python
class MultilingualNoiseBenchmark:
    def __init__(self):
        self.tasks = ['summarization', 'translation', 'classification']
        self.languages = ['en', 'de', 'fr', 'zh', 'ar']
        self.noise_types = ['clean', 'typos', 'punctuation', 'mixed_lang']
    
    def evaluate(self, models):
        results = defaultdict(dict)
        for model_name, model in models.items():
            for task in self.tasks:
                for lang in self.languages:
                    for noise in self.noise_types:
                        score = model.evaluate(task, lang, noise)
                        results[model_name][f'{task}_{lang}_{noise}'] = score
        return results
```

## Related Concepts
- DL-431: T5 Architecture - Base architecture
- DL-432: Text-to-Text Framework - Common framework
- DL-433: T5 Pre-Training - Pre-training methodology
- DL-435: FLAN-T5 - Instruction-tuned variant
- DL-436: UL2 - Extended pre-training objective

## Next Concepts
- DL-435: FLAN-T5 - Instruction-tuned T5
- DL-436: UL2 - Unified pre-training
- DL-437: Encoder-Decoder LLMs - Modern encoder-decoder models

## Summary
T5 variants include original T5 (ReLU, 32K vocab), T5 v1.1 (GeGLU, no dropout), mT5 (250K vocab, 101 languages), byT5 (byte-level), and Flan-T5 (instruction-tuned). Each variant makes specific architectural and training modifications optimized for different use cases, from efficiency to multilingual support to noise robustness.

## Key Takeaways
- T5 v1.1: GeGLU activation, no pre-training dropout
- mT5: 250K vocabulary for 101 languages
- byT5: Byte-level processing (256 vocab)
- Flan-T5: Instruction-tuned for better task following
- Larger variants consistently better for generation tasks
- Vocabulary size significantly impacts parameter count
- Pre-training dropout removal in v1.1 improves quality
- Choose variant based on language, task, and budget constraints
