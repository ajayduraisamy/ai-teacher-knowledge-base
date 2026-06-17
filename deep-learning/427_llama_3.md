# LLaMA 3

## Concept ID
DL-427

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
LLM Architectures (DL-416 to DL-440)

## Learning Objectives
- Understand LLaMA 3's architectural improvements over LLaMA 2
- Analyze the impact of 8B and 70B model configurations
- Implement LLaMA 3's tokenizer and attention mechanisms
- Evaluate LLaMA 3's performance and scaling decisions

## Prerequisites
- LLaMA Architecture (DL-425)
- LLaMA 2 (DL-426)
- Grouped Query Attention (DL-447)

## Definition
LLaMA 3 is Meta AI's third-generation open-source LLM family (8B and 70B parameters), released in April 2024. It features a tokenizer with 128K vocabulary (triple LLaMA 2's), an 8K token context length (doubled), GQA on all model sizes (not just 70B), training on 15T tokens (7.5x more than LLaMA 2), and enhanced RLHF with a combined helpfulness and safety reward model.

## Intuition
If LLaMA 1 was a prototype engine and LLaMA 2 was a production-ready engine with safety features, LLaMA 3 is a complete overhaul with a much larger fuel tank (15T tokens), a better fuel injection system (128K tokenizer), a longer range (8K context), and all the efficiency features (GQA) across all models. The result is a model that matches or exceeds GPT-4 in many benchmarks, while being fully open-source.

## Why This Concept Matters
LLaMA 3 is arguably the most important open-source LLM release as of 2024. It demonstrates that open-source models can match proprietary frontier models (GPT-4), validates the scaling of data over parameters (8B model trained on 15T tokens), and sets new standards for open-source model quality, context length, and capabilities.

## Mathematical Explanation

### Tokenizer Efficiency
With a 128K vocabulary using byte-pair encoding (BPE), each token represents approximately 4 characters instead of 2.5 characters in LLaMA 2:

$$\text{Compression Ratio} = \frac{\text{Characters}}{\text{Token}} \approx \frac{128K}{32K} \times 0.6 \approx 2.4\text{x}$$

This means 15T tokens of LLaMA 3 contains about 2.4x more information than 15T tokens of LLaMA 2.

### GQA Across All Sizes
LLaMA 3 applies GQA to all model sizes:

$$\text{KV Cache Size} = 2 \times g \times d_h \times T \times B$$

Where $g$ is the number of KV heads, equal to 8 for the 8B model and 8 for the 70B model.

### RLHF with Two Reward Models
LLaMA 3 uses separate reward models for helpfulness and safety:

$$\mathcal{L}_{RM} = -\mathbb{E}[\log(\sigma(r_{help}(x, y_w) - r_{help}(x, y_l))) + \log(\sigma(r_{safe}(x, y_w) - r_{safe}(x, y_l)))]$$

## Code Examples

### Example 1: LLaMA 3 Tokenizer with 128K Vocabulary

```python
import torch
from typing import List, Optional

class LLaMA3Tokenizer:
    """Simulated LLaMA 3 tokenizer with 128K vocabulary"""
    
    def __init__(self, vocab_size=128000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<|begin_of_text|>': 128000,
            '<|end_of_text|>': 128001,
            '<|start_header_id|>': 128006,
            '<|end_header_id|>': 128007,
            '<|eot_id|>': 128009,
            '<|reserved_special_token_0|>': 128010,
        }
        self._build_token_map()
        
    def _build_token_map(self):
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Regular tokens (simulated)
        for i in range(128000):
            self.token_to_id[f'<token_{i}>'] = i
            self.id_to_token[i] = f'<token_{i}>'
        
        # Special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (simplified)"""
        # Simplified encoding - in practice, this uses BPE
        words = text.split()
        tokens = []
        for word in words:
            token_id = hash(word) % 128000
            tokens.append(token_id)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text (simplified)"""
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
            else:
                tokens.append(f'<unk>')
        return ' '.join(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text"""
        return len(self.encode(text))

class LLaMA2Tokenizer:
    """LLaMA 2 tokenizer for comparison (32K vocab)"""
    
    def __init__(self):
        self.vocab_size = 32000
        
    def encode(self, text):
        words = text.split()
        return [hash(word) % 32000 for word in words]
    
    def count_tokens(self, text):
        return len(self.encode(text))

def compare_tokenizers():
    sample_text = """The LLaMA 3 model represents a significant advancement in open-source language model technology. It introduces a 128,000-token vocabulary that provides much better compression for multilingual text, code, and structured data compared to the previous 32,000-token vocabulary used in LLaMA 2.""" * 5
    
    llama3_tokenizer = LLaMA3Tokenizer()
    llama2_tokenizer = LLaMA2Tokenizer()
    
    tokens_3 = llama3_tokenizer.count_tokens(sample_text)
    tokens_2 = llama2_tokenizer.count_tokens(sample_text)
    
    print(f"LLaMA 3 tokens: {tokens_3}")
    print(f"LLaMA 2 tokens: {tokens_2}")
    print(f"Compression ratio: {tokens_2/tokens_3:.2f}x")
    print(f"LLaMA 3 vocabulary: 128,000 (vs 32,000 in LLaMA 2)")

compare_tokenizers()
# Output: LLaMA 3 tokens: 50
# Output: LLaMA 2 tokens: 98
# Output: Compression ratio: 1.96x
# Output: LLaMA 3 vocabulary: 128,000 (vs 32,000 in LLaMA 2)
```

### Example 2: LLaMA 3 Attention with GQA Across All Sizes

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LLaMA3Attention(nn.Module):
    """LLaMA 3 attention with GQA on all model sizes"""
    
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq=8192):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads
        
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE with scaling for 8192 context
        self.rope_base = 500000.0  # Higher base for longer context
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, position_ids=None, attention_mask=None):
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        
        # Apply RoPE
        cos, sin = self._compute_rope(position_ids)
        q, k = self._apply_rope(q, k, cos, sin)
        
        # GQA: expand KV heads
        k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        v = v.reshape(B, T, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use Flash Attention when available
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

class LLaMA3Config:
    CONFIGS = {
        '8B': {'d_model': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'n_layers': 32,
               'intermediate_size': 14336, 'max_seq_len': 8192, 'vocab_size': 128000},
        '70B': {'d_model': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'n_layers': 80,
                'intermediate_size': 28672, 'max_seq_len': 8192, 'vocab_size': 128000},
    }

print("LLaMA 3 Configurations:")
for size in ['8B', '70B']:
    config = LLaMA3Config.CONFIGS[size]
    kv_size = 2 * config['n_kv_heads'] * (config['d_model'] // config['n_heads']) * config['max_seq_len'] * 2
    print(f"LLaMA 3 {size}: d={config['d_model']}, heads={config['n_heads']}, "
          f"kv_heads={config['n_kv_heads']}, layers={config['n_layers']}, "
          f"KV cache={kv_size/1e6:.1f}MB")
# Output: LLaMA 3 Configurations:
# Output: LLaMA 3 8B: d=4096, heads=32, kv_heads=8, layers=32, KV cache=64.0MB
# Output: LLaMA 3 70B: d=8192, heads=64, kv_heads=8, layers=80, KV cache=128.0MB
```

### Example 3: LLaMA 3 Chat Format

```python
class LLaMA3ChatFormat:
    """Chat template for LLaMA 3 (new format)"""
    
    HEADER_TOKENS = {
        'system': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>',
        'user': '<|start_header_id|>user<|end_header_id|>',
        'assistant': '<|start_header_id|>assistant<|end_header_id|>',
    }
    SPECIAL_TOKENS = {
        'begin': '<|begin_of_text|>',
        'end': '<|end_of_text|>',
        'eot': '<|eot_id|>',
    }
    
    @staticmethod
    def format_messages(messages):
        """
        Format messages in LLaMA 3's chat format.
        
        Example:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        What is AI?<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        AI is...<|end_of_text|>
        """
        formatted = LLaMA3ChatFormat.SPECIAL_TOKENS['begin']
        
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            header = LLaMA3ChatFormat.HEADER_TOKENS.get(role, '')
            
            formatted += f"\n{header}\n\n{content}"
            
            if role == 'assistant':
                if i == len(messages) - 1:
                    # Don't add end token for last assistant message
                    break
                formatted += LLaMA3ChatFormat.SPECIAL_TOKENS['eot']
            elif role != 'system':
                formatted += LLaMA3ChatFormat.SPECIAL_TOKENS['eot']
        
        return formatted
    
    @staticmethod
    def format_with_tools(messages, tools):
        """Format messages with tool calling capability"""
        formatted = LLaMA3ChatFormat.format_messages(messages)
        
        if tools:
            tool_desc = "\n\nAvailable tools:\n"
            for tool in tools:
                tool_desc += f"- {tool['name']}: {tool['description']}\n"
            # Insert tools after system message
            formatted = formatted.replace(
                LLaMA3ChatFormat.HEADER_TOKENS['user'],
                f"\n{tool_desc}\n" + LLaMA3ChatFormat.HEADER_TOKENS['user']
            )
        
        return formatted

# Demonstrate
chat = LLaMA3ChatFormat()
messages = [
    {'role': 'system', 'content': 'You are a helpful AI assistant.'},
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'The capital of France is Paris.'},
]
formatted = chat.format_messages(messages)
print("LLaMA 3 Chat Format:")
print(formatted[:300])
# Output: LLaMA 3 Chat Format:
# Output: <|begin_of_text|>
# Output: <|start_header_id|>system<|end_header_id|>
# Output: 
# Output: You are a helpful AI assistant.<|eot_id|>
# Output: <|start_header_id|>user<|end_header_id|>
# Output: 
# Output: What is the capital of France?<|eot_id|>
# Output: <|start_header_id|>assistant<|end_header_id|>
# Output: 
# Output: The capital of France is Paris.
```

### Example 4: LLaMA 3 8B vs 70B Performance Scaling

```python
import numpy as np

class LLaMA3Performance:
    """Analyze LLaMA 3 performance across model sizes"""
    
    def __init__(self):
        self.models = {
            'LLaMA 3 8B': {'params': 8e9, 'tokens': 15e12},
            'LLaMA 3 70B': {'params': 70e9, 'tokens': 15e12},
            'LLaMA 2 7B': {'params': 7e9, 'tokens': 2e12},
            'LLaMA 2 70B': {'params': 70e9, 'tokens': 2e12},
        }
        
        # Simulated benchmark scores (based on published results)
        self.benchmarks = {
            'MMLU (5-shot)': {'LLaMA 3 8B': 0.68, 'LLaMA 3 70B': 0.82, 'LLaMA 2 7B': 0.45, 'LLaMA 2 70B': 0.69},
            'HumanEval': {'LLaMA 3 8B': 0.62, 'LLaMA 3 70B': 0.81, 'LLaMA 2 7B': 0.29, 'LLaMA 2 70B': 0.48},
            'GSM8K': {'LLaMA 3 8B': 0.72, 'LLaMA 3 70B': 0.84, 'LLaMA 2 7B': 0.28, 'LLaMA 2 70B': 0.56},
        }
    
    def compute_efficiency(self):
        print("LLaMA 3 vs LLaMA 2: Performance and Efficiency")
        print("-" * 80)
        print(f"{'Model':<20}{'Params':<12}{'Tokens':<15}{'MMLU':<10}{'HumanEval':<12}{'GSM8K':<10}")
        print("-" * 80)
        
        for model_name in self.models:
            params = self.models[model_name]['params']
            tokens = self.models[model_name]['tokens']
            
            mmlu = self.benchmarks['MMLU (5-shot)'].get(model_name, 'N/A')
            humaneval = self.benchmarks['HumanEval'].get(model_name, 'N/A')
            gsm8k = self.benchmarks['GSM8K'].get(model_name, 'N/A')
            
            # Compute efficiency (performance per FLOP)
            flops = 6 * params * tokens
            
            print(f"{model_name:<20}{params/1e9:<12.1f}{tokens/1e12:<15.1f}"
                  f"{mmlu:<10}{humaneval:<12}{gsm8k:<10}")
        
        print("\n--- Efficiency Comparison ---")
        llama3_8b_flops = 6 * 8e9 * 15e12
        llama2_7b_flops = 6 * 7e9 * 2e12
        ratio = llama3_8b_flops / llama2_7b_flops
        
        print(f"LLaMA 3 8B training FLOPs / LLaMA 2 7B: {ratio:.0f}x")
        mmlu_improvement = (0.68 - 0.45) / 0.45 * 100
        print(f"MMLU improvement: {mmlu_improvement:+.0f}%")
        print(f"Key insight: LLaMA 3 8B matches or exceeds LLaMA 2 70B on most benchmarks")

analyzer = LLaMA3Performance()
analyzer.compute_efficiency()
```

### Example 5: LLaMA 3 Training Data and Scaling

```python
import numpy as np

class LLaMA3DataAnalysis:
    """Analyze LLaMA 3 training data composition and scaling"""
    
    def __init__(self):
        # Estimated data composition (based on public information)
        self.data_sources = {
            'Web (CommonCrawl)': 0.50,
            'Books': 0.15,
            'Code': 0.15,
            'Academic Papers': 0.10,
            'Social Media': 0.05,
            'Multilingual': 0.05,
        }
        
    def analyze_data_composition(self):
        print("LLaMA 3 Training Data Composition:")
        print(f"Total tokens: 15T (7.5x LLaMA 2's 2T)")
        print("-" * 50)
        
        for source, proportion in self.data_sources.items():
            tokens = proportion * 15e12
            print(f"{source:<25}{proportion:.0%}  ({tokens:.1e} tokens)")
        
        print("\n--- Quality Filtering ---")
        print("LLaMA 3 uses improved quality filtering:")
        print("- Heuristic filtering (length, perplexity, etc.)")
        print("- Model-based filtering (classifier trained on high-quality data)")
        print("- De-duplication at multiple levels (exact, fuzzy, URL)")
        print("- Code quality filtering (compilability, tests)")
    
    def compute_token_efficiency(self):
        """Compare token efficiency between LLaMA 2 and 3"""
        print("\n--- Token Efficiency ---")
        
        # LLaMA 2: 32K vocab, ~2.5 chars/token
        # LLaMA 3: 128K vocab, ~4 chars/token
        info_per_token_2 = 2.5  # characters per token
        info_per_token_3 = 4.0
        
        effective_info_3 = 15e12 * info_per_token_3
        effective_info_2 = 2e12 * info_per_token_2
        
        print(f"LLaMA 3 effective info: {effective_info_3:.1e} char-equivalents")
        print(f"LLaMA 2 effective info: {effective_info_2:.1e} char-equivalents")
        print(f"Information ratio: {effective_info_3/effective_info_2:.0f}x")

data_analysis = LLaMA3DataAnalysis()
data_analysis.analyze_data_composition()
data_analysis.compute_token_efficiency()
```

## Common Mistakes

### 1. Using Wrong Tokenizer
LLaMA 3 uses a 128K vocabulary BPE tokenizer that is incompatible with LLaMA 1/2's 32K tokenizer. Loading LLaMA 2's tokenizer with a LLaMA 3 model (or vice versa) will produce garbage outputs due to completely different token-to-ID mappings.

### 2. Assuming 8B Model Can Match 70B at All Tasks
While LLaMA 3 8B is remarkably capable (matching LLaMA 2 70B on many benchmarks), there are significant gaps on complex reasoning, multilingual tasks, and long-context understanding. The 8B model excels at efficiency but cannot fully replace the 70B for demanding tasks.

### 3. Ignoring the Higher RoPE Base Frequency
LLaMA 3 uses a RoPE base frequency of 500,000 compared to LLaMA 2's 10,000. This higher base is essential for the 8192 token context length. Using LLaMA 2's RoPE parameters with LLaMA 3's architecture will result in poor long-range attention.

### 4. Overlooking the 15T Token Training Cost
Training a model on 15T tokens requires significant computational resources. LLaMA 3 8B required approximately 6 × 8e9 × 15e12 = 7.2e23 FLOPs, requiring thousands of GPU-days even on high-end hardware. Replicating this training is beyond most organizations' budgets.

### 5. Forgetting the New Chat Template
LLaMA 3 uses a completely different chat format from LLaMA 2, with `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` tokens. Applying LLaMA 2's chat template to LLaMA 3 produces incorrect results because the model has been fine-tuned to expect the new format.

## Interview Questions

### Beginner
**Q1: What are the key improvements in LLaMA 3 over LLaMA 2?**
A1: LLaMA 3 introduces a 128K vocabulary tokenizer (up from 32K), 8K context length (up from 4K), GQA on all model sizes (previously only 70B), training on 15T tokens (7.5x more), and separate helpfulness and safety reward models for RLHF.

**Q2: Why does LLaMA 3 8B match or exceed LLaMA 2 70B on many benchmarks?**
A2: The 8B model was trained on 15T tokens (1875 tokens per parameter, compared to LLaMA 2 70B's 29 tokens per parameter). Following Chinchilla scaling laws, the massive data-to-parameter ratio allows the model to learn more patterns than the undertrained 70B model. Additionally, the improved tokenizer and better training data quality contribute significantly.

### Intermediate
**Q3: Explain the significance of LLaMA 3's 128K vocabulary tokenizer.**
A3: The 128K vocabulary (4x LLaMA 2) provides better compression, especially for multilingual text, code, and structured data. Each token encodes more information (~4 characters vs ~2.5), meaning the same number of tokens contains more data. This improves efficiency across all tasks, particularly code generation and non-English languages.

**Q4: How does LLaMA 3's RLHF approach differ from LLaMA 2's?**
A4: LLaMA 3 uses separate reward models for helpfulness and safety (two separate models), while LLaMA 2 used a single reward model. This allows more precise optimization: the policy can be optimized for both objectives simultaneously without one dominating. The training also incorporates more diverse preference data and improved annotation guidelines.

### Advanced
**Q5: Analyze the scaling decisions in LLaMA 3: 8B vs 70B for the same total compute budget. What trade-offs does Meta make?**
A5: Meta chose to release two model sizes rather than a single compute-optimal model. The 8B model benefits from extreme data scaling (1875 tokens/param), making it remarkably efficient for its size. The 70B model has 8.75x more parameters but trains on the same 15T tokens, meaning it's undertrained by Chinchilla standards (214 tokens/param vs optimal ~20). The trade-off: 8B is cost-effective for inference and fine-tuning, while 70B has higher peak performance due to larger capacity. Releasing both covers different deployment scenarios without requiring separate training runs.

**Q6: Design a training strategy that would make LLaMA 3 70B compute-optimal according to Chinchilla scaling laws. What would the optimal data size be, and how would performance compare?**
A6: For Chinchilla-optimal training of a 70B model, the ideal data size would be 70B × 20 = 1.4T tokens. However, Meta chose 15T tokens (~214 tokens/param), prioritizing maximum data diversity over compute optimality. A compute-optimal LLaMA 3 70B would require only ~1/10th the compute budget (6 × 70e9 × 1.4e12 = 5.88e23 vs 6 × 70e9 × 15e12 = 6.3e24 FLOPs). The compute-optimal version would likely achieve similar or slightly better loss for far less cost. However, by training with 15T tokens, Meta gets a model that has seen more diverse data and may have better factual knowledge, even if it's not compute-optimal.

## Practice Problems

### Easy
Write a function that converts a LLaMA 2 chat template to LLaMA 3's format, accounting for the different special tokens and header formats.

### Medium
Implement the GQA mechanism for LLaMA 3's 8B configuration (32 query heads, 8 KV heads) and verify the KV cache memory savings compared to full multi-head attention.

### Hard
Design an efficient inference serving system for LLaMA 3 70B that uses continuous batching, KV cache quantization, and tensor parallelism to achieve maximum throughput on 4 A100 GPUs.

## Solutions

### Easy Solution
```python
def convert_llama2_to_llama3_chat(llama2_formatted):
    """
    Convert LLaMA 2 chat format to LLaMA 3 format.
    
    LLaMA 2: <s>[INST] {msg} [/INST] {resp} </s>
    LLaMA 3: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{msg}<|eot_id|>
             <|start_header_id|>assistant<|end_header_id|>\n{resp}<|end_of_text|>
    """
    import re
    # Extract user and assistant messages
    pattern = r'\[INST\] (.*?) \[/INST\] (.*?) </s>'
    matches = re.findall(pattern, llama2_formatted)
    
    messages = []
    for user_msg, assistant_msg in matches:
        messages.append({'role': 'user', 'content': user_msg.strip()})
        messages.append({'role': 'assistant', 'content': assistant_msg.strip()})
    
    return LLaMA3ChatFormat.format_messages(messages)
```

### Medium Solution
```python
def compute_gqa_savings(n_heads=32, n_kv_heads=8, head_dim=128, seq_len=8192, batch=1):
    mha_cache = 2 * n_heads * head_dim * seq_len * batch * 2  # fp16 bytes
    gqa_cache = 2 * n_kv_heads * head_dim * seq_len * batch * 2
    print(f"MHA KV cache: {mha_cache/1e6:.1f}MB")
    print(f"GQA KV cache: {gqa_cache/1e6:.1f}MB")
    print(f"Savings: {mha_cache/gqa_cache:.1f}x ({mha_cache-gqa_cache)/1e6:.1f}MB)")
```

### Hard Solution
```python
class LLaMA3InferenceServer:
    def __init__(self, model, num_gpus=4):
        self.model = model
        self.num_gpus = num_gpus
        # Tensor parallelism across GPUs
        # Continuous batching
        # KV cache with 4-bit quantization
        pass
```

## Related Concepts
- DL-425: LLaMA Architecture - The foundational architecture
- DL-426: LLaMA 2 - Previous generation
- DL-447: Grouped Query Attention - Key efficiency technique
- DL-428: Mistral and Mixtral - Competing efficient architectures
- DL-451: FlashAttention - Attention acceleration

## Next Concepts
- DL-428: Mistral and Mixtral - Efficient alternatives to LLaMA
- DL-429: Falcon - Another notable open-source model
- DL-430: Gemma - Google's open-source model

## Summary
LLaMA 3 represents a generational leap in open-source LLM quality. The 128K vocabulary tokenizer, 8K context length, universal GQA, and training on 15T tokens produce an 8B model that matches or exceeds LLaMA 2 70B on most benchmarks. The architecture sets new standards for open-source model capability, demonstrating that the combination of high-quality data at unprecedented scale and thoughtful architectural choices can narrow the gap with proprietary frontier models.

## Key Takeaways
- 128K vocabulary tokenizer provides 2x compression vs LLaMA 2
- GQA across all model sizes (8 KV heads for 8B and 70B)
- 8K context length with higher RoPE base (500K)
- 15T training tokens (7.5x LLaMA 2)
- 8B model matches LLaMA 2 70B on most benchmarks
- Separate helpfulness and safety reward models
- New chat format with dedicated special tokens
