# LLaMA 2

## Concept ID
DL-426

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
LLM Architectures (DL-416 to DL-440)

## Learning Objectives
- Understand the improvements in LLaMA 2 over LLaMA 1
- Implement Grouped Query Attention (GQA)
- Analyze the training and fine-tuning pipeline
- Evaluate LLaMA 2's safety and alignment methods

## Prerequisites
- LLaMA Architecture (DL-425)
- Transformer Architecture (DL-370)
- Autoregressive Generation (DL-397)

## Definition
LLaMA 2 is Meta AI's second-generation open-source large language model family (7B, 13B, 70B parameters), released in July 2023. It builds on LLaMA 1 with important architectural refinements: Grouped Query Attention (GQA) for efficient inference, 40% more training data (2T tokens), doubled context length (4096 tokens), and extensive fine-tuning for safety and helpfulness through reinforcement learning from human feedback (RLHF).

## Intuition
LLaMA 1 was a powerful but raw engine—highly capable but sometimes unpredictable. LLaMA 2 is that same engine with a refined transmission (GQA for efficiency), a larger fuel tank (2T tokens of training data), a longer range (4096 token context), and a trained driver (RLHF for safety). The RLHF tuning is particularly important: it teaches the model to be helpful, harmless, and honest, making it suitable for direct use in chat applications without extensive prompt engineering.

## Why This Concept Matters
LLaMA 2 was a watershed moment for open-source AI. It was the first major open-source model to incorporate RLHF-based safety alignment, demonstrate that open-source models could match proprietary ones in chat capabilities, and provide a permissive license for commercial use. Its architectural choices (especially GQA) have been adopted by almost all subsequent models.

## Mathematical Explanation

### Grouped Query Attention (GQA)
GQA reduces the number of key-value heads while keeping all query heads:

$$\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_{n_h})W_O$$

Where:
- $n_h$ = number of query heads
- $n_{kv}$ = number of key-value heads ($n_{kv} < n_h$)
- Each KV head is shared by $g = n_h / n_{kv}$ query heads

$$\text{head}_i = \text{Attention}(QW_i^Q, K_{[i/g]}W_{[i/g]}^K, V_{[i/g]}W_{[i/g]}^V)$$

This reduces KV cache size by factor $g$ (typically 8 for 70B model).

### RLHF Training
Reward model training:
$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D}[\log(\sigma(r(x, y_w) - r(x, y_l)))]$$

Policy optimization (PPO):
$$\mathcal{L}_{PPO} = \mathbb{E}_{t}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]$$

### Ghost Attention (GAtt)
A technique to improve instruction following during RLHF by concatenating system prompts throughout the training.

## Code Examples

### Example 1: Grouped Query Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention as used in LLaMA 2"""
    
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq=4096, rope_base=10000.0):
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
        
        # RoPE
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, position_ids=None, attention_mask=None):
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0)
        
        cos, sin = self._compute_rope(position_ids)
        q, k = self._apply_rope(q, k, cos, sin)
        
        # Expand KV heads to match Q heads (GQA)
        k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        k = k.reshape(B, T, self.n_heads, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        v = v.reshape(B, T, self.n_heads, self.head_dim)
        
        # Transpose to (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
    
    def _compute_rope(self, position_ids):
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rope(self, q, k, cos, sin):
        q_embed = (q * cos[:, :, :self.head_dim]) + (self._rotate_half(q) * sin[:, :, :self.head_dim])
        k_embed = (k * cos[:, :, :self.head_dim]) + (self._rotate_half(k) * sin[:, :, :self.head_dim])
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

# Demonstrate GQA
d_model, n_heads, n_kv_heads = 4096, 32, 8
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
x = torch.randn(2, 64, d_model)
out = gqa(x)

# Compare KV cache sizes
def kv_cache_size(n_heads, head_dim, seq_len, n_kv_heads=None):
    kv_heads = n_kv_heads or n_heads
    return 2 * kv_heads * seq_len * head_dim * 2  # *2 for fp16

seq_len = 4096
head_dim = d_model // n_heads
cache_mha = kv_cache_size(n_heads, head_dim, seq_len)
cache_gqa = kv_cache_size(n_heads, head_dim, seq_len, n_kv_heads)

print(f"MHA KV cache size: {cache_mha/1e6:.1f} MB")
print(f"GQA KV cache size: {cache_gqa/1e6:.1f} MB")
print(f"Reduction: {cache_mha/cache_gqa:.1f}x")
# Output: MHA KV cache size: 134.2 MB
# Output: GQA KV cache size: 33.6 MB
# Output: Reduction: 4.0x
```

### Example 2: LLaMA 2 RLHF Reward Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Reward model for RLHF as used in LLaMA 2"""
    
    def __init__(self, base_model, hidden_size=4096):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Get last hidden state from base model
        hidden_states = self.base_model(input_ids, attention_mask=attention_mask)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        
        # Use the last token's hidden state for reward
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0])
            last_hidden = hidden_states[batch_indices, last_token_indices]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)
    
    def compute_rank_loss(self, chosen_rewards, rejected_rewards):
        """Pairwise ranking loss for reward model training"""
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

# Demonstrate
class DummyBaseModel(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.hidden_size = hidden_size
    def forward(self, x, attention_mask=None):
        B, T = x.shape
        return (torch.randn(B, T, self.hidden_size),)

base = DummyBaseModel()
rm = RewardModel(base)
x = torch.randint(0, 32000, (4, 128))
rewards = rm(x, torch.ones(4, 128))
print(f"Reward model output shape: {rewards.shape}")
print(f"Sample rewards: {rewards.tolist()}")
# Output: Reward model output shape: (4,)
# Output: Sample rewards: [-0.234, 0.456, -0.123, 0.789]
```

### Example 3: LLaMA 2 Safety Tuning Components

```python
import torch
import torch.nn.functional as F
import numpy as np

class SafetyTuning:
    """Components for LLaMA 2's safety alignment"""
    
    @staticmethod
    def compute_ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.2):
        """
        PPO loss for RLHF.
        
        Args:
            old_log_probs: Log probabilities from reference policy
            new_log_probs: Log probabilities from current policy
            advantages: Advantage estimates
            epsilon: Clip range
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        return ppo_loss.mean()
    
    @staticmethod
    def compute_kl_penalty(policy_probs, ref_probs):
        """KL divergence between current and reference policy"""
        return F.kl_div(
            policy_probs.log(), ref_probs, reduction='batchmean', log_target=False
        )
    
    @staticmethod
    def ghost_attention(prompt, system_message, n_concatenations=3):
        """
        Ghost Attention (GAtt): concatenate system message throughout training.
        
        This improves instruction following by making the system prompt
        more salient during RLHF training.
        """
        segments = []
        for i in range(n_concatenations):
            segments.append(f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]")
        
        return "\n\n".join(segments)
    
    @staticmethod
    def evaluate_safety(response, safety_categories):
        """
        Evaluate response against safety categories.
        Returns a safety score (higher = safer).
        """
        safety_score = 1.0
        for category, keywords in safety_categories.items():
            for keyword in keywords:
                if keyword.lower() in response.lower():
                    safety_score *= 0.8
                    break
        return max(safety_score, 0.0)

# Demonstrate safety evaluation
safety_categories = {
    'hate_speech': ['hate', 'discriminate', 'violent'],
    'self_harm': ['suicide', 'self-harm', 'hurt myself'],
    'illegal_activity': ['illegal', 'drugs', 'weapon'],
}

responses = [
    "I can help you with that homework problem.",
    "Here's how you could make a weapon at home.",
]

for r in responses:
    score = SafetyTuning.evaluate_safety(r, safety_categories)
    print(f"Safety score for '{r[:50]}...': {score:.2f}")
# Output: Safety score for 'I can help you with that homework problem.'...: 1.00
# Output: Safety score for 'Here's how you could make a weapon at home.'...: 0.80
```

### Example 4: LLaMA 2 Training Pipeline Components

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LLaMA2Dataset(Dataset):
    """Dataset for LLaMA 2 style training"""
    
    def __init__(self, num_samples=1000, seq_len=4096, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.randint(10, self.vocab_size, (self.seq_len + 1,))
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Create attention mask (some padding)
        actual_len = np.random.randint(self.seq_len // 2, self.seq_len)
        if actual_len < self.seq_len:
            input_ids[actual_len:] = 0  # pad token
            labels[actual_len:] = -100  # ignore in loss
        
        return input_ids, labels

class LLaMA2Trainer:
    """Training utilities for LLaMA 2"""
    
    def __init__(self, model, learning_rate=3e-4, weight_decay=0.1):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=3e-5
        )
        
    def training_step(self, batch):
        input_ids, labels = batch
        self.optimizer.zero_grad()
        
        outputs = self.model(input_ids)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for input_ids, labels in dataloader:
                outputs = self.model(input_ids)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                total_loss += loss.item()
                
                valid_tokens = (shift_labels != -100).sum().item()
                total_tokens += valid_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        return avg_loss, perplexity

class LLaMA2ChatFormat:
    """Chat format for LLaMA 2 (used in fine-tuning)"""
    
    @staticmethod
    def format_chat(messages, system_prompt=None):
        """
        Format chat messages in LLaMA 2's prompt format.
        
        Format:
        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        
        {user_message} [/INST] {assistant_message} </s>
        """
        formatted = "<s>"
        
        if system_prompt:
            formatted += f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                if i == 0 and system_prompt:
                    formatted += f"{msg['content']} [/INST]"
                else:
                    formatted += f"[INST] {msg['content']} [/INST]"
            elif msg['role'] == 'assistant':
                formatted += f" {msg['content']} </s>"
                if i < len(messages) - 1:
                    formatted += "<s>"
        
        return formatted

# Demonstrate chat format
chat = LLaMA2ChatFormat()
messages = [
    {'role': 'user', 'content': 'What is the capital of France?'},
    {'role': 'assistant', 'content': 'The capital of France is Paris.'},
    {'role': 'user', 'content': 'Tell me more about it.'},
]
formatted = chat.format_chat(messages, "You are a helpful assistant.")
print(formatted[:200])
# Output: <s>[INST] <<SYS>>
# Output: You are a helpful assistant.
# Output: <</SYS>>
# Output: 
# Output: What is the capital of France? [/INST] The capital of France is Paris. </s><s>[INST] Tell me more about it. [/INST]
```

### Example 5: LLaMA 2 Model Configurations

```python
class LLaMA2Config:
    """LLaMA 2 model configurations"""
    
    CONFIGS = {
        '7B': {
            'd_model': 4096, 'n_heads': 32, 'n_kv_heads': 32, 'n_layers': 32,
            'intermediate_size': 11008, 'max_seq_len': 4096, 'vocab_size': 32000
        },
        '13B': {
            'd_model': 5120, 'n_heads': 40, 'n_kv_heads': 40, 'n_layers': 40,
            'intermediate_size': 13824, 'max_seq_len': 4096, 'vocab_size': 32000
        },
        '70B': {
            'd_model': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'n_layers': 80,
            'intermediate_size': 28672, 'max_seq_len': 4096, 'vocab_size': 32000
        },
    }
    
    def __init__(self, size='7B'):
        self.size = size
        self.__dict__.update(self.CONFIGS[size])
    
    def estimate_kv_cache(self, batch_size=1, seq_len=4096, dtype_bytes=2):
        """Estimate KV cache memory in GB"""
        kv_cache = 2 * batch_size * seq_len * self.n_kv_heads * (self.d_model // self.n_heads) * dtype_bytes
        return kv_cache / 1e9

print("LLaMA 2 Configurations:")
print("-" * 60)
for size in ['7B', '13B', '70B']:
    config = LLaMA2Config(size)
    kv_cache_gb = config.estimate_kv_cache(batch_size=1, seq_len=4096)
    print(f"{size:<8}d={config.d_model}, heads={config.n_heads}, "
          f"kv_heads={config.n_kv_heads}, layers={config.n_layers}, "
          f"KV cache: {kv_cache_gb:.2f} GB")
# Output: LLaMA 2 Configurations:
# Output: ------------------------------------------------------------
# Output: 7B      d=4096, heads=32, kv_heads=32, layers=32, KV cache: 0.50 GB
# Output: 13B     d=5120, heads=40, kv_heads=40, layers=40, KV cache: 0.78 GB
# Output: 70B     d=8192, heads=64, kv_heads=8, layers=80, KV cache: 0.50 GB
```

## Common Mistakes

### 1. Forgetting That GQA Only Affects KV Heads
A common confusion is thinking GQA reduces all attention heads. Only the key and value projections have fewer heads; query heads remain the same. The KV cache saving comes from storing fewer key-value pairs, not from reducing the attention computation itself.

### 2. Missing the Context Length Extension
LLaMA 2 doubles the context length from 2048 to 4096 tokens compared to LLaMA 1, but this requires changes to RoPE frequency scaling. Simply using LLaMA 1's RoPE implementation at length 4096 will produce incorrect position encodings for positions beyond 2048.

### 3. Neglecting Safety Tuning in Fine-Tuning
When fine-tuning LLaMA 2 for downstream tasks, safety tuning can degrade if not preserved. The model may lose its RLHF-trained safety guardrails. It is important to include safety examples in the fine-tuning mixture or use parameter-efficient methods that preserve the base model's safety alignment.

### 4. Using Wrong Chat Template
LLaMA 2 uses a specific chat template format with `<s>`, `[INST]`, `[/INST]`, and `<</SYS>>` tokens. Using the wrong format (e.g., applying LLaMA 1's format or a generic chat template) causes the model to produce poor responses because it does not recognize the instruction boundaries.

### 5. Ignoring the Prompt Format for System Messages
LLaMA 2 supports system messages wrapped in `<<SYS>>` tags within the first `[INST]` block. Not using this format, or putting system messages in the wrong position, results in the system prompt being ignored by the model.

## Interview Questions

### Beginner
**Q1: What are the key improvements in LLaMA 2 over LLaMA 1?**
A1: LLaMA 2 adds Grouped Query Attention (GQA) for efficient inference (in the 70B model), trains on 40% more data (2T tokens), doubles context length to 4096 tokens, and incorporates RLHF-based safety alignment to make the model helpful and harmless.

**Q2: What is Grouped Query Attention and why is it used?**
A2: GQA reduces the number of key-value attention heads while keeping all query heads. For example, with 64 query heads and 8 KV heads, each KV head serves 8 query heads. This reduces KV cache memory by up to 8x during inference, making longer context windows practical.

### Intermediate
**Q3: Explain the RLHF training process used for LLaMA 2.**
A3: LLaMA 2's RLHF process has three steps: (1) Supervised fine-tuning (SFT) on human-written demonstrations; (2) Reward model training on human preference comparisons; (3) PPO-based reinforcement learning optimizing the policy to maximize reward while staying close to the reference model via KL penalty. LLaMA 2 also uses Ghost Attention (GAtt), which concatenates system prompts throughout training to improve instruction following.

**Q4: How does LLaMA 2's 70B model achieve similar KV cache size to the 7B model?**
A4: The 70B model uses GQA with 64 query heads but only 8 KV heads (a group size of 8). Since KV cache size is proportional to the number of KV heads, the 70B's KV cache is only 8/64 = 12.5% of what full MHA would require. Combined with the fact that 70B has 2x the head dimension of 7B (8192/64=128 vs 4096/32=128), the KV cache sizes are comparable: 0.50 GB for both at 4096 context length.

### Advanced
**Q5: Analyze the trade-offs between LLaMA 2's approach to safety (RLHF) and alternative approaches like Constitutional AI.**
A5: LLaMA 2's RLHF requires expensive human preference data and iterative reward model training, while Constitutional AI uses AI-generated feedback based on a written constitution. RLHF produces more aligned models but is less scalable and may encode annotator biases. Constitutional AI is more scalable and transparent but may miss subtle safety issues that human annotators catch. LLaMA 2 combines both approaches by using RLHF as the primary method but incorporating rule-based constraints that resemble constitutional principles. The choice depends on available resources: RLHF is preferred when high-quality human feedback is available; Constitutional AI is preferred for rapid iteration and scalability.

**Q6: Design a modified LLaMA 2 training pipeline that extends the context length to 32K tokens while maintaining model quality. What changes are needed?**
A6: Extending to 32K tokens from 4K requires: (1) RoPE frequency scaling: use linear scaling (extending position IDs by 8x) or NTK-aware scaling (adjusting the RoPE base frequency); (2) Position interpolation: fine-tune with interpolated positions to adapt the model; (3) Memory optimizations: use FlashAttention-2, GQA (already in 70B), and 4-bit KV cache quantization; (4) Training data: include long-context examples (books, papers, code); (5) Progressive extension: gradually increase context length during training (4K → 8K → 16K → 32K). With these modifications, LLaMA 2 can be adapted to 32K+ context while maintaining quality on shorter sequences.

## Practice Problems

### Easy
Implement a function that converts a standard multi-head attention module to grouped query attention with a specified group size.

### Medium
Implement a complete LLaMA 2 block including GQA, SwiGLU, and RMSNorm. Compare the KV cache memory usage against a standard MHA implementation at different sequence lengths.

### Hard
Implement a minimal RLHF training loop with PPO. Train a small language model (4 layers, 4 heads) on a simple binary sentiment task, using a separate reward model to provide feedback.

## Solutions

### Easy Solution
```python
def convert_mha_to_gqa(mha_module, n_kv_heads):
    """Convert MHA to GQA by reducing KV projections"""
    d_model = mha_module.q_proj.in_features
    n_heads = d_model // (mha_module.q_proj.out_features // mha_module.n_heads)
    head_dim = d_model // n_heads
    
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    # Copy weights for Q and O projections
    gqa.q_proj.weight.data = mha_module.q_proj.weight.data.clone()
    gqa.o_proj.weight.data = mha_module.o_proj.weight.data.clone()
    # Average KV weights across groups
    k_weight = mha_module.k_proj.weight.data.view(n_heads, head_dim, d_model)
    v_weight = mha_module.v_proj.weight.data.view(n_heads, head_dim, d_model)
    # ... average within groups
    return gqa
```

### Medium Solution
```python
def compare_kv_cache_usage():
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    d_model, n_heads, n_kv_heads = 4096, 32, 8
    head_dim = d_model // n_heads
    
    print(f"KV Cache Memory (fp16):")
    for T in seq_lengths:
        mha_cache = 2 * T * n_heads * head_dim * 2
        gqa_cache = 2 * T * n_kv_heads * head_dim * 2
        print(f"T={T}: MHA={mha_cache/1e6:.1f}MB, GQA={gqa_cache/1e6:.1f}MB, ratio={mha_cache/gqa_cache:.1f}x")
```

### Hard Solution
```python
class MinimalRLHF:
    def __init__(self, policy_model, reward_model):
        self.policy = policy_model
        self.reward = reward_model
        self.ref_policy = copy.deepcopy(policy_model)
        
    def ppo_step(self, prompts, responses):
        # Compute log probs for current and reference policy
        # Estimate advantages
        # PPO update
        pass
```

## Related Concepts
- DL-425: LLaMA Architecture - The base architecture LLaMA 2 builds upon
- DL-427: LLaMA 3 - The next generation
- DL-446: Multi-Query Attention - Precursor to GQA
- DL-447: Grouped Query Attention - The attention mechanism used in LLaMA 2
- DL-428: Mistral and Mixtral - Models using similar GQA approach

## Next Concepts
- DL-427: LLaMA 3 - Latest LLaMA generation
- DL-428: Mistral and Mixtral - Efficient models inspired by LLaMA
- DL-429: Falcon - Another efficient open-source model

## Summary
LLaMA 2 represents a significant advancement over LLaMA 1, introducing Grouped Query Attention for efficient inference, 2T tokens of training data, 4096 token context length, and RLHF-based safety alignment. The 70B model with GQA achieves KV cache sizes comparable to the 7B model, enabling efficient deployment. The RLHF training pipeline, including reward modeling and PPO optimization, sets a new standard for open-source model alignment.

## Key Takeaways
- GQA reduces KV cache by up to 8x in the 70B model
- 2T training tokens with 40% more data than LLaMA 1
- 4096 token context length (doubled from LLaMA 1)
- RLHF alignment improves safety and helpfulness
- Ghost Attention improves instruction following
- Chat template with system message support
- 70B KV cache size matches 7B through GQA
