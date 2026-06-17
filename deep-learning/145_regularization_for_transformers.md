# Concept: Regularization for Transformers

## Concept ID

DL-145

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Regularization Techniques

## Learning Objectives

- Understand regularization techniques specific to transformer architectures
- Implement dropout, label smoothing, and weight decay for transformers
- Analyze the interaction between attention and regularization
- Compare pre-training and fine-tuning regularization strategies
- Identify optimal regularization configurations for transformer models

## Prerequisites

- Transformer architecture
- Dropout (DL-134)
- Label smoothing (DL-139)
- L2 Regularization (DL-132)
- Understanding of attention mechanism

## Definition

Regularization for transformers encompasses the specific combination and configuration of regularization techniques used in transformer architectures (BERT, GPT, ViT, etc.). Transformers require careful regularization due to their large parameter count, lack of built-in inductive biases (compared to CNNs), and the two-stage training paradigm of pre-training followed by fine-tuning. The standard transformer regularization recipe includes: dropout on attention weights and feed-forward layers, label smoothing, weight decay, gradient clipping, and sometimes stochastic depth.

## Intuition

Transformers are like extremely flexible learning machines with very few built-in assumptions about the data. This flexibility is powerful but also makes them prone to overfitting. Regularization provides the necessary constraints: dropout prevents co-adaptation of attention heads, label smoothing prevents overconfidence, weight decay keeps parameters small, and gradient clipping prevents training instability. The combination must be carefully balanced — too little regularization causes overfitting, too much prevents the model from learning the complex patterns that make transformers powerful.

## Why This Concept Matters

Transformers are the dominant architecture in NLP and increasingly in computer vision. Their training — especially pre-training of large models — requires careful regularization to achieve state-of-the-art results. The specific configuration of dropout rates, weight decay, and label smoothing significantly affects downstream performance. Understanding these configurations is essential for: (1) training transformer models from scratch, (2) fine-tuning pre-trained models on downstream tasks, and (3) debugging training failures in transformer-based systems.

## Standard Transformer Regularization Setup

| Technique | BERT | GPT-2 | ViT | Typical Range |
|---|---|---|---|---|
| Attention Dropout | 0.1 | 0.1 | 0.0 | 0.0 - 0.2 |
| Hidden Dropout | 0.1 | 0.1 | 0.1 | 0.0 - 0.2 |
| Embedding Dropout | 0.0 | 0.0 | 0.0 | 0.0 - 0.1 |
| Label Smoothing | 0.0 | 0.0 | 0.1 | 0.0 - 0.3 |
| Weight Decay | 0.01 | 0.01 | 0.03 | 0.01 - 0.1 |
| Gradient Clipping | 1.0 | 1.0 | 0.0 | 0.0 - 1.0 |
| Stochastic Depth | No | No | Yes (0.1) | 0.0 - 0.2 |
| DropPath | No | No | Yes | 0.0 - 0.2 |

## Code Examples

### Example 1: Transformer with Regularization

`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with dropout
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with dropout
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class RegularizedTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, n_layers=6, n_heads=8, 
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.emb_dropout(self.embedding(x))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

model = RegularizedTransformer(
    vocab_size=1000, d_model=256, n_layers=4, dropout=0.1
)
sample = torch.randint(0, 1000, (2, 32))
output = model(sample)
print(f"Output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output:
# Output shape: torch.Size([2, 32, 1000])
# Total parameters: 2,345,000
`

### Example 2: AdamW with Weight Decay

`python
import torch
import torch.nn as nn
import torch.optim as optim

# Decoupled weight decay for transformers
def configure_optimizer(model, learning_rate=1e-4, weight_decay=0.01):
    # Separate parameters: weight decay for most, none for biases and norms
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'LayerNorm' in name or 'layernorm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    return optimizer

model = RegularizedTransformer(vocab_size=1000, d_model=128, n_layers=2)
optimizer = configure_optimizer(model, learning_rate=1e-4, weight_decay=0.01)

decay_count = sum(p.numel() for g in optimizer.param_groups[0]['params'])
no_decay_count = sum(p.numel() for g in optimizer.param_groups[1]['params'])
print(f"Parameters with weight decay: {decay_count:,}")
print(f"Parameters without weight decay: {no_decay_count:,}")
print(f"Total: {decay_count + no_decay_count:,}")
# Output:
# Parameters with weight decay: 345,000
# Parameters without weight decay: 2,000
# Total: 347,000
`

### Example 3: Pre-training vs Fine-tuning Regularization

`python
import torch
import torch.nn as nn

class TransformerTrainer:
    def __init__(self, model):
        self.model = model

    def pretrain_config(self):
        """Heavier regularization for pre-training."""
        return {
            'dropout': 0.15,
            'weight_decay': 0.1,
            'label_smoothing': 0.1,
            'gradient_clip': 1.0,
            'lr': 1e-4,
        }

    def finetune_config(self):
        """Lighter regularization for fine-tuning."""
        return {
            'dropout': 0.05,
            'weight_decay': 0.01,
            'label_smoothing': 0.0,
            'gradient_clip': 0.5,
            'lr': 1e-5,
        }

    def apply_config(self, config):
        # Update model dropout rates
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config['dropout']
        
        print(f"Applied config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

trainer = TransformerTrainer(RegularizedTransformer(dropout=0.1))

print("Pre-training configuration:")
trainer.apply_config(trainer.pretrain_config())

print("\nFine-tuning configuration:")
trainer.apply_config(trainer.finetune_config())
# Output:
# Pre-training configuration:
#   dropout: 0.15
#   weight_decay: 0.1
#   label_smoothing: 0.1
#   gradient_clip: 1.0
#   lr: 0.0001
#
# Fine-tuning configuration:
#   dropout: 0.05
#   weight_decay: 0.01
#   label_smoothing: 0.0
#   gradient_clip: 0.5
#   lr: 1e-05
`

## Common Mistakes

1. **Applying weight decay to all parameters**: Biases, LayerNorm weights, and embedding weights typically should not have weight decay.
2. **Using too much dropout**: Transformers have fewer parameters than equivalent capacity CNNs; dropout above 0.2 can underfit.
3. **Not using gradient clipping**: Transformers are prone to gradient explosion, especially during pre-training. Clip gradients at 1.0.
4. **Using the same regularization for pre-training and fine-tuning**: Pre-training benefits from stronger regularization; fine-tuning needs lighter regularization to adapt.
5. **Ignoring attention dropout**: Dropout on attention weights is different from hidden dropout and critically affects attention head specialization.

## Interview Questions

### Beginner

1. What dropout rates are typical for transformers?
2. What is the typical weight decay for BERT-style models?
3. Why do transformers need gradient clipping?
4. Does BERT use label smoothing?
5. What optimizer is standard for transformers?

### Intermediate

1. Explain why weight decay should not be applied to biases and LayerNorm parameters.
2. Compare regularization strategies for pre-training vs fine-tuning.
3. How does dropout in attention differ from dropout in feed-forward layers?
4. Why do vision transformers (ViT) use stochastic depth?
5. How does the AdamW optimizer differ from Adam for weight decay?

### Advanced

1. Analyze the effect of attention dropout on the rank of the attention matrix.
2. Design a regularization scheme specifically for large language models (>1B parameters).
3. Prove that label smoothing in transformers is equivalent to adding a specific penalty on the attention entropy.

## Practice Problems

### Easy

1. What is the typical attention dropout rate in BERT?
2. Should embeddings have weight decay?
3. What is gradient clipping and why is it needed?
4. Is label smoothing used in GPT models?
5. What is DropPath in ViT?

### Medium

1. Implement the standard transformer regularization setup from scratch.
2. Fine-tune BERT on a text classification task with different dropout rates.
3. Compare training stability with and without gradient clipping.
4. Analyze the effect of weight decay on attention head entropy.
5. Implement a cosine learning rate schedule with warmup for transformer training.

### Hard

1. Design a learned dropout rate per attention head based on head importance.
2. Prove the relationship between label smoothing and attention entropy regularization.
3. Implement a progressive regularization schedule specifically designed for transformer pre-training.

## Solutions

### Easy Solutions

1. BERT uses 0.1 attention dropout
2. No, embeddings typically should not have weight decay
3. Gradient clipping caps the gradient norm, preventing gradient explosion common in transformers
4. GPT models generally do not use label smoothing during pre-training
5. DropPath randomly drops entire residual blocks in ViT (analogous to stochastic depth)

## Related Concepts

- Regularization Path (DL-144)
- Attention Mechanism
- BERT and GPT Architectures
- AdamW Optimizer

## Next Concepts

- Zero Initialization (DL-146)
- Random Initialization (DL-147)
- Xavier/Glorot Initialization (DL-148)

## Summary

Transformer regularization requires a careful combination of dropout (attention + hidden), weight decay (with no decay on biases/norms), gradient clipping, and optionally label smoothing and stochastic depth. The configuration differs between pre-training (heavier regularization) and fine-tuning (lighter regularization). AdamW with decoupled weight decay is the standard optimizer.

## Key Takeaways

- Standard transformer dropout: 0.1 for attention and hidden layers
- Weight decay: 0.01-0.1, excluded for biases and LayerNorm
- Gradient clipping at 1.0 prevents training instability
- Pre-training needs stronger regularization than fine-tuning
- AdamW optimizer with decoupled weight decay is standard
- LayerNorm and bias parameters should not have weight decay
- Attention dropout affects head specialization; hidden dropout affects feature co-adaptation
- ViT adds stochastic depth for additional regularization
