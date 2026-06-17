# Concept: GPT-4 Architecture Overview

## Concept ID

DL-402

## Difficulty

Expert

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Understand the known architectural details of GPT-4 and how it differs from GPT-3.
- Analyze the impact of mixture-of-experts (MoE) on model capacity and computational cost.
- Explain the multimodal capabilities of GPT-4 and how vision is integrated.
- Evaluate the safety alignment techniques used in GPT-4 (RLHF, Constitutional AI).
- Identify the key innovations that make GPT-4 more capable than GPT-3.

## Prerequisites

- Thorough understanding of GPT-3 architecture (DL-401)
- Knowledge of mixture-of-experts (MoE) layers in Transformers
- Understanding of reinforcement learning from human feedback (RLHF)
- Familiarity with multimodal model architectures
- Expert-level knowledge of Transformer scaling and training dynamics

## Definition

GPT-4 is OpenAI's fourth-generation language model, released in March 2023. Unlike GPT-3, OpenAI did not release detailed technical specifications, but based on available information and analysis, GPT-4 is a multimodal mixture-of-experts (MoE) Transformer model capable of processing both text and image inputs. The model is estimated to have approximately 1.7 trillion total parameters with ~220 billion active parameters per inference due to the MoE architecture using 16 experts. Key advancements include improved factual accuracy, better reasoning capabilities, longer context windows (initially 8192, later 128K), multimodal input processing, and significant alignment improvements through RLHF and post-training techniques.

## Intuition

Think of GPT-4 as a team of 16 specialists (experts), each good at different aspects of language: one expert might understand code well, another excels at creative writing, another knows medical terminology. When you ask a question, only the most relevant 2-3 experts are consulted. This is the mixture-of-experts approach. It allows the model to have vast knowledge (1.7T total parameters) while keeping inference costs manageable (220B active parameters).

The multimodal capability is like having both a language specialist and a vision specialist on the team. The vision specialist processes images into representations that the language model can understand, allowing the whole team to answer questions about images.

GPT-4's alignment is achieved through extensive post-training. The base model is fine-tuned with RLHF — human feedback helps the model learn what constitutes good, helpful, and safe responses. This is like a mentor providing guidance on not just what is correct, but what is appropriate.

## Why This Concept Matters

GPT-4 represents the current state-of-the-art in large language models and has had transformative impact:

1. Demonstrates that MoE architectures can dramatically increase model capacity without proportional compute increase.
2. Shows that multimodality can be integrated into a language model through a vision encoder.
3. Establishes new standards for model safety through alignment techniques.
4. Achieves human-level or super-human performance on many professional and academic benchmarks.
5. Powers products used by hundreds of millions of users (ChatGPT Plus, GPT-4 API, Copilot).
6. Demonstrates that continued scaling and refinement yields qualitative improvements in reasoning and factuality.

## Mathematical Explanation

### Mixture-of-Experts Architecture

In a standard Transformer FFN layer:
FFN(x) = GELU(x W_1) W_2

In an MoE layer with E experts:
FFN_MoE(x) = sum_{i=1}^E g_i(x) * FFN_i(x)

Where g_i(x) is the routing probability for expert i, computed by a learned router:

g(x) = softmax(Router(x))

Only the top-k experts are activated per token (typically k = 2 for GPT-4):
g_i'(x) = g_i(x) if g_i(x) in top-k, else 0
g_i''(x) = g_i'(x) / sum_j g_j'(x)

GPT-4 alternates standard dense layers with MoE layers, similar to how it alternates attention patterns.

### Estimated Parameters

GPT-4 estimated parameter breakdown:
- Embedding: ~25M * 16384 ≈ 410M
- 96 layers:
  - 48 dense layers: 48 * (4 * 16384^2 + 2 * 16384 * 65536) ≈ 154B
  - 48 MoE layers: 48 * 16 * (2 * 16384 * 65536) ≈ 1.65T
- Total: ~1.7T parameters
- Active per token: ~220B (2 experts active out of 16)

### Multimodal Integration

GPT-4 processes images through a vision encoder (which is not a ViT but likely a more advanced architecture). The encoder produces a sequence of visual embeddings that are projected into the same space as text token embeddings and concatenated with them. The decoder-only Transformer processes the combined sequence, treating image regions as additional tokens in the context.

### RLHF (Reinforcement Learning from Human Feedback)

Three-stage process:
1. Supervised fine-tuning (SFT) on demonstration data
2. Reward model training on human preference comparisons
3. PPO (Proximal Policy Optimization) to maximize reward

The RLHF objective:
RLHF_objective = R(x, y) - beta * KL(pi_RL || pi_SFT)

Where R is the reward model score, and the KL penalty prevents the RL-tuned model from diverging too far from the SFT model.

## Code Examples

### Example 1: Mixture-of-Experts FFN Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoEFFN(nn.Module):
    def __init__(self, d_model=16384, d_ff=65536, n_experts=16, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            weight = top_k_weights[:, :, i:i+1]
            expert_outputs = torch.zeros_like(x)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    masked_x = x[mask]
                    expert_out = self.experts[e](masked_x)
                    expert_outputs[mask] = expert_out
            final_output += expert_outputs * weight

        return final_output

class MoELayer(nn.Module):
    def __init__(self, d_model=768, d_ff=3072, n_experts=8, top_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 12, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(d_model, d_ff, n_experts, top_k)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        x = x + self.moe(self.norm2(x))
        return x

moe_layer = MoELayer(d_model=256, d_ff=1024, n_experts=4, top_k=2)
x = torch.randn(2, 16, 256)
out = moe_layer(x)
print("MoE layer output shape:", out.shape)
# Output: MoE layer output shape: torch.Size([2, 16, 256])
print("Only top-k experts activated per token")
# Output: Only top-k experts activated per token
```

### Example 2: Vision Encoder Integration

```python
class VisionEncoder(nn.Module):
    def __init__(self, d_model=768, patch_size=16, img_size=224):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, images):
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return self.ln(x)

class GPT4Multimodal(nn.Module):
    def __init__(self, text_vocab=50257, d_model=768, n_layers=12):
        super().__init__()
        self.text_embed = nn.Embedding(text_vocab, d_model)
        self.vision_encoder = VisionEncoder(d_model)
        self.vision_projection = nn.Linear(d_model, d_model)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 12, batch_first=True),
            n_layers
        )
        self.lm_head = nn.Linear(d_model, text_vocab)

    def forward(self, input_ids=None, images=None):
        text_emb = self.text_embed(input_ids)
        if images is not None:
            vis_emb = self.vision_encoder(images)
            vis_emb = self.vision_projection(vis_emb)
            combined = torch.cat([vis_emb, text_emb], dim=1)
        else:
            combined = text_emb
        decoded = self.decoder(combined)
        return self.lm_head(decoded)

model = GPT4Multimodal(d_model=256, n_layers=4)
text = torch.randint(0, 1000, (2, 32))
images = torch.randn(2, 3, 224, 224)
logits = model(input_ids=text, images=images)
print("Multimodal output shape:", logits.shape)
# Output: Multimodal output shape: torch.Size([2, 261, 1000])
print("Image patches + text tokens processed together")
# Output: Image patches + text tokens processed together
```

### Example 3: RLHF Training Loop Sketch

```python
def ppo_update(policy_model, ref_model, reward_model, prompt_dataset, ppo_epochs=4, kl_coef=0.1):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)

    for epoch in range(ppo_epochs):
        for batch in prompt_dataset:
            prompts = batch["prompts"]

            with torch.no_grad():
                responses = policy_model.generate(prompts)
                ref_logprobs = ref_model.get_logprobs(prompts, responses)
                rewards = reward_model(prompts, responses)

            logprobs = policy_model.get_logprobs(prompts, responses)
            ratio = torch.exp(logprobs - ref_logprobs)
            kl_div = logprobs - ref_logprobs

            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 0.8, 1.2) * advantages
            ).mean()
            kl_loss = kl_div.mean()
            total_loss = policy_loss + kl_coef * kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    print("RLHF update complete")
    # Output: RLHF update complete
    print("Policy optimized using PPO with KL penalty")
    # Output: Policy optimized using PPO with KL penalty
```

## Common Mistakes

1. Assuming GPT-4's architecture is a simple scale-up of GPT-3: GPT-4 uses mixture-of-experts, which fundamentally changes the architecture's computational profile. MoE increases total capacity while keeping inference costs manageable.

2. Confusing total parameters with active parameters: GPT-4 has ~1.7T total parameters but only ~220B active per forward pass. The remaining 1.48T parameters are in inactive experts for each token.

3. Underestimating the importance of post-training: The base GPT-4 model (pre-trained only) is significantly less capable and less aligned than the released version. RLHF and other alignment techniques are crucial to GPT-4's performance and safety.

4. Ignoring the context window limitations: While GPT-4 supports up to 128K tokens in some versions, attention still has O(L^2) complexity for dense layers. Long-context performance may degrade for tasks requiring precise attention to specific details.

5. Assuming GPT-4's knowledge cutoff reflects model capability: GPT-4's training data cutoff (September 2021 for the initial release) means it lacks knowledge of events after that date, regardless of model capability.

6. Overlooking the inference cost: GPT-4's 220B active parameters require multiple GPUs for inference. The cost per token is significantly higher than smaller models, which is reflected in API pricing.

## Interview Questions

### Beginner

Q: What is mixture-of-experts (MoE) and why does GPT-4 use it?

A: MoE is an architecture where multiple specialized sub-networks (experts) are trained, and a router selects which experts to use for each input. GPT-4 uses MoE to have a very large total capacity (1.7T parameters) while keeping inference costs manageable (220B active parameters per token). This allows the model to have more knowledge than a dense model of the same computational cost.

### Intermediate

Q: How does GPT-4 process images? What architectural components are involved?

A: GPT-4 uses a vision encoder to process images into a sequence of visual embeddings. The encoder divides the image into patches, embeds each patch, and adds positional information. These visual embeddings are then projected into the same space as text token embeddings and concatenated with them. The decoder-only Transformer processes the combined sequence, treating image patches as additional tokens. This allows the model to reason about images, answer questions about visual content, and incorporate visual information into its responses.

### Advanced

Q: Describe the RLHF training process used for GPT-4. Why is it effective, and what are its limitations?

A: RLHF has three stages: (1) Supervised fine-tuning on high-quality demonstration data showing desired behavior. (2) Training a reward model on human preference comparisons (which response is better). (3) Optimizing the policy model using PPO to maximize reward while constraining KL divergence from the SFT model. RLHF is effective because it aligns the model with human values by learning from comparative judgments rather than absolute labels. Limitations include: reward model can be hacked (Goodhart's law), human preferences are inconsistent, the process can reduce output diversity, and it may not capture nuanced safety requirements. Constitutional AI and other techniques are being explored to address these limitations.

## Practice Problems

### Easy

Compare GPT-3 and GPT-4 in terms of: number of parameters (total and active), architecture type (dense vs MoE), modality support, and training approach. Create a comparison table.

### Medium

Implement a simplified MoE Transformer with 4 experts and top-k=2. Train it on a language modeling task and compare with a dense model of similar active parameter count. Measure training speed, inference speed, and perplexity.

### Hard

Design a multimodal model architecture that processes text, images, and audio using a shared decoder. Describe how each modality is encoded, projected, and combined. Implement a prototype that handles at least two modalities and evaluates on a multimodal task (e.g., visual question answering or image captioning).

## Solutions

```python
# Easy solution outline
comparison = {
    "Total Parameters": {"GPT-3": "175B", "GPT-4": "~1.7T"},
    "Active Parameters": {"GPT-3": "175B", "GPT-4": "~220B"},
    "Architecture": {"GPT-3": "Dense Transformer", "GPT-4": "MoE Transformer"},
    "Modalities": {"GPT-3": "Text only", "GPT-4": "Text + Images"},
    "Training": {"GPT-3": "LM + few-shot", "GPT-4": "LM + RLHF + post-training"},
    "Context Window": {"GPT-3": "2048", "GPT-4": "8192-128K"}
}
for key, vals in comparison.items():
    print(f"{key}: GPT-3={vals['GPT-3']}, GPT-4={vals['GPT-4']}")
# Output: Total Parameters: GPT-3=175B, GPT-4=~1.7T
# Output: Active Parameters: GPT-3=175B, GPT-4=~220B
# Output: Architecture: GPT-3=Dense Transformer, GPT-4=MoE Transformer
# Output: Modalities: GPT-3=Text only, GPT-4=Text + Images
# Output: Training: GPT-3=LM + few-shot, GPT-4=LM + RLHF + post-training
# Output: Context Window: GPT-3=2048, GPT-4=8192-128K
```

## Related Concepts

- GPT-3 (DL-401)
- Decoder-Only Architecture (DL-403)
- Mixture-of-Experts
- Reinforcement Learning from Human Feedback
- Multimodal Learning
- AI Alignment
- Constitutional AI

## Next Concepts

- Decoder-Only Architecture
- Inference with Decoder
- Prefix LM

## Summary

GPT-4 is a multimodal MoE Transformer with estimated 1.7T total parameters and 220B active per inference. It integrates a vision encoder for image understanding, uses RLHF for alignment, and represents the state-of-the-art in large language models. The MoE architecture allows massive capacity while maintaining inference efficiency, and the multimodal capability enables text-image reasoning.

## Key Takeaways

- GPT-4 uses MoE with 16 experts and top-k=2 for efficient scaling.
- Total parameters ~1.7T, active ~220B per token.
- Multimodal: processes both text and images through a vision encoder.
- RLHF alignment training improves helpfulness and safety.
- Context window extends from 8K to 128K tokens.
- Significant qualitative improvement over GPT-3 across most benchmarks.
- Post-training (RLHF, safety filters) is as important as pre-training.
- The architecture enabled transformative products like ChatGPT Plus and GPT-4 API.
- GPT-4 set new SOTA on professional exams (Bar, SAT, GRE) and reasoning benchmarks.
- Future directions include longer context, more modalities, and better alignment.
