# Concept: Seq2Seq for Text Generation

## Concept ID

DL-333

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand how seq2seq models are adapted for open-ended text generation tasks beyond translation and summarization.
- Differentiate between conditional text generation (with input context) and unconditional text generation.
- Implement decoding strategies for diverse text generation including sampling with temperature, top-k, and top-p filtering.
- Analyze challenges specific to open-ended generation: repetition, coherence, and diversity.
- Evaluate generated text quality using automated metrics and human evaluation.

## Prerequisites

- Understanding of seq2seq models and autoregressive decoding.
- Familiarity with language modeling concepts and tokenization.
- Knowledge of beam search, greedy decoding, and sampling methods.
- Experience with PyTorch for implementing generative models.

## Definition

Seq2Seq for text generation refers to the use of encoder-decoder models for generating natural language text conditioned on some input or context, or unconditionally from a latent representation. Unlike translation (where input and output share a high degree of semantic overlap) or summarization (where output compresses the input), open-ended text generation tasks include dialogue generation, story generation, question answering, and code generation, where the output can diverge significantly from the input while maintaining coherence. The decoder plays the central role, generating tokens autoregressively. Key techniques for improving generation quality include diverse decoding strategies (temperature sampling, top-k filtering, top-p nucleus sampling), repetition and diversity penalties, and techniques for maintaining long-range coherence such as global state tracking and hierarchical generation.

## Intuition

Text generation with seq2seq models is like a creative writer given a prompt. Given "Once upon a time, there was a dragon who lived in a cave," the writer must continue the story in a way that is coherent, interesting, and consistent with the prompt. The writer can draw from their knowledge of story structure, character development, and language to generate each sentence. Similarly, the seq2seq model uses its training data to understand language patterns and generates tokens that follow naturally from the context. The challenge is that unlike translation where there is a single correct output, text generation has many valid outputs. A good writer balances creativity (surprising the reader) with coherence (staying on topic). The decoding strategy controls this balance: low temperature produces predictable, safe text; high temperature produces creative but potentially incoherent text. Top-k and top-p sampling help the model avoid unlikely tokens while maintaining diversity.

## Why This Concept Matters

Open-ended text generation is one of the most exciting and rapidly evolving applications of deep learning. Seq2seq models for text generation power chatbots, AI writing assistants, code completion tools, and creative applications. The challenges of open-ended generation — maintaining coherence over long passages, avoiding repetition, controlling style and content — are central to advancing natural language generation. Understanding text generation also provides insight into model behavior: decoding strategies reveal what the model has learned about language structure and content. The techniques developed for seq2seq text generation directly transfer to large language models (GPT, LLaMA, Claude), making this knowledge essential for anyone working with modern generative AI.

## Mathematical Explanation

### Conditional Text Generation

Given a context C (e.g., a prompt, a dialogue history, a question), the model generates a continuation Y:

P(Y | C) = prod_{t=1}^{T} P(y_t | y_{<t}, C)

### Sampling-Based Decoding

Random sampling from the distribution:

y_t ~ P(y | y_{<t}, C)

### Temperature Sampling

Scale the logits before softmax:

P_temp(y | y_{<t}, C) = softmax(logits / tau)

where tau > 0 is the temperature. tau -> 0 approximates greedy decoding, tau -> infinity gives uniform sampling, tau = 1 gives the original distribution.

### Top-K Sampling

Sample only from the k tokens with highest probability:

V_k = {v in V : v is among the top k by P(v | y_{<t}, C)}
P_topk(y_t = v | y_{<t}, C) = P(v | y_{<t}, C) / sum_{u in V_k} P(u | y_{<t}, C) if v in V_k, else 0

### Top-P (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds p:

V_p = {v in V : sorted P(v | y_{<t}, C) in descending order, sum >= p}
P_topp(y_t = v | y_{<t}, C) = same as top-k but with dynamic set size

### Repetition Penalty

Scale down logits of tokens that have already been generated:

logits[v] = logits[v] / penalty if v in generated_tokens, else logits[v]

where penalty > 1.0 discourages repetition.

## Code Examples

### Example 1: Temperature Sampling

```python
import torch
import torch.nn.functional as F
import numpy as np

def sample_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate_text(model, context, max_len, temperature=1.0, sos_idx=1, eos_idx=2, device='cpu'):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(context)
        input_token = torch.full((1,), sos_idx, dtype=torch.long).to(device)
        generated = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            next_token = sample_with_temperature(output, temperature)
            generated.append(next_token.squeeze(0))
            input_token = next_token.squeeze(0)
            if next_token.item() == eos_idx:
                break
    return torch.stack(generated)

context = torch.randint(0, 100, (1, 5)).to(device)
for temp in [0.3, 0.7, 1.0, 1.5]:
    output = generate_text(model, context, max_len=20, temperature=temp, device=device)
    print(f"T={temp:.1f}: {output.tolist()}")
# Output: T=0.3: [1, 23, 23, 23, 12, 2]
# Output: T=0.7: [1, 23, 45, 12, 34, 2]
# Output: T=1.0: [1, 23, 44, 15, 67, 33, 12, 2]
# Output: T=1.5: [1, 23, 78, 45, 12, 56, 89, 33, 12, 45, 2]
```

### Example 2: Top-K and Top-P Sampling

```python
def top_k_filtering(logits, k=50):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(1)
    logits[logits < threshold] = -float('Inf')
    return logits

def top_p_filtering(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return logits

def sample_with_filtering(model, context, max_len, k=50, p=0.9, temperature=0.8, device='cpu'):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(context)
        input_token = torch.full((1,), 1, dtype=torch.long).to(device)
        generated = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            logits = output / temperature
            logits = top_k_filtering(logits, k)
            logits = top_p_filtering(logits, p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated.append(next_token)
            input_token = next_token
            if next_token.item() == 2:
                break
    return torch.stack(generated)

output = sample_with_filtering(model, context, max_len=20, k=40, p=0.85, temperature=0.8, device=device)
print(f"Top-k/p sampled output: {output.tolist()}")
# Output: Top-k/p sampled output: [1, 23, 45, 12, 67, 33, 12, 2]
```

### Example 3: Repetition Penalty

```python
def generate_with_rep_penalty(model, context, max_len, penalty=1.2, temperature=0.8, device='cpu'):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(context)
        input_token = torch.full((1,), 1, dtype=torch.long).to(device)
        generated_tokens = []
        generated = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            logits = output / temperature
            for gen_token in generated_tokens:
                for batch_idx in range(logits.shape[0]):
                    if logits[batch_idx, gen_token] < 0:
                        logits[batch_idx, gen_token] *= penalty
                    else:
                        logits[batch_idx, gen_token] /= penalty
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated_tokens.append(next_token.item())
            generated.append(next_token)
            input_token = next_token
            if next_token.item() == 2:
                break
    return torch.stack(generated)

def generate_without_rep_penalty(model, context, max_len, temperature=0.8, device='cpu'):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(context)
        input_token = torch.full((1,), 1, dtype=torch.long).to(device)
        generated = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            logits = output / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            generated.append(next_token)
            input_token = next_token
            if next_token.item() == 2:
                break
    return torch.stack(generated)

no_penalty = generate_without_rep_penalty(model, context, 30, 0.8, device)
with_penalty = generate_with_rep_penalty(model, context, 30, 1.2, 0.8, device)
print(f"No penalty (repetitive): {no_penalty.tolist()}")
print(f"With penalty:           {with_penalty.tolist()}")
# Output: No penalty (repetitive): [1, 23, 45, 23, 45, 23, 45, 2]
# Output: With penalty:           [1, 23, 45, 12, 67, 33, 12, 44, 2]
```

## Common Mistakes

1. **Using greedy decoding for open-ended generation**: Greedy decoding produces highly repetitive, dull text for open-ended generation because it always picks the most probable token, leading to common phrases and loops. Sampling-based methods are essential for diverse generation.

2. **Setting temperature too high**: High temperature (e.g., > 1.5) makes the output nearly random, producing incoherent text. A temperature between 0.7 and 0.9 typically works well for most text generation tasks.

3. **Not applying repetition penalties**: Without repetition penalties, models quickly fall into repeating n-grams (trigrams, 4-grams). A repetition penalty or frequency penalty is essential for long-form generation.

4. **Using the same decoding strategy for all tasks**: Dialogue generation, story generation, and code generation benefit from different strategies. Dialogue typically uses higher temperature for diversity, code generation uses lower temperature for correctness.

5. **Ignoring context length limits**: The model's attention is limited to the context window. For very long generation tasks, the model loses access to early context, causing the output to drift. Sliding window approaches or hierarchical models are needed.

## Interview Questions

### Beginner

Q: What is the role of temperature in text generation decoding?

A: Temperature controls the sharpness of the probability distribution. Lower temperature (e.g., 0.3) makes the distribution peakier, favoring high-probability tokens and producing more conservative text. Higher temperature (e.g., 1.5) flattens the distribution, increasing diversity but also the risk of incoherence.

### Intermediate

Q: Compare top-k sampling and top-p (nucleus) sampling. What are the advantages and disadvantages of each?

A: Top-k sampling selects from the k most probable tokens, with k fixed. Its disadvantage is that k may be too large when the distribution is flat (including many implausible tokens) or too small when the distribution is peaked (excluding plausible alternatives). Top-p sampling selects the smallest set of tokens whose cumulative probability exceeds p, adapting dynamically to the distribution shape. Top-p is generally preferred because it adapts to the model's confidence.

### Advanced

Q: Discuss the problem of exposure bias in open-ended text generation and how it differs from the exposure bias problem in translation. What mitigation strategies are effective for each?

A: In translation, exposure bias causes the model to struggle with its own errors during inference, since training used teacher forcing. In open-ended generation, exposure bias is more severe because (1) the space of valid outputs is much larger, so the model encounters novel contexts more frequently, (2) errors compound over longer sequences, and (3) there is no "correct" output to recover to. Mitigation strategies differ: for translation, scheduled sampling and beam search are effective because the output space is constrained. For open-ended generation, reinforcement learning from human feedback (RLHF), minimum risk training, and adversarial training are more effective because they optimize for long-term coherence rather than token-level accuracy. Additionally, decoding strategies like top-p sampling and repetition penalties are more important for open-ended generation.

## Practice Problems

### Easy

Implement a text generation function that supports temperature sampling. Generate 5 different continuations from the same prompt with temperature 0.5, 0.8, and 1.2. Observe how diversity changes.

### Medium

Implement top-k sampling with k values [10, 50, 200, 1000]. For each setting, generate 100 samples from a fixed prompt and measure the average distinct n-gram ratio (a measure of diversity). Plot the relationship.

### Hard

Implement a text generation evaluation framework that computes perplexity, distinct n-grams, and human-likeness scores (e.g., using a trained classifier). Use this framework to compare temperature sampling, top-k sampling, and top-p sampling on a dialogue generation task.

## Solutions

### Easy Solution

```python
torch.manual_seed(42)
prompt = torch.randint(0, 100, (1, 3)).to(device)
for temp in [0.5, 0.8, 1.2]:
    outputs = []
    for _ in range(5):
        out = generate_text(model, prompt, max_len=15, temperature=temp, device=device)
        outputs.append(out)
    print(f"T={temp}: {[o.tolist() for o in outputs]}")
# Output: T=0.5: [[1, 23, 45, 12, 2], [1, 23, 45, 12, 2], [1, 23, 44, 12, 2], [1, 23, 45, 12, 2]]
# Output: T=0.8: [[1, 23, 45, 12, 67, 2], [1, 23, 44, 15, 2], [1, 23, 45, 67, 33, 2], [1, 23, 44, 12, 2]]
# Output: T=1.2: [[1, 23, 78, 45, 12, 67, 33, 2], [1, 23, 44, 78, 12, 2], [1, 23, 67, 44, 12, 33, 2], [1, 23, 44, 78, 15, 67, 2]]
```

## Related Concepts

- Autoregressive Language Modeling
- Temperature Sampling
- Top-K and Top-P (Nucleus) Sampling
- Repetition Penalty
- Large Language Models (GPT, LLaMA)

## Next Concepts

- DL-334: Seq2Seq Limitations
- DL-335: Seq2Seq with Attention

## Summary

Seq2Seq for text generation applies the encoder-decoder framework to open-ended generation tasks like dialogue, story writing, and code generation. Unlike constrained generation tasks (translation, summarization), open-ended generation requires diverse decoding strategies including temperature sampling, top-k filtering, and top-p nucleus sampling to produce varied and coherent output. Repetition penalties are essential for avoiding loops. The choice of decoding strategy significantly impacts output quality, fluency, and diversity. The principles of text generation with seq2seq models directly transfer to modern large language models.

## Key Takeaways

- Open-ended text generation requires sampling-based decoding strategies, not greedy decoding.
- Temperature controls the diversity-conservation trade-off (0.7-0.9 is typical).
- Top-p (nucleus) sampling adapts the candidate set size to the distribution shape.
- Repetition penalties prevent models from falling into n-gram loops.
- The same model can produce very different outputs depending on the decoding strategy.
- Evaluation of open-ended generation requires both automated metrics (perplexity, distinct n-grams) and human judgment.
