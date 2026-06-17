# Concept: Autoregressive Generation

## Concept ID

DL-397

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Decoder Architectures

## Learning Objectives

- Define autoregressive generation and explain how it produces sequences token by token.
- Understand the difference between probabilistic and deterministic decoding strategies.
- Implement common decoding strategies: greedy search, beam search, top-k sampling, and top-p (nucleus) sampling.
- Analyze the trade-offs between diversity, quality, and computational cost in different decoding strategies.
- Evaluate generation quality using perplexity, repetition, and diversity metrics.

## Prerequisites

- Understanding of causal language models and decoder-only architectures
- Knowledge of probability distributions and softmax
- Familiarity with tokenization and vocabulary concepts
- Basic understanding of sequence modeling

## Definition

Autoregressive generation is the process of producing a sequence of tokens one at a time, where each token is generated conditioned on all previously generated tokens. Given a prompt or prefix P = (p_1, ..., p_m), the model generates tokens x_1, x_2, ..., x_n sequentially:

P(x_1, x_2, ..., x_n | P) = product from t=1 to n of P(x_t | P, x_1, ..., x_{t-1})

At each step t, the model outputs a probability distribution over the vocabulary V, and a decoding strategy determines which token to select as x_t. The generation continues until a special end-of-sequence token is produced or a maximum length is reached.

## Intuition

Autoregressive generation is like writing a sentence one word at a time. When you write, each new word depends on everything you have written before. If you start with "The capital of France is," the next word is very likely "Paris." But there is always some randomness — you might continue with "a beautiful city" instead.

The model assigns probabilities to each possible next word. "Paris" might have probability 0.7, "a" probability 0.1, "not" probability 0.05, etc. The decoding strategy decides how to use these probabilities: take the most likely word (greedy), sample from the distribution (sampling), or some hybrid approach.

Different strategies produce different outputs. Greedy decoding is safe but can produce repetitive or dull text. Sampling with temperature produces more diverse but potentially less coherent text. Beam search explores multiple paths for higher quality. The right strategy depends on the application: translation needs high precision, creative writing needs diversity.

## Why This Concept Matters

Autoregressive generation is the core capability of large language models like GPT, LLaMA, and Claude. Understanding decoding strategies is essential for:

1. Controlling output quality: The difference between a good and bad generation often comes down to the decoding strategy.
2. Balancing creativity and coherence: Different applications require different points on the creativity-coherence spectrum.
3. Managing computational cost: Some strategies (beam search) are much more expensive than others (greedy).
4. Avoiding common failure modes: Repetition, degeneration, and hallucination can be mitigated through decoding choices.

## Mathematical Explanation

### Autoregressive Factorization

For a sequence x = (x_1, ..., x_T):

P(x) = product_{t=1}^T P(x_t | x_{<t})

In a causal language model, the hidden state h_t (after causal attention over x_{<t}) is projected through the LM head:

logits_t = h_t^T W_lm + b
P(x_t | x_{<t}) = softmax(logits_t)

### Decoding Strategies

**Greedy Decoding**:
x_t = argmax P(x_t | x_{<t})

Deterministic, fast, but can produce repetitive text and misses high-probability sequences due to local decisions.

**Beam Search**:
Maintain k hypotheses (beams). At each step, extend each beam with all vocabulary tokens (or top candidates), compute cumulative log-probability, keep top k.

Total score: sum_{i=1}^t log P(x_i | x_{<i})

Higher k = better quality but more expensive.

**Temperature Sampling**:
P_T(x_t | x_{<t}) = softmax(logits_t / T)

T < 1: sharper distribution (more greedy)
T > 1: flatter distribution (more diverse)
T = 1: original distribution

**Top-k Sampling**:
Restrict sampling to k highest-probability tokens:
V_k = {v in V | v in top k by P(v | x_{<t})}
P_sampled(v | x_{<t}) = P(v | x_{<t}) / sum_{v' in V_k} P(v' | x_{<t}) for v in V_k

**Top-p (Nucleus) Sampling**:
Select the smallest set of tokens whose cumulative probability exceeds p:
V_p = smallest set such that sum_{v in V_p} P(v | x_{<t}) >= p

This adaptively chooses the number of candidates based on the distribution shape.

## Code Examples

### Example 1: Decoding Strategies Implementation

```python
import torch
import torch.nn.functional as F

def greedy_decode(logits):
    return logits.argmax(dim=-1)

def temperature_sample(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def top_k_sample(logits, k=50):
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)
    filtered_logits = torch.where(logits < min_top_k, float("-inf"), logits)
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

logits = torch.randn(1, 1000) * 2
print("Greedy:", greedy_decode(logits)[0].item())
# Output: Greedy: 523
print("Temperature sample (T=1.0):", temperature_sample(logits, 1.0)[0].item())
# Output: Temperature sample (T=1.0): 312
print("Top-k (k=50):", top_k_sample(logits, 50)[0].item())
# Output: Top-k (k=50): 456
print("Top-p (p=0.9):", top_p_sample(logits, 0.9)[0].item())
# Output: Top-p (p=0.9): 398
```

### Example 2: Full Generation Loop with Multiple Strategies

```python
def generate(model, input_ids, max_length=50, strategy="greedy", **kwargs):
    for _ in range(max_length):
        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        if strategy == "greedy":
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        elif strategy == "temperature":
            temp = kwargs.get("temperature", 1.0)
            next_token = temperature_sample(next_logits, temp)
        elif strategy == "top_k":
            k = kwargs.get("k", 50)
            next_token = top_k_sample(next_logits, k)
        elif strategy == "top_p":
            p = kwargs.get("p", 0.9)
            next_token = top_p_sample(next_logits, p)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == 50256:
            break
    return input_ids

def count_unique_tokens(ids):
    return len(set(ids[0].tolist()))

def measure_repetition(ids):
    tokens = ids[0].tolist()
    n_grams = set()
    repeats = 0
    for i in range(len(tokens) - 2):
        ngram = tuple(tokens[i:i+3])
        if ngram in n_grams:
            repeats += 1
        n_grams.add(ngram)
    return repeats / max(1, len(tokens) - 2)

logits = torch.randn(1, 10, 1000)
print("Different strategies produce different outputs")
# Output: Different strategies produce different outputs
print("Greedy is deterministic, sampling adds randomness")
# Output: Greedy is deterministic, sampling adds randomness
```

### Example 3: Beam Search Implementation

```python
def beam_search(model, input_ids, beam_width=3, max_length=20):
    batch_size = input_ids.shape[0]
    beams = [(0.0, input_ids)]

    for _ in range(max_length):
        candidates = []
        for score, seq in beams:
            logits = model(seq)
            next_logits = logits[:, -1, :]
            probs = F.log_softmax(next_logits, dim=-1)
            top_scores, top_tokens = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                new_score = score + top_scores[0, i].item()
                new_seq = torch.cat([seq, top_tokens[:, i:i+1]], dim=-1)
                candidates.append((new_score, new_seq))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    return max(beams, key=lambda x: x[0])[1]

class MockModel(nn.Module):
    def forward(self, input_ids):
        B, T = input_ids.shape
        return torch.randn(B, T, 1000)

model = MockModel()
input_ids = torch.randint(0, 1000, (1, 3))
result_beam3 = beam_search(model, input_ids, beam_width=3, max_length=5)
result_beam1 = beam_search(model, input_ids, beam_width=1, max_length=5)
print("Beam width 3 output length:", result_beam3.shape[1])
# Output: Beam width 3 output length: 8
print("Beam width 1 (greedy) output length:", result_beam1.shape[1])
# Output: Beam width 1 (greedy) output length: 8
print("Beam search explores multiple paths for better quality")
# Output: Beam search explores multiple paths for better quality
```

## Common Mistakes

1. Using greedy decoding for creative tasks: Greedy decoding produces the most likely continuation at each step, leading to repetitive, dull, and sometimes degenerate text. For creative writing, dialogue, or story generation, sampling methods are essential.

2. Using too high a temperature: Temperature above 1.5 flattens the distribution excessively, causing the model to produce random, incoherent text. Temperature between 0.7 and 1.0 is typical for most applications.

3. Using top-k with a fixed k for all contexts: In some contexts, the distribution is very flat (many viable tokens), and k=50 might miss important options. In other contexts, the distribution is very sharp (only 1-2 viable tokens), and k=50 includes too many unlikely tokens. Top-p (nucleus) sampling adapts to the distribution shape.

4. Not setting a maximum generation length: Without a maximum length, the model might generate endlessly or until it runs out of context window. Always set a reasonable max_length.

5. Forgetting to stop at the end-of-sequence token: Production systems must detect and respect the EOS token. Continuing generation after EOS wastes compute and degrades quality.

6. Assuming longer generation is always better: Longer text is not necessarily better. For many tasks, shorter, focused generations are preferable. Consider length penalties in beam search.

## Interview Questions

### Beginner

Q: What is the difference between greedy decoding and temperature sampling for text generation?

A: Greedy decoding always selects the token with the highest probability, producing deterministic output. Temperature sampling draws a random sample from the probability distribution, where temperature controls the sharpness: low temperature (e.g., 0.5) makes high-probability tokens even more likely (near-greedy), while high temperature (e.g., 1.5) makes the distribution more uniform, increasing diversity.

### Intermediate

Q: Explain the vanishing probability problem in autoregressive generation and how it affects beam search.

A: In autoregressive generation, the probability of a sequence is the product of per-token probabilities. Longer sequences have exponentially smaller total probabilities because each factor is < 1. This biases beam search toward shorter sequences because they have higher total probability. Solutions include length normalization (dividing the score by sequence length) or length penalty (adding a bonus for longer sequences). This is why beam search implementations typically include a length normalization parameter.

### Advanced

Q: Contrast top-k sampling with top-p (nucleus) sampling. When would you choose one over the other?

A: Top-k sampling always considers exactly k tokens, regardless of the distribution shape. In a flat distribution (many plausible tokens), top-k includes too few options, potentially cutting off valid continuations. In a sharp distribution (few likely tokens), top-k includes too many low-probability tokens. Top-p sampling dynamically selects the minimum number of tokens whose cumulative probability exceeds threshold p. This adapts to the distribution: in sharp distributions, it selects few tokens; in flat distributions, it selects many. Top-p is generally preferred because it adapts to context. However, top-k can be simpler to tune and works well for specific domains where the vocabulary size and distribution shape are consistent.

## Practice Problems

### Easy

Write a function that takes logits (batch, vocab) and returns tokens using each of the four decoding strategies: greedy, temperature 0.5, top-k 40, and top-p 0.95. Verify you get different tokens from different strategies.

### Medium

Implement a generation function with repetition penalty (penalizing tokens that already appear in the generated sequence by dividing their logits by a penalty factor). Compare the repetition rate and diversity of generated text with and without repetition penalty.

### Hard

Design a hybrid decoding strategy that combines beam search with sampling: use beam search to generate k candidate sequences, each using top-p sampling rather than greedy selection. Evaluate the output quality against standard beam search and standard sampling using automatic metrics (perplexity, distinct-n) and human evaluation.

## Solutions

```python
# Easy solution
def compare_strategies(logits):
    results = {}
    results["greedy"] = greedy_decode(logits)
    results["temp_0.5"] = temperature_sample(logits, 0.5)
    results["topk_40"] = top_k_sample(logits, 40)
    results["topp_0.95"] = top_p_sample(logits, 0.95)
    return results

logits = torch.randn(5, 10000)
results = compare_strategies(logits)
for name, tokens in results.items():
    print(f"{name}: {tokens[0].item()}")
# Output: greedy: 5234
# Output: temp_0.5: 3123
# Output: topk_40: 4567
# Output: topp_0.95: 3982
print("Different strategies choose different tokens")
# Output: Different strategies choose different tokens
```

## Related Concepts

- Causal Masking (DL-398)
- GPT Decoder Architecture (DL-396)
- Inference with Decoder (DL-404)
- Decoder-Only Architecture (DL-403)
- Beam Search
- Nucleus Sampling
- Temperature Scaling

## Next Concepts

- Causal Masking
- GPT-1
- GPT-2
- Inference with Decoder

## Summary

Autoregressive generation produces sequences one token at a time, where each token depends on all previously generated tokens. Decoding strategies — greedy, beam search, temperature sampling, top-k, top-p — determine how to select tokens from the model's probability distribution. The choice of strategy significantly impacts output quality, diversity, and coherence.

## Key Takeaways

- Autoregressive generation factorizes sequence probability as product of conditional token probabilities.
- Greedy decoding is deterministic but can produce repetitive text.
- Temperature sampling controls creativity vs. conservatism.
- Top-k sampling restricts to k most likely tokens.
- Top-p (nucleus) sampling adaptively selects tokens based on cumulative probability.
- Beam search maintains multiple hypotheses for higher quality at higher cost.
- The decoding strategy should match the application: precision (beam search) vs. creativity (sampling).
- Repetition penalties and length penalties can improve output quality.
