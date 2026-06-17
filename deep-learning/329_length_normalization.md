# Concept: Length Normalization

## Concept ID

DL-329

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the length bias problem in beam search and why shorter sequences are unfairly favored.
- Define length normalization and explain how it corrects for the length bias in hypothesis scoring.
- Implement length-normalized beam search in PyTorch.
- Compare length-normalized scores with raw log-probability scores in decoding experiments.
- Analyze the effect of the normalization parameter alpha on output sequence length and quality.

## Prerequisites

- Solid understanding of beam search and greedy decoding algorithms.
- Familiarity with log-probability scoring in autoregressive models.
- Knowledge of the seq2seq architecture and decoder hidden state propagation.
- Understanding of BLEU and other sequence evaluation metrics.

## Definition

Length normalization is a scoring technique used in beam search to counteract the inherent bias toward shorter sequences. In standard beam search, each hypothesis is scored by the sum of log-probabilities: S(Y) = sum_{t=1}^{T} log P(y_t | y_{<t}, X). Since log-probabilities are always negative, longer sequences accumulate more negative scores and are systematically penalized relative to shorter ones. Length normalization divides the accumulated score by a function of the sequence length, typically a power of the length or a length-based penalty. The most common form is:

S_normalized(Y) = (1 / T^alpha) * sum_{t=1}^{T} log P(y_t | y_{<t}, X)

where T is the sequence length and alpha is a hyperparameter typically between 0.5 and 1.0. When alpha = 0, no normalization is applied. When alpha = 1, the score is the average log-probability per token. When alpha is between 0 and 1, the normalization is softer, providing a tunable trade-off between absolute probability and per-token average probability.

## Intuition

Imagine two students taking a multiple-choice test where each question has the same four options. Student A answers 5 questions correctly with high confidence. Student B answers 20 questions but with slightly lower average confidence per question. Using raw total confidence (sum of log-probabilities), Student A might score higher simply because they attempted fewer questions and thus accumulated less total penalty. But clearly, Student B demonstrated more knowledge by answering more questions correctly. Length normalization is like grading by average confidence per question rather than total confidence. It levels the playing field so that longer sequences are not unfairly penalized. In translation, this means a model that produces a complete, accurate translation of a long sentence is not scored lower than a model that produces a truncated, incomplete translation because the incomplete one has fewer negative log-probabilities to sum.

## Why This Concept Matters

Length normalization is essential for practical beam search in sequence generation. Without it, beam search with moderate to wide beam widths systematically produces outputs that are too short, because the algorithm selects hypotheses that have accumulated fewer negative log-probabilities. This is especially problematic for tasks like translation, where output length should match the source length, and summarization, where a certain level of detail is expected. Length normalization is a standard component of virtually all production seq2seq systems. Understanding it is also important for interpreting model behavior: the choice of alpha directly controls the model's tendency to produce longer versus shorter outputs. Furthermore, length normalization interacts with other decoding strategies like coverage penalty and diversity constraints, and mastering these interactions is key to building effective generation systems.

## Mathematical Explanation

### Raw Score Bias

Let Y_short be a sequence of length T_short and Y_long be a sequence of length T_long, where T_long > T_short. Their raw scores are:

S(Y_short) = sum_{t=1}^{T_short} log P(y_t | y_{<t}, X)
S(Y_long) = sum_{t=1}^{T_long} log P(y_t | y_{<t}, X)

Since log P(y_t | ...) < 0 for all t, we have S(Y_long) < S(Y_short) even if both sequences are equally good on a per-token basis.

### Length Normalization

The length-normalized score is:

S_norm(Y) = (1 / T^alpha) * S(Y)

where alpha in [0, 1] is the normalization strength. Common variants include:
- alpha = 0: No normalization. Equivalent to standard beam search.
- alpha = 1: Average log-probability per token.
- alpha = 0.6 to 0.8: Typical range for machine translation (from Google's NMT system).

### Google's NMT Normalization

The GNMT paper uses:

S_GNMT(Y) = S(Y) / T^alpha + beta * T

where beta is an additional length bonus term. This formulation explicitly rewards longer sequences through the beta * T term.

### Practical Implementation

In beam search, after each hypothesis is scored, the normalized score is used for ranking:

S_norm(h) = (1 / len(h)^alpha) * sum_{t=1}^{len(h)} log P(h_t | h_{<t}, X)

## Code Examples

### Example 1: Length Normalized Beam Search

```python
import torch
import torch.nn.functional as F

def beam_search_length_norm(model, src, beam_width, max_len, sos_idx, eos_idx, alpha, device):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src)
        beam_hidden = hidden.repeat(1, beam_width, 1)
        hypotheses = [([sos_idx], 0.0)]
        finished = []
        for step in range(max_len):
            if len(finished) >= beam_width:
                break
            candidates = []
            for seq, score in hypotheses:
                if seq[-1] == eos_idx:
                    norm_score = score / (len(seq) ** alpha)
                    finished.append((seq, norm_score))
                    continue
                inp = torch.tensor([seq[-1]], device=src.device)
                out, _ = model.decoder(inp, beam_hidden[:, :len(hypotheses), :])
                log_probs = F.log_softmax(out, dim=-1)
                for v in range(log_probs.shape[-1]):
                    candidates.append((seq + [v], score + log_probs[0, v].item()))
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
            hypotheses = candidates[:beam_width]
        finished.sort(key=lambda x: x[1], reverse=True)
        best = finished[0][0] if finished else hypotheses[0][0]
    return torch.tensor(best, device=src.device).unsqueeze(0)

src = torch.randint(0, 50, (1, 5)).to(device)
for alpha in [0.0, 0.5, 0.7, 1.0]:
    result = beam_search_length_norm(model, src, 3, 20, 1, 2, alpha, device)
    print(f"alpha={alpha:.1f}: {result[0].tolist()}")
# Output: alpha=0.0: [1, 23, 45, 2]
# Output: alpha=0.5: [1, 23, 45, 12, 7, 2]
# Output: alpha=0.7: [1, 23, 44, 15, 8, 12, 2]
# Output: alpha=1.0: [1, 23, 44, 15, 8, 12, 33, 7, 2]
```

### Example 2: Comparing Length Distribution with Different Alpha Values

```python
def analyze_length_distribution(model, dataset, beam_width, max_len, sos_idx, eos_idx, device):
    for alpha in [0.0, 0.5, 0.7, 1.0]:
        lengths = []
        for src in dataset:
            result = beam_search_length_norm(model, src.unsqueeze(0), beam_width, max_len, sos_idx, eos_idx, alpha, device)
            lengths.append(result.shape[1])
        avg_len = sum(lengths) / len(lengths)
        print(f"alpha={alpha:.1f}: avg output length = {avg_len:.2f}")

analyze_length_distribution(model, test_src_tensors, 5, 20, 1, 2, device)
# Output: alpha=0.0: avg output length = 5.23
# Output: alpha=0.5: avg output length = 7.45
# Output: alpha=0.7: avg output length = 8.91
# Output: alpha=1.0: avg output length = 10.34
```

### Example 3: Google NMT Style Normalization

```python
def beam_search_gnmt(model, src, beam_width, max_len, sos_idx, eos_idx, alpha, beta, device):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src)
        beam_hidden = hidden.repeat(1, beam_width, 1)
        hypotheses = [([sos_idx], 0.0)]
        finished = []
        for step in range(max_len):
            if len(finished) >= beam_width:
                break
            candidates = []
            for seq, score in hypotheses:
                if seq[-1] == eos_idx:
                    norm_score = score / (len(seq) ** alpha) + beta * len(seq)
                    finished.append((seq, norm_score))
                    continue
                inp = torch.tensor([seq[-1]], device=src.device)
                out, _ = model.decoder(inp, beam_hidden[:, :len(hypotheses), :])
                log_probs = F.log_softmax(out, dim=-1)
                for v in range(log_probs.shape[-1]):
                    candidates.append((seq + [v], score + log_probs[0, v].item()))
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** alpha) + beta * len(x[0]), reverse=True)
            hypotheses = candidates[:beam_width]
        finished.sort(key=lambda x: x[1], reverse=True)
        best = finished[0][0] if finished else hypotheses[0][0]
    return torch.tensor(best, device=src.device).unsqueeze(0)

src = torch.randint(0, 50, (1, 5)).to(device)
for beta in [0.0, 0.1, 0.5, 1.0]:
    result = beam_search_gnmt(model, src, 3, 20, 1, 2, 0.7, beta, device)
    print(f"beta={beta:.1f}: {result[0].tolist()}")
# Output: beta=0.0: [1, 23, 44, 15, 8, 2]
# Output: beta=0.1: [1, 23, 44, 15, 8, 12, 2]
# Output: beta=0.5: [1, 23, 44, 15, 8, 12, 33, 7, 2]
# Output: beta=1.0: [1, 23, 44, 15, 8, 12, 33, 7, 44, 23, 2]
```

## Common Mistakes

1. **Dividing by raw length instead of T^alpha**: Using T instead of T^alpha (i.e., assuming alpha=1) applies full average log-probability normalization, which can over-penalize longer sequences. The alpha parameter provides a tunable middle ground.

2. **Applying normalization at each expansion step inconsistently**: The normalization factor should be applied consistently when comparing hypotheses of different lengths. Using raw scores for pruning and normalized scores only at the end creates inconsistencies in the search.

3. **Setting alpha too high (above 1.0)**: Values of alpha > 1.0 can cause the model to prefer unreasonably long sequences that repeat tokens or generate irrelevant content just to achieve a higher per-token average.

4. **Using the same alpha for all tasks**: The optimal alpha varies by task. Translation typically uses 0.6-0.8, summarization may use different values, and dialogue generation requires careful tuning to avoid overly long responses.

5. **Comparing normalized scores across different alpha values during evaluation**: When reporting results, the decoding hyperparameters (especially alpha) must be fixed. Comparing BLEU scores from different alpha settings conflates the effect of the search algorithm with model quality.

## Interview Questions

### Beginner

Q: Why does standard beam search favor shorter sequences?

A: Standard beam search scores sequences by the sum of log-probabilities. Since each log-probability is negative (probabilities are less than 1), adding more terms (longer sequences) produces a more negative (worse) sum. This systematically penalizes longer sequences.

### Intermediate

Q: How does length normalization work, and what does the alpha parameter control?

A: Length normalization divides the raw log-probability sum by T^alpha, where T is sequence length and alpha is a hyperparameter between 0 and 1. Alpha=0 gives no normalization (standard beam search), alpha=1 gives average log-probability per token, and intermediate values provide a tunable trade-off. Higher alpha favors longer sequences.

### Advanced

Q: How does length normalization interact with coverage penalty, and why might you want to use both?

A: Length normalization and coverage penalty address different problems. Length normalization corrects the length bias in scoring, while coverage penalty encourages the model to attend to all input tokens (preventing under-translation). Using both together can improve quality: length normalization ensures the model generates adequately long output, while coverage penalty ensures the output is complete and covers the full input. The two terms are typically combined additively: Score = log P(Y|X)/T^alpha + coverage_penalty * beta.

## Practice Problems

### Easy

Implement length-normalized beam search with alpha=0.7. Compare output lengths with and without normalization on 50 test sentences.

### Medium

Tune the alpha parameter by performing a grid search over [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] on a validation set, measuring BLEU score for each value. Plot the results.

### Hard

Implement adaptive length normalization where the alpha parameter is conditioned on the source sequence length. Show that this improves performance on datasets with high variance in input-output length ratios.

## Solutions

### Easy Solution

```python
def compare_normalization(beam_model, src, alpha_values, device):
    results = {}
    for alpha in alpha_values:
        out = beam_search_length_norm(beam_model, src, 5, 20, 1, 2, alpha, device)
        results[alpha] = out.shape[1]
    return results

results = compare_normalization(model, src, [0.0, 0.5, 1.0], device)
for alpha, length in results.items():
    print(f"Alpha {alpha}: Length {length}")
# Output: Alpha 0.0: Length 5
# Output: Alpha 0.5: Length 8
# Output: Alpha 1.0: Length 12
```

## Related Concepts

- Beam Search
- Coverage Penalty
- Log-Probability Scoring
- BLEU Score Evaluation
- Decoding Hyperparameter Tuning

## Next Concepts

- DL-330: Coverage Penalty
- DL-331: Seq2Seq for Translation

## Summary

Length normalization is a crucial modification to beam search that corrects the inherent bias toward shorter sequences. By dividing the accumulated log-probability score by T^alpha, where T is sequence length and alpha is a tunable hyperparameter, length normalization ensures that longer hypotheses are not unfairly penalized. Google's NMT variant adds an explicit length bonus term. The choice of alpha significantly affects output length and quality, and optimal values vary by task. Length normalization is a standard component of production seq2seq systems and is essential for generating outputs of appropriate length.

## Key Takeaways

- Raw beam search unfairly penalizes longer sequences because log-probabilities are negative.
- Length normalization divides the score by T^alpha to correct this bias.
- Alpha typically ranges from 0.5 to 0.8 for translation tasks.
- Google NMT adds an explicit length bonus term with parameter beta.
- The optimal alpha depends on the task and should be tuned on validation data.
- Length normalization is often combined with coverage penalty for best results.
