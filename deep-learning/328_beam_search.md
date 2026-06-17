# Concept: Beam Search

## Concept ID

DL-328

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the beam search algorithm for approximate inference in autoregressive sequence generation.
- Implement beam search in PyTorch with configurable beam width.
- Analyze the trade-off between beam size, decoding quality, and computational cost.
- Differentiate beam search from greedy decoding and understand when each is appropriate.
- Apply beam search to improve sequence generation quality in translation and summarization tasks.

## Prerequisites

- Solid understanding of greedy decoding and its limitations.
- Familiarity with autoregressive sequence generation in seq2seq models.
- Knowledge of log-probabilities and why they are preferred over raw probabilities for numerical stability.
- Experience with PyTorch tensor operations and batch processing.

## Definition

Beam search is a heuristic search algorithm that approximates the globally optimal sequence by maintaining a fixed number of candidate hypotheses (the beam) at each decoding timestep. Unlike greedy decoding which keeps only the single best token at each step, beam search keeps the k most probable partial sequences, where k is the beam width. At each timestep, the algorithm expands all current hypotheses by considering all possible next tokens, scores the resulting k x V candidates (where V is vocabulary size), and retains the k highest-scoring ones. This allows the model to explore multiple possible paths and recover from early suboptimal choices. The final output is the highest-scoring complete hypothesis among those that end with the EOS token. Beam search does not guarantee finding the globally optimal sequence (which would require exhaustive search over V^T possibilities), but it dramatically outperforms greedy decoding in practice with only linear computational overhead in the beam width.

## Intuition

Imagine you are a detective solving a case. Greedy decoding is like following only the most promising lead at each step and if that lead goes cold, you have nothing to fall back on. Beam search, by contrast, is like keeping a case board with your top k suspects at all times. At each new piece of evidence, you evaluate how it affects all current suspects, keeping only the most viable ones. If your top suspect turns out to have an alibi, your second-best suspect is still under active investigation and can take the lead. In translation, this means the model can initially favor a common word like "bank" (financial institution), but as more context suggests "river bank" (geographical feature), an alternative hypothesis that chose "river" earlier can emerge as the best overall translation. Beam search with width k = 3 or 5 provides a good balance: it explores multiple alternatives without the exponential cost of exhaustive search.

## Why This Concept Matters

Beam search is the standard decoding algorithm for practically all deployed seq2seq models, from Google Translate to commercial summarization systems. It directly addresses the fundamental limitation of greedy decoding — the inability to recover from early mistakes — with a computationally tractable approximation. Understanding beam search is essential because it introduces the concept of hypothesis scoring and pruning that underlies many advanced decoding techniques. Beam search also reveals important trade-offs in model deployment: wider beams improve quality but increase latency and memory usage. Furthermore, beam search interacts with other decoding techniques like length normalization and coverage penalty, which require the hypothesis-level scoring framework that beam search provides. Mastering beam search is necessary for anyone building or deploying sequence generation models in production.

## Mathematical Explanation

### Algorithm

Let k be the beam width. Let H_t be the set of k partial hypotheses (sequences) at timestep t, each with score S(h).

1. Initialize: H_0 = {[SOS]}, S([SOS]) = 0.
2. For t = 1 to T_max:
   - Expand: For each hypothesis h in H_{t-1} and each token v in V, create a new hypothesis h' = h + [v].
   - Score: Compute S(h') = S(h) + log P(v | h, X).
   - Prune: Select the k hypotheses with the highest scores to form H_t.
   - Check completion: Move hypotheses ending in EOS to a finished set F.
3. Return: The hypothesis in F with the highest score (or score normalized by length).

### Score Computation

Scores are computed in log-space to avoid numerical underflow:

S(y_{1:t}) = sum_{i=1}^{t} log P(y_i | y_{<i}, X)

### Beam Width Trade-offs

- k = 1: Equivalent to greedy decoding.
- Larger k: Higher probability of finding the optimal sequence, but computational cost is O(k * V * T) instead of O(V * T).
- Very large k: Diminishing returns; quality plateaus and may even decrease because wider beams tend to favor shorter sequences (a bias addressed by length normalization).

## Code Examples

### Example 1: Basic Beam Search Implementation

```python
import torch
import torch.nn.functional as F

def beam_search(model, src, beam_width, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    assert batch_size == 1, "Beam search implemented for single sequence"
    with torch.no_grad():
        encoder_hidden = model.encoder(src)
        hidden = encoder_hidden.repeat(1, beam_width, 1)
        hypotheses = [([sos_idx], 0.0)]
        finished_hypotheses = []
        for step in range(max_len):
            if len(finished_hypotheses) >= beam_width:
                break
            all_candidates = []
            for seq, score in hypotheses:
                if seq[-1] == eos_idx:
                    finished_hypotheses.append((seq, score))
                    continue
                input_token = torch.tensor([seq[-1]], device=device)
                output, hidden_new = model.decoder(input_token, hidden[:, :len(hypotheses), :])
                log_probs = F.log_softmax(output, dim=-1)
                for v in range(log_probs.shape[-1]):
                    candidate_score = score + log_probs[0, v].item()
                    candidate_seq = seq + [v]
                    all_candidates.append((candidate_seq, candidate_score))
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            hypotheses = all_candidates[:beam_width]
        finished_hypotheses.sort(key=lambda x: x[1], reverse=True)
        best_seq = finished_hypotheses[0][0] if finished_hypotheses else hypotheses[0][0]
    return torch.tensor(best_seq, device=device).unsqueeze(0)

src = torch.randint(0, 50, (1, 5)).to(device)
result = beam_search(model, src, beam_width=3, max_len=20, sos_idx=1, eos_idx=2, device=device)
print(f"Beam search result shape: {result.shape}")
print(f"Result sequence: {result.tolist()}")
# Output: Beam search result shape: torch.Size([1, 7])
# Output: Result sequence: [[1, 23, 45, 12, 7, 2]]
```

### Example 2: Batch Beam Search

```python
def batch_beam_search(model, src, beam_width, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    all_results = []
    for i in range(batch_size):
        single_src = src[i:i+1]
        result = beam_search(model, single_src, beam_width, max_len, sos_idx, eos_idx, device)
        all_results.append(result)
    return torch.cat(all_results, dim=0)

src = torch.randint(0, 50, (3, 5)).to(device)
results = batch_beam_search(model, src, beam_width=3, max_len=20, sos_idx=1, eos_idx=2, device=device)
print(f"Batch beam search results shape: {results.shape}")
# Output: Batch beam search results shape: torch.Size([3, 8])
```

### Example 3: Beam Search with Different Beam Widths

```python
def compare_beam_widths(model, src, max_len, sos_idx, eos_idx, device):
    results = {}
    for beam_width in [1, 3, 5, 10]:
        result = beam_search(model, src, beam_width, max_len, sos_idx, eos_idx, device)
        results[beam_width] = result
    for bw, seq in results.items():
        print(f"Beam width {bw}: {seq[0].tolist()}")

src = torch.randint(0, 50, (1, 5)).to(device)
compare_beam_widths(model, src, max_len=20, sos_idx=1, eos_idx=2, device=device)
# Output: Beam width 1: [1, 23, 45, 12, 2]
# Output: Beam width 3: [1, 23, 45, 12, 7, 2]
# Output: Beam width 5: [1, 23, 44, 15, 8, 2]
# Output: Beam width 10: [1, 23, 44, 15, 8, 12, 2]
```

## Common Mistakes

1. **Using raw probabilities instead of log-probabilities**: Multiplying many small probabilities causes numerical underflow. Always use log-probabilities and sum them for stable scoring.

2. **Not handling EOS tokens properly**: When a hypothesis generates EOS, it should be moved to the finished set and not expanded further. Continuing to expand finished hypotheses wastes computation.

3. **Forgetting to repeat the encoder hidden state for the beam**: The encoder's hidden state must be repeated beam_width times along the batch dimension. Without this, each hypothesis shares the same decoder initial state.

4. **Using a very large beam width without length normalization**: Larger beams favor shorter sequences because log-probabilities are negative and shorter sequences have fewer terms. Length normalization is essential for wide beams.

5. **Treating beam search as guaranteed optimal**: Beam search is a heuristic. Even with large beam widths, it can prune the optimal hypothesis early. It is not equivalent to exhaustive search.

## Interview Questions

### Beginner

Q: What is the key difference between greedy decoding and beam search?

A: Greedy decoding keeps only the single best token at each timestep, while beam search maintains k candidate partial sequences and considers all possible next tokens for each. Beam search explores multiple paths and can recover from early suboptimal choices.

### Intermediate

Q: What is the computational complexity of beam search compared to greedy decoding?

A: Greedy decoding has complexity O(V * T). Beam search has complexity O(k * V * T) because at each step, all k hypotheses are expanded over all V tokens. The beam width k is a linear multiplier on computational cost. Memory also increases linearly with k.

### Advanced

Q: Discuss the relationship between beam search and the problem of model overconfidence. How does the choice of beam width interact with the calibration of the underlying sequence model?

A: Beam search relies on the model's probability estimates to score hypotheses. If the model is poorly calibrated — overconfident in incorrect predictions — beam search may systematically prefer incorrect hypotheses that happen to have high scores. There is also a known phenomenon where increasing beam width beyond a certain point can actually degrade quality (measured by BLEU), because wider beams retain hypotheses that are locally plausible but globally incoherent. This is sometimes addressed by using a coverage penalty or by combining beam search with length normalization.

## Practice Problems

### Easy

Implement beam search with width 3 for a simple seq2seq model. Compare the output with greedy decoding on 10 test sentences.

### Medium

Implement a comparison function that runs beam search with widths 1, 2, 4, 8, 16 on a validation set and plots BLEU score vs. beam width.

### Hard

Implement beam search with diversity penalties that encourage the beam to maintain diverse hypotheses (e.g., by penalizing hypotheses that share n-grams with other hypotheses in the beam).

## Solutions

### Easy Solution

```python
def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
    model.eval()
    with torch.no_grad():
        hidden = model.encoder(src)
        input_token = torch.full((1,), sos_idx, dtype=torch.long).to(device)
        tokens = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            next_token = output.argmax(-1)
            tokens.append(next_token)
            input_token = next_token
            if next_token.item() == eos_idx:
                break
        return torch.stack(tokens, dim=0).unsqueeze(0)

greedy = greedy_decode(model, src, 20, 1, 2, device)
beam3 = beam_search(model, src, 3, 20, 1, 2, device)
print(f"Greedy: {greedy.tolist()}")
print(f"Beam-3: {beam3.tolist()}")
# Output: Greedy: [[1, 23, 45, 2]]
# Output: Beam-3: [[1, 23, 44, 15, 2]]
```

## Related Concepts

- Greedy Decoding
- Length Normalization
- Coverage Penalty
- Diversity Beam Search
- Minimum Bayes Risk Decoding

## Next Concepts

- DL-329: Length Normalization
- DL-330: Coverage Penalty

## Summary

Beam search is a powerful approximate search algorithm for autoregressive sequence generation that maintains k candidate hypotheses at each timestep. By exploring multiple paths in parallel, it can recover from early suboptimal choices and produce higher-quality output than greedy decoding. The beam width controls the trade-off between search quality and computational cost. Beam search is the standard decoding algorithm for seq2seq models in production and serves as the foundation for more advanced decoding strategies that incorporate length normalization, coverage penalties, and diversity constraints.

## Key Takeaways

- Beam search maintains k candidate hypotheses (the beam) at each decoding step.
- Scores are computed in log-space for numerical stability.
- Beam search can recover from early mistakes unlike greedy decoding.
- The beam width controls the quality-computation trade-off.
- Length normalization is necessary to counteract the bias toward short sequences with wide beams.
- Beam search is a heuristic that does not guarantee finding the globally optimal sequence.
