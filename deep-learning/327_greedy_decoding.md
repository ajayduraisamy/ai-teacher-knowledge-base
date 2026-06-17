# Concept: Greedy Decoding

## Concept ID

DL-327

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand the greedy decoding algorithm for autoregressive sequence generation in seq2seq models.
- Implement greedy decoding in PyTorch for both training-free inference and validation.
- Analyze the strengths and limitations of greedy decoding compared to more advanced search strategies.
- Recognize the problem of error propagation in greedy decoding and its impact on output quality.
- Evaluate scenarios where greedy decoding is sufficient versus when beam search is necessary.

## Prerequisites

- Understanding of the decoder RNN and autoregressive generation in seq2seq models.
- Familiarity with softmax probability distributions and argmax selection.
- Basic knowledge of sequence generation tasks like translation and summarization.

## Definition

Greedy decoding is the simplest decoding strategy for autoregressive sequence generation. At each timestep, the decoder selects the token with the highest probability from the output distribution and feeds it as input to the next timestep. Formally, at timestep \( t \):

\[
\hat{y}_t = \arg\max_{v \in V} P(y_t = v \mid y_{<t}, X)
\]

The generated token \( \hat{y}_t \) is then used as input for predicting \( y_{t+1} \). This process continues until the end-of-sequence token (EOS) is generated or a maximum length is reached. Greedy decoding makes a locally optimal choice at each step with no consideration of future consequences — it greedily maximizes the probability at each individual timestep rather than considering the joint probability of the entire sequence.

## Intuition

Greedy decoding is like navigating a maze by always taking the path that looks most promising from your current position, without considering what lies ahead. At each intersection, you pick the opening that seems widest or brightest, hoping it leads to the exit. Sometimes this works perfectly — the locally optimal choice is indeed the globally correct one. But other times, the locally optimal choice leads to a dead end, while a slightly less obvious choice would have opened up a clear path to the goal. In translation, this means the model might pick a common word early on that later forces an ungrammatical construction, when a less frequent but more appropriate word would have led to a better translation overall. Greedy decoding is fast and simple — it requires only a single forward pass per token — but it cannot recover from early mistakes and never considers alternative hypotheses.

## Why This Concept Matters

Greedy decoding is the default inference algorithm for most seq2seq models. Understanding it is essential because it establishes the baseline against which all other decoding strategies are compared. Despite its simplicity, greedy decoding performs surprisingly well for many tasks, especially when the model is confident and the output sequences are short. However, its limitations — particularly the inability to recover from early errors and the tendency to produce repetitive or generic output — motivate the development of more sophisticated decoding strategies like beam search, sampling-based methods, and length normalization. Greedy decoding also serves as the foundation for understanding the trade-off between decoding quality and computational cost, which is a central consideration in deploying sequence generation models in production. For many real-world applications, greedy decoding provides a fast, acceptable-quality baseline, and more expensive decoding is only used when additional quality is required.

## Mathematical Explanation

### Algorithm

Given a trained seq2seq model with decoder \( D \), start token SOS, end token EOS, and maximum length \( T_{\max} \):

1. Initialize \( t = 0 \), \( y_0 = \text{SOS} \), \( s_0 = \text{context vector from encoder} \).
2. For \( t = 1 \) to \( T_{\max} \):
   - Compute output distribution: \( o_t = D(y_{t-1}, s_{t-1}) \)
   - Select greedy token: \( \hat{y}_t = \arg\max_v \text{softmax}(o_t)[v] \)
   - If \( \hat{y}_t = \text{EOS} \), stop.
   - Update decoder state: \( s_t = \text{updated hidden state} \)
3. Return sequence \( (\hat{y}_1, \ldots, \hat{y}_{T}) \).

### Probability Interpretation

The greedy search selects the sequence that maximally increases the partial probability at each step:

\[
P(y_{1:T} \mid X) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, X)
\]

Greedy decoding selects the token that maximizes each factor individually:

\[
\hat{y}_t = \arg\max_{y_t} P(y_t \mid \hat{y}_{<t}, X)
\]

However, the globally optimal sequence \( y^*_{1:T} = \arg\max_{y_{1:T}} P(y_{1:T} \mid X) \) is not guaranteed to be found by greedy decoding because the greedy choice at step \( t \) may lead to low-probability continuations later.

### When Greedy is Optimal

If the model has the property that for all \( t \), the argmax token at step \( t \) leads to a continuation that is at least as good as any alternative, then greedy search is optimal. This property — the "optimal substructure" of the search problem — does not generally hold for seq2seq models.

## Code Examples

### Example 1: Basic Greedy Decoding

```python
import torch
import torch.nn.functional as F

def greedy_decode(model, src, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    with torch.no_grad():
        hidden = model.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long).to(device)
        output_sequence = [input_token]
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            probs = F.softmax(output, dim=-1)
            next_token = probs.argmax(dim=-1)
            output_sequence.append(next_token)
            input_token = next_token
            if (next_token == eos_idx).all():
                break
        output_sequence = torch.stack(output_sequence, dim=1)
    return output_sequence

sos_idx, eos_idx = 1, 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src = torch.randint(0, 50, (2, 5)).to(device)
result = greedy_decode(model, src, max_len=20, sos_idx=sos_idx, eos_idx=eos_idx, device=device)
print(f"Greedy decoded shape: {result.shape}")
print(f"Sequence 0: {result[0].tolist()}")
# Output: Greedy decoded shape: torch.Size([2, 8])
# Output: Sequence 0: [1, 23, 45, 12, 7, 2, 0, 0]
```

### Example 2: Greedy Decoding with an LSTM Seq2Seq Model

```python
def greedy_decode_lstm(model, src, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    with torch.no_grad():
        hidden, cell = model.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long).to(device)
        output_sequences = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            next_token = output.argmax(dim=-1)
            output_sequences.append(next_token)
            input_token = next_token
            if (next_token == eos_idx).all():
                break
        output_sequences = torch.stack(output_sequences, dim=1)
        sos_tensor = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(device)
        output_sequences = torch.cat([sos_tensor, output_sequences], dim=1)
    return output_sequences

src = torch.randint(0, 100, (1, 7)).to(device)
output = greedy_decode_lstm(lstm_model, src, max_len=15, sos_idx=1, eos_idx=2, device=device)
print(f"LSTM greedy output: {output.shape}")
print(f"Tokens: {output[0].tolist()}")
# Output: LSTM greedy output: torch.Size([1, 6])
# Output: Tokens: [1, 45, 67, 23, 12, 2]
```

### Example 3: Comparison of Greedy vs. Random Sampling

```python
def random_sample_decode(model, src, max_len, sos_idx, eos_idx, device, temperature=1.0):
    model.eval()
    batch_size = src.shape[0]
    with torch.no_grad():
        hidden = model.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long).to(device)
        output_sequence = []
        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            probs = F.softmax(output / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            output_sequence.append(next_token)
            input_token = next_token
            if (next_token == eos_idx).all():
                break
        output_sequence = torch.stack(output_sequence, dim=1)
    return output_sequence

greedy_result = greedy_decode(model, src, max_len=20, sos_idx=1, eos_idx=2, device=device)
sampled_result = random_sample_decode(model, src, max_len=20, sos_idx=1, eos_idx=2, device=device)

print(f"Greedy:   {greedy_result[0].tolist()}")
print(f"Sampled:  {sampled_result[0].tolist()}")
# Output: Greedy:   [1, 23, 45, 12, 2]
# Output: Sampled:  [1, 23, 44, 15, 67, 3, 2]
```

## Common Mistakes

1. **Not including SOS and EOS tokens in the output**: The generated sequence should include a SOS token at the beginning (added by the implementation) and stop when EOS is generated. Some implementations forget to prepend SOS or continue generating past EOS.

2. **Using teacher forcing during inference**: Teacher forcing uses ground truth tokens as decoder input. During inference, ground truth is unavailable, and the model must use its own predictions. Accidentally leaving teacher forcing enabled during inference gives unrealistically good results.

3. **Stopping only on EOS without a max length guard**: If the model never generates EOS (which can happen with poorly trained models), the decoding loop runs indefinitely. Always include a maximum length limit.

4. **Assuming monotonic EOS generation**: In batch decoding, different sequences in the batch may generate EOS at different timesteps. A naive loop that breaks when any sequence hits EOS truncates other sequences prematurely. Each sequence should be tracked independently.

5. **Not using torch.no_grad() during inference**: Greedy decoding (and all inference) should be wrapped in `torch.no_grad()` to disable gradient computation. Without this, memory usage grows linearly with sequence length as the computation graph is preserved.

## Interview Questions

### Beginner

Q: How does greedy decoding work in a seq2seq model?

A: At each timestep, the decoder outputs a probability distribution over the vocabulary. Greedy decoding selects the token with the highest probability as the next token and feeds it back as input to the decoder for the next timestep. This continues until EOS is generated or max length is reached.

### Intermediate

Q: What are the limitations of greedy decoding, and when might it fail?

A: Greedy decoding makes locally optimal choices that may lead to globally suboptimal sequences. It cannot recover from early mistakes, tends to produce repetitive text, and often generates generic or short outputs because it always chooses the most probable (common) tokens. It fails most dramatically on tasks requiring long-range coherence or where early ambiguous tokens have high probability but lead to dead ends.

### Advanced

Q: How does greedy decoding relate to the concept of exposure bias? Can greedy decoding itself be considered a cause of exposure bias?

A: Exposure bias is the mismatch between training (teacher forcing) and inference (free-running) conditions. Greedy decoding is the standard inference strategy that reveals this mismatch. During training with teacher forcing, the decoder never experiences the consequences of its own argmax choices. During greedy inference, the decoder conditions on its own argmax tokens, which may be out-of-distribution compared to ground truth tokens. So greedy decoding doesn't cause exposure bias, but it is the context in which exposure bias becomes problematic. Mitigations like scheduled sampling and beam search address the consequences of this mismatch.

## Practice Problems

### Easy

Implement a greedy decoding function for a GRU-based seq2seq model. The function should take a source sequence, decode greedily, and return the sequence of token indices including SOS and stopping at EOS.

### Medium

Compare greedy decoding performance against beam search (width 3, 5, 10) on a small English-to-French translation dataset. Measure BLEU scores and average decoding time per sentence. Plot the trade-off.

### Hard

Implement an oracle experiment: given a trained seq2seq model and a reference target sequence, compute how many tokens greedy decoding would need to get "correct" (matching the reference) at each position. Compare this with the model's confidence at those positions. Write a short analysis.

## Solutions

### Easy Solution

```python
def greedy_decode_gru(model, src, max_len, sos_idx, eos_idx, device):
    model.eval()
    batch_size = src.shape[0]
    with torch.no_grad():
        hidden = model.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long).to(device)
        tokens = [input_token]
        for _ in range(max_len):
            logits, hidden = model.decoder(input_token, hidden)
            next_token = logits.argmax(dim=-1)
            tokens.append(next_token)
            input_token = next_token
            if (next_token == eos_idx).all():
                break
        return torch.stack(tokens, dim=1)
```

## Related Concepts

- Beam Search
- Random Sampling and Top-K Sampling
- Autoregressive Generation
- Exposure Bias
- Teacher Forcing

## Next Concepts

- DL-328: Beam Search
- DL-329: Length Normalization
- DL-330: Coverage Penalty

## Summary

Greedy decoding is the simplest and fastest decoding strategy for autoregressive sequence generation. At each timestep, it selects the token with the highest probability from the output distribution. While computationally efficient and easy to implement, greedy decoding suffers from several limitations: it cannot revise early mistakes, tends to produce repetitive or generic output, and is not guaranteed to find the globally optimal sequence. Despite these limitations, greedy decoding serves as an essential baseline for evaluating more sophisticated decoding strategies and is adequate for many applications where speed is prioritized over marginal quality gains.

## Key Takeaways

- Greedy decoding selects the highest-probability token at each timestep.
- It is fast (single forward pass per token) and simple to implement.
- It cannot recover from early errors and may produce suboptimal sequences.
- Greedy decoding reveals the exposure bias problem in seq2seq models.
- It serves as the baseline against which beam search and sampling methods are compared.
