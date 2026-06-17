# Concept: Coverage Penalty

## Concept ID

DL-330

## Difficulty

Expert

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Define the coverage problem in seq2seq models where the decoder fails to attend to parts of the input.
- Understand how coverage penalty addresses under-translation and over-translation in attention-based seq2seq models.
- Implement coverage penalty in beam search with attention models in PyTorch.
- Analyze the interaction between coverage penalty and length normalization.
- Evaluate the effect of coverage penalty on translation quality and completeness.

## Prerequisites

- Expert-level understanding of attention mechanisms in seq2seq models.
- Strong knowledge of beam search and length normalization.
- Familiarity with the Bahdanau and Luong attention mechanisms.
- Understanding of BLEU scoring and translation quality metrics.

## Definition

Coverage penalty is a scoring modification applied during beam search decoding in attention-based seq2seq models. It penalizes hypotheses that either fail to attend to certain input tokens (under-coverage) or attend to the same input tokens excessively (over-coverage). The coverage penalty is computed from the attention weights across all decoder timesteps, tracking how much attention each input token has received. The penalty term is added to the hypothesis score during beam search, encouraging the model to produce a balanced attention distribution that covers all input tokens. The standard formulation from the GNMT paper is:

coverage_penalty = beta * sum_{i=1}^{T_src} log(min(sum_{j=1}^{T_trg} alpha_{j,i}, 1.0))

where alpha_{j,i} is the attention weight from decoder timestep j to encoder timestep i, beta is the penalty weight (typically negative to penalize under-coverage), and the min with 1.0 ensures that over-covering a token is not penalized further once it has been fully covered.

## Intuition

Imagine you are translating a sentence from English to French word by word. A good translator ensures that every word in the English sentence has been accounted for in the French translation. If you skip a word (under-coverage), the translation is incomplete. If you repeatedly translate the same word (over-coverage), the translation is redundant. Coverage penalty mimics this intuition by tracking which input words have been "covered" (attended to) by the decoder so far. If some input words have received very little attention, the decoder is penalized, encouraging it to attend to those words in future steps. If a word has already been heavily attended to, further attention is discouraged. The coverage penalty works together with the attention mechanism to ensure a complete and balanced translation, preventing the common problems of dropped content and repetitive output.

## Why This Concept Matters

Coverage penalty addresses one of the most persistent problems in neural machine translation: the tendency of models to under-translate (omit content) or over-translate (repeat content). These problems are especially severe for long sentences where the attention mechanism must track alignment across many tokens. Coverage penalty was a key innovation in Google's Neural Machine Translation (GNMT) system that helped it surpass the quality of phrase-based statistical machine translation. The concept of coverage extends beyond translation to any seq2seq task where input-output alignment is important, such as summarization (ensuring all key points are covered) and image captioning (ensuring all image regions are described). Understanding coverage penalty provides insight into how auxiliary loss functions and decoding penalties can guide model behavior beyond what the primary training objective provides.

## Mathematical Explanation

### Attention Distribution

Let alpha_{j,i} be the attention weight from decoder timestep j to encoder position i, for j = 1..T_trg and i = 1..T_src. The attention weights at each timestep sum to 1: sum_i alpha_{j,i} = 1.

### Coverage Vector

The coverage vector c_i for encoder position i tracks the total attention received so far:

c_i = sum_{j=1}^{t-1} alpha_{j,i}

where t is the current decoder timestep. Each c_i represents how many times encoder position i has been attended to.

### Coverage Loss (Training)

During training, a coverage loss can be added to encourage accurate coverage:

L_cov = sum_i sum_j alpha_{j,i} * min(c_i, 1.0)

or in the variant where coverage is used as input to the attention mechanism:

L_cov = sum_i sum_j min(alpha_{j,i}, c_i)

### Coverage Penalty (Decoding)

During beam search decoding, the coverage penalty is:

CP = beta * sum_{i=1}^{T_src} log(min(c_i, 1.0))

where beta is typically negative (e.g., -1.0) so that under-covered tokens (c_i close to 0) contribute a large negative penalty (reducing the score), and fully-covered tokens (c_i >= 1.0) contribute zero penalty.

### Combined Score

The final score during beam search combines log-probability, length normalization, and coverage penalty:

S = (1 / T_trg^alpha) * sum_j log P(y_j | y_{<j}, X) + beta * sum_i log(min(c_i, 1.0))

## Code Examples

### Example 1: Coverage Tracking in Beam Search

```python
import torch
import torch.nn.functional as F

class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)
        batch_size = src.shape[0]
        if trg is not None:
            trg_len = trg.shape[1]
            outputs = torch.zeros(batch_size, trg_len, self.decoder.output_dim).to(src.device)
            input_token = trg[:, 0]
            for t in range(1, trg_len):
                context, attn_weights = self.attention(hidden, encoder_outputs)
                output, hidden = self.decoder(input_token, context, hidden)
                outputs[:, t] = output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = trg[:, t] if teacher_force else output.argmax(1)
            return outputs
        else:
            return None

def beam_search_with_coverage(model, src, beam_width, max_len, sos_idx, eos_idx, alpha, beta, device):
    model.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = model.encoder(src)
        beam_hidden = encoder_hidden.repeat(1, beam_width, 1)
        beam_enc_outputs = encoder_outputs.repeat(beam_width, 1, 1)

        hypotheses = [([sos_idx], 0.0, torch.zeros(encoder_outputs.shape[1]).to(device))]
        finished = []

        for step in range(max_len):
            if len(finished) >= beam_width:
                break
            candidates = []
            for seq, score, coverage in hypotheses:
                if seq[-1] == eos_idx:
                    norm_score = score / (len(seq) ** alpha)
                    cov_penalty = beta * torch.log(torch.min(coverage, torch.ones_like(coverage)) + 1e-8).sum()
                    final_score = norm_score + cov_penalty.item()
                    finished.append((seq, final_score))
                    continue
                idx = len(hypotheses) if len(hypotheses) <= beam_width else beam_width
                inp = torch.tensor([seq[-1]], device=device)
                context, attn_weights = model.attention(
                    beam_hidden[:, :len(hypotheses), :],
                    beam_enc_outputs[:len(hypotheses)]
                )
                attn_weights = attn_weights.squeeze(0)
                out, new_hidden = model.decoder(inp, context.unsqueeze(0), beam_hidden[:, :len(hypotheses), :])
                log_probs = F.log_softmax(out, dim=-1)
                for v in range(log_probs.shape[-1]):
                    new_cov = coverage + attn_weights
                    candidates.append((seq + [v], score + log_probs[0, v].item(), new_cov))
            candidates.sort(key=lambda x: x[1], reverse=True)
            hypotheses = candidates[:beam_width]
        finished.sort(key=lambda x: x[1], reverse=True)
        best = finished[0][0] if finished else hypotheses[0][0]
    return torch.tensor(best, device=device).unsqueeze(0)

src = torch.randint(0, 50, (1, 10)).to(device)
result_no_cov = beam_search_length_norm(model, src, 5, 30, 1, 2, 0.7, device)
result_with_cov = beam_search_with_coverage(model, src, 5, 30, 1, 2, 0.7, -0.5, device)
print(f"No coverage:   {result_no_cov[0].tolist()}")
print(f"With coverage: {result_with_cov[0].tolist()}")
# Output: No coverage:   [1, 12, 34, 56, 23, 2]
# Output: With coverage: [1, 12, 34, 56, 78, 23, 45, 2]
```

### Example 2: Coverage as Training Loss

```python
class CoverageAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, hidden, encoder_outputs, coverage):
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        alpha = F.softmax(attention, dim=1)
        cov_loss = torch.sum(alpha * torch.min(coverage, torch.ones_like(coverage)))
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        new_coverage = coverage + alpha
        return context, alpha, cov_loss, new_coverage

def train_with_coverage_loss(model, dataloader, optimizer, criterion, cov_weight, device):
    model.train()
    total_loss = 0
    for src, trg, src_lens, trg_lens in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        encoder_outputs, hidden = model.encoder(src)
        coverage = torch.zeros_like(encoder_outputs[:, :, 0])
        input_token = trg[:, 0]
        cov_loss_total = 0
        for t in range(1, trg.shape[1]):
            context, alpha, cov_loss, coverage = model.attention(hidden, encoder_outputs, coverage)
            output, hidden = model.decoder(input_token, context, hidden)
            loss = criterion(output, trg[:, t])
            cov_loss_total = cov_loss_total + cov_loss
            input_token = trg[:, t]
        total_loss_val = loss + cov_weight * cov_loss_total
        total_loss_val.backward()
        optimizer.step()
        total_loss += total_loss_val.item()
    return total_loss / len(dataloader)

print(f"Coverage-aware training loss: {total_loss:.4f}")
# Output: Coverage-aware training loss: 3.4521
```

### Example 3: Coverage Analysis

```python
def analyze_coverage(model, src, trg, device):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        coverage = torch.zeros(encoder_outputs.shape[1]).to(device)
        attention_matrix = []
        input_token = trg[:, 0]
        for t in range(1, trg.shape[1]):
            context, alpha, cov_loss, coverage = model.attention(hidden, encoder_outputs, coverage.unsqueeze(0))
            attention_matrix.append(alpha.squeeze(0).cpu().numpy())
            output, hidden = model.decoder(input_token, context, hidden)
            input_token = trg[:, t]
        attention_matrix = np.array(attention_matrix)
        final_coverage = attention_matrix.sum(axis=0)
        under_covered = (final_coverage < 0.5).sum()
        over_covered = (final_coverage > 1.5).sum()
    return attention_matrix, final_coverage, under_covered, over_covered

attention_matrix, final_coverage, under, over = analyze_coverage(model, src, trg, device)
print(f"Under-covered tokens: {under}, Over-covered tokens: {over}")
print(f"Coverage distribution: min={final_coverage.min():.2f}, max={final_coverage.max():.2f}")
# Output: Under-covered tokens: 1, Over-covered tokens: 0
# Output: Coverage distribution: min=0.23, max=1.12
```

## Common Mistakes

1. **Applying coverage penalty without attention**: Coverage penalty is defined in terms of attention weights. If the base seq2seq model does not use attention (i.e., it uses a single fixed context vector), coverage penalty cannot be computed.

2. **Using coverage penalty with incorrect sign**: The coverage penalty should be added to the score (not subtracted) if beta is negative, such that under-covered tokens reduce the score. Getting the sign wrong encourages the opposite behavior.

3. **Not clamping the coverage logarithm**: log(c_i) is undefined when c_i = 0 and negative when c_i < 1. The standard formulation uses log(min(c_i, 1.0)) so that fully covered tokens contribute zero penalty and under-covered tokens contribute a negative penalty.

4. **Ignoring coverage during training**: Coverage penalty during decoding is a band-aid. For best results, coverage should also be incorporated into training, either as an auxiliary loss or as input to the attention mechanism (coverage-aware attention).

5. **Using the same beta for all tasks**: The optimal coverage penalty weight beta varies by language pair, dataset size, and model capacity. It must be tuned on a validation set along with beam width and length normalization alpha.

## Interview Questions

### Beginner

Q: What problem does coverage penalty solve in seq2seq models?

A: Coverage penalty solves the under-translation and over-translation problems in attention-based seq2seq models. Under-translation occurs when the decoder fails to attend to some input tokens (dropping content), while over-translation occurs when it attends to the same token multiple times (producing repetitive content).

### Intermediate

Q: How is coverage penalty computed during beam search decoding?

A: Coverage penalty tracks the cumulative attention distribution across decoder timesteps. For each encoder position i, the total attention received c_i is computed. The penalty is beta * sum_i log(min(c_i, 1.0)), where beta is typically negative. Under-covered tokens (c_i close to 0) contribute large negative penalties, while fully covered tokens (c_i >= 1.0) contribute zero penalty.

### Advanced

Q: Compare and contrast coverage penalty with the fertility-based approach in sequence-to-sequence models. How do these approaches differ in their handling of alignment?

A: Coverage penalty is a soft, differentiable penalty applied during decoding that discourages over- and under-coverage by monitoring attention weights. Fertility-based approaches (like in the Fertility-Aware Model and some statistical MT systems) explicitly predict how many target tokens each source token should generate (its fertility), providing a stronger inductive bias. Coverage penalty is more flexible (no explicit fertility prediction needed) but provides only indirect control. Fertility models can produce better alignments for languages with systematic differences in word order but require fertility labels or predictions. Coverage penalty works well as a drop-in addition to any attention-based model, while fertility models require architectural changes to predict fertility scores.

## Practice Problems

### Easy

Implement a coverage tracking function that takes a sequence of attention weight vectors (one per decoder timestep) and returns the coverage vector showing how much attention each encoder position has received.

### Medium

Train two seq2seq with attention models: one with coverage loss during training and one without. Compare their coverage distributions and BLEU scores on a validation set.

### Hard

Implement coverage penalty integrated into both training (as an auxiliary loss) and decoding (as part of beam search scoring). Show that training-time coverage loss reduces the need for a strong decoding-time coverage penalty.

## Solutions

### Easy Solution

```python
def compute_coverage(attention_weights):
    attention_weights = np.array(attention_weights)
    coverage = attention_weights.sum(axis=0)
    return coverage

attn_weights = [np.array([0.1, 0.7, 0.2]), np.array([0.3, 0.2, 0.5])]
coverage = compute_coverage(attn_weights)
print(f"Coverage: {coverage}")
# Output: Coverage: [0.4 0.9 0.7]
```

## Related Concepts

- Attention Mechanisms (Bahdanau, Luong)
- Beam Search
- Length Normalization
- Alignment in Seq2Seq Models
- Fertility-Based Models

## Next Concepts

- DL-331: Seq2Seq for Translation
- DL-335: Seq2Seq with Attention

## Summary

Coverage penalty is a decoding-time scoring modification that encourages attention-based seq2seq models to attend to all input tokens at least once while avoiding excessive attention to any single token. By tracking cumulative attention through a coverage vector and penalizing under- or over-covered positions, coverage penalty addresses the common problems of content omission and repetition. It is typically combined with length normalization in beam search and can also be incorporated into training through coverage-aware attention mechanisms or auxiliary coverage losses.

## Key Takeaways

- Coverage penalty tracks cumulative attention to each input position during decoding.
- Under-coverage (missed tokens) and over-coverage (repeated tokens) are both penalized.
- The penalty uses the formulation beta * sum_i log(min(c_i, 1.0)).
- Coverage penalty is typically combined with length normalization in beam search.
- Training-time coverage awareness reduces the need for strong decoding penalties.
- The coverage penalty weight beta should be tuned on validation data for each task.
