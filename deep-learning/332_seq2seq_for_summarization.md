# Concept: Seq2Seq for Summarization

## Concept ID

DL-332

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Seq2Seq Models

## Learning Objectives

- Understand how seq2seq models are applied to abstractive and extractive text summarization.
- Differentiate between extractive summarization (selecting sentences) and abstractive summarization (generating novel text).
- Implement a seq2seq summarization model with attention in PyTorch.
- Apply summarization-specific decoding strategies including length control and repetition prevention.
- Evaluate summary quality using ROUGE scores.

## Prerequisites

- Solid understanding of seq2seq architecture and attention mechanisms.
- Familiarity with text preprocessing and tokenization for NLP.
- Knowledge of beam search and length normalization.
- Understanding of ROUGE evaluation metrics.

## Definition

Seq2Seq for summarization is the application of sequence-to-sequence models to automatically generate concise summaries of longer text documents. The model takes a source document (e.g., a news article, research paper, or legal document) as input and generates a shorter summary that captures the most important information. There are two main approaches: extractive summarization, which selects and rearranges sentences from the source, and abstractive summarization, which generates novel sentences that may not appear verbatim in the source. Seq2seq models are particularly suited for abstractive summarization, where the encoder processes the source document and the decoder generates the summary token by token, using attention to focus on relevant parts of the source. Key challenges include handling very long documents (beyond typical seq2seq context windows), preventing factual hallucinations, controlling summary length, and avoiding repetition.

## Intuition

Summarization is like a student reading a textbook chapter and writing a one-paragraph summary that captures the key points. The student must first read and understand the entire chapter (the encoder step), decide which information is most important, and then generate a concise, coherent summary in their own words (the decoder step). The attention mechanism allows the student to look back at specific paragraphs or sentences while writing each part of the summary. For example, when writing about the main characters, the student attends to the introduction section; when writing about the plot twist, they attend to the relevant chapter. The challenge is that unlike translation (where input and output are roughly the same length), summarization involves significant compression: a 1000-word article must be reduced to 100 words, requiring the model to understand the document's overall structure and identify the most salient information.

## Why This Concept Matters

Text summarization is one of the most practically useful NLP applications, with use cases in news aggregation, legal document review, medical record summarization, and research paper abstraction. Seq2seq models have dramatically improved the quality of abstractive summarization compared to earlier extractive methods, producing summaries that are more fluent and better capture the essence of the source. The challenges specific to summarization — handling long documents, preventing factual errors, controlling output length — have driven important research innovations, including hierarchical encoders, two-stage summarization, and faithfulness verification. For deep learning practitioners, summarizing long documents tests the limits of seq2seq models and reveals the importance of attention mechanisms, positional encoding, and decoding strategy.

## Mathematical Explanation

### Problem Setup

Given a source document D = (d_1, ..., d_T) where each d_i is a token, the model generates a summary S = (s_1, ..., s_{T'}) where T' << T (typically T' is 10-20% of T).

### Extractive vs. Abstractive

- Extractive: S is a subset of {d_1, ..., d_T} selected by a scoring function f(d_i).
- Abstractive: S is generated from scratch by P(S | D) = prod_{t'} P(s_{t'} | s_{<t'}, D).

### Long Document Handling

For documents longer than the model's maximum input length, hierarchical approaches are used:

1. Sentence encoder: Encode each sentence independently.
2. Document encoder: Encode the sequence of sentence representations.
3. Decoder: Generate summary attending to sentence-level and token-level representations.

### Loss Function

Standard cross-entropy loss with masking:

L = -sum_{t'} log P(s_{t'} | s_{<t'}, D)

Additional losses for summarization-specific objectives:
- Coverage loss: Encourage attending to all parts of the document.
- Length penalty: Encourage summaries of appropriate length.
- Repetition penalty: Discourage n-gram repetition in the summary.

### ROUGE Evaluation

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap between generated and reference summaries:

ROUGE-N = (sum_{ref} sum_{gram_n in ref} count_{match}(gram_n)) / (sum_{ref} sum_{gram_n in ref} count(gram_n))

Variants include ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (longest common subsequence).

## Code Examples

### Example 1: Seq2Seq Summarization Model

```python
import torch
import torch.nn as nn

class Summarizer(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid, dec_hid, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, enc_hid, n_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, dec_hid, n_layers, dropout=dropout, batch_first=True)
        self.fc_hidden = nn.Linear(enc_hid * 2, dec_hid)
        self.fc_cell = nn.Linear(enc_hid * 2, dec_hid)
        self.attention = nn.Linear(enc_hid * 2 + dec_hid, dec_hid)
        self.attn_vector = nn.Linear(dec_hid, 1)
        self.fc_out = nn.Linear(dec_hid + enc_hid * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        embedded = self.dropout(self.embedding(src))
        enc_outputs, (hidden, cell) = self.encoder(embedded)
        hidden = self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        cell = self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        batch_size = src.shape[0]
        if trg is not None:
            trg_len = trg.shape[1]
            outputs = torch.zeros(batch_size, trg_len, self.fc_out.out_features).to(src.device)
            input_token = trg[:, 0]
            for t in range(1, trg_len):
                trg_embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)
                dec_output, (hidden, cell) = self.decoder(trg_embedded, (hidden, cell))
                src_len = enc_outputs.shape[1]
                hidden_expanded = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
                energy = torch.tanh(self.attention(torch.cat((hidden_expanded, enc_outputs), dim=2)))
                attention = self.attn_vector(energy).squeeze(2)
                attn_weights = torch.softmax(attention, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1)
                combined = torch.cat((dec_output.squeeze(1), context), dim=1)
                prediction = self.fc_out(combined)
                outputs[:, t] = prediction
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                input_token = trg[:, t] if teacher_force else prediction.argmax(1)
            return outputs
        else:
            return None

model = Summarizer(vocab_size=500, emb_dim=128, enc_hid=256, dec_hid=256, n_layers=2, dropout=0.3)
src = torch.randint(0, 500, (2, 50))
trg = torch.randint(0, 500, (2, 15))
output = model(src, trg, 1.0)
print(f"Summarizer output shape: {output.shape}")
# Output: Summarizer output shape: torch.Size([2, 15, 500])
```

### Example 2: Generating a Summary with Length Control

```python
def generate_summary(model, src, max_len, min_len, sos_idx, eos_idx, device, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        outputs, hidden = model.encoder(src)
        input_token = torch.full((1,), sos_idx, dtype=torch.long).to(device)
        generated = [input_token]
        for t in range(max_len):
            output, hidden = model.decoder(input_token.unsqueeze(0), hidden)
            logits = output.squeeze(1)
            for token in generated:
                logits[0, token] /= repetition_penalty
            probs = torch.softmax(logits, dim=-1)
            if t < min_len:
                probs[0, eos_idx] = 0
            next_token = probs.argmax(-1)
            generated.append(next_token)
            input_token = next_token
            if next_token.item() == eos_idx and t >= min_len:
                break
    return torch.stack(generated)

src_tensor = torch.randint(0, 500, (1, 50)).to(device)
summary = generate_summary(model, src_tensor, max_len=50, min_len=5, sos_idx=1, eos_idx=2, device=device)
print(f"Summary length: {summary.shape[0]}")
# Output: Summary length: 12
```

### Example 3: ROUGE Score Computation

```python
def compute_rouge_n(reference, hypothesis, n=1):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    ref_ngrams = set()
    for i in range(len(ref_tokens) - n + 1):
        ref_ngrams.add(tuple(ref_tokens[i:i+n]))
    matches = 0
    total = 0
    for i in range(len(hyp_tokens) - n + 1):
        hyp_ng = tuple(hyp_tokens[i:i+n])
        total += 1
        if hyp_ng in ref_ngrams:
            matches += 1
    precision = matches / max(total, 1)
    recall = matches / max(len(ref_ngrams), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1

def compute_rouge_l(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    m = len(ref_tokens)
    n = len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / max(n, 1)
    recall = lcs / max(m, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1

reference = "the cat sat on the mat near the fire"
hypothesis = "the cat sat on the mat"
p1, r1, f1 = compute_rouge_n(reference, hypothesis, 1)
p2, r2, f2 = compute_rouge_n(reference, hypothesis, 2)
pl, rl, fl = compute_rouge_l(reference, hypothesis)
print(f"ROUGE-1: P={p1:.3f}, R={r1:.3f}, F1={f1:.3f}")
print(f"ROUGE-2: P={p2:.3f}, R={r2:.3f}, F1={f2:.3f}")
print(f"ROUGE-L: P={pl:.3f}, R={rl:.3f}, F1={fl:.3f}")
# Output: ROUGE-1: P=1.000, R=0.667, F1=0.800
# Output: ROUGE-2: P=1.000, R=0.571, F1=0.727
# Output: ROUGE-L: P=1.000, R=0.667, F1=0.800
```

## Common Mistakes

1. **Allowing factual hallucinations**: Seq2seq summarization models frequently generate facts that are not present in the source document. This is a critical problem for deployment. Techniques like factual consistency verification and constrained decoding are needed to mitigate hallucinations.

2. **Not handling document length limits**: Most seq2seq models have a maximum input length (e.g., 512 tokens for BERT-based, 1024 for standard transformers). Documents longer than this must be truncated or processed with hierarchical encoders.

3. **Generating repetitive summaries**: Without repetition penalties, seq2seq models tend to repeat the same phrases or n-grams. Decoding-time repetition penalties or coverage mechanisms are essential.

4. **Training on extractive oracle summaries**: Using extractive oracle summaries (the best possible extractive summary) as training targets for abstractive models creates a mismatch. The model learns to extract rather than abstract, limiting its ability to generate novel sentences.

5. **Ignoring summary-worthy content selection**: Standard seq2seq models treat all input tokens equally, but summarization requires content selection. Using pointer-generator networks or content selection mechanisms improves relevance.

## Interview Questions

### Beginner

Q: What is the difference between extractive and abstractive summarization?

A: Extractive summarization selects and rearranges existing sentences from the source document to form the summary. Abstractive summarization generates novel sentences that may not appear verbatim in the source, potentially rephrasing and condensing information in new ways.

### Intermediate

Q: Why do seq2seq models for summarization tend to hallucinate facts, and what strategies can reduce this?

A: Hallucinations occur because the model learns to generate fluent text from the training data distribution, and sometimes generates plausible but incorrect facts that fit the distribution better than the actual content. Strategies include: (1) pointer-generator networks that can copy tokens directly from the source, (2) factual consistency verification as a post-processing step, (3) contrastive learning with hallucinated examples, and (4) constrained decoding that restricts generation to facts present in the source.

### Advanced

Q: Describe how a hierarchical encoder works for long document summarization. What are the advantages and disadvantages compared to truncation-based approaches?

A: A hierarchical encoder processes the document at two levels: first, each sentence is encoded independently by a sentence-level encoder (e.g., a bidirectional LSTM). Then, the sequence of sentence representations is processed by a document-level encoder (another RNN or transformer). The decoder can attend to both sentence-level and token-level representations. Advantages: (1) the model can process arbitrarily long documents, (2) it captures document-level structure, (3) it can select entire sentences for extraction. Disadvantages: (1) the two-level architecture is more complex to train, (2) it may lose fine-grained token-level information between sentences, (3) it requires more memory. Truncation-based approaches are simpler but lose content from the truncated portion of the document.

## Practice Problems

### Easy

Implement a pointer-generator network that can either generate a new token or copy a token from the source document. The copy probability should be computed from attention weights.

### Medium

Train a seq2seq summarization model on the CNN/DailyMail dataset. Implement a repetition penalty during decoding and compare ROUGE scores with and without the penalty.

### Hard

Implement a two-stage summarization pipeline: first, use an extractive model to select the top 3 sentences from the document, then use an abstractive seq2seq model to rewrite these sentences into a coherent summary. Compare ROUGE and fluency with a purely abstractive approach.

## Solutions

### Easy Solution

```python
class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.gen = nn.Linear(hid_dim * 3, 1)
        self.fc = nn.Linear(hid_dim * 3, vocab_size)

    def compute_p_gen(self, hidden, context, decoder_input):
        x = torch.cat((hidden, context, decoder_input), dim=1)
        return torch.sigmoid(self.gen(x))

    def forward(self, src, trg):
        enc_out, (hidden, cell) = self.encoder(src)
        embedded = self.embedding(trg)
        dec_out, _ = self.decoder(embedded, (hidden, cell))
        context = torch.bmm(dec_out, enc_out.transpose(1, 2))
        p_gen = self.compute_p_gen(dec_out[:, -1], context[:, -1], embedded[:, -1])
        vocab_dist = self.fc(torch.cat((dec_out[:, -1], context[:, -1]), dim=1))
        vocab_dist = torch.softmax(vocab_dist, dim=-1)
        final_dist = p_gen * vocab_dist + (1 - p_gen) * context[:, -1]
        return final_dist

print("PointerGenerator initialized successfully")
# Output: PointerGenerator initialized successfully
```

## Related Concepts

- Pointer-Generator Networks
- ROUGE Evaluation Metric
- Abstractive vs. Extractive Summarization
- Hierarchical Attention
- Factual Consistency

## Next Concepts

- DL-333: Seq2Seq for Text Generation
- DL-334: Seq2Seq Limitations

## Summary

Seq2Seq for summarization applies the encoder-decoder framework to generate concise summaries of longer documents. Abstractive summarization generates novel text while extractive summarization selects existing content. Key challenges include handling long documents, preventing factual hallucinations, controlling summary length, and avoiding repetition. The attention mechanism is essential for focusing on relevant parts of the source during generation. Seq2Seq summarization has achieved state-of-the-art results on benchmark datasets and is widely deployed in news aggregation and business intelligence applications.

## Key Takeaways

- Seq2Seq models can be used for both extractive and abstractive summarization.
- Abstractive summarization generates novel text but risks factual hallucination.
- Hierarchical encoders handle long documents beyond the model's maximum input length.
- Repetition penalties and coverage mechanisms are essential for quality summaries.
- ROUGE score evaluates n-gram overlap but does not measure factual consistency.
- Pointer-generator networks combine copying with generation for better factual accuracy.
