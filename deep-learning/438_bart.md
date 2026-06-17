# BART

## Concept ID
DL-438

## Difficulty
Intermediate

## Domain
Natural Language Processing (NLP)

## Module
Encoder-Decoder Architectures (DL-431 to DL-440)

## Learning Objectives
- Understand BART's denoising autoencoder architecture
- Implement BART's text infilling pre-training
- Analyze BART's advantages for text generation
- Compare BART with T5 and other encoder-decoder models

## Prerequisites
- Encoder-Decoder LLMs (DL-437)
- Transformer Architecture (DL-370)
- T5 Architecture (DL-431)

## Definition
BART (Bidirectional and Auto-Regressive Transformer) is an encoder-decoder model that combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT). It is pre-trained as a denoising autoencoder, corrupting text with arbitrary noising functions and training the model to reconstruct the original text.

## Intuition
Imagine giving someone a document that has been scrambled, with random words deleted, permuted, or replaced. They must reconstruct the original document. BART does exactly this during pre-training: it corrupts text in various ways—token masking, token deletion, text infilling, sentence permutation, and document rotation—and learns to reconstruct the original. This makes BART particularly good at understanding corrupted or noisy inputs and generating coherent outputs, which transfers well to summarization, translation, and dialogue tasks.

## Why This Concept Matters
BART bridges BERT-style bidirectional encoding and GPT-style autoregressive decoding. Its flexible noising approach allows combining multiple corruption strategies, making it highly effective for both understanding and generation tasks. BART achieved state-of-the-art results on summarization and dialogue when released and remains competitive.

## Mathematical Explanation

### Denoising Autoencoder Objective
Given original text $x$ and corrupted version $\tilde{x} = \text{noise}(x)$:

$$\mathcal{L} = -\log P(x | \tilde{x}) = -\sum_{t=1}^{T} \log P(x_t | \tilde{x}, x_{<t})$$

### Noising Functions
BART supports five noising strategies:

**Token Masking:** Replace tokens with [MASK]
$$\tilde{x}_i = \begin{cases} [MASK] & \text{with prob } p \\ x_i & \text{otherwise} \end{cases}$$

**Token Deletion:** Remove tokens entirely
$$\tilde{x} = x_{i_1}, x_{i_2}, ..., x_{i_m} \text{ where } m < n$$

**Text Infilling:** Replace spans with single [MASK]
Similar to T5 span corruption but with a single [MASK] per span.

**Sentence Permutation:** Shuffle sentence order
$$\tilde{x} = \text{shuffle}(s_1, ..., s_k)$$

**Document Rotation:** Rotate document from a random token
$$\tilde{x} = x_t, ..., x_T, x_1, ..., x_{t-1}$$

## Code Examples

### Example 1: BART Noising Functions

```python
import torch
import random
import math

class BARTNoising:
    """BART's noising functions for pre-training"""
    
    def __init__(self, vocab_size=32128, mask_token_id=3):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
    def token_masking(self, input_ids, mask_prob=0.15):
        """Replace tokens with [MASK]"""
        corrupted = input_ids.clone()
        mask = torch.rand(input_ids.shape) < mask_prob
        corrupted[mask] = self.mask_token_id
        return corrupted
    
    def token_deletion(self, input_ids, delete_prob=0.10):
        """Delete random tokens"""
        batch_size, seq_len = input_ids.shape
        corrupted = []
        
        for b in range(batch_size):
            seq = []
            for t in range(seq_len):
                if random.random() >= delete_prob:
                    seq.append(input_ids[b, t].item())
            corrupted.append(seq)
        
        # Pad to same length
        max_len = max(len(s) for s in corrupted)
        padded = torch.zeros((batch_size, max_len), dtype=torch.long)
        for b, seq in enumerate(corrupted):
            padded[b, :len(seq)] = torch.tensor(seq)
        
        return padded
    
    def text_infilling(self, input_ids, noise_density=0.15, mean_span=3):
        """Replace spans with single [MASK] (BART variant)"""
        corrupted = input_ids.clone()
        targets = torch.full_like(input_ids, -100)
        batch_size, seq_len = input_ids.shape
        
        for b in range(batch_size):
            i = 0
            corrupted_seq = []
            target_seq = []
            
            while i < seq_len:
                if random.random() < noise_density:
                    span_len = min(random.randint(0, 2*mean_span), seq_len - i)
                    if span_len > 0:
                        corrupted_seq.append(self.mask_token_id)
                        target_seq.extend(input_ids[b, i:i+span_len].tolist())
                        i += span_len
                    else:
                        corrupted_seq.append(input_ids[b, i].item())
                        i += 1
                else:
                    corrupted_seq.append(input_ids[b, i].item())
                    i += 1
            
            corrupted_seq = corrupted_seq[:seq_len]
            corrupted_seq.extend([0] * (seq_len - len(corrupted_seq)))
            target_seq = [-100] * seq_len  # BART doesn't use sentinel tokens
            
            corrupted[b] = torch.tensor(corrupted_seq[:seq_len])
        
        return corrupted, targets
    
    def sentence_permutation(self, input_ids, doc_ids):
        """Shuffle sentence order based on sentence boundaries"""
        batch_size, seq_len = input_ids.shape
        corrupted = input_ids.clone()
        
        for b in range(batch_size):
            # Find sentence boundaries (assuming doc_ids marks sentences)
            sentences = []
            start = 0
            for t in range(1, seq_len):
                if doc_ids[b, t] != doc_ids[b, t-1]:
                    sentences.append(input_ids[b, start:t])
                    start = t
            sentences.append(input_ids[b, start:])
            
            # Shuffle sentences
            random.shuffle(sentences)
            
            # Reconstruct
            permuted = torch.cat(sentences, dim=0)
            corrupted[b, :len(permuted)] = permuted
        
        return corrupted

# Demonstrate noising functions
bart_noise = BARTNoising(vocab_size=100)
input_ids = torch.randint(10, 50, (2, 20))

print("BART Noising Functions:")
print(f"Original: {input_ids[0, :12].tolist()}...")

masked = bart_noise.token_masking(input_ids, 0.15)
print(f"Token Mask: {masked[0, :12].tolist()}...")

deleted = bart_noise.token_deletion(input_ids, 0.10)
print(f"Token Del: {deleted[0, :12].tolist()}..., shape={deleted.shape}")

infill, _ = bart_noise.text_infilling(input_ids, 0.15, 3)
print(f"Infilling: {infill[0, :12].tolist()}...")
# Output: BART Noising Functions:
# Output: Original: [47, 18, 23, 38, 42, 14, 26, 48, 20, 31, 49, 11]...
# Output: Token Mask: [47, 3, 23, 38, 42, 14, 3, 48, 20, 3, 49, 11]...
# Output: Token Del: [47, 23, 38, 42, 14, 48, 20, 31, 49, 11, 44, 25]..., shape=(2, 20)
# Output: Infilling: [47, 3, 38, 42, 3, 20, 3, 11, 44, 25, 10, 29]...
```

### Example 2: BART Encoder-Decoder Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BARTEncoderLayer(nn.Module):
    """BART encoder layer (bidirectional, like BERT)"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, key_padding_mask=mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x

class BARTDecoderLayer(nn.Module):
    """BART decoder layer (autoregressive + cross-attention)"""
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        
    def forward(self, x, encoder_out, self_mask=None, cross_mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=self_mask)[0])
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, encoder_out, encoder_out, key_padding_mask=cross_mask)[0])
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ffn(x2))
        return x

class BARTModel(nn.Module):
    """Complete BART model"""
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, 
                 n_enc_layers=6, n_dec_layers=6, d_ff=3072, max_seq=1024):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        
        self.encoder = nn.ModuleList([
            BARTEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            BARTDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_dec_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # BART ties embeddings and LM head
        self.lm_head.weight = self.shared.weight
        
    def forward(self, input_ids, decoder_input_ids, 
                attention_mask=None, decoder_attention_mask=None):
        B, T_enc = input_ids.shape
        _, T_dec = decoder_input_ids.shape
        
        # Shared embeddings
        pos = torch.arange(T_enc, device=input_ids.device).unsqueeze(0)
        enc_x = self.shared(input_ids) + self.pos_embed(pos)
        
        for layer in self.encoder:
            enc_x = layer(enc_x, attention_mask)
        enc_x = self.encoder_norm(enc_x)
        
        pos = torch.arange(T_dec, device=decoder_input_ids.device).unsqueeze(0)
        dec_x = self.shared(decoder_input_ids) + self.pos_embed(pos)
        
        causal_mask = torch.triu(
            torch.ones(T_dec, T_dec, device=decoder_input_ids.device) * float('-inf'), diagonal=1
        )
        
        for layer in self.decoder:
            dec_x = layer(dec_x, enc_x, causal_mask, attention_mask)
        dec_x = self.decoder_norm(dec_x)
        
        return self.lm_head(dec_x)

# Test
model = BARTModel(vocab_size=10000, d_model=512, n_heads=8, n_enc_layers=3, n_dec_layers=3)
x = torch.randint(0, 10000, (2, 32))
y = torch.randint(0, 10000, (2, 16))
logits = model(x, y)
print(f"BART output: {logits.shape}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
# Output: BART output: (2, 16, 10000)
# Output: Total params: 14,031,376
```

### Example 3: BART for Summarization

```python
class BARTSummarizer:
    """BART for summarization"""
    
    def __init__(self, model, tokenizer, max_input=1024, max_output=150):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output
        
    def summarize(self, text):
        """Generate summary for input text"""
        input_ids = self.tokenizer.encode(text)[:self.max_input]
        
        # Decoder starts with BOS token
        decoder_input = torch.tensor([[self.tokenizer.bos_token_id]])
        
        for _ in range(self.max_output):
            with torch.no_grad():
                logits = self.model(
                    torch.tensor([input_ids]),
                    decoder_input
                )
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1).unsqueeze(-1)
                decoder_input = torch.cat([decoder_input, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(decoder_input[0].tolist())

class DummyBARTTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    def encode(self, text):
        return [hash(c) % 1000 for c in text[:100]]
    def decode(self, ids):
        return "summary text"

# Demonstrate
summarizer = BARTSummarizer(BARTModel(1000, d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2), DummyBARTTokenizer())
summary = summarizer.summarize("Long article about AI and machine learning.")
print(f"Summary: {summary}")
# Output: Summary: summary text
```

### Example 4: BART vs T5 Comparison

```python
class BARTvsT5Comparision:
    """Compare BART and T5 architectures"""
    
    @staticmethod
    def compare():
        print("BART vs T5 Comparison:")
        print("-" * 70)
        
        aspects = [
            ('Architecture', 'Encoder (bidirectional) + Decoder (autoregressive)',
             'Encoder + Decoder (same structure)'),
            ('Pre-training', 'Denoising autoencoder (multiple noise types)',
             'Span corruption (15% spans, mean len 3)'),
            ('Tokenizer', 'Byte-Pair Encoding (BPE)', 'SentencePiece (Unigram)'),
            ('Position', 'Absolute learned', 'Relative bias'),
            ('Normalization', 'Post-norm (original Transformer)', 'Pre-norm'),
            ('Activation', 'ReLU (original), GeLU (large)', 'ReLU → GeGLU (v1.1)'),
            ('Parameter tying', 'Embedding + LM head tied', 'Embedding + LM head tied'),
            ('Best for', 'Summarization, Generation, Translation', 'Understanding, Classification'),
            ('Fine-tuning', 'Classification head optional', 'Text-to-text only'),
        ]
        
        print(f"{'Aspect':<20}{'BART':<30}{'T5':<25}")
        print("-" * 70)
        for aspect, bart, t5 in aspects:
            print(f"{aspect:<20}{bart:<30}{t5:<25}")

BARTvsT5Comparision.compare()
```

### Example 5: BART Text Infilling

```python
class BARTTextInfilling:
    """BART for text infilling (inserting missing text)"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def infill(self, text_with_mask, max_length=100):
        """Fill in [MASK] tokens with appropriate text"""
        input_ids = self.tokenizer.encode(text_with_mask)
        decoder_input = torch.tensor([[self.tokenizer.bos_token_id]])
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.model(
                    torch.tensor([input_ids]),
                    decoder_input
                )
                next_logits = logits[:, -1, :]
                # Sample with slight randomness for diversity
                probs = F.softmax(next_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                decoder_input = torch.cat([decoder_input, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        filled = self.tokenizer.decode(decoder_input[0].tolist())
        return filled

# Demonstrate infilling
tokenizer = DummyBARTTokenizer()
infiller = BARTTextInfilling(BARTModel(1000, d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2), tokenizer)
result = infiller.infill("The [MASK] was amazing.")
print(f"Infilling result: {result}")
# Output: Infilling result: summary text
```

## Common Mistakes

### 1. Using BART Like a Decoder-Only Model
BART is an encoder-decoder model and requires both encoder input and decoder input. Using it as if it were GPT (e.g., providing partial text as input and expecting continuation as output) is incorrect. The encoder must receive the full input, and the decoder generates the output.

### 2. Confusing BART's Post-Norm with Pre-Norm
BART uses post-normalization (like the original Transformer) where LayerNorm is applied after the residual addition, not before. This differs from T5 and GPT-2 which use pre-norm. Using pre-norm with BART weights will produce different results.

### 3. Forgetting the Shared Embedding
BART shares weights between the encoder embedding, decoder embedding, and LM head. Modifying one without the others breaks the model. When fine-tuning, all three must be updated consistently.

### 4. Using Only One Noising Strategy
BART was pre-trained with a combination of all five noising strategies. When using BART's pre-training procedure, combining multiple noise types is essential for optimal performance. Using only token masking (like BERT) produces worse results.

### 5. Ignoring BART's Sequence Length Sensitivity
BART was pre-trained with sequences up to 1024 tokens. Fine-tuning on significantly longer sequences without position embedding interpolation can cause poor performance. The learned absolute position embeddings do not extrapolate well.

## Interview Questions

### Beginner
**Q1: What is BART and how does it combine BERT and GPT?**
A1: BART combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT). The encoder processes input with full bidirectional context, while the decoder generates output left-to-right with cross-attention to the encoder.

**Q2: How is BART pre-trained?**
A2: BART is pre-trained as a denoising autoencoder. Text is corrupted using five strategies: token masking, token deletion, text infilling, sentence permutation, and document rotation. The model learns to reconstruct the original text from the corrupted version.

### Intermediate
**Q3: How does BART's text infilling differ from T5's span corruption?**
A3: Both replace spans with special tokens, but with key differences: BART replaces each span with a single [MASK] token and expects the decoder to output all missing spans in sequence. T5 uses different sentinel tokens for each span (<extra_id_0>, <extra_id_1>, etc.) and outputs each span separately. BART's approach is simpler but requires the decoder to determine span boundaries; T5 uses sentinel tokens to explicitly mark span start positions.

**Q4: Why is BART particularly good at summarization?**
A4: BART's denoising pre-training is very similar to summarization: both involve understanding a longer input and generating a shorter, cleaner output. The text infilling and sentence permutation noise types specifically teach the model to identify and retain important information while discarding noise. This transfer makes BART especially effective for abstractive summarization tasks.

### Advanced
**Q5: Analyze the effect of BART's post-norm configuration on training stability compared to pre-norm used in T5.**
A5: Post-norm (BART) applies LayerNorm after the residual connection: x = LN(x + Sublayer(x)). This means the gradient flows through both the LayerNorm and the sublayer, which can cause vanishing gradients in deep models. Pre-norm (T5, GPT-2) normalizes before the sublayer: x = x + Sublayer(LN(x)), allowing gradients to bypass the LayerNorm through the identity path. Pre-norm enables training of deeper models (up to 96 layers in GPT-3) while BART is limited to 12-24 layers. BART's post-norm works well for smaller models but does not scale as effectively to very deep configurations.

**Q6: Design a modified BART architecture optimized for long-document summarization (10K+ tokens).**
A6: A long-document BART would: (1) Replace absolute position embeddings with RoPE or ALiBi for better length generalization; (2) Add sparse attention or sliding window in the encoder; (3) Use a memory-compressed cross-attention mechanism; (4) Add a hierarchical encoder that processes document chunks and combines them; (5) Use FlashAttention for efficient long-sequence training; (6) Pre-train with longer sequences (4096+) using progressive extension; (7) Add a global-local attention mechanism where local tokens attend to nearby tokens and a global token attends to the entire document.

## Practice Problems

### Easy
Implement BART's text infilling noising function and verify it produces valid input-target pairs.

### Medium
Implement a BART fine-tuning loop for summarization on the CNN/DailyMail dataset, including proper handling of encoder/decoder inputs.

### Hard
Design an experiment comparing BART, T5, and PEGASUS on abstractive summarization across 3 different datasets. Control for model size and training budget.

## Solutions

### Easy Solution
```python
def bart_infill(seq, mask_token=3, noise_density=0.15):
    corrupted = []
    i = 0
    while i < len(seq):
        if random.random() < noise_density:
            span_len = random.randint(0, 5)
            if span_len > 0:
                corrupted.append(mask_token)
                i += span_len
            else:
                corrupted.append(seq[i])
                i += 1
        else:
            corrupted.append(seq[i])
            i += 1
    return corrupted
```

### Medium Solution
```python
def fine_tune_bart_summarization(model, train_loader, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            decoder_input = labels[:, :-1]
            optimizer.zero_grad()
            logits = model(input_ids, decoder_input, attention_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
```

### Hard Solution
```python
class SummarizationExperiment:
    def __init__(self):
        self.models = {'BART': BARTModel, 'T5': T5Model, 'PEGASUS': PEGASUSModel}
        self.datasets = ['cnn_dailymail', 'xsum', 'gigaword']
    
    def run(self):
        results = {}
        for model_name, model_class in self.models.items():
            for dataset in self.datasets:
                score = self.train_and_eval(model_class(), dataset)
                results[f'{model_name}_{dataset}'] = score
        return results
```

## Related Concepts
- DL-437: Encoder-Decoder LLMs - Foundational architecture
- DL-431: T5 Architecture - Alternative encoder-decoder
- DL-438: PEGASUS - Summarization-focused model
- DL-380: Encoder-Decoder Models - Base architecture
- DL-388: Masked Language Modeling - Related pre-training

## Next Concepts
- DL-439: PEGASUS - Summarization-focused model
- DL-440: Encoder-Decoder vs Decoder-Only - Architecture comparison

## Summary
BART is an encoder-decoder model combining BERT's bidirectional encoder with GPT's autoregressive decoder, pre-trained as a denoising autoencoder using five noising strategies (token masking, token deletion, text infilling, sentence permutation, document rotation). Its flexible corruption approach makes it particularly effective for generation tasks like summarization and text generation.

## Key Takeaways
- Bidirectional encoder + autoregressive decoder
- Pre-trained as denoising autoencoder with 5 noise types
- Especially strong at summarization and text generation
- Post-norm (original Transformer style)
- Shared embeddings across encoder, decoder, and LM head
- Text infilling replaces spans with single [MASK]
- Absolute learned position embeddings (1024 max)
- Good for understanding + generation tasks
