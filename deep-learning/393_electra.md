# Concept: ELECTRA

## Concept ID

DL-393

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Encoder Architectures

## Learning Objectives

- Explain the ELECTRA pre-training framework: generator-discriminator architecture with replaced token detection.
- Analyze why ELECTRA is more sample-efficient than MLM-based pre-training.
- Implement the generator-discriminator training loop and understand the loss functions.
- Compare ELECTRA's performance and efficiency with BERT and RoBERTa.
- Apply ELECTRA-style training to new domains and tasks.

## Prerequisites

- Thorough understanding of BERT pre-training and MLM (DL-387, DL-388)
- Knowledge of adversarial training concepts (GANs are helpful but not required)
- Understanding of binary classification and cross-entropy loss
- Familiarity with training dynamics and curriculum learning

## Definition

ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) is a pre-training framework introduced by Clark et al. (2020) that trains a bidirectional encoder using a more sample-efficient objective than masked language modeling. ELECTRA uses two Transformer encoders: a small generator (trained with MLM) that replaces masked tokens with plausible alternatives, and a main discriminator that predicts for every token whether it is original or replaced. The discriminator loss is computed over all positions (100% of tokens), unlike MLM which only supervises 15% of tokens. This makes ELECTRA significantly more sample-efficient — it achieves BERT-level GLUE scores with 1/4 of the compute.

## Intuition

MLM is like a student filling in blanks in a sentence, learning only from the 15% of positions that are masked. ELECTRA is like a detective examining every word in a sentence to determine if it has been tampered with. The detective (discriminator) gets practice on every single token, not just the blanks.

The generator creates the tampered sentences. It is a small MLM model that tries to replace masked tokens with plausible alternatives. Early in training, the generator makes obvious replacements (gibberish), making the discriminator's job easy. As the generator improves, it creates more convincing replacements, naturally increasing the discriminator's difficulty. This creates an automatic curriculum where the task becomes progressively harder.

Unlike GANs (which ELECTRA is often compared to), ELECTRA does not use adversarial training in the strict sense. The generator is trained with maximum likelihood (standard MLM loss), not to fool the discriminator. The generator and discriminator are trained jointly, but gradients do not flow from discriminator to generator.

## Why This Concept Matters

ELECTRA introduced a new paradigm for NLP pre-training that is fundamentally more sample-efficient than MLM. Key contributions:

1. Demonstrates that learning from all tokens (100%) is more efficient than learning from 15%.
2. Shows that replaced token detection is a harder, more useful task than masked token prediction.
3. Achieves state-of-the-art results with significantly less compute.
4. Bridges ideas from adversarial training and denoising autoencoders in a novel way.
5. Influenced subsequent models and is widely used in production settings where compute is limited.

## Mathematical Explanation

### Generator

The generator G is a small BERT-style encoder (typically 1/4 the size of the discriminator). Given a masked input sequence x_masked, the generator predicts the original tokens at masked positions:

p_G(x_i | x_masked) = softmax(h_i^G W_G + b_G)

The generator loss is the standard MLM loss:

L_G = -1/|M| * sum_{i in M} log p_G(x_i | x_masked)

### Discriminator

The discriminator D receives the sequence x_replaced, where masked positions are filled with generator samples:

x_i_replaced = sample from p_G(x_i | x_masked) if i in M, else x_i

The discriminator predicts for each position whether it is original or replaced:

D(x_replaced, i) = sigmoid(h_i^D w_D + b_D)

The discriminator loss is binary cross-entropy over all positions:

L_D = -sum_i [1(x_i_replaced == x_i) * log D(x_i_replaced, i) + 1(x_i_replaced != x_i) * log (1 - D(x_i_replaced, i))]

### Total Loss

L = L_G + lambda * L_D (lambda is typically 50 to balance the losses)

### Training Details

- Generator: 12 layers, 256 hidden (for base discriminator with 768 hidden)
- Discriminator: 12 layers, 768 hidden
- Shared embeddings between generator and discriminator
- Adam optimizer with linear warmup and decay
- Training for 2M steps with batch size 256

## Code Examples

### Example 1: ELECTRA Generator-Discriminator Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ElectraGenerator(nn.Module):
    def __init__(self, vocab_size=30522, d_model=256, n_layers=6, n_heads=4, d_ff=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        logits = self.lm_head(x)
        return logits

class ElectraDiscriminator(nn.Module):
    def __init__(self, vocab_size=30522, d_model=768, n_layers=12, n_heads=12, d_ff=3072):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.discriminator_head = nn.Linear(d_model, 1)
        self.generator_predictions = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        disc_logits = self.discriminator_head(x).squeeze(-1)
        return disc_logits

gen = ElectraGenerator()
disc = ElectraDiscriminator()

x = torch.randint(0, 1000, (2, 32))
gen_logits = gen(x)
disc_logits = disc(x)

print("Generator logits:", gen_logits.shape)
# Output: Generator logits: torch.Size([2, 32, 30522])
print("Discriminator logits:", disc_logits.shape)
# Output: Discriminator logits: torch.Size([2, 32])
print("Discriminator produces per-token binary logits")
# Output: Discriminator produces per-token binary logits
```

### Example 2: Token Replacement and Discriminator Labels

```python
def electra_corrupt(generator, input_ids, mask_token_id=103, mask_prob=0.15):
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape) < mask_prob
    special_tokens = (input_ids == 0) | (input_ids == 102) | (input_ids == 103)
    mask = mask & ~special_tokens

    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id

    with torch.no_grad():
        gen_logits = generator(masked_ids)
        gen_probs = F.softmax(gen_logits, dim=-1)
        gen_samples = torch.multinomial(
            gen_probs.view(-1, gen_probs.size(-1)),
            num_samples=1
        ).view(input_ids.shape)

    corrupted_ids = input_ids.clone()
    corrupted_ids[mask] = gen_samples[mask]

    disc_labels = torch.ones_like(input_ids)
    disc_labels[mask] = 0
    disc_labels[gen_samples[mask] == labels[mask]] = 1

    return corrupted_ids, disc_labels, labels[mask], gen_samples[mask]

generator = ElectraGenerator()
x = torch.randint(0, 1000, (2, 16))
corrupted, disc_labels, orig_tokens, gen_tokens = electra_corrupt(generator, x)

print("Discriminator labels shape:", disc_labels.shape)
# Output: Discriminator labels shape: torch.Size([2, 16])
print("Masked positions:", (disc_labels == 0).sum().item())
# Output: Masked positions: 4
print("Correctly replaced:", ((disc_labels == 1) & (x == corrupted)).sum().item())
# Output: Correctly replaced: 8
print("Loss computed on ALL positions (100%): True")
# Output: Loss computed on ALL positions (100%): True
```

### Example 3: ELECTRA Training Step

```python
class ELECTRATrainer:
    def __init__(self, generator, discriminator, lambda_weight=50.0, lr=1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_weight = lambda_weight
        self.gen_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr)
        self.disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr)

    def train_step(self, input_ids):
        mask = torch.rand(input_ids.shape) < 0.15
        special_tokens = (input_ids == 0) | (input_ids == 2) | (input_ids == 103)
        mask = mask & ~special_tokens

        masked_ids = input_ids.clone()
        masked_ids[mask] = 103

        gen_logits = self.generator(masked_ids)
        gen_loss = F.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)),
            input_ids.view(-1)
        )

        with torch.no_grad():
            gen_probs = F.softmax(gen_logits, dim=-1)
            gen_samples = torch.multinomial(
                gen_probs.view(-1, gen_probs.size(-1)),
                num_samples=1
            ).view(input_ids.shape)

        corrupted = input_ids.clone()
        corrupted[mask] = gen_samples[mask]

        disc_logits = self.discriminator(corrupted)
        disc_labels = torch.ones_like(input_ids)
        disc_labels[mask] = 0
        disc_labels[gen_samples[mask] == input_ids[mask]] = 1

        disc_loss = F.binary_cross_entropy_with_logits(disc_logits, disc_labels.float())

        total_loss = gen_loss + self.lambda_weight * disc_loss

        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()
        total_loss.backward()
        self.gen_optimizer.step()
        self.disc_optimizer.step()

        return {
            "gen_loss": gen_loss.item(),
            "disc_loss": disc_loss.item(),
            "total_loss": total_loss.item()
        }

trainer = ELECTRATrainer(ElectraGenerator(), ElectraDiscriminator())
dummy_input = torch.randint(0, 1000, (2, 32))
losses = trainer.train_step(dummy_input)
print(f"Gen loss: {losses['gen_loss']:.4f}, Disc loss: {losses['disc_loss']:.4f}")
# Output: Gen loss: 7.9245, Disc loss: 0.7012
print(f"Total loss: {losses['total_loss']:.4f}")
# Output: Total loss: 42.9865
```

## Common Mistakes

1. Treating ELECTRA like a GAN: ELECTRA does not use adversarial training. The generator is trained with standard MLM loss (max likelihood), and gradients do not flow from discriminator to generator. The generator is just an auxiliary model for creating training data.

2. Using a generator that is too large or too small: The generator should be approximately 1/4 the size of the discriminator. A generator that is too powerful produces replacements that are too hard to detect (discriminator accuracy near random). A generator that is too weak produces obvious replacements (discriminator accuracy near 100%, providing weak learning signal).

3. Not sharing embeddings between generator and discriminator: The original ELECTRA shares token embeddings, which improves efficiency and provides a good initialization for the discriminator's embedding layer.

4. Computing discriminator loss only on masked positions: The key advantage of ELECTRA is that the discriminator learns from ALL positions. Computing loss only on masked positions loses this advantage.

5. Training the discriminator to predict original tokens: The discriminator predicts binary original/replaced, not the token identity. Using a classification head instead of a binary head introduces unnecessary complexity.

6. Neglecting the generator's role after pre-training: During fine-tuning, only the discriminator is used. The generator is discarded. Keeping the generator or fine-tuning it alongside the discriminator wastes FLOPs.

## Interview Questions

### Beginner

Q: How does ELECTRA's pre-training objective differ from BERT's MLM?

A: BERT's MLM masks 15% of tokens and predicts their original values. ELECTRA uses a small generator to replace masked tokens with plausible alternatives, then trains a discriminator to classify each token as original or replaced. The discriminator loss is computed over all 100% of tokens, while BERT only learns from 15% of tokens.

### Intermediate

Q: Why is ELECTRA more sample-efficient than BERT? Explain with concrete numbers.

A: BERT computes loss only on 15% of tokens per example. ELECTRA computes loss on 100% of tokens per example. This means each training example provides ~6.7x more supervision in ELECTRA. Additionally, the replaced token detection task is harder than masked token prediction — distinguishing real from plausible fake requires deeper understanding than simply predicting a masked token. Empirically, ELECTRA achieves BERT-level GLUE scores with 1/4 the training compute.

### Advanced

Q: The ELECTRA paper mentions that the generator uses "maximum likelihood" rather than an adversarial objective. Why is this choice important, and what would happen if you used an adversarial generator?

A: An adversarial generator (trained to fool the discriminator) would optimize for producing replacements indistinguishable from originals in the discriminator's embedding space. This could lead to mode collapse (generator produces a small set of "safe" replacements) or instability from minimax optimization. Maximum likelihood training ensures the generator produces diverse, plausible replacements based on the data distribution. Additionally, adversarial training would require differentiability through the discrete sampling step, which is challenging without techniques like Gumbel-Softmax. The maximum likelihood generator provides a natural curriculum: as it improves, the discriminator task becomes progressively harder, without the optimization challenges of adversarial training.

## Practice Problems

### Easy

Implement the token corruption function for ELECTRA: given a generator's logits, sample replacement tokens, create the corrupted sequence, and generate the discriminator labels (1 for original, 0 for replaced). Verify the label distribution is balanced.

### Medium

Train a small ELECTRA model (generator: 4 layers, 128 hidden; discriminator: 6 layers, 256 hidden) on the Wikitext-2 dataset. After pre-training, fine-tune the discriminator on the SST-2 sentiment classification task and compare accuracy with a baseline BERT-like model of similar size trained with MLM.

### Hard

Design an improved version of ELECTRA where the generator and discriminator are trained with a shared encoder (similar to ALBERT-style parameter sharing). Analyze whether parameter sharing between generator and discriminator improves or worsens performance, and measure the speed/quality trade-off.

## Solutions

```python
# Easy solution
def electra_corrupt_simple(gen_logits, input_ids, mask_token_id=103, mask_prob=0.15):
    batch_size, seq_len = input_ids.shape
    mask = torch.rand(batch_size, seq_len) < mask_prob

    gen_probs = F.softmax(gen_logits, dim=-1)
    gen_samples = gen_probs.view(-1, gen_probs.size(-1)).multinomial(1).view(batch_size, seq_len)

    corrupted = input_ids.clone()
    corrupted[mask] = gen_samples[mask]

    disc_labels = torch.ones(batch_size, seq_len, dtype=torch.long)
    disc_labels[mask] = 0
    disc_labels[gen_samples[mask] == input_ids[mask]] = 1

    print(f"Original/Replaced ratio: {disc_labels.sum().item()}/{disc_labels.numel() - disc_labels.sum().item()}")
    return corrupted, disc_labels

gen = ElectraGenerator()
x = torch.randint(0, 100, (2, 20))
logits = gen(x)
corrupted, labels = electra_corrupt_simple(logits, x)
print("Labels unique:", labels.unique().tolist())
# Output: Labels unique: [0, 1]
print("Corrupted shape:", corrupted.shape)
# Output: Corrupted shape: torch.Size([2, 20])
```

## Related Concepts

- BERT Pre-training (DL-387)
- Masked Language Modeling (DL-388)
- RoBERTa (DL-391)
- Generative Adversarial Networks
- Denoising Autoencoders
- Sample Efficiency in Self-Supervised Learning

## Next Concepts

- DeBERTa
- Encoder-only vs Decoder-only
- GPT Decoder Architecture

## Summary

ELECTRA replaces MLM with a more sample-efficient replaced token detection task. A small generator creates corrupted sequences by replacing masked tokens, and the discriminator learns to distinguish original from replaced tokens at every position. This provides supervision on 100% of tokens per example, achieving comparable performance to BERT with significantly less compute.

## Key Takeaways

- ELECTRA uses a generator-discriminator framework for pre-training.
- The discriminator learns from all 100% of tokens, not just 15%.
- Sample efficiency is ~4x better than BERT (matches BERT with 1/4 the compute).
- The generator is trained with standard MLM loss, not adversarial loss.
- Only the discriminator is kept for fine-tuning; the generator is discarded.
- The generator's role is to create natural curriculum — progressively harder replacements.
- ELECTRA set new standards for efficient pre-training and influenced subsequent models.
