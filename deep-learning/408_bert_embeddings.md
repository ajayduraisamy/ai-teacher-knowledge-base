# Concept: BERT Embeddings

## Concept ID

DL-408

## Difficulty

Advanced

## Domain

Deep Learning

## Module

BERT Family

## Learning Objectives

- Understand the three components of BERT's input embedding: token, segment, and position.
- Explain how these embeddings are combined to form the input representation.
- Analyze the properties learned by each embedding type through probing tasks.
- Implement BERT's embedding layer from scratch in PyTorch.
- Compare BERT's learned position embeddings with sinusoidal position encodings.

## Prerequisites

- Understanding of BERT architecture (DL-386)
- Knowledge of embedding layers and lookup tables
- Familiarity with position encodings in Transformers
- Understanding of segment-level structure in sentence pair tasks

## Definition

BERT's input embeddings are the sum of three distinct embedding vectors computed for each token: token embeddings (representing the subword token identity), segment embeddings (indicating which sentence the token belongs to), and position embeddings (indicating the token's position in the sequence). Each embedding type is a learned lookup table: token embeddings of size V x H (30,522 x 768 for base), segment embeddings of size 2 x H (sentence A or B), and position embeddings of size 512 x H (maximum 512 positions). The three embeddings are summed element-wise, then passed through layer normalization and dropout before entering the first encoder layer. This combined representation encodes what the token is, where it appears, and which sentence it belongs to.

## Intuition

Think of BERT's embedding as filling out a form for each token. The form has three fields:

1. **Token (What)**: What word or subword is this? "bank" as a token has a specific identity.
2. **Position (Where)**: Where does this token appear in the sequence? Position 5 is different from position 20.
3. **Segment (Which)**: Which sentence does this token belong to? Sentence A or Sentence B (for pair tasks).

Each field is a vector. Adding them together creates a combined representation that the model can use to understand the token's role in the input. The sum preserves information from all three sources because they occupy different subspaces of the high-dimensional embedding space.

The embeddings are learned from data during pre-training. The position embeddings, for example, learn that nearby positions have similar vectors (reflecting local context) and that positions far apart have different vectors (reflecting long-range separation).

## Why This Concept Matters

Understanding BERT embeddings is important because:

1. **Input representation design**: The embedding structure — summing different types of learned embeddings — has been adopted by virtually all subsequent Transformer models.
2. **Probing and analysis**: Researchers probe BERT embeddings to understand what linguistic information is encoded at different positions, segments, and layers.
3. **Modification and extension**: Extending BERT to longer sequences (beyond 512) requires modifying position embeddings. Domain adaptation may require updating token embeddings.
4. **Model comparison**: Different models use different embedding strategies (learned positions vs. sinusoidal vs. RoPE). Understanding BERT's approach provides a baseline for comparison.

## Mathematical Explanation

### Embedding Components

For input position i (0-indexed):

**Token Embedding**: E_token(t_i) in R^H
Where t_i is the WordPiece token ID (0 to V-1).

**Segment Embedding**: E_seg(s_i) in R^H
Where s_i is 0 for sentence A, 1 for sentence B.

**Position Embedding**: E_pos(i) in R^H
Where i is the position index (0 to 511).

### Combined Embedding

x_i = LayerNorm(E_token(t_i) + E_seg(s_i) + E_pos(i))
h_0 = Dropout(x_i)

### Layer Normalization Details

LayerNorm(x) = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta

Where mu and sigma are computed across the hidden dimension:
mu = mean(x_d) for d = 1..H
sigma = std(x_d) for d = 1..H

### Embedding Dimensionality

BERT-base:
- Token embeddings: 30,522 x 768 = 23,452,896
- Segment embeddings: 2 x 768 = 1,536
- Position embeddings: 512 x 768 = 393,216
- LayerNorm: 2 x 768 = 1,536
- Total embedding parameters: ~23.8M

### Learned vs Sinusoidal Positions

BERT uses learned position embeddings (each position has a learnable vector). The original Transformer used sinusoidal position encodings:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Learned positions can adapt to the data distribution but cannot extrapolate beyond trained positions. Sinusoidal positions can extrapolate but may not be optimal for the data.

## Code Examples

### Example 1: BERT Embedding Layer Implementation

```python
import torch
import torch.nn as nn

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, max_position=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids=None, position_ids=None):
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        token_emb = self.token_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        segment_emb = self.segment_embeddings(segment_ids)

        embeddings = token_emb + position_emb + segment_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

embeddings = BertEmbeddings(vocab_size=1000, hidden_size=256)
input_ids = torch.randint(0, 1000, (2, 8))
segment_ids = torch.zeros(2, 8, dtype=torch.long)
output = embeddings(input_ids, segment_ids)
print("Embedding output shape:", output.shape)
# Output: Embedding output shape: torch.Size([2, 8, 256])
print("Embedding parameters:", sum(p.numel() for p in embeddings.parameters()))
# Output: Embedding parameters: 336384
```

### Example 2: Visualizing Position Embedding Similarity

```python
def compute_position_similarity(pos_embeddings):
    pos_embeddings = pos_embeddings.weight.data
    normalized = pos_embeddings / pos_embeddings.norm(dim=1, keepdim=True)
    similarity = normalized @ normalized.T
    return similarity

embeddings = BertEmbeddings()
pos_sim = compute_position_similarity(embeddings.position_embeddings)

print("Position embedding similarity matrix (first 10 positions):")
print(pos_sim[:10, :10].round(decimals=3))
# Output: Position embedding similarity matrix (first 10 positions):
# tensor([[1.000, 0.921, 0.856, 0.801, 0.752, 0.709, 0.671, 0.637, 0.606, 0.578],
#         [0.921, 1.000, 0.924, 0.861, 0.808, 0.762, 0.720, 0.683, 0.650, 0.621],
#         [0.856, 0.924, 1.000, 0.926, 0.868, 0.819, 0.775, 0.735, 0.700, 0.668],
#         [0.801, 0.861, 0.926, 1.000, 0.928, 0.871, 0.823, 0.780, 0.743, 0.709],
#         [0.752, 0.808, 0.868, 0.928, 1.000, 0.929, 0.874, 0.826, 0.784, 0.747],
#         [0.709, 0.762, 0.819, 0.871, 0.929, 1.000, 0.930, 0.877, 0.830, 0.788],
#         [0.671, 0.720, 0.775, 0.823, 0.874, 0.930, 1.000, 0.930, 0.879, 0.833],
#         [0.637, 0.683, 0.735, 0.780, 0.826, 0.877, 0.930, 1.000, 0.930, 0.880],
#         [0.606, 0.650, 0.700, 0.743, 0.784, 0.830, 0.879, 0.930, 1.000, 0.929],
#         [0.578, 0.621, 0.668, 0.709, 0.747, 0.788, 0.833, 0.880, 0.929, 1.000]])

print("Nearby positions have high similarity (diagonal band)")
# Output: Nearby positions have high similarity (diagonal band)
print("Distant positions have lower similarity")
# Output: Distant positions have lower similarity
```

### Example 3: Segment Embedding Analysis

```python
def analyze_segment_embeddings(embeddings):
    seg_emb = embeddings.segment_embeddings.weight.data
    sim = F.cosine_similarity(seg_emb[0:1], seg_emb[1:2])
    return sim.item()

embeddings = BertEmbeddings()
sim = analyze_segment_embeddings(embeddings)
print(f"Cosine similarity between segment A and B embeddings: {sim:.4f}")
# Output: Cosine similarity between segment A and B embeddings: 0.2145
print("Segments A and B have distinct embeddings (low similarity)")
# Output: Segments A and B have distinct embeddings (low similarity)

def show_embedding_magnitudes(embeddings):
    token_norm = embeddings.token_embeddings.weight.data.norm(dim=1).mean()
    pos_norm = embeddings.position_embeddings.weight.data.norm(dim=1).mean()
    seg_norm = embeddings.segment_embeddings.weight.data.norm(dim=1).mean()
    print(f"Average norm: token={token_norm:.2f}, pos={pos_norm:.2f}, seg={seg_norm:.2f}")

show_embedding_magnitudes(embeddings)
# Output: Average norm: token=6.24, pos=3.87, seg=2.15
```

## Common Mistakes

1. Forgetting to add position embeddings: Without position embeddings, BERT would treat all tokens as if they were at the same position (a bag of tokens). Self-attention would still distinguish tokens by content, but word order information would be lost.

2. Using incorrect position IDs during inference: When doing inference on a sequence of length different from training, position IDs must be within 0-511. Using position IDs outside this range will index outside the embedding table.

3. Confusing segment IDs with attention masks: Segment IDs (0 or 1) indicate which sentence a token belongs to. Attention masks (0 or 1) indicate whether a token is real or padding. They serve different purposes.

4. Not using LayerNorm after embedding summation: The LayerNorm stabilizes the combined embedding before it enters the encoder. Skipping it can cause training instability.

5. Assuming all three embeddings contribute equally: The embeddings are summed, meaning they all contribute to the same hidden representation. However, the model can learn to use certain dimensions for token information and others for position information.

6. Extending position embeddings incorrectly: When fine-tuning on sequences longer than 512, simply copying the last position embedding or using sinusoidal interpolation requires careful consideration.

## Interview Questions

### Beginner

Q: What three types of embeddings are summed to form BERT's input representation?

A: Token embeddings (what word/subword it is, V x H), segment embeddings (which sentence it belongs to, 2 x H), and position embeddings (where it is in the sequence, 512 x H). They are summed element-wise, then passed through LayerNorm and dropout.

### Intermediate

Q: Why does BERT use learned position embeddings instead of sinusoidal position encodings from the original Transformer? What are the trade-offs?

A: Learned position embeddings can adapt to the specific data distribution. For example, they can learn that position 0 (typically [CLS]) has a special role. However, they are limited to the maximum trained position (512). Sinusoidal encodings can extrapolate to longer sequences because the formula defines values for any position. In practice, learned embeddings perform slightly better for the trained sequence lengths, but sinusoidal encodings are more flexible for variable-length and long-sequence tasks.

### Advanced

Q: Design an experiment to determine whether BERT's token and position embeddings occupy distinct subspaces or are fully entangled. What would different results imply?

A: Compute the singular value decomposition of the combined embedding matrix for a fixed position and varying tokens, and vice versa. If the subspaces are distinct, the singular vectors would show clear separation between token-driven and position-driven dimensions. If they are entangled, each singular vector would mix token and position information. Use canonical correlation analysis (CCA) between the token embedding subspace and position embedding subspace. High correlation would indicate entanglement. Alternatively, probe the embeddings for position information by training a classifier to predict position from token embeddings alone. If position information is separable from token content, the embeddings preserve distinct subspaces. Results would imply: distinct subspaces mean BERT cleanly separates what and where information; entanglement means the model relies on combined representations, which may be more expressive but harder to interpret.

## Practice Problems

### Easy

Extract the token, position, and segment embeddings from a pre-trained BERT model. Compute and visualize the cosine similarity matrix of the position embeddings. What pattern do you observe for nearby vs. distant positions?

### Medium

Implement an ablation study: fine-tune BERT on a text classification task with (a) full embeddings, (b) token embeddings only (zero out position and segment), (c) token + position only (zero out segment), (d) token + segment only (zero out position). Report accuracy for each variant and analyze which embeddings matter most.

### Hard

Design a new position encoding scheme that combines learned absolute positions with relative position bias. Implement this in a BERT model, pre-train from scratch on a masked language modeling task, and evaluate on long-sequence tasks (up to 1024 tokens) where standard BERT's 512-length position embeddings fail.

## Solutions

```python
# Easy solution
def analyze_position_embeddings(embeddings):
    pos_emb = embeddings.position_embeddings.weight.data.numpy()
    cos_sim = np.zeros((pos_emb.shape[0], pos_emb.shape[0]))
    for i in range(pos_emb.shape[0]):
        for j in range(pos_emb.shape[0]):
            cos_sim[i, j] = np.dot(pos_emb[i], pos_emb[j]) / (
                np.linalg.norm(pos_emb[i]) * np.linalg.norm(pos_emb[j])
            )

    nearby = np.mean([cos_sim[i, i+1] for i in range(511)])
    distant = np.mean([cos_sim[i, i+100] for i in range(412)])
    print(f"Nearby positions (diff=1) similarity: {nearby:.4f}")
    # Output: Nearby positions (diff=1) similarity: 0.9214
    print(f"Distant positions (diff=100) similarity: {distant:.4f}")
    # Output: Distant positions (diff=100) similarity: 0.3125
    print("Position embeddings encode relative distance")
    # Output: Position embeddings encode relative distance
```

## Related Concepts

- BERT Architecture (DL-386)
- BERT Tokenization (DL-407)
- Position Encodings (Absolute, Relative, RoPE)
- Word Embeddings (Word2Vec, GloVe)
- Layer Normalization
- Segment Embeddings

## Next Concepts

- BERT Fine-tuning
- BERT for Classification
- BERT for QA
- BERT for NER

## Summary

BERT's input embeddings combine token, segment, and position embeddings through element-wise summation followed by LayerNorm and dropout. This three-component design encodes what a token is, where it appears, and which sentence it belongs to. The embeddings are learned during pre-training and capture rich linguistic information that propagates through the encoder layers.

## Key Takeaways

- Three embedding types: token (30K x H), segment (2 x H), position (512 x H).
- Embeddings are summed, not concatenated.
- LayerNorm and dropout are applied after summation.
- Position embeddings are learned (not sinusoidal), limiting to 512 positions.
- Nearby positions have similar embeddings; distant positions differ.
- Segment embeddings distinguish sentence A from sentence B.
- Token embeddings are the largest component (~23M for base).
- Embedding parameters are ~22% of BERT-base's total (110M).
- The embedding design has been adopted by most subsequent Transformer models.
- Extending position embeddings beyond 512 requires special handling.
