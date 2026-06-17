# Concept: Embedding Layer

## Concept ID

DL-040

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Neural Network Building Blocks

## Learning Objectives

- Understand the purpose and mechanism of embedding layers
- Implement embeddings using PyTorch's `nn.Embedding`
- Explain how embeddings map discrete tokens to continuous vectors
- Analyze the properties of learned embedding spaces

## Prerequisites

DL-031 (Dense / Fully Connected Layer), DL-001 (Perceptron), DL-012 (Matrix Multiplication)

## Definition

An embedding layer is a lookup table that maps discrete tokens (indices) to continuous dense vectors. Given a vocabulary of size V and an embedding dimension d, the embedding layer stores a matrix **E** ∈ ℝ^{V × d}. Token i is mapped to the i-th row of **E**: embedding(i) = **E**[i, :]. The embedding vectors are learned during training through backpropagation.

## Intuition

Think of embeddings as converting discrete IDs into meaningful coordinates in a continuous space. Words with similar meanings end up close together in this space. For example, "king" and "queen" might be near each other, and the vector difference "king - man + woman" approximates "queen." The embedding layer simply retrieves the right row from a learned table.

## Why This Concept Matters

Embeddings are fundamental to modern NLP and recommendation systems:
- **Word embeddings**: Map words to vectors (Word2Vec, GloVe, fastText)
- **Token embeddings**: Used in all transformer language models
- **Entity embeddings**: Map categorical features in tabular data
- **Graph embeddings**: Map nodes to vectors
- **Multi-modal embeddings**: Map images, text, and other modalities to shared space

## Mathematical Explanation

Given vocabulary size V and embedding dimension d:

Embedding matrix: **E** ∈ ℝ^{V × d}

For a batch of token indices **I** ∈ ℕ^{batch × seq_len}:

Output = **E**[**I**, :] ∈ ℝ^{batch × seq_len × d}

The gradient with respect to the embedding matrix is sparse: only the rows corresponding to the input indices receive non-zero gradients.

∂L/∂**E**[i, :] = Σ_{t: I[t] = i} ∂L/∂output[t, :]

This sparse gradient is the key property that makes embedding layers efficient despite large vocabulary sizes.

## Code Examples

### Example 1: Basic embedding

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10, embedding_dim=4)
# Vocabulary size 10, each token mapped to 4D vector

indices = torch.tensor([1, 3, 5, 7])
vectors = embedding(indices)
print("Indices:", indices)
print("Embedding vectors:\n", vectors)
print("Shape:", vectors.shape)
# Output:
# Indices: tensor([1, 3, 5, 7])
# Embedding vectors:
#  tensor([[-0.1234,  0.5678, -0.9012,  0.3456],
#          [ 0.7890, -0.2345,  0.6789, -0.1234],
#          [ 0.4567,  0.8901, -0.3456,  0.1234],
#          [-0.5678,  0.9012, -0.2345,  0.6789]], requires_grad=True)
# Shape: torch.Size([4, 4])
```

### Example 2: Batch of sequences

```python
embedding = nn.Embedding(10, 3)
batch_indices = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (batch=2, seq=3)
vectors = embedding(batch_indices)
print("Input shape:", batch_indices.shape)
print("Output shape:", vectors.shape)  # (2, 3, 3)
print("Vectors:\n", vectors)
# Output:
# Input shape: torch.Size([2, 3])
# Output shape: torch.Size([2, 3, 3])
# Vectors:
#  tensor([[[-0.1234,  0.5678, -0.9012],
#           [ 0.3456, -0.7890,  0.1234],
#           [-0.5678,  0.9012, -0.3456]],
#          [[ 0.7890, -0.1234,  0.5678],
#           [-0.9012,  0.3456, -0.7890],
#           [ 0.1234, -0.5678,  0.9012]]], grad_fn=<EmbeddingBackward0>)
```

### Example 3: Using padding_idx

```python
embedding = nn.Embedding(10, 4, padding_idx=0)
# index 0 is reserved for padding — its gradient is always zero

batch = torch.tensor([[1, 2, 0, 3],  # 0 = padding
                       [4, 0, 0, 5]])
vectors = embedding(batch)
print("Output for padding (index 0):\n", vectors[0, 2])  # All zeros
# Output:
# Output for padding (index 0):
#  tensor([0., 0., 0., 0.], grad_fn=<SliceBackward0>)
```

### Example 4: Loading pre-trained embeddings (GloVe style)

```python
# Simulating pre-trained embeddings
pretrained_vectors = torch.tensor([
    [0.1, 0.2, 0.3],  # word 0: "cat"
    [0.4, 0.5, 0.6],  # word 1: "dog"
    [0.7, 0.8, 0.9],  # word 2: "bird"
])

embedding = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
# freeze=True prevents further training

indices = torch.tensor([0, 2])
vectors = embedding(indices)
print("Loaded embedding vectors:\n", vectors)
print("Requires grad:", vectors.requires_grad)
# Output:
# Loaded embedding vectors:
#  tensor([[0.1000, 0.2000, 0.3000],
#          [0.7000, 0.8000, 0.9000]])
# Requires grad: False
```

### Example 5: Visualizing embedding similarity

```python
embedding = nn.Embedding(8, 2)  # 2D for easy visualization
all_indices = torch.arange(8)
all_vectors = embedding(all_indices).detach()

# Compute cosine similarity
normed = all_vectors / all_vectors.norm(dim=1, keepdim=True)
similarity = normed @ normed.T

print("Cosine similarity matrix (first 4 tokens):\n", similarity[:4, :4])
# Output:
# Cosine similarity matrix (first 4 tokens):
#  tensor([[ 1.0000, -0.1234,  0.5678,  0.2345],
#          [-0.1234,  1.0000, -0.3456,  0.7890],
#          [ 0.5678, -0.3456,  1.0000, -0.4567],
#          [ 0.2345,  0.7890, -0.4567,  1.0000]])
```

### Example 6: Parameter count and memory

```python
# Large vocabulary embedding
vocab_size = 50000
emb_dim = 512
embedding = nn.Embedding(vocab_size, emb_dim)
print(f"Parameters: {sum(p.numel() for p in embedding.parameters()):,}")
print(f"Memory: {sum(p.numel() for p in embedding.parameters()) * 4 / 1024 / 1024:.2f} MB (float32)")
# Output:
# Parameters: 25,600,000
# Memory: 97.66 MB (float32)
```

## Common Mistakes

1. **Confusing vocabulary size and embedding dimension**: `num_embeddings` is the number of tokens, `embedding_dim` is the vector size. They are independent.

2. **Using embeddings for non-integer inputs**: Embedding layers require integer indices. If you pass floats, PyTorch raises an error.

3. **Forgetting that index 0 is a valid token**: Unless you set `padding_idx=0`, index 0 is a regular trainable embedding. Common mistake: assuming 0 is automatically padding.

4. **Training embeddings from scratch with small data**: With limited data, pre-trained embeddings (GloVe, FastText, BERT) are much more effective than randomly initialized ones.

5. **Using too large embedding dimensions**: Larger dim = more parameters but not always better. The embedding matrix for V=50k, d=1024 has ~50M+ parameters.

6. **Not handling out-of-vocabulary tokens**: If an index exceeds `num_embeddings - 1`, you get an error. Always handle OOV tokens by mapping them to a special [UNK] token.

7. **Forgetting that embeddings produce gradients**: If using `nn.Embedding.from_pretrained` without `freeze=True`, the embeddings will be updated during training.

## Interview Questions

### Beginner - 5

1. What is an embedding layer?
2. How do you create an embedding layer in PyTorch for a vocabulary of 10,000 tokens with 300-dimensional vectors?
3. What is the shape of the embedding matrix?
4. How does the forward pass of an embedding layer work?
5. What does `padding_idx` do in `nn.Embedding`?

### Intermediate - 5

1. How does backpropagation work through an embedding layer?
2. Compare randomly initialized embeddings vs. pre-trained embeddings.
3. What is the relationship between embedding layers and one-hot encoding followed by a linear layer?
4. How do you handle out-of-vocabulary words when using embedding layers?
5. Explain how embeddings capture semantic similarity.

### Advanced - 3

1. Implement a subword embedding layer using BPE or WordPiece tokenization.
2. Derive and implement adaptive embedding (where embedding dimension can vary per token based on frequency).
3. How would you implement a multi-modal embedding layer that maps both text and images into a shared space?

## Practice Problems

### Easy - 5

1. Create an embedding layer with vocabulary size 100 and embedding dimension 16.
2. Pass indices [0, 5, 10] through the embedding and print the output shape.
3. Create an embedding with `padding_idx=0` and verify the zero vector output.
4. Load a random matrix as a pre-trained embedding using `nn.Embedding.from_pretrained`.
5. Compute the cosine similarity between two embeddings from the same layer.

### Medium - 5

1. Implement an embedding layer from scratch (as a lookup table) and compare gradients with PyTorch's.
2. Train a word embedding model (CBOW or Skip-gram) on a small corpus using `nn.Embedding`.
3. Visualize 2D embeddings of 10 tokens and analyze their clustering.
4. Compare training with frozen vs. trainable pre-trained embeddings.
5. Implement an embedding layer that supports weighted sum of multiple embedding vectors per position.

### Hard - 3

1. Implement an adaptive embedding layer where rare tokens share sub-vectors from a smaller base embedding matrix.
2. Build a contextual embedding layer (like BERT) that outputs different vectors for the same token based on context.
3. Implement a hash-based embedding layer for extremely large vocabularies (e.g., 10M tokens) with sub-linear memory.

## Solutions

### Easy - 1
```python
emb = nn.Embedding(100, 16)
```

### Easy - 2
```python
emb = nn.Embedding(100, 16)
indices = torch.tensor([0, 5, 10])
out = emb(indices)
print(out.shape)  # (3, 16)
```

### Easy - 3
```python
emb = nn.Embedding(5, 3, padding_idx=0)
out = emb(torch.tensor([[0, 1, 2]]))
print(out[0, 0])  # Should be [0, 0, 0]
```

## Related Concepts

DL-031 Dense / Fully Connected Layer, DL-048 Softmax Output, DL-051 Feature Hierarchy, DL-052 Information Flow

## Next Concepts

DL-041 Residual Connection, DL-053 Computational Graph

## Summary

An embedding layer maps discrete tokens to continuous vector representations via a learned lookup table. It is the primary mechanism for neural networks to process categorical data, especially text. Embeddings capture semantic relationships in their vector space geometry and are a cornerstone of modern NLP.

## Key Takeaways

- Maps integer token IDs to dense vectors via lookup table
- Embedding matrix shape: (vocab_size, embedding_dim)
- Sparse gradients: only rows corresponding to input tokens are updated
- `padding_idx` reserves a token for zero-vector padding
- Pre-trained embeddings transfer knowledge from large corpora
- Embedding dimension is a hyperparameter (typical: 50-1024)
- Equivalent to one-hot encoding + linear projection
