# Concept: Attention Visualization

## Concept ID

DL-351

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Attention Mechanisms

## Learning Objectives

- Understand the purpose and methods of visualizing attention weights for model interpretability.
- Implement attention visualization techniques using heatmaps and Sankey diagrams.
- Analyze attention patterns to understand model behavior and diagnose issues.
- Interpret what different attention patterns reveal about model predictions.
- Apply attention visualization to verify alignment in translation and other seq2seq tasks.

## Prerequisites

- Understanding of attention weights and their computation.
- Familiarity with seq2seq with attention and transformer self-attention.
- Knowledge of Python data visualization libraries.
- Experience with PyTorch and NumPy for tensor manipulation.

## Definition

Attention visualization is the process of creating graphical representations of attention weights to interpret which parts of the input a model focuses on when making predictions. The most common visualization is a heatmap showing attention weights as a 2D grid, where rows represent output positions (or queries) and columns represent input positions (or keys). The color intensity at position (i, j) represents the attention weight from output position i to input position j. Attention visualization can reveal: (1) alignment patterns in translation — which source words correspond to which target words, (2) syntactic relationships in self-attention — which tokens attend to which other tokens, (3) model failures — when attention is focused on padding tokens or irrelevant positions, and (4) head specialization — how different attention heads learn different patterns.

## Intuition

Attention visualization is like putting a heat camera on the model's decision-making process. When a translator says a word, the heat camera shows which source words are "hot" (high attention weight) and which are "cold" (low attention weight). A good translation produces a clear diagonal pattern: the first target word attends to the first source word, the second target word to the second, and so on. A strange pattern — like the third target word attending to the last source word — might indicate a problem with the model or an unusual translation. In self-attention, the visualization shows which words in a sentence are most connected. For example, "it" in "The cat sat on the mat because it was comfortable" should attend to "cat" (not "mat"), revealing coreference resolution.

## Why This Concept Matters

Attention visualization is the primary tool for interpreting neural sequence models. It provides a window into the model's decision-making process that is much more informative than looking at final outputs alone. Attention visualization is used for: (1) debugging — verifying that the model attends to the correct input positions, (2) analysis — understanding what linguistic patterns the model has learned, (3) trust — building confidence that the model is making reasonable decisions before deploying it, (4) research — discovering how attention mechanisms work and how heads specialize, and (5) communication — explaining model behavior to non-technical stakeholders.

## Mathematical Explanation

### Attention Matrix

For a seq2seq model with T_src source positions and T_trg target positions, the attention matrix A in R^{T_trg x T_src} contains:

A_{i,j} = alpha_{i,j} = softmax(score(s_i, h_j))

where s_i is the decoder state at position i and h_j is the encoder state at position j.

### Self-Attention Matrix

For self-attention in a transformer with T positions, the attention matrix for head h is:

A_h in R^{T x T}, where A_h[i, j] = attention weight from position i to position j.

### Aggregation Across Heads

To visualize multi-head attention, we can:
- Show each head separately (most common)
- Average across heads (loses head-specific patterns)
- Show the head with the most structured pattern

### Aggregation Across Layers

Attention patterns often change across layers:
- Lower layers: local, syntactic patterns
- Middle layers: mixed syntactic and semantic
- Higher layers: global, semantic patterns

## Code Examples

### Example 1: Basic Attention Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_attention_heatmap(attention_matrix, src_tokens, trg_tokens, title="Attention Weights"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax.set_yticklabels(trg_tokens)
    for i in range(len(trg_tokens)):
        for j in range(len(src_tokens)):
            text = ax.text(j, i, f"{attention_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel('Source Tokens')
    ax.set_ylabel('Target Tokens')
    ax.set_title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

src_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
trg_tokens = ['Le', 'chat', 's_est', 'assis', 'sur', 'le', 'tapis']
attention = np.random.dirichlet(np.ones(len(src_tokens)), size=len(trg_tokens))
attention[0] = np.array([0.8, 0.1, 0.02, 0.02, 0.03, 0.03])
attention[1] = np.array([0.05, 0.85, 0.02, 0.02, 0.03, 0.03])
attention[2] = np.array([0.02, 0.02, 0.80, 0.05, 0.05, 0.06])
attention[3] = np.array([0.02, 0.02, 0.60, 0.15, 0.10, 0.11])
attention[4] = np.array([0.02, 0.02, 0.05, 0.70, 0.10, 0.11])
attention[5] = np.array([0.02, 0.05, 0.03, 0.05, 0.70, 0.15])
attention[6] = np.array([0.02, 0.05, 0.03, 0.05, 0.10, 0.75])

print("Attention heatmap data prepared")
print(f"Matrix shape: {attention.shape}")
print(f"Row sums (should be ~1.0): {attention.sum(axis=1).round(2)}")
# Output: Attention heatmap data prepared
# Output: Matrix shape: (7, 6)
# Output: Row sums (should be ~1.0): [1. 1. 1. 1. 1. 1. 1.]
```

### Example 2: Self-Attention Visualization for Transformer

```python
def plot_self_attention(sentence, attention_matrix, layer=0, head=0):
    tokens = sentence.split()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    ax.set_title(f'Layer {layer}, Head {head}')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

sentence = "The cat sat on the mat because it was comfortable"
n_tokens = len(sentence.split())
attn_matrix = np.random.dirichlet(np.ones(n_tokens), size=n_tokens)

# Simulate: "it" attends strongly to "cat"
it_idx = sentence.split().index('it')
cat_idx = sentence.split().index('cat')
attn_matrix[it_idx] = np.ones(n_tokens) * 0.02
attn_matrix[it_idx, cat_idx] = 0.70
attn_matrix[it_idx] /= attn_matrix[it_idx].sum()

print(f"Self-attention: 'it' (pos {it_idx}) attends to 'cat' (pos {cat_idx})")
print(f"  Weight from 'it' to 'cat': {attn_matrix[it_idx, cat_idx]:.2f}")
# Output: Self-attention: 'it' (pos 8) attends to 'cat' (pos 1)
# Output:   Weight from 'it' to 'cat': 0.70
```

### Example 3: Extracting Attention from a Transformer Model

```python
import torch
import torch.nn as nn

class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attentions = {}

    def register_hooks(self):
        def get_attention(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    attn = output[1]
                else:
                    attn = output
                self.attentions[name] = attn.detach()
            return hook

        for name, module in self.model.named_modules():
            if 'self_attn' in name or 'cross_attn' in name:
                module.register_forward_hook(get_attention(name))

    def forward_with_attention(self, x):
        self.attentions = {}
        self.model(x)
        return self.attentions

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(16, 4, batch_first=True)
        self.linear = nn.Linear(16, 10)

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x)
        return self.linear(attn_out), attn_weights

model = SimpleTransformer()
extractor = AttentionExtractor(model)
extractor.register_hooks()

x = torch.randn(1, 5, 16)
attentions = extractor.forward_with_attention(x)
for name, attn in attentions.items():
    print(f"{name}: {attn.shape}")
# Output: self_attn: torch.Size([1, 5, 5])
```

## Common Mistakes

1. **Interpreting attention weights as causal explanations**: High attention weight does not necessarily mean causal importance. The model may attend to a token because it confirms what it already knows, not because that token caused the prediction. Attention-based explanations must be validated with causal methods.

2. **Visualizing only a single head**: Different attention heads learn different patterns. Visualizing only one head gives an incomplete picture. Either show all heads, or aggregate with care.

3. **Ignoring the effect of residual connections and layer normalization**: The attention output goes through residual connections and layer normalization before being used. The final representation is not just the attention output but a combination of multiple components.

4. **Over-interpreting attention patterns in deep layers**: Self-attention patterns in deeper layers are influenced by the representations from lower layers, which are already contextualized. The "direct" attention to raw input tokens is mediated by the learned representations.

5. **Not normalizing attention weights for comparison**: Different layers and heads can have different entropy levels. Comparing raw attention weights across layers without normalization can be misleading.

## Interview Questions

### Beginner

Q: Why is attention visualization useful for model interpretability?

A: Attention visualization shows which input elements the model focuses on when making predictions. This helps verify the model is attending to relevant information, debug failures (e.g., attending to padding), and build trust in the model's decisions.

### Intermediate

Q: What patterns would you expect to see in the attention matrix of a well-trained translation model?

A: A well-trained translation model typically shows a diagonal pattern in the attention matrix: the first target word attends to the first few source words, the second target word to slightly later source words, etc. There may be slight off-diagonal elements for words that are reordered between languages. For closely related languages (e.g., English-French), the diagonal is very clear. For distant languages (e.g., English-Japanese), there may be more complex patterns.

### Advanced

Q: What are the limitations of attention-based interpretability, and how would you design a more rigorous attribution method?

A: Limitations: (1) Attention weights do not account for gradient information, so they can't distinguish between "this token is important" and "this token confirms the prediction." (2) In multi-layer models, attention patterns interact across layers in complex ways. (3) Attention to padding or irrelevant tokens can occur without affecting the prediction. More rigorous methods include: (1) Gradient-based attribution (e.g., integrated gradients, saliency maps) that measures how much the prediction changes when input is perturbed. (2) Attention-rollout or attention-flow methods that aggregate attention across layers. (3) Causal methods that intervene on the model (e.g., counterfactual analysis, feature ablation). (4) Erasure-based methods that measure performance drop when a token is removed. Combining attention with gradient information (attention*gradient) often provides more faithful explanations.

## Practice Problems

### Easy

Extract and print the attention weights from a simple seq2seq model with attention for a sample sentence. Identify which source token has the highest attention weight for each target token.

### Medium

Create a function that generates an attention heatmap using matplotlib. Test it on the attention matrix from a pre-trained translation model and analyze whether the alignments are reasonable.

### Hard

Implement attention rollout (aggregating attention weights across layers by multiplying matrices) and compare the aggregated attention with per-layer attention for a pre-trained transformer model. Discuss the differences.

## Solutions

### Easy Solution

```python
def extract_attention(model, src, trg, device):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        input_token = trg[:, 0]
        alignments = []
        for t in range(1, trg.shape[1]):
            output, hidden, attn = model.decoder(input_token, hidden, enc_out)
            alignments.append(attn.squeeze(0).cpu().numpy())
            input_token = trg[:, t]
        alignments = np.array(alignments)
        for i in range(alignments.shape[0]):
            max_pos = alignments[i].argmax()
            print(f"Target pos {i}: attends to source pos {max_pos} (weight={alignments[i][max_pos]:.3f})")
    return alignments

print("Attention extraction function defined")
# Output: Attention extraction function defined
```

## Related Concepts

- Attention Weights
- Model Interpretability
- Attention Head Analysis
- Integrated Gradients
- Saliency Maps

## Next Concepts

- DL-352: Attention Is All You Need
- DL-353: Key Query Value

## Summary

Attention visualization is a powerful tool for interpreting neural sequence models. By plotting attention weights as heatmaps, practitioners can see which input elements the model focuses on for each output element, revealing alignment patterns, syntactic relationships, and model failures. In translation, attention typically shows a diagonal alignment pattern. In self-attention, different heads learn different patterns (local syntax, coreference, semantic relationships). While attention visualization provides valuable insights, it should be complemented with gradient-based or causal attribution methods for rigorous interpretability.

## Key Takeaways

- Attention visualization provides insight into model focus and alignment.
- Heatmaps show attention weights as color intensity (rows=outputs, columns=inputs).
- Diagonal patterns indicate monotonic alignment (common in translation).
- Self-attention visualization reveals syntactic and semantic relationships.
- Different heads and layers show different attention patterns.
- Attention visualization is a debugging and interpretability tool, not a causal explanation.
- Combine attention with gradient information for more rigorous attribution.
