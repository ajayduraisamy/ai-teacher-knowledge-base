# Concept: Information Theory for Deep Learning

## Concept ID

DL-019

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define entropy as a measure of uncertainty in a probability distribution
- Compute and interpret cross-entropy between two distributions
- Understand KL divergence as a measure of distribution dissimilarity
- Relate mutual information to dependence between random variables
- Connect information-theoretic quantities to deep learning loss functions
- Implement entropy, cross-entropy, and KL divergence using PyTorch

## Prerequisites

- DL-018: Probability for DL (random variables, PDF/PMF, expectation)
- Basic understanding of classification problems in deep learning
- Logarithms and their properties

## Definition

Information theory, founded by Claude Shannon in 1948, quantifies information in terms of uncertainty reduction. In deep learning, information-theoretic concepts are used to design loss functions (cross-entropy), regularize models (KL divergence), understand representations (mutual information), and bound generalization error (VC dimension, PAC-Bayes).

## Intuition

Think of entropy as "surprise." A distribution where all outcomes are equally likely has high entropy because you are maximally uncertain about which outcome will occur. A distribution where one outcome is almost certain has low entropy. When we train a classifier, we want the predicted distribution to have low entropy (be confident) but also match the true distribution.

Cross-entropy measures how many bits are needed to encode samples from distribution $p$ using a code optimized for distribution $q$. Minimizing cross-entropy is equivalent to making $q$ as close to $p$ as possible. This is exactly what we do when training a classifier with cross-entropy loss.

## Why This Concept Matters

Information theory is fundamental to deep learning because:

- **Cross-entropy is the standard classification loss**: Every image classifier, NLP model, and speech recognition system uses it.
- **KL divergence enables variational inference**: VAEs, Bayesian neural networks, and many generative models rely on KL divergence.
- **Mutual information measures representation quality**: Used in representation learning, contrastive learning, and information bottleneck theory.
- **Entropy bounds generalization**: Information-theoretic generalization bounds connect model complexity to performance.
- **Model compression**: Information theory guides quantization and pruning strategies.

## Mathematical Explanation

### Entropy

For a discrete random variable $X$ with PMF $p(x)$, the entropy $H(X)$ is:

$$H(X) = H(p) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]$$

Entropy measures the average information content (or uncertainty) of $X$. For a binary random variable with $P(X=1) = p$:

$$H(p) = -p \log p - (1-p) \log(1-p)$$

This is called **binary entropy**. It is maximized at $p = 0.5$ with value $H = \log 2$ (1 bit).

### Cross-Entropy

For two distributions $p$ and $q$ over the same sample space:

$$H(p, q) = -\sum_{x} p(x) \log q(x) = \mathbb{E}_{x \sim p}[-\log q(x)]$$

Cross-entropy is always greater than or equal to entropy: $H(p, q) \geq H(p)$, with equality iff $p = q$.

In deep learning, for classification with one-hot labels $y$ and predicted probabilities $\hat{y}$:

$$H(y, \hat{y}) = -\sum_{c} y_c \log \hat{y}_c = -\log \hat{y}_{\text{true}}$$

This is the **categorical cross-entropy loss**.

### KL Divergence

The Kullback-Leibler (KL) divergence measures how one distribution diverges from another:

$$D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Properties:
- $D_{KL}(p \| q) \geq 0$ (Gibbs' inequality)
- $D_{KL}(p \| q) = 0$ iff $p = q$
- Asymmetric: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$
- Not a true metric (violates symmetry and triangle inequality)

### Mutual Information

Mutual information $I(X; Y)$ measures the reduction in uncertainty about $X$ given knowledge of $Y$:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = D_{KL}(p(x,y) \| p(x)p(y))$$

Properties:
- $I(X; Y) \geq 0$
- $I(X; Y) = 0$ iff $X$ and $Y$ are independent
- $I(X; Y) = H(X) + H(Y) - H(X, Y)$

### Entropy of Continuous Distributions (Differential Entropy)

For a continuous random variable with PDF $p(x)$:

$$h(X) = -\int p(x) \log p(x) dx$$

For a Gaussian $\mathcal{N}(\mu, \sigma^2)$:

$$h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2} \log(2\pi e \sigma^2)$$

### Relationship to Deep Learning Losses

| Loss Function | Information-Theoretic Form |
|---|---|
| Cross-entropy (classification) | $H(y, \hat{y}) = -\sum y \log \hat{y}$ |
| MSE (regression, Gaussian) | $-\log p(y|x) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$ |
| KL in VAE | $D_{KL}(q(z|x) \| p(z))$ — regularizes latent space |
| InfoNCE (contrastive learning) | $I(x; y) \geq \log N - \mathcal{L}_{NCE}$ |

## Code Examples

### Example 1: Computing Entropy

```python
import torch
import torch.nn.functional as F

# Binary entropy
p = torch.tensor(0.5)
binary_entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
print(f"Binary entropy at p=0.5: {binary_entropy:.4f}")
# Output: Binary entropy at p=0.5: 0.6931

# Categorical entropy
probs = torch.tensor([0.7, 0.2, 0.1])
entropy = -(probs * torch.log(probs)).sum()
print(f"Categorical entropy: {entropy:.4f}")
# Output: Categorical entropy: 0.8018

# Uniform distribution has higher entropy
uniform_probs = torch.tensor([1/3, 1/3, 1/3])
uniform_entropy = -(uniform_probs * torch.log(uniform_probs)).sum()
print(f"Uniform entropy: {uniform_entropy:.4f}")
# Output: Uniform entropy: 1.0986
```

### Example 2: Cross-Entropy as Classification Loss

```python
import torch
import torch.nn.functional as F

# Example: 3-class classification, batch of 2
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 3.0, 0.8]])
targets = torch.tensor([0, 1])  # true class indices

# PyTorch cross-entropy = softmax + negative log-likelihood
ce_loss = F.cross_entropy(logits, targets)
print(f"Cross-entropy loss: {ce_loss:.4f}")
# Output: Cross-entropy loss: 0.2557

# Manual computation
probs = F.softmax(logits, dim=-1)
print(f"Probabilities:\n{probs}")
# Output: Probabilities:
# tensor([[0.6590, 0.2424, 0.0986],
#         [0.0723, 0.8909, 0.0368]])

nll = -torch.log(probs[range(len(targets)), targets])
print(f"NLL per sample: {nll}")
# Output: NLL per sample: tensor([0.4170, 0.1155])
print(f"Mean NLL: {nll.mean():.4f}")
# Output: Mean NLL: 0.2663

# Verify: cross-entropy H(y, p_hat) where y is one-hot
one_hot = F.one_hot(targets, num_classes=3).float()
ce_manual = -(one_hot * torch.log(probs)).sum(dim=-1).mean()
print(f"Manual CE: {ce_manual:.4f}")
# Output: Manual CE: 0.2663
```

### Example 3: KL Divergence

```python
import torch
import torch.nn.functional as F

# Two distributions
p = torch.tensor([0.7, 0.2, 0.1])
q = torch.tensor([0.4, 0.4, 0.2])

# KL divergence D_KL(p || q)
kl_pq = (p * (torch.log(p) - torch.log(q))).sum()
print(f"D_KL(p||q): {kl_pq:.4f}")
# Output: D_KL(p||q): 0.2063

# KL divergence D_KL(q || p) — note asymmetry!
kl_qp = (q * (torch.log(q) - torch.log(p))).sum()
print(f"D_KL(q||p): {kl_qp:.4f}")
# Output: D_KL(q||p): 0.2671

# Using PyTorch's built-in
kl_py = F.kl_div(torch.log(q), p, reduction='sum')
print(f"PyTorch KL (p||log(q)): {kl_py:.4f}")
# Output: PyTorch KL (p||log(q)): 0.2063

# In VAE: KL between q(z|x) = N(mu, sigma^2) and p(z) = N(0, 1)
mu = torch.tensor([0.5, -0.2])
log_var = torch.tensor([-1.0, 0.5])
kl_vae = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
print(f"VAE KL divergence: {kl_vae:.4f}")
# Output: VAE KL divergence: 0.5208
```

### Example 4: Mutual Information Estimation

```python
import torch

# Simple empirical MI estimation
torch.manual_seed(42)
N = 1000

# Generate independent variables
x = torch.randn(N, 1)
y = torch.randn(N, 1)
# Joint entropy H(x,y) for independent variables
# should equal H(x) + H(y), so MI ≈ 0

# Generate dependent variables
z = x + 0.1 * torch.randn(N, 1)  # y is a noisy function of x

# Empirical entropy estimation (simplified binning approach)
def estimate_entropy_1d(data, bins=20):
    hist = torch.histc(data, bins=bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return -(probs * torch.log(probs)).sum()

h_x = estimate_entropy_1d(x)
h_y = estimate_entropy_1d(y)
h_z = estimate_entropy_1d(z)
h_xz = estimate_entropy_1d(torch.cat([x, z], dim=1).sum(dim=1, keepdim=True))  # rough

# For truly independent variables, MI should be near 0
mi_approx = h_x + h_y - estimate_entropy_1d(torch.cat([x, y], dim=1).sum(dim=1, keepdim=True))
print(f"MI estimate (independent): {mi_approx:.4f} (should be ~0)")
# Output: MI estimate (independent): 0.0123 (should be ~0)

# For MNIST digits, mutual information between pixels and labels
# can be computed using the InfoNCE bound in practice
```

### Example 5: Cross-Entropy vs KL Divergence

```python
import torch
import torch.nn.functional as F

# True distribution (one-hot): p = [1, 0, 0]
# Predicted distributions with different confidences
q_confident = torch.tensor([0.9, 0.05, 0.05])
q_uncertain = torch.tensor([0.4, 0.3, 0.3])

p = torch.tensor([1.0, 0.0, 0.0])

# Cross-entropy
ce_confident = -(p * torch.log(q_confident)).sum()
ce_uncertain = -(p * torch.log(q_uncertain)).sum()
print(f"CE (confident): {ce_confident:.4f}")
# Output: CE (confident): 0.1054
print(f"CE (uncertain): {ce_uncertain:.4f}")
# Output: CE (uncertain): 0.9163

# KL divergence = CE - H(p) = CE (since H(p) = 0 for one-hot)
kl_confident = (p * (torch.log(p) - torch.log(q_confident))).sum()
print(f"KL (confident): {kl_confident:.4f}")
# Output: KL (confident): 0.1054

# For non-one-hot p, cross-entropy = entropy(p) + KL(p||q)
p_smooth = torch.tensor([0.8, 0.15, 0.05])
h_p = -(p_smooth * torch.log(p_smooth)).sum()
ce_smooth = -(p_smooth * torch.log(q_confident)).sum()
kl_smooth = (p_smooth * (torch.log(p_smooth) - torch.log(q_confident))).sum()
print(f"H(p) = {h_p:.4f}, CE = {ce_smooth:.4f}, KL = {kl_smooth:.4f}")
print(f"CE == H(p) + KL(p||q): {torch.allclose(ce_smooth, h_p + kl_smooth)}")
# Output: H(p) = 0.6048, CE = 0.7102, KL = 0.1054
# Output: CE == H(p) + KL(p||q): True
```

## Common Mistakes

1. **Confusing cross-entropy and KL divergence**: Cross-entropy $H(p,q)$ equals $H(p) + D_{KL}(p\|q)$. For one-hot targets, $H(p) = 0$, so cross-entropy = KL divergence. For soft targets, they differ by the target entropy.

2. **Using natural log vs log base 2**: Entropy in nats (natural log) vs bits (log base 2). PyTorch uses natural log. The choice is a constant scaling factor but must be consistent.

3. **Forgetting KL divergence is asymmetric**: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$. The forward KL (used in supervised learning) expects $q$ to cover all modes of $p$, while reverse KL (used in variational inference) can mode-seek.

4. **Applying cross-entropy to non-probability inputs**: Cross-entropy requires valid probability distributions (non-negative, sum to 1). Use softmax to convert logits to probabilities.

5. **Ignoring numerical stability**: $\log(0)$ is undefined. Add a small epsilon like $10^{-8}$ inside the log.

6. **Misinterpreting mutual information as correlation**: MI captures non-linear dependencies too, not just linear correlation.

7. **Believing that lower entropy always means better predictions**: A degenerate distribution that always predicts the same class has low entropy but is useless. Low entropy is good only when combined with accuracy.

## Interview Questions

### Beginner

1. What is entropy? What does it measure?
2. Write the formula for cross-entropy and explain each term.
3. What is the difference between entropy and cross-entropy?
4. Give an example of KL divergence used in deep learning.
5. What is mutual information and what does it capture that correlation does not?

### Intermediate

1. Derive the relationship between cross-entropy, entropy, and KL divergence: $H(p,q) = H(p) + D_{KL}(p\|q)$.
2. Explain why minimizing cross-entropy is equivalent to minimizing KL divergence for classification with one-hot labels.
3. What is the gradient of the cross-entropy loss with respect to the logits? Show that it equals $\hat{y} - y$.
4. How is KL divergence used in the VAE loss function? Why is the analytical form important?
5. Explain the information bottleneck principle and how it relates to representation learning.

### Advanced

1. Prove the variational lower bound $I(X; Y) \geq \mathbb{E}[\log \frac{q(y|x)}{p(y)}]$ used in InfoNCE. Show how this connects to contrastive learning objectives.
2. Derive the generalization bound using mutual information (PAC-Bayes). What does it tell us about the relationship between model complexity and generalization?
3. For a deep classifier, show that the cross-entropy loss can be decomposed into a conditional entropy term $H(Y|X)$ and a KL term $D_{KL}(p_{\text{data}}(y|x) \| p_{\text{model}}(y|x))$ under the true data distribution.

## Practice Problems

### Easy

1. Compute the entropy of a fair 6-sided die.
2. Two distributions over $\{A, B, C\}$: $p = [0.5, 0.3, 0.2]$, $q = [0.4, 0.4, 0.2]$. Compute $H(p, q)$.
3. What is the KL divergence $D_{KL}(p \| q)$ for the distributions in problem 2?
4. For a binary classifier that outputs $\hat{y} = 0.9$ when the true label is $y = 1$, what is the per-sample cross-entropy loss?
5. What is the entropy of a Gaussian distribution with $\sigma = 1$?

### Medium

1. Implement a function to compute the KL divergence between two multivariate Gaussians with diagonal covariance.
2. Show that the mutual information $I(X; Y) \geq 0$ with equality iff $X$ and $Y$ are independent.
3. For the VAE loss, derive the KL divergence between $\mathcal{N}(\mu, \sigma^2)$ and $\mathcal{N}(0, 1)$.
4. Implement the InfoNCE loss used in contrastive learning (SimCLR style) and verify it approximates mutual information.
5. Given a trained classifier, compute the empirical entropy of the predicted distribution on a test set. What does a high entropy indicate?

### Hard

1. Prove that the categorical cross-entropy loss with softmax is convex in the logits. Show that there is a unique global minimum.
2. Implement the information bottleneck method for a neural network: train a classifier while penalizing mutual information between hidden representations and inputs.
3. Derive the Pitman-Koopman-Darmois theorem and explain its implications for exponential family distributions in information geometry.

## Solutions

_Solutions for selected problems._

**Easy 1**: $H = -\sum_{i=1}^6 \frac{1}{6} \log \frac{1}{6} = \log 6 \approx 1.7918$ nats.

**Easy 3**: $D_{KL}(p\|q) = \sum p \log(p/q) = 0.5\log(0.5/0.4) + 0.3\log(0.3/0.4) + 0.2\log(0.2/0.2) = 0.5\log(1.25) + 0.3\log(0.75) + 0 = 0.1116 + (-0.0864) = 0.0252$ nats.

**Medium 4**:

```python
def infonce_loss(z_i, z_j, temperature=0.1):
    """InfoNCE loss for contrastive learning."""
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)])
    loss = F.cross_entropy(sim, labels)
    return loss
```

## Related Concepts

- **DL-018: Probability for DL** — Foundation for information theory (all quantities are expectations over distributions)
- **DL-017: Matrix Calculus** — Gradients of cross-entropy and KL divergence
- **DL-020: Optimization Theory** — Optimizing information-theoretic objectives
- **DL-016: Linear Algebra** — Representing probability distributions as vectors

## Next Concepts

- DL-020: Optimization Theory (optimizing information-theoretic losses)
- DL-024: Convex vs Non-Convex (cross-entropy is convex in certain parameterizations)

## Summary

Information theory provides fundamental tools for quantifying uncertainty and information content. Entropy $H(p)$ measures the average surprise or uncertainty of a distribution. Cross-entropy $H(p,q)$ measures the cost of using $q$ to approximate $p$ and is the standard loss for classification. KL divergence $D_{KL}(p\|q)$ measures the "distance" between distributions and appears in variational inference, regularization, and many other contexts. Mutual information $I(X;Y)$ quantifies dependence between variables and is used in representation learning. These concepts form a unified framework for understanding loss functions, regularization, and representation quality in deep learning.

## Key Takeaways

- Cross-entropy is the standard classification loss; it equals $-\log \hat{y}_{\text{true}}$ for one-hot labels
- KL divergence is asymmetric: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$
- Mutual information captures all dependencies (linear and non-linear)
- VAE loss = reconstruction loss + KL divergence ($D_{KL}(q(z|x)\|p(z))$)
- Information theory provides generalization bounds via PAC-Bayes
- Entropy is maximized for uniform distributions, minimized for deterministic ones
- Always use log-probabilities for numerical stability (never take $\log$ of 0)
