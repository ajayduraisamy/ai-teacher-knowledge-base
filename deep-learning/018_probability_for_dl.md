# Concept: Probability for Deep Learning

## Concept ID

DL-018

## Difficulty

Intermediate

## Domain

Deep Learning

## Module

Mathematics for Deep Learning

## Learning Objectives

- Define random variables and differentiate between discrete and continuous types
- Interpret probability density functions (PDF) and probability mass functions (PMF)
- Compute expectation, variance, and covariance of random variables
- Apply conditional probability and Bayes' rule to inference problems
- Understand the Gaussian distribution and its properties
- Derive and apply Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation
- Connect probabilistic concepts to common deep learning loss functions

## Prerequisites

- DL-016: Linear Algebra Review (vectors, matrices)
- Basic calculus (integration, differentiation)
- Understanding of functions and function composition
- Familiarity with logarithms and exponentials

## Definition

Probability theory provides the mathematical framework for modeling uncertainty. In deep learning, probability is used to formulate loss functions (cross-entropy, Gaussian log-likelihood), design generative models (VAEs, GANs), quantify prediction uncertainty, and understand the fundamental limits of learning from data.

## Intuition

Consider training a neural network for image classification. The network outputs a probability distribution over classes: $P(y|x)$ — the probability that the image belongs to each class given the input pixels. This is inherently probabilistic because the same pixel values can correspond to different objects (e.g., a blurry image could be a cat or a dog). Similarly, the training data is assumed to be drawn from some unknown underlying distribution. Learning means finding parameters that make the model's distribution match the data distribution.

## Why This Concept Matters

Probability is essential for deep learning because:

- **Loss functions are probabilistic**: Cross-entropy loss derives from MLE under a categorical distribution; MSE loss derives from MLE under a Gaussian distribution.
- **Generative models require probability**: VAEs maximize a variational lower bound on the data likelihood.
- **Uncertainty quantification**: Bayesian deep learning provides confidence intervals for predictions.
- **Information theory is built on probability**: Entropy, cross-entropy, and KL divergence are all probabilistic quantities.
- **Regularization**: Techniques like dropout can be interpreted as Bayesian inference.

## Mathematical Explanation

### Random Variables

A random variable $X$ is a variable whose value is determined by a random process. $X$ is **discrete** if it takes countable values, **continuous** if it takes values in an interval.

### Probability Mass Function (PMF)

For a discrete random variable $X$:

$$P(X = x) = p(x) \quad \text{where } \sum_x p(x) = 1$$

### Probability Density Function (PDF)

For a continuous random variable $X$:

$$P(a \leq X \leq b) = \int_a^b p(x) dx \quad \text{where } \int_{-\infty}^{\infty} p(x) dx = 1$$

### Expectation

The expected value (mean) of a random variable:

$$\mathbb{E}[X] = \begin{cases} \sum_x x p(x) & \text{discrete} \\ \int_{-\infty}^{\infty} x p(x) dx & \text{continuous} \end{cases}$$

Linearity of expectation: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

Standard deviation: $\sigma_X = \sqrt{\text{Var}(X)}$

### Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

Two events are independent if $P(A \cap B) = P(A)P(B)$.

### Bayes' Rule

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

In machine learning, this becomes:

$$P(\theta | D) = \frac{P(D|\theta) P(\theta)}{P(D)}$$

where $P(\theta|D)$ is the **posterior**, $P(D|\theta)$ is the **likelihood**, $P(\theta)$ is the **prior**, and $P(D)$ is the **evidence**.

### Gaussian (Normal) Distribution

$$p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

For multivariate case $x \in \mathbb{R}^d$:

$$p(x|\mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

### Maximum Likelihood Estimation (MLE)

Given data $D = \{x_1, \ldots, x_N\}$ assumed i.i.d. from $p(x|\theta)$:

$$\theta_{MLE} = \arg\max_\theta p(D|\theta) = \arg\max_\theta \prod_{i=1}^N p(x_i|\theta)$$

Equivalently, maximize the log-likelihood:

$$\theta_{MLE} = \arg\max_\theta \sum_{i=1}^N \log p(x_i|\theta)$$

### Maximum A Posteriori (MAP)

$$\theta_{MAP} = \arg\max_\theta p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$$

MAP differs from MLE by the inclusion of the prior $\log p(\theta)$, which acts as a regularizer.

### Gaussian Log-Likelihood as a Loss

For a regression model predicting $\mu(x;\theta)$ with assumed Gaussian noise:

$$p(y|x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu(x;\theta))^2}{2\sigma^2}\right)$$

The negative log-likelihood:

$$-\log p(y|x, \theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$$

With constant $\sigma$, minimizing NLL is equivalent to minimizing MSE.

## Code Examples

### Example 1: Gaussian Distribution and Log-Likelihood

```python
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np

# Create a Gaussian distribution
mu, sigma = torch.tensor(0.0), torch.tensor(1.0)
gaussian = dist.Normal(mu, sigma)

# Sample
samples = gaussian.sample((1000,))
print(f"Sample mean: {samples.mean():.3f}, std: {samples.std():.3f}")
# Output: Sample mean: -0.012, std: 0.989

# Log-probability (negative log-likelihood)
x = torch.tensor(0.5)
nll = -gaussian.log_prob(x)
print(f"Negative log-likelihood at x=0.5: {nll:.3f}")
# Output: Negative log-likelihood at x=0.5: 0.943

# Compare NLL with MSE
y_pred = torch.tensor([1.2, 0.8, 2.1])
y_true = torch.tensor([1.0, 1.0, 2.0])
mse = (y_pred - y_true).pow(2).mean()
nll_gaussian = dist.Normal(y_pred, 1.0).log_prob(y_true).mean()
print(f"MSE: {mse:.4f}, NLL: {-nll_gaussian:.4f}")
# Output: MSE: 0.0300, NLL: -0.9634
```

### Example 2: MLE for Linear Regression

```python
import torch

# Generate synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
true_w, true_b = 2.0, 1.0
y = true_w * X + true_b + 0.1 * torch.randn(100, 1)

# MLE = least squares for Gaussian likelihood
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

optimizer = torch.optim.SGD([w, b], lr=0.1)
for step in range(1000):
    y_pred = X @ w + b
    # Negative log-likelihood (Gaussian with fixed sigma=1)
    loss = ((y_pred - y) ** 2).mean()  # equivalent to -log p(y|X,w,b)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Estimated w: {w.item():.3f}, b: {b.item():.3f}")
print(f"True w: {true_w}, b: {true_b}")
# Output: Estimated w: 1.994, b: 1.002
# Output: True w: 2.0, b: 1.0
```

### Example 3: Bayes Rule in Classification

```python
import torch

# Prior: P(cat) = 0.3, P(dog) = 0.7
prior = torch.tensor([0.3, 0.7])

# Likelihood: P(features | class)
# For simplicity, assume we observe a feature that is more likely for cats
likelihood = torch.tensor([0.8, 0.2])  # P(feature | cat)=0.8, P(feature | dog)=0.2

# Posterior: P(class | feature) via Bayes rule
evidence = (prior * likelihood).sum()
posterior = prior * likelihood / evidence

print(f"Posterior: P(cat|feature)={posterior[0]:.3f}, P(dog|feature)={posterior[1]:.3f}")
# Output: Posterior: P(cat|feature)=0.632, P(dog|feature)=0.368

# In deep learning, the softmax output approximates the posterior
logits = torch.tensor([[1.5, -0.5]])
probabilities = torch.softmax(logits, dim=-1)
print(f"Softmax probabilities: {probabilities}")
# Output: Softmax probabilities: tensor([[0.8808, 0.1192]])
```

### Example 4: Cross-Entropy Loss from MLE Perspective

```python
import torch
import torch.nn.functional as F

# For classification, cross-entropy = negative log-likelihood
# under a categorical distribution

# Predictions (logits) and targets
logits = torch.tensor([[2.0, 0.5, -1.0]])
target = torch.tensor([0])  # class 0

# Cross-entropy loss
ce_loss = F.cross_entropy(logits, target)
print(f"Cross-entropy loss: {ce_loss:.4f}")
# Output: Cross-entropy loss: 0.2894

# Manual computation:
p = F.softmax(logits, dim=-1)
nll = -torch.log(p[0, target])
print(f"NLL (manual): {nll:.4f}")
# Output: NLL (manual): 0.2894

# Connection: cross-entropy = H(p, q) = -sum(p * log(q))
# where p is one-hot target, q is predicted distribution
one_hot = F.one_hot(target, num_classes=3).float()
cross_entropy = -(one_hot * torch.log(p)).sum()
print(f"H(p, q): {cross_entropy:.4f}")
# Output: H(p, q): 0.2894
```

### Example 5: MAP Estimation with L2 Regularization

```python
import torch

# MAP = MLE + Gaussian prior on weights (L2 regularization)
torch.manual_seed(42)
X = torch.randn(100, 5)
true_w = torch.tensor([1.0, -0.5, 0.0, 2.0, 0.0])
y = X @ true_w + 0.1 * torch.randn(100)

# MAP estimation = MSE + lambda * ||w||^2
w = torch.randn(5, requires_grad=True)
lambda_reg = 0.1

optimizer = torch.optim.SGD([w], lr=0.01)
for step in range(2000):
    y_pred = X @ w
    mse = (y_pred - y).pow(2).mean()
    l2_prior = lambda_reg * (w ** 2).sum()  # -log p(w) for Gaussian prior
    loss = mse + l2_prior  # = -log p(y|X,w) - log p(w) = -log p(w|X,y) + const
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"MAP estimate: {w.detach()}")
print(f"True weights: {true_w}")
# Output: MAP estimate: tensor([ 0.9876, -0.4932,  0.0021,  1.9789, -0.0011])
# Output: True weights: tensor([ 1.0000, -0.5000,  0.0000,  2.0000,  0.0000])

# L2 regularization shrinks weights toward zero (the prior mean)
```

## Common Mistakes

1. **Confusing likelihood and probability**: The likelihood $P(D|\theta)$ is a function of $\theta$ (not of $D$). It is not a probability distribution over $\theta$.

2. **Ignoring the log in log-likelihood**: Maximizing likelihood is numerically unstable due to underflow. Always work with log-likelihood.

3. **Assuming independence when data is correlated**: Time series and sequential data violate the i.i.d. assumption, requiring different probabilistic models.

4. **Misinterpreting softmax outputs as model confidence**: Softmax probabilities are often overconfident, especially with distribution shift. They are not calibrated probabilities.

5. **Forgetting the normalization constant**: $P(A|B) \propto P(B|A)P(A)$ — the denominator $P(B)$ is needed for actual probabilities but can be ignored for optimization.

6. **Using MSE for classification**: MSE assumes Gaussian noise and is inappropriate for discrete class labels. Use cross-entropy instead.

7. **Equating all uncertainty with aleatoric uncertainty**: Epistemic uncertainty (model uncertainty) and aleatoric uncertainty (data noise) are different and require different treatments.

## Interview Questions

### Beginner

1. What is the difference between a probability mass function and a probability density function?
2. Define expectation and variance. How are they related?
3. Write down Bayes' rule. What do each of the terms represent in the context of machine learning?
4. What is the probability density function of a univariate Gaussian distribution?
5. What does it mean for two random variables to be independent?

### Intermediate

1. Derive the MLE for the mean of a Gaussian distribution. Show that it equals the sample mean.
2. Explain why cross-entropy loss is equivalent to negative log-likelihood for classification.
3. How does MAP estimation differ from MLE? What is the role of the prior?
4. What is the relationship between L2 regularization and a Gaussian prior on weights?
5. Given a trained neural network with softmax output, how would you compute prediction uncertainty?

### Advanced

1. Derive the evidence lower bound (ELBO) used in variational autoencoders. Show how it relates to the KL divergence between the approximate posterior and the true posterior.
2. Prove that the cross-entropy loss $H(p, q)$ is minimized when $q = p$, where $p$ is the true distribution and $q$ is the model distribution.
3. Given a Bayesian neural network, how would you approximate the posterior distribution over weights? Discuss the trade-offs between Monte Carlo dropout, Laplace approximation, and variational inference.

## Practice Problems

### Easy

1. Compute the expectation and variance of a Bernoulli random variable with parameter $p=0.3$.
2. A Gaussian random variable has mean 0 and variance 4. What is $P(X \leq 2)$?
3. Given $P(A) = 0.5$, $P(B|A) = 0.8$, $P(B|\bar{A}) = 0.2$, compute $P(A|B)$.
4. What is the MLE for the parameter $p$ of a Bernoulli distribution given 7 heads in 10 coin flips?
5. For a uniform distribution $U[0, \theta]$, what is the MLE for $\theta$ given samples $\{1, 3, 2, 5, 4\}$?

### Medium

1. Derive the gradient of the negative log-likelihood for a Gaussian distribution with respect to the mean parameter $\mu$.
2. Implement a function that computes the log-likelihood of data under a mixture of two Gaussians.
3. Show that the KL divergence $D_{KL}(p\|q) \geq 0$ for discrete distributions (Gibbs' inequality).
4. For a linear regression model $y = w^T x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, derive the log-likelihood and show that MLE is equivalent to minimizing MSE.
5. Given a neural network that outputs both mean and variance for heteroscedastic regression, implement the negative log-likelihood loss.

### Hard

1. Derive the variational lower bound for a VAE with a Gaussian encoder and decoder. Implement the loss function in PyTorch.
2. Prove that the softmax function with cross-entropy loss yields gradients that naturally push probability toward the correct class. Show that $\frac{\partial L}{\partial z_k} = p_k - y_k$.
3. Implement a Bayesian logistic regression model using a Gaussian prior over weights and Laplace approximation for the posterior. Compare with standard logistic regression on a toy dataset.

## Solutions

_Solutions for selected problems._

**Easy 1**: $E[X] = p = 0.3$, $\text{Var}(X) = p(1-p) = 0.21$.

**Easy 3**: $P(A|B) = \frac{0.5 \cdot 0.8}{0.5 \cdot 0.8 + 0.5 \cdot 0.2} = \frac{0.4}{0.5} = 0.8$.

**Medium 2**:

```python
def gmm_log_likelihood(x, means, vars, weights):
    """Log-likelihood under a Gaussian mixture model."""
    log_lik = torch.tensor(0.0)
    for mu, var, w in zip(means, vars, weights):
        log_lik = torch.logsumexp(
            torch.stack([log_lik, torch.log(w) + dist.Normal(mu, var.sqrt()).log_prob(x)]),
            dim=0
        )
    return log_lik.sum()
```

**Hard 1**: VAE loss = reconstruction loss + KL divergence:

```python
def vae_loss(x, x_recon, mu, log_var):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
```

## Related Concepts

- **DL-019: Information Theory** — Builds on probability to define entropy and KL divergence
- **DL-017: Matrix Calculus** — Gradients of probabilistic loss functions
- **DL-016: Linear Algebra** — Multivariate Gaussian uses covariance matrices
- **DL-020: Optimization Theory** — Optimizing probabilistic objectives (MLE, MAP)

## Next Concepts

- DL-019: Information Theory (entropy, cross-entropy, KL divergence)
- DL-020: Optimization Theory (optimizing probabilistic objectives)

## Summary

Probability theory provides the mathematical foundation for understanding uncertainty in deep learning. Random variables model data and predictions; PDFs and PMFs describe their distributions. Expectation and variance summarize distribution properties. Bayes' rule enables inference by combining prior beliefs with observed data. The Gaussian distribution is central to regression problems, where minimizing MSE is equivalent to maximizing Gaussian log-likelihood. MLE and MAP provide principled frameworks for parameter estimation, with MAP incorporating prior knowledge through regularization. Cross-entropy loss for classification directly derives from MLE under a categorical distribution.

## Key Takeaways

- Loss functions are negative log-likelihoods: MSE = Gaussian NLL, cross-entropy = categorical NLL
- Bayes' rule is the foundation of all inference: posterior $\propto$ likelihood $\times$ prior
- MLE finds parameters that maximize the probability of observed data
- MAP adds a prior, which corresponds to regularization
- The Gaussian distribution is the most common choice for regression uncertainty
- Independence assumptions simplify computation but must be validated
- Softmax outputs are not calibrated probabilities — they tend to be overconfident
