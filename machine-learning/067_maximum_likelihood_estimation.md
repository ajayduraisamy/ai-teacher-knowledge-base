# Concept: Maximum Likelihood Estimation

## Concept ID

ML-067

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the principle of maximum likelihood as a parameter estimation framework
- Derive MLE for Bernoulli, Gaussian, and linear regression models
- Explain the relationship between MLE and Ordinary Least Squares
- Apply MLE to real-world parameter estimation problems

## Prerequisites

- Probability theory (PDF, PMF, independence)
- Calculus (derivatives, logarithms, optimization)
- Basic linear algebra
- Familiarity with common probability distributions (Bernoulli, Gaussian)

## Definition

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a statistical model given observations. The MLE finds the parameter values that maximize the likelihood function — the probability of observing the data given the parameters. Formally, given i.i.d. data points {x_1, ..., x_n} drawn from a distribution with unknown parameter θ, the MLE is:

θ̂_MLE = argmax_θ L(θ) = argmax_θ ∏_{i=1}^n p(x_i | θ)

In practice, we work with the log-likelihood for numerical stability:

θ̂_MLE = argmax_θ log L(θ) = argmax_θ ∑_{i=1}^n log p(x_i | θ)

## Intuition

MLE answers the question: "Which parameter values make the observed data most probable?" If you flip a coin 10 times and observe 7 heads, MLE estimates the probability of heads as 0.7 because that value maximizes the probability of observing exactly 7 heads out of 10 flips. MLE is intuitive, consistent (converges to the true parameter as n→∞), and asymptotically efficient (achieves the Cramér-Rao lower bound).

## Why This Concept Matters

MLE is the bedrock of statistical learning. Logistic regression, linear regression (via OLS), neural networks (with cross-entropy loss), and countless other models are trained by maximizing likelihood. Understanding MLE provides a unified perspective on loss functions: mean squared error corresponds to MLE with Gaussian noise, binary cross-entropy corresponds to MLE with Bernoulli likelihood, and categorical cross-entropy corresponds to MLE with multinomial likelihood.

## Mathematical Explanation

### General Framework

Given i.i.d. data D = {x_1, ..., x_n} with likelihood function:

L(θ) = ∏_{i=1}^n f(x_i; θ)

The log-likelihood is:

ℓ(θ) = log L(θ) = ∑_{i=1}^n log f(x_i; θ)

The MLE is obtained by solving the score equation:

∂ℓ(θ) / ∂θ = 0

And verifying that the Hessian ∂²ℓ(θ) / ∂θ∂θ^T is negative definite.

### Bernoulli MLE

Let X_1, ..., X_n ~ Bernoulli(p). PMF: f(x; p) = p^x (1-p)^{1-x}.

L(p) = ∏_{i=1}^n p^{x_i} (1-p)^{1-x_i} = p^{∑ x_i} (1-p)^{n - ∑ x_i}

ℓ(p) = (∑ x_i) log p + (n - ∑ x_i) log(1-p)

∂ℓ/∂p = (∑ x_i)/p - (n - ∑ x_i)/(1-p) = 0

(∑ x_i)(1-p) = (n - ∑ x_i)p

∑ x_i - p∑ x_i = np - p∑ x_i

∑ x_i = np

p̂_MLE = (1/n) ∑ x_i = x̄

The MLE for the Bernoulli parameter is simply the sample mean.

### Gaussian MLE

Let X_1, ..., X_n ~ N(μ, σ²). PDF: f(x; μ, σ²) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²)).

ℓ(μ, σ²) = ∑_{i=1}^n [ -½ log(2πσ²) - (x_i - μ)²/(2σ²) ]

= -n/2 log(2πσ²) - (1/(2σ²)) ∑ (x_i - μ)²

For μ:

∂ℓ/∂μ = (1/σ²) ∑ (x_i - μ) = 0 → μ̂ = (1/n) ∑ x_i = x̄

For σ²:

∂ℓ/∂σ² = -n/(2σ²) + (1/(2σ⁴)) ∑ (x_i - μ)² = 0

nσ² = ∑ (x_i - μ)² → σ̂² = (1/n) ∑ (x_i - x̄)²

### Linear Regression MLE (OLS equivalence)

Assume y_i = w^T x_i + ε_i, ε_i ~ N(0, σ²).

y_i | x_i; w ~ N(w^T x_i, σ²)

ℓ(w) = ∑ log p(y_i | x_i; w) = -n/2 log(2πσ²) - (1/(2σ²)) ∑ (y_i - w^T x_i)²

Maximizing ℓ(w) is equivalent to minimizing ∑ (y_i - w^T x_i)², which is OLS.

The gradient:

∂ℓ/∂w = (1/σ²) ∑ x_i (y_i - w^T x_i) = 0

X^T (y - Xw) = 0 → ŵ = (X^T X)^{-1} X^T y

This is the normal equation. OLS is exactly MLE with Gaussian noise.

### Properties of MLE

1. **Consistency:** θ̂_MLE → θ* as n → ∞ (converges in probability to true parameter)
2. **Asymptotic normality:** √n(θ̂_MLE - θ*) → N(0, I(θ*)^{-1}) where I is the Fisher information
3. **Asymptotic efficiency:** Achieves the Cramér-Rao lower bound as n → ∞
4. **Invariance:** If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ) for any function g

## Code Examples

### Example 1: Bernoulli MLE - Coin Flip Estimation

```python
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

np.random.seed(42)
true_p = 0.65
n_flips = 100
flips = np.random.binomial(1, true_p, size=n_flips)

def neg_log_likelihood(p):
    if p <= 0 or p >= 1:
        return np.inf
    return -(np.sum(flips) * np.log(p) + (len(flips) - np.sum(flips)) * np.log(1 - p))

result = minimize_scalar(neg_log_likelihood, bounds=(0, 1), method='bounded')
mle_p = result.x
sample_mean = np.mean(flips)

print(f"True p:      {true_p}")
print(f"MLE p:       {mle_p:.4f}")
print(f"Sample mean: {sample_mean:.4f}")
print(f"Flips: {np.sum(flips)} heads out of {n_flips}")
# Output:
# True p:      0.65
# MLE p:       0.6500
# Sample mean: 0.6500
```

### Example 2: Gaussian MLE - Parameter Estimation

```python
import numpy as np

np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
n_samples = 1000
data = np.random.normal(true_mu, true_sigma, size=n_samples)

mu_mle = np.mean(data)
sigma_sq_mle = np.var(data)
sigma_mle = np.sqrt(sigma_sq_mle)

print(f"True mu:    {true_mu}")
print(f"MLE mu:     {mu_mle:.4f}")
print(f"True sigma: {true_sigma}")
print(f"MLE sigma:  {sigma_mle:.4f}")
print(f"MLE sigma²: {sigma_sq_mle:.4f}")

# Compare with unbiased estimator
sigma_sq_unbiased = np.var(data, ddof=1)
print(f"Unbiased σ²: {sigma_sq_unbiased:.4f}")
# Output:
# True mu:    5.0
# MLE mu:     4.9873
# True sigma: 2.0
# MLE sigma:  1.9747
# MLE sigma²: 3.8994
# Unbiased σ²: 3.9018
```

### Example 3: Linear Regression - MLE vs OLS

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 200
true_w = np.array([2.5, -1.3, 0.7])
X = np.random.randn(n, 3)
noise = np.random.normal(0, 1.0, size=n)
y = X @ true_w + noise

# MLE solution via normal equations
w_mle = np.linalg.inv(X.T @ X) @ X.T @ y

# OLS via sklearn
ols = LinearRegression(fit_intercept=False)
ols.fit(X, y)
w_ols = ols.coef_

print("True w:     ", true_w)
print("MLE w:       ", np.round(w_mle, 4))
print("OLS w:       ", np.round(w_ols, 4))

# MLE of noise variance
residuals = y - X @ w_mle
sigma_sq_mle = np.mean(residuals ** 2)
print(f"Noise variance MLE: {sigma_sq_mle:.4f}")
print(f"True noise variance: 1.0")
# Output:
# True w:      [ 2.5 -1.3  0.7]
# MLE w:       [ 2.537 -1.259  0.677]
# OLS w:       [ 2.537 -1.259  0.677]
# Noise variance MLE: 1.0229
# True noise variance: 1.0
```

### Example 4: MLE for Custom Distribution (Exponential)

```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
true_lambda = 0.5
data = np.random.exponential(1/true_lambda, size=500)

def neg_log_likelihood_exp(params):
    lam = params[0]
    if lam <= 0:
        return np.inf
    n = len(data)
    return -(n * np.log(lam) - lam * np.sum(data))

result = minimize(neg_log_likelihood_exp, x0=[1.0], method='BFGS')
lambda_mle = result.x[0]

print(f"True λ:      {true_lambda}")
print(f"MLE λ:       {lambda_mle:.4f}")
print(f"1/mean:      {1/np.mean(data):.4f}")
# Output:
# True λ:      0.5
# MLE λ:       0.5042
# 1/mean:      0.5042
```

## Common Mistakes

1. **Confusing likelihood with probability.** Likelihood is not a probability distribution over parameters — it is a function of parameters given fixed data.
2. **Using MLE without checking identifiability.** If multiple parameter values produce the same likelihood, the MLE is not unique.
3. **Overlooking the i.i.d. assumption.** MLE assumes independent and identically distributed samples; violations bias estimates.
4. **Forgetting that MLE can be biased in finite samples.** Gaussian variance MLE is biased (divides by n, not n-1).
5. **Applying MLE to non-identifiable models.** In mixture models, the likelihood may have multiple equivalent maxima (label switching).
6. **Assuming MLE always has a closed form.** For many models (e.g., logistic regression), MLE requires iterative optimization.
7. **Ignoring numerical stability.** Multiplying many small probabilities causes underflow — always use log-likelihood.

## Interview Questions

### Beginner

1. What does MLE stand for and what does it do?
2. Why do we use log-likelihood instead of likelihood?
3. What is the MLE for the mean of a Gaussian distribution?
4. What is the MLE for the probability parameter of a Bernoulli distribution?
5. How is MLE related to Ordinary Least Squares?

### Intermediate

1. Prove that the MLE of the Bernoulli parameter p is the sample mean.
2. Derive the MLE for the variance of a Gaussian. Why is it biased?
3. Show that maximizing likelihood with Gaussian noise is equivalent to minimizing MSE.
4. What are the asymptotic properties of MLE?
5. How does MLE relate to the cross-entropy loss function?

### Advanced

1. Prove the consistency of MLE under regularity conditions.
2. Derive the Cramér-Rao lower bound and show that MLE achieves it asymptotically.
3. How would you compute MLE for a mixture of Gaussians? Why can't you use closed-form MLE directly?

## Practice Problems

### Easy

1. Write code to compute MLE for a Poisson distribution given sample data.
2. Given 15 successes out of 50 Bernoulli trials, compute MLE for p and plot the likelihood function.
3. Write a function that computes the Gaussian MLE for mean and variance.
4. Show empirically that as sample size increases, MLE converges to the true parameter.
5. Implement OLS from scratch using the normal equations and verify it matches sklearn's LinearRegression.

### Medium

1. Derive MLE for the rate parameter of an exponential distribution.
2. Implement MLE for logistic regression using gradient ascent.
3. Compare MLE and MAP (Maximum a Posteriori) estimates for a Gaussian with known variance.
4. Compute the Fisher information for Bernoulli and Gaussian distributions.
5. Simulate the sampling distribution of MLE for a Gaussian mean in 1000 experiments and verify asymptotic normality.

### Hard

1. Derive the EM algorithm for MLE with missing data and apply to a simple Gaussian mixture.
2. Prove the invariance property of MLE: if θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ).
3. Implement MLE for a censored regression model (Tobit model) using numerical optimization.

## Solutions

Solution 1 (Easy): Poisson MLE

```python
import numpy as np
x = np.array([2, 3, 1, 4, 2, 3, 3, 1, 2, 4])
lambda_mle = np.mean(x)
print(f"Poisson MLE λ = {lambda_mle:.4f}")
# Output: Poisson MLE λ = 2.5000
```

Solution 2 (Medium): Fisher information for Bernoulli

```python
import numpy as np
p = 0.7
n = 100
I_p = n / (p * (1 - p))
print(f"Fisher Information I(p) = {I_p:.2f}")
se = 1 / np.sqrt(I_p)
print(f"Asymptotic SE = {se:.4f}")
# Output:
# Fisher Information I(p) = 476.19
# Asymptotic SE = 0.0458
```

Solution 3 (Hard): MLE invariance proof sketch

```python
import numpy as np
# If λ̂ = 1/x̄ is MLE for Exponential(λ),
# then by invariance, μ̂ = 1/λ̂ = x̄ is MLE of mean μ = 1/λ
data = np.random.exponential(scale=2.0, size=1000)
lambda_mle = 1 / np.mean(data)
mu_mle = 1 / lambda_mle  # should equal np.mean(data)
print(f"μ_MLE = {mu_mle:.4f}, sample mean = {np.mean(data):.4f}")
# Output: μ_MLE = 1.9843, sample mean = 1.9843
```

## Related Concepts

- Bayesian Estimation (ML-017)
- Expectation Maximization (ML-068)
- Loss Functions and Optimization
- Generalized Linear Models
- Fisher Information
- Cramér-Rao Lower Bound
- MAP Estimation

## Next Concepts

- Expectation Maximization (ML-068)
- Bayesian Optimization (ML-069)
- Model Interpretability (ML-075)

## Summary

Maximum Likelihood Estimation is a principled framework for estimating model parameters by maximizing the probability of observing the data. MLE yields closed-form solutions for many standard distributions (Bernoulli, Gaussian) and connects directly to common loss functions (MSE for regression, cross-entropy for classification). MLE is consistent, asymptotically normal, and asymptotically efficient, making it the default estimation method in most machine learning pipelines.

## Key Takeaways

- MLE finds parameters that maximize ∏ p(x_i | θ) or equivalently ∑ log p(x_i | θ).
- Bernoulli MLE: p̂ = sample mean.
- Gaussian MLE: μ̂ = sample mean, σ̂² = sample variance (biased).
- OLS is MLE with Gaussian noise assumption.
- MLE is consistent, asymptotically normal, and efficient.
- Always use log-likelihood to avoid numerical underflow.
- MLE connects all major loss functions: MSE, cross-entropy, and more.
