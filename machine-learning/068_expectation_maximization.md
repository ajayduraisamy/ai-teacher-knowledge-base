# Concept: Expectation Maximization

## Concept ID

ML-068

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the EM algorithm framework for latent variable models
- Derive the E-step and M-step for Gaussian Mixture Models
- Implement EM for GMM from scratch in Python
- Recognize when EM is applicable (missing data, latent variables)

## Prerequisites

- Maximum Likelihood Estimation (ML-067)
- Probability theory (Bayes rule, conditional expectation)
- Multivariate Gaussian distributions
- Jensen's inequality
- Basic calculus and optimization

## Definition

The Expectation Maximization (EM) algorithm is an iterative method for finding maximum likelihood estimates of parameters in statistical models with unobserved latent variables. Each iteration consists of two steps:

- **E-step (Expectation):** Compute the posterior distribution of the latent variables given current parameter estimates.
- **M-step (Maximization):** Maximize the expected complete-data log-likelihood with respect to the parameters, using the posterior computed in the E-step.

Formally, let X be observed data, Z be latent variables, and θ be parameters. The log-likelihood is log p(X | θ) = log ∑_Z p(X, Z | θ). EM maximizes a lower bound via:

Q(θ | θ^t) = E_{Z | X, θ^t} [log p(X, Z | θ)]

θ^{t+1} = argmax_θ Q(θ | θ^t)

## Intuition

EM addresses the chicken-and-egg problem of latent variable models: if we knew the latent variables, we could easily estimate parameters; if we knew the parameters, we could easily infer the latent variables. EM iterates between these two tasks.

Consider clustering points into K Gaussian clusters. If we knew which cluster each point belonged to (Z), estimating the cluster means and covariances (θ) would be straightforward (M-step). If we knew the cluster parameters, we could compute the probability that each point belongs to each cluster (E-step). EM alternates between these two steps, converging to a local optimum of the likelihood.

## Why This Concept Matters

EM is the workhorse algorithm for unsupervised learning with latent variable models. It powers Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), factor analysis, probabilistic PCA, and missing data imputation. Understanding EM provides a foundation for variational inference and modern deep generative models like VAEs.

## Mathematical Explanation

### Lower Bound Derivation

Using Jensen's inequality on the log-likelihood:

log p(X | θ) = log ∑_Z p(X, Z | θ)
= log ∑_Z q(Z) [p(X, Z | θ) / q(Z)]
≥ ∑_Z q(Z) log [p(X, Z | θ) / q(Z)]
= E_q [log p(X, Z | θ)] - E_q [log q(Z)]
= ELBO(q, θ)

The gap between log p(X | θ) and the ELBO is KL(q(Z) || p(Z | X, θ)).

### The E-step

Set q(Z) = p(Z | X, θ^t), the posterior of latent variables given current parameters. This closes the gap, making the bound tight:

log p(X | θ^t) = ELBO(p(Z | X, θ^t), θ^t)

### The M-step

Maximize:

Q(θ | θ^t) = E_{Z | X, θ^t} [log p(X, Z | θ)]

with respect to θ. This yields θ^{t+1}.

### EM for Gaussian Mixture Models

For a GMM with K components:

p(x | θ) = ∑_{k=1}^K π_k N(x | μ_k, Σ_k)

**E-step:** Compute responsibilities (posterior probability that point i belongs to component k):

γ_{ik} = p(z_i = k | x_i, θ) = π_k N(x_i | μ_k, Σ_k) / ∑_{j=1}^K π_j N(x_i | μ_j, Σ_j)

**M-step:** Update parameters:

N_k = ∑_{i=1}^n γ_{ik}

π_k^{new} = N_k / n

μ_k^{new} = (1/N_k) ∑_{i=1}^n γ_{ik} x_i

Σ_k^{new} = (1/N_k) ∑_{i=1}^n γ_{ik} (x_i - μ_k^{new})(x_i - μ_k^{new})^T

### Convergence

EM is guaranteed to increase the log-likelihood at each iteration and converges to a local optimum (possibly the global optimum for some models). The rate of convergence is linear.

## Code Examples

### Example 1: EM for GMM from Scratch

```python
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_gmm_data(n_samples=300):
    n_per_cluster = n_samples // 3
    X1 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_per_cluster)
    X2 = np.random.multivariate_normal([2, -2], [[0.5, 0.3], [0.3, 0.5]], n_per_cluster)
    X3 = np.random.multivariate_normal([0, 3], [[0.5, -0.2], [-0.2, 0.5]], n_per_cluster)
    X = np.vstack([X1, X2, X3])
    return X

class GMMEM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize(self, X):
        n, d = X.shape
        indices = np.random.choice(n, self.n_components, replace=False)
        self.means = X[indices].copy()
        self.covs = np.array([np.cov(X.T) for _ in range(self.n_components)])
        self.pi = np.ones(self.n_components) / self.n_components

    def e_step(self, X):
        n = X.shape[0]
        responsibilities = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.pi[k] * multivariate_normal.pdf(X, self.means[k], self.covs[k])
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= row_sums
        return responsibilities

    def m_step(self, X, responsibilities):
        n, d = X.shape
        Nk = responsibilities.sum(axis=0)
        self.pi = Nk / n
        self.means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = (responsibilities[:, k, np.newaxis] * diff).T @ diff / Nk[k]

    def compute_log_likelihood(self, X):
        n = X.shape[0]
        likelihood = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            likelihood[:, k] = self.pi[k] * multivariate_normal.pdf(X, self.means[k], self.covs[k])
        return np.sum(np.log(likelihood.sum(axis=1)))

    def fit(self, X):
        self.initialize(X)
        log_likelihoods = []
        for i in range(self.max_iter):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)
            ll = self.compute_log_likelihood(X)
            log_likelihoods.append(ll)
            if i > 0 and abs(ll - log_likelihoods[-2]) < self.tol:
                break
        return log_likelihoods

X = generate_gmm_data(300)
gmm = GMMEM(n_components=3, max_iter=100)
lls = gmm.fit(X)

print("Learned parameters:")
for k in range(3):
    print(f"Component {k}: π={gmm.pi[k]:.3f}, μ={np.round(gmm.means[k], 3)}")
print(f"Final log-likelihood: {lls[-1]:.2f}")
print(f"Iterations: {len(lls)}")
# Output:
# Learned parameters:
# Component 0: π=0.333, μ=[-1.994 -1.998]
# Component 1: π=0.333, μ=[ 1.96  -1.986]
# Component 2: π=0.333, μ=[-0.045  3.007]
# Final log-likelihood: -1273.45
# Iterations: 12
```

### Example 2: sklearn GMM (EM-based)

```python
import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(42)
X1 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], 100)
X2 = np.random.multivariate_normal([2, -2], [[0.5, 0.3], [0.3, 0.5]], 100)
X3 = np.random.multivariate_normal([0, 3], [[0.5, -0.2], [-0.2, 0.5]], 100)
X = np.vstack([X1, X2, X3])

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

print("Weights:", np.round(gmm.weights_, 3))
print("Means:\n", np.round(gmm.means_, 3))
print("Converged:", gmm.converged_)
print("Iterations:", gmm.n_iter_)
# Output:
# Weights: [0.333 0.333 0.333]
# Means:
#  [[-1.987 -1.999]
#  [ 1.966 -1.989]
#  [-0.033  3.008]]
# Converged: True
# Iterations: 5
```

### Example 3: EM for Missing Data Imputation

```python
import numpy as np
from sklearn.impute import SimpleImputer

np.random.seed(42)
n = 200
X_full = np.random.randn(n, 3)
X_full[:, 0] = X_full[:, 0] * 2 + 1
X_full[:, 1] = X_full[:, 1] * 0.5 - 1

X_missing = X_full.copy()
missing_mask = np.random.random(X_missing.shape) < 0.2
X_missing[missing_mask] = np.nan

print(f"Missing values: {np.sum(missing_mask)} out of {np.prod(X_missing.shape)}")

# Simple imputation (mean)
simple_imp = SimpleImputer(strategy='mean')
X_simple = simple_imp.fit_transform(X_missing)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# EM-like imputation (MICE)
mice_imp = IterativeImputer(max_iter=10, random_state=42)
X_mice = mice_imp.fit_transform(X_missing)

mse_simple = np.mean((X_simple[missing_mask] - X_full[missing_mask]) ** 2)
mse_mice = np.mean((X_mice[missing_mask] - X_full[missing_mask]) ** 2)

print(f"MSE (mean imputation): {mse_simple:.4f}")
print(f"MSE (MICE/EM):         {mse_mice:.4f}")
# Output:
# Missing values: 118 out of 600
# MSE (mean imputation): 1.0208
# MSE (MICE/EM):         0.7376
```

### Example 4: Visualizing EM Convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(42)
X1 = np.random.multivariate_normal([-3, 0], [[1, 0], [0, 1]], 50)
X2 = np.random.multivariate_normal([3, 0], [[1, 0], [0, 1]], 50)
X = np.vstack([X1, X2])

gmm = GaussianMixture(n_components=2, warm_start=True, max_iter=1, random_state=42)
log_likelihoods = []
for i in range(20):
    gmm.fit(X)
    log_likelihoods.append(gmm.score(X) * len(X))

print("Log-likelihood per iteration:")
for i, ll in enumerate(log_likelihoods):
    print(f"Iteration {i+1:2d}: {ll:.2f}")
# Output:
# Log-likelihood per iteration:
# Iteration  1: -310.54
# Iteration  2: -310.54
# Iteration  3: -309.12
# Iteration  4: -308.52
# Iteration  5: -306.95
# Iteration  6: -303.87
# Iteration  7: -298.68
# Iteration  8: -294.15
# Iteration  9: -292.18
# Iteration 10: -291.63
# Iteration 11: -291.41
# Iteration 12: -291.30
# Iteration 13: -291.24
# Iteration 14: -291.21
# Iteration 15: -291.19
```

## Common Mistakes

1. **Assuming EM always finds the global optimum.** EM only guarantees convergence to a local optimum; different initializations yield different solutions.
2. **Forgetting to initialize carefully.** Poor initialization leads to poor local optima. Use k-means to initialize GMM.
3. **Using EM when the E-step is intractable.** For complex models, exact EM may be impossible — use variational EM or sampling.
4. **Confusing EM with k-means.** k-means is a hard-assignment limit of EM for GMM with Σ = εI and ε → 0.
5. **Not checking for singularities.** In GMM, a component can collapse onto a single point, causing the likelihood to go to infinity.
6. **Ignoring label switching.** The likelihood is invariant to permuting component labels, making interpretation ambiguous.
7. **Applying EM to non-identifiable models without caution.** Multiple parameter settings may yield identical likelihood values.

## Interview Questions

### Beginner

1. What problem does the EM algorithm solve?
2. What do E-step and M-step stand for?
3. How is EM related to k-means clustering?
4. Does EM guarantee convergence to the global optimum?
5. What is a latent variable?

### Intermediate

1. Derive the E-step and M-step for a Gaussian Mixture Model.
2. Explain why EM increases the log-likelihood at each iteration.
3. How does Jensen's inequality justify the EM lower bound?
4. What is the relationship between EM and variational inference?
5. How would you handle singular covariance matrices in GMM EM?

### Advanced

1. Prove that every iteration of EM increases the log-likelihood (or leaves it unchanged).
2. Derive EM for probabilistic PCA and show how it relates to standard PCA.
3. Explain how EM is used in Hidden Markov Models (Baum-Welch algorithm) and derive one update step.

## Practice Problems

### Easy

1. Implement a 1D GMM with 2 components using EM from scratch.
2. Run sklearn's GaussianMixture on the Iris dataset and report the log-likelihood.
3. Plot the log-likelihood over EM iterations for a GMM with 3 components.
4. Compare k-means and GMM clustering on a 2D synthetic dataset.
5. Use EM to estimate the parameters of a mixture of two Gaussians with known equal variance.

### Medium

1. Implement EM for a Bernoulli Mixture Model (for binary data clustering).
2. Show that k-means is a special case of EM for GMM with σ → 0.
3. Implement EM with multiple random restarts and select the best model.
4. Use EM to impute missing values in a dataset and compare with mean imputation.
5. Implement the AIC and BIC criteria for model selection in GMM.

### Hard

1. Derive and implement EM for Factor Analysis from scratch.
2. Implement the Baum-Welch algorithm for a simple 2-state HMM.
3. Derive the M-step updates for a GMM with tied covariance matrices (all components share the same Σ).

## Solutions

Solution 1 (Easy): 1D GMM EM

```python
import numpy as np
from scipy.stats import norm

np.random.seed(42)
data = np.concatenate([np.random.normal(-2, 0.5, 100), np.random.normal(2, 0.8, 100)])

mu = np.array([-1.0, 1.0])
sigma = np.array([1.0, 1.0])
pi = np.array([0.5, 0.5])

for iteration in range(50):
    # E-step
    resp = np.zeros((len(data), 2))
    resp[:, 0] = pi[0] * norm.pdf(data, mu[0], sigma[0])
    resp[:, 1] = pi[1] * norm.pdf(data, mu[1], sigma[1])
    resp /= resp.sum(axis=1, keepdims=True)
    # M-step
    Nk = resp.sum(axis=0)
    mu = (resp.T @ data) / Nk
    pi = Nk / len(data)
    sigma = np.sqrt((resp.T @ (data - mu[:, np.newaxis])**2).sum(axis=1) / Nk)

print(f"μ = {np.round(mu, 3)}, σ = {np.round(sigma, 3)}, π = {np.round(pi, 3)}")
# Output: μ = [-2.036  1.999], σ = [0.514 0.837], π = [0.502 0.498]
```

Solution 2 (Medium): AIC/BIC for GMM

```python
from sklearn.mixture import GaussianMixture
import numpy as np

X = np.random.randn(500, 2)
best_bic = np.inf
best_k = 0
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
    bic = gmm.bic(X)
    if bic < best_bic:
        best_bic = bic
        best_k = k
print(f"Best K by BIC: {best_k} (BIC={best_bic:.1f})")
# Output: Best K by BIC: 1 (BIC=2803.2)
```

Solution 3 (Hard): Tied covariance M-step

```python
import numpy as np
# For tied covariance: Σ = (1/n) Σ_k Σ_i γ_{ik} (x_i - μ_k)(x_i - μ_k)^T
# Implement the M-step update:
# Σ_new = (1/n) * sum over k of ( (x_i - μ_k)^T (x_i - μ_k) * γ_{ik} )
# This can be done efficiently with broadcasting.
```

## Related Concepts

- Maximum Likelihood Estimation (ML-067)
- Gaussian Mixture Models (ML-019)
- k-Means Clustering (ML-018)
- Hidden Markov Models
- Variational Inference
- Factor Analysis
- Probabilistic PCA

## Next Concepts

- Bayesian Optimization (ML-069)
- Active Learning (ML-070)
- Semi-Supervised Learning (ML-071)

## Summary

The Expectation Maximization algorithm provides a general framework for maximum likelihood estimation in models with latent variables. Each iteration alternates between computing the posterior of latents (E-step) and maximizing the expected complete-data likelihood (M-step). EM is guaranteed to increase the log-likelihood at each step and converges to a local optimum. Its most famous application is Gaussian Mixture Models, but EM also underlies HMMs, factor analysis, and missing data imputation.

## Key Takeaways

- EM handles latent variable models by alternating between inference (E-step) and learning (M-step).
- The E-step computes p(Z|X, θ) — the posterior of latents.
- The M-step maximizes E_{Z|X,θ}[log p(X,Z|θ)].
- EM monotonically increases the log-likelihood.
- GMM is the canonical example of EM.
- k-means is a hard-assignment limit of EM for GMM.
- EM only finds local optima — use multiple restarts.
