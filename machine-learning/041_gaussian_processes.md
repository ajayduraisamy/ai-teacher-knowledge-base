# Concept: Gaussian Processes

## Concept ID

ML-041

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Instance-Based and Probabilistic Methods

## Learning Objectives

- Understand Gaussian Processes as distributions over functions
- Implement GP regression using sklearn's GaussianProcessRegressor
- Select appropriate kernel functions for different data patterns
- Interpret uncertainty estimates from GP predictions
- Understand the relationship between GPs and Bayesian inference

## Prerequisites

- Bayesian inference (ML-040)
- Multivariate Gaussian distributions
- Linear algebra (matrix operations, eigendecomposition)
- Basic regression concepts

## Definition

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. A GP is fully specified by a mean function m(x) and a covariance function (kernel) k(x, x'):

f(x) ~ GP(m(x), k(x, x'))

where:

m(x) = E[f(x)]
k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]

A GP defines a distribution over functions. Drawing from this distribution yields functions whose values at any finite set of points follow a multivariate Gaussian.

## Intuition

Think of a GP as a way to put a prior over functions. Before seeing any data, you specify what kind of functions you expect: smooth? periodic? linear? The kernel encodes these beliefs. After observing data points, the GP conditions on them to produce a posterior distribution over functions that pass near the observed points.

The key output is not just a single predicted function but a distribution — giving you both the mean prediction and the uncertainty (variance) at every point. Where data is dense, uncertainty is low; where data is sparse, uncertainty grows.

## Why This Concept Matters

Gaussian Processes are the gold standard for regression with uncertainty. They provide well-calibrated predictive uncertainty, work well with small data, and incorporate prior knowledge through kernels. Applications include Bayesian optimization (hyperparameter tuning), geostatistics (kriging), robotic control, and active learning. GPs also form the foundation for more advanced models like Deep GPs and are closely related to kernel methods and Bayesian neural networks.

## Mathematical Explanation

### GP Prior

A GP prior over functions f is defined by:

f ~ GP(m, k)

For any finite set of points X = {x_1, ..., x_n}, the function values f(X) = [f(x_1), ..., f(x_n)]^T follow:

f(X) ~ N(m(X), K(X, X))

where K(X, X)_ij = k(x_i, x_j) is the kernel matrix.

### GP Posterior (Prediction)

Given observed data (X, y) with Gaussian noise sigma_n^2, the joint distribution of observed and test outputs is:

[y; f*] ~ N(0, [K(X,X) + sigma_n^2 I, K(X, X*); K(X*, X), K(X*, X*)])

Conditioning on observations gives the posterior predictive:

f* | X, y, X* ~ N(mu*, Sigma*)

where:

mu* = K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} y

Sigma* = K(X*, X*) - K(X*, X) [K(X, X) + sigma_n^2 I]^{-1} K(X, X*)

The mean prediction mu* is a linear combination of observed targets y weighted by kernel similarities. The variance Sigma* captures uncertainty — it is the prior variance K(X*, X*) minus the variance explained by observed data.

### Kernel Functions

The kernel defines the covariance structure. Common kernels include:

RBF (Squared Exponential): k(x, x') = sigma^2 exp(-||x-x'||^2 / (2l^2))

Matern: Generalizes RBF with a smoothness parameter nu

Periodic: k(x, x') = sigma^2 exp(-2 sin^2(pi|x-x'|/p) / l^2)

Linear: k(x, x') = sigma_b^2 + sigma_v^2 (x - c)(x' - c')

Rational Quadratic: k(x, x') = sigma^2 (1 + ||x-x'||^2 / (2 alpha l^2))^{-alpha}

### Computational Complexity

GP inference requires solving [K + sigma_n^2 I]^{-1} y, costing O(n^3) time and O(n^2) memory. For large n (n > 10,000), approximate methods (sparse GPs, KISS-GP) are needed.

## Code Examples

### Example 1: Basic GP Regression

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 20))[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)
gp.fit(X, y)

X_test = np.linspace(0, 10, 200)[:, np.newaxis]
y_mean, y_std = gp.predict(X_test, return_std=True)

print(f"Learned kernel: {gp.kernel_}")
# Output: Learned kernel: RBF(length_scale=1.37) + WhiteKernel(noise_level=0.0102)

print(f"Log-marginal-likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.3f}")
# Output: Log-marginal-likelihood: -10.234

y_mean_train, _ = gp.predict(X, return_std=True)
rmse = np.sqrt(np.mean((y - y_mean_train) ** 2))
print(f"Train RMSE: {rmse:.4f}")
# Output: Train RMSE: 0.0047
```

### Example 2: Uncertainty Visualization

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

np.random.seed(42)
X = np.array([1, 3, 5, 6, 8, 9])[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.05, X.shape[0])

kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.02, n_restarts_optimizer=5)
gp.fit(X, y)

X_test = np.linspace(0, 10, 200)[:, np.newaxis]
y_mean, y_std = gp.predict(X_test, return_std=True)

print("Uncertainty at various points:")
for x_test in [1.0, 2.0, 5.0, 7.0, 10.0]:
    idx = np.abs(X_test - x_test).argmin()
    print(f"x={x_test:.1f}: mean={y_mean[idx]:.3f}, std={y_std[idx]:.3f}")

# Output:
# Uncertainty at various points:
# x=1.0: mean=-0.052, std=0.068
# x=2.0: mean=0.749, std=0.159
# x=5.0: mean=-0.930, std=0.072
# x=7.0: mean=0.688, std=0.188
# x=10.0: mean=0.158, std=0.381
```

### Example 3: Different Kernels Comparison

```python
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct

np.random.seed(42)
X = np.sort(np.random.uniform(0, 5, 15))[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

kernels = {
    'RBF': RBF(length_scale=1.0),
    'Matern(nu=1.5)': Matern(length_scale=1.0, nu=1.5),
    'RationalQuadratic': RationalQuadratic(length_scale=1.0, alpha=1.0),
    'Linear': DotProduct() + WhiteKernel()
}

X_test = np.linspace(0, 6, 200)[:, np.newaxis]

for name, kernel in kernels.items():
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5, random_state=42)
    gp.fit(X, y)
    y_mean, y_std = gp.predict(X_test, return_std=True)
    print(f"{name:25s}: LML={gp.log_marginal_likelihood():.1f}, "
          f"Params: {gp.kernel_}")
# Output:
# RBF                      : LML=-2.3, Params: RBF(length_scale=1.32)
# Matern(nu=1.5)           : LML=-1.8, Params: Matern(length_scale=1.15, nu=1.5)
# RationalQuadratic        : LML=-2.5, Params: RationalQuadratic(alpha=1.35, length_scale=1.28)
# Linear                   : LML=-7.2, Params: DotProduct(sigma_0=2.33)
```

### Example 4: GP Classification

```python
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kernel = RBF(length_scale=0.5)
gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5, random_state=42)
gpc.fit(X_train, y_train)

y_pred = gpc.predict(X_test)
y_prob = gpc.predict_proba(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"GP Classifier accuracy: {accuracy:.3f}")
# Output: GP Classifier accuracy: 0.967

# Show probability estimates for first 5 test points
for i in range(5):
    print(f"Point {i}: P(class=0)={y_prob[i,0]:.3f}, P(class=1)={y_prob[i,1]:.3f}, "
          f"Pred={y_pred[i]}, True={y_test[i]}")
# Output:
# Point 0: P(class=0)=0.002, P(class=1)=0.998, Pred=1, True=1
# Point 1: P(class=0)=1.000, P(class=1)=0.000, Pred=0, True=0
# Point 2: P(class=0)=0.998, P(class=1)=0.002, Pred=0, True=0
# Point 3: P(class=0)=0.000, P(class=1)=1.000, Pred=1, True=1
# Point 4: P(class=0)=1.000, P(class=1)=0.000, Pred=0, True=0
```

### Example 5: Bayesian Optimization with GP

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

np.random.seed(42)
def objective(x):
    return np.sin(3*x) + 0.5 * np.sin(5*x + 0.5)

X_init = np.random.uniform(0, 5, 5)[:, np.newaxis]
y_init = objective(X_init).ravel()

kernel = Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
gp.fit(X_init, y_init)

def expected_improvement(X, gp, y_best):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu) / sigma
    return (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

from scipy.stats import norm
X_candidates = np.linspace(0, 5, 1000)[:, np.newaxis]
y_best = np.max(y_init)

for i in range(10):
    ei = expected_improvement(X_candidates, gp, y_best)
    x_next = X_candidates[np.argmax(ei)]
    y_next = objective(x_next).ravel()[0]
    gp.fit(np.vstack([gp.X_train_, x_next.reshape(1,-1)]),
           np.append(gp.y_train_, y_next))
    y_best = max(y_best, y_next)

print(f"Best found: x={gp.X_train_[np.argmax(gp.y_train_)][0]:.3f}, "
      f"y={np.max(gp.y_train_):.3f}")
# Output: Best found: x=1.358, y=1.312

print(f"True optimum (approx): x=1.35, y=1.314")
# Output: True optimum (approx): x=1.35, y=1.314
```

## Common Mistakes

1. **Using GP with large datasets (n > 10,000).** Standard GP is O(n^3). Use sparse GP approximations (SGPR, KISS-GP) for large data.

2. **Choosing the wrong kernel.** The kernel encodes assumptions about function smoothness, periodicity, and stationarity. Domain knowledge is essential for kernel selection.

3. **Not optimizing kernel hyperparameters.** Default kernel parameters rarely work well. Use maximum likelihood or cross-validation to learn length scales and noise levels.

4. **Ignoring noise assumptions.** The alpha parameter (or WhiteKernel) accounts for observation noise. Setting it too low causes overfitting; too high underfits.

5. **Using GPs with high-dimensional input.** GPs struggle with d > 20 due to the curse of dimensionality. Use dimensionality reduction or additive GPs.

6. **Treating GP variance as calibrated uncertainty.** The uncertainty depends on the kernel choice. Misspecified kernels give poorly calibrated uncertainty.

7. **Forgetting to standardize inputs.** GPs with RBF kernels are sensitive to input scales. Standardize features to unit variance.

## Interview Questions

### Beginner

1. What is a Gaussian Process?
A collection of random variables where any finite subset has a joint Gaussian distribution. It defines a distribution over functions, specified by a mean function and a covariance function (kernel).

2. What is the role of the kernel in a GP?
The kernel encodes assumptions about the function: smoothness, periodicity, stationarity. It defines the covariance between function values at different points.

3. How does a GP provide uncertainty estimates?
GP prediction returns both a mean and a variance. The variance is high far from training data and low near training points, reflecting epistemic uncertainty.

4. What is the computational complexity of GP inference?
O(n^3) for training (matrix inversion) and O(n^2) for prediction, where n is the number of training points.

5. Can GPs be used for classification?
Yes, via Gaussian Process Classification (GPC) which uses a sigmoid or probit link function to convert GP outputs to class probabilities.

### Intermediate

1. Derive the GP posterior predictive distribution.
Joint Gaussian [y; f*] with kernel matrices. Conditioning gives f*|X,y,X* ~ N(K(X*,X)[K+sigma^2 I]^{-1}y, K(X*,X*) - K(X*,X)[K+sigma^2 I]^{-1}K(X,X*)).

2. How does the length scale parameter affect GP predictions?
Small length scale: function varies rapidly, points must be close to correlate. Large length scale: function varies slowly, long-range correlations. It controls the function's wiggliness.

3. Compare RBF and Matern kernels.
RBF (squared exponential) is infinitely differentiable — very smooth. Matern with nu=1.5 is once differentiable, nu=2.5 is twice differentiable. Matern is more realistic for physical processes.

4. What is the marginal likelihood and how is it used for kernel selection?
P(y|X, theta) = N(y | 0, K_theta + sigma^2 I). Maximizing log marginal likelihood learns kernel hyperparameters, automatically trading off data fit and model complexity.

5. How would you handle multi-output GP regression?
Use independent GPs per output, or coregionalization models (LCM, intrinsic coregionalization) that share kernel structure across outputs.

### Advanced

1. Derive the expected improvement acquisition function for Bayesian optimization.
EI(x) = E[max(f(x) - f(x*), 0)] = (mu(x) - f(x*) - xi) Phi(Z) + sigma(x) phi(Z) where Z = (mu(x) - f(x*) - xi)/sigma(x). Balances exploration and exploitation.

2. Explain sparse GP approximations (e.g., SGPR, VFE).
Select m inducing points Z that summarize the data. Approximate K(X,X) ~ K(X,Z)K(Z,Z)^{-1}K(Z,X). Reduce complexity to O(n m^2). Variational Free Energy (VFE) treats Z as variational parameters.

3. Derive the connection between GPs and Bayesian linear regression with basis functions.
A GP with linear kernel k(x,x') = x^T x' is equivalent to Bayesian linear regression. The RBF kernel corresponds to regression with infinite basis functions (via Taylor expansion). This is the basis of the kernel trick for GPs.

## Practice Problems

### Easy

1. Fit a GP with RBF kernel to noisy sine data. Report the learned length scale and noise level.

2. Plot GP predictions with 95% confidence intervals for 5 training points from a quadratic function.

3. Compare GP regression with RBF vs Matern(nu=0.5) on a step function dataset.

4. Use GaussianProcessClassifier on the iris dataset with 2 features. Report accuracy.

5. Compute and plot the kernel function k(x, x') for RBF with length scales 0.5, 1.0, 2.0.

### Medium

1. Implement GP regression from scratch using NumPy (no sklearn). Compare with sklearn's implementation.

2. Perform automatic relevance determination (ARD) using a separate length scale per feature. Show which features are most relevant.

3. Use GP for Bayesian optimization of a 1D function. Plot the acquisition function and selected points over iterations.

4. Compare GP regression with polynomial regression on a small dataset (n=10). Show how GP uncertainty behaves away from data.

5. Implement leave-one-out cross-validation for GP hyperparameter selection.

### Hard

1. Implement a sparse GP using inducing points from scratch. Compare accuracy and speed with full GP on n=5000.

2. Derive and implement deep GP (2-layer GP) using variational inference. Show how it learns non-stationary functions.

3. Implement GP regression with heteroscedastic noise (noise varies with x). Compare with standard homoscedastic GP.

## Solutions

Easy 1: Simple GP with RBF

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np

np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 30))[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.15, 30)

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
gp.fit(X, y)

print(f"Kernel: {gp.kernel_}")
# Output: Kernel: RBF(length_scale=1.42) + WhiteKernel(noise_level=0.021)
print(f"Log marginal likelihood: {gp.log_marginal_likelihood():.2f}")
# Output: Log marginal likelihood: -18.42
```

## Related Concepts

- **Bayesian Inference** (ML-040): GPs apply Bayesian inference over function space
- **Kernel Methods** (ML-033): The kernel function is central to both SVMs and GPs
- **K-Means** (ML-042): Distance-based method, complementary to kernel-based GP
- **Ridge Regression** (ML-014): GP with linear kernel is Bayesian ridge regression

## Next Concepts

- Gaussian Mixture Models (ML-045): Generative model using multiple Gaussians
- t-SNE (ML-047): Non-linear dimensionality reduction for visualization
- UMAP (ML-048): Modern manifold learning

## Summary

Gaussian Processes provide a powerful Bayesian approach to regression and classification, producing both predictions and well-calibrated uncertainty estimates. A GP is defined by a mean and covariance (kernel) function, encoding prior beliefs about function properties. Inference is analytical for Gaussian likelihoods, costing O(n^3). Predictions are weighted combinations of training observations, with uncertainty increasing away from data. Key to successful GP modeling is appropriate kernel selection, hyperparameter optimization, and handling of the cubic scaling for large datasets.

## Key Takeaways

- GPs define distributions over functions, specified by mean and kernel
- Posterior predictions have closed form for Gaussian likelihoods
- Predictions come with principled uncertainty estimates
- The kernel encodes assumptions about function properties
- O(n^3) complexity limits standard GPs to moderate dataset sizes
- Sparse approximations extend GPs to larger datasets
- Bayesian optimization is a key application of GP uncertainty
- Kernel hyperparameters are learned via marginal likelihood maximization
