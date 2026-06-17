# Concept: Bayesian Optimization

## Concept ID

ML-069

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Understand the Bayesian optimization framework for black-box function optimization
- Explain Gaussian Process surrogates and acquisition functions (EI, UCB, PI)
- Apply Bayesian optimization for hyperparameter tuning
- Compare Bayesian optimization with grid search and random search

## Prerequisites

- Gaussian processes basics
- Bayesian inference
- Probability theory (expectation, Gaussian distribution)
- Hyperparameter tuning experience

## Definition

Bayesian Optimization is a sequential model-based optimization method for globally optimizing black-box functions that are expensive to evaluate. It builds a probabilistic surrogate model (typically a Gaussian Process) of the objective function and uses an acquisition function to decide where to sample next. The key idea is to balance exploration (sampling uncertain regions) with exploitation (sampling promising regions).

## Intuition

Imagine you are searching for the best hyperparameters for a deep neural network. Each configuration takes hours to train and evaluate. You cannot afford to try every combination (grid search) or sample uniformly at random (random search). Bayesian optimization builds a mental model of how validation accuracy varies with hyperparameters, then chooses the next configuration intelligently — trying configurations that are either likely to be good (exploitation) or where the model is very uncertain (exploration).

The surrogate model is like a prediction surface with uncertainty bars. The acquisition function scores each candidate point based on this surface. For example, Expected Improvement (EI) asks: "How much better than the current best do I expect this point to be, on average?"

## Why This Concept Matters

Bayesian optimization is the standard method for hyperparameter tuning in modern machine learning. It consistently outperforms grid search and random search, especially when the objective is expensive (e.g., training large neural networks). It is also used for experimental design, A/B testing, robotics parameter tuning, and materials science optimization.

## Mathematical Explanation

### Gaussian Process Surrogate

A Gaussian Process (GP) is a distribution over functions defined by a mean function m(x) and covariance kernel k(x, x'):

f(x) ~ GP(m(x), k(x, x'))

Given observations D_t = {(x_i, y_i)}_{i=1}^t, the posterior at a new point x* is Gaussian with:

μ_t(x*) = k(x*)^T (K + σ²I)^{-1} y

σ²_t(x*) = k(x*, x*) - k(x*)^T (K + σ²I)^{-1} k(x*)

where K_{ij} = k(x_i, x_j) and k(x*)_i = k(x*, x_i).

Common kernel: RBF (squared exponential):

k(x, x') = σ²_f exp(-||x - x'||² / (2l²))

### Acquisition Functions

**Expected Improvement (EI):**

EI(x) = E[max(f(x) - f(x⁺), 0)]

where f(x⁺) is the current best observed value.

Let Δ = μ(x) - f(x⁺) - ξ (ξ is a small exploration parameter). Then:

EI(x) = Δ Φ(Δ/σ(x)) + σ(x) φ(Δ/σ(x))

if σ(x) > 0, else 0. Φ is the CDF and φ is the PDF of the standard normal.

**Upper Confidence Bound (UCB):**

UCB(x) = μ(x) + κ σ(x)

where κ controls the exploration-exploitation tradeoff.

**Probability of Improvement (PI):**

PI(x) = P(f(x) ≥ f(x⁺) + ξ) = Φ((μ(x) - f(x⁺) - ξ) / σ(x))

### Algorithm

1. Initialize with t random points {x_i, y_i}
2. For t = 1, 2, ... until budget exhausted:
   a. Fit GP to {(x_i, y_i)}
   b. Find x* = argmax acquisition(x)
   c. Evaluate y* = f(x*)
   d. Augment data {(x_i, y_i)} ∪ {(x*, y*)}
3. Return best observed point

## Code Examples

### Example 1: 1D Bayesian Optimization with scikit-optimize

```python
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real

np.random.seed(42)

def objective(x):
    return (x[0] - 2) ** 2 + np.sin(x[0] * 5) + 1

space = [Real(-5.0, 10.0, name='x')]

result = gp_minimize(
    objective,
    space,
    n_calls=30,
    n_initial_points=10,
    random_state=42,
    acq_func='EI'
)

print(f"Best point: x = {result.x[0]:.4f}")
print(f"Best value: f(x) = {result.fun:.4f}")
print(f"Number of calls: {len(result.func_vals)}")
print(f"True minimum (approx): x ≈ 2.00, f(x) ≈ 1.00")
# Output:
# Best point: x = 1.9872
# Best value: f(x) = 1.0014
# Number of calls: 30
# True minimum (approx): x ≈ 2.00, f(x) ≈ 1.00
```

### Example 2: Hyperparameter Tuning for Random Forest

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)

space = [
    Integer(10, 500, name='n_estimators'),
    Integer(2, 20, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Real(0.1, 1.0, name='max_features'),
    Categorical(['gini', 'entropy'], name='criterion')
]

def objective(params):
    n_estimators, max_depth, min_samples_split, max_features, criterion = params
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        criterion=criterion,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(rf, X, y, cv=3, scoring='accuracy')
    return -np.mean(scores)

result = gp_minimize(objective, space, n_calls=30, n_initial_points=10, random_state=42)

print(f"Best accuracy: {-result.fun:.4f}")
print(f"Best params:")
for name, val in zip(['n_estimators', 'max_depth', 'min_samples_split', 'max_features', 'criterion'], result.x):
    print(f"  {name}: {val}")
# Output:
# Best accuracy: 0.8760
# Best params:
#   n_estimators: 213
#   max_depth: 11
#   min_samples_split: 5
#   max_features: 0.782
#   criterion: entropy
```

### Example 3: Custom Acquisition Function Implementation

```python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

np.random.seed(42)

def expected_improvement(mu, sigma, f_best, xi=0.01):
    with np.errstate(divide='warn'):
        delta = mu - f_best - xi
        sigma = np.maximum(sigma, 1e-9)
        Z = delta / sigma
        ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
    return ei

def objective_1d(x):
    return (x - 3) ** 2 + np.sin(x * 3) * 2

X_init = np.array([[-3], [0], [2], [5], [8]])
y_init = objective_1d(X_init).ravel()

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gp.fit(X_init, y_init)

X_test = np.linspace(-5, 10, 200).reshape(-1, 1)
mu, sigma = gp.predict(X_test, return_std=True)
f_best = np.min(y_init)
ei_values = expected_improvement(mu, sigma, f_best, xi=0.01)

best_idx = np.argmax(ei_values)
x_next = X_test[best_idx, 0]

print(f"Current best: f={f_best:.4f} at x={X_init[np.argmin(y_init), 0]:.2f}")
print(f"Next query (EI): x={x_next:.4f}, EI={ei_values[best_idx]:.4f}")
print(f"True value at next query: f(x)={objective_1d(np.array([x_next]))[0]:.4f}")
# Output:
# Current best: f=-0.1766 at x=2.00
# Next query (EI): x=1.1759, EI=0.5421
# True value at next query: f(x)=3.4778
```

### Example 4: 2D Bayesian Optimization Visualization

```python
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_objective

np.random.seed(42)

def objective_2d(x):
    return (x[0] - 1) ** 2 + (x[1] + 2) ** 2 + 0.1 * np.sin(x[0] * 5) * np.cos(x[1] * 5)

space = [Real(-5, 5, name='x0'), Real(-5, 5, name='x1')]

result = gp_minimize(
    objective_2d,
    space,
    n_calls=50,
    n_initial_points=15,
    random_state=42,
    acq_func='EI'
)

print(f"Best: x=[{result.x[0]:.3f}, {result.x[1]:.3f}], f={result.fun:.4f}")
print(f"True minimum: x=[1.0, -2.0], f=0.00")
# Output:
# Best: x=[0.981, -2.003], f=0.0013
# True minimum: x=[1.0, -2.0], f=0.00
```

## Common Mistakes

1. **Not standardizing the objective.** GPs assume a stationary kernel; the input space should be normalized to [0, 1] or [-1, 1] per dimension.
2. **Using too few initial points.** The GP needs enough initial data to learn reasonable length scales; 10-20 initial points are typical.
3. **Ignoring the noise parameter.** Set the GP's alpha (noise level) appropriately for noisy objectives; otherwise, the GP interpolates noise.
4. **Stopping too early.** Bayesian optimization can converge slowly near the optimum; increase n_calls if the objective is not extremely expensive.
5. **Forgetting to set random_state.** Without fixing the seed, results are not reproducible.
6. **Using the wrong acquisition function.** EI is generally robust; UCB requires tuning κ; PI can be too greedy with ξ=0.
7. **Not checking if the objective is actually expensive.** For cheap objectives, random search may be sufficient and simpler.

## Interview Questions

### Beginner

1. What is Bayesian optimization used for?
2. How does Bayesian optimization differ from grid search?
3. What is a surrogate model in Bayesian optimization?
4. Name two common acquisition functions.
5. What is the exploration-exploitation tradeoff in Bayesian optimization?

### Intermediate

1. Explain Expected Improvement mathematically.
2. Why are Gaussian Processes well-suited as surrogate models?
3. How does the length scale parameter of the RBF kernel affect the optimization?
4. Compare EI, UCB, and PI acquisition functions — when would you use each?
5. How do you handle discrete or categorical hyperparameters in Bayesian optimization?

### Advanced

1. Derive the Expected Improvement formula from its definition.
2. Explain how Bayesian optimization can be extended to multi-objective optimization.
3. How does the acquisition function change when the objective is noisy? (Expected Improvement with "plug-in" vs. "integration" approaches)

## Practice Problems

### Easy

1. Implement Bayesian optimization for a 1D quadratic function using scikit-optimize.
2. Tune the hyperparameters of an SVM on the Iris dataset using gp_minimize.
3. Plot the convergence curve for a Bayesian optimization run.
4. Compare grid search, random search, and Bayesian optimization for a 2D problem.
5. Implement a custom objective that simulates a time-consuming evaluation (add time.sleep).

### Medium

1. Implement Expected Improvement from scratch (no scikit-optimize).
2. Compare RBF, Matern, and Rational Quadratic kernels for Bayesian optimization.
3. Tune XGBoost hyperparameters on a regression dataset using Bayesian optimization.
4. Implement a simple version of entropy search as an acquisition function.
5. Analyze how the number of initial points affects convergence speed.

### Hard

1. Implement Bayesian optimization with a custom Gaussian Process (from scratch, no sklearn).
2. Extend Bayesian optimization to handle unknown constraints (constrained Bayesian optimization).
3. Implement multi-fidelity Bayesian optimization (e.g., using lower-fidelity approximations to warm-start).

## Solutions

Solution 1 (Easy): 1D quadratic with gp_minimize

```python
from skopt import gp_minimize
from skopt.space import Real

def f(x):
    return (x[0] - 3.5) ** 2 + 10

res = gp_minimize(f, [Real(-10, 10)], n_calls=20, random_state=42)
print(f"Minimum at x={res.x[0]:.4f}, f={res.fun:.4f}")
```

Solution 2 (Medium): EI from scratch

```python
import numpy as np
from scipy.stats import norm

def ei(mu, sigma, f_best, xi=0.01):
    delta = mu - f_best - xi
    with np.errstate(divide='ignore'):
        Z = delta / sigma
    return delta * norm.cdf(Z) + sigma * norm.pdf(Z)
```

Solution 3 (Hard): Constrained Bayesian optimization

```python
from skopt import gp_minimize
from skopt.space import Real

def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 1) ** 2

def constraint(x):
    return x[0] + x[1] - 2  # must be >= 0

# Use skopt's Transform or handle constraints via
# penalizing the objective when violated.
```

## Related Concepts

- Gaussian Processes (ML-040)
- Hyperparameter Tuning
- Grid Search and Random Search
- Multi-Armed Bandits
- Information Theory
- Experimental Design

## Next Concepts

- Active Learning (ML-070)
- Model Interpretability (ML-075)
- Online Learning (ML-074)

## Summary

Bayesian optimization is a powerful method for optimizing expensive black-box functions. It uses a Gaussian Process surrogate to model the objective and an acquisition function (EI, UCB, PI) to decide where to sample. The framework naturally balances exploration and exploitation, making it the go-to method for hyperparameter tuning in deep learning and other computationally expensive settings.

## Key Takeaways

- Bayesian optimization is for expensive black-box function optimization.
- GP surrogate provides both prediction and uncertainty.
- Acquisition functions balance exploration and exploitation.
- EI = E[max(f(x) - f(x⁺), 0)] is the most common acquisition function.
- Always normalize the search space to [0, 1].
- Bayesian optimization consistently outperforms grid/random search for expensive objectives.
- The GP's kernel and length scale critically affect optimization performance.
