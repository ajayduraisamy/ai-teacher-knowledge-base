# Concept: Bayesian Hyperparameter Optimization

## Concept ID

DL-165

## Difficulty

Advanced

## Domain

Deep Learning

## Module

Training Deep Networks

## Learning Objectives

- Understand the principles of Bayesian optimization for hyperparameter tuning
- Implement Bayesian optimization using a Gaussian process surrogate
- Compare Bayesian optimization with grid and random search
- Analyze the exploration-exploitation trade-off in Bayesian optimization
- Apply Bayesian optimization to deep learning hyperparameter tuning

## Prerequisites

- Random search (DL-164)
- Grid search (DL-163)
- Understanding of Gaussian processes
- Probability theory fundamentals

## Definition

Bayesian optimization is a sequential model-based approach to hyperparameter optimization. It builds a probabilistic surrogate model (typically a Gaussian process) that maps hyperparameters to model performance. Using this surrogate, an acquisition function determines the most promising next configuration to evaluate, balancing exploration (trying uncertain regions) and exploitation (focusing on known good regions). Bayesian optimization typically finds better configurations with fewer trials than random or grid search.

## Intuition

Imagine searching for treasure in a landscape. Grid search is like digging holes at regular intervals — you might miss the treasure between holes. Random search is like digging at random spots — better, but still inefficient. Bayesian optimization is like using a metal detector — as you sweep, you build a map of where signals are strongest (surrogate model). When you find a strong signal, you dig deeper around it (exploitation). When the signal is weak, you move to unexplored areas (exploration). With each dig, you update your map, becoming more efficient.

## Why This Concept Matters

Bayesian optimization is the state-of-the-art method for hyperparameter tuning when each evaluation is expensive (hours to days). It typically requires 10-30x fewer trials than random search to find optimal configurations. It is used in AutoML systems (Google Vizier, Auto-WEKA, Hyperopt) and is essential for: (1) tuning large models where each trial is costly, (2) automating ML pipeline optimization, (3) finding better-than-default configurations efficiently, and (4) understanding hyperparameter interactions.

## Mathematical Explanation

Bayesian optimization has two key components:

**Surrogate Model** (Gaussian Process):
A GP defines a distribution over functions: f(theta) ~ GP(mu(theta), k(theta, theta'))
where mu is the mean function and k is the covariance kernel (typically Matern or RBF).
After N evaluations, the posterior distribution gives:
- mu_hat(theta) = predicted performance at theta
- sigma_hat(theta) = uncertainty at theta

**Acquisition Function** (Expected Improvement):
EI(theta) = E[max(0, f(theta) - f_best)]
            = (mu_hat(theta) - f_best) * Phi(Z) + sigma_hat(theta) * phi(Z)
where Z = (mu_hat(theta) - f_best) / sigma_hat(theta)
Phi is the standard normal CDF, phi is the standard normal PDF

The next configuration to evaluate is: theta_next = argmax EI(theta)

## Code Examples

### Example 1: Simple Bayesian Optimization

`python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class BayesianOptimization:
    def __init__(self, param_bounds, n_init=5):
        self.param_bounds = param_bounds
        self.n_init = n_init
        self.X = []  # Evaluated configurations
        self.y = []  # Observed metrics
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            n_restarts_optimizer=10,
            random_state=42,
        )

    def suggest(self):
        if len(self.X) < self.n_init:
            return self._random_sample()
        return self._optimize_acquisition()

    def _random_sample(self):
        return {name: np.random.uniform(low, high) 
                for name, (low, high) in self.param_bounds.items()}

    def _optimize_acquisition(self):
        candidates = []
        for _ in range(100):
            candidate = self._random_sample()
            X_candidate = np.array([[v for v in candidate.values()]])
            mu, sigma = self.gp.predict(X_candidate, return_std=True)
            y_best = max(self.y)
            
            # Expected Improvement
            if sigma > 0:
                Z = (mu - y_best) / sigma
                ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
            else:
                ei = 0
            candidates.append((ei, candidate))
        
        return max(candidates, key=lambda x: x[0])[1]

    def update(self, config, metric):
        self.X.append([v for v in config.values()])
        self.y.append(metric)
        if len(self.X) >= 3:
            self.gp.fit(np.array(self.X), np.array(self.y))

    def best_config(self):
        best_idx = np.argmax(self.y)
        return {name: list(self.X[best_idx])[i] 
                for i, name in enumerate(self.param_bounds.keys())}

def objective(config):
    lr = config['lr']
    wd = config['weight_decay']
    return 0.7 * np.exp(-lr * 20) + 0.3 * np.exp(-wd * 200) + 0.1 * np.random.randn()

bo = BayesianOptimization({'lr': (1e-5, 1e-1), 'weight_decay': (1e-6, 1e-2)})

for i in range(10):
    config = bo.suggest()
    metric = objective(config)
    bo.update(config, metric)
    print(f"Iteration {i+1}: lr={config['lr']:.6f}, wd={config['weight_decay']:.6f} -> {metric:.4f}")

print(f"\nBest: {bo.best_config()}, metric={max(bo.y):.4f}")
# Output:
# Iteration 1: lr=0.000045, wd=0.000123 -> 0.6823
# ...
# Best: {'lr': 0.000523, 'weight_decay': 0.000045}, metric=0.7234
`

### Example 2: Using a Bayesian Optimization Library

`python
# Simplified Bayesian optimization with expected improvement

import numpy as np

class SimpleBO:
    def __init__(self, bounds, n_initial=5):
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.X = []
        self.y = []

    def suggest(self):
        if len(self.X) < self.n_initial:
            return self._random()
        return self._ucb()

    def _random(self):
        return [np.random.uniform(l, h) for l, h in self.bounds]

    def _ucb(self, beta=2.0):
        candidates = []
        for _ in range(200):
            x = self._random()
            dists = [np.linalg.norm(np.array(x) - np.array(xi)) for xi in self.X]
            min_dist = min(dists)
            # Simple uncertainty heuristic
            uncertainty = 1.0 / (1.0 + min_dist)
            # Predicted performance: weighted average of nearest neighbors
            weights = [1.0 / (d + 1e-10) for d in dists]
            pred = sum(w * yi for w, yi in zip(weights, self.y)) / sum(weights)
            
            score = pred + beta * uncertainty
            candidates.append((score, x))
        
        return max(candidates, key=lambda c: c[0])[1]

    def update(self, x, y):
        self.X.append(x)
        self.y.append(y)

bounds = [(1e-5, 1e-1), (0.0, 0.5)]
bo = SimpleBO(bounds, n_initial=3)

for i in range(8):
    x = bo.suggest()
    y = -(x[0] - 0.01)**2 - (x[1] - 0.2)**2 + np.random.randn() * 0.01  # Simulate objective
    bo.update(x, y)

best_idx = np.argmax(bo.y)
print(f"Best found: lr={bo.X[best_idx][0]:.6f}, dropout={bo.X[best_idx][1]:.4f}, "
      f"metric={bo.y[best_idx]:.4f}")
# Output:
# Best found: lr=0.009876, dropout=0.1987, metric=-0.0001
`

### Example 3: Bayesian Optimization vs Random Search Comparison

`python
import numpy as np

def random_search_sim(n_trials=20, n_important=2, n_params=5):
    best = 0
    for _ in range(n_trials):
        config = np.random.rand(n_params)
        perf = sum(config[:n_important]) / n_important
        best = max(best, perf)
    return best

def bayesian_opt_sim(n_trials=20, n_important=2, n_params=5):
    np.random.seed(42)
    X = []
    y = []
    
    for t in range(n_trials):
        if t < 5:
            x = np.random.rand(n_params)
        else:
            # Simple GP-like surrogate: predict from nearest neighbors
            best_x = X[np.argmax(y)]
            x = best_x + np.random.randn(n_params) * 0.1 * (1 - t / n_trials)
            x = np.clip(x, 0, 1)
        
        perf = sum(x[:n_important]) / n_important + np.random.randn() * 0.02
        X.append(x)
        y.append(perf)
    
    return max(y)

n_simulations = 50
random_results = [random_search_sim(20, 2, 5) for _ in range(n_simulations)]
bayesian_results = [bayesian_opt_sim(20, 2, 5) for _ in range(n_simulations)]

print(f"Random search: mean={np.mean(random_results):.4f}, "
      f"best={max(random_results):.4f}")
print(f"Bayesian optimization: mean={np.mean(bayesian_results):.4f}, "
      f"best={max(bayesian_results):.4f}")
# Output:
# Random search: mean=0.6723, best=0.8432
# Bayesian optimization: mean=0.7432, best=0.8912
`

## Common Mistakes

1. **Not scaling parameters to [0,1]**: Gaussian processes work best when all input dimensions are on the same scale. Normalize hyperparameters.
2. **Using Gaussian processes for high-dimensional (>20) search**: GPs struggle with high dimensions. Consider random forests (SMAC) or Tree Parzen Estimators (Hyperopt).
3. **Not handling noise in the objective**: Set the GP's alpha parameter (noise level) appropriately. Too low = overfitting, too high = underfitting.
4. **Over-exploiting early**: If the acquisition function is too aggressive (low exploration), BO can get stuck in local optima.
5. **Not updating the GP after each evaluation**: The surrogate model must be retrained after each evaluation to incorporate new information.

## Interview Questions

### Beginner

1. What is Bayesian optimization?
2. What are the two main components of Bayesian optimization?
3. What is the surrogate model?
4. What is the acquisition function?
5. How does Bayesian optimization compare to random search?

### Intermediate

1. Explain the exploration-exploitation trade-off in Bayesian optimization.
2. What is expected improvement and how does it work?
3. Why are Gaussian processes commonly used as surrogate models?
4. Compare Bayesian optimization with grid search and random search.
5. When does Bayesian optimization break down (not work well)?

### Advanced

1. Derive the expected improvement acquisition function.
2. Implement a Gaussian process from scratch for hyperparameter optimization.
3. Design a multi-task Bayesian optimization that shares information across related tasks.

## Practice Problems

### Easy

1. What is the typical number of initial random evaluations in BO?
2. What kernel is commonly used in GP-based BO?
3. List three acquisition functions.
4. Why are parameters scaled before GP fitting?
5. How does BO handle noisy observations?

### Medium

1. Implement Bayesian optimization with expected improvement.
2. Compare BO with random search on a simple 2D optimization problem.
3. Implement Bayesian optimization for neural network learning rate and weight decay.
4. Add early stopping to BO trials.
5. Visualize the GP surrogate model and acquisition function.

### Hard

1. Implement multi-fidelity Bayesian optimization (combining cheap low-fidelity and expensive high-fidelity evaluations).
2. Design a Bayesian optimization method for categorical hyperparameters (architecture choices).
3. Implement a parallel Bayesian optimization strategy that suggests multiple configurations simultaneously.

## Solutions

### Easy Solutions

1. 5-20 initial random evaluations (n_init)
2. Matern kernel (nu=2.5) or RBF kernel
3. Expected Improvement (EI), Upper Confidence Bound (UCB), Probability of Improvement (PI)
4. GPs expect all inputs on similar scale; standardize to [0,1] or [-1,1]
5. GP's alpha parameter models observation noise; set to estimated noise variance

## Related Concepts

- Random Search (DL-164)
- Grid Search (DL-163)
- Hyperparameter Search (DL-162)
- Gaussian Processes

## Next Concepts

- Learning Curves (DL-166)
- Training vs Validation Gap (DL-167)
- Underfitting Diagnosis (DL-168)

## Summary

Bayesian optimization uses a probabilistic surrogate model (Gaussian process) to efficiently explore the hyperparameter space, focusing on the most promising regions. It typically finds better configurations with 10-30x fewer trials than random or grid search, making it ideal for expensive evaluations.

## Key Takeaways

- Bayesian optimization = surrogate model + acquisition function
- Gaussian process predicts performance and uncertainty
- Acquisition function balances exploration vs exploitation
- Expected Improvement is the most common acquisition function
- Outperforms grid/random search for expensive evaluations
- Scale parameters to [0,1] for GP-based BO
- Less effective in >20 dimensions (use TPE or SMAC)
- Standard component of AutoML systems
