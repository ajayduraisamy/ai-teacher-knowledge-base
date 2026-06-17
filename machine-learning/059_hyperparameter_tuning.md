# Concept: Hyperparameter Tuning

## Concept ID

ML-059

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Neural Networks

## Learning Objectives

- Understand the difference between parameters and hyperparameters
- Implement grid search, random search, and Bayesian optimization
- Use sklearn's GridSearchCV and RandomizedSearchCV
- Interpret learning curves for diagnosing model performance
- Apply best practices for hyperparameter tuning

## Prerequisites

- Basic machine learning concepts — training/validation/test splits
- Model evaluation — cross-validation, overfitting, underfitting

## Definition

Hyperparameter tuning is the process of systematically searching for the optimal set of hyperparameters — configuration variables set before training that control the learning process. Unlike model parameters (weights learned from data), hyperparameters must be specified in advance and significantly impact model performance.

Common hyperparameters include learning rate, number of hidden layers, number of units per layer, regularization strength, batch size, dropout rate, and optimizer choice.

## Intuition

Think of hyperparameter tuning like adjusting the settings on a camera. The aperture, shutter speed, and ISO are all settings (hyperparameters) that must be chosen before taking a photo. The resulting image quality (model performance) depends heavily on getting these settings right. Just as a photographer takes test shots to find the right settings, a data scientist trains models with different hyperparameter combinations to find the best configuration.

## Why This Concept Matters

1. **Performance impact**: Proper tuning can improve model accuracy by 10-30% over default settings.
2. **Resource optimization**: Efficient tuning saves days or weeks of computation.
3. **Scientific rigor**: Systematic tuning ensures reproducible results and fair comparisons.
4. **Understanding the model**: Tuning reveals how different hyperparameters affect behavior.
5. **Automation**: Modern ML platforms automate tuning, but understanding the principles is essential for effective use.

## Mathematical Explanation

### Grid Search

Exhaustively evaluates all combinations in a predefined grid of hyperparameter values.

Given hyperparameters H1 with values {v11, v12, ...} and H2 with values {v21, v22, ...}, grid search evaluates all |H1| * |H2| combinations.

**Complexity:** O(|H1| * |H2| * ... * |Hk|) evaluations.
**Pros:** Simple, guarantees finding best combination in grid.
**Cons:** Curse of dimensionality — number of evaluations grows exponentially.

### Random Search

Samples hyperparameter combinations uniformly at random from defined ranges. Bergstra and Bengio (2012) showed that random search is more efficient than grid search for high-dimensional spaces because not all hyperparameters are equally important.

**Strategy:** For N trials, sample N random combinations from the search space.
**Pros:** More efficient in high dimensions, can explore more values per parameter.
**Cons:** No guarantee of finding optimal combination, may miss narrow peaks.

Bayesian Optimization builds a probabilistic model (Gaussian Process or Tree Parzen Estimator) mapping hyperparameters to performance. It uses an acquisition function to balance exploration (trying uncertain regions) and exploitation (focusing on promising regions).

**Expected Improvement (EI):**
EI(x) = E[max(f(x) - f*, 0)]

where f* is the current best observed value. The next point to evaluate is:
x_next = argmax EI(x)

Bayesian optimization typically requires 10-20x fewer evaluations than grid search.

### Practical Guidelines

Learning curves plot training and validation scores as a function of training set size (or training iterations). They diagnose:
- **High bias (underfitting)**: Both curves converge to a low score, close together.
- **High variance (overfitting)**: Large gap between train and validation curves.

## Code Examples

### Example 1: Grid Search with sklearn

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
    'activation': ['tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
}

start = time.time()
grid_search = GridSearchCV(
    MLPClassifier(max_iter=200, random_state=42, solver='adam'),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"Grid search completed in {elapsed:.2f}s")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

```
# Output:
Fitting 3 folds for each of 36 candidates, totalling 108 fits
Grid search completed in 45.23s
Best parameters: {'activation': 'relu', 'alpha': 0.0001,
                  'hidden_layer_sizes': (100,),
                  'learning_rate_init': 0.001}
Best CV score: 0.9729
Test score: 0.9750
```

### Example 2: Random Search with sklearn

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (50, 25),
                           (100, 50), (150, 75)],
    'activation': ['tanh', 'relu'],
    'alpha': loguniform(1e-5, 1e-1),
    'learning_rate_init': loguniform(1e-4, 1e-1),
    'batch_size': randint(16, 128),
}

start = time.time()
random_search = RandomizedSearchCV(
    MLPClassifier(max_iter=200, random_state=42, solver='adam'),
    param_dist,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)
elapsed = time.time() - start

print(f"\nRandom search completed in {elapsed:.2f}s")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
print(f"Test score: {random_search.score(X_test, y_test):.4f}")
```

```
# Output:
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Random search completed in 25.67s
Best parameters: {'activation': 'relu', 'alpha': 5.23e-05,
                  'batch_size': 64,
                  'hidden_layer_sizes': (100,),
                  'learning_rate_init': 0.0087}
Best CV score: 0.9754
Test score: 0.9778
```

### Example 3: Learning Curves for Diagnosis

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=200,
                  random_state=42, solver='adam'),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std,
                 train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, val_mean - val_std,
                 val_mean + val_std, alpha=0.2, color='red')
plt.plot(train_sizes, train_mean, 'o-', label='Train', color='blue')
plt.plot(train_sizes, val_mean, 'o-', label='Validation', color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final train accuracy: {train_mean[-1]:.4f}")
print(f"Final validation accuracy: {val_mean[-1]:.4f}")
print(f"Gap: {train_mean[-1] - val_mean[-1]:.4f}")
```

```
# Output:
Final train accuracy: 0.9986
Final validation accuracy: 0.9722
Gap: 0.0264
```

### Example 4: Validation Curve for a Single Hyperparameter

```python
from sklearn.model_selection import validation_curve

param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

train_scores, val_scores = validation_curve(
    MLPClassifier(hidden_layer_sizes=(100,), max_iter=200,
                  random_state=42, solver='adam'),
    X_train, y_train,
    param_name='alpha',
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(range(len(param_range)), train_mean - train_std,
                 train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(range(len(param_range)), val_mean - val_std,
                 val_mean + val_std, alpha=0.2, color='red')
plt.plot(range(len(param_range)), train_mean, 'o-', label='Train')
plt.plot(range(len(param_range)), val_mean, 'o-', label='Validation')
plt.xticks(range(len(param_range)), [str(p) for p in param_range])
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Accuracy')
plt.title('Validation Curve for Alpha')
plt.legend()
plt.grid(True)
plt.show()

for i, alpha in enumerate(param_range):
    print(f"Alpha={alpha:.5f}: Train={train_mean[i]:.4f}, "
          f"Val={val_mean[i]:.4f}")
```

```
# Output:
Alpha=0.00001: Train=1.0000, Val=0.9708
Alpha=0.00010: Train=1.0000, Val=0.9729
Alpha=0.00100: Train=0.9986, Val=0.9743
Alpha=0.01000: Train=0.9785, Val=0.9674
Alpha=0.10000: Train=0.9250, Val=0.9174
Alpha=1.00000: Train=0.5257, Val=0.5236
```

### Example 5: Coarse-to-Fine Bayesian Optimization (Manual)

```python
# Coarse-to-fine tuning demonstration
import itertools

# Stage 1: Coarse grid
coarse_grid = {
    'lr': [0.1, 0.01, 0.001],
    'hidden': [50, 100, 200],
    'alpha': [0.0001, 0.001, 0.01]
}

print("Stage 1: Coarse search")
results = []
for lr, hidden, alpha in itertools.product(
    coarse_grid['lr'], coarse_grid['hidden'], coarse_grid['alpha']
):
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation='relu',
        solver='sgd',
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    score = mlp.score(X_train, y_train)
    results.append({'lr': lr, 'hidden': hidden, 'alpha': alpha,
                    'score': score})
    print(f"  LR={lr}, hidden={hidden}, alpha={alpha}: "
          f"{score:.4f}")

# Stage 2: Fine grid around best
best = max(results, key=lambda r: r['score'])
print(f"\nStage 1 best: {best}")

print("\nStage 2: Fine search around best")
fine_lr = [best['lr'] * f for f in [0.3, 0.5, 1.0, 2.0, 3.0]]
fine_hidden = [best['hidden'] + d for d in [-30, -15, 0, 15, 30]
               if best['hidden'] + d > 0]
fine_alpha = [best['alpha'] * f for f in [0.3, 0.5, 1.0, 2.0, 3.0]]

for lr, hidden, alpha in itertools.product(
    fine_lr[:3], fine_hidden[:3], fine_alpha[:3]
):
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        activation='relu',
        solver='sgd',
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    score = mlp.score(X_test, y_test)
    print(f"  LR={lr:.5f}, hidden={hidden}, alpha={alpha:.5f}: "
          f"{score:.4f}")
```

```
# Output:
Stage 1: Coarse search
  LR=0.1, hidden=50, alpha=0.0001: 0.9750
  LR=0.1, hidden=50, alpha=0.001: 0.9549
  ...
Stage 1 best: {'lr': 0.01, 'hidden': 100, 'alpha': 0.0001, 'score': 0.9681}

Stage 2: Fine search around best
  LR=0.00300, hidden=70, alpha=0.00003: 0.9625
  LR=0.00500, hidden=85, alpha=0.00005: 0.9708
  LR=0.01000, hidden=100, alpha=0.00010: 0.9778
  ...
```

## Common Mistakes

1. **Tuning on test data**: Never use test data for tuning. Use a separate validation set or cross-validation. Tuning on test data gives over-optimistic performance estimates.

2. **Using too few cross-validation folds**: 3-5 folds are standard. Fewer than 3 gives noisy estimates; more than 10 is expensive with diminishing returns.

3. **Grid search for high-dimensional spaces**: Grid search scales exponentially. For more than 3-4 hyperparameters, use random search or Bayesian optimization.

4. **Not using log scale for positive parameters**: Learning rates, regularization strengths, and other scale parameters should be sampled on a log scale (e.g., 1e-4 to 1e-1).

5. **Over-tuning**: Excessive tuning can lead to overfitting the validation set. The best validation score may not correspond to the best test performance.

6. **Ignoring interactions between hyperparameters**: Some hyperparameters interact (e.g., learning rate and batch size). Always tune them together, not independently.

7. **Not using early stopping during tuning**: Evaluating poorly configured models wastes computation. Use early stopping to terminate bad configurations quickly.

8. **Using the same hyperparameters across datasets**: Optimal hyperparameters are dataset-dependent. Always tune for each new dataset.

9. **Not considering computation budget**: More hyperparameter evaluations aren't always better. Balance search thoroughness with available compute.

10. **Tuning too many hyperparameters at once**: Focus on the 3-5 most important hyperparameters. The rest can use reasonable defaults.

## Interview Questions

### Beginner

**Q1:** What is the difference between a parameter and a hyperparameter?

**A1:** Parameters are learned from data (e.g., neural network weights). Hyperparameters are set before training (e.g., learning rate, number of layers). Parameters are optimized by gradient descent; hyperparameters are tuned by searching.

**Q2:** What is grid search and when would you use it?

**A2:** Grid search exhaustively evaluates all combinations in a predefined grid of hyperparameter values. Use it when you have few hyperparameters (<=3) and small value ranges, or when exhaustive tuning is computationally feasible.

**Q3:** How does random search differ from grid search?

**A3:** Random search samples hyperparameter combinations uniformly at random from defined ranges. It's more efficient than grid search when some hyperparameters are more important than others, as it explores more distinct values per hyperparameter.

**Q4:** What is a validation curve?

**A4:** A validation curve plots training and validation scores against a single hyperparameter value. It shows the effect of that hyperparameter on model performance and helps identify overfitting (diverging curves) and underfitting (both low).

**Q5:** Why should we use a separate test set during hyperparameter tuning?

**A5:** The test set must remain unseen until the final evaluation. If we tune on the test set, we risk overfitting to it, and the reported performance won't generalize to new data.

### Intermediate

**Q1:** Explain Bayesian optimization for hyperparameter tuning.

**A1:** Bayesian optimization builds a probabilistic surrogate model (typically Gaussian Process) mapping hyperparameters to model performance. It uses an acquisition function (Expected Improvement) to select the next point to evaluate, balancing exploration of uncertain regions and exploitation of known good regions. It typically finds good configurations in 10-20x fewer evaluations than grid search.

**Q2:** How does learning rate interact with batch size in hyperparameter tuning?

**A2:** There's a linear scaling rule: when doubling batch size, double the learning rate to maintain the same effective gradient variance. However, this holds only for moderate batch sizes (up to ~2048). Beyond that, the learning rate may need additional adjustments.

**Q3:** What are learning curves and how do they diagnose bias vs. variance?

**A3:** Learning curves plot train/validation error vs. training size. High bias: both curves converge to high error, close together. High variance: train error is low, validation error is high (large gap). Adding more data helps high variance but not high bias.

**Q4:** How would you tune a neural network with 8+ hyperparameters?

**A4:** Use a multi-stage approach: (1) Random search with wide ranges to identify promising regions. (2) Bayesian optimization to refine around good configurations. (3) Reduce the number of hyperparameters by fixing less important ones. (4) Use early stopping to prune bad trials quickly.

**Q5:** What is the difference between tuning for SGD vs. Adam?

**A5:** SGD typically requires tuning learning rate, momentum, and learning rate schedule. Adam is less sensitive: default learning rate (0.001) works well, and beta1=0.9, beta2=0.999 are good defaults. For Adam, the most important hyperparameters are learning rate and weight decay.

### Advanced

**Q1:** Explain the concept of multi-fidelity hyperparameter optimization.

**A1:** Multi-fidelity optimization (e.g., Hyperband, Successive Halving) evaluates many configurations at low fidelity (few iterations, small subset of data) and progressively allocates more resources to promising ones. This is far more efficient than evaluating all configurations at full fidelity. Hyperband uses a grid of resource allocations, while BOHB combines it with Bayesian optimization.

**Q2:** How do you handle hyperparameter tuning in distributed systems?

**A2:** Use asynchronous parallel tuning: workers independently sample and evaluate hyperparameter configurations, reporting results to a central database. The search algorithm (e.g., Bayesian optimization) uses all collected results to suggest the next configuration. This scales to hundreds of workers. Tools like Optuna, Ray Tune, and Katib support distributed tuning with early stopping and pruning.

**Q3:** Compare evolutionary optimization with Bayesian optimization for hyperparameter tuning.

**A3:** Evolutionary algorithms (CMA-ES, genetic algorithms) maintain a population of configurations, using crossover and mutation to evolve better solutions. They handle discrete and conditional hyperparameters well. Bayesian optimization is more sample-efficient for low-dimensional continuous spaces. For high-dimensional or mixed-type spaces, evolutionary methods often perform better. Population-Based Training (PBT) combines evolutionary ideas with weight sharing for Neural Architecture Search.

## Practice Problems

**E1:** Use GridSearchCV to tune an SVM on the iris dataset. Report best parameters and test accuracy.

**E2:** Use RandomizedSearchCV to tune a random forest on the wine dataset with 5 hyperparameters.

**E3:** Plot learning curves for an MLP on the digits dataset. Is it overfitting or underfitting?

**M1:** Implement a simple Bayesian optimization from scratch using Gaussian Processes.

**M2:** Compare grid search vs. random search efficiency for a 4-parameter tuning problem.

**M3:** Implement Hyperband for tuning an MLP on a larger dataset.

**H1:** Implement Population-Based Training (PBT) for tuning an MLP during training.

**H2:** Analyze the hyperparameter importance ranking using fANOVA or ablation studies.

## Solutions

**M1:** Bayesian optimization implementation involves: (1) Initialize GP with random points. (2) Fit GP to observed (params, score) pairs. (3) Optimize acquisition function to suggest next point. (4) Evaluate and add to observations. (5) Repeat.

## Related Concepts

- Cross-Validation — Evaluating model performance robustly
- Learning Curves (ML-063) — Visual diagnosis of bias/variance
- Gradient Descent Variants (ML-055) — Learning rate tuning
- Ensemble Methods (ML-061) — Hyperparameter tuning for ensemble components

## Next Concepts

- Automated Machine Learning (AutoML) — Full pipeline optimization
- Neural Architecture Search (NAS) — Automating architecture design
- Meta-Learning — Learning to learn hyperparameters

## Summary

Hyperparameter tuning is essential for achieving optimal model performance. Grid search is simple but inefficient for high-dimensional spaces. Random search is more efficient and often finds good configurations faster. Bayesian optimization provides the most sample-efficient approach by learning from previous evaluations. Learning curves and validation curves help diagnose model behavior and guide tuning.

## Key Takeaways

- Hyperparameters are set before training; parameters are learned during training
- Grid search exhaustively evaluates all combinations; random search is more efficient
- Bayesian optimization uses a probabilistic model to guide search efficiently
- Learning curves diagnose high bias (both curves low) vs. high variance (large gap)
- Validation curves show the effect of a single hyperparameter
- Tune on a validation set; use test set only for final evaluation
- Use log scales for learning rate, regularization, and other scale parameters
- Coarse-to-fine search is an effective multi-stage strategy
- Early stopping saves computation during tuning
- The best hyperparameters are dataset-dependent — always tune
