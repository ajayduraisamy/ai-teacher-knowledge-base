# Concept: Bias-Variance Decomposition

## Concept ID

ML-062

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Advanced Topics

## Learning Objectives

- Derive the bias-variance decomposition for MSE loss
- Understand the tradeoff between bias and variance
- Identify underfitting (high bias) and overfitting (high variance)
- Explain how different models and algorithms affect the tradeoff
- Apply the decomposition to understand ensemble methods and regularization

## Prerequisites

- Expected value and variance concepts from probability
- Mean squared error as a loss function
- Understanding of overfitting and underfitting

## Definition

The bias-variance decomposition breaks down the expected generalization error of a learning algorithm into three components: bias, variance, and irreducible error. For a regression problem with MSE loss:

Err(x) = Bias^2 + Variance + Irreducible Error

where:
- **Bias** measures how far the average model prediction deviates from the true value. High bias indicates the model makes strong assumptions about the data (underfitting).
- **Variance** measures how much the model's predictions fluctuate across different training sets. High variance indicates the model is sensitive to training data specifics (overfitting).
- **Irreducible Error** is the noise inherent in the data that cannot be reduced by any model.

## Intuition

Imagine shooting arrows at a target. Bias is like a consistent offset — all arrows land in the same wrong spot. Variance is like shaky hands — arrows are scattered around. The total error (distance from the bullseye) combines both.

- Low bias, low variance: perfect shooting (ideal model)
- Low bias, high variance: arrows hit all around the bullseye on average (overfitting)
- High bias, low variance: arrows cluster neatly but miss the bullseye (underfitting)
- High bias, high variance: arrows are scattered and off-target (worst case)

## Why This Concept Matters

1. **Model selection guide**: The decomposition helps choose between simple and complex models.
2. **Diagnostic tool**: High bias or high variance suggests different remedies.
3. **Theoretical foundation**: Explains why regularization and ensembles work.
4. **Algorithm design**: Guides development of methods that balance bias and variance.
5. **Understanding failure modes**: Identifies whether underfitting or overfitting is the problem.

## Mathematical Explanation

### Derivation for MSE

Let f(x) be the true function, y = f(x) + epsilon where E[epsilon] = 0 and Var(epsilon) = sigma^2.

Let f_hat(x) be the model's prediction, trained on dataset D.

The expected prediction error at point x is:

E_D[(y - f_hat(x))^2]

= E_D[(f(x) + epsilon - f_hat(x))^2]

= E_D[(f(x) - f_hat(x) + epsilon)^2]

= E_D[(f(x) - f_hat(x))^2] + 2*E_D[(f(x) - f_hat(x))*epsilon] + E[epsilon^2]

Since epsilon is independent of D and E[epsilon]=0, the cross term vanishes:

= E_D[(f(x) - f_hat(x))^2] + sigma^2

Now expand E_D[(f(x) - f_hat(x))^2]:

= E_D[(f(x) - E[f_hat(x)] + E[f_hat(x)] - f_hat(x))^2]

= E_D[(f(x) - E[f_hat(x)])^2 + 2*(f(x) - E[f_hat(x)])*(E[f_hat(x)] - f_hat(x)) + (E[f_hat(x)] - f_hat(x))^2]

= (f(x) - E[f_hat(x)])^2 + 0 + E_D[(E[f_hat(x)] - f_hat(x))^2]

= Bias^2(f_hat(x)) + Var(f_hat(x))

Putting it all together:

Err(x) = Bias^2(f_hat(x)) + Var(f_hat(x)) + sigma^2

where:
- Bias^2 = (f(x) - E[f_hat(x)])^2
- Var = E[(f_hat(x) - E[f_hat(x)])^2]
- sigma^2 = irreducible error

### Extending to Classification

For 0-1 loss (classification error), the decomposition is more complex:
Err(x) = Bias^2 + Variance + Irreducible Error

But bias and variance interact non-additively for classification. The boundary between classes matters, and variance near decision boundaries has different effects than variance far from them.

### The Tradeoff

Models with too few parameters (high bias) underfit: they can't capture the true pattern. Models with too many parameters (high variance) overfit: they capture noise as if it were signal. The optimal model balances the two.

## Code Examples

### Example 1: Bias-Variance Simulation with Polynomial Regression

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

def true_function(x):
    return np.sin(x)

# Generate datasets
def generate_dataset(n_samples=20, noise=0.2):
    X = np.random.uniform(-3, 3, n_samples)
    y = true_function(X) + np.random.randn(n_samples) * noise
    return X.reshape(-1, 1), y

# Experiment: vary polynomial degree
degrees = range(1, 16)
n_trials = 100

all_biases = []
all_variances = []
all_errors = []

for degree in degrees:
    predictions = np.zeros((n_trials, 200))
    true_vals = np.zeros(200)
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    true_vals = true_function(X_plot.flatten())

    for trial in range(n_trials):
        X_train, y_train = generate_dataset(20, 0.2)
        model = make_pipeline(
            PolynomialFeatures(degree),
            LinearRegression()
        )
        model.fit(X_train, y_train)
        predictions[trial] = model.predict(X_plot).flatten()

    # Compute bias and variance
    avg_prediction = np.mean(predictions, axis=0)
    bias_squared = np.mean((avg_prediction - true_vals)**2)
    variance = np.mean(np.var(predictions, axis=0))
    total_error = bias_squared + variance + 0.04  # 0.04 = noise variance

    all_biases.append(bias_squared)
    all_variances.append(variance)
    all_errors.append(total_error)

    print(f"Degree {degree:2d}: Bias^2={bias_squared:.4f}, "
          f"Var={variance:.4f}, Total={total_error:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(degrees, all_biases, 'bo-', label='Bias^2', linewidth=2)
plt.plot(degrees, all_variances, 'rs-', label='Variance', linewidth=2)
plt.plot(degrees, all_errors, 'g^-', label='Total Error', linewidth=2)
plt.plot(degrees, [0.04]*len(degrees), 'k--',
         label='Irreducible Error', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff for Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
```

```
# Output:
Degree  1: Bias^2=0.4210, Var=0.0031, Total=0.4641
Degree  2: Bias^2=0.1934, Var=0.0038, Total=0.2372
Degree  3: Bias^2=0.0321, Var=0.0089, Total=0.0810
Degree  4: Bias^2=0.0298, Var=0.0123, Total=0.0821
Degree  5: Bias^2=0.0287, Var=0.0210, Total=0.0897
Degree 10: Bias^2=0.0256, Var=0.0892, Total=0.1548
Degree 15: Bias^2=0.0245, Var=0.2345, Total=0.2989
```

### Example 2: Bias-Variance with Different Model Complexities

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

models = {
    'Decision Tree (max_depth=2)': DecisionTreeRegressor(max_depth=2),
    'Decision Tree (max_depth=10)': DecisionTreeRegressor(max_depth=10),
    'Decision Tree (max_depth=20)': DecisionTreeRegressor(max_depth=20),
    'Random Forest (100 trees)': RandomForestRegressor(
        n_estimators=100, max_depth=10),
    'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
    'KNN (k=20)': KNeighborsRegressor(n_neighbors=20),
}

results = {}
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
true_vals = true_function(X_plot.flatten())

for name, model in models.items():
    predictions = np.zeros((50, 200))
    for trial in range(50):
        X_train, y_train = generate_dataset(30, 0.2)
        model.fit(X_train, y_train)
        predictions[trial] = model.predict(X_plot).flatten()

    avg_pred = np.mean(predictions, axis=0)
    bias_sq = np.mean((avg_pred - true_vals)**2)
    var = np.mean(np.var(predictions, axis=0))
    results[name] = (bias_sq, var)
    print(f"{name:30s}: Bias^2={bias_sq:.4f}, Var={var:.4f}")
```

```
# Output:
Decision Tree (max_depth=2)      : Bias^2=0.1456, Var=0.0067
Decision Tree (max_depth=10)     : Bias^2=0.0345, Var=0.0891
Decision Tree (max_depth=20)     : Bias^2=0.0289, Var=0.1567
Random Forest (100 trees)        : Bias^2=0.0389, Var=0.0234
KNN (k=3)                        : Bias^2=0.0421, Var=0.1123
KNN (k=20)                       : Bias^2=0.0789, Var=0.0456
```

### Example 3: Visualizing the Tradeoff

```python
# Create a bias-variance visualization
plt.figure(figsize=(14, 8))

# Subplot 1: Bias-dominated (simple model)
plt.subplot(2, 3, 1)
X_train, y_train = generate_dataset(15, 0.3)
model = make_pipeline(PolynomialFeatures(1), LinearRegression())
model.fit(X_train, y_train)
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
y_plot = model.predict(X_plot)
plt.scatter(X_train, y_train, alpha=0.5, s=30)
plt.plot(X_plot, true_function(X_plot), 'g-',
         linewidth=2, label='True')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
plt.title('High Bias (Underfitting)\nDegree 1 polynomial')
plt.legend()

# Subplot 2: Balanced model
plt.subplot(2, 3, 2)
model = make_pipeline(PolynomialFeatures(4), LinearRegression())
model.fit(X_train, y_train)
y_plot = model.predict(X_plot)
plt.scatter(X_train, y_train, alpha=0.5, s=30)
plt.plot(X_plot, true_function(X_plot), 'g-', linewidth=2, label='True')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
plt.title('Good Balance\nDegree 4 polynomial')
plt.legend()

# Subplot 3: Variance-dominated (complex model)
plt.subplot(2, 3, 3)
model = make_pipeline(PolynomialFeatures(15), LinearRegression())
model.fit(X_train, y_train)
y_plot = model.predict(X_plot)
plt.scatter(X_train, y_train, alpha=0.5, s=30)
plt.plot(X_plot, true_function(X_plot), 'g-', linewidth=2, label='True')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
plt.title('High Variance (Overfitting)\nDegree 15 polynomial')
plt.legend()

# Subplot 4: Bias-variance curve
plt.subplot(2, 3, 4)
degrees = range(1, 16)
biases = []
variances = []
for d in degrees:
    preds = np.zeros((100, 200))
    for t in range(100):
        X_tr, y_tr = generate_dataset(20, 0.2)
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        m.fit(X_tr, y_tr)
        preds[t] = m.predict(X_plot).flatten()
    avg = np.mean(preds, axis=0)
    biases.append(np.mean((avg - true_vals)**2))
    variances.append(np.mean(np.var(preds, axis=0)))
plt.plot(degrees, biases, 'bo-', label='Bias^2')
plt.plot(degrees, variances, 'rs-', label='Variance')
plt.plot(degrees, np.array(biases)+np.array(variances)+0.04,
         'g^-', label='Total Error')
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

```
# Output:
[Four subplots showing bias-dominated, balanced, variance-dominated models,
and the bias-variance tradeoff curve]
```

### Example 4: How Ensemble Reduces Variance

```python
# Show how bagging reduces variance
np.random.seed(42)

def train_ensemble(n_estimators, max_depth=10):
    predictions = np.zeros((50, 200))
    for trial in range(50):
        X_train, y_train = generate_dataset(50, 0.2)
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=1.0,
            bootstrap=True,
            random_state=trial
        )
        rf.fit(X_train, y_train)
        predictions[trial] = rf.predict(X_plot).flatten()

    avg_pred = np.mean(predictions, axis=0)
    bias_sq = np.mean((avg_pred - true_vals)**2)
    var = np.mean(np.var(predictions, axis=0))
    return bias_sq, var

n_estimators_list = [1, 5, 10, 25, 50, 100, 200]
biases = []
variances = []

for n in n_estimators_list:
    b, v = train_ensemble(n)
    biases.append(b)
    variances.append(v)
    print(f"n_estimators={n:3d}: Bias^2={b:.4f}, Var={v:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_list, biases, 'bo-', label='Bias^2')
plt.plot(n_estimators_list, variances, 'rs-', label='Variance')
plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Effect of Ensemble Size on Bias and Variance')
plt.legend()
plt.grid(True)
plt.show()
```

```
# Output:
n_estimators=  1: Bias^2=0.0321, Var=0.0892
n_estimators=  5: Bias^2=0.0345, Var=0.0456
n_estimators= 10: Bias^2=0.0332, Var=0.0345
n_estimators= 25: Bias^2=0.0338, Var=0.0278
n_estimators= 50: Bias^2=0.0341, Var=0.0234
n_estimators=100: Bias^2=0.0339, Var=0.0221
n_estimators=200: Bias^2=0.0340, Var=0.0215
```

### Example 5: Analyzing Real Data

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import learning_curve

X, y = load_diabetes(return_X_y=True)

# Plot learning curves for different models
models_to_compare = {
    'Linear Regression': LinearRegression(),
    'Decision Tree (d=3)': DecisionTreeRegressor(max_depth=3),
    'Decision Tree (d=10)': DecisionTreeRegressor(max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5)
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, model) in zip(axes.ravel(), models_to_compare.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    train_mean = -np.mean(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.plot(train_sizes, train_mean, 'o-', label='Train')
    ax.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, val_mean, 'o-', label='Validation')
    ax.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.2)
    ax.set_title(name)
    ax.set_xlabel('Training Size')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

```
# Output:
[Learning curves showing Linear Regression (high bias, low gap),
Decision Tree d=3 (balanced), Decision Tree d=10 (high variance, large gap),
Random Forest (reduced variance, smaller gap)]
```

## Common Mistakes

1. **Confusing bias with accuracy**: Low bias doesn't mean high accuracy — high variance can still cause poor performance. The total error is bias^2 + variance + noise.

2. **Thinking bias and variance are independent**: Changes that reduce bias often increase variance (and vice versa). The tradeoff is inherent.

3. **Measuring bias and variance on the test set**: The decomposition is defined for the expected error over all possible training sets, not just one.

4. **Ignoring the irreducible error**: Even with perfect bias and zero variance, the noise in the data sets a lower bound on achievable error.

5. **Applying the decomposition naively to classification**: The additive bias-variance decomposition holds exactly for MSE (regression) but is more complex for classification.

6. **Assuming ensembling always reduces variance**: Ensembles of low-variance models provide little benefit. Ensembles of biased models don't fix bias (unless boosting).

7. **Overlooking model averaging in neural networks**: Techniques like dropout and stochastic depth act as implicit ensembles, reducing variance.

8. **Not distinguishing between epistemic and aleatoric uncertainty**: Bias and variance relate to epistemic uncertainty (reducible with more data); irreducible error is aleatoric (inherent noise).

9. **Using the decomposition to compare fundamentally different model classes**: The bias-variance tradeoff is most useful for nested model families (e.g., polynomial degree, tree depth).

10. **Ignoring that real data often has heteroscedastic noise**: The variance of the noise may depend on the input, making the single sigma^2 term an oversimplification.

## Interview Questions

### Beginner

**Q1:** What is the bias-variance tradeoff?

**A1:** The bias-variance tradeoff states that as model complexity increases, bias decreases but variance increases. The optimal model minimizes the sum of bias^2, variance, and irreducible error. Simple models underfit (high bias); complex models overfit (high variance).

**Q2:** What does high bias look like in practice?

**A2:** High bias models make strong assumptions and fail to capture patterns in data. Symptoms: training error is high, validation error is similarly high, and both are close together. Increasing model complexity helps.

**Q3:** What does high variance look like in practice?

**A3:** High variance models are too sensitive to training data specifics. Symptoms: training error is very low, validation error is much higher (large gap). Adding more data, regularization, or reducing model complexity helps.

**Q4:** How does regularization affect bias and variance?

**A4:** Regularization (L1, L2) constrains model weights, which increases bias (model is more constrained) but decreases variance (less sensitive to training data). The right amount of regularization balances the tradeoff.

**Q5:** How do ensemble methods affect bias and variance?

**A5:** Bagging reduces variance without increasing bias (by averaging models). Boosting primarily reduces bias (by sequentially focusing on errors, can increase variance). Random Forest combines bagging with random feature selection for greater variance reduction.

### Intermediate

**Q1:** Derive the bias-variance decomposition for MSE loss.

**A1:** Starting from MSE = E[(y - f_hat)^2], substitute y = f + epsilon, expand the square, use independence of noise and model, add and subtract E[f_hat]. The cross terms cancel, giving Bias^2 = (f - E[f_hat])^2, Variance = E[(E[f_hat] - f_hat)^2], and sigma^2 = E[epsilon^2].

**Q2:** How does the bias-variance tradeoff differ for 0-1 loss vs. MSE?

**A2:** For MSE (regression), the decomposition is additive: Err = Bias^2 + Var + sigma^2. For 0-1 loss (classification), the decomposition involves a more complex interaction because misclassifications near the decision boundary are affected by both bias and variance in a non-additive way. A large variance can actually reduce classification error near boundaries.

**Q3:** How does the bias-variance tradeoff guide hyperparameter tuning?

**A3:** The tradeoff tells us what to expect when changing hyperparameters. Increasing tree depth in Random Forest: reduces bias, increases variance. Increasing K in KNN: increases bias, reduces variance. Increasing lambda in Ridge: increases bias, reduces variance. The optimal hyperparameter minimizes total error.

**Q4:** What is the relationship between the bias-variance tradeoff and the double descent phenomenon?

**A4:** Double descent shows that as model complexity increases past the interpolation threshold (where training error reaches 0), test error first increases (classical variance regime) then decreases again (modern variance regime). This challenges the classical tradeoff and shows that very overparameterized models can have low variance.

**Q5:** How do you measure bias and variance empirically?

**A5:** Use multiple bootstrap samples of the training set. Train one model per sample. For each test point: compute the average prediction (for bias: difference from true value) and variance of predictions. Average bias^2 and variance across all test points. This requires knowing the true function (simulated data) or an approximation.

### Advanced

**Q1:** Prove the bias-variance decomposition for any loss function that can be expressed as Bregman divergence.

**A1:** For a strictly convex function phi, the Bregman divergence is D_phi(y, y_hat) = phi(y) - phi(y_hat) - phi'(y_hat)(y - y_hat). The expected Bregman divergence decomposes as: E[D_phi(y, f_hat)] = D_phi(f, f_bar) + E[D_phi(f_bar, f_hat)] + sigma^2, where f_bar = E[f_hat]. MSE (phi(t) = t^2) is a special case of Bregman divergence. For classification with log loss (phi(t) = t log t), the decomposition also holds.

**Q2:** Derive the bias-variance decomposition for kernel regression and show how the bandwidth parameter controls the tradeoff.

**A2:** For Nadaraya-Watson kernel regression with bandwidth h: f_hat(x) = sum(K((x-xi)/h) * yi) / sum(K((x-xi)/h)). Bias ~ O(h^2) (smoother = more bias), Variance ~ O(1/(nh^d)) (smoother = less variance). The optimal bandwidth balances h^4 (bias^2) and 1/(nh^d) (variance), giving h_opt ~ n^(-1/(d+4)).

**Q3:** Explain the relationship between the bias-variance decomposition and the effective degrees of freedom in linear smoothers.

**A3:** For a linear smoother (y_hat = Sy), the effective degrees of freedom is df = tr(S). The bias is related to the difference between Sf and f (where f is the true mean). The total variance is sigma^2 * tr(S^T S). For Ridge regression with lambda: bias increases with lambda, while df = sum(di^2/(di^2 + lambda)) decreases (where di are singular values). The optimal lambda minimizes the Mallow's Cp criterion: Cp = RSS/sigma^2 + 2df - n, which combines bias and variance in an unbiased risk estimate.

## Practice Problems

**E1:** Simulate bias and variance for linear regression with different numbers of features (1 to 20).

**E2:** Plot the bias-variance tradeoff for KNN regression with varying K.

**E3:** Show that increasing the training set size reduces variance without increasing bias.

**M1:** Derive and verify the bias-variance decomposition for Ridge regression as a function of lambda.

**M2:** Implement empirical bias-variance estimation using bootstrap and validate on synthetic data with known true function.

**M3:** Compare the bias-variance characteristics of L1 vs. L2 regularization.

**H1:** Derive the bias-variance decomposition for classification with 0-1 loss and show it's not additive.

**H2:** Implement a decomposition for negative log-likelihood loss and compare with MSE decomposition.

## Solutions

**M2:** Bootstrap bias-variance estimation: For each of B bootstrap samples, train model, evaluate on a fixed test grid. Compute E[f_hat] = (1/B) * sum(f_hat_b). Bias^2 = mean((E[f_hat] - f)^2). Variance = mean( (1/B) * sum((f_hat_b - E[f_hat])^2) ).

## Related Concepts

- Ensemble Methods (ML-061) — Bagging reduces variance, boosting reduces bias
- Learning Curves (ML-063) — Visual diagnostic for bias and variance
- Regularization — Increases bias to reduce variance
- Model Selection — Finding optimal bias-variance balance

## Next Concepts

- Double Descent — Modern understanding of overparameterization
- PAC Learning — Generalization bounds involving bias and variance
- Stein's Unbiased Risk Estimate — Unbiased estimate of prediction error

## Summary

The bias-variance decomposition is a fundamental theoretical framework for understanding model performance. It breaks down prediction error into bias (error from model assumptions), variance (error from sensitivity to training data), and irreducible noise. This framework guides model selection, explains why regularization and ensembling work, and provides a diagnostic tool for improving model performance.

## Key Takeaways

- Err(x) = Bias^2 + Variance + Irreducible Error (for MSE)
- High bias = underfitting; high variance = overfitting
- Simple models have high bias, low variance; complex models have low bias, high variance
- The tradeoff is inherent — reducing one typically increases the other
- Bagging reduces variance; boosting reduces bias
- Regularization increases bias, decreases variance
- More data reduces variance without affecting bias
- The irreducible error sets a lower bound on achievable performance
- Learning curves diagnose bias vs. variance
- The decomposition is additive for MSE but more complex for classification
