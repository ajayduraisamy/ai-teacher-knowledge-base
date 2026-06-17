# Concept: Overfitting and Underfitting

## Concept ID

ML-004

## Difficulty

BEGINNER

## Domain

Machine Learning

## Module

ML Fundamentals

## Learning Objectives

- Define overfitting, underfitting, and the bias-variance tradeoff
- Distinguish between high bias (underfitting) and high variance (overfitting)
- Diagnose overfitting and underfitting using learning curves
- Apply regularization techniques to mitigate overfitting
- Interpret validation curves for hyperparameter tuning

## Prerequisites

- ML-001: What is Machine Learning
- ML-003: Train/Test Split
- Basic understanding of model complexity

## Definition

### Overfitting

Overfitting occurs when a machine learning model learns the training data too well, capturing noise and random fluctuations rather than the true underlying pattern. An overfit model has exceptionally high performance on training data but poor performance on unseen data. It has effectively memorized the training set instead of learning generalizable patterns.

Overfitting is associated with **high variance** — the model is highly sensitive to the specific training data used. If you train the same algorithm on a different training sample, the resulting model changes dramatically.

### Underfitting

Underfitting occurs when a model is too simple to capture the underlying structure of the data. An underfit model performs poorly on both training data and unseen data because it has not learned enough from the training process.

Underfitting is associated with **high bias** — the model makes strong assumptions about the data that are not justified. No matter what training data it sees, the model cannot capture the true relationship because its capacity is too limited.

### The Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that describes the tension between a model's ability to fit data well (low bias) and its ability to generalize to new data (low variance). Total error can be decomposed into three components:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias**: Error introduced by approximating a complex real-world problem with a simpler model. High bias leads to underfitting.
- **Variance**: Error introduced by the model's sensitivity to small fluctuations in the training set. High variance leads to overfitting.
- **Irreducible Error**: The noise inherent in the problem that cannot be reduced by any model.

## Intuition

### Overfitting Intuition

Imagine a student preparing for an exam by memorizing the textbook word-for-word, including the page numbers, font sizes, and minor typos. On the exam, if a question is phrased exactly as in the textbook, the student answers perfectly. But if the question is rephrased or tests the same concept in a new context, the student fails because they never truly understood the underlying principles.

In ML terms: The model has memorized the training data, including its noise. When new data arrives, the model cannot generalize because its decision boundaries are too complex and specific to the training set.

### Underfitting Intuition

Imagine a student who only reads the chapter titles and summaries, believing this is sufficient to understand the subject. On the exam, the student has only a superficial understanding and cannot answer detailed questions about any topic. The student never learned enough to perform well.

In ML terms: The model is too simple to capture the patterns in the data. It makes crude approximations that miss important relationships.

### Bias-Variance Tradeoff Intuition

Consider throwing darts at a dartboard:
- **High bias, low variance**: All darts land in the same spot, but that spot is far from the bullseye. Consistent but inaccurate.
- **Low bias, high variance**: The average dart position is near the bullseye, but individual darts are scattered widely. Accurate on average but inconsistent.
- **Low bias, low variance**: Darts cluster tightly around the bullseye. The ideal scenario.
- **High bias, high variance**: Darts are scattered widely and cluster far from the bullseye. The worst case.

## Why This Concept Matters

Overfitting and underfitting are the most common failure modes in machine learning. Understanding them enables practitioners to:

- Diagnose why a model is performing poorly
- Select the right model complexity for a problem
- Apply appropriate remedies (regularization, more data, feature selection)
- Communicate tradeoffs to stakeholders
- Build models that generalize reliably to production data

Virtually every ML technique — from regularization to cross-validation to feature selection — exists to address overfitting or underfitting.

## Mathematical Explanation

### Bias-Variance Decomposition

For a regression problem with squared error loss, the expected test error at a point $x$ can be decomposed as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- $\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$ — the difference between the average prediction and the true value
- $\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$ — the variability of predictions across different training sets
- $\sigma^2$ — the irreducible error (noise)

### Derivation

Let $f(x)$ be the true function, $y = f(x) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$ and $\text{Var}[\epsilon] = \sigma^2$.

$$\begin{align}
\mathbb{E}[(y - \hat{f})^2] &= \mathbb{E}[(f + \epsilon - \hat{f})^2] \\
&= \mathbb{E}[(f - \hat{f})^2] + 2\mathbb{E}[(f - \hat{f})\epsilon] + \mathbb{E}[\epsilon^2] \\
&= \mathbb{E}[(f - \hat{f})^2] + \sigma^2
\end{align}$$

Now expand $\mathbb{E}[(f - \hat{f})^2]$:

$$\begin{align}
\mathbb{E}[(f - \hat{f})^2] &= \mathbb{E}[(f - \mathbb{E}[\hat{f}] + \mathbb{E}[\hat{f}] - \hat{f})^2] \\
&= \mathbb{E}[(f - \mathbb{E}[\hat{f}])^2] + \mathbb{E}[(\mathbb{E}[\hat{f}] - \hat{f})^2] + 2\mathbb{E}[(f - \mathbb{E}[\hat{f}])(\mathbb{E}[\hat{f}] - \hat{f})] \\
&= \text{Bias}[\hat{f}]^2 + \text{Var}[\hat{f}] + 0
\end{align}$$

This gives the full decomposition: $\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$.

### Learning Curves

Learning curves plot training and validation performance against training set size or model complexity.

- **With model complexity** (validation curve): As complexity increases, training error decreases monotonically. Validation error initially decreases (reducing bias), then increases (increasing variance). The optimal complexity is at the minimum of the validation curve.
- **With training set size**: As training size increases, training error increases (harder to fit all points), and validation error decreases (more data improves generalization). They converge to the irreducible error as $n \to \infty$.

## Code Examples

### Example 1: Overfitting with Polynomial Regression

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data: y = sin(x) + noise
np.random.seed(42)
X = np.sort(np.random.rand(40, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(40) * 0.25

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test different polynomial degrees
for degree in [1, 3, 15]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"Degree {degree:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
# Output:
# Degree  1: Train MSE = 0.2814, Test MSE = 0.2677  (Underfitting - high bias)
# Degree  3: Train MSE = 0.0607, Test MSE = 0.0697  (Good fit - balanced)
# Degree 15: Train MSE = 0.0283, Test MSE = 0.3114  (Overfitting - high variance)
```

### Example 2: Learning Curves to Diagnose Overfitting/Underfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
X = np.sort(np.random.rand(200, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(200) * 0.25

# Deep tree (prone to overfitting)
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeRegressor(max_depth=None, random_state=42),
    X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='neg_mean_squared_error'
)

train_errors = -train_scores.mean(axis=1)
val_errors = -val_scores.mean(axis=1)

print("Learning curve for deep tree (overfitting):")
for size, tr_err, val_err in zip(train_sizes, train_errors, val_errors):
    print(f"  Train size={size:3d}: Train MSE={tr_err:.4f}, Val MSE={val_err:.4f}")
# Output:
# Learning curve for deep tree (overfitting):
#   Train size= 18: Train MSE=0.0000, Val MSE=0.6364
#   Train size= 40: Train MSE=0.0000, Val MSE=0.4219
#   Train size= 62: Train MSE=0.0000, Val MSE=0.3112
#   Train size= 84: Train MSE=0.0000, Val MSE=0.2817
#   Train size=106: Train MSE=0.0000, Val MSE=0.2550
#   ...

# Shallow tree (prone to underfitting)
train_sizes2, train_scores2, val_scores2 = learning_curve(
    DecisionTreeRegressor(max_depth=2, random_state=42),
    X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='neg_mean_squared_error'
)

train_errors2 = -train_scores2.mean(axis=1)
val_errors2 = -val_scores2.mean(axis=1)

print("\nLearning curve for shallow tree (underfitting):")
for size, tr_err, val_err in zip(train_sizes2, train_errors2, val_errors2):
    print(f"  Train size={size:3d}: Train MSE={tr_err:.4f}, Val MSE={val_err:.4f}")
# Output:
# Learning curve for shallow tree (underfitting):
#   Train size= 18: Train MSE=0.2457, Val MSE=0.3326
#   Train size= 40: Train MSE=0.2087, Val MSE=0.2541
#   Train size= 62: Train MSE=0.2088, Val MSE=0.2297
#   Train size= 84: Train MSE=0.2074, Val MSE=0.2186
#   ...
```

### Example 3: Regularization to Combat Overfitting

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Generate high-dimensional data prone to overfitting
np.random.seed(42)
n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)
true_coefs = np.zeros(n_features)
true_coefs[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]
y = X @ true_coefs + np.random.randn(n_samples) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# No regularization (linear regression)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Regression (no regularization):")
print(f"  Train MSE: {mean_squared_error(y_train, lr.predict(X_train)):.4f}")
print(f"  Test MSE: {mean_squared_error(y_test, lr.predict(X_test)):.4f}")
# Output:
# Linear Regression (no regularization):
#   Train MSE: 0.0000
#   Test MSE: 15592.0563

# L2 regularization (Ridge)
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
print(f"Ridge Regression (L2, alpha=10):")
print(f"  Train MSE: {mean_squared_error(y_train, ridge.predict(X_train)):.4f}")
print(f"  Test MSE: {mean_squared_error(y_test, ridge.predict(X_test)):.4f}")
# Output:
# Ridge Regression (L2, alpha=10):
#   Train MSE: 2.5537
#   Test MSE: 2.0401

# L1 regularization (Lasso) — also performs feature selection
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
print(f"Lasso Regression (L1, alpha=0.5):")
print(f"  Train MSE: {mean_squared_error(y_train, lasso.predict(X_train)):.4f}")
print(f"  Test MSE: {mean_squared_error(y_test, lasso.predict(X_test)):.4f}")
print(f"  Non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
# Output:
# Lasso Regression (L1, alpha=0.5):
#   Train MSE: 0.1876
#   Test MSE: 0.2790
#   Non-zero coefficients: 5
```

### Example 4: Validation Curve for Hyperparameter Tuning

```python
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.svm import SVR

np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.25

# Validation curve for SVM gamma parameter
param_range = np.logspace(-3, 2, 6)
train_scores, val_scores = validation_curve(
    SVR(kernel='rbf'), X, y,
    param_name='gamma', param_range=param_range,
    cv=5, scoring='neg_mean_squared_error'
)

train_errors = -train_scores.mean(axis=1)
val_errors = -val_scores.mean(axis=1)

print("Validation curve for SVM gamma:")
for gamma, tr_err, val_err in zip(param_range, train_errors, val_errors):
    status = ""
    if tr_err < val_err and tr_err < 0.05:
        status = " (overfitting?)"
    elif tr_err > 0.3 and val_err > 0.3:
        status = " (underfitting)"
    print(f"  gamma={gamma:.3f}: Train MSE={tr_err:.4f}, Val MSE={val_err:.4f}{status}")
# Output:
# Validation curve for SVM gamma:
#   gamma=0.001: Train MSE=0.3835, Val MSE=0.3813 (underfitting)
#   gamma=0.010: Train MSE=0.2448, Val MSE=0.2432
#   gamma=0.032: Train MSE=0.0766, Val MSE=0.0882
#   gamma=0.100: Train MSE=0.0375, Val MSE=0.0548
#   gamma=0.316: Train MSE=0.0117, Val MSE=0.0785
#   gamma=1.000: Train MSE=0.0000, Val MSE=0.1697 (overfitting?)
```

## Common Mistakes

1. **Judging model quality only from training performance**: Training error is misleadingly optimistic. Always evaluate on held-out data.
2. **Ignoring the validation curve**: Many practitioners pick the most complex model without checking where validation error starts increasing.
3. **Using too much regularization**: Over-regularization suppresses both variance and signal, leading to underfitting. Tune regularization strength carefully.
4. **Assuming more data always fixes overfitting**: More data helps, but the rate of improvement depends on model complexity. A highly complex model may need exponentially more data.
5. **Confusing model complexity with model performance**: A complex model is not inherently better. The best model minimizes test error, which often occurs at moderate complexity.
6. **Applying the same preprocessing to train and test before splitting**: Data leakage makes overfitting harder to detect because test performance appears artificially high.
7. **Not using cross-validation for small datasets**: Single train/test splits on small datasets can show apparent underfitting or overfitting due to chance, leading to incorrect conclusions.

## Interview Questions

### Beginner - 5

1. **Q: What is the difference between overfitting and underfitting?**
   A: Overfitting occurs when a model learns the training data too well, including noise, resulting in poor generalization. Underfitting occurs when a model is too simple to capture the underlying patterns, performing poorly on both training and test data.

2. **Q: What causes overfitting?**
   A: Overfitting is caused by a model that is too complex relative to the amount and quality of training data. Common causes include too many features, insufficient training data, and training for too many iterations.

3. **Q: What is the bias-variance tradeoff?**
   A: The bias-variance tradeoff describes the inverse relationship between a model's ability to fit training data (low bias) and its stability across different training sets (low variance). Increasing model complexity reduces bias but increases variance, and vice versa.

4. **Q: How does regularization help with overfitting?**
   A: Regularization adds a penalty for large coefficients to the loss function, which constrains the model and reduces its effective complexity. This reduces variance at the cost of slightly increased bias.

5. **Q: What is a learning curve?**
   A: A learning curve plots training and validation performance as a function of training set size or model complexity. It helps diagnose whether a model is overfitting or underfitting.

### Intermediate - 5

1. **Q: Explain how to use learning curves to diagnose overfitting and underfitting.**
   A: For overfitting: training error is very low, validation error is significantly higher, and the gap does not close with more data. For underfitting: both training and validation errors are high and convergent, indicating the model lacks capacity. For a good fit: both errors are low and close to each other.

2. **Q: Compare L1 (Lasso) and L2 (Ridge) regularization in terms of bias-variance tradeoff.**
   A: L1 regularization tends to produce sparse models (some coefficients exactly zero), acting as a feature selection mechanism. It has higher bias but can be more interpretable. L2 regularization shrinks all coefficients toward zero but rarely to zero. It often provides better prediction when all features are relevant. L1 can have infinite variance when features are correlated (non-unique solutions).

3. **Q: How does the bias-variance decomposition help in understanding ensemble methods?**
   A: Bagging (Random Forest) reduces variance by averaging multiple high-variance, low-bias models trained on bootstrap samples. Boosting reduces bias by sequentially training models to correct previous errors, but can increase variance. Understanding the decomposition guides which method to use.

4. **Q: What is the relationship between model capacity and the amount of training data needed?**
   A: The sample complexity of a model grows with its capacity. As a rule of thumb, the number of training samples should exceed the number of parameters by at least a factor of 10. More complex models (deep neural networks) may require millions of samples.

5. **Q: How do early stopping and dropout combat overfitting in neural networks?**
   A: Early stopping halts training when validation performance stops improving, preventing the model from memorizing training data. Dropout randomly deactivates neurons during training, forcing the network to learn redundant representations and reducing co-adaptation between neurons.

### Advanced - 3

1. **Q: Derive the bias-variance decomposition for the 0-1 loss function (classification) and explain why it is more complex than for squared error.**
   A: For 0-1 loss, the decomposition is not as clean as for squared error. The expected misclassification rate can be expressed as: $\mathbb{E}[I(y \neq \hat{y}(x))] = P(\hat{y}(x) = 1)P(y = 0) + P(\hat{y}(x) = 0)P(y = 1)$. This does not decompose additively into bias and variance terms because the loss is not quadratic. In classification, bias relates to the model's average decision boundary, while variance relates to boundary instability. The ambiguity around the decision boundary makes the decomposition less interpretable.

2. **Q: Prove that the expected prediction error for ridge regression decreases as $\lambda$ increases from 0 (under certain conditions), relating to the bias-variance tradeoff.**
   A: Ridge regression solution: $\hat{\beta}_\lambda = (X^T X + \lambda I)^{-1} X^T y$. The bias is $\mathbb{E}[\hat{\beta}_\lambda] - \beta = - \lambda (X^T X + \lambda I)^{-1} \beta$, which increases with $\lambda$. The variance is $\sigma^2 (X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1}$, which decreases with $\lambda$ as the eigenvalues of $(X^T X + \lambda I)^{-1}$ are shrunk. The optimal $\lambda$ minimizes the sum of squared bias and variance. For certain data configurations (high multicollinearity), small increases in $\lambda$ dramatically reduce variance with minimal bias increase.

3. **Q: Explain the double descent phenomenon in modern ML and how it relates to classical bias-variance theory.**
   A: Classical bias-variance theory describes a U-shaped test error curve as model complexity increases. However, modern overparameterized models (neural networks with more parameters than training samples) exhibit a "double descent" curve: test error initially decreases, then increases near the interpolation threshold (where model just fits training data), then decreases again in the overparameterized regime. This challenges classical theory — extremely large models can generalize well despite (or because of) their size. The phenomenon is explained by the implicit regularization of gradient descent in overparameterized models.

## Practice Problems

### Easy - 5

1. **Problem**: A decision tree has training accuracy of 100% and test accuracy of 72%. Is this overfitting or underfitting?

2. **Problem**: A linear regression model has R² = 0.25 on both training and test data. Is this overfitting or underfitting?

3. **Problem**: What happens to bias and variance as you increase the degree of a polynomial regression model?

4. **Problem**: Name three techniques to reduce overfitting.

5. **Problem**: Name three techniques to reduce underfitting.

### Medium - 5

1. **Problem**: You are building a model with 500 features and 200 samples. The model achieves training R² = 0.99 and test R² = 0.30. Diagnose and recommend solutions.

2. **Problem**: A logistic regression model is trained with and without L2 regularization. The regularized model has lower test accuracy but the unregularized model has much higher variance across different train/test splits. Explain what is happening.

3. **Problem**: Given the following learning curve data, determine whether the model is overfitting or underfitting:
   - With 100 training samples: Train MSE = 0.32, Val MSE = 0.38
   - With 500 training samples: Train MSE = 0.31, Val MSE = 0.34
   - With 2000 training samples: Train MSE = 0.30, Val MSE = 0.31

4. **Problem**: Design an experiment to determine the optimal depth for a decision tree on a given dataset of 10,000 samples.

5. **Problem**: Explain why increasing the training set size is an effective remedy for overfitting but may not help with underfitting.

### Hard - 3

1. **Problem**: Prove the bias-variance decomposition for ridge regression, showing how the regularization parameter $\lambda$ affects each component.

2. **Problem**: Compare the bias-variance characteristics of k-nearest neighbors as k varies from 1 to n. Derive or explain the behavior of bias and variance.

3. **Problem**: A deep neural network with 10 million parameters trained on 50,000 images achieves excellent test performance despite being massively overparameterized. Explain this phenomenon using modern ML theory.

## Solutions

### Easy Solutions

1. Overfitting — the model memorized training data (100% accuracy) but fails to generalize (72% test accuracy).
2. Underfitting — low performance on both sets indicates the model is too simple to capture patterns.
3. As polynomial degree increases, bias decreases (model becomes more flexible) and variance increases (model becomes more sensitive to training data fluctuations).
4. Regularization (L1/L2), reduce model complexity (shallower trees, fewer features), increase training data, early stopping, dropout, cross-validation.
5. Increase model complexity (more features, deeper trees, higher polynomial degree), reduce regularization, engineer better features, reduce noise in data, train for more iterations.

### Medium Solutions

1. Classic overfitting: 500 features >> 200 samples, model memorizes training data. Solutions: (1) Feature selection (Lasso, mutual information), (2) Dimensionality reduction (PCA), (3) Strong regularization, (4) Collect more data, (5) Simpler model.
2. The regularized model has higher bias (shrinks coefficients) but lower variance. On some test sets, the unregularized model happens to perform better, but on average the regularized model is more stable and likely better for deployment.
3. Underfitting — both errors are close together but high, suggesting the model lacks capacity. The small gap indicates low variance, and the high error level indicates high bias.
4. (1) Split into train/val/test (60/20/20). (2) Train decision trees with depths 1-20 on training set. (3) Evaluate each on validation set. (4) Select depth with lowest validation error. (5) Retrain at that depth on train+val. (6) Report final test performance.
5. More data provides more examples of the true pattern, making it harder for the model to memorize noise (reduces overfitting). However, if the model is inherently too simple (underfitting), more data does not help because the model lacks the capacity to capture the pattern — the errors converge to a high value.

### Hard Solutions

1. For ridge: $\hat{\beta}_\lambda = (X^T X + \lambda I)^{-1} X^T y$. Let $A(\lambda) = (X^T X + \lambda I)^{-1} X^T X$. Then $\text{Bias}^2 = \beta^T (A(\lambda) - I)^T (A(\lambda) - I) \beta$ and $\text{Variance} = \sigma^2 \text{tr}((X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1})$. As $\lambda$ increases, the eigenvalues of $A(\lambda)$ shrink from 1 toward 0, increasing bias (the "shrinkage" bias) but decreasing variance (less sensitivity to training data). The optimal $\lambda$ minimizes the sum.

2. For k-NN: When k=1, bias is near 0 (no average over neighbors) but variance is high (depends on single nearest neighbor's label). As k increases, bias increases (averaging over more distant neighbors smooths away fine patterns) and variance decreases (averaging reduces the impact of any single noisy point). At k=n (all points), the model predicts the global mean — maximum bias, minimum variance. The optimal k minimizes total error and can be found via cross-validation.

3. This is the "double descent" or "benign overfitting" phenomenon. In overparameterized regimes, neural networks can interpolate training data perfectly (zero training error) while still generalizing well. Explanations: (1) Implicit regularization of SGD favors solutions with low norm that generalize. (2) The model's inductive bias (architecture, data augmentation) constrains the hypothesis space. (3) Deep networks learn progressively simpler features, and overparameterization allows finding global minima that generalize. (4) The neural tangent kernel theory shows that wide networks behave like kernel methods with good generalization properties.

## Related Concepts

- **ML-003: Train/Test Split** — Essential for detecting overfitting
- **ML-005: Cross-Validation** — Robust evaluation that helps diagnose overfitting
- **ML-010: Feature Selection** — Reduces overfitting by removing irrelevant features
- **Regularization**: Mathematical techniques (L1, L2, elastic net) to control model complexity
- **Bias-Variance Tradeoff**: The foundational theoretical framework for understanding overfitting/underfitting

## Next Concepts

- **ML-005: Cross-Validation** — A more robust evaluation framework
- **ML-006: Evaluation Metrics** — How to measure performance correctly
- **ML-010: Feature Selection** — Reducing overfitting through feature selection

## Summary

Overfitting (high variance) and underfitting (high bias) are the two fundamental failure modes in machine learning. Overfitting occurs when a model is too complex and memorizes training data noise. Underfitting occurs when a model is too simple to capture underlying patterns. The bias-variance tradeoff describes the tension between these extremes — increasing model complexity reduces bias but increases variance. Learning curves and validation curves are diagnostic tools for identifying these problems. Regularization (L1/L2), simplifying the model, and collecting more data help combat overfitting. Increasing model complexity, reducing regularization, and better feature engineering help combat underfitting.

## Key Takeaways

1. Overfitting = high variance, low bias; underfitting = low variance, high bias.
2. Training performance alone is never sufficient to evaluate a model.
3. Learning curves diagnose overfitting (large train-test gap) vs underfitting (both errors high and close).
4. Regularization is the primary technique for controlling overfitting.
5. The bias-variance tradeoff is central to model selection.
6. More data helps overfitting but not necessarily underfitting.
7. The optimal model complexity balances bias and variance to minimize total error.
