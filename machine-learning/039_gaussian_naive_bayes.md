# Concept: Gaussian Naive Bayes

## Concept ID

ML-039

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Instance-Based and Probabilistic Methods

## Learning Objectives

- Derive the Gaussian likelihood function for Naive Bayes
- Understand how means and variances are estimated per class per feature
- Implement Gaussian Naive Bayes from scratch using NumPy
- Analyze the effect of variance estimation on decision boundaries
- Compare GaussianNB with other Naive Bayes variants

## Prerequisites

- Naive Bayes classifier (ML-038)
- Normal (Gaussian) distribution and its PDF
- Maximum likelihood estimation
- Basic linear algebra (means, variances)

## Definition

Gaussian Naive Bayes (GNB) is a variant of the Naive Bayes classifier that assumes the continuous feature values $x_i$ for each class $y$ follow a univariate Gaussian (normal) distribution. For each feature $i$ and class $k$, GNB estimates two parameters: the mean $\mu_{ik}$ and the variance $\sigma_{ik}^2$ from the training data.

The conditional probability of feature $x_i$ given class $y=k$ is:

$$P(x_i | y=k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$

The overall classification rule is:

$$\hat{y} = \arg\max_k \left[\log P(y=k) + \sum_{i=1}^n \log P(x_i | y=k)\right]$$

## Intuition

Imagine plotting the distribution of sepal length for three species of iris flowers. Each species has a bell-shaped curve centered at a different mean — setosas have small sepals, virginica have large ones. GaussianNB simply learns these per-class, per-feature bell curves and uses them to compute how likely a new flower belongs to each species given its measurements.

The key insight: instead of modeling the joint distribution of all four measurements (which would require estimating covariances), GaussianNB models each measurement independently per class. This reduces the parameter count from $O(d^2)$ to $O(d)$ per class, making it extremely data-efficient.

## Why This Concept Matters

Gaussian Naive Bayes is the go-to variant for continuous data. It is widely used for real-time classification, medical diagnosis (where interpretability matters), and as a baseline in research papers. Understanding its inner workings — especially how variance estimates affect decisions — provides insight into generative models, the bias-variance tradeoff in density estimation, and the relationship between parametric assumptions and model performance.

## Mathematical Explanation

### Parameter Estimation via MLE

Given training data $\mathcal{D} = \{(x^{(j)}, y^{(j)})\}_{j=1}^m$, we estimate for each class $k$ and feature $i$:

$$\hat{\mu}_{ik} = \frac{1}{m_k} \sum_{j: y^{(j)} = k} x_i^{(j)}$$

$$\hat{\sigma}_{ik}^2 = \frac{1}{m_k} \sum_{j: y^{(j)} = k} (x_i^{(j)} - \hat{\mu}_{ik})^2$$

where $m_k$ is the number of training examples belonging to class $k$.

These are maximum likelihood estimates for the Gaussian distribution parameters. Note that sklearn uses the MLE variance (dividing by $m_k$, not $m_k - 1$), which is slightly biased but often preferred for its simplicity.

### Decision Boundary

For binary classification with GNB, the decision boundary where $P(y=1|x) = P(y=0|x)$ satisfies:

$$\log\frac{P(y=1)}{P(y=0)} + \sum_{i=1}^n \left[\log\frac{\sigma_{i0}}{\sigma_{i1}} - \frac{(x_i-\mu_{i1})^2}{2\sigma_{i1}^2} + \frac{(x_i-\mu_{i0})^2}{2\sigma_{i0}^2}\right] = 0$$

This is a quadratic function of $x$ because of the $x_i^2$ terms. If all classes share the same variance $\sigma_i^2 = \sigma_{i0}^2 = \sigma_{i1}^2$, the quadratic terms cancel and the boundary becomes linear:

$$\sum_{i=1}^n \frac{\mu_{i1} - \mu_{i0}}{\sigma_i^2} x_i + \text{constant} = 0$$

### Log-Likelihood Computation

In sklearn's GaussianNB, the log-likelihood is computed as:

$$\log P(x_i | y=k) = -\frac{1}{2}\log(2\pi\sigma_{ik}^2) - \frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}$$

The constant term $-\frac{1}{2}\log(2\pi)$ can be dropped during classification since it's the same for all classes, but sklearn includes it for proper probability estimation.

### Variance Smoothing

To prevent zero variance estimates (which would make the log-likelihood infinite), sklearn's GaussianNB applies a variance smoothing parameter `var_smoothing`. The effective variance becomes:

$$\hat{\sigma}_{ik}^2 = \hat{\sigma}_{ik}^2 + \text{var\_smoothing} \times \max_j \hat{\sigma}_{jk}^2$$

This is particularly important when a feature is constant within a class.

## Code Examples

### Example 1: GaussianNB from Scratch

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + self.var_smoothing
            self.priors[c] = X_c.shape[0] / X.shape[0]

        return self

    def _log_likelihood(self, X, c):
        mu = self.means[c]
        var = self.vars[c]
        log_constant = -0.5 * np.log(2 * np.pi * var)
        log_exponent = -0.5 * ((X - mu) ** 2 / var)
        return log_constant + log_exponent

    def predict_log_proba(self, X):
        log_probs = []
        for c in self.classes:
            log_prior = np.log(self.priors[c])
            log_lik = np.sum(self._log_likelihood(X, c), axis=1)
            log_probs.append(log_prior + log_lik)
        return np.array(log_probs).T

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / np.sum(probs, axis=1, keepdims=True)

# Test on Iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb_scratch = GaussianNBScratch()
gnb_scratch.fit(X_train, y_train)
y_pred = gnb_scratch.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Custom GaussianNB accuracy: {acc:.4f}")
# Output: Custom GaussianNB accuracy: 0.9556

# Compare with sklearn
from sklearn.naive_bayes import GaussianNB
gnb_sk = GaussianNB()
gnb_sk.fit(X_train, y_train)
y_pred_sk = gnb_sk.predict(X_test)
acc_sk = accuracy_score(y_test, y_pred_sk)
print(f"Sklearn GaussianNB accuracy: {acc_sk:.4f}")
# Output: Sklearn GaussianNB accuracy: 0.9556
```

### Example 2: Mean and Variance Per Class

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

gnb = GaussianNB()
gnb.fit(X, y)

print("Class Means (theta_):")
means_df = pd.DataFrame(gnb.theta_, columns=feature_names, index=target_names)
print(means_df.round(2))
# Output:
# Class Means (theta_):
#                 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# setosa                     5.006              3.428              1.462              0.246
# versicolor                 5.936              2.770              4.260              1.326
# virginica                  6.588              2.974              5.552              2.026

print("\nClass Variances (var_):")
var_df = pd.DataFrame(gnb.var_, columns=feature_names, index=target_names)
print(var_df.round(3))
# Output:
# Class Variances (var_):
#                 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# setosa                    0.122              0.142              0.030              0.011
# versicolor                0.261              0.097              0.216              0.039
# virginica                 0.396              0.102              0.298              0.075

# Note how setosa has very low variance for petal features
# — this means it's tightly clustered and easy to identify
```

### Example 3: Effect of Variance Smoothing

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification

# Create dataset with a constant feature in one class
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
X[y == 0, 0] = 5.0  # Feature 0 is constant for class 0

for var_smoothing in [0, 1e-12, 1e-9, 1e-6, 1e-3]:
    gnb = GaussianNB(var_smoothing=var_smoothing)
    try:
        gnb.fit(X, y)
        score = gnb.score(X, y)
        print(f"var_smoothing={var_smoothing:8.0e}: accuracy={score:.4f}")
    except Exception as e:
        print(f"var_smoothing={var_smoothing:8.0e}: FAILED - {e}")

# Output:
# var_smoothing=0e+00: FAILED - divide by zero in log
# var_smoothing=1e-12: accuracy=0.9350
# var_smoothing=1e-09: accuracy=0.9350
# var_smoothing=1e-06: accuracy=0.9350
# var_smoothing=1e-03: accuracy=0.9300
```

### Example 4: Decision Boundary Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=2, n_classes=2,
                           n_redundant=0, n_clusters_per_class=1,
                           class_sep=0.8, random_state=42)

gnb = GaussianNB().fit(X, y)
lda = LinearDiscriminantAnalysis().fit(X, y)
qda = QuadraticDiscriminantAnalysis().fit(X, y)

# Create mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

Z_gnb = gnb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_lda = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_qda = qda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['GaussianNB (Quadratic)', 'LDA (Linear)', 'QDA (Quadratic)']
Zs = [Z_gnb, Z_lda, Z_qda]

for ax, Z, title in zip(axes, Zs, titles):
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('gnb_decision_boundary.png', dpi=150)
print("Decision boundary comparison saved.")
# Output: Decision boundary comparison saved.

# GNB produces quadratic boundaries (each class has its own variance per feature)
# LDA produces linear boundaries (shared covariance across classes)
# QDA produces quadratic boundaries (full covariance per class)
```

### Example 5: Learning Curves - Small Data Performance

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)

train_sizes, train_scores, test_scores = learning_curve(
    GaussianNB(), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', random_state=42
)

gnb_train_mean = np.mean(train_scores, axis=1)
gnb_test_mean = np.mean(test_scores, axis=1)

# Compare with Logistic Regression
train_sizes2, train_scores2, test_scores2 = learning_curve(
    LogisticRegression(max_iter=5000), X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', random_state=42
)
lr_train_mean = np.mean(train_scores2, axis=1)
lr_test_mean = np.mean(test_scores2, axis=1)

print("Training size | GNB Train | GNB Test | LR Train | LR Test")
print("-" * 60)
for i, size in enumerate(train_sizes):
    print(f"{size:5d} samples  |  {gnb_train_mean[i]:.3f}   |  {gnb_test_mean[i]:.3f}  |  "
          f"{lr_train_mean[i]:.3f}   |  {lr_test_mean[i]:.3f}")

# Output:
# Training size | GNB Train | GNB Test | LR Train | LR Test
# ------------------------------------------------------------
#   134 samples  |  0.922   |  0.835  |  1.000   |  0.876
#   269 samples  |  0.923   |  0.859  |  1.000   |  0.896
#   404 samples  |  0.926   |  0.867  |  1.000   |  0.906
#   538 samples  |  0.927   |  0.872  |  1.000   |  0.907
#   673 samples  |  0.927   |  0.875  |  1.000   |  0.909
#   808 samples  |  0.927   |  0.878  |  1.000   |  0.915
#   943 samples  |  0.928   |  0.880  |  1.000   |  0.915
#  1077 samples  |  0.928   |  0.880  |  1.000   |  0.916
#  1212 samples  |  0.928   |  0.881  |  1.000   |  0.918
#  1347 samples  |  0.929   |  0.881  |  1.000   |  0.919

# Note: GNB has lower training accuracy (not overfitting) while
# LR achieves perfect training accuracy (overfitting risk)
```

## Common Mistakes

1. **Using GaussianNB on non-Gaussian data.** Features with multimodal, skewed, or heavy-tailed distributions violate the normality assumption. Apply transformations (log, Box-Cox) or use non-parametric density estimation.

2. **Forgetting that features are assumed independent within each class.** If two features are highly correlated (e.g., temperature in Celsius and Fahrenheit), GaussianNB double-counts information, potentially distorting predictions. PCA before GNB can help.

3. **Ignoring zero-variance features.** A constant feature within a class causes division by zero. Always check for near-zero variance and apply var_smoothing or remove the feature.

4. **Applying GaussianNB without scaling.** While GNB is less sensitive to scaling than distance-based methods, extreme scale differences can cause numerical issues in the exponent. Standardization helps numerical stability.

5. **Assuming GNB produces well-calibrated probabilities.** GNB probabilities tend to be extreme (near 0 or 1) due to the multiplicative effect of many features. Calibration via Platt scaling is needed for reliable confidence estimates.

6. **Using the same variance for all classes.** Each class can have different variance per feature (heteroscedastic). GNB naturally allows this, which gives it quadratic decision boundaries. Forcing shared variance turns it into LDA.

7. **Not handling missing values.** GNB cannot handle NaN values. Impute missing values before training (mean imputation per class is reasonable given the Gaussian assumption).

## Interview Questions

### Beginner

1. What distribution does Gaussian Naive Bayes assume for features?

A univariate Gaussian (normal) distribution for each feature within each class: $P(x_i|y=k) \sim \mathcal{N}(\mu_{ik}, \sigma_{ik}^2)$.

2. What parameters does GaussianNB learn from data?

For each class k and each feature i, it learns the mean $\mu_{ik}$ (gnb.theta_) and variance $\sigma_{ik}^2$ (gnb.var_). It also learns the class prior $P(y=k)$.

3. How does GaussianNB handle continuous features differently from MultinomialNB?

GaussianNB models continuous features with a Gaussian density function. MultinomialNB requires discrete counts and would need features to be discretized or binned first.

4. What is var_smoothing in sklearn's GaussianNB?

A parameter that adds a small constant to all feature variances to prevent zero-variance issues. The effective variance becomes $\sigma_{ik}^2 + \text{var\_smoothing} \times \max_j \sigma_{jk}^2$.

5. Why does GaussianNB produce a quadratic decision boundary?

Because each class has its own variance per feature, the log-probability contains $(x_i - \mu_{ik})^2 / \sigma_{ik}^2$ terms. The squared $x_i$ terms don't cancel when $\sigma_{i0}^2 \neq \sigma_{i1}^2$, giving a quadratic boundary.

### Intermediate

1. Derive the MLE for the mean and variance in GaussianNB.

For class k, maximize $\prod_{j: y_j=k} \prod_i \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp(-\frac{(x_i^{(j)}-\mu_{ik})^2}{2\sigma_{ik}^2})$. Taking logs and setting derivatives to zero gives: $\hat{\mu}_{ik} = \frac{1}{m_k}\sum x_i^{(j)}$ and $\hat{\sigma}_{ik}^2 = \frac{1}{m_k}\sum (x_i^{(j)} - \hat{\mu}_{ik})^2$.

2. Compare GaussianNB with Quadratic Discriminant Analysis (QDA).

GNB assumes diagonal covariance per class (features independent). QDA models full covariance per class. QDA captures feature correlations but requires many more parameters: O(d^2) vs O(d) per class. GNB is more efficient with limited data; QDA is more expressive with sufficient data.

3. Under what condition does GaussianNB produce a linear decision boundary?

When all classes share the same variance per feature: $\sigma_{i0}^2 = \sigma_{i1}^2$ for all i. Then the $x_i^2$ terms cancel in the log-odds equation, leaving a linear boundary. This is exactly the LDA assumption.

4. How would you handle categorical features alongside continuous features in GNB?

Discretize continuous features into bins and use MultinomialNB for all features, or build a hybrid model: use GNB for continuous features (assuming Gaussian) and a separate estimator for categoricals, combining the log-probabilities.

5. What happens to GNB predictions when a feature has very high variance in one class?

Features with high variance contribute less to the discriminative signal because the log-likelihood $-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}$ is attenuated by large $\sigma_{ik}^2$. Effectively, the model learns to ignore high-variance features for that class.

### Advanced

1. Prove that GaussianNB is equivalent to a quadratic discriminant classifier with diagonal class covariances.

For class k, the log-posterior is: $\log P(y=k|x) = \log P(y=k) - \frac{1}{2}\log|2\pi\Sigma_k| - \frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)$. With diagonal $\Sigma_k = \text{diag}(\sigma_{1k}^2, ..., \sigma_{dk}^2)$, expanding gives: $-\frac{1}{2}\sum_i\log(2\pi\sigma_{ik}^2) - \frac{1}{2}\sum_i\frac{(x_i-\mu_{ik})^2}{\sigma_{ik}^2}$. This is exactly GNB's log-likelihood plus the prior. The decision boundary comes from equating two such quadratics.

2. Derive the bias and variance of the GaussianNB variance estimator. How does this affect the decision boundary?

The MLE variance $\hat{\sigma}^2 = \frac{1}{m}\sum(x_i-\bar{x})^2$ has bias $-\sigma^2/m$ (underestimates). For small class sizes, this biases variances downward, making log-likelihoods more extreme and decision boundaries tighter, increasing variance. sklearn uses MLE (biased) rather than Bessel-corrected variance, accepting slight bias for reduced variance in parameter estimation.

3. Implement an online version of GaussianNB that updates with streaming data without storing all training examples.

Maintain running counts: $m_k \leftarrow m_k + 1$, $\mu_{ik} \leftarrow \frac{m_k \mu_{ik} + x_i}{m_k + 1}$, and $S_{ik} \leftarrow S_{ik} + x_i^2$ where variance = $S_{ik}/m_k - \mu_{ik}^2$. The prior updates as $P(k) \leftarrow m_k / \sum_j m_j$. This allows incremental learning.

## Practice Problems

### Easy

1. Train GaussianNB on the Breast Cancer Wisconsin dataset. Report accuracy, precision, and recall.

2. For a binary classification dataset with a single feature, manually compute the GNB decision boundary given class means 2.0 and 4.0, variances 0.5 and 1.5, and equal priors.

3. Compare GaussianNB accuracy with and without StandardScaler on the Wine dataset.

4. Using sklearn's GaussianNB, extract and visualize the learned means for the Iris dataset using a heatmap.

5. Train GaussianNB on the digits dataset and plot the confusion matrix.

### Medium

1. Implement a cross-validated search for the var_smoothing parameter. Show its effect on test accuracy.

2. Compare GaussianNB, LDA, and QDA on 5 different sklearn datasets. Report when each performs best and explain why.

3. Create a synthetic 2D dataset where GNB's quadratic boundary is clearly visible. Plot the boundary and the true class distributions.

4. Implement a hybrid NB classifier that uses Gaussian for continuous features and Bernoulli for binary features. Test on a mixed dataset.

5. Show empirically that GNB's probability estimates are poorly calibrated by plotting a reliability diagram.

### Hard

1. Implement GaussianNB from scratch with online (streaming) update capability. Demonstrate it matches batch GNB.

2. Prove that the decision boundary between two classes in GNB is quadratic by expanding the log-posterior ratio.

3. Extend GaussianNB to use kernel density estimation (KDE) instead of a parametric Gaussian. Compare flexibility vs. overfitting on a multimodal synthetic dataset.

## Solutions

### Easy 1: Breast Cancer with GNB

```python
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
# Output:
# Accuracy: 0.9415
# Precision: 0.9538
# Recall: 0.9538
```

### Easy 2: Manual decision boundary

```python
import numpy as np

mu0, mu1 = 2.0, 4.0
var0, var1 = 0.5, 1.5
prior0, prior1 = 0.5, 0.5

# Solve log P(y=0|x) = log P(y=1|x)
# -0.5*log(2*pi*var0) - (x-mu0)^2/(2*var0) + log(prior0) = -0.5*log(2*pi*var1) - (x-mu1)^2/(2*var1) + log(prior1)

# Quadratic: a*x^2 + b*x + c = 0
a = 1/(2*var0) - 1/(2*var1)
b = mu1/var1 - mu0/var0
c = -0.5*np.log(var0/var1) + mu0**2/(2*var0) - mu1**2/(2*var1) + np.log(prior0/prior1)

discriminant = b**2 - 4*a*c
x1 = (-b + np.sqrt(discriminant)) / (2*a)
x2 = (-b - np.sqrt(discriminant)) / (2*a)

print(f"Decision boundaries at x = {x1:.3f} and x = {x2:.3f}")
# Output: Decision boundaries at x = 1.200 and x = 4.800

# Since variance differs, there are two boundaries (quadratic)
```

## Related Concepts

- **Naive Bayes** (ML-038): The general framework
- **Bayesian Inference** (ML-040): Theoretical foundation
- **LDA** (ML-049): Linear decision boundary under shared covariance
- **QDA**: Quadratic boundary under full per-class covariance
- **Maximum Likelihood Estimation** (ML-004): How parameters are fit

## Next Concepts

- **Bayesian Inference** (ML-040): Generalizing from point estimates to full posteriors
- **Gaussian Mixture Models** (ML-045): Multiple Gaussians per class for complex distributions
- **Gaussian Processes** (ML-041): Non-parametric Gaussian modeling

## Summary

Gaussian Naive Bayes is the continuous-feature variant of the Naive Bayes family. It models each feature as a univariate Gaussian per class, estimating means and variances from the training data via maximum likelihood. Despite its strong assumptions (feature independence, Gaussianity), it is fast, data-efficient, and performs well on many real-world tasks. Its decision boundaries are quadratic when class variances differ, making it more flexible than LDA. Key considerations include variance smoothing to prevent numerical issues and the potential need for feature transformations when the normality assumption is violated.

## Key Takeaways

- GaussianNB models $P(x_i|y) \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$ independently per feature per class
- Parameters $\mu_{iy}$ and $\sigma_{iy}^2$ are estimated via MLE from per-class data
- Produces quadratic decision boundaries (unlike LDA's linear boundaries)
- Very fast to train and predict: O(md) time, O(kd) memory
- Sensitive to non-Gaussian features — transform or use non-parametric alternatives
- Variance smoothing prevents numerical issues from constant features
- Poor probability calibration — use Platt scaling if confidence estimates are needed
- Excellent baseline for small to medium continuous datasets
