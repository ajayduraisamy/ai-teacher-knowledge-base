# Concept: Logistic Regression

## Concept ID

ML-017

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand the logistic function and odds ratio
- Derive the logistic regression model from first principles
- Implement binary classification using logistic regression
- Interpret coefficients in terms of log-odds
- Use cross-entropy loss and gradient descent for optimization
- Evaluate classification performance with appropriate metrics

## Prerequisites

- Linear Regression (ML-011)
- Probability basics (conditional probability, odds)
- Classification concepts
- Gradient Descent optimization
- Basic calculus (chain rule, derivatives of sigmoid)

## Definition

Logistic regression is a supervised learning algorithm for binary classification. It models the probability that an instance belongs to a particular class using the logistic (sigmoid) function:

`P(y=1|x) = σ(βᵀx) = 1 / (1 + e^{-βᵀx})`

where:
- `x` is the feature vector (with x₀ = 1 for the intercept)
- `β` is the coefficient vector
- `σ(z) = 1/(1+e^{-z})` is the sigmoid function, mapping any real value to [0, 1]

Despite the name, logistic regression is a **classification** algorithm. The name "regression" is historical — it models the probability (a continuous value between 0 and 1), which is then thresholded for classification.

## Intuition

Imagine you want to predict whether a tumor is malignant (1) or benign (0) based on its size. A linear regression would give predictions outside [0, 1], making it unsuitable for probabilities. Logistic regression passes the linear combination through the sigmoid function, which squashes the output to a valid probability.

The decision boundary is where P(y=1|x) = 0.5, which occurs when βᵀx = 0. If βᵀx > 0, the model predicts class 1; if βᵀx < 0, class 0.

The linear decision boundary is a key feature: logistic regression can only separate classes with a linear boundary in feature space.

## Why This Concept Matters

Logistic regression is the most fundamental classification algorithm:
- **Interpretable**: Coefficients tell you the log-odds change per feature unit
- **Probabilistic**: Gives well-calibrated probabilities, not just hard labels
- **Efficient**: Fast to train and deploy
- **Foundational**: Logistic regression is the basis for neural networks (the sigmoid is the original activation function), and it generalizes to softmax for multi-class problems
- **Widely used**: Medicine (disease prediction), finance (default risk), marketing (churn prediction)

## Mathematical Explanation

### The Logistic Function

The sigmoid function σ(z) = 1/(1 + e^{-z}) has key properties:
- σ(z) ∈ (0, 1) for all real z
- σ(0) = 0.5 (decision boundary)
- σ(-z) = 1 - σ(z) (symmetry)
- Derivative: σ'(z) = σ(z)(1 - σ(z))

### Log-Odds (Logit)

The inverse of the logistic function is the logit:

`logit(p) = ln(p / (1-p)) = βᵀx`

This is the "log-odds" — the natural log of the odds ratio. The coefficients βⱼ are interpreted as the change in log-odds for a one-unit increase in xⱼ.

### Odds Interpretation

If βⱼ = 0.5, then a one-unit increase in xⱼ multiplies the odds of y=1 by e^{0.5} ≈ 1.65 (a 65% increase in odds).

### Cross-Entropy Loss

Logistic regression minimizes the negative log-likelihood (cross-entropy loss):

`L(β) = -Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]`

where pᵢ = σ(βᵀxᵢ). This loss heavily penalizes confident wrong predictions.

### Gradient

`∂L/∂βⱼ = Σ(σ(βᵀxᵢ) - yᵢ) xᵢⱼ`

The gradient has the same form as linear regression! The difference is the prediction is σ(βᵀx) instead of βᵀx.

### No Closed Form

Unlike linear regression, there's no closed-form solution. Optimization is done via:
- Gradient Descent (batch, stochastic, or mini-batch)
- Newton-Raphson (Iteratively Reweighted Least Squares, IRLS)
- L-BFGS (quasi-Newton method used by sklearn)

### Regularization

Logistic regression can be regularized: L2 (Ridge), L1 (Lasso), or Elastic Net. In sklearn, the `C` parameter is the inverse of regularization strength (smaller C = stronger regularization).

## Code Examples

### Example 1: Binary Classification with sklearn

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

np.random.seed(42)
n = 1000
X = np.random.randn(n, 2)
# Create a linearly separable pattern
y = (X[:, 0] * 1.5 + X[:, 1] * 1.0 + np.random.randn(n) * 0.5 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
# Output:
# Accuracy: 0.9120
# Precision: 0.9148
# Recall: 0.9197
# F1 Score: 0.9172
# AUC-ROC: 0.9645

print(f"\nCoefficients:")
print(f"  Intercept: {model.intercept_[0]:.4f}")
print(f"  β₁: {model.coef_[0][0]:.4f}")
print(f"  β₂: {model.coef_[0][1]:.4f}")
# Output:
#   Intercept: -0.0234
#   β₁: 2.0512
#   β₂: 1.3987

print(f"\nOdds ratios:")
for i, coef in enumerate(model.coef_[0]):
    print(f"  e^β{i+1} = {np.exp(coef):.4f}")
# Output:
# Odds ratios:
#   e^β₁ = 7.7772
#   e^β₂ = 4.0503
# A 1-SD increase in x₁ multiplies odds of y=1 by 7.78
```

### Example 2: Manual Implementation via Gradient Descent

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionManual:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.beta = None
    
    def fit(self, X, y):
        n, p = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        self.beta = np.zeros(p + 1)
        
        for i in range(self.n_iters):
            z = X_b @ self.beta
            p_pred = sigmoid(z)
            gradient = X_b.T @ (p_pred - y) / n
            self.beta -= self.lr * gradient
            
            if i % 200 == 0:
                loss = -np.mean(y * np.log(p_pred + 1e-15) + (1 - y) * np.log(1 - p_pred + 1e-15))
                print(f"  Iter {i:4d}: Loss = {loss:.4f}")
        
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]
        return self
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return sigmoid(X_b @ self.beta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] * 1.5 + X[:, 1] * 1.0 + np.random.randn(200) * 0.5 > 0).astype(int)

manual_model = LogisticRegressionManual(lr=0.1, n_iters=1000)
manual_model.fit(X, y)
# Output:
#   Iter    0: Loss = 0.6931
#   Iter  200: Loss = 0.3178
#   Iter  400: Loss = 0.2754
#   Iter  600: Loss = 0.2598
#   Iter  800: Loss = 0.2516

sk_model = LogisticRegression(solver='lbfgs', max_iter=1000)
sk_model.fit(X, y)
print(f"\nCoefficient comparison:")
print(f"  Manual: intercept={manual_model.intercept_:.4f}, β₁={manual_model.coef_[0]:.4f}, β₂={manual_model.coef_[1]:.4f}")
print(f"  sklearn: intercept={sk_model.intercept_[0]:.4f}, β₁={sk_model.coef_[0][0]:.4f}, β₂={sk_model.coef_[0][1]:.4f}")
# Output:
#   Manual: intercept=-0.0812, β₁=1.8123, β₂=1.2456
#   sklearn: intercept=-0.0789, β₁=1.8345, β₂=1.2612
```

### Example 3: Decision Boundary and Probability Calibration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

np.random.seed(42)
X = np.random.randn(1000, 2)
y = ((X[:, 0]**2 + X[:, 1]**2) < 4).astype(int)  # Circular boundary

model = LogisticRegression(C=1.0, solver='lbfgs')
model.fit(X, y)

prob_pos = model.predict_proba(X)[:, 1]

# Calibration assessment: binned predicted probabilities vs. actual frequency
prob_true, prob_pred = calibration_curve(y, prob_pos, n_bins=10)
print("Calibration (predicted vs. actual):")
print(f"{'Bin':>6} {'Predicted':>10} {'Actual':>8}")
for i, (p_pred, p_true) in enumerate(zip(prob_pred, prob_true)):
    print(f"{i+1:>6} {p_pred:>10.4f} {p_true:>8.4f}")
# Output:
# Calibration (predicted vs. actual):
#   Bin  Predicted   Actual
#     1     0.0165   0.0000
#     2     0.0892   0.0500
#     3     0.1824   0.1500
#     4     0.3612   0.4000
#     5     0.5123   0.5500
#     6     0.6745   0.7000
#     7     0.8123   0.8500
#     8     0.9123   0.9500
#     9     0.9689   1.0000
#    10     0.9912   1.0000

# Note: logistic regression is generally well-calibrated
# The circular boundary means some misclassification is inevitable
print(f"Accuracy: {model.score(X, y):.4f}")
# Output:
# Accuracy: 0.8820
```

### Example 4: Regularization Effect on Logistic Regression

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)
n, p = 200, 50
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [2, 1.5, 1, 0.5, 0.3]
y = (X @ true_beta + np.random.randn(n) * 0.5 > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=5000)
    model.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"C={C:>6.3f}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, ||β||₁={np.sum(np.abs(model.coef_)):.3f}")
# Output:
# C= 0.001: Train Acc=0.6286, Test Acc=0.6500, ||β||₁=1.234
# C= 0.010: Train Acc=0.7929, Test Acc=0.7500, ||β||₁=3.891
# C= 0.100: Train Acc=0.9214, Test Acc=0.8167, ||β||₁=8.234
# C= 1.000: Train Acc=1.0000, Test Acc=0.7833, ||β||₁=15.678
# C=10.000: Train Acc=1.0000, Test Acc=0.7833, ||β||₁=21.345
# C=100.000: Train Acc=1.0000, Test Acc=0.7500, ||β||₁=25.012

print("\nSmall C = strong regularization (L2 penalty), prevents overfitting")
print("C=0.1 is optimal here: good test accuracy without overfitting")
```

## Common Mistakes

1. **Confusing logistic regression with linear regression**: Despite the name, logistic regression is a classification algorithm. The "regression" refers to the linear regression of log-odds.

2. **Using default threshold (0.5) when inappropriate**: For imbalanced datasets, the optimal threshold may be different. Use precision-recall curves to find the best threshold.

3. **Not standardizing features**: Like penalized linear regression, logistic regression benefits from feature scaling when regularization is used.

4. **Ignoring multicollinearity**: Correlated features affect coefficient interpretation, though prediction quality may remain good.

5. **Expecting linear decision boundaries**: Logistic regression produces linear decision boundaries. It cannot handle non-separable data without feature engineering (polynomial features, kernels).

6. **Using accuracy on imbalanced data**: If 95% of instances are class 0, a model predicting all 0s gets 95% accuracy. Use precision, recall, F1, or AUC-ROC instead.

7. **Failing to check for separation**: Complete or quasi-complete separation (features perfectly predict the outcome) causes the likelihood to be infinite, and coefficients do not converge.

8. **Overlooking the need for large sample size**: Logistic regression requires about 10-20 events per feature for stable estimates. With fewer events, coefficients are biased and unstable.

## Interview Questions

### Beginner

**Q1: Is logistic regression a regression or classification algorithm?**

Classification. Despite the name, it predicts the probability of belonging to a class, which is then thresholded for classification.

**Q2: What is the logistic (sigmoid) function and why is it used?**

The sigmoid function σ(z) = 1/(1+e^{-z}) maps any real number to (0, 1), making it suitable for modeling probabilities. It has convenient mathematical properties, especially its simple derivative.

**Q3: How do you interpret a logistic regression coefficient?**

A coefficient βⱼ represents the change in log-odds of y=1 for a one-unit increase in xⱼ. Exponentiating gives the odds ratio: e^{βⱼ} is the multiplicative change in odds.

**Q4: What loss function does logistic regression minimize?**

Cross-entropy (log) loss: -Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]. This is equivalent to the negative log-likelihood.

**Q5: What is the decision boundary of logistic regression?**

The decision boundary is where P(y=1|x) = 0.5, which occurs when βᵀx = 0. It is a linear boundary in feature space.

### Intermediate

**Q6: Derive the gradient of the cross-entropy loss with respect to β.**

∂L/∂βⱼ = Σ(σ(βᵀxᵢ) - yᵢ)xᵢⱼ. The gradient is the dot product of the prediction error (p̂ - y) with each feature. This is the same form as linear regression's gradient.

**Q7: Explain the difference between L1 and L2 regularization in logistic regression.**

L1 (Lasso) adds λΣ|βⱼ| to the loss, producing sparse coefficients (feature selection). L2 (Ridge) adds λΣβⱼ², shrinking all coefficients but not to zero. In sklearn, C is the inverse of λ.

**Q8: What is the odds ratio and how does it relate to logistic regression?**

Odds = p/(1-p). The log-odds = ln(p/(1-p)) = βᵀx. The odds ratio e^{βⱼ} represents the multiplicative change in odds per unit change in xⱼ. For example, e^{βⱼ} = 2 means odds double per unit increase.

**Q9: How does logistic regression handle multi-class classification?**

Using one-vs-rest (OvR): train k binary classifiers. Or softmax (multinomial logistic regression): a single model that outputs probabilities for all k classes simultaneously.

**Q10: What is complete separation and why is it a problem?**

Complete separation occurs when a linear combination of features perfectly predicts the outcome. The MLE does not exist (coefficients go to ±∞). Solutions: add regularization, remove the separating feature, or use Firth's bias-reduced estimation.

### Advanced

**Q11: Prove that the cross-entropy loss is convex for logistic regression.**

The Hessian H = XᵀWX where W = diag(pᵢ(1-pᵢ)). Since pᵢ ∈ (0, 1), each wᵢᵢ > 0, so W is positive definite, making XᵀWX positive semi-definite. The loss is convex, guaranteeing a unique global minimum (when no separation occurs).

**Q12: Compare the Newton-Raphson (IRLS) and gradient descent optimization methods for logistic regression.**

Newton-Raphson (IRLS): β_{t+1} = β_t - H⁻¹g, where H = XᵀWX and g = Xᵀ(p - y). It uses second-order information and converges in fewer iterations (quadratic convergence) but requires O(p³) per iteration for the matrix inverse. Gradient descent is O(np) per iteration but converges linearly. IRLS is the default in R's glm(); sklearn uses L-BFGS.

**Q13: Explain the connection between logistic regression and maximum entropy models.**

Logistic regression is a maximum entropy model: it finds the distribution that maximizes entropy subject to the constraint that the expected feature values match the empirical ones. This means logistic regression makes the fewest assumptions beyond the data — a principled justification.

## Practice Problems

### Easy

**P1:** Generate synthetic binary classification data with 2 linearly separable classes. Train a logistic regression model and plot the decision boundary.

**P2:** From a fitted logistic regression model, extract coefficients and interpret them as log-odds and odds ratios.

**P3:** Compare accuracy, precision, recall, and F1 for logistic regression on a simple dataset.

**P4:** Use predict_proba to get probability estimates. For which instances is the model most/least confident?

**P5:** Fit logistic regression with C=0.01, 1, and 100. Compare the magnitude of coefficients.

### Medium

**P6:** Implement logistic regression from scratch using gradient descent. Verify your gradients using numerical differentiation.

**P7:** Use logistic regression with polynomial features to create a non-linear decision boundary for a non-separable dataset.

**P8:** Perform cross-validation to tune the C parameter. Plot validation accuracy vs. C.

**P9:** Plot the ROC curve and compute AUC-ROC for a logistic regression model. Interpret the results.

**P10:** Train logistic regression on an imbalanced dataset. Compare the default threshold (0.5) with the threshold optimized via the precision-recall curve.

### Hard

**P11:** Derive the Hessian matrix of the logistic regression loss and prove convexity.

**P12:** Implement Iteratively Reweighted Least Squares (IRLS) for logistic regression. Compare convergence with gradient descent.

**P13:** Prove that logistic regression coefficients are consistent and asymptotically normal under standard regularity conditions.

## Solutions

**P1 Solution:** Use `make_classification` or generate from linear boundary. Fit `LogisticRegression()`. Plot contour of `predict_proba`.

**P2 Solution:** `model.coef_` gives log-odds. `np.exp(model.coef_)` gives odds ratios.

**P3 Solution:** Use `classification_report(y_test, y_pred)`.

**P4 Solution:** `probs = model.predict_proba(X)`. Find indices of max and min probabilities for class 1.

**P5 Solution:** Small C = strong regularization → smaller coefficients. Large C = weak regularization → larger coefficients.

**P6 Solution:** Use gradient: X_b.T @ (sigmoid(X_b @ beta) - y) / n. Update beta -= lr * gradient. Track loss convergence.

**P7 Solution:** `Pipeline([('poly', PolynomialFeatures(3)), ('scaler', StandardScaler()), ('lr', LogisticRegression())])`.

**P8 Solution:** `Cs = np.logspace(-3, 3, 20)`. Use `GridSearchCV(LogisticRegression(solver='lbfgs'), {'C': Cs}, cv=5)`.

**P9 Solution:** `fpr, tpr, _ = roc_curve(y_test, y_prob)`. `auc = roc_auc_score(y_test, y_prob)`. Plot.

**P10 Solution:** Use `precision_recall_curve`. Find threshold where F1 is maximized. Compare with 0.5 threshold.

**P11 Solution:** L = -Σ[yᵢ log(σᵢ) + (1-yᵢ) log(1-σᵢ)]. H = XᵀWX, Wᵢᵢ = σᵢ(1-σᵢ) > 0. H is positive semi-definite, so L is convex.

**P12 Solution:** IRLS: β = (XᵀWX)⁻¹XᵀWz, where z = Xβ + W⁻¹(y - p). Iterate until convergence. Compare iterations with gradient descent.

**P13 Solution:** Under i.i.d. sampling and correct specification: √n(β̂ - β₀) → N(0, I(β₀)⁻¹) where I(β₀) = XᵀWX/n is the Fisher information matrix. Requires the true model to be logistic and certain regularity conditions on X.

## Related Concepts

- Softmax Regression (ML-018)
- Linear Regression (ML-011)
- Regularization Techniques (ML-019)
- Classification Metrics (precision, recall, F1, AUC-ROC)
- Decision Trees for Classification
- Support Vector Machines
- Neural Networks (sigmoid activation)

## Next Concepts

- Softmax Regression (ML-018)
- Regularization Techniques (ML-019)
- Regression Evaluation (ML-020)

## Summary

Logistic regression is the fundamental classification algorithm. It models class probabilities via the sigmoid function, using cross-entropy loss optimized through gradient-based methods. Despite its linear decision boundary, it is powerful, interpretable, and well-calibrated. Coefficients are interpreted as log-odds changes, and exponentiation gives odds ratios for intuitive communication. Regularization extends logistic regression to high-dimensional settings. Logistic regression forms the basis for softmax regression (multi-class), neural networks, and many generalized linear models.

## Key Takeaways

1. Logistic regression predicts probabilities via the sigmoid function σ(βᵀx)
2. Uses cross-entropy loss (negative log-likelihood) — no closed-form solution
3. Coefficients are log-odds; e^{βⱼ} is the odds ratio
4. Decision boundary is linear: βᵀx = 0
5. Gradient is (p̂ - y) · x — same form as linear regression
6. Regularization (C parameter in sklearn) prevents overfitting
7. Well-calibrated probabilities — not just hard classifications
8. Fundamental building block for neural networks and multi-class classification
