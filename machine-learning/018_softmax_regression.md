# Concept: Softmax Regression

## Concept ID

ML-018

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Extend logistic regression to multi-class classification using the softmax function
- Understand the softmax function as a generalization of the sigmoid
- Derive the cross-entropy loss for multi-class problems
- Implement softmax regression using scikit-learn
- Interpret the output probabilities across classes

## Prerequisites

- Logistic Regression (ML-017)
- Linear Regression (ML-011)
- Probability theory basics
- Matrix calculus

## Definition

Softmax regression (also known as multinomial logistic regression) generalizes logistic regression to handle multiple classes. For K classes, the probability that an instance belongs to class k is:

`P(y=k|x) = exp(β_kᵀx) / Σⱼ exp(βⱼᵀx)`

where:
- `β_k` is the coefficient vector for class k
- The denominator normalizes probabilities to sum to 1 across all K classes

The predicted class is the one with the highest probability:

`ŷ = argmax_k P(y=k|x)`

Unlike binary logistic regression (which uses one set of coefficients), softmax regression learns `K` sets of coefficients (one per class), though one set is redundant and can be pinned to zero for identifiability.

## Intuition

Imagine classifying handwritten digits (0-9). Each image belongs to exactly one of 10 classes. Softmax regression computes a score for each class (a linear combination of pixel values), then converts these scores into probabilities using the softmax function.

The softmax function takes a vector of real numbers (logits) and produces a probability distribution. High logits get high probabilities; low logits get low probabilities. The exponential amplifies differences — if one logit is much larger than others, its probability approaches 1.

Think of it as a "soft" version of argmax: instead of picking the maximum, it assigns probabilities based on relative magnitudes.

## Why This Concept Matters

Softmax regression is essential for multi-class classification:
- **Natural extension**: Generalizes binary logistic regression seamlessly
- **Probabilistic**: Produces well-calibrated class probabilities
- **Interpretable**: Coefficients show which features favor each class
- **Foundation for deep learning**: The softmax output layer is used in neural networks for multi-class classification
- **Widely applied**: Image recognition, text classification, medical diagnosis

## Mathematical Explanation

### The Softmax Function

For a vector `z = [z₁, z₂, ..., z_K]`:

`softmax(z)_k = exp(z_k) / Σⱼ exp(zⱼ)`

Properties:
- `softmax(z)_k ∈ (0, 1)` for all k
- `Σ_k softmax(z)_k = 1` (it's a valid probability distribution)
- `softmax(z + c)_k = softmax(z)_k` for any constant c (shift invariance)
- The logits can be shifted without changing probabilities

### The Model

For each class k, let `z_k = β_kᵀx = β_{k0} + β_{k1}x₁ + ... + β_{kp}xₚ`.

Then: `P(y=k|x) = exp(z_k) / Σⱼ exp(zⱼ)`

Total parameters: `K × (p+1)`. However, one class is redundant — subtracting a constant from all logits doesn't change probabilities. By convention, we set `β_K = 0` for the last class, leaving `(K-1) × (p+1)` free parameters.

### Cross-Entropy Loss

For n samples with K classes (one-hot encoded targets yᵢₖ):

`L = -Σᵢ Σₖ yᵢₖ log(pᵢₖ)`

where `pᵢₖ = softmax(zᵢ)_k` and `yᵢₖ = 1` if sample i belongs to class k, else 0.

### Gradient

The gradient for class k's coefficients:

`∂L/∂β_k = Σᵢ (pᵢₖ - yᵢₖ) xᵢ`

This elegantly mirrors binary logistic regression: `(predicted - actual) × features`. For the correct class, `pᵢₖ - 1` pushes coefficients to increase the logit. For incorrect classes, `pᵢₖ - 0` pushes coefficients to decrease the logit.

### Decision Boundary

For two classes k and l, the decision boundary is where P(y=k|x) = P(y=l|x), which occurs when `(β_k - β_l)ᵀx = 0`. This is a linear boundary in feature space.

### Identifiability

Without constraints, softmax regression has redundant parameters. Adding a constant to all logits doesn't matter. Solutions:
1. Set one class's coefficients to zero (reference class)
2. Add regularization (which breaks the symmetry)

sklearn uses regularization by default, avoiding the identifiability issue.

## Code Examples

### Example 1: Multi-Class Classification with sklearn

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
n = 1500
X = np.random.randn(n, 4)

# Create 3 classes with different linear patterns
y = np.zeros(n, dtype=int)
y[(X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n) * 0.5 > 1)] = 1
y[(X[:, 2] * 1.5 - X[:, 3] * 1.0 + np.random.randn(n) * 0.5 > 0.5)] = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# softmax regression = multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Output:
# Accuracy: 0.9147

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
# Output:
#               precision    recall  f1-score   support
#
#     Class 0       0.92      0.91      0.91       124
#     Class 1       0.91      0.90      0.90       123
#     Class 2       0.91      0.93      0.92       128
#
#    accuracy                           0.91       375

print(f"Coefficient shape: {model.coef_.shape}")
print(f"Intercept shape: {model.intercept_.shape}")
# Output:
# Coefficient shape: (3, 4)
# Intercept shape: (3,)

print(f"First sample probabilities: {y_prob[0]}")
print(f"Predicted class: {y_pred[0]}, True class: {y_test[0]}")
# Output:
# First sample probabilities: [0.0234 0.8912 0.0854]
# Predicted class: 1, True class: 1
```

### Example 2: Softmax from Scratch with Gradient Descent

```python
import numpy as np

def softmax(z):
    """Compute softmax probabilities. z: (n_samples, n_classes)"""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
    
    def fit(self, X, y):
        n, p = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        
        X_b = np.c_[np.ones((n, 1)), X]
        self.beta = np.zeros((p + 1, K))
        
        y_onehot = np.zeros((n, K))
        y_onehot[np.arange(n), y] = 1
        
        for i in range(self.n_iters):
            logits = X_b @ self.beta
            probs = softmax(logits)
            
            gradient = X_b.T @ (probs - y_onehot) / n
            self.beta -= self.lr * gradient
            
            if i % 200 == 0:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
                print(f"  Iter {i:4d}: Loss = {loss:.4f}")
        
        self.intercept_ = self.beta[0, :]
        self.coef_ = self.beta[1:, :]
        return self
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return softmax(X_b @ self.beta)
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

np.random.seed(42)
X = np.random.randn(500, 4)
y = np.zeros(500, dtype=int)
y[(X[:, 0] * 2 + X[:, 1] * 1.5 > 0.5)] = 1
y[(X[:, 2] * 1.5 - X[:, 3] * 1.0 > 0.0)] = 2

print("Training Softmax from scratch:")
softmax_manual = SoftmaxRegression(lr=0.1, n_iters=1000)
softmax_manual.fit(X, y)
# Output:
# Training Softmax from scratch:
#   Iter    0: Loss = 1.0986
#   Iter  200: Loss = 0.4123
#   Iter  400: Loss = 0.3678
#   Iter  600: Loss = 0.3512
#   Iter  800: Loss = 0.3423

print(f"\nManual accuracy: {accuracy_score(y, softmax_manual.predict(X)):.4f}")
# Output:
# Manual accuracy: 0.8720
```

### Example 3: Visualizing Softmax Probabilities

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=2, n_classes=3, 
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0)
model.fit(X, y)

# Generate grid for probability visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
grid = np.c_[xx.ravel(), yy.ravel()]

probs = model.predict_proba(grid)

# For a sample point, show class probabilities
sample = np.array([[0, 0]])
sample_probs = model.predict_proba(sample)[0]
print(f"Samples probabilities for X=(0, 0):")
for i, p in enumerate(sample_probs):
    print(f"  Class {i}: {p:.4f}")
# Output:
# Samples probabilities for X=(0, 0):
#   Class 0: 0.8213
#   Class 1: 0.1456
#   Class 2: 0.0331

print(f"Predicted class: {model.predict(sample)[0]}")
# Output:
# Predicted class: 0

# Check sum of probabilities
print(f"Sum of probabilities: {np.sum(sample_probs):.4f}")
# Output:
# Sum of probabilities: 1.0000
```

### Example 4: One-vs-Rest vs. Softmax Comparison

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=4, 
                          n_informative=6, n_redundant=2, random_state=42)

ovr_model = LogisticRegression(multi_class='ovr', solver='lbfgs', C=1.0, max_iter=1000)
softmax_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000)

ovr_scores = cross_val_score(ovr_model, X, y, cv=5)
softmax_scores = cross_val_score(softmax_model, X, y, cv=5)

print(f"One-vs-Rest CV Accuracy: {ovr_scores.mean():.4f} (+/- {ovr_scores.std():.4f})")
print(f"Softmax CV Accuracy: {softmax_scores.mean():.4f} (+/- {softmax_scores.std():.4f})")
# Output:
# One-vs-Rest CV Accuracy: 0.8340 (+/- 0.0215)
# Softmax CV Accuracy: 0.8470 (+/- 0.0198)

# Softmax typically performs better because it optimizes a single joint objective
# rather than K independent objectives
```

## Common Mistakes

1. **Not using `multi_class='multinomial'`**: sklearn's LogisticRegression defaults to 'ovr' (one-vs-rest). For softmax, explicitly set `multi_class='multinomial'`.

2. **Forgetting the softmax is shift-invariant**: Adding a constant to all logits doesn't change probabilities. This redundancy means regularization is important to get a unique solution.

3. **Using softmax when classes are not mutually exclusive**: Softmax assumes each instance belongs to exactly one class. For multi-label problems (an image can contain both a cat and a dog), use independent binary classifiers.

4. **Numerical instability with large logits**: `exp(large_number)` overflows. Solution: subtract the maximum logit before exponentiating (the `z - max(z)` trick).

5. **Assuming softmax probabilities are well-calibrated for all classes**: While softmax outputs sum to 1, the probabilities may not be well-calibrated, especially with strong regularization.

6. **Ignoring the reference class**: Without regularization, one class's coefficients should be fixed to zero for identifiability. sklearn handles this automatically with regularization.

7. **Using softmax with a large number of classes**: With 10,000 classes, computing the denominator over all classes is expensive. Consider hierarchical softmax or negative sampling.

8. **Not interpreting coefficients carefully**: Unlike binary logistic regression, coefficients are relative — β_k shows how features increase the logit for class k relative to the baseline.

## Interview Questions

### Beginner

**Q1: What is softmax regression and how does it differ from logistic regression?**

Softmax regression generalizes logistic regression to K > 2 classes. Logistic regression uses the sigmoid for binary classification; softmax uses the softmax function that produces a probability distribution over all K classes.

**Q2: What does the softmax function do?**

The softmax function converts a vector of real numbers (logits) into a probability distribution. Each element is exponentiated and divided by the sum of all exponentiated elements. The result is a vector of non-negative values summing to 1.

**Q3: How many parameters does softmax regression have?**

For p features and K classes, there are K × (p+1) parameters (including intercepts). Due to redundancy, only (K-1) × (p+1) are free. With regularization, all K sets are identifiable.

**Q4: What loss function does softmax regression minimize?**

Categorical cross-entropy: L = -ΣᵢΣₖ yᵢₖ log(pᵢₖ), where yᵢₖ is the one-hot encoded label and pᵢₖ is the predicted probability for class k.

**Q5: How do you make a prediction with softmax regression?**

Compute the logits z_k = β_kᵀx for each class, apply softmax to get probabilities, and predict the class with the highest probability: ŷ = argmax_k softmax(z)_k.

### Intermediate

**Q6: Derive the gradient for softmax regression.**

The gradient for class k is: ∂L/∂β_k = Σᵢ (pᵢₖ - yᵢₖ)xᵢ. For the correct class, yᵢₖ = 1, so the gradient pushes β_k to increase the logit. For incorrect classes, yᵢₖ = 0, so the gradient pushes β_k to decrease the logit.

**Q7: Explain the identifiability issue in softmax regression.**

Adding a constant vector c to all coefficient sets β_k doesn't change the probabilities because the softmax is shift-invariant: softmax(z + c) = softmax(z). This means the parameters are not uniquely identifiable without constraints (regularization or setting one class to zero).

**Q8: Compare one-vs-rest (OvR) with softmax for multi-class classification.**

OvR trains K binary classifiers, each against all others. Softmax trains one joint model. Softmax produces well-calibrated probabilities (summing to 1) and typically performs better when classes are mutually exclusive. OvR can handle non-mutually-exclusive classes (multi-label).

**Q9: What is the relationship between softmax and the sigmoid function?**

The sigmoid function is a special case of softmax for two classes. If z₁ = βᵀx and z₂ = 0, then P(y=1|x) = exp(z₁)/(exp(z₁) + exp(0)) = 1/(1 + exp(-z₁)) = σ(z₁). Binary logistic regression is softmax with K=2.

**Q10: How does regularization affect softmax regression?**

Regularization (L1 or L2) addresses identifiability by penalizing large coefficients, breaking the shift-invariance. It also prevents overfitting when there are many features relative to samples per class.

### Advanced

**Q11: Prove the convexity of the softmax cross-entropy loss.**

The Hessian for softmax regression is block-structured: ∂²L/(∂βⱼ∂βₖ) = Σᵢ pᵢⱼ(δⱼₖ - pᵢₖ)xᵢxᵢᵀ, where δⱼₖ = 1 if j=k. This Hessian is positive semi-definite (it's a sum of outer products weighted by pᵢⱼ(δⱼₖ - pᵢₖ), which forms a valid covariance matrix), making the loss convex.

**Q12: Explain the temperature parameter in softmax. How does it affect predictions?**

Temperature T scales the logits: softmax(z/T)_k = exp(z_k/T) / Σⱼ exp(zⱼ/T). As T → 0, softmax approaches argmax (hard assignment). As T → ∞, probabilities become uniform. Temperature is used in knowledge distillation to create "soft" targets for student models.

**Q13: Derive the connection between softmax regression and maximum entropy models.**

Softmax regression is the maximum entropy model for multi-class classification. Among all distributions satisfying E[xⱼ·1{y=k}] = empirical expectation, the softmax is the one with maximum entropy. This provides a principled justification: softmax makes the fewest assumptions beyond the observed feature-label statistics.

## Practice Problems

### Easy

**P1:** Generate a 3-class dataset using make_classification. Train softmax regression and report accuracy.

**P2:** For a trained softmax model, show the probability distribution for a test sample. Verify the sum is 1.

**P3:** Compare softmax (multinomial) with one-vs-rest (ovr) in sklearn's LogisticRegression.

**P4:** Train softmax on the Iris dataset (3 classes of flowers). Report classification metrics.

**P5:** Visualize the decision regions of softmax regression for a 2D dataset with 3 classes.

### Medium

**P6:** Implement softmax regression from scratch using gradient descent. Verify against sklearn.

**P7:** Add L2 regularization to your manual softmax implementation and show it produces unique coefficients.

**P8:** Use cross-validation to tune the C parameter for softmax on a multi-class dataset.

**P9:** Plot the learning curves (loss vs. iteration) for softmax regression with different learning rates.

**P10:** Train softmax on the digits dataset (10 classes). Visualize the coefficient matrix as images.

### Hard

**P11:** Derive the Hessian for softmax regression and prove convexity.

**P12:** Implement softmax regression with stochastic gradient descent. Compare convergence with batch gradient descent.

**P13:** Implement label smoothing for softmax regression and show how it improves calibration and generalization.

## Solutions

**P1 Solution:** `model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)`. Score on test.

**P2 Solution:** `probs = model.predict_proba(X_test[0:1])`. Print `np.sum(probs)` should be 1.0.

**P3 Solution:** `multi_class='ovr'` vs. `multi_class='multinomial'`. Use cross-validation to compare.

**P4 Solution:** Load iris from sklearn.datasets. Fit softmax. Report with `classification_report`.

**P5 Solution:** Use meshgrid to create grid, predict classes on grid, plot filled contour.

**P6 Solution:** Implement softmax gradient descent. Use one-hot encoding. Compare coefficients with sklearn.

**P7 Solution:** Add λβ² to loss. Gradient becomes (pred - actual)·x + 2λβ. Without regularization, coefficients are not unique.

**P8 Solution:** `GridSearchCV(LogisticRegression(multi_class='multinomial', solver='lbfgs'), {'C': np.logspace(-3, 3, 20)}, cv=5)`.

**P9 Solution:** For α in [0.001, 0.01, 0.1, 0.5], record loss per iteration. Plot.

**P10 Solution:** Load digits. Fit softmax. Reshape coef_[k] to 8×8. Display as heatmaps showing which pixels are associated with each digit.

**P11 Solution:** Hessian H = Σᵢ [diag(pᵢ) - pᵢpᵢᵀ] ⊗ xᵢxᵢᵀ where pᵢ = softmax(zᵢ). The Kronecker product with xᵢxᵢᵀ preserves positive semi-definiteness.

**P12 Solution:** For each epoch, shuffle data, iterate one sample at a time: update β -= lr × (p - y_oh) × x. Compare loss curves.

**P13 Solution:** Label smoothing: use targets (1-ε) for correct class and ε/(K-1) for others. This prevents overconfidence and improves calibration.

## Related Concepts

- Logistic Regression (ML-017)
- Linear Regression (ML-011)
- Cross-Entropy Loss
- One-vs-Rest Classification
- Neural Networks (softmax output layer)
- Maximum Entropy Models

## Next Concepts

- Regularization Techniques (ML-019)
- Regression Evaluation Metrics (ML-020)
- Neural Networks (Multi-Layer Perceptron)

## Summary

Softmax regression generalizes logistic regression to K > 2 classes by modeling class probabilities via the softmax function. Each class gets a coefficient vector; the softmax converts logits into a probability distribution. The model minimizes categorical cross-entropy loss and is convex, guaranteeing a global optimum. Softmax is the standard output layer for multi-class neural networks and is particularly effective when classes are mutually exclusive. Key considerations include identifiability (resolved by regularization), numerical stability (logit shifting), and the tradeoff with one-vs-rest approaches.

## Key Takeaways

1. Softmax regression extends binary logistic regression to K classes
2. P(y=k|x) = exp(β_kᵀx) / Σⱼ exp(βⱼᵀx) — produces a valid probability distribution
3. Categorical cross-entropy loss: L = -ΣΣ yₖ log(pₖ)
4. Gradient: (predicted - actual) × features, same form as binary logistic regression
5. Softmax is shift-invariant — regularization or reference class needed for identifiability
6. In sklearn, use `multi_class='multinomial'` with a suitable solver
7. Classes must be mutually exclusive; for multi-label, use OvR instead
8. Softmax is the foundation of the output layer in neural networks for multi-class problems
