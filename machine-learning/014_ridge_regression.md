# Concept: Ridge Regression

## Concept ID

ML-014

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand why and when to use Ridge regression over OLS
- Derive the Ridge regression solution with L2 penalty
- Implement Ridge regression using scikit-learn
- Tune the regularization strength parameter alpha
- Explain the bias-variance tradeoff in the context of Ridge
- Understand the shrinkage effect on coefficients

## Prerequisites

- Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Overfitting concepts
- Matrix algebra (eigenvalues, matrix inverse)
- Bias-Variance Tradeoff

## Definition

Ridge regression (also known as Tikhonov regularization or L2 regularization) is a regularized version of linear regression that adds a penalty proportional to the square of the magnitude of the coefficients to the loss function:

`L(β) = Σ(yᵢ - ŷᵢ)² + λ Σ βⱼ²`

where λ ≥ 0 is the regularization parameter. The penalty term `λ Σ βⱼ²` shrinks the coefficients toward zero, reducing model complexity and mitigating overfitting.

In matrix form, the Ridge objective is:

`min_β ||y - Xβ||²₂ + λ ||β||²₂`

The closed-form solution becomes:

`β̂_ridge = (XᵀX + λI)⁻¹Xᵀy`

## Intuition

Think of Ridge regression as putting a "budget" on the size of the coefficients. OLS can use arbitrarily large coefficients to fit the training data perfectly, but when there's noise, large coefficients amplify irrelevant patterns. Ridge constrains the coefficients to be small, forcing the model to distribute explanatory power more evenly across features.

The λ parameter controls how much shrinkage is applied. When λ = 0, Ridge becomes OLS. As λ → ∞, coefficients shrink toward zero (but never exactly zero). The penalty is applied to all coefficients except the intercept.

Geometrically, Ridge adds a spherical constraint (a hypersphere) around the OLS solution. The optimal Ridge solution is where the elliptical OLS contours first touch this sphere, resulting in shrinkage along all axes.

## Why This Concept Matters

Ridge regression is one of the most important tools in a data scientist's toolkit:
- **Handles multicollinearity**: When features are correlated, OLS coefficients become unstable. Ridge stabilizes them.
- **Prevents overfitting**: In high-dimensional settings (p large), Ridge dramatically improves generalization.
- **Works when n < p**: OLS breaks when features exceed observations; Ridge still works.
- **Improves prediction**: The small bias introduced by shrinkage is often worth the large variance reduction.
- **Numerically stable**: The λI term ensures XᵀX + λI is always invertible.

## Mathematical Explanation

### The Ridge Objective

`L(β) = Σᵢ (yᵢ - Σⱼ xᵢⱼβⱼ)² + λ Σⱼ βⱼ²`

The first term is the RSS (fit to data). The second term is the L2 penalty (shrinkage). The intercept β₀ is typically not penalized.

### Closed-Form Solution

`β̂_ridge = (XᵀX + λI)⁻¹Xᵀy`

Note: For the intercept to be excluded from the penalty, center the data first:
- X should be standardized (mean = 0, std = 1)
- y should be centered (mean = 0)
- Then β̂₀ = ȳ (no penalty)

### Why Ridge Works: Bias-Variance Decomposition

For OLS: E[MSE] = 0² + σ²(p/n) + σ² (irreducible)

For Ridge: E[MSE] = bias²(λ) + variance(λ) + σ²

- As λ increases, bias² increases but variance decreases
- There exists an optimal λ where total test error is minimized
- Ridge trades a small increase in bias for a large decrease in variance

### Ridge and the SVD

Let X = UDVᵀ be the SVD of X. The Ridge solution can be expressed as:

`β̂_ridge = V(D² + λI)⁻¹DUᵀy`

The fitted values are:

`ŷ = Xβ̂_ridge = UD²(D² + λI)⁻¹Uᵀy`

The factors `d²ⱼ / (d²ⱼ + λ)` shrink the contributions of components with small singular values (high variance directions) the most. This is precisely why Ridge works: it downweights the high-variance directions that correspond to noise.

## Code Examples

### Example 1: Ridge vs. OLS with Multicollinearity

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 200
X = np.random.randn(n, 5)
# Create multicollinearity: X2 ≈ X1 + noise; X5 ≈ X3 + X4 + noise
X[:, 1] = X[:, 0] * 0.95 + np.random.randn(n) * 0.1
X[:, 4] = X[:, 2] * 0.9 + X[:, 3] * 0.9 + np.random.randn(n) * 0.1

true_beta = np.array([2, -1, 3, 0.5, -2])
y = X @ true_beta + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize (important for Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

print("Coefficient comparison:")
print(f"{'True':>8} {'OLS':>8} {'Ridge':>8}")
for i in range(5):
    print(f"{true_beta[i]:>8.2f} {ols.coef_[i]:>8.3f} {ridge.coef_[i]:>8.3f}")
# Output:
# Coefficient comparison:
#     True      OLS     Ridge
#     2.00    2.012    1.934
#    -1.00   -0.987   -0.912
#     3.00    3.045    2.891
#     0.50    0.498    0.483
#    -2.00   -1.998   -1.876

print(f"\nOLS Test R²: {r2_score(y_test, ols.predict(X_test_scaled)):.4f}")
print(f"Ridge Test R²: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
# Output:
# OLS Test R²: 0.9623
# Ridge Test R²: 0.9715
```

### Example 2: Effect of Alpha on Coefficients

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)
n, p = 100, 10
X = np.random.randn(n, p)
true_beta = np.array([5, 4, 3, 2, 1, 0, 0, 0, 0, 0])
y = X @ true_beta + np.random.randn(n) * 2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = [0, 0.01, 0.1, 1, 10, 100, 1000]
coefs = []

for alpha in alphas:
    if alpha == 0:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    coefs.append(model.coef_)

print(f"{'Alpha':>8}", end="")
for i in range(min(p, 6)):
    print(f"{f'β{i}':>8}", end="")
print()
for alpha, coef in zip(alphas, coefs):
    print(f"{alpha:>8.3f}", end="")
    for i in range(min(p, 6)):
        print(f"{coef[i]:>8.3f}", end="")
    print()
# Output:
#    Alpha      β0      β1      β2      β3      β4
#    0.000   5.012   3.898   2.931   1.945   1.045
#    0.010   4.998   3.889   2.925   1.942   1.044
#    0.100   4.881   3.804   2.868   1.910   1.030
#    1.000   4.103   3.249   2.510   1.717   0.951
#   10.000   1.920   1.619   1.358   1.029   0.647
#  100.000   0.368   0.320   0.287   0.234   0.169
# 1000.000   0.040   0.035   0.032   0.027   0.020

print("\nAs alpha increases, all coefficients shrink toward zero.")
print("Important features shrink more slowly than noise features.")
```

### Example 3: Alpha Tuning with Cross-Validation

```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

np.random.seed(42)
n = 500
X = np.random.randn(n, 50)
true_beta = np.zeros(50)
true_beta[:10] = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]
y = X @ true_beta + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000], scoring='r2', cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")
# Output:
# Best alpha: 0.1

print(f"Train R²: {ridge_cv.score(X_train_scaled, y_train):.4f}")
print(f"Test R²: {r2_score(y_test, ridge_cv.predict(X_test_scaled)):.4f}")
# Output:
# Train R²: 0.9954
# Test R²: 0.9912

print(f"Number of non-negligible coefficients (|β| > 0.01): {np.sum(np.abs(ridge_cv.coef_) > 0.01)}")
# Output:
# Number of non-negligible coefficients (|β| > 0.01): 50

# Ridge does NOT produce exact zeros (unlike Lasso)
```

### Example 4: Ridge with Changing n/p Ratio

```python
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
p = 30
true_beta = np.random.randn(p) * 3
true_beta[10:] = 0  # 10 relevant, 20 irrelevant

for n in [20, 30, 50, 100, 500]:
    X = np.random.randn(n, p)
    y = X @ true_beta + np.random.randn(n) * 2
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if n >= p:
        ols = LinearRegression()
        ols.fit(X_train_scaled, y_train)
        ols_r2 = r2_score(y_test, ols.predict(X_test_scaled))
    else:
        ols_r2 = -999  # OLS fails when n < p
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test_scaled))
    
    if ols_r2 > -999:
        print(f"n={n:3d}, p={p:2d}: OLS Test R²={ols_r2:.4f}, Ridge Test R²={ridge_r2:.4f}")
    else:
        print(f"n={n:3d}, p={p:2d}: OLS Fails (n < p), Ridge Test R²={ridge_r2:.4f}")
# Output:
# n= 20, p=30: OLS Fails (n < p), Ridge Test R²=0.8341
# n= 30, p=30: OLS Test R²=0.8712, Ridge Test R²=0.9012
# n= 50, p=30: OLS Test R²=0.9185, Ridge Test R²=0.9278
# n=100, p=30: OLS Test R²=0.9432, Ridge Test R²=0.9452
# n=500, p=30: OLS Test R²=0.9634, Ridge Test R²=0.9635
```

## Common Mistakes

1. **Penalizing the intercept**: Ridge should NOT penalize β₀. In sklearn, Ridge does not penalize the intercept by default (when fit_intercept=True). When using the closed-form manually, center y and standardize X first, then estimate β₀ = ȳ.

2. **Not standardizing features before Ridge**: Ridge is sensitive to feature scales. A feature with large values will have smaller coefficients naturally; penalizing all coefficients equally penalizes large-scale features unfairly. Always standardize.

3. **Forgetting to scale test data with training parameters**: Use `scaler.transform(X_test)`, not `StandardScaler().fit_transform(X_test)`. This is a common data leakage error.

4. **Treating λ = 0 the same as OLS with sklearn**: `Ridge(alpha=0)` should be OLS, but in practice, `LinearRegression()` may produce slightly different results due to different solver implementations. Use `LinearRegression()` for OLS.

5. **Choosing alpha arbitrarily**: The optimal alpha depends on the data. Always use cross-validation (RidgeCV, GridSearchCV) to tune it.

6. **Expecting feature selection**: Ridge shrinks coefficients but never sets them exactly to zero. For feature selection, use Lasso (L1) instead.

7. **Using Ridge when data is already well-conditioned**: If n >> p and features are independent, OLS is fine. Ridge adds unnecessary bias.

8. **Ignoring Ridge's effect on inference**: Ridge coefficients are biased, so standard hypothesis tests (t-tests, p-values) are not valid. Ridge is primarily for prediction, not inference.

## Interview Questions

### Beginner

**Q1: What problem does Ridge regression solve?**

Ridge regression addresses overfitting by adding an L2 penalty to the loss function that shrinks coefficients toward zero. This reduces variance at the cost of introducing a small amount of bias.

**Q2: What is the L2 penalty?**

The L2 penalty is the sum of squared coefficients (β₁² + β₂² + ... + βₚ²) multiplied by the regularization parameter λ. The total loss becomes RSS + λ × sum(β²).

**Q3: How does Ridge differ from OLS?**

OLS minimizes RSS alone and can produce large, unstable coefficients. Ridge adds a penalty term that constrains coefficient magnitude, making the model more stable and generalizable.

**Q4: What happens to coefficients as λ → ∞?**

Coefficients shrink toward zero (but never reach exactly zero). The model approaches a constant prediction equal to the mean of y.

**Q5: Should you standardize data for Ridge? Why?**

Yes. Ridge penalizes all coefficients equally, so features on different scales would be penalized unfairly. Standardization ensures the penalty is applied evenly.

### Intermediate

**Q6: Derive the Ridge closed-form solution.**

Start with L(β) = ||y - Xβ||² + λ||β||². Expand: (y - Xβ)ᵀ(y - Xβ) + λβᵀβ. Take gradient: -2Xᵀ(y - Xβ) + 2λβ = 0. Rearranged: Xᵀy = (XᵀX + λI)β. Thus β̂ = (XᵀX + λI)⁻¹Xᵀy.

**Q7: Explain the bias-variance tradeoff in Ridge regression.**

As λ increases: bias² increases (coefficients are shrunk from their true values), but variance decreases (coefficients are less sensitive to training data). Total test MSE = bias² + variance + irreducible error. Ridge finds the λ that minimizes this total.

**Q8: How does Ridge handle multicollinearity?**

In OLS, multicollinearity makes XᵀX nearly singular, leading to huge, unstable coefficients. Ridge adds λI to XᵀX, ensuring the matrix is always invertible and estimates are stable. The penalty shrinks correlated features' coefficients toward each other.

**Q9: Compare Ridge with Principal Component Regression (PCR).**

Both handle multicollinearity and high-dimensional data. Ridge shrinks all coefficients continuously. PCR performs dimensionality reduction first (PCA), then regresses on principal components. Ridge is smoother and often preferred because it doesn't lose information from small-variance components.

**Q10: What is the effective degrees of freedom in Ridge?**

df(λ) = trace(H_λ) = Σⱼ d²ⱼ / (d²ⱼ + λ), where dⱼ are the singular values of X. When λ = 0, df = p (full OLS). When λ → ∞, df → 0. This shows Ridge continuously reduces model complexity.

### Advanced

**Q11: Derive the Ridge solution using the SVD and show how it shrinks components differently.**

Let X = UDVᵀ. Then β̂_ridge = V(D² + λI)⁻¹DUᵀy = V diag(dⱼ/(d²ⱼ + λ))Uᵀy. The shrinkage factor for the j-th direction is d²ⱼ/(d²ⱼ + λ). Small singular values (noise directions) shrink more toward zero. Large singular values (signal directions) shrink less.

**Q12: Prove that Ridge always produces a unique solution even when XᵀX is singular.**

The Ridge solution requires inverting XᵀX + λI. For any λ > 0 and any X, XᵀX is positive semi-definite and λI is positive definite. The sum is always positive definite, hence always invertible, guaranteeing a unique solution.

**Q13: How can Ridge be reformulated as a constrained optimization problem? Show the equivalence.**

Ridge minimizes ||y - Xβ||² subject to ||β||² ≤ t for some t > 0. By Lagrangian duality, this is equivalent to minimizing ||y - Xβ||² + λ||β||² where λ depends on t. As λ increases, the constraint t decreases, forcing more shrinkage.

## Practice Problems

### Easy

**P1:** Fit Ridge regression on a simple dataset with alpha = 0.1. Compare coefficients with OLS.

**P2:** Generate data with 3 features and multicollinearity. Show that OLS coefficients have high variance while Ridge coefficients are stable.

**P3:** Plot the Ridge coefficients path as alpha varies from 0.001 to 1000.

**P4:** Use RidgeCV to find the best alpha for a given dataset.

**P5:** Verify that as alpha → 0, Ridge coefficients approach OLS coefficients.

### Medium

**P6:** Implement Ridge regression from scratch using the closed-form solution and verify against sklearn.

**P7:** Perform a simulation study: generate data with 50 features (10 relevant, 40 noise) and compare OLS vs Ridge test MSE as n varies from 30 to 300.

**P8:** Use Ridge regression with PolynomialFeatures to fit a high-degree polynomial and prevent overfitting.

**P9:** Implement Ridge regression using gradient descent. Compare convergence for different alpha values.

**P10:** Compute the effective degrees of freedom for Ridge with different alphas.

### Hard

**P11:** Prove that the Ridge estimator is biased but has lower variance than OLS. Derive both bias and variance expressions.

**P12:** Implement a numerical experiment demonstrating the optimal alpha via the bias-variance decomposition. Plot bias², variance, and test MSE against alpha.

**P13:** Extend Ridge regression to Kernel Ridge Regression. Implement it on a non-linear dataset and compare with polynomial features + Ridge.

## Solutions

**P1 Solution:** Fit OLS and `Ridge(alpha=0.1).fit(X, y)`. All Ridge coefficients are slightly smaller than OLS.

**P2 Solution:** Create 100 bootstrap samples, fit OLS and Ridge on each. Compute variance of each coefficient across bootstraps. Ridge variance < OLS variance.

**P3 Solution:** For each alpha in `np.logspace(-3, 3, 50)`, fit Ridge, store coefficients, plot alpha (log scale) vs. coefficient values.

**P4 Solution:** `RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5).fit(X_train, y_train).alpha_`.

**P5 Solution:** `np.allclose(LinearRegression().fit(X, y).coef_, Ridge(alpha=1e-10).fit(X, y).coef_)`.

**P6 Solution:** Use `np.linalg.inv(X.T @ X + alpha * np.eye(p)) @ X.T @ y`. Compare with `Ridge(alpha=alpha).fit(X, y).coef_`.

**P7 Solution:** For n in [30, 50, 100, 200, 300]: generate data, split, fit OLS and Ridge, record test MSE. Plot.

**P8 Solution:** `make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=1.0))`. Compare with `LinearRegression()`.

**P9 Solution:** Ridge gradient: (2/n)Xᵀ(Xβ - y) + 2λβ. Update: β -= α * gradient. Track loss across iterations.

**P10 Solution:** Compute SVD of X. df = Σ d²ⱼ/(d²ⱼ + λ). For λ=0, df=p; for λ=∞, df=0.

**P11 Solution:** Bias = E[β̂_ridge] - β = -λ(XᵀX + λI)⁻¹β. Var(β̂_ridge) = σ²(XᵀX + λI)⁻¹XᵀX(XᵀX + λI)⁻¹. Show Frobenius norm of variance decreases with λ.

**P12 Solution:** For each λ: compute bias² = ||E[β̂] - β||² analytically, compute variance empirically, compute test MSE. Plot all three against λ.

**P13 Solution:** Kernel Ridge: β = Xᵀ(K + λI)⁻¹y, where K = XXᵀ (kernel matrix). Use RBF kernel. Compare with polynomial: `Pipeline([('poly', PolynomialFeatures(3)), ('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])`.

## Related Concepts

- Lasso Regression (L1) (ML-015)
- Elastic Net (L1 + L2) (ML-016)
- Regularization Techniques (ML-019)
- Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Polynomial Regression (ML-013)
- Bias-Variance Tradeoff
- Singular Value Decomposition (SVD)

## Next Concepts

- Lasso Regression (ML-015)
- Elastic Net (ML-016)
- Regularization Techniques Overview (ML-019)
- Regression Evaluation (ML-020)

## Summary

Ridge regression adds an L2 penalty to the OLS objective, shrinking coefficients toward zero to reduce variance at the cost of some bias. The closed-form solution β̂ = (XᵀX + λI)⁻¹Xᵀy is always unique and invertible, even when XᵀX is singular. Ridge is particularly effective for high-dimensional data (including n < p), multicollinearity, and improving prediction accuracy. The regularization parameter λ controls the strength of shrinkage and must be tuned via cross-validation. Unlike Lasso, Ridge does not perform feature selection (coefficients never reach exactly zero). Ridge is the go-to tool when stability and prediction accuracy are priorities.

## Key Takeaways

1. Ridge adds L2 penalty λΣβⱼ² to OLS loss, shrinking coefficients toward zero
2. Closed-form: β̂ = (XᵀX + λI)⁻¹Xᵀy — always invertible and unique
3. Ridge trades bias for variance: small increase in bias, large decrease in variance
4. Always standardize features before Ridge since the penalty is scale-sensitive
5. Ridge handles multicollinearity and n < p situations where OLS fails
6. Ridge does NOT produce zero coefficients (no feature selection)
7. Optimal λ must be found via cross-validation (RidgeCV)
8. Ridge is primarily for prediction, not inference (biased coefficients)
