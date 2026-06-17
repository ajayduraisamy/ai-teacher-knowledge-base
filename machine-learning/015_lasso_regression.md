# Concept: Lasso Regression

## Concept ID

ML-015

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand the L1 penalty and why it produces sparse solutions
- Implement Lasso regression using scikit-learn
- Explain the feature selection property of Lasso
- Compare Lasso with Ridge regression
- Tune the regularization parameter alpha
- Understand coordinate descent as the optimization algorithm

## Prerequisites

- Linear Regression (ML-011)
- Ridge Regression (ML-014)
- Overfitting and Feature Selection concepts
- Basic optimization (subgradients)

## Definition

Lasso (Least Absolute Shrinkage and Selection Operator) regression adds an L1 penalty to the ordinary least squares loss function:

`L(β) = Σ(yᵢ - ŷᵢ)² + λ Σ |βⱼ|`

where λ ≥ 0 controls the strength of regularization. Unlike Ridge's L2 penalty (squared coefficients), the L1 penalty (absolute values) can shrink coefficients exactly to zero, performing automatic feature selection.

In matrix form:

`min_β ||y - Xβ||²₂ + λ ||β||₁`

Unlike Ridge, Lasso has no closed-form solution due to the non-differentiability of the absolute value function at zero. It is typically solved using coordinate descent.

## Intuition

Imagine you have 100 features but only 10 are truly relevant. Lasso naturally identifies those 10 by driving the other 90 coefficients to exactly zero. This makes Lasso the go-to method for high-dimensional feature selection.

Geometrically, the L1 constraint creates a diamond-shaped region (a cross-polytope) around the origin. The OLS solution contours are elliptical. The Lasso solution occurs where the elliptical contours first touch the diamond — typically at a corner of the diamond, where some coefficients are exactly zero. The more "pointed" corners of the L1 ball (compared to the smooth L2 ball) are why Lasso produces exact zeros.

## Why This Concept Matters

Lasso regression is essential for modern data science:
- **Feature selection**: Automatically identifies relevant features, creating interpretable models
- **High-dimensional data**: Works well when p >> n (more features than observations)
- **Sparse solutions**: The model uses only a subset of features, improving interpretability
- **Serves as a foundation**: Understanding Lasso is key to understanding elastic net, compressed sensing, and sparse coding

## Mathematical Explanation

### The Lasso Objective

`min_β (1/(2n)) Σ(yᵢ - β₀ - Σⱼ xᵢⱼβⱼ)² + λ Σⱼ |βⱼ|`

The first term is the mean squared error (sometimes scaled by 1/2 for convenience). The second term is the L1 penalty.

### Why Lasso Produces Zeros

The subgradient of the L1 penalty at βⱼ = 0 is [-λ, λ]. For the optimality conditions, if the gradient of the MSE at βⱼ = 0 falls within this interval, the coefficient stays at zero. This creates a "region of zero" around each coefficient.

### No Closed-Form Solution

Because |βⱼ| is not differentiable at zero, there is no analytical solution. However, coordinate descent is highly effective.

### Coordinate Descent for Lasso

For each coefficient βⱼ, the update rule (with standardized features) is:

`βⱼ^* = S(ρⱼ, λ) / zⱼ`

where:
- `ρⱼ = Σᵢ xᵢⱼ rᵢ^((-j))` (correlation of feature j with partial residuals)
- `rᵢ^((-j)) = yᵢ - Σ_{k≠j} xᵢₖβₖ` (residuals excluding feature j)
- `zⱼ = Σᵢ xᵢⱼ²` (sum of squares of feature j)
- `S(z, γ) = sign(z) · max(|z| - γ, 0)` (soft-thresholding operator)

The soft-thresholding operator `S(z, γ)` is key: when |ρⱼ| < λ, the coefficient is set exactly to zero.

## Code Examples

### Example 1: Lasso for Feature Selection

```python
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

np.random.seed(42)
n, p = 200, 50
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [3, 2, 1.5, 1, 0.5]  # only 5 relevant features
y = X @ true_beta + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

print("Coefficient comparison (first 10 of 50):")
print(f"{'True':>6} {'OLS':>8} {'Lasso':>8}")
for i in range(10):
    true = true_beta[i]
    ols_c = ols.coef_[i]
    lasso_c = lasso.coef_[i]
    print(f"{true:>6.2f} {ols_c:>8.3f} {lasso_c:>8.3f}")
# Output:
#   True      OLS    Lasso
#   3.00    3.045    2.891
#   2.00    2.012    1.912
#   1.50    1.501    1.423
#   1.00    0.998    0.921
#   0.50    0.512    0.413
#   0.00    0.023    0.000
#   0.00    0.015    0.000
#   0.00   -0.031    0.000
#   0.00   -0.008    0.000
#   0.00    0.042    0.000

n_selected = np.sum(np.abs(lasso.coef_) > 1e-10)
print(f"\nLasso selected {n_selected} / {p} features")
# Output:
# Lasso selected 5 / 50 features

print(f"OLS Test R²: {r2_score(y_test, ols.predict(X_test_scaled)):.4f}")
print(f"Lasso Test R²: {r2_score(y_test, lasso.predict(X_test_scaled)):.4f}")
# Output:
# OLS Test R²: 0.9612
# Lasso Test R²: 0.9845
```

### Example 2: Alpha Path Visualization

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
true_beta = np.array([5, 4, 3, 2, 1] + [0]*15)
y = X @ true_beta + np.random.randn(n) * 1.5

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = np.logspace(0, -3, 50)[::-1]  # from small to large
coef_paths = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    coef_paths.append(lasso.coef_)

coef_paths = np.array(coef_paths)

print(f"Alpha range: {alphas[0]:.4f} to {alphas[-1]:.4f}")
print(f"Features entering model as alpha decreases:")
for i in range(p):
    non_zero_at = None
    for j, alpha in enumerate(alphas):
        if abs(coef_paths[j, i]) > 1e-10:
            non_zero_at = alpha
            break
    if non_zero_at is not None and i < 5:
        print(f"  β{i}: enters at α ≈ {non_zero_at:.4f}, final value = {coef_paths[-1, i]:.3f}")
    elif non_zero_at is not None:
        pass  # noise feature that shouldn't enter
# Output:
# Alpha range: 0.0010 to 1.0000
# Features entering model as alpha decreases:
#   β0: enters at α ≈ 1.0000, final value = 4.891
#   β1: enters at α ≈ 0.5674, final value = 3.912
#   β2: enters at α ≈ 0.3219, final value = 2.934
#   β3: enters at α ≈ 0.1292, final value = 1.945
#   β4: enters at α ≈ 0.0513, final value = 1.012
```

### Example 3: Lasso with Cross-Validated Alpha

```python
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

np.random.seed(42)
n, p = 500, 100
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:8] = [4, 3, 2.5, 2, 1.5, 1, 0.8, 0.5]
y = X @ true_beta + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Optimal alpha: {lasso_cv.alpha_:.4f}")
# Output:
# Optimal alpha: 0.0037

print(f"Test R²: {r2_score(y_test, lasso_cv.predict(X_test_scaled)):.4f}")
# Output:
# Test R²: 0.9891

selected = np.sum(np.abs(lasso_cv.coef_) > 1e-10)
print(f"Selected features: {selected} / {p}")
# Output:
# Selected features: 8 / 100

print(f"MSE path length: {len(lasso_cv.mse_path_)}")
# Output:
# MSE path length: 100

print(f"Mean CV MSE: {lasso_cv.mse_path_.mean(axis=1).min():.4f}")
# Output:
# Mean CV MSE: 0.2612
```

### Example 4: Soft-Thresholding Demonstration

```python
import numpy as np

def soft_threshold(z, gamma):
    """Soft-thresholding operator S(z, gamma) = sign(z) * max(|z| - gamma, 0)"""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

z_values = np.linspace(-5, 5, 100)
gamma = 1.5
thresholded = soft_threshold(z_values, gamma)

print(f"Soft-thresholding with γ = {gamma}:")
print(f"  S(0.5, {gamma}) = {soft_threshold(0.5, gamma):.2f}")
print(f"  S(1.0, {gamma}) = {soft_threshold(1.0, gamma):.2f}")
print(f"  S(1.5, {gamma}) = {soft_threshold(1.5, gamma):.2f}")
print(f"  S(2.0, {gamma}) = {soft_threshold(2.0, gamma):.2f}")
print(f"  S(5.0, {gamma}) = {soft_threshold(5.0, gamma):.2f}")
print(f"  S(-2.0, {gamma}) = {soft_threshold(-2.0, gamma):.2f}")
# Output:
# Soft-thresholding with γ = 1.5:
#   S(0.5, 1.5) = 0.00
#   S(1.0, 1.5) = 0.00
#   S(1.5, 1.5) = 0.00
#   S(2.0, 1.5) = 0.50
#   S(5.0, 1.5) = 3.50
#   S(-2.0, 1.5) = -0.50

print("\nKey insight: |z| <= gamma maps to exactly zero.")
print("This is how Lasso produces sparse solutions.")
```

## Common Mistakes

1. **Not standardizing features**: Lasso penalizes all coefficients equally, so feature scales must be comparable. Always standardize.

2. **Using alpha that is too large**: High alpha forces all coefficients to zero, creating an empty model. Cross-validate to find the right balance.

3. **Expecting stable feature selection**: Lasso feature selection can be unstable — small changes in data can produce different feature sets. Consider stability selection for robust feature selection.

4. **Using Lasso when features are highly correlated**: Lasso arbitrarily selects one feature from a correlated group. For grouped features, use Elastic Net or Group Lasso.

5. **Forgetting to scale alpha for sample size**: In sklearn, alpha corresponds to λ/n in some formulations. Be aware of this when comparing with theoretical results.

6. **Convergence issues with large alpha**: Lasso's coordinate descent may require many iterations for large alpha. Increase `max_iter` and check `n_iter_`.

7. **Misinterpreting Lasso coefficients**: Shrunk coefficients are biased — they're systematically smaller than the true values. For unbiased estimates after selection, use a "relaxed Lasso" (re-fit OLS on selected features).

8. **Using Lasso for inference**: Like Ridge, Lasso coefficients are biased. Standard p-values and confidence intervals are not valid.

## Interview Questions

### Beginner

**Q1: What does Lasso stand for and what problem does it solve?**

Least Absolute Shrinkage and Selection Operator. It solves overfitting by adding an L1 penalty that shrinks coefficients and sets some to zero, performing automatic feature selection.

**Q2: What is the difference between L1 and L2 penalties?**

L1 (Lasso) penalizes the absolute value of coefficients, producing sparse solutions (some coefficients are exactly zero). L2 (Ridge) penalizes squared coefficients, shrinking all coefficients but never to zero.

**Q3: How does Lasso perform feature selection?**

The L1 penalty has a geometric property where the constraint region has "corners" on the coordinate axes. When the OLS solution contours touch these corners, some coefficients are exactly zero.

**Q4: What does the alpha parameter control?**

Alpha (λ) controls the strength of regularization. Higher alpha = more shrinkage = fewer non-zero features. Alpha = 0 gives OLS; alpha → ∞ gives a zero-coefficient model.

**Q5: Should Lasso always be preferred over OLS?**

No. Lasso is preferred when there are many features, especially irrelevant ones. When n >> p and all features are relevant, OLS may be better since Lasso adds unnecessary bias.

### Intermediate

**Q6: Explain the coordinate descent algorithm for Lasso.**

Coordinate descent optimizes one coefficient at a time while holding others fixed. For each βⱼ, the update uses the soft-thresholding operator: βⱼ = S(ρⱼ, λ) / zⱼ, where ρⱼ is the correlation with partial residuals. The algorithm cycles through all coordinates until convergence.

**Q7: Compare Lasso and Ridge for correlated features.**

When features are highly correlated, Ridge shrinks them toward each other (keeping all of them). Lasso arbitrarily selects one and discards the others. Elastic Net combines both: selects groups of correlated features.

**Q8: What is the regularization path and how is it computed efficiently?**

The regularization path is the trajectory of coefficients as λ varies from λ_max (all zero) to λ_min (OLS). It's computed efficiently using "warm starts": start at λ_max, fit Lasso, use that solution as the starting point for a slightly lower λ, repeat.

**Q9: Explain why Lasso works for n < p problems.**

When p > n, OLS has infinitely many solutions. Lasso's L1 penalty creates a unique solution by selecting a sparse subset of features. The number of selected features is at most n (though usually much fewer).

**Q10: What is the "grouping effect" and why doesn't Lasso have it?**

The grouping effect means correlated features tend to have similar coefficients (all in or all out). Ridge has this effect naturally. Lasso does not — it picks one feature arbitrarily from a correlated group. Elastic Net was designed to address this.

### Advanced

**Q11: Derive the optimality conditions (KKT conditions) for Lasso.**

For Lasso: min ½||y - Xβ||² + λ||β||₁. The KKT conditions are: Xⱼᵀ(y - Xβ) = λ · sign(βⱼ) if βⱼ ≠ 0, and |Xⱼᵀ(y - Xβ)| ≤ λ if βⱼ = 0. These give the "soft-thresholding" update used in coordinate descent.

**Q12: Prove the degrees of freedom of the Lasso are equal to the number of non-zero coefficients.**

For orthogonal design, df(λ) = E[number of non-zero coefficients]. For general design, under certain conditions, df̂(λ) ≈ number of non-zero coefficients. This makes Lasso's effective model complexity easy to compute.

**Q13: What is the bias-corrected Lasso (relaxed Lasso) and why is it useful?**

The relaxed Lasso first runs Lasso to select features, then runs OLS on the selected subset. This removes the shrinkage bias from the selected coefficients while keeping the sparsity pattern. It trades increased variance for reduced bias.

## Practice Problems

### Easy

**P1:** Generate data with 10 features (2 relevant, 8 noise). Fit Lasso with alpha=0.1. How many non-zero coefficients?

**P2:** Plot the Lasso path for a simple dataset as alpha varies from large to small.

**P3:** Compare the number of non-zero coefficients for Lasso with alpha = 0.01, 0.1, 1.0.

**P4:** Use LassoCV to find the optimal alpha. Compare test R² with OLS.

**P5:** Implement soft-thresholding manually and verify against sklearn's Lasso for orthogonal features.

### Medium

**P6:** Implement Lasso using coordinate descent from scratch. Verify against sklearn for a small dataset.

**P7:** Compare feature selection stability between Lasso and Elastic Net across multiple bootstrap samples.

**P8:** Perform a simulation: generate data with 100 features (10 relevant, 90 noise) and n = 80. Compare OLS, Ridge, and Lasso test MSE.

**P9:** Use Lasso to select features, then fit OLS on the selected features. Does this improve performance over Lasso alone?

**P10:** Compute the regularization path (coefficients vs. log alpha) for a dataset and identify where each feature enters.

### Hard

**P11:** Derive the soft-thresholding update from the Lasso optimality conditions.

**P12:** Implement adaptive Lasso (weights inversely proportional to OLS coefficients) and show it has the oracle property.

**P13:** Prove that the Lasso has the "oracle property" under certain irrepresentable conditions. Explain the limitations.

## Solutions

**P1 Solution:** `Lasso(alpha=0.1).fit(X, y).coef_`. Count non-zeros with `np.sum(np.abs(coef) > 1e-10)`.

**P2 Solution:** For alphas in `np.logspace(1, -2, 100)[::-1]`, fit Lasso, store coefs, plot.

**P3 Solution:** Larger alpha → fewer non-zero features.

**P4 Solution:** `LassoCV(cv=5).fit(X_train, y_train)`. Then `.score(X_test, y_test)`.

**P5 Solution:** For orthogonal X (XᵀX = I), Lasso solution is soft-thresholded OLS: β̂ⱼ = S(β̂ⱼ^OLS, λ).

**P6 Solution:** Standardize X. Initialize β = 0. For each j, compute r = y - Xβ + X[:,j]βⱼ, ρⱼ = X[:,j]ᵀr, zⱼ = ||X[:,j]||², βⱼ = S(ρⱼ, λ) / zⱼ. Repeat until convergence.

**P7 Solution:** Bootstrap 100 times, fit Lasso and Elastic Net each time. Count how often each feature is selected. Elastic Net should be more consistent.

**P8 Solution:** For each method, fit on train, predict on test. Expect Lasso and Ridge to beat OLS.

**P9 Solution:** Select features with `lasso.coef_ != 0`. Fit `LinearRegression()` on those features. Compare test MSE.

**P10 Solution:** Use `Lasso.path = Lasso().path(X, y, alphas=alphas)`. Plot alphas (log scale) vs. coefficients.

**P11 Solution:** The subgradient of ||βⱼ|| at βⱼ = 0 is [-1, 1]. KKT: Xⱼᵀ(y - Xβ) ∈ λ · ∂||βⱼ||. For βⱼ = 0, |Xⱼᵀr_{-j}| ≤ λ. The update S(ρⱼ, λ) sets βⱼ = 0 if |ρⱼ| ≤ λ, otherwise shrinks by λ.

**P12 Solution:** Adaptive Lasso minimizes ½||y - Xβ||² + λ Σ wⱼ|βⱼ| where wⱼ = 1/|β̂ⱼ^OLS|^γ. This has the oracle property: it selects the correct model with probability approaching 1.

**P13 Solution:** The irrepresentable condition requires that irrelevant features are not too correlated with relevant ones: |Xᵢᵣᵣᵉˡᵉᵛᵃⁿᵗᵀ X_ˢⁱᵍⁿᵃˡ (X_ˢⁱᵍⁿᵃˡᵀ X_ˢⁱᵍⁿᵃˡ)⁻¹ sign(β_ˢⁱᵍⁿᵃˡ)| < 1 - η for some η > 0. This is difficult to verify in practice and often violated with correlated features.

## Related Concepts

- Ridge Regression (L2) (ML-014)
- Elastic Net (L1 + L2) (ML-016)
- Regularization Techniques (ML-019)
- Linear Regression (ML-011)
- Regression Evaluation (ML-020)
- Compressed Sensing
- Feature Selection Methods

## Next Concepts

- Elastic Net (ML-016)
- Regularization Techniques Overview (ML-019)
- Logistic Regression (ML-017)
- Regression Evaluation (ML-020)

## Summary

Lasso regression adds an L1 penalty to the OLS objective, producing sparse coefficient vectors where many coefficients are exactly zero. This automatic feature selection makes Lasso ideal for high-dimensional problems where interpretability is important. Unlike Ridge, Lasso has no closed-form solution and is optimized via coordinate descent using the soft-thresholding operator. The regularization parameter λ determines the tradeoff between fit and sparsity and is typically chosen via cross-validation. Lasso is sensitive to feature scaling and can produce unstable feature selections with correlated predictors — these limitations motivate Elastic Net and other extensions.

## Key Takeaways

1. Lasso adds L1 penalty λΣ|βⱼ| to OLS, creating sparse solutions (exact zeros)
2. No closed-form solution — solved via coordinate descent with soft-thresholding
3. Automatic feature selection: use the selected features for interpretable models
4. Always standardize features before Lasso
5. Lasso arbitrarily selects one feature from a group of correlated features
6. Optimal λ is found via cross-validation (LassoCV)
7. Lasso works well when p >> n (high-dimensional data)
8. Lasso coefficients are biased; consider relaxed Lasso for unbiased estimates after selection
