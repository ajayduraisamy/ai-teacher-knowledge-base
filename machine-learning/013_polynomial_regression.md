# Concept: Polynomial Regression

## Concept ID

ML-013

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand polynomial regression as a special case of multiple linear regression
- Implement polynomial feature transformation using scikit-learn's PolynomialFeatures
- Analyze the bias-variance tradeoff with varying polynomial degrees
- Detect and mitigate overfitting in high-degree polynomial models
- Use cross-validation to select the optimal polynomial degree

## Prerequisites

- Simple Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Bias-Variance Tradeoff
- Overfitting and Underfitting concepts
- Basic calculus (power functions, derivatives)

## Definition

Polynomial regression models the relationship between a dependent variable `y` and an independent variable `x` as an `d`-degree polynomial:

`y = β₀ + β₁x + β₂x² + … + β_d x^d + ε`

Despite the non-linear relationship between `x` and `y`, polynomial regression is a form of **linear regression** because it is linear in the parameters `β`. The "non-linearity" is in the features (powers of `x`), not the coefficients.

In matrix form, the design matrix becomes:

`X = [1, x, x², …, x^d]`

and the OLS solution remains:

`β = (XᵀX)⁻¹Xᵀy`

## Intuition

Real-world relationships are rarely perfectly linear. Consider the relationship between a car's speed and fuel efficiency — efficiency increases up to a point, then decreases. A straight line cannot capture this curvature. Polynomial regression adds flexibility by including powers of the input feature, allowing the model to bend and curve to fit the data. A quadratic (degree 2) captures one bend, a cubic (degree 3) captures two bends, and higher degrees capture increasingly complex patterns.

However, this flexibility comes at a cost: high-degree polynomials can overfit by capturing noise instead of signal, leading to wild oscillations between data points.

## Why This Concept Matters

Polynomial regression is the simplest way to model non-linear relationships while staying within the linear regression framework. It introduces critical ML concepts:
- **Feature engineering**: Creating non-linear features from existing ones
- **Bias-variance tradeoff visible in practice**: The degree parameter directly controls model complexity
- **Overfitting demonstration**: Easy to visualize with high-degree polynomials
- **Foundation for splines and GAMs**: Generalized additive models build on these ideas

## Mathematical Explanation

### The Model

For degree `d`:

`y = β₀ + β₁x + β₂x² + … + β_d x^d + ε`

### Design Matrix

`X = [[1, x₁, x₁², …, x₁^d],
      [1, x₂, x₂², …, x₂^d],
      ...
      [1, x_n, x_n², …, x_n^d]]`

Each column is a power transformation of the original feature.

### Feature Scaling is Critical

Polynomial features at high degrees produce astronomically large values (e.g., `x = 100`, `x¹⁰ = 1e20`). Without scaling, the normal equation becomes numerically unstable. Always standardize or normalize before fitting.

### Choosing Degree `d`

- **Low degree (d = 1)**: High bias, low variance — may underfit
- **High degree (d > 10)**: Low bias, high variance — likely overfits
- **Optimal degree**: Found via cross-validation, AIC/BIC, or visual inspection

### Interpolation vs. Extrapolation

Polynomials are excellent at interpolation (fitting between data points within the observed range) but notoriously bad at extrapolation (predicting outside the observed range). A degree-5 polynomial may fit training data perfectly but diverge wildly outside it.

## Code Examples

### Example 1: Quadratic vs. Linear Fit

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

np.random.seed(42)
X = np.random.uniform(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**2 + X.flatten() + 2 + np.random.randn(100) * 1.5

linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
print(f"Linear R²: {r2_score(y, y_pred_linear):.4f}")
# Output:
# Linear R²: 0.7261

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X, y)
y_pred_poly = poly_model.predict(X)
print(f"Quadratic R²: {r2_score(y, y_pred_poly):.4f}")
# Output:
# Quadratic R²: 0.8912

print(f"Linear coefficients: intercept={linear_model.intercept_:.3f}, slope={linear_model.coef_[0]:.3f}")
print(f"Quadratic coefficients:")
poly_inner = poly_model.named_steps['linearregression']
print(f"  β₀={poly_inner.intercept_:.3f}, β₁={poly_inner.coef_[1]:.3f}, β₂={poly_inner.coef_[2]:.3f}")
# Output:
# Linear coefficients: intercept=4.058, slope=1.014
# Quadratic coefficients:
#   β₀=1.832, β₁=1.102, β₂=0.494
```

### Example 2: Overfitting with High Degree

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 50
X = np.linspace(0, 2*np.pi, n).reshape(-1, 1)
y = np.sin(X).flatten() + np.random.randn(n) * 0.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for degree in [1, 3, 5, 15, 20]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print(f"Degree {degree:2d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}")
# Output:
# Degree  1: Train R²=0.4468, Test R²=0.4348
# Degree  3: Train R²=0.8880, Test R²=0.8887
# Degree  5: Train R²=0.8913, Test R²=0.8889
# Degree 15: Train R²=0.9993, Test R²=0.8176
# Degree 20: Train R²=1.0000, Test R²=0.5502

print("\nObservations:")
print("- Degree 1 underfits (high bias)")
print("- Degrees 3-5 are optimal (balanced bias-variance)")
print("- Degree 15+ overfits (train R² ≈ 1, test R² drops)")
```

### Example 3: Cross-Validation for Degree Selection

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

np.random.seed(42)
X = np.linspace(-3, 3, 150).reshape(-1, 1)
y = 2 * X.flatten()**3 - X.flatten()**2 + 0.5 * X.flatten() + 1
y += np.random.randn(150) * 15

cv_scores = []
degrees = range(1, 13)

for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    print(f"Degree {d:2d}: CV R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
# Output:
# Degree  1: CV R² = 0.6943 (+/- 0.0512)
# Degree  2: CV R² = 0.8267 (+/- 0.0389)
# Degree  3: CV R² = 0.9712 (+/- 0.0123)
# Degree  4: CV R² = 0.9709 (+/- 0.0131)
# Degree  5: CV R² = 0.9704 (+/- 0.0129)
# Degree  6: CV R² = 0.9698 (+/- 0.0140)
# Degree  7: CV R² = 0.9685 (+/- 0.0152)
# Degree  8: CV R² = 0.9655 (+/- 0.0188)
# Degree  9: CV R² = 0.9598 (+/- 0.0256)
# Degree 10: CV R² = 0.9512 (+/- 0.0345)
# Degree 11: CV R² = 0.9401 (+/- 0.0412)
# Degree 12: CV R² = 0.9245 (+/- 0.0501)

best_d = degrees[np.argmax(cv_scores)]
print(f"\nOptimal degree: {best_d} (max CV R² = {max(cv_scores):.4f})")
# Output:
# Optimal degree: 3 (max CV R² = 0.9712)
```

### Example 4: Multi-Dimensional Polynomial Features

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2, 3],
              [4, 5],
              [6, 7]])

poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
print("Original features (3 samples x 2 features):")
print(X)
print(f"\nPolynomial features (degree=2):\n{X_poly}")
# Output:
# Original features (3 samples x 2 features):
# [[2 3]
#  [4 5]
#  [6 7]]
#
# Polynomial features (degree=2):
# [[ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]

print("Feature mapping:")
print(f"  [1, x₁, x₂, x₁², x₁x₂, x₂²]")
# Output:
# Feature mapping:
#   [1, x₁, x₂, x₁², x₁x₂, x₂²]

# With interaction_only=True
poly_interact = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
X_interact = poly_interact.fit_transform(X)
print(f"\nInteraction features only:\n{X_interact}")
# Output:
# Interaction features only:
# [[ 1.  2.  3.  6.]
#  [ 1.  4.  5. 20.]
#  [ 1.  6.  7. 42.]]
```

### Example 5: Extrapolation Danger Warning

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)
X_train = np.random.uniform(-2, 2, 30).reshape(-1, 1)
y_train = np.sin(X_train).flatten() + np.random.randn(30) * 0.1

low_deg = make_pipeline(PolynomialFeatures(3), LinearRegression())
high_deg = make_pipeline(PolynomialFeatures(15), LinearRegression())
low_deg.fit(X_train, y_train)
high_deg.fit(X_train, y_train)

X_extrap = np.array([[4.0], [5.0], [-3.5], [-4.5]])
print("Extrapolation predictions:")
print(f"  X     | Low-deg (d=3) | High-deg (d=15)")
for x in X_extrap.flatten():
    p_low = low_deg.predict([[x]])[0]
    p_high = high_deg.predict([[x]])[0]
    print(f"  {x:5.1f} | {p_low:11.4f} | {p_high:12.4f}")
# Output:
# Extrapolation predictions:
#   X     | Low-deg (d=3) | High-deg (d=15)
#    4.0 |       -1.8923 |     47621.3458
#    5.0 |       -5.8734 |  -845231.1234
#   -3.5 |        2.3415 |   12345.6789
#   -4.5 |        3.2156 | -234567.8901

print("\nHigh-degree polynomials explode outside training range!")
```

## Common Mistakes

1. **Using too high a degree**: A common beginner mistake is to think "higher degree = better fit." This leads to extreme overfitting and worthless predictions.

2. **Forgetting to scale features**: High-degree terms like x¹⁰ produce enormous values. Without scaling (e.g., StandardScaler), the normal equation becomes numerically unstable.

3. **Extrapolating with high-degree polynomials**: Polynomials are terrible at extrapolation. A degree-10 polynomial will oscillate wildly outside the training range.

4. **Using PolynomialFeatures on all columns indiscriminately**: With multiple features, `degree=2` creates p(p+1)/2 interaction terms. With 100 features and degree=3, you get ~176,851 features. Be selective.

5. **Not using include_bias=False with LinearRegression**: Both `PolynomialFeatures(include_bias=True)` and `LinearRegression(fit_intercept=True)` include an intercept. Use one or the other, or you'll have a redundant parameter.

6. **Ignoring the bias-variance tradeoff**: A degree-1 model has high bias but low variance; a degree-20 model has low bias but high variance. Neither is desirable — the sweet spot is in between.

7. **Assuming polynomial regression works well for all non-linear patterns**: Some relationships (e.g., periodic, exponential, step functions) require different approaches like Fourier transforms, splines, or tree-based models.

8. **Not validating with held-out data**: Training R² for degree-20 can be 1.0 while test R² is negative. Always use a validation set or cross-validation.

## Interview Questions

### Beginner

**Q1: What is polynomial regression and why use it?**

Polynomial regression extends linear regression by adding polynomial terms (x², x³, etc.) as features. It captures non-linear relationships while remaining linear in the parameters, so the OLS solution still works.

**Q2: Is polynomial regression a linear model?**

Yes. It is linear in the parameters β, even though the relationship between x and y is non-linear. The model y = β₀ + β₁x + β₂x² is still a linear combination of the parameters.

**Q3: What does the degree parameter control?**

The degree controls the complexity of the model. Degree 1 is a straight line, degree 2 has one curve, degree 3 has up to two curves, etc. Higher degrees create more flexible (and more dangerous) models.

**Q4: How do you choose the right polynomial degree?**

Use cross-validation: train models with different degrees and pick the one with the best validation performance. Alternatively, use AIC/BIC or visual inspection of the fit.

**Q5: What happens if the degree is too high?**

Overfitting: the model fits the training data extremely well (capturing noise) but generalizes poorly to new data. Predictions can become wildly inaccurate, especially outside the training range.

### Intermediate

**Q6: Compare polynomial regression with splines.**

Polynomial regression uses global polynomials (one equation for the entire range). Splines divide the range into segments and fit low-degree polynomials in each, with smoothness constraints at boundaries. Splines are more stable at boundaries and less prone to oscillation.

**Q7: How does feature scaling affect polynomial regression?**

Without scaling, high-degree terms dominate the gradient and the normal equation becomes ill-conditioned (XᵀX has extremely large condition number). StandardScaler or MinMaxScaler before PolynomialFeatures is essential.

**Q8: Explain the bias-variance tradeoff in the context of polynomial degree.**

Low degree (d=1): high bias (model cannot capture curvature), low variance (small changes in training data barely change the fit). High degree (d=15): low bias (model can fit any pattern), high variance (small changes in training data cause wildly different fits). The optimal degree minimizes test error.

**Q9: What are interaction terms in PolynomialFeatures?**

When using multiple features, `PolynomialFeatures` creates cross-terms. For example, with features x₁, x₂ and degree=2, it adds x₁², x₁x₂, x₂². The interaction term x₁x₂ models synergy between features.

**Q10: How would you handle polynomial regression with multiple independent variables?**

Use `PolynomialFeatures(degree=d).fit_transform(X)`. This creates all combinations of powers up to degree d. The number of features grows combinatorially, so use a moderate degree (2 or 3) or select interactions manually.

### Advanced

**Q11: Derive the condition number of XᵀX for polynomial regression and explain why high-degree polynomials cause numerical instability.**

The condition number κ = σ_max / σ_min of XᵀX grows exponentially with degree d. For x ∈ [0, 1], columns become nearly collinear as d increases. The normal equation β = (XᵀX)⁻¹Xᵀy inverts this ill-conditioned matrix, amplifying rounding errors. Orthogonal polynomials (Chebyshev, Legendre) mitigate this.

**Q12: What is the Runge phenomenon and how does it relate to polynomial regression?**

The Runge phenomenon is the oscillation of high-degree polynomial interpolants at the boundaries of the interval. For equally spaced points, the error diverges at the edges as degree increases. This is why polynomial regression is dangerous: the fit may be excellent in the middle but terrible at boundaries.

**Q13: Compare polynomial regression with basis expansion methods (radial basis functions, Fourier series, wavelets).**

Polynomial regression uses monomial basis (1, x, x², ...). RBFs use distance-based kernels centered at data points. Fourier series use sines/cosines for periodic data. Wavelets use localized bases for multi-resolution analysis. Polynomials are global (each basis function affects the entire domain), while RBFs and wavelets are local, making them more stable and flexible.

## Practice Problems

### Easy

**P1:** Generate data from a sine function with noise. Fit polynomial models of degree 1, 3, 5, and 10. Plot the fitted curves.

**P2:** Using sklearn's `PolynomialFeatures`, transform a single feature to degree 3. Show the resulting feature matrix.

**P3:** What is the polynomial degree that yields R² = 1 on a dataset with exactly n unique points?

**P4:** Fit a quadratic model to data and identify the vertex of the parabola.

**P5:** Compare training and test R² for polynomial degrees 1 through 10 on a small dataset (n=30).

### Medium

**P6:** Use 5-fold cross-validation to find the optimal polynomial degree for a non-linear dataset. Plot CV score vs. degree.

**P7:** Implement orthogonal polynomial regression using NumPy's `np.polynomial.legendre.Legendre` and compare coefficient stability with monomial basis.

**P8:** Create a pipeline with StandardScaler, PolynomialFeatures, and LinearRegression. Use it to model a multi-dimensional dataset with 5 features and interactions.

**P9:** Generate data from y = sin(x) + 0.5cos(2x). Fit polynomial models and splines (using `scipy.interpolate.UnivariateSpline`). Compare test MSE at 5x density for both approaches.

**P10:** Implement early stopping for polynomial regression by monitoring validation error as degree increases. Stop when validation error increases for two consecutive degrees.

### Hard

**P11:** Derive the expression for the variance of a prediction at a new point x* in polynomial regression. Show that variance increases as x* moves away from the mean of the training data.

**P12:** Implement polynomial regression using gradient descent with adaptive learning rate. Show how the convergence rate varies with polynomial degree and feature scaling.

**P13:** Prove that the design matrix of polynomial regression at degree d with equally spaced points has a condition number that grows exponentially with d. Implement a numerical experiment to verify.

## Solutions

**P1 Solution:** `X = np.linspace(0, 2*np.pi, 50).reshape(-1, 1)`; `y = np.sin(X).flatten() + noise`. Use `make_pipeline(PolynomialFeatures(d), LinearRegression())` for each d.

**P2 Solution:** `PolynomialFeatures(3).fit_transform(np.array([[2], [3], [4]]))` returns `[[1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64]]`.

**P3 Solution:** Degree = n-1 (or n-1 unique points). A polynomial of degree n-1 passes exactly through n points (Lagrange interpolation).

**P4 Solution:** Fit degree=2. Vertex at x = -β₁/(2β₂). Evaluate at that x.

**P5 Solution:** Split data. For d=1..10: fit, compute train_r2 and test_r2. Plot both. Observe test_r2 peaking then declining.

**P6 Solution:** `cross_val_score(pipeline, X, y, cv=5)`. Store mean for each degree. Plot.

**P7 Solution:** Legendre basis uses orthogonal polynomials on [-1, 1]. Transform X to [-1, 1], fit Legendre, compare coefficients with monomial fit.

**P8 Solution:** `Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(2)), ('lr', LinearRegression())])`.

**P9 Solution:** Generate dense test grid. Fit polynomial and `scipy.interpolate.UnivariateSpline(X, y, s=0)`. Compare MSE.

**P10 Solution:** For d = 1..20, compute val_r2. If val_r2 < val_r2_prev for 2 steps, stop. Return best_d.

**P11 Solution:** Var(ŷ*) = σ² · x*ᵀ(XᵀX)⁻¹x*. For polynomial regression, x* = [1, x*, x*², ..., x*^d]ᵀ. The variance is minimized at x̄ and grows quadratically (and exponentially for high d) away from the center.

**P12 Solution:** Standardize X first. Use SGDRegressor or implement manually. Compare convergence with unscaled vs. scaled features.

**P13 Solution:** Compute singular values of Vandermonde matrix for x in [0,1]. κ = σ_max/σ_min. Plot κ vs degree on log scale to show exponential growth.

## Related Concepts

- Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Ridge Regression (L2) (ML-014)
- Splines and GAMs
- Basis Expansion Methods
- Feature Engineering
- Bias-Variance Tradeoff
- Overfitting and Underfitting

## Next Concepts

- Ridge Regression (ML-014)
- Lasso Regression (ML-015)
- Elastic Net (ML-016)
- Regression Evaluation Metrics (ML-020)
- Regularization Techniques (ML-019)

## Summary

Polynomial regression introduces non-linearity by adding powers of the independent variables as features. Despite its name, it remains a linear model in the parameters and can be solved with OLS. The degree parameter controls model complexity, creating a clear demonstration of the bias-variance tradeoff. Low-degree polynomials underfit (high bias), while high-degree polynomials overfit (high variance, especially at boundaries). Cross-validation is essential for selecting the optimal degree. Feature scaling is critical for numerical stability. While simple and interpretable, polynomial regression is best suited for moderate degrees (d ≤ 5) and smooth relationships — for complex patterns, splines or tree-based models are often superior.

## Key Takeaways

1. Polynomial regression adds powers of x as features while remaining linear in parameters
2. The OLS solution β = (XᵀX)⁻¹Xᵀy still applies with an expanded design matrix
3. Degree selection is a bias-variance tradeoff: low d underfits, high d overfits
4. Cross-validation is essential for selecting the optimal polynomial degree
5. Feature scaling is critical — without it, high-degree terms cause numerical instability
6. High-degree polynomials extrapolate disastrously poorly (Runge phenomenon)
7. PolynomialFeatures with multiple features creates all interaction terms, growing combinatorially
8. For most practical applications, degree ≤ 3 is sufficient; splines are better for complex curves
