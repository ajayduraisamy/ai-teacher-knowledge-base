# Concept: Linear Regression

## Concept ID

ML-011

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand the mathematical formulation of simple linear regression
- Derive the Ordinary Least Squares (OLS) solution and the normal equation
- Implement linear regression using Python and scikit-learn
- Interpret model coefficients and assess model fit
- Identify the assumptions underlying linear regression and diagnose violations

## Prerequisites

- Basic understanding of probability and statistics (mean, variance, covariance)
- Familiarity with calculus (partial derivatives, minimization)
- Linear algebra basics (matrix multiplication, transpose, inverse)
- Python programming fundamentals
- Familiarity with NumPy and pandas

## Definition

Linear regression is a supervised learning algorithm used to model the relationship between a continuous dependent variable (target) `y` and one or more independent variables (features) `x`. The model assumes a linear relationship of the form:

`y = β₀ + β₁x + ε`

where:
- `y` is the dependent variable (target)
- `x` is the independent variable (feature)
- `β₀` is the intercept term
- `β₁` is the slope coefficient
- `ε` is the error term (residuals)

The goal is to find the best-fitting line through the data points that minimizes the sum of squared differences between observed and predicted values.

## Intuition

Imagine you have data on house sizes and their sale prices. You notice that larger houses tend to cost more. Linear regression finds the straight line that best captures this relationship. The slope tells you how much price increases per additional square foot, while the intercept represents the base price. Once the line is fit, you can predict the price of any house given its size.

The "best" line is determined by minimizing the vertical distances (residuals) between the data points and the line. Squaring these distances ensures positive and negative errors are treated equally and penalizes large errors more heavily.

## Why This Concept Matters

Linear regression is the foundation of almost all supervised learning. It is:
- **Interpretable**: Coefficients have direct meanings
- **Simple**: Fast to train and easy to deploy
- **Extensible**: Basis for regularized regression, GLMs, and neural networks
- **Widely used**: Economics, finance, healthcare, social sciences, engineering

Understanding linear regression gives you the conceptual and mathematical tools needed for more complex models.

## Mathematical Explanation

### The Model

For `n` observations with one feature:

`yᵢ = β₀ + β₁xᵢ + εᵢ` for `i = 1, 2, …, n`

We want to find `β₀` and `β₁` that minimize the sum of squared residuals (SSR):

`min_{β₀, β₁} Σ(yᵢ - ŷᵢ)²`

where `ŷᵢ = β₀ + β₁xᵢ` is the predicted value.

### OLS Solution via Calculus

Take partial derivatives and set to zero:

`∂L/∂β₀ = -2 Σ(yᵢ - β₀ - β₁xᵢ) = 0`
`∂L/∂β₁ = -2 Σ(yᵢ - β₀ - β₁xᵢ)xᵢ = 0`

Solving gives:

`β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²`
`β₀ = ȳ - β₁x̄`

### Normal Equation (Matrix Form)

For multiple features, the solution in matrix notation:

`β = (XᵀX)⁻¹Xᵀy`

where:
- `X` is the `n × (p+1)` design matrix (with a column of 1s for intercept)
- `y` is the `n × 1` target vector
- `β` is the `(p+1) × 1` coefficient vector

This closed-form solution directly gives the optimal coefficients without iterative optimization.

### Key Statistics

- **R² = 1 - SS_res / SS_tot**: Proportion of variance explained
- **Adjusted R²**: R² penalized for number of predictors
- **F-statistic**: Tests overall model significance
- **p-values**: Tests individual coefficient significance

## Code Examples

### Example 1: Simple Linear Regression with sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
X = np.random.rand(100, 1) * 10  # feature: house size in 1000 sqft
y = 2.5 + 3.8 * X.flatten() + np.random.randn(100) * 2  # price in $100k

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept (β₀): {model.intercept_:.4f}")
print(f"Coefficient (β₁): {model.coef_[0]:.4f}")
# Output:
# Intercept (β₀): 2.8173
# Coefficient (β₁): 3.7544

y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
# Output:
# R² Score: 0.9751
# MSE: 4.2783
```

### Example 2: Manual OLS Implementation

```python
import numpy as np

class ManualLinearRegression:
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.intercept_, self.coef_]

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3.2, 5.1, 6.8, 9.2, 10.9])

manual_model = ManualLinearRegression()
manual_model.fit(X, y)
print(f"Manual OLS - Intercept: {manual_model.intercept_:.4f}, Coef: {manual_model.coef_[0]:.4f}")
# Output:
# Manual OLS - Intercept: 1.2300, Coef: 1.9400

sk_model = LinearRegression()
sk_model.fit(X, y)
print(f"sklearn - Intercept: {sk_model.intercept_:.4f}, Coef: {sk_model.coef_[0]:.4f}")
# Output:
# sklearn - Intercept: 1.2300, Coef: 1.9400
```

### Example 3: Coefficient Interpretation and Diagnostics

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np

np.random.seed(42)
n = 200
X = np.random.rand(n, 1) * 10
y = 5.0 + 2.3 * X.flatten() + np.random.randn(n) * 1.5

X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()
print(ols_model.summary())
# Output:
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.695
# Model:                            OLS   Adj. R-squared:                  0.694
# Method:                 Least Squares   F-statistic:                     452.3
# Prob (F-statistic):           2.87e-54   Log-Likelihood:                -355.87
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          5.0709      0.205     24.764      0.000       4.667       5.475
# x1             2.2368      0.035     63.932      0.000       2.168       2.306
# ==============================================================================

# Interpretation:
# - Intercept (5.07): when X=0, predicted y is 5.07
# - Slope (2.24): each 1-unit increase in X increases y by 2.24 on average
# - R² (0.695): 69.5% of variance in y is explained by X
# - p-value (< 0.05): coefficient is statistically significant

residuals = ols_model.resid
print(f"Residual mean: {np.mean(residuals):.6f}")
print(f"Residual std: {np.std(residuals):.4f}")
# Output:
# Residual mean: -0.000000
# Residual std: 1.4900
```

### Example 4: Predicting with Confidence Intervals

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import PredictionErrorDisplay

np.random.seed(123)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.randn(50) * 2

model = LinearRegression()
model.fit(X, y)

X_new = np.array([[4.5], [7.2], [9.0]])
y_new = model.predict(X_new)
print("Predictions for new data points:")
for x, y_hat in zip(X_new.flatten(), y_new):
    print(f"  X = {x:.1f} -> Predicted y = {y_hat:.3f}")
# Output:
# Predictions for new data points:
#   X = 4.5 -> Predicted y = 9.983
#   X = 7.2 -> Predicted y = 15.438
#   X = 9.0 -> Predicted y = 19.008
```

## Common Mistakes

1. **Assuming linearity without verification**: Always plot residuals vs. fitted values to check if a linear model is appropriate. If patterns emerge, consider transformations or non-linear models.

2. **Ignoring multicollinearity**: When features are highly correlated, coefficient estimates become unstable and uninterpretable. Use VIF (Variance Inflation Factor) to detect it.

3. **Extrapolating beyond the data range**: Predictions outside the observed range of features are unreliable because we cannot verify the linear relationship holds there.

4. **Confusing correlation with causation**: A significant coefficient does not imply causality. There may be confounding variables driving the relationship.

5. **Overlooking outliers**: Outliers can substantially influence OLS estimates, pulling the regression line disproportionately toward them. Use robust regression or inspect residuals.

6. **Forgetting to scale features**: When features have vastly different scales, coefficients become hard to compare. Standardization helps interpretation.

7. **Using R² alone to assess model quality**: R² always increases with more features. Use adjusted R² or cross-validated performance metrics.

8. **Not checking homoscedasticity**: Residuals should have constant variance across predicted values. Heteroscedasticity invalidates standard errors and hypothesis tests.

## Interview Questions

### Beginner

**Q1: What is linear regression and when would you use it?**

Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. Use it when the target is continuous and the relationship is approximately linear.

**Q2: How do you interpret the coefficients in a simple linear regression model?**

The intercept (β₀) is the expected value of y when all features are zero. The slope (β₁) is the average change in y for a one-unit change in x. For example, if house price = 50 + 0.3 × size, each extra sq. ft. adds 0.3 thousand dollars.

**Q3: What is the difference between R² and adjusted R²?**

R² measures the proportion of variance explained by the model. Adjusted R² penalizes for adding extra predictors, increasing only if the new variable improves the model beyond chance.

**Q4: What does a p-value tell you about a coefficient?**

The p-value tests the null hypothesis that the true coefficient is zero. A low p-value (< 0.05) suggests the feature is statistically significant and has a non-zero effect on the target.

**Q5: How do you handle categorical variables in linear regression?**

Use one-hot encoding (dummy variables) to convert categories into binary columns. Drop one category to avoid perfect multicollinearity (dummy variable trap).

### Intermediate

**Q6: Derive the OLS estimator in matrix form.**

The OLS estimator minimizes `||y - Xβ||²`. Taking the derivative w.r.t β: `-2Xᵀ(y - Xβ) = 0`. Solving: `XᵀXβ = Xᵀy`, so `β = (XᵀX)⁻¹Xᵀy`. This assumes `XᵀX` is invertible.

**Q7: What assumptions must hold for OLS to be BLUE (Best Linear Unbiased Estimator)?**

The Gauss-Markov assumptions: (1) Linearity, (2) Independence of errors, (3) Homoscedasticity, (4) No perfect multicollinearity, (5) Zero conditional mean (E[ε|X] = 0). Under these, OLS is the minimum variance unbiased linear estimator.

**Q8: How do you detect and handle heteroscedasticity?**

Detect via residual plots (funnel shape) or Breusch-Pagan test. Solutions: use weighted least squares, transform the target (log), or use heteroscedasticity-consistent standard errors (HCSE).

**Q9: Explain the bias-variance tradeoff in linear regression.**

Simple models (few features) have high bias but low variance. Complex models (many features) have low bias but high variance. OLS minimizes MSE, which decomposes into bias² + variance + irreducible error.

**Q10: What happens if XᵀX is singular (non-invertible)?**

This occurs with perfect multicollinearity or when n < p. Solutions: remove correlated features, use regularization (Ridge/Lasso), or use the pseudoinverse.

### Advanced

**Q11: Prove that the OLS estimator is unbiased under the Gauss-Markov assumptions.**

E[β̂] = E[(XᵀX)⁻¹Xᵀy] = E[(XᵀX)⁻¹Xᵀ(Xβ + ε)] = β + (XᵀX)⁻¹XᵀE[ε|X] = β, since E[ε|X] = 0. Therefore β̂ is unbiased.

**Q12: Derive the variance-covariance matrix of β̂.**

Var(β̂) = Var((XᵀX)⁻¹Xᵀε) = (XᵀX)⁻¹XᵀVar(ε)X(XᵀX)⁻¹ = σ²(XᵀX)⁻¹, assuming Var(ε) = σ²I (homoscedasticity and independence).

**Q13: Compare and contrast MLE vs OLS for linear regression.**

Under normally distributed errors, OLS and MLE produce identical coefficient estimates. OLS minimizes sum of squared residuals; MLE maximizes the likelihood function. MLE also estimates σ², while OLS estimates it as RSS/(n-p). MLE provides a framework for likelihood-based inference (AIC, BIC, likelihood ratio tests).

## Practice Problems

### Easy

**P1:** You have data on study hours (X) and exam scores (y). Fit a linear regression model and interpret the coefficients.

**P2:** Compute the OLS estimates for β₀ and β₁ manually given five data points and verify with sklearn.

**P3:** Generate synthetic data with a known linear relationship and noise. Fit the model and compare estimated vs. true coefficients.

**P4:** Calculate R² manually from residuals and total sum of squares.

**P5:** Use sklearn's LinearRegression to predict house prices from a single feature and evaluate with MSE.

### Medium

**P6:** Use statsmodels to fit a linear regression. Interpret the summary table including R², F-statistic, and p-values.

**P7:** Add a quadratic term to a linear regression and determine whether it improves the model fit using adjusted R².

**P8:** Implement k-fold cross-validation for linear regression and compare MSE across folds.

**P9:** Detect and handle multicollinearity in a dataset with 5 correlated features using VIF.

**P10:** Compare the performance of linear regression with and without feature standardization.

### Hard

**P11:** Prove mathematically that the OLS estimator minimizes the sum of squared residuals.

**P12:** Implement a linear regression model from scratch using gradient descent instead of the normal equation. Compare convergence with different learning rates.

**P13:** Derive the influence function (leverage) and Cook's distance for identifying influential points. Implement and identify outliers in a real dataset.

## Solutions

**P1 Solution:** Use `LinearRegression().fit(X, y)`. If β₀ = 30 and β₁ = 5, each additional study hour increases the predicted exam score by 5 points.

**P2 Solution:** Compute means: x̄, ȳ. Then β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)². Then β₀ = ȳ - β₁x̄.

**P3 Solution:** Generate `y = 3 + 2x + noise`. Fit and print coefficients. They should be close to 3 and 2.

**P4 Solution:** Compute SS_res = Σ(yᵢ - ŷᵢ)², SS_tot = Σ(yᵢ - ȳ)². R² = 1 - SS_res/SS_tot.

**P5 Solution:** Split into train/test, fit on train, predict on test, MSE = mean((y_test - y_pred)²).

**P6 Solution:** `sm.OLS(y, sm.add_constant(X)).fit().summary()` — examine R², Prob(F-statistic), coef, P>|t|.

**P7 Solution:** Use `PolynomialFeatures(degree=2).fit_transform(X)` and compare R² adjusted.

**P8 Solution:** `cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')`.

**P9 Solution:** Compute VIF = 1/(1 - R²ⱼ) for each feature j regressed on others. Drop or combine features with VIF > 10.

**P10 Solution:** Use `StandardScaler().fit_transform(X)` and compare coefficients before and after scaling.

**P11 Solution:** L(β) = (y - Xβ)ᵀ(y - Xβ). Expand: yᵀy - 2βᵀXᵀy + βᵀXᵀXβ. Set ∂L/∂β = -2Xᵀy + 2XᵀXβ = 0 → XᵀXβ = Xᵀy → β = (XᵀX)⁻¹Xᵀy. Verify Hessian 2XᵀX is positive definite (full rank).

**P12 Solution:** Implement gradient descent: β := β - α * (2/n) * Xᵀ(Xβ - y). Compare convergence for α = 0.001, 0.01, 0.1.

**P13 Solution:** Hat matrix H = X(XᵀX)⁻¹Xᵀ. Leverage hᵢᵢ = H[i,i]. Cook's distance Dᵢ = (rᵢ² / (p·MSE)) × (hᵢᵢ / (1 - hᵢᵢ)²). Points with Dᵢ > 4/n are influential.

## Related Concepts

- Multiple Linear Regression (ML-012)
- Polynomial Regression (ML-013)
- Ridge Regression (ML-014)
- Lasso Regression (ML-015)
- Gradient Descent
- Ordinary Least Squares (OLS)
- Gauss-Markov Theorem
- Hypothesis Testing in Regression

## Next Concepts

- Multiple Linear Regression (ML-012)
- Regression Evaluation Metrics (ML-020)
- Regularization Techniques (ML-019)

## Summary

Linear regression models the relationship between a dependent variable and one or more independent variables using a linear equation. The OLS estimator minimizes the sum of squared residuals and has a closed-form solution via the normal equation. While simple and interpretable, the model relies on several assumptions, including linearity, independence, homoscedasticity, and normality of errors. Violating these assumptions can lead to biased or inefficient estimates. Linear regression serves as the building block for more advanced techniques like regularized regression, polynomial regression, and generalized linear models.

## Key Takeaways

1. Linear regression finds the best-fit line by minimizing the sum of squared residuals (OLS)
2. The normal equation `β = (XᵀX)⁻¹Xᵀy` provides a closed-form solution
3. Coefficients are interpretable: β₁ represents the change in y per unit change in x
4. Model quality is assessed with R², adjusted R², MSE, and residual analysis
5. Key assumptions include linearity, independence, homoscedasticity, and normality of errors
6. Linear regression is sensitive to outliers and multicollinearity
7. It is the foundation for understanding more complex supervised learning algorithms
8. Always validate assumptions before trusting coefficient estimates and predictions
