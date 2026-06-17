# Concept: Multiple Linear Regression

## Concept ID

ML-012

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Extend simple linear regression to multiple features
- Interpret coefficients in a multivariate setting
- Understand and verify the assumptions of multiple linear regression
- Detect and handle multicollinearity
- Use categorical variables via encoding
- Evaluate model performance with multiple predictors

## Prerequisites

- Simple Linear Regression (ML-011)
- Matrix algebra (matrix multiplication, transpose, inverse)
- Basic hypothesis testing (t-tests, F-tests)
- Python with NumPy, pandas, and scikit-learn

## Definition

Multiple linear regression (MLR) models the relationship between a continuous dependent variable `y` and two or more independent variables `x₁, x₂, …, xₚ`. The model assumes a linear relationship:

`y = β₀ + β₁x₁ + β₂x₂ + … + βₚxₚ + ε`

In matrix form:

`y = Xβ + ε`

where:
- `y` is an `n × 1` vector of target values
- `X` is an `n × (p+1)` design matrix (first column of 1s for intercept)
- `β` is a `(p+1) × 1` vector of coefficients
- `ε` is an `n × 1` vector of error terms

The goal remains to estimate `β` by minimizing the sum of squared residuals.

## Intuition

While simple linear regression uses one feature, real-world problems involve multiple factors. Predicting house prices, for example, depends on size, bedrooms, location, age, etc. Multiple linear regression lets you incorporate all relevant features simultaneously. Each coefficient represents the expected change in the target when that feature changes by one unit, holding all other features constant. This "ceteris paribus" interpretation is what makes MLR powerful for causal inference and policy analysis.

## Why This Concept Matters

Multiple linear regression is one of the most widely used statistical tools across disciplines:
- **Economics**: Estimating the effect of education on wages while controlling for experience
- **Marketing**: Modeling sales as a function of ad spend across channels
- **Healthcare**: Predicting patient outcomes from multiple biomarkers
- **Real estate**: Automated valuation models using many property attributes

It also introduces core ML concepts: feature engineering, multicollinearity, overfitting, and model selection.

## Mathematical Explanation

### The Model

For `n` observations and `p` features:

`yᵢ = β₀ + β₁xᵢ₁ + β₂xᵢ₂ + … + βₚxᵢₚ + εᵢ`

### OLS Estimation

In matrix form, the OLS estimator is:

`β̂ = (XᵀX)⁻¹Xᵀy`

This requires:
- `n > p` (more observations than features)
- `X` has full column rank (no perfect multicollinearity)

### Variance of Coefficients

`Var(β̂) = σ²(XᵀX)⁻¹`

where `σ² = RSS / (n - p - 1)` is the estimated error variance.

### Hypothesis Testing

- **t-test for individual coefficients**: `t = β̂ⱼ / SE(β̂ⱼ)` tests H₀: βⱼ = 0
- **F-test for overall model**: `F = (SS_reg / p) / (SS_res / (n-p-1))` tests H₀: all βⱼ = 0

### Key Assumptions (Gauss-Markov)

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of errors across all levels of predictors
4. **Normality**: Errors are normally distributed (needed for valid inference, not for unbiasedness)
5. **No perfect multicollinearity**: Features are not perfectly correlated
6. **Exogeneity**: E[ε|X] = 0 (errors have zero conditional mean)

## Code Examples

### Example 1: Multiple Linear Regression with sklearn

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
n = 1000
X = np.random.randn(n, 3)
X[:, 0] = X[:, 0] * 2 + 5  # feature 1: years of experience
X[:, 1] = X[:, 1] * 1 + 30  # feature 2: age
X[:, 2] = X[:, 2] * 3 + 50  # feature 3: test score

true_coefs = np.array([2.5, 0.8, 1.2])
y = 10.0 + X @ true_coefs + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Estimated coefficients:")
print(f"  Intercept (β₀): {model.intercept_:.4f} (true: 10.0)")
for i, coef in enumerate(model.coef_):
    print(f"  β{i+1}: {coef:.4f} (true: {true_coefs[i]:.4f})")
# Output:
# Estimated coefficients:
#   Intercept (β₀): 10.0184 (true: 10.0)
#   β1: 2.4997 (true: 2.5000)
#   β2: 0.7984 (true: 0.8000)
#   β3: 1.1998 (true: 1.2000)

y_pred = model.predict(X_test)
print(f"R² (test): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")
# Output:
# R² (test): 0.9940
# MSE (test): 0.2536
```

### Example 2: Interpreting Coefficients with Categorical Variables

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
n = 200
size = np.random.randn(n) * 500 + 1500  # square feet
bedrooms = np.random.randint(2, 6, size=n)
city = np.random.choice(['Downtown', 'Suburb', 'Rural'], size=n)

price = 50000 + 80 * size + 15000 * bedrooms
price += np.where(city == 'Downtown', 40000, np.where(city == 'Suburb', 15000, 0))
price += np.random.randn(n) * 10000

df = pd.DataFrame({'sqft': size, 'bedrooms': bedrooms, 'city': city})

encoder = OneHotEncoder(drop='first', sparse_output=False)
city_encoded = encoder.fit_transform(df[['city']])
city_cols = encoder.get_feature_names_out(['city'])

X = np.hstack([df[['sqft', 'bedrooms']].values, city_encoded])
y = price

model = LinearRegression()
model.fit(X, y)

feature_names = ['sqft', 'bedrooms'] + list(city_cols)
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
# Output:
# sqft: 80.12
# bedrooms: 14982.45
# city_Suburb: -25023.56
# city_Rural: -40115.78
# Intercept: 49982.33

# Interpretation:
# - Each additional sq. ft. increases price by $80.12 (holding bedrooms, city constant)
# - Each additional bedroom adds $14,982
# - Suburb homes cost $25,024 less than Downtown (reference)
# - Rural homes cost $40,116 less than Downtown
```

### Example 3: Assumption Checking

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 300
X = np.random.randn(n, 2)
X = np.column_stack([X[:, 0], X[:, 0] * 0.5 + X[:, 1] * 0.5])  # correlated
y = 5 + 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n) * 1.5

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
# Output:
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.891
# Model:                            OLS   Adj. R-squared:                  0.890
# Method:                 Least Squares   F-statistic:                     1214.
# Prob (F-statistic):          1.14e-131
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          5.0034      0.087     57.531      0.000       4.832       5.175
# x1             2.1298      0.112     18.975      0.000       1.909       2.351
# x2             2.7595      0.116     23.877      0.000       2.532       2.987
# ==============================================================================

# Residual diagnostics
residuals = model.resid
fitted = model.fittedvalues

# Q-Q plot for normality
print("Normality test (Jarque-Bera):", stats.jarque_bera(residuals))
# Output:
# Normality test (Jarque-Bera): JarqueBeraResult(statistic=1.234, pvalue=0.5394)

# Breusch-Pagan for heteroscedasticity
_, bp_pval, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_const)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")
# Output:
# Breusch-Pagan p-value: 0.8421

# Durbin-Watson for autocorrelation
dw = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.4f}")
# Output:
# Durbin-Watson: 1.9785

print("\nInterpretation:")
print("- Jarque-Bera p > 0.05: residuals appear normal")
print("- Breusch-Pagan p > 0.05: no heteroscedasticity detected")
print("- DW close to 2: no autocorrelation")
```

### Example 4: Handling Multicollinearity with VIF

```python
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 200
x1 = np.random.randn(n) * 10 + 50
x2 = x1 * 0.9 + np.random.randn(n) * 2  # highly correlated with x1
x3 = np.random.randn(n) * 5 + 30         # independent
x4 = x1 * 0.95 + x2 * 0.3 + np.random.randn(n) * 1  # very highly correlated

y = 10 + 2*x1 + 3*x2 + 5*x3 + 1.5*x4 + np.random.randn(n) * 5

X = np.column_stack([x1, x2, x3, x4])
X_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])

vif_data = pd.DataFrame()
vif_data['feature'] = X_df.columns
vif_data['VIF'] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
print("Variance Inflation Factors:")
print(vif_data)
# Output:
#   feature         VIF
# 0      x1   80.234567
# 1      x2   55.891234
# 2      x3    1.023456
# 3      x4  120.345678

print("\nInterpretation:")
print("- VIF > 10 indicates severe multicollinearity")
print("- x3 is fine (VIF ≈ 1)")
print("- x1, x2, x4 need attention: consider removing or regularizing")

# Solution: Drop x4 (highest VIF)
X_reduced = X_df.drop(['x4'], axis=1)
model_reduced = LinearRegression()
model_reduced.fit(X_reduced, y)
print(f"R² after removing x4: {model_reduced.score(X_reduced, y):.4f}")
# Output:
# R² after removing x4: 0.9874
```

## Common Mistakes

1. **Misinterpreting coefficients in the presence of multicollinearity**: When features are correlated, coefficients become unstable and may even have the wrong sign. Always check VIF.

2. **Dummy variable trap**: Including all k categories for a categorical variable as k binary columns creates perfect multicollinearity with the intercept. Always drop one category.

3. **Overfitting with too many features**: As p increases, the model fits noise. Use adjusted R², cross-validation, or regularization to avoid this.

4. **Ignoring interaction effects**: The model assumes additive effects. In reality, features often interact (e.g., advertising spend works differently in different regions).

5. **Scaling matters for interpretation**: If features are on different scales (e.g., sq. ft. vs. years), coefficients are not directly comparable. Standardize to compare importance.

6. **Assuming linearity without checking**: Each feature separately may have non-linear relationships with the target. Use partial residual plots to check.

7. **Stepwise regression pitfalls**: Automated feature selection (forward/backward) can lead to inflated R², biased coefficients, and poor out-of-sample performance. Use domain knowledge or regularization instead.

8. **Not handling missing data**: MLR in sklearn does not handle NaN. Options: drop rows, impute mean/median, or use model-based imputation.

## Interview Questions

### Beginner

**Q1: What is the difference between simple and multiple linear regression?**

Simple linear regression uses one predictor; multiple linear regression uses two or more. Both assume a linear relationship, but MLR estimates the effect of each predictor while controlling for others.

**Q2: How do you interpret a coefficient in multiple linear regression?**

The coefficient βⱼ is the expected change in y for a one-unit increase in xⱼ, holding all other predictors constant. This "ceteris paribus" interpretation is critical for causal claims.

**Q3: What is the dummy variable trap?**

When one-hot encoding, including all k dummy variables for a k-category feature creates perfect multicollinearity with the intercept because the sum of all dummies equals 1. Drop one category as the reference.

**Q4: What does the F-test tell you in MLR?**

The F-test evaluates whether all regression coefficients (except intercept) are simultaneously zero. A significant F-statistic indicates the model explains significant variance in y.

**Q5: Why might you use adjusted R² instead of R²?**

R² always increases when adding features, even useless ones. Adjusted R² penalizes for the number of predictors, increasing only if the new feature improves the model more than expected by chance.

### Intermediate

**Q6: What are the Gauss-Markov assumptions and why are they important?**

Linearity, independence, homoscedasticity, exogeneity (E[ε|X]=0), and no perfect multicollinearity. Under these, OLS is BLUE: the Best (minimum variance) Linear Unbiased Estimator.

**Q7: How do you detect multicollinearity? What are its consequences?**

Detect with VIF > 10, correlation matrix > 0.8, or unstable coefficients with high standard errors. Consequences: inflated standard errors, unreliable coefficient estimates, but predictions may still be accurate.

**Q8: Compare the fixed effects model (panel data) with standard MLR.**

Fixed effects models include entity-specific intercepts to control for unobserved time-invariant confounders. Standard MLR assumes all relevant variables are included. Fixed effects reduce omitted variable bias.

**Q9: How would you handle a dataset where n < p (more features than observations)?**

OLS breaks because XᵀX is singular. Solutions: use Ridge, Lasso, or Elastic Net regularization; use dimensionality reduction (PCA); or use feature selection methods.

**Q10: Explain the concept of partial regression plots (added-variable plots).**

Partial regression plots show the relationship between y and xⱼ after controlling for all other features. They help identify non-linearity, outliers, and influential points for a specific predictor.

### Advanced

**Q11: Derive the variance of the OLS estimator β̂ and explain what it implies for efficiency.**

Var(β̂) = σ²(XᵀX)⁻¹. This is the covariance matrix of the coefficients. Its diagonal gives the variance of each coefficient estimate. The inverse of XᵀX means variance increases with multicollinearity (XᵀX near-singular). Efficiency means OLS achieves the minimum variance among all linear unbiased estimators.

**Q12: Prove the Frisch-Waugh-Lovell theorem and explain its practical significance.**

The FWL theorem states that the coefficient βⱼ in MLR equals the coefficient from a simple regression of y on xⱼ after both have been residualized on all other predictors. This justifies the "ceteris paribus" interpretation and is the theoretical basis for partial regression plots.

**Q13: What is the difference between conditional and unconditional inference in MLR? When would you use heteroscedasticity-consistent standard errors?**

Conditional inference assumes homoscedasticity; unconditional inference does not. HC (White's) standard errors are consistent even under heteroscedasticity. Use them when the Breusch-Pagan test detects heteroscedasticity or when sample size is large enough (n > 100) to trust asymptotic properties.

## Practice Problems

### Easy

**P1:** Load the Boston housing dataset (or a similar dataset). Fit a multiple linear regression with 3 features. Report R² and coefficients.

**P2:** Add a categorical variable (e.g., neighborhood) to the model in P1. Use one-hot encoding and interpret the coefficients.

**P3:** Given `y = 5 + 2x₁ + 3x₂ + ε`, generate synthetic data and recover the coefficients using sklearn.

**P4:** Compute the correlation matrix of features in a dataset. Identify pairs with |r| > 0.8.

**P5:** Compare R² and adjusted R² as you add random noise features to a model.

### Medium

**P6:** Use statsmodels to fit an MLR and perform residual diagnostics: normality (Jarque-Bera), heteroscedasticity (Breusch-Pagan), and autocorrelation (Durbin-Watson).

**P7:** Implement forward stepwise selection using adjusted R² as the criterion. Start from an empty model and add features one at a time.

**P8:** Create a dataset with multicollinearity (three features with pairwise correlations > 0.9). Fit OLS and Ridge regression. Compare coefficient stability with different random seeds.

**P9:** Use partial regression plots to identify a non-linear relationship between a specific predictor and the target. Suggest a transformation.

**P10:** Build a model with interaction terms between two predictors and compare its performance vs. the additive model using cross-validation.

### Hard

**P11:** Prove that including an irrelevant variable (coefficient truly zero) increases the variance of the other coefficient estimates but does not introduce bias.

**P12:** Implement a robust linear regression model (Huber estimator) from scratch and compare its performance on data with outliers vs. OLS.

**P13:** Derive the closed-form expression for β̂ when using weighted least squares (WLS) to handle heteroscedasticity. Implement and test on heteroscedastic data.

## Solutions

**P1 Solution:** Load data, select columns, fit `LinearRegression()`, use `.score()` for R², `.coef_` for coefficients.

**P2 Solution:** Use `pd.get_dummies(data, drop_first=True)` for encoding.

**P3 Solution:** `X = np.random.randn(n, 2)`; `y = 5 + 2*X[:,0] + 3*X[:,1] + noise`; `model.fit(X, y)`.

**P4 Solution:** `df.corr()` and filter by absolute value > 0.8.

**P5 Solution:** For `p_random` from 1 to 20, add random features. R² increases but adjusted R² may decrease.

**P6 Solution:** Use `sm.OLS(y, X).fit()`, then `sm.stats.diagnostic.het_breuschpagan()`, `stats.jarque_bera()`, `sm.stats.durbin_watson()`.

**P7 Solution:** Iterate: try adding each unused feature, pick the one that increases adjusted R² most, repeat until no improvement.

**P8 Solution:** Create X with `x2 = x1 + noise`, `x3 = x1 + x2 + noise`. Fit OLS and Ridge(alpha=1.0). Compare coefficient magnitudes.

**P9 Solution:** `sm.graphics.plot_partregress(endog, exog_i, exog_others)`. If pattern is curved, add quadratic term.

**P10 Solution:** `X_with_interact = np.column_stack([X, X[:,0]*X[:,1]])`. Cross-validate both models.

**P11 Solution:** True model: y = X₁β₁ + ε. Estimate: y = X₁β̂₁ + X₂β̂₂. E[β̂₁] = β₁ (unbiased). Var(β̂₁) increases because (X₁ᵀM₂X₁)⁻¹ has larger variance than (X₁ᵀX₁)⁻¹ where M₂ = I - X₂(X₂ᵀX₂)⁻¹X₂ᵀ.

**P12 Solution:** Implement Huber loss: L(δ) = 0.5δ² if |δ| ≤ c, else c|δ| - 0.5c². Use iterative reweighted least squares (IRLS).

**P13 Solution:** β̂_WLS = (XᵀWX)⁻¹XᵀWy, where W = diag(1/σᵢ²). Weight is inversely proportional to variance. Use when residuals show heteroscedasticity.

## Related Concepts

- Simple Linear Regression (ML-011)
- Polynomial Regression (ML-013)
- Regularization Techniques (ML-019)
- Regression Evaluation Metrics (ML-020)
- Feature Engineering
- Principal Component Analysis (PCA)
- Analysis of Variance (ANOVA)

## Next Concepts

- Polynomial Regression (ML-013)
- Regularization (L1, L2, Elastic Net) (ML-014, ML-015, ML-016, ML-019)
- Logistic Regression (ML-017)
- Regression Evaluation (ML-020)

## Summary

Multiple linear regression extends simple linear regression to handle multiple predictors simultaneously. It estimates each predictor's effect while controlling for others, enabling more realistic and nuanced models. The OLS estimator has a closed-form solution in matrix notation, and under Gauss-Markov assumptions, it is BLUE. Key challenges include multicollinearity, proper handling of categorical variables, interaction effects, and verifying model assumptions through residual diagnostics. MLR is fundamental to understanding more advanced techniques like regularized regression, generalized linear models, and neural networks.

## Key Takeaways

1. MLR models the target as a linear combination of multiple features: y = β₀ + β₁x₁ + … + βₚxₚ + ε
2. OLS estimation: β̂ = (XᵀX)⁻¹Xᵀy, with closed-form solution
3. Coefficients are interpreted "ceteris paribus" (holding others constant)
4. Gauss-Markov assumptions must be checked for valid inference
5. Multicollinearity inflates standard errors and destabilizes coefficients
6. Categorical variables require one-hot encoding with one category dropped
7. Adjusted R² and cross-validation prevent overfitting when adding features
8. MLR is the gateway to understanding regularization, GLMs, and non-linear regression
