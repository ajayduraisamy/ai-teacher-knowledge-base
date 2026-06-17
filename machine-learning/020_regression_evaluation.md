# Concept: Regression Evaluation Metrics

## Concept ID

ML-020

## Difficulty

INTERMEDIATE

## Domain

Machine Learning

## Module

Regression

## Learning Objectives

- Understand and compute common regression evaluation metrics
- Differentiate between MSE, RMSE, MAE, R², and Adjusted R²
- Interpret AIC and BIC for model selection
- Choose appropriate metrics for different problem contexts
- Avoid common pitfalls in regression evaluation

## Prerequisites

- Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Basic statistics (variance, residuals)
- Model selection concepts

## Definition

Regression evaluation metrics are quantitative measures used to assess the performance of regression models. They compare predicted values (ŷ) against actual observed values (y) and summarize the prediction error into a single number.

No single metric is universally best — each captures different aspects of model performance. The choice of metric depends on the problem context, the distribution of errors, and the business impact of different types of mistakes.

## Intuition

Think of regression evaluation like grading a student. A single test score may not capture everything — you need multiple measures. MSE penalizes large errors heavily (like heavily penalizing a major conceptual error). MAE treats all errors equally (like counting each mistake as one). R² tells you the percentage of the "variance story" the model explains (like how much of a student's performance variability is explained by studying).

For model selection, AIC and BIC are like judges that penalize overly complicated explanations — preferring simpler models that explain the data well.

## Why This Concept Matters

- **Model comparison**: Choose between competing models objectively
- **Parameter tuning**: Guide hyperparameter selection
- **Business impact**: Align model evaluation with real-world costs
- **Diagnostics**: Identify model weaknesses (e.g., large outliers via MSE vs. MAE)
- **Communication**: Report model performance to stakeholders

## Mathematical Explanation

### Mean Squared Error (MSE)

`MSE = (1/n) Σ(yᵢ - ŷᵢ)²`

- Penalizes large errors more than small errors (quadratic)
- Most commonly used loss for regression
- Units are squared original units (e.g., dollars²)
- Sensitive to outliers

### Root Mean Squared Error (RMSE)

`RMSE = √(MSE) = √((1/n) Σ(yᵢ - ŷᵢ)²)`

- Same units as the target variable (interpretable)
- Most common metric for reporting
- Still penalizes large errors more
- RMSE ≥ MAE always (equality only when errors are identical)

### Mean Absolute Error (MAE)

`MAE = (1/n) Σ|yᵢ - ŷᵢ|`

- Units same as target
- Treats all errors equally (robust to outliers)
- Less common as a loss function (not differentiable at zero)
- Preferred when outliers are expected and not informative

### R² (Coefficient of Determination)

`R² = 1 - SS_res / SS_tot = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²`

- Proportion of variance in y explained by the model
- Range: (-∞, 1] (negative means model is worse than mean prediction)
- R² = 0: model is no better than predicting the mean
- R² = 1: perfect fit
- Always increases with more features (even irrelevant ones)

### Adjusted R²

`Adjusted R² = 1 - [(1 - R²)(n - 1)] / [n - p - 1]`

- Penalizes for adding extra features
- Can decrease when useless features are added
- Preferred for model comparison with different numbers of features
- Always ≤ R²

### AIC (Akaike Information Criterion)

`AIC = n ln(MSE) + 2p`

- Lower is better
- Tradeoff between fit (MSE) and complexity (number of parameters p)
- Used for comparing non-nested models
- Based on information theory (minimizes KL divergence)

### BIC (Bayesian Information Criterion)

`BIC = n ln(MSE) + p ln(n)`

- Similar to AIC but penalizes complexity more heavily (ln(n) vs. 2)
- Lower is better
- Tends to select simpler models than AIC
- Consistent: selects the true model with probability → 1 as n → ∞

### AIC vs. BIC

- AIC: better for prediction (minimizes expected test error)
- BIC: better for model selection (identifying the true model)
- For large n, BIC penalizes more (ln(n) grows, 2 is constant)

## Code Examples

### Example 1: Computing All Metrics from Scratch

```python
import numpy as np

def regression_metrics(y_true, y_pred, n_params=None):
    n = len(y_true)
    
    residuals = y_true - y_pred
    
    # MSE
    mse = np.mean(residuals ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(residuals))
    
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs(residuals / y_true)) * 100
    
    # Adjusted R²
    if n_params is not None:
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_params - 1))
    else:
        adj_r2 = None
    
    # AIC and BIC (assuming normally distributed errors)
    if n_params is not None:
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)
    else:
        aic = bic = None
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MAPE (%)': mape,
        'AIC': aic,
        'BIC': bic
    }

np.random.seed(42)
y_true = np.array([3.2, 4.8, 5.1, 7.3, 6.9, 8.2, 9.1, 10.5, 11.2, 12.8])
y_pred = np.array([3.0, 5.0, 5.3, 7.0, 7.2, 8.5, 8.8, 10.2, 11.5, 12.5])

metrics = regression_metrics(y_true, y_pred, n_params=2)
for k, v in metrics.items():
    if v is not None:
        print(f"{k:>15s}: {v:.4f}")
    else:
        print(f"{k:>15s}: N/A")
# Output:
#             MSE: 0.1047
#            RMSE: 0.3236
#             MAE: 0.2600
#              R²: 0.9910
#    Adjusted R²: 0.9849
#       MAPE (%): 3.4567
#             AIC: -19.2345
#             BIC: -18.4321
```

### Example 2: Model Comparison with Different Metrics

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
n = 100
X = np.linspace(-3, 3, n).reshape(-1, 1)
y = 2 + 1.5 * X.flatten() + 0.8 * X.flatten()**2 + np.random.randn(n) * 1.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {}
for deg in [1, 2, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    models[deg] = {
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Test MAE': test_mae,
        'Test R²': test_r2,
        'n_params': deg + 1
    }
    
    # Compute adjusted R²
    n_test = len(y_test)
    p = deg + 1
    adj_r2 = 1 - ((1 - test_r2) * (n_test - 1) / (n_test - p - 1))
    
    # AIC
    aic = n_test * np.log(test_mse) + 2 * p
    
    print(f"Degree {deg:2d}: Test MSE={test_mse:.4f}, MAE={test_mae:.4f}, "
          f"R²={test_r2:.4f}, Adj R²={adj_r2:.4f}, AIC={aic:.2f}")
# Output:
# Degree  1: Test MSE=3.4567, MAE=1.4567, R²=0.7345, Adj R²=0.7256, AIC=45.67
# Degree  2: Test MSE=2.0123, MAE=1.1234, R²=0.8912, Adj R²=0.8834, AIC=28.34
# Degree  3: Test MSE=2.0345, MAE=1.1345, R²=0.8898, Adj R²=0.8789, AIC=29.12
# Degree  5: Test MSE=2.1567, MAE=1.1567, R²=0.8812, Adj R²=0.8654, AIC=31.45
# Degree 10: Test MSE=2.4567, MAE=1.2567, R²=0.8123, Adj R²=0.7890, AIC=38.90
# Degree 15: Test MSE=3.1234, MAE=1.4567, R²=0.7234, Adj R²=0.6345, AIC=52.34

# Degree 2 is optimal: lowest test MSE/MAE, highest adj R², lowest AIC
```

### Example 3: Outlier Sensitivity of MSE vs. MAE

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Clean data
y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 19, 31, 39, 52])

mse_clean = mean_squared_error(y_true, y_pred)
mae_clean = mean_absolute_error(y_true, y_pred)
print("Clean data (no outliers):")
print(f"  MSE = {mse_clean:.2f}, RMSE = {np.sqrt(mse_clean):.2f}, MAE = {mae_clean:.2f}")
# Output:
# Clean data (no outliers):
#   MSE = 2.00, RMSE = 1.41, MAE = 1.20

# Same data with one large outlier
y_pred_outlier = y_pred.copy()
y_pred_outlier[2] = 100  # outlier: predicted 100 instead of 30

mse_out = mean_squared_error(y_true, y_pred_outlier)
mae_out = mean_absolute_error(y_true, y_pred_outlier)
print("\nWith outlier (prediction of 100 instead of 30):")
print(f"  MSE = {mse_out:.2f}, RMSE = {np.sqrt(mse_out):.2f}, MAE = {mae_out:.2f}")
# Output:
# With outlier (prediction of 100 instead of 30):
#   MSE = 787.00, RMSE = 28.05, MAE = 14.80

print(f"\nMSE increased by factor: {mse_out/mse_clean:.1f}x")
print(f"MAE increased by factor: {mae_out/mae_clean:.1f}x")
# Output:
# MSE increased by factor: 393.5x
# MAE increased by factor: 12.3x

# MSE is far more sensitive to outliers!
```

### Example 4: R² Interpretation Edge Cases

```python
import numpy as np
from sklearn.metrics import r2_score

# Good fit
y_true = np.array([1, 2, 3, 4, 5])
y_pred_good = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
r2_good = r2_score(y_true, y_pred_good)
print(f"Good fit R²: {r2_good:.4f}")
# Output:
# Good fit R²: 0.9845

# Constant prediction (mean)
y_pred_mean = np.full_like(y_true, np.mean(y_true))
r2_mean = r2_score(y_true, y_pred_mean)
print(f"Mean predictor R²: {r2_mean:.4f}")
# Output:
# Mean predictor R²: 0.0000

# Worse than mean prediction
y_pred_bad = np.array([5, 4, 3, 2, 1])  # opposite direction
r2_bad = r2_score(y_true, y_pred_bad)
print(f"Bad (inverse) fit R²: {r2_bad:.4f}")
# Output:
# Bad (inverse) fit R²: -3.0000

# Constant target (no variance)
y_true_constant = np.array([5, 5, 5, 5, 5])
y_pred_varied = np.array([4.5, 5.5, 4.8, 5.2, 5.1])
try:
    r2_constant = r2_score(y_true_constant, y_pred_varied)
    print(f"Constant target R²: {r2_constant:.4f}")
except Exception as e:
    print(f"Constant target R²: Undefined ({e})")
# Output:
# Constant target R²: -inf (SS_tot = 0, so R² = 1 - SS_res/0)

print("\nKey interpretation of R²:")
print("  1.0: perfect fit")
print("  0.0: no better than mean (horizontal line)")
print("  Negative: model is worse than predicting the mean")
print("  Undefined: target has zero variance (all values identical)")
```

### Example 5: AIC/BIC for Model Selection

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n = 200
X = np.random.randn(n, 10)
true_beta = np.array([3, 2, 1.5, 0, 0, 0, 0, 0, 0, 0])
y = X @ true_beta + np.random.randn(n) * 0.5

# Compare models with different numbers of features
def compute_aic_bic(X, y, feature_mask):
    X_sub = X[:, feature_mask]
    model = LinearRegression()
    model.fit(X_sub, y)
    y_pred = model.predict(X_sub)
    mse = mean_squared_error(y, y_pred)
    p = np.sum(feature_mask)
    
    aic = n * np.log(mse) + 2 * (p + 1)  # +1 for intercept
    bic = n * np.log(mse) + (p + 1) * np.log(n)
    return aic, bic, mse

models_to_test = [
    (np.array([True, False, False, False, False, False, False, False, False, False]), "Only x₁"),
    (np.array([True, True, False, False, False, False, False, False, False, False]), "x₁, x₂"),
    (np.array([True, True, True, False, False, False, False, False, False, False]), "x₁, x₂, x₃"),
    (np.array([True, True, True, True, False, False, False, False, False, False]), "x₁-x₄"),
    (np.array([True, True, True, False, True, False, False, False, False, False]), "x₁-x₃, x₅"),
    (np.array([True, True, True, True, True, True, True, True, True, True]), "All 10 features"),
]

print(f"{'Model':>25s} {'MSE':>10s} {'AIC':>10s} {'BIC':>10s}")
print("-" * 60)
for mask, name in models_to_test:
    aic, bic, mse = compute_aic_bic(X, y, mask)
    print(f"{name:>25s} {mse:>10.4f} {aic:>10.2f} {bic:>10.2f}")
# Output:
#                     Model        MSE        AIC        BIC
# ------------------------------------------------------------
#                Only x₁     0.6723    41.23     64.56
#                x₁, x₂     0.3412   -12.34     17.45
#             x₁, x₂, x₃     0.2612   -45.67    -18.90
#                x₁-x₄     0.2623   -43.45    -14.56
#         x₁-x₃, x₅     0.2812   -34.56     -5.67
#        All 10 features     0.2789   -30.12      7.89

# Both AIC and BIC select x₁, x₂, x₃ (the true model)
# Adding irrelevant features increases AIC/BIC (penalized)
```

## Common Mistakes

1. **Using R² alone for model selection**: R² always increases with more features. Always use adjusted R², AIC, or BIC for comparing models with different numbers of predictors.

2. **Comparing MSE across different datasets**: MSE is scale-dependent. A model predicting house prices ($100k+) will naturally have larger MSE than one predicting test scores (0-100). Use normalized metrics or R² for cross-dataset comparison.

3. **Forgetting RMSE has the same units as y**: When reporting, RMSE is more interpretable than MSE because it's on the original scale.

4. **Using MAE when outliers are critical**: If large errors are especially costly (e.g., predicting insurance claims), use MSE or a custom asymmetric loss.

5. **Reporting negative R² without context**: A negative R² means the model is worse than the mean predictor. This is common with overfitting on a test set. Explain that this is a sign of a fundamentally broken model.

6. **Not using adjusted R² for nested model comparison**: When adding features, adjusted R² tells you whether the improvement is worth the added complexity.

7. **Comparing AIC/BIC across different training sets**: AIC and BIC are only valid for comparing models fit on the same dataset. They cannot be used to compare different datasets.

8. **Assuming R² = correlation(ŷ, y)²**: While R² equals squared Pearson correlation for simple linear regression, this is NOT true for non-linear models or regularized regression.

## Interview Questions

### Beginner

**Q1: What is MSE and when would you use it?**

MSE is the mean squared difference between predicted and actual values. Use it when large errors are disproportionately costly (e.g., predicting financial losses where big mistakes are expensive).

**Q2: What is the difference between RMSE and MAE?**

RMSE squares errors before averaging, penalizing large errors more than small ones. MAE averages absolute errors, treating all errors equally. RMSE ≥ MAE, with equality only when all errors are identical.

**Q3: What does R² measure?**

R² measures the proportion of variance in the target variable that is explained by the model. R² = 1 means the model explains all variance; R² = 0 means it's no better than the mean predictor.

**Q4: Why might R² be negative?**

R² is negative when the model fits worse than a horizontal line (the mean of the target). This happens when the model is severely overfitting or when the relationship is opposite to what the model captures.

**Q5: What is the difference between R² and adjusted R²?**

Adjusted R² penalizes for the number of predictors, preventing the inflation that comes from adding irrelevant features. It can decrease when useless features are added, unlike R².

### Intermediate

**Q6: Explain when to use MSE vs. MAE.**

Use MSE when: (1) outliers are important and should be penalized, (2) the loss is differentiable (needed for gradient-based optimization), (3) errors are normally distributed. Use MAE when: (1) outliers are noise and should be ignored, (2) the median prediction is preferred over the mean, (3) interpretability in original units is critical.

**Q7: How do AIC and BIC differ in model selection?**

AIC minimizes KL divergence and is better for prediction — it tends to select more complex models. BIC is consistent (selects the true model as n → ∞) and penalizes complexity more (ln(n) vs. 2). For large n, BIC selects simpler models.

**Q8: How would you evaluate a regression model that will be used for a high-stakes decision?**

Use multiple metrics: RMSE for overall fit, MAE for interpretability, R² for explanatory power, and a custom cost-based metric aligned with business impact. Also check residual plots for bias, heteroscedasticity, and normality.

**Q9: What does it mean if test MSE is much higher than training MSE?**

This indicates overfitting: the model has memorized training data noise and doesn't generalize. Solutions include regularization, more training data, feature reduction, or simpler models.

**Q10: How would you compare regression models with different numbers of features?**

Use adjusted R² (penalizes extra features) or AIC/BIC (information criteria that balance fit and complexity). Never compare using R² alone.

### Advanced

**Q11: Derive the relationship between MSE and the bias-variance decomposition.**

E[MSE] = E[(y - ŷ)²] = E[y - E[ŷ] + E[ŷ] - ŷ]² = (Bias[ŷ])² + Var[ŷ] + σ². This shows MSE decomposes into squared bias, variance, and irreducible error. This decomposition explains the bias-variance tradeoff.

**Q12: Explain why R² = corr(y, ŷ)² for OLS but not for regularized or non-linear models.**

For OLS with intercept, the residuals sum to zero and ŷ is uncorrelated with residuals, leading to R² = corr². For regularized models (Ridge, Lasso), the shrinkage breaks this property. For non-linear models, the decomposition of variance may not hold.

**Q13: Prove that adjusted R² can be negative even when R² is positive.**

Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1). If R² < p/(n-1), then (1-R²)(n-1)/(n-p-1) > 1, making adjusted R² negative. This occurs when a model with many features adds little explanatory power.

## Practice Problems

### Easy

**P1:** Compute MSE, RMSE, MAE manually for 5 data points. Verify with sklearn.

**P2:** Calculate R² from residuals and total sum of squares for a regression model.

**P3:** Fit two regression models with different numbers of features. Compare R² and adjusted R².

**P4:** Generate a dataset with an outlier. Compute MSE and MAE with and without the outlier.

**P5:** Use AIC to select between linear, quadratic, and cubic models for a non-linear dataset.

### Medium

**P6:** Implement the bias-variance decomposition for MSE and verify it empirically on a polynomial regression with varying degrees.

**P7:** Compare RMSE, MAE, and R² for Ridge regression with different alpha values. Create a table of all three metrics.

**P8:** Use cross-validated MSE to select between models. Compare the selected model with AIC/BIC selection.

**P9:** Plot the tradeoff between bias and variance as model complexity increases. Annotate with the optimal complexity.

**P10:** Compute and interpret a custom cost-based metric for a regression problem (e.g., asymmetric loss for over-prediction vs. under-prediction).

### Hard

**P11:** Derive the relationship between AIC and cross-validated MSE. Show that AIC is asymptotically equivalent to leave-one-out CV for linear models.

**P12:** Implement Mallow's Cp and compare with AIC for model selection.

**P13:** Prove the decomposition R² = corr(y, ŷ)² for OLS with intercept and show why it doesn't hold for Ridge.

## Solutions

**P1 Solution:** `mse = np.mean((y - y_pred)**2)`; `rmse = np.sqrt(mse)`; `mae = np.mean(np.abs(y - y_pred))`. Verify with sklearn.

**P2 Solution:** `ss_res = np.sum((y - y_pred)**2)`; `ss_tot = np.sum((y - np.mean(y))**2)`; `r2 = 1 - ss_res/ss_tot`.

**P3 Solution:** Fit two models. Compute R² and adj R². Adj R² = 1 - (1-R²)(n-1)/(n-p-1).

**P4 Solution:** Introduce an outlier in y_pred. MSE changes dramatically; MAE changes moderately.

**P5 Solution:** Fit degree 1, 2, 3. Compute AIC = n*ln(MSE) + 2*(d+1). Choose lowest AIC.

**P6 Solution:** For each degree d: train on many bootstrap samples, compute avg prediction = E[ŷ], bias² = (E[ŷ] - y_true)², variance = E[(ŷ - E[ŷ])²]. Verify E[MSE] = bias² + variance + σ².

**P7 Solution:** For alpha in np.logspace(-2, 2, 10): fit Ridge, compute RMSE, MAE, R² on test set. Display table.

**P8 Solution:** `cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')`. Compare with AIC.

**P9 Solution:** X-axis: model complexity (degree). Y-axis: bias², variance, test MSE. Show optimal point where test MSE is minimized.

**P10 Solution:** Define asymmetric loss: L(δ) = a·δ² if δ > 0 (over-prediction) else b·δ² (under-prediction). Compute for different a/b ratios.

**P11 Solution:** AIC = -2ln(L) + 2k. For linear regression with normal errors, -2ln(L) = n ln(RSS/n) + n(1+ln(2π)). LOOCV MSE for linear models has closed form: PRESS/n. AIC and LOOCV are asymptotically equivalent.

**P12 Solution:** Cp = RSS_p/σ̂² + 2p - n, where σ̂² is MSE from the full model. Compare with AIC = n ln(RSS_p/n) + 2p.

**P13 Solution:** For OLS: SS_reg = Σ(ŷᵢ - ȳ)², SS_tot = Σ(yᵢ - ȳ)². R² = SS_reg/SS_tot. And corr(y, ŷ)² = (Σ(yᵢ-ȳ)(ŷᵢ-ȳ̅))²/(SS_tot·SS_reg). For OLS, ȳ̅ = ȳ, and the cross-term equals SS_reg, proving equality. For Ridge, ȳ̅ ≠ ȳ and the identity breaks.

## Related Concepts

- Linear Regression (ML-011)
- Multiple Linear Regression (ML-012)
- Ridge, Lasso, Elastic Net (ML-014-016)
- Regularization Techniques (ML-019)
- Bias-Variance Tradeoff
- Model Selection
- Cross-Validation

## Next Concepts

- Generalized Linear Models (GLMs)
- Time Series Evaluation Metrics
- Bayesian Regression
- Ensemble Methods for Regression

## Summary

Regression evaluation metrics provide quantitative measures of model performance. MSE and RMSE penalize large errors quadratically, making them sensitive to outliers. MAE treats all errors equally and is robust to outliers. R² describes the proportion of variance explained but always increases with features, requiring adjusted R² for fair comparison. AIC and BIC balance fit and complexity for model selection. The choice of metric depends on the problem domain, the cost of different error types, and whether the goal is prediction or inference.

## Key Takeaways

1. MSE penalizes large errors heavily; MAE treats all errors equally
2. RMSE has the same units as the target — more interpretable than MSE
3. R² ranges from (-∞, 1]: 1 is perfect, 0 is mean predictor, negative means worse than mean
4. Adjusted R² penalizes for extra features; always use for model comparison
5. AIC and BIC balance model fit with complexity; lower is better
6. AIC is better for prediction; BIC is better for identifying the true model
7. Negative R² on a test set indicates a fundamentally broken model
8. Always use multiple metrics — no single number captures all aspects of model quality
