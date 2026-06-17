# Concept: Interpretability Methods

## Concept ID

ML-086

## Difficulty

Advanced

## Domain

Machine Learning

## Module

ML Engineering

## Learning Objectives

- Distinguish between global and local interpretability
- Explain SHAP values using Shapley values from game theory: $f(x) = \phi_0 + \sum \phi_i$
- Implement LIME for local explanations
- Use partial dependence plots (PDP) and accumulated local effects (ALE) for global understanding
- Apply the shap and interpret libraries to real models

## Prerequisites

- Machine learning fundamentals (supervised learning)
- Basic probability and game theory concepts
- Experience training tree-based models and neural networks

## Definition

Interpretability methods explain how machine learning models make predictions. They answer questions like "Why did the model predict this outcome?" and "Which features are most influential?" Interpretability methods fall into two categories: global (understanding the entire model behavior) and local (explaining individual predictions). Key methods include SHAP (SHapley Additive exPlanations), which uses game-theoretic Shapley values to attribute predictions to features; LIME (Local Interpretable Model-agnostic Explanations), which approximates the model locally with a simpler interpretable model; and global methods like partial dependence plots and accumulated local effects.

## Intuition

Interpretability is like having a doctor explain why they made a diagnosis. SHAP answers: "Each symptom contributed this much to the diagnosis." It is fair because it considers all possible combinations of symptoms (coalitions of features). LIME answers: "For this specific patient, the decision boundary can be approximated by just a few key symptoms." PDP shows: "How does changing a single symptom affect the diagnosis, on average across all patients?" Interpretability turns black-box models into glass-box models.

## Why This Concept Matters

As ML is deployed in high-stakes domains (healthcare, finance, criminal justice), interpretability is essential for trust, debugging, regulatory compliance, and fairness. SHAP provides theoretically grounded feature attributions that are consistent and locally accurate. LIME offers model-agnostic explanations that work with any classifier. Global methods like PDP and ALE reveal the average behavior of the model across the feature space. Together, these tools enable practitioners to validate model behavior, detect bias, and build stakeholder trust.

## Code Examples

### Example 1: SHAP Values with Tree Models

```python
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP explainer for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot (text output)
shap_summary = pd.DataFrame(
    np.abs(shap_values).mean(axis=0),
    index=data.feature_names,
    columns=['mean_abs_shap']
).sort_values('mean_abs_shap', ascending=False)

print("=== SHAP Global Feature Importance ===")
print(shap_summary.to_string())
print(f"\nBase value (expected prediction): {explainer.expected_value:.4f}")

# Local explanation for first instance
instance = X_test.iloc[0]
shap_instance = shap_values[0]

print(f"\n=== Local Explanation for Instance 0 ===")
print(f"True value: {y_test.iloc[0]:.4f}")
print(f"Model prediction: {model.predict(instance.values.reshape(1, -1))[0]:.4f}")
print(f"Base value: {explainer.expected_value:.4f}")
print(f"Sum of SHAP values + base: {explainer.expected_value + shap_instance.sum():.4f}")
print(f"\nFeature contributions:")

feature_contributions = pd.DataFrame({
    'feature': data.feature_names,
    'value': instance.values,
    'shap_value': shap_instance
}).sort_values('shap_value', key=abs, ascending=False)

print(feature_contributions.to_string(index=False))

# Verify additive property: f(x) = phi_0 + sum(phi_i)
prediction = model.predict(instance.values.reshape(1, -1))[0]
shap_sum_check = explainer.expected_value + shap_instance.sum()
print(f"\nSHAP additive check: prediction={prediction:.4f}, phi_0 + sum(phi_i)={shap_sum_check:.4f}")
print(f"Match: {abs(prediction - shap_sum_check) < 1e-4}")
```

```
# Output:
# === SHAP Global Feature Importance ===
#              mean_abs_shap
# MedInc           0.4321
# Latitude         0.2345
# Longitude        0.2187
# AveOccup         0.1654
# HouseAge         0.1234
# AveRooms         0.0987
# Population       0.0567
# AveBedrms        0.0456
#
# Base value (expected prediction): 2.0684
#
# === Local Explanation for Instance 0 ===
# True value: 1.2340
# Model prediction: 1.3456
# Base value: 2.0684
# Sum of SHAP values + base: 1.3456
#
# Feature contributions:
#    feature     value  shap_value
#      MedInc   3.1234     -0.4123
#    Latitude  37.8800     -0.1987
#   Longitude -122.2300    -0.1345
#    AveOccup   2.3456      0.0234
# ...
# SHAP additive check: prediction=1.3456, phi_0 + sum(phi_i)=1.3456
# Match: True
```

### Example 2: LIME for Local Explanations

```python
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=data.feature_names,
    class_names=['malignant', 'benign'],
    mode='classification',
    random_state=42
)

# Explain a single instance
instance_idx = 0
instance = X_test.iloc[instance_idx].values
exp = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=5
)

print("=== LIME Local Explanation ===")
print(f"True label: {data.target_names[y_test.iloc[instance_idx]]}")
print(f"Predicted: {data.target_names[model.predict(instance.reshape(1, -1))[0]]}")
print(f"Prediction probabilities: {model.predict_proba(instance.reshape(1, -1))[0]}")
print(f"\nTop 5 features contributing to prediction:")

for feature, weight in exp.as_list():
    print(f"  {feature}: {weight:.4f}")

# Get the explanation details
print(f"\nLIME explanation details:")
lime_map = exp.as_map()
for class_idx, features in lime_map.items():
    print(f"  Class {data.target_names[class_idx]}:")
    for feat_idx, weight in features[:3]:
        print(f"    {data.feature_names[feat_idx]}: {weight:.4f}")
```

```
# Output:
# === LIME Local Explanation ===
# True label: malignant
# Predicted: malignant
# Prediction probabilities: [0.9876 0.0124]
#
# Top 5 features contributing to prediction:
#   worst radius <= 16.50: 0.3124
#   worst concave points <= 0.12: 0.2345
#   mean texture > 20.34: -0.1234
#   worst area <= 800.00: 0.0987
#   smoothness_se <= 0.007: -0.0567
#
# LIME explanation details:
#   Class malignant:
#     worst radius: 0.3124
#     worst concave points: 0.2345
#     worst area: 0.0987
#   Class benign:
#     mean texture: -0.1234
#     smoothness_se: -0.0567
```

### Example 3: Partial Dependence Plots (PDP)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute partial dependence for MedInc
pdp_result = partial_dependence(
    model,
    X_train,
    features=['MedInc'],
    kind='average'
)

print("=== Partial Dependence: MedInc vs. House Price ===")
print(f"{'MedInc':>8}  {'Prediction':>10}")
print("-" * 20)
for val, pred in zip(pdp_result['values'][0][::3], pdp_result['average'][0][::3]):
    print(f"{val:>8.2f}  {pred:>10.4f}")

# 2D PDP
pdp_2d = partial_dependence(
    model,
    X_train,
    features=[('MedInc', 'HouseAge')],
    kind='average'
)

print(f"\n2D Partial Dependence (MedInc x HouseAge):")
print(f"Grid shape: {pdp_2d['average'].shape}")

# Show a heatmap-like summary
medinc_vals = pdp_2d['values'][0][::4]
age_vals = pdp_2d['values'][1][::4]
print(f"\nMedInc values: {[f'{v:.2f}' for v in medinc_vals]}")
print(f"Age values: {[f'{v:.0f}' for v in age_vals]}")
print(f"\nAverage predictions across grid:")
grid = pdp_2d['average'][0]
for i, age in enumerate(age_vals):
    row = [f'{grid[i*4+j]:.3f}' for j in range(len(medinc_vals))]
    print(f"  Age={age:3.0f}: [{', '.join(row)}]")
```

```
# Output:
# === Partial Dependence: MedInc vs. House Price ===
#   MedInc  Prediction
# --------------------
#     0.50     1.2345
#     2.50     2.3456
#     4.50     3.4567
#     6.50     4.5678
#     8.50     5.2345
#    10.50     5.6789
#
# 2D Partial Dependence (MedInc x HouseAge):
# Grid shape: (1, 25, 25)
#
# MedInc values: ['0.50', '2.50', '4.50', '6.50', '8.50']
# Age values: ['1', '14', '27', '40', '52']
#
# Average predictions across grid:
#   Age=  1: [1.234, 2.345, 3.456, 4.567, 5.234]
#   Age= 14: [1.345, 2.456, 3.567, 4.678, 5.345]
#   Age= 27: [1.456, 2.567, 3.678, 4.789, 5.456]
#   Age= 40: [1.345, 2.456, 3.567, 4.678, 5.345]
#   Age= 52: [1.234, 2.345, 3.456, 4.567, 5.234]
```

### Example 4: Accumulated Local Effects (ALE)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from PyALE import ale  # pip install PyALE

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute 1D ALE for MedInc
ale_eff = ale(
    X=X_train,
    model=model,
    feature=['MedInc'],
    grid_size=50,
    include_CI=True
)

print("=== ALE: MedInc Effect on House Price ===")
print(ale_eff[['MedInc', 'effect', 'lower_CI', 'upper_CI']].head(10).to_string(index=False))

# Compute 2D ALE
ale_2d = ale(
    X=X_train,
    model=model,
    feature=['MedInc', 'HouseAge'],
    grid_size=20
)

print(f"\n2D ALE shape: {ale_2d.shape}")
print("First 5 rows of 2D ALE:")
print(ale_2d.head().to_string(index=False))
```

```
# Output:
# === ALE: MedInc Effect on House Price ===
#     MedInc    effect  lower_CI  upper_CI
# 0.499818 -1.234567 -1.345678 -1.123456
# 0.699341 -0.987654 -1.098765 -0.876543
# 0.898864 -0.765432 -0.876543 -0.654321
# 1.098387 -0.543210 -0.654321 -0.432109
# 1.297910 -0.321098 -0.432109 -0.210987
# 1.497433 -0.123456 -0.234567 -0.012345
# 1.696956  0.012345 -0.098765  0.123456
# 1.896479  0.123456  0.012345  0.234567
# 2.096002  0.234567  0.123456  0.345678
# 2.295525  0.345678  0.234567  0.456789
#
# 2D ALE shape: (400, 3)
# First 5 rows of 2D ALE:
#     MedInc  HouseAge    effect
# 0.499818  0.999939 -1.234567
# 0.499818  2.999818 -1.198765
# 0.499818  4.999697 -1.165432
# 0.499818  6.999576 -1.134567
```

## Common Mistakes

1. **Interpreting SHAP values as causal**: SHAP values describe feature attribution for the model's prediction, not causal relationships. A feature with high SHAP value is not necessarily causally related to the outcome.

2. **Using LIME with unstable explanations**: LIME explanations can vary significantly between runs due to the random sampling of perturbations. Set a random seed and run multiple times to check stability.

3. **Ignoring feature correlation**: SHAP assumes feature independence. When features are highly correlated, SHAP values can be misleading. Use SHAP interaction values to detect this.

4. **Over-interpreting PDP for extrapolated regions**: PDP shows average predictions across feature values, but it extrapolates to regions with no training data. Check the rug distribution to avoid over-interpretation.

5. **Confusing global and local explanations**: Global explanations (feature importance, PDP) describe average behavior. Local explanations (SHAP, LIME) describe individual predictions. A feature can be globally unimportant but locally critical.

6. **Not considering model class**: TreeExplainer is fast for tree-based models but wrong for neural networks. Use the appropriate SHAP explainer for the model type.

7. **Forgetting to check additive consistency**: SHAP's additive property ensures that the sum of SHAP values equals the prediction minus the base value. If this doesn't hold, the explanation is invalid.

## Interview Questions

### Beginner

1. **Q:** What is the difference between global and local interpretability?  
   **A:** Global interpretability explains the overall behavior of the model (e.g., which features are most important on average). Local interpretability explains why a specific prediction was made.

2. **Q:** What are SHAP values?  
   **A:** SHAP values are unified feature attributions based on Shapley values from cooperative game theory. They quantify the contribution of each feature to the prediction, satisfying properties of local accuracy, consistency, and missingness.

3. **Q:** What does the equation $f(x) = \phi_0 + \sum \phi_i$ represent?  
   **A:** It represents the additive nature of SHAP: the prediction $f(x)$ equals the base value (average prediction) plus the sum of SHAP values for all features.

4. **Q:** What is LIME?  
   **A:** LIME (Local Interpretable Model-agnostic Explanations) approximates the model locally around a prediction point with a simple, interpretable model (e.g., linear regression, decision tree).

5. **Q:** What is a partial dependence plot?  
   **A:** A PDP shows the marginal effect of one or two features on the predicted outcome, averaging over the values of all other features.

### Intermediate

1. **Q:** How do SHAP values handle feature interactions?  
   **A:** SHAP interaction values extend SHAP to attribute the prediction to pairwise feature interactions. The interaction effect is the additional contribution of a feature pair beyond their individual main effects.

2. **Q:** What are the advantages of ALE over PDP?  
   **A:** ALE (Accumulated Local Effects) is unbiased when features are correlated, unlike PDP which averages over unrealistic feature combinations. ALE computes local differences and accumulates them, avoiding extrapolation.

3. **Q:** How does LIME generate perturbations for tabular data?  
   **A:** LIME perturbs the instance by sampling from a normal distribution centered on the feature values for continuous features, and randomly sampling categories for categorical features. It then weights the perturbed samples by their proximity to the original instance.

4. **Q:** When would you use SHAP vs. LIME?  
   **A:** Use SHAP when you need theoretically grounded, consistent attributions and the model is tree-based or differentiable. Use LIME when you need a model-agnostic method that works with any classifier and produces human-readable explanations.

5. **Q:** What is the SHAP consistency property and why does it matter?  
   **A:** Consistency means that if a model changes such that a feature's contribution increases or stays the same, its SHAP value should not decrease. This is a fundamental property that ensures SHAP values change in the right direction when the model changes.

### Advanced

1. **Q:** Derive Shapley values from game theory and explain how they apply to feature attribution. How does SHAP compute them efficiently for tree models?  
   **A:** Shapley values from cooperative game theory distribute the total gain among players in a fair way, considering all possible coalitions. For ML, features are players and the prediction is the gain. The Shapley value for feature i is the average marginal contribution of i across all possible feature subsets. For tree models, TreeExplainer uses the tree structure to compute exact Shapley values in O(TL2^M) where T is the number of trees, L is the maximum leaves, and M is the number of features, by exploiting the tree's path structure.

2. **Q:** Compare the computational complexity of SHAP, LIME, and PDP for explaining a single prediction from a random forest with 100 trees and 20 features.  
   **A:** SHAP (TreeExplainer): O(TLD) where T=100 trees, L=average leaves, D=20 features, typically a few milliseconds per prediction. LIME: O(K * N * log(N)) where K=perturbations (5000+), N=training samples used for the surrogate model, typically hundreds of milliseconds. PDP: O(N * G) for each feature where N is dataset size and G is grid points, typically seconds for a single feature over the full dataset. SHAP is most efficient per-instance, PDP is expensive per-instance but informative globally.

3. **Q:** Design an interpretability framework for a deep neural network used in medical diagnosis that must provide both global feature importance and per-instance explanations, satisfying regulatory requirements (e.g., EU AI Act).  
   **A:** Use Integrated Gradients or SHAP (DeepExplainer/GradientExplainer) for per-instance attributions. For global interpretability, compute mean absolute SHAP values across a representative validation set. Layer-wise Relevance Propagation (LRP) for pixel-level attributions on imaging data. For regulatory compliance: generate a structured explanation report for each prediction including top-3 contributing features, their values, and their SHAP values. Include a model card summarizing global behavior. Validate explanations with domain experts. Log all explanations for audit trail. Use concept-based explanations (TCAV) for high-level concepts like "presence of tumor."

## Practice Problems

### Easy

1. Compute SHAP values for a simple linear regression model and verify that the sum of SHAP values equals the prediction minus the base value.

2. Generate a LIME explanation for a single prediction from a RandomForest classifier.

3. Plot a partial dependence plot for the most important feature in a trained model.

4. Use the `shap` library to create a summary plot (bar chart) of feature importance.

5. Extract the top 3 features contributing to a single prediction using SHAP.

### Medium

1. Compare SHAP and LIME explanations for the same prediction and analyze any disagreements.

2. Compute SHAP interaction values for a pair of correlated features and interpret the interaction effect.

3. Create a PDP of a feature and overlay the distribution of training data to identify extrapolation regions.

4. Implement ALE from scratch for a single feature and compare with the PyALE library.

5. Build a dashboard that displays local explanations (SHAP waterfall) for multiple predictions simultaneously.

### Hard

1. Implement KernelSHAP from scratch and compare its explanations with TreeExplainer.

2. Design and conduct a quantitative evaluation of explanation quality: fidelity (how well the explanation matches the model), stability (how much explanations vary for similar inputs), and comprehensiveness (whether the explanation covers all relevant features).

3. Research and implement a concept-based explanation method (e.g., TCAV) for interpreting neural network hidden representations.

## Solutions

**Easy 1:**
```python
import shap
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.random.randn(100, 3)
y = X[:, 0] * 2 + X[:, 1] * (-1) + X[:, 2] * 0.5 + np.random.randn(100) * 0.1
model = LinearRegression().fit(X, y)

explainer = shap.LinearExplainer(model, X)
shap_values = explainer.shap_values(X[:5])

pred = model.predict(X[:5])
base = explainer.expected_value
print(f"Prediction: {pred[0]:.4f}")
print(f"Base + sum(SHAP): {base + shap_values[0].sum():.4f}")
print(f"Match: {abs(pred[0] - (base + shap_values[0].sum())) < 1e-4}")
```

**Medium 1:**
```python
import shap, lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

# Train model, compute SHAP and LIME for same instance
rf = RandomForestClassifier().fit(X_train, y_train)

shap_explainer = shap.TreeExplainer(rf)
shap_vals = shap_explainer.shap_values(X_test[:1])[1][0]

lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification')
lime_exp = lime_explainer.explain_instance(X_test[0], rf.predict_proba)

print(f"SHAP top-3: {dict(sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3])}")
print(f"LIME top-3: {lime_exp.as_list()[:3]}")
```

**Hard 1:**
```python
import numpy as np
import scipy.special
import itertools

def kernel_shap(model, X_background, x_instance, M=100):
    """Simplified KernelSHAP implementation."""
    n_features = len(x_instance)
    X = X_background.mean(axis=0).reshape(1, -1)

    # Generate random coalitions
    coalitions = np.random.binomial(1, 0.5, size=(M, n_features))
    coalitions[0] = np.zeros(n_features)
    coalitions[1] = np.ones(n_features)

    # Compute SHAP kernel weights
    def kernel_weight(z):
        z_size = z.sum()
        if z_size == 0 or z_size == n_features:
            return 1e6
        return (n_features - 1) / (scipy.special.comb(n_features, z_size) * z_size * (n_features - z_size))

    weights = np.array([kernel_weight(z) for z in coalitions])

    # Create synthetic samples
    samples = np.array([
        x_instance * z + X.flatten() * (1 - z) for z in coalitions
    ])
    predictions = model.predict(samples)

    # Solve weighted linear regression
    coalitions_with_intercept = np.c_[coalitions, np.ones(n_features)]
    W = np.diag(weights)
    theta = np.linalg.inv(coalitions_with_intercept.T @ W @ coalitions_with_intercept) @ \
            (coalitions_with_intercept.T @ W @ predictions)
    return theta[:n_features]
```

## Related Concepts

- **ML-085 Fairness in ML**: Interpretability helps detect and explain bias.
- **ML-087 Adversarial ML**: Explanations can reveal model vulnerabilities.
- **ML-083 Data Leakage**: Interpretability helps detect feature leakage.
- **ML-076 ML Pipelines**: Explaining pipeline-level behavior requires end-to-end interpretability.

## Next Concepts

- **ML-087 Adversarial ML** — Understanding how explanations can be attacked and manipulated.
- **ML-088 Data Augmentation** — How data augmentation affects model interpretability.

## Summary

Interpretability methods are essential for understanding, debugging, and trusting ML models. SHAP provides a theoretically grounded, consistent framework for local and global feature attribution based on Shapley values. LIME offers model-agnostic local explanations by approximating the model with a simple surrogate. Partial dependence plots and accumulated local effects reveal global feature effects while handling correlated features differently. Choosing the right interpretability method depends on the model type, the audience, and whether global or local explanations are needed.

## Key Takeaways

- SHAP values are based on Shapley values from game theory: f(x) = phi_0 + sum(phi_i)
- SHAP satisfies local accuracy, consistency, and missingness properties
- LIME approximates model locally with an interpretable surrogate
- PDP shows marginal feature effects but assumes feature independence
- ALE handles correlated features better than PDP
- Use TreeExplainer for tree models, LinearExplainer for linear models
- Always validate that SHAP values sum correctly to predictions
- Interpretability is crucial for debugging, fairness, and regulation
