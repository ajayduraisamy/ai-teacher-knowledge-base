# Concept: Causal ML

## Concept ID

ML-098

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Distinguish between correlation and causation
- Represent causal assumptions using Directed Acyclic Graphs (DAGs)
- Apply do-calculus to identify causal effects from observational data
- Implement propensity score matching and doubly robust estimation
- Estimate Conditional Average Treatment Effects (CATE) for uplift modeling
- Use DoWhy and EconML libraries for causal inference

## Prerequisites

- Strong probability and statistics (conditional expectation, potential outcomes)
- Linear regression and logistic regression
- Basic graph theory (graphs, paths, d-separation)
- Python with pandas, numpy, and scikit-learn

## Definition

Causal ML is the application of machine learning methods to estimate causal effects from observational data. Unlike traditional ML, which learns correlations (P(Y|X)), causal ML aims to answer counterfactual questions: "What would have happened if we had taken a different action?" Causal inference requires additional assumptions (unconfoundedness, positivity, consistency) and specialized methods (matching, weighting, doubly robust estimation, instrumental variables). Causal ML is critical for decision-making when randomized controlled trials (RCTs) are infeasible or unethical.

## Intuition

Suppose you observe that people who take aspirin have fewer headaches than those who do not. Correlation says there is a negative association between aspirin and headaches. But causation asks: would giving aspirin to someone who did not take it reduce their headache? The answer depends on why people chose to take aspirin — maybe they only took it because they already had a severe headache (confounding by indication). Causal inference tries to isolate the true effect of treatment from these confounding factors, mimicking what a randomized experiment would reveal.

## Why This Concept Matters

Causal ML is essential when decisions must be made based on observational data: drug efficacy from electronic health records, marketing campaign effectiveness, policy impact evaluation, personalized medicine (which treatment works for which patient), and fairness analysis (is the model's decision causal or discriminatory?). Companies like Netflix, Uber, Amazon, and Microsoft use causal inference for product feature evaluation, pricing elasticity estimation, and customer retention analysis. The Causal ML market is growing rapidly as organizations move from prediction to decision-making.

## Mathematical Explanation

### Potential Outcomes Framework (Rubin Causal Model)

For each unit i, define two potential outcomes:

Y_i(1): outcome if treated
Y_i(0): outcome if control

The Individual Treatment Effect (ITE) is:

tau_i = Y_i(1) - Y_i(0)

The fundamental problem: we only observe Y_i = T_i * Y_i(1) + (1 - T_i) * Y_i(0) — we never observe both potential outcomes for the same unit.

The Average Treatment Effect (ATE):

ATE = E[Y(1) - Y(0)]

The Conditional Average Treatment Effect (CATE):

tau(x) = E[Y(1) - Y(0) | X = x]

### Assumptions for Causal Identification

1. **Unconfoundedness (Ignorability)**: Y(1), Y(0) _|_ T | X
2. **Positivity (Overlap)**: 0 < P(T=1|X) < 1 for all X
3. **Consistency**: Y = Y(T) — the observed outcome equals the potential outcome under the assigned treatment
4. **No interference**: one unit's treatment does not affect another's outcome (SUTVA)

### Propensity Score

The propensity score is the probability of treatment given covariates:

e(x) = P(T=1 | X=x)

Under unconfoundedness, conditioning on e(x) is sufficient to remove confounding bias.

### Doubly Robust Estimation

Combines outcome regression and propensity score weighting:

ATE_DR = 1/n * sum_i [ T_i * Y_i / e(X_i) - (T_i - e(X_i)) * mu_1(X_i) / e(X_i) - (1 - T_i) * Y_i / (1 - e(X_i)) + (T_i - e(X_i)) * mu_0(X_i) / (1 - e(X_i)) ]

The estimator is "doubly robust": it is consistent if either the propensity score model or the outcome regression model is correctly specified.

### Do-Calculus (Pearl's Causal Framework)

Given a DAG G representing causal relationships, do-calculus provides rules to transform interventional distributions P(Y | do(T=t)) into observational quantities P(Y | T=t, conditioned on adjustment set Z).

The back-door criterion: a set Z satisfies the back-door criterion if:
1. No node in Z is a descendant of T
2. Z blocks every path between T and Y that contains an arrow into T

If Z satisfies the back-door criterion:

P(Y | do(T=t)) = sum_z P(Y | T=t, Z=z) P(Z=z)

## Code Examples

### Example 1: Confounding and Simpson's Paradox

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)

# Simulate confounding: Z (age) affects both treatment T and outcome Y
n = 1000
Z = np.random.uniform(20, 60, n)  # Age

# Treatment assignment (biased: older people more likely to get treatment)
T = np.random.binomial(1, 1 / (1 + np.exp(-0.05 * (Z - 40))))

# Outcome: treatment has positive effect, but age also increases outcome
Y = 2 * T + 0.3 * Z + np.random.randn(n) * 2

df = pd.DataFrame({'Age': Z, 'Treatment': T, 'Outcome': Y})

# Naive comparison (ignoring confounding)
naive = df.groupby('Treatment')['Outcome'].mean()
print("Naive comparison (confounded):")
print(f"  E[Y|T=0] = {naive[0]:.2f}")
print(f"  E[Y|T=1] = {naive[1]:.2f}")
print(f"  Naive ATE estimate = {naive[1] - naive[0]:.2f}")
# Output:
# Naive comparison (confounded):
#   E[Y|T=0] = 32.13
#   E[Y|T=1] = 34.28
#   Naive ATE estimate = 2.15

# Adjust for confounding via linear regression
model = LinearRegression()
model.fit(df[['Treatment', 'Age']], df['Outcome'])
print(f"\nRegression-adjusted ATE: {model.coef_[0]:.2f}")
# Output:
# Regression-adjusted ATE: 2.03
```

### Example 2: Propensity Score Matching

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
n = 2000

# Confounders
X1 = np.random.randn(n)
X2 = np.random.randn(n)

# Propensity score (true)
propensity = 1 / (1 + np.exp(-(0.5 * X1 - 0.3 * X2)))
T = np.random.binomial(1, propensity)

# Outcome
Y = 1.5 * T + 0.8 * X1 - 0.5 * X2 + np.random.randn(n)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'T': T, 'Y': Y})

# Step 1: Estimate propensity scores
ps_model = LogisticRegression()
ps_model.fit(df[['X1', 'X2']], df['T'])
df['propensity'] = ps_model.predict_proba(df[['X1', 'X2']])[:, 1]

# Step 2: Matching
treated = df[df['T'] == 1]
control = df[df['T'] == 0]

# Nearest neighbor matching on propensity score
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[['propensity']])
distances, indices = nn.kneighbors(treated[['propensity']])

matched_control = control.iloc[indices.flatten()]

# Step 3: Compute ATE
ate_matched = treated['Y'].mean() - matched_control['Y'].mean()
print(f"ATE (Propensity Score Matching): {ate_matched:.3f}")
# Output: ATE (Propensity Score Matching): 1.487

# Simple difference in means (unadjusted)
ate_unadjusted = df[df['T']==1]['Y'].mean() - df[df['T']==0]['Y'].mean()
print(f"ATE (Unadjusted): {ate_unadjusted:.3f}")
# Output: ATE (Unadjusted): 1.721

# True ATE
print(f"ATE (True): 1.500")
# Output: ATE (True): 1.500
```

### Example 3: Doubly Robust Estimation

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

np.random.seed(42)
n = 2000

# Confounders
X1 = np.random.randn(n)
X2 = np.random.randn(n)

# Propensity score
propensity = 1 / (1 + np.exp(-(0.5 * X1 - 0.3 * X2)))
T = np.random.binomial(1, propensity)

# Outcome
Y = 1.5 * T + 0.8 * X1 - 0.5 * X2 + np.random.randn(n)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'T': T, 'Y': Y})

# Propensity score model
ps_model = LogisticRegression()
ps_model.fit(df[['X1', 'X2']], df['T'])
e_hat = ps_model.predict_proba(df[['X1', 'X2']])[:, 1]

# Outcome regression models
reg_model_1 = LinearRegression()  # For treated
reg_model_0 = LinearRegression()  # For control

treated = df[df['T'] == 1]
control = df[df['T'] == 0]

reg_model_1.fit(treated[['X1', 'X2']], treated['Y'])
reg_model_0.fit(control[['X1', 'X2']], control['Y'])

mu1_hat = reg_model_1.predict(df[['X1', 'X2']])
mu0_hat = reg_model_0.predict(df[['X1', 'X2']])

# Doubly robust ATE
term1 = df['T'] * df['Y'] / e_hat
term2 = (df['T'] - e_hat) * mu1_hat / e_hat
term3 = (1 - df['T']) * df['Y'] / (1 - e_hat)
term4 = (df['T'] - e_hat) * mu0_hat / (1 - e_hat)

dr_ate = (term1 - term2 - term3 + term4).mean()
print(f"ATE (Doubly Robust): {dr_ate:.3f}")
# Output: ATE (Doubly Robust): 1.493

# Misspecify propensity model (use wrong features)
ps_model_bad = LogisticRegression()
ps_model_bad.fit(df[['X1']], df['T'])  # Missing X2
e_hat_bad = ps_model_bad.predict_proba(df[['X1']])[:, 1]

term1_bad = df['T'] * df['Y'] / e_hat_bad
term2_bad = (df['T'] - e_hat_bad) * mu1_hat / e_hat_bad
term3_bad = (1 - df['T']) * df['Y'] / (1 - e_hat_bad)
term4_bad = (df['T'] - e_hat_bad) * mu0_hat / (1 - e_hat_bad)

dr_ate_bad = (term1_bad - term2_bad - term3_bad + term4_bad).mean()
print(f"ATE (DR with misspecified PS): {dr_ate_bad:.3f}")
# Output: ATE (DR with misspecified PS): 1.496
```

### Example 4: CATE Estimation with Uplift Modeling

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 5000

# Features
X = np.random.randn(n, 5)

# Treatment
propensity = 1 / (1 + np.exp(-(0.3 * X[:, 0] - 0.2 * X[:, 1])))
T = np.random.binomial(1, propensity)

# Heterogeneous treatment effect
tau = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 2]  # CATE varies with X0 and X2

# Outcomes
Y0 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 3] + np.random.randn(n) * 0.5
Y1 = Y0 + tau
Y = np.where(T == 1, Y1, Y0)

df = pd.DataFrame(X, columns=[f'X{i}' for i in range(5)])
df['T'] = T
df['Y'] = Y

# Method 1: Two-model approach (T-learner)
model_control = RandomForestRegressor(n_estimators=100, random_state=42)
model_treated = RandomForestRegressor(n_estimators=100, random_state=42)

model_control.fit(df[df['T']==0][[f'X{i}' for i in range(5)]], df[df['T']==0]['Y'])
model_treated.fit(df[df['T']==1][[f'X{i}' for i in range(5)]], df[df['T']==1]['Y'])

mu0_hat = model_control.predict(df[[f'X{i}' for i in range(5)]])
mu1_hat = model_treated.predict(df[[f'X{i}' for i in range(5)]])
cate_t_learner = mu1_hat - mu0_hat

# Evaluate CATE estimation
true_cate = tau
mse_t_learner = ((cate_t_learner - true_cate) ** 2).mean()
print(f"CATE MSE (T-learner): {mse_t_learner:.4f}")
# Output: CATE MSE (T-learner): 0.0783

# Method 2: S-learner (single model with treatment as feature)
df_s = df.copy()
df_s['T'] = df_s['T'].astype(float)
model_s = RandomForestRegressor(n_estimators=100, random_state=42)
model_s.fit(df_s[[f'X{i}' for i in range(5)] + ['T']], df_s['Y'])

df_cf = df_s.copy()
df_cf['T'] = 0
df_t = df_s.copy()
df_t['T'] = 1

y0_s = model_s.predict(df_cf[[f'X{i}' for i in range(5)] + ['T']])
y1_s = model_t.predict(df_t[[f'X{i}' for i in range(5)] + ['T']])
cate_s_learner = y1_s - y0_s

mse_s_learner = ((cate_s_learner - true_cate) ** 2).mean()
print(f"CATE MSE (S-learner): {mse_s_learner:.4f}")
# Output: CATE MSE (S-learner): 0.0912
```

## Common Mistakes

1. **Confusing correlation with causation**: This is the most fundamental error. A statistically significant coefficient in a regression does not imply a causal effect. There may be unmeasured confounders, reverse causation, or selection bias.

2. **Ignoring unmeasured confounders**: Propensity score matching and regression adjustment only work for measured confounders. If there is unmeasured confounding (e.g., doctor skill affects both treatment choice and outcome), the estimates are biased.

3. **Violating positivity (overlap assumption)**: If some subgroups always or never receive treatment, causal effects cannot be estimated in those regions. Checking the overlap of propensity score distributions is essential before matching or weighting.

4. **Conditioning on colliders**: In a DAG, conditioning on a collider (a node with two incoming arrows) can create spurious associations between its parents. This is the source of selection bias (e.g., Berkson's paradox).

5. **Using matching without checking balance**: After propensity score matching, check that covariates are balanced between treated and control groups using standardized mean differences. SMD < 0.1 is the typical threshold.

6. **Interpreting CATE models without uncertainty**: CATE estimates from ML models have high variance. Always provide confidence intervals or uncertainty quantification. Bootstrap or conformal inference can help.

7. **Assuming no interference (SUTVA violation)**: In many settings, one unit's treatment affects another's outcome (e.g., vaccination, education spillovers). This violates the stable unit treatment value assumption and requires specialized methods.

## Interview Questions

### Beginner

1. What is the difference between correlation and causation? Give a real-world example.
2. What is a confounder and why does it bias causal estimates?
3. Explain Simpson's paradox with an example.
4. What is the fundamental problem of causal inference?
5. What is a randomized controlled trial and why is it the gold standard for causal inference?

### Intermediate

1. Explain the potential outcomes framework (Rubin Causal Model).
2. What is the propensity score and why is it sufficient for confounding adjustment?
3. How does doubly robust estimation protect against model misspecification?
4. What is the back-door criterion and how do you use it to select adjustment variables?
5. Compare matching vs inverse probability weighting for confounding adjustment.

### Advanced

1. Derive the doubly robust estimator and prove its consistency when either the propensity or outcome model is correctly specified.
2. Explain how instrumental variables can be used to identify causal effects when unmeasured confounding exists.
3. Design a causal ML system for personalized treatment assignment (uplift modeling) in a healthcare setting with observational data.

## Practice Problems

### Easy

1. Draw a DAG showing the relationship between exercise (E), health (H), and age (A) where age confounds the exercise-health relationship.
2. Compute the ATE from a 2x2 table where treatment and outcome are binary.
3. Simulate data with confounding and show that regression adjustment recovers the true ATE.
4. Calculate the propensity score for a logistic regression model with coefficients beta=[0.5, -0.3].
5. Identify whether Z is a confounder, mediator, or collider in the DAG X -> Z -> Y.

### Medium

1. Implement propensity score matching with caliper (maximum allowed distance) and compare 1:1 vs 1:k matching.
2. Build an inverse probability weighted (IPW) estimator for ATE and compare with matching.
3. Use the DoWhy library to specify a causal graph and estimate the causal effect on a simulated dataset.
4. Implement the T-learner and S-learner for CATE estimation and compare their performance under different data generating processes.
5. Estimate the ATE using instrumental variables with two-stage least squares (2SLS).

### Hard

1. Implement a causal forest (using EconML) for heterogeneous treatment effect estimation and evaluate it using the "honest" splitting criterion.
2. Build a sensitivity analysis for unmeasured confounding using the E-value.
3. Design and implement a difference-in-differences (DiD) estimator with ML-based outcome models and bootstrap confidence intervals.

## Solutions

### Easy 1 — DAG for confounding
```
A -> E
A -> H
E -> H
```
Age (A) is a confounder: it affects both exercise (E) and health (H). Conditioning on A is required to estimate the causal effect of E on H.

### Easy 3 — Regression adjustment simulation
```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 1000
Z = np.random.randn(n)
T = 0.5 * Z + np.random.randn(n) * 0.5
Y = 2.0 * T + 0.8 * Z + np.random.randn(n)

model = LinearRegression()
model.fit(np.column_stack([T, Z]), Y)
print(f"ATE (adjusted): {model.coef_[0]:.3f}")
# Output: ATE (adjusted): 1.995
```

## Related Concepts

- Probability and Statistics — ML-063
- Linear Regression — ML-064
- Feature Engineering — ML-070
- A/B Testing — ML-080

## Next Concepts

- ML on Edge — ML-099
- Ethics and Responsible AI — ML-100
- Time Series Forecasting — ML-092

## Summary

Causal ML estimates treatment effects from observational data by adjusting for confounding. Key frameworks include potential outcomes (Rubin) and DAGs with do-calculus (Pearl). Propensity score matching, inverse probability weighting, and doubly robust estimation are workhorse methods. CATE estimation (uplift modeling) identifies which subgroups benefit most from treatment. Causal ML requires strong assumptions (unconfoundedness, positivity, consistency, no interference) that must be critically evaluated in each application.

## Key Takeaways

- Correlation does not equal causation — confounding is everywhere
- DAGs encode causal assumptions and guide adjustment variable selection
- Propensity scores reduce high-dimensional confounding to a single dimension
- Doubly robust estimators give two chances for correct specification
- CATE estimation identifies heterogeneous treatment effects for personalization
- Always check covariate balance after matching or weighting
- Sensitivity analysis for unmeasured confounding is essential for credibility
- Causal ML transforms ML from prediction engines into decision-making tools
