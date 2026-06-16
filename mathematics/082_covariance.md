# Concept: Covariance

## Concept ID

MATH-082

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define covariance as a measure of joint variability between two variables
- Compute covariance using the standard formula
- Interpret positive, negative, and zero covariance
- Understand the covariance matrix and its properties
- Apply covariance in AI/ML contexts such as PCA and Mahalanobis distance

## Prerequisites

- Variance (MATH-080)
- Mean (MATH-077)
- Basic matrix algebra
- Understanding of expectation

## Definition

**Covariance** measures how two random variables change together. It quantifies the direction of the linear relationship between two variables. Positive covariance means the variables tend to increase together; negative covariance means one tends to increase when the other decreases.

**Population covariance:**
$$
\text{Cov}(X, Y) = \frac{1}{N}\sum_{i=1}^N (x_i - \mu_X)(y_i - \mu_Y)
$$

**Sample covariance:**
$$
s_{XY} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})
$$

The **covariance matrix** (variance-covariance matrix) for a $d$-dimensional random vector $\mathbf{X}$ is a $d \times d$ symmetric matrix:
$$
\Sigma = \begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \dots & \text{Cov}(X_1, X_d) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \dots & \text{Cov}(X_2, X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_d, X_1) & \text{Cov}(X_d, X_2) & \dots & \text{Var}(X_d)
\end{bmatrix}
$$

## Intuition

Imagine plotting two variables on a scatter plot. For each point, compute how far it is from the $X$ mean (horizontally) and from the $Y$ mean (vertically). Multiply these two distances. If most points fall in the upper-right and lower-left quadrants (product positive), covariance is positive. If points fall in upper-left and lower-right quadrants (product negative), covariance is negative.

Covariance tells you: "When $X$ is above its mean, is $Y$ typically above or below its mean?"

## Why This Concept Matters

Covariance is the foundation for understanding relationships between variables. It is essential for:

- **Dimensionality reduction:** PCA is based on the eigendecomposition of the covariance matrix.
- **Portfolio theory:** Covariance between asset returns determines diversification benefits.
- **Multivariate statistics:** MANOVA, discriminant analysis, and factor analysis all use the covariance matrix.
- **Signal processing:** Covariance matrices capture dependencies in multi-channel signals.
- **Machine learning:** Gaussian processes, linear discriminant analysis, and Mahalanobis distance all rely on covariance.

## Historical Background

The concept of covariance was developed by Karl Pearson and Francis Galton in the late 19th century. Galton studied the relationship between parents' and children's heights, which led to the concept of correlation and regression. Pearson formalised covariance in his 1896 paper "Mathematical Contributions to the Theory of Evolution."

The covariance matrix was developed in the early 20th century as multivariate statistics emerged. Hotelling (1933) used the covariance matrix in principal component analysis, and Mahalanobis (1936) used it to define the Mahalanobis distance.

## Real World Examples

**Finance:** The covariance between stock returns of Apple and Microsoft is positive because both respond similarly to market conditions. The covariance between airline stocks and oil prices is typically negative (higher oil hurts airlines).

**Healthcare:** The covariance between age and blood pressure is positive — older people tend to have higher blood pressure.

**Marketing:** The covariance between advertising spend and sales revenue is positive — more advertising typically correlates with more sales.

**Education:** The covariance between hours studied and exam score is positive — more study time associates with higher scores.

**Environmental science:** The covariance between temperature and ice cream sales is positive; between temperature and coat sales is negative.

## AI/ML Relevance

**Principal Component Analysis (PCA):** PCA performs eigendecomposition on the covariance matrix of the data:
$$
\Sigma = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T
$$
The eigenvectors of $\Sigma$ are the principal components, and the eigenvalues are the variances along those directions.

**Mahalanobis distance:** A distance measure that accounts for the covariance structure of the data:
$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$
Unlike Euclidean distance, Mahalanobis distance accounts for correlations between features.

**Gaussian distributions:** The multivariate normal distribution is parameterised by the mean vector $\boldsymbol{\mu}$ and covariance matrix $\Sigma$:
$$
f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

**Feature relationship analysis:** The covariance matrix reveals which features are correlated. Near-zero covariance suggests features may carry independent information; high covariance suggests redundancy.

**Gaussian Processes:** The kernel function defines the covariance between function values at different points. The covariance matrix $K(X, X)$ is the core of GP computation.

**Linear Discriminant Analysis (LDA):** Uses within-class and between-class covariance matrices to find the discriminant directions that best separate classes.

## Mathematical Explanation

Covariance is defined as the expected product of deviations from the means:

$$
\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - \mu_X \mu_Y
$$

This computational formula is often easier to use:
$$
\text{Cov}(X, Y) = \frac{1}{n}\sum x_i y_i - \bar{x}\bar{y}
$$

**Properties of covariance:**

- $\text{Cov}(X, X) = \text{Var}(X)$ — covariance of a variable with itself is its variance.
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ — covariance is symmetric.
- $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$.
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$.
- If $X$ and $Y$ are independent, $\text{Cov}(X, Y) = 0$ (but the converse is not necessarily true — zero covariance does not imply independence).

**Covariance matrix properties:**
- Symmetric: $\Sigma = \Sigma^T$.
- Positive semi-definite: All eigenvalues are non-negative.
- $\mathbf{v}^T \Sigma \mathbf{v} = \text{Var}(\mathbf{v}^T \mathbf{X}) \geq 0$ for any vector $\mathbf{v}$.

## Formula(s)

**Population covariance:**
$$
\sigma_{XY} = \text{Cov}(X, Y) = \frac{1}{N}\sum_{i=1}^N (x_i - \mu_X)(y_i - \mu_Y)
$$

**Sample covariance:**
$$
s_{XY} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})
$$

**Computational formula:**
$$
\text{Cov}(X, Y) = \frac{\sum x_i y_i}{n} - \left(\frac{\sum x_i}{n}\right)\left(\frac{\sum y_i}{n}\right)
$$

**Covariance matrix:**
$$
\Sigma = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T
$$

## Properties

- **Range:** Covariance can range from $-\infty$ to $\infty$, making it difficult to interpret magnitude (that is why we use correlation).
- **Symmetry:** $\text{Cov}(X, Y) = \text{Cov}(Y, X)$.
- **Bilinearity:** $\text{Cov}(aX + bY, Z) = a\text{Cov}(X, Z) + b\text{Cov}(Y, Z)$.
- **Variance-covariance relationship:** $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$.
- **Independence:** $X \perp Y \implies \text{Cov}(X, Y) = 0$ (zero covariance), but zero covariance does not imply independence.
- **Units:** Covariance is in the product of the units of $X$ and $Y$. If $X$ is in metres and $Y$ in kg, covariance is in metre-kilograms.

## Step-by-Step Worked Examples

### Example 1: Computing Covariance

**Problem:** Compute the covariance between $X$ and $Y$ for the data:
$$\begin{array}{c|c}
X & Y \\
\hline
2 & 4 \\
4 & 6 \\
6 & 8 \\
8 & 10 \\
10 & 12
\end{array}$$

**Solution:**

Step 1: Compute means.
$$
\bar{x} = \frac{2 + 4 + 6 + 8 + 10}{5} = 6
$$
$$
\bar{y} = \frac{4 + 6 + 8 + 10 + 12}{5} = 8
$$

Step 2: Compute products of deviations.
- $(2-6)(4-8) = (-4)(-4) = 16$
- $(4-6)(6-8) = (-2)(-2) = 4$
- $(6-6)(8-8) = (0)(0) = 0$
- $(8-6)(10-8) = (2)(2) = 4$
- $(10-6)(12-8) = (4)(4) = 16$

Sum = $16 + 4 + 0 + 4 + 16 = 40$

Step 3: Divide by $n-1$ (sample covariance).
$$
s_{XY} = \frac{40}{5-1} = \frac{40}{4} = 10
$$

The positive covariance (10) confirms that $X$ and $Y$ increase together.

### Example 2: Negative Covariance

**Problem:** Compute covariance for:
$$\begin{array}{c|c}
X & Y \\
\hline
1 & 10 \\
2 & 8 \\
3 & 6 \\
4 & 4 \\
5 & 2
\end{array}$$

**Solution:**

Step 1: Means.
$$
\bar{x} = 3, \quad \bar{y} = 6
$$

Step 2: Products.
- $(1-3)(10-6) = (-2)(4) = -8$
- $(2-3)(8-6) = (-1)(2) = -2$
- $(3-3)(6-6) = (0)(0) = 0$
- $(4-3)(4-6) = (1)(-2) = -2$
- $(5-3)(2-6) = (2)(-4) = -8$

Sum = $-8 - 2 + 0 - 2 - 8 = -20$

Step 3: Sample covariance.
$$
s_{XY} = \frac{-20}{4} = -5
$$

The negative covariance (-5) confirms $X$ and $Y$ move in opposite directions.

### Example 3: Covariance Matrix

**Problem:** Given the data matrix with 3 features:
$$\begin{array}{c|c|c}
X_1 & X_2 & X_3 \\
\hline
2 & 5 & 3 \\
4 & 7 & 5 \\
6 & 9 & 7 \\
8 & 11 & 9
\end{array}$$
Compute the sample covariance matrix.

**Solution:**

Step 1: Means.
$\bar{x}_1 = 5$, $\bar{x}_2 = 8$, $\bar{x}_3 = 6$.

Step 2: Deviations matrix.
$$
\mathbf{D} = \begin{bmatrix}
-3 & -3 & -3 \\
-1 & -1 & -1 \\
1 & 1 & 1 \\
3 & 3 & 3
\end{bmatrix}
$$

Step 3: Compute $\mathbf{D}^T\mathbf{D}$.
$$
\mathbf{D}^T\mathbf{D} = \begin{bmatrix}
20 & 20 & 20 \\
20 & 20 & 20 \\
20 & 20 & 20
\end{bmatrix}
$$

Step 4: Divide by $n-1 = 3$.
$$
\Sigma = \frac{1}{3}\begin{bmatrix}
20 & 20 & 20 \\
20 & 20 & 20 \\
20 & 20 & 20
\end{bmatrix} = \begin{bmatrix}
\frac{20}{3} & \frac{20}{3} & \frac{20}{3} \\
\frac{20}{3} & \frac{20}{3} & \frac{20}{3} \\
\frac{20}{3} & \frac{20}{3} & \frac{20}{3}
\end{bmatrix}
$$

All features are perfectly correlated in this contrived example, so all covariances equal the variance.

## Visual Interpretation

On a scatter plot, covariance corresponds to the orientation of the point cloud:

- **Positive covariance:** The cloud slopes upward (bottom-left to top-right).
- **Negative covariance:** The cloud slopes downward (top-left to bottom-right).
- **Zero covariance:** The cloud is roughly circular or aligned with the axes.

The magnitude of covariance depends on the spread of the data as well as the strength of the relationship. Two tightly correlated variables with large individual variances will have larger covariance than the same relationship with smaller variances.

The covariance matrix can be visualised as a heatmap, where diagonal entries (variances) are the largest and off-diagonal entries show pairwise relationships.

## Common Mistakes

1. **Interpreting covariance magnitude:** A large covariance magnitude does not necessarily mean a strong relationship — it depends on the scales of the variables. Always use correlation (standardised covariance) for strength.

2. **Confusing zero covariance with independence:** Zero covariance means no linear relationship, but non-linear relationships can exist with zero covariance (e.g., $Y = X^2$ over a symmetric range).

3. **Forgetting the sample correction:** Using $n$ instead of $n-1$ for sample covariance introduces bias.

4. **Ignoring units:** Covariance has units that are the product of the units of the two variables. Comparing covariances across different variable pairs is meaningless without standardisation.

5. **Assuming the covariance matrix is diagonal:** Many real-world datasets have correlated features. Ignoring off-diagonal elements loses important information.

6. **Computing covariance for categorical variables:** Covariance is meaningful only for continuous (or at least interval) variables.

7. **Misinterpreting sign reversal:** Changing the sign of one variable flips the sign of covariance.

## Interview Questions

### Beginner - 5

1. **Q:** What does covariance measure?
   **A:** Covariance measures how two variables vary together — the direction of their linear relationship.

2. **Q:** What does positive covariance mean?
   **A:** When one variable is above its mean, the other tends to be above its mean as well; they increase together.

3. **Q:** What does negative covariance mean?
   **A:** When one variable is above its mean, the other tends to be below its mean; they move in opposite directions.

4. **Q:** What is the covariance of a variable with itself?
   **A:** $\text{Cov}(X, X) = \text{Var}(X)$.

5. **Q:** What is the range of covariance?
   **A:** Covariance has no fixed range — it can be any real number from $-\infty$ to $\infty$.

### Intermediate - 5

1. **Q:** What is the covariance matrix and what are its properties?
   **A:** A $d \times d$ symmetric matrix with variances on the diagonal and covariances off the diagonal. It is positive semi-definite.

2. **Q:** How is covariance used in PCA?
   **A:** PCA eigendecomposes the covariance matrix (or correlation matrix). Eigenvectors give principal directions; eigenvalues give variance explained.

3. **Q:** What is the relationship between covariance and correlation?
   **A:** Correlation is standardised covariance: $\rho_{XY} = \text{Cov}(X, Y) / (\sigma_X \sigma_Y)$.

4. **Q:** Why is the covariance matrix always positive semi-definite?
   **A:** For any vector $\mathbf{v}$, $\mathbf{v}^T \Sigma \mathbf{v} = \text{Var}(\mathbf{v}^T \mathbf{X}) \geq 0$.

5. **Q:** How does covariance relate to the variance of a sum?
   **A:** $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$.

### Advanced - 3

1. **Q:** Derive the expression for the covariance of two linear combinations.
   **A:** $\text{Cov}(a^T X, b^T Y) = a^T \Sigma_{XY} b$, where $\Sigma_{XY}$ is the cross-covariance matrix.

2. **Q:** Explain the role of the inverse covariance matrix (precision matrix) in Gaussian graphical models.
   **A:** The precision matrix $\Sigma^{-1}$ encodes conditional independence relationships. Zero entries in $\Sigma^{-1}$ indicate conditional independence given all other variables.

3. **Q:** How is the covariance matrix estimated in high-dimensional settings ($p \gg n$)?
   **A:** The sample covariance is singular when $p > n$. Regularisation methods (ridge-type shrinkage, graphical lasso, factor models) are used to obtain well-conditioned estimates.

## Practice Problems

### Easy - 5

1. Compute $\text{Cov}(X, Y)$ for $\{(1,2), (2,3), (3,4)\}$.

2. If $\text{Cov}(X, Y) = 5$ and $\text{Cov}(Y, Z) = -3$, find $\text{Cov}(2X, 3Y)$.

3. Compute $\text{Cov}(X, X)$ for $X = \{1, 3, 5\}$.

4. If $\text{Var}(X) = 4$, $\text{Var}(Y) = 9$, $\text{Cov}(X, Y) = 3$, find $\text{Var}(X+Y)$.

5. Given $\sum (x_i - \bar{x})(y_i - \bar{y}) = 30$ and $n = 10$, compute the sample covariance.

### Medium - 5

1. Show that $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$.

2. Given data: $(1,1), (2,2), (3,4), (4,3), (5,5)$. Compute sample covariance.

3. If $\Sigma = \begin{bmatrix} 4 & 1 \\ 1 & 9 \end{bmatrix}$, find $\text{Var}(X_1 + X_2)$.

4. Prove $\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$.

5. For random variables $X, Y, Z$, express $\text{Cov}(X+Y, Z)$ in terms of simple covariances.

### Hard - 3

1. Prove that $\text{Var}(\bar{x}) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \text{Cov}(x_i, x_j)$.

2. Derive the expression for the partial covariance (covariance of residuals after regression).

3. Given the eigendecomposition $\Sigma = V \Lambda V^T$, show that PCA projects the data onto the eigenvectors of the covariance matrix.

## Solutions

**Easy:**

1. $\bar{x}=2$, $\bar{y}=3$. Products: $(1-2)(2-3)=1$, $(2-2)(3-3)=0$, $(3-2)(4-3)=1$. Sum $=2$. Population $\text{Cov}=2/3$, Sample $=2/2=1$.

2. $\text{Cov}(2X, 3Y) = 2 \cdot 3 \cdot \text{Cov}(X,Y) = 6 \cdot 5 = 30$.

3. $\text{Cov}(X,X) = \text{Var}(X) = ((1-3)^2+(3-3)^2+(5-3)^2)/3 = (4+0+4)/3 = 8/3$.

4. $\text{Var}(X+Y) = 4 + 9 + 2(3) = 19$.

5. $s_{XY} = 30/(10-1) = 30/9 = 10/3 \approx 3.33$.

**Medium:**

1. $\text{Cov}(aX+b, cY+d) = E[(aX+b - a\mu_X - b)(cY+d - c\mu_Y - d)] = E[a(X-\mu_X) \cdot c(Y-\mu_Y)] = ac \cdot E[(X-\mu_X)(Y-\mu_Y)] = ac \cdot \text{Cov}(X,Y)$.

2. $\bar{x}=3$, $\bar{y}=3$. Products: $(-2)(-2)=4$, $(-1)(-1)=1$, $(0)(1)=0$, $(1)(0)=0$, $(2)(2)=4$. Sum $=9$. Sample Cov $=9/4 = 2.25$.

3. $\text{Var}(X_1 + X_2) = \sigma_{11} + \sigma_{22} + 2\sigma_{12} = 4 + 9 + 2(1) = 15$.

4. $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY - \mu_X Y - \mu_Y X + \mu_X\mu_Y] = E[XY] - \mu_X E[Y] - \mu_Y E[X] + \mu_X\mu_Y = E[XY] - \mu_X\mu_Y - \mu_X\mu_Y + \mu_X\mu_Y = E[XY] - \mu_X\mu_Y$.

5. $\text{Cov}(X+Y, Z) = \text{Cov}(X,Z) + \text{Cov}(Y,Z)$ by bilinearity.

**Hard:**

1. $\text{Var}(\bar{x}) = \text{Var}(\frac{1}{n}\sum x_i) = \frac{1}{n^2}\text{Var}(\sum x_i) = \frac{1}{n^2}[\sum \text{Var}(x_i) + \sum_{i\neq j} \text{Cov}(x_i, x_j)] = \frac{1}{n^2}\sum_i\sum_j \text{Cov}(x_i, x_j)$.

2. If we regress $X$ on $Z$ and $Y$ on $Z$, the residuals are $X^* = X - \hat{X}(Z)$ and $Y^* = Y - \hat{Y}(Z)$. The partial covariance is $\text{Cov}(X^*, Y^*)$, which measures the relationship between $X$ and $Y$ after removing the effect of $Z$.

3. PCA aims to find $\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \text{Var}(\mathbf{v}^T\mathbf{X}) = \arg\max_{\|\mathbf{v}\|=1} \mathbf{v}^T\Sigma\mathbf{v}$. By the Rayleigh quotient theorem, this is maximised by the eigenvector corresponding to the largest eigenvalue of $\Sigma$. Subsequent PCs are the remaining eigenvectors, with variances equal to the eigenvalues.

## Related Concepts

- Variance (MATH-080) — covariance of a variable with itself
- Correlation (MATH-083) — standardised covariance
- Covariance Matrix — collection of all pairwise covariances
- Principal Component Analysis — uses eigendecomposition of covariance matrix
- Mahalanobis Distance — distance using inverse covariance matrix
- Multivariate Normal Distribution — parameterised by mean vector and covariance matrix

## Next Concepts

- Correlation (MATH-083) — interpretable version of covariance
- Skewness (MATH-084) — third moment, extending the moment framework
- Confidence Intervals (MATH-086) — using variability for inference

## Summary

Covariance measures the direction of the linear relationship between two variables. Positive covariance indicates variables move together; negative indicates they move in opposite directions. The covariance matrix collects all pairwise covariances and variances, and is positive semi-definite and symmetric. Covariance is fundamental to PCA (which eigendecomposes the covariance matrix), Mahalanobis distance (which uses its inverse), and multivariate Gaussian distributions. However, covariance magnitude is hard to interpret because it depends on units — this motivates correlation.

## Key Takeaways

- $\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - \mu_X\mu_Y$.
- Positive covariance: $X$ and $Y$ increase together. Negative: they move oppositely.
- $\text{Cov}(X, X) = \text{Var}(X)$.
- The covariance matrix is symmetric and positive semi-definite.
- Zero covariance does not imply independence (only linear independence).
- PCA eigendecomposes the covariance matrix.
- Mahalanobis distance uses $\Sigma^{-1}$ for scale-invariant distance.
- Covariance magnitude is scale-dependent — use correlation for standardised strength.
