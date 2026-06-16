# Concept: Correlation

## Concept ID

MATH-083

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define Pearson correlation coefficient and interpret its value
- Distinguish between Pearson and Spearman correlation
- Understand that correlation does not imply causation
- Apply correlation for feature selection and multicollinearity detection in ML pipelines
- Visualise and interpret the correlation matrix

## Prerequisites

- Covariance (MATH-082)
- Standard Deviation (MATH-081)
- Mean (MATH-077)
- Basic understanding of scatter plots

## Definition

**Correlation** (Pearson correlation coefficient) measures the strength and direction of a linear relationship between two variables. It is the standardised form of covariance, scaled to the range $[-1, 1]$.

**Pearson correlation coefficient:**
$$
r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
$$

**Spearman rank correlation** measures the monotonic relationship between variables using ranks:
$$
\rho = 1 - \frac{6\sum d_i^2}{n(n^2 - 1)}
$$
where $d_i$ is the difference between the ranks of $x_i$ and $y_i$.

## Intuition

Correlation answers the question: "When I know the value of one variable, how well can I predict the value of another variable using a straight line?"

- $r = 1$: Perfect positive linear relationship. All points lie on a line with positive slope.
- $r = -1$: Perfect negative linear relationship. All points lie on a line with negative slope.
- $r = 0$: No linear relationship. The scatter plot shows no linear trend.

The magnitude $|r|$ tells you the strength. Values near $\pm 0.9$ indicate a very strong relationship; values near $\pm 0.1$ indicate a very weak one.

## Why This Concept Matters

Correlation is one of the most widely used statistics in all of science. It is the foundation of:

- **Feature selection:** Removing highly correlated features to reduce redundancy.
- **Multicollinearity detection:** Identifying when regression predictors are highly correlated.
- **Exploratory data analysis:** Understanding relationships between variables before modelling.
- **Portfolio diversification:** Investing in assets with low or negative correlation.
- **Psychometrics:** Measuring test-retest reliability and inter-rater agreement.

The correlation coefficient is also crucial for understanding that association does not imply causation.

## Historical Background

The Pearson correlation coefficient was developed by Karl Pearson from an idea by Francis Galton. Galton studied the relationship between parents' and children's heights and introduced the concept of "co-relation" in 1888. Pearson refined this into the product-moment correlation coefficient in 1896.

Spearman's rank correlation was developed by Charles Spearman in 1904 as a non-parametric alternative. Spearman was a psychologist who needed a correlation measure for ordinal data and relationships that were monotonic but not necessarily linear.

The phrase "correlation does not imply causation" was popularised by statisticians in the early 20th century, though the concept was understood earlier.

## Real World Examples

**Healthcare:** The correlation between smoking and lung cancer is strong and positive ($r > 0.7$ in many studies). This relationship, established through decades of epidemiological research, is causal.

**Finance:** The correlation between the S&P 500 and international stock indices is typically $0.5-0.8$, suggesting moderate to strong positive relationship during normal markets but approaching 1.0 during crises.

**Education:** The correlation between hours studied and exam score is typically moderate ($r \approx 0.4-0.6$).

**Sports:** The correlation between a basketball player's height and their number of rebounds is positive but modest ($r \approx 0.3-0.5$).

**Weather:** The correlation between altitude and temperature is strongly negative -- higher elevations are colder ($r \approx -0.8$ to $-0.9$).

## AI/ML Relevance

**Feature selection via correlation:** Features that are highly correlated with the target variable are valuable predictors. Features that are highly correlated with each other can be removed to reduce redundancy (multicollinearity).

**Multicollinearity detection:** In linear regression, high correlation between predictors inflates standard errors of coefficients. The Variance Inflation Factor (VIF) is computed from correlations:
$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$
where $R_j^2$ is the $R^2$ from regressing predictor $j$ on all others.

**Correlation matrix visualisation:** Heatmaps of the correlation matrix are standard in exploratory data analysis (EDA). Libraries like seaborn make this easy. The correlation matrix provides a quick overview of pairwise relationships, highlighting potential multicollinearity issues.

**Dimensionality reduction:** Before applying PCA, the correlation matrix (instead of covariance matrix) is used when features are on different scales. This ensures all features contribute equally to the principal components.

**Algorithm design:** Decision trees naturally handle correlated features (unlike linear models). Understanding correlation helps choose the right algorithm for the data.

**Cluster analysis:** Correlation can be used as a similarity measure between variables (R-mode clustering) or between observations (Q-mode clustering).

**Time series analysis:** Autocorrelation (correlation with lagged values) is used in ARIMA models to capture temporal dependencies.

## Mathematical Explanation

Pearson correlation is the covariance divided by the product of standard deviations:

$$
r = \frac{\sigma_{XY}}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}
$$

Geometrically, $r$ is the cosine of the angle between the centred vectors $\mathbf{x} - \bar{x}\mathbf{1}$ and $\mathbf{y} - \bar{y}\mathbf{1}$:

$$
r = \cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

where $\mathbf{u}_i = x_i - \bar{x}$ and $\mathbf{v}_i = y_i - \bar{y}$.

**Relationship to linear regression:** In simple linear regression $Y = \beta_0 + \beta_1 X + \epsilon$, the slope is:
$$
\beta_1 = r \frac{\sigma_Y}{\sigma_X}
$$
The $R^2$ of the regression equals $r^2$.

**Spearman correlation:** For ranks $R(x_i)$ and $R(y_i)$, Spearman's $\rho$ is just Pearson's $r$ applied to the ranks. It detects any monotonic relationship (not just linear).

**Kendall's tau:** Another rank-based correlation measure:
$$
\tau = \frac{\text{concordant pairs} - \text{discordant pairs}}{n(n-1)/2}
$$

## Formula(s)

**Pearson correlation:**
$$
r_{XY} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}
$$

**Computational formula:**
$$
r_{XY} = \frac{n\sum x_i y_i - (\sum x_i)(\sum y_i)}{\sqrt{[n\sum x_i^2 - (\sum x_i)^2][n\sum y_i^2 - (\sum y_i)^2]}}
$$

**Spearman rank correlation:**
$$
\rho = 1 - \frac{6\sum_{i=1}^n d_i^2}{n(n^2 - 1)}
$$

## Properties

- **Range:** $-1 \leq r \leq 1$.
- **Symmetry:** $r_{XY} = r_{YX}$.
- **Scale invariance:** $r(aX + b, cY + d) = \text{sign}(ac) \cdot r_{XY}$.
- **Unitless:** Correlation has no units.
- **$r = \pm 1$:** All points lie exactly on a straight line.
- **$r = 0$:** No linear relationship (but non-linear relationships may exist).
- **$r = \sqrt{R^2}$** in simple linear regression.
- **Robustness:** Pearson $r$ is sensitive to outliers. Spearman's $\rho$ is more robust.

## Step-by-Step Worked Examples

### Example 1: Computing Pearson Correlation

**Problem:** Compute Pearson $r$ for the data: $(1,2), (2,4), (3,6), (4,8), (5,10)$.

**Solution:**

Step 1: Compute means.
$\bar{x} = 3$, $\bar{y} = 6$

Step 2: Compute deviations and products.
- $(1-3)(2-6) = (-2)(-4) = 8$
- $(2-3)(4-6) = (-1)(-2) = 2$
- $(3-3)(6-6) = 0$
- $(4-3)(8-6) = (1)(2) = 2$
- $(5-3)(10-6) = (2)(4) = 8$

$\sum (x_i - \bar{x})(y_i - \bar{y}) = 8 + 2 + 0 + 2 + 8 = 20$

Step 3: Compute squared deviations.
$\sum (x_i - \bar{x})^2 = 4 + 1 + 0 + 1 + 4 = 10$
$\sum (y_i - \bar{y})^2 = 16 + 4 + 0 + 4 + 16 = 40$

Step 4: Compute $r$.
$$
r = \frac{20}{\sqrt{10 \cdot 40}} = \frac{20}{\sqrt{400}} = \frac{20}{20} = 1
$$

Perfect positive correlation (data lies on line $y = 2x$).

### Example 2: Moderate Correlation

**Problem:** Compute $r$ for: $(1,3), (2,1), (3,5), (4,2), (5,4)$.

**Solution:**

Step 1: Means. $\bar{x} = 3$, $\bar{y} = 3$.

Step 2: Products.
- $(1-3)(3-3) = (-2)(0) = 0$
- $(2-3)(1-3) = (-1)(-2) = 2$
- $(3-3)(5-3) = (0)(2) = 0$
- $(4-3)(2-3) = (1)(-1) = -1$
- $(5-3)(4-3) = (2)(1) = 2$

Sum of products $= 0 + 2 + 0 - 1 + 2 = 3$

Step 3: Squared deviations.
$\sum (x_i - \bar{x})^2 = 4 + 1 + 0 + 1 + 4 = 10$
$\sum (y_i - \bar{y})^2 = 0 + 4 + 4 + 1 + 1 = 10$

Step 4:
$$
r = \frac{3}{\sqrt{10 \cdot 10}} = \frac{3}{10} = 0.3
$$

Moderate positive correlation.

### Example 3: Correlation vs Non-Linear Relationship

**Problem:** Compute Pearson $r$ for $X = \{-2, -1, 0, 1, 2\}$, $Y = \{4, 1, 0, 1, 4\}$ (quadratic relationship $Y = X^2$).

**Solution:**

Step 1: Means. $\bar{x} = 0$, $\bar{y} = (4+1+0+1+4)/5 = 2$.

Step 2: Products.
- $(-2-0)(4-2) = (-2)(2) = -4$
- $(-1-0)(1-2) = (-1)(-1) = 1$
- $(0-0)(0-2) = (0)(-2) = 0$
- $(1-0)(1-2) = (1)(-1) = -1$
- $(2-0)(4-2) = (2)(2) = 4$

Sum of products $= -4 + 1 + 0 - 1 + 4 = 0$

Step 3: $r = 0$.

Despite a perfect quadratic relationship ($Y = X^2$), Pearson correlation is zero because the relationship is non-linear. This demonstrates why we should always plot data, not just compute correlation.

## Visual Interpretation

The classic Anscombe's quartet demonstrates four different datasets with identical correlation coefficients ($r = 0.816$) but dramatically different relationships when visualised. This underscores the importance of plotting data before interpreting correlation.

A scatter plot with a clear elliptical shape indicates linear correlation. The narrower the ellipse, the stronger the correlation. The slope of the ellipse's major axis indicates the sign.

The correlation matrix heatmap uses colour intensity to show the strength of relationships. Darker colours (red or blue typically) indicate stronger correlations; lighter colours indicate weaker ones.

## Common Mistakes

1. **Correlation implies causation:** The most famous mistake. Two variables can be correlated without one causing the other. Spurious correlations (e.g., ice cream sales and drowning incidents are both correlated with temperature) are common.

2. **Ignoring non-linear relationships:** A zero Pearson correlation does not mean no relationship -- the relationship might be non-linear (e.g., quadratic, sinusoidal).

3. **Using $r$ for non-linear data:** Pearson correlation only measures linear relationships. Use Spearman for monotonic relationships.

4. **Outlier sensitivity:** A single outlier can dramatically change Pearson's $r$. Always check scatter plots.

5. **Confusing correlation strength with slope:** A correlation of 0.9 does not mean the slope is 0.9. The slope depends on the ratio of standard deviations.

6. **Averaging correlations:** You cannot simply average correlation coefficients. Use Fisher's z-transformation before averaging.

7. **Ignoring range restriction:** Correlation computed on a restricted range underestimates the true correlation in the full population.

8. **Multiple testing:** Computing many correlations increases the chance of false positives. Adjust significance thresholds (Bonferroni, FDR).

## Interview Questions

### Beginner - 5

1. **Q:** What is the range of the Pearson correlation coefficient?
   **A:** $-1 \leq r \leq 1$. -1 is perfect negative, 0 is no linear, 1 is perfect positive.

2. **Q:** What does a correlation of 0 mean?
   **A:** There is no linear relationship between the variables. A non-linear relationship may still exist.

3. **Q:** How is correlation different from covariance?
   **A:** Correlation is covariance standardised by the product of standard deviations. Correlation is unitless and bounded between -1 and 1.

4. **Q:** Does correlation imply causation?
   **A:** No. Correlation does not imply causation. A third variable may cause both, or the relationship may be coincidental.

5. **Q:** What is the difference between Pearson and Spearman correlation?
   **A:** Pearson measures linear relationships. Spearman measures monotonic relationships using ranks and is robust to outliers.

### Intermediate - 5

1. **Q:** How is correlation used for feature selection in ML?
   **A:** Features highly correlated with the target are selected; features highly correlated with each other are removed to reduce multicollinearity.

2. **Q:** What is the relationship between $r$ and $R^2$ in linear regression?
   **A:** In simple linear regression, $R^2 = r^2$, the proportion of variance in $Y$ explained by $X$.

3. **Q:** How does an outlier affect Pearson's $r$?
   **A:** A single outlier can inflate or deflate $r$ dramatically. Outliers can create a false correlation or mask a real one.

4. **Q:** What is the Variance Inflation Factor?
   **A:** $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ is from regressing predictor $j$ on all others. VIF > 5 or 10 indicates problematic multicollinearity.

5. **Q:** When would you use Spearman correlation instead of Pearson?
   **A:** When data is ordinal, the relationship is monotonic but not linear, or when outliers are present.

### Advanced - 3

1. **Q:** Prove that $-1 \leq r \leq 1$ using the Cauchy-Schwarz inequality.
   **A:** By Cauchy-Schwarz: $|\sum (x_i - \bar{x})(y_i - \bar{y})| \leq \sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}$. Dividing gives $|r| \leq 1$.

2. **Q:** Explain the concept of partial correlation and its use in graphical models.
   **A:** Partial correlation measures the correlation between two variables after removing the effect of a third variable. Zero partial correlation implies conditional independence in Gaussian graphical models.

3. **Q:** Derive the Fisher z-transformation and explain why it is needed.
   **A:** $z = \frac{1}{2}\ln\frac{1+r}{1-r}$. The distribution of $r$ is not normal. Fisher's $z$ is approximately normal with variance $1/(n-3)$, enabling hypothesis testing and confidence intervals.

## Practice Problems

### Easy - 5

1. Compute Pearson $r$ for $(1,1), (2,2), (3,3)$.

2. If $r = 0.64$, what is $R^2$?

3. Interpret $r = -0.85$ for hours of TV watched and test scores.

4. Compute Spearman $\rho$ for $(1,1), (2,4), (3,9)$.

5. If $\text{Cov}(X,Y) = 6$, $\sigma_X = 2$, $\sigma_Y = 4$, find $r$.

### Medium - 5

1. Given data: $(2,4), (3,6), (4,8), (5,10), (6,12)$. Compute $r$.

2. Show that $r = 1$ if and only if $y_i = a + bx_i$ with $b > 0$ for all $i$.

3. Explain the difference between $r = 0.5$ and $r = -0.5$.

4. A dataset has $n=30$, $r = 0.4$. Is this statistically significant at $\alpha = 0.05$? (Hint: use $t = r\sqrt{(n-2)/(1-r^2)}$)

5. Compute the correlation matrix for data: $(1,2,3), (2,4,6), (3,6,9)$.

### Hard - 3

1. Prove that $r^2 = R^2$ in simple linear regression.

2. Derive the asymptotic distribution of Pearson's $r$ under the null hypothesis $\rho = 0$.

3. Show that the partial correlation $\rho_{XY|Z}$ can be expressed in terms of simple correlations.

## Solutions

**Easy:**

1. $r = 1$ (perfect positive, all points on line $y=x$).

2. $R^2 = 0.64^2 = 0.4096$ (about 41% of variance explained).

3. Strong negative correlation: more TV watched associates with lower test scores.

4. Ranks of X: 1,2,3. Ranks of Y: 1,2,3. $d_i = 0$. $\rho = 1$.

5. $r = 6/(2 \cdot 4) = 6/8 = 0.75$.

**Medium:**

1. $\bar{x}=4$, $\bar{y}=8$. All products: $(-2)(-4)=8$, $(-1)(-2)=2$, $(0)(0)=0$, $(1)(2)=2$, $(2)(4)=8$. Sum=20. Squared deviations X: $4+1+0+1+4=10$. Y: $16+4+0+4+16=40$. $r = 20/\sqrt{400}=1$.

2. If $y_i = a+bx_i$ with $b>0$, then $(y_i - \bar{y}) = b(x_i - \bar{x})$. Then $r = \sum b(x_i-\bar{x})^2 / (\sqrt{\sum(x_i-\bar{x})^2}\sqrt{b^2\sum(x_i-\bar{x})^2}) = b/|b| = 1$.

3. Same strength (magnitude 0.5) but opposite direction. $r=0.5$: positive relationship. $r=-0.5$: negative relationship.

4. $t = 0.4\sqrt{(30-2)/(1-0.16)} = 0.4\sqrt{28/0.84} \approx 0.4\sqrt{33.33} \approx 0.4(5.77) \approx 2.31$. With $df=28$, $t_{0.025,28} \approx 2.048$. Since $2.31 > 2.048$, it is significant.

5. All variables are perfectly correlated ($r=1$ between any pair).

**Hard:**

1. In simple linear regression $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$. $R^2 = \frac{\sum(\hat{y}_i - \bar{y})^2}{\sum(y_i - \bar{y})^2} = \frac{\hat{\beta}_1^2\sum(x_i-\bar{x})^2}{\sum(y_i-\bar{y})^2} = \frac{(\sum(x_i-\bar{x})(y_i-\bar{y})/\sum(x_i-\bar{x})^2)^2 \sum(x_i-\bar{x})^2}{\sum(y_i-\bar{y})^2} = \frac{(\sum(x_i-\bar{x})(y_i-\bar{y}))^2}{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2} = r^2$.

2. Under $H_0: \rho=0$, the test statistic $t = r\sqrt{(n-2)/(1-r^2)}$ follows a $t$-distribution with $n-2$ degrees of freedom. For large $n$, $r \sim N(0, 1/n)$ approximately.

3. $\rho_{XY|Z} = \frac{\rho_{XY} - \rho_{XZ}\rho_{YZ}}{\sqrt{(1-\rho_{XZ}^2)(1-\rho_{YZ}^2)}}$. This removes the linear effect of $Z$ from both $X$ and $Y$.

## Related Concepts

- Covariance (MATH-082) — unstandardised correlation
- Coefficient of Determination ($R^2$) — squared correlation
- Linear Regression — models linear relationships
- Spearman Correlation — rank-based alternative
- Correlation Matrix — collection of pairwise correlations
- Partial Correlation — conditional correlation
- Autocorrelation — correlation with lagged values

## Next Concepts

- Skewness (MATH-084) — asymmetry in distributions
- Kurtosis (MATH-085) — tail heaviness
- Hypothesis Testing (MATH-087) — testing correlation significance

## Summary

Correlation measures the strength and direction of a linear relationship between two variables. Pearson's $r$ ranges from $-1$ to $1$, with $\pm 1$ indicating perfect linear relationships and $0$ indicating no linear relationship. Spearman's $\rho$ extends this to monotonic relationships using ranks. Correlation is a standardised, unitless version of covariance. In AI/ML, correlation is used for feature selection, multicollinearity detection, EDA, and dimensionality reduction. Correlation does not imply causation -- this is a critical distinction. Always visualise data before interpreting correlation coefficients.

## Key Takeaways

- Pearson $r \in [-1, 1]$ measures linear relationships.
- $r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$.
- Spearman $\rho$ uses ranks for monotonic relationships.
- Correlation does not imply causation.
- In linear regression, $R^2 = r^2$.
- Use correlation matrices and heatmaps for EDA.
- High inter-feature correlation indicates multicollinearity.
- Pearson $r$ is sensitive to outliers; Spearman is robust.
- Always visualise data (Anscombe's quartet).
