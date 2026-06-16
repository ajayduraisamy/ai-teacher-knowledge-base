# Concept: Variance

## Concept ID

MATH-080

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define variance as the average squared deviation from the mean
- Distinguish between population variance $\sigma^2$ and sample variance $s^2$
- Understand Bessel's correction ($n-1$) and its purpose
- Interpret variance in the context of the bias-variance tradeoff
- Apply variance in AI/ML contexts including PCA and model evaluation

## Prerequisites

- Mean (MATH-077)
- Basic probability
- Summation notation
- Understanding of expectation

## Definition

**Variance** measures the spread of a dataset around its mean. It is the average of the squared deviations from the mean. Variance quantifies how far values are spread out from the central value.

**Population variance:**
$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2
$$

**Sample variance (with Bessel's correction):**
$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2
$$

The square root of variance is the standard deviation $\sigma = \sqrt{\sigma^2}$.

## Intuition

Imagine two dartboards. On the first, all darts cluster tightly around the bullseye. On the second, darts are scattered widely across the board. The variance captures this difference — low variance means points are close to the centre; high variance means they are spread out.

The reason we square the deviations is twofold: (1) squaring makes all deviations positive so they don't cancel out, and (2) squaring penalises larger deviations more heavily than smaller ones. A point 2 units away contributes 4 to the variance; a point 4 units away contributes 16.

## Why This Concept Matters

Variance is fundamental to understanding uncertainty, risk, and variability. In finance, variance measures investment risk. In manufacturing, variance indicates product inconsistency. In experimental science, variance determines whether observed effects are meaningful or just noise.

Variance is also the reason we need Bessel's correction. When we estimate population variance from a sample, we systematically underestimate it because the sample mean is closer to the sample points than the population mean would be. Dividing by $n-1$ instead of $n$ corrects this bias.

## Historical Background

The concept of variance was formalised by Ronald Fisher in his 1918 paper "The Correlation Between Relatives on the Supposition of Mendelian Inheritance." Fisher introduced the term "variance" as a measure of dispersion and developed analysis of variance (ANOVA) as a statistical method.

The earlier concept of "mean square error" was used by Legendre and Gauss in the method of least squares (1805-1809). Bessel's correction was introduced by Friedrich Bessel in the 19th century to correct for bias in sample variance estimation.

## Real World Examples

**Finance:** The variance of daily stock returns measures the risk of holding a stock. High variance means high volatility and uncertainty.

**Manufacturing:** A factory measures the variance in bottle fill volumes. Low variance indicates consistent filling; high variance suggests machine calibration issues.

**Education:** The variance of test scores within a class indicates how much student performance differs. Low variance suggests uniform understanding; high variance shows diverse ability levels.

**Weather:** The variance of daily temperatures around the seasonal mean captures climate variability.

**Healthcare:** Variance in patient recovery times helps hospitals plan resource allocation.

## AI/ML Relevance

**Bias-variance tradeoff:** This is one of the most fundamental concepts in machine learning. Total error = Bias$^2$ + Variance + Irreducible error.

- High bias: model underfits (too simple; pays little attention to training data)
- High variance: model overfits (too complex; learns noise in training data)
- The goal is to find the sweet spot that minimises total error.

**Explained variance in PCA:** Principal Component Analysis finds directions of maximum variance in the data. The explained variance ratio tells us how much of the total variance each principal component captures.

**Variance of predictions:** In ensemble methods, averaging reduces the variance of predictions without increasing bias. If $M$ independent models each have variance $\sigma^2$, the ensemble mean has variance $\sigma^2 / M$.

**Feature selection:** Features with very low variance provide little information. VarianceThreshold in scikit-learn removes features below a variance threshold.

**Regularisation:** L2 regularisation (Ridge) penalises large weights, which reduces the variance of the model at the cost of increased bias.

**Batch normalisation:** Reduces internal covariate shift by normalising each layer's activations to have zero mean and unit variance.

**Gradient variance:** In stochastic gradient descent, the variance of gradient estimates affects convergence speed and stability.

## Mathematical Explanation

Variance is defined as the expected squared deviation from the mean:

$$
\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
$$

This alternative formula (computational formula) is useful for calculations:

$$
\sigma^2 = \frac{\sum x_i^2}{n} - \left(\frac{\sum x_i}{n}\right)^2 = \overline{x^2} - (\bar{x})^2
$$

**Bessel's correction:** For sample variance, using $n-1$ makes $s^2$ an unbiased estimator of $\sigma^2$:

$$
E[s^2] = E\left[\frac{1}{n-1}\sum (x_i - \bar{x})^2\right] = \sigma^2
$$

The proof uses the fact that $\sum (x_i - \bar{x})^2 = \sum (x_i - \mu)^2 - n(\bar{x} - \mu)^2$, and $E[\sum (x_i - \mu)^2] = n\sigma^2$, $E[n(\bar{x} - \mu)^2] = \sigma^2$.

**Properties of variance:**

- $\text{Var}(c) = 0$ for any constant $c$.
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$.
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$.
- If $X$ and $Y$ are independent: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

## Formula(s)

**Population variance:**
$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2
$$

**Sample variance:**
$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2
$$

**Computational formula:**
$$
\sigma^2 = E[X^2] - (E[X])^2
$$
$$
s^2 = \frac{1}{n-1}\left(\sum x_i^2 - \frac{(\sum x_i)^2}{n}\right)
$$

**Variance of a random variable:**
$$
\text{Var}(X) = \int (x - \mu)^2 f(x)\,dx
$$

## Properties

- **Non-negative:** $\sigma^2 \geq 0$. Variance is zero only when all values are identical.
- **Units:** Variance is in squared units of the original data (if $x$ is in metres, $\sigma^2$ is in square metres).
- **Scale sensitivity:** Multiplying data by $a$ multiplies variance by $a^2$.
- **Shift invariance:** Adding a constant $b$ to all data does not change variance: $\text{Var}(X+b) = \text{Var}(X)$.
- **Decomposition:** Total variance = Between-group variance + Within-group variance (ANOVA).
- **Additivity for independent variables:** $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ if $X$ and $Y$ are independent.

## Step-by-Step Worked Examples

### Example 1: Population Variance

**Problem:** Find the population variance of $\{2, 4, 6, 8, 10\}$.

**Solution:**

Step 1: Compute the population mean $\mu$.
$$
\mu = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = 6
$$

Step 2: Compute squared deviations.
- $(2-6)^2 = (-4)^2 = 16$
- $(4-6)^2 = (-2)^2 = 4$
- $(6-6)^2 = 0^2 = 0$
- $(8-6)^2 = 2^2 = 4$
- $(10-6)^2 = 4^2 = 16$

Step 3: Average the squared deviations.
$$
\sigma^2 = \frac{16 + 4 + 0 + 4 + 16}{5} = \frac{40}{5} = 8
$$

### Example 2: Sample Variance (Bessel's Correction)

**Problem:** Compute the sample variance of $\{2, 4, 6, 8, 10\}$.

**Solution:**

Step 1: Compute the sample mean $\bar{x} = 6$ (same as above).

Step 2: Compute squared deviations (same as above): $16, 4, 0, 4, 16$, sum = 40.

Step 3: Divide by $n-1 = 4$ instead of $n = 5$.
$$
s^2 = \frac{40}{5-1} = \frac{40}{4} = 10
$$

Note that $s^2 = 10$ is larger than $\sigma^2 = 8$. This is Bessel's correction — the sample variance is unbiased for the population variance.

### Example 3: Variance and Bias-Variance Tradeoff

**Problem:** A model trained on 5 different training sets yields predictions for a fixed test point: $\{12, 13, 12.5, 11.5, 14\}$. The true value is 10. Compute the bias and variance of the model's predictions.

**Solution:**

Step 1: Compute the mean of predictions.
$$
\bar{\hat{y}} = \frac{12 + 13 + 12.5 + 11.5 + 14}{5} = \frac{63}{5} = 12.6
$$

Step 2: Compute bias.
$$
\text{Bias} = \bar{\hat{y}} - y_{\text{true}} = 12.6 - 10 = 2.6
$$

Step 3: Compute variance of predictions.
Squared deviations from $\bar{\hat{y}} = 12.6$:
- $(12 - 12.6)^2 = 0.36$
- $(13 - 12.6)^2 = 0.16$
- $(12.5 - 12.6)^2 = 0.01$
- $(11.5 - 12.6)^2 = 1.21$
- $(14 - 12.6)^2 = 1.96$

$$
\text{Variance} = \frac{0.36 + 0.16 + 0.01 + 1.21 + 1.96}{5} = \frac{3.7}{5} = 0.74
$$

The model has bias 2.6 (systematically overpredicts) and variance 0.74 (moderate spread).

## Visual Interpretation

Variance is visualised by the spread of data points around the mean. In a scatter plot, low variance appears as tight clustering; high variance as wide dispersion.

In a probability distribution, variance is related to the width of the distribution. A narrow, tall distribution has low variance. A wide, flat distribution has high variance. For the normal distribution, approximately 68% of data falls within $\pm \sigma$ of the mean, 95% within $\pm 2\sigma$, and 99.7% within $\pm 3\sigma$.

The variance is the second central moment of the distribution (after the mean, which is the first moment). Higher moments (skewness, kurtosis) build on the mean and variance.

## Common Mistakes

1. **Confusing $n$ and $n-1$:** Using $n$ for sample variance gives a biased estimator that underestimates population variance. Always use $n-1$ for sample data.

2. **Interpreting variance in original units:** Variance is in squared units. For interpretability, use the standard deviation (same units as the data).

3. **Assuming variance is additive for non-independent variables:** $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ only when $X$ and $Y$ are uncorrelated. For correlated variables, include the covariance term.

4. **Ignoring variance when comparing groups:** Two groups can have the same mean but very different variances. Reporting only the mean hides this important information.

5. **Using variance for outlier detection without caution:** Large squared deviations dominate variance calculation. A single extreme outlier can inflate variance dramatically.

6. **Believing zero variance means "no variation":** Zero variance means all values are identical. This is rare in real data and may indicate data entry errors.

7. **Computing variance without computing the mean first:** The mean is required for variance. Using an incorrect mean propagates error into the variance.

## Interview Questions

### Beginner - 5

1. **Q:** What is variance in simple terms?
   **A:** Variance measures how spread out data is from the mean — the average squared distance from the centre.

2. **Q:** Why do we square the deviations when computing variance?
   **A:** To make all deviations positive (so they don't cancel) and to penalise larger deviations more heavily.

3. **Q:** What is the difference between population variance and sample variance?
   **A:** Population variance divides by $N$; sample variance divides by $n-1$ (Bessel's correction).

4. **Q:** What does a variance of zero mean?
   **A:** All values in the dataset are identical — there is no spread.

5. **Q:** How does adding a constant affect variance?
   **A:** Adding a constant does not change variance: $\text{Var}(X+b) = \text{Var}(X)$.

### Intermediate - 5

1. **Q:** What is Bessel's correction and why do we use it?
   **A:** Bessel's correction divides by $n-1$ instead of $n$ to make the sample variance an unbiased estimator of population variance.

2. **Q:** Explain the bias-variance tradeoff.
   **A:** Total error = Bias$^2$ + Variance + Irreducible error. Simple models have high bias (underfit); complex models have high variance (overfit). The tradeoff balances these.

3. **Q:** How is variance used in Principal Component Analysis?
   **A:** PCA finds orthogonal directions that maximise the variance of projected data. The first PC captures the most variance, the second the next most, etc.

4. **Q:** What is the computational formula for variance and when is it useful?
   **A:** $\sigma^2 = E[X^2] - (E[X])^2 = \overline{x^2} - (\bar{x})^2$. It avoids computing each deviation separately.

5. **Q:** How does ensemble averaging reduce variance?
   **A:** If $M$ independent models each have variance $\sigma^2$, the ensemble mean has variance $\sigma^2/M$, reducing overall variance.

### Advanced - 3

1. **Q:** Prove that the sample variance $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$ is unbiased for $\sigma^2$.
   **A:** $E[\sum (x_i - \bar{x})^2] = E[\sum (x_i - \mu)^2 - n(\bar{x} - \mu)^2] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2$, so $E[s^2] = \sigma^2$.

2. **Q:** Derive the variance of the sample mean $\bar{x}$.
   **A:** $\text{Var}(\bar{x}) = \text{Var}(\frac{1}{n}\sum x_i) = \frac{1}{n^2}\sum \text{Var}(x_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}$.

3. **Q:** In deep learning, explain the role of variance in gradient descent convergence.
   **A:** High gradient variance causes noisy updates and slow convergence. Techniques like momentum, Adam, and learning rate scheduling reduce effective variance. Batch normalisation reduces variance across layers.

## Practice Problems

### Easy - 5

1. Compute the population variance of $\{1, 3, 5\}$.

2. Compute the sample variance of $\{1, 3, 5\}$.

3. If $\text{Var}(X) = 4$, what is $\text{Var}(2X)$?

4. A dataset has $\sum (x_i - \bar{x})^2 = 50$, $n = 10$. Find the sample variance.

5. If all values in a dataset are equal to 7, what is the variance?

### Medium - 5

1. Given $\sum x_i = 100$, $\sum x_i^2 = 1200$, $n = 10$, compute the sample variance.

2. Show that $\text{Var}(X + c) = \text{Var}(X)$ for any constant $c$.

3. Two models on a test point: Model A predictions $\{9, 10, 11\}$ (true = 10). Model B predictions $\{5, 10, 15\}$ (true = 10). Compute bias and variance for both.

4. Prove that $\sigma^2 = E[X^2] - \mu^2$.

5. In PCA, the first principal component has variance 5 and the second has variance 2. If total variance is 10, what are the explained variance ratios?

### Hard - 3

1. Prove that the sample variance converges in probability to the population variance (consistency of $s^2$).

2. Derive the variance of a Bernoulli random variable and explain its relationship to the variance of a Binomial.

3. Given $n$ observations with sample variance $s^2$, show that adding a new observation $x_{n+1}$ changes the variance predictably. Derive the update formula.

## Solutions

**Easy:**

1. $\mu = (1+3+5)/3 = 3$. $\sigma^2 = ((1-3)^2 + (3-3)^2 + (5-3)^2)/3 = (4+0+4)/3 = 8/3 \approx 2.67$.

2. $\bar{x} = 3$. $s^2 = (4+0+4)/(3-1) = 8/2 = 4$.

3. $\text{Var}(2X) = 4 \times \text{Var}(X) = 16$.

4. $s^2 = 50/(10-1) = 50/9 \approx 5.56$.

5. $\text{Var} = 0$ (all values are the same).

**Medium:**

1. $\bar{x} = 100/10 = 10$. $s^2 = (1/(10-1))(1200 - 100^2/10) = (1/9)(1200 - 1000) = 200/9 \approx 22.22$.

2. $\text{Var}(X+c) = E[(X+c - (\mu+c))^2] = E[(X-\mu)^2] = \text{Var}(X)$.

3. Model A: $\bar{\hat{y}} = 10$, Bias $= 0$, Variance $= ((9-10)^2 + (10-10)^2 + (11-10)^2)/3 = 2/3 \approx 0.67$.
   Model B: $\bar{\hat{y}} = 10$, Bias $= 0$, Variance $= ((5-10)^2 + (10-10)^2 + (15-10)^2)/3 = 50/3 \approx 16.67$.
   Same bias, very different variance.

4. $\sigma^2 = E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - 2\mu^2 + \mu^2 = E[X^2] - \mu^2$.

5. PC1 explained variance $= 5/10 = 0.5$ (50%). PC2 explained variance $= 2/10 = 0.2$ (20%). Total explained $= 70\%$.

**Hard:**

1. $s_n^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$. By the law of large numbers, $\bar{x} \xrightarrow{p} \mu$, and $\frac{1}{n}\sum x_i^2 \xrightarrow{p} E[X^2]$. By Slutsky's theorem, $s_n^2 \xrightarrow{p} E[X^2] - \mu^2 = \sigma^2$.

2. For Bernoulli($p$): $E[X] = p$, $E[X^2] = p$. $\text{Var}(X) = p - p^2 = p(1-p)$. For Binomial($n,p$): $Y = \sum_{i=1}^n X_i$, $\text{Var}(Y) = np(1-p)$ by additivity of variance for independent variables.

3. Original: $\bar{x}_n = \frac{1}{n}\sum_{i=1}^n x_i$, $S_n = \sum_{i=1}^n (x_i - \bar{x}_n)^2$. After adding $x_{n+1}$: $\bar{x}_{n+1} = \frac{n\bar{x}_n + x_{n+1}}{n+1}$. New sum of squares: $S_{n+1} = S_n + (x_{n+1} - \bar{x}_n)^2 \cdot \frac{n}{n+1}$. New variance: $s_{n+1}^2 = S_{n+1}/n$ (dividing by $n$ for sample of size $n+1$).

## Related Concepts

- Mean (MATH-077) — central value from which variance measures spread
- Standard Deviation (MATH-081) — square root of variance, same units as data
- Covariance (MATH-082) — measures joint variability between two variables
- Bessel's Correction — the $n-1$ factor for unbiased estimation
- Bias-Variance Tradeoff — fundamental ML concept
- Principal Component Analysis — uses variance for dimensionality reduction

## Next Concepts

- Standard Deviation (MATH-081) — more interpretable version of variance
- Covariance (MATH-082) — extending variance to two variables
- Correlation (MATH-083) — normalised covariance
- Confidence Intervals (MATH-086) — using variance to quantify estimation uncertainty

## Summary

Variance quantifies the spread of data around the mean as the average squared deviation. Population variance $\sigma^2$ divides by $N$; sample variance $s^2$ divides by $n-1$ using Bessel's correction to provide an unbiased estimate. Variance is always non-negative, is in squared units, and is invariant to additive shifts. The computational formula $\sigma^2 = E[X^2] - \mu^2$ simplifies calculations. In AI/ML, variance is central to the bias-variance tradeoff, PCA (explained variance), regularisation, ensemble methods, and batch normalisation.

## Key Takeaways

- Variance measures spread as the average squared distance from the mean.
- Population variance: $\sigma^2 = \frac{1}{N}\sum (x_i - \mu)^2$.
- Sample variance: $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$ (Bessel's correction).
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$.
- Variance is in squared units — use standard deviation for interpretability.
- The bias-variance tradeoff is a foundational ML concept.
- PCA selects directions maximising variance.
- Ensemble averaging reduces prediction variance by factor $1/M$.
- Zero variance means all values are identical.
