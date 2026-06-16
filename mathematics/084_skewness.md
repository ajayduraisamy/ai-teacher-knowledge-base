# Concept: Skewness

## Concept ID

MATH-084

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define skewness as the third standardised moment
- Distinguish between positive, negative, and zero skewness
- Interpret skewness in the context of data distributions
- Apply skewness for checking data normality in AI/ML pipelines
- Understand transformations for correcting skewness

## Prerequisites

- Mean (MATH-077)
- Variance (MATH-080)
- Standard Deviation (MATH-081)
- Understanding of moments

## Definition

**Skewness** is a measure of the asymmetry of a probability distribution about its mean. It quantifies how much the distribution deviates from symmetry. The skewness of a perfectly symmetric distribution (such as the normal distribution) is zero.

**Population skewness:**
$$
\gamma_1 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \mu}{\sigma}\right)^3
$$

**Sample skewness (adjusted):**
$$
g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3
$$

## Intuition

Skewness tells you whether the tail of the distribution is longer on the right side or the left side.

- **Positive skewness (right-skewed):** The right tail is longer or fatter. Most data is concentrated on the left. Mean $>$ Median $>$ Mode.
- **Negative skewness (left-skewed):** The left tail is longer or fatter. Most data is concentrated on the right. Mean $<$ Median $<$ Mode.
- **Zero skewness:** The distribution is symmetric. Mean $\approx$ Median $\approx$ Mode.

Imagine a distribution of incomes: most people earn between $30,000 and $80,000, but a small number earn millions. This creates a long right tail -- positive skewness. The mean income is pulled to the right by the wealthy minority, while the median better represents the typical person.

## Why This Concept Matters

Understanding skewness is critical for:

- **Choosing appropriate statistical tests:** Many parametric tests assume normality. Skewed data violates this assumption.
- **Model selection:** Skewed features may perform poorly in linear models without transformation.
- **Risk assessment:** In finance, positive skewness means potential for extreme positive outcomes; negative skewness means risk of extreme negative outcomes.
- **Data preprocessing:** Identifying skewness guides the choice of transformations (log, Box-Cox).
- **Outlier detection:** High skewness can indicate the presence of outliers in one tail.

## Historical Background

Skewness was introduced as a formal statistical concept by Karl Pearson in 1895. Pearson developed the moment-based measure $\gamma_1$ as part of his system of frequency curves. He also introduced the Pearson mode skewness:
$$
\text{Skewness} = \frac{\text{Mean} - \text{Mode}}{\text{Standard Deviation}}
$$

Later, Ronald Fisher (1930) worked on the sampling distribution of skewness and developed the adjusted sample skewness $g_1$ to reduce bias in small samples.

## Real World Examples

**Income distribution:** Nearly always positively skewed. Most people earn near the median, but a small number earn extremely high incomes, creating a long right tail.

**Insurance claims:** Claim amounts are positively skewed. Most claims are small, but a few catastrophic claims are very large.

**Human lifetime:** Age at death for a population with good healthcare is negatively skewed (left-skewed). Most people die at older ages, with a left tail of premature deaths.

**Test scores:** A very easy exam produces negatively skewed scores (most students score high, with a left tail of low scores). A very hard exam produces positively skewed scores.

**Housing prices:** Positively skewed -- most homes are moderately priced, but a small number of luxury properties have extremely high prices.

## AI/ML Relevance

**Checking data normality:** Many ML models assume normally distributed features (linear regression, LDA, Gaussian Naive Bayes). Skewness $\approx 0$ indicates approximate symmetry, a necessary condition for normality.

**Feature transformations:** When features have high skewness, transformations can make them more symmetric:
- **Log transform:** $x' = \log(x)$ for positive right-skewed data.
- **Square root transform:** $x' = \sqrt{x}$ for moderate right skew.
- **Box-Cox transform:** $x' = (x^\lambda - 1)/\lambda$ for $\lambda \neq 0$, $\log(x)$ for $\lambda = 0$. The optimal $\lambda$ is estimated to minimise skewness.
- **Yeo-Johnson transform:** Like Box-Cox but handles zero and negative values.

**Model evaluation:** Skewness of residuals indicates model misspecification. Residuals should be approximately symmetric (skewness near 0) for well-specified linear models.

**Anomaly detection:** High skewness in feature distributions can make distance-based anomaly detection (like Mahalanobis distance) less reliable because the normality assumption is violated.

**Gradient dynamics:** In deep learning, the skewness of gradient distributions can affect optimisation. Heavy-tailed gradients (high positive skewness) may require adaptive learning rates.

**Credit scoring:** Credit loss distributions are highly right-skewed -- most loans are repaid, but a few default with partial recovery. Modelling this skewness is essential for risk management.

## Mathematical Explanation

Skewness is the third standardised moment:
$$
\gamma_1 = E\left[\left(\frac{X - \mu}{\sigma}\right)^3\right] = \frac{E[(X - \mu)^3]}{\sigma^3}
$$

The third power preserves the sign of deviations, so positive deviations (above the mean) and negative deviations (below the mean) do not cancel out as they do in the mean.

**Pearson's moment coefficient of skewness:**
$$
\gamma_1 = \frac{\frac{1}{n}\sum (x_i - \mu)^3}{\left(\frac{1}{n}\sum (x_i - \mu)^2\right)^{3/2}}
$$

**Pearson's mode skewness:**
$$
\text{Skewness}_P = \frac{\bar{x} - \text{Mode}}{s}
$$

**Pearson's median skewness (second skewness coefficient):**
$$
\text{Skewness}_{P2} = \frac{3(\bar{x} - \text{Median})}{s}
$$

**Relationship with moments:**
- First moment (mean): location
- Second moment (variance): spread
- Third moment (skewness): asymmetry
- Fourth moment (kurtosis): tail weight

## Formula(s)

**Population skewness:**
$$
\gamma_1 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \mu}{\sigma}\right)^3
$$

**Sample skewness (Fisher-Pearson adjusted):**
$$
g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3
$$

**Pearson median skewness:**
$$
\text{Skewness} = \frac{3(\bar{x} - \text{Median})}{s}
$$

**Box-Cox transformation:**
$$
x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \\ \log(x), & \lambda = 0 \end{cases}
$$

## Properties

- **Range:** Skewness can be any real number. Typical values for moderate skewness are between $-2$ and $2$.
- **Zero for symmetric distributions:** Normal, uniform, and $t$-distributions have zero skewness.
- **Invariance to location shifts:** $\text{Skewness}(X + c) = \text{Skewness}(X)$.
- **Scale invariance:** $\text{Skewness}(aX) = \text{Skewness}(X)$ for $a > 0$. For $a < 0$, the sign flips.
- **Sensitive to sample size:** Sample skewness has high variance for small $n$ and requires large samples for reliable estimation.
- **Interpretation guidelines:** $|g_1| < 0.5$ is approximately symmetric; $0.5 \leq |g_1| < 1$ is moderately skewed; $|g_1| \geq 1$ is highly skewed.

## Step-by-Step Worked Examples

### Example 1: Computing Skewness

**Problem:** Compute the population skewness for $\{1, 2, 3, 4, 10\}$.

**Solution:**

Step 1: Compute mean.
$$
\mu = \frac{1 + 2 + 3 + 4 + 10}{5} = \frac{20}{5} = 4
$$

Step 2: Compute standard deviation.
$$
\sigma^2 = \frac{(1-4)^2 + (2-4)^2 + (3-4)^2 + (4-4)^2 + (10-4)^2}{5}
$$
$$
\sigma^2 = \frac{9 + 4 + 1 + 0 + 36}{5} = \frac{50}{5} = 10
$$
$$
\sigma = \sqrt{10} \approx 3.162
$$

Step 3: Compute cubed standardised deviations.
- $((1-4)/3.162)^3 = (-0.949)^3 = -0.855$
- $((2-4)/3.162)^3 = (-0.632)^3 = -0.253$
- $((3-4)/3.162)^3 = (-0.316)^3 = -0.032$
- $((4-4)/3.162)^3 = (0)^3 = 0$
- $((10-4)/3.162)^3 = (1.897)^3 = 6.827$

Step 4: Average.
$$
\gamma_1 = \frac{-0.855 - 0.253 - 0.032 + 0 + 6.827}{5} = \frac{5.687}{5} = 1.137
$$

The positive skewness (1.137) indicates right skew, confirmed by the outlier 10 pulling the tail to the right.

### Example 2: Negative Skewness

**Problem:** Compute skewness for $\{1, 7, 8, 9, 10\}$.

**Solution:**

Step 1: Mean.
$$
\mu = \frac{1 + 7 + 8 + 9 + 10}{5} = 7
$$

Step 2: Variance and SD.
$$
\sigma^2 = \frac{(1-7)^2 + (7-7)^2 + (8-7)^2 + (9-7)^2 + (10-7)^2}{5}
$$
$$
\sigma^2 = \frac{36 + 0 + 1 + 4 + 9}{5} = \frac{50}{5} = 10
$$
$$
\sigma = \sqrt{10} \approx 3.162
$$

Step 3: Cubed standardised deviations.
- $((1-7)/3.162)^3 = (-1.897)^3 = -6.827$
- $((7-7)/3.162)^3 = 0$
- $((8-7)/3.162)^3 = (0.316)^3 = 0.032$
- $((9-7)/3.162)^3 = (0.632)^3 = 0.253$
- $((10-7)/3.162)^3 = (0.949)^3 = 0.855$

Step 4: Average.
$$
\gamma_1 = \frac{-6.827 + 0 + 0.032 + 0.253 + 0.855}{5} = \frac{-5.687}{5} = -1.137
$$

Negative skewness (indicates left skew, with the outlier 1 pulling the left tail).

### Example 3: Symmetric Distribution

**Problem:** Compute skewness for $\{1, 2, 3, 4, 5\}$.

**Solution:**

Step 1: $\mu = 3$.

Step 2: $\sigma^2 = (4+1+0+1+4)/5 = 2$, $\sigma = \sqrt{2} \approx 1.414$.

Step 3: Cubed deviations.
- $((1-3)/1.414)^3 = (-1.414)^3 = -2.828$
- $((2-3)/1.414)^3 = (-0.707)^3 = -0.354$
- $((3-3)/1.414)^3 = 0$
- $((4-3)/1.414)^3 = (0.707)^3 = 0.354$
- $((5-3)/1.414)^3 = (1.414)^3 = 2.828$

Step 4: $\gamma_1 = (-2.828 - 0.354 + 0 + 0.354 + 2.828)/5 = 0/5 = 0$.

Zero skewness confirms symmetry.

## Visual Interpretation

On a histogram, positive skewness appears as a long right tail: the histogram has a peak on the left and gradually descends to the right. Negative skewness is the mirror image: peak on the right with a long left tail.

In a box plot, positive skewness shows as the median closer to the bottom of the box, with a longer upper whisker. Negative skewness shows the median near the top of the box with a longer lower whisker.

For a density plot, right-skewed distributions peak left of centre and trail off to the right. Left-skewed distributions peak right of centre and trail off to the left.

## Common Mistakes

1. **Confusing skewness with kurtosis:** Skewness measures asymmetry; kurtosis measures tail weight. A distribution can have zero skewness but high kurtosis (heavy tails).

2. **Assuming zero skewness implies normality:** Zero skewness is necessary but not sufficient for normality. A symmetric distribution can still be non-normal (e.g., uniform, $t$ with low df).

3. **Using raw skewness for small samples:** Sample skewness is biased and has high variance for $n < 30$. Use the adjusted Fisher-Pearson coefficient.

4. **Ignoring skewness before modelling:** Linear models assume symmetric residuals. Skewed features can violate this and degrade performance.

5. **Applying log transform to negative values:** The log transform is defined only for positive values. Use Yeo-Johnson for data with zeros or negatives.

6. **Over-interpreting small skewness:** Skewness near zero (say $< 0.5$ in absolute value) is common even for normal samples due to sampling variability.

7. **Equating skewness with the presence of outliers:** Skewness can be high without outliers (e.g., exponential distribution) and low with outliers (if outliers are balanced on both sides).

## Interview Questions

### Beginner - 5

1. **Q:** What is skewness?
   **A:** Skewness measures the asymmetry of a distribution. Positive skew means a longer right tail; negative skew means a longer left tail.

2. **Q:** What is the skewness of a normal distribution?
   **A:** Zero. The normal distribution is perfectly symmetric.

3. **Q:** How does skewness affect the relationship between mean and median?
   **A:** For positive skew, mean $>$ median. For negative skew, mean $<$ median. For symmetric, mean $\approx$ median.

4. **Q:** What does a positively skewed income distribution tell us?
   **A:** Most people earn moderate incomes, but a small number earn very high incomes, creating a long right tail.

5. **Q:** Why is skewness measured using the third power?
   **A:** The third power preserves sign (positive devations stay positive, negative stay negative), so asymmetry is not cancelled out.

### Intermediate - 5

1. **Q:** How do you interpret a skewness value of 1.5?
   **A:** The distribution is highly positively skewed. The right tail is substantially longer than the left tail.

2. **Q:** What is the Box-Cox transformation and when do you use it?
   **A:** Box-Cox transforms data to make it more normal: $x^{(\lambda)} = (x^\lambda - 1)/\lambda$. Use it to reduce skewness in positive features.

3. **Q:** How does skewness affect linear regression?
   **A:** Skewed residuals violate the normality assumption, affecting confidence intervals and $p$-values. Skewed features can lead to unstable coefficient estimates.

4. **Q:** What is the difference between sample skewness and population skewness?
   **A:** Sample skewness uses $s$ (sample SD) instead of $\sigma$ and includes an adjustment factor $n/((n-1)(n-2))$ to reduce bias.

5. **Q:** How can you detect skewness without computing a coefficient?
   **A:** Visual inspection of histograms, box plots (check which whisker is longer), and Q-Q plots.

### Advanced - 3

1. **Q:** Derive the sampling distribution of sample skewness under normality.
   **A:** Under normality, the sample skewness $g_1$ has approximately $E[g_1] = 0$ and $\text{Var}(g_1) \approx 6/n$ for large $n$. This is used to test whether the population skewness differs from zero.

2. **Q:** Explain how the skewness of gradients affects the convergence of stochastic optimisation methods.
   **A:** Heavy-tailed (high positive skewness) gradient distributions can cause SGD to have high variance updates. Methods like Adam and gradient clipping mitigate this by normalising the gradient scale.

3. **Q:** Prove that for any distribution with finite third moment, if the PDF is symmetric about $\mu$, then $\gamma_1 = 0$.
   **A:** If $f(\mu + d) = f(\mu - d)$ for all $d$, then $E[(X-\mu)^3] = \int (x-\mu)^3 f(x) dx$. The integrand is odd about $\mu$, so the integral over symmetric limits is zero.

## Practice Problems

### Easy - 5

1. Classify the skewness of: $\{1, 1, 1, 2, 100\}$.

2. For a left-skewed distribution, which is larger: mean or median?

3. What is the Pearson median skewness if $\bar{x} = 50$, median $= 55$, $s = 10$?

4. A distribution has $\gamma_1 = 0$. Is it necessarily normal?

5. The cube of a negative number is __________. (positive/negative)

### Medium - 5

1. Compute $\gamma_1$ for $\{2, 4, 6, 8, 20\}$.

2. Explain the relationship: positive skew $\implies$ mean $>$ median $>$ mode.

3. Log-transform $X = \{1, 2, 5, 10, 100\}$ and compute the skewness before and after.

4. A dataset has $\bar{x} = 100$, median $= 90$, $s = 20$. Compute and interpret the Pearson median skewness.

5. True or false: Multiplying all values by $-1$ changes negative skewness to positive. Explain.

### Hard - 3

1. Derive the formula for the skewness of an exponential distribution with rate $\lambda$.

2. Prove that the sample skewness $g_1$ is a biased but consistent estimator of $\gamma_1$.

3. Given two distributions $X$ and $Y$ with known skewness, derive the skewness of $X + Y$ under independence.

## Solutions

**Easy:**

1. Right-skewed (positive skew). The outlier 100 creates a long right tail.

2. Mean $<$ Median for left-skewed data.

3. $3(50-55)/10 = 3(-5)/10 = -15/10 = -1.5$. Negative skew.

4. No. Zero skewness is necessary but not sufficient. A uniform distribution has zero skewness but is not normal.

5. Negative.

**Medium:**

1. $\mu = 8$. $\sigma^2 = ((2-8)^2+(4-8)^2+(6-8)^2+(8-8)^2+(20-8)^2)/5 = (36+16+4+0+144)/5 = 200/5 = 40$. $\sigma = \sqrt{40} \approx 6.325$. Cubed normalised deviations: $((-6)/6.325)^3 = -0.85$, $((-4)/6.325)^3 = -0.25$, $((-2)/6.325)^3 = -0.03$, $0$, $((12)/6.325)^3 = 6.83$. Sum $= 5.70$. $\gamma_1 = 5.70/5 = 1.14$. Positive skew.

2. For right-skewed data, the right tail pulls the mean right of the median. The median is less affected. The mode (peak) is at the left.

3. Before: compute $\gamma_1$ (will be positive). After log transform: compute $\gamma_1$ (should be closer to 0).

4. $3(100-90)/20 = 3(10)/20 = 30/20 = 1.5$. Positive skewness (right-skewed).

5. True. Negating flips the sign because $((-x) - (-\mu))^3 = (-(x-\mu))^3 = -(x-\mu)^3$.

**Hard:**

1. For exponential $f(x) = \lambda e^{-\lambda x}$, $x \geq 0$: $E[X] = 1/\lambda$, $E[X^2] = 2/\lambda^2$, $E[X^3] = 6/\lambda^3$. $\mu = 1/\lambda$, $\sigma^2 = 1/\lambda^2$, $E[(X-\mu)^3] = E[X^3] - 3\mu E[X^2] + 2\mu^3 = 6/\lambda^3 - 3(1/\lambda)(2/\lambda^2) + 2/\lambda^3 = 6/\lambda^3 - 6/\lambda^3 + 2/\lambda^3 = 2/\lambda^3$. $\gamma_1 = (2/\lambda^3) / (1/\lambda^3) = 2$.

2. $E[g_1] \neq \gamma_1$ for finite $n$ (bias), but as $n \to \infty$, $g_1 \xrightarrow{p} \gamma_1$ (consistency) by the law of large numbers and Slutsky's theorem, provided the sixth moment is finite.

3. For independent $X$ and $Y$: $E[(X+Y - (\mu_X+\mu_Y))^3] = E[(X-\mu_X)^3] + E[(Y-\mu_Y)^3]$ because cross-terds vanish. So $\gamma_1(X+Y) = (\gamma_1(X)\sigma_X^3 + \gamma_1(Y)\sigma_Y^3) / (\sigma_X^2 + \sigma_Y^2)^{3/2}$.

## Related Concepts

- Mean (MATH-077) — first moment, affected by skewness
- Median (MATH-078) — robust central measure for skewed data
- Moment (statistics) — general framework including skewness
- Kurtosis (MATH-085) — fourth moment, tail weight
- Normal Distribution — zero skewness baseline
- Box-Cox Transformation — reduces skewness
- Quantile-Quantile (Q-Q) Plot — visual skewness assessment

## Next Concepts

- Kurtosis (MATH-085) — tail heaviness
- Confidence Interval (MATH-086)
- Hypothesis Testing (MATH-087)

## Summary

Skewness measures the asymmetry of a distribution. Positive skewness (right-skewed) has a longer right tail; negative skewness (left-skewed) has a longer left tail. The normal distribution has zero skewness. Skewness is the third standardised moment. In AI/ML, detecting and correcting skewness is essential for preprocessing, model selection, and assumption checking. Transformations like Box-Cox and log transforms reduce skewness, improving model performance.

## Key Takeaways

- Skewness $\gamma_1 = E[((X-\mu)/\sigma)^3]$ measures asymmetry.
- Positive skew: right tail longer, mean $>$ median.
- Negative skew: left tail longer, mean $<$ median.
- Zero skewness is necessary but not sufficient for normality.
- Sample skewness $g_1$ uses correction factor for bias.
- In ML, check feature skewness; apply Box-Cox or log transforms.
- Skewness affects linear model assumptions and outlier detection.
- $|g_1| < 0.5$: symmetric; $0.5-1$: moderate skew; $>1$: high skew.
