# Concept: Kurtosis

## Concept ID

MATH-085

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define kurtosis as the fourth standardised moment
- Distinguish between leptokurtic, mesokurtic, and platykurtic distributions
- Interpret excess kurtosis relative to the normal distribution
- Apply kurtosis for outlier detection and tail risk assessment
- Understand the role of kurtosis in AI/ML contexts

## Prerequisites

- Skewness (MATH-084)
- Variance (MATH-080)
- Standard Deviation (MATH-081)
- Understanding of moments and normal distribution

## Definition

**Kurtosis** is a measure of the "tailedness" of a probability distribution. It quantifies how much mass is in the tails compared to the normal distribution. The kurtosis of any univariate normal distribution is 3.

**Excess kurtosis** is kurtosis relative to the normal distribution:
$$
\text{Excess Kurtosis} = \text{Kurtosis} - 3
$$

**Population kurtosis:**
$$
\gamma_2 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \mu}{\sigma}\right)^4
$$

**Excess kurtosis:**
$$
\text{Kurtosis}_{\text{excess}} = \gamma_2 - 3
$$

**Sample excess kurtosis (adjusted):**
$$
g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}
$$

## Intuition

Kurtosis tells you about the weight of the tails and the peakedness of the distribution. A high kurtosis distribution has heavy tails (more extreme outliers) and a sharp peak. A low kurtosis distribution has light tails (fewer outliers) and a flatter peak.

Think of kurtosis as an outlier detector: distributions with high kurtosis produce more extreme outliers than the normal distribution.

- **Leptokurtic (excess kurtosis $> 0$):** Heavy tails, sharp peak. More outliers than normal.
- **Mesokurtic (excess kurtosis $= 0$):** Medium tails, normal-like. The normal distribution is the classic example.
- **Platykurtic (excess kurtosis $< 0$):** Light tails, flat peak. Fewer and less extreme outliers than normal.

## Why This Concept Matters

Kurtosis is important for:

- **Risk management:** High kurtosis means higher probability of extreme events (tail risk). This is critical in finance, insurance, and cybersecurity.
- **Outlier detection:** High kurtosis suggests the data has many outliers relative to a normal distribution.
- **Model selection:** Some models are sensitive to kurtosis (e.g., linear models with normality assumptions).
- **Hypothesis testing:** Tests like the Jarque-Bera test use kurtosis (and skewness) to check normality.
- **Portfolio theory:** Investors prefer positive skewness and prefer low kurtosis (fewer extreme negative returns).

## Historical Background

The term "kurtosis" was introduced by Karl Pearson in 1905 from the Greek word "kyrtos" meaning "curved" or "bulging." Pearson classified distributions based on their shape relative to the normal distribution.

The concept was further developed by R.A. Fisher in the 1920s and 1930s, who derived the sampling distribution of kurtosis and developed the adjusted sample kurtosis formula.

The Jarque-Bera test, which jointly tests skewness and kurtosis for normality, was developed by Carlos Jarque and Anil Bera in 1980.

## Real World Examples

**Finance:** Stock returns often exhibit excess kurtosis (leptokurtic). Most daily returns are small, but crashes and rallies produce extreme returns (black swan events). The 2008 financial crisis is a classic example of tail risk predicted by high kurtosis.

**Insurance:** Claim sizes are leptokurtic. Most claims are small, but catastrophic events (hurricanes, earthquakes) produce extremely large claims that are far more common than a normal distribution would predict.

**Manufacturing:** Quality control metrics are often platykurtic when the process is well-controlled -- deviations cluster within a narrow range with few extreme values.

**Biology:** Human height has approximately mesokurtic distribution (close to normal).

**Cybersecurity:** Network traffic features often have high kurtosis -- most traffic is normal, but there are extreme spikes during attacks.

## AI/ML Relevance

**Outlier detection:** High kurtosis indicates that extreme values are more common than expected under normality. This is used in anomaly detection systems -- if a feature shows high kurtosis, the 3-sigma rule will miss many true outliers.

**Tail risk in financial ML:** In quantitative finance, models must account for the heavy tails (high kurtosis) of asset returns. Gaussian assumption underestimates crash probability. GARCH models and extreme value theory explicitly model this.

**Heavy-tailed gradient noise:** In deep learning, the distribution of stochastic gradients often has heavy tails (high kurtosis). This affects optimisation -- heavy-tailed gradient noise can cause large parameter updates, necessitating gradient clipping or adaptive methods like Adam.

**Normality testing:** The Jarque-Bera test statistic combines skewness and kurtosis:
$$
JB = \frac{n}{6}\left(g_1^2 + \frac{(g_2)^2}{4}\right)
$$
Under the null hypothesis of normality, JB follows a $\chi^2_2$ distribution.

**Feature engineering:** High kurtosis in features may indicate the need for transformations (power transforms, winsorisation) to make the data more suitable for linear models.

**Robust statistics:** When kurtosis is high, robust estimators (median, MAD) are preferred over mean and standard deviation.

**Bayesian methods:** The choice of prior distributions is influenced by kurtosis. Laplace priors (heavy-tailed) are used for sparse models; Gaussian priors (medium-tailed) for regularised models.

## Mathematical Explanation

Kurtosis is the fourth standardised moment:
$$
\gamma_2 = E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] = \frac{E[(X - \mu)^4]}{\sigma^4}
$$

The fourth power heavily weights observations far from the mean. A value 3 standard deviations away contributes $3^4 = 81$ to the kurtosis numerator, compared to $3^2 = 9$ for variance.

**Why subtract 3?** The normal distribution has kurtosis exactly 3. By subtracting 3, we set the normal distribution as the baseline (excess kurtosis = 0). Positive values indicate heavier tails than normal; negative values indicate lighter tails.

**Interpretation:**
- Excess kurtosis $> 0$ (leptokurtic): More outliers than normal. Data has a higher probability of extreme values.
- Excess kurtosis $= 0$ (mesokurtic): Similar tail behaviour to normal.
- Excess kurtosis $< 0$ (platykurtic): Fewer outliers than normal. Data has lower probability of extreme values.

**Relationship to tails (not peakedness):** Modern understanding (Westfall, 2014) emphasises that kurtosis primarily measures tail weight, not peakedness. A distribution can have high kurtosis due to heavy tails, even if the peak is not sharp.

## Formula(s)

**Population kurtosis:**
$$
\gamma_2 = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \mu}{\sigma}\right)^4
$$

**Excess kurtosis:**
$$
\text{Kurtosis}_{\text{excess}} = \gamma_2 - 3
$$

**Sample excess kurtosis:**
$$
g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)}\frac{\sum (x_i - \bar{x})^4}{s^4} - \frac{3(n-1)^2}{(n-2)(n-3)}
$$

**Jarque-Bera statistic:**
$$
JB = \frac{n}{6}\left(g_1^2 + \frac{g_2^2}{4}\right) \sim \chi^2_2
$$

## Properties

- **Lower bound:** For any distribution, excess kurtosis $\geq -2$. The minimum is achieved by a Bernoulli distribution with $p = 1/2$.
- **Scale invariance:** $\text{Kurtosis}(aX + b) = \text{Kurtosis}(X)$ for $a \neq 0$.
- **Not location or scale dependent:** Kurtosis depends only on the shape of the distribution.
- **Sensitivity to outliers:** Kurtosis is highly sensitive to extreme values. A single extreme observation can dramatically inflate kurtosis.
- **High variance:** Sample kurtosis has very high sampling variance, especially for heavy-tailed distributions. Large samples ($n > 100$) are needed for reliable estimation.
- **Relation to tail behaviour:** For distributions with power-law tails $f(x) \sim |x|^{-\alpha-1}$, kurtosis is finite only if $\alpha > 4$.

## Step-by-Step Worked Examples

### Example 1: Computing Kurtosis

**Problem:** Compute the excess kurtosis for $\{1, 2, 3, 4, 15\}$.

**Solution:**

Step 1: Mean.
$$
\mu = \frac{1 + 2 + 3 + 4 + 15}{5} = \frac{25}{5} = 5
$$

Step 2: Variance and SD.
$$
\sigma^2 = \frac{(1-5)^2 + (2-5)^2 + (3-5)^2 + (4-5)^2 + (15-5)^2}{5}
$$
$$
\sigma^2 = \frac{16 + 9 + 4 + 1 + 100}{5} = \frac{130}{5} = 26
$$
$$
\sigma = \sqrt{26} \approx 5.099
$$

Step 3: Fourth powers of standardised deviations.
- $((1-5)/5.099)^4 = (-0.784)^4 = 0.378$
- $((2-5)/5.099)^4 = (-0.588)^4 = 0.120$
- $((3-5)/5.099)^4 = (-0.392)^4 = 0.024$
- $((4-5)/5.099)^4 = (-0.196)^4 = 0.001$
- $((15-5)/5.099)^4 = (1.961)^4 = 14.79$

Step 4: Average.
$$
\gamma_2 = \frac{0.378 + 0.120 + 0.024 + 0.001 + 14.79}{5} = \frac{15.313}{5} = 3.063
$$

Step 5: Excess kurtosis.
$$
\gamma_2 - 3 = 3.063 - 3 = 0.063
$$

This distribution is approximately mesokurtic (slightly leptokurtic).

### Example 2: Leptokurtic vs Platykurtic

**Problem:** Compare the kurtosis of $U = \text{Uniform}(0,1)$ (platykurtic) and $T = t$-distribution with 5 df (leptokurtic).

**Solution:**

For Uniform(0,1):
$$
E[U] = 0.5, \quad \text{Var}(U) = 1/12, \quad \sigma = 1/\sqrt{12}
$$
$$
\gamma_2 = \frac{E[(U - 0.5)^4]}{\sigma^4} = \frac{1/80}{(1/12)^2} = \frac{1/80}{1/144} = \frac{144}{80} = 1.8
$$
Excess kurtosis $= 1.8 - 3 = -1.2$ (platykurtic).

For $t$-distribution with 5 df:
Excess kurtosis $= 6/(\nu - 4) = 6/(5-4) = 6$ (extremely leptokurtic).

### Example 3: Computing Sample Excess Kurtosis

**Problem:** Compute the adjusted sample excess kurtosis for $\{2, 4, 6, 8\}$.

**Solution:**

Step 1: $\bar{x} = 5$.

Step 2: $s^2 = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2)/(4-1) = (9+1+1+9)/3 = 20/3$.
$s = \sqrt{20/3} \approx 2.582$.

Step 3: Standardised fourth powers.
- $((2-5)/2.582)^4 = (-1.162)^4 = 1.822$
- $((4-5)/2.582)^4 = (-0.387)^4 = 0.022$
- $((6-5)/2.582)^4 = (0.387)^4 = 0.022$
- $((8-5)/2.582)^4 = (1.162)^4 = 1.822$

Sum $= 3.688$.

Step 4: Apply adjustment.
$$
\frac{n(n+1)}{(n-1)(n-2)(n-3)} \times \text{sum} = \frac{4 \cdot 5}{3 \cdot 2 \cdot 1} \times 3.688 = \frac{20}{6} \times 3.688 = 12.293
$$

Step 5: Subtract second term.
$$
\frac{3(n-1)^2}{(n-2)(n-3)} = \frac{3 \cdot 9}{2 \cdot 1} = \frac{27}{2} = 13.5
$$
$$
g_2 = 12.293 - 13.5 = -1.207
$$

Negative excess kurtosis (platykurtic) for this small symmetric dataset.

## Visual Interpretation

On a density plot, kurtosis differences appear as:

- **Leptokurtic:** Tall, sharp peak with heavier tails (more area in the tails).
- **Platykurtic:** Flatter, broader peak with lighter tails (less area in the tails).
- **Mesokurtic:** Moderate peak and medium tails (normal-like).

A Q-Q plot shows kurtosis as deviations in the tails. Leptokurtic data has points deviating from the line in both tails (S-shaped). Platykurtic data has points closer to zero in the tails than expected.

In a box plot, leptokurtic data shows many more outliers (points beyond the whiskers) than expected.

## Common Mistakes

1. **Confusing kurtosis with peakedness:** Modern statistics emphasises that kurtosis primarily measures tail weight, not peakedness. A distribution can have a sharp peak but low kurtosis if the tails are light.

2. **Forgetting to subtract 3:** Most software reports excess kurtosis (kurtosis minus 3). Using raw kurtosis when excess kurtosis is expected (or vice versa) causes confusion.

3. **Interpreting kurtosis for small samples:** Sample kurtosis has high variance for small $n$. A small sample from a normal distribution can show positive or negative excess kurtosis by chance.

4. **Assuming high kurtosis means bimodal:** Bimodality does not necessarily imply high or low kurtosis. Kurtosis measures tails, not modality.

5. **Ignoring kurtosis in risk models:** Using only variance to measure risk ignores tail behaviour. Two portfolios can have the same variance but very different kurtosis.

6. **Using unadjusted sample kurtosis for small samples:** Raw sample kurtosis is biased. Use the adjusted formula $g_2$ for accurate estimation.

7. **Assuming platykurtic means "no outliers":** Platykurtic distributions have fewer outliers than normal, but can still have some outliers.

## Interview Questions

### Beginner - 5

1. **Q:** What is kurtosis?
   **A:** Kurtosis measures the tail heaviness of a distribution. It tells you how outlier-prone the distribution is.

2. **Q:** What is the kurtosis of a normal distribution?
   **A:** The raw kurtosis is 3. The excess kurtosis is 0.

3. **Q:** What does positive excess kurtosis mean?
   **A:** The distribution has heavier tails than normal (leptokurtic), meaning more extreme outliers.

4. **Q:** What does negative excess kurtosis mean?
   **A:** The distribution has lighter tails than normal (platykurtic), meaning fewer extreme outliers.

5. **Q:** What is the difference between mesokurtic, leptokurtic, and platykurtic?
   **A:** Mesokurtic (excess = 0, normal-like), leptokurtic (excess > 0, heavy tails), platykurtic (excess < 0, light tails).

### Intermediate - 5

1. **Q:** Why do we subtract 3 from kurtosis?
   **A:** To set the normal distribution as the baseline. Excess kurtosis = raw kurtosis - 3 makes interpretation relative to normal.

2. **Q:** How is kurtosis used in the Jarque-Bera test?
   **A:** The Jarque-Bera statistic combines skewness and kurtosis: $JB = (n/6)(g_1^2 + g_2^2/4) \sim \chi^2_2$. Large values reject normality.

3. **Q:** How does kurtosis affect financial risk models?
   **A:** High kurtosis means tail risk is higher than Gaussian models predict. Value-at-Risk (VaR) and expected shortfall must account for this.

4. **Q:** What is the minimum possible excess kurtosis?
   **A:** For any distribution, excess kurtosis $\geq -2$, achieved by Bernoulli(0.5) with equal probability on two values.

5. **Q:** How does kurtosis relate to outlier detection in ML?
   **A:** Features with high kurtosis produce outliers more frequently than normal. Standard outlier detection (3-sigma) will miss true outliers in leptokurtic data.

### Advanced - 3

1. **Q:** Prove that for any distribution, $\gamma_2 \geq \gamma_1^2 + 1$.
   **A:** By the Cauchy-Schwarz inequality applied to $Z = (X-\mu)/\sigma$ and $Z^2$: $(E[Z^3])^2 \leq E[Z^2]E[Z^4] = 1 \cdot \gamma_2$. Hence $\gamma_2 \geq \gamma_1^2$.

2. **Q:** Derive the kurtosis of a $t$-distribution and explain its behaviour as degrees of freedom vary.
   **A:** For $t_\nu$ with $\nu > 4$: excess kurtosis $= 6/(\nu-4)$. As $\nu \to \infty$, kurtosis $\to 0$ (approaches normal). For $\nu \leq 4$, kurtosis is infinite (fourth moment does not exist).

3. **Q:** Discuss the role of kurtosis in Bayesian neural network posteriors.
   **A:** The posterior over weights can have high kurtosis, indicating that many weights are near zero (Dirac-like) with occasional large values. This motivates Laplace approximation (Gaussian) versus more flexible approximations.

## Practice Problems

### Easy - 5

1. A distribution has raw kurtosis 5. What is its excess kurtosis?

2. Classify: Uniform distribution (excess kurtosis = -1.2).

3. If $\gamma_2 = 3$, what is the distribution called?

4. Compute $\gamma_2$ for $\{5, 5, 5, 5\}$.

5. True or false: High kurtosis means more outliers.

### Medium - 5

1. Compute excess kurtosis for $\{1, 1, 2, 3, 10\}$.

2. Explain why the $t$-distribution with 3 df has infinite kurtosis.

3. What is the Jarque-Bera test for normality?

4. Compare the kurtosis of Normal(0,1) and Logistic(0,1).

5. Show that excess kurtosis $\geq -2$ with the Bernoulli distribution.

### Hard - 3

1. Prove that the sample excess kurtosis $g_2$ is a consistent estimator of the population excess kurtosis.

2. Derive the relationship between kurtosis and the variance of the sample variance.

3. Show that for a mixture of two normal distributions with different variances, kurtosis can be greater than 3 even if each component has kurtosis 3.

## Solutions

**Easy:**

1. Excess kurtosis $= 5 - 3 = 2$ (leptokurtic).

2. Platykurtic (excess kurtosis $< 0$).

3. Mesokurtic (same as normal).

4. All values equal $\mu = 5$. $\sigma = 0$. Kurtosis is undefined (division by zero). The distribution has no spread.

5. True. Higher kurtosis means heavier tails and more extreme outliers relative to normal.

**Medium:**

1. $\mu = 17/5 = 3.4$. $\sigma^2 = ((1-3.4)^2+(1-3.4)^2+(2-3.4)^2+(3-3.4)^2+(10-3.4)^2)/5 = (5.76+5.76+1.96+0.16+43.56)/5 = 57.2/5 = 11.44$. $\sigma \approx 3.382$. Fourth powers: $((-2.4/3.382)^4 + (-2.4/3.382)^4 + (-1.4/3.382)^4 + (-0.4/3.382)^4 + (6.6/3.382)^4)/5$. Approx: $(0.254+0.254+0.029+0.0002+14.48)/5 = 15.017/5 = 3.003$. Excess = $3.003 - 3 = 0.003$.

2. The $t$-distribution with $\nu$ df has finite $k$-th moment only if $k < \nu$. For $\nu = 3$, the fourth moment is infinite, so kurtosis is undefined (infinite).

3. The Jarque-Bera test jointly tests null hypothesis of normality. $JB = (n/6)(g_1^2 + g_2^2/4)$ under $H_0$ follows $\chi^2_2$. Large values reject normality.

4. Normal(0,1): excess kurtosis = 0. Logistic(0,1): excess kurtosis = 1.2. Logistic has heavier tails than normal.

5. For Bernoulli($p$): $P(X=1)=p$, $P(X=0)=1-p$. $\mu = p$, $\sigma^2 = p(1-p)$. $\gamma_2 = ((1-p)^4 p + p^4(1-p))/(p(1-p))^2$. The minimum is at $p=0.5$, giving $\gamma_2 = 1$, excess = $-2$.

**Hard:**

1. $g_2$ is a function of sample moments. By the law of large numbers, $\frac{1}{n}\sum (x_i - \bar{x})^k \xrightarrow{p} E[(X-\mu)^k]$ for $k = 2,4$ provided moments exist. Then $g_2 \xrightarrow{p} \gamma_2 - 3$ by Slutsky's theorem.

2. $\text{Var}(s^2) = \frac{1}{n}(\mu_4 - \frac{n-3}{n-1}\mu_2^2)$ where $\mu_k = E[(X-\mu)^k]$. In terms of kurtosis: $\text{Var}(s^2) = \frac{\sigma^4}{n}(\gamma_2 + \frac{2n}{n-1})$. Higher kurtosis increases the variance of $s^2$.

3. Mixture $p N(\mu_1, \sigma_1^2) + (1-p) N(\mu_2, \sigma_2^2)$. If $\sigma_1 \neq \sigma_2$, the mixture creates heavier tails than either component. The fourth moment of the mixture exceeds the weighted average of the fourth moments, producing excess kurtosis $> 0$.

## Related Concepts

- Skewness (MATH-084) — third moment, asymmetry
- Moment (statistics) — general framework
- Normal Distribution — baseline (kurtosis = 3)
- Jarque-Bera Test — joint normality test
- Heavy-tailed Distributions — distributions with high kurtosis
- Tail Risk — risk of extreme events

## Next Concepts

- Confidence Interval (MATH-086) — using moments for inference
- Hypothesis Testing (MATH-087) — testing distributional assumptions

## Summary

Kurtosis measures the tail heaviness of a distribution relative to the normal distribution. Excess kurtosis $> 0$ indicates leptokurtic (heavy-tailed), $= 0$ indicates mesokurtic (normal-like), and $< 0$ indicates platykurtic (light-tailed). Kurtosis is the fourth standardised moment and is highly sensitive to outliers. In AI/ML, kurtosis is used for outlier detection, normality testing, financial risk modelling, and understanding gradient noise. The sample kurtosis requires adjustment for bias and has high variance, requiring large samples for reliable estimation.

## Key Takeaways

- Kurtosis $\gamma_2 = E[((X-\mu)/\sigma)^4]$ measures tail weight.
- Excess kurtosis $= \gamma_2 - 3$ (normal baseline = 0).
- Leptokurtic (excess $> 0$): heavy tails, more outliers.
- Mesokurtic (excess $= 0$): normal-like tails.
- Platykurtic (excess $< 0$): light tails, fewer outliers.
- Kurtosis is scale and location invariant.
- Jarque-Bera test combines skewness and kurtosis for normality.
- High kurtosis indicates tail risk in finance.
- Sample kurtosis has high variance; use adjustment and large samples.
- Kurtosis primarily measures tails, not peakedness.
