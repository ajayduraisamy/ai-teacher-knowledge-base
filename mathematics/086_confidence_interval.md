# Concept: Confidence Interval

## Concept ID

MATH-086

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define and interpret confidence intervals for the population mean
- Distinguish between $z$-intervals and $t$-intervals
- Understand how confidence level and sample size affect interval width
- Apply confidence intervals in AI/ML for model evaluation uncertainty
- Explain the frequentist interpretation of confidence intervals

## Prerequisites

- Mean (MATH-077)
- Variance (MATH-080)
- Standard Deviation (MATH-081)
- Normal Distribution
- Sampling distributions
- Central Limit Theorem

## Definition

A **confidence interval (CI)** is a range of values, computed from sample data, that is likely to contain the true population parameter with a specified level of confidence. It provides an estimate of uncertainty around a point estimate.

**Confidence interval for the mean (known $\sigma$):**
$$
\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
$$

**Confidence interval for the mean (unknown $\sigma$, using $t$-distribution):**
$$
\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
$$

where $\alpha = 1 - \text{confidence level}$, $z_{\alpha/2}$ is the standard normal critical value, and $t_{\alpha/2, n-1}$ is the $t$-distribution critical value with $n-1$ degrees of freedom.

## Intuition

Imagine you are trying to estimate the average height of all people in a country. You measure 100 people and find the sample mean is 170 cm. You know that if you repeated this sampling many times, the sample means would vary around the true population mean. A confidence interval captures this sampling variability.

A 95% confidence interval $[168, 172]$ means: "If we repeated this sampling process many times and computed a confidence interval each time, approximately 95% of those intervals would contain the true population mean."

Crucially, the confidence is in the procedure, not in any single interval. It is incorrect to say "there is a 95% probability the true mean lies in this interval." The interval either contains the true mean or it does not. The 95% refers to the long-run success rate of the method.

## Why This Concept Matters

Confidence intervals are fundamental to inferential statistics. They provide:

- **Uncertainty quantification:** A point estimate alone (e.g., $\bar{x} = 170$) is rarely useful without knowing its precision.
- **Hypothesis testing equivalence:** A 95% CI that does not contain a null value is equivalent to rejecting $H_0$ at $\alpha = 0.05$.
- **Clinical significance:** CIs show the range of plausible effect sizes, not just whether an effect exists.
- **Sample size planning:** CIs help determine how many samples are needed for a desired precision.
- **Meta-analysis:** CIs are the building blocks for combining results across studies.

## Historical Background

The concept of confidence intervals was introduced by Jerzy Neyman in 1937, building on earlier work by Ronald Fisher (who developed fiducial inference) and William Gosset (Student's $t$-distribution, 1908).

Neyman's approach was purely frequentist: the confidence interval is a procedure that, when repeated, captures the true parameter with a given frequency. This contrasted with Fisher's fiducial approach and Bayesian credible intervals.

The $t$-distribution was developed by Gosset (under the pseudonym "Student") while working at Guinness Brewery. He needed to make inferences about beer quality from small samples, leading to the development of the $t$-test and $t$-interval.

## Real World Examples

**Clinical trials:** A 95% CI for the difference in recovery rates between a new drug and placebo might be $[0.05, 0.15]$, suggesting the drug improves recovery by 5% to 15%.

**Election polling:** A poll might report that 52% of voters support Candidate A with a 95% CI of $[49\%, 55\%]$. This indicates the race is too close to call.

**Quality control:** A factory measures the mean diameter of ball bearings. A 99% CI of $[9.98, 10.02]$ mm confirms the process is within specification.

**Education:** A study estimates that a new teaching method improves test scores by 8 points, 95% CI $[3, 13]$. The interval shows the improvement is meaningful but the precise magnitude is uncertain.

**Environmental science:** An estimate of mean global temperature increase is $1.2^\circ$C, 95% CI $[1.0, 1.4]^\circ$C.

## AI/ML Relevance

**Confidence intervals for model accuracy:** When evaluating a classifier on a test set, accuracy is a point estimate. A confidence interval provides the plausible range of the true accuracy:
$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$
where $\hat{p}$ is the observed accuracy and $n$ is the test set size.

**Prediction intervals vs confidence intervals:** A prediction interval for a new observation is wider than a confidence interval for the mean because it includes individual-level variability:
$$
\hat{y} \pm t_{\alpha/2, n-2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum (x_i - \bar{x})^2}}
$$

**Bootstrap confidence intervals:** For complex models (decision trees, neural networks), analytical CIs are rarely available. The bootstrap method resamples the data many times to estimate the sampling distribution of a statistic:
$$
\text{CI}_{\text{bootstrap}} = [\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]
$$
where $\hat{\theta}_{(k)}$ is the $k$-th percentile of bootstrap estimates.

**Bayesian credible intervals:** In Bayesian ML, the posterior distribution directly provides credible intervals (the Bayesian analogue of confidence intervals). A 95% credible interval is an interval containing 95% of the posterior probability mass.

**A/B testing in ML:** When comparing two models, CIs for the difference in metrics (accuracy, AUC, MSE) determine if the difference is practically significant.

**Uncertainty in deep learning:** Monte Carlo dropout and ensemble methods produce distributions of predictions from which CIs can be derived.

## Mathematical Explanation

The confidence interval for the mean is derived from the sampling distribution of $\bar{x}$.

**Known $\sigma$:** By the Central Limit Theorem, for large $n$:
$$
\frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)
$$

Rearranging:
$$
P\left(-z_{\alpha/2} \leq \frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \leq z_{\alpha/2}\right) = 1 - \alpha
$$
$$
P\left(\bar{x} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{x} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) = 1 - \alpha
$$

**Unknown $\sigma$:** When $\sigma$ is estimated by $s$, we use the $t$-distribution:
$$
\frac{\bar{x} - \mu}{s/\sqrt{n}} \sim t_{n-1}
$$

The $t$-distribution has heavier tails than the normal, producing wider intervals. As $n \to \infty$, $t_{n-1} \to N(0,1)$.

**Factors affecting CI width:**
- **Confidence level:** Higher confidence (99% vs 95%) increases width.
- **Sample size:** Larger $n$ decreases width (width $\propto 1/\sqrt{n}$).
- **Variability:** Larger $\sigma$ or $s$ increases width.
- **Critical value:** $t$-intervals are wider than $z$-intervals for small $n$.

## Formula(s)

**CI for mean (known $\sigma$):**
$$
\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
$$

**CI for mean (unknown $\sigma$):**
$$
\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}
$$

**CI for proportion:**
$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

**CI for variance:**
$$
\left[\frac{(n-1)s^2}{\chi^2_{1-\alpha/2, n-1}}, \frac{(n-1)s^2}{\chi^2_{\alpha/2, n-1}}\right]
$$

**CI for difference in means (two independent samples):**
$$
(\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, \text{df}} \cdot s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

**Prediction interval for a new observation:**
$$
\hat{y} \pm t_{\alpha/2, n-2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum (x_i - \bar{x})^2}}
$$

## Properties

- **Width is proportional to $z_{\alpha/2}$:** Higher confidence $\implies$ larger $z_{\alpha/2}$ $\implies$ wider interval.
- **Width is proportional to $1/\sqrt{n}$:** Quadrupling the sample size halves the width.
- **Width is proportional to $\sigma$:** More variable data produces wider intervals.
- **Exactness for normal data:** $t$-intervals are exact when data is normal. For non-normal data, they are approximate and rely on CLT for large $n$.
- **The interval is random, the parameter is fixed:** In frequentist statistics, the parameter $\mu$ is fixed; the interval boundaries are random.
- **Duality with hypothesis testing:** A $100(1-\alpha)\%$ CI contains all values of $\mu$ that would not be rejected at level $\alpha$.

## Step-by-Step Worked Examples

### Example 1: $z$-Interval (Known $\sigma$)

**Problem:** A sample of $n = 100$ students has mean height $\bar{x} = 170$ cm. The population standard deviation is known to be $\sigma = 15$ cm. Construct a 95% CI for the population mean.

**Solution:**

Step 1: Identify parameters.
$n = 100$, $\bar{x} = 170$, $\sigma = 15$, $\alpha = 0.05$.

Step 2: Find the critical value.
For 95% confidence, $z_{0.025} = 1.96$.

Step 3: Compute the margin of error.
$$
ME = z_{\alpha/2} \frac{\sigma}{\sqrt{n}} = 1.96 \times \frac{15}{\sqrt{100}} = 1.96 \times 1.5 = 2.94
$$

Step 4: Construct the CI.
$$
CI = 170 \pm 2.94 = [167.06, 172.94]
$$

Interpretation: We are 95% confident that the true population mean height lies between 167.06 cm and 172.94 cm.

### Example 2: $t$-Interval (Unknown $\sigma$)

**Problem:** A sample of $n = 16$ batteries has mean life $\bar{x} = 500$ hours and sample standard deviation $s = 40$ hours. Construct a 99% CI for the mean battery life.

**Solution:**

Step 1: Parameters.
$n = 16$, $\bar{x} = 500$, $s = 40$, $\alpha = 0.01$.

Step 2: Find the $t$ critical value.
$df = n - 1 = 15$. For 99% confidence, $\alpha/2 = 0.005$.
$t_{0.005, 15} = 2.947$ (from $t$-table).

Step 3: Margin of error.
$$
ME = t_{\alpha/2, 15} \cdot \frac{s}{\sqrt{n}} = 2.947 \times \frac{40}{\sqrt{16}} = 2.947 \times 10 = 29.47
$$

Step 4: CI.
$$
CI = 500 \pm 29.47 = [470.53, 529.47]
$$

### Example 3: CI for a Proportion

**Problem:** In a survey of $n = 400$ voters, 220 support Candidate A. Construct a 90% CI for the true proportion of supporters.

**Solution:**

Step 1: Sample proportion.
$\hat{p} = 220/400 = 0.55$.

Step 2: Critical value.
For 90% confidence, $z_{0.05} = 1.645$.

Step 3: Margin of error.
$$
ME = 1.645 \times \sqrt{\frac{0.55 \times 0.45}{400}} = 1.645 \times \sqrt{0.00061875} = 1.645 \times 0.02487 = 0.0409
$$

Step 4: CI.
$$
CI = 0.55 \pm 0.0409 = [0.5091, 0.5909]
$$

We are 90% confident that the true support lies between 50.9% and 59.1%.

## Visual Interpretation

A confidence interval can be visualised as horizontal bars on a plot. Imagine 100 different samples from the same population, each with its own 95% CI. Approximately 95 of those intervals will contain the true population mean (shown as a vertical line), and about 5 will miss it.

On a forest plot (common in meta-analysis), each study's effect size is shown with its CI. Studies with narrow CIs are more precise. The diamond at the bottom shows the combined estimate.

For regression, confidence bands around the regression line show the uncertainty of the predicted mean. These bands are narrower near $\bar{x}$ and widen at the extremes.

## Common Mistakes

1. **The probabilistic misinterpretation:** Saying "there is a 95% probability that the true mean lies in this interval" is incorrect. The interval either contains $\mu$ or does not. The 95% refers to the procedure's success rate over repeated sampling.

2. **Confusing confidence interval with prediction interval:** A CI for the mean is narrower than a PI for a new observation. Using a CI when a PI is needed underestimates uncertainty.

3. **Using $z$ instead of $t$ for small samples:** When $\sigma$ is unknown and $n$ is small, the $t$-distribution is required. Using $z$ produces intervals that are too narrow.

4. **Ignoring assumptions:** $t$-intervals assume the data (or sampling distribution) is approximately normal. For small $n$ with non-normal data, the CI may be unreliable.

5. **Overlapping CIs does not mean non-significance:** Two CIs can overlap substantially while a test still finds a significant difference. Use the CI for the difference directly.

6. **Confusing confidence level with precision:** A 99% CI is wider (less precise) than a 95% CI. Higher confidence does not mean better.

7. **Applying CIs to non-random samples:** Confidence intervals require probability sampling. Convenience samples do not produce valid CIs.

## Interview Questions

### Beginner - 5

1. **Q:** What is a confidence interval?
   **A:** A range of values likely to contain the true population parameter, computed from sample data with a specified confidence level.

2. **Q:** What does 95% confidence mean?
   **A:** If we repeated the sampling procedure many times, 95% of the computed intervals would contain the true parameter.

3. **Q:** What factors affect the width of a confidence interval?
   **A:** Confidence level (higher = wider), sample size (larger = narrower), and variability (more = wider).

4. **Q:** What is the difference between a $z$-interval and a $t$-interval?
   **A:** $z$-intervals assume known $\sigma$ or large $n$; $t$-intervals use $s$ (sample SD) and are wider, especially for small $n$.

5. **Q:** What happens to CI width when sample size quadruples?
   **A:** The width is halved (width $\propto 1/\sqrt{n}$).

### Intermediate - 5

1. **Q:** How do you interpret a 95% CI for a proportion?
   **A:** We are 95% confident that the true population proportion falls within the interval. About 95% of such intervals from repeated sampling would contain the true proportion.

2. **Q:** What is the relationship between CIs and hypothesis tests?
   **A:** A 95% CI contains all values that would not be rejected by a two-sided test at $\alpha = 0.05$.

3. **Q:** How would you construct a CI for model accuracy?
   **A:** Use the normal approximation for a proportion: $\hat{p} \pm z_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})/n}$, where $\hat{p}$ is accuracy.

4. **Q:** Why is the $t$-distribution used instead of the normal for small samples?
   **A:** The $t$-distribution has heavier tails, accounting for the additional uncertainty from estimating $\sigma$ with $s$.

5. **Q:** How do bootstrap CIs work?
   **A:** Resample data with replacement $B$ times. Compute the statistic for each resample. The CI is the $(\alpha/2, 1-\alpha/2)$ percentiles of the bootstrap distribution.

### Advanced - 3

1. **Q:** Derive the CI for the mean using the CLT and explain when it is exact vs approximate.
   **A:** The CLT gives $\sqrt{n}(\bar{x} - \mu)/\sigma \xrightarrow{d} N(0,1)$. The CI is approximate for non-normal data. It is exact when data is normal and $\sigma$ is known.

2. **Q:** Explain the difference between a frequentist confidence interval and a Bayesian credible interval.
   **A:** A frequentist CI is random (varies by sample) and covers the fixed parameter with a given frequency. A Bayesian credible interval is fixed (given the data) and contains 95% of the posterior probability mass.

3. **Q:** Derive the prediction interval for a new observation and explain why it is wider than the CI for the mean.
   **A:** $\hat{y}_0 \pm t_{\alpha/2}s\sqrt{1 + 1/n + (x_0 - \bar{x})^2/\sum (x_i - \bar{x})^2}$. It is wider because it includes both the uncertainty of the mean estimate and the individual-level variability $\sigma^2$.

## Practice Problems

### Easy - 5

1. Find $z_{0.025}$ for a 95% CI.

2. A 95% CI is $[10, 20]$. What is the margin of error?

3. What is the $t$ critical value for 90% confidence with $df = 10$?

4. If $n = 25$, $\bar{x} = 50$, $s = 10$, compute the standard error.

5. What sample size gives a standard error of 2 when $\sigma = 10$?

### Medium - 5

1. Construct a 95% CI for $\mu$ given $n=36$, $\bar{x}=80$, $\sigma=12$.

2. A 99% CI is $[45, 65]$ with $n=64$, $\bar{x}=55$. Find $s$.

3. Test scores: $n=9$, $\bar{x}=75$, $s=12$. Construct a 90% CI.

4. In a poll of 500 voters, 270 support a policy. Construct a 95% CI for the true proportion.

5. Explain why increasing $n$ from 100 to 400 reduces the CI width by half.

### Hard - 3

1. Derive the CI for the difference in two population means with unequal variances (Welch's $t$-interval).

2. Given bootstrap estimates $\hat{\theta}^*_1, \dots, \hat{\theta}^*_{1000}$, construct a 95% percentile bootstrap CI.

3. Prove that the length of the $z$-interval for the mean is $2z_{\alpha/2}\sigma/\sqrt{n}$ and derive the sample size needed to achieve a desired margin of error $E$.

## Solutions

**Easy:**

1. $z_{0.025} = 1.96$.

2. Margin of error $= (20 - 10)/2 = 5$.

3. $t_{0.05, 10} \approx 1.812$.

4. $SE = s/\sqrt{n} = 10/\sqrt{25} = 10/5 = 2$.

5. $n = (\sigma/SE)^2 = (10/2)^2 = 25$.

**Medium:**

1. $ME = 1.96 \times 12/\sqrt{36} = 1.96 \times 2 = 3.92$. CI $= [80 - 3.92, 80 + 3.92] = [76.08, 83.92]$.

2. $ME = (65-45)/2 = 10$. $ME = z_{0.005} \cdot s/\sqrt{n}$. For 99%, $z_{0.005} \approx 2.576$. $10 = 2.576 \times s/8$. $s = 10 \times 8/2.576 \approx 31.06$.

3. $df=8$, $t_{0.05,8} \approx 1.86$. $ME = 1.86 \times 12/\sqrt{9} = 1.86 \times 4 = 7.44$. CI $= [75-7.44, 75+7.44] = [67.56, 82.44]$.

4. $\hat{p} = 270/500 = 0.54$. $ME = 1.96 \times \sqrt{0.54(0.46)/500} = 1.96 \times \sqrt{0.0004968} \approx 1.96 \times 0.0223 \approx 0.0437$. CI $= [0.4963, 0.5837]$.

5. CI width $\propto 1/\sqrt{n}$. When $n$ quadruples (100 to 400), $\sqrt{n}$ doubles, so width halves.

**Hard:**

1. Welch's $t$-interval: $(\bar{x}_1 - \bar{x}_2) \pm t_{\alpha/2, \nu} \sqrt{s_1^2/n_1 + s_2^2/n_2}$, where $\nu \approx \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1-1) + (s_2^2/n_2)^2/(n_2-1)}$.

2. Sort the 1000 bootstrap estimates: $\hat{\theta}^*_{(1)} \leq \cdots \leq \hat{\theta}^*_{(1000)}$. The 95% percentile CI is $[\hat{\theta}^*_{(25)}, \hat{\theta}^*_{(975)}]$.

3. Length $L = 2z_{\alpha/2}\sigma/\sqrt{n}$. For desired margin of error $E = L/2$: $E = z_{\alpha/2}\sigma/\sqrt{n}$, so $n = (z_{\alpha/2}\sigma/E)^2$.

## Related Concepts

- Hypothesis Testing (MATH-087) — dual to confidence intervals
- Standard Error — building block of CIs
- Central Limit Theorem — justifies normal approximation
- $t$-Distribution — used when $\sigma$ is unknown
- Bootstrap — non-parametric CI construction
- Bayesian Credible Interval — Bayesian analogue
- Prediction Interval — CI for individual observations

## Next Concepts

- Hypothesis Testing (MATH-087) — using intervals for decisions
- Power Analysis — planning studies for desired precision

## Summary

A confidence interval quantifies the uncertainty around a point estimate by providing a range of plausible parameter values. The $100(1-\alpha)\%$ CI for the mean is $\bar{x} \pm z_{\alpha/2}\sigma/\sqrt{n}$ (known $\sigma$) or $\bar{x} \pm t_{\alpha/2}s/\sqrt{n}$ (unknown $\sigma$). The interval width depends on the confidence level, sample size, and variability. The frequentist interpretation is about the procedure, not any single interval. In AI/ML, CIs are used for model evaluation, prediction intervals, bootstrap inference, and A/B testing.

## Key Takeaways

- CI for mean: $\bar{x} \pm \text{critical value} \times \text{standard error}$.
- 95% CI means 95% of such intervals contain the true parameter.
- Use $t$-distribution when $\sigma$ is unknown (most practical cases).
- Width $\propto 1/\sqrt{n}$: quadruple $n$ to halve the width.
- CIs are dual to hypothesis tests.
- Bootstrap CIs are useful when analytical formulas are unavailable.
- CIs quantify estimation uncertainty; PIs quantify prediction uncertainty.
- Higher confidence $\implies$ wider interval (less precise).
