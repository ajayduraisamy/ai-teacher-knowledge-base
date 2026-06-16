# Concept: Hypothesis Testing

## Concept ID

MATH-087

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define null and alternative hypotheses and their roles
- Distinguish between Type I error ($\alpha$) and Type II error ($\beta$)
- Interpret $p$-values correctly
- Apply $t$-tests and $\chi^2$ tests to practical problems
- Use hypothesis testing for A/B testing and model comparison in AI/ML

## Prerequisites

- Confidence Intervals (MATH-086)
- Normal Distribution
- Mean (MATH-077)
- Variance (MATH-080)
- Standard Deviation (MATH-081)

## Definition

**Hypothesis testing** is a statistical procedure that uses sample data to evaluate a claim (hypothesis) about a population parameter. It provides a framework for making decisions under uncertainty.

**Null hypothesis ($H_0$):** The default assumption, typically representing "no effect" or "no difference." It is the hypothesis we seek evidence against.

**Alternative hypothesis ($H_1$ or $H_a$):** The competing claim, representing an effect, difference, or relationship. It is the hypothesis we seek evidence for.

**Test statistic:** A value computed from sample data, used to decide between $H_0$ and $H_1$.

**$p$-value:** The probability of observing a test statistic as extreme as (or more extreme than) the one observed, assuming $H_0$ is true.

**Significance level ($\alpha$):** The threshold for rejecting $H_0$, typically set at 0.05. If $p$-value $< \alpha$, reject $H_0$.

## Intuition

Think of a criminal trial. The null hypothesis is "the defendant is innocent." The prosecution presents evidence. The jury decides: is the evidence strong enough to reject innocence beyond a reasonable doubt?

- **Type I error:** Convicting an innocent person (false positive).
- **Type II error:** Acquitting a guilty person (false negative).

In hypothesis testing, we assume $H_0$ is true and ask: "If $H_0$ were true, how likely would we observe this extreme data?" If the answer is "very unlikely" ($p < \alpha$), we reject $H_0$.

## Why This Concept Matters

Hypothesis testing is the backbone of scientific inference:

- **Scientific discovery:** Does a new drug work? Does a teaching method improve learning? Hypothesis testing provides evidence.
- **Quality control:** Is a manufacturing process within specification?
- **A/B testing:** Does a new website design increase conversion rates?
- **Feature selection:** Is a feature significantly related to the target?
- **Model comparison:** Is one ML model significantly better than another?

Without hypothesis testing, we cannot distinguish genuine effects from random noise.

## Historical Background

The modern framework of hypothesis testing was developed by:

- **Ronald Fisher (1925):** Introduced the $p$-value and the concept of significance testing in "Statistical Methods for Research Workers." He proposed $\alpha = 0.05$ as a conventional threshold.

- **Jerzy Neyman and Egon Pearson (1933):** Formalised the framework with $H_0$ vs $H_1$, Type I and Type II errors, and the concept of power. They emphasised the decision-theoretic aspect.

- **William Gosset (Student, 1908):** Developed the $t$-test for small-sample inference while working at Guinness Brewery.

- **Karl Pearson (1900):** Developed the $\chi^2$ goodness-of-fit test.

The Fisher vs Neyman-Pearson debate (significance testing vs hypothesis testing) shaped modern statistics. Current practice combines elements from both approaches.

## Real World Examples

**Medicine:** A clinical trial tests whether a new drug reduces blood pressure more than a placebo. $H_0$: no difference. $H_1$: the drug is effective. If $p < 0.05$, the drug is declared effective.

**Manufacturing:** A factory tests whether a new production method reduces defect rates. $H_0$: defect rate unchanged. $H_1$: defect rate decreased.

**Marketing:** An e-commerce site runs an A/B test. $H_0$: no difference in conversion rates. $H_1$: the new design has higher conversion. If $p < 0.05$, they adopt the new design.

**Education:** A school tests whether a new curriculum improves test scores. $H_0$: no improvement. $H_1$: scores are higher with the new curriculum.

**Agriculture:** A farmer tests whether a new fertiliser increases crop yield. $H_0$: no effect. $H_1$: fertiliser increases yield.

## AI/ML Relevance

**A/B testing for model comparison:** When comparing two ML models (e.g., baseline vs proposed), we test:
$$
H_0: \mu_{\text{baseline}} = \mu_{\text{proposed}} \quad \text{vs} \quad H_1: \mu_{\text{proposed}} > \mu_{\text{baseline}}
$$
where $\mu$ is the metric of interest (accuracy, AUC, MSE). A paired $t$-test or McNemar's test is used.

**Statistical significance of improvements:** A 0.5% accuracy improvement may be statistically significant (if $n$ is large) but practically meaningless. Always consider effect size alongside $p$-value.

**Feature significance tests:** In linear regression, the $t$-test tests whether each coefficient is significantly different from zero:
$$
H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \neq 0
$$
The $F$-test tests whether the overall model explains significant variance.

**Model selection:** The likelihood ratio test compares nested models:
$$
D = -2\log(L_{\text{simple}}/L_{\text{complex}}) \sim \chi^2_{df}
$$
where $df$ is the difference in parameter count.

**Chi-squared test for independence:** Tests whether two categorical variables are independent. Used in feature selection for categorical data.

**Multiple testing correction:** When testing many hypotheses (e.g., thousands of features), the Bonferroni correction ($\alpha/m$) or FDR control (Benjamini-Hochberg) prevents false positives.

## Mathematical Explanation

**General procedure:**

1. State $H_0$ and $H_1$.
2. Choose significance level $\alpha$.
3. Select test statistic and determine its sampling distribution under $H_0$.
4. Compute the test statistic from the data.
5. Compute the $p$-value.
6. If $p < \alpha$, reject $H_0$; otherwise, fail to reject $H_0$.

**One-sample $t$-test:**
$$
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \sim t_{n-1}
$$
Tests whether the population mean differs from a hypothesized value $\mu_0$.

**Two-sample $t$-test (independent):**
$$
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}} \sim t_{n_1+n_2-2}
$$
Tests whether two population means differ.

**Paired $t$-test:**
$$
t = \frac{\bar{d}}{s_d/\sqrt{n}} \sim t_{n-1}
$$
where $d_i = x_{1i} - x_{2i}$ are paired differences.

**Chi-squared test for independence:**
$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \sim \chi^2_{(r-1)(c-1)}
$$

**Error types:**
- **Type I error ($\alpha$):** Reject $H_0$ when $H_0$ is true (false positive).
- **Type II error ($\beta$):** Fail to reject $H_0$ when $H_0$ is false (false negative).
- **Power ($1 - \beta$):** Probability of correctly rejecting $H_0$.

## Formula(s)

**One-sample $t$-test:**
$$
t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \quad df = n-1
$$

**Two-sample $t$-test (pooled):**
$$
t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}} \quad s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}
$$

**Welch's $t$-test (unequal variances):**
$$
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}
$$

**Chi-squared test:**
$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

**Z-test for proportion:**
$$
z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}
$$

**P-value from test statistic:**
$$
p = P(|T| > |t_{\text{obs}}|) \quad \text{(two-tailed)}
$$

## Properties

- **$\alpha$ is chosen before the test:** Typically 0.05, 0.01, or 0.10.
- **$p$-value is not the probability $H_0$ is true:** It is the probability of the data (or more extreme) given $H_0$.
- **Failing to reject $H_0$ is not accepting $H_0$:** It means insufficient evidence to reject.
- **$p$-value depends on sample size:** With large $n$, even tiny effects become significant.
- **One-tailed vs two-tailed:** Two-tailed tests are more conservative and should be used unless there is a strong directional hypothesis.
- **Power increases with:** Larger $\alpha$, larger effect size, larger $n$, lower variability.

## Step-by-Step Worked Examples

### Example 1: One-Sample $t$-Test

**Problem:** A manufacturer claims their batteries last 100 hours. A sample of 10 batteries has $\bar{x} = 95$ hours, $s = 8$ hours. Test at $\alpha = 0.05$ whether the mean battery life differs from 100 hours.

**Solution:**

Step 1: State hypotheses.
$H_0: \mu = 100$
$H_1: \mu \neq 100$ (two-tailed)

Step 2: Test statistic.
$$
t = \frac{95 - 100}{8/\sqrt{10}} = \frac{-5}{2.529} = -1.977
$$

Step 3: Degrees of freedom.
$df = 10 - 1 = 9$

Step 4: Critical value.
$t_{0.025, 9} = 2.262$ (from $t$-table).

Step 5: Decision.
$|t| = 1.977 < 2.262$, so we fail to reject $H_0$.

Step 6: Conclusion.
There is insufficient evidence to conclude the mean battery life differs from 100 hours.

### Example 2: Two-Sample $t$-Test

**Problem:** Compare test scores between two teaching methods.
Method A ($n=12$): $\bar{x}_1=78$, $s_1=10$.
Method B ($n=10$): $\bar{x}_2=85$, $s_2=8$.
Test at $\alpha=0.05$ if Method B is better.

**Solution:**

Step 1: Hypotheses.
$H_0: \mu_A = \mu_B$
$H_1: \mu_A < \mu_B$ (one-tailed)

Step 2: Pooled variance.
$$
s_p^2 = \frac{(12-1)10^2 + (10-1)8^2}{12+10-2} = \frac{1100 + 576}{20} = \frac{1676}{20} = 83.8
$$
$s_p = \sqrt{83.8} \approx 9.154$

Step 3: Test statistic.
$$
t = \frac{78 - 85}{9.154\sqrt{1/12 + 1/10}} = \frac{-7}{9.154 \times 0.4249} = \frac{-7}{3.889} = -1.80
$$

Step 4: $df = 20$. One-tailed critical value: $t_{0.05, 20} = 1.725$.

Step 5: Since $|-1.80| > 1.725$, we reject $H_0$.

Step 6: Method B produces significantly higher scores at $\alpha = 0.05$.

### Example 3: $\chi^2$ Test for Independence

**Problem:** Test whether gender and voting preference are independent.
$$\begin{array}{c|cc|c}
& \text{Candidate X} & \text{Candidate Y} & \text{Total} \\
\hline
\text{Male} & 30 & 20 & 50 \\
\text{Female} & 25 & 25 & 50 \\
\hline
\text{Total} & 55 & 45 & 100
\end{array}$$

**Solution:**

Step 1: $H_0$: Gender and voting preference are independent.
$H_1$: They are not independent.

Step 2: Expected frequencies (under independence).
$E_{ij} = (\text{row total} \times \text{column total}) / \text{grand total}$
- Male-X: $50 \times 55 / 100 = 27.5$
- Male-Y: $50 \times 45 / 100 = 22.5$
- Female-X: $50 \times 55 / 100 = 27.5$
- Female-Y: $50 \times 45 / 100 = 22.5$

Step 3: $\chi^2$ statistic.
$$
\chi^2 = \frac{(30-27.5)^2}{27.5} + \frac{(20-22.5)^2}{22.5} + \frac{(25-27.5)^2}{27.5} + \frac{(25-22.5)^2}{22.5}
$$
$$
\chi^2 = \frac{2.5^2}{27.5} + \frac{(-2.5)^2}{22.5} + \frac{(-2.5)^2}{27.5} + \frac{2.5^2}{22.5}
$$
$$
\chi^2 = \frac{6.25}{27.5} + \frac{6.25}{22.5} + \frac{6.25}{27.5} + \frac{6.25}{22.5} = 0.227 + 0.278 + 0.227 + 0.278 = 1.01
$$

Step 4: $df = (2-1)(2-1) = 1$. Critical value $\chi^2_{0.05, 1} = 3.841$.

Step 5: $1.01 < 3.841$, fail to reject $H_0$.

Step 6: No significant association between gender and voting preference.

## Visual Interpretation

The $p$-value can be visualised as the area in the tails of the sampling distribution beyond the observed test statistic. For a two-tailed $t$-test, the $p$-value is the total area in both tails beyond $\pm |t_{\text{obs}}|$.

A power curve shows the probability of rejecting $H_0$ as a function of the true effect size. Steeper curves indicate more powerful tests.

A Q-Q plot of $p$-values (p-value histogram) is used in multiple testing to assess whether the overall distribution of $p$-values is uniform (under $H_0$) or enriched for small values (under $H_1$).

## Common Mistakes

1. **Misinterpreting $p$-value:** A $p$-value is NOT the probability that $H_0$ is true. It is $P(\text{data or more extreme} | H_0)$.

2. **$p$-hacking:** Running multiple tests or analyses until finding a significant result, then reporting only that result. This inflates Type I error.

3. **Confusing statistical significance with practical significance:** A very small $p$-value does not mean the effect is large or important, especially with large $n$.

4. **Using one-tailed tests inappropriately:** One-tailed tests have more power but should only be used when the direction of effect is truly known in advance.

5. **Failing to check assumptions:** $t$-tests assume normality and equal variance. Violations can invalidate results.

6. **Multiple testing without correction:** Testing hundreds of hypotheses at $\alpha = 0.05$ will produce many false positives by chance.

7. **Equating "fail to reject $H_0$" with "accept $H_0$":** Absence of evidence is not evidence of absence.

## Interview Questions

### Beginner - 5

1. **Q:** What is the null hypothesis?
   **A:** The default assumption, usually of no effect or no difference. It is the hypothesis we seek evidence against.

2. **Q:** What is a Type I error?
   **A:** Rejecting the null hypothesis when it is actually true (false positive).

3. **Q:** What is a $p$-value?
   **A:** The probability of observing data as extreme as the sample, assuming $H_0$ is true.

4. **Q:** What does $\alpha = 0.05$ mean?
   **A:** There is a 5% risk of rejecting $H_0$ when it is true (Type I error).

5. **Q:** What is the difference between one-tailed and two-tailed tests?
   **A:** One-tailed tests for an effect in one direction; two-tailed tests for any difference.

### Intermediate - 5

1. **Q:** How does sample size affect hypothesis testing?
   **A:** Larger $n$ increases power (ability to detect small effects) and makes even tiny effects statistically significant.

2. **Q:** What is the relationship between CIs and hypothesis tests?
   **A:** A 95% CI contains all values of $\mu$ that would not be rejected at $\alpha = 0.05$ by a two-tailed test.

3. **Q:** How would you compare two ML models using hypothesis testing?
   **A:** Use a paired $t$-test on cross-validation results or McNemar's test for classification.

4. **Q:** What is the Bonferroni correction?
   **A:** Dividing $\alpha$ by the number of tests ($\alpha/m$) to control the family-wise error rate.

5. **Q:** What is statistical power and why does it matter?
   **A:** Power $= 1 - \beta$, the probability of correctly rejecting a false $H_0$. Low power means the study may miss real effects.

### Advanced - 3

1. **Q:** Derive the likelihood ratio test statistic and explain its asymptotic distribution.
   **A:** $\Lambda = -2\log(L(H_0)/L(H_1)) \sim \chi^2_{df}$ under $H_0$ by Wilks' theorem, where $df$ is the difference in parameter dimension.

2. **Q:** Explain the Neyman-Pearson lemma and its implications for optimal tests.
   **A:** The most powerful test of $H_0$ vs $H_1$ rejects for large values of the likelihood ratio $L(H_1)/L(H_0)$. This provides the theoretical foundation for likelihood-based tests.

3. **Q:** Discuss the problem of multiple hypothesis testing in high-dimensional genomics/ML and compare FWER vs FDR control.
   **A:** FWER (Bonferroni) controls the probability of any false positive. FDR (Benjamini-Hochberg) controls the expected proportion of false positives among rejected hypotheses. FDR is less conservative and more powerful when many true effects exist.

## Practice Problems

### Easy - 5

1. If $p = 0.03$ and $\alpha = 0.05$, do we reject $H_0$?

2. What is the probability of a Type I error if $\alpha = 0.01$?

3. In a one-sample $t$-test with $n=20$, what are the degrees of freedom?

4. If we fail to reject $H_0$, what error might we be making?

5. A test has $p = 0.08$ at $\alpha = 0.05$. What is the conclusion?

### Medium - 5

1. Compute the $t$-statistic: $\bar{x}=55$, $\mu_0=50$, $s=10$, $n=25$.

2. Given $n=30$, $\bar{x}=100$, $s=15$, test $H_0: \mu=95$ vs $H_1: \mu \neq 95$ at $\alpha=0.05$.

3. Two samples: $n_1=15$, $\bar{x}_1=20$, $s_1=4$; $n_2=15$, $\bar{x}_2=24$, $s_2=5$. Test if means differ at $\alpha=0.05$.

4. Explain what power $= 0.80$ means.

5. In A/B testing with 10,000 users per variant, the new design shows $p = 0.04$ for conversion rate increase. Is this practically significant if the conversion rate increased from 5.0% to 5.1%?

### Hard - 3

1. Derive the two-sample $t$-test statistic and its degrees of freedom under the equal variance assumption.

2. Prove that the expected value of the $\chi^2$ test statistic under $H_0$ equals the degrees of freedom.

3. Discuss the Behrens-Fisher problem and how Welch's $t$-test addresses it.

## Solutions

**Easy:**

1. Yes, since $p = 0.03 < 0.05$.

2. $\alpha = 0.01$, so the Type I error probability is 0.01.

3. $df = n-1 = 19$.

4. Type II error (false negative) -- we failed to detect a real effect.

5. Fail to reject $H_0$ (insufficient evidence at $\alpha = 0.05$).

**Medium:**

1. $t = (55-50)/(10/\sqrt{25}) = 5/2 = 2.5$.

2. $t = (100-95)/(15/\sqrt{30}) = 5/2.739 = 1.825$. $df = 29$. Two-tailed critical $t_{0.025,29} \approx 2.045$. $1.825 < 2.045$, fail to reject $H_0$.

3. $s_p^2 = (14\cdot16 + 14\cdot25)/(28) = (224+350)/28 = 574/28 = 20.5$. $s_p \approx 4.528$. $t = (20-24)/(4.528\sqrt{1/15+1/15}) = -4/(4.528 \cdot 0.365) = -4/1.653 = -2.42$. $df=28$, two-tailed critical $t_{0.025,28} \approx 2.048$. $|t|=2.42 > 2.048$, reject $H_0$.

4. If $H_0$ is false, there is an 80% chance the test will correctly reject it.

5. The result is statistically significant ($p<0.05$) but the practical improvement (1% relative increase from 5.0% to 5.1%) may not be worth implementing.

**Hard:**

1. The test statistic is $t = (\bar{x}_1 - \bar{x}_2 - \Delta_0)/(s_p\sqrt{1/n_1+1/n_2})$. Under $H_0$ and the equal variance normality assumption, $t \sim t_{n_1+n_2-2}$. The derivation uses the fact that $\bar{x}_1 - \bar{x}_2 \sim N(\mu_1-\mu_2, \sigma^2(1/n_1+1/n_2))$ and $(n_1+n_2-2)s_p^2/\sigma^2 \sim \chi^2_{n_1+n_2-2}$.

2. For a $2 \times 2$ table, $E[\chi^2] = E[\sum (O-E)^2/E]$. Under $H_0$, $O \sim \text{Poisson}(E)$ or multinomial with $E[O] = E$. For each cell, $E[(O-E)^2] = \text{Var}(O) = E$ (approximately). So $E[(O-E)^2/E] \approx 1$, and the sum of $rc$ cells has expected value $rc - 1$ (the degrees of freedom), with the $-1$ accounting for constraints.

3. The Behrens-Fisher problem is testing the difference of two means when variances are unequal. Welch's solution approximates the degrees of freedom using the Satterthwaite approximation, avoiding the equal variance assumption of the pooled $t$-test.

## Related Concepts

- Confidence Intervals (MATH-086) — dual to hypothesis tests
- Type I & II Errors — error framework
- $p$-value — evidence against $H_0$
- Statistical Power — probability of detecting effects
- $t$-test — most common hypothesis test
- $\chi^2$ Test — categorical data hypothesis test
- ANOVA — extension of $t$-test to multiple groups
- Multiple Testing — correction for many simultaneous tests

## Next Concepts

- Effect Size — measuring practical significance
- Bayesian Inference — alternative to frequentist testing
- Experimental Design — planning studies for valid inference

## Summary

Hypothesis testing provides a rigorous framework for making decisions under uncertainty. The null hypothesis ($H_0$) represents the default claim; the alternative ($H_1$) represents the effect of interest. The $p$-value quantifies the evidence against $H_0$. Type I error (false positive) is controlled by $\alpha$; Type II error (false negative) relates to power. Common tests include the $t$-test for means and the $\chi^2$ test for categorical data. In AI/ML, hypothesis testing is used for A/B testing, model comparison, feature selection, and evaluating statistical significance of improvements.

## Key Takeaways

- $H_0$: no effect; $H_1$: effect exists.
- Type I error: rejecting true $H_0$ ($\alpha$).
- Type II error: failing to reject false $H_0$ ($\beta$).
- $p$-value $< \alpha$: reject $H_0$.
- $p$-value is NOT probability $H_0$ is true.
- $t$-test compares means; $\chi^2$ test compares frequencies.
- Statistical significance $\neq$ practical significance.
- Multiple testing requires correction (Bonferroni, FDR).
- Power $= 1 - \beta$: ability to detect real effects.
- In ML: use for model comparison, feature selection, A/B testing.
