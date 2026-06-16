# Concept: Binomial Distribution

## Concept ID

MATH-073

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

1. Define the Binomial distribution and derive its probability mass function
2. Compute probabilities, mean, variance, and cumulative probabilities for Binomial random variables
3. Understand the Binomial distribution as the sum of independent Bernoulli trials
4. Apply the Binomial distribution to model count of successes in real-world scenarios
5. Connect the Binomial distribution to accuracy metrics, A/B testing, and hypothesis testing in machine learning

## Prerequisites

- Bernoulli distribution (MATH-072)
- Combinatorics: combinations $\binom{n}{k}$
- Independence of random variables
- Expected value and variance of a sum of independent variables
- Basic probability axioms

## Definition

The Binomial distribution is the discrete probability distribution of the number of successes in a fixed number of $n$ independent Bernoulli trials, each with the same probability of success $p$. A random variable $X$ following a Binomial distribution is denoted as

$$X \sim \text{Binomial}(n, p)$$

The probability mass function (PMF) is given by

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

## Intuition

The Binomial distribution answers the question: "If I repeat a binary experiment $n$ times independently, with success probability $p$ each time, how many successes will I observe?" It is the natural extension of the Bernoulli distribution from one trial to many trials.

Imagine flipping a fair coin 10 times and counting the number of heads. The result could be anywhere from 0 to 10, but some outcomes (like 5 heads) are more likely than others (like 0 heads or 10 heads). The Binomial distribution tells us exactly how likely each possible count is.

The name "binomial" comes from the binomial theorem: $\sum_{k=0}^n \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1^n = 1$, confirming that the probabilities sum to 1.

## Why This Concept Matters

The Binomial distribution is one of the most widely used discrete distributions in statistics and machine learning:

- It models accuracy, precision, recall, and other binary metrics in classification
- It is the foundation of A/B testing and hypothesis testing for proportions
- It underpins confidence intervals for success probabilities
- It appears in the analysis of experimental design, clinical trials, and quality control
- The normal approximation to the Binomial (for large $n$) provides the theoretical basis for many statistical tests used in ML model evaluation

## Historical Background

The Binomial distribution was first studied by Jacob Bernoulli in his work *Ars Conjectandi* (1713), though the binomial coefficients had been known much earlier (Pascal's triangle dates to ancient Chinese mathematics). Bernoulli proved the first version of the Law of Large Numbers specifically for Binomial random variables, showing that the observed proportion of successes converges to the true probability $p$ as the number of trials increases. Later, Abraham de Moivre (1738) derived the normal approximation to the Binomial distribution, which was a precursor to the Central Limit Theorem. Pierre-Simon Laplace further developed the theory, and the distribution has since become a cornerstone of statistical inference.

## Real World Examples

1. **Quality Control**: A factory produces electronic chips with a 5% defect rate. An inspector tests 20 chips. The number of defective chips $X$ follows $X \sim \text{Binomial}(20, 0.05)$.

2. **Clinical Trials**: A new drug is effective for 70% of patients. In a trial with 50 patients, the number of patients who respond positively follows $X \sim \text{Binomial}(50, 0.7)$.

3. **Survey Sampling**: A political poll surveys 1000 voters. If 52% of the population supports a candidate, the number of supporters in the sample follows $X \sim \text{Binomial}(1000, 0.52)$.

4. **Sports Analytics**: A basketball player makes 80% of free throws. In a game with 15 free throw attempts, the number of made shots follows $X \sim \text{Binomial}(15, 0.8)$.

5. **Genetics**: A genetic trait is inherited with probability 25% (recessive trait). In a family with 4 children, the number of children expressing the trait follows $X \sim \text{Binomial}(4, 0.25)$.

## AI/ML Relevance

The Binomial distribution is deeply connected to model evaluation and experimental comparison in machine learning.

**Model Accuracy as a Binomial Variable**: When evaluating a binary classifier on $n$ test examples, if the true error rate is $p$, the number of misclassified examples $X$ follows $X \sim \text{Binomial}(n, p)$. The observed accuracy is $1 - X/n$. This allows us to compute confidence intervals for model accuracy:

$$\text{Accuracy} \pm z_{\alpha/2} \sqrt{\frac{\text{Accuracy} \times (1 - \text{Accuracy})}{n}}$$

**Comparing Two Models**: When comparing two classifiers on the same test set, McNemar's test (based on the Binomial distribution) is used to determine if the difference in performance is statistically significant.

**A/B Testing**: In A/B testing, two versions of a system (e.g., a recommender system or a webpage) are compared. The number of conversions in each group follows a Binomial distribution. A two-proportion z-test (based on the normal approximation to the Binomial) determines if the difference in conversion rates is statistically significant.

**Precision and Recall**: Metrics like precision, recall, and F1-score are calculated from counts that follow Binomial distributions. Understanding their variance helps in reporting these metrics with confidence intervals.

**Bagging and Ensemble Methods**: In bootstrap aggregating (bagging), each bootstrap sample is drawn with replacement from the original dataset. The number of times a particular example appears in a bootstrap sample of size $n$ follows $\text{Binomial}(n, 1/n) \approx \text{Poisson}(1)$.

**Learning Theory**: In PAC (Probably Approximately Correct) learning theory, the Binomial distribution is used to bound the probability that the empirical error deviates from the true error by more than a given threshold, leading to sample complexity bounds.

## Mathematical Explanation

### Probability Mass Function

For $X \sim \text{Binomial}(n, p)$,

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

The binomial coefficient $\binom{n}{k}$ counts the number of ways to arrange $k$ successes among $n$ trials.

### Derivation

Consider $n$ independent Bernoulli trials $Y_1, \ldots, Y_n$ each with $P(Y_i = 1) = p$. Then $X = \sum_{i=1}^n Y_i$ is the total number of successes. For $X = k$, we need exactly $k$ of the $Y_i$ to equal 1 and $n-k$ to equal 0. The probability of any specific sequence with $k$ successes is $p^k (1-p)^{n-k}$. Since there are $\binom{n}{k}$ such sequences:

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

### Cumulative Distribution Function

The CDF is:

$$F(k) = P(X \leq k) = \sum_{i=0}^k \binom{n}{i} p^i (1-p)^{n-i}$$

There is no closed-form expression for the CDF, but it can be expressed in terms of the regularized incomplete beta function:

$$P(X \leq k) = I_{1-p}(n-k, k+1)$$

### Expected Value

$$\mathbb{E}[X] = \sum_{i=1}^n \mathbb{E}[Y_i] = \sum_{i=1}^n p = np$$

### Variance

$$\text{Var}[X] = \sum_{i=1}^n \text{Var}[Y_i] = \sum_{i=1}^n p(1-p) = np(1-p)$$

The variance follows from the independence of the Bernoulli trials.

### Moment-Generating Function

$$M_X(t) = \mathbb{E}[e^{tX}] = \mathbb{E}\left[e^{t\sum Y_i}\right] = \prod_{i=1}^n \mathbb{E}[e^{tY_i}] = \left(1 - p + p e^{t}\right)^n$$

### Skewness

The skewness of $\text{Binomial}(n, p)$ is:

$$\gamma_1 = \frac{1 - 2p}{\sqrt{np(1-p)}}$$

The distribution is symmetric when $p = 0.5$, positively skewed when $p < 0.5$, and negatively skewed when $p > 0.5$. As $n$ increases, the skewness decreases (the distribution becomes more symmetric).

### Mode

The mode(s) of the Binomial distribution occur at

$$\lfloor (n+1)p \rfloor \quad \text{and} \quad \lfloor (n+1)p \rfloor - 1 \text{ if } (n+1)p \text{ is an integer}$$

## Formula(s)

1. **PMF**: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$

2. **Mean**: $\mathbb{E}[X] = np$

3. **Variance**: $\text{Var}[X] = np(1-p)$

4. **Standard Deviation**: $\sigma_X = \sqrt{np(1-p)}$

5. **MGF**: $M_X(t) = (1 - p + pe^t)^n$

6. **CDF**: $F(k) = \sum_{i=0}^k \binom{n}{i} p^i (1-p)^{n-i}$

## Properties

1. **Sum of independent Binomials**: If $X \sim \text{Binomial}(n_1, p)$ and $Y \sim \text{Binomial}(n_2, p)$ are independent, then $X + Y \sim \text{Binomial}(n_1 + n_2, p)$.

2. **Normal approximation**: For large $n$, $\text{Binomial}(n, p) \approx \mathcal{N}(np, np(1-p))$. This approximation works well when $np \geq 5$ and $n(1-p) \geq 5$.

3. **Poisson approximation**: For large $n$ and small $p$ (rare events), $\text{Binomial}(n, p) \approx \text{Poisson}(\lambda = np)$. This works well when $n \geq 100$, $p \leq 0.05$, and $np \leq 10$.

4. **Relation to Bernoulli**: $\text{Binomial}(1, p) = \text{Bernoulli}(p)$.

5. **Symmetry**: If $X \sim \text{Binomial}(n, p)$, then $n - X \sim \text{Binomial}(n, 1-p)$.

6. **Reproductivity**: The Binomial family is reproductive under addition when the $p$ parameter is the same across distributions.

7. **Exponential family**: The Binomial distribution (with known $n$) belongs to the exponential family.

8. **Conjugate prior**: The Beta distribution is the conjugate prior for $p$ when $n$ is known.

## Step-by-Step Worked Examples

### Example 1: Fair Coin Tosses

A fair coin is tossed 10 times. Let $X$ be the number of heads.

**Step 1**: Identify parameters. $n = 10$, $p = 0.5$.

**Step 2**: Distribution. $X \sim \text{Binomial}(10, 0.5)$.

**Step 3**: Probability of exactly 5 heads.

$$P(X = 5) = \binom{10}{5} (0.5)^5 (0.5)^5 = \binom{10}{5} (0.5)^{10}$$

$$\binom{10}{5} = \frac{10!}{5!5!} = 252$$

$$P(X = 5) = 252 \times \frac{1}{1024} \approx 0.2461$$

**Step 4**: Probability of at most 2 heads.

$$P(X \leq 2) = P(X = 0) + P(X = 1) + P(X = 2)$$

$$P(X = 0) = \binom{10}{0} (0.5)^{10} = \frac{1}{1024} \approx 0.0010$$

$$P(X = 1) = \binom{10}{1} (0.5)^{10} = \frac{10}{1024} \approx 0.0098$$

$$P(X = 2) = \binom{10}{2} (0.5)^{10} = \frac{45}{1024} \approx 0.0439$$

$$P(X \leq 2) \approx 0.0010 + 0.0098 + 0.0439 = 0.0547$$

**Step 5**: Mean and variance.

$$\mathbb{E}[X] = 10 \times 0.5 = 5, \quad \text{Var}[X] = 10 \times 0.5 \times 0.5 = 2.5$$

### Example 2: Quality Control

A factory produces light bulbs with a 2% defect rate. An inspector checks 100 bulbs. What is the probability that at most 1 bulb is defective?

**Step 1**: Parameters. $n = 100$, $p = 0.02$.

**Step 2**: Distribution. $X \sim \text{Binomial}(100, 0.02)$.

**Step 3**: Since $n$ is large and $p$ is small, we can use the Poisson approximation with $\lambda = np = 2$.

**Step 4**: Exact Binomial calculation.

$$P(X \leq 1) = P(X = 0) + P(X = 1)$$

$$P(X = 0) = \binom{100}{0} (0.02)^0 (0.98)^{100} = (0.98)^{100} \approx 0.1326$$

$$P(X = 1) = \binom{100}{1} (0.02)^1 (0.98)^{99} = 100 \times 0.02 \times (0.98)^{99} \approx 0.2707$$

$$P(X \leq 1) \approx 0.1326 + 0.2707 = 0.4033$$

**Step 5**: Poisson approximation.

$$P(X \leq 1) \approx e^{-2} \left(\frac{2^0}{0!} + \frac{2^1}{1!}\right) = e^{-2}(1 + 2) = 3e^{-2} \approx 0.4060$$

**Step 6**: Interpret. The approximation is close to the exact value (0.4033 vs 0.4060). There is about a 40% chance of finding at most 1 defective bulb in a batch of 100.

### Example 3: Model Accuracy and Confidence Intervals

A binary classifier achieves 85% accuracy on 200 test examples. Compute a 95% confidence interval for the true accuracy.

**Step 1**: Parameters. $\hat{p} = 0.85$, $n = 200$.

**Step 2**: The number of correct predictions $X \sim \text{Binomial}(200, p)$ where $p$ is the true accuracy.

**Step 3**: Using the normal approximation to the Binomial. The standard error is:

$$\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = \sqrt{\frac{0.85 \times 0.15}{200}} = \sqrt{\frac{0.1275}{200}} \approx 0.0252$$

**Step 4**: For a 95% confidence interval, $z_{0.025} = 1.96$.

$$\text{CI} = \hat{p} \pm 1.96 \times \text{SE} = 0.85 \pm 1.96 \times 0.0252 = 0.85 \pm 0.0494$$

$$\text{CI} = (0.8006, 0.8994)$$

**Step 5**: Interpret. We are 95% confident that the true accuracy of the classifier is between 80.06% and 89.94%.

**Step 6**: Verify the normal approximation conditions. $n\hat{p} = 200 \times 0.85 = 170 \geq 5$ and $n(1-\hat{p}) = 200 \times 0.15 = 30 \geq 5$. The approximation is valid.

### Example 4: A/B Testing

A website tests two landing pages. Version A gets 120 conversions out of 1000 visitors. Version B gets 150 conversions out of 1000 visitors. Is the difference statistically significant at $\alpha = 0.05$?

**Step 1**: Conversion rates. $\hat{p}_A = 0.12$, $\hat{p}_B = 0.15$.

**Step 2**: Pooled proportion under null hypothesis (no difference).

$$\hat{p}_{\text{pooled}} = \frac{120 + 150}{1000 + 1000} = \frac{270}{2000} = 0.135$$

**Step 3**: Standard error of the difference.

$$\text{SE} = \sqrt{\hat{p}_{\text{pooled}}(1-\hat{p}_{\text{pooled}})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)} = \sqrt{0.135 \times 0.865 \times \left(\frac{1}{1000} + \frac{1}{1000}\right)}$$

$$\text{SE} = \sqrt{0.135 \times 0.865 \times 0.002} = \sqrt{0.00023355} \approx 0.01528$$

**Step 4**: Test statistic (z-score).

$$z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}} = \frac{0.15 - 0.12}{0.01528} \approx 1.963$$

**Step 5**: Compare to critical value $z_{0.025} = 1.96$. Since $1.963 > 1.96$, we reject the null hypothesis at $\alpha = 0.05$.

**Step 6**: Interpret. Version B has a statistically significantly higher conversion rate than Version A at the 5% significance level.

### Example 5: Probability of Rare Event

A medical treatment has a 95% success rate. If 20 patients are treated, what is the probability that all 20 succeed? What is the probability that at least 18 succeed?

**Step 1**: Parameters. $n = 20$, $p = 0.95$.

**Step 2**: Let $X$ be the number of successes. $X \sim \text{Binomial}(20, 0.95)$.

**Step 3**: Probability that all 20 succeed.

$$P(X = 20) = \binom{20}{20} (0.95)^{20} (0.05)^0 = (0.95)^{20} \approx 0.3585$$

**Step 4**: Probability that at least 18 succeed.

$$P(X \geq 18) = P(X = 18) + P(X = 19) + P(X = 20)$$

$$P(X = 18) = \binom{20}{18} (0.95)^{18} (0.05)^2 = 190 \times (0.95)^{18} \times 0.0025 \approx 0.1887$$

$$P(X = 19) = \binom{20}{19} (0.95)^{19} (0.05)^1 = 20 \times (0.95)^{19} \times 0.05 \approx 0.3774$$

$$P(X = 20) \approx 0.3585$$

$$P(X \geq 18) \approx 0.1887 + 0.3774 + 0.3585 = 0.9246$$

**Step 5**: Interpret. There is a 92.46% probability that at least 18 out of 20 patients succeed with the treatment.

## Visual Interpretation

The Binomial distribution can be visualized as a histogram with bars at $k = 0, 1, \ldots, n$, where the height of each bar is $P(X = k)$.

**Shape depends on $p$**:
- When $p = 0.5$, the histogram is symmetric and bell-shaped (for moderate to large $n$)
- When $p < 0.5$, the histogram is skewed right (long tail toward higher values)
- When $p > 0.5$, the histogram is skewed left (long tail toward lower values)
- As $n$ increases, the distribution becomes more symmetric and more concentrated around $np$ (by the Law of Large Numbers)

**Effect of $n$**: As $n$ grows, the distribution becomes more bell-shaped and narrow relative to its mean. The coefficient of variation $\frac{\sqrt{np(1-p)}}{np} = \sqrt{\frac{1-p}{np}}$ decreases, meaning the relative spread shrinks.

The normal approximation overlays a Normal curve with mean $np$ and variance $np(1-p)$ over the Binomial histogram, showing a close match when $n$ is large enough.

## Common Mistakes

1. **Confusing $n$ and $k$**: The Binomial coefficient $\binom{n}{k}$ counts ways to choose $k$ successes from $n$ trials, not the other way around. A common error is writing $\binom{k}{n}$.

2. **Applying Binomial to dependent trials**: The trials must be independent. Drawing without replacement from a finite population produces dependent Bernoulli variables, which follow the Hypergeometric distribution, not Binomial.

3. **Ignoring the fixed $n$ assumption**: The Binomial distribution requires a fixed number of trials. If the number of trials is random (e.g., continue until $r$ successes), use the Negative Binomial distribution instead.

4. **Miscomputing $\binom{n}{k}$ for large $n$**: When $n$ is large, $\binom{n}{k}$ can be astronomically large. Use the normal or Poisson approximation, or use log-probabilities to avoid numerical overflow.

5. **Assuming normal approximation is always valid**: The normal approximation $np \geq 5$ and $n(1-p) \geq 5$ rule must be checked. For small $n$ or extreme $p$, exact Binomial calculations are necessary.

6. **Confusing the Binomial distribution with its sampling distribution**: The Binomial distribution models the count $X$ itself, not the distribution of the estimator $\hat{p} = X/n$ (though the latter can be derived from the former).

7. **Order matters incorrectly**: The Binomial coefficient accounts for all possible orderings of successes and failures. Forgetting the coefficient and writing only $p^k(1-p)^{n-k}$ gives the probability of one specific sequence, not all sequences with $k$ successes.

8. **Applying the Poisson approximation without checking conditions**: The Poisson approximation $(\lambda = np)$ works well only when $n$ is large, $p$ is small, and $np$ is moderate (typically $np \leq 10$).

## Interview Questions

### Beginner

1. **Q**: What are the parameters of a Binomial distribution?
   **A**: $n$ (number of trials) and $p$ (probability of success on each trial).

2. **Q**: What is the expected value of $\text{Binomial}(n, p)$?
   **A**: $\mathbb{E}[X] = np$.

3. **Q**: What is the variance of $\text{Binomial}(n, p)$?
   **A**: $\text{Var}[X] = np(1-p)$.

4. **Q**: If $X \sim \text{Binomial}(5, 0.5)$, compute $P(X = 0)$.
   **A**: $P(X = 0) = \binom{5}{0} (0.5)^0 (0.5)^5 = (0.5)^5 = 0.03125$.

5. **Q**: What is the relationship between the Bernoulli and Binomial distributions?
   **A**: $\text{Binomial}(1, p) = \text{Bernoulli}(p)$. The Binomial with $n=1$ is exactly the Bernoulli. More generally, the sum of $n$ independent $\text{Bernoulli}(p)$ variables is $\text{Binomial}(n, p)$.

### Intermediate

1. **Q**: Derive the moment-generating function of a Binomial random variable.
   **A**: For $X = \sum_{i=1}^n Y_i$ where $Y_i \sim \text{Bernoulli}(p)$ are independent, $M_X(t) = \prod_{i=1}^n M_{Y_i}(t) = (1-p+pe^t)^n$.

2. **Q**: State the conditions for using the normal approximation to the Binomial distribution. What continuity correction should be applied?
   **A**: The normal approximation is valid when $np \geq 5$ and $n(1-p) \geq 5$. The continuity correction adjusts for approximating a discrete distribution with a continuous one: $P(X \leq k) \approx \Phi\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)$.

3. **Q**: If $X \sim \text{Binomial}(n, p)$, what is the distribution of $n - X$?
   **A**: $n - X \sim \text{Binomial}(n, 1-p)$. This follows because $n-X$ counts the number of failures, and each failure has probability $1-p$.

4. **Q**: In a binary classification problem, how would you compute a confidence interval for the model's accuracy?
   **A**: The number of correct predictions $X \sim \text{Binomial}(n, p)$ where $p$ is true accuracy. The observed accuracy $\hat{p} = X/n$ approximates $p$. Using the normal approximation, a 95% CI is $\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/n}$. For small $n$, use the exact Clopper-Pearson (Beta-based) interval.

5. **Q**: Explain how the Binomial distribution is used in A/B testing to compare two conversion rates.
   **A**: Under the null hypothesis of equal conversion rates $p_A = p_B$, the pooled proportion estimates the common $p$. The test statistic $z = (\hat{p}_B - \hat{p}_A) / \sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}$ follows approximately $N(0,1)$ under null, using the normal approximation to the Binomial. A significant $z$ indicates a real difference.

### Advanced

1. **Q**: Prove that the Binomial distribution belongs to the exponential family when $n$ is known. Identify the natural parameter and sufficient statistic.
   **A**: For $X \sim \text{Binomial}(n, p)$, the PMF is $f(x; p) = \binom{n}{x} \exp\left[x\log\left(\frac{p}{1-p}\right) + n\log(1-p)\right]$. With natural parameter $\theta = \log(p/(1-p))$, $f(x; \theta) = \binom{n}{x} \exp\left[x\theta - n\log(1+e^\theta)\right]$. The sufficient statistic is $T(X) = X$, and the log-partition function is $A(\theta) = n\log(1+e^\theta)$.

2. **Q**: Derive the bias and variance of the MLE $\hat{p} = X/n$ for a Binomial model. Is $\hat{p}$ the uniformly minimum variance unbiased estimator (UMVUE)?
   **A**: $\mathbb{E}[\hat{p}] = \mathbb{E}[X]/n = p$, so $\hat{p}$ is unbiased. $\text{Var}[\hat{p}] = \text{Var}[X]/n^2 = p(1-p)/n$. By the Lehmann-Scheffé theorem, since $X$ is a complete sufficient statistic for $p$ and $\hat{p}$ is unbiased, $\hat{p}$ is the UMVUE.

3. **Q**: In PAC learning theory, the Binomial distribution is used to bound the generalization error. Derive the Chernoff bound for $\text{Binomial}(n, p)$ and explain its use in deriving sample complexity.
   **A**: For $X \sim \text{Binomial}(n, p)$, the Chernoff bound states: $P(X \geq (1+\delta)np) \leq \left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^{np}$ for $\delta > 0$, and $P(X \leq (1-\delta)np) \leq \left(\frac{e^{-\delta}}{(1-\delta)^{1-\delta}}\right)^{np}$ for $0 < \delta < 1$. These bounds are used in PAC learning to show that the empirical error converges to the true error exponentially fast in the number of training examples, giving sample complexity bounds like $n \geq \frac{1}{2\epsilon^2}\log\left(\frac{2}{\delta}\right)$ for the Hoeffding version.

## Practice Problems

### Easy

1. A die is rolled 4 times. Let $X$ be the number of times a 6 appears. Find $P(X = 2)$.

2. A multiple-choice test has 10 questions, each with 4 options. If a student guesses randomly, find the probability of getting exactly 3 correct answers.

3. If $X \sim \text{Binomial}(8, 0.3)$, compute $\mathbb{E}[X]$ and $\text{Var}[X]$.

4. A coin is biased with $P(\text{heads}) = 0.6$. If tossed 5 times, find $P(X \geq 4)$ where $X$ is the number of heads.

5. If $X \sim \text{Binomial}(12, 0.25)$, compute $P(X > 9)$.

### Medium

1. A pharmaceutical company claims a drug is 90% effective. In a trial with 15 patients, what is the probability that at least 13 patients respond positively?

2. For $X \sim \text{Binomial}(n, p)$, find the value of $k$ that maximizes $P(X = k)$ (the mode) in terms of $n$ and $p$.

3. A classifier achieves 92% accuracy on 300 test samples. Construct a 99% confidence interval for the true accuracy.

4. In an A/B test, Version A has 45 conversions out of 500 visitors and Version B has 62 conversions out of 500 visitors. Test whether Version B is significantly better at $\alpha = 0.05$.

5. For $X \sim \text{Binomial}(n, p)$, show that $\mathbb{E}[X^2] = np(1-p) + n^2p^2$.

### Hard

1. Let $X \sim \text{Binomial}(n_1, p)$ and $Y \sim \text{Binomial}(n_2, p)$ be independent. Find the conditional distribution of $X$ given $X + Y = m$.

2. Derive the Fisher information for $p$ in a Binomial$(n, p)$ model. Show that the Cramér-Rao lower bound for an unbiased estimator of $p$ is $p(1-p)/n$.

3. In the context of bagging, each bootstrap sample of size $n$ is drawn with replacement from $n$ original examples. Show that the number of times a particular example appears in the bootstrap sample follows $\text{Binomial}(n, 1/n)$. What is the probability that an example is not selected? What is the limit as $n \to \infty$?

## Solutions

### Easy Solutions

1. $n = 4$, $p = 1/6$. $P(X = 2) = \binom{4}{2}(1/6)^2(5/6)^2 = 6 \times (1/36) \times (25/36) \approx 0.1157$.

2. $n = 10$, $p = 0.25$. $P(X = 3) = \binom{10}{3}(0.25)^3(0.75)^7 \approx 120 \times 0.015625 \times 0.13348 \approx 0.2503$.

3. $\mathbb{E}[X] = 8 \times 0.3 = 2.4$. $\text{Var}[X] = 8 \times 0.3 \times 0.7 = 1.68$.

4. $P(X \geq 4) = P(X=4) + P(X=5) = \binom{5}{4}(0.6)^4(0.4) + \binom{5}{5}(0.6)^5 = 5(0.1296)(0.4) + 0.07776 = 0.2592 + 0.07776 = 0.33696$.

5. $P(X > 9) = P(X=10) + P(X=11) + P(X=12)$. $P(X=10) = \binom{12}{10}(0.25)^{10}(0.75)^2 \approx 0.000354$. $P(X=11) = \binom{12}{11}(0.25)^{11}(0.75) \approx 0.0000107$. $P(X=12) = (0.25)^{12} \approx 0.00000006$. Sum $\approx 0.000365$.

### Medium Solutions

1. $P(X \geq 13) = \sum_{k=13}^{15} \binom{15}{k}(0.9)^k(0.1)^{15-k}$. $P(X=13) = \binom{15}{13}(0.9)^{13}(0.1)^2 \approx 0.2669$. $P(X=14) = \binom{15}{14}(0.9)^{14}(0.1) \approx 0.3432$. $P(X=15) = (0.9)^{15} \approx 0.2059$. Sum $\approx 0.8159$.

2. The mode is at $\lfloor (n+1)p \rfloor$ and also at $(n+1)p-1$ if $(n+1)p$ is an integer. This is derived by comparing $P(X=k+1)/P(X=k) = \frac{(n-k)p}{(k+1)(1-p)}$ and finding where the ratio crosses 1.

3. $\hat{p} = 0.92$, $\text{SE} = \sqrt{0.92 \times 0.08 / 300} \approx 0.0157$. For 99% CI, $z = 2.576$. CI = $0.92 \pm 2.576 \times 0.0157 = 0.92 \pm 0.0404 = (0.8796, 0.9604)$.

4. $\hat{p}_A = 45/500 = 0.09$, $\hat{p}_B = 62/500 = 0.124$. Pooled $\hat{p} = 107/1000 = 0.107$. $\text{SE} = \sqrt{0.107 \times 0.893 \times (2/500)} \approx 0.01955$. $z = (0.124 - 0.09)/0.01955 \approx 1.74$. Critical value $z_{0.05} = 1.645$ (one-tailed). Since $1.74 > 1.645$, we reject the null at $\alpha = 0.05$ (one-tailed).

5. $\mathbb{E}[X^2] = \text{Var}[X] + (\mathbb{E}[X])^2 = np(1-p) + (np)^2 = np(1-p) + n^2p^2$.

### Hard Solutions

1. $X + Y \sim \text{Binomial}(n_1 + n_2, p)$. $$P(X = k \mid X+Y = m) = \frac{P(X=k)P(Y=m-k)}{P(X+Y=m)} = \frac{\binom{n_1}{k}\binom{n_2}{m-k}}{\binom{n_1+n_2}{m}}$$. This is the Hypergeometric distribution with parameters $(n_1+n_2, n_1, m)$.

2. The log-likelihood is $\ell(p) = x\log p + (n-x)\log(1-p)$. The score function is $\frac{\partial \ell}{\partial p} = \frac{x}{p} - \frac{n-x}{1-p}$. The Fisher information is $\mathcal{I}(p) = \mathbb{E}\left[\left(\frac{\partial \ell}{\partial p}\right)^2\right] = \frac{n}{p(1-p)}$. The Cramér-Rao lower bound is $1/\mathcal{I}(p) = p(1-p)/n$, which equals $\text{Var}(\hat{p})$, so $\hat{p}$ is efficient.

3. For each bootstrap draw, the probability of selecting a specific example is $1/n$. Over $n$ draws with replacement, the count follows $\text{Binomial}(n, 1/n)$. The probability of not being selected is $(1 - 1/n)^n$. As $n \to \infty$, $(1 - 1/n)^n \to e^{-1} \approx 0.368$. This means each bootstrap sample omits about 36.8% of the original examples, which is a key property used in bagging.

## Related Concepts

- **Bernoulli Distribution (MATH-072)**: The Binomial with $n=1$ is Bernoulli.
- **Poisson Distribution (MATH-074)**: The Poisson approximates the Binomial when $n$ is large and $p$ is small.
- **Normal Distribution (MATH-075)**: The Normal approximates the Binomial when $n$ is large and $p$ is not too extreme.
- **Central Limit Theorem (MATH-076)**: The CLT justifies the normal approximation to the Binomial.
- **Hypergeometric Distribution**: Models sampling without replacement (dependent Bernoulli trials).
- **Negative Binomial Distribution**: Models number of trials until $r$ successes.
- **Beta Distribution**: Conjugate prior for $p$ in the Binomial model.
- **Logistic Regression**: Uses the Binomial likelihood for grouped binary data.

## Next Concepts

- Poisson Distribution (MATH-074)
- Normal Distribution (MATH-075)
- Central Limit Theorem (MATH-076)
- Hypothesis Testing for Proportions
- Logistic Regression
- Generalized Linear Models

## Summary

The Binomial distribution models the number of successes in $n$ independent Bernoulli trials with common success probability $p$. Its PMF is $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$, with mean $np$ and variance $np(1-p)$. It is a fundamental distribution in statistics and machine learning, underpinning model evaluation (accuracy, precision, recall), A/B testing, confidence intervals for proportions, and sample size determination. The Binomial connects to the Bernoulli distribution as its building block, to the Poisson distribution through the rare-event approximation, and to the Normal distribution through the large-sample approximation. Mastering the Binomial distribution is essential for understanding statistical inference, experimental design, and model comparison in machine learning.

## Key Takeaways

- The Binomial distribution models the count of successes in $n$ independent trials with success probability $p$
- PMF: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$, $k = 0, 1, \ldots, n$
- Mean $\mathbb{E}[X] = np$, Variance $\text{Var}[X] = np(1-p)$
- Binomial is the sum of $n$ independent Bernoulli($p$) variables
- Normal approximation: for large $n$, $\text{Binomial}(n,p) \approx \mathcal{N}(np, np(1-p))$
- Poisson approximation: for large $n$ and small $p$, $\text{Binomial}(n,p) \approx \text{Poisson}(np)$
- Used for confidence intervals for accuracy, A/B testing, and hypothesis testing for proportions
- The MLE of $p$ is $\hat{p} = X/n$, which is unbiased and efficient
