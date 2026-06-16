# Concept: Normal Distribution

## Concept ID

MATH-075

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Probability

## Learning Objectives

1. Define the Normal distribution and its probability density function
2. Interpret the parameters mu (mean) and sigma (standard deviation)
3. Standardize a Normal random variable and use the standard Normal table
4. Apply the 68-95-99.7 empirical rule
5. Connect the Normal distribution to weight initialization, noise models, and Gaussian processes in machine learning

## Prerequisites

- Continuous random variables
- Probability density function (PDF) and cumulative distribution function (CDF)
- Expected value and variance of a continuous random variable
- Integration techniques (including Gaussian integrals)
- Central Limit Theorem (conceptual understanding)

## Definition

The Normal (or Gaussian) distribution is a continuous probability distribution that is symmetric about its mean and characterized by a bell-shaped density curve. A random variable $X$ following a Normal distribution with mean $\mu$ and variance $\sigma^2$ is denoted as

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

The probability density function (PDF) is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right), \quad -\infty < x < \infty$$

The standard Normal distribution has $\mu = 0$ and $\sigma = 1$, denoted as $Z \sim \mathcal{N}(0, 1)$.

## Intuition

The Normal distribution is the most important probability distribution in statistics. It arises naturally whenever we average or sum many independent random variables (the Central Limit Theorem). The bell-shaped curve describes the distribution of many natural phenomena: heights, blood pressure, measurement errors, and IQ scores.

The parameter $\mu$ determines the centre (location) of the bell curve, while $\sigma$ determines the spread (scale). A smaller $\sigma$ gives a taller, narrower curve; a larger $\sigma$ gives a shorter, wider curve. The curve is symmetric around $\mu$, and about 68% of the probability mass lies within one standard deviation of the mean.

Think of the Normal distribution as the "default" distribution for continuous data when no other information is available. Many statistical methods assume Normality because of its mathematical convenience and theoretical justification via the CLT.

## Why This Concept Matters

The Normal distribution is fundamental to virtually all of statistics and machine learning:

- It is the limiting distribution of many other distributions (Binomial, Poisson, etc.) via the Central Limit Theorem
- It underpins hypothesis testing, confidence intervals, and linear regression
- Many machine learning algorithms assume Normally distributed errors (linear regression, Gaussian processes)
- It is used for weight initialization in neural networks
- It models noise in signal processing and measurement systems
- The multivariate Normal distribution is the foundation of Gaussian mixture models, factor analysis, and variational autoencoders

## Historical Background

The Normal distribution was first discovered by Abraham de Moivre in 1733 as an approximation to the Binomial distribution. Carl Friedrich Gauss (1777-1855) independently derived the distribution in 1809 while studying the method of least squares and measurement errors, which is why it is also called the Gaussian distribution. Pierre-Simon Laplace also contributed significantly, connecting it to the Central Limit Theorem. Adolphe Quetelet applied the Normal distribution to social science data (human heights, chest measurements). The term "Normal distribution" was popularized by Karl Pearson around 1900, though the name is somewhat misleading (it implies that non-Normal distributions are "abnormal"). The distribution is sometimes called the "bell curve" in popular culture.

## Real World Examples

1. **Human Heights**: The heights of adult women in a population follow approximately $\mathcal{N}(162, 7.5^2)$ cm. About 68% of women have heights between 154.5 cm and 169.5 cm.

2. **IQ Scores**: Standardized IQ tests are designed so that scores follow $\mathcal{N}(100, 15^2)$. An IQ of 130 is two standard deviations above the mean, placing someone in the top 2.5%.

3. **Measurement Error**: Electronic scales produce readings with Normally distributed errors. If a scale has standard deviation $\sigma = 0.5$ g, then 95% of measurements of a 100 g weight will fall within $100 \pm 1.96 \times 0.5 = 100 \pm 0.98$ g.

4. **Stock Returns**: Daily stock returns are approximately Normally distributed (though they exhibit heavier tails in reality). A stock with mean daily return 0.05% and standard deviation 1.5% has about a 16% chance of a negative return on any given day.

5. **Blood Pressure**: Systolic blood pressure in a healthy adult population approximately follows $\mathcal{N}(120, 15^2)$ mmHg. A reading above 150 mmHg ($z = 2$) is in the top 2.5%.

## AI/ML Relevance

The Normal distribution appears throughout machine learning in numerous ways.

**Weight Initialization**: Neural network weights are often initialized from a Normal distribution. For example, Xavier (Glorot) initialization draws weights from $\mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$ to maintain variance across layers. He initialization for ReLU networks uses $\mathcal{N}(0, 2/n_{\text{in}})$. Proper initialization prevents vanishing/exploding gradients.

**Noise Models**: In generative modelling, noise is often sampled from a Normal distribution:
- Denoising autoencoders add Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ to inputs
- Diffusion models (e.g., DDPM) gradually add Gaussian noise to data and learn to reverse the process
- Variational autoencoders (VAEs) assume the latent code follows $\mathcal{N}(0, I)$

**Gaussian Processes**: A Gaussian process (GP) is a distribution over functions where any finite set of function values follows a multivariate Normal distribution. GPs are used for:
- Bayesian optimization (hyperparameter tuning)
- Regression with uncertainty estimates
- Time series forecasting
- Reinforcement learning (modelling unknown reward functions)

**Linear Regression**: The classic linear regression model assumes errors are Normally distributed:

$$y_i = \beta^T x_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

This makes the MLE equivalent to least squares and enables statistical inference (confidence intervals, hypothesis tests) on the coefficients.

**Loss Functions**: Mean squared error (MSE) loss for regression derives from the Normal log-likelihood:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \propto -\log \mathcal{L}(\mu, \sigma^2 \mid \text{data})$$

**Gaussian Mixture Models (GMMs)**: A GMM models data as a weighted sum of $K$ Normal distributions:

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

GMMs are used for clustering, density estimation, and anomaly detection.

**Batch Normalization**: This technique normalizes the activations of each layer to have zero mean and unit variance (approximately following a standard Normal), which stabilizes and accelerates training.

**Normalizing Flows**: These models transform a simple base distribution (usually $\mathcal{N}(0, I)$) into a complex target distribution through a series of invertible transformations.

## Mathematical Explanation

### Probability Density Function

The PDF of $X \sim \mathcal{N}(\mu, \sigma^2)$ is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right), \quad -\infty < x < \infty$$

The PDF is always positive, symmetric about $\mu$, and integrates to 1 over the real line.

### Cumulative Distribution Function

The CDF $F(x) = P(X \leq x)$ does not have a closed-form expression in elementary functions. It is expressed in terms of the error function:

$$F(x) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$$

where $\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$.

For the standard Normal, the CDF is denoted $\Phi(z)$:

$$\Phi(z) = P(Z \leq z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^z e^{-t^2/2} dt = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$

### Standardization

Any Normal random variable can be transformed to the standard Normal:

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

This is called standardization or z-score transformation. It allows us to compute probabilities for any Normal distribution using standard Normal tables or the error function.

### Expected Value

$$\mathbb{E}[X] = \int_{-\infty}^\infty x \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx = \mu$$

### Variance

$$\text{Var}[X] = \mathbb{E}[(X-\mu)^2] = \int_{-\infty}^\infty (x-\mu)^2 \cdot \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx = \sigma^2$$

### Moment-Generating Function

$$M_X(t) = \mathbb{E}[e^{tX}] = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

### Characteristic Function

$$\varphi_X(t) = \mathbb{E}[e^{itX}] = \exp\left(i\mu t - \frac{\sigma^2 t^2}{2}\right)$$

### Skewness and Kurtosis

- Skewness: $\gamma_1 = 0$ (the Normal distribution is perfectly symmetric)
- Excess kurtosis: $\gamma_2 = 0$ (the Normal distribution is the reference for kurtosis)

### Moments

The central moments of the Normal distribution are:

$$\mathbb{E}[(X-\mu)^k] = \begin{cases} 0 & k \text{ odd} \\ \sigma^k (k-1)!! & k \text{ even} \end{cases}$$

where $(k-1)!! = (k-1)(k-3)(k-5)\cdots 1$.

For example:
- $\mathbb{E}[(X-\mu)^4] = 3\sigma^4$
- $\mathbb{E}[(X-\mu)^6] = 15\sigma^6$

## Formula(s)

1. **PDF**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$

2. **Standard Normal PDF**: $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$

3. **CDF**: $F(x) = \Phi\left(\frac{x-\mu}{\sigma}\right) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$

4. **Standardization**: $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

5. **MGF**: $M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$

6. **68-95-99.7 Rule**: $P(|X-\mu| < \sigma) \approx 0.6827$, $P(|X-\mu| < 2\sigma) \approx 0.9545$, $P(|X-\mu| < 3\sigma) \approx 0.9973$

## Properties

1. **Linear transformation**: If $X \sim \mathcal{N}(\mu, \sigma^2)$, then for any constants $a, b$, $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$.

2. **Sum of independent Normals**: If $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are independent, then $X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.

3. **Reproductivity**: The Normal family is closed under addition and linear transformations.

4. **Maximum entropy**: Among all continuous distributions on $\mathbb{R}$ with given mean $\mu$ and variance $\sigma^2$, the Normal distribution $\mathcal{N}(\mu, \sigma^2)$ has maximum entropy.

5. **Exponential family**: The Normal distribution belongs to the exponential family with natural parameters $(\mu/\sigma^2, -1/(2\sigma^2))$.

6. **Central Limit Theorem**: The sum (or average) of $n$ i.i.d. random variables with finite variance converges in distribution to a Normal distribution as $n \to \infty$.

7. **Stein's lemma**: For $X \sim \mathcal{N}(\mu, \sigma^2)$ and any differentiable function $g$ with $\mathbb{E}[|g'(X)|] < \infty$, $\mathbb{E}[g(X)(X-\mu)] = \sigma^2 \mathbb{E}[g'(X)]$.

8. **Closure under conditioning**: The conditional distribution of a multivariate Normal is also Normal.

## Step-by-Step Worked Examples

### Example 1: Computing Probabilities

Suppose IQ scores follow $X \sim \mathcal{N}(100, 15^2)$. Find the probability that a randomly selected person has an IQ between 85 and 115.

**Step 1**: Standardize the values.

$$z_1 = \frac{85 - 100}{15} = -1, \quad z_2 = \frac{115 - 100}{15} = 1$$

**Step 2**: Express the probability in terms of the standard Normal CDF.

$$P(85 < X < 115) = P(-1 < Z < 1) = \Phi(1) - \Phi(-1)$$

**Step 3**: Use the standard Normal table. $\Phi(1) \approx 0.8413$, $\Phi(-1) = 1 - \Phi(1) \approx 0.1587$.

$$P(85 < X < 115) \approx 0.8413 - 0.1587 = 0.6826$$

**Step 4**: Interpret. About 68.3% of people have IQs between 85 and 115, consistent with the 68-95-99.7 rule.

### Example 2: Finding Quantiles

What IQ score corresponds to the top 5% of the population?

**Step 1**: Find $x$ such that $P(X > x) = 0.05$, or $P(X \leq x) = 0.95$.

**Step 2**: Standardize. $P(X \leq x) = \Phi((x-100)/15) = 0.95$.

**Step 3**: Find $z_{0.95}$ such that $\Phi(z_{0.95}) = 0.95$. From tables, $z_{0.95} \approx 1.645$.

**Step 4**: Unstandardize.

$$\frac{x - 100}{15} = 1.645 \implies x = 100 + 1.645 \times 15 = 124.675$$

**Step 5**: Interpret. An IQ of approximately 125 or higher is in the top 5% of the population.

### Example 3: Weight Initialization in Neural Networks

A neural network layer has 256 input neurons and 128 output neurons. Using Xavier initialization, weights are drawn from $\mathcal{N}(0, 2/(256 + 128)) = \mathcal{N}(0, 2/384)$. What proportion of weights exceeds 0.1 in magnitude?

**Step 1**: Standard deviation. $\sigma = \sqrt{2/384} \approx 0.0722$.

**Step 2**: Compute z-score for $x = 0.1$.

$$z = \frac{0.1 - 0}{0.0722} \approx 1.385$$

**Step 3**: Probability of a weight exceeding 0.1.

$$P(X > 0.1) = P(Z > 1.385) = 1 - \Phi(1.385) \approx 1 - 0.917 = 0.083$$

**Step 4**: Interpret. About 8.3% of weights will exceed 0.1 in magnitude. This is reasonable — Xavier initialization keeps weights small enough to prevent exploding activations but large enough for effective learning.

### Example 4: Gaussian Noise in Denoising Autoencoders

A denoising autoencoder adds Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.1^2)$ to input pixel values. Find the probability that the noise magnitude exceeds 0.2.

**Step 1**: Parameters. $\mu = 0$, $\sigma = 0.1$.

**Step 2**: Probability of $|\epsilon| > 0.2$.

$$P(|\epsilon| > 0.2) = P(\epsilon < -0.2) + P(\epsilon > 0.2)$$

Standardize: $z = 0.2 / 0.1 = 2$.

$$P(|\epsilon| > 0.2) = 2\Phi(-2) \approx 2 \times 0.02275 = 0.0455$$

**Step 3**: Interpret. Only about 4.55% of noise values exceed 0.2, so most noise is modest relative to the pixel range.

### Example 5: Confidence Interval for Regression Coefficients

A linear regression model estimates a coefficient $\hat{\beta}_1 = 0.75$ with standard error $\text{SE}(\hat{\beta}_1) = 0.25$. Construct a 95% confidence interval for the true $\beta_1$.

**Step 1**: The estimator $\hat{\beta}_1 \approx \mathcal{N}(\beta_1, 0.25^2)$.

**Step 2**: For a 95% CI, $z_{0.025} = 1.96$.

**Step 3**: Compute the interval.

$$\text{CI} = \hat{\beta}_1 \pm z_{0.025} \times \text{SE} = 0.75 \pm 1.96 \times 0.25 = 0.75 \pm 0.49$$

$$\text{CI} = (0.26, 1.24)$$

**Step 4**: Interpret. We are 95% confident that $\beta_1$ lies between 0.26 and 1.24. Since the interval does not include 0, the coefficient is statistically significant.

## Visual Interpretation

The Normal PDF is the iconic bell-shaped curve, characterized by:

- **Symmetry**: The curve is perfectly symmetric about $\mu$
- **Unimodality**: The curve has a single peak at $x = \mu$, where the density is highest
- **Inflection points**: The curve changes concavity at $x = \mu \pm \sigma$
- **Asymptotic tails**: The curve approaches but never touches the x-axis as $x \to \pm \infty$
- **Scale**: At $\mu \pm \sigma$, the density is approximately 60.7% of the peak height

The standard Normal PDF is $\phi(z) = e^{-z^2/2}/\sqrt{2\pi}$ with peak height approximately 0.399.

The empirical (68-95-99.7) rule is the most practical visual guide:
- 68% of the area lies within $\mu \pm \sigma$
- 95% of the area lies within $\mu \pm 2\sigma$ (more precisely, $\mu \pm 1.96\sigma$)
- 99.7% of the area lies within $\mu \pm 3\sigma$

## Common Mistakes

1. **Confusing the PDF with probability**: The PDF $f(x)$ is a density, not a probability. $P(X = x) = 0$ for any specific value $x$ (continuous distribution). Probabilities are only defined over intervals.

2. **Ignoring the sigma in the PDF denominator**: Forgetting the $1/\sigma$ factor leads to an incorrect density that does not integrate to 1. The full term $1/(\sigma\sqrt{2\pi})$ is necessary for normalization.

3. **Misapplying the 68-95-99.7 rule**: This rule applies to the Normal distribution only. Applying it to non-Normal data (e.g., heavy-tailed distributions) is incorrect.

4. **Assuming Normality without checking**: Many statistical tests assume Normality, but real data may be skewed or heavy-tailed. Always check with Q-Q plots, histograms, or normality tests (Shapiro-Wilk, Kolmogorov-Smirnov).

5. **Confusing sigma and sigma-squared**: The Normal distribution is parametrized by both $\mu$ (mean) and $\sigma^2$ (variance). The standard deviation is $\sigma$, and using $\sigma^2$ in place of $\sigma$ in formulas like standardization leads to errors.

6. **Standardization mistake**: When computing $Z = (X - \mu)/\sigma$, the denominator is the standard deviation $\sigma$, not the variance $\sigma^2$.

7. **Two-tailed vs one-tailed probabilities**: When computing $P(X > x)$, use $1 - \Phi(z)$. For $P(|X| > x)$, double the one-tailed probability.

8. **Forgetting the continuity correction**: When approximating discrete distributions (like Binomial) with the Normal, the continuity correction improves accuracy.

## Interview Questions

### Beginner

1. **Q**: What are the two parameters of a Normal distribution?
   **A**: The mean $\mu$ (location) and the variance $\sigma^2$ (or standard deviation $\sigma$, scale).

2. **Q**: Write the PDF of the standard Normal distribution.
   **A**: $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.

3. **Q**: What is the 68-95-99.7 rule?
   **A**: About 68% of data falls within 1 standard deviation, 95% within 2, and 99.7% within 3 standard deviations of the mean.

4. **Q**: How do you standardize a Normal random variable?
   **A**: $Z = (X - \mu) / \sigma$, which transforms $X \sim \mathcal{N}(\mu, \sigma^2)$ to $Z \sim \mathcal{N}(0, 1)$.

5. **Q**: What is $P(Z > 1.96)$ for a standard Normal?
   **A**: $P(Z > 1.96) \approx 0.025$.

### Intermediate

1. **Q**: Show that the Normal distribution belongs to the exponential family.
   **A**: $f(x; \mu, \sigma^2) = \exp\left(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right)$. Natural parameters: $\eta_1 = \mu/\sigma^2$, $\eta_2 = -1/(2\sigma^2)$. Sufficient statistics: $T_1(x) = x$, $T_2(x) = x^2$.

2. **Q**: What is the relationship between the Normal CDF $\Phi(z)$ and the error function $\text{erf}(z)$?
   **A**: $\Phi(z) = \frac{1}{2}[1 + \text{erf}(z/\sqrt{2})]$. Equivalently, $\text{erf}(z) = 2\Phi(z\sqrt{2}) - 1$.

3. **Q**: If $X \sim \mathcal{N}(0, 1)$, find $\mathbb{E}[X^4]$.
   **A**: $\mathbb{E}[X^4] = 3!! = 3 \times 1 = 3$.

4. **Q**: Derive the MLE of $\mu$ and $\sigma^2$ for a Normal distribution.
   **A**: Log-likelihood: $\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2$. Solving $\partial\ell/\partial\mu = 0$ gives $\hat{\mu} = \bar{x}$. Solving $\partial\ell/\partial\sigma^2 = 0$ gives $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ (note: this is biased; the unbiased version divides by $n-1$).

5. **Q**: In Gaussian processes, what is the role of the covariance function (kernel)?
   **A**: The covariance function $k(x, x')$ defines the prior over functions by specifying the covariance between function values at any two points. It encodes assumptions about smoothness, periodicity, and stationarity. Common kernels include RBF, Matérn, and periodic kernels.

### Advanced

1. **Q**: Prove that the Normal distribution maximizes entropy among all continuous distributions on $\mathbb{R}$ with given mean and variance.
   **A**: Use variational calculus. Maximize $H[f] = -\int f(x)\log f(x) dx$ subject to $\int f(x) dx = 1$, $\int x f(x) dx = \mu$, $\int (x-\mu)^2 f(x) dx = \sigma^2$. The Lagrangian yields $f(x) \propto \exp(-\lambda_1 x - \lambda_2 (x-\mu)^2)$, which simplifies to the Normal PDF after solving for the Lagrange multipliers.

2. **Q**: Derive Stein's lemma and explain its use in risk estimation and reinforcement learning.
   **A**: Stein's lemma: For $X \sim \mathcal{N}(\mu, \sigma^2)$ and differentiable $g$ with $\mathbb{E}[|g'(X)|] < \infty$, $\mathbb{E}[g(X)(X-\mu)] = \sigma^2\mathbb{E}[g'(X)]$. Proof via integration by parts. Used in Stein's unbiased risk estimate (SURE) for denoising and in policy gradient methods for reinforcement learning.

3. **Q**: In variational autoencoders, the latent space is typically assumed to follow $\mathcal{N}(0, I)$. Explain how the reparameterization trick enables gradient-based training and why the Normal distribution is the natural choice.
   **A**: The reparameterization trick expresses $z \sim \mathcal{N}(\mu, \sigma^2)$ as $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$. This allows gradients to flow through the sampling operation via $\mu$ and $\sigma$. The Normal is natural because: (1) it has maximum entropy for given mean and variance, (2) it is closed under linear transformations, (3) the KL divergence between two Normals has a closed form, enabling analytical computation of the VAE loss.

## Practice Problems

### Easy

1. If $X \sim \mathcal{N}(50, 10^2)$, find $P(X > 60)$.

2. For a standard Normal $Z$, find $P(-1.5 < Z < 1.5)$.

3. What is the z-score for $x = 75$ if $\mu = 60$ and $\sigma = 10$?

4. If $X \sim \mathcal{N}(100, 20^2)$, find $P(80 < X < 120)$.

5. For $Z \sim \mathcal{N}(0, 1)$, find $z$ such that $P(Z < z) = 0.90$.

### Medium

1. The heights of adult men follow $\mathcal{N}(175, 8^2)$ cm. Find the probability that a randomly selected man is taller than 185 cm.

2. A test score distribution is $\mathcal{N}(500, 100^2)$. What score corresponds to the 95th percentile?

3. Derive the moment-generating function of $\mathcal{N}(\mu, \sigma^2)$ and use it to find $\mathbb{E}[X^3]$.

4. The sample mean $\bar{X}$ of $n = 25$ i.i.d. $\mathcal{N}(10, 4^2)$ observations follows what distribution? Find $P(\bar{X} > 11)$.

5. A 95% confidence interval for $\mu$ based on a sample of size $n = 16$ from $\mathcal{N}(\mu, \sigma^2)$ is $(12.5, 17.5)$. Find $\bar{x}$ and the margin of error.

### Hard

1. Let $X_1, \ldots, X_n$ be i.i.d. $\mathcal{N}(\mu, \sigma^2)$. Derive the joint distribution of $\bar{X}$ and $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$. Show that $\bar{X}$ and $S^2$ are independent.

2. Prove that for a Normal sample, the sample mean $\bar{X}$ is the UMVUE of $\mu$.

3. In a Gaussian process regression model with RBF kernel $k(x, x') = \sigma_f^2 \exp(-(x-x')^2/(2\ell^2))$, explain the role of $\sigma_f^2$ (signal variance) and $\ell$ (length scale). How does the posterior mean behave as $\ell \to 0$ and $\ell \to \infty$?

## Solutions

### Easy Solutions

1. $z = (60-50)/10 = 1$. $P(X > 60) = P(Z > 1) = 1 - \Phi(1) \approx 0.1587$.

2. $P(-1.5 < Z < 1.5) = \Phi(1.5) - \Phi(-1.5) = 2\Phi(1.5) - 1 \approx 2(0.9332) - 1 = 0.8664$.

3. $z = (75-60)/10 = 1.5$.

4. $z_1 = (80-100)/20 = -1$, $z_2 = (120-100)/20 = 1$. $P = \Phi(1) - \Phi(-1) \approx 0.6826$.

5. $z_{0.90} \approx 1.282$.

### Medium Solutions

1. $z = (185-175)/8 = 1.25$. $P = 1 - \Phi(1.25) \approx 1 - 0.8944 = 0.1056$.

2. $z_{0.95} \approx 1.645$. Score $= 500 + 1.645 \times 100 = 664.5$.

3. $M_X(t) = \exp(\mu t + \sigma^2 t^2/2)$. Third derivative at $t = 0$: $M_X'''(0) = \mathbb{E}[X^3] = \mu^3 + 3\mu\sigma^2$.

4. $\bar{X} \sim \mathcal{N}(10, 4^2/25) = \mathcal{N}(10, 0.8^2)$. $z = (11-10)/0.8 = 1.25$. $P(\bar{X} > 11) = P(Z > 1.25) \approx 0.1056$.

5. $\bar{x} = (12.5 + 17.5)/2 = 15$. Margin of error $= (17.5 - 12.5)/2 = 2.5$.

### Hard Solutions

1. By Cochran's theorem, $\bar{X}$ and $S^2$ are independent for Normal samples. $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$ and $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$. The independence follows from the fact that $\bar{X}$ is a function of the projection onto the equiangular line, while $S^2$ depends on the orthogonal complement.

2. $\bar{X}$ is unbiased and a function of the complete sufficient statistic $T = \sum X_i$. By the Lehmann-Scheffé theorem, $\bar{X}$ is the UMVUE of $\mu$.

3. $\sigma_f^2$ controls the vertical scale (prior variance) of the function. $\ell$ controls the smoothness: small $\ell$ means rapid variation (short-range correlation), large $\ell$ means very smooth functions (long-range correlation). As $\ell \to 0$, the kernel becomes a delta function and predictions default to the prior mean (no learning from data). As $\ell \to \infty$, function values become perfectly correlated (linear behaviour), and the GP reduces to a linear model.

## Related Concepts

- **Standard Normal Distribution**: The Normal with $\mu = 0$, $\sigma = 1$, serving as the reference distribution.
- **Central Limit Theorem (MATH-076)**: The CLT explains why the Normal distribution appears so frequently as a limiting distribution.
- **Multivariate Normal Distribution**: Generalization to vectors, used in GMMs, factor analysis, and GP regression.
- **Binomial Distribution (MATH-073)**: The Normal approximates the Binomial for large $n$.
- **Poisson Distribution (MATH-074)**: The Normal approximates the Poisson for large $\lambda$.
- **Chi-squared Distribution**: The sum of squared standard Normals follows a chi-squared distribution.
- **Student's t-Distribution**: Used for inference on the mean when $\sigma$ is unknown.
- **Error Function (erf)**: Directly related to the Normal CDF.
- **Gaussian Processes**: Non-parametric Bayesian models using the multivariate Normal.

## Next Concepts

- Central Limit Theorem (MATH-076)
- Multivariate Normal Distribution
- Gaussian Processes
- Bayesian Linear Regression
- Gaussian Mixture Models
- Variational Autoencoders

## Summary

The Normal (Gaussian) distribution is the most important continuous probability distribution in statistics and machine learning. Its PDF is $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-(x-\mu)^2/(2\sigma^2))$, with mean $\mu$ and variance $\sigma^2$. The distribution is symmetric, unimodal, and characterized by the 68-95-99.7 empirical rule. Standardization via $Z = (X-\mu)/\sigma$ transforms any Normal to the standard Normal. The Normal distribution appears throughout machine learning: in weight initialization, noise models, Gaussian processes, linear regression assumptions, variational autoencoders, and as the limiting distribution in the Central Limit Theorem.

## Key Takeaways

- The Normal distribution is defined by mean $\mu$ and variance $\sigma^2$
- PDF: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp(-(x-\mu)^2/(2\sigma^2))$
- Standardization: $Z = (X-\mu)/\sigma \sim \mathcal{N}(0, 1)$
- 68-95-99.7 rule: approximately 68%, 95%, and 99.7% of mass within $1\sigma$, $2\sigma$, and $3\sigma$
- Linear transformations and sums of independent Normals preserve Normality
- The Normal has maximum entropy among all distributions with given mean and variance
- It is the limiting distribution in the Central Limit Theorem
- Used for weight initialization, noise models, GP regression, MSE loss, and variational inference in ML
- The CDF relates to the error function: $\Phi(z) = \frac{1}{2}[1 + \text{erf}(z/\sqrt{2})]$
- MLE of $\mu$ is $\bar{x}$; MLE of $\sigma^2$ is $\frac{1}{n}\sum(x_i - \bar{x})^2$
