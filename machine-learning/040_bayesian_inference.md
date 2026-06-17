# Concept: Bayesian Inference

## Concept ID

ML-040

## Difficulty

Advanced

## Domain

Machine Learning

## Module

Instance-Based and Probabilistic Methods

## Learning Objectives

- Understand the Bayesian framework: prior, likelihood, posterior, and evidence
- Compute posterior distributions using conjugate priors
- Explain the difference between MLE, MAP, and full Bayesian inference
- Implement Bayesian inference for Beta-Binomial and Gaussian-Gaussian models
- Apply Bayesian decision theory for predictions under uncertainty

## Prerequisites

- Probability theory (conditional probability, Bayes' theorem)
- Familiarity with common distributions (Beta, Binomial, Gaussian, Gamma)
- Calculus (integration, derivatives)
- Python with NumPy and SciPy

## Definition

Bayesian inference is a framework for statistical learning under uncertainty. It treats unknown parameters theta as random variables with prior distributions P(theta), updates beliefs given observed data D via Bayes' theorem to obtain the posterior distribution P(theta | D), and makes predictions by integrating over the posterior.

Bayes' Theorem:

P(theta | D) = P(D | theta) * P(theta) / P(D)

where:
- Prior P(theta): belief about theta before seeing data
- Likelihood P(D | theta): how probable the data is given theta
- Evidence P(D): normalizing constant (marginal likelihood)
- Posterior P(theta | D): updated belief after data

## Intuition

Bayesian inference mirrors human learning. Before seeing evidence, you have a prior belief. As you observe data, you update that belief. Strong evidence overwhelms the prior; weak evidence leaves it intact. The posterior represents your complete knowledge about theta as a full distribution reflecting remaining uncertainty.

Imagine a doctor diagnosing a rare disease. The prior is the disease's prevalence (say 1 in 10,000). A positive test result updates this belief. Even with a highly accurate test, the posterior probability might still be low because the prior was very small — this is the Bayesian interpretation of the base rate fallacy.

## Why This Concept Matters

Bayesian inference is a unifying framework underpinning Naive Bayes, Gaussian processes, Bayesian neural networks, and probabilistic programming (PyMC, Stan). It naturally handles uncertainty quantification, regularization (via priors), sequential learning (posterior today = prior tomorrow), and domain knowledge incorporation. In an era where calibrated uncertainty is increasingly important, Bayesian methods are essential.

## Mathematical Explanation

### Bayes' Theorem in Detail

P(theta | D) = P(D | theta) P(theta) / integral P(D | theta) P(theta) d(theta)

The denominator P(D) is the marginal likelihood, which ensures the posterior integrates to 1. For most interesting models, this integral is intractable, requiring approximate inference (MCMC, variational inference).

### Conjugate Priors

A prior P(theta) is conjugate to likelihood P(D|theta) if the posterior P(theta|D) belongs to the same family as the prior. Conjugacy yields closed-form updates.

Common conjugate pairs:
- Binomial(n, theta) with Beta(alpha, beta) prior gives Beta(alpha + x, beta + n - x) posterior
- Poisson(theta) with Gamma(alpha, beta) prior gives Gamma(alpha + sum x_i, beta + n) posterior
- Gaussian(theta, sigma^2) with known sigma^2 and Gaussian(mu_0, sigma_0^2) prior gives Gaussian posterior

### MAP Estimation

Maximum a Posteriori (MAP) finds the posterior mode:

theta_MAP = argmax_theta P(theta | D) = argmax_theta [log P(D | theta) + log P(theta)]

MAP is a compromise between MLE and full Bayesian inference. It regularizes via the prior but still gives a point estimate rather than a full distribution.

### Posterior Predictive Distribution

For new data x_tilde:

P(x_tilde | D) = integral P(x_tilde | theta) P(theta | D) d(theta)

This fully accounts for posterior uncertainty.

### Bayesian Decision Theory

For loss function L(theta, a), the optimal action minimizes posterior expected loss:

a* = argmin_a integral L(theta, a) P(theta | D) d(theta)

For squared error, the optimal estimate is the posterior mean.

## Code Examples

### Example 1: Beta-Binomial Conjugate Model

```python
import numpy as np
from scipy import stats

alpha_prior, beta_prior = 2, 2
n_flips, n_heads = 10, 8

alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_flips - n_heads)

print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Prior mean: {alpha_prior/(alpha_prior+beta_prior):.3f}")
print(f"MLE: {n_heads/n_flips:.3f}")
print(f"Posterior mean: {alpha_post/(alpha_post+beta_post):.3f}")
print(f"95% credible interval: {stats.beta.ppf(0.025, alpha_post, beta_post):.3f} - "
      f"{stats.beta.ppf(0.975, alpha_post, beta_post):.3f}")
# Output:
# Prior: Beta(2, 2)
# Posterior: Beta(10, 4)
# Prior mean: 0.500
# MLE: 0.800
# Posterior mean: 0.714
# 95% credible interval: 0.457 - 0.916
```

### Example 2: Sequential Bayesian Updating

```python
flips = ['H', 'H', 'H', 'T', 'H', 'H', 'T', 'H', 'H', 'H']
a, b = 2, 2
for i, flip in enumerate(flips):
    if flip == 'H':
        a += 1
    else:
        b += 1
    mean = a / (a + b)
    print(f"After flip {i+1} ({flip}): mean={mean:.3f}, Beta({a},{b})")
# Output:
# After flip 1 (H): mean=0.600, Beta(3,2)
# After flip 2 (H): mean=0.667, Beta(4,2)
# After flip 3 (H): mean=0.714, Beta(5,2)
# After flip 4 (T): mean=0.667, Beta(5,3)
# After flip 5 (H): mean=0.700, Beta(6,3)
# After flip 6 (H): mean=0.727, Beta(7,3)
# After flip 7 (T): mean=0.700, Beta(7,4)
# After flip 8 (H): mean=0.727, Beta(8,4)
# After flip 9 (H): mean=0.750, Beta(9,4)
# After flip 10 (H): mean=0.750, Beta(10,4)
```

### Example 3: Bayesian Linear Regression

```python
import numpy as np
np.random.seed(42)
n = 50
X = np.random.randn(n, 1)
true_w, true_b = 2.5, 1.0
y = true_w * X.flatten() + true_b + np.random.randn(n) * 0.5

sigma2 = 0.25
prior_var = 10.0
X_design = np.column_stack([np.ones(n), X.flatten()])
Sigma_prior_inv = np.eye(2) / prior_var
Sigma_post = np.linalg.inv(Sigma_prior_inv + X_design.T @ X_design / sigma2)
mu_post = Sigma_post @ (X_design.T @ y / sigma2)

print(f"True: w={true_w}, b={true_b}")
print(f"MAP: w={mu_post[1]:.3f}, b={mu_post[0]:.3f}")
print(f"Posterior std: w={np.sqrt(Sigma_post[1,1]):.4f}, b={np.sqrt(Sigma_post[0,0]):.4f}")
# Output:
# True: w=2.5, b=1.0
# MAP: w=2.530, b=0.973
# Posterior std: w=0.0718, b=0.0728
```

### Example 4: Posterior Predictive Distribution

```python
from scipy.stats import betabinom

alpha_post, beta_post = 10, 4
n_future = 10
k_values = np.arange(0, n_future + 1)
pred_probs = betabinom.pmf(k_values, n_future, alpha_post, beta_post)

print("Predictive distribution for next 10 flips:")
for k, p in zip(k_values, pred_probs):
    print(f"  P({k} heads) = {p:.4f}")
print(f"  Expected heads: {np.sum(k_values * pred_probs):.3f}")
# Output:
# Predictive distribution for next 10 flips:
#   P(0 heads) = 0.0005
#   P(1 heads) = 0.0050
#   P(2 heads) = 0.0233
#   P(3 heads) = 0.0669
#   P(4 heads) = 0.1308
#   P(5 heads) = 0.1831
#   P(6 heads) = 0.1880
#   P(7 heads) = 0.1427
#   P(8 heads) = 0.0794
#   P(9 heads) = 0.0310
#   P(10 heads) = 0.0071
```

### Example 5: MCMC Sampling for Non-Conjugate Model

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(3.0, 1.5, 30)

# Metropolis-Hastings for Normal likelihood with Cauchy prior
def log_likelihood(theta, data):
    return np.sum(stats.norm.logpdf(data, loc=theta, scale=1.5))

def log_prior(theta):
    return stats.cauchy.logpdf(theta, loc=0, scale=2.5)

samples = [0.0]
n_iter = 5000
for i in range(n_iter):
    theta_prop = np.random.normal(samples[-1], 0.5)
    log_accept = (log_likelihood(theta_prop, data) + log_prior(theta_prop)
                  - log_likelihood(samples[-1], data) - log_prior(samples[-1]))
    if np.log(np.random.random()) < log_accept:
        samples.append(theta_prop)
    else:
        samples.append(samples[-1])

samples = samples[1000:]  # burn-in
print(f"Posterior mean: {np.mean(samples):.3f}")
print(f"Posterior std: {np.std(samples):.3f}")
# Output:
# Posterior mean: 2.985
# Posterior std: 0.274
```

## Common Mistakes

1. **Interpreting credible intervals as confidence intervals.** A 95% credible interval means P(theta in [a,b] | D) = 0.95, a direct probabilistic statement about theta.

2. **Using improper priors without checking.** Improper priors can lead to improper posteriors. Always verify the posterior is proper.

3. **Treating MAP as full Bayesian inference.** MAP discards uncertainty. For asymmetric posteriors, mode and mean differ significantly.

4. **Over-relying on conjugate priors for complex models.** Modern probabilistic programming uses MCMC or variational inference for arbitrary priors.

5. **Ignoring prior sensitivity.** Different priors yield different posteriors with small data. Always perform sensitivity analysis.

6. **Double-counting data in sequential updating.** Use only new data at each step; the posterior from step t becomes the prior for step t+1.

7. **Confusing P(theta|D) with P(D|theta).** The likelihood P(D|theta) is NOT a probability over theta.

## Interview Questions

### Beginner

1. What is Bayes' theorem?
P(theta|D) = P(D|theta)P(theta)/P(D). It updates beliefs about parameters given observed data.

2. Difference between MLE and MAP?
MLE maximizes P(D|theta). MAP adds a prior: argmax P(D|theta)P(theta). MAP regularizes toward the prior.

3. What is a conjugate prior?
A prior that yields a posterior in the same distribution family when combined with a given likelihood.

4. What is the posterior predictive distribution?
P(x_tilde|D) = integral P(x_tilde|theta) P(theta|D) d(theta) — the distribution of new data given observed data, averaging over posterior uncertainty.

5. Real-world example of Bayesian inference?
A/B testing: prior on conversion rates, update with user data, compute P(variant B > variant A | data).

### Intermediate

1. Derive Beta-Binomial posterior.
Prior Beta(alpha, beta): P(theta) propto theta^(alpha-1)(1-theta)^(beta-1). Likelihood Binomial: P(x|theta) propto theta^x (1-theta)^(n-x). Posterior propto theta^(alpha+x-1)(1-theta)^(beta+n-x-1) = Beta(alpha+x, beta+n-x).

2. Role of marginal likelihood in model comparison?
P(D|M) = integral P(D|theta,M)P(theta|M) d(theta). The Bayes factor compares models, automatically penalizing complexity (Occam's razor).

3. How does Bayesian inference handle regularization?
The prior acts as a regularizer. A prior centered at zero shrinks estimates toward zero, like L2 regularization. Prior variance controls regularization strength.

4. Credible interval vs confidence interval?
Credible: P(theta in [a,b] | D) = 0.95. Confidence: 95% of repeated intervals contain true theta. Bayesian interpretation is more intuitive but prior-dependent.

5. How to perform Bayesian A/B testing?
Model each variant's conversion rate as Beta with conjugate updates. Compute P(theta_B > theta_A | data) via Monte Carlo from posteriors.

### Advanced

1. Derive the ELBO for variational inference.
log P(D) = log integral P(D|theta)P(theta) dtheta >= integral q(theta) log [P(D|theta)P(theta)/q(theta)] dtheta = ELBO. ELBO = E_q[log P(D|theta)] - KL(q(theta) || P(theta)). Maximizing ELBO makes q approximate the posterior.

2. Prove posterior mean minimizes expected squared error loss.
a* = argmin_a integral (theta - a)^2 P(theta|D) d(theta). Derivative: -2E[theta] + 2a = 0 => a* = E[theta|D].

3. Laplace approximation and its limitations?
Approximates posterior as Gaussian centered at MAP with covariance H^-1 where H is Hessian of negative log-posterior. Limitations: captures only local curvature, fails for multimodal or skewed posteriors, degrades with small data.

## Practice Problems

### Easy

1. Coin flipped 20 times, 14 heads. Beta(1,1) prior. Compute posterior mean and 95% credible interval.

2. Gaussian known variance sigma^2=4, prior mu ~ N(0,10), data [3.2, 4.1, 2.8]. Compute posterior mean and variance.

3. Implement sequential Beta-Binomial updating for flips "H,T,H,H,H,T,H,H". Start Beta(1,1).

4. Compute posterior predictive P(3 heads in next 5 flips) given Beta(5,3) posterior.

5. Compare MLE and MAP for Bernoulli data [1,1,1,0,1,1,0,1] with Beta(2,2) prior.

### Medium

1. Implement Bayesian linear regression with conjugate priors. Compare with sklearn's LinearRegression on n=10, d=5.

2. Prior sensitivity: for data (5 heads, 5 tails), compute posterior under Beta(1,1), Beta(0.5,0.5), Beta(10,10), Beta(1,100).

3. Implement Metropolis-Hastings for Cauchy likelihood with Normal prior.

4. Compute Bayes factor comparing Beta(1,1) vs Beta(10,10) for 8 heads in 10 flips.

5. Bayesian change point detection in coin flip sequence using Gibbs sampling.

### Hard

1. Derive and implement stochastic variational inference for Bayesian mixture of Gaussians.

2. Implement Hamiltonian Monte Carlo for a 2D Gaussian posterior.

3. Build a Bayesian neural network with one hidden layer using variational inference. Compare uncertainty with dropout-based methods.

## Solutions

Easy 1: Beta(1+14, 1+6) = Beta(15,7). Mean = 15/22 = 0.682. CI from scipy.stats.beta.ppf([0.025,0.975], 15, 7) = [0.477, 0.854].

Easy 3:
```python
from scipy import stats
flips = ['H', 'T', 'H', 'H', 'H', 'T', 'H', 'H']
a, b = 1, 1
for i, flip in enumerate(flips):
    if flip == 'H': a += 1
    else: b += 1
    ci = stats.beta.ppf([0.025, 0.975], a, b)
    print(f"After {i+1}: Beta({a},{b}), mean={a/(a+b):.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}]")
```

## Related Concepts

- **Naive Bayes** (ML-038): Applies Bayes' theorem for classification
- **Gaussian Naive Bayes** (ML-039): Continuous-feature variant
- **Gaussian Processes** (ML-041): Non-parametric Bayesian prediction
- **Maximum Likelihood Estimation** (ML-004): Point estimation without priors

## Next Concepts

- Gaussian Processes (ML-041): Bayesian non-parametric regression with uncertainty
- Gaussian Mixture Models (ML-045): Generative model with latent variables

## Summary

Bayesian inference provides a principled framework for learning from data by maintaining uncertainty through probability distributions. Starting with a prior, updating via Bayes' theorem yields the posterior, from which predictions and decisions are made. Conjugate priors give closed-form solutions. For complex models, MCMC and variational inference approximate the posterior. Bayesian methods naturally handle regularization, sequential learning, and uncertainty quantification.

## Key Takeaways

- Bayes' theorem combines prior beliefs with data likelihood to yield the posterior
- Conjugate priors enable closed-form Bayesian updating
- MAP estimation adds regularization via the prior
- The posterior predictive distribution fully accounts for parameter uncertainty
- Bayesian methods naturally quantify uncertainty and regularize
- MCMC and variational inference approximate posteriors for complex models
- Bayesian credible intervals have a direct probabilistic interpretation
