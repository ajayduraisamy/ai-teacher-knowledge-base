# Concept: KL Divergence

## Concept ID

MATH-090

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Information Theory

## Learning Objectives

- Define Kullback-Leibler (KL) divergence as a measure of how one probability distribution diverges from another
- Understand why $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ (asymmetry) and its implications
- Prove and apply Gibbs' inequality: $D_{KL}(P\|Q) \geq 0$ with equality iff $P = Q$
- Relate KL divergence to entropy and cross entropy via $D_{KL}(P\|Q) = H(P,Q) - H(P)$
- Apply KL divergence in variational inference, knowledge distillation, and model compression

## Prerequisites

- Entropy (MATH-088)
- Cross Entropy (MATH-089)
- Probability (MATH-065)
- Bayes Theorem (MATH-069)
- Jensen's inequality and convexity of $-\log x$
- Basic calculus (differentiation and integration)

## Definition

The **Kullback-Leibler divergence** (also called **relative entropy**) measures how one probability distribution $Q$ diverges from a reference distribution $P$. For two discrete probability distributions $P$ and $Q$ over the same sample space $\mathcal{X}$, it is defined as:

$$
D_{KL}(P \| Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
$$

For continuous distributions with densities $p$ and $q$:

$$
D_{KL}(p \| q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} \, dx
$$

Conventions: $0 \log \frac{0}{0} = 0$, $0 \log \frac{0}{q} = 0$, $p \log \frac{p}{0} = \infty$ (if $p > 0$). This means KL divergence is infinite if $Q$ assigns zero probability to an outcome that $P$ considers possible.

KL divergence is not a true distance metric because it is not symmetric and does not satisfy the triangle inequality. It is best understood as a directed measure of divergence or "information gain" when moving from $Q$ to $P$.

## Intuition

KL divergence answers: "How surprised would I be to see data from $P$ if I believed $Q$?" Or equivalently: "How many extra bits of information are needed to encode samples from $P$ when using a code designed for $Q$?"

If $P$ and $Q$ are identical, KL divergence is zero — no extra information is needed. If they differ, the divergence grows. When $Q$ assigns very low probability to outcomes that $P$ considers likely, the divergence becomes large (potentially infinite).

The asymmetry $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ is intuitive: the cost of mistakenly believing $Q$ when reality is $P$ differs from the cost of believing $P$ when reality is $Q$. In the coding interpretation, $D_{KL}(P\|Q)$ is the extra bits when the true distribution is $P$ but the code is optimised for $Q$. Reversing the roles gives a different number.

Think of KL divergence as a "distance" that is highly sensitive to differences in the tails of distributions: $D_{KL}(P\|Q)$ blows up if $Q$ has thin tails relative to $P$ (because $Q(x) \to 0$ while $P(x) > 0$), while $D_{KL}(Q\|P)$ blows up if $P$ has thin tails relative to $Q$.

## Why This Concept Matters

KL divergence is arguably the most important tool for comparing probability distributions in machine learning:

- **Variational inference:** The entire field of variational Bayes is built on minimising $D_{KL}(Q\|P)$ where $Q$ is an approximating distribution and $P$ is the true posterior. The evidence lower bound (ELBO) is derived from this.
- **Model compression:** Knowledge distillation trains a student model to match a teacher's output distribution by minimising their KL divergence.
- **Generative modelling:** Variational autoencoders (VAEs) use KL divergence to regularise the latent space. Generative adversarial networks (GANs) relate to Jensen-Shannon divergence, a symmetrised variant.
- **Information theory:** KL divergence is the fundamental measure of "distance" between distributions in information theory. It appears in Sanov's theorem, the method of types, and large deviations theory.
- **Statistical inference:** The likelihood ratio test statistic is asymptotically distributed as $\chi^2$ under the null, and the large-deviations rate function is a KL divergence.
- **Reinforcement learning:** Trust region policy optimisation (TRPO) and proximal policy optimisation (PPO) constrain policy updates using KL divergence.

## Historical Background

**Solomon Kullback** (1907–1994) and **Richard Leibler** (1914–2003) introduced the divergence measure in their 1951 paper "On Information and Sufficiency," published in the Annals of Mathematical Statistics. Both worked at the National Security Agency (NSA) and developed the concept for cryptographic applications — specifically, for distinguishing encrypted messages from random noise.

The concept was independently known by others: **Harold Jeffreys** had used the symmetrised version (now called Jensen-Shannon divergence) in 1946. Andrey Kolmogorov also recognised its importance. **I.J. Good** (who worked with Turing at Bletchley Park) connected KL divergence to Bayesian inference.

Kullback wrote the influential book "Information Theory and Statistics" (1959), establishing KL divergence as a fundamental tool for statistical inference: likelihood ratio tests, sufficient statistics, and the information inequality.

The measure has many names: KL divergence, Kullback-Leibler distance (a misnomer, as it is not a metric), relative entropy, information divergence, and discrimination information.

## Real World Examples

1. **Variational autoencoders:** A VAE learns a latent representation by minimising $D_{KL}(Q(z|x)\|P(z))$, where $Q(z|x)$ is the encoder's approximate posterior and $P(z)$ is the prior (typically a standard normal). This regularises the latent space to match the prior.

2. **Knowledge distillation:** A large teacher network produces soft probabilities $\hat{y}_{\text{teacher}}$ over classes. A smaller student network minimises $D_{KL}(\hat{y}_{\text{teacher}}\|\hat{y}_{\text{student}})$ to mimic the teacher's behaviour, often with a temperature parameter to soften the distributions.

3. **Spam filtering:** A spam classifier learns $P(\text{spam}|\text{words})$. The KL divergence between word distributions in spam vs. non-spam emails identifies the most discriminative words. Words with high $D_{KL}(\text{word distribution}_{\text{spam}}\|\text{word distribution}_{\text{non-spam}})$ are strong indicators.

4. **A/B testing:** When comparing two conversion rate distributions, KL divergence quantifies how different the two experimental conditions are. A/B test results with high KL divergence indicate practically meaningful differences.

5. **Model selection:** The Akaike Information Criterion (AIC) is derived from KL divergence. The expected KL divergence between the true model and the estimated model is approximated by AIC = $-2\log L + 2k$, where $k$ is the number of parameters.

## AI/ML Relevance

**Variational inference and ELBO:** Given a probabilistic model with latent variables $z$ and observed data $x$, the true posterior $P(z|x)$ is often intractable. We approximate it with a variational distribution $Q(z)$ by minimising $D_{KL}(Q(z)\|P(z|x))$. This is intractable directly, so we maximise the ELBO:

$$
\log P(x) - D_{KL}(Q(z)\|P(z|x)) = \mathbb{E}_{z\sim Q}[\log P(x|z)] - D_{KL}(Q(z)\|P(z)) = \text{ELBO}
$$

Maximising the ELBO is equivalent to minimising $D_{KL}(Q(z)\|P(z|x))$, making KL divergence the core of variational inference.

**Knowledge distillation loss:** The student model minimises:
$$
\mathcal{L}_{\text{KD}} = \alpha \cdot H(y, \hat{y}_{\text{student}}) + \beta \cdot D_{KL}(\hat{y}_{\text{teacher}}^\tau \| \hat{y}_{\text{student}}^\tau)
$$
where $\tau$ is a temperature that softens the distributions, $H$ is cross-entropy with hard labels, and $\alpha, \beta$ balance the losses.

**Policy gradients and TRPO:** Trust region policy optimisation constrains the policy update such that $D_{KL}(\pi_{\text{old}}\|\pi_{\text{new}}) \leq \delta$, ensuring stable learning. This is a KL-constrained optimisation.

**Bayesian deep learning:** KL divergence arises naturally in Bayesian neural networks where we approximate the true weight posterior with a simpler distribution. The variational loss is $\mathcal{L} = -\mathbb{E}_{Q}[\log P(D|w)] + D_{KL}(Q(w)\|P(w))$.

**Information bottleneck:** The information bottleneck method finds a representation $Z$ that maximises $I(Z;Y)$ while minimising $I(Z;X)$, using KL divergence to measure these mutual informations. The trade-off is $\min I(X;Z) - \beta I(Z;Y)$.

## Mathematical Explanation

**Relation to entropy and cross entropy:**
$$
D_{KL}(P\|Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)} = \sum_i P(i)\log P(i) - \sum_i P(i)\log Q(i) = -H(P) + H(P, Q)
$$

So $D_{KL}(P\|Q) = H(P,Q) - H(P)$. This confirms the intuition: KL divergence is the "extra bits" beyond the optimal encoding.

**Gibbs' inequality:** $D_{KL}(P\|Q) \geq 0$, with equality iff $P = Q$.

Proof using Jensen's inequality: Since $-\log x$ is convex,
$$
D_{KL}(P\|Q) = \sum_i P(i) \left(-\log\frac{Q(i)}{P(i)}\right) = \mathbb{E}_{P}\left[-\log\frac{Q}{P}\right] \geq -\log\mathbb{E}_{P}\left[\frac{Q}{P}\right] = -\log\sum_i P(i)\frac{Q(i)}{P(i)} = -\log\sum_i Q(i) = -\log 1 = 0
$$

Equality holds when the function $-\log x$ is affine on the support of $P$, i.e., when $Q/P$ is constant, which means $P = Q$.

**Asymmetry:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ generally. For example, if $P = [1, 0]$ and $Q = [0.5, 0.5]$, then:
- $D_{KL}(P\|Q) = 1\log\frac{1}{0.5} + 0\log\frac{0}{0.5} = 1\cdot 1 = 1$ nat.
- $D_{KL}(Q\|P) = 0.5\log\frac{0.5}{1} + 0.5\log\frac{0.5}{0} = 0.5(-0.693) + \infty = \infty$.

The asymmetry reflects the different penalties: $D_{KL}(P\|Q)$ heavily penalises $Q$ underestimating $P$'s support.

**Chain rule (joint distributions):**
$$
D_{KL}(P(x,y)\|Q(x,y)) = D_{KL}(P(x)\|Q(x)) + D_{KL}(P(y|x)\|Q(y|x))
$$

**Data processing inequality:** For a Markov chain $X \to Y \to Z$,
$$
D_{KL}(P(x)\|Q(x)) \geq D_{KL}(P(y)\|Q(y)) \geq D_{KL}(P(z)\|Q(z))
$$

Processing can only reduce the ability to distinguish between distributions.

**Jensen-Shannon divergence:** A symmetrised and bounded version:
$$
D_{JS}(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)
$$
where $M = (P + Q)/2$. $D_{JS}$ is bounded by $\log 2$ (or 1 nat). Its square root is a true metric.

## Formula(s)

**KL divergence (discrete):**
$$
D_{KL}(P \| Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
$$

**KL divergence (continuous):**
$$
D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

**Relation to cross entropy:**
$$
D_{KL}(P\|Q) = H(P, Q) - H(P)
$$

**ELBO decomposition:**
$$
\log P(x) = \text{ELBO} + D_{KL}(Q(z)\|P(z|x))
$$

**Gaussian KL divergence (closed form):**
For $P = \mathcal{N}(\mu_1, \sigma_1^2)$ and $Q = \mathcal{N}(\mu_2, \sigma_2^2)$:
$$
D_{KL}(P\|Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

For multivariate Gaussians $P = \mathcal{N}(\mu_1, \Sigma_1)$, $Q = \mathcal{N}(\mu_2, \Sigma_2)$:
$$
D_{KL}(P\|Q) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T\Sigma_2^{-1}(\mu_2 - \mu_1)\right]
$$

## Properties

- **Non-negativity:** $D_{KL}(P\|Q) \geq 0$, with equality iff $P = Q$ (Gibbs' inequality).
- **Asymmetry:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general.
- **No triangle inequality:** KL divergence does not satisfy the triangle inequality and is therefore not a metric.
- **Additivity for independent distributions:** If $P(x,y) = P(x)P(y)$ and $Q(x,y) = Q(x)Q(y)$, then $D_{KL}(P\|Q) = D_{KL}(P_x\|Q_x) + D_{KL}(P_y\|Q_y)$.
- **Chain rule:** $D_{KL}(P(x,y)\|Q(x,y)) = D_{KL}(P(x)\|Q(x)) + D_{KL}(P(y|x)\|Q(y|x))$.
- **Convexity:** $D_{KL}(P\|Q)$ is convex in the pair $(P, Q)$.
- **Invariance under reparameterisation:** KL divergence is invariant under smooth, invertible transformations of the sample space.
- **Infinite penalty for zero:** If $Q(x)=0$ but $P(x)>0$, then $D_{KL}(P\|Q) = \infty$. This is the "zero-forcing" property.
- **Lower semicontinuity:** KL divergence is lower semicontinuous in the topology of weak convergence.

## Step-by-Step Worked Examples

### Example 1: KL Divergence Between Two Binary Distributions

**Problem:** Compute $D_{KL}(P\|Q)$ and $D_{KL}(Q\|P)$ for $P = [0.6, 0.4]$ and $Q = [0.8, 0.2]$.

**Solution:**

Step 1: $D_{KL}(P\|Q)$.
$$
D_{KL}(P\|Q) = 0.6\log\frac{0.6}{0.8} + 0.4\log\frac{0.4}{0.2}
$$

Using natural logs:
- $\log\frac{0.6}{0.8} = \log 0.75 \approx -0.2877$
- $\log\frac{0.4}{0.2} = \log 2 \approx 0.6931$

$$
D_{KL}(P\|Q) = 0.6(-0.2877) + 0.4(0.6931) = -0.1726 + 0.2772 = 0.1046 \text{ nats}
$$

Step 2: $D_{KL}(Q\|P)$.
$$
D_{KL}(Q\|P) = 0.8\log\frac{0.8}{0.6} + 0.2\log\frac{0.2}{0.4}
$$
- $\log\frac{0.8}{0.6} = \log 1.333 \approx 0.2877$
- $\log\frac{0.2}{0.4} = \log 0.5 \approx -0.6931$

$$
D_{KL}(Q\|P) = 0.8(0.2877) + 0.2(-0.6931) = 0.2302 - 0.1386 = 0.0916 \text{ nats}
$$

**Answer:** $D_{KL}(P\|Q) \approx 0.105$ nats, $D_{KL}(Q\|P) \approx 0.092$ nats. They are different, confirming asymmetry.

### Example 2: KL Divergence Between Two Gaussians

**Problem:** Compute $D_{KL}(P\|Q)$ where $P = \mathcal{N}(0, 1)$ and $Q = \mathcal{N}(1, 2)$ (standard normal vs. normal with $\mu=1$, $\sigma=2$).

**Solution:**

Step 1: Apply the Gaussian KL divergence formula.
$$
D_{KL}(P\|Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

Step 2: Substitute $\mu_1 = 0$, $\sigma_1 = 1$, $\mu_2 = 1$, $\sigma_2 = 2$.
$$
D_{KL}(P\|Q) = \log\frac{2}{1} + \frac{1^2 + (0 - 1)^2}{2 \cdot 2^2} - \frac{1}{2}
$$

Step 3: Compute each term.
- $\log\frac{2}{1} = \ln 2 \approx 0.6931$
- $\frac{1 + 1}{2 \cdot 4} = \frac{2}{8} = 0.25$
- $-\frac{1}{2} = -0.5$

Step 4: Sum.
$$
D_{KL}(P\|Q) = 0.6931 + 0.25 - 0.5 = 0.4431 \text{ nats}
$$

**Answer:** $D_{KL}(\mathcal{N}(0,1)\|\mathcal{N}(1,2)) \approx 0.443$ nats.

Interpretation: On average, observations from the standard normal are about 0.443 nats more surprising under the $\mathcal{N}(1, 2)$ distribution than under the true $\mathcal{N}(0, 1)$.

### Example 3: KL Divergence in Variational Inference

**Problem:** In a variational autoencoder, the encoder produces $Q(z|x) = \mathcal{N}(\mu, \sigma^2)$ where $\mu = 0.5$, $\sigma = 0.8$ for a specific input. The prior is $P(z) = \mathcal{N}(0, 1)$. Compute the KL divergence $D_{KL}(Q(z|x)\|P(z))$, which forms the regularisation term in the ELBO.

**Solution:**

Step 1: Use the Gaussian KL formula (note the order: $Q$ is first, $P$ is second).
$$
D_{KL}(Q\|P) = \log\frac{\sigma_P}{\sigma_Q} + \frac{\sigma_Q^2 + (\mu_Q - \mu_P)^2}{2\sigma_P^2} - \frac{1}{2}
$$

Step 2: Substitute $\mu_Q = 0.5$, $\sigma_Q = 0.8$, $\mu_P = 0$, $\sigma_P = 1$.
$$
D_{KL}(Q\|P) = \log\frac{1}{0.8} + \frac{0.8^2 + (0.5 - 0)^2}{2 \cdot 1^2} - \frac{1}{2}
$$

Step 3: Compute.
- $\log(1/0.8) = \log 1.25 \approx 0.2231$
- $\frac{0.64 + 0.25}{2} = \frac{0.89}{2} = 0.445$
- $-\frac{1}{2} = -0.5$

Step 4: Sum.
$$
D_{KL}(Q\|P) = 0.2231 + 0.445 - 0.5 = 0.1681 \text{ nats}
$$

**Answer:** $D_{KL}(Q\|P) \approx 0.168$ nats.

Interpretation: The encoder's approximate posterior is reasonably close to the prior. This KL term penalises the VAE if the latent representations deviate too much from the standard normal prior, enforcing a structured latent space. The total loss is $-E_{z\sim Q}[\log P(x|z)] + D_{KL}(Q(z|x)\|P(z))$.

### Example 4: KL Divergence for Knowledge Distillation

**Problem:** A teacher model outputs $\hat{y}_T = [0.1, 0.7, 0.2]$ for a 3-class problem. A student model outputs $\hat{y}_S = [0.2, 0.6, 0.2]$. Compute the KL divergence between the teacher and student distributions with temperature $\tau = 2.0$ (softened) and $\tau = 1.0$ (original).

**Solution:**

Step 1: Apply temperature scaling: $p_i^\tau = \frac{p_i^{1/\tau}}{\sum_j p_j^{1/\tau}}$.

For $\tau = 2.0$:
Teacher: $[0.1^{0.5}, 0.7^{0.5}, 0.2^{0.5}] = [0.316, 0.837, 0.447]$, sum $= 1.600$. Normalised: $\hat{y}_T^\tau = [0.198, 0.523, 0.279]$.

Student: $[0.2^{0.5}, 0.6^{0.5}, 0.2^{0.5}] = [0.447, 0.775, 0.447]$, sum $= 1.669$. Normalised: $\hat{y}_S^\tau = [0.268, 0.464, 0.268]$.

Step 2: Compute $D_{KL}(\hat{y}_T^\tau \| \hat{y}_S^\tau)$.
$$
D_{KL} = 0.198\log\frac{0.198}{0.268} + 0.523\log\frac{0.523}{0.464} + 0.279\log\frac{0.279}{0.268}
$$
- $0.198\log(0.739) = 0.198(-0.302) = -0.0598$
- $0.523\log(1.127) = 0.523(0.119) = 0.0623$
- $0.279\log(1.041) = 0.279(0.040) = 0.0112$

Sum: $-0.0598 + 0.0623 + 0.0112 = 0.0137$ nats.

Step 3: For $\tau = 1.0$, compute $D_{KL}(\hat{y}_T \| \hat{y}_S)$.
$$
D_{KL} = 0.1\log\frac{0.1}{0.2} + 0.7\log\frac{0.7}{0.6} + 0.2\log\frac{0.2}{0.2}
$$
- $0.1\log(0.5) = 0.1(-0.693) = -0.0693$
- $0.7\log(1.167) = 0.7(0.154) = 0.1078$
- $0.2\log(1) = 0$

Sum: $0.0385$ nats.

**Answer:** With $\tau=2.0$, $D_{KL} \approx 0.014$ nats. With $\tau=1.0$, $D_{KL} \approx 0.039$ nats. Higher temperature softens the distributions, reducing KL divergence and encouraging the student to learn the relative probabilities between all classes, not just the top prediction.

## Visual Interpretation

KL divergence can be visualised as the expected log-likelihood ratio under $P$. If we plot two distributions $P$ and $Q$ as histograms, the KL divergence is large where $P$ assigns high probability but $Q$ does not. The asymmetry is clear: $D_{KL}(P\|Q)$ is dominated by regions where $P$ has mass and $Q$ is small; $D_{KL}(Q\|P)$ is dominated by regions where $Q$ has mass and $P$ is small.

For the binary entropy function, the KL divergence between Bernoulli distributions with parameters $p$ and $q$ is:
$$
D_{KL}(p\|q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}
$$

This function is convex in both $p$ and $q$, zero when $p = q$, and approaches infinity as $q \to 0$ or $q \to 1$ when $p$ is between.

In variational inference, the ELBO can be visualised as the gap between the true log-evidence $\log P(x)$ and the variational lower bound. The gap is exactly $D_{KL}(Q(z)\|P(z|x))$, which is reduced during optimisation.

## Common Mistakes

1. **Treating KL divergence as a metric:** KL divergence is not symmetric and does not satisfy the triangle inequality. It is a "divergence," not a "distance." Always specify the order: $D_{KL}(P\|Q)$ is the divergence from $Q$ to $P$.

2. **Assuming $D_{KL}(P\|Q) = D_{KL}(Q\|P)$:** This is almost never true. The two values can differ dramatically, especially when the distributions have different support.

3. **Infinite values from zero probabilities:** If $Q(x)=0$ but $P(x)>0$ for any $x$, $D_{KL}(P\|Q) = \infty$. This is mathematically correct but practically problematic. Smoothing techniques (adding small probabilities) or using Jensen-Shannon divergence can help.

4. **Confusing KL divergence direction in ELBO:** The ELBO minimises $D_{KL}(Q\|P)$ (the reverse KL), not $D_{KL}(P\|Q)$ (forward KL). Forward KL leads to "mode covering" behaviour (mass-covering), while reverse KL leads to "mode seeking" behaviour. Choosing the wrong direction fundamentally changes the approximation quality.

5. **Using KL divergence for distributions with different support:** If $P$ and $Q$ have different support (e.g., one is defined on $[0,1]$ and the other on $[0,2]$), KL divergence may be undefined or infinite. Ensure distributions share the same sample space.

6. **Numerical issues with log of ratios:** Computing $\log(P/Q)$ directly can be numerically unstable when $P$ and $Q$ are very small. Use numerically stable formulations like $\log P - \log Q$ or library functions designed for this (e.g., `scipy.special.kl_div`).

7. **Ignoring that $D_{KL}(P\|Q)$ penalises $Q$ for being confident where $P$ is uncertain:** KL divergence is zero-forcing in the reverse direction. This means variational inference with $D_{KL}(Q\|P)$ tends to underestimate variance (producing overconfident, under-dispersed approximations).

## Interview Questions

### Beginner - 5

1. **Q:** What is KL divergence?
   **A:** KL divergence $D_{KL}(P\|Q) = \sum P(x)\log(P(x)/Q(x))$ measures how much distribution $Q$ diverges from distribution $P$. It quantifies the information loss when using $Q$ to approximate $P$.

2. **Q:** Is KL divergence symmetric? Why does this matter?
   **A:** No. $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general. This matters because the direction must be chosen based on the application: forward KL is mass-covering, reverse KL is mode-seeking.

3. **Q:** What is the range of KL divergence?
   **A:** $0 \leq D_{KL}(P\|Q) \leq \infty$. Zero occurs only when $P = Q$. Infinity occurs when $Q$ assigns zero probability to an outcome that $P$ considers possible.

4. **Q:** What is the relationship between KL divergence, entropy, and cross entropy?
   **A:** $D_{KL}(P\|Q) = H(P,Q) - H(P)$. Cross entropy minus entropy equals KL divergence.

5. **Q:** State Gibbs' inequality.
   **A:** $D_{KL}(P\|Q) \geq 0$ with equality if and only if $P = Q$.

### Intermediate - 5

1. **Q:** Derive the evidence lower bound (ELBO) using KL divergence.
   **A:** $\log P(x) = D_{KL}(Q(z)\|P(z|x)) + \mathbb{E}_{z\sim Q}[\log P(x|z)] - D_{KL}(Q(z)\|P(z))$. Since $D_{KL} \geq 0$, $\log P(x) \geq \text{ELBO} = \mathbb{E}_{Q}[\log P(x|z)] - D_{KL}(Q(z)\|P(z))$. The gap between $\log P(x)$ and the ELBO is exactly $D_{KL}(Q(z)\|P(z|x))$.

2. **Q:** Explain the difference between forward KL ($D_{KL}(P\|Q)$) and reverse KL ($D_{KL}(Q\|P)$) in variational inference.
   **A:** Forward KL penalises $Q$ where $P$ has mass, forcing $Q$ to cover all modes of $P$ (mass-covering / zero-avoiding). Reverse KL penalises $Q$ having mass where $P$ does not, making $Q$ seek a single mode and underestimate variance (mode-seeking / zero-forcing). For a multimodal $P$, forward KL may be spread across modes while reverse KL collapses to one mode.

3. **Q:** How is KL divergence used in knowledge distillation?
   **A:** A student model is trained to minimise $D_{KL}(\hat{y}_{\text{teacher}}^\tau \| \hat{y}_{\text{student}}^\tau)$, where $\tau$ is a temperature parameter that softens the distributions. Higher temperatures reveal the relative structure of the teacher's output (e.g., which incorrect classes are more plausible), helping the student learn beyond hard labels.

4. **Q:** What is the KL divergence between two Gaussian distributions? Write the formula and explain each term.
   **A:** $D_{KL}(\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)) = \log(\sigma_2/\sigma_1) + (\sigma_1^2 + (\mu_1-\mu_2)^2)/(2\sigma_2^2) - 1/2$. The first term penalises different scales, the second penalises different means and variances, and the third is a normalisation constant.

5. **Q:** Prove the data processing inequality for KL divergence.
   **A:** For a Markov chain $X \to Y \to Z$, the joint distribution $P(x,z) = \sum_y P(x)P(y|x)P(z|y)$. By the chain rule and non-negativity of KL divergence, $D_{KL}(P(x)\|Q(x)) = D_{KL}(P(x,y)\|Q(x,y)) - D_{KL}(P(y|x)\|Q(y|x)) \geq D_{KL}(P(y)\|Q(y))$ (since the second term is $\geq 0$). This shows that processing cannot increase the distinguishability of distributions.

### Advanced - 3

1. **Q:** Prove that maximising the ELBO is equivalent to minimising $D_{KL}(Q(z)\|P(z|x))$ and discuss the implications of using reverse KL in variational inference.
   **A:** $\log P(x) = \log\int P(x|z)P(z)dz = \log\int Q(z)\frac{P(x|z)P(z)}{Q(z)}dz \geq \int Q(z)\log\frac{P(x|z)P(z)}{Q(z)}dz$ (by Jensen). Rearranging: $\log P(x) \geq \mathbb{E}_Q[\log P(x|z)] - D_{KL}(Q(z)\|P(z)) = \text{ELBO}$. The gap is $\log P(x) - \text{ELBO} = D_{KL}(Q(z)\|P(z|x))$. Since $D_{KL} \geq 0$, maximising ELBO minimises $D_{KL}(Q\|P(z|x))$. This reverse KL means the variational posterior $Q$ will be zero-forcing: it will underestimate variance and may miss modes of the true posterior, but it produces a tractable approximation.

2. **Q:** Compare and contrast KL divergence with the Wasserstein distance (Earth Mover's distance) in the context of generative modelling. When would you prefer one over the other?
   **A:** KL divergence is not a metric (asymmetric, no triangle inequality), infinite for non-overlapping support, and sensitive to tail probabilities. Wasserstein distance is a true metric, finite for distributions with different support, and captures the underlying geometry of the sample space. In GANs, KL divergence can lead to mode collapse and training instability when the generator and data distributions have non-overlapping support. Wasserstein GANs (WGAN) use the Wasserstein distance, which provides smooth, meaningful gradients even when distributions are far apart. However, KL divergence has a direct information-theoretic interpretation and is easier to compute in closed form for exponential family distributions (e.g., Gaussians). Preference: Wasserstein for generative models where support mismatch is common; KL for variational inference, knowledge distillation, and any scenario where the reference distribution has full support.

3. **Q:** Derive the forward KL ($D_{KL}(P\|Q)$) minimisation as maximum likelihood estimation. Show the connection explicitly.
   **A:** Given data $\{x^{(n)}\}_{n=1}^N$ drawn i.i.d. from $P_{\text{data}}$, and a parametric model $Q_\theta$, MLE maximises $\frac{1}{N}\sum_n \log Q_\theta(x^{(n)})$. As $N \to \infty$, this converges by the law of large numbers to $\mathbb{E}_{x\sim P_{\text{data}}}[\log Q_\theta(x)] = \sum_x P_{\text{data}}(x)\log Q_\theta(x) = -H(P_{\text{data}}, Q_\theta)$. Minimising $H(P_{\text{data}}, Q_\theta)$ is equivalent to $H(P_{\text{data}}, Q_\theta) = H(P_{\text{data}}) + D_{KL}(P_{\text{data}}\|Q_\theta)$, so MLE minimises $D_{KL}(P_{\text{data}}\|Q_\theta)$. This is why maximum likelihood is "mass-covering" — $Q_\theta$ must assign probability everywhere $P_{\text{data}}$ has mass. In contrast, variational inference minimises $D_{KL}(Q_\theta\|P)$, which is "mode-seeking."

## Practice Problems

### Easy - 5

1. Compute $D_{KL}(P\|Q)$ for $P = [0.5, 0.5]$, $Q = [0.5, 0.5]$.

2. Compute $D_{KL}(P\|Q)$ for $P = [1, 0]$, $Q = [0.5, 0.5]$.

3. If $H(P) = 1.5$ nats and $H(P,Q) = 2.3$ nats, what is $D_{KL}(P\|Q)$?

4. True or false: $D_{KL}(P\|Q) \geq 0$ for any probability distributions $P$ and $Q$.

5. What is $D_{KL}(P\|Q)$ if $Q$ has a zero probability for an outcome where $P$ has positive probability?

### Medium - 5

1. Compute $D_{KL}(P\|Q)$ and $D_{KL}(Q\|P)$ for $P = [0.2, 0.3, 0.5]$ and $Q = [0.3, 0.3, 0.4]$.

2. Two Gaussians: $P = \mathcal{N}(2, 3)$ and $Q = \mathcal{N}(0, 1)$. Compute $D_{KL}(P\|Q)$.

3. Prove that $D_{KL}(P\|Q) = \sum_i P(i)\log P(i) - \sum_i P(i)\log Q(i)$ and hence relate it to entropy and cross entropy.

4. In a VAE, the encoder outputs $\mu = -0.2$, $\sigma = 0.5$ for a given input. The prior is $\mathcal{N}(0, 1)$. Compute $D_{KL}(Q(z|x)\|P(z))$.

5. For Bernoulli distributions with parameters $p$ and $q$, show that $D_{KL}(p\|q) = p\log(p/q) + (1-p)\log((1-p)/(1-q))$. At what value of $q$ (given $p$) does the divergence go to infinity?

### Hard - 3

1. Prove Jensen's inequality version of Gibbs' inequality: $D_{KL}(P\|Q) \geq 0$ by using $\log x \leq x - 1$.

2. Derive the closed-form KL divergence for multivariate Gaussian distributions. Show all steps including the trace identity.

3. In trust region policy optimisation (TRPO), the update is constrained by $D_{KL}(\pi_{\text{old}}\|\pi_{\text{new}}) \leq \delta$. Explain why this constraint is used and how it relates to natural gradient descent. Show that the natural gradient direction is $F^{-1}\nabla L$, where $F$ is the Fisher information matrix, which is also the Hessian of $D_{KL}(\pi_{\text{old}}\|\pi_{\text{new}})$ at $\pi_{\text{new}} = \pi_{\text{old}}$.

## Solutions

**Easy:**

1. $D_{KL} = 0.5\log(0.5/0.5) + 0.5\log(0.5/0.5) = 0$. The distributions are identical.

2. $D_{KL} = 1\log(1/0.5) + 0\log(0/0.5) = 1\cdot\log 2 + 0 = 0.693$ nats. Note: $0\log 0 = 0$ by convention.

3. $D_{KL} = H(P,Q) - H(P) = 2.3 - 1.5 = 0.8$ nats.

4. True. This is Gibbs' inequality.

5. $D_{KL}(P\|Q) = \infty$ because there is a term $P(x)\log(P(x)/0) = \infty$. This reflects that $Q$ assigns zero probability to an event that occurs under $P$.

**Medium:**

1. $D_{KL}(P\|Q) = 0.2\log(0.2/0.3) + 0.3\log(0.3/0.3) + 0.5\log(0.5/0.4) = 0.2(-0.4055) + 0 + 0.5(0.2231) = -0.0811 + 0.1116 = 0.0305$ nats. $D_{KL}(Q\|P) = 0.3\log(0.3/0.2) + 0.3\log(0.3/0.3) + 0.4\log(0.4/0.5) = 0.3(0.4055) + 0 + 0.4(-0.2231) = 0.1217 - 0.0892 = 0.0325$ nats. They differ.

2. $D_{KL}(P\|Q) = \log(1/\sqrt{3}) + (3 + (2-0)^2)/(2\cdot1) - 1/2 = -0.5493 + (3+4)/2 - 0.5 = -0.5493 + 3.5 - 0.5 = 2.4507$ nats.

3. $D_{KL}(P\|Q) = \sum P\log(P/Q) = \sum P\log P - \sum P\log Q = -H(P) + H(P,Q)$. This is the fundamental relationship.

4. $D_{KL} = \log(1/0.5) + (0.25 + 0.04)/(2\cdot1) - 0.5 = 0.6931 + 0.29/2 - 0.5 = 0.6931 + 0.145 - 0.5 = 0.3381$ nats.

5. $D_{KL}(p\|q) \to \infty$ when $q \to 0$ (if $p > 0$) or when $q \to 1$ (if $p < 1$). For a fixed $p$, the divergence blows up when $Q$ assigns near-zero probability to an outcome that $P$ considers possible.

**Hard:**

1. Using $\log x \leq x - 1$ for $x > 0$ (with equality at $x=1$): let $x = Q(i)/P(i)$. Then $\log(Q/P) \leq Q/P - 1$, so $-\log(Q/P) \geq 1 - Q/P$. Multiply by $P(i)$ and sum: $-\sum P(i)\log(Q(i)/P(i)) \geq \sum P(i)(1 - Q(i)/P(i)) = \sum(P(i) - Q(i)) = 0$. Therefore $D_{KL}(P\|Q) = -\sum P(i)\log(Q(i)/P(i)) \geq 0$. Equality holds when $Q(i)/P(i) = 1$ for all $i$, i.e., $P = Q$.

2. For $P = \mathcal{N}(\mu_1, \Sigma_1)$, $Q = \mathcal{N}(\mu_2, \Sigma_2)$ with $p, q$-dimensional:
$D_{KL}(P\|Q) = \mathbb{E}_P[\log p(x) - \log q(x)]$.
$\log p(x) = -\frac{d}{2}\log 2\pi - \frac{1}{2}\log|\Sigma_1| - \frac{1}{2}(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)$.
$\log q(x) = -\frac{d}{2}\log 2\pi - \frac{1}{2}\log|\Sigma_2| - \frac{1}{2}(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)$.
Taking expectations: $\mathbb{E}_P[(x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)] = \mathbb{E}_P[\text{tr}(\Sigma_1^{-1}(x-\mu_1)(x-\mu_1)^T)] = \text{tr}(\Sigma_1^{-1}\Sigma_1) = d$.
$\mathbb{E}_P[(x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)] = \mathbb{E}_P[\text{tr}(\Sigma_2^{-1}(x-\mu_2)(x-\mu_2)^T)] = \text{tr}(\Sigma_2^{-1}(\Sigma_1 + (\mu_1-\mu_2)(\mu_1-\mu_2)^T))$.
Combining: $D_{KL}(P\|Q) = \frac{1}{2}[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)]$.

3. TRPO constrains the policy update to a trust region where the KL divergence between old and new policies is bounded. This prevents the large, destructive updates that can occur with vanilla policy gradients. The constraint $D_{KL}(\pi_{\text{old}}\|\pi_{\text{new}}) \leq \delta$ is approximated by a second-order expansion: $D_{KL}(\pi_{\text{old}}\|\pi_{\text{new}}) \approx \frac{1}{2}(\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})$, where $F$ is the Fisher information matrix $F = \mathbb{E}[\nabla\log\pi(a|s)\nabla\log\pi(a|s)^T]$. The natural gradient direction $F^{-1}\nabla L$ follows the steepest descent in the KL geometry rather than the Euclidean parameter space, making optimisation more efficient and stable.

## Related Concepts

- Entropy (MATH-088) — $H(P) = -D_{KL}(P\|U) + \log|\mathcal{X}|$ where $U$ is uniform
- Cross Entropy (MATH-089) — $H(P,Q) = H(P) + D_{KL}(P\|Q)$
- Mutual Information (MATH-091) — $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y))$
- Information Gain (MATH-092) — expected reduction in KL divergence after observing a feature
- Jensen-Shannon Divergence — symmetrised, bounded KL divergence
- Fisher Information — local curvature of KL divergence
- Variational Inference — optimisation of KL divergence between approximate and true posteriors

## Next Concepts

- Mutual Information (MATH-091) — $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y))$
- Information Gain (MATH-092) — expected reduction in entropy via KL divergence
- Information Bottleneck — KL-based trade-off between compression and prediction
- Maximum Mean Discrepancy — alternative to KL divergence for two-sample testing
- Wasserstein Distance — metric alternative to KL divergence

## Summary

KL divergence $D_{KL}(P\|Q) = \sum P(x)\log(P(x)/Q(x))$ is a directed measure of how one probability distribution diverges from another. It is non-negative (Gibbs' inequality), zero iff $P=Q$, asymmetric, and can be infinite. It relates to entropy and cross entropy via $D_{KL}(P\|Q) = H(P,Q) - H(P)$. KL divergence is the fundamental objective in variational inference (ELBO), knowledge distillation, trust region policy optimisation, and Bayesian deep learning. The forward KL ($D_{KL}(P\|Q)$, mass-covering) is minimised in maximum likelihood estimation, while reverse KL ($D_{KL}(Q\|P)$, mode-seeking) is minimised in variational inference. Its closed-form expression for Gaussians makes it widely applicable in latent variable models.

## Key Takeaways

- $D_{KL}(P\|Q) = \sum P(x)\log(P(x)/Q(x)) \geq 0$ (Gibbs' inequality), $0$ iff $P=Q$.
- Asymmetric: $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ — always specify the order.
- $D_{KL}(P\|Q) = H(P,Q) - H(P)$.
- Forward KL (MLE): mass-covering; Reverse KL (VI): mode-seeking.
- $D_{KL}(P\|Q) = \infty$ if $Q$ assigns zero to an event $P$ considers possible.
- Closed form for Gaussians: $D_{KL}(\mathcal{N}_1\|\mathcal{N}_2) = \log(\sigma_2/\sigma_1) + (\sigma_1^2 + (\mu_1-\mu_2)^2)/(2\sigma_2^2) - 1/2$.
- Core loss in: variational inference (ELBO), knowledge distillation, TRPO/PPO.
- ELBO $= \mathbb{E}_Q[\log P(x|z)] - D_{KL}(Q(z)\|P(z)) \leq \log P(x)$.
