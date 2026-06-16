# Concept: Mutual Information

## Concept ID

MATH-091

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Information Theory

## Learning Objectives

- Define mutual information as a measure of dependence between random variables
- Express mutual information in terms of entropy, conditional entropy, and KL divergence
- Understand the relationship $I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = D_{KL}(P(x,y)\|P(x)P(y))$
- Compute mutual information for discrete and Gaussian distributions
- Apply mutual information to feature selection, representation learning, and the information bottleneck

## Prerequisites

- Entropy (MATH-088)
- KL Divergence (MATH-090)
- Probability (MATH-065)
- Conditional Probability (MATH-068)
- Bayes Theorem (MATH-069)
- Random Variable (MATH-070)
- Joint and marginal distributions

## Definition

**Mutual information** (MI) quantifies the amount of information obtained about one random variable by observing another. It measures the reduction in uncertainty of $X$ given knowledge of $Y$, or vice versa.

For two discrete random variables $X$ and $Y$ with joint distribution $P(x,y)$ and marginal distributions $P(x)$ and $P(y)$, mutual information is defined as:

$$
I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}
$$

For continuous random variables with joint density $f(x,y)$ and marginal densities $f(x)$, $f(y)$:

$$
I(X; Y) = \int_{\mathcal{X}} \int_{\mathcal{Y}} f(x, y) \log \frac{f(x, y)}{f(x)f(y)} \, dx \, dy
$$

Mutual information measures the departure from independence: if $X$ and $Y$ are independent, $P(x,y) = P(x)P(y)$ and $I(X;Y) = 0$. The more dependent $X$ and $Y$ are, the larger the mutual information.

## Intuition

Mutual information answers: "How much does knowing $Y$ tell me about $X$?" If $X$ and $Y$ are independent, knowing $Y$ gives no information about $X$ — MI is zero. If $Y$ is a deterministic function of $X$ (or vice versa), knowing $Y$ reveals $X$ completely — MI equals $H(X)$ (the total uncertainty of $X$).

MI is symmetric: $I(X;Y) = I(Y;X)$. Unlike correlation, which only measures linear dependence, MI captures any kind of statistical dependence — linear, nonlinear, periodic, or otherwise. Two variables can have zero correlation but high MI if they have a nonlinear relationship (e.g., $Y = X^2$).

Think of MI as the "shared information" between two variables. Visualise a Venn diagram where the areas of the circles represent $H(X)$ and $H(Y)$. The overlap region is $I(X;Y)$, the information common to both. The non-overlapping parts are $H(X|Y)$ (information unique to $X$) and $H(Y|X)$ (information unique to $Y$). The union is $H(X,Y)$ (the joint entropy).

## Why This Concept Matters

Mutual information is a cornerstone of information-theoretic approaches in machine learning:

- **Feature selection:** MI quantifies how relevant a feature is to the target variable. Features with high MI with the target are valuable; those with low MI can be discarded. MI-based feature selection captures nonlinear relationships that correlation-based methods miss.
- **Representation learning:** InfoGAN maximises MI between latent codes and generated images to learn disentangled representations. Variational information bottleneck (VIB) learns representations that compress away irrelevant information while preserving predictive information.
- **Independent component analysis:** Infomax ICA maximises MI between transformed variables and a nonlinear function to separate mixed signals.
- **Neural coding:** In neuroscience, MI measures how much information a neural response carries about a stimulus.
- **Fairness:** MI between sensitive attributes and predictions quantifies algorithmic bias. Fair learning can incorporate MI constraints.
- **Cluster evaluation:** Normalised mutual information (NMI) evaluates clustering quality by measuring the agreement between cluster assignments and ground-truth labels, normalised to $[0, 1]$.

## Historical Background

Mutual information was developed alongside modern information theory by Claude Shannon in his 1948 paper. Shannon defined MI as the rate of information transmission through a communication channel, calling it the "rate of transmission." The channel capacity is the maximum mutual information between channel input and output over all possible input distributions.

The concept has roots in earlier work: the likelihood ratio statistic used in hypothesis testing is related to the information-theoretic divergence. The term "mutual information" was standardised after Shannon's work.

R.A. Fisher's concept of "information" in a statistic (Fisher information) is related but distinct: Fisher information measures the curvature of the log-likelihood, while mutual information measures dependence between variables. The two are connected through the Cramér-Rao bound and the de Bruijn identity.

Modern developments include the information bottleneck method (Tishby, Pereira, Bialek, 1999), which uses MI to find optimal representations, and its extension to deep learning (information bottleneck theory of deep learning, Tishby and Zaslavsky, 2015).

## Real World Examples

1. **Medical diagnosis:** Mutual information between symptoms and diseases identifies which symptoms are most informative. A symptom like "fever" has moderate MI with many diseases, while "Jaundice" has high MI with liver disease specifically. Doctors use this information to prioritise diagnostic tests.

2. **Feature selection for text classification:** In spam filtering, MI between each word and the class label (spam vs. ham) ranks words by informativeness. Words like "free," "win," and "click" have high MI with spam. This MI-based selection often outperforms frequency-based or correlation-based methods.

3. **Genomics:** MI is used to infer gene regulatory networks. High MI between the expression levels of two genes suggests they may be co-regulated or functionally related. The ARACNE algorithm (Algorithm for the Reconstruction of Accurate Cellular Networks) uses MI to reconstruct genetic interaction networks.

4. **Image registration:** In medical imaging (CT, MRI, PET), images from different modalities must be aligned. MI between pixel intensities of the two images is maximised when they are properly registered, as corresponding anatomical structures produce coordinated intensity patterns across modalities.

5. **Stock market analysis:** MI between the returns of different stocks measures non-linear dependencies beyond simple correlation. This is used for portfolio diversification: stocks with low pairwise MI provide better diversification benefits.

## AI/ML Relevance

**Feature selection:** Given features $X_1, \dots, X_d$ and target $Y$, the MI $I(X_i; Y)$ ranks each feature's relevance. Unlike correlation, MI detects non-linear relationships. The minimum Redundancy Maximum Relevance (mRMR) criterion selects features that maximise $I(X_i; Y) - \beta \sum_{j} I(X_i; X_j)$, balancing relevance with redundancy.

**InfoGAN:** InfoGAN extends the generative adversarial network framework by maximising the MI between a subset of latent variables $c$ and the generated samples $G(z,c)$:
$$
\min_G \max_D V(D,G) - \lambda I(c; G(z,c))
$$
This forces the latent code $c$ to capture semantically meaningful features of the data (e.g., digit identity, rotation, thickness in MNIST).

**Information bottleneck (IB):** The IB method finds a representation $Z$ of input $X$ that is maximally informative about target $Y$ while being maximally compressive about $X$:
$$
\min I(X; Z) - \beta I(Z; Y)
$$
where $\beta$ controls the trade-off between compression and prediction. The variational information bottleneck (VIB) provides a scalable deep learning implementation.

**Information bottleneck theory of deep learning:** Tishby and colleagues proposed that deep networks go through two phases: an initial "fitting" phase (increasing $I(Z;Y)$) and a later "compression" phase (decreasing $I(X;Z)$). This theory provides a information-theoretic perspective on generalisation and the role of stochastic gradient descent.

**Normalised mutual information (NMI) for clustering:** Given cluster assignments $C$ and ground-truth labels $L$:
$$
\text{NMI}(L, C) = \frac{2 I(L; C)}{H(L) + H(C)}
$$
NMI ranges from 0 (no agreement) to 1 (perfect agreement). It is a standard metric for clustering evaluation, invariant to label permutations.

**Estimation challenges:** Estimating MI from finite samples is difficult. The plug-in estimator (using empirical frequencies) is biased upward. Common approaches include:
- k-nearest neighbour-based estimators (Kraskov, Stögbauer, Grassberger, 2004)
- Kernel density estimation-based MI
- Variational lower bounds (MINE — Mutual Information Neural Estimation, Belghazi et al., 2018)
- Binned histograms with correction for finite-size bias

## Mathematical Explanation

**Relationship with entropy:**
$$
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
$$

Proof: Starting from the definition,
$$
I(X;Y) = \sum_{x,y} P(x,y)\log\frac{P(x,y)}{P(x)P(y)} = \sum_{x,y} P(x,y)\log\frac{P(x|y)}{P(x)} = -\sum_x P(x)\log P(x) + \sum_{x,y} P(x,y)\log P(x|y)
$$
$$
I(X;Y) = H(X) - \left(-\sum_{x,y} P(x,y)\log P(x|y)\right) = H(X) - H(X|Y)
$$

The other forms follow by symmetry and the chain rule for entropy.

**Relationship with KL divergence:**
$$
I(X;Y) = D_{KL}(P(x,y) \| P(x)P(y))
$$

Mutual information is the KL divergence between the joint distribution $P(x,y)$ and the product of the marginals $P(x)P(y)$. This directly shows that $I(X;Y) = 0$ iff $X$ and $Y$ are independent, since $P(x,y) = P(x)P(y)$.

**Chain rule for mutual information:**
$$
I(X_1, \dots, X_n; Y) = \sum_{i=1}^{n} I(X_i; Y | X_{i-1}, \dots, X_1)
$$

For three variables:
$$
I(X; Y, Z) = I(X; Y) + I(X; Z | Y)
$$

**Conditional mutual information:**
$$
I(X; Y | Z) = H(X|Z) - H(X|Y,Z) = \sum_z P(z) I(X;Y|Z=z)
$$

This measures the information shared between $X$ and $Y$ after accounting for $Z$.

**Data processing inequality:** For a Markov chain $X \to Y \to Z$,
$$
I(X; Y) \geq I(X; Z)
$$

Processing $Y$ can only reduce (not increase) the information about $X$. This is the fundamental limitation of any data processing pipeline.

**Gaussian mutual information:** For jointly Gaussian $(X,Y)$ with correlation coefficient $\rho$:
$$
I(X;Y) = -\frac{1}{2}\log(1 - \rho^2)
$$

In bits: $I(X;Y) = -\frac{1}{2}\log_2(1 - \rho^2)$. For multivariate Gaussians:
$$
I(X;Y) = \frac{1}{2}\log\frac{|\Sigma_X| \cdot |\Sigma_Y|}{|\Sigma|}
$$
where $\Sigma$ is the joint covariance matrix, and $\Sigma_X, \Sigma_Y$ are marginal covariance matrices.

## Formula(s)

**Mutual information (discrete):**
$$
I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}
$$

**Mutual information in terms of entropy:**
$$
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
$$

**Mutual information as KL divergence:**
$$
I(X;Y) = D_{KL}(P(x,y) \| P(x)P(y))
$$

**Conditional mutual information:**
$$
I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
$$

**Chain rule:**
$$
I(X; Y, Z) = I(X;Y) + I(X;Z|Y)
$$

**Gaussian mutual information:**
$$
I(X;Y) = -\frac{1}{2}\log(1 - \rho^2)
$$

**Normalised mutual information:**
$$
\text{NMI}(X,Y) = \frac{2I(X;Y)}{H(X) + H(Y)}
$$

**Information bottleneck objective:**
$$
\min I(X;Z) - \beta I(Z;Y)
$$

## Properties

- **Non-negativity:** $I(X;Y) \geq 0$, with equality iff $X$ and $Y$ are independent.
- **Symmetry:** $I(X;Y) = I(Y;X)$.
- **Upper bound:** $I(X;Y) \leq \min(H(X), H(Y))$, with equality iff $X$ is a function of $Y$ (or vice versa).
- **Invariance under invertible transformations:** For any bijections $f, g$, $I(f(X); g(Y)) = I(X;Y)$.
- **Additivity for independent pairs:** If $(X_1,Y_1)$ and $(X_2,Y_2)$ are independent, then $I(X_1,X_2; Y_1,Y_2) = I(X_1;Y_1) + I(X_2;Y_2)$.
- **Data processing inequality:** For $X \to Y \to Z$, $I(X;Y) \geq I(X;Z)$.
- **Convex-concave:** $I(X;Y)$ is convex in $P(y|x)$ for fixed $P(x)$ and concave in $P(x)$ for fixed $P(y|x)$.
- **Relationship to correlation:** For Gaussian variables, $I(X;Y) = -\frac{1}{2}\log(1-\rho^2)$, which is a monotonic function of $\rho^2$. For non-Gaussian variables, $I(X;Y)$ captures non-linear dependencies that $\rho$ misses.
- **Fano's inequality:** $H(e) \geq H(X|Y)$ where $e$ is the indicator of error in estimating $X$ from $Y$, connecting MI to error probability.

## Step-by-Step Worked Examples

### Example 1: Mutual Information Between Two Binary Variables

**Problem:** Compute $I(X;Y)$ given the following joint distribution for binary $X, Y$:
$$
P(X=0, Y=0) = 0.3, \quad P(X=0, Y=1) = 0.2, \quad P(X=1, Y=0) = 0.1, \quad P(X=1, Y=1) = 0.4
$$

**Solution:**

Step 1: Compute marginal distributions.
$$
P(X=0) = 0.3 + 0.2 = 0.5, \quad P(X=1) = 0.1 + 0.4 = 0.5
$$
$$
P(Y=0) = 0.3 + 0.1 = 0.4, \quad P(Y=1) = 0.2 + 0.4 = 0.6
$$

Step 2: Compute $H(X)$.
$$
H(X) = -[0.5\log_2 0.5 + 0.5\log_2 0.5] = 1 \text{ bit}
$$

Step 3: Compute $H(X|Y)$.
$$
H(X|Y=0) = -[P(X=0|Y=0)\log P(X=0|Y=0) + P(X=1|Y=0)\log P(X=1|Y=0)]
$$
$$
P(X=0|Y=0) = 0.3/0.4 = 0.75, \quad P(X=1|Y=0) = 0.1/0.4 = 0.25
$$
$$
H(X|Y=0) = -[0.75\log_2 0.75 + 0.25\log_2 0.25] = -[0.75(-0.415) + 0.25(-2)] = -[-0.311 - 0.5] = 0.811 \text{ bits}
$$

$$
H(X|Y=1) = -[P(X=0|Y=1)\log P(X=0|Y=1) + P(X=1|Y=1)\log P(X=1|Y=1)]
$$
$$
P(X=0|Y=1) = 0.2/0.6 = 0.333, \quad P(X=1|Y=1) = 0.4/0.6 = 0.667
$$
$$
H(X|Y=1) = -[0.333\log_2 0.333 + 0.667\log_2 0.667] = -[0.333(-1.585) + 0.667(-0.585)] = -[-0.528 - 0.390] = 0.918 \text{ bits}
$$

Step 4: $H(X|Y) = P(Y=0)H(X|Y=0) + P(Y=1)H(X|Y=1) = 0.4(0.811) + 0.6(0.918) = 0.324 + 0.551 = 0.875$ bits.

Step 5: $I(X;Y) = H(X) - H(X|Y) = 1 - 0.875 = 0.125$ bits.

**Answer:** $I(X;Y) \approx 0.125$ bits.

Interpretation: Knowing $Y$ reduces uncertainty about $X$ by 0.125 bits. Since $H(X) = 1$ bit, this is a relatively weak dependence.

### Example 2: Mutual Information Using KL Divergence

**Problem:** Compute $I(X;Y)$ via $D_{KL}(P(x,y)\|P(x)P(y))$ for the same distribution as Example 1.

**Solution:**

Step 1: Compute the product of marginals $P(x)P(y)$.
- $P(X=0)P(Y=0) = 0.5 \times 0.4 = 0.2$
- $P(X=0)P(Y=1) = 0.5 \times 0.6 = 0.3$
- $P(X=1)P(Y=0) = 0.5 \times 0.4 = 0.2$
- $P(X=1)P(Y=1) = 0.5 \times 0.6 = 0.3$

Step 2: Compute the KL divergence.
$$
I(X;Y) = \sum_x\sum_y P(x,y)\log_2\frac{P(x,y)}{P(x)P(y)}
$$
$$
= 0.3\log_2\frac{0.3}{0.2} + 0.2\log_2\frac{0.2}{0.3} + 0.1\log_2\frac{0.1}{0.2} + 0.4\log_2\frac{0.4}{0.3}
$$
$$
= 0.3\log_2 1.5 + 0.2\log_2 0.667 + 0.1\log_2 0.5 + 0.4\log_2 1.333
$$
$$
= 0.3(0.585) + 0.2(-0.585) + 0.1(-1) + 0.4(0.415)
$$
$$
= 0.176 - 0.117 - 0.1 + 0.166 = 0.125 \text{ bits}
$$

**Answer:** $I(X;Y) \approx 0.125$ bits, confirming the result from the entropy approach.

### Example 3: Mutual Information for Gaussian Variables

**Problem:** Two stock returns $X$ and $Y$ have a bivariate normal distribution with correlation $\rho = 0.6$. Compute their mutual information in bits.

**Solution:**

Step 1: Apply the Gaussian MI formula.
$$
I(X;Y) = -\frac{1}{2}\log_2(1 - \rho^2)
$$

Step 2: Substitute $\rho = 0.6$.
$$
I(X;Y) = -\frac{1}{2}\log_2(1 - 0.36) = -\frac{1}{2}\log_2 0.64
$$

Step 3: Compute.
$$
\log_2 0.64 = \frac{\ln 0.64}{\ln 2} \approx \frac{-0.4463}{0.6931} \approx -0.644
$$

Step 4: Final calculation.
$$
I(X;Y) = -\frac{1}{2}(-0.644) = 0.322 \text{ bits}
$$

**Answer:** $I(X;Y) \approx 0.322$ bits.

Interpretation: Knowing one stock return reduces uncertainty about the other by 0.322 bits. If $\rho = 0$, MI would be 0 bits (independence). If $\rho = 0.9$, MI would be $-\frac{1}{2}\log_2(0.19) \approx 1.198$ bits. If $\rho = 1$, MI becomes infinite (perfect linear relationship removes all uncertainty).

### Example 4: Feature Selection with Mutual Information

**Problem:** A dataset has 3 features $X_1, X_2, X_3$ and a binary target $Y$. The empirical MI values are:
- $I(X_1; Y) = 0.3$ bits
- $I(X_2; Y) = 0.5$ bits
- $I(X_3; Y) = 0.1$ bits
- $I(X_1; X_2) = 0.4$ bits
- $I(X_1; X_3) = 0.05$ bits
- $I(X_2; X_3) = 0.1$ bits

Select the best pair of features using both (a) top-k selection and (b) mRMR criterion with $\beta = 0.5$.

**Solution:**

(a) Top-k selection: Choose the two features with highest individual MI: $X_2$ (0.5) and $X_1$ (0.3).

(b) mRMR criterion: Score = $I(X_i; Y) - \beta \sum_{j \in \text{selected}} I(X_i; X_j)$.

Step 1: First feature: pick the one with highest MI — $X_2$ (0.5).

Step 2: Second feature evaluation:
- For $X_1$: Score = $I(X_1; Y) - \beta \cdot I(X_1; X_2) = 0.3 - 0.5(0.4) = 0.3 - 0.2 = 0.1$
- For $X_3$: Score = $I(X_3; Y) - \beta \cdot I(X_3; X_2) = 0.1 - 0.5(0.1) = 0.1 - 0.05 = 0.05$

Step 3: $X_1$ has a higher mRMR score (0.1 > 0.05).

**Answer:** Top-k selects $\{X_2, X_1\}$. mRMR also selects $\{X_2, X_1\}$ because $X_1$ provides more unique information about $Y$ despite having higher redundancy with $X_2$. If $I(X_3; X_2)$ were very small (say 0.01), mRMR might prefer $X_3$ despite its lower individual MI, because it adds more complementary information.

## Visual Interpretation

The Venn diagram interpretation is the most intuitive: two overlapping circles represent the entropy of $X$ and $Y$. The overlap is $I(X;Y)$. The exclusive part of $X$ is $H(X|Y)$ and the exclusive part of $Y$ is $H(Y|X)$. The union is $H(X,Y)$.

For a scatter plot of samples $(x_i, y_i)$, MI quantifies how tightly the points cluster around any curve (not just a line). High MI means the relationship is nearly deterministic (points fall along a thin curve). Low MI means the points are spread out in a cloud.

The joint histogram of $(X,Y)$ provides a visual estimate of $P(x,y)$. MI compares this to the product $P(x)P(y)$ computed from marginal histograms. The ratio $P(x,y)/(P(x)P(y))$ indicates "surprising" coincidences: values >1 mean the pair occurs more often than under independence; values <1 mean less often.

For Gaussians, MI is a monotonic function of $|\rho|$, making the correlation circle plot a visual proxy. But for non-Gaussian relationships (e.g., $Y = X^2$), scatter plots with zero correlation but high MI demonstrate the superiority of MI.

## Common Mistakes

1. **Confusing mutual information with correlation:** Correlation only measures linear dependence. Two variables can have zero correlation but high MI (e.g., $Y = X^2$ with symmetric $X$). MI captures all forms of statistical dependence.

2. **Using plug-in estimators without bias correction:** The naive empirical estimator $I_{\text{emp}} = \sum \hat{P}(x,y) \log(\hat{P}(x,y)/(\hat{P}(x)\hat{P}(y)))$ is biased upward, especially when the number of bins is large relative to sample size. Always use bias-corrected estimators (e.g., Miller-Madow correction, shuffling-based correction, or kNN-based methods).

3. **Forgetting that MI is bounded:** $I(X;Y) \leq \min(H(X), H(Y))$. If $X$ has low entropy (e.g., a nearly deterministic variable), MI with $Y$ is automatically limited. Normalised MI ($\text{NMI} = 2I/(H(X)+H(Y))$) accounts for this.

4. **Assuming MI is symmetric to conditional MI:** MI is symmetric ($I(X;Y) = I(Y;X)$), but conditional MI is not symmetric in the same way. $I(X;Y|Z)$ measures the information between $X$ and $Y$ after accounting for $Z$, and $I(X;Y|Z) \neq I(X;Z|Y)$ generally.

5. **Discretising continuous variables poorly:** When computing MI for continuous variables, the choice of binning strategy significantly affects results. Too few bins lose information; too many bins create sparse estimates with high bias. Adaptive binning (e.g., equal-frequency bins) or kNN estimators are preferred.

6. **Overinterpreting small MI values:** MI depends on the units (bits vs. nats) and the base entropy. A small absolute MI might be significant if $H(X)$ is small. Always consider the relative reduction: $I(X;Y)/H(X)$ gives the fraction of explained uncertainty.

7. **Ignoring the difference between $I(X;Y)$ and $I(X;Y|Z)$:** It is possible for $I(X;Y) = 0$ but $I(X;Y|Z) > 0$ (e.g., $X$ and $Y$ are independent but conditionally dependent given $Z$, as in the "explaining away" phenomenon). This is the opposite of Simpson's paradox in information theory.

## Interview Questions

### Beginner - 5

1. **Q:** What is mutual information?
   **A:** Mutual information $I(X;Y)$ measures how much information one random variable provides about another. It is the reduction in uncertainty of $X$ given $Y$: $I(X;Y) = H(X) - H(X|Y)$.

2. **Q:** What is the range of mutual information?
   **A:** $0 \leq I(X;Y) \leq \min(H(X), H(Y))$. Zero indicates independence; the upper bound indicates one variable is a deterministic function of the other.

3. **Q:** What is the relationship between mutual information and KL divergence?
   **A:** $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y))$. It measures how much the joint distribution differs from the product of marginals (independence).

4. **Q:** Is mutual information symmetric?
   **A:** Yes, $I(X;Y) = I(Y;X)$. Knowing $Y$ gives as much information about $X$ as knowing $X$ gives about $Y$.

5. **Q:** What is normalised mutual information (NMI)?
   **A:** NMI scales MI to $[0,1]$: $\text{NMI} = 2I(X;Y)/(H(X) + H(Y))$. It is commonly used in clustering evaluation.

### Intermediate - 5

1. **Q:** How does mutual information differ from Pearson correlation?
   **A:** Correlation only captures linear dependence and ranges from $-1$ to $1$. MI captures all statistical dependencies (linear, nonlinear, periodic) and is always non-negative. Two variables can have zero correlation but high MI (e.g., $Y = X^2$ with $X$ symmetric around 0).

2. **Q:** How is mutual information used in feature selection?
   **A:** Features are ranked by $I(\text{feature}; \text{target})$. High MI features are informative. The mRMR criterion extends this by subtracting a redundancy term: $\max I(X_i; Y) - \beta \sum_{j \in S} I(X_i; X_j)$, balancing relevance with non-redundancy.

3. **Q:** State the data processing inequality for mutual information.
   **A:** For a Markov chain $X \to Y \to Z$, $I(X;Y) \geq I(X;Z)$. Any processing of $Y$ to produce $Z$ cannot increase the information about $X$.

4. **Q:** Compute the mutual information for two independent standard normal variables.
   **A:** Since $\rho = 0$, $I(X;Y) = -\frac{1}{2}\log(1 - 0) = 0$. Variables are independent, so no information is shared.

5. **Q:** What is the information bottleneck principle?
   **A:** The information bottleneck finds a representation $Z$ of $X$ that is maximally informative about $Y$ while being maximally compressive: $\min I(X;Z) - \beta I(Z;Y)$. The trade-off parameter $\beta$ controls compression vs. prediction.

### Advanced - 3

1. **Q:** Derive the relationship $I(X;Y) = H(X) - H(X|Y)$ starting from the definition $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y))$.
   **A:** $I(X;Y) = \sum P(x,y)\log\frac{P(x,y)}{P(x)P(y)} = \sum P(x,y)\log\frac{P(x|y)}{P(x)} = \sum P(x,y)[\log P(x|y) - \log P(x)] = \sum_y P(y)\sum_x P(x|y)\log P(x|y) - \sum_x P(x)\log P(x) = -H(X|Y) + H(X) = H(X) - H(X|Y)$. This derivation uses $P(x,y) = P(y)P(x|y)$ and the definitions of conditional and marginal entropy.

2. **Q:** Explain the Kraskov-Stögbauer-Grassberger (KSG) estimator for MI between continuous variables. Why is it preferred over naive binning?
   **A:** The KSG estimator uses k-nearest neighbour distances to estimate MI without explicit binning. It works by counting neighbours in the joint space and comparing to neighbours in marginal spaces. The estimator is: $I(X;Y) \approx \psi(k) + \psi(N) - \langle \psi(n_x+1) + \psi(n_y+1) \rangle$, where $\psi$ is the digamma function, $N$ is sample size, $n_x$ is the number of points within a certain distance in $X$-space, and $n_y$ similarly for $Y$. The KSG estimator is adaptive, has lower bias than binning, and works well for continuous variables with nonlinear relationships. It avoids the curse of dimensionality associated with fixed binning schemes.

3. **Q:** In the context of the information bottleneck theory of deep learning, explain the "fitting" and "compression" phases. How does this relate to generalisation?
   **A:** The information bottleneck theory (Tishby & Zaslavsky, 2015) analyses neural network layers as representations $Z$ of the input $X$ that predict output $Y$. During SGD training, two phases are observed in the $(I(X;Z), I(Z;Y))$ plane: (1) **Fitting phase:** Early training rapidly increases $I(Z;Y)$ as the network learns to predict the output, while $I(X;Z)$ also increases as the representation encodes more input information. (2) **Compression phase:** Later training decreases $I(X;Z)$ as the network discards irrelevant input information while maintaining $I(Z;Y)$ — it compresses the representation to what is strictly necessary for prediction. This compression is driven by the stochasticity of SGD (the "noise" in gradient updates). The theory connects compression to generalisation: representations with lower $I(X;Z)$ generalise better because they are less sensitive to irrelevant input variations. While the original theory has been debated (some argue compression is due to quantisation effects of common activation functions like ReLU), the information-theoretic perspective remains influential for understanding deep learning.

## Practice Problems

### Easy - 5

1. Compute $I(X;Y)$ for independent variables $X$ and $Y$.

2. If $Y = f(X)$ is a deterministic function, what is $I(X;Y)$?

3. True or false: $I(X;Y) = I(Y;X)$.

4. For Gaussian variables with $\rho = 0.8$, compute $I(X;Y)$ in nats.

5. If $H(X) = 3$ bits and $H(X|Y) = 1$ bit, what is $I(X;Y)$?

### Medium - 5

1. Given the joint distribution:
   $$P = \begin{bmatrix} 0.2 & 0.1 \\ 0.1 & 0.6 \end{bmatrix}$$
   where rows are $X=\{0,1\}$ and columns are $Y=\{0,1\}$, compute $I(X;Y)$.

2. For jointly Gaussian $(X,Y)$ with $I(X;Y) = 0.5$ nats, find $\rho$.

3. Prove that $I(X;Y) = H(X) + H(Y) - H(X,Y)$.

4. In a clustering evaluation, $H(L) = 2.0$, $H(C) = 1.8$, and $I(L;C) = 1.2$ (all in bits). Compute the NMI.

5. For a Markov chain $X \to Y \to Z$ where $I(X;Y) = 0.8$ bits and $I(X;Z) = 0.3$ bits, verify the data processing inequality.

### Hard - 3

1. Prove the chain rule for mutual information: $I(X; Y, Z) = I(X;Y) + I(X;Z|Y)$.

2. Derive the closed-form expression for MI between two jointly Gaussian random vectors using the covariance matrix determinant formula.

3. The MINE estimator uses a neural network to lower-bound mutual information: $I(X;Y) \geq \sup_{\theta} \mathbb{E}[-h_\theta(x,y)] - \log \mathbb{E}[e^{-h_\theta(x,y')}]$ for a specific class of functions $h_\theta$. Derive this bound from the Donsker-Varadhan representation of KL divergence.

## Solutions

**Easy:**

1. $I(X;Y) = 0$ bits. The joint distribution factorises, so $P(x,y)=P(x)P(y)$, making $\log(P(x,y)/(P(x)P(y))) = \log 1 = 0$.

2. $I(X;Y) = H(X) = H(Y)$ (assuming $f$ is bijective). If $Y$ fully determines $X$, then $H(X|Y) = 0$, so $I(X;Y) = H(X)$.

3. True. Mutual information is symmetric.

4. $I(X;Y) = -\frac{1}{2}\ln(1 - 0.64) = -\frac{1}{2}\ln 0.36 = -0.5(-1.0217) = 0.5108$ nats.

5. $I(X;Y) = H(X) - H(X|Y) = 3 - 1 = 2$ bits.

**Medium:**

1. Marginals: $P(X=0)=0.3$, $P(X=1)=0.7$, $P(Y=0)=0.3$, $P(Y=1)=0.7$. $I = 0.2\log(0.2/0.09) + 0.1\log(0.1/0.21) + 0.1\log(0.1/0.21) + 0.6\log(0.6/0.49) = 0.2(1.152) + 0.1(-0.742) + 0.1(-0.742) + 0.6(0.292) = 0.2304 - 0.0742 - 0.0742 + 0.1752 = 0.2572$ bits.

2. $\rho^2 = 1 - e^{-2 \times 0.5} = 1 - e^{-1} \approx 0.6321$. $\rho \approx 0.795$.

3. $I(X;Y) = H(X) - H(X|Y) = H(X) - [H(X,Y) - H(Y)] = H(X) + H(Y) - H(X,Y)$.

4. $\text{NMI} = 2 \times 1.2 / (2.0 + 1.8) = 2.4 / 3.8 \approx 0.632$.

5. $I(X;Y) = 0.8 \geq I(X;Z) = 0.3$, confirming the data processing inequality. Information about $X$ decreases as it passes through the chain.

**Hard:**

1. $I(X;Y,Z) = H(X) - H(X|Y,Z) = H(X) - [H(X|Y) + H(X|Y,Z) - H(X|Y)] = [H(X)-H(X|Y)] + [H(X|Y)-H(X|Y,Z)] = I(X;Y) + I(X;Z|Y)$. The second step uses $H(X|Y,Z) \leq H(X|Y)$ with equality adjustment.

2. For jointly Gaussian vectors $X \in \mathbb{R}^p$, $Y \in \mathbb{R}^q$ with joint covariance $\Sigma = \begin{bmatrix} \Sigma_X & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_Y \end{bmatrix}$. The marginal entropies are $h(X) = \frac{1}{2}\log((2\pi e)^p |\Sigma_X|)$ and $h(Y) = \frac{1}{2}\log((2\pi e)^q |\Sigma_Y|)$. The joint entropy is $h(X,Y) = \frac{1}{2}\log((2\pi e)^{p+q} |\Sigma|)$. Then $I(X;Y) = h(X) + h(Y) - h(X,Y) = \frac{1}{2}\log\frac{|\Sigma_X|\cdot|\Sigma_Y|}{|\Sigma|}$. This can be expressed as $I(X;Y) = -\frac{1}{2}\log|I - \Sigma_X^{-1}\Sigma_{XY}\Sigma_Y^{-1}\Sigma_{YX}|$.

3. The Donsker-Varadhan representation states: $D_{KL}(P\|Q) = \sup_{T} \mathbb{E}_P[T] - \log\mathbb{E}_Q[e^T]$, where the supremum is over all measurable functions $T$. Setting $P = P(x,y)$, $Q = P(x)P(y)$, and parameterising $T_\theta(x,y) = -h_\theta(x,y)$ gives $I(X;Y) = \sup_\theta \mathbb{E}_{P(x,y)}[-h_\theta(x,y)] - \log\mathbb{E}_{P(x)P(y)}[e^{-h_\theta(x,y)}]$. Since $\mathbb{E}[e^{-h}] \geq e^{-\mathbb{E}[h]}$ (by Jensen), this is a lower bound. MINE trains a neural network $h_\theta$ to approximate this supremum, providing a differentiable estimate of MI that can be optimised via gradient descent.

## Related Concepts

- Entropy (MATH-088) — $H(X)$ measures total uncertainty; $I(X;Y) = H(X) - H(X|Y)$
- KL Divergence (MATH-090) — $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y))$
- Information Gain (MATH-092) — $IG(T,a) = I(T;A)$ where $A$ is the split attribute
- Conditional Entropy — $H(X|Y) = H(X) - I(X;Y)$
- Joint Entropy — $H(X,Y) = H(X) + H(Y) - I(X;Y)$
- Correlation — linear dependence; MI generalises to all dependence
- Data Processing Inequality — $I(X;Y) \geq I(f(X);g(Y))$ for any functions $f,g$

## Next Concepts

- Information Gain (MATH-092) — MI between feature and target in decision tree context
- Information Bottleneck — trade-off between $I(X;Z)$ and $I(Z;Y)$
- Total Correlation — multivariate generalisation of MI
- Transfer Entropy — directed (causal) version of MI for time series
- Conditional Mutual Information — $I(X;Y|Z)$ for partial dependence

## Summary

Mutual information $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y)) = H(X) - H(X|Y)$ measures the amount of information shared between two random variables. It is symmetric, non-negative, and zero iff $X$ and $Y$ are independent. Unlike correlation, MI captures all forms of statistical dependence. It is bounded above by $\min(H(X), H(Y))$ and satisfies the data processing inequality. Gaussian MI has the closed form $-\frac{1}{2}\log(1-\rho^2)$. In machine learning, MI is used for feature selection (including mRMR), representation learning (InfoGAN, information bottleneck), clustering evaluation (NMI), and interpretability. Estimating MI from finite data requires bias correction; popular methods include KSG estimation and neural lower bounds (MINE).

## Key Takeaways

- $I(X;Y) = D_{KL}(P(x,y)\|P(x)P(y)) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)$.
- $0 \leq I(X;Y) \leq \min(H(X), H(Y))$; $0$ iff independent, $\min(H(X),H(Y))$ iff one determines the other.
- Symmetric: $I(X;Y) = I(Y;X)$.
- Captures all dependencies (linear and nonlinear), unlike correlation.
- Gaussian closed form: $I(X;Y) = -\frac{1}{2}\log(1-\rho^2)$.
- Data processing inequality: $I(X;Y) \geq I(f(X);g(Y))$.
- Feature selection: maximise $I(\text{feature}; \text{target})$, minimise redundancy.
- NMI: $\text{NMI} = 2I/(H(X)+H(Y))$, standard clustering metric.
- InfoGAN: $\max I(c; G(z,c))$ for disentangled representations.
- Estimation requires bias correction (KSG, MINE).
