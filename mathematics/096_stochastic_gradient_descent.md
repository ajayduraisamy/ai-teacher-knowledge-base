# Concept: Stochastic Gradient Descent (SGD)

## Concept ID

MATH-096

## Difficulty

Intermediate

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Distinguish between batch, mini-batch, and stochastic gradient descent
- Formulate the SGD update rule with mini-batch gradient estimation
- Analyze the bias-variance trade-off in gradient estimation as a function of batch size
- Compare convergence rates of SGD and full-batch GD for convex and non-convex objectives
- Select appropriate batch sizes for different training scenarios
- Implement SGD with proper learning rate schedules for practical deep learning

## Prerequisites

- Gradient Descent: update rule, convergence properties, learning rate
- Probability: expectation, variance, unbiased estimators
- Calculus: gradients, chain rule (backpropagation)
- Linear algebra: matrix operations, norms

## Definition

**Stochastic Gradient Descent (SGD)** is an iterative optimization method that uses a randomly selected subset of data (a **mini-batch**) to estimate the gradient, rather than computing the exact gradient over the full dataset. The update rule is:

$$
\theta_{t+1} = \theta_t - \alpha_t \hat{\nabla} L(\theta_t)
$$

where $\hat{\nabla} L(\theta_t)$ is an unbiased estimate of the true gradient $\nabla L(\theta_t)$, computed from a mini-batch $\mathcal{B}_t$:

$$
\hat{\nabla} L(\theta_t) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla \ell(y_i, f_\theta(x_i))
$$

**Special cases:**
- **Batch GD**: $|\mathcal{B}_t| = N$ (full dataset)
- **Pure SGD**: $|\mathcal{B}_t| = 1$ (single sample)
- **Mini-batch SGD**: $1 < |\mathcal{B}_t| < N$ (small subset)

## Intuition

Imagine you are trying to determine the average height of everyone in a city. The exact average requires measuring every person---expensive and slow. Instead, you measure a random sample of 100 people. The sample average is noisy but close to the true average, and you can compute it 1,000 times faster.

SGD applies the same principle to gradient computation. Instead of computing the gradient over all training examples (which could be millions), it uses a small random batch. Each gradient estimate is noisy, but the noise averages out over many iterations. The speedup per iteration allows many more updates in the same wall-clock time, often leading to faster convergence overall.

The noise in SGD is not just a nuisance---it can be beneficial. The stochasticity helps the optimizer escape sharp minima and saddle points, potentially leading to better generalization.

## Why This Concept Matters

SGD is the default training algorithm for deep learning. Every major deep learning framework (PyTorch, TensorFlow, JAX) implements SGD as a core optimizer. Understanding SGD is essential for:

- Choosing batch sizes (32, 64, 128, 256---these standard values exist for good reasons)
- Setting learning rate schedules (SGD requires more careful scheduling than adaptive methods)
- Diagnosing training instability (noisy loss curves are normal)
- Understanding generalization (SGD's implicit regularization)
- Scaling to large datasets (SGD is the only feasible approach for million-example datasets)

## Historical Background

Stochastic approximation methods date back to Herbert Robbins and Sutton Monro's 1951 paper introducing the Robbins-Monro algorithm for root-finding with noisy observations. This provided the theoretical foundation for stochastic optimization.

The connection to machine learning was made in the 1980s and 1990s. The 1998 paper by Leon Bottou "Online Learning and Stochastic Approximations" formalized SGD for neural network training. Bottou's subsequent work with Yann LeCun established SGD as the practical choice for large-scale learning.

The 2010s saw SGD become the dominant optimizer thanks to three developments:
1. **GPU computing**: Enabled fast mini-batch processing
2. **Large datasets**: ImageNet (1.2M images) made batch GD impractical
3. **Theoretical advances**: Understanding of SGD's generalization properties

The 2012 AlexNet paper used SGD with momentum to achieve breakthrough image classification results, cementing SGD's place in deep learning history.

## Real World Examples

- **ImageNet training**: Modern vision models are trained with SGD on 1.2 million images using batch sizes of 256--1024. Computing the full gradient would require processing all 1.2M images before each update.
- **Language model pretraining**: BERT was pretrained with Adam (which extends SGD) on 3.3 billion words. Full-batch computation is infeasible.
- **Recommendation systems**: YouTube's recommendation model trains on billions of user interactions daily using SGD-based optimizers.
- **Reinforcement learning**: Policy gradient methods use SGD on trajectories, where each trajectory is a "mini-batch" of environment interactions.
- **Federated learning**: SGD is performed on user devices, with each device's local data forming a natural mini-batch.

## AI/ML Relevance

**Why SGD is necessary:** For a dataset with $N = 10^6$ examples and a model with $d = 10^7$ parameters, computing the full gradient requires $O(Nd) = 10^{13}$ operations per update. Using batch size $B = 256$ reduces this to $O(Bd) = 2.56 \times 10^9$---about 4,000x fewer operations.

**Gradient noise and generalization:** The variance of the SGD gradient estimate is:

$$
\text{Var}(\hat{\nabla} L) = \frac{1}{B} \Sigma(\theta)
$$

where $\Sigma(\theta)$ is the covariance of per-sample gradients. This noise has been shown to bias SGD toward flat minima, which correlate with better generalization.

**Concrete example---Logistic regression:** For binary classification with logistic loss $L(w) = \frac{1}{N}\sum \log(1 + e^{-y_i w^T x_i})$, the gradient is:

$$
\nabla L(w) = \frac{1}{N}\sum_{i=1}^N \frac{-y_i x_i}{1 + e^{y_i w^T x_i}}
$$

SGD approximates this using a mini-batch:

$$
\hat{\nabla} L(w) = \frac{1}{B}\sum_{i \in \mathcal{B}} \frac{-y_i x_i}{1 + e^{y_i w^T x_i}}
$$

Each iteration costs $O(Bd)$ instead of $O(Nd)$, enabling training on massive datasets.

## Mathematical Explanation

### Unbiased Gradient Estimation

Let $\xi_t$ be the random variable representing the mini-batch sampled at iteration $t$. The SGD update is:

$$
\theta_{t+1} = \theta_t - \alpha_t g(\theta_t, \xi_t)
$$

where $g(\theta_t, \xi_t) = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)$. By linearity of expectation:

$$
\mathbb{E}[g(\theta_t, \xi_t)] = \nabla L(\theta_t)
$$

### Variance of the Gradient Estimate

The covariance of the gradient estimate is:

$$
\text{Cov}(g(\theta, \xi)) = \frac{1}{B} \Sigma(\theta)
$$

where $\Sigma(\theta) = \frac{1}{N-1}\sum_{i=1}^N (\nabla \ell_i(\theta) - \nabla L(\theta))(\nabla \ell_i(\theta) - \nabla L(\theta))^T$

### Convergence Analysis

Under standard assumptions ($L$-smoothness, bounded variance $\mathbb{E}\|g - \nabla f\|^2 \leq \sigma^2$):

**Convex case** (with decaying learning rate $\alpha_t = 1/\sqrt{t}$):

$$
\mathbb{E}[f(\bar{\theta}_T) - f(\theta^*)] = O\left(\frac{1}{\sqrt{T}}\right)
$$

**Strongly convex case** (with $\alpha_t = 1/\mu t$):

$$
\mathbb{E}[f(\theta_T) - f(\theta^*)] = O\left(\frac{1}{T}\right)
$$

## Formula(s)

**SGD Update**:

$$
\theta_{t+1} = \theta_t - \alpha_t \left(\frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)\right)
$$

**Expected Squared Gradient Norm**:

$$
\mathbb{E}[\|g(\theta)\|^2] = \|\nabla L(\theta)\|^2 + \frac{1}{B}\text{Tr}(\Sigma(\theta))
$$

**Convergence for Convex SGD** (with $\alpha_t = \alpha/\sqrt{t}$):

$$
\frac{1}{T}\sum_{t=1}^T \mathbb{E}[f(\theta_t) - f(\theta^*)] \leq \frac{R^2}{2\alpha\sqrt{T}} + \frac{\alpha\sigma^2}{2\sqrt{T}}
$$

## Properties

1. **Unbiased gradient**: $\mathbb{E}[\hat{\nabla} L] = \nabla L$ (no systematic bias in gradient estimates).
2. **Noisy descent**: Individual updates may increase the loss, but the expected direction is downhill.
3. **Linear scaling rule**: Increasing batch size $B$ by $k$ allows increasing learning rate by $k$ (up to a bound).
4. **Decaying learning rates**: Constant learning rates cause SGD to converge to a noise ball; decaying rates are needed for exact convergence.
5. **Implicit regularization**: SGD's noise biases toward flat minima that generalize better.
6. **Epoch vs. iteration distinction**: One epoch = one pass through all data. The number of SGD iterations per epoch is $N/B$.
7. **Gradient noise scale**: The ratio of gradient variance to gradient norm determines SGD's behavior.

## Step-by-Step Worked Examples

### Example 1: SGD for Linear Regression

**Problem**: Consider 3 data points: $(x_1, y_1) = (1, 2)$, $(x_2, y_2) = (2, 4)$, $(x_3, y_3) = (3, 6)$. Model: $y = wx$. Loss: mean squared error. Perform SGD with batch size $B = 1$, learning rate $\alpha = 0.1$, starting from $w_0 = 0$. Show 3 iterations.

**Solution**:

The per-sample loss and gradient:

$$
\ell_i(w) = (w x_i - y_i)^2
$$

$$
\nabla \ell_i(w) = 2(w x_i - y_i)x_i
$$

**Iteration 1**: Sample $(x_1, y_1) = (1, 2)$

$$
g_1 = 2(w_0 \cdot 1 - 2) \cdot 1 = 2(0 - 2) = -4
$$

$$
w_1 = w_0 - 0.1(-4) = 0.4
$$

**Iteration 2**: Sample $(x_3, y_3) = (3, 6)$

$$
g_2 = 2(w_1 \cdot 3 - 6) \cdot 3 = 2(1.2 - 6) \cdot 3 = 2(-4.8) \cdot 3 = -28.8
$$

$$
w_2 = 0.4 - 0.1(-28.8) = 0.4 + 2.88 = 3.28
$$

**Iteration 3**: Sample $(x_2, y_2) = (2, 4)$

$$
g_3 = 2(w_2 \cdot 2 - 4) \cdot 2 = 2(6.56 - 4) \cdot 2 = 2(2.56) \cdot 2 = 10.24
$$

$$
w_3 = 3.28 - 0.1(10.24) = 3.28 - 1.024 = 2.256
$$

After 3 iterations, $w \approx 2.256$, approaching the true slope of 2. Note the noisy path: $w$ went $0 \to 0.4 \to 3.28 \to 2.256$, overshooting and correcting.

### Example 2: Batch Size Comparison

**Problem**: For the same data as Example 1, compare SGD with $B = 1$, $B = 2$, and $B = 3$ for one epoch. Start at $w = 0$, $\alpha = 0.1$.

**Solution**:

**$B = 1$ (pure SGD)**: 3 updates per epoch
- Update 1 (sample 1): $g = -4$, $w = 0.4$
- Update 2 (sample 3): $g = -28.8$, $w = 3.28$
- Update 3 (sample 2): $g = 10.24$, $w = 2.256$
Final after 1 epoch: $w = 2.256$

**$B = 2$ (mini-batch)**:
- Mini-batch 1: samples (1, 2) and (2, 4)
  $g = \frac{1}{2}[2(0-2)(1) + 2(0-4)(2)] = \frac{1}{2}[-4 - 16] = -10$
  $w = 0 - 0.1(-10) = 1.0$
- Mini-batch 2: sample (3, 6)
  $g = 2(1.0 \cdot 3 - 6) \cdot 3 = 2(-3)(3) = -18$
  $w = 1.0 - 0.1(-18) = 2.8$
Final after 1 epoch: $w = 2.8$

**$B = 3$ (full batch)**:
- Single batch: all 3 samples
  $g = \frac{1}{3}[-4 - 16 - 36] = -\frac{56}{3} \approx -18.67$
  $w = 0 - 0.1(-18.67) = 1.867$
Final after 1 epoch: $w = 1.867$

Larger batches give more accurate updates (less variance) but fewer updates per epoch.

### Example 3: Learning Rate Decay

**Problem**: Run SGD on $f(x) = x^2$ with constant LR $\alpha = 0.1$ and decaying LR $\alpha_t = 0.1/t$, starting at $x_0 = 5$. Compare after 10 iterations.

**Solution**:

$f'(x) = 2x$, SGD update: $x_{t+1} = x_t - \alpha_t(2x_t) = x_t(1 - 2\alpha_t)$

**Constant LR $\alpha = 0.1$**:

| $t$ | $x_t$ | $f(x_t)$ |
|-----|-------|----------|
| 0   | 5.000 | 25.0000 |
| 1   | 4.000 | 16.0000 |
| 2   | 3.200 | 10.2400 |
| 3   | 2.560 | 6.5536 |
| 4   | 2.048 | 4.1943 |
| 5   | 1.638 | 2.6844 |
| 6   | 1.311 | 1.7180 |
| 7   | 1.049 | 1.0995 |
| 8   | 0.839 | 0.7037 |
| 9   | 0.671 | 0.4504 |
| 10  | 0.537 | 0.2882 |

Converges to 0 geometrically. For the noise-free case, constant and decaying LR give the same trajectory. With noise, constant LR leads to a noise ball around the optimum; decaying LR enables exact convergence.

### Example 4: SGD with Gradient Noise

**Problem**: Consider $f(x) = x^2$ but with noisy gradients $g_t = 2x_t + \epsilon_t$ where $\epsilon_t \sim \mathcal{N}(0, 1)$. Compare constant $\alpha = 0.1$ vs decaying $\alpha_t = 0.1/t$ over 50 iterations starting from $x_0 = 5$.

**Solution** (illustrative):

With **constant LR**, the iterates bounce around $x = 0$ but never settle exactly. The long-run average converges to a stationary distribution centered at $0$ with variance proportional to $\alpha$.

With **decaying LR**, the early iterations (large $\alpha$) make rapid progress, and later iterations (small $\alpha$) precisely converge. The Polyak-Ruppert averaged iterate $\bar{x}_T = \frac{1}{T}\sum_{t=1}^T x_t$ achieves the optimal $O(1/T)$ convergence rate.

## Visual Interpretation

Imagine the loss landscape's contour plot with the full-batch gradient showing the exact downhill direction as a perfect arrow. SGD's gradient is a noisy arrow: it generally points downhill but jitters around the true direction.

With batch size $B = 1$, the arrow is very noisy---pointing in the correct direction about 60-70% of the time. With $B = 32$, the arrow is much more reliable. The noise level is proportional to $1/\sqrt{B}$.

The trajectory of SGD looks like a random walk superimposed on a downhill path. It explores the landscape while descending, which helps in non-convex optimization.

In terms of loss curves:
- Full-batch GD: smooth, monotonically decreasing (for convex objectives)
- SGD ($B > 1$): noisy but generally decreasing
- Pure SGD ($B = 1$): very noisy, may increase temporarily

## Common Mistakes

1. **Using too large a batch size**: Large batches reduce gradient noise but also reduce the number of updates per epoch and can hurt generalization. Very large batches require careful learning rate tuning.

2. **Not shuffling data**: Using data in a fixed order creates correlations between consecutive gradients. Always shuffle data at each epoch.

3. **Constant learning rate for too long**: Without decay, SGD with noise never converges exactly---it bounces around the minimum.

4. **Confusing epoch and iteration**: An epoch equals $N/B$ iterations. The number of epochs determines how many times the model has seen each example.

5. **Setting batch size too small for GPU efficiency**: Modern GPUs process batches of 256 or 512 as fast as batches of 1 due to parallelization.

6. **Ignoring gradient accumulation**: When GPU memory limits batch size, gradients can be accumulated over several forward/backward passes before updating.

7. **Thinking SGD always converges with constant LR**: For noisy problems, constant LR leads to convergence only to a neighborhood of the optimum.

8. **Neglecting the linear scaling rule**: If you increase batch size by $k$, you should typically increase the learning rate by $k$ (within bounds).

## Interview Questions

### Beginner - 5

**Q1**: What does "stochastic" mean in stochastic gradient descent?
**A**: "Stochastic" refers to the random sampling of mini-batches. The gradient estimate is random rather than deterministic.

**Q2**: Why does SGD use mini-batches instead of the full dataset?
**A**: Computing the gradient over the full dataset is computationally expensive for large datasets. Mini-batches provide an unbiased, computationally cheaper gradient estimate.

**Q3**: What is the difference between an epoch and an iteration?
**A**: An iteration is one parameter update (one mini-batch). An epoch is one complete pass through the entire training dataset.

**Q4**: Is SGD guaranteed to reduce the loss at every step?
**A**: No. The gradient estimate is noisy, so the loss may increase on individual steps. Only the expected loss decreases.

**Q5**: What is a typical batch size?
**A**: Common choices are 32, 64, 128, or 256. These balance GPU efficiency with gradient accuracy.

### Intermediate - 5

**Q1**: How does batch size affect gradient estimation variance?
**A**: Variance decreases as $1/B$. Larger batches give more accurate gradient estimates but reduce the number of updates per epoch.

**Q2**: What is the linear scaling rule for learning rate and batch size?
**A**: If batch size is multiplied by $k$, the learning rate should also be multiplied by $k$ (up to a maximum). This maintains the variance of the parameter update.

**Q3**: Why might SGD generalize better than full-batch GD?
**A**: SGD's gradient noise biases toward flatter minima, which are less sensitive to parameter perturbations and generalize better.

**Q4**: What happens if you never decay the learning rate in SGD?
**A**: The iterates converge to a neighborhood of the optimum (a "noise ball") but never reach the exact optimum.

**Q5**: Explain the trade-off between batch size and learning rate.
**A**: Larger batches give more accurate gradients, allowing larger learning rates. But the linear scaling rule has an upper bound; beyond a certain point, larger batches with proportionally larger LRs fail.

### Advanced - 3

**Q1**: Derive the optimal batch size for SGD under a fixed time budget, considering both gradient computation and communication costs in distributed training.
**A**: The optimal batch size balances gradient variance reduction against the cost of processing more samples per update. In distributed settings, communication overhead ($O(d)$ per worker) must be amortized over computation ($O(Bd)$ per worker).

**Q2**: Explain the relationship between SGD and Langevin dynamics, and how this leads to the Bayesian interpretation of SGD.
**A**: When the learning rate is constant, SGD follows a discretized Langevin diffusion. The stationary distribution of the iterates concentrates around minima with probability proportional to $\exp(-f(\theta)/T)$ where $T \propto \alpha$.

**Q3**: Prove that SGD with decreasing learning rates converges almost surely to a stationary point under standard assumptions.
**A**: Under Robbins-Monro conditions ($\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$) and standard smoothness/variance assumptions, SGD converges almost surely. The proof uses the supermartingale convergence theorem.

## Practice Problems

### Easy - 5

**P1**: For a dataset of 10,000 examples and batch size 128, how many iterations per epoch?
**P2**: Is SGD's gradient estimate biased or unbiased?
**P3**: If the gradient variance is $\sigma^2 = 10$ with $B = 1$, what is the variance with $B = 64$?
**P4**: Why must data be shuffled before each epoch?
**P5**: What are the three variants of gradient descent based on batch size?

### Medium - 5

**P1**: Derive the variance of the SGD gradient estimate in terms of per-sample gradient covariance.
**P2**: For $f(x) = \frac{1}{N}\sum_{i=1}^N (x - a_i)^2$, show that SGD with a single sample has the same expected update as full-batch GD but with added noise.
**P3**: Given that doubling batch size allows doubling learning rate, show that the expected parameter update magnitude remains approximately constant.
**P4**: Compare the FLOPs for one full-batch GD update vs. one SGD update for $N = 10^6$ and $d = 10^6$, $B = 256$.
**P5**: Explain why SGD benefits from cyclic learning rates rather than monotonic decay.

### Hard - 3

**P1**: Prove the convergence rate $O(1/\sqrt{T})$ of SGD for convex functions with bounded gradients.
**P2**: Derive the central limit theorem for SGD iterates: show that $\sqrt{T}(\theta_T - \theta^*)$ converges to a Gaussian distribution.
**P3**: Analyze the effect of batch size on the spectral properties of the gradient noise covariance and its relationship to the Hessian.

## Solutions

### Easy - Solutions

**S1**: $N/B = 10000/128 \approx 78.125$, so 78 or 79 iterations per epoch.
**S2**: Unbiased. $\mathbb{E}[\hat{\nabla} L] = \nabla L$ because mini-batches are sampled uniformly.
**S3**: Variance scales as $1/B$, so $\sigma^2_{64} = 10/64 \approx 0.156$.
**S4**: Without shuffling, consecutive mini-batches are correlated. Shuffling ensures independence and unbiased gradient estimates.
**S5**: Batch GD ($B = N$), mini-batch SGD ($1 < B < N$), and pure SGD ($B = 1$).

### Medium - Solutions

**S1**: $\text{Var}(\hat{\nabla} L(\theta)) = \frac{1}{B} \Sigma(\theta)$, where $\Sigma(\theta)$ is the per-sample gradient covariance.
**S2**: Full gradient: $f'(x) = \frac{2}{N}\sum(x - a_i)$. SGD: $g_t = 2(x_t - a_i)$. $\mathbb{E}[g_t] = f'(x_t)$. Unbiased.
**S3**: Parameter update $\Delta \theta = \alpha \cdot g(\theta)$. With double LR and batch size, the expected squared update magnitude remains approximately constant.
**S4**: Full-batch: $O(Nd) = 10^{12}$ FLOPs. SGD: $O(Bd) = 256 \times 10^6$ FLOPs. SGD is about 3,900x cheaper per iteration.
**S5**: Cyclic learning rates combine exploration benefits of high LR (escaping sharp minima) with convergence of low LR (fine-tuning).

### Hard - Solutions

**S1**: Using convexity and bounded gradients: $\mathbb{E}[f(\theta_t) - f(\theta^*)] \leq \frac{1}{2\alpha_t}(\mathbb{E}\|\theta_t - \theta^*\|^2 - \mathbb{E}\|\theta_{t+1} - \theta^*\|^2) + \frac{\alpha_t G^2}{2}$. Summing with $\alpha_t = \alpha/\sqrt{t}$ yields $O(1/\sqrt{T})$.

**S2**: Under appropriate conditions, $\sqrt{T}(\theta_T - \theta^*)$ converges to $\mathcal{N}(0, \Sigma^*)$ where $\Sigma^* = H^{-1} S H^{-1}$, $H = \nabla^2 f(\theta^*)$, and $S$ is the gradient noise covariance.

**S3**: The gradient noise covariance is $C = \frac{1}{B}\Sigma(\theta)$. The alignment of $C$ with $H$ determines escaping behavior. Anisotropic noise aligned with sharp directions helps escape sharp minima.

## Related Concepts

- **Batch Gradient Descent**: The deterministic precursor to SGD.
- **Mini-batch SGD**: The practical variant using batch sizes between 1 and N.
- **Momentum**: Accelerates SGD by smoothing gradient estimates.
- **Adam**: Adaptive learning rate extension of SGD.
- **Learning Rate Scheduling**: Decay strategies for SGD convergence.
- **Variance Reduction**: Methods that reduce SGD noise (SVRG, SAGA).
- **Distributed SGD**: Parallelizing SGD across multiple workers.
- **Implicit Regularization**: The bias of SGD toward flat minima.

## Next Concepts

- **Momentum**: Accelerating SGD with velocity accumulation.
- **RMSProp**: Adaptive per-parameter learning rates.
- **Adam**: Combining momentum with adaptive learning rates.
- **Learning Rate Scheduling**: Systematic strategies for adjusting learning rates.

## Summary

Stochastic Gradient Descent (SGD) is the default optimization algorithm for deep learning. It estimates the gradient using a random mini-batch of data, trading gradient accuracy for computational efficiency. The gradient estimate is unbiased with variance inversely proportional to batch size.

SGD's noise is beneficial for generalization, biasing toward flat minima. However, it requires decaying learning rates for exact convergence. Key hyperparameters include batch size (typically 32--256), learning rate, and decay schedule.

Compared to full-batch GD, SGD converges more slowly per iteration (sublinear vs. linear) but much faster per unit of computation on large datasets. Understanding SGD is essential for effective deep learning practice.

## Key Takeaways

- SGD uses random mini-batches to estimate gradients, trading accuracy for speed.
- The gradient estimate is unbiased: $\mathbb{E}[\hat{\nabla} L] = \nabla L$.
- Variance scales as $1/B$; larger batches give more accurate gradients.
- Decaying learning rates are necessary for exact convergence.
- SGD's noise provides implicit regularization toward flat minima.
- The linear scaling rule: increase LR proportionally with batch size.
- One epoch = N/B iterations; shuffling data each epoch is critical.
- Convergence rate: $O(1/\sqrt{T})$ for convex, $O(1/T)$ for strongly convex.
- Batch size choice (32--256) balances gradient accuracy with GPU efficiency.
- Understanding SGD is prerequisite to mastering advanced optimizers.
