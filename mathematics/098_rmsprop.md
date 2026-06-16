# Concept: RMSProp

## Concept ID

MATH-098

## Difficulty

Advanced

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Derive the RMSProp update rule from the need for per-parameter adaptive learning rates
- Explain how RMSProp normalizes gradients by their root-mean-square history
- Analyze the role of the decay rate $\beta$ in controlling the adaptation timescale
- Compare RMSProp with AdaGrad and identify scenarios where each is preferred
- Implement RMSProp for training neural networks with vanishing or exploding gradients
- Understand RMSProp's role as a precursor to Adam

## Prerequisites

- Gradient Descent: update rule, learning rate sensitivity
- SGD: mini-batch gradients, gradient noise
- Momentum: exponential moving averages, velocity accumulation
- Backpropagation: vanishing and exploding gradient problems

## Definition

**RMSProp** (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm that maintains a running average of squared gradients to normalize each parameter's learning rate. The update rule is:

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

where:
- $g_t = \nabla L(\theta_t)$ is the gradient at step $t$
- $E[g^2]_t$ is the running average of squared gradients (second moment)
- $\beta \in [0, 1)$ is the decay rate (typically $\beta = 0.9$)
- $\alpha$ is the global learning rate
- $\epsilon$ is a small constant (typically $10^{-8}$) for numerical stability

Each parameter $\theta_i$ gets its own effective learning rate $\alpha / \sqrt{E[g_i^2]_t + \epsilon}$, which adapts based on the historical gradient magnitude for that parameter.

## Intuition

Imagine you are hiking down a mountain with varying terrain. In some directions, the slope is steep (large gradient), and you need to take small, careful steps to avoid falling. In other directions, the slope is gentle (small gradient), and you can take larger strides.

RMSProp automatically adjusts your step size per direction. Parameters that consistently receive large gradients (steep directions) get smaller effective learning rates. Parameters with small gradients (flat directions) get larger effective learning rates. This balances progress across all parameters.

The running average $E[g^2]_t$ acts like a memory of recent gradient magnitudes. When a parameter's recent gradients have been large (indicating high curvature or steepness), the denominator grows, shrinking the step for that parameter. When gradients have been small, the denominator shrinks, allowing larger steps.

Unlike AdaGrad (which accumulates squared gradients from the entire training history), RMSProp uses an exponentially decaying average that forgets old gradients. This is crucial for non-stationary objectives common in deep learning, where gradient statistics change as the model learns.

## Why This Concept Matters

RMSProp addresses a fundamental limitation of standard SGD: the same learning rate applies to all parameters. In deep neural networks, different layers and different parameters naturally operate at different scales. The gradients in early layers can be orders of magnitude smaller than those in later layers (vanishing gradients) or larger (exploding gradients).

RMSProp automatically handles this by adapting per-parameter learning rates. This makes it especially effective for:
- **Recurrent neural networks (RNNs)** where vanishing/exploding gradients are severe
- **Deep networks** with heterogeneous layer sizes
- **Problems with sparse features** where some parameters are updated rarely
- **Non-stationary objectives** where gradient statistics evolve during training

RMSProp was a critical step toward fully adaptive methods and directly inspired the Adam optimizer, which combines RMSProp's adaptive learning rates with momentum.

## Historical Background

RMSProp was proposed by Geoffrey Hinton in Lecture 6 of his Coursera course "Neural Networks for Machine Learning" in 2012. It was never published as a peer-reviewed paper but became one of the most widely used optimization algorithms in deep learning.

RMSProp was developed as an improvement to AdaGrad (Duchi, Hazan, and Singer, 2011). AdaGrad adapts per-parameter learning rates by accumulating the sum of squared gradients from the entire training history. While elegant, AdaGrad's learning rates monotonically decrease to zero, eventually stopping learning entirely. This works for convex problems but fails for deep learning, where gradient statistics change over time.

RMSProp replaced the unbounded sum with an exponentially weighted moving average, preventing the learning rates from decaying to zero. This simple modification made adaptive optimization practical for deep neural networks.

RMSProp was later extended by Adam (Kingma and Ba, 2014), which added momentum via the first moment estimate and bias correction.

## Real World Examples

- **Speech recognition**: RMSProp is effective for training RNNs and LSTMs for speech-to-text systems, where vanishing gradients are a major challenge.
- **Language modeling**: Training recurrent language models benefits from RMSProp's handling of long-range dependencies.
- **Reinforcement learning**: Many RL algorithms (e.g., A3C) use RMSProp because of its ability to handle non-stationary gradient distributions.
- **Generative models**: Training GANs with RMSProp provides more stable convergence than SGD in many cases.
- **Time series forecasting**: Recurrent forecast models benefit from RMSProp's adaptive learning rates across different temporal scales.

## AI/ML Relevance

**Vanishing/exploding gradients in RNNs**: RNNs process sequences, and gradients must flow backward through many time steps. The gradients for early time steps are products of many Jacobians, often causing them to vanish (approach zero) or explode (grow exponentially). RMSProp's per-parameter normalization helps maintain appropriate update sizes across all time steps.

**Handling sparse features**: In recommendation systems and NLP, many features are sparse (e.g., one-hot encoded words). Parameters for rare features receive gradients infrequently. RMSProp gives these parameters larger effective learning rates when they do receive gradients, because their running average $E[g^2]$ remains small.

**Concrete example**: Training a sentiment analysis RNN. Output layer gradients may be 100x larger than input embedding gradients. SGD would either make embedding learning too slow (small LR) or output learning unstable (large LR). RMSProp automatically scales updates appropriately.

### Comparison with AdaGrad

AdaGrad: $G_t = \sum_{\tau=1}^t g_\tau^2$, $\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} g_t$

The cumulative sum $G_t$ grows monotonically, so learning rates always decrease. After many iterations, $G_t$ is so large that effective learning rates approach zero, stopping learning.

RMSProp: $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$

The running average can both increase and decrease. If gradient magnitudes decrease, $E[g^2]_t$ decays, allowing learning rates to increase again. This is essential for non-convex deep learning.

## Mathematical Explanation

### Per-Parameter Adaptation

Let $g_{t,i}$ be the $i$-th component of the gradient at step $t$. The per-parameter update is:

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\alpha}{\sqrt{E[g_i^2]_t + \epsilon}} g_{t,i}
$$

where $E[g_i^2]_t = \beta E[g_i^2]_{t-1} + (1-\beta) g_{t,i}^2$.

### Effective Learning Rate Interpretation

The effective learning rate for parameter $i$ is $\alpha / \sigma_{t,i}$ where $\sigma_{t,i} = \sqrt{E[g_i^2]_t}$ is the RMS (root mean square) of recent gradients. Parameters with large RMS gradient get small effective steps; parameters with small RMS gradient get large effective steps.

### Relationship to Natural Gradient

RMSProp can be seen as a diagonal approximation to natural gradient descent. Natural gradient uses the Fisher information matrix $F$: $\theta_{t+1} = \theta_t - \alpha F^{-1} g_t$. RMSProp approximates $F$ as a diagonal matrix with entries proportional to $E[g_i^2]_t$, the expected squared gradients.

### Scale Invariance

Consider multiplying the loss by a constant $c$. Gradients also multiply by $c$, so $E[g^2]$ multiplies by $c^2$, and $\sqrt{E[g^2]}$ multiplies by $c$. The effective step $\alpha g / \sqrt{E[g^2]}$ is invariant to this scaling. RMSProp is approximately scale-invariant, meaning it automatically adjusts to the scale of the objective.

## Formula(s)

**RMSProp Update**:

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

**Element-wise Form**:

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\alpha}{\sqrt{v_{t,i} + \epsilon}} g_{t,i}
$$

where $v_{t,i} = \beta v_{t-1,i} + (1 - \beta) g_{t,i}^2$.

**RMSProp with Momentum** (common variant):

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t + \epsilon}} m_t
$$

(The momentum variant combines RMSProp's adaptive learning rates with momentum. This is essentially Adam without bias correction.)

## Properties

1. **Per-parameter learning rates**: Each parameter gets an individually adapted learning rate.
2. **Non-monotonic adaptation**: Learning rates can increase if gradient magnitudes decrease, unlike AdaGrad.
3. **Scale invariance**: Approximately invariant to constant scaling of the loss function.
4. **Suitable for non-stationary**: The running average adapts to changing gradient statistics.
5. **Robust to hyperparameters**: Less sensitive to learning rate choice than SGD.
6. **Stabilizes RNN training**: Particularly effective for recurrent architectures.
7. **No bias correction**: Unlike Adam, RMSProp does not correct for initialization bias of the running average (but this matters less in practice).

## Step-by-Step Worked Examples

### Example 1: RMSProp for a Simple 2D Function

**Problem**: Minimize $f(x, y) = 0.1x^2 + 10y^2$ using RMSProp. Start at $(x_0, y_0) = (4, 2)$. Use $\alpha = 0.5$, $\beta = 0.9$, $\epsilon = 10^{-8}$. Perform 5 iterations.

**Solution**:

Gradient: $\nabla f = (0.2x, 20y)^T$

Initialize: $v_0 = (0, 0)$

**Iteration 1**:
$g_1 = (0.2 \cdot 4, 20 \cdot 2) = (0.8, 40)$
$v_1 = 0.9(0, 0) + 0.1(0.8^2, 40^2) = (0.064, 160)$
$\theta_1 = (4, 2) - 0.5 \cdot (0.8/\sqrt{0.064}, 40/\sqrt{160})$
$= (4, 2) - 0.5 \cdot (0.8/0.253, 40/12.649)$
$= (4, 2) - 0.5 \cdot (3.162, 3.162)$
$= (4, 2) - (1.581, 1.581)$
$= (2.419, 0.419)$

Notice both entries received the same effective step magnitude (3.162 before LR scaling, 1.581 after) despite very different raw gradients (0.8 vs 40)! The $y$ gradient was 50x larger, but RMSProp normalized it.

**Iteration 2**:
$g_2 = (0.2 \cdot 2.419, 20 \cdot 0.419) = (0.484, 8.38)$
$v_2 = 0.9(0.064, 160) + 0.1(0.484^2, 8.38^2)$
$= 0.9(0.064, 160) + 0.1(0.234, 70.224)$
$= (0.0576 + 0.0234, 144 + 7.022)$
$= (0.081, 151.022)$
$\theta_2 = (2.419, 0.419) - 0.5 \cdot (0.484/\sqrt{0.081}, 8.38/\sqrt{151.022})$
$= (2.419, 0.419) - 0.5 \cdot (0.484/0.285, 8.38/12.289)$
$= (2.419, 0.419) - 0.5 \cdot (1.699, 0.682)$
$= (2.419, 0.419) - (0.850, 0.341)$
$= (1.569, 0.078)$

**Iteration 3**:
$g_3 = (0.2 \cdot 1.569, 20 \cdot 0.078) = (0.314, 1.56)$
$v_3 = 0.9(0.081, 151.022) + 0.1(0.314^2, 1.56^2)$
$= (0.0729 + 0.0099, 135.920 + 0.243)$
$= (0.0828, 136.163)$
$\theta_3 = (1.569, 0.078) - 0.5 \cdot (0.314/\sqrt{0.0828}, 1.56/\sqrt{136.163})$
$= (1.569, 0.078) - 0.5 \cdot (0.314/0.288, 1.56/11.669)$
$= (1.569, 0.078) - 0.5 \cdot (1.091, 0.134)$
$= (1.569, 0.078) - (0.545, 0.067)$
$= (1.024, 0.011)$

After 3 iterations, $(x, y) = (1.024, 0.011)$, approaching $(0, 0)$ with both coordinates converging at similar rates despite very different curvatures.

### Example 2: RMSProp vs. SGD for Ill-Conditioned Problem

**Problem**: Compare SGD (LR = 0.1) and RMSProp ($\alpha = 0.5$, $\beta = 0.9$) on $f(x, y) = x^2 + 100y^2$, starting at $(4, 2)$. Show 10 iterations.

**Solution**:

**SGD with $\alpha = 0.1$**:
Update: $(x_{t+1}, y_{t+1}) = (x_t, y_t) - 0.1(2x_t, 200y_t) = (0.8x_t, y_t - 20y_t)$

$y$ update: $y_{t+1} = y_t - 20y_t = -19y_t$. This diverges! The learning rate is too large for the $y$ direction (curvature 200). We must use $\alpha \leq 0.01$ for stability in $y$, but then $x$ converges very slowly.

With $\alpha = 0.005$ (safe for both):
$x_t = 4(0.99)^t$ (decays slowly)
$y_t = 2(-1)^t$ (oscillates between +2 and -2, never converging to 0!)

SGD fails to converge $y$ to 0 even with safe LR because the $y$-oscillation never damps.

**RMSProp with $\alpha = 0.5$, $\beta = 0.9$**:

Initialize $v_0 = (0, 0)$

| t | $x_t$ | $y_t$ | $v_x$ | $v_y$ | effective LR $x$ | effective LR $y$ |
|---|-------|-------|-------|-------|-----------------|-----------------|
| 0 | 4.000 | 2.000 | 0 | 0 | - | - |
| 1 | 2.000 | 0.000 | 6.4 | 4000 | 0.198 | 0.008 |
| 2 | 1.000 | 0.000 | 3.6 | 3600 | 0.278 | 0.008 |
| 3 | 0.500 | 0.000 | 2.1 | 3240 | 0.345 | 0.009 |
| 4 | 0.250 | 0.000 | 1.2 | 2916 | 0.408 | 0.009 |
| 5 | 0.125 | 0.000 | 0.7 | 2624 | 0.454 | 0.010 |

RMSProp converges $y$ to 0 in one step and then decays $x$ geometrically. The effective learning rate for $y$ is automatically reduced to handle the high curvature, while $x$ gets a larger effective rate.

### Example 3: RMSProp for a 1D Problem with Varying Gradient Magnitudes

**Problem**: Minimize $f(x) = 10x^2$ for $x < 0$ and $f(x) = 0.1x^2$ for $x \geq 0$ (piecewise quadratic with different curvatures). Start at $x_0 = -5$. Compare SGD ($\alpha = 0.1$) and RMSProp ($\alpha = 1$, $\beta = 0.9$). Show 15 iterations.

**Solution**:

For $x < 0$: $\nabla f = 20x$
For $x \geq 0$: $\nabla f = 0.2x$

**SGD**:
At $x = -5$, $f'(x) = 20(-5) = -100$.
$x_1 = -5 - 0.1(-100) = -5 + 10 = 5$

Now $x > 0$: $f'(x) = 0.2 \cdot 5 = 1$
$x_2 = 5 - 0.1(1) = 4.9$
$x_3 = 4.9 - 0.1(0.98) = 4.802$
...

SGD takes a huge step from -5 to +5 (overshooting), then slowly descends from +5 toward 0 because the gradient on the right side is very small.

**RMSProp**:
Initialize $v = 0$.

Iteration 1: $g = 20(-5) = -100$
$v_1 = 0.9(0) + 0.1(10000) = 1000$
$x_1 = -5 - 1 \cdot (-100)/\sqrt{1000} = -5 + 100/31.62 = -5 + 3.162 = -1.838$

Iteration 2: $g = 20(-1.838) = -36.76$
$v_2 = 0.9(1000) + 0.1(1351) = 900 + 135.1 = 1035.1$
$x_2 = -1.838 - 1 \cdot (-36.76)/\sqrt{1035.1} = -1.838 + 36.76/32.17 = -1.838 + 1.143 = -0.695$

Iteration 3: $g = 20(-0.695) = -13.9$
$v_3 = 0.9(1035.1) + 0.1(193.2) = 931.6 + 19.32 = 950.9$
$x_3 = -0.695 + 13.9/\sqrt{950.9} = -0.695 + 13.9/30.84 = -0.695 + 0.451 = -0.244$

Now $x$ crosses to positive side at some point and continues with adapted rates. RMSProp avoids the massive overshoot because the denominator $\sqrt{E[g^2]}$ normalizes the step, preventing the large gradient from causing a huge jump.

### Example 4: RMSProp with Momentum Variant

**Problem**: Apply RMSProp with momentum ($\beta_1 = 0.9$, $\beta_2 = 0.9$, $\alpha = 0.1$) to $f(x, y) = x^2 + 50y^2$ starting at $(10, 5)$. Show 5 iterations.

**Solution**:

Initialize $m_0 = (0, 0)$, $v_0 = (0, 0)$.

Gradient: $\nabla f = (2x, 100y)$

**Iteration 1**:
$g_1 = (20, 500)$
$m_1 = 0.9(0,0) + 0.1(20, 500) = (2, 50)$
$v_1 = 0.9(0,0) + 0.1(400, 250000) = (40, 25000)$
$\theta_1 = (10, 5) - 0.1(2/\sqrt{40}, 50/\sqrt{25000}) = (10, 5) - 0.1(0.316, 0.316) = (9.968, 4.968)$

**Iteration 2**:
$g_2 = (19.936, 496.8)$
$m_2 = 0.9(2, 50) + 0.1(19.936, 496.8) = (1.8 + 1.994, 45 + 49.68) = (3.794, 94.68)$
$v_2 = 0.9(40, 25000) + 0.1(397.4, 246810) = (36 + 39.74, 22500 + 24681) = (75.74, 47181)$
$\theta_2 = (9.968, 4.968) - 0.1(3.794/\sqrt{75.74}, 94.68/\sqrt{47181}) = (9.968, 4.968) - 0.1(0.436, 0.436) = (9.924, 4.924)$

The parameters converge to $(0, 0)$ with both coordinates receiving balanced effective steps.

## Visual Interpretation

Consider the contour plot of $f(x, y) = x^2 + 100y^2$. The contours are extremely elongated ellipses (high condition number). Without adaptation, SGD oscillates wildly in the $y$ direction while barely moving in $x$.

RMSProp normalizes the gradient by its RMS history. In the $y$ direction, where gradients are large (due to high curvature), $E[g^2]$ accumulates quickly, reducing the effective learning rate. In the $x$ direction, where gradients are small, $E[g^2]$ stays small, allowing larger steps.

The result is that the effective contour plot becomes approximately circular from the optimizer's perspective. The update steps are roughly equal in all directions, eliminating the zig-zag behavior.

The exponential moving average means that if gradient statistics change (e.g., the optimizer enters a region with different curvature), RMSProp adapts within about $1/(1-\beta)$ iterations.

## Common Mistakes

1. **Using $\epsilon$ too large**: A large $\epsilon$ (e.g., $10^{-4}$) reduces the effect of adaptation, making RMSProp behave more like SGD. The standard $\epsilon = 10^{-8}$ ensures robust adaptation.

2. **Confusing $\beta$ with momentum**: In vanilla RMSProp, $\beta$ controls the decay of squared gradients. Some implementations add a separate momentum parameter. Always check whether your framework's RMSProp includes momentum.

3. **Not tuning $\alpha$**: Although RMSProp is more robust to learning rate than SGD, the global $\alpha$ still needs tuning. Starting with $\alpha = 0.001$ is common.

4. **Expecting monotonic loss decrease**: Like SGD, RMSProp with mini-batches can increase loss on individual steps due to gradient noise.

5. **Using RMSProp when AdaGrad might be better**: For convex problems with sparse features, AdaGrad's cumulative (non-decaying) sum of squared gradients may be preferable.

6. **Forgetting bias correction**: RMSProp's $E[g^2]_t$ is initialized at 0, causing bias toward small denominators (large steps) early in training. Adam corrects this; vanilla RMSProp does not.

7. **Applying to all problems by default**: While RMSProp works well for RNNs, Adam is generally preferred for most deep learning tasks.

## Interview Questions

### Beginner - 5

**Q1**: What does RMSProp stand for?
**A**: Root Mean Square Propagation—it propagates updates scaled by the root mean square of recent gradients.

**Q2**: How does RMSProp differ from standard SGD?
**A**: RMSProp uses a per-parameter adaptive learning rate based on the running average of squared gradients, while SGD uses the same learning rate for all parameters.

**Q3**: What problem does RMSProp solve?
**A**: RMSProp handles heterogeneous gradient scales across parameters, preventing vanishing/exploding updates in deep networks.

**Q4**: What is the role of $\beta$ in RMSProp?
**A**: $\beta$ controls the decay rate of the squared gradient moving average. Typical value is 0.9.

**Q5**: Why is $\epsilon$ added in the denominator?
**A**: To prevent division by zero when the running average of squared gradients is very small (e.g., at initialization or for rarely-updated parameters).

### Intermediate - 5

**Q1**: How does RMSProp differ from AdaGrad?
**A**: AdaGrad accumulates all past squared gradients (monotonic sum), causing learning rates to decrease to zero. RMSProp uses an exponentially decaying average, allowing learning rates to increase again if appropriate.

**Q2**: Why is RMSProp particularly effective for RNNs?
**A**: RNNs suffer from vanishing/exploding gradients across time steps. RMSProp's per-parameter normalization keeps updates well-scaled regardless of gradient magnitude.

**Q3**: Explain the scale invariance property of RMSProp.
**A**: If the loss is multiplied by constant $c$, gradients are multiplied by $c$, squared gradients by $c^2$, so $\sqrt{E[g^2]}$ is multiplied by $c$. The update $\alpha g / \sqrt{E[g^2]}$ becomes $c\alpha g / c\sqrt{E[g^2]} = \alpha g / \sqrt{E[g^2]}$, which is invariant.

**Q4**: How does the momentum variant of RMSProp work?
**A**: It maintains both a first-moment (momentum) and second-moment (RMSProp) estimate. The momentum smooths gradient updates while RMSProp provides adaptation, giving the benefits of both.

**Q5**: What happens if $\beta = 0$ in RMSProp?
**A**: The running average is replaced by the current gradient squared: $E[g^2]_t = g_t^2$. The effective step becomes $\alpha g_t / |g_t| = \alpha \cdot \text{sign}(g_t)$, reducing to sign descent (only direction, not magnitude used).

### Advanced - 3

**Q1**: Derive RMSProp as a diagonal approximation to natural gradient descent and discuss the limitations of this approximation.
**A**: Natural gradient uses $\theta_{t+1} = \theta_t - \alpha F^{-1} g_t$ where $F = \mathbb{E}_{p(y|x,\theta)}[\nabla \log p \nabla \log p^T]$ is the Fisher information matrix. RMSProp approximates $F$ as diagonal with entries $E[g_i^2]$. Limitations: (1) diagonal approximation ignores off-diagonal correlations between parameters, (2) uses gradient outer product rather than true Fisher, (3) the running average with decay $\beta$ gives a biased estimate.

**Q2**: Analyze the convergence properties of RMSProp for non-convex optimization and explain why formal convergence guarantees are weaker than for SGD.
**A**: RMSProp lacks formal convergence guarantees for non-convex objectives in the same way as SGD because the adaptive step size violates the standard assumptions (e.g., bounded step sizes, Robbins-Monro conditions). The effective step size $\alpha / \sqrt{v_t}$ depends on the gradient path, creating a complex feedback loop. Recent work provides convergence guarantees under additional assumptions (e.g., gradient clipping, bounded gradients).

**Q3**: Compare the computational and memory costs of RMSProp vs. SGD and discuss when the additional cost is justified.
**A**: RMSProp stores one additional vector (the running average of squared gradients) compared to SGD, doubling the memory per parameter. For models with billions of parameters (e.g., LLMs), this memory overhead is significant. The computational cost is $O(d)$ per iteration, matching SGD. The additional cost is justified when gradient scales vary significantly across parameters (RNNs, deep networks, sparse features).

## Practice Problems

### Easy - 5

**P1**: What is the update for $E[g^2]_t$ given $t=0$ initialization?
**P2**: For a single parameter with $g_1 = 0.5$, compute $E[g^2]_1$ with $\beta = 0.9$.
**P3**: What is the effective step size for a parameter with $\alpha = 0.1$, $\sqrt{E[g^2]} = 2$, $\epsilon = 10^{-8}$?
**P4**: Why does RMSProp handle non-stationary objectives better than AdaGrad?
**P5**: If a parameter consistently has $|g_t| = 1$, what is the steady-state value of $\sqrt{E[g^2]}$?

### Medium - 5

**P1**: For $f(x) = x^2$, apply RMSProp with $\alpha = 1$, $\beta = 0.9$, starting at $x = 3$. Compute 5 iterations.
**P2**: Compare the trajectories of SGD and RMSProp for $f(x, y) = x^2 + 100y^2$ starting at $(5, 1)$.
**P3**: Show that RMSProp is approximately scale-invariant (multiplying loss by constant $c$ leaves update unchanged).
**P4**: Derive the steady-state value of $\sqrt{E[g^2]}$ given a constant gradient magnitude $g$.
**P5**: Explain why $\beta$ should be higher for noisier gradients.

### Hard - 3

**P1**: Prove that RMSProp with $\beta = 0$ reduces to sign descent: $\theta_{t+1} = \theta_t - \alpha \cdot \text{sign}(g_t)$.
**P2**: Analyze the effect of $\epsilon$ on the effective learning rate for parameters with near-zero gradients.
**P3**: Derive a bound on the learning rate $\alpha$ that ensures convergence of RMSProp for convex quadratic functions.

## Solutions

### Easy - Solutions

**S1**: $E[g^2]_1 = \beta \cdot 0 + (1-\beta) g_1^2 = (1-\beta)g_1^2$.
**S2**: $E[g^2]_1 = 0.9(0) + 0.1(0.25) = 0.025$.
**S3**: $\alpha / \sqrt{E[g^2] + \epsilon} = 0.1 / \sqrt{4 + 10^{-8}} \approx 0.1 / 2 = 0.05$.
**S4**: AdaGrad's cumulative sum grows monotonically, so learning rates only decrease. RMSProp's running average can increase if gradients decrease, allowing learning rates to recover.
**S5**: $E[g^2]_{ss} = (1-\beta) \cdot 1^2 / (1-\beta) = 1$. So $\sqrt{E[g^2]} = 1$.

### Medium - Solutions

**S1**: $f'(x) = 2x$. $\alpha = 1$, $\beta = 0.9$.
$x_0 = 3$, $v_0 = 0$
$g_1 = 6$, $v_1 = 0 + 0.1(36) = 3.6$, $x_1 = 3 - 6/\sqrt{3.6} = 3 - 3.162 = -0.162$
$g_2 = -0.324$, $v_2 = 0.9(3.6) + 0.1(0.105) = 3.24 + 0.0105 = 3.2505$, $x_2 = -0.162 - (-0.324)/\sqrt{3.2505} = -0.162 + 0.1797 = 0.0177$
(converging to 0 rapidly)

**S2**: For SGD with safe LR: $y$ oscillates indefinitely, $x$ decays slowly. For RMSProp: $y$ converges quickly due to automatic LR reduction, $x$ gets larger effective LR.

**S3**: If $L \to cL$, then $g \to cg$, $g^2 \to c^2 g^2$, $E[g^2] \to c^2 E[g^2]$, $\sqrt{E[g^2]} \to c\sqrt{E[g^2]}$, and $\alpha g / \sqrt{E[g^2]} \to \alpha(cg)/(c\sqrt{E[g^2]}) = \alpha g / \sqrt{E[g^2]}$. Invariant.

**S4**: $v_t = \beta v_{t-1} + (1-\beta)g^2$. At steady state, $v_\infty = \beta v_\infty + (1-\beta)g^2$, so $v_\infty(1-\beta) = (1-\beta)g^2$, $v_\infty = g^2$, $\sqrt{v_\infty} = |g|$.

**S5**: Noisier gradients require more smoothing to get reliable estimates of the underlying gradient magnitude. Higher $\beta$ gives a longer averaging window, reducing the impact of individual noisy samples on $E[g^2]$.

### Hard - Solutions

**S1**: With $\beta = 0$, $E[g^2]_t = g_t^2$. Then $\theta_{t+1} = \theta_t - \alpha g_t / \sqrt{g_t^2 + \epsilon} \approx \theta_t - \alpha g_t / |g_t| = \theta_t - \alpha \cdot \text{sign}(g_t)$, valid for $\epsilon \ll g_t^2$.

**S2**: Near-zero gradients, $\sqrt{E[g^2]} \approx \sqrt{\epsilon}$. The effective step is $\alpha g / \sqrt{\epsilon}$. Setting $\epsilon$ too large gives artificially large steps for small-gradient parameters; too small risks division by zero or numerical instability.

**S3**: For quadratic $f(x) = \frac{1}{2}x^T A x$, RMSProp update is $x_{t+1} = x_t - \alpha D_t^{-1} A x_t$ where $D_t = \text{diag}(\sqrt{v_t})$. Convergence requires $\rho(I - \alpha D^{-1} A) < 1$. Since $D_t$ depends on the trajectory, a rigorous bound is complex, but a sufficient condition is $\alpha < 2 / \lambda_{\max}(D^{-1} A)$.

## Related Concepts

- **AdaGrad**: The predecessor of RMSProp using cumulative squared gradients.
- **Adam**: Combines RMSProp with momentum and bias correction.
- **AdaDelta**: An extension of RMSProp that also adapts the learning rate based on parameter updates.
- **SGD with Momentum**: The base optimizer that RMSProp's adaptation extends.
- **Natural Gradient Descent**: The theoretical framework that RMSProp approximates.
- **Gradient Sign Descent**: The limiting case of RMSProp with $\beta = 0$ and $\epsilon \to 0$.

## Next Concepts

- **Adam**: The optimizer combining RMSProp with momentum and bias correction.
- **Learning Rate Scheduling**: Complementary strategies for adjusting learning rates.
- **AdamW**: Adam with decoupled weight decay for improved regularization.

## Summary

RMSProp is an adaptive learning rate optimization algorithm that maintains a running average of squared gradients $E[g^2]_t$ and divides each parameter's gradient by $\sqrt{E[g^2]_t + \epsilon}$ before applying the learning rate. This gives each parameter an individually adapted step size.

The exponential moving average (controlled by $\beta$) allows RMSProp to adapt to changing gradient statistics, unlike AdaGrad's monotonically decreasing learning rates. RMSProp handles vanishing/exploding gradients, heterogeneous parameter scales, and non-stationary objectives.

RMSProp is particularly effective for RNNs and deep networks. It was instrumental in the development of Adam, which adds momentum and bias correction. Understanding RMSProp is essential for understanding the landscape of adaptive optimization methods.

## Key Takeaways

- RMSProp normalizes gradients by their root mean square: $\theta_{t+1} = \theta_t - \alpha g_t / \sqrt{E[g^2]_t + \epsilon}$.
- $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$ is an exponential moving average of squared gradients.
- Per-parameter learning rates adapt to gradient magnitudes automatically.
- Unlike AdaGrad, learning rates can increase if gradients shrink.
- $\beta = 0.9$ is the standard decay rate for the moving average.
- RMSProp is approximately scale-invariant.
- Effective for RNNs, deep networks, and sparse features.
- Less sensitive to learning rate choice than SGD.
- Precursor to Adam, which adds momentum and bias correction.
- Memory cost: one additional vector per parameter (stores $E[g^2]$).
