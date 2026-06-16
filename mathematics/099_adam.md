# Concept: Adam

## Concept ID

MATH-099

## Difficulty

Advanced

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Derive the Adam update rule as a combination of momentum and RMSProp with bias correction
- Explain the role of each hyperparameter ($\alpha$, $\beta_1$, $\beta_2$, $\epsilon$) in the Adam algorithm
- Compute one full Adam update step manually for a simple function
- Analyze the bias correction mechanism and why it is necessary
- Compare Adam with SGD, momentum, and RMSProp in terms of convergence behavior
- Identify scenarios where Adam outperforms or underperforms other optimizers

## Prerequisites

- Momentum: first moment estimation, exponential moving average
- RMSProp: second moment estimation, per-parameter learning rate adaptation
- SGD: mini-batch gradient estimation, learning rate scheduling
- Bias correction: understanding initialization bias in moving averages

## Definition

**Adam** (Adaptive Moment Estimation) is a first-order gradient-based optimization algorithm that computes adaptive per-parameter learning rates by combining ideas from momentum and RMSProp. Adam maintains two running averages:

1. **First moment estimate** $m_t$ (the mean of gradients, like momentum):
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$

2. **Second moment estimate** $v_t$ (the uncentered variance of gradients, like RMSProp):
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$

Both estimates are biased toward zero at initialization (since $m_0 = v_0 = 0$). Adam corrects this bias:

3. **Bias correction**:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

4. **Parameter update**:
   $$
   \theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

**Default hyperparameters** (from the original paper):
- $\alpha = 0.001$ (learning rate)
- $\beta_1 = 0.9$ (first moment decay rate)
- $\beta_2 = 0.999$ (second moment decay rate)
- $\epsilon = 10^{-8}$ (numerical stability)

## Intuition

Adam can be understood as RMSProp with momentum, plus corrections for initialization bias.

Imagine driving a car with both a speedometer and an accelerometer. The **first moment** $m_t$ is like your velocity: it smoothes out bumps (gradient noise) and gives you a consistent direction. The **second moment** $v_t$ is like the roughness of the road: if the road is bumpy (large gradient variance), you drive slower (smaller effective learning rate). On smooth roads (consistent gradients), you can drive faster.

Bias correction is needed because both estimates start at zero. In the first few steps, $m_t$ and $v_t$ are too small. Without correction, the first steps would be tiny (momentum not yet built up) and the effective learning rate would be too large (since dividing by a too-small $\sqrt{v_t}$). Bias correction fixes this by scaling up early estimates.

The result is an optimizer that:
- Moves in a smoothed direction (like momentum)
- Adjusts step size per parameter (like RMSProp)
- Has robust default hyperparameters
- Works well out-of-the-box for a wide range of deep learning problems

## Why This Concept Matters

Adam is the default optimizer for most deep learning applications. Since its introduction in 2014, it has become the most widely used optimization algorithm in machine learning. Understanding Adam is essential because:

- It consistently outperforms SGD on many tasks, especially transformers and large language models
- Its hyperparameters are more robust and require less tuning than SGD
- It handles sparse gradients, noisy objectives, and non-stationary targets effectively
- It has become the baseline optimizer for comparing new optimization methods

Adam's design represents the culmination of decades of optimization research, combining the best ideas from momentum, adaptive learning rates, and bias correction into a single practical algorithm.

## Historical Background

Adam was introduced by Diederik Kingma and Jimmy Ba in their 2014 paper "Adam: A Method for Stochastic Optimization." The paper has become one of the most cited in machine learning, with over 100,000 citations.

Adam built on several previous developments:
- **SGD with Momentum** (Polyak, 1964): Velocity accumulation
- **AdaGrad** (Duchi et al., 2011): Per-parameter learning rates with cumulative squared gradients
- **RMSProp** (Hinton, 2012): Per-parameter learning rates with running average of squared gradients
- **AdaDelta** (Zeiler, 2012): Adaptive learning rates without a global learning rate

Adam's key innovation was combining first and second moment estimates with bias correction. The bias correction was crucial for making adaptive methods work well in practice, especially in the early training steps.

Subsequent developments include:
- **AdamW** (Loshchilov & Hutter, 2017): Decoupled weight decay for better regularization
- **Nadam** (Dozat, 2016): Adam with Nesterov momentum
- **AMSGrad** (Reddi et al., 2018): A variant addressing convergence issues with adaptive methods
- **LAMB** (You et al., 2019): Layer-wise adaptive moments for large-batch training

## Real World Examples

- **Large language models**: GPT, BERT, and their variants are typically pretrained with Adam or AdamW. The adaptive learning rates handle the varying scales across transformer layers.
- **Image generation**: Stable Diffusion and DALL-E use Adam variants for training diffusion models.
- **Reinforcement learning**: Deep Q-Networks and policy gradient methods frequently use Adam.
- **Graph neural networks**: Training GNNs on irregular graph structures benefits from Adam's robustness.
- **Federated learning**: FedAvg often uses Adam as the local optimizer on client devices.

## AI/ML Relevance

**Why Adam dominates**: Adam combines the benefits of two successful ideas:
- **Momentum** (via $m_t$): Accelerates convergence in consistent gradient directions, damps oscillations.
- **Adaptive learning rates** (via $v_t$): Handles heterogeneous parameter scales, automatically adjusts step sizes.

**Robust default hyperparameters**: The defaults ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$) work well across many tasks. This reduces the need for extensive hyperparameter tuning, a major practical advantage.

**Concrete example**: Training BERT-base (110M parameters). Using SGD would require careful learning rate tuning, gradient clipping, and potentially warmup schedules. Adam with default settings converges reliably without gradient clipping, and the learning rate 0.001 works well with linear warmup and decay.

**When Adam may not be best**: 
- **Computer vision with CNNs**: SGD with momentum often achieves better test accuracy than Adam, possibly because Adam's adaptive rates find sharper minima.
- **Very large models**: Adam's memory cost (2x parameters for $m$ and $v$) is significant. Adafactor reduces this.
- **When generalization is critical**: Adam may converge to sharper minima than SGD, leading to worse generalization. AdamW and SGD with careful tuning can outperform.

### The Adam Update Algorithm (Pseudo-code)

```
Initialize: theta_0, m_0 = 0, v_0 = 0, t = 0
while not converged:
    t = t + 1
    g_t = gradient(L, theta_{t-1})
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - alpha * m_hat / (sqrt(v_hat) + epsilon)
```

## Mathematical Explanation

### First Moment (Momentum)

$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ is an exponential moving average of gradients. Expanding:

$$
m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
$$

The weights sum to $1 - \beta_1^t$, not 1. Hence the bias correction factor $1/(1 - \beta_1^t)$.

### Second Moment (Adaptive Learning Rates)

$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ is an exponential moving average of squared gradients:

$$
v_t = (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} g_i^2
$$

The effective learning rate for parameter $j$ is $\alpha / (\sqrt{\hat{v}_{t,j}} + \epsilon)$.

### Bias Correction Derivation

The expected value of $m_t$ at step $t$ is:

$$
\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbb{E}[g_i]
$$

If the gradient distribution is stationary, $\mathbb{E}[g_i] \approx \mathbb{E}[g]$, then:

$$
\mathbb{E}[m_t] \approx (1 - \beta_1) \mathbb{E}[g] \sum_{i=1}^t \beta_1^{t-i} = (1 - \beta_1^t) \mathbb{E}[g]
$$

Thus $m_t$ is biased toward zero by factor $(1 - \beta_1^t)$. Dividing by $(1 - \beta_1^t)$ corrects this bias. The same derivation applies to $v_t$.

### Convergence Analysis

For convex problems, Adam achieves $O(1/\sqrt{T})$ regret bound (same as SGD) under appropriate conditions. The analysis requires the bias correction and decreasing learning rates.

## Formula(s)

**Complete Adam Update**:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**Default Values**:

$$
\beta_1 = 0.9, \quad \beta_2 = 0.999, \quad \epsilon = 10^{-8}, \quad \alpha = 0.001
$$

**AdamW (decoupled weight decay)**:

$$
\theta_{t+1} = \theta_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

where $\lambda$ is the weight decay coefficient.

## Properties

1. **Per-parameter learning rates**: Each parameter gets an individually adapted step size.
2. **Bias-corrected estimates**: Initialization bias is corrected, ensuring good behavior from the first step.
3. **Approximately scale-invariant**: Invariant to diagonal rescaling of gradients.
4. **Robust hyperparameters**: Defaults work well across many problems.
5. **Limits to simpler optimizers**: As $\beta_1 \to 0$, Adam approximates RMSProp. As $\beta_2 \to 0$, Adam approximates SGD with momentum.
6. **Suitable for sparse gradients**: Handles features with varying update frequencies.
7. **Theoretical regret bound**: $O(\sqrt{T})$ for convex objectives.

## Step-by-Step Worked Examples

### Example 1: Complete Manual Adam Update (The Capstone Example)

**Problem**: Perform Adam optimization for $f(x, y) = x^2 + 2y^2$ starting at $\theta = (3, 2)$. Use $\alpha = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Compute **3 complete iterations** showing every intermediate value.

**Solution**:

Initialize: $m_0 = (0, 0)$, $v_0 = (0, 0)$, $t = 0$

Gradient: $\nabla f(x, y) = (2x, 4y)$

---

**Iteration 1 ($t = 1$)**:

Step 1: Compute gradient at $\theta_0 = (3, 2)$
$$
g_1 = (2 \cdot 3, 4 \cdot 2) = (6, 8)
$$

Step 2: Update biased first moment
$$
m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9(0, 0) + 0.1(6, 8) = (0.6, 0.8)
$$

Step 3: Update biased second moment
$$
v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.999(0, 0) + 0.001(36, 64) = (0.036, 0.064)
$$

Step 4: Bias correction
$$
\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{(0.6, 0.8)}{1 - 0.9} = \frac{(0.6, 0.8)}{0.1} = (6, 8)
$$

$$
\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{(0.036, 0.064)}{1 - 0.999} = \frac{(0.036, 0.064)}{0.001} = (36, 64)
$$

Step 5: Update parameters
$$
\theta_1 = \theta_0 - \alpha \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon}
$$

For $x$:
$$
\frac{\hat{m}_{1,x}}{\sqrt{\hat{v}_{1,x}} + \epsilon} = \frac{6}{\sqrt{36} + 10^{-8}} = \frac{6}{6} = 1
$$
$$
x_1 = 3 - 0.1 \cdot 1 = 2.9
$$

For $y$:
$$
\frac{\hat{m}_{1,y}}{\sqrt{\hat{v}_{1,y}} + \epsilon} = \frac{8}{\sqrt{64} + 10^{-8}} = \frac{8}{8} = 1
$$
$$
y_1 = 2 - 0.1 \cdot 1 = 1.9
$$

Result: $\theta_1 = (2.9, 1.9)$, loss $= 2.9^2 + 2(1.9)^2 = 8.41 + 7.22 = 15.63$ (down from $9 + 8 = 17$).

Note: At $t = 1$, Adam's bias correction makes $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, so the update is $\alpha \cdot g_1 / |g_1| = \alpha \cdot \text{sign}(g_1)$. The first step is a sign descent step.

---

**Iteration 2 ($t = 2$)**:

Step 1: Gradient at $\theta_1 = (2.9, 1.9)$
$$
g_2 = (5.8, 7.6)
$$

Step 2: Update first moment
$$
m_2 = \beta_1 m_1 + (1 - \beta_1) g_2 = 0.9(0.6, 0.8) + 0.1(5.8, 7.6) = (0.54 + 0.58, 0.72 + 0.76) = (1.12, 1.48)
$$

Step 3: Update second moment
$$
g_2^2 = (33.64, 57.76)
$$
$$
v_2 = \beta_2 v_1 + (1 - \beta_2) g_2^2 = 0.999(0.036, 0.064) + 0.001(33.64, 57.76)
$$
$$
v_2 = (0.035964 + 0.03364, 0.063936 + 0.05776) = (0.069604, 0.121696)
$$

Step 4: Bias correction
$$
1 - \beta_1^2 = 1 - 0.9^2 = 1 - 0.81 = 0.19
$$
$$
\hat{m}_2 = \frac{(1.12, 1.48)}{0.19} = (5.8947, 7.7895)
$$

$$
1 - \beta_2^2 = 1 - 0.999^2 = 1 - 0.998001 = 0.001999
$$
$$
\hat{v}_2 = \frac{(0.069604, 0.121696)}{0.001999} = (34.819, 60.878)
$$

Step 5: Update parameters
$$
\frac{\hat{m}_{2,x}}{\sqrt{\hat{v}_{2,x}} + \epsilon} = \frac{5.8947}{\sqrt{34.819} + 10^{-8}} = \frac{5.8947}{5.900} \approx 0.999
$$
$$
x_2 = 2.9 - 0.1 \cdot 0.999 = 2.9 - 0.0999 = 2.8001
$$

$$
\frac{\hat{m}_{2,y}}{\sqrt{\hat{v}_{2,y}} + \epsilon} = \frac{7.7895}{\sqrt{60.878} + 10^{-8}} = \frac{7.7895}{7.803} \approx 0.998
$$
$$
y_2 = 1.9 - 0.1 \cdot 0.998 = 1.9 - 0.0998 = 1.8002
$$

Result: $\theta_2 = (2.8001, 1.8002)$

---

**Iteration 3 ($t = 3$)**:

Step 1: Gradient at $\theta_2 = (2.8001, 1.8002)$
$$
g_3 = (5.6002, 7.2008)
$$

Step 2: Update first moment
$$
m_3 = \beta_1 m_2 + (1 - \beta_1) g_3 = 0.9(1.12, 1.48) + 0.1(5.6002, 7.2008)
$$
$$
m_3 = (1.008 + 0.5600, 1.332 + 0.7201) = (1.5680, 2.0521)
$$

Step 3: Update second moment
$$
g_3^2 = (31.362, 51.851)
$$
$$
v_3 = 0.999(0.069604, 0.121696) + 0.001(31.362, 51.851)
$$
$$
v_3 = (0.069534 + 0.031362, 0.121574 + 0.051851) = (0.100896, 0.173425)
$$

Step 4: Bias correction
$$
1 - \beta_1^3 = 1 - 0.9^3 = 1 - 0.729 = 0.271
$$
$$
\hat{m}_3 = \frac{(1.5680, 2.0521)}{0.271} = (5.786, 7.572)
$$

$$
1 - \beta_2^3 = 1 - 0.999^3 = 1 - 0.997002999 = 0.002997
$$
$$
\hat{v}_3 = \frac{(0.100896, 0.173425)}{0.002997} = (33.660, 57.860)
$$

Step 5: Update parameters
$$
\frac{\hat{m}_{3,x}}{\sqrt{\hat{v}_{3,x}} + \epsilon} = \frac{5.786}{\sqrt{33.660} + 10^{-8}} = \frac{5.786}{5.802} \approx 0.997
$$
$$
x_3 = 2.8001 - 0.1 \cdot 0.997 = 2.8001 - 0.0997 = 2.7004
$$

$$
\frac{\hat{m}_{3,y}}{\sqrt{\hat{v}_{3,y}} + \epsilon} = \frac{7.572}{\sqrt{57.860} + 10^{-8}} = \frac{7.572}{7.607} \approx 0.995
$$
$$
y_3 = 1.8002 - 0.1 \cdot 0.995 = 1.8002 - 0.0995 = 1.7007
$$

| t | $x_t$ | $y_t$ | $f(x_t, y_t)$ |
|---|-------|-------|---------------|
| 0 | 3.0000 | 2.0000 | 17.0000 |
| 1 | 2.9000 | 1.9000 | 15.6300 |
| 2 | 2.8001 | 1.8002 | 14.2894 |
| 3 | 2.7004 | 1.7007 | 12.9789 |

The parameters steadily decrease toward $(0, 0)$ with both coordinates receiving approximately equal-sized updates despite $y$ having twice the curvature.

### Example 2: Adam vs. SGD vs. Momentum vs. RMSProp

**Problem**: Compare Adam, SGD (LR=0.1), Momentum ($\beta=0.9$, LR=0.1), and RMSProp ($\beta=0.9$, LR=0.5) on $f(x, y) = x^2 + 100y^2$ starting at $(5, 1)$. Show 5 iterations.

**Solution**:

Gradient: $\nabla f = (2x, 200y)$

**SGD** ($\alpha = 0.1$):
$x_{t+1} = x_t(1 - 0.2)$, $y_{t+1} = y_t(1 - 20) = -19y_t$
$x$ converges (0.8 factor), $y$ diverges ($-19$ factor)! Not usable.

**SGD** ($\alpha = 0.005$):
$x_{t+1} = 0.99x_t$, $y_{t+1} = y_t - 200(0.005)y_t = 0y_t = 0$
$y$ converges in 1 step! $x$ converges at $0.99^t$, very slow.

**Adam** ($\alpha = 0.1$):
$y$ has large gradients ($200 \times 1 = 200$ in first step), so $v_t$ for $y$ grows quickly. The effective step for $y$ is $\alpha \cdot \hat{m}_y / (\sqrt{\hat{v}_y} + \epsilon) \approx 0.1 \cdot 200 / 200 = 0.1$, which is bounded. Both $x$ and $y$ converge steadily.

| t | SGD (safe LR) | Adam |
|---|---------------|------|
| 0 | (5.00, 1.00) | (5.00, 1.00) |
| 1 | (4.95, 0.00) | (4.90, 0.90) |
| 2 | (4.90, 0.00) | (4.80, 0.80) |
| 3 | (4.85, 0.00) | (4.71, 0.71) |
| 4 | (4.80, 0.00) | (4.61, 0.61) |
| 5 | (4.76, 0.00) | (4.52, 0.52) |

Adam makes steady progress in both coordinates. SGD collapses $y$ in one step but barely moves $x$.

### Example 3: Bias Correction Effect (First Few Steps)

**Problem**: Consider optimizing $f(x) = x$ starting at $x = 10$ (gradient is always 1). Show the effect of bias correction for the first 5 steps. $\alpha = 1$, $\beta_1 = 0.9$.

**Solution**:

Without bias correction:
$m_1 = 0.9(0) + 0.1(1) = 0.1$, $x_1 = 10 - 1 \cdot 0.1 = 9.9$
$m_2 = 0.9(0.1) + 0.1(1) = 0.19$, $x_2 = 9.9 - 0.19 = 9.71$
$m_3 = 0.9(0.19) + 0.1(1) = 0.271$, $x_3 = 9.71 - 0.271 = 9.439$
$m_4 = 0.9(0.271) + 0.1(1) = 0.344$, $x_4 = 9.439 - 0.344 = 9.095$
$m_5 = 0.9(0.344) + 0.1(1) = 0.410$, $x_5 = 9.095 - 0.410 = 8.685$

With bias correction:
$\hat{m}_1 = 0.1/0.1 = 1$, $x_1 = 10 - 1 \cdot 1 = 9$
$\hat{m}_2 = 0.19/0.19 = 1$, $x_2 = 9 - 1 = 8$
$\hat{m}_3 = 0.271/0.271 = 1$, $x_3 = 8 - 1 = 7$
$\hat{m}_4 = 0.344/0.344 = 1$, $x_4 = 7 - 1 = 6$
$\hat{m}_5 = 0.410/0.410 = 1$, $x_5 = 6 - 1 = 5$

Without bias correction, the step sizes start too small (0.1 instead of 1) and gradually approach the correct value. With bias correction, every step correctly uses $\hat{m}_t = 1 = \mathbb{E}[g]$.

After many steps, $\beta_1^t \to 0$, so $1/(1-\beta_1^t) \to 1$ and bias correction becomes negligible. The correction matters most in the first ~$1/(1-\beta_1) = 10$ iterations.

### Example 4: Second Moment Adaptation with Changing Gradient Scale

**Problem**: $f(x)$ has gradient $g = 10$ for $t \leq 5$ and $g = 0.1$ for $t > 5$. Show how $v_t$ and the effective learning rate adapt. $\alpha = 0.1$, $\beta_2 = 0.9$ (using faster decay for illustration).

**Solution**:

$v_0 = 0$

Phase 1 ($g = 10$):
$v_1 = 0.9(0) + 0.1(100) = 10$, $\sqrt{v_1} \approx 3.16$, effective LR $\approx 0.1/3.16 \approx 0.032$
$v_2 = 0.9(10) + 0.1(100) = 9 + 10 = 19$, $\sqrt{v_2} \approx 4.36$, effective LR $\approx 0.023$
$v_3 = 0.9(19) + 0.1(100) = 17.1 + 10 = 27.1$, $\sqrt{v_3} \approx 5.21$, effective LR $\approx 0.019$
$v_4 = 0.9(27.1) + 0.1(100) = 24.39 + 10 = 34.39$, $\sqrt{v_4} \approx 5.86$, effective LR $\approx 0.017$
$v_5 = 0.9(34.39) + 0.1(100) = 30.95 + 10 = 40.95$, $\sqrt{v_5} \approx 6.40$, effective LR $\approx 0.016$

Phase 2 ($g = 0.1$):
$v_6 = 0.9(40.95) + 0.1(0.01) = 36.855 + 0.001 = 36.856$, $\sqrt{v_6} \approx 6.07$, effective LR $\approx 0.016$
$v_7 = 0.9(36.856) + 0.001 = 33.170$, $\sqrt{v_7} \approx 5.76$, effective LR $\approx 0.017$
$v_8 = 0.9(33.170) + 0.001 = 29.854$, $\sqrt{v_8} \approx 5.46$, effective LR $\approx 0.018$
$v_9 = 0.9(29.854) + 0.001 = 26.870$, $\sqrt{v_9} \approx 5.18$, effective LR $\approx 0.019$
$v_{10} = 0.9(26.870) + 0.001 = 24.184$, $\sqrt{v_{10}} \approx 4.92$, effective LR $\approx 0.020$

When gradients drop, $v_t$ decays gradually (with time constant $1/(1-\beta_2) = 10$ steps), and the effective learning rate slowly increases. This adaptation time constant is controlled by $\beta_2$.

## Visual Interpretation

Adam's update can be visualized as three components working together:

1. **Direction** ($\hat{m}_t / \sqrt{\hat{v}_t}$): The ratio of the smoothed gradient to the RMS gradient. This gives a direction that is approximately a normalized gradient, with larger steps in directions where the signal-to-noise ratio is high.

2. **Magnitude** ($\alpha$): The global learning rate scales the overall step size.

3. **Per-parameter scaling** ($1/\sqrt{\hat{v}_t}$): Parameters with large gradient variance get smaller steps.

On a contour plot, Adam's trajectory is smoother than SGD (due to momentum) and more isotropic than RMSProp (due to bias correction early on). The steps are approximately the same size in all directions, eliminating the zig-zag behavior of SGD in ill-conditioned problems.

## Common Mistakes

1. **Not using bias correction**: Some implementations omit bias correction, leading to very small initial steps. The standard Adam includes bias correction; check your framework.

2. **Using Adam with too large weight decay**: In many frameworks, weight decay is implemented as L2 regularization added to the loss, which interacts poorly with Adam's adaptive rates. Use AdamW for decoupled weight decay.

3. **Neglecting learning rate tuning**: Despite Adam's robust defaults, $\alpha$ still needs tuning for optimal performance. The default $\alpha = 0.001$ is a starting point, not an optimal value for all tasks.

4. **Confusing Adam with AdamW**: AdamW decouples weight decay from the adaptive updates. Using standard Adam with weight decay effectively applies different regularization strengths to different parameters.

5. **Using Adam when SGD might generalize better**: For computer vision tasks, SGD with momentum often achieves better test accuracy despite worse training loss. Adam's adaptive rates may find sharper minima.

6. **Setting $\beta_2$ too low**: Low $\beta_2$ makes the second moment estimate noisy, causing erratic learning rates. The default $\beta_2 = 0.999$ provides a long averaging window.

7. **Forgetting $\epsilon$ matters**: The default $\epsilon = 10^{-8}$ affects parameters with near-zero gradients. Larger $\epsilon$ (e.g., $10^{-6}$) can help with numerical stability in some cases but reduces adaptation.

8. **Applying Adam on extremely large models**: Storing $m$ and $v$ doubles memory. For models with billions of parameters, this is significant. Consider Adafactor or SM3 for memory-efficient alternatives.

## Interview Questions

### Beginner - 5

**Q1**: What does Adam stand for?
**A**: Adaptive Moment Estimation. It adaptively estimates the first moment (mean) and second moment (uncentered variance) of gradients.

**Q2**: What are the two main ideas Adam combines?
**A**: Momentum (first moment estimate) and RMSProp (second moment estimate for adaptive learning rates).

**Q3**: What are the default values for $\beta_1$, $\beta_2$, $\alpha$, and $\epsilon$?
**A**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\alpha = 0.001$, $\epsilon = 10^{-8}$.

**Q4**: Why does Adam include bias correction?
**A**: The first and second moment estimates ($m_t$, $v_t$) are initialized at zero, biasing them toward zero in early iterations. Bias correction scales them up to compensate.

**Q5**: How does Adam differ from RMSProp?
**A**: Adam adds momentum (first moment estimate) and bias correction to RMSProp. RMSProp only maintains the second moment estimate.

### Intermediate - 5

**Q1**: Explain the role of $\beta_1$ and $\beta_2$ in Adam.
**A**: $\beta_1$ controls the exponential decay rate for the first moment (momentum). Higher $\beta_1$ = smoother gradient direction. $\beta_2$ controls the decay rate for the second moment (RMSProp adaptation). Higher $\beta_2$ = longer memory of gradient magnitudes.

**Q2**: What happens if $\beta_1 = 0$ in Adam?
**A**: $m_t = (1-0)g_t = g_t$, so there is no momentum. Adam reduces to RMSProp (with bias correction for the second moment).

**Q3**: Why might Adam generalize worse than SGD for some tasks?
**A**: Adam's adaptive learning rates can lead to sharper minima (where the Hessian has large eigenvalues). Sharp minima generalize worse because small parameter changes cause large output changes. SGD's constant learning rate biases toward flatter minima.

**Q4**: How does the effective learning rate differ between parameters with large vs. small gradients in Adam?
**A**: Parameters with large gradients have large $v_t$, so their effective LR $\alpha/\sqrt{v_t}$ is small. Parameters with small gradients have small $v_t$, giving larger effective LR. This balances progress across parameters.

**Q5**: What is AdamW and why was it introduced?
**A**: AdamW decouples weight decay from the adaptive gradient updates. In standard Adam, L2 regularization interacts poorly with adaptive rates. AdamW applies weight decay directly to the parameters after the Adam update, improving generalization.

### Advanced - 3

**Q1**: Prove that the bias correction in Adam ensures $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$ under stationary gradient distributions.
**A**: $\mathbb{E}[m_t] = \mathbb{E}[(1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i] = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} \mathbb{E}[g_i]$. Under stationarity, $\mathbb{E}[g_i] = \mu$, so $\mathbb{E}[m_t] = (1-\beta_1)\mu \sum_{i=1}^t \beta_1^{t-i} = (1-\beta_1)\mu \cdot (1-\beta_1^t)/(1-\beta_1) = \mu(1-\beta_1^t)$. Since $\hat{m}_t = m_t/(1-\beta_1^t)$, $\mathbb{E}[\hat{m}_t] = \mu = \mathbb{E}[g_t]$. The bias is exactly corrected.

**Q2**: Derive the regret bound for Adam under convex, bounded gradient assumptions, and explain why the $\beta_2^{1/2}$ correction in some variants is needed for convergence.
**A**: The regret bound analysis uses the fact that $\sum \alpha_t \hat{m}_t / \sqrt{\hat{v}_t} \cdot g_t$ telescopes under convexity. The original Adam analysis had a flaw: the inverse of $\sqrt{v_t}$ can increase if $v_t$ decreases, potentially violating the decreasing step size assumption. AMSGrad fixes this by maintaining the maximum of past $v_t$ values, ensuring the step size monotonically decreases.

**Q3**: Explain the memory and computation trade-offs between Adam, SGD with momentum, and second-order methods for training large models.
**A**: SGD with momentum stores 1 extra vector (velocity): $O(2d)$ memory. Adam stores 2 extra vectors ($m$, $v$): $O(3d)$ memory. Second-order methods (K-FAC, Shampoo) store block-diagonal approximations: $O(kd)$ where $k$ depends on block size. For a 1B parameter model: SGD (8 GB for momentum), Adam (12 GB for $m$+$v$), K-FAC (potentially 50+ GB). Adam's cost is acceptable for most models but significant for the largest LLMs. LAMB and Adafactor reduce memory by factorizing the second moment.

## Practice Problems

### Easy - 5

**P1**: Compute $m_1$ given $m_0 = 0$, $g_1 = (3, 4)$, $\beta_1 = 0.9$.
**P2**: Compute $\hat{m}_1$ for the above. What is the bias correction factor?
**P3**: If $\beta_2 = 0.999$, what is $1 - \beta_2^{100}$ approximately?
**P4**: What are the two moment estimates Adam maintains?
**P5**: If $\alpha = 0.01$ and $\hat{m}_1/\sqrt{\hat{v}_1} = (2, 3)$, what is the parameter update?

### Medium - 5

**P1**: For $f(x, y) = x^2 + y^2$, perform 2 iterations of Adam starting at $(4, 3)$ with $\alpha = 0.2$, defaults for other hyperparameters.
**P2**: Show that as $t \to \infty$, the bias correction factors $1/(1-\beta_1^t)$ and $1/(1-\beta_2^t)$ approach 1.
**P3**: Prove that Adam with $\beta_1 = 0$ is equivalent to RMSProp with bias correction.
**P4**: Compare the effective step size of Adam and SGD for a parameter with consistent gradient $g = 100$ vs $g = 0.01$.
**P5**: Explain why Adam might diverge on some problems and how to fix it.

### Hard - 3

**P1**: Derive the convergence proof for Adam under convex, Lipschitz, and bounded gradient assumptions, identifying where the proof requires monotonic step sizes.
**P2**: Analyze the interaction between Adam's adaptive learning rates and gradient noise: derive the stationary distribution of iterates for a simple quadratic.
**P3**: Prove that AdamW with decoupled weight decay is equivalent to Adam with L2 regularization only when weight decay and learning rate satisfy a specific relationship.

## Solutions

### Easy - Solutions

**S1**: $m_1 = 0.9(0, 0) + 0.1(3, 4) = (0.3, 0.4)$.
**S2**: $\hat{m}_1 = (0.3, 0.4)/(1-0.9^1) = (0.3, 0.4)/0.1 = (3, 4)$. The correction factor is $1/(1-0.9) = 10$.
**S3**: $1 - 0.999^{100} \approx 1 - 0.9048 \approx 0.0952$. The bias correction factor is about 10.5.
**S4**: $m_t$ (first moment / momentum) and $v_t$ (second moment / uncentered variance of gradients).
**S5**: $\Delta \theta = -\alpha \cdot (\hat{m}_1/\sqrt{\hat{v}_1}) = -0.01 \cdot (2, 3) = (-0.02, -0.03)$.

### Medium - Solutions

**S1**: $g_1 = (8, 6)$. $m_1 = (0.8, 0.6)$. $v_1 = (0.064, 0.036)$. $\hat{m}_1 = (8, 6)$. $\hat{v}_1 = (64, 36)$. $\theta_1 = (4, 3) - 0.2(8/8, 6/6) = (3.8, 2.8)$.
$g_2 = (7.6, 5.6)$. $m_2 = 0.9(0.8, 0.6) + 0.1(7.6, 5.6) = (0.72+0.76, 0.54+0.56) = (1.48, 1.10)$. $v_2 = 0.999(0.064, 0.036) + 0.001(57.76, 31.36) = (0.064+0.058, 0.036+0.031) = (0.122, 0.067)$. $\hat{m}_2 = (1.48, 1.10)/0.19 = (7.789, 5.789)$. $\hat{v}_2 = (0.122, 0.067)/0.001999 = (61.0, 33.5)$. $\theta_2 = (3.8, 2.8) - 0.2(7.789/7.81, 5.789/5.79) = (3.8, 2.8) - 0.2(0.997, 1.0) = (3.6003, 2.600)$.

**S2**: As $t \to \infty$, $\beta_1^t \to 0$ (since $0 \leq \beta_1 < 1$), so $1 - \beta_1^t \to 1$ and $1/(1-\beta_1^t) \to 1$. Same for $\beta_2$.

**S3**: With $\beta_1 = 0$: $m_t = (1-0)g_t = g_t$, $\hat{m}_t = g_t/(1-0^t) = g_t$ (since $0^t = 0$ for $t \geq 1$). $v_t$ and $\hat{v}_t$ follow RMSProp. Update: $\theta_{t+1} = \theta_t - \alpha g_t/(\sqrt{\hat{v}_t} + \epsilon)$, which is RMSProp with bias-corrected second moment.

**S4**: For a parameter with consistent gradient $g$, $\hat{v}_t \to g^2$ (steady state). Effective step: $\alpha g / (|g| + \epsilon) \approx \alpha \cdot \text{sign}(g)$. The effective step magnitude is approximately $\alpha$ regardless of $g$, as long as $|g| \gg \epsilon$. Very different from SGD where step = $\alpha g$.

**S5**: Adam can diverge if $v_t$ decreases while $m_t$ remains large, leading to increasing effective step sizes. Fixes: increase $\beta_2$ (slower decay of $v$), decrease $\alpha$, use gradient clipping, or use AMSGrad which maintains the maximum of past $v_t$.

### Hard - Solutions

**S1**: The regret $R(T) = \sum f_t(\theta_t) - f_t(\theta^*)$ is bounded using convexity: $R(T) \leq \sum \langle g_t, \theta_t - \theta^* \rangle$. The Adam update gives $\theta_{t+1} - \theta^* = \theta_t - \theta^* - \alpha_t \hat{m}_t / \sqrt{\hat{v}_t}$. Taking norms and using convexity yields a telescoping sum. The key issue: if $\sqrt{\hat{v}_t} < \sqrt{\hat{v}_{t-1}}$, the step size increases, potentially breaking the proof. AMSGrad fixes this.

**S2**: For quadratic $f(x) = \frac{1}{2}ax^2$ with gradient noise $\sigma^2$, Adam's update is approximately $x_{t+1} = x_t(1 - \alpha a / \sqrt{v_t})$. The second moment $v_t$ converges in distribution to a random variable with mean $\mathbb{E}[g^2] = a^2 x^2 + \sigma^2$. The stationary distribution of $x_t$ has variance proportional to $\alpha^2 \sigma^2 / (a \sqrt{\mathbb{E}[g^2]})$.

**S3**: AdamW: $\theta_{t+1} = \theta_t - \alpha (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_t)$. Standard Adam with L2: $\theta_{t+1} = \theta_t - \alpha (\hat{m}_t + \lambda \theta_t) / (\sqrt{\hat{v}_t} + \epsilon)$ (where $\lambda \theta_t$ is part of the gradient). These are equivalent only when $\sqrt{\hat{v}_t}$ is constant (all parameters have same curvature). In general, AdamW applies uniform weight decay while standard Adam applies stronger decay to parameters with smaller gradients.

## Related Concepts

- **RMSProp**: The adaptive learning rate component that Adam extends.
- **SGD with Momentum**: The velocity accumulation component.
- **AdamW**: Decoupled weight decay for better regularization.
- **Nadam**: Adam with Nesterov momentum.
- **AMSGrad**: Convergent variant using maximum of past second moments.
- **Adafactor**: Memory-efficient Adam for large models.
- **LAMB**: Layer-wise adaptive moments for large-batch training.
- **AdaBound**: Dynamic bounds on learning rates to transition from Adam to SGD.

## Next Concepts

- **Learning Rate Scheduling**: Complementary strategies for adjusting $\alpha$ during training.
- **AdamW**: The standard modern variant with decoupled weight decay.
- **LAMB**: For large-batch distributed training of large models.

## Summary

Adam (Adaptive Moment Estimation) combines momentum and RMSProp with bias correction to provide a robust, adaptive optimization algorithm. It maintains exponentially decaying averages of past gradients ($m_t$, first moment) and past squared gradients ($v_t$, second moment), applies bias correction to account for zero initialization, and updates parameters using the ratio $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ scaled by the learning rate.

Adam's default hyperparameters ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) work well across a wide range of deep learning tasks, making it the default optimizer in most frameworks. The bias correction ensures effective steps from the very first iteration.

Adam is not universally optimal: SGD with momentum sometimes generalizes better for computer vision, and AdamW improves regularization. However, for most deep learning applications, Adam remains the go-to optimizer.

## Key Takeaways

- Adam = Momentum (first moment) + RMSProp (second moment) + bias correction.
- Update: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\theta_{t+1} = \theta_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.
- Bias correction: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$.
- Defaults: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.
- Per-parameter learning rates adapt to gradient magnitudes.
- Robust to sparse gradients, noisy objectives, and hyperparameter choices.
- Memory cost: 2x parameter count (stores $m$ and $v$).
- AdamW improves regularization by decoupling weight decay.
- May find sharper minima than SGD, sometimes hurting generalization.
- Most widely used optimizer in deep learning.
