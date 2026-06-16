# Concept: Momentum

## Concept ID

MATH-097

## Difficulty

Advanced

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Derive the momentum update rule from the physical analogy of a heavy ball rolling downhill
- Explain how momentum accumulates velocity to dampen oscillations and accelerate convergence
- Analyze the effect of the momentum coefficient $\beta$ on convergence speed and stability
- Implement momentum-based gradient descent for multivariate optimization
- Distinguish between standard momentum and Nesterov accelerated gradient (NAG)
- Determine appropriate momentum schedules for different optimization landscapes

## Prerequisites

- Gradient Descent: update rule, convergence analysis, learning rate selection
- SGD: mini-batch gradient estimation, stochastic noise
- Convex functions: strong convexity, condition number
- Basic physics: velocity, friction, inertia

## Definition

**Momentum** is an extension of gradient descent that accelerates convergence by accumulating a velocity vector in the direction of persistent gradient signals. The update rule introduces a velocity term $v_t$ that retains a fraction of the previous update:

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_{t+1}
$$

where:
- $\beta \in [0, 1)$ is the **momentum coefficient** (typically $\beta = 0.9$)
- $\alpha$ is the **learning rate**
- $v_t$ is the **velocity** (a running average of past gradients)

The velocity accumulates gradients over time. In directions where the gradient consistently points the same way (downhill), velocity builds up, producing larger steps. In directions where the gradient oscillates (e.g., across a narrow valley), the oscillating components cancel out, reducing step size in that direction and damping oscillations.

## Intuition

Imagine rolling a heavy ball down a hill. The ball does not stop at every change in slope. Instead, it accumulates momentum: it speeds up when the slope is consistently downhill, and it resists abrupt direction changes due to its inertia. If the ball enters a narrow valley with steep sides, it bounces less than a lighter object would, because the momentum from the downward direction carries it forward while the oscillating sideways components partially cancel.

Standard gradient descent is like a frictionless walker who takes a step based only on the current slope, stopping and reassessing at each step. This walker zig-zags down narrow valleys. Momentum adds mass to the walker, smoothing the trajectory and accelerating progress along consistent downhill directions.

The momentum coefficient $\beta$ controls the effective mass. A high $\beta$ (e.g., 0.99) gives a heavy ball that resists direction changes but may overshoot. A low $\beta$ (e.g., 0.5) gives a lighter ball that responds more quickly to gradient changes but provides less smoothing.

## Why This Concept Matters

Momentum addresses two fundamental problems with standard gradient descent:

1. **Slow convergence in narrow ravines**: When the loss landscape has a long, narrow valley (high condition number), gradient descent zig-zags across the valley walls. Momentum cancels the oscillating perpendicular components while accelerating along the valley floor.

2. **Escaping plateaus and small local minima**: The accumulated velocity can carry the optimizer through flat regions and out of shallow local minima, improving solution quality.

In practice, momentum is the default extension to SGD used in virtually all deep learning frameworks. Understanding momentum is essential before studying adaptive methods (RMSProp, Adam) that build upon this idea.

## Historical Background

The concept of momentum in optimization was introduced by Boris Polyak in 1964 in his work on the "heavy ball method" (Polyak, 1964). He showed that adding a momentum term accelerates convergence for convex quadratic functions, particularly when the condition number is high.

Independently, Yuri Nesterov developed the "accelerated gradient method" in 1983, which uses a clever look-ahead step (Nesterov Accelerated Gradient, NAG) that achieves the optimal $O(1/t^2)$ convergence rate for smooth convex functions. While related to momentum, NAG uses a different update order.

In deep learning, momentum was popularized by Rumelhart, Hinton, and Williams in their 1986 backpropagation paper. Sutskever et al. (2013) showed that Nesterov momentum provides benefits for deep neural network training, and it remains widely used today alongside Adam.

## Real World Examples

- **Robotics**: Control systems use momentum-like terms in PID controllers to smooth trajectories and reduce oscillation.
- **Finance**: Moving averages of stock prices (like the 50-day moving average) are analogous to momentum, smoothing out daily fluctuations.
- **Physics simulation**: Verlet integration uses velocity terms to simulate particle motion with inertia.
- **Navigation**: A GPS route planner that penalizes sharp turns is using a momentum-like smoothing.
- **Economics**: Adaptive expectations models use momentum-like accumulation of past price changes.

## AI/ML Relevance

**Convergence acceleration**: In deep learning, loss landscapes often have narrow valleys (ravines) where gradient descent oscillates. Momentum dramatically improves convergence in these cases.

**Typical settings**: 
- Standard momentum: $\beta = 0.9$ (default in PyTorch and TensorFlow)
- Sometimes $\beta$ is scheduled: start at 0.5 and increase to 0.99
- Learning rate is typically reduced when using momentum

**Relationship with adaptive methods**: Adaptive methods (RMSProp, Adam) use momentum-like averaging of gradients (first moment) and squared gradients (second moment). Understanding pure momentum is prerequisite to understanding Adam.

**Concrete example**: Training a deep convolutional network on ImageNet. Without momentum, training might require 90 epochs to reach target accuracy. With momentum ($\beta = 0.9$), the same accuracy can be reached in 60-70 epochs.

### Nesterov Accelerated Gradient (NAG)

NAG modifies the momentum update by computing the gradient at a look-ahead position:

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t - \alpha \beta v_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_{t+1}
$$

The gradient is evaluated at $\theta_t - \alpha \beta v_t$ (where momentum would carry us), not at the current $\theta_t$. This "peek ahead" corrects overshooting and provides faster convergence theoretically and empirically.

## Mathematical Explanation

### Convergence Analysis

For a $\mu$-strongly convex, $L$-smooth quadratic function, the momentum method achieves linear convergence with rate:

$$
\rho = \max(|1 - \alpha \lambda_{\min}|, |1 - \alpha \lambda_{\max}|)
$$

for gradient descent, versus momentum's rate determined by the roots of a characteristic equation. For an optimal choice of parameters, momentum accelerates the convergence especially when the condition number $\kappa = L/\mu$ is large.

The characteristic equation for momentum on a quadratic $f(x) = \frac{1}{2}x^T A x$ is:

$$
x_{t+1} = x_t - \alpha \nabla f(x_t) + \beta (x_t - x_{t-1})
$$

This gives the recurrence:

$$
x_{t+1} = (1 + \beta) x_t - \beta x_{t-1} - \alpha A x_t
$$

The convergence rate depends on the eigenvalues of $A$ and the parameters $\alpha, \beta$.

### Nesterov's Optimal Rate

For $L$-smooth convex functions, Nesterov's accelerated gradient achieves:

$$
f(x_t) - f(x^*) = O\left(\frac{1}{t^2}\right)
$$

compared to $O(1/t)$ for standard gradient descent. This is the optimal convergence rate for first-order methods on smooth convex functions.

## Formula(s)

**Standard Momentum (Heavy Ball)**:

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_{t+1}
$$

**Nesterov Accelerated Gradient**:

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t - \alpha \beta v_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_{t+1}
$$

**Equivalent Form (decoupled weight decay)**:

$$
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t) + \beta (\theta_t - \theta_{t-1})
$$

**Velocity as Exponential Moving Average**:

$$
v_t = \sum_{i=0}^{t} \beta^{t-i} \nabla L(\theta_i)
$$

## Properties

1. **Velocity accumulation**: Persistent gradient signals accumulate, while oscillating signals cancel.
2. **Phase lead**: Nesterov momentum provides a correction that reduces overshooting.
3. **Critical damping**: Optimal $\beta$ and $\alpha$ give convergence as fast as second-order methods for quadratics.
4. **Noise reduction**: The exponential averaging smooths stochastic gradient noise.
5. **Overshooting risk**: High momentum can overshoot the minimum, requiring careful tuning.
6. **Bias initialization**: $v_0 = 0$ means early iterations are biased toward zero velocity.
7. **Invariance to gradient scaling**: Momentum is not scale-invariant; parameter initialization matters.

## Step-by-Step Worked Examples

### Example 1: Momentum vs. GD for a Quadratic Function

**Problem**: Minimize $f(x, y) = x^2 + 100y^2$ (condition number $\kappa = 100$) using standard GD and momentum. Starting at $(x_0, y_0) = (5, 1)$, $\alpha = 0.01$, $\beta = 0.9$. Run 15 iterations.

**Solution**:

$\nabla f(x, y) = [2x, 200y]^T$

**Standard GD**:
$(x_{t+1}, y_{t+1}) = (x_t, y_t) - 0.01(2x_t, 200y_t) = (x_t(1 - 0.02), y_t(1 - 2)) = (0.98x_t, -y_t)$

The $y$ coordinate oscillates (sign flips each step) while damping slowly:

| t | $x_t$ | $y_t$ |
|---|-------|-------|
| 0 | 5.000 | 1.000 |
| 1 | 4.900 | -1.000 |
| 2 | 4.802 | 1.000 |
| 3 | 4.706 | -1.000 |
| 4 | 4.612 | 1.000 |
| 5 | 4.520 | -1.000 |

The $y$ component oscillates between $\pm 1$ indefinitely without converging to 0!

**Momentum**:
$v_{t+1} = \beta v_t + \nabla f(x_t, y_t)$
$(x_{t+1}, y_{t+1}) = (x_t, y_t) - \alpha v_{t+1}$

Initial: $v_0 = (0, 0)$

Iteration 1:
$v_1 = 0.9(0, 0) + (10, 200) = (10, 200)$
$(x_1, y_1) = (5, 1) - 0.01(10, 200) = (4.9, -1.0)$

Iteration 2:
$v_2 = 0.9(10, 200) + (9.8, -200) = (9 + 9.8, 180 - 200) = (18.8, -20)$
$(x_2, y_2) = (4.9, -1.0) - 0.01(18.8, -20) = (4.712, -0.8)$

Iteration 3:
$v_3 = 0.9(18.8, -20) + (9.424, -160) = (16.92 + 9.424, -18 - 160) = (26.344, -178)$
$(x_3, y_3) = (4.712, -0.8) - 0.01(26.344, -178) = (4.449, 0.98)$

| t | $x_t$ | $y_t$ |
|---|-------|-------|
| 0 | 5.000 | 1.000 |
| 1 | 4.900 | -1.000 |
| 2 | 4.712 | -0.800 |
| 3 | 4.449 | 0.980 |
| 4 | 4.128 | -0.557 |
| 5 | 3.769 | 0.898 |

Momentum's $y$ component shows decaying oscillation (amplitudes: 1.0, 0.8, 0.98, 0.557, 0.898...) while GD's $y$ stays at $\pm 1.0$ indefinitely. Momentum gradually damps the oscillation.

### Example 2: Momentum Effect on a Simple 1D Problem

**Problem**: Minimize $f(x) = x^2$ starting at $x_0 = 10$ with $\alpha = 0.1$. Compare GD ($\beta = 0$), moderate momentum ($\beta = 0.7$), and high momentum ($\beta = 0.95$).

**Solution**:

GD: $x_{t+1} = x_t - 0.1(2x_t) = 0.8x_t$

Momentum: $v_{t+1} = \beta v_t + 2x_t$, $x_{t+1} = x_t - 0.1v_{t+1}$

**$t = 0$**: $x_0 = 10$, $v_0 = 0$

**Iteration 1**:
- GD: $x_1 = 0.8(10) = 8.0$
- $\beta = 0.7$: $v_1 = 0.7(0) + 20 = 20$, $x_1 = 10 - 0.1(20) = 8.0$
- $\beta = 0.95$: $v_1 = 0.95(0) + 20 = 20$, $x_1 = 10 - 0.1(20) = 8.0$

All same at first step (velocity starts at 0).

**Iteration 2**:
$v_2 = \beta v_1 + 2x_1$
- GD: $x_2 = 0.8(8.0) = 6.4$
- $\beta = 0.7$: $v_2 = 0.7(20) + 2(8.0) = 14 + 16 = 30$, $x_2 = 8.0 - 0.1(30) = 5.0$
- $\beta = 0.95$: $v_2 = 0.95(20) + 2(8.0) = 19 + 16 = 35$, $x_2 = 8.0 - 0.1(35) = 4.5$

**Iteration 3**:
- GD: $x_3 = 0.8(6.4) = 5.12$
- $\beta = 0.7$: $v_3 = 0.7(30) + 2(5.0) = 21 + 10 = 31$, $x_3 = 5.0 - 0.1(31) = 1.9$
- $\beta = 0.95$: $v_3 = 0.95(35) + 2(4.5) = 33.25 + 9 = 42.25$, $x_3 = 4.5 - 0.1(42.25) = 0.275$

| t | GD ($\beta=0$) | $\beta=0.7$ | $\beta=0.95$ |
|---|----------------|-------------|--------------|
| 0 | 10.000 | 10.000 | 10.000 |
| 1 | 8.000 | 8.000 | 8.000 |
| 2 | 6.400 | 5.000 | 4.500 |
| 3 | 5.120 | 1.900 | 0.275 |
| 4 | 4.096 | 0.130 | -1.919 |
| 5 | 3.277 | -0.614 | -2.518 |

Higher momentum gives faster initial progress but overshoots. $\beta=0.95$ reaches near $x=0$ by iteration 3 but then overshoots to negative. The optimal momentum balances speed with overshooting.

### Example 3: Momentum for Ill-Conditioned Quadratic

**Problem**: Show that momentum dampens oscillations for $f(x, y) = 0.5x^2 + 50y^2$ (condition number $\kappa = 100$). Use $\alpha = 0.02$, $\beta = 0.8$, start at $(x, y) = (4, 2)$.

**Solution**:

Gradient: $\nabla f = (x, 100y)$

GD without momentum: $x_{t+1} = x_t(1 - 0.02)$, $y_{t+1} = y_t(1 - 0.02 \cdot 100) = y_t(1 - 2) = -y_t$. The $y$ component oscillates indefinitely between $\pm 2$.

Momentum update:
$v_{t+1} = \beta v_t + \nabla f(x_t, y_t)$
$(x_{t+1}, y_{t+1}) = (x_t, y_t) - \alpha v_{t+1}$

t=0: $v_0 = (0, 0)$, $(x, y) = (4, 2)$
t=1: $v_1 = (0.8(0)+4, 0.8(0)+200) = (4, 200)$, $(x_1, y_1) = (4, 2) - 0.02(4, 200) = (3.92, -2)$
t=2: $v_2 = 0.8(4, 200) + (3.92, -200) = (3.2+3.92, 160-200) = (7.12, -40)$, $(x_2, y_2) = (3.92, -2) - 0.02(7.12, -40) = (3.7776, -1.2)$
t=3: $v_3 = 0.8(7.12, -40) + (3.7776, -120) = (5.696+3.7776, -32-120) = (9.4736, -152)$, $(x_3, y_3) = (3.7776, -1.2) - 0.02(9.4736, -152) = (3.5881, 1.84)$

| t | $x_t$ | $y_t$ | $y$ amplitude change |
|---|-------|-------|---------------------|
| 0 | 4.000 | 2.000 | - |
| 1 | 3.920 | -2.000 | full |
| 2 | 3.778 | -1.200 | reduced |
| 3 | 3.588 | 1.840 | - |
| 4 | 3.359 | -0.584 | further reduced |

The $y$ oscillation amplitudes decay because momentum cancels the oscillating components. Without momentum, $y$ would remain at $\pm 2$ forever.

### Example 4: Nesterov vs Standard Momentum

**Problem**: Compare standard momentum and Nesterov AG for $f(x) = x^2$ starting at $x_0 = 10$, $\alpha = 0.1$, $\beta = 0.9$. Run 5 iterations.

**Solution**:

**Standard Momentum**:
$v_{t+1} = 0.9v_t + 2x_t$
$x_{t+1} = x_t - 0.1v_{t+1}$

t=0: $x=10$, $v=0$
t=1: $v_1 = 0 + 20 = 20$, $x_1 = 10 - 2 = 8$
t=2: $v_2 = 0.9(20) + 16 = 18 + 16 = 34$, $x_2 = 8 - 3.4 = 4.6$
t=3: $v_3 = 0.9(34) + 9.2 = 30.6 + 9.2 = 39.8$, $x_3 = 4.6 - 3.98 = 0.62$
t=4: $v_4 = 0.9(39.8) + 1.24 = 35.82 + 1.24 = 37.06$, $x_4 = 0.62 - 3.706 = -3.086$
t=5: $v_5 = 0.9(37.06) - 6.172 = 33.354 - 6.172 = 27.182$, $x_5 = -3.086 - 2.718 = -5.804$

Significant overshoot (to -5.804).

**Nesterov AG**:
$v_{t+1} = 0.9v_t + \nabla f(x_t - 0.1 \cdot 0.9 \cdot v_t) = 0.9v_t + 2(x_t - 0.09v_t)$
$x_{t+1} = x_t - 0.1v_{t+1}$

t=0: $x=10$, $v=0$
t=1: $v_1 = 0 + 2(10 - 0) = 20$, $x_1 = 10 - 2 = 8$
t=2: look-ahead = $8 - 0.09(20) = 6.2$, $v_2 = 0.9(20) + 2(6.2) = 18 + 12.4 = 30.4$, $x_2 = 8 - 3.04 = 4.96$
t=3: look-ahead = $4.96 - 0.09(30.4) = 4.96 - 2.736 = 2.224$, $v_3 = 0.9(30.4) + 2(2.224) = 27.36 + 4.448 = 31.808$, $x_3 = 4.96 - 3.1808 = 1.7792$
t=4: look-ahead = $1.7792 - 0.09(31.808) = 1.7792 - 2.8627 = -1.0835$, $v_4 = 0.9(31.808) + 2(-1.0835) = 28.6272 - 2.167 = 26.4602$, $x_4 = 1.7792 - 2.6460 = -0.8668$
t=5: look-ahead = $-0.8668 - 0.09(26.4602) = -0.8668 - 2.3814 = -3.2482$, $v_5 = 0.9(26.4602) + 2(-3.2482) = 23.8142 - 6.4964 = 17.3178$, $x_5 = -0.8668 - 1.7318 = -2.5986$

Nesterov overshoots less ($-2.60$ vs $-5.80$ at t=5). The look-ahead correction reduces overshooting while maintaining fast convergence.

## Visual Interpretation

Visualize a contour plot of $f(x, y) = x^2 + 100y^2$. The contours are extremely elongated ellipses. Standard gradient descent takes a zig-zag path: each step goes left (in $x$) but overshoots in $y$, forcing a correction next step that again overshoots.

Momentum's trajectory is smoother. The velocity vector points generally downhill. When the gradient in $y$ flips sign, the velocity only partially responds, so the $y$ oscillation damps out. The $x$ velocity accumulates, producing larger steps along the valley floor.

The effect is like comparing a drunk person staggering downhill (GD) versus a bicyclist coasting downhill (momentum). The bicyclist's momentum smooths out wobbles and maintains speed.

For Nesterov, imagine the bicyclist leaning forward before pedaling. This anticipatory correction prevents oversteering and keeps the trajectory closer to the optimal path.

## Common Mistakes

1. **Setting momentum too high ($\beta > 0.99$)**: Causes unstable oscillations and divergence. The optimizer may overshoot the minimum repeatedly without converging.

2. **Not reducing learning rate when adding momentum**: Momentum effectively increases the step size. Adding momentum to an existing GD setup without reducing $\alpha$ often causes divergence. A good rule: when adding $\beta = 0.9$, reduce $\alpha$ by a factor of $1/(1-\beta) = 10$.

3. **Confusing momentum with learning rate**: Momentum is not a replacement for learning rate tuning. Both need to be set appropriately, and their interaction is nonlinear.

4. **Applying momentum naively with weight decay**: In PyTorch, weight decay is applied differently with momentum. The decoupled weight decay formulation (AdamW) properly separates regularization from momentum.

5. **Using constant momentum schedule**: Sometimes starting with low momentum ($\beta = 0.5$) and increasing to $\beta = 0.99$ over time (momentum scheduling) works better than constant high momentum.

6. **Ignoring velocity initialization bias**: Since $v_0 = 0$, the early iterations have reduced effective step size. This bias diminishes over time but matters in the first few iterations.

7. **Applying Nesterov incorrectly**: Nesterov momentum requires computing the gradient at a look-ahead position. Implementing it naively without proper gradient computation order gives incorrect results.

## Interview Questions

### Beginner - 5

**Q1**: What problem does momentum solve in gradient descent?
**A**: Momentum solves two problems: (1) slow convergence in narrow valleys due to oscillations, and (2) slow progress on flat plateaus where gradients are small.

**Q2**: What does the momentum coefficient $\beta$ control?
**A**: $\beta$ controls how much of the previous velocity is retained. Higher $\beta$ means more smoothing and faster buildup in consistent directions, but more overshooting risk.

**Q3**: What is a typical value for $\beta$?
**A**: $\beta = 0.9$ is the most common default. Values range from 0.5 (low momentum) to 0.99 (high momentum).

**Q4**: How does momentum affect the learning rate?
**A**: Momentum effectively amplifies the step size. When adding momentum with $\beta = 0.9$, the learning rate typically needs to be reduced by a factor of about 10.

**Q5**: What happens at $\beta = 0$?
**A**: Momentum reduces to standard gradient descent: $v_{t+1} = \nabla L(\theta_t)$, $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$.

### Intermediate - 5

**Q1**: How does momentum help with ill-conditioned problems (high condition number)?
**A**: In directions with high curvature, gradients oscillate in sign. Momentum cancels these oscillations by averaging. In directions with low curvature, gradients are consistently pointing downhill, so momentum accumulates velocity for larger steps.

**Q2**: What is the difference between standard momentum and Nesterov accelerated gradient?
**A**: Standard momentum computes the gradient at the current position, then applies velocity. Nesterov first applies the velocity to get a look-ahead position, then computes the gradient at that position. This correction reduces overshooting.

**Q3**: How does the velocity vector behave as an exponential moving average of gradients?
**A**: $v_t = \sum_{i=0}^{t} \beta^{t-i} \nabla L(\theta_i)$. Recent gradients get weight close to 1, while gradients from $k$ steps ago get weight $\beta^k \approx e^{-k(1-\beta)}$, an exponential decay with time constant $1/(1-\beta)$.

**Q4**: Why might high momentum cause divergence?
**A**: High momentum (e.g., $\beta > 0.99$) means the velocity decays very slowly. If the gradient changes direction, the velocity may carry the optimizer past the minimum and up the opposite slope, potentially leading to growing oscillations.

**Q5**: How do you tune $\alpha$ and $\beta$ together?
**A**: A common grid search: try $\beta \in \{0.5, 0.9, 0.95, 0.99\}$ and $\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$. Scale $\alpha$ down as $\beta$ increases. The optimal $\alpha$ for $\beta=0.9$ is typically about $1/10$ of the optimal $\alpha$ for GD.

### Advanced - 3

**Q1**: Derive the convergence rate of the heavy-ball method for a strongly convex quadratic and compare with gradient descent.
**A**: For $f(x) = \frac{1}{2}x^T A x$ with eigenvalues $\lambda_1 \leq \ldots \leq \lambda_n$, the optimal parameters are $\alpha = \frac{4}{(\sqrt{\lambda_1} + \sqrt{\lambda_n})^2}$ and $\beta = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$ where $\kappa = \lambda_n/\lambda_1$. The convergence factor is $\rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$, compared to GD's $\rho_{\text{GD}} = \frac{\kappa - 1}{\kappa + 1}$. For $\kappa = 100$, $\rho_{\text{GD}} \approx 0.98$ while $\rho_{\text{momentum}} \approx 0.82$, giving dramatically faster convergence.

**Q2**: Explain why Nesterov's method achieves the optimal $O(1/t^2)$ rate for smooth convex optimization and how this relates to the oracle model.
**A**: Nesterov's method achieves the lower bound established for first-order oracle models, which states that no first-order method can achieve a better rate than $O(1/t^2)$ for smooth convex functions. The acceleration comes from the look-ahead step that makes the method a second-order discretization of a differential equation, effectively using gradient information from two consecutive iterations.

**Q3**: Describe the relationship between momentum and the Polyak-Lojasiewicz condition for linear convergence without strong convexity.
**A**: The PL condition $\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f(x^*))$ is sufficient for linear convergence of gradient descent but requires careful treatment with momentum. Under the PL condition, momentum can achieve linear convergence with a better rate than GD, though the analysis is more nuanced because momentum can overshoot. The key insight is that PL functions have unique minima but may not be convex, and momentum helps navigate the non-convex regions.

## Practice Problems

### Easy - 5

**P1**: If $v_t = 10$ and $\beta = 0.9$, what is the contribution of $v_t$ to $v_{t+1}$?
**P2**: For $f(x) = 3x^2$, starting at $x = 4$, $\alpha = 0.1$, $\beta = 0.9$, compute the first momentum update.
**P3**: What is the physical analogy for momentum in optimization?
**P4**: If $\beta = 0.95$, approximately how many steps does it take for a gradient signal to decay to half its original weight?
**P5**: What does Nesterov's look-ahead step compute?

### Medium - 5

**P1**: For $f(x) = 0.5x^2 + 5y^2$, compare GD and momentum ($\beta = 0.8$, $\alpha = 0.1$) starting at $(2, 2)$ for 5 iterations.
**P2**: Derive the effective step size of momentum in terms of $\alpha$ and $\beta$ for a constant gradient.
**P3**: Show that momentum with $\beta = 0$ is equivalent to standard gradient descent.
**P4**: For a constant gradient $g$, find the asymptotic velocity and effective step size.
**P5**: Explain why momentum can overshoot the minimum and how Nesterov's method reduces this.

### Hard - 3

**P1**: Prove that for a quadratic function, the heavy-ball method with optimal parameters converges with rate $\rho = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$.
**P2**: Derive the update equations for Nesterov's accelerated gradient and show its $O(1/t^2)$ convergence rate for smooth convex functions.
**P3**: Analyze the stability region of the heavy-ball method for quadratics: find the set of $(\alpha, \beta)$ that guarantee convergence.

## Solutions

### Easy - Solutions

**S1**: $0.9 \times 10 = 9$. The previous velocity contributes 9 to the new velocity.
**S2**: $f'(x) = 6x$. $v_1 = 0.9(0) + 6(4) = 24$. $x_1 = 4 - 0.1(24) = 4 - 2.4 = 1.6$.
**S3**: A heavy ball rolling downhill. The ball accumulates speed (velocity) and resists direction changes due to inertia.
**S4**: The weight decays as $\beta^k$. Half-life: $\beta^k = 0.5 \implies k = \ln(0.5)/\ln(0.95) \approx 13.5$ steps.
**S5**: Nesterov computes the gradient at $\theta_t - \alpha \beta v_t$, the position where momentum would carry the parameters, rather than at the current $\theta_t$.

### Medium - Solutions

**S1**: $\nabla f = (x, 10y)$.
GD: $x_{t+1} = 0.9x_t$, $y_{t+1} = y_t - 0.1(10y_t) = 0$, so GD converges in y in 1 step.

Momentum: More complex trajectory. The $y$ component will show damping as momentum cancels oscillations.
**S2**: For constant gradient $g$, $v_{t+1} = \beta v_t + g$, $v_\infty = g/(1-\beta)$. The effective step is $\alpha g/(1-\beta)$. Momentum effectively increases the learning rate by $1/(1-\beta)$.
**S3**: With $\beta = 0$: $v_{t+1} = 0 \cdot v_t + \nabla L(\theta_t) = \nabla L(\theta_t)$. Then $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$, which is standard GD.
**S4**: $v_\infty = g/(1-\beta)$. The effective step is $\alpha g/(1-\beta)$. For $\beta = 0.9$, this is $10\alpha g$---a 10x amplification.
**S5**: Momentum accumulates velocity, which can carry the optimizer past the minimum. Nesterov computes gradient at the look-ahead position, detecting the upcoming slope change earlier and correcting the step accordingly.

### Hard - Solutions

**S1**: For $f(x) = \frac{1}{2}x^T A x$, the momentum recurrence is $x_{t+1} = x_t - \alpha(A x_t) + \beta(x_t - x_{t-1})$. Taking the eigendecomposition, each mode $i$ evolves independently. The characteristic equation is $\lambda^2 - (1 + \beta - \alpha \lambda_i)\lambda + \beta = 0$. The spectral radius is minimized by choosing $\alpha$ and $\beta$ so both roots have equal magnitude, giving $\rho = (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$.

**S2**: Nesterov's method uses $y_t = x_t + \beta_t(x_t - x_{t-1})$, $x_{t+1} = y_t - \alpha \nabla f(y_t)$, with $\beta_t = (t-1)/(t+2)$. The proof uses an estimate sequence technique to establish $f(x_t) - f(x^*) \leq O(1/t^2)$.

**S3**: For quadratic $f$ with eigenvalues $\lambda_i$, the update matrix has eigenvalues satisfying $|\lambda| \leq 1$ when $\alpha \leq 2(1+\beta)/\lambda_{\max}$ and $\beta \leq 1$. The full stability region is $\{(\alpha, \beta): 0 \leq \beta < 1, 0 < \alpha < 2(1+\beta)/\lambda_{\max}\}$.

## Related Concepts

- **Gradient Descent**: The base algorithm that momentum extends.
- **Nesterov Accelerated Gradient**: A variant with look-ahead correction.
- **Adam**: Combines momentum (first moment) with adaptive learning rates.
- **RMSProp**: Adaptive per-parameter learning rates.
- **Polyak Heavy Ball**: The original momentum method.
- **Conjugate Gradient**: Another acceleration technique for quadratic optimization.
- **Learning Rate Scheduling**: Often paired with momentum.
- **Condition Number**: The key difficulty that momentum addresses.

## Next Concepts

- **RMSProp**: Adaptive per-parameter learning rate based on gradient magnitude.
- **Adam**: The culmination of momentum and adaptive learning rates.
- **Learning Rate Scheduling**: Systematic LR adjustment strategies.

## Summary

Momentum accelerates gradient descent by accumulating a velocity vector that smooths gradient updates and damps oscillations. The update rule $v_{t+1} = \beta v_t + \nabla L(\theta_t)$, $\theta_{t+1} = \theta_t - \alpha v_{t+1}$ introduces velocity as an exponential moving average of gradients.

The momentum coefficient $\beta \in [0, 1)$ controls the trade-off between smoothing and responsiveness. Higher $\beta$ provides more smoothing but risks overshooting. Standard momentum ($\beta = 0.9$) is the default in deep learning frameworks.

Nesterov Accelerated Gradient improves upon standard momentum by computing gradients at a look-ahead position, achieving the optimal $O(1/t^2)$ convergence rate for smooth convex functions. Momentum is the foundation upon which adaptive optimizers like Adam are built.

## Key Takeaways

- Momentum accumulates velocity to accelerate convergence and dampen oscillations.
- Velocity is an exponential moving average: $v_t = \sum \beta^{t-i} \nabla L(\theta_i)$.
- Default $\beta = 0.9$; higher values risk overshooting.
- Momentum effectively amplifies the learning rate by $1/(1-\beta)$.
- Critical for ill-conditioned problems with high condition numbers.
- Nesterov AG computes gradient at a look-ahead position to reduce overshooting.
- Nesterov achieves $O(1/t^2)$ convergence rate (optimal for smooth convex).
- Momentum should be paired with reduced learning rates.
- Momentum is the foundation for adaptive methods (Adam).
- Understanding momentum is essential for effective deep learning training.
