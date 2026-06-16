# Concept: Gradient Descent

## Concept ID

MATH-095

## Difficulty

Intermediate

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Derive the gradient descent update rule from the first-order Taylor approximation
- Implement gradient descent for univariate and multivariate functions
- Analyze the effect of learning rate on convergence and divergence
- Distinguish between full-batch, mini-batch, and stochastic gradient descent
- Apply convergence criteria based on gradient norm and function value change
- Visualize gradient descent trajectories on 2D loss contour plots

## Prerequisites

- Multivariate calculus: gradient vector, partial derivatives, directional derivatives
- Linear algebra: vector norms, dot products
- Convex functions: bowl-shaped functions, global minima
- Optimization: stationary points, necessary conditions for optimality

## Definition

**Gradient descent** is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. Starting from an initial point $x_0$, the algorithm repeatedly takes steps proportional to the negative gradient of the function at the current point:

$$
x_{t+1} = x_t - \alpha \nabla f(x_t)
$$

where $\alpha > 0$ is the **learning rate** (or step size), $\nabla f(x_t)$ is the gradient of $f$ evaluated at $x_t$, and $t$ indexes the iteration number.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(x_t) \in \mathbb{R}^n$ points in the direction of steepest ascent. By moving in the opposite direction, gradient descent reduces the function value at each step (for sufficiently small $\alpha$).

## Intuition

Imagine you are standing on a mountain ridge in thick fog and need to reach the valley floor. You cannot see the entire landscape, but you can feel the slope beneath your feet. Gradient descent tells you: take a step in the steepest downhill direction you can sense locally. Repeat until the ground feels flat.

The gradient is your local compass. It points uphill. By moving opposite to it, you descend. The learning rate determines how large a step you take. Step too small, and you take forever to reach the valley. Step too large, and you might overstep the valley and climb up the opposite wall—or even end up on a different mountain entirely.

For convex functions like a bowl, gradient descent will always find the bottom. For non-convex functions like a mountain range, it will find the nearest valley—which might be a local minimum rather than the global deepest point.

## Why This Concept Matters

Gradient descent is the single most important optimization algorithm in machine learning. Every neural network—from tiny models on edge devices to massive language models with billions of parameters—is trained using some variant of gradient descent. Its simplicity, scalability, and effectiveness have made it the default algorithm for virtually all differentiable optimization in deep learning.

Understanding gradient descent is essential for:
- Setting and adjusting learning rates
- Diagnosing convergence problems
- Choosing between optimization variants (SGD, Adam, etc.)
- Implementing training loops from scratch
- Debugging vanishing/exploding gradient issues
- Understanding why and when training fails

## Historical Background

The method of steepest descent was first analyzed by Augustin-Louis Cauchy in 1847. Cauchy proposed using the gradient (or its discrete analogue) to solve systems of equations arising from least squares problems. However, the method was limited by manual computation and saw only sporadic use for over a century.

The modern revival began in the 1940s and 1950s with the development of electronic computers. Early work by Haskell Curry and George Forsythe established convergence properties. The 1960s saw formal analysis of convergence rates for convex functions, particularly by Boris Polyak.

The algorithm remained a niche tool in numerical optimization until the 1980s, when Rumelhart, Hinton, and Williams popularized backpropagation—an efficient way to compute gradients for neural networks. Backpropagation combined with gradient descent (often with momentum) became the standard training algorithm.

The 2010s revolutionized gradient descent with the introduction of mini-batch processing for large-scale machine learning. The 2012 AlexNet paper demonstrated that GPU-accelerated gradient descent could train deep networks for image classification, sparking the deep learning revolution.

## Real World Examples

- **Self-driving cars**: The perception system is trained by gradient descent to minimize prediction error between detected objects and ground truth.
- **Language models**: GPT and BERT are trained by gradient descent on massive text corpora to predict masked words or next tokens.
- **Recommender systems**: Netflix and YouTube use gradient descent to optimize collaborative filtering and neural recommendation models.
- **Drug discovery**: Molecular property prediction models are trained via gradient descent to predict binding affinity.
- **Robotics**: Reinforcement learning agents use gradient descent (via policy gradient methods) to optimize control policies.

## AI/ML Relevance

Gradient descent is the universal training algorithm for deep learning. The training loop for any neural network follows the same pattern:

1. Forward pass: compute predictions $\hat{y} = f_\theta(x)$
2. Compute loss: $L = \frac{1}{N}\sum_i \ell(y_i, \hat{y}_i)$
3. Backward pass: compute $\nabla_\theta L$ via backpropagation
4. Parameter update: $\theta \leftarrow \theta - \alpha \nabla_\theta L$

This process repeats for thousands or millions of iterations until convergence.

**Variants in practice:**
- **Stochastic Gradient Descent (SGD)**: Uses a single random sample per update.
- **Mini-batch SGD**: Uses a small batch (e.g., 32-512 samples) per update—the most common variant.
- **SGD with Momentum**: Adds velocity to smooth updates and accelerate convergence.
- **Adaptive methods (Adam, RMSProp)**: Per-parameter learning rate adaptation.

**Concrete application**: Training a linear regression model on housing data. The loss is $L(w) = \frac{1}{N}\sum_{i=1}^N (w^T x_i - y_i)^2$, and the gradient is $\nabla L(w) = \frac{2}{N}\sum_{i=1}^N (w^T x_i - y_i)x_i$. Gradient descent iteratively updates $w$ along this gradient until convergence to the minimum.

## Mathematical Explanation

### Derivation from Taylor Expansion

Consider the first-order Taylor expansion of $f$ at $x_t$:

$$
f(x_t + \Delta x) \approx f(x_t) + \nabla f(x_t)^T \Delta x
$$

We want to choose $\Delta x$ to minimize $f(x_t + \Delta x)$, so we want $\nabla f(x_t)^T \Delta x$ as negative as possible. For a fixed step norm $\|\Delta x\|$, the inner product $\nabla f(x_t)^T \Delta x$ is minimized when $\Delta x$ points opposite to $\nabla f(x_t)$:

$$
\Delta x = -\alpha \nabla f(x_t)
$$

where $\alpha$ controls the step size. This gives the gradient descent update.

### Line Search

Instead of a fixed learning rate, line search selects $\alpha_t$ to approximately minimize $f(x_t - \alpha \nabla f(x_t))$. The **Armijo condition** (backtracking line search) shrinks $\alpha$ until sufficient decrease is achieved:

$$
f(x_t - \alpha \nabla f(x_t)) \leq f(x_t) - c\alpha \|\nabla f(x_t)\|^2
$$

for some constant $c \in (0, 1)$ (typically $c = 10^{-4}$).

### Convergence Analysis

For an $L$-smooth function (gradient is $L$-Lipschitz continuous: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$), gradient descent with $\alpha \leq 1/L$ guarantees:

$$
f(x_{t+1}) \leq f(x_t) - \frac{\alpha}{2}\|\nabla f(x_t)\|^2
$$

If $f$ is also $\mu$-strongly convex, the convergence is linear:

$$
f(x_t) - f(x^*) \leq \left(1 - \frac{\mu}{L}\right)^t (f(x_0) - f(x^*))
$$

For merely convex functions, the rate is sublinear:

$$
f(x_t) - f(x^*) = O\left(\frac{1}{t}\right)
$$

## Formula(s)

**Basic Update**:

$$
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
$$

**Batch GD Loss**:

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(y_i, f_\theta(x_i))
$$

**Gradient of Batch Loss**:

$$
\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla \ell(y_i, f_\theta(x_i))
$$

**Descent Direction Proof**:

For sufficiently small $\alpha$, $f(x_{t+1}) < f(x_t)$ because:

$$
f(x_{t+1}) - f(x_t) \approx -\alpha \|\nabla f(x_t)\|^2 \leq 0
$$

**Armijo Backtracking Line Search**:

Find $\alpha$ such that $f(x_t - \alpha \nabla f(x_t)) \leq f(x_t) - c\alpha \|\nabla f(x_t)\|^2$

## Properties

1. **Descent property**: For sufficiently small $\alpha > 0$, $f(x_{t+1}) < f(x_t)$ unless $\nabla f(x_t) = 0$.
2. **Linear convergence for strongly convex functions**: The error decreases geometrically with factor $(1 - \mu/L)$.
3. **Sublinear convergence for convex functions**: The error decreases as $O(1/t)$.
4. **Sensitivity to conditioning**: Convergence slows with increasing condition number $\kappa = L/\mu$.
5. **Stationary point convergence**: Under standard assumptions, $\lim_{t \to \infty} \|\nabla f(x_t)\| = 0$.
6. **Invariance to rotation**: Gradient descent is not rotation-invariant; preconditioning can dramatically improve convergence.
7. **Learning rate threshold**: Divergence occurs if $\alpha > 2/L$ for $L$-smooth functions.

## Step-by-Step Worked Examples

### Example 1: Minimizing $f(x) = x^2$ (Univariate)

**Problem**: Minimize $f(x) = x^2$ using gradient descent with $\alpha = 0.5$, starting at $x_0 = 4$. Perform 10 iterations.

**Solution**:

$f(x) = x^2$, $f'(x) = 2x$

Update rule: $x_{t+1} = x_t - \alpha f'(x_t) = x_t - 0.5 \cdot 2x_t = x_t - x_t = 0$

Wait—with $\alpha = 0.5$, the update is $x_{t+1} = x_t - 0.5(2x_t) = x_t - x_t = 0$. This converges in one step! Let's use a different learning rate to show the iterative nature.

Let's use $\alpha = 0.2$:

| $t$ | $x_t$ | $f'(x_t) = 2x_t$ | $\Delta x = -\alpha f'(x_t)$ | $x_{t+1}$ | $f(x_{t+1})$ |
|-----|-------|------------------|------------------------------|-----------|--------------|
| 0   | 4.00  | 8.00             | $-1.60$                      | 2.40      | 5.7600       |
| 1   | 2.40  | 4.80             | $-0.96$                      | 1.44      | 2.0736       |
| 2   | 1.44  | 2.88             | $-0.576$                     | 0.864     | 0.7465       |
| 3   | 0.864 | 1.728            | $-0.3456$                    | 0.5184    | 0.2687       |
| 4   | 0.5184| 1.0368           | $-0.2074$                    | 0.3110    | 0.0967       |
| 5   | 0.3110| 0.6220           | $-0.1244$                    | 0.1866    | 0.0348       |
| 6   | 0.1866| 0.3732           | $-0.0746$                    | 0.1120    | 0.0125       |
| 7   | 0.1120| 0.2240           | $-0.0448$                    | 0.0672    | 0.0045       |
| 8   | 0.0672| 0.1344           | $-0.0269$                    | 0.0403    | 0.0016       |
| 9   | 0.0403| 0.0806           | $-0.0161$                    | 0.0242    | 0.0006       |
| 10  | 0.0242| 0.0484           | $-0.0097$                    | 0.0145    | 0.0002       |

The function value decreases from 16 to 0.0002 in 10 iterations, approaching the minimum at $x = 0$.

### Example 2: The Effect of Learning Rate

**Problem**: Apply gradient descent to $f(x) = x^2$ with different learning rates: $\alpha = 0.05$ (too small), $\alpha = 0.9$ (large but stable), and $\alpha = 1.1$ (divergent). Start at $x_0 = 5$ for 5 iterations.

**Solution**:

$f'(x) = 2x$, update: $x_{t+1} = x_t - \alpha(2x_t) = x_t(1 - 2\alpha)$

**Case 1: $\alpha = 0.05$** (small, slow convergence)

$x_1 = 5(1 - 0.1) = 4.5$
$x_2 = 4.5(0.9) = 4.05$
$x_3 = 4.05(0.9) = 3.645$
$x_4 = 3.645(0.9) = 3.2805$
$x_5 = 3.2805(0.9) = 2.9525$

After 5 steps, still far from 0. Slow decay.

**Case 2: $\alpha = 0.9$** (aggressive but stable)

$x_1 = 5(1 - 1.8) = 5(-0.8) = -4$
$x_2 = -4(1 - 1.8) = -4(-0.8) = 3.2$
$x_3 = 3.2(-0.8) = -2.56$
$x_4 = -2.56(-0.8) = 2.048$
$x_5 = 2.048(-0.8) = -1.6384$

The iterates oscillate around 0 but converge (since $|1 - 2\alpha| = |1 - 1.8| = 0.8 < 1$). This is oscillatory convergence.

**Case 3: $\alpha = 1.1$** (divergent)

$x_1 = 5(1 - 2.2) = 5(-1.2) = -6$
$x_2 = -6(-1.2) = 7.2$
$x_3 = 7.2(-1.2) = -8.64$
$x_4 = -8.64(-1.2) = 10.368$
$x_5 = 10.368(-1.2) = -12.4416$

The iterates grow in magnitude, diverging to infinity. The condition for convergence is $|1 - 2\alpha| < 1$, which gives $\alpha < 1$ for this function.

### Example 3: Multivariate Gradient Descent

**Problem**: Minimize $f(x, y) = x^2 + 2y^2$ using gradient descent with $\alpha = 0.25$, starting at $(x_0, y_0) = (3, 2)$. Perform 5 iterations.

**Solution**:

$\nabla f(x, y) = [2x, 4y]^T$

Update: $(x_{t+1}, y_{t+1}) = (x_t, y_t) - 0.25(2x_t, 4y_t) = (x_t - 0.5x_t, y_t - y_t)$

Wait: $y_t - 0.25(4y_t) = y_t - y_t = 0$. So $y$ converges immediately.

Let's trace:
- $t = 0$: $(3, 2)$, $\nabla f = (6, 8)$, update = $(-1.5, -2.0)$, new = $(1.5, 0)$
- $t = 1$: $(1.5, 0)$, $\nabla f = (3, 0)$, update = $(-0.75, 0)$, new = $(0.75, 0)$
- $t = 2$: $(0.75, 0)$, $\nabla f = (1.5, 0)$, update = $(-0.375, 0)$, new = $(0.375, 0)$
- $t = 3$: $(0.375, 0)$, $\nabla f = (0.75, 0)$, update = $(-0.1875, 0)$, new = $(0.1875, 0)$
- $t = 4$: $(0.1875, 0)$, $\nabla f = (0.375, 0)$, update = $(-0.09375, 0)$, new = $(0.09375, 0)$

The $y$ coordinate converges in one step because the learning rate $\alpha = 0.25$ equals $1/4$, which is the reciprocal of the Hessian eigenvalue for $y$ ($\partial^2 f/\partial y^2 = 4$). The $x$ coordinate decays geometrically with factor $0.5$ per iteration.

### Example 4: Gradient Descent with Backtracking Line Search

**Problem**: Apply gradient descent with backtracking line search to $f(x) = x^4 - 8x^2 + 16$, starting at $x_0 = 3$. Use initial $\alpha = 1$, Armijo constant $c = 0.1$, shrinkage factor $\beta = 0.5$.

**Solution**:

$f(x) = (x^2 - 4)^2$, $f'(x) = 4x^3 - 16x = 4x(x^2 - 4)$

**Iteration 1**: $x_0 = 3$, $f(3) = 25$, $f'(3) = 4(3)(9-4) = 60$

Try $\alpha = 1$:
$x_{\text{trial}} = 3 - 1(60) = -57$, $f(-57) = (3249 - 4)^2 = 3245^2 = 10,530,025$

Armijo condition: $f(-57) \leq f(3) - 0.1(1)(60)^2 = 25 - 360 = -335$? No, $-335 < 10,530,025$ is true but we need $f(-57) \leq -335$. Since $10,530,025 \not\leq -335$, reject.

Try $\alpha = 0.5$:
$x_{\text{trial}} = 3 - 0.5(60) = -27$, $f(-27) = (729 - 4)^2 = 725^2 = 525,625$

Condition: $f(-27) \leq -335$? No.

Try $\alpha = 0.25$:
$x_{\text{trial}} = 3 - 0.25(60) = -12$, $f(-12) = (144 - 4)^2 = 140^2 = 19,600$

Condition: $19,600 \leq -335$? No.

Try $\alpha = 0.125$:
$x_{\text{trial}} = 3 - 0.125(60) = -4.5$, $f(-4.5) = (20.25 - 4)^2 = 16.25^2 \approx 264.06$

Condition: $264.06 \leq -335$? No.

Try $\alpha = 0.0625$:
$x_{\text{trial}} = 3 - 0.0625(60) = -0.75$, $f(-0.75) = (0.5625 - 4)^2 = (-3.4375)^2 \approx 11.82$

Condition: $11.82 \leq -335$? No.

Try $\alpha = 0.03125$:
$x_{\text{trial}} = 3 - 0.03125(60) = 1.125$, $f(1.125) = (1.2656 - 4)^2 = (-2.7344)^2 \approx 7.48$

Condition: $7.48 \leq 25 - 0.1(0.03125)(3600) = 25 - 11.25 = 13.75$? Yes! $7.48 \leq 13.75$.

Accept $\alpha = 0.03125$. $x_1 = 1.125$, $f = 7.48$.

**Iteration 2**: $x_1 = 1.125$, $f'(1.125) = 4(1.125)(1.2656 - 4) = 4.5(-2.7344) \approx -12.30$

Try $\alpha = 0.03125$ (previous accepted value):
$x_{\text{trial}} = 1.125 - 0.03125(-12.30) = 1.125 + 0.3844 = 1.5094$, $f(1.5094) = (2.2783 - 4)^2 = 2.96$

Condition: $2.96 \leq 7.48 - 0.1(0.03125)(151.29) = 7.48 - 0.473 = 7.007$? Yes, $2.96 \leq 7.007$.

Accept. The algorithm converges toward $x = \pm 2$, where $f = 0$.

## Visual Interpretation

Imagine the contour plot of $f(x, y) = x^2 + 2y^2$. The contours are ellipses centered at $(0, 0)$. The $x$-axis contours are spaced closely (steep direction), while the $y$-axis contours are spaced more widely because the curvature in $y$ is higher ($\partial^2 f/\partial y^2 = 4$ vs. $\partial^2 f/\partial x^2 = 2$).

Gradient descent takes steps perpendicular to the contour lines, moving toward the center. With a fixed learning rate, the path shows the characteristic zig-zag pattern when the condition number $\kappa > 1$. In this case, $\kappa = 4/2 = 2$, so there is mild zig-zagging.

For a function with a very skewed valley like $f(x, y) = x^2 + 100y^2$, the condition number is $\kappa = 100$, and gradient descent would bounce back and forth along the narrow valley, taking many iterations to converge.

The learning rate determines the step length. Visualize:
- $\alpha$ too small: tiny steps, many iterations needed
- $\alpha$ just right: steady progress toward minimum
- $\alpha$ too large: overshooting, oscillation
- $\alpha$ excessive: divergence, iterates fly away from the minimum

## Common Mistakes

1. **Setting the learning rate too high**: Causes divergence. The iterates oscillate with increasing amplitude instead of converging. Always check loss curves for signs of divergence (loss spiking upward).

2. **Setting the learning rate too low**: The algorithm converges extremely slowly, potentially stopping before reaching the minimum due to limited iterations or numerical tolerance.

3. **Assuming gradient descent always finds the global minimum**: For non-convex functions, gradient descent converges to a local minimum or saddle point. Only for convex functions is the global minimum guaranteed.

4. **Failing to normalize features**: Features on different scales create ill-conditioned optimization landscapes. Gradient descent converges much faster when all features are on a similar scale (mean 0, variance 1).

5. **Using a single global learning rate for all parameters**: Different parameters may benefit from different learning rates. Adaptive methods (Adam, RMSProp) address this automatically.

6. **Not monitoring convergence**: Running a fixed number of iterations without checking whether the gradient norm or loss has stabilized wastes computation. Use early stopping based on convergence criteria.

7. **Confusing gradient descent with backpropagation**: Backpropagation computes gradients efficiently via the chain rule; gradient descent uses those gradients to update parameters. They are complementary but distinct algorithms.

8. **Ignoring the stochastic case**: In practice, mini-batch SGD is used instead of full-batch GD for large datasets. The convergence behavior differs fundamentally due to gradient noise.

## Interview Questions

### Beginner - 5

**Q1**: What does the learning rate control in gradient descent?  
**A**: The learning rate controls the step size taken in the direction opposite to the gradient. It determines how much the parameters change per iteration.

**Q2**: Why does gradient descent move opposite to the gradient?  
**A**: The gradient points in the direction of steepest ascent. To minimize the function, we move in the opposite direction—steepest descent.

**Q3**: What happens if the learning rate is too large?  
**A**: The algorithm may overshoot the minimum, oscillate, or diverge (loss increases to infinity).

**Q4**: What is a stationary point?  
**A**: A point where $\nabla f(x) = 0$. Gradient descent stops at stationary points. These can be minima, maxima, or saddle points.

**Q5**: Is gradient descent guaranteed to reduce the loss every iteration?  
**A**: For full-batch GD with a sufficiently small learning rate on a smooth function, yes. But for mini-batch SGD, the loss can increase in individual iterations due to gradient noise.

### Intermediate - 5

**Q1**: How does the condition number of the Hessian affect gradient descent?  
**A**: A high condition number (ratio of largest to smallest Hessian eigenvalue) causes gradient descent to zig-zag along narrow valleys, converging slowly. The convergence rate depends on $\mu/L$, the ratio of strong convexity to smoothness.

**Q2**: What is the difference between batch, mini-batch, and stochastic gradient descent?  
**A**: Batch GD uses all data per update. SGD uses one sample per update (very noisy). Mini-batch GD uses a subset (e.g., 32–512), balancing gradient accuracy and computational efficiency. Mini-batch is the standard in practice.

**Q3**: Explain backtracking line search and when it's useful.  
**A**: Backtracking line search adaptively selects the learning rate by starting with a large value and shrinking it until the Armijo condition is satisfied. It's useful when the optimal fixed learning rate is unknown or varies during optimization.

**Q4**: What is the convergence rate of gradient descent for convex functions?  
**A**: For $L$-smooth convex functions, $f(x_t) - f(x^*) = O(1/t)$. For $\mu$-strongly convex functions, the rate is linear: $f(x_t) - f(x^*) = O((1-\mu/L)^t)$.

**Q5**: Why is feature normalization important for gradient descent?  
**A**: Without normalization, features on different scales create elliptical contour lines with high curvature in some directions and low curvature in others. Gradient descent zig-zags, requiring many iterations. Normalization circularizes the contours.

### Advanced - 3

**Q1**: Derive the convergence rate of gradient descent for $L$-smooth convex functions using the Polyak-Lojasiewicz (PL) condition.  
**A**: The PL condition $\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f(x^*))$ is weaker than strong convexity but still yields linear convergence. For gradient descent with $\alpha \leq 1/L$: $f(x_{t+1}) - f(x^*) \leq (1 - \alpha\mu)(f(x_t) - f(x^*))$. The proof uses the descent lemma $f(x_{t+1}) \leq f(x_t) - \frac{\alpha}{2}\|\nabla f(x_t)\|^2$ combined with the PL condition.

**Q2**: Explain the implicit bias of gradient descent for overparameterized linear models.  
**A**: For linear regression with more parameters than data points ($d > n$), gradient descent converges to the minimum $\ell_2$ norm solution among all interpolating solutions. Specifically, if initialized at $0$, it converges to $\arg\min\{\|w\|_2 : Xw = y\}$. This implicit bias toward low-norm solutions explains generalization in overparameterized regimes.

**Q3**: How does gradient descent behave near saddle points in high dimensions?  
**A**: Gradient descent slows down near saddle points because the gradient is small. For strict saddle points (where the Hessian has at least one negative eigenvalue), gradient descent can escape given sufficiently small learning rates—though the escape time may be long. This is in contrast to Newton methods, which are attracted to saddle points.

## Practice Problems

### Easy - 5

**P1**: Perform 3 iterations of gradient descent on $f(x) = 2x^2$ starting at $x_0 = 3$ with $\alpha = 0.1$.

**P2**: For $f(x) = x^2 - 4x + 4$, find the minimum analytically and then verify that gradient descent with $\alpha = 0.3$ starting from $x_0 = 0$ converges to it.

**P3**: What is the gradient of $f(x, y) = 3x^2 + 4xy + y^2$ at $(1, 2)$?

**P4**: Starting from $x_0 = 5$, perform one iteration of gradient descent on $f(x) = e^x$ with $\alpha = 0.01$.

**P5**: If $f'(x) = 6x$ and $x_t = 2$, what is $x_{t+1}$ with $\alpha = 0.2$?

### Medium - 5

**P1**: For $f(x) = x^4 - 8x^2$, find the update equation for gradient descent and determine the range of $\alpha$ that ensures local convergence near $x = 0$.

**P2**: Implement gradient descent for $f(x, y) = 3x^2 - 2xy + y^2$ starting at $(2, 2)$ with $\alpha = 0.1$ for 5 iterations.

**P3**: Compare gradient descent with $\alpha = 1/L$ for $f(x) = \frac{1}{2}x^T Q x$ where $Q = \text{diag}(1, 100)$, starting from $(1, 1)$. Compute 3 iterations.

**P4**: Show that gradient descent on $f(x) = \|Ax - b\|_2^2$ converges to the least squares solution $(A^T A)^{-1} A^T b$ under appropriate conditions.

**P5**: Prove that $f(x_{t+1}) \leq f(x_t)$ for gradient descent with $\alpha \leq 1/L$ when $f$ is $L$-smooth.

### Hard - 3

**P1**: Prove that gradient descent on a $\mu$-strongly convex, $L$-smooth function achieves linear convergence: $\|x_{t+1} - x^*\|^2 \leq (1 - \alpha\mu)\|x_t - x^*\|^2$ for $\alpha \leq 1/L$.

**P2**: Design a learning rate schedule that achieves the optimal $O(1/t^2)$ convergence rate for convex functions (Polyak's heavy-ball method or Nesterov's accelerated gradient).

**P3**: Analyze the convergence of gradient descent with random initialization for overparameterized two-layer neural networks in the neural tangent kernel (NTK) regime.

## Solutions

### Easy - Solutions

**S1**: $f'(x) = 4x$. $x_1 = 3 - 0.1(12) = 1.8$, $x_2 = 1.8 - 0.1(7.2) = 1.08$, $x_3 = 1.08 - 0.1(4.32) = 0.648$.

**S2**: $f(x) = (x-2)^2$, minimum at $x = 2$. $f'(x) = 2(x-2)$. $x_1 = 0 - 0.3(2(-2)) = 0 + 1.2 = 1.2$. $x_2 = 1.2 - 0.3(2(-0.8)) = 1.2 + 0.48 = 1.68$. Converges to 2.

**S3**: $\partial f/\partial x = 6x + 4y = 6 + 8 = 14$. $\partial f/\partial y = 4x + 2y = 4 + 4 = 8$. Gradient $(14, 8)$.

**S4**: $f'(x) = e^x$, $f'(5) = e^5 \approx 148.41$. $x_1 = 5 - 0.01(148.41) = 5 - 1.4841 = 3.5159$.

**S5**: $x_{t+1} = 2 - 0.2(12) = 2 - 2.4 = -0.4$.

### Medium - Solutions

**S1**: Update: $x_{t+1} = x_t - \alpha(4x_t^3 - 16x_t) = x_t - 4\alpha x_t(x_t^2 - 4)$. Near $x = 0$, $f'(x) \approx -16x$ (linearizing), so $x_{t+1} \approx x_t + 16\alpha x_t = x_t(1 + 16\alpha)$. Convergence requires $|1 + 16\alpha| < 1$, which is impossible for $\alpha > 0$. So $x = 0$ is a local maximum, and GD diverges from it.

**S2**: $\nabla f = (6x - 2y, -2x + 2y)^T$.
- Iter 1: $(2, 2)$, $\nabla f = (8, 0)$, update: $(2, 2) - 0.1(8, 0) = (1.2, 2)$
- Iter 2: $(1.2, 2)$, $\nabla f = (3.2, 1.6)$, update: $(1.2, 2) - 0.1(3.2, 1.6) = (0.88, 1.84)$
- Iter 3: $(0.88, 1.84)$, $\nabla f = (1.6, 1.92)$, update: $(0.88, 1.84) - 0.1(1.6, 1.92) = (0.72, 1.648)$
- Iter 4: $(0.72, 1.648)$, $\nabla f = (1.024, 1.856)$, update: $(0.72, 1.648) - 0.1(1.024, 1.856) = (0.6176, 1.4624)$
- Iter 5: $(0.6176, 1.4624)$, $\nabla f = (0.8128, 1.6896)$, update: $(0.6176, 1.4624) - 0.1(0.8128, 1.6896) = (0.5363, 1.2934)$

**S3**: $f(x) = 0.5(1^2 + 100 \cdot 1^2) = 0.5(101) = 50.5$.
- $\nabla f = Qx = (1, 100)$
- Iter 1: $(1, 1) - 0.01(1, 100) = (0.99, 0)$, loss = $0.5(0.99^2 + 100(0)^2) = 0.4901$
- Iter 2: $(0.99, 0) - 0.01(0.99, 0) = (0.9801, 0)$
- Iter 3: $(0.9801, 0)$
The $y$ component reaches 0 in one iteration (optimal step for that direction), while $x$ decays slowly.

**S4**: $f(x) = \|Ax - b\|^2 = (Ax-b)^T(Ax-b)$. $\nabla f = 2A^T(Ax-b)$. GD: $x_{t+1} = x_t - 2\alpha A^T(Ax_t - b)$. For $\alpha \leq 1/\lambda_{\max}(A^T A)$, GD converges to the least squares solution.

**S5**: For $L$-smooth functions, $f(y) \leq f(x) + \nabla f(x)^T(y-x) + \frac{L}{2}\|y-x\|^2$. Let $y = x_{t+1} = x_t - \alpha \nabla f(x_t)$. Then $f(x_{t+1}) \leq f(x_t) - \alpha\|\nabla f(x_t)\|^2 + \frac{L\alpha^2}{2}\|\nabla f(x_t)\|^2 = f(x_t) - \alpha(1 - \frac{L\alpha}{2})\|\nabla f(x_t)\|^2$. For $\alpha \leq 1/L$, we have $1 - L\alpha/2 \geq 1/2$, so $f(x_{t+1}) \leq f(x_t) - \frac{\alpha}{2}\|\nabla f(x_t)\|^2 \leq f(x_t)$.

### Hard - Solutions

**S1**: From $L$-smoothness: $f(x_{t+1}) \leq f(x_t) - \frac{\alpha}{2}\|\nabla f(x_t)\|^2$ (as above). From strong convexity: $\|\nabla f(x)\|^2 \geq 2\mu(f(x) - f(x^*))$ and $f(x) - f(x^*) \leq \frac{1}{2\mu}\|\nabla f(x)\|^2$. Combining: $f(x_{t+1}) - f(x^*) \leq (f(x_t) - f(x^*)) - \frac{\alpha}{2} \cdot 2\mu(f(x_t) - f(x^*)) = (1 - \alpha\mu)(f(x_t) - f(x^*))$. This yields linear convergence.

**S2**: Nesterov's accelerated gradient achieves $O(1/t^2)$ using momentum-like updates: $y_{t+1} = x_t - \alpha \nabla f(x_t)$, $x_{t+1} = y_{t+1} + \beta_t(y_{t+1} - y_t)$ with $\beta_t = (t-1)/(t+2)$. This is optimal for first-order convex optimization.

**S3**: In the NTK regime, the network function evolves linearly with respect to parameters. Gradient descent achieves zero training loss and converges to the minimum norm solution in the reproducing kernel Hilbert space defined by the NTK.

## Related Concepts

- **Stochastic Gradient Descent (SGD)**: Mini-batch variant essential for large-scale learning.
- **Momentum**: Accelerates gradient descent by accumulating velocity.
- **Newton's Method**: Second-order optimization using Hessian information.
- **Conjugate Gradient**: Direction-based method for quadratic optimization.
- **Backpropagation**: Efficient gradient computation via the chain rule.
- **Learning Rate Schedules**: Systematic adjustment of learning rate during training.
- **Line Search**: Adaptive step size selection.
- **Preconditioning**: Scaling the gradient to improve conditioning.

## Next Concepts

- **Stochastic Gradient Descent**: The workhorse algorithm for deep learning with mini-batch sampling.
- **Momentum**: Overcoming gradient descent's slow convergence in narrow valleys.
- **RMSProp and Adam**: Adaptive learning rate methods that automatically tune per-parameter step sizes.
- **Learning Rate Scheduling**: Systematic strategies for adjusting learning rates during training.

## Summary

Gradient descent is a first-order iterative optimization algorithm that minimizes differentiable functions by moving in the direction opposite to the gradient. The update rule $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$ is the foundation of virtually all neural network training.

The learning rate $\alpha$ controls step size and is the most critical hyperparameter. Too small leads to slow convergence; too large causes divergence. Convergence is linear for strongly convex functions ($O((1-\mu/L)^t)$) and sublinear for convex functions ($O(1/t)$).

In practice, mini-batch stochastic gradient descent replaces full-batch GD for computational efficiency. Understanding gradient descent is prerequisite to mastering all advanced optimizers (momentum, RMSProp, Adam) and diagnosing training dynamics.

## Key Takeaways

- Gradient descent updates parameters opposite the gradient direction.
- Learning rate $\alpha$ is the critical hyperparameter for stability and speed.
- Convergence is guaranteed for convex functions with appropriate step sizes.
- The condition number $\kappa = L/\mu$ determines convergence speed.
- Full-batch GD uses all data; mini-batch SGD uses subsets.
- Feature normalization is essential for efficient gradient descent.
- Gradient descent only converges to stationary points (minima, maxima, or saddles).
- For non-convex functions, global optimality is not guaranteed.
- Backpropagation computes gradients; gradient descent uses them.
- Monitoring gradient norm and loss curves is essential for diagnosing issues.
