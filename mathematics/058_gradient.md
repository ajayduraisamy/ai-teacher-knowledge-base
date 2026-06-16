# Concept: Gradient

## Concept ID

MATH-058

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the gradient $\nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n})$ as the vector of all first-order partial derivatives.
- Interpret the gradient geometrically: direction of steepest ascent, orthogonal to level sets.
- Apply gradient descent: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$ for optimisation.
- Compute gradients of scalar-valued functions (loss functions) with respect to parameters.
- Understand variants of gradient descent: SGD, momentum, Adam, RMSprop.
- Visualise optimisation landscapes and the role of the gradient in navigating them.

## Prerequisites

- Partial Derivative (MATH-056) — gradient components are partial derivatives.
- Derivative (MATH-055) — the gradient generalises the derivative to multivariable functions.
- Chain Rule (MATH-057) — gradients through composite functions use the chain rule.
- Vector (MATH-002) — the gradient is a vector; vector operations apply.
- Dot product (MATH-016) — directional derivative as $\nabla f \cdot \mathbf{v}$.

## Definition

The **gradient** of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of its first-order partial derivatives:

$$\nabla f(x_1, x_2, \dots, x_n) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right)$$

Alternative notations: $\text{grad } f$, $\frac{\partial f}{\partial \mathbf{x}}$, $\nabla_{\mathbf{x}} f$.

For $f(x, y)$ in $\mathbb{R}^2$:
$$\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$$

**Directional Derivative:** The derivative of $f$ at $\mathbf{x}$ in the direction of a unit vector $\mathbf{v}$ is:

$$D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{v}$$

This measures the rate of change of $f$ in the direction $\mathbf{v}$.

**Steepest Ascent:** The gradient points in the direction of the greatest rate of increase of $f$. The directional derivative is maximised when $\mathbf{v}$ points in the same direction as $\nabla f$.

**Orthogonality to Level Sets:** The gradient at a point is orthogonal (perpendicular) to the level set (contour line/surface) passing through that point.

## Intuition

Imagine you are standing on a hillside described by $f(x, y)$. The gradient at your position points directly uphill — the direction of steepest ascent. Its magnitude tells you how steep the hill is at that point. If you want to go downhill fastest, you walk in the opposite direction of the gradient (negative gradient).

The gradient is a vector that encodes two pieces of information for each coordinate direction:
- **Direction (sign):** Whether $f$ increases (+) or decreases (-) as you move in that coordinate direction.
- **Magnitude:** How sensitive $f$ is to changes in that coordinate (the slope in that direction).

When the gradient is zero ($\nabla f = \mathbf{0}$), you are at a critical point: a local minimum, maximum, or saddle point. The terrain is flat in all directions.

**The key insight for optimisation:** If we want to minimise a function $L(\theta)$, we should move in the direction opposite to $\nabla L(\theta)$, because that is the direction of steepest descent. Gradient descent iterates this process: compute the gradient, take a step downhill, repeat.

## Why This Concept Matters

The gradient is the fundamental tool for optimisation in high-dimensional spaces:

1. **Machine Learning Optimisation.** Nearly all neural network training uses gradient-based methods. The gradient tells us how to adjust each parameter to reduce the loss.

2. **Signal Processing.** Image gradients (Sobel operator, Canny edge detection) identify edges by computing $\nabla I$ for an image $I(x, y)$.

3. **Physics.** Force equals the negative gradient of potential energy: $\mathbf{F} = -\nabla U$. Electric field is the gradient of potential: $\mathbf{E} = -\nabla V$. Heat flows down temperature gradients.

4. **Computer Graphics.** Normal vectors for 3D surfaces are computed as gradients of implicit surface functions: $\mathbf{n} = \nabla F / \|\nabla F\|$ for $F(x, y, z) = 0$.

5. **Robotics.** Gradient-based motion planning uses potential fields where the gradient attracts the robot to the goal and repels it from obstacles.

## Historical Background

The concept of the gradient emerged in the 18th and 19th centuries as multivariable calculus developed. Leonhard Euler (1707-1783) and Joseph-Louis Lagrange (1736-1813) laid the groundwork for the calculus of variations, which implicitly used gradient-like concepts.

The term "gradient" was introduced in the late 19th century by James Clerk Maxwell (1831-1879) in his work on electromagnetism. Maxwell used the gradient operator $\nabla$ (nabla) in his equations of electromagnetism, where the electric field is the gradient of the electric potential.

The notation $\nabla f$ (nabla $f$) was introduced by William Rowan Hamilton (1805-1865), who originally used $\nabla$ as a differential operator. The symbol $\nabla$ is an inverted Greek capital delta ($\Delta$), suggesting "difference."

The application of gradients to optimisation — gradient descent — was proposed by Augustin-Louis Cauchy in 1847. Cauchy suggested using the gradient to find minima of functions, anticipating modern machine learning by over 150 years.

The modern explosion in gradient-based methods came with the development of backpropagation (Werbos, 1974; Rumelhart, Hinton, Williams, 1986) and stochastic gradient descent (Robbins-Monro, 1951). Today, gradient-based optimisation is the backbone of deep learning.

## Real World Examples

**Example 1: Finding the Minimum of a Surface.** The function $f(x, y) = x^2 + 2y^2$ has gradient $\nabla f = (2x, 4y)$. Starting at $(1, 1)$, the gradient is $(2, 4)$ — pointing uphill. Moving in the opposite direction $(-2, -4)$ with step size $\alpha = 0.1$: $(1, 1) - 0.1(2, 4) = (0.8, 0.6)$. After many steps, we approach $(0, 0)$, the global minimum.

**Example 2: Temperature Distribution.** A metal plate has temperature $T(x, y) = 100 - (x^2 + 2y^2)$. The heat flux (flow of heat) is proportional to $-\nabla T = (2x, 4y)$. Heat flows from hot regions to cold regions, following the negative gradient of temperature.

**Example 3: Robot Path Planning.** A robot wants to reach a goal while avoiding obstacles. A potential field $U(q) = U_{\text{att}}(q) + U_{\text{rep}}(q)$ is defined over the configuration space. The gradient $\nabla U(q)$ gives the force direction. The robot moves along $-\nabla U(q)$, attracted to the goal and repelled by obstacles.

**Example 4: Image Edge Detection.** For a grayscale image $I(x, y)$, edges occur where $\|\nabla I\|$ is large. The Sobel operator approximates $\partial I/\partial x$ and $\partial I/\partial y$ via convolution with $3 \times 3$ kernels:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$$
The gradient magnitude is $\sqrt{G_x^2 + G_y^2}$; the direction is $\text{atan2}(G_y, G_x)$.

**Example 5: Financial Optimisation.** In portfolio optimisation, we minimise risk $\sigma^2(w) = w^T \Sigma w$ subject to return constraints. The gradient $\nabla \sigma^2 = 2\Sigma w$ tells us how the portfolio risk changes when we adjust each asset weight.

## AI/ML Relevance

The gradient is the central computational object in machine learning:

**1. Gradient Descent.** The core algorithm: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. The gradient points in the direction of steepest ascent of the loss; moving opposite to it reduces the loss. Without the gradient, we would have no way to know which direction to adjust parameters.

**2. Stochastic Gradient Descent (SGD).** Instead of computing the gradient on the full dataset (which is expensive), SGD estimates the gradient from a random minibatch:
$$\nabla L(\theta) \approx \frac{1}{|B|} \sum_{i \in B} \nabla L_i(\theta)$$
The stochastic gradient is an unbiased estimate of the true gradient: $\mathbb{E}[\nabla L_B] = \nabla L$. The variance of this estimate slows convergence but enables escaping sharp local minima.

**3. Gradient Descent Variants.** Modern optimisers modify the gradient update:

- **Momentum:** $v_{t+1} = \beta v_t + \nabla L(\theta_t)$, $\theta_{t+1} = \theta_t - \alpha v_{t+1}$. Accumulates gradient history to smooth oscillations and accelerate convergence in directions of consistent gradient.

- **Nesterov Accelerated Gradient:** $v_{t+1} = \beta v_t + \nabla L(\theta_t - \beta v_t)$, $\theta_{t+1} = \theta_t - \alpha v_{t+1}$. Looks ahead to where the momentum is taking us, then corrects.

- **AdaGrad:** $\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \varepsilon}} \odot \nabla L(\theta_t)$, where $G_t$ accumulates squared gradients. Adapts learning rate per parameter.

- **RMSprop:** $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$, $\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \varepsilon}} g_t$. Uses moving average of squared gradients.

- **Adam:** Combines momentum and RMSprop with bias correction:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

**4. Gradient Clipping.** When gradients are very large (exploding gradients), we clip them:
$$g \leftarrow \frac{g}{\|g\|} \cdot \min(\|g\|, \text{clip\_threshold})$$
This prevents parameter updates from being dominated by a single large gradient.

**5. Visualising Optimisation Landscapes.** The loss landscape $L(\theta)$ is a high-dimensional surface. We can visualise 2D slices by varying two directions: the gradient direction and a random direction:
$$L(\theta_0 + \alpha \nabla L + \beta \mathbf{r})$$
This reveals the curvature, minima, and saddle points of the loss.

**6. Gradient Checking.** To verify correct backpropagation implementation, we compare analytical gradients with numerical gradients:
$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + \varepsilon e_i) - L(\theta - \varepsilon e_i)}{2\varepsilon}$$

**7. Natural Gradient.** The gradient in Euclidean space may not be optimal when the parameter space has curvature (Riemannian geometry). The natural gradient adjusts for the Fisher information matrix:
$$\theta_{t+1} = \theta_t - \alpha F(\theta_t)^{-1} \nabla L(\theta_t)$$
This is invariant to parameter reparameterisations.

**8. Gradient in Deep Learning Theory:**
- **Neural Tangent Kernel:** $K(x, x') = \nabla_\theta f(x; \theta)^T \nabla_\theta f(x'; \theta)$ — the inner product of gradients at different inputs.
- **Gradient Flow:** In the limit of infinitesimally small learning rates, gradient descent becomes gradient flow: $\frac{d\theta}{dt} = -\nabla L(\theta(t))$.
- **Flat Minima:** Gradients near flat minima have small magnitude, and solutions in flat regions generalise better (empirically).

**9. Per-Parameter Gradients.** The gradient $\frac{\partial L}{\partial w_{ij}^{(\ell)}}$ tells us how much a specific weight in layer $\ell$ connecting neuron $j$ to neuron $i$ affects the loss. Gradients for different layers often have different magnitudes, motivating layer-wise learning rate adaptation.

## Mathematical Explanation

**Directional Derivative and Steepest Ascent:**
The directional derivative in direction $\mathbf{v}$ (unit vector) is:
$$D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{v} = \|\nabla f(\mathbf{x})\| \cos \theta$$
where $\theta$ is the angle between $\nabla f$ and $\mathbf{v}$.

The directional derivative is maximised when $\cos \theta = 1$, i.e., when $\mathbf{v}$ points in the same direction as $\nabla f$. The maximum value is $\|\nabla f(\mathbf{x})\|$.

Similarly, the steepest descent direction is $-\nabla f(\mathbf{x})$.

**Gradient Orthogonal to Level Sets:**
If $\mathbf{r}(t)$ is a curve along a level set $f(\mathbf{r}(t)) = c$, then:
$$\frac{d}{dt} f(\mathbf{r}(t)) = \nabla f(\mathbf{r}(t)) \cdot \mathbf{r}'(t) = 0$$
Thus $\nabla f$ is orthogonal to the tangent vector $\mathbf{r}'(t)$ of the level set.

**Gradient Descent as Euler Discretisation of Gradient Flow:**
The continuous-time gradient flow is $\frac{d\theta}{dt} = -\nabla L(\theta)$. Euler discretisation with step $\alpha$ gives:
$$\theta_{t+1} - \theta_t = -\alpha \nabla L(\theta_t) \implies \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

**Relationship to Hessian:**
The Hessian $H = \nabla^2 L$ (matrix of second derivatives) describes how the gradient changes locally. Near a point $\theta_0$:
$$\nabla L(\theta) \approx \nabla L(\theta_0) + H(\theta_0)(\theta - \theta_0)$$

**Convergence Rate of Gradient Descent:**
For a convex function with $L$-Lipschitz gradient and strong convexity $\mu$:
$$\|\theta_t - \theta^*\| \leq \left(1 - \frac{\mu}{L}\right)^t \|\theta_0 - \theta^*\|$$
The rate depends on the condition number $\kappa = L/\mu$ — ill-conditioned problems (high $\kappa$) converge slowly.

## Formula(s)

**Gradient Definition:**
$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right)$$

**Directional Derivative:**
$$D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{v}$$

**Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

**Stochastic Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \frac{1}{|B|} \sum_{i \in B} \nabla L_i(\theta_t)$$

**Momentum:**
$$v_{t+1} = \beta v_t + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_{t+1}$$

**Adam Optimiser:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

**Natural Gradient:**
$$\theta_{t+1} = \theta_t - \alpha F(\theta_t)^{-1} \nabla L(\theta_t)$$

**Gradient of Loss w.r.t. Parameters (Linear Regression):**
$$\nabla L = \frac{2}{N} X^T (X\theta - y)$$

## Properties

1. **Direction of Steepest Ascent:** $\nabla f(\mathbf{x})$ points in the direction of greatest increase of $f$ at $\mathbf{x}$.

2. **Magnitude Equals Slope:** $\|\nabla f(\mathbf{x})\|$ equals the maximum rate of increase (slope in the steepest direction).

3. **Orthogonality to Level Sets:** $\nabla f$ is perpendicular to the level set $\{\mathbf{x} : f(\mathbf{x}) = c\}$ passing through $\mathbf{x}$.

4. **Linearity:** $\nabla(af + bg) = a\nabla f + b\nabla g$ for constants $a, b$.

5. **Product Rule:** $\nabla(fg) = f \nabla g + g \nabla f$.

6. **Chain Rule:** $\nabla(f \circ g)(\mathbf{x}) = f'(g(\mathbf{x})) \nabla g(\mathbf{x})$, where $f: \mathbb{R} \to \mathbb{R}$ and $g: \mathbb{R}^n \to \mathbb{R}$.

7. **Zero Gradient at Extrema:** At local minima, maxima, and saddle points, $\nabla f = \mathbf{0}$.

8. **Gradient is a Vector Field:** $\nabla f$ assigns a vector to each point in the domain, creating a vector field.

9. **Conservative Vector Fields:** If $\mathbf{F} = \nabla f$ for some $f$, then $\mathbf{F}$ is conservative: line integrals are path-independent, and $\oint \mathbf{F} \cdot d\mathbf{r} = 0$.

10. **Second Derivative Test:** The Hessian at a critical point determines its nature: positive definite $\to$ local min, negative definite $\to$ local max, indefinite $\to$ saddle.

## Step-by-Step Worked Examples

### Example 1: Computing the Gradient

Find the gradient of $f(x, y, z) = x^2 y + y e^z + z \sin x$ at the point $(1, 0, \pi)$.

**Step 1:** Compute partial derivatives.
$$\frac{\partial f}{\partial x} = 2xy + z \cos x$$
$$\frac{\partial f}{\partial y} = x^2 + e^z$$
$$\frac{\partial f}{\partial z} = y e^z + \sin x$$

**Step 2:** Evaluate at $(1, 0, \pi)$.
$$\frac{\partial f}{\partial x}(1, 0, \pi) = 2(1)(0) + \pi \cos(1) = \pi \cos(1) \approx 1.697$$
$$\frac{\partial f}{\partial y}(1, 0, \pi) = 1 + e^\pi \approx 1 + 23.14 = 24.14$$
$$\frac{\partial f}{\partial z}(1, 0, \pi) = 0 \cdot e^\pi + \sin(1) \approx 0.8415$$

**Answer:** $\nabla f(1, 0, \pi) = (\pi \cos(1), 1 + e^\pi, \sin(1)) \approx (1.697, 24.14, 0.8415)$.

### Example 2: Gradient Descent for a Simple Function

Perform two steps of gradient descent on $f(x, y) = x^2 + 2y^2$ starting at $(3, 2)$ with $\alpha = 0.1$.

**Step 1:** Compute gradient: $\nabla f = (2x, 4y)$.

**Step 2:** Step 1: $\theta_1 = \theta_0 - \alpha \nabla f(\theta_0)$.
$$\theta_1 = (3, 2) - 0.1(6, 8) = (3 - 0.6, 2 - 0.8) = (2.4, 1.2)$$

**Step 3:** Compute gradient at $\theta_1$: $\nabla f(2.4, 1.2) = (4.8, 4.8)$.

**Step 4:** Step 2: $\theta_2 = (2.4, 1.2) - 0.1(4.8, 4.8) = (2.4 - 0.48, 1.2 - 0.48) = (1.92, 0.72)$.

**Answer:** After step 1: $(2.4, 1.2)$. After step 2: $(1.92, 0.72)$. The values approach the minimum at $(0, 0)$, with $f$ decreasing from $f(3, 2) = 17$ to $f(2.4, 1.2) = 8.64$ to $f(1.92, 0.72) = 4.7232$.

### Example 3: Gradient of Loss for Linear Regression

Consider linear regression $y = Wx + b$ with MSE loss $L = \frac{1}{2N} \sum_{i=1}^N (Wx_i + b - y_i)^2$. Compute $\nabla L$ (gradient w.r.t. $W$ and $b$).

**Step 1:** Gradient w.r.t. $W$:
$$\frac{\partial L}{\partial W} = \frac{1}{N} \sum_{i=1}^N (Wx_i + b - y_i) \cdot x_i = \frac{1}{N} X^T (XW + b - y)$$

**Step 2:** Gradient w.r.t. $b$:
$$\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^N (Wx_i + b - y_i)$$

**Step 3:** The full gradient is the vector of these components.

**Answer:** $\nabla L = \left(\frac{1}{N} X^T (XW + b - y), \frac{1}{N} \sum_i (Wx_i + b - y_i)\right)$.

### Example 4: Directional Derivative

Find the directional derivative of $f(x, y) = x^2 y$ at $(1, 2)$ in the direction $\mathbf{v} = (3, 4)$.

**Step 1:** Compute gradient: $\nabla f = (2xy, x^2) = (4, 1)$ at $(1, 2)$.

**Step 2:** Normalise the direction vector: $\|\mathbf{v}\| = \sqrt{9 + 16} = 5$, so $\mathbf{u} = (3/5, 4/5)$.

**Step 3:** Directional derivative: $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = (4, 1) \cdot (3/5, 4/5) = 12/5 + 4/5 = 16/5 = 3.2$.

**Answer:** The rate of change of $f$ at $(1, 2)$ in the direction $(3, 4)$ is $3.2$.

### Example 5: Gradient of a Neural Network Loss

For a 2-layer network $L = \frac{1}{2}(W_2 \sigma(W_1 x) - y)^2$, compute $\partial L / \partial W_1$ and write it in terms of $\nabla L$.

**Step 1:** Define $z = W_1 x$, $h = \sigma(z)$, $\hat{y} = W_2 h$, $L = \frac{1}{2}(\hat{y} - y)^2$.

**Step 2:** By backpropagation (chain rule):
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial W_1}$$
$$= (\hat{y} - y) \cdot W_2^T \cdot \sigma'(z) \cdot x^T$$

**Step 3:** The gradient of $L$ with respect to all parameters is:
$$\nabla L = \left(\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2}\right)$$

**Answer:** $\partial L / \partial W_1 = (\hat{y} - y) W_2^T \sigma'(W_1 x) x^T$, and the full gradient is the collection of all partial derivatives.

### Example 6: Gradient Descent with Momentum

Minimise $f(x, y) = x^2 + 10y^2$ starting at $(1, 1)$ with $\alpha = 0.1$, $\beta = 0.9$, 2 steps.

**Step 1:** Initialise $v_0 = (0, 0)$, $\theta_0 = (1, 1)$.

**Step 2:** Compute gradient: $\nabla f(1, 1) = (2, 20)$.

**Step 3:** Update velocity: $v_1 = 0.9 \cdot (0, 0) + (2, 20) = (2, 20)$.

**Step 4:** Update parameters: $\theta_1 = (1, 1) - 0.1 \cdot (2, 20) = (0.8, -1)$.

**Step 5:** Compute gradient: $\nabla f(0.8, -1) = (1.6, -20)$.

**Step 6:** Update velocity: $v_2 = 0.9 \cdot (2, 20) + (1.6, -20) = (1.8 + 1.6, 18 - 20) = (3.4, -2)$.

**Step 7:** Update parameters: $\theta_2 = (0.8, -1) - 0.1 \cdot (3.4, -2) = (0.46, -0.8)$.

**Answer:** After 2 steps: $(0.46, -0.8)$. Momentum helps accelerate convergence in the $y$-direction (which has a steeper gradient).

## Visual Interpretation

**Gradient as an Arrow Field:**
For $f(x, y) = x^2 + y^2$, the gradient $\nabla f = (2x, 2y)$ produces arrows pointing radially outward from the origin. The magnitude increases with distance from the origin:
```
      ↑
      |
   ↖ ↑ ↗
← ← o → →
   ↙ ↓ ↘
      |
      ↓
```

**Gradient Descent on a 1D Function:**
```
Loss
  |\
  | \     . step 1
  |  \   .
  |   \ .
  |    \. step 2
  |     *--- step 3
  +---------------> Weight
```
At each step, we compute the slope (derivative) and move downhill.

**Gradient Descent on a 2D Surface (Contour Map):**
```
y
  |   ..... (contours of constant f)
  |  .     .
  | .   ↑   .
  |.    | grad .
  |.    |      .
  |.    ↓ move  .
  | .  opposite  .
  |  .     .     .
  |   ..... (ellipses become circles near min)
  +----------------> x
```
The gradient is perpendicular to contour lines. Moving opposite to the gradient crosses contours most efficiently.

**Loss Landscape Visualisation:**
```
Loss
  \           /
   \    /\   /
    \  /  \ /
     \/    \
     saddle   min
```
Local minima, saddle points, and steep regions are all navigated by following the gradient.

## Common Mistakes

1. **Confusing the gradient with the derivative.** The gradient is a vector (collection of partial derivatives), not a scalar. For $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is in $\mathbb{R}^n$, while the derivative $f'(x)$ would be a $1 \times n$ row vector (Jacobian).

2. **Assuming the gradient always points to the minimum.** The gradient points in the direction of steepest ascent from the current point, which is the direction of fastest increase at that specific point. It does not necessarily point directly toward the global minimum (unless the function is quadratic with spherical contours).

3. **Using too large a learning rate.** If $\alpha$ is too large, gradient descent can overshoot and diverge. The maximum stable learning rate depends on the Lipschitz constant of the gradient: $\alpha < 2/L$.

4. **Using too small a learning rate.** If $\alpha$ is too small, convergence is very slow. The algorithm may get stuck before reaching the minimum.

5. **Forgetting that the gradient is zero at critical points.** Gradient descent stops at points where $\nabla f = 0$ (local minima, maxima, or saddle points). In non-convex optimisation (neural networks), this often means a saddle point rather than a minimum.

6. **Not normalising input features.** Gradient descent converges much faster when all input features are on approximately the same scale (e.g., zero mean, unit variance). Otherwise, the gradient is dominated by features with larger scales.

7. **Treating stochastic and full gradient descent the same.** SGD with mini-batches has noisy gradients — the variance of the gradient estimate causes the loss to bounce around rather than smoothly decrease. Learning rate schedules and momentum help mitigate this.

8. **Ignoring gradient clipping.** For deep networks (especially RNNs), gradients can explode. Not clipping gradients can cause numerical overflow and training collapse.

9. **Thinking natural gradient and Euclidean gradient are the same.** The natural gradient adjusts for the curvature of the parameter space (Riemannian metric). In standard gradient descent, we implicitly assume Euclidean geometry, which may not be appropriate for parameters with different scales or correlations.

10. **Equating gradient magnitude with proximity to minimum.** A small gradient does not necessarily mean you are near a minimum — you could be on a plateau or near a saddle point. Conversely, a large gradient does not mean you are far from a minimum — some functions have steep slopes even near the minimum (e.g., $f(x) = x^2$ near 0 has gradient $2x$, which approaches 0 as $x \to 0$).

## Interview Questions

### Beginner

1. **What is the gradient of a function?**
   *Answer: The gradient $\nabla f$ of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of its first-order partial derivatives: $\nabla f = (\partial f / \partial x_1, \dots, \partial f / \partial x_n)$. It points in the direction of steepest ascent and its magnitude is the rate of change in that direction.*

2. **What is the relationship between the gradient and gradient descent?**
   *Answer: Gradient descent minimises a function by iteratively moving in the opposite direction of the gradient: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$. The gradient points uphill, so moving opposite to it moves downhill toward a minimum.*

3. **Compute $\nabla f$ for $f(x, y) = 3x^2 + 4y^2$.**
   *Answer: $\nabla f = (6x, 8y)$.*

4. **What does $\|\nabla f(\mathbf{x})\| = 0$ mean?**
   *Answer: It means all partial derivatives are zero at $\mathbf{x}$, so $\mathbf{x}$ is a critical point — potentially a local minimum, local maximum, or saddle point.*

5. **What is a directional derivative?**
   *Answer: The directional derivative $D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{v}$ measures the rate of change of $f$ at $\mathbf{x}$ in the direction of unit vector $\mathbf{v}$. It is maximised when $\mathbf{v}$ points in the direction of the gradient.*

### Intermediate

1. **Explain stochastic gradient descent (SGD) and why it is used instead of full batch gradient descent.**
   *Answer: SGD estimates the gradient using a random mini-batch: $\nabla L_B(\theta) = \frac{1}{|B|} \sum_{i \in B} \nabla L_i(\theta)$. This is unbiased: $\mathbb{E}[\nabla L_B] = \nabla L$. SGD is used because: (1) computing the full gradient on large datasets is expensive (millions of examples). (2) The noise in SGD can help escape sharp local minima and saddle points. (3) SGD often converges faster in wall-clock time because updates are more frequent. (4) It generalises better empirically.*  

2. **How does momentum improve gradient descent?**
   *Answer: Momentum accumulates past gradients: $v_{t+1} = \beta v_t + \nabla L(\theta_t)$, $\theta_{t+1} = \theta_t - \alpha v_{t+1}$. Benefits: (1) Accelerates convergence in directions of consistent gradient (e.g., ravines). (2) Dampens oscillations in directions where gradients oscillate. (3) Helps escape plateaus and shallow local minima. (4) Reduces the variance of SGD. The parameter $\beta$ (typically 0.9) controls how much past gradients influence the current update.*

3. **What is the relationship between the gradient and level sets?**
   *Answer: The gradient is orthogonal (perpendicular) to level sets. For a level set $f(\mathbf{x}) = c$, any tangent vector $\mathbf{t}$ satisfies $\nabla f \cdot \mathbf{t} = 0$ (since $f$ is constant along the level set). This means the direction of maximum increase (gradient) is perpendicular to the direction of no change (level set). This is why, on a contour map of a mountain, the steepest path is perpendicular to the contour lines.*

4. **Why does gradient clipping help in training RNNs?**
   *Answer: RNNs are prone to exploding gradients because the gradient through $T$ time steps involves $T$ multiplications of the recurrent weight matrix $W$. If the spectral radius of $W$ exceeds 1, the gradient grows exponentially: $\prod_{t=1}^T W^T$ explodes. Gradient clipping caps the gradient norm: $g \leftarrow g \cdot \min(1, \text{threshold}/\|g\|)$. This prevents a single large gradient from destabilising training, while preserving the gradient direction.*

5. **Derive the update rule for Adam from first principles.**
   *Answer: Adam combines momentum and RMSprop. (1) Compute biased first moment estimate: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ (exponential moving average of gradients). (2) Compute biased second moment estimate: $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ (exponential moving average of squared gradients). (3) Correct bias: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$ — needed because $m_t$ and $v_t$ are initialised at 0. (4) Update: $\theta_{t+1} = \theta_t - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$. The effective learning rate is $\alpha / (\sqrt{\hat{v}_t} + \varepsilon)$, which is per-parameter: parameters with large gradients get smaller updates (normalised by their gradient history).*

### Advanced

1. **Derive the convergence rate of gradient descent for strongly convex functions with Lipschitz gradients. Explain the condition number and its effect.**
   *Answer: For a $\mu$-strongly convex function with $L$-Lipschitz gradient: $\frac{\mu}{2}\|\theta - \theta^*\|^2 \leq f(\theta) - f(\theta^*) \leq \frac{L}{2}\|\theta - \theta^*\|^2$. Gradient descent with $\alpha = 1/L$ gives: $\|\theta_{t+1} - \theta^*\|^2 \leq (1 - \mu/L)\|\theta_t - \theta^*\|^2$. Thus $\|\theta_t - \theta^*\| \leq (1 - \mu/L)^{t/2}\|\theta_0 - \theta^*\|$ — linear convergence. The ratio $\kappa = L/\mu$ is the condition number. Ill-conditioned problems (high $\kappa$) converge slowly because the loss surface is much steeper in some directions than others. Preconditioning (scaling variables) reduces $\kappa$ and accelerates convergence. For $\kappa = 10^6$ (not uncommon in poorly-scaled problems), gradient descent needs $\sim 10^6$ iterations, while with optimal preconditioning ($\kappa = 1$), one iteration may suffice.*

2. **Explain the concept of gradient flow and its connection to neural network training dynamics in the infinite-width limit (Neural Tangent Kernel).**
   *Answer: Gradient flow is gradient descent in the limit of infinitesimal step size: $\frac{d\theta}{dt} = -\nabla L(\theta(t))$. In function space, the network output evolves as $\frac{df(x; \theta(t))}{dt} = \nabla_\theta f(x; \theta(t))^T \frac{d\theta}{dt} = -\nabla_\theta f(x)^T \nabla_\theta L = -\frac{1}{N} \sum_i K_t(x, x_i)(f(x_i) - y_i)$, where $K_t(x, x') = \nabla_\theta f(x; \theta(t))^T \nabla_\theta f(x'; \theta(t))$ is the Neural Tangent Kernel (NTK). In the infinite-width limit, the NTK becomes constant ($K_t \to K^*$), and the network evolves under kernel gradient descent: $\frac{df_t(x)}{dt} = -\frac{1}{N} \sum_i K^*(x, x_i)(f_t(x_i) - y_i)$. This is a linear ODE in function space, showing that infinitely wide neural networks trained by gradient descent converge to the kernel regression solution with the NTK kernel. The gradient flow perspective reveals that deep learning in the infinite-width limit is equivalent to kernel methods.*

3. **Design a custom optimiser that combines the benefits of Adam (adaptive learning rates) with Nesterov momentum (look-ahead gradient). Derive the update equations.**
   *Answer: The NAdam optimiser combines Nesterov momentum with Adam. The key insight: Nesterov computes the gradient at the "look-ahead" position $\theta_t - \beta m_{t-1}$, while Adam uses the gradient at $\theta_t$. NAdam: (1) Compute look-ahead: $\bar{\theta}_t = \theta_t - \beta m_{t-1}$. (2) Compute gradient at look-ahead: $g_t = \nabla L(\bar{\theta}_t)$. (3) Update first moment: $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$. (4) Update second moment: $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$. (5) Bias correction: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$. (6) Nesterov correction: $\bar{m}_t = (1-\beta_1)g_t + \beta_1 \hat{m}_t$ (approximates Nesterov momentum). (7) Update: $\theta_{t+1} = \theta_t - \alpha \bar{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$. This combines the adaptive learning rates of Adam with the look-ahead property of Nesterov, which can provide faster convergence and better generalisation. Most deep learning frameworks implement NAdam as a built-in optimiser.*

## Practice Problems

### Easy

1. Compute $\nabla f$ for $f(x, y) = \sin(xy)$.
2. Find the gradient of $f(x, y, z) = x^2 + y^2 + z^2$ at $(1, 2, 3)$.
3. Compute the directional derivative of $f(x, y) = x^2 y$ at $(1, 1)$ in the direction $(1, 0)$.
4. Perform one step of gradient descent on $f(x) = (x-3)^2$ starting at $x = 0$ with $\alpha = 0.1$.
5. For $f(x, y) = x^2 + y^2$, what is the direction of steepest descent at $(1, 1)$?

### Medium

1. Perform two steps of gradient descent on $f(x, y) = 2x^2 + 3y^2$ starting at $(2, 2)$ with $\alpha = 0.1$.
2. Compute $\nabla L$ for $L(w, b) = \frac{1}{N} \sum_{i=1}^N (wx_i + b - y_i)^2$.
3. Show that $\nabla f$ is orthogonal to level curves for $f(x, y) = x^2 + y^2$.
4. Find the direction of steepest ascent of $f(x, y) = e^{-(x^2 + y^2)}$ at $(1, 1)$.
5. For $f(x, y) = x^2 y + y^3$, compute $\nabla f$ and all second-order partial derivatives.

### Hard

1. Prove that gradient descent with exact line search converges monotonically for convex quadratic functions.
2. Derive the natural gradient update for a Bernoulli logistic regression model.
3. Implement (in pseudocode) a gradient descent optimiser with Adam, including bias correction, for a neural network training loop. Show how the gradient flows from the loss through each parameter.

## Solutions

### Easy Solutions

**1.** $\nabla f = (y \cos(xy), x \cos(xy))$.

**2.** $\nabla f = (2x, 2y, 2z) = (2, 4, 6)$ at $(1, 2, 3)$.

**3.** $\nabla f(1, 1) = (2xy, x^2) = (2, 1)$. Unit direction: $(1, 0)$. Directional derivative: $(2, 1) \cdot (1, 0) = 2$.

**4.** $f'(x) = 2(x-3)$. At $x = 0$, $f'(0) = -6$. Step: $x_1 = 0 - 0.1(-6) = 0.6$.

**5.** $\nabla f = (2x, 2y) = (2, 2)$ at $(1, 1)$. Direction of steepest descent: $-\nabla f / \|\nabla f\| = (-1/\sqrt{2}, -1/\sqrt{2})$.

### Medium Solutions

**1.** $\nabla f = (4x, 6y)$. Step 1: $(2, 2) - 0.1(8, 12) = (1.2, 0.8)$. Step 2: $\nabla f(1.2, 0.8) = (4.8, 4.8)$. $(1.2, 0.8) - 0.1(4.8, 4.8) = (0.72, 0.32)$.

**2.** $\frac{\partial L}{\partial w} = \frac{2}{N} \sum (wx_i + b - y_i) x_i$, $\frac{\partial L}{\partial b} = \frac{2}{N} \sum (wx_i + b - y_i)$. So $\nabla L = (\frac{2}{N} X^T (Xw + b - y), \frac{2}{N} \mathbf{1}^T (Xw + b - y))$.

**3.** A level curve is $x^2 + y^2 = c$, a circle. The gradient is $\nabla f = (2x, 2y)$, which at any point $(x, y)$ points radially outward. The radius vector is orthogonal to the tangent of the circle, so $\nabla f$ is orthogonal to the level curve.

**4.** $\nabla f = (-2x e^{-(x^2+y^2)}, -2y e^{-(x^2+y^2)}) = (-2e^{-2}, -2e^{-2})$ at $(1, 1)$. Direction: $(-1/\sqrt{2}, -1/\sqrt{2})$. Interestingly, this is toward the origin (the maximum of the function).

**5.** $\nabla f = (2xy, x^2 + 3y^2)$. $f_{xx} = 2y$, $f_{xy} = 2x$, $f_{yx} = 2x$, $f_{yy} = 6y$.

### Hard Solutions

**1.** For $f(\theta) = \frac{1}{2} \theta^T A \theta - b^T \theta + c$ with $A$ symmetric positive definite, gradient descent with exact line search (choosing $\alpha_t$ to minimise $f(\theta_t - \alpha \nabla f(\theta_t))$) yields $\alpha_t = \frac{\|\nabla f(\theta_t)\|^2}{\nabla f(\theta_t)^T A \nabla f(\theta_t)}$. The convergence is: $\|\theta_{t+1} - \theta^*\|_A^2 \leq \left(1 - \frac{1}{\kappa}\right) \|\theta_t - \theta^*\|_A^2$, where $\kappa = \lambda_{\max}(A)/\lambda_{\min}(A)$ is the condition number and $\|x\|_A^2 = x^T A x$ is the energy norm. The monotonic decrease follows from the fact that exact line search ensures the steepest descent direction reduces $f$ at each step.

**2.** For logistic regression $p(y=1|x) = \sigma(w^T x)$, the loss is $L = -\sum_i [y_i \log \sigma(w^T x_i) + (1-y_i) \log(1-\sigma(w^T x_i))]$. The gradient is $\nabla L = \sum_i (\sigma(w^T x_i) - y_i) x_i$. The Fisher information matrix is $F = \mathbb{E}_{p(y|x)}[\nabla L \nabla L^T] = \sum_i \sigma(w^T x_i)(1-\sigma(w^T x_i)) x_i x_i^T$. The natural gradient update is $w_{t+1} = w_t - \alpha F^{-1} \nabla L$, which is invariant to linear reparameterisations of the input.

**3.** Pseudocode:
```
def adam_update(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    new_params = []
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g**2
        m_hat = m[i] / (1 - b1**t)
        v_hat = v[i] / (1 - b2**t)
        p_new = p - lr * m_hat / (sqrt(v_hat) + eps)
        new_params.append(p_new)
    return new_params, m, v

# Training loop:
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = forward(batch)  # forward pass
        grads = backward(loss, params)  # backprop -> chain rule
        params, m, v = adam_update(params, grads, m, v, t)
        t += 1
```
The gradient flows from the loss (scalar) backward through each operation (chain rule), producing $\partial L / \partial w_i$ for every parameter. Adam then adaptively scales each gradient component.

## Related Concepts

- **Partial Derivative** (MATH-056) — Each component of the gradient is a partial derivative.
- **Derivative** (MATH-055) — The gradient generalises the derivative to multiple variables.
- **Chain Rule** (MATH-057) — Gradients through compositions use the chain rule.
- **Directional Derivative** — The dot product of the gradient with a direction vector.
- **Hessian Matrix** — The matrix of second partial derivatives; describes gradient change.
- **Vector** (MATH-002) — The gradient is a vector.
- **Linear Transformation** (MATH-036) — The gradient defines a linear approximation: $f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \mathbf{h}$.

## Next Concepts

- **Divergence and Curl** — Differential operators on vector fields ($\nabla \cdot \mathbf{F}$, $\nabla \times \mathbf{F}$).
- **Laplacian** — $\nabla^2 f = \nabla \cdot \nabla f = \sum \partial^2 f / \partial x_i^2$.
- **Jacobian Matrix** — The gradient generalised to vector-valued functions.
- **Hessian-Free Optimisation** — Methods that compute Hessian-vector products without forming the full Hessian.
- **Riemannian Gradient Descent** — Gradient descent on manifolds, accounting for non-Euclidean geometry.

## Summary

The gradient $\nabla f$ is the vector of first-order partial derivatives, pointing in the direction of steepest ascent and orthogonal to level sets. Gradient descent $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$ is the foundational algorithm for minimising loss functions in machine learning. Variants like SGD, momentum, and Adam modify the update to improve convergence, handle noise, and adapt learning rates per parameter. The gradient is the central object in optimisation, connecting calculus to the practical training of neural networks with millions of parameters.

## Key Takeaways

- $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)$ — a vector of all partial derivatives.
- The gradient points in the direction of steepest ascent; $-\nabla f$ points in the direction of steepest descent.
- The gradient is orthogonal to level sets (contours of constant $f$).
- Gradient descent: $\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$.
- SGD uses mini-batches for noisy but cheap gradient estimates.
- Momentum accelerates convergence by accumulating past gradients.
- Adam adapts learning rates per parameter using first and second moment estimates.
- The magnitude $\|\nabla f\|$ equals the maximum rate of change.
- At critical points (min, max, saddle), $\nabla f = \mathbf{0}$.
- Gradient clipping prevents exploding gradients in deep networks and RNNs.
- The condition number $\kappa = L/\mu$ determines gradient descent convergence rate.
- Natural gradient adjusts for parameter space curvature via the Fisher information matrix.
- Gradient flow is the continuous-time limit of gradient descent.
- The Neural Tangent Kernel connects gradient descent in infinite-width networks to kernel regression.
