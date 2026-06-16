# Concept: Hessian Matrix

## Concept ID

MATH-060

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the Hessian matrix as the matrix of all second-order partial derivatives of a scalar-valued function
- Compute the Hessian matrix for multivariable functions
- Use the Hessian to classify critical points (minima, maxima, saddle points)
- Apply the Hessian in second-order optimisation methods
- Understand the relationship between the Hessian and the curvature of the loss landscape
- Interpret the condition number of the Hessian for optimisation convergence

## Prerequisites

- Multivariable calculus (first-order partial derivatives, gradients)
- Linear algebra (eigenvalues, eigenvectors, positive definiteness)
- The Jacobian matrix (MATH-059)
- Taylor series in multiple variables
- Basic optimisation concepts (gradient descent, critical points)

## Definition

Let $f: \mathbb{R}^n \to \mathbb{R}$ be a scalar-valued function that is twice differentiable. The **Hessian matrix** of $f$, denoted $\mathbf{H}(f)$ or $\nabla^2 f$, is an $n \times n$ symmetric matrix of all second-order partial derivatives:

$$
\mathbf{H}(f) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}_{n \times n}
$$

The $(i,j)$ entry is $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$. By Clairaut's theorem (also known as Schwarz's theorem), if all second-order partial derivatives are continuous, then mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ for all $i, j$, making the Hessian symmetric.

## Intuition

The Hessian captures the local curvature of a function $f$ at a given point. While the gradient $\nabla f$ tells us the direction and rate of steepest ascent (like the slope of a tangent plane), the Hessian tells us how the slope itself is changing — it describes the "bending" or "curvature" of the function.

Imagine standing on a surface $z = f(x, y)$. The gradient points downhill. But is the downhill direction getting steeper or shallower? Is the valley narrow and steep or wide and gentle? The Hessian answers these questions. Its eigenvalues correspond to the principal curvatures of the surface in orthogonal directions. A positive eigenvalue means the function curves upward in that direction (concave up), while a negative eigenvalue means it curves downward (concave down).

This curvature information is critical for determining the nature of critical points (where $\nabla f = 0$):
- If the Hessian is **positive definite** (all eigenvalues $> 0$), the point is a **local minimum**
- If the Hessian is **negative definite** (all eigenvalues $< 0$), the point is a **local maximum**
- If the Hessian has both positive and negative eigenvalues, the point is a **saddle point**

The magnitude of the eigenvalues tells us how "sharp" the extremum is, which directly affects optimisation dynamics.

## Why This Concept Matters

The Hessian is fundamental to understanding and optimising complex functions. In classical optimisation, Newton's method uses the Hessian to achieve quadratic convergence near minima. In machine learning, the Hessian of the loss function governs the behaviour of gradient-based optimisation algorithms.

The loss landscape of neural networks — the function mapping parameters to loss — is a high-dimensional surface whose geometry determines training difficulty. The Hessian reveals the curvature of this landscape: flat directions (small eigenvalues) correspond to parameter changes that barely affect the loss, while sharp directions (large eigenvalues) correspond to sensitive parameters. The ratio of the largest to smallest eigenvalue — the **condition number** of the Hessian — dictates how quickly gradient descent converges, with ill-conditioned problems requiring many iterations.

The concept of "flat minima" (where the Hessian has small eigenvalues, indicating broad basins of attraction) has been linked to better generalisation in deep learning, motivating techniques like Sharpness-Aware Minimisation (SAM).

## Historical Background

The Hessian is named after the German mathematician **Ludwig Otto Hesse** (1811–1874), who introduced the matrix in the 19th century while working on the theory of algebraic curves and surfaces. Hesse was a student of Jacobi and a contemporary of Riemann and Weierstrass.

Hesse's primary contributions were in analytical geometry and the theory of determinants. He studied the second-order partial derivatives of functions defined by algebraic equations, particularly in the context of finding inflection points and classifying singularities of curves. His work laid the foundation for the second-derivative test in multivariable calculus.

The term "Hesse matrix" was coined later by James Joseph Sylvester, who recognised the importance of this construction. The Hessian became a standard tool in differential geometry (where it describes the second fundamental form of a surface) and optimisation theory (where it is the key to second-order methods).

Carl Gustav Jacob Jacobi, Hesse's teacher, had earlier studied determinants of partial derivatives (now called Jacobians), and Hesse naturally extended this work to second-order derivatives.

## Real World Examples

1. **Economics: Portfolio Optimisation** — In Markowitz portfolio theory, the risk of a portfolio is a quadratic function of asset weights. The Hessian of this risk function is the covariance matrix of asset returns. Minimising risk subject to return constraints requires solving a system involving this Hessian.

2. **Engineering: Structural Analysis** — The potential energy of a deformed structure is a function of nodal displacements. The Hessian of this energy (the stiffness matrix) determines the structure's response to loads. Its positive definiteness guarantees stability.

3. **Robotics: Motion Planning** — In trajectory optimisation for robotic arms, the cost function includes terms for energy, smoothness, and obstacle avoidance. The Hessian of this cost function determines the convergence properties of the optimisation algorithm used for trajectory generation.

4. **Physics: Phase Transitions** — In statistical mechanics, the free energy's Hessian with respect to order parameters determines the stability of thermodynamic phases. A zero eigenvalue of the Hessian indicates a second-order phase transition.

5. **Medical Imaging: Image Registration** — When aligning medical images, the cost function's Hessian determines how reliably the optimisation will converge. Regions with ill-conditioned Hessians correspond to sliding motions (e.g., lungs against the ribcage) where registration is inherently ambiguous.

## AI/ML Relevance

The Hessian is profoundly important in machine learning for several key applications:

1. **Second-Order Optimisation** — Newton's method updates parameters as $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{H}^{-1} \nabla L(\boldsymbol{\theta}_t)$, where $\mathbf{H}$ is the Hessian of the loss. This can achieve quadratic convergence near minima but requires computing and inverting the $n \times n$ Hessian, which is $O(n^3)$ and infeasible for large models. Quasi-Newton methods (L-BFGS, BFGS) approximate the Hessian using gradient information.

2. **Loss Landscape Visualisation** — by computing the Hessian eigenvectors corresponding to the largest and smallest eigenvalues, researchers can visualise 2D slices of the loss landscape. Projects like "Loss Landscapes" and "Filter Normalisation" have revealed that deep networks often have remarkably connected minimisers.

3. **Flat Minima and Generalisation** — The eigenvalues of the Hessian at convergence correlate with generalisation. "Flat minima" (where most Hessian eigenvalues are small) tend to generalise better than "sharp minima" (large eigenvalues). This insight has led to algorithms like Sharpness-Aware Minimisation (SAM) that explicitly penalise large Hessian eigenvalues.

4. **Saddle Point Detection** — High-dimensional non-convex optimisation is plagued by saddle points where the gradient is zero but the Hessian has both positive and negative eigenvalues. The Hessian's eigenvalue structure determines whether gradient descent will escape a saddle point, with negative curvature directions providing escape routes. This is formalised in the "strict saddle" property.

5. **Natural Gradient Descent** — The Fisher Information Matrix (FIM) is related to the Hessian of the KL divergence. For exponential families, the FIM equals the Hessian of the log-partition function. Natural gradient descent uses the FIM (or its empirical approximation) to define a geometry-aware update: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{F}^{-1} \nabla L(\boldsymbol{\theta}_t)$.

6. **Condition Number and Convergence** — The condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ of the Hessian determines the convergence rate of gradient descent. For a quadratic objective, gradient descent converges as $\left(\frac{\kappa-1}{\kappa+1}\right)^2$. Ill-conditioned problems ($\kappa \gg 1$) converge very slowly, motivating preconditioning techniques.

7. **Hessian-Free Optimisation** — For models with many parameters, forming the full Hessian is impractical. Hessian-free methods (Martens, 2010) compute Hessian-vector products directly using automatic differentiation without forming the full matrix, enabling second-order updates at $O(n)$ cost per product.

## Mathematical Explanation

### Hessian as the Jacobian of the Gradient

The Hessian can be defined as the Jacobian of the gradient function. If $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ maps a point to its gradient vector, then:

$$
\mathbf{H}(f) = \mathbf{J}_{\nabla f}
$$

This means $H_{ij} = \frac{\partial}{\partial x_j} \left( \frac{\partial f}{\partial x_i} \right) = \frac{\partial^2 f}{\partial x_i \partial x_j}$.

### Taylor Expansion with Hessian

The second-order Taylor expansion of $f$ around $\mathbf{x}_0$ is:

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2} (\mathbf{x} - \mathbf{x}_0)^T \mathbf{H}(f)(\mathbf{x}_0) (\mathbf{x} - \mathbf{x}_0)
$$

This quadratic approximation is the foundation of second-order optimisation methods.

### Critical Point Classification

At a critical point $\mathbf{x}^*$ where $\nabla f(\mathbf{x}^*) = 0$:

- If $\mathbf{H}(\mathbf{x}^*)$ is **positive definite** (all eigenvalues $> 0$): $\mathbf{x}^*$ is a **local minimum**
- If $\mathbf{H}(\mathbf{x}^*)$ is **negative definite** (all eigenvalues $< 0$): $\mathbf{x}^*$ is a **local maximum**
- If $\mathbf{H}(\mathbf{x}^*)$ has both positive and negative eigenvalues: $\mathbf{x}^*$ is a **saddle point**
- If $\mathbf{H}(\mathbf{x}^*)$ has zero eigenvalues (semidefinite): the test is inconclusive (degenerate critical point)

### Positive Definiteness Check

A symmetric matrix $\mathbf{A}$ is positive definite if any of the following equivalent conditions hold:
- All eigenvalues $\lambda_i > 0$
- All leading principal minors (determinants of upper-left submatrices) are positive
- $\mathbf{x}^T \mathbf{A} \mathbf{x} > 0$ for all non-zero $\mathbf{x}$

For a $2 \times 2$ Hessian $\begin{bmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{bmatrix}$:
- Positive definite: $f_{xx} > 0$ and $f_{xx} f_{yy} - f_{xy}^2 > 0$
- Negative definite: $f_{xx} < 0$ and $f_{xx} f_{yy} - f_{xy}^2 > 0$
- Saddle: $f_{xx} f_{yy} - f_{xy}^2 < 0$

### Hessian-Vector Product

For large-scale problems, we rarely form the full Hessian. Instead, we compute Hessian-vector products (HVPs) directly:

$$
\mathbf{H}(\mathbf{x}) \mathbf{v} = \nabla \left( \nabla f(\mathbf{x})^T \mathbf{v} \right)
$$

This can be computed with two automatic differentiation passes (one forward, one reverse) at $O(n)$ cost, enabling Hessian-free optimisation.

## Formula(s)

1. **Hessian Matrix Definition**:
   $$
   H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
   $$

2. **Second-Order Taylor Expansion**:
   $$
   f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^T \mathbf{H}(\mathbf{x}_0) \Delta\mathbf{x} + O(\|\Delta\mathbf{x}\|^3)
   $$

3. **Newton's Method Update**:
   $$
   \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \mathbf{H}(\boldsymbol{\theta}_t)^{-1} \nabla L(\boldsymbol{\theta}_t)
   $$

4. **Hessian Condition Number**:
   $$
   \kappa = \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}(\mathbf{H})}
   $$

5. **Gradient Descent Convergence Rate** (quadratic case):
   $$
   \rho = \left(\frac{\kappa - 1}{\kappa + 1}\right)^2
   $$

6. **Hessian-Vector Product**:
   $$
   \mathbf{H}(\mathbf{x}) \mathbf{v} = \nabla_{\mathbf{x}} \left( \nabla_{\mathbf{x}} f(\mathbf{x})^T \mathbf{v} \right)
   $$

7. **Hessian in Neural Networks** (for a scalar loss $L$ with respect to parameters $\boldsymbol{\theta}$):
   $$
   \mathbf{H}_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
   $$

## Properties

1. **Symmetry**: If $f$ is twice continuously differentiable, the Hessian is symmetric: $H_{ij} = H_{ji}$.

2. **Quadratic Form**: The quadratic form $\mathbf{v}^T \mathbf{H} \mathbf{v}$ measures the second directional derivative in direction $\mathbf{v}$: $\frac{d^2}{dt^2} f(\mathbf{x} + t\mathbf{v})\big|_{t=0} = \mathbf{v}^T \mathbf{H} \mathbf{v}$.

3. **Convexity**: A function $f$ is convex if and only if its Hessian is positive semidefinite everywhere.

4. **Chain Rule with Hessian**: For $f(g(\mathbf{x}))$ where $g: \mathbb{R}^n \to \mathbb{R}$,
   $$
   \nabla^2 (f \circ g) = f''(g(\mathbf{x})) \nabla g(\mathbf{x}) \nabla g(\mathbf{x})^T + f'(g(\mathbf{x})) \nabla^2 g(\mathbf{x})
   $$

5. **Invariance under Linear Transformations**: If $\mathbf{y} = \mathbf{A}\mathbf{x}$, then the Hessian of $f(\mathbf{Ax})$ is $\mathbf{A}^T \nabla^2 f(\mathbf{Ax}) \mathbf{A}$.

6. **Spectral Theorem**: Since the Hessian is symmetric, it is orthogonally diagonalisable: $\mathbf{H} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$, where $\mathbf{Q}$'s columns are eigenvectors and $\boldsymbol{\Lambda}$ contains eigenvalues.

7. **Rank Deficiency**: Zero eigenvalues indicate directions in which the function is locally flat (no curvature). This relates to overparameterisation in neural networks.

8. **Trace and Volume**: The trace of the Hessian $\operatorname{tr}(\mathbf{H}) = \sum_i \frac{\partial^2 f}{\partial x_i^2} = \nabla^2 f$ (the Laplacian) measures the mean curvature.

## Step-by-Step Worked Examples

### Example 1: Classifying Critical Points of a 2D Function

**Problem**: Find and classify all critical points of $f(x, y) = x^3 - 3x + y^2 + 2y$.

**Solution**:

Step 1: Find critical points by setting the gradient to zero.

$$
\frac{\partial f}{\partial x} = 3x^2 - 3 = 0 \implies x^2 = 1 \implies x = \pm 1
$$

$$
\frac{\partial f}{\partial y} = 2y + 2 = 0 \implies y = -1
$$

Critical points: $(1, -1)$ and $(-1, -1)$.

Step 2: Compute the Hessian matrix.

$$
\frac{\partial^2 f}{\partial x^2} = 6x, \quad \frac{\partial^2 f}{\partial x \partial y} = 0, \quad \frac{\partial^2 f}{\partial y^2} = 2
$$

$$
\mathbf{H}(x, y) = \begin{bmatrix} 6x & 0 \\ 0 & 2 \end{bmatrix}
$$

Step 3: Classify each critical point.

At $(1, -1)$: $\mathbf{H} = \begin{bmatrix} 6 & 0 \\ 0 & 2 \end{bmatrix}$.
- $f_{xx} = 6 > 0$ and $\det(\mathbf{H}) = 6 \cdot 2 - 0 = 12 > 0$ → **Local minimum**.
- Eigenvalues: $\lambda_1 = 6$, $\lambda_2 = 2$, both positive.

At $(-1, -1)$: $\mathbf{H} = \begin{bmatrix} -6 & 0 \\ 0 & 2 \end{bmatrix}$.
- $\det(\mathbf{H}) = -12 < 0$ → **Saddle point**.
- Eigenvalues: $\lambda_1 = -6$, $\lambda_2 = 2$, mixed signs.

Step 4: Evaluate function at critical points.

$f(1, -1) = 1 - 3 + 1 - 2 = -3$ (minimum value)

$f(-1, -1) = -1 + 3 + 1 - 2 = 1$

### Example 2: Newton's Method for a Simple Neural Network

**Problem**: Consider a single-neuron network with input $x$, weight $w$, bias $b$, and squared error loss $L(w, b) = \frac{1}{2}(\sigma(wx + b) - y)^2$, where $y$ is the target and $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid. For a single data point $(x=1, y=0)$, compute the Hessian of the loss at $(w=0, b=0)$ and perform one Newton update.

**Solution**:

Step 1: Write the output and loss. Let $z = w \cdot 1 + b = w + b$, $\sigma(z) = \frac{1}{1+e^{-z}}$.
$L(w, b) = \frac{1}{2}(\sigma(z) - 0)^2 = \frac{1}{2}\sigma(z)^2$.

At $(w, b) = (0, 0)$, $z = 0$, $\sigma(0) = 0.5$, $L = \frac{1}{2}(0.25) = 0.125$.

Step 2: Compute the gradient.

$\frac{\partial L}{\partial w} = \sigma(z) \cdot \sigma'(z) \cdot \frac{\partial z}{\partial w} = \sigma(z)\sigma'(z)$

$\frac{\partial L}{\partial b} = \sigma(z)\sigma'(z)$

Since $\sigma'(z) = \sigma(z)(1-\sigma(z))$, at $z=0$: $\sigma'(0) = 0.5 \cdot 0.5 = 0.25$.

So $\nabla L(0,0) = \begin{bmatrix} 0.5 \cdot 0.25 \\ 0.5 \cdot 0.25 \end{bmatrix} = \begin{bmatrix} 0.125 \\ 0.125 \end{bmatrix}$.

Step 3: Compute the Hessian.

$\frac{\partial^2 L}{\partial w^2} = \frac{\partial}{\partial w}[\sigma(z)\sigma'(z)] = \sigma'(z) \cdot \sigma'(z) \cdot \frac{\partial z}{\partial w} + \sigma(z) \cdot \sigma''(z) \cdot \frac{\partial z}{\partial w}$

$\sigma''(z) = \sigma'(z)(1-2\sigma(z)) = \sigma(z)(1-\sigma(z))(1-2\sigma(z))$

At $z=0$:
- $\sigma(0) = 0.5$, $\sigma'(0) = 0.25$, $\sigma''(0) = 0.25(1-1) = 0$

So $\frac{\partial^2 L}{\partial w^2} = 0.25 \cdot 0.25 + 0.5 \cdot 0 = 0.0625$

Similarly, $\frac{\partial^2 L}{\partial b^2} = 0.0625$ and $\frac{\partial^2 L}{\partial w \partial b} = 0.0625$.

$$
\mathbf{H} = \begin{bmatrix} 0.0625 & 0.0625 \\ 0.0625 & 0.0625 \end{bmatrix}
$$

Step 4: Newton update. $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta} - \mathbf{H}^{-1} \nabla L$.

But $\det(\mathbf{H}) = 0.0625 \cdot 0.0625 - 0.0625 \cdot 0.0625 = 0$! The Hessian is singular.

This happens because at initialization, $z=0$ for both inputs (since we used a single scalar $x=1$ shared by both parameters). The Hessian has rank 1 — both parameters affect the output identically at this point. Newton's method fails because $\mathbf{H}$ is not invertible.

In practice, we add a damping term: $\mathbf{H}_{\text{damped}} = \mathbf{H} + \lambda \mathbf{I}$, giving $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta} - (\mathbf{H} + \lambda \mathbf{I})^{-1} \nabla L$. This is the Levenberg-Marquardt method.

With $\lambda = 0.1$:
$$
\mathbf{H}_{\text{damped}} = \begin{bmatrix} 0.1625 & 0.0625 \\ 0.0625 & 0.1625 \end{bmatrix}, \quad \det = 0.1625^2 - 0.0625^2 \approx 0.0225
$$

$$
\mathbf{H}_{\text{damped}}^{-1} = \frac{1}{0.0225} \begin{bmatrix} 0.1625 & -0.0625 \\ -0.0625 & 0.1625 \end{bmatrix} \approx \begin{bmatrix} 7.22 & -2.78 \\ -2.78 & 7.22 \end{bmatrix}
$$

$$
\boldsymbol{\theta}_{\text{new}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 7.22 & -2.78 \\ -2.78 & 7.22 \end{bmatrix} \begin{bmatrix} 0.125 \\ 0.125 \end{bmatrix} = -\begin{bmatrix} 7.22 - 2.78 \\ -2.78 + 7.22 \end{bmatrix} \cdot 0.125 = \begin{bmatrix} -0.555 \\ -0.555 \end{bmatrix}
$$

The new loss: $z = -0.555 - 0.555 = -1.11$, $\sigma(-1.11) \approx 0.248$, $L_{\text{new}} = 0.5 \cdot 0.248^2 \approx 0.0308$, significantly lower than the original $0.125$.

### Example 3: Eigenvalues of the Hessian and Curvature

**Problem**: Consider the function $f(x, y) = x^2 + 4y^2 + 2xy$. Compute the Hessian, its eigenvalues and eigenvectors, and interpret the curvature.

**Solution**:

Step 1: Compute first-order partials.

$$
f_x = 2x + 2y, \quad f_y = 8y + 2x
$$

Step 2: Compute the Hessian.

$$
f_{xx} = 2, \quad f_{xy} = 2, \quad f_{yy} = 8
$$

$$
\mathbf{H} = \begin{bmatrix} 2 & 2 \\ 2 & 8 \end{bmatrix}
$$

Step 3: Find eigenvalues. Solve $\det(\mathbf{H} - \lambda \mathbf{I}) = 0$.

$$
\det\left(\begin{bmatrix} 2-\lambda & 2 \\ 2 & 8-\lambda \end{bmatrix}\right) = (2-\lambda)(8-\lambda) - 4 = 16 - 10\lambda + \lambda^2 - 4 = \lambda^2 - 10\lambda + 12 = 0
$$

$$
\lambda = \frac{10 \pm \sqrt{100 - 48}}{2} = \frac{10 \pm \sqrt{52}}{2} = 5 \pm \sqrt{13}
$$

So $\lambda_1 = 5 + \sqrt{13} \approx 8.606$, $\lambda_2 = 5 - \sqrt{13} \approx 1.394$.

Both eigenvalues are positive, so the Hessian is positive definite → $f$ is convex with a unique global minimum.

Step 4: Find eigenvectors.

For $\lambda_1 \approx 8.606$:

$$
\begin{bmatrix} 2-8.606 & 2 \\ 2 & 8-8.606 \end{bmatrix} \mathbf{v} = \begin{bmatrix} -6.606 & 2 \\ 2 & -0.606 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

From the first row: $-6.606 v_1 + 2 v_2 = 0 \implies v_2 = 3.303 v_1$. So $\mathbf{v}_1 \approx \begin{bmatrix} 1 \\ 3.303 \end{bmatrix}$ (normalised: $[0.29, 0.957]$).

For $\lambda_2 \approx 1.394$:

$$
\begin{bmatrix} 0.606 & 2 \\ 2 & 6.606 \end{bmatrix} \mathbf{v} = \mathbf{0}
$$

$0.606 v_1 + 2 v_2 = 0 \implies v_2 = -0.303 v_1$. So $\mathbf{v}_2 \approx \begin{bmatrix} 1 \\ -0.303 \end{bmatrix}$ (normalised: $[0.957, -0.29]$).

Step 5: Interpret.

The curvature in the direction $\mathbf{v}_1 \approx [0.29, 0.957]$ is $\lambda_1 \approx 8.606$ — very steep upward curvature. The curvature in the direction $\mathbf{v}_2 \approx [0.957, -0.29]$ is $\lambda_2 \approx 1.394$ — gentler upward curvature.

The condition number is $\kappa = 8.606 / 1.394 \approx 6.17$. Gradient descent would converge slowly in the $\mathbf{v}_1$ direction and faster in $\mathbf{v}_2$, leading to a zigzag path typical of ill-conditioned problems.

### Example 4: Hessian of Cross-Entropy Loss

**Problem**: For logistic regression with $K$ classes, the cross-entropy loss for a single data point $(\mathbf{x}, y)$ where $y$ is the true class (one-hot encoded) is:

$$
L(\mathbf{W}) = -\sum_{k=1}^K y_k \log p_k, \quad p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \mathbf{z} = \mathbf{W}\mathbf{x}
$$

Derive the Hessian of $L$ with respect to the logits $\mathbf{z}$.

**Solution**:

Step 1: Write the loss in terms of logits.

$L(\mathbf{z}) = -\sum_{k=1}^K y_k \log\left(\frac{e^{z_k}}{\sum_j e^{z_j}}\right) = -\sum_k y_k z_k + \log\left(\sum_j e^{z_j}\right)\sum_k y_k$

Since $\sum_k y_k = 1$ (one-hot), $L(\mathbf{z}) = -\sum_k y_k z_k + \log\left(\sum_j e^{z_j}\right)$.

Step 2: Compute the gradient with respect to $z_i$.

$$
\frac{\partial L}{\partial z_i} = -y_i + \frac{e^{z_i}}{\sum_j e^{z_j}} = -y_i + p_i
$$

So $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$, a convenient expression.

Step 3: Compute the Hessian entries.

$$
\frac{\partial^2 L}{\partial z_i \partial z_j} = \frac{\partial p_i}{\partial z_j}
$$

For $i = j$:
$$
\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \sum_j e^{z_j} - e^{z_i} \cdot e^{z_i}}{(\sum_j e^{z_j})^2} = p_i - p_i^2 = p_i(1-p_i)
$$

For $i \neq j$:
$$
\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot \sum_j e^{z_j} - e^{z_i} e^{z_j}}{(\sum_j e^{z_j})^2} = -p_i p_j
$$

Thus the Hessian is:

$$
\frac{\partial^2 L}{\partial z_i \partial z_j} = p_i(\delta_{ij} - p_j)
$$

where $\delta_{ij}$ is the Kronecker delta.

In matrix form:
$$
\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T
$$

This is the covariance matrix of a multinomial distribution with probabilities $\mathbf{p}$. It is positive semidefinite (and actually singular since $\mathbf{H}\mathbf{1} = \mathbf{0}$ because probabilities sum to 1). This Hessian appears in the Fisher Information Matrix for logistic regression and is fundamental to natural gradient optimisation for classification.

## Visual Interpretation

The Hessian's geometric meaning is best understood through the concept of curvature. For a 1D function $f(x)$, the second derivative $f''(x)$ measures curvature — whether the function bends upward ($f'' > 0$), downward ($f'' < 0$), or is flat ($f'' = 0$).

For a 2D function $f(x, y)$, the Hessian describes the curvature in every direction. If we take a unit vector $\mathbf{v}$, the second directional derivative in that direction is $\mathbf{v}^T \mathbf{H} \mathbf{v}$. This is the curvature of the slice of the surface $z = f(x, y)$ cut by the plane containing the vertical direction and $\mathbf{v}$.

Visualise the loss landscape of a neural network as a high-dimensional surface. At any point:
- The gradient is the direction of steepest ascent
- The Hessian eigenvalues are the principal curvatures
- The eigenvectors are the directions of those curvatures
- Positive curvature means the surface bends upward; negative curvature means downward

A local minimum is like a bowl: all curvatures are positive (bowl opens upward). A local maximum is like an inverted bowl: all curvatures are negative. A saddle point is like a mountain pass: positive curvature in one direction and negative in another.

The magnitude of curvature matters: a steep, narrow valley (large eigenvalue) means gradient descent will oscillate or diverge if the learning rate is too large. A gentle, wide valley (small eigenvalue) means gradient descent will make slow progress. This is why normalising the Hessian (preconditioning) can dramatically accelerate convergence.

For visualising loss landscapes in deep learning, researchers often plot 2D slices defined by two Hessian eigenvectors. These visualisations reveal the characteristic "mountain range" structure of neural network loss landscapes, with interconnected minima connected by low-loss valleys.

## Common Mistakes

1. **Assuming the Hessian is always positive definite**: Many functions have regions where the Hessian is indefinite (mixed eigenvalues) or singular (zero eigenvalues). Not all critical points are minima — saddle points are common in high dimensions.

2. **Confusing the Hessian with the Jacobian**: The Jacobian ($m \times n$ matrix of first derivatives) and Hessian ($n \times n$ matrix of second derivatives) serve different purposes. The Hessian is the Jacobian of the gradient.

3. **Forgetting symmetry**: While the Hessian should be symmetric by Clairaut's theorem (if mixed partials are continuous), numerical errors in computation can break symmetry. Always symmetrise when using the Hessian numerically.

4. **Ignoring the quadratic term in Taylor expansions**: The first-order expansion (gradient only) is insufficient for understanding local behaviour near critical points. The Hessian is essential for determining the type of critical point.

5. **Inverting a near-singular Hessian**: In high-dimensional optimisation, the Hessian is often ill-conditioned or singular. Direct inversion amplifies noise in near-zero eigenvalue directions. Regularisation (adding $\lambda \mathbf{I}$) is essential.

6. **Assuming gradient descent is immune to Hessian effects**: Even without explicitly computing the Hessian, its condition number determines gradient descent convergence rate. A poorly conditioned problem will converge slowly regardless of learning rate tuning.

7. **Applying the second-derivative test incorrectly**: For a 2D function, $f_{xx} > 0$ and $\det(\mathbf{H}) > 0$ indicates a local minimum, but $f_{xx} < 0$ and $\det(\mathbf{H}) > 0$ indicates a local maximum. If $\det(\mathbf{H}) < 0$, it is a saddle point regardless of $f_{xx}$.

8. **Forgetting that the Hessian depends on the point**: The Hessian is a function of position, not a constant. The curvature of the loss landscape changes as training progresses. The Hessian at initialization is very different from the Hessian at convergence.

9. **Assuming the Hessian diagonal is sufficient**: Diagonal approximations (like in AdaGrad) capture only the axis-aligned curvature and miss off-diagonal interactions. For correlated parameters, the full Hessian structure is important.

10. **Confusing the Hessian of the loss with the Fisher Information Matrix**: While related, they are not the same. The FIM is $\mathbb{E}[\nabla \log p \cdot \nabla \log p^T]$, while the Hessian of the negative log-likelihood is $\mathbb{E}[-\nabla^2 \log p]$. They coincide for exponential families but differ in general.

## Interview Questions

### Beginner

1. **Q**: What is the Hessian matrix?
   **A**: The Hessian is an $n \times n$ symmetric matrix of all second-order partial derivatives of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$.

2. **Q**: How can you determine if a critical point is a local minimum using the Hessian?
   **A**: A critical point where $\nabla f = 0$ is a local minimum if the Hessian is positive definite (all eigenvalues $> 0$).

3. **Q**: What does it mean if the Hessian has both positive and negative eigenvalues at a critical point?
   **A**: The point is a saddle point — a minimum in some directions and a maximum in others.

4. **Q**: Compute the Hessian of $f(x, y) = x^2 + 3y^2$.
   **A**: $\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & 6 \end{bmatrix}$, positive definite everywhere.

5. **Q**: What is the condition number of the Hessian and why does it matter for optimisation?
   **A**: The condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ measures the ratio of maximum to minimum curvature. A high condition number slows gradient descent convergence.

### Intermediate

1. **Q**: Derive the second-order Taylor expansion of $f$ around $\mathbf{x}_0$ using the Hessian.
   **A**: $f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T \mathbf{H}(\mathbf{x}_0) \Delta\mathbf{x} + O(\|\Delta\mathbf{x}\|^3)$, where $\Delta\mathbf{x} = \mathbf{x} - \mathbf{x}_0$.

2. **Q**: In the context of neural network loss landscapes, what distinguishes a "sharp" minimum from a "flat" minimum, and why does flatness correlate with better generalisation?
   **A**: A sharp minimum has large Hessian eigenvalues (high curvature in some directions), so small parameter changes cause large loss increases. A flat minimum has small eigenvalues. Flat minima generalise better because the loss doesn't change much when parameters shift slightly, making the solution more robust to the train-test distribution shift.

3. **Q**: Why is forming the full Hessian $O(n^2)$ and inverting it $O(n^3)$ prohibitive for deep learning models with millions of parameters?
   **A**: For $n = 10^6$ parameters, the Hessian has $10^{12}$ entries, requiring 4 TB of memory in single precision. Inversion requires $10^{18}$ operations. Even storing the Hessian is impossible; only Hessian-vector products (computed in $O(n)$ time via autodiff) are feasible.

4. **Q**: How does the Hessian of the cross-entropy loss for logistic regression relate to the Fisher Information Matrix?
   **A**: The Hessian is $\mathbf{H} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$, which is exactly the Fisher Information Matrix for the multinomial distribution. Natural gradient descent for classification uses this structure to define a geometry-aware update.

5. **Q**: For the function $f(x, y) = x^2 - y^2 + 2xy$, find and classify the critical point.
   **A**: $\nabla f = (2x+2y, -2y+2x) = (0,0) \implies x=0, y=0$. $\mathbf{H} = \begin{bmatrix} 2 & 2 \\ 2 & -2 \end{bmatrix}$, $\det = -4 - 4 = -8 < 0$, so $(0,0)$ is a saddle point.

### Advanced

1. **Q**: Derive the exact relationship between the Hessian of the expected loss and the Fisher Information Matrix for a probabilistic model $p(\mathbf{x}|\boldsymbol{\theta})$. Under what conditions are they equal?
   **A**: The Hessian of the negative log-likelihood is $\mathbf{H}_{NLL} = -\frac{1}{N}\sum_i \nabla^2 \log p(\mathbf{x}_i|\boldsymbol{\theta})$. The Fisher Information is $\mathbf{F} = \mathbb{E}_{p(\mathbf{x}|\boldsymbol{\theta})}[ \nabla \log p(\mathbf{x}|\boldsymbol{\theta}) \nabla \log p(\mathbf{x}|\boldsymbol{\theta})^T]$. Using the identity $\mathbb{E}[\nabla^2 \log p] = -\mathbb{E}[(\nabla \log p)(\nabla \log p)^T]$ (the Fisher identity), they are equal in expectation when the model is correctly specified. However, $\mathbf{H}_{NLL}$ is an empirical average over observed data, while $\mathbf{F}$ is an expectation under the model. In practice, the empirical Fisher $\hat{\mathbf{F}} = \frac{1}{N}\sum_i \nabla \log p_i \nabla \log p_i^T$ is a Gauss-Newton approximation to the Hessian.

2. **Q**: In Sharpness-Aware Minimisation (SAM), the objective is $\min_{\boldsymbol{\theta}} \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L(\boldsymbol{\theta} + \boldsymbol{\epsilon})$. Show that for small $\rho$, the SAM objective approximates $L(\boldsymbol{\theta}) + \frac{\rho}{2} \|\nabla L(\boldsymbol{\theta})\|_2 + \frac{\rho^2}{2} \lambda_{\max}(\mathbf{H}(\boldsymbol{\theta})) + O(\rho^3)$, where $\lambda_{\max}$ is the top eigenvalue of the Hessian. Explain how this penalises sharp minima.
   **A**: Using a second-order Taylor expansion around $\boldsymbol{\theta}$:
   $$
   L(\boldsymbol{\theta} + \boldsymbol{\epsilon}) \approx L(\boldsymbol{\theta}) + \nabla L^T \boldsymbol{\epsilon} + \frac{1}{2} \boldsymbol{\epsilon}^T \mathbf{H} \boldsymbol{\epsilon}
   $$
   For $\|\boldsymbol{\epsilon}\|_2 \leq \rho$, the worst-case $\boldsymbol{\epsilon}$ that maximises this is $\boldsymbol{\epsilon} = \rho \frac{\nabla L}{\|\nabla L\|}$ to first order, and $\boldsymbol{\epsilon} = \rho \mathbf{v}_{\max}$ (the top eigenvector of $\mathbf{H}$) to second order when $\nabla L = 0$ (near a minimum). At a critical point ($\nabla L = 0$), the maximum perturbation direction is the top eigenvector, giving value $\frac{\rho^2}{2} \lambda_{\max}(\mathbf{H})$. So the SAM objective penalises large curvature (sharp minima) by adding $\frac{\rho^2}{2} \lambda_{\max}(\mathbf{H})$ to the loss.

3. **Q**: Explain the concept of "Hessian-free optimisation" and describe how to compute a Hessian-vector product using automatic differentiation. Show that the cost of computing $\mathbf{H}\mathbf{v}$ is roughly the same as computing $\nabla L$.
   **A**: Hessian-free optimisation (Martens, 2010) uses the conjugate gradient method to solve $\mathbf{H}\mathbf{p} = -\nabla L$ without forming $\mathbf{H}$. Each CG iteration requires one HVP. The key insight is that $\mathbf{H}\mathbf{v} = \nabla_{\boldsymbol{\theta}} ( \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})^T \mathbf{v} )$. Computing $\nabla L$ costs one forward and one backward pass. Let $g = \nabla L$. Then $g^T \mathbf{v}$ is a scalar. Taking the gradient of this scalar (another backward pass) gives $\mathbf{H}\mathbf{v}$, all in $O(n)$ time. In reverse-mode AD frameworks, this is implemented as:
   ```
   def hvp(loss, params, v):
       grad = torch.autograd.grad(loss, params, create_graph=True)
       grad_v = sum(g.dot(v_i) for g, v_i in zip(grad, v))
       hvp = torch.autograd.grad(grad_v, params)
       return hvp
   ```
   The total cost is roughly 2-3x a forward-backward pass, regardless of $n$, making HVPs tractable for models with millions of parameters.

## Practice Problems

### Easy

1. Compute the Hessian matrix of $f(x, y) = 3x^2 + 4xy + 2y^2$.

2. For $f(x, y) = e^{x+y}$, find $\mathbf{H}(f)$ and verify it is positive definite everywhere.

3. Find the Hessian of $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$.

4. Compute the Hessian of $f(x, y, z) = \ln(1 + x^2 + y^2 + z^2)$.

5. For $f(x, y) = x^3 - 3xy^2$, find the Hessian at $(0, 0)$ and $(1, 0)$.

### Medium

1. Classify all critical points of $f(x, y) = x^4 + y^4 - 4xy$.

2. The loss for linear regression with $N$ data points is $L(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^N (\mathbf{w}^T \mathbf{x}_i - y_i)^2$. Show that the Hessian is $\frac{1}{N} \mathbf{X}^T \mathbf{X}$ where $\mathbf{X}$ is the design matrix, and discuss its properties.

3. For $f(x, y) = x^2 e^y$, compute the Hessian and evaluate it at $(1, 0)$. Determine if the function is convex near this point.

4. A neural network's loss function $L(\theta_1, \theta_2) = (\theta_1^2 - 4)^2 + \theta_2^2$ has multiple local minima. Find all critical points and classify them using the Hessian.

5. Show that the Hessian of a quadratic function $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$ is constant and equals $\mathbf{A}$.

### Hard

1. For a deep neural network with ReLU activations, the Hessian is piecewise constant within linear regions. Prove that the Hessian of a ReLU network (with respect to parameters) has rank at most the width of the network, and explain the implications for optimisation.

2. Consider the function $f(x, y) = \frac{1}{2}(x^2 + \epsilon y^2)$ where $\epsilon$ is very small. Compute the condition number of the Hessian. How many iterations of gradient descent with optimal learning rate would be needed to reduce the error by a factor of $10^{-6}$? Compare this to Newton's method.

3. In the context of the Neural Tangent Kernel (NTK), the NTK matrix is $\Theta(\mathbf{x}_i, \mathbf{x}_j) = \nabla_{\boldsymbol{\theta}} f(\mathbf{x}_i)^T \nabla_{\boldsymbol{\theta}} f(\mathbf{x}_j)$. Show that the Hessian of the squared error loss for a neural network can be expressed in terms of the NTK and second-order terms involving $\nabla^2_{\boldsymbol{\theta}} f(\mathbf{x}_i)$. Under what conditions does the Hessian simplify to the NTK?

## Solutions

### Easy Solutions

**Solution 1**: $f_{xx} = 6$, $f_{xy} = 4$, $f_{yy} = 4$. $\mathbf{H} = \begin{bmatrix} 6 & 4 \\ 4 & 4 \end{bmatrix}$.

**Solution 2**: $f_x = e^{x+y}$, $f_y = e^{x+y}$. All second derivatives equal $e^{x+y}$. $\mathbf{H} = \begin{bmatrix} e^{x+y} & e^{x+y} \\ e^{x+y} & e^{x+y} \end{bmatrix}$. Eigenvalues: $2e^{x+y}$ and $0$. Positive semidefinite (one eigenvalue zero). The function is convex but not strictly convex.

**Solution 3**: $\nabla f = \mathbf{A}^T (\mathbf{A}\mathbf{x} - \mathbf{b})$. $\mathbf{H} = \mathbf{A}^T \mathbf{A}$, constant. Positive semidefinite.

**Solution 4**: 
$f_x = \frac{2x}{1+x^2+y^2+z^2}$, $f_{xx} = \frac{2(1+x^2+y^2+z^2) - 2x \cdot 2x}{(1+x^2+y^2+z^2)^2} = \frac{2(1 - x^2 + y^2 + z^2)}{(1+x^2+y^2+z^2)^2}$.
$f_{xy} = \frac{-4xy}{(1+x^2+y^2+z^2)^2}$. Similar for other mixed partials.

**Solution 5**: $f_{xx} = 6x$, $f_{xy} = -6y$, $f_{yy} = -6x$.
At $(0,0)$: $\mathbf{H} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$ (zero matrix, degenerate).
At $(1,0)$: $\mathbf{H} = \begin{bmatrix} 6 & 0 \\ 0 & -6 \end{bmatrix}$ (saddle point).

### Medium Solutions

**Solution 1**: $\nabla f = (4x^3 - 4y, 4y^3 - 4x) = (0,0)$.
Critical points: $(0,0)$, $(1,1)$, $(-1,-1)$.
$\mathbf{H} = \begin{bmatrix} 12x^2 & -4 \\ -4 & 12y^2 \end{bmatrix}$.
At $(0,0)$: $\mathbf{H} = \begin{bmatrix} 0 & -4 \\ -4 & 0 \end{bmatrix}$, $\det = -16 < 0$, saddle.
At $(1,1)$: $\mathbf{H} = \begin{bmatrix} 12 & -4 \\ -4 & 12 \end{bmatrix}$, $\det = 144 - 16 = 128 > 0$, $f_{xx} = 12 > 0$, local min.
At $(-1,-1)$: same as $(1,1)$, local min.

**Solution 2**: $L = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ where $\mathbf{X}$ is $N \times n$. $\nabla L = \frac{1}{N} \mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$. $\mathbf{H} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$. It is positive semidefinite (positive definite if $\mathbf{X}$ has full column rank). Eigenvalues are the squared singular values of $\mathbf{X}/\sqrt{N}$.

**Solution 3**: $f_x = 2xe^y$, $f_y = x^2 e^y$.
$f_{xx} = 2e^y$, $f_{xy} = 2xe^y$, $f_{yy} = x^2 e^y$.
At $(1,0)$: $\mathbf{H} = \begin{bmatrix} 2 & 2 \\ 2 & 1 \end{bmatrix}$, $\det = 2 - 4 = -2 < 0$. Indefinite — not convex (saddle point).

**Solution 4**: $\nabla L = (4\theta_1(\theta_1^2 - 4), 2\theta_2) = (0,0)$.
Critical points: $(2,0)$, $(-2,0)$, $(0,0)$.
$\mathbf{H} = \begin{bmatrix} 12\theta_1^2 - 16 & 0 \\ 0 & 2 \end{bmatrix}$.
At $(\pm2,0)$: $\mathbf{H} = \begin{bmatrix} 32 & 0 \\ 0 & 2 \end{bmatrix}$, pos def, local minima.
At $(0,0)$: $\mathbf{H} = \begin{bmatrix} -16 & 0 \\ 0 & 2 \end{bmatrix}$, indefinite, saddle point.

**Solution 5**: $\nabla f = \frac{1}{2}(\mathbf{A} + \mathbf{A}^T)\mathbf{x} + \mathbf{b}$. If $\mathbf{A}$ is symmetric, $\nabla f = \mathbf{A}\mathbf{x} + \mathbf{b}$. Then $\mathbf{H} = \mathbf{A}$, constant.

### Hard Solutions

**Solution 1**: A ReLU network computes a piecewise linear function, so its output is linear within each region. The Hessian of the loss $\frac{1}{2}\|f(\mathbf{x};\boldsymbol{\theta}) - y\|^2$ is $\nabla f \nabla f^T + (f-y)\nabla^2 f$. Within a linear region, $\nabla^2 f = 0$, so $\mathbf{H} = \nabla f \nabla f^T$, which is rank-1. This means the Hessian has at most rank equal to the number of active linear regions or the network width. Implication: the loss landscape is extremely degenerate with many zero-curvature directions, making second-order methods challenging.

**Solution 2**: $\mathbf{H} = \begin{bmatrix} 1 & 0 \\ 0 & \epsilon \end{bmatrix}$, with eigenvalues $\lambda_{\max} = 1$, $\lambda_{\min} = \epsilon$. Condition number $\kappa = 1/\epsilon$.
Gradient descent with optimal learning rate $\eta^* = \frac{2}{\lambda_{\max} + \lambda_{\min}} = \frac{2}{1+\epsilon}$ converges with factor $\rho = \frac{\kappa-1}{\kappa+1} = \frac{1-\epsilon}{1+\epsilon} \approx 1 - 2\epsilon$.
To reduce error by $10^{-6}$, we need $\rho^k \leq 10^{-6}$, so $k \geq \frac{-6\ln 10}{\ln \rho} \approx \frac{13.82}{2\epsilon} = 6.91/\epsilon$.
For $\epsilon = 0.01$, this is ~691 iterations. Newton's method achieves $\rho = 0$ (one-step convergence) for quadratics.

**Solution 3**: For squared error $L = \frac{1}{2}\sum_i (f(\mathbf{x}_i) - y_i)^2$:
$\nabla_{\boldsymbol{\theta}} L = \sum_i (f_i - y_i) \nabla_{\boldsymbol{\theta}} f_i$
$\nabla^2_{\boldsymbol{\theta}} L = \sum_i \nabla_{\boldsymbol{\theta}} f_i \nabla_{\boldsymbol{\theta}} f_i^T + \sum_i (f_i - y_i) \nabla^2_{\boldsymbol{\theta}} f_i$
The first term is the empirical NTK. The second term involves the Hessian of the network outputs. In the NTK regime (infinite width), the network is linearised around initialisation, and the second term vanishes. The Hessian simplifies to the NTK, which remains constant during training in the infinite-width limit.

## Related Concepts

- **Jacobian Matrix**: The Hessian is the Jacobian of the gradient (MATH-059)
- **Gradient**: First derivative vector; the Hessian gives second-order information
- **Second Derivative Test**: Classification of critical points using the Hessian
- **Convex Optimisation**: The Hessian's positive semidefiniteness defines convexity
- **Taylor Series**: The Hessian appears in the second-order Taylor expansion
- **Eigenvalues and Eigenvectors**: Diagonalising the Hessian reveals principal curvatures
- **Newton's Method**: Second-order optimisation using the Hessian inverse
- **Condition Number**: Ratio of extreme Hessian eigenvalues, governing optimisation speed

## Next Concepts

- **Multiple Integrals**: Integration techniques in higher dimensions (MATH-064)
- **Optimisation Theory**: Advanced gradient-based and second-order optimisation
- **Fisher Information Matrix**: Related to the Hessian of the log-likelihood
- **Natural Gradient Descent**: Optimisation using the Fisher-Riemannian metric
- **Loss Landscape Analysis**: Using Hessian eigenvalues to study neural network training
- **Differential Geometry**: The Hessian as the second fundamental form

## Summary

The Hessian matrix $\mathbf{H}(f)$ is a symmetric $n \times n$ matrix of second-order partial derivatives of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$. It captures the local curvature of $f$, with its eigenvalues representing principal curvatures and its eigenvectors indicating curvature directions. The Hessian is essential for classifying critical points, understanding convexity, and designing second-order optimisation algorithms.

In machine learning, the Hessian of the loss function governs optimisation dynamics. Its condition number determines gradient descent convergence rates, while its eigenvalue structure reveals properties of the loss landscape such as flat vs. sharp minima. Advanced techniques like Hessian-free optimisation and Sharpness-Aware Minimisation leverage Hessian information without explicitly forming the full matrix, making them practical for large-scale deep learning.

The Hessian bridges first-order and second-order analysis, providing a deeper understanding of function behaviour beyond what the gradient alone can offer.

## Key Takeaways

- The Hessian is the matrix of all second-order partial derivatives of a scalar function
- It is symmetric when mixed partials are continuous (Clairaut's theorem)
- The Hessian determines the nature of critical points: positive definite → minimum, negative definite → maximum, indefinite → saddle
- The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ governs gradient descent convergence
- Newton's method uses $\mathbf{H}^{-1}$ for second-order updates but is $O(n^3)$
- Hessian-vector products can be computed in $O(n)$ without forming the full matrix
- Flat minima (small Hessian eigenvalues) correlate with better generalisation
- The loss landscape of neural networks reveals saddle points, degenerate valleys, and interconnected minima
- SAM explicitly penalises large Hessian eigenvalues to find flatter minima
- The Hessian appears in the second-order Taylor expansion as the quadratic term
