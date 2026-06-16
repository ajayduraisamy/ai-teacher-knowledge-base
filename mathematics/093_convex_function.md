# Concept: Convex Function

## Concept ID

MATH-093

## Difficulty

Intermediate

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Define convex functions formally using the Jensen inequality criterion
- Verify convexity of univariate and multivariate functions using first- and second-order conditions
- Distinguish between convex, strictly convex, and strongly convex functions
- Apply Jensen's inequality to derive fundamental bounds in probability and information theory
- Identify convex loss functions and their role in guaranteeing global optimality in machine learning
- Recognize non-convex landscapes in deep learning and explain why local minima still work

## Prerequisites

- Basic calculus: differentiation, partial derivatives, gradient, Hessian matrix
- Linear algebra: vector spaces, norms, inner products
- Set theory: affine sets, convex sets, line segments
- Single-variable optimization: stationary points, second derivative test

## Definition

A function $f: \mathbb{R}^n \to \mathbb{R}$ defined on a convex domain $\mathcal{D} \subseteq \mathbb{R}^n$ is **convex** if, for all $x, y \in \mathcal{D}$ and for all $\theta \in [0, 1]$, the following inequality holds:

$$
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
$$

The function is **strictly convex** if the inequality is strict for all $x \neq y$ and $\theta \in (0, 1)$. The function is **strongly convex** with parameter $\mu > 0$ if $f(x) - \frac{\mu}{2} \|x\|^2$ is convex.

Equivalently, for a twice-differentiable function, convexity is equivalent to the Hessian matrix being positive semidefinite everywhere on the domain:

$$
\nabla^2 f(x) \succeq 0 \quad \forall x \in \mathcal{D}
$$

## Intuition

A convex function describes a bowl-shaped surface. The line segment joining any two points on the graph lies above or on the graph itself. This "bowl shape" guarantees that any local minimum is automatically a global minimum—there are no hidden valleys or deceptive basins.

Think of a convex function as a landscape where water always flows to a single lowest point. If you place a marble anywhere on the surface, it will eventually settle at the bottom. This property is the mathematical foundation of reliable optimization: algorithms can descend the surface without fear of getting trapped in suboptimal hollows.

For twice-differentiable convex functions, the curvature is everywhere non-negative. The function curves upward like $x^2$, never downward like $-x^2$. This curvature constraint (positive semidefinite Hessian) means the function is at least as curved as a flat plane.

## Why This Concept Matters

Convex functions are the bedrock of optimization theory. When an optimization problem involves minimizing a convex function over a convex set, every local minimum is a global minimum, and the solution can be found efficiently and reliably. This guarantee is why convex optimization is considered "solved" in a practical sense—there exist algorithms with polynomial-time convergence guarantees.

In machine learning, convex loss functions like mean squared error and logistic loss ensure that the training objective has a unique, globally optimal solution when the model is linear. This property is why linear regression and logistic regression are so well-behaved.

However, modern deep learning relies on non-convex neural network objectives. Understanding convexity helps practitioners appreciate why deep learning optimization is fundamentally harder—and why techniques like adaptive optimizers, careful initialization, and learning rate scheduling are necessary to navigate the non-convex landscape.

## Historical Background

The concept of convexity dates back to ancient geometry. Archimedes studied convex curves in his work on the sphere and cylinder. However, the formal mathematical treatment of convex functions began in the late 19th and early 20th centuries.

Otto Hölder and Otto Stolz introduced early definitions in the 1880s. Johan Jensen, a Danish mathematician, published his seminal paper on convex functions in 1906, establishing the inequality that now bears his name. Jensen's work unified many disparate inequalities in analysis and probability under a single framework.

The mid-20th century saw convex analysis flourish through the work of Werner Fenchel, Jean-Jacques Moreau, and R. Tyrrell Rockafellar. Rockafellar's 1970 book "Convex Analysis" remains the definitive reference. The field gained enormous practical importance in the 1990s when Stephen Boyd and Lieven Vandenberghe's work on convex optimization showed that a wide class of engineering problems could be formulated and solved as convex programs.

In machine learning, convexity played a central role in the development of support vector machines (1990s) and remains important for understanding generalization and optimization guarantees.

## Real World Examples

- **Economics**: Utility functions in microeconomics are often assumed to be concave (the negative of convex), reflecting diminishing marginal returns. Cost functions exhibiting economies of scale are convex.
- **Engineering**: The energy of a physical system is typically a convex function of the state variables, ensuring stable equilibrium. Spring potential energy $E(x) = \frac{1}{2}kx^2$ is convex.
- **Finance**: Portfolio optimization uses convex risk measures. The variance of a portfolio is a convex function of the asset weights, making mean-variance optimization a convex problem.
- **Control Theory**: Quadratic cost functions in linear-quadratic regulators (LQR) are convex, guaranteeing optimal control laws exist.
- **Signal Processing**: The $\ell_1$ norm used in compressed sensing is convex, enabling efficient recovery of sparse signals via convex optimization.

## AI/ML Relevance

Convex functions are central to machine learning in two fundamental ways.

**Convex loss functions** form the backbone of classical supervised learning. The mean squared error (MSE) $L(y, \hat{y}) = (y - \hat{y})^2$ is convex in the prediction $\hat{y}$. The logistic loss $L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})$ is convex in $\hat{y}$. When combined with a linear model $\hat{y} = w^T x + b$, the resulting objective function is convex in the parameters $w$ and $b$. This convexity guarantees that gradient descent converges to the global minimum regardless of initialization.

**Non-convex objectives in deep learning**: Neural networks with hidden layers produce loss landscapes that are highly non-convex. The composition of linear transformations with non-linear activation functions breaks convexity. However, recent research shows that despite the presence of many local minima, most are of similar quality to the global minimum. Saddle points, not local minima, are the bigger challenge in high-dimensional non-convex optimization.

Convex analysis also appears in:
- **Regularization**: $\ell_1$ (LASSO) and $\ell_2$ (ridge) regularizers are convex, preserving convexity of the overall objective.
- **Support vector machines**: The hinge loss is convex, and the SVM training problem is convex quadratic programming.
- **Adversarial robustness**: Convex relaxation is used to certify robustness of neural networks.
- **Optimization theory**: Convergence proofs for gradient-based methods rely on convexity or strong convexity assumptions.

## Mathematical Explanation

### First-Order Condition

A differentiable function $f$ is convex if and only if for all $x, y \in \mathcal{D}$:

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x)
$$

This inequality says that the first-order Taylor approximation at any point $x$ is a global underestimator of the function. The function lies above its tangent line (or tangent plane in higher dimensions).

### Second-Order Condition

A twice-differentiable function $f$ is convex if and only if its Hessian matrix is positive semidefinite everywhere:

$$
\nabla^2 f(x) \succeq 0 \quad \forall x \in \mathcal{D}
$$

For a univariate function, this reduces to $f''(x) \geq 0$ for all $x$.

### Epigraph Characterization

A function $f$ is convex if and only if its epigraph $\{(x, t) \mid f(x) \leq t\}$ is a convex set. This geometric characterization connects convex functions to convex sets.

### Operations that Preserve Convexity

- **Non-negative weighted sums**: If $f_1, \ldots, f_m$ are convex and $w_i \geq 0$, then $\sum w_i f_i$ is convex.
- **Composition with affine map**: If $f$ is convex, then $f(Ax + b)$ is convex.
- **Pointwise maximum**: If $f_1, \ldots, f_m$ are convex, then $\max_i f_i(x)$ is convex.
- **Composition with convex non-decreasing function**: If $g$ is convex and non-decreasing, and $h$ is convex, then $g(h(x))$ is convex.

## Formula(s)

**Jensen's Inequality**:

$$
f\left(\sum_{i=1}^n \theta_i x_i\right) \leq \sum_{i=1}^n \theta_i f(x_i)
$$

where $\theta_i \geq 0$ and $\sum \theta_i = 1$.

**Strong Convexity** (with parameter $\mu > 0$):

$$
f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2} \|y - x\|^2
$$

**Convex Conjugate (Fenchel Transform)**:

$$
f^*(y) = \sup_{x \in \mathcal{D}} \left( y^T x - f(x) \right)
$$

## Properties

1. **Global minimum property**: Any local minimum of a convex function is a global minimum.
2. **Uniqueness under strict convexity**: If $f$ is strictly convex, the global minimum is unique.
3. **Subgradient existence**: Convex functions have at least one subgradient at every point in the interior of their domain.
4. **Continuity**: Convex functions on $\mathbb{R}^n$ are continuous on the interior of their domain.
5. **Lipschitz property**: Convex functions with bounded subgradients on a compact set are Lipschitz continuous.
6. **Level sets**: All sublevel sets $\{x \mid f(x) \leq c\}$ of a convex function are convex sets.
7. **Second derivative non-negative**: For univariate convex functions, $f''(x) \geq 0$ everywhere.
8. **Monotone gradient**: The gradient of a differentiable convex function is a monotone operator: $(\nabla f(x) - \nabla f(y))^T (x - y) \geq 0$.

## Step-by-Step Worked Examples

### Example 1: Verifying Convexity of $f(x) = x^2$

**Problem**: Show that $f(x) = x^2$ is convex using the definition.

**Solution**:

Let $x, y \in \mathbb{R}$ and $\theta \in [0, 1]$. Compute:

$$
f(\theta x + (1 - \theta) y) = (\theta x + (1 - \theta) y)^2
$$

$$
= \theta^2 x^2 + 2\theta(1 - \theta)xy + (1 - \theta)^2 y^2
$$

Now compute the right side of the convexity inequality:

$$
\theta f(x) + (1 - \theta) f(y) = \theta x^2 + (1 - \theta) y^2
$$

We need to verify:

$$
\theta^2 x^2 + 2\theta(1 - \theta)xy + (1 - \theta)^2 y^2 \leq \theta x^2 + (1 - \theta) y^2
$$

Rearranging:

$$
\theta x^2 + (1 - \theta) y^2 - \theta^2 x^2 - 2\theta(1 - \theta)xy - (1 - \theta)^2 y^2 \geq 0
$$

$$
\theta(1 - \theta)x^2 + \theta(1 - \theta)y^2 - 2\theta(1 - \theta)xy \geq 0
$$

$$
\theta(1 - \theta)(x^2 - 2xy + y^2) \geq 0
$$

$$
\theta(1 - \theta)(x - y)^2 \geq 0
$$

Since $\theta \in [0, 1]$ and $(x - y)^2 \geq 0$, the inequality holds. Thus $f(x) = x^2$ is convex.

### Example 2: Second Derivative Test for $f(x) = e^x$

**Problem**: Show $f(x) = e^x$ is convex using the second derivative test.

**Solution**:

Compute the first and second derivatives:

$$
f'(x) = e^x
$$

$$
f''(x) = e^x
$$

For all $x \in \mathbb{R}$, $e^x > 0$. Therefore $f''(x) \geq 0$ for all $x$, confirming that $f(x) = e^x$ is convex on $\mathbb{R}$.

The function grows exponentially, yet its curvature remains positive everywhere. This demonstrates that convexity does not impose boundedness.

### Example 3: Verifying Convexity of $f(x, y) = x^2 + y^2$

**Problem**: Show that $f(x, y) = x^2 + y^2$ is convex using the Hessian test.

**Solution**:

Compute the gradient:

$$
\nabla f(x, y) = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
$$

Compute the Hessian:

$$
\nabla^2 f(x, y) = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
$$

The Hessian is constant and equal to $2I$. For any vector $v = [v_1, v_2]^T$:

$$
v^T \nabla^2 f \, v = v^T (2I) v = 2(v_1^2 + v_2^2) \geq 0
$$

The Hessian is positive definite (eigenvalues are both 2), so the function is strictly convex. This is a quadratic bowl with a unique global minimum at $(0, 0)$.

### Example 4: Non-convex Function $f(x) = x^3$

**Problem**: Show that $f(x) = x^3$ is not convex.

**Solution**:

Compute derivatives:

$$
f'(x) = 3x^2
$$

$$
f''(x) = 6x
$$

For $x < 0$, $f''(x) < 0$, violating the second-order condition. Therefore $f(x) = x^3$ is not convex.

To see this geometrically: take $x = -2, y = 2, \theta = 0.5$:

$$
f(0.5(-2) + 0.5(2)) = f(0) = 0
$$

$$
0.5 f(-2) + 0.5 f(2) = 0.5(-8) + 0.5(8) = 0
$$

This gives equality. But try $x = -1, y = 2, \theta = 0.5$:

$$
f(0.5(-1) + 0.5(2)) = f(0.5) = 0.125
$$

$$
0.5 f(-1) + 0.5 f(2) = 0.5(-1) + 0.5(8) = 3.5
$$

The inequality $0.125 \leq 3.5$ holds, but for $x = -2, y = 1, \theta = 0.5$:

$$
f(-0.5) = -0.125
$$

$$
0.5 f(-2) + 0.5 f(1) = 0.5(-8) + 0.5(1) = -3.5
$$

Now $-0.125 \leq -3.5$ is false. The function is not convex.

### Example 5: Jensen's Inequality in Probability

**Problem**: Let $X$ be a random variable with finite expectation. Use Jensen's inequality to show that $E[X^2] \geq (E[X])^2$.

**Solution**:

The function $f(x) = x^2$ is convex (shown in Example 1). Applying Jensen's inequality in its probabilistic form:

$$
f(E[X]) \leq E[f(X)]
$$

Substituting $f(x) = x^2$:

$$
(E[X])^2 \leq E[X^2]
$$

This is the well-known fact that variance is non-negative: $\text{Var}(X) = E[X^2] - (E[X])^2 \geq 0$.

## Visual Interpretation

Consider the graph of a convex function $f(x) = x^2$. Draw any two points on the curve, say $(-1, 1)$ and $(2, 4)$. The line segment connecting them lies entirely above the curve between these points. This is the geometric meaning of the convexity inequality.

For a convex function, the tangent line at any point lies below the function. At $x = -1$ for $f(x) = x^2$, the tangent line is $y = -2x - 1$. Since $x^2 \geq -2x - 1$ for all $x$ (equality only at $x = -1$), the tangent is a global lower bound.

For multivariate functions, visualize a bowl in 3D. The function $f(x, y) = x^2 + y^2$ creates a parabolic bowl. Any cross-section through the bowl produces a convex curve. The contour lines (level sets) are concentric circles centered at the origin.

A non-convex function like $f(x) = \sin(x)$ oscillates, with tangent lines sometimes above and sometimes below the curve. The Hessian alternates sign, creating alternating regions of convexity and concavity.

## Common Mistakes

1. **Assuming the sum of convex functions is always convex**: While true for non-negative weighted sums, subtracting convex functions (negative weights) can produce non-convex functions. For example, $f(x) = x^2$ and $g(x) = 2x^2$ are convex, but $f(x) - g(x) = -x^2$ is concave.

2. **Confusing convex functions with convex sets**: A convex function is not the same as a function defined on a convex set. The convexity inequality is about the function values, not just the domain. All convex functions must have convex domains, but having a convex domain does not make a function convex.

3. **Assuming local minimum guarantees global minimum without convexity**: Many beginners think gradient descent always finds the global minimum. This is only guaranteed for convex objectives. In non-convex landscapes, gradient descent can converge to local minima or saddle points.

4. **Believing convex functions must be smooth**: Convex functions need not be differentiable. The absolute value function $f(x) = |x|$ is convex but not differentiable at $x = 0$. It has a subgradient at that point (any value in $[-1, 1]$).

5. **Misapplying the second derivative test for multivariate functions**: For $f: \mathbb{R}^n \to \mathbb{R}$, checking $f''_{ii}(x) \geq 0$ is insufficient. All principal minors of the Hessian must have non-negative determinants (Sylvester's criterion).

6. **Thinking convex functions must be bounded below**: While convex functions with compact domains do attain minima on closed bounded sets, functions like $e^x$ are convex but have no minimum on $\mathbb{R}$ (the infimum is 0, but it is never attained).

7. **Overlooking domain restrictions**: A function may be convex on a restricted domain. For example, $f(x) = \log(x)$ is convex on $\mathbb{R}^+$ (its second derivative $-1/x^2$ is negative? Wait, $\log(x)$ is actually concave. Let's verify: $f'(x) = 1/x$, $f''(x) = -1/x^2 < 0$. So $\log$ is concave, not convex. This common mistake highlights the importance of careful computation.

## Interview Questions

### Beginner - 5

**Q1**: Is the function $f(x) = 3x^2 + 2x + 1$ convex?  
**A**: Yes. $f'(x) = 6x + 2$, $f''(x) = 6 > 0$. The second derivative is positive everywhere, so the function is strictly convex.

**Q2**: What is Jensen's inequality?  
**A**: For a convex function $f$, the function of the average is less than or equal to the average of the function: $f(E[X]) \leq E[f(X)]$.

**Q3**: Can a convex function have multiple local minima?  
**A**: No. For convex functions, any local minimum is a global minimum. However, there may be a flat region of minima (if the function is not strictly convex).

**Q4**: What does the graph of a convex function look like?  
**A**: A bowl shape (like $x^2$). The line segment between any two points on the graph lies above the graph.

**Q5**: Is $f(x) = |x|$ convex?  
**A**: Yes. Although it is not differentiable at $x = 0$, it satisfies the definition of convexity: $|\theta x + (1 - \theta)y| \leq \theta|x| + (1 - \theta)|y|$ by the triangle inequality.

### Intermediate - 5

**Q1**: Prove that if $f$ and $g$ are convex, then $\max(f, g)$ is convex.  
**A**: For any $x, y$ and $\theta \in [0, 1]$: $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ and similarly for $g$. Since $\max(f, g) \geq f$ and $\max(f, g) \geq g$, we have $\max(f(\theta x + (1-\theta)y), g(\theta x + (1-\theta)y)) \leq \theta\max(f(x), g(x)) + (1-\theta)\max(f(y), g(y))$ because the right side is at least as large as each individual convex combination.

**Q2**: Why is MSE convex but the cross-entropy loss for neural networks is not necessarily convex?  
**A**: MSE is convex in the prediction $\hat{y}$, and a linear model makes the composition convex. For neural networks, the parameters appear inside non-linear activation functions, creating non-convex compositions.

**Q3**: What is strong convexity and why does it matter for optimization?  
**A**: A function is $\mu$-strongly convex if $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2}\|y-x\|^2$. Strong convexity guarantees a unique minimum and gives linear convergence rates for gradient descent, unlike the sublinear rates for merely convex functions.

**Q4**: How can you check if a multivariate function is convex using the Hessian?  
**A**: Compute the Hessian matrix and check that it is positive semidefinite everywhere (all eigenvalues $\geq 0$). For a $2 \times 2$ Hessian, this requires the diagonal entries $\geq 0$ and the determinant $\geq 0$.

**Q5**: Give an example of a function that is convex but not differentiable.  
**A**: $f(x) = |x|$, $f(x) = \max(0, x)$ (ReLU), and $f(x) = \|x\|_1$ (L1 norm) are all convex but not differentiable at certain points.

### Advanced - 3

**Q1**: Explain the relationship between convex conjugate (Fenchel dual) and the Legendre transform. How is this used in duality theory for optimization?  
**A**: The Fenchel conjugate $f^*(y) = \sup_x (y^T x - f(x))$ generalizes the Legendre transform. For a convex function, $f^*$ is also convex, and $f^{**} = f$ if $f$ is closed and convex. This duality underpins Lagrangian duality: the dual problem maximizes the conjugate of the objective, providing lower bounds on the primal optimum. Strong duality holds under Slater's condition.

**Q2**: In deep learning, the loss landscape is highly non-convex. Why do optimization methods still find good solutions?  
**A**: Several factors contribute: (1) In high dimensions, local minima are rare compared to saddle points, and gradient methods can escape saddle points. (2) Overparameterization creates many equivalent minima. (3) The loss landscape of deep networks has been shown to have connected sublevel sets where minima are connected by simple paths. (4) Implicit regularization from SGD biases solutions toward flatter minima that generalize better.

**Q3**: Derive the convergence rate of gradient descent for a $\mu$-strongly convex, $L$-smooth function.  
**A**: For such functions, gradient descent with step size $\alpha \leq 1/L$ satisfies: $\|x_{k+1} - x^*\|^2 \leq (1 - \alpha\mu)\|x_k - x^*\|^2$. This geometric (linear) convergence follows from the inequality $\langle \nabla f(x), x - x^*\rangle \geq \frac{\mu}{L}\|\nabla f(x)\|^2 + \frac{\mu}{2}\|x - x^*\|^2$, which combines strong convexity and smoothness.

## Practice Problems

### Easy - 5

**P1**: Determine if $f(x) = 5x^4 + 3x^2$ is convex.

**P2**: Is $f(x) = \sqrt{x}$ defined on $[0, \infty)$ convex or concave?

**P3**: Compute the Hessian of $f(x, y) = x^2 + 2y^2 + 3xy$ and check convexity.

**P4**: Show that $f(x) = -\log(x)$ on $(0, \infty)$ is convex.

**P5**: Is the sum of two convex functions always convex?

### Medium - 5

**P1**: Prove that the pointwise maximum of convex functions is convex.

**P2**: Show that $f(x) = \log(1 + e^x)$ is convex. This is the softplus activation.

**P3**: For $f(x) = \frac{1}{2}x^T A x + b^T x + c$, what condition on $A$ guarantees convexity?

**P4**: Let $f_i$ be convex. Prove that $g(x) = \log(\sum e^{f_i(x)})$ (log-sum-exp) is convex.

**P5**: Find all $\alpha \in \mathbb{R}$ such that $f(x) = x^\alpha$ on $(0, \infty)$ is convex.

### Hard - 3

**P1**: Prove that if $f$ is convex and differentiable, then $\nabla f$ is a monotone operator.

**P2**: Show that the negative entropy $f(x) = \sum_{i=1}^n x_i \log x_i$ on the probability simplex is convex.

**P3**: Derive the conjugate (Fenchel dual) of $f(x) = \frac{1}{2}x^T Q x$ where $Q \succ 0$, and verify that $f^{**} = f$.

## Solutions

### Easy - Solutions

**S1**: $f'(x) = 20x^3 + 6x$, $f''(x) = 60x^2 + 6 \geq 6 > 0$. Strictly convex.

**S2**: $f'(x) = \frac{1}{2\sqrt{x}}$, $f''(x) = -\frac{1}{4x^{3/2}} < 0$. Concave.

**S3**: $\nabla^2 f = \begin{bmatrix} 2 & 3 \\ 3 & 4 \end{bmatrix}$. Determinant $= 2(4) - 3(3) = 8 - 9 = -1 < 0$. Indefinite, not convex.

**S4**: $f'(x) = -1/x$, $f''(x) = 1/x^2 > 0$ for $x > 0$. Convex.

**S5**: Yes, if the weights are non-negative. Convexity is preserved under non-negative weighted sums.

### Medium - Solutions

**S1**: Let $f = \max_i f_i$ where each $f_i$ is convex. For any $x, y$ and $\theta \in [0,1]$: $f_i(\theta x + (1-\theta)y) \leq \theta f_i(x) + (1-\theta)f_i(y) \leq \theta f(x) + (1-\theta)f(y)$. Taking max over $i$ gives $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$.

**S2**: $f'(x) = \frac{e^x}{1+e^x} = \sigma(x)$ (the sigmoid). $f''(x) = \sigma(x)(1-\sigma(x)) > 0$ for all $x$. Hence convex.

**S3**: $\nabla^2 f = A$. Convex if $A \succeq 0$ (positive semidefinite).

**S4**: The Hessian of log-sum-exp can be shown to be $ \text{diag}(z)^{-1} - zz^T$ where $z_i = e^{f_i(x)} / \sum e^{f_j(x)}$. This matrix is positive semidefinite, establishing convexity.

**S5**: $f''(x) = \alpha(\alpha-1)x^{\alpha-2}$. For $f$ to be convex, $\alpha(\alpha-1) \geq 0$, so $\alpha \leq 0$ or $\alpha \geq 1$.

### Hard - Solutions

**S1**: For convex $f$, the subgradient inequality gives $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ and $f(x) \geq f(y) + \nabla f(y)^T(x-y)$. Adding: $0 \geq \nabla f(x)^T(y-x) + \nabla f(y)^T(x-y) = (\nabla f(x) - \nabla f(y))^T(y-x) = -(\nabla f(x) - \nabla f(y))^T(x-y)$. Thus $(\nabla f(x) - \nabla f(y))^T(x-y) \geq 0$.

**S2**: The Hessian of negative entropy is $\text{diag}(1/x_i)$, which is positive definite on the interior of the simplex. The function is convex on the simplex.

**S3**: $f^*(y) = \sup_x (y^T x - \frac{1}{2}x^T Q x)$. Setting gradient to zero: $y - Qx = 0 \Rightarrow x = Q^{-1}y$. Then $f^*(y) = y^T Q^{-1}y - \frac{1}{2}(Q^{-1}y)^T Q (Q^{-1}y) = \frac{1}{2}y^T Q^{-1}y$. Then $f^{**}(x) = \frac{1}{2}x^T Q x = f(x)$.

## Related Concepts

- **Concave Function**: $f$ is concave if $-f$ is convex. Concave functions model diminishing returns.
- **Convex Set**: A set $C$ where $\theta x + (1-\theta)y \in C$ for all $x, y \in C$, $\theta \in [0,1]$.
- **Subgradient**: A generalization of gradient for non-differentiable convex functions.
- **Epigraph**: The set of points above the graph of a function; convex epigraph $\iff$ convex function.
- **Convex Optimization**: Minimizing convex functions over convex sets.
- **Lagrangian Duality**: Convex analysis provides the foundation for duality theory.
- **Legendre-Fenchel Transform**: The convex conjugate relating primal and dual problems.
- **Bregman Divergence**: A distance-like measure generated by a strictly convex function.
- **Quasiconvex Function**: A weaker form where sublevel sets are convex (but the function itself may not satisfy Jensen's inequality).

## Next Concepts

- **Optimization**: The practical problem of minimizing or maximizing convex (and non-convex) functions.
- **Gradient Descent**: The foundational iterative algorithm for minimizing differentiable functions.
- **KKT Conditions**: Necessary conditions for optimality in constrained optimization.
- **Stochastic Optimization**: Extending convex optimization to handle large datasets and noisy gradients.

## Summary

Convex functions are functions where the line segment between any two points lies above the graph. This property, formalized through Jensen's inequality, guarantees that every local minimum is a global minimum. Convexity can be verified through first-order conditions (gradient inequality), second-order conditions (positive semidefinite Hessian), or the epigraph characterization.

Key classes include strictly convex (unique minimum), strongly convex (quadratic lower bound), and non-smooth but convex functions (like $\ell_1$ norm). Operations preserving convexity enable building complex convex functions from simple building blocks.

In machine learning, convex loss functions underpin linear models and are essential for optimization guarantees. For deep learning, understanding convexity clarifies why training is fundamentally harder and why specialized optimizers are needed.

## Key Takeaways

- A convex function satisfies $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ for all $\theta \in [0,1]$.
- For twice-differentiable functions, convexity $\iff$ Hessian $\succeq 0$ everywhere.
- Every local minimum of a convex function is a global minimum.
- Jensen's inequality: $f(E[X]) \leq E[f(X)]$ for convex $f$.
- Examples of convex functions: $x^2$, $e^x$, $-\log x$, $\|x\|_2^2$, $\|x\|_1$, $\max(0, x)$.
- Convex loss functions (MSE, logistic loss, hinge loss) guarantee globally convergent training for linear models.
- Neural network objectives are non-convex due to hidden layers and non-linear activations.
- Strong convexity provides linear convergence rates for gradient-based optimization.
- Subgradients generalize gradients for non-differentiable convex functions.
- Understanding convexity is essential for choosing appropriate optimization algorithms and diagnosing training behavior.
