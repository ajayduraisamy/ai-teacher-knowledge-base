# Concept: Optimization

## Concept ID

MATH-094

## Difficulty

Intermediate

## Domain

Mathematics

## Module

Optimization

## Learning Objectives

- Formulate optimization problems with objective functions, decision variables, and constraints
- Distinguish between unconstrained and constrained optimization, and between convex and non-convex problems
- Derive and apply first-order necessary conditions ($\nabla f = 0$) and second-order sufficient conditions (Hessian positive definite)
- Understand closed-form versus iterative solution approaches and when each is appropriate
- Apply optimization concepts to machine learning training as loss minimization
- Design objective functions with regularization and constraints for specific learning tasks

## Prerequisites

- Multivariate calculus: gradients, Hessians, Taylor expansions
- Linear algebra: matrix operations, eigenvalues, positive definiteness
- Convex functions: definition, properties, global minimum guarantee
- Basic probability: expectations, loss functions

## Definition

**Optimization** is the mathematical discipline of finding the best available solution from a set of feasible alternatives. Formally, an optimization problem is defined as:

$$
\begin{aligned}
&\text{minimize}_{x \in \mathcal{D}} \quad f(x) \\
&\text{subject to} \quad g_i(x) \leq 0, \quad i = 1, \ldots, m \\
&\quad \quad \quad \quad h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}
$$

where $f(x)$ is the **objective function**, $x \in \mathbb{R}^n$ are the **decision variables**, $\mathcal{D}$ is the **domain**, $g_i(x) \leq 0$ are **inequality constraints**, and $h_j(x) = 0$ are **equality constraints**. When there are no constraints, the problem is **unconstrained optimization**.

A point $x^*$ is a **global minimum** if $f(x^*) \leq f(x)$ for all feasible $x$. It is a **local minimum** if there exists a neighborhood $\mathcal{N}$ of $x^*$ such that $f(x^*) \leq f(x)$ for all feasible $x \in \mathcal{N}$.

## Intuition

Optimization is about making the best decision under constraints. Imagine you are designing a shipping network: you want to minimize delivery costs while ensuring every package arrives within 24 hours. The cost function depends on routes, truck assignments, and schedules; the 24-hour requirement is a constraint.

In unconstrained optimization, picture a mountain range where you want to find the lowest valley. You can walk anywhere. In constrained optimization, you are restricted to a fenced area—find the lowest point inside the fence.

The gradient $\nabla f$ points in the direction of steepest ascent. To minimize, we move opposite to the gradient—descending toward valleys. At the bottom, the gradient is zero (flat ground), and the curvature (Hessian) tells us whether we are in a valley (positive curvature), on a ridge (negative curvature), or on a saddle (mixed curvature).

## Why This Concept Matters

Optimization is the computational engine of machine learning. Every time a model is trained, an optimization algorithm minimizes a loss function. Without optimization, there would be no learning—just random parameters.

Beyond ML, optimization drives:
- **Engineering design**: minimizing weight while maintaining structural integrity
- **Finance**: maximizing portfolio return for a given risk level
- **Logistics**: minimizing transportation costs subject to capacity constraints
- **Energy systems**: minimizing power generation costs while meeting demand
- **Drug discovery**: optimizing molecular properties subject to synthesis constraints

Understanding optimization theory helps practitioners choose the right algorithm, diagnose convergence issues, set learning rates, and design objective functions that are amenable to efficient solution.

## Historical Background

Optimization has ancient roots. Euclid's problem of finding the shortest path between two points (a straight line) and Heron's principle of reflection (light takes the shortest path) are early instances.

The calculus revolution brought formal methods. Newton and Leibniz developed techniques for finding minima and maxima using derivatives. In the 18th century, Euler and Lagrange developed the calculus of variations, extending optimization to functions of functions.

The modern era began in the 1940s with George Dantzig's simplex method for linear programming, which solved optimization problems with linear objectives and constraints. The 1950s saw the development of nonlinear programming by Kuhn, Tucker, and Karush (KKT conditions). The 1960s brought gradient-based methods for unconstrained optimization.

The 1980s and 1990s witnessed the rise of interior-point methods (Karmarkar, 1984) and the formalization of convex optimization by Boyd and Vandenberghe. In parallel, the machine learning revolution of the 2000s made stochastic gradient descent the most widely used optimization algorithm in the world.

## Real World Examples

- **Route planning**: GPS navigation solves a shortest-path optimization problem. Google Maps minimizes travel time subject to road network constraints.
- **Manufacturing**: A factory optimizes production quantities to maximize profit, constrained by raw materials, labor hours, and machine capacity.
- **Aerospace**: Aircraft wing design optimizes lift-to-drag ratio subject to structural stress limits.
- **Energy**: Power grid operators minimize generation costs while meeting electricity demand and respecting transmission line capacities.
- **Portfolio management**: Mean-variance optimization selects asset weights to minimize portfolio variance for a target return.

## AI/ML Relevance

**Training as optimization**: Every supervised learning model is trained by solving an optimization problem:

$$
\theta^* = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^N L(y_i, f_\theta(x_i)) + \lambda R(\theta)
$$

where $L$ is the loss function, $R$ is a regularizer, and $\theta$ are model parameters. The entire field of deep learning is driven by advances in optimization algorithms.

**Key optimization challenges in ML**:
- **High dimensionality**: Modern models have millions or billions of parameters. The optimization landscape exists in a space so vast that enumerating critical points is impossible.
- **Non-convexity**: Neural network objectives are non-convex, meaning local minima may exist. However, empirical evidence shows that most local minima are of similar quality.
- **Stochasticity**: Training uses mini-batches, introducing noise into gradient estimates. This noise can help escape sharp minima and improve generalization.
- **Generalization gap**: The optimization objective (training loss) differs from the true goal (test performance). This creates a tension between minimizing training loss and avoiding overfitting.

**Hyperparameter optimization**: Beyond training, the model architecture and training hyperparameters themselves are optimized (learning rate, batch size, network depth, etc.). Methods include grid search, random search, Bayesian optimization, and evolutionary algorithms.

## Mathematical Explanation

### Unconstrained Optimization

For an unconstrained problem $\min_{x \in \mathbb{R}^n} f(x)$, the **first-order necessary condition** (FONC) for a local minimum is:

$$
\nabla f(x^*) = 0
$$

The gradient must vanish at any interior local minimum. Points satisfying $\nabla f(x) = 0$ are called **stationary points**.

The **second-order necessary condition** (SONC) for a local minimum is:

$$
\nabla^2 f(x^*) \succeq 0 \quad \text{(positive semidefinite)}
$$

The **second-order sufficient condition** (SOSC) for an isolated local minimum is:

$$
\nabla f(x^*) = 0 \quad \text{and} \quad \nabla^2 f(x^*) \succ 0 \quad \text{(positive definite)}
$$

### Constrained Optimization

For constrained problems, the **KKT conditions** (Karush-Kuhn-Tucker) provide necessary conditions for optimality. For the problem:

$$
\begin{aligned}
\min_x &\quad f(x) \\
\text{s.t.} &\quad g_i(x) \leq 0, \quad i = 1, \ldots, m \\
&\quad h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}
$$

The Lagrangian is:

$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$

KKT conditions:
1. Stationarity: $\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = 0$
2. Primal feasibility: $g_i(x^*) \leq 0$, $h_j(x^*) = 0$
3. Dual feasibility: $\lambda_i^* \geq 0$
4. Complementary slackness: $\lambda_i^* g_i(x^*) = 0$

### Closed-Form vs. Iterative Solutions

**Closed-form solutions** exist when the optimality conditions yield an explicit formula. For example, linear regression:

$$
\min_w \|Xw - y\|_2^2 \implies w^* = (X^T X)^{-1} X^T y
$$

Closed-form solutions are exact (up to numerical precision) but may be computationally expensive for large problems due to matrix inversion.

**Iterative methods** start with an initial guess and refine it progressively:

$$
x_{k+1} = x_k + \alpha_k d_k
$$

where $\alpha_k$ is a step size and $d_k$ is a search direction (typically the negative gradient). These methods scale to high dimensions and handle non-linear, non-convex problems.

## Formula(s)

**Gradient Descent Update**:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

**Newton's Method**:

$$
x_{k+1} = x_k - \alpha (\nabla^2 f(x_k))^{-1} \nabla f(x_k)
$$

**Taylor Expansion for Optimality Conditions**:

$$
f(x^* + \delta) \approx f(x^*) + \nabla f(x^*)^T \delta + \frac{1}{2} \delta^T \nabla^2 f(x^*) \delta
$$

**Condition Number**:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

of the Hessian. Ill-conditioned problems ($\kappa \gg 1$) converge slowly for gradient descent.

## Properties

1. **Convexity**: If $f$ is convex, any stationary point is a global minimum.
2. **Unimodality**: Convex functions have a single valley (unique minimum if strictly convex).
3. **Well-posedness**: An optimization problem is well-posed if a minimum exists, is unique, and depends continuously on the data.
4. **Existence**: A continuous function on a compact set attains its minimum (Weierstrass theorem).
5. **Duality**: Every optimization problem has a dual problem that provides a lower bound on the optimal value. Strong duality holds when the optimal values coincide.
6. **Sensitivity**: Lagrange multipliers $\lambda_i^*$ measure the sensitivity of the optimal value to constraint perturbations.

## Step-by-Step Worked Examples

### Example 1: Minimizing a Quadratic Function Analytically

**Problem**: Find the minimum of $f(x) = 3x^2 - 12x + 7$.

**Solution**:

Step 1: Compute first derivative:

$$
f'(x) = 6x - 12
$$

Step 2: Set to zero (FONC):

$$
6x - 12 = 0 \implies x^* = 2
$$

Step 3: Check second derivative (SONC/SOSC):

$$
f''(x) = 6 > 0
$$

The Hessian is positive everywhere, confirming $x^* = 2$ is a strict local (and global) minimum.

Step 4: Compute minimum value:

$$
f(2) = 3(4) - 12(2) + 7 = 12 - 24 + 7 = -5
$$

The minimum value is $-5$ at $x = 2$.

### Example 2: Multivariate Optimization

**Problem**: Minimize $f(x, y) = x^2 + 2y^2 - 2xy - 4x - 6y$.

**Solution**:

Step 1: Compute gradient:

$$
\nabla f(x, y) = \begin{bmatrix} 2x - 2y - 4 \\ 4y - 2x - 6 \end{bmatrix}
$$

Step 2: Set $\nabla f = 0$:

$$
\begin{aligned}
2x - 2y - 4 &= 0 \\
-2x + 4y - 6 &= 0
\end{aligned}
$$

Step 3: Solve the linear system. From the first equation: $x - y = 2$, so $x = y + 2$. Substituting into the second:

$$
-2(y + 2) + 4y - 6 = 0 \implies -2y - 4 + 4y - 6 = 0 \implies 2y - 10 = 0 \implies y = 5
$$

Then $x = 7$. The stationary point is $(7, 5)$.

Step 4: Compute Hessian:

$$
\nabla^2 f = \begin{bmatrix} 2 & -2 \\ -2 & 4 \end{bmatrix}
$$

Step 5: Check positive definiteness:

- Leading principal minor 1: $2 > 0$
- Determinant: $2(4) - (-2)(-2) = 8 - 4 = 4 > 0$

The Hessian is positive definite, confirming a strict local minimum.

Step 6: Minimum value:

$$
f(7, 5) = 49 + 2(25) - 2(35) - 28 - 30 = 49 + 50 - 70 - 28 - 30 = -29
$$

### Example 3: Constrained Optimization with Lagrange Multipliers

**Problem**: Minimize $f(x, y) = x^2 + y^2$ subject to $x + y = 1$.

**Solution**:

Step 1: Form the Lagrangian:

$$
\mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 1)
$$

Step 2: Compute partial derivatives and set to zero:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= 2x + \lambda = 0 \implies x = -\lambda/2 \\
\frac{\partial \mathcal{L}}{\partial y} &= 2y + \lambda = 0 \implies y = -\lambda/2 \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= x + y - 1 = 0
\end{aligned}
$$

Step 3: From $x = y = -\lambda/2$, substitute into the constraint:

$$
-\lambda/2 - \lambda/2 = 1 \implies -\lambda = 1 \implies \lambda = -1
$$

Thus $x = 1/2$, $y = 1/2$.

Step 4: Minimum value:

$$
f(1/2, 1/2) = (1/4) + (1/4) = 1/2
$$

The point $(0.5, 0.5)$ is the closest point on the line $x + y = 1$ to the origin—intuitively correct.

### Example 4: Ridge Regression as Optimization

**Problem**: Find $w$ minimizing $f(w) = \|Xw - y\|_2^2 + \lambda \|w\|_2^2$.

**Solution**:

Step 1: Expand the objective:

$$
f(w) = (Xw - y)^T (Xw - y) + \lambda w^T w = w^T X^T X w - 2y^T X w + y^T y + \lambda w^T w
$$

Step 2: Compute gradient:

$$
\nabla f(w) = 2X^T X w - 2X^T y + 2\lambda w = 2(X^T X + \lambda I) w - 2X^T y
$$

Step 3: Set to zero:

$$
(X^T X + \lambda I) w = X^T y
$$

Step 4: Solve:

$$
w^* = (X^T X + \lambda I)^{-1} X^T y
$$

This is the closed-form ridge regression solution. The $\lambda I$ term ensures the matrix is invertible even when $X^T X$ is singular, and it shrinks the coefficients toward zero.

## Visual Interpretation

Consider the contour plot of a quadratic function $f(x, y) = x^2 + 2y^2$. The contours are ellipses centered at the origin. The gradient at any point points perpendicular to the contour lines, toward the direction of steepest ascent.

For gradient descent, the optimization path zig-zags perpendicularly through contour lines, moving toward the center. The curvature (ratio of Hessian eigenvalues) determines how direct the path is. For circular contours ($\kappa = 1$), gradient descent goes straight to the center. For highly elliptical contours ($\kappa \gg 1$), it oscillates.

The condition number $\kappa$ visualizes as the ratio of the longest to shortest axis of the contour ellipses. Ill-conditioned problems have long, narrow valleys where gradient descent bounces between walls.

For constrained optimization with $x + y = 1$, visualize the constraint as a line through the plane. The feasible set is that line. The objective $x^2 + y^2$ is a bowl centered at the origin. The optimum is the point on the line closest to the origin—the tangency point between the line and the smallest circular contour that touches it.

## Common Mistakes

1. **Confusing necessary and sufficient conditions**: $\nabla f = 0$ is necessary for a local minimum but not sufficient—it could be a maximum or saddle point. Always check the Hessian.

2. **Assuming all optimization problems have closed-form solutions**: Most practical problems, especially in deep learning, require iterative methods. Only simple cases (linear regression, quadratic programming) have closed forms.

3. **Ignoring constraints**: Unconstrained optimization can yield infeasible solutions. For example, minimizing variance without a return constraint can lead to all-cash portfolios.

4. **Setting learning rate too high or too low**: In iterative optimization, a large step size causes divergence; too small a step size leads to slow convergence. The optimal step size balances stability and speed.

5. **Assuming convexity guarantees fast convergence**: Even for convex problems, gradient descent converges slowly for ill-conditioned problems with high condition numbers.

6. **Overlooking saddle points**: In high dimensions, saddle points (where gradient is zero but Hessian has mixed eigenvalues) are more common than local minima. First-order methods can stall near saddle points.

7. **Forgetting normalization**: Highly different scales across variables create ill-conditioned problems. Feature scaling (standardization) is essential for efficient optimization.

8. **Treating training loss as the final objective**: The optimization objective (training loss) differs from the true goal (test performance). Over-optimizing training loss can lead to overfitting.

## Interview Questions

### Beginner - 5

**Q1**: What is the difference between a local and global minimum?  
**A**: A local minimum has the smallest function value in its immediate neighborhood; a global minimum has the smallest value over the entire domain.

**Q2**: What does it mean for $\nabla f(x^*) = 0$ at an optimum?  
**A**: The gradient being zero means the function is flat at that point—there is no direction of descent. This is a necessary condition for an unconstrained local minimum.

**Q3**: What is the role of the Hessian in optimization?  
**A**: The Hessian determines the curvature. Positive definite $\implies$ local minimum, negative definite $\implies$ local maximum, indefinite $\implies$ saddle point.

**Q4**: What is a constraint in optimization?  
**A**: A constraint is a condition that feasible solutions must satisfy, either an inequality ($g_i(x) \leq 0$) or equality ($h_j(x) = 0$).

**Q5**: Why is training a neural network considered optimization?  
**A**: Training minimizes a loss function (measuring prediction error) by adjusting the network parameters. This is an unconstrained optimization problem.

### Intermediate - 5

**Q1**: What is the KKT condition and why is it important?  
**A**: The KKT conditions generalize Lagrange multipliers to inequality constraints. They provide necessary conditions for optimality in constrained nonlinear programming and form the basis for many optimization algorithms.

**Q2**: How does the condition number of the Hessian affect gradient descent convergence?  
**A**: A high condition number (ratio of largest to smallest eigenvalue) creates elongated contour ellipses. Gradient descent zig-zags along the valley, converging slowly. Preconditioning or Newton methods address this.

**Q3**: Compare batch gradient descent with stochastic gradient descent from an optimization perspective.  
**A**: Batch GD computes the exact gradient using all data, guaranteeing descent each step but expensive per iteration. SGD uses a mini-batch, giving a noisy gradient estimate. The noise helps escape sharp minima but prevents exact convergence to the optimum.

**Q4**: What is the difference between convex and non-convex optimization in terms of guarantees?  
**A**: In convex optimization, every local minimum is global, and gradient descent with appropriate step size converges to the global optimum. In non-convex optimization, gradient descent may converge to local minima or saddle points with no global optimality guarantee.

**Q5**: Explain the role of regularization in the optimization objective.  
**A**: Regularization adds a penalty term (e.g., $\lambda\|w\|_2^2$) to the loss function. It modifies the optimization landscape, often making it more convex and better conditioned, while preventing overfitting by discouraging large parameters.

### Advanced - 3

**Q1**: Derive the convergence rate of gradient descent for a strongly convex, L-smooth function.  
**A**: For $\mu$-strongly convex and $L$-smooth $f$, gradient descent with step size $\alpha = 1/L$ satisfies $f(x_{k+1}) - f(x^*) \leq (1 - \mu/L)^k (f(x_0) - f(x^*))$. The ratio $\mu/L$ determines the convergence speed. This linear (geometric) convergence is the best achievable with first-order methods.

**Q2**: What is the implicit regularization effect of SGD, and how does it relate to the optimization landscape?  
**A**: SGD's gradient noise biases the optimization toward flat minima (where the Hessian has small eigenvalues). Flat minima generalize better because small parameter perturbations don't significantly affect predictions. This implicit regularization is absent in full-batch GD, which can converge to sharp minima.

**Q3**: How does the convergence theory differ between convex and non-convex optimization for deep neural networks?  
**A**: For convex problems, we have global convergence guarantees. For deep neural networks, standard theory only guarantees convergence to a stationary point ($\nabla f \to 0$) under smoothness and bounded variance assumptions. Recent work shows that overparameterized networks have benign optimization landscapes where gradient descent converges to global minima despite non-convexity.

## Practice Problems

### Easy - 5

**P1**: Find the minimum of $f(x) = x^2 - 6x + 10$.

**P2**: Minimize $f(x, y) = x^2 + y^2 - 2x + 4y$.

**P3**: Check if $f(x) = x^3 - 3x$ has a local minimum at $x = 1$.

**P4**: Minimize $f(x) = e^{x^2 - 4x}$.

**P5**: For $f(x) = \|Ax - b\|_2^2$, derive the closed-form solution.

### Medium - 5

**P1**: Minimize $f(x, y) = 2x^2 + 3y^2 + 4xy - 2x$ and classify the stationary point.

**P2**: Use Lagrange multipliers to minimize $f(x, y) = x^2 + y^2$ subject to $2x + 3y = 6$.

**P3**: For $f(x) = \frac{1}{2}x^T Q x + c^T x$ with $Q \succ 0$, derive the Newton update and compare with gradient descent.

**P4**: Show that ridge regression can be derived as constrained optimization: $\min \|Xw - y\|^2$ subject to $\|w\|_2 \leq t$.

**P5**: Find the minimizer of $f(x) = x_1^2 + 2x_2^2 + 3x_3^2$ subject to $x_1 + x_2 + x_3 = 1$.

### Hard - 3

**P1**: Prove that for convex $f$, any stationary point is a global minimum.

**P2**: For $f(x) = \log(1 + e^{-ax})$ where $a \neq 0$, find the minimum and discuss when it exists.

**P3**: Derive the dual of the Lasso problem $\min_w \frac{1}{2}\|Xw - y\|_2^2 + \lambda \|w\|_1$ and identify when strong duality holds.

## Solutions

### Easy - Solutions

**S1**: $f'(x) = 2x - 6 = 0 \implies x = 3$. $f''(x) = 2 > 0$, so $x = 3$ is a minimum. $f(3) = 9 - 18 + 10 = 1$.

**S2**: $\nabla f = [2x - 2, 2y + 4]^T$. Setting to zero: $x = 1$, $y = -2$. Hessian is $2I$, positive definite. Minimum: $f(1, -2) = 1 + 4 - 2 - 8 = -5$.

**S3**: $f'(x) = 3x^2 - 3 = 0$ at $x = 1$. $f''(x) = 6x$, $f''(1) = 6 > 0$. So $x = 1$ is a local minimum. (Note: $x = -1$ is a local maximum.)

**S4**: Minimizing $e^{g(x)}$ is equivalent to minimizing $g(x) = x^2 - 4x$. $g'(x) = 2x - 4 = 0 \implies x = 2$. $f(2) = e^{4 - 8} = e^{-4}$.

**S5**: $f(w) = (Aw - b)^T(Aw - b)$. $\nabla f = 2A^T(Aw - b) = 0 \implies A^T A w = A^T b$, so $w = (A^T A)^{-1} A^T b$ (assuming $A^T A$ invertible).

### Medium - Solutions

**S1**: $\nabla f = [4x + 4y - 2, 6y + 4x]^T$. Setting to zero: $4x + 4y = 2$, $4x + 6y = 0$. Subtracting: $-2y = 2 \implies y = -1$, $x = 1.5$. Hessian: $\begin{bmatrix} 4 & 4 \\ 4 & 6 \end{bmatrix}$. Determinant $= 24 - 16 = 8 > 0$, leading minor $4 > 0$. Positive definite, so local minimum.

**S2**: $\mathcal{L} = x^2 + y^2 + \lambda(2x + 3y - 6)$. $\partial/\partial x: 2x + 2\lambda = 0$, $\partial/\partial y: 2y + 3\lambda = 0$, $\partial/\partial\lambda: 2x + 3y = 6$. From first two: $x = -\lambda$, $y = -3\lambda/2$. Substituting: $-2\lambda - 9\lambda/2 = 6 \implies -13\lambda/2 = 6 \implies \lambda = -12/13$. Then $x = 12/13$, $y = 18/13$.

**S3**: Newton update: $x_{k+1} = x_k - Q^{-1}(Qx_k + c) = -Q^{-1}c$. This converges in one step (exact for quadratics). Gradient descent: $x_{k+1} = x_k - \alpha(Qx_k + c)$, which converges linearly.

**S4**: The Lagrangian is $\mathcal{L}(w, \lambda) = \|Xw - y\|^2 + \lambda(\|w\|^2 - t^2)$. Setting gradient to zero gives $(X^T X + \lambda I)w = X^T y$. The constraint $\|w\| \leq t$ corresponds to a specific $\lambda$ via the KKT conditions.

**S5**: $\mathcal{L} = x_1^2 + 2x_2^2 + 3x_3^2 + \lambda(x_1 + x_2 + x_3 - 1)$. Derivatives: $2x_1 + \lambda = 0$, $4x_2 + \lambda = 0$, $6x_3 + \lambda = 0$. So $x_1 = -\lambda/2$, $x_2 = -\lambda/4$, $x_3 = -\lambda/6$. Constraint: $-\lambda(1/2 + 1/4 + 1/6) = 1 \implies -\lambda(6/12 + 3/12 + 2/12) = 1 \implies -11\lambda/12 = 1 \implies \lambda = -12/11$. Then $x_1 = 6/11$, $x_2 = 3/11$, $x_3 = 2/11$.

### Hard - Solutions

**S1**: Let $f$ be convex and differentiable, and let $x^*$ satisfy $\nabla f(x^*) = 0$. By the first-order characterization of convexity: $f(y) \geq f(x^*) + \nabla f(x^*)^T(y - x^*) = f(x^*)$ for all $y$. Thus $x^*$ is a global minimum.

**S2**: $f(x) = \log(1 + e^{-ax})$. $f'(x) = \frac{-ae^{-ax}}{1 + e^{-ax}} = -a\sigma(-ax)$. If $a > 0$, $f'(x) < 0$ for all $x$, so $f$ is decreasing. No finite minimum exists; $\lim_{x \to \infty} f(x) = 0$. If $a < 0$, $f'(x) > 0$, increasing, no minimum. Non-trivial minima require $a = 0$ (flat).

**S3**: The Lasso primal: $\min_w \frac{1}{2}\|Xw - y\|^2 + \lambda\|w\|_1$. The dual is: $\max_\alpha -\frac{1}{2}\|\alpha\|_2^2$ subject to $\|X^T\alpha\|_\infty \leq \lambda$. Strong duality holds because the primal is convex and satisfies Slater's condition (no inequality constraints other than the convex regularizer).

## Related Concepts

- **Gradient Descent**: The primary iterative algorithm for unconstrained optimization.
- **Convex Optimization**: The subset of optimization problems with convex objectives and convex feasible sets.
- **Linear Programming**: Optimization with linear objective and linear constraints.
- **Quadratic Programming**: Optimization with quadratic objective and linear constraints.
- **Lagrange Multipliers**: Method for handling equality constraints.
- **KKT Conditions**: Necessary conditions for constrained optimality.
- **Duality Theory**: Every primal problem has a dual, providing bounds and algorithmic insights.
- **Calculus of Variations**: Optimization over function spaces.
- **Combinatorial Optimization**: Optimization over discrete structures.
- **Metaheuristics**: Evolutionary algorithms, simulated annealing for non-convex problems.

## Next Concepts

- **Gradient Descent**: The foundational first-order optimization algorithm.
- **Stochastic Gradient Descent**: Extending gradient descent with mini-batch sampling.
- **Momentum**: Accelerating gradient descent with velocity accumulation.
- **Adaptive Methods (Adam, RMSProp)**: Per-parameter learning rate adaptation for robust optimization.

## Summary

Optimization is the mathematical framework for finding the best solution from available alternatives. Problems are characterized by an objective function, decision variables, and optional constraints. Solutions are classified as local or global minima, with the gradient and Hessian providing necessary and sufficient optimality conditions.

Unconstrained optimization handles problems without restrictions, solved via closed-form (when available) or iterative methods. Constrained optimization adds feasibility requirements, addressed through Lagrange multipliers and KKT conditions.

In machine learning, optimization is the computational engine behind training. Every learning algorithm solves an optimization problem to find parameters that minimize a loss function. The choice of optimization algorithm—batch GD, SGD, Adam—significantly impacts training speed and final model quality.

## Key Takeaways

- Optimization = minimizing or maximizing an objective function subject to constraints.
- FONC: $\nabla f(x^*) = 0$ for unconstrained local minima.
- SOSC: $\nabla^2 f(x^*) \succ 0$ confirms a strict local minimum.
- Convex objectives guarantee global optimality of stationary points.
- Closed-form solutions exist only for simple problems (linear regression, quadratic forms).
- Iterative methods (gradient descent, Newton) are essential for high-dimensional optimization.
- KKT conditions generalize Lagrange multipliers to inequality constraints.
- Machine learning training is fundamentally an optimization problem.
- Regularization modifies the objective to improve generalization.
- The condition number of the Hessian determines gradient descent convergence speed.
