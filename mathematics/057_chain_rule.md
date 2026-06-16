# Concept: Chain Rule

## Concept ID

MATH-057

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- State and apply the single-variable chain rule: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$.
- Extend the chain rule to multivariable functions: $\frac{d}{dt}f(x(t), y(t)) = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$.
- Apply the chain rule to functions of multiple variables and multiple paths (tree diagrams).
- Understand the chain rule as the foundation of backpropagation in neural networks.
- Compute gradients through arbitrary computational graphs using the chain rule.
- Derive the backpropagation equations for a multi-layer neural network.

## Prerequisites

- Derivative (MATH-055) — the chain rule is a rule for computing derivatives.
- Partial Derivative (MATH-056) — the multivariable chain rule uses partial derivatives.
- Composite functions (MATH-047) — the chain rule differentiates compositions.
- Basic understanding of neural network architectures (layers, activations, loss functions).

## Definition

**Single-Variable Chain Rule:** If $y = f(u)$ and $u = g(x)$, then the derivative of the composite function $f \circ g$ with respect to $x$ is:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = f'(g(x)) \cdot g'(x)$$

In words: the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function.

**Leibniz Notation:**
$$\frac{d}{dx}[f(g(x))] = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Multivariable Chain Rule (One Dependent, Two Independent):** If $z = f(x, y)$, $x = x(t)$, $y = y(t)$, then:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

**General Form (Several Paths):** If $z = f(x_1, x_2, \dots, x_n)$ and each $x_i = x_i(t_1, t_2, \dots, t_m)$, then:

$$\frac{\partial z}{\partial t_j} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j}$$

This is the sum over all paths from $z$ to $t_j$ in the dependency graph.

## Intuition

Think of the chain rule as describing how changes propagate through a system. If $z$ depends on $y$, and $y$ depends on $x$, then a small change in $x$ causes a change in $y$, which in turn causes a change in $z$. The chain rule multiplies these "sensitivity factors" and sums over all paths.

**Analogy: Factory Production.** Suppose a factory's profit $P$ depends on the number of workers $N$, which depends on the hourly wage $w$. If we increase the wage, how does profit change? The chain rule gives:
$$\frac{dP}{dw} = \frac{dP}{dN} \cdot \frac{dN}{dw}$$
The first factor is "how much profit changes per additional worker" (marginal profit), and the second is "how many workers change per wage increase" (labour supply elasticity).

**Multivariable Intuition.** If $z = f(x, y)$ and both $x$ and $y$ depend on $t$, then changing $t$ affects $z$ through two channels: via changes in $x$ and via changes in $y$. The total effect is the sum of the two effects: (sensitivity of $z$ to $x$) times (rate of change of $x$) plus (sensitivity of $z$ to $y$) times (rate of change of $y$).

This is exactly what happens in backpropagation: the loss depends on many parameters through many intermediate variables. The chain rule tells us to sum over all paths from the loss to each parameter, multiplying derivatives along each path.

## Why This Concept Matters

The chain rule is **the single most important calculus concept for deep learning**. Here is why:

1. **Backpropagation IS the Chain Rule.** Backpropagation is simply the chain rule applied to neural networks. Every gradient computed during training uses the chain rule. Without it, we could not train multi-layer networks.

2. **Automatic Differentiation.** Modern deep learning frameworks (PyTorch, TensorFlow, JAX) implement reverse-mode automatic differentiation, which is just systematic application of the chain rule on a computational graph.

3. **Gradient Computation Efficiency.** The chain rule allows computing all gradients in $O(n)$ time (where $n$ is the number of parameters) rather than $O(n^2)$ or worse.

4. **Scalability to Deep Networks.** The chain rule makes training arbitrarily deep networks computationally feasible. Each layer adds one more factor in the chain of derivatives.

5. **Scientific Modelling.** Differential equations, fluid dynamics, thermodynamics, and many other fields model how quantities change through chains of dependencies.

## Historical Background

The chain rule has been implicit in calculus since its inception. Newton and Leibniz both understood the concept, though they expressed it differently.

Gottfried Wilhelm Leibniz (1646-1716) developed the notation $\frac{dy}{dx}$ and the chain rule $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$ as a natural consequence of treating derivatives as ratios of differentials. His notation made the chain rule almost trivial — if $dy/du$ and $du/dx$ are fractions, their product is $dy/dx$.

In the 18th century, Leonhard Euler (1707-1783) and Joseph-Louis Lagrange (1736-1813) developed the multivariable chain rule. Euler's work on partial differential equations required understanding how functions of multiple variables change when their arguments change.

The rigorous proof of the chain rule using limits was provided by Augustin-Louis Cauchy (1789-1857) in the 19th century. He showed that $\lim_{x\to a} \frac{f(g(x)) - f(g(a))}{x - a} = f'(g(a)) \cdot g'(a)$ under appropriate differentiability conditions.

The connection to neural networks was established in the 1970s-1980s. Paul Werbos (1974) first described backpropagation as an application of the chain rule to neural networks. David Rumelhart, Geoffrey Hinton, and Ronald Williams (1986) popularised the algorithm, showing how the chain rule enables efficient gradient computation through multi-layer networks. This work sparked the neural network revolution.

## Real World Examples

**Example 1: Weather Balloon.** The temperature $T$ a weather balloon experiences depends on altitude $h$, and altitude depends on time $t$: $T(h) = T_0 - Lh$ (where $L$ is the lapse rate), and $h(t) = \frac{1}{2}at^2$ (constant ascent). The rate of temperature change at the balloon is:
$$\frac{dT}{dt} = \frac{dT}{dh} \cdot \frac{dh}{dt} = (-L) \cdot (at) = -aLt$$

**Example 2: Economic Elasticity.** Revenue $R = p q(p)$ where $q(p)$ is the quantity demanded at price $p$. The marginal revenue is:
$$\frac{dR}{dp} = 1 \cdot q(p) + p \cdot q'(p) = q(p) + p \cdot q'(p)$$
This combines product rule and chain rule: changing price affects revenue directly (selling at a higher price) and indirectly (changing quantity demanded).

**Example 3: Chemical Reaction Rate.** The concentration $C$ of a product depends on temperature $T$, and temperature follows a programmed schedule $T(t)$: $C(T) = C_0 e^{-E_a/(RT)}$ (Arrhenius equation). The rate of concentration change is:
$$\frac{dC}{dt} = \frac{dC}{dT} \cdot \frac{dT}{dt} = C_0 e^{-E_a/(RT)} \cdot \frac{E_a}{RT^2} \cdot \frac{dT}{dt}$$

**Example 4: Robot Arm Kinematics.** The end-effector position $(x, y)$ of a 2-joint robot arm depends on joint angles $\theta_1, \theta_2$: $x = L_1\cos\theta_1 + L_2\cos(\theta_1 + \theta_2)$, $y = L_1\sin\theta_1 + L_2\sin(\theta_1 + \theta_2)$. If the joint angles follow trajectories $\theta_1(t), \theta_2(t)$, the end-effector velocity is:
$$\frac{dx}{dt} = \frac{\partial x}{\partial \theta_1} \frac{d\theta_1}{dt} + \frac{\partial x}{\partial \theta_2} \frac{d\theta_2}{dt}$$
$$\frac{dy}{dt} = \frac{\partial y}{\partial \theta_1} \frac{d\theta_1}{dt} + \frac{\partial y}{\partial \theta_2} \frac{d\theta_2}{dt}$$
This is the Jacobian matrix times the joint velocity vector.

**Example 5: Drug Metabolism.** A drug is administered orally (amount $A(t)$ in stomach) and absorbed into the bloodstream (amount $B(t)$), then eliminated: $\frac{dB}{dt} = k_a A - k_e B$. If the efficacy $E$ of the drug depends on $B$, the rate of change of efficacy is:
$$\frac{dE}{dt} = \frac{dE}{dB} \cdot \frac{dB}{dt} = \frac{dE}{dB} (k_a A - k_e B)$$

## AI/ML Relevance

The chain rule is **the fundamental theorem of deep learning**. Every neural network training loop is an exercise in applying the chain rule:

**1. Backpropagation is the Chain Rule.** Consider a simple feedforward network:
$$x \to z_1 = W_1 x + b_1 \to h_1 = \sigma(z_1) \to z_2 = W_2 h_1 + b_2 \to \hat{y} = z_2 \to L = \frac{1}{2}(\hat{y} - y)^2$$

The gradient for $W_1$ is computed via the chain rule:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

Each factor is a local derivative:
- $\partial L / \partial \hat{y} = \hat{y} - y$
- $\partial \hat{y} / \partial h_1 = W_2^T$
- $\partial h_1 / \partial z_1 = \sigma'(z_1)$
- $\partial z_1 / \partial W_1 = x^T$

**2. Computational Graphs.** Any neural network can be represented as a directed acyclic graph (DAG) where nodes are operations and edges are data flows. The chain rule is applied backward through this graph:

```
L → ∂L/∂L = 1
 ↓
 ∂L/∂ŷ = ŷ - y
 ↓
 ∂L/∂z₂ = ∂L/∂ŷ · 1
 ↓↙↘
 ∂L/∂W₂ = ∂L/∂ŷ · h₁^T    ∂L/∂h₁ = W₂^T · ∂L/∂ŷ
 ↓
 ∂L/∂z₁ = ∂L/∂h₁ · σ'(z₁)
 ↓↙↘
 ∂L/∂W₁ = ∂L/∂z₁ · x^T    ∂L/∂b₁ = ∂L/∂z₁
```

**3. Gradient Flow Through Deep Networks.** For an $L$-layer network, the gradient for the first layer involves a product of $L$ Jacobians:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_L} \cdot \prod_{i=2}^L \frac{\partial h_i}{\partial h_{i-1}} \cdot \frac{\partial h_1}{\partial W_1}$$

Each term $\partial h_i / \partial h_{i-1}$ is a matrix multiplication (weight matrix times activation derivative). This is why the chain rule explains the vanishing gradient problem: if these matrices have eigenvalues $< 1$, the product tends to zero as $L$ increases.

**4. The Multivariable Chain Rule in Action.** For a network with multiple branches (e.g., ResNet with skip connections), the chain rule sums over all paths:
$$\frac{\partial L}{\partial x} = \sum_{\text{paths } p} \prod_{\text{edges } e \in p} \frac{\partial f_e}{\partial \text{input}}$$

Skip connections create additional paths in the computational graph, which helps gradients flow more directly to early layers — this is precisely why ResNets can train very deep networks.

**5. Automatic Differentiation.** Modern ML frameworks implement three modes of the chain rule:
- **Forward-mode AD:** Compute $\partial y / \partial x$ by propagating derivatives forward. Efficient when $y$ is a scalar and $x$ is large (many inputs).
- **Reverse-mode AD (backpropagation):** Compute $\partial y / \partial x$ by propagating derivatives backward. Efficient when $x$ is large and $y$ is a scalar (exactly the neural network case).
- **Jacobian-Vector Products:** The forward and backward passes can be combined to efficiently compute products of Jacobians with vectors, which is useful for Hessian-vector products.

**6. Applications Beyond Neural Networks.**
- **Variational Autoencoders (VAEs):** The reparameterisation trick uses the chain rule to backpropagate through random sampling: $z = \mu + \sigma \cdot \varepsilon$, $\frac{\partial z}{\partial \mu} = 1$.
- **Neural ODEs:** The adjoint sensitivity method computes gradients through ODE solutions using a continuous version of the chain rule.
- **Reinforcement Learning:** Policy gradient methods use the chain rule to compute gradients through stochastic policies.
- **Generative Models:** Normalising flows use the chain rule to compute log-determinants of Jacobians through invertible transformations.

**7. Concrete Example: 2-Layer Network Backpropagation.**

Consider: $h = \sigma(W_1 x + b_1)$, $\hat{y} = W_2 h + b_2$, $L = \frac{1}{2}(\hat{y} - y)^2$.

Forward pass:
$$z_1 = W_1 x + b_1$$
$$h = \sigma(z_1)$$
$$z_2 = W_2 h + b_2$$
$$\hat{y} = z_2$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

Backward pass (chain rule):
$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2} = (\hat{y} - y) \cdot h^T$$
$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial \hat{y}} \cdot 1 = \hat{y} - y$$
$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} = W_2^T (\hat{y} - y)$$
$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z_1} = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$$
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1} = (W_2^T (\hat{y} - y) \odot \sigma'(z_1)) \cdot x^T$$
$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot 1 = W_2^T (\hat{y} - y) \odot \sigma'(z_1)$$

The backward pass flows the error signal $\hat{y} - y$ back through the network, applying the chain rule at each step.

## Mathematical Explanation

**Proof of Single-Variable Chain Rule:**
If $f$ is differentiable at $g(x)$ and $g$ is differentiable at $x$, then:
$$\frac{d}{dx}[f(g(x))] = \lim_{h\to 0} \frac{f(g(x+h)) - f(g(x))}{h}$$
$$= \lim_{h\to 0} \frac{f(g(x+h)) - f(g(x))}{g(x+h) - g(x)} \cdot \frac{g(x+h) - g(x)}{h}$$
$$= f'(g(x)) \cdot g'(x)$$

provided $g(x+h) \neq g(x)$ for small $h$. When $g$ is constant near $x$, a more careful proof is needed, but the result holds in general.

**Tree Diagram for Multivariable Chain Rule:**
```
        ┌── x ──┐
        │       │
z ──────┤       ├── t
        │       │
        └── y ──┘
```

For $z = f(x, y)$ with $x = x(t)$, $y = y(t)$:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}$$

**General Tree Diagram:**
```
        ┌── x₁ ──┐
        │        │
        ├── x₂ ──┤
        │        │
z ──────┤  ...   ├── tⱼ
        │        │
        ├── xₙ ──┤
        │        │
        └────────┘
```

$$\frac{\partial z}{\partial t_j} = \sum_{i=1}^n \frac{\partial z}{\partial x_i} \frac{\partial x_i}{\partial t_j}$$

Each path from $z$ to $t_j$ through $x_i$ contributes one term.

**Chain Rule for Vector-Valued Functions:**
If $f: \mathbb{R}^m \to \mathbb{R}^n$ and $g: \mathbb{R}^k \to \mathbb{R}^m$, then:
$$J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$$
where $J$ denotes the Jacobian matrix (matrix of all first-order partial derivatives). The chain rule says the Jacobian of the composition is the product of the Jacobians.

## Formula(s)

**Single-Variable Chain Rule:**
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**Leibniz Form:**
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Multivariable (One Variable Path):**
$$\frac{d}{dt}f(x(t), y(t)) = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

**Multivariable (Multiple Variables):**
$$\frac{\partial}{\partial t_j} f(x_1(\mathbf{t}), \dots, x_n(\mathbf{t})) = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \frac{\partial x_i}{\partial t_j}$$

**Vector-Valued (Jacobian Product):**
$$J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$$

**Backpropagation Gradient for Layer l:**
$$\frac{\partial L}{\partial W^{(\ell)}} = \delta^{(\ell)} \cdot (h^{(\ell-1)})^T$$
where $\delta^{(\ell)}$ is the error signal at layer $\ell$.

## Properties

1. **Chaining Property:** The derivative of a composition is the product of the derivatives of the constituent functions, composed appropriately.

2. **Linearity of Chain Rule for Multiple Paths:** The chain rule sums contributions from all paths in the dependency graph. Each path contributes the product of derivatives along that path.

3. **Associativity:** $(f \circ (g \circ h))' = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x) = ((f \circ g) \circ h)'(x)$ — chaining is associative.

4. **Inverse Function Relationship:** If $f$ is invertible and differentiable, then $(f^{-1})'(y) = 1/f'(f^{-1}(y))$, which follows from differentiating $f(f^{-1}(y)) = y$.

5. **Product Rule via Chain Rule:** The product rule $(fg)' = f'g + fg'$ can be derived from the chain rule by considering $f \cdot g$ as a composition with the multiplication function $M(u, v) = uv$.

6. **Higher-Order Chain Rule (Fa\`a di Bruno's Formula):** The $n$-th derivative of a composition involves Bell polynomials and sums over partitions — there is no simple formula beyond first order.

7. **Computational Graph Interpretation:** The chain rule on a DAG computes gradients by backpropagating error signals — each node receives error from its downstream nodes, multiplies by its local derivative, and passes the result upstream.

8. **Gradient Flow:** The chain rule preserves the information bottleneck: if one path has zero derivative, that entire path contributes nothing to the gradient.

## Step-by-Step Worked Examples

### Example 1: Basic Single-Variable Chain Rule

Find $h'(x)$ for $h(x) = \sin(x^3 + 2x)$.

**Step 1:** Identify outer function $f(u) = \sin u$ and inner function $u = g(x) = x^3 + 2x$.

**Step 2:** $f'(u) = \cos u = \cos(x^3 + 2x)$.

**Step 3:** $g'(x) = 3x^2 + 2$.

**Step 4:** Apply the chain rule: $h'(x) = f'(g(x)) \cdot g'(x) = \cos(x^3 + 2x) \cdot (3x^2 + 2)$.

**Answer:** $h'(x) = (3x^2 + 2) \cos(x^3 + 2x)$.

### Example 2: Multivariable Chain Rule with One Parameter

Let $z = f(x, y) = x^2 y + \sin y$, where $x = t^2$ and $y = e^t$. Find $dz/dt$ at $t = 0$.

**Step 1:** Compute partial derivatives: $\partial z / \partial x = 2xy$, $\partial z / \partial y = x^2 + \cos y$.

**Step 2:** Compute derivatives of intermediate variables: $dx/dt = 2t$, $dy/dt = e^t$.

**Step 3:** Apply the multivariable chain rule:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt} = (2xy)(2t) + (x^2 + \cos y)(e^t)$$

**Step 4:** Substitute $x = t^2$, $y = e^t$:
$$= (2t^2 \cdot e^t)(2t) + (t^4 + \cos(e^t))(e^t) = 4t^3 e^t + e^t t^4 + e^t \cos(e^t)$$

**Step 5:** Evaluate at $t = 0$: $z = 0 + 0 + 1 \cdot \cos(1) = \cos(1) \approx 0.5403$.

**Answer:** $dz/dt|_{t=0} = \cos(1) \approx 0.5403$.

### Example 3: Backpropagation Through a 2-Layer Network (Sigmoid)

Consider: $h = \sigma(W_1 x + b_1)$, $\hat{y} = W_2 h + b_2$, $L = \frac{1}{2}(\hat{y} - y)^2$. Given $W_1 = 1$, $b_1 = 0$, $W_2 = 2$, $b_2 = 1$, $x = 1$, $y = 3$, compute $\partial L / \partial W_1$ by hand.

**Step 1:** Forward pass.
$$z_1 = 1 \cdot 1 + 0 = 1$$
$$h = \sigma(1) = \frac{1}{1+e^{-1}} \approx 0.7311$$
$$z_2 = 2 \cdot 0.7311 + 1 = 2.4622$$
$$\hat{y} = 2.4622$$
$$L = \frac{1}{2}(2.4622 - 3)^2 = \frac{1}{2}(-0.5378)^2 \approx 0.1446$$

**Step 2:** Backward pass (chain rule).
$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 2.4622 - 3 = -0.5378$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot h = -0.5378 \cdot 0.7311 \approx -0.3932$$

$$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial \hat{y}} = -0.5378$$

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \cdot W_2 = -0.5378 \cdot 2 = -1.0756$$

$$\frac{\partial h}{\partial z_1} = \sigma'(1) = \sigma(1)(1 - \sigma(1)) = 0.7311 \cdot 0.2689 \approx 0.1966$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z_1} = -1.0756 \cdot 0.1966 \approx -0.2115$$

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x = -0.2115 \cdot 1 = -0.2115$$

$$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot 1 = -0.2115$$

**Answer:** $\partial L / \partial W_1 \approx -0.2115$. Gradient descent would update $W_1 \leftarrow W_1 - \alpha(-0.2115) = W_1 + 0.2115\alpha$ (increase the weight since the prediction was too low).

### Example 4: Chain Rule with Multiple Paths (Tree Diagram)

Let $z = f(x, y) = x^2 + xy$, where $x = s + t$, $y = s - t$. Find $\partial z / \partial s$ and $\partial z / \partial t$.

**Step 1:** Dependency graph: $z$ depends on $x$ and $y$, each depending on $s$ and $t$.

**Step 2:** Partial derivatives of $z$:
$$\partial z / \partial x = 2x + y, \quad \partial z / \partial y = x$$

**Step 3:** Partial derivatives of $x$ and $y$:
$$\partial x / \partial s = 1, \quad \partial x / \partial t = 1$$
$$\partial y / \partial s = 1, \quad \partial y / \partial t = -1$$

**Step 4:** Apply the chain rule for $\partial z / \partial s$:
$$\frac{\partial z}{\partial s} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial s} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial s} = (2x + y) \cdot 1 + x \cdot 1 = 3x + y$$

**Step 5:** Apply the chain rule for $\partial z / \partial t$:
$$\frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot \frac{\partial x}{\partial t} + \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial t} = (2x + y) \cdot 1 + x \cdot (-1) = x + y$$

**Step 6:** Substitute $x = s + t$, $y = s - t$:
$$\partial z / \partial s = 3(s + t) + (s - t) = 4s + 2t$$
$$\partial z / \partial t = (s + t) + (s - t) = 2s$$

**Verification:** Direct substitution gives $z = (s+t)^2 + (s+t)(s-t) = 2s^2 + 2st$. Then $\partial z / \partial s = 4s + 2t$, $\partial z / \partial t = 2s$. Correct!

**Answer:** $\partial z / \partial s = 4s + 2t$, $\partial z / \partial t = 2s$.

### Example 5: Gradient Through a Computation Graph with Branching

Compute $dL/dx$ for $L = f(g(x)) + h(g(x))$ where $f(u) = u^2$, $g(x) = 3x$, $h(v) = \sin(v)$.

**Step 1:** Notice that $g(x)$ feeds into two different functions. The chain rule must sum over both paths.

**Step 2:** Let $u = g(x) = 3x$. Then $L = f(u) + h(u)$.

**Step 3:** By the chain rule summing over paths:
$$\frac{dL}{dx} = \frac{dL}{du} \cdot \frac{du}{dx} = \left(\frac{df}{du} + \frac{dh}{du}\right) \cdot \frac{du}{dx}$$

**Step 4:** Compute each derivative:
$$\frac{df}{du} = 2u = 2(3x) = 6x$$
$$\frac{dh}{du} = \cos u = \cos(3x)$$
$$\frac{du}{dx} = 3$$

**Step 5:** Combine:
$$\frac{dL}{dx} = (6x + \cos(3x)) \cdot 3 = 18x + 3\cos(3x)$$

**Answer:** $dL/dx = 18x + 3\cos(3x)$.

## Visual Interpretation

**Tree Diagram for Chain Rule:**
```
      z
     / \
    x   y
   / \ / \
  s   t   s   t
```
Each path from root $z$ to leaf $t$ contributes a product of edge derivatives. Sum over all paths.

**Backpropagation Visual:**
```
Forward Pass:
x → z₁ = W₁x+b₁ → h₁ = σ(z₁) → z₂ = W₂h₁+b₂ → ŷ → L

Backward Pass (Chain Rule):
L → ∂L/∂ŷ → ∂L/∂z₂ → ∂L/∂h₁ → ∂L/∂z₁ → ∂L/∂W₁, ∂L/∂b₁
          ↘            ↘
        ∂L/∂W₂, ∂L/∂b₂
```
Each backward step is an application of the chain rule.

**Error Signal Flow:**
```
Layer 1          Layer 2          Layer 3
    ↑              ↑               ↑
δ₁ = (W₂ᵀδ₂)   δ₂ = (W₃ᵀδ₃)    δ₃ =  ŷ - y
    ⊙ σ'(z₁)     ⊙ σ'(z₂)
    ↑              ↑               ↑
  ∂L/∂W₁        ∂L/∂W₂          ∂L/∂W₃
  = δ₁xᵀ        = δ₂h₁ᵀ         = δ₃h₂ᵀ
```
The error $\delta$ is backpropagated: each layer receives $\delta$ from the layer above, multiplies by the transposed weight and activation derivative, then passes it backward.

## Common Mistakes

1. **Forgetting the inner derivative.** The most common mistake: $\frac{d}{dx} \sin(x^2) = \cos(x^2)$, missing the factor $2x$. Always multiply by the derivative of the inner function.

2. **Treating $f'(g(x))$ as the derivative of the composition.** $f'(g(x))$ is the derivative of $f$ evaluated at $g(x)$ — this is not the same as $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. The missing $g'(x)$ factor is a very common error.

3. **Forgetting to sum over all paths in multivariable chain rule.** When a variable affects the final output through multiple paths, the chain rule sums the contributions. Backpropagation through skip connections (ResNet) requires summing gradients from both the direct and skip paths.

4. **Confusing total and partial derivatives.** In $\frac{d}{dt}f(x(t), y(t))$, we use $\partial f/\partial x$ for the partial derivative of $f$, and $d/dt$ for the total derivative with respect to $t$. Mixing these notations can cause confusion.

5. **Not respecting the chain rule order in matrix calculus.** For vector-valued functions, the chain rule involves matrix multiplication: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial W}$. The dimensions must match for matrix multiplication, and the order matters.

6. **Forgetting to transpose weight matrices in backpropagation.** When backpropagating $\partial L / \partial h = W^T \cdot \partial L / \partial \hat{y}$, the weight matrix must be transposed. This is because $\hat{y} = Wh$ gives $\partial \hat{y} / \partial h = W^T$ (in numerator layout).

7. **Assuming the chain rule only applies to scalar functions.** The chain rule generalises to vector and matrix functions via Jacobians. In automatic differentiation, the chain rule is applied through tensor operations.

8. **Applying the chain rule when the inner function does not depend on the variable.** If $g(x)$ does not depend on $x$ (constant), then $g'(x) = 0$ and the chain rule gives 0, which is correct since the composition is constant.

9. **Not handling element-wise operations correctly.** For activation functions applied element-wise (ReLU, sigmoid), the derivative in backpropagation is the element-wise product (Hadamard product): $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \odot \sigma'(z)$.

10. **Confusing the chain rule with the product rule.** The product rule $(fg)' = f'g + fg'$ is different from the chain rule $f(g(x))' = f'(g(x))g'(x)$. They serve different purposes: the product rule differentiates products; the chain rule differentiates compositions.

## Interview Questions

### Beginner

1. **State the chain rule for a single-variable composite function.**
   *Answer: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$. In Leibniz notation: $\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$.*

2. **Differentiate $h(x) = (3x^2 + 1)^4$.**
   *Answer: Outer: $u^4$, inner: $3x^2 + 1$. $h'(x) = 4(3x^2 + 1)^3 \cdot 6x = 24x(3x^2 + 1)^3$.*

3. **What is the multivariable chain rule for $z = f(x(t), y(t))$?**
   *Answer: $\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$. This sums over all paths from $z$ to $t$.*

4. **Find $\frac{d}{dx} \ln(\sin x)$.**
   *Answer: $\frac{1}{\sin x} \cdot \cos x = \frac{\cos x}{\sin x} = \cot x$.*

5. **Differentiate $e^{\cos x}$.**
   *Answer: $e^{\cos x} \cdot (-\sin x) = -\sin x \cdot e^{\cos x}$.*

### Intermediate

1. **Explain why backpropagation is an application of the chain rule.**
   *Answer: Backpropagation computes the gradient of the loss $L$ with respect to each weight $w$ by applying the chain rule: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdot \dots \cdot \frac{\partial h_1}{\partial w}$. Each factor is a local derivative at each layer. The backward pass multiplies these derivatives together, summing over paths when there are multiple dependencies. This systematic application of the chain rule on a computational graph enables efficient gradient computation for networks with millions of parameters.*

2. **Let $z = f(x, y)$ where $x = r\cos\theta$ and $y = r\sin\theta$. Express $\partial z / \partial r$ and $\partial z / \partial \theta$ in terms of $\partial z / \partial x$ and $\partial z / \partial y$.**
   *Answer: Using the chain rule: $\frac{\partial z}{\partial r} = \frac{\partial z}{\partial x} \frac{\partial x}{\partial r} + \frac{\partial z}{\partial y} \frac{\partial y}{\partial r} = \frac{\partial z}{\partial x} \cos\theta + \frac{\partial z}{\partial y} \sin\theta$. $\frac{\partial z}{\partial \theta} = \frac{\partial z}{\partial x} \frac{\partial x}{\partial \theta} + \frac{\partial z}{\partial y} \frac{\partial y}{\partial \theta} = \frac{\partial z}{\partial x} (-r\sin\theta) + \frac{\partial z}{\partial y} (r\cos\theta) = -r\frac{\partial z}{\partial x} \sin\theta + r\frac{\partial z}{\partial y} \cos\theta$.*

3. **Derive the gradient for $W_1$ in a 2-layer network with ReLU activation through the chain rule.**
   *Answer: Forward: $z_1 = W_1 x$, $h = \text{ReLU}(z_1)$, $\hat{y} = W_2 h$, $L = \frac{1}{2}(\hat{y} - y)^2$. Backward: $\partial L / \partial \hat{y} = \hat{y} - y$, $\partial L / \partial W_2 = (\hat{y} - y) h^T$. $\partial L / \partial h = W_2^T (\hat{y} - y)$. $\partial L / \partial z_1 = \partial L / \partial h \odot \mathbf{1}_{z_1 > 0}$. $\partial L / \partial W_1 = (\partial L / \partial z_1) \cdot x^T$.*

4. **What is the difference between forward-mode and reverse-mode automatic differentiation?**
   *Answer: Forward-mode AD propagates derivatives forward from inputs to outputs: $\dot{y} = \sum_i (\partial y / \partial x_i) \dot{x}_i$. It computes one column of the Jacobian per forward pass — efficient when $y$ is large and $x$ is small. Reverse-mode AD (backpropagation) propagates derivatives backward from outputs to inputs: $\bar{x}_i = \sum_j (\partial y_j / \partial x_i) \bar{y}_j$. It computes one row of the Jacobian per backward pass — efficient when $y$ is small and $x$ is large, which is exactly the neural network case (scalar loss, many parameters). Reverse-mode requires storing intermediate values from the forward pass, increasing memory usage.*

5. **If $f(x) = \int_a^{g(x)} h(t) dt$, find $f'(x)$.**
   *Answer: By the Fundamental Theorem of Calculus combined with the chain rule: $f'(x) = h(g(x)) \cdot g'(x)$. The inner function is the upper limit $g(x)$, so the chain rule gives the derivative of the integral as the integrand evaluated at the upper limit times the derivative of the upper limit.*

### Advanced

1. **Prove the multivariable chain rule $\frac{d}{dt}f(x(t), y(t)) = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$ using the definition of differentiability.**
   *Answer: Since $f$ is differentiable at $(x(t), y(t))$, we can write: $f(x+h, y+k) - f(x, y) = \frac{\partial f}{\partial x} h + \frac{\partial f}{\partial y} k + \varepsilon_1 h + \varepsilon_2 k$, where $\varepsilon_1, \varepsilon_2 \to 0$ as $(h, k) \to (0, 0)$. Let $h = x(t+\Delta t) - x(t)$, $k = y(t+\Delta t) - y(t)$. Then $\frac{f(x(t+\Delta t), y(t+\Delta t)) - f(x(t), y(t))}{\Delta t} = \frac{\partial f}{\partial x} \frac{h}{\Delta t} + \frac{\partial f}{\partial y} \frac{k}{\Delta t} + \varepsilon_1 \frac{h}{\Delta t} + \varepsilon_2 \frac{k}{\Delta t}$. As $\Delta t \to 0$, $h/\Delta t \to dx/dt$, $k/\Delta t \to dy/dt$, and $\varepsilon_1, \varepsilon_2 \to 0$, giving the chain rule.*

2. **Derive the backpropagation equations for a Residual Network (ResNet) block: $y = x + F(x, W)$. Explain how skip connections affect gradient flow.**
   *Answer: A ResNet block: $y = x + F(x, W)$. During backpropagation: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left(I + \frac{\partial F}{\partial x}\right)$. The term $I$ (identity) from the skip connection ensures that the gradient can flow directly to the input without passing through $F$. This is critical for training very deep networks. Without the skip connection, the gradient is $\frac{\partial L}{\partial y} \cdot \frac{\partial F}{\partial x}$, which can vanish if $F$ has contracting properties. The skip connection adds a direct gradient path, preventing vanishing gradients. For a stack of $L$ ResNet blocks: $\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_{L+1}} \cdot \prod_{i=1}^L \left(I + \frac{\partial F_i}{\partial x_i}\right) = \frac{\partial L}{\partial x_{L+1}} \cdot \left(I + \sum_i \frac{\partial F_i}{\partial x_i} + \text{higher-order terms}\right)$. The identity term ensures gradients can propagate directly through all $L$ blocks.*

3. **Derive the adjoint sensitivity method for Neural ODEs: $\frac{dh}{dt} = f(h(t), t, \theta)$. Show how the chain rule is used to compute gradients through the ODE solver.**
   *Answer: In Neural ODEs, the forward pass solves $\frac{dh}{dt} = f(h(t), t, \theta)$ from $t_0$ to $t_1$ with initial state $h(t_0)$. The loss $L(h(t_1))$ depends on the final state. To compute gradients, define the adjoint $a(t) = \partial L / \partial h(t)$ (a row vector), which satisfies its own ODE: $\frac{da}{dt} = -a(t) \cdot \frac{\partial f}{\partial h}(h(t), t, \theta)$. This ODE is solved backward in time from $t_1$ to $t_0$ with initial condition $a(t_1) = \partial L / \partial h(t_1)$. The gradient with respect to parameters is: $\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t) \cdot \frac{\partial f}{\partial \theta}(h(t), t, \theta) dt$. This is the continuous analogue of the chain rule: instead of multiplying discrete Jacobians, we solve an ODE that accumulates gradient information continuously. The adjoint method requires solving three ODEs: the forward state, backward adjoint, and parameter gradient — all of which are computed by the ODE solver, and the chain rule is applied through each solver step.*

## Practice Problems

### Easy

1. Differentiate $h(x) = \cos(5x)$.
2. Find $h'(x)$ for $h(x) = (2x^3 - x)^5$.
3. Compute $\frac{d}{dx} e^{\sin x}$.
4. Differentiate $h(x) = \ln(x^2 + 1)$.
5. Find $\frac{d}{dx} \sqrt{1 + x^2}$.

### Medium

1. Let $z = f(x, y) = x^2 + y^2$, $x = t^2$, $y = t^3$. Find $dz/dt$.
2. Differentiate $h(x) = \sin^2(3x)$.
3. For $f(x, y) = e^{xy}$, $x = \cos t$, $y = \sin t$, find $df/dt$.
4. Backpropagate through one step: given $\hat{y} = wx + b$, $L = (\hat{y} - y)^2$, find $\partial L/\partial w$ and $\partial L/\partial b$.
5. Let $z = f(x, y)$ where $x = s^2 + t$, $y = s - t^2$. Express $\partial z/\partial s$ and $\partial z/\partial t$ in terms of $\partial z/\partial x$ and $\partial z/\partial y$.

### Hard

1. Derive the backpropagation equations for a 3-layer network with tanh activations and MSE loss. Write all gradients explicitly.
2. Prove Fa\`a di Bruno's formula for the $n$-th derivative of $f(g(x))$ (optional: just derive up to $n = 3$).
3. Implement the chain rule for a computation graph with branching: $a = f(x) = x^2$, $b = g(x) = e^x$, $c = h(a, b) = a + \sin(b)$. Compute $dc/dx$ and verify with direct substitution.

## Solutions

### Easy Solutions

**1.** $h'(x) = -\sin(5x) \cdot 5 = -5\sin(5x)$.

**2.** $h'(x) = 5(2x^3 - x)^4 \cdot (6x^2 - 1)$.

**3.** $\frac{d}{dx} e^{\sin x} = e^{\sin x} \cdot \cos x$.

**4.** $h'(x) = \frac{1}{x^2 + 1} \cdot 2x = \frac{2x}{x^2 + 1}$.

**5.** $\frac{d}{dx} (1 + x^2)^{1/2} = \frac{1}{2}(1 + x^2)^{-1/2} \cdot 2x = \frac{x}{\sqrt{1 + x^2}}$.

### Medium Solutions

**1.** $\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt} = (2x)(2t) + (2y)(3t^2) = 4xt + 6yt^2 = 4(t^2)(t) + 6(t^3)(t^2) = 4t^3 + 6t^5$.

**2.** $h'(x) = 2\sin(3x) \cdot \cos(3x) \cdot 3 = 6\sin(3x)\cos(3x) = 3\sin(6x)$.

**3.** $\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} = (ye^{xy})(-\sin t) + (xe^{xy})(\cos t) = e^{xy}(-y\sin t + x\cos t) = e^{\cos t \sin t}(-\sin t \cdot \sin t + \cos t \cdot \cos t) = e^{\cos t \sin t}(\cos^2 t - \sin^2 t) = e^{\cos t \sin t} \cos(2t)$.

**4.** $\partial L / \partial \hat{y} = 2(\hat{y} - y)$. $\partial L / \partial w = 2(\hat{y} - y) \cdot x$, $\partial L / \partial b = 2(\hat{y} - y) \cdot 1$.

**5.** $\frac{\partial z}{\partial s} = \frac{\partial z}{\partial x} \cdot 2s + \frac{\partial z}{\partial y} \cdot 1$. $\frac{\partial z}{\partial t} = \frac{\partial z}{\partial x} \cdot 1 + \frac{\partial z}{\partial y} \cdot (-2t)$.

### Hard Solutions

**1.** Forward: $z_1 = W_1 x + b_1$, $h_1 = \tanh(z_1)$, $z_2 = W_2 h_1 + b_2$, $h_2 = \tanh(z_2)$, $z_3 = W_3 h_2 + b_3$, $\hat{y} = z_3$, $L = \frac{1}{2}(\hat{y} - y)^2$. Backward: $\delta_3 = \hat{y} - y$, $\partial L / \partial W_3 = \delta_3 h_2^T$, $\partial L / \partial b_3 = \delta_3$. $\delta_2 = (W_3^T \delta_3) \odot \text{sech}^2(z_2)$, $\partial L / \partial W_2 = \delta_2 h_1^T$, $\partial L / \partial b_2 = \delta_2$. $\delta_1 = (W_2^T \delta_2) \odot \text{sech}^2(z_1)$, $\partial L / \partial W_1 = \delta_1 x^T$, $\partial L / \partial b_1 = \delta_1$.

**2.** For $n = 1$: $(f \circ g)'(x) = f'(g(x)) g'(x)$. For $n = 2$: $(f \circ g)''(x) = f''(g(x))(g'(x))^2 + f'(g(x)) g''(x)$. For $n = 3$: $(f \circ g)'''(x) = f'''(g(x))(g'(x))^3 + 3 f''(g(x)) g'(x) g''(x) + f'(g(x)) g'''(x)$. The general formula involves Bell polynomials: $(f \circ g)^{(n)}(x) = \sum_{k=1}^n f^{(k)}(g(x)) B_{n,k}(g'(x), g''(x), \dots)$.

**3.** By chain rule: $dc/dx = \partial c/\partial a \cdot da/dx + \partial c/\partial b \cdot db/dx = (1) \cdot (2x) + (\cos(b)) \cdot (e^x) = 2x + e^x \cos(e^x)$. Verification by direct substitution: $c(x) = x^2 + \sin(e^x)$. Then $c'(x) = 2x + \cos(e^x) \cdot e^x = 2x + e^x \cos(e^x)$. Matches.

## Related Concepts

- **Derivative** (MATH-055) — The chain rule is a rule for computing derivatives.
- **Partial Derivative** (MATH-056) — The multivariable chain rule uses partial derivatives.
- **Gradient** (MATH-058) — The gradient of a composition is computed via the chain rule.
- **Composite Function** (MATH-047) — The chain rule differentiates composite functions.
- **Limits** (MATH-053) — The chain rule is proved using limits.
- **Backpropagation** — Direct application of the chain rule to neural networks.

## Next Concepts

- **Directional Derivative** — Derivative in an arbitrary direction, generalising partial derivatives.
- **Jacobian Matrix** — Matrix of all partial derivatives of a vector-valued function.
- **Automatic Differentiation** — Systematic application of the chain rule for computing derivatives.
- **Neural ODEs** — Continuous-depth networks using the adjoint method (continuous chain rule).
- **Normalising Flows** — Invertible transformations using the chain rule for density estimation.

## Summary

The chain rule is the rule for differentiating compositions of functions. In single-variable calculus, $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$. In multivariable calculus, $\frac{d}{dt}f(x(t), y(t)) = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$. The chain rule lies at the heart of backpropagation — the algorithm that makes deep learning possible by efficiently computing gradients through multi-layer neural networks. Every gradient computation in modern machine learning is an application of the chain rule on a computational graph, with derivatives propagating backward from the loss to each parameter.

## Key Takeaways

- Single-variable chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$.
- Multivariable chain rule sums over all paths: $\frac{d}{dt}f(x(t), y(t)) = \sum_i \frac{\partial f}{\partial x_i} \frac{dx_i}{dt}$.
- Backpropagation IS the chain rule applied to neural networks — each layer's gradient is the product of downstream error and local derivative.
- The chain rule explains gradient flow: products of Jacobians give the overall gradient through deep networks.
- Skip connections (ResNet) add direct paths for gradient flow via the chain rule, preventing vanishing gradients.
- Automatic differentiation implements forward and reverse modes of the chain rule.
- For vector-valued functions, the chain rule involves matrix multiplication of Jacobians.
- The adjoint method (Neural ODEs) is the continuous version of the chain rule.
- Faa di Bruno's formula generalises the chain rule to higher-order derivatives.
- Understanding the chain rule is essential for designing, debugging, and optimising neural network architectures.
