# Concept: Partial Derivative

## Concept ID

MATH-056

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the partial derivative $\frac{\partial f}{\partial x}$ for a multivariable function $f(x, y, \dots)$ as the derivative with respect to one variable while holding others constant.
- Compute first-order and higher-order partial derivatives (including mixed partials $\frac{\partial^2 f}{\partial x \partial y}$).
- State and apply Clairaut's theorem on equality of mixed partials.
- Interpret partial derivatives as slopes in coordinate directions and as components of the gradient.
- Compute gradients of loss functions with respect to individual weights in neural networks.
- Understand the role of partial derivatives in backpropagation and parameter optimisation.

## Prerequisites

- Derivative (MATH-055) — partial derivatives extend the derivative concept to multiple variables.
- Limits (MATH-053) — the definition uses a limit.
- Functions of several variables — domain, range, graphing in 3D.
- Basic differentiation rules (product, quotient, chain rule) for single-variable functions.

## Definition

For a function $f(x, y, z, \dots)$ of several variables, the **partial derivative** with respect to $x$ is the derivative of $f$ with respect to $x$, treating all other variables as constants:

$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x + h, y, z, \dots) - f(x, y, z, \dots)}{h}$$

provided this limit exists. Alternative notations: $f_x$, $f_x(x, y, \dots)$, $\frac{\partial}{\partial x} f(x, y, \dots)$, $D_1 f$.

Similarly, the partial derivative with respect to $y$ is:

$$\frac{\partial f}{\partial y} = \lim_{k \to 0} \frac{f(x, y + k, z, \dots) - f(x, y, z, \dots)}{k}$$

**Geometric Interpretation:** For $f(x, y)$, $\frac{\partial f}{\partial x}(a, b)$ is the slope of the tangent line to the curve formed by intersecting the surface $z = f(x, y)$ with the plane $y = b$ at the point $(a, b, f(a, b))$. Similarly, $\frac{\partial f}{\partial y}(a, b)$ is the slope of the tangent line in the $x = a$ plane.

**Higher-Order Partial Derivatives:**
- Second-order: $\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$, $\frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right)$.
- Mixed partials: $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)$, $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$.

**Clairaut's Theorem (Schwarz's Theorem):** If the mixed partial derivatives $\frac{\partial^2 f}{\partial x \partial y}$ and $\frac{\partial^2 f}{\partial y \partial x}$ are continuous at a point, then they are equal at that point:

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

This means the order of differentiation does not matter for "nice" (sufficiently smooth) functions.

## Intuition

Imagine you are standing on a hilly terrain described by $f(x, y)$, where $x$ points east and $y$ points north. The partial derivative $\partial f / \partial x$ is the slope you would experience if you walked due east, keeping your north-south position fixed. Similarly, $\partial f / \partial y$ is the slope walking due north.

The key idea: to isolate the effect of changing just one variable, hold all others constant. This is exactly what we need in machine learning when we want to know how the loss changes when we adjust a single weight, while keeping all other weights fixed.

If $\frac{\partial f}{\partial x}(a, b) > 0$, the function increases as $x$ increases (moving east). If negative, it decreases. The magnitude tells us how sensitive $f$ is to changes in $x$ at that point.

Partial derivatives generalise the derivative to higher dimensions. While a single-variable function has one derivative at each point, a function of $n$ variables has $n$ partial derivatives, which are collected into the gradient vector.

## Why This Concept Matters

Partial derivatives are essential whenever we have functions of multiple variables — which is almost everywhere in science and engineering:

1. **Multivariable Optimisation:** Most real-world optimisation problems involve many variables. Partial derivatives tell us how the objective changes with respect to each variable individually.

2. **Gradient-Based Machine Learning:** Neural networks have thousands to billions of parameters. Training requires computing $\frac{\partial L}{\partial w_i}$ for every weight $w_i$ — each of these is a partial derivative holding all other weights constant.

3. **Physics:** Temperature $T(x, y, z, t)$ depends on position and time. The partial derivative $\partial T / \partial x$ is the temperature gradient in the $x$-direction, appearing in heat equation: $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$.

4. **Economics:** Production functions $Q(K, L)$ depend on capital $K$ and labour $L$. The marginal product of labour is $\partial Q / \partial L$ — how much output changes when adding one more worker.

5. **Differential Geometry:** Curvature, geodesics, and other geometric quantities involve partial derivatives of coordinate functions.

## Historical Background

Partial derivatives emerged naturally as calculus was extended from one to several variables. Jean le Rond d'Alembert (1717-1783) used partial derivatives in his work on fluid dynamics and the wave equation. Leonhard Euler (1707-1783) extensively developed the calculus of multiple variables, including partial differential equations.

Alexis Clairaut (1713-1765) discovered the theorem on equality of mixed partials in 1740 while working on the shape of the Earth. He realised that for "well-behaved" functions, the order of differentiation does not matter — a result that greatly simplifies multivariable calculus.

In the 19th century, Carl Gustav Jacob Jacobi (1804-1851) formalised the notation $\frac{\partial f}{\partial x}$ for partial derivatives, which became standard. Hermann Schwarz (1843-1921) proved the rigorous version of Clairaut's theorem requiring continuity of mixed partials.

The theory of partial differential equations (PDEs) — equations involving partial derivatives — became a major field in the 19th and 20th centuries, with applications to heat flow, wave propagation, quantum mechanics, and general relativity. The Navier-Stokes equations (fluid dynamics), Maxwell's equations (electromagnetism), and the Schr\"odinger equation (quantum mechanics) are all PDEs.

## Real World Examples

**Example 1: Temperature Distribution.** A metal plate has temperature $T(x, y) = 100 - x^2 - 2y^2$ degrees Celsius at point $(x, y)$. The partial derivative $\partial T / \partial x = -2x$ tells us how temperature changes moving east. At $(1, 1)$, $\partial T / \partial x = -2$°C per unit distance — temperature decreases as we move east. $\partial T / \partial y = -4y$, so at $(1, 1)$, temperature decreases at 4°C per unit moving north. The plate is hottest at the origin.

**Example 2: Production Function.** A factory's output is $Q(K, L) = 10 K^{0.3} L^{0.7}$ where $K$ is capital (in dollars) and $L$ is labour (in hours). The marginal product of labour is $\partial Q / \partial L = 7 K^{0.3} L^{-0.3} = 7 (K/L)^{0.3}$. With $K = 1,000,000$ and $L = 1000$, $\partial Q / \partial L = 7(1000)^{0.3} \approx 7(7.94) \approx 55.6$ units per additional labour hour.

**Example 3: Ideal Gas Law.** Pressure $P = nRT / V$ depends on temperature $T$ and volume $V$. $\partial P / \partial T = nR/V$ tells us how pressure changes with temperature at constant volume. $\partial P / \partial V = -nRT/V^2$ tells us how pressure changes with volume at constant temperature (always negative — pressure decreases as volume increases).

**Example 4: Stock Option Pricing.** The Black-Scholes model gives option price $C(S, t)$ depending on stock price $S$ and time $t$. $\partial C / \partial S$ (Delta) measures sensitivity to stock price — a key quantity for hedging. $\partial C / \partial t$ (Theta) measures time decay.

**Example 5: Gradient of Loss.** For linear regression with MSE loss $L = \frac{1}{N} \sum_{i=1}^N (w_0 + w_1 x_i - y_i)^2$, the partial derivatives are: $\partial L / \partial w_0 = \frac{2}{N} \sum (w_0 + w_1 x_i - y_i)$ and $\partial L / \partial w_1 = \frac{2}{N} \sum (w_0 + w_1 x_i - y_i) x_i$. These are used in the normal equations and gradient descent.

## AI/ML Relevance

Partial derivatives are the fundamental computational building block of deep learning:

**1. Gradients of Loss w.r.t. Each Parameter.** During training, we compute $\frac{\partial L}{\partial w_i}$ for every weight $w_i$ in the network. Each is a partial derivative: we differentiate $L$ with respect to $w_i$ while holding all other weights constant. This tells us how much a tiny nudge to $w_i$ would affect the loss.

**2. Backpropagation Computes All Partial Derivatives.** Backpropagation efficiently computes all $\frac{\partial L}{\partial w_i}$ using the chain rule. For a network with $n$ parameters, naive differentiation would require $O(n^2)$ operations, but backpropagation computes all $n$ partial derivatives in $O(n)$ time.

**3. Partial Derivatives in Convolutional Layers.** For a convolutional layer with kernel $K$ and input $X$, the output is $Y = X * K$. The partial derivative $\partial L / \partial K_{ij}$ tells us how to update each kernel element. This is computed by convolving the input with the upstream gradient.

**4. Gradient of Loss w.r.t. Inputs.** In adversarial machine learning, we compute $\partial L / \partial x$ (the partial derivative of the loss with respect to the input) to find perturbations that fool the network:
$$x_{\text{adv}} = x + \varepsilon \cdot \text{sign}(\nabla_x L(x, y))$$

**5. Hessian Matrix.** The Hessian $H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}$ contains second-order partial derivatives. It measures curvature of the loss landscape and is used in:
- Second-order optimisation (Newton's method, natural gradient).
- Analysing convergence properties.
- Understanding saddle points vs. local minima.

**6. Mixed Partials in Physics-Informed Neural Networks.** PINNs solve PDEs by minimising:
$$L = \|u_{\text{NN}} - u_{\text{data}}\|^2 + \left\|\frac{\partial u_{\text{NN}}}{\partial t} - \mathcal{N}[u_{\text{NN}}]\right\|^2$$
The PDE residual involves partial derivatives $\partial u / \partial t$, $\partial^2 u / \partial x^2$, etc., computed via automatic differentiation.

**7. Gradient of Regularisation Terms.** L2 regularisation adds $\frac{\lambda}{2} \sum_i w_i^2$ to the loss. Its partial derivative is $\partial / \partial w_i [\frac{\lambda}{2} w_i^2] = \lambda w_i$, which is why L2 regularisation is also called "weight decay."

**8. Batch Normalisation Gradients.** Batch norm involves partial derivatives through the normalisation transformation:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_B^2 + \varepsilon}} + \text{terms from } \frac{\partial \mu_B}{\partial x_i}, \frac{\partial \sigma_B^2}{\partial x_i}$$

## Mathematical Explanation

**Computing Partial Derivatives:** To compute $\frac{\partial f}{\partial x}$ for $f(x, y, \dots)$, treat all variables except $x$ as constants and differentiate with respect to $x$ using all the usual single-variable rules.

**Example with $f(x, y) = x^2 y + 3xy^2 + \sin(xy)$:**
- $\frac{\partial f}{\partial x}$: treat $y$ as constant. $2x \cdot y + 3y^2 \cdot 1 + \cos(xy) \cdot y = 2xy + 3y^2 + y\cos(xy)$.
- $\frac{\partial f}{\partial y}$: treat $x$ as constant. $x^2 \cdot 1 + 3x \cdot 2y + \cos(xy) \cdot x = x^2 + 6xy + x\cos(xy)$.

**Higher-Order Partial Derivatives:**
For $f(x, y) = x^3 y^2 + e^x \sin y$:
- $f_x = 3x^2 y^2 + e^x \sin y$
- $f_y = 2x^3 y + e^x \cos y$
- $f_{xx} = 6x y^2 + e^x \sin y$
- $f_{yy} = 2x^3 - e^x \sin y$
- $f_{xy} = 6x^2 y + e^x \cos y$ (differentiate $f_x$ w.r.t. $y$)
- $f_{yx} = 6x^2 y + e^x \cos y$ (differentiate $f_y$ w.r.t. $x$)
By Clairaut's theorem, $f_{xy} = f_{yx}$.

**Clairaut's Theorem (Formal Statement):** Let $f$ be defined on an open set $U \subset \mathbb{R}^2$. If the mixed partial derivatives $f_{xy}$ and $f_{yx}$ exist and are continuous on $U$, then $f_{xy} = f_{yx}$ at all points in $U$.

**Counterexample (where Clairaut fails):**
$$f(x, y) = \begin{cases} \frac{x^3 y - x y^3}{x^2 + y^2} & (x, y) \neq (0, 0) \\ 0 & (x, y) = (0, 0) \end{cases}$$
At $(0, 0)$: $f_x(0, 0) = 0$, $f_y(0, 0) = 0$, but $f_{xy}(0, 0) = 1$ and $f_{yx}(0, 0) = -1$. The mixed partials are not continuous at $(0, 0)$, so Clairaut's theorem does not apply.

**Chain Rule for Partial Derivatives (will be covered more in MATH-057):**
If $z = f(x(t), y(t))$, then $\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$.

## Formula(s)

**Definition:**
$$\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}$$

**Gradient (vector of partial derivatives):**
$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

**Clairaut's Theorem:**
$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} \quad \text{(if continuous)}$$

**Chain Rule for $f(x(t), y(t))$:**
$$\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

**Hessian Matrix:**
$$H(f) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**Gradient Descent Update (per parameter):**
$$\theta_i^{(t+1)} = \theta_i^{(t)} - \alpha \frac{\partial L}{\partial \theta_i}$$

## Properties

1. **Linearity:** $\frac{\partial}{\partial x}(af + bg) = a \frac{\partial f}{\partial x} + b \frac{\partial g}{\partial x}$.

2. **Product Rule:** $\frac{\partial}{\partial x}(f g) = \frac{\partial f}{\partial x} g + f \frac{\partial g}{\partial x}$.

3. **Quotient Rule:** $\frac{\partial}{\partial x}\left(\frac{f}{g}\right) = \frac{f_x g - f g_x}{g^2}$.

4. **Equality of Mixed Partials (Clairaut):** For $C^2$ functions, the order of differentiation in mixed partials does not matter.

5. **Partial Derivative of a Constant:** If $f$ does not depend on $x$, then $\partial f / \partial x = 0$.

6. **Geometric Meaning:** $\partial f / \partial x$ is the slope in the $x$-direction; $\partial f / \partial y$ is the slope in the $y$-direction.

7. **Total vs. Partial:** The partial derivative $\partial f / \partial x$ differs from the total derivative $df/dx$ when $f$ depends on other variables that themselves depend on $x$.

8. **Hessian Symmetry:** By Clairaut's theorem, the Hessian matrix of a $C^2$ function is symmetric: $H_{ij} = H_{ji}$.

9. **Critical Points:** At a critical point, all first partial derivatives are zero: $\nabla f = \mathbf{0}$.

10. **Second Derivative Test:** The nature of a critical point (min, max, saddle) is determined by the eigenvalues of the Hessian matrix.

## Step-by-Step Worked Examples

### Example 1: Basic Partial Derivatives

Compute all first-order partial derivatives of $f(x, y, z) = x^2 y + y e^z + z \sin x$.

**Step 1:** $\partial f / \partial x$: treat $y$ and $z$ as constants.
$$\frac{\partial f}{\partial x} = 2xy + 0 + z \cos x = 2xy + z \cos x$$

**Step 2:** $\partial f / \partial y$: treat $x$ and $z$ as constants.
$$\frac{\partial f}{\partial y} = x^2 + e^z + 0 = x^2 + e^z$$

**Step 3:** $\partial f / \partial z$: treat $x$ and $y$ as constants.
$$\frac{\partial f}{\partial z} = 0 + y e^z + \sin x = y e^z + \sin x$$

**Answer:**
$$f_x = 2xy + z \cos x, \quad f_y = x^2 + e^z, \quad f_z = y e^z + \sin x$$

### Example 2: Mixed Partials and Clairaut's Theorem

For $f(x, y) = x^3 y^2 + \ln(x + y)$, show $f_{xy} = f_{yx}$.

**Step 1:** Compute $f_x = 3x^2 y^2 + \frac{1}{x+y}$.
**Step 2:** Compute $f_y = 2x^3 y + \frac{1}{x+y}$.
**Step 3:** Compute $f_{xy} = \frac{\partial}{\partial y}[f_x] = 6x^2 y - \frac{1}{(x+y)^2}$.
**Step 4:** Compute $f_{yx} = \frac{\partial}{\partial x}[f_y] = 6x^2 y - \frac{1}{(x+y)^2}$.
**Step 5:** $f_{xy} = f_{yx}$, confirming Clairaut's theorem.

**Answer:** $f_{xy} = f_{yx} = 6x^2 y - \frac{1}{(x+y)^2}$.

### Example 3: Partial Derivative in Gradient Descent for Linear Regression

Consider MSE loss $L = \frac{1}{2}(w_1 x + w_0 - y)^2$ for one data point $(x, y)$. Compute $\partial L / \partial w_1$ and $\partial L / \partial w_0$.

**Step 1:** Let $u = w_1 x + w_0 - y$, so $L = \frac{1}{2} u^2$.
**Step 2:** $\partial L / \partial u = u = w_1 x + w_0 - y$.
**Step 3:** $\partial u / \partial w_1 = x$, $\partial u / \partial w_0 = 1$.
**Step 4:** Chain rule:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial w_1} = (w_1 x + w_0 - y) \cdot x$$
$$\frac{\partial L}{\partial w_0} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial w_0} = w_1 x + w_0 - y$$

**Answer:**
$$\frac{\partial L}{\partial w_1} = x(w_1 x + w_0 - y), \quad \frac{\partial L}{\partial w_0} = w_1 x + w_0 - y$$

### Example 4: Higher-Order Partial Derivatives

Compute $f_{xx}$, $f_{yy}$, $f_{xy}$, $f_{yx}$ for $f(x, y) = e^{2x} \cos y$.

**Step 1:** $f_x = 2e^{2x} \cos y$, $f_y = -e^{2x} \sin y$.

**Step 2:** $f_{xx} = \frac{\partial}{\partial x}[2e^{2x} \cos y] = 4e^{2x} \cos y$.

**Step 3:** $f_{yy} = \frac{\partial}{\partial y}[-e^{2x} \sin y] = -e^{2x} \cos y$.

**Step 4:** $f_{xy} = \frac{\partial}{\partial y}[2e^{2x} \cos y] = -2e^{2x} \sin y$.

**Step 5:** $f_{yx} = \frac{\partial}{\partial x}[-e^{2x} \sin y] = -2e^{2x} \sin y$.

**Answer:** $f_{xx} = 4e^{2x} \cos y$, $f_{yy} = -e^{2x} \cos y$, $f_{xy} = f_{yx} = -2e^{2x} \sin y$.

### Example 5: Partial Derivatives of a Neural Network Loss

Consider a 2-layer network with 1 hidden neuron: $\hat{y} = w_2 \sigma(w_1 x + b_1) + b_2$ with MSE loss $L = \frac{1}{2}(\hat{y} - y)^2$. Compute $\partial L / \partial w_1$.

**Step 1:** Define intermediate variables.
$$z = w_1 x + b_1, \quad h = \sigma(z), \quad \hat{y} = w_2 h + b_2$$

**Step 2:** Apply chain rule (backpropagation).
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

**Step 3:** Compute each factor.
$$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$
$$\frac{\partial \hat{y}}{\partial h} = w_2$$
$$\frac{\partial h}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) \quad \text{(for sigmoid)}$$
$$\frac{\partial z}{\partial w_1} = x$$

**Step 4:** Combine.
$$\frac{\partial L}{\partial w_1} = (\hat{y} - y) \cdot w_2 \cdot \sigma(w_1 x + b_1)(1 - \sigma(w_1 x + b_1)) \cdot x$$

**Answer:** $\frac{\partial L}{\partial w_1} = (\hat{y} - y) w_2 \sigma(w_1 x + b_1)(1 - \sigma(w_1 x + b_1)) x$.

## Visual Interpretation

**Surface and Tangent Lines:**
The graph of $z = f(x, y)$ is a surface in 3D. At a point $(x_0, y_0, z_0)$:
- $\partial f / \partial x$ is the slope of the curve formed by slicing the surface with the plane $y = y_0$.
- $\partial f / \partial y$ is the slope of the curve formed by slicing the surface with the plane $x = x_0$.

These two slopes define the tangent plane at the point.

**Contour Map Interpretation:**
A contour map shows level curves $f(x, y) = c$. The partial derivatives relate to the spacing of contours:
- If $\partial f / \partial x$ is large, contours are closely spaced in the $x$-direction (steep terrain).
- If $\partial f / \partial x = 0$, the contour is locally horizontal (level in the $x$-direction).

**Gradient Components:**
The gradient vector $\nabla f = (f_x, f_y)$ points in the direction of steepest ascent. On a contour map, the gradient is perpendicular to level curves. Each component $f_x$ and $f_y$ represents the slope in the $x$ and $y$ directions respectively.

**Visualisation for Backpropagation:**
```
    ∂L/∂w1     ∂L/∂w2     ∂L/∂w3
      ↑          ↑          ↑
    ∂h1/∂w1   ∂h2/∂w2   ∂h3/∂w3
      ↑          ↑          ↑
L ←── h1  ←──── h2  ←──── h3  ←── ... ←── x
      ↑          ↑          ↑
    ∂L/∂h1    ∂L/∂h2    ∂L/∂h3
```
Each partial derivative is computed locally and multiplied along the backward pass.

## Common Mistakes

1. **Treating all other variables as zero instead of constants.** When computing $\partial / \partial x$ of $f(x, y) = x^2 y$, the correct derivative is $2xy$, not $2x$ (which would treat $y$ as zero). Remember: constant means keep it as-is, not zero.

2. **Confusing partial derivatives with total derivatives.** If $f(x, y)$ and $y = y(x)$, the total derivative $df/dx = \partial f/\partial x + (\partial f/\partial y)(dy/dx)$, not just $\partial f/\partial x$. This confusion is common in backpropagation when variables are shared.

3. **Forgetting the chain rule for functions inside compositions.** For $f(x, y) = \sin(xy)$, $\partial f/\partial x = \cos(xy) \cdot y$, not $\cos(xy)$. The inner derivative $y$ (from $\partial(xy)/\partial x$) is required.

4. **Assuming mixed partials always commute.** Clairaut's theorem requires continuity of the mixed partials. Counterexamples exist where $f_{xy} \neq f_{yx}$ when the mixed partials are not continuous, though these are rare in practice.

5. **Ignoring the chain rule in backpropagation.** When computing $\partial L / \partial W$ for a multi-layer network, each layer's derivative depends on upstream gradients via the chain rule. Simply differentiating each layer independently gives incorrect results.

6. **Writing $\frac{df}{dx}$ for a multivariable function.** The notation $\partial f / \partial x$ (partial derivative) is used for multivariable functions to distinguish from the total derivative $df/dx$, which includes dependencies through other variables.

7. **Not accounting for shared parameters.** If the same weight $w$ appears in multiple terms of the loss, $\partial L / \partial w$ is the sum of contributions from each term. Forgetting to sum over all contributions is a common error.

8. **Thinking the gradient is a scalar.** For a function of $n$ variables, the gradient is an $n$-dimensional vector, not a single number. Each component is a partial derivative.

9. **Applying power rule when exponent depends on the variable being differentiated.** For $f(x, y) = y^x$, $\partial f / \partial x$ uses logarithmic differentiation: $f_x = y^x \ln y$, not $x y^{x-1}$ (which would incorrectly treat $x$ as the exponent constant).

10. **Confusing the order in higher-order mixed partials.** $\frac{\partial^2 f}{\partial x \partial y}$ means differentiate with respect to $y$ first, then $x$: $\frac{\partial}{\partial x}(\frac{\partial f}{\partial y})$. Some textbooks use the opposite convention.

## Interview Questions

### Beginner

1. **What is a partial derivative?**
   *Answer: A partial derivative of a multivariable function $f(x, y, \dots)$ is the derivative with respect to one variable, treating all other variables as constants. For example, $\partial f / \partial x = \lim_{h\to 0} (f(x+h, y) - f(x, y))/h$.*

2. **Compute $\partial f / \partial x$ for $f(x, y) = x^2 y + y^3$.**
   *Answer: $\partial f / \partial x = 2xy$ (treat $y$ as constant).*

3. **What is Clairaut's theorem?**
   *Answer: Clairaut's theorem states that if the mixed partial derivatives $f_{xy}$ and $f_{yx}$ are continuous at a point, then they are equal: $\partial^2 f / \partial x \partial y = \partial^2 f / \partial y \partial x$.*

4. **How do you compute $\partial f / \partial y$ for $f(x, y) = e^{xy}$?**
   *Answer: Treat $x$ as constant: $\partial f / \partial y = e^{xy} \cdot x = x e^{xy}$.*

5. **What does $\partial f / \partial x (a, b)$ represent geometrically?**
   *Answer: It is the slope of the tangent line to the curve formed by intersecting the surface $z = f(x, y)$ with the plane $y = b$ at the point $(a, b, f(a, b))$.*

### Intermediate

1. **Explain how partial derivatives are used in backpropagation.**
   *Answer: Backpropagation computes the partial derivative of the loss with respect to each network parameter. For a parameter $w$, the chain rule decomposes $\partial L / \partial w$ into a product of partial derivatives along the computational path from $w$ to $L$. Each step is a local partial derivative (e.g., $\partial h / \partial z$ for activation, $\partial z / \partial w$ for linear transformation). These are combined via the chain rule, and gradients are accumulated backward through the network.*

2. **For $f(x, y) = \sin(xy)$, compute $f_x$, $f_y$, $f_{xx}$, $f_{xy}$, and verify $f_{xy} = f_{yx}$.**
   *Answer: $f_x = y\cos(xy)$, $f_y = x\cos(xy)$. $f_{xx} = -y^2\sin(xy)$. $f_{xy} = \cos(xy) - xy\sin(xy)$ (differentiate $f_x$ w.r.t. $y$). $f_{yx} = \cos(xy) - xy\sin(xy)$ (differentiate $f_y$ w.r.t. $x$). Indeed $f_{xy} = f_{yx}$.*

3. **What is the Hessian matrix and why is it important in optimisation?**
   *Answer: The Hessian $H_{ij} = \partial^2 L / \partial \theta_i \partial \theta_j$ is the matrix of second-order partial derivatives. It characterises the local curvature of the loss landscape. Positive definite Hessian at a critical point indicates a local minimum; negative definite indicates a local maximum; indefinite indicates a saddle point. Newton's method uses the Hessian for faster convergence: $\theta_{t+1} = \theta_t - H^{-1} \nabla L$. However, computing the full Hessian is $O(n^2)$ in parameters, so approximations (L-BFGS, diagonal Hessian) are used in practice.*

4. **Compute $\frac{\partial^2 f}{\partial x \partial y}$ for $f(x, y) = x^y$.**
   *Answer: Write $f = e^{y \ln x}$. $f_x = e^{y \ln x} \cdot (y/x) = y x^{y-1}$. Then $f_{xy} = \frac{\partial}{\partial y}[y x^{y-1}]$. Using logarithmic differentiation: $\frac{\partial}{\partial y}[y x^{y-1}] = x^{y-1} + y x^{y-1} \ln x = x^{y-1}(1 + y \ln x)$.*

5. **Why is it important that we can compute partial derivatives efficiently for neural networks with millions of parameters?**
   *Answer: Efficient computation is critical because: (1) Each training step requires gradients for all parameters. (2) Backpropagation computes all $n$ partial derivatives in $O(n)$ time (same order as forward pass). (3) Without efficient computation, training networks with millions of parameters would be computationally infeasible. (4) Automatic differentiation frameworks (PyTorch, TensorFlow, JAX) implement reverse-mode AD, which computes all partial derivatives in a single backward pass.*

### Advanced

1. **State and prove the chain rule for a function $f(x(t), y(t))$ using the definition of partial derivatives.**
   *Answer: The multivariable chain rule: $\frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$. Proof: $\frac{df}{dt} = \lim_{h\to 0} \frac{f(x(t+h), y(t+h)) - f(x(t), y(t))}{h}$. Add and subtract $f(x(t), y(t+h))$ in the numerator: $= \lim_{h\to 0} \frac{f(x(t+h), y(t+h)) - f(x(t), y(t+h))}{h} + \lim_{h\to 0} \frac{f(x(t), y(t+h)) - f(x(t), y(t))}{h}$. In the first term, define $\Delta x = x(t+h) - x(t)$. As $h\to 0$, $\Delta x \to 0$, and the term becomes $\frac{\partial f}{\partial x}(x(t), y(t)) \cdot \frac{dx}{dt}$. The second term is $\frac{\partial f}{\partial y}(x(t), y(t)) \cdot \frac{dy}{dt}$, provided $f$ is differentiable.*

2. **Derive the update equations for the parameters of a single-neuron network with cross-entropy loss and sigmoid activation, showing all partial derivatives.**
   *Answer: Let $z = w^T x + b$, $a = \sigma(z) = 1/(1+e^{-z})$, and loss $L = -y \ln a - (1-y) \ln(1-a)$ for binary classification. Compute $\partial L / \partial a = -\frac{y}{a} + \frac{1-y}{1-a} = \frac{a-y}{a(1-a)}$. $\partial a / \partial z = a(1-a)$. So $\partial L / \partial z = \frac{a-y}{a(1-a)} \cdot a(1-a) = a - y$. Then $\partial L / \partial w_i = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_i} = (a-y) x_i$. $\partial L / \partial b = (a-y) \cdot 1 = a - y$. Thus the gradient descent updates are: $w_i \leftarrow w_i - \alpha (a-y) x_i$, $b \leftarrow b - \alpha (a-y)$. This is remarkably simple: the update for each weight is proportional to the prediction error $(a-y)$ times the input $x_i$.*

3. **Construct a function where $f_{xy}(0, 0) \neq f_{yx}(0, 0)$. Explain why Clairaut's theorem fails.**
   *Answer: The standard counterexample is: $f(x, y) = \begin{cases} \frac{x^3 y - x y^3}{x^2 + y^2} & (x, y) \neq (0,0) \\ 0 & (x, y) = (0,0) \end{cases}$. At $(0,0)$: $f_x(0,0) = \lim_{h\to0} (f(h,0)-0)/h = \lim_{h\to0} 0/h = 0$. Similarly $f_y(0,0) = 0$. Now $f_x(x,y) = \frac{3x^2 y - y^3}{x^2+y^2} - \frac{2x(x^3 y - xy^3)}{(x^2+y^2)^2}$ for $(x,y)\neq(0,0)$. Then $f_{xy}(0,0) = \lim_{k\to0} (f_x(0,k)-f_x(0,0))/k = \lim_{k\to0} ((-k^3)/(k^2) - 0)/k = \lim_{k\to0} (-k)/k = -1$. Similarly $f_{yx}(0,0) = \lim_{h\to0} (f_y(h,0)-f_y(0,0))/h = \lim_{h\to0} (h^3/h^2 - 0)/h = \lim_{h\to0} h/h = 1$. So $f_{xy}(0,0) = -1 \neq 1 = f_{yx}(0,0)$. Clairaut's theorem fails because the mixed partials are not continuous at $(0,0)$.*

## Practice Problems

### Easy

1. Compute $\partial f / \partial x$ and $\partial f / \partial y$ for $f(x, y) = 3x^2 y - 2xy^3$.
2. Find $\partial f / \partial z$ for $f(x, y, z) = x^2 + y^2 + z^2$.
3. Compute $\partial f / \partial x$ for $f(x, y) = \ln(x^2 + y^2)$.
4. Find $\partial f / \partial y$ for $f(x, y) = e^{x} \cos(y)$.
5. Compute $\partial^2 f / \partial x^2$ for $f(x, y) = x^3 y + x^2$.

### Medium

1. Compute all second-order partial derivatives of $f(x, y) = x^2 e^y + y \sin x$.
2. For $f(x, y) = \tan^{-1}(y/x)$, compute $f_x$ and $f_y$.
3. Find $\partial L / \partial w_2$ for $L = \frac{1}{2}(w_2 \sigma(w_1 x) - y)^2$.
4. Compute $f_{xy}$ for $f(x, y) = \cos(x^2 y)$.
5. Show that $f(x, y) = e^{ax} \cos(by)$ satisfies Laplace's equation $\frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} = 0$ for a particular relationship between $a$ and $b$.

### Hard

1. For $f(x, y) = \int_x^y e^{-t^2} dt$, compute $\partial f / \partial x$ and $\partial f / \partial y$ (use the Fundamental Theorem of Calculus).
2. Derive the backpropagation equations for a 3-layer network with ReLU activations and MSE loss. Compute all partial derivatives of the loss with respect to each weight.
3. Prove that if $f$ is $C^2$ (twice continuously differentiable) on an open set $U$, then the Hessian matrix is symmetric, i.e., $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$.

## Solutions

### Easy Solutions

**1.** $f_x = 6xy - 2y^3$, $f_y = 3x^2 - 6xy^2$.

**2.** $\partial f / \partial z = 2z$.

**3.** $f_x = \frac{2x}{x^2 + y^2}$.

**4.** $f_y = -e^x \sin y$.

**5.** $f_{xx} = \frac{\partial}{\partial x}[3x^2 y + 2x] = 6xy + 2$.

### Medium Solutions

**1.** $f_x = 2xe^y + y\cos x$, $f_y = x^2 e^y + \sin x$. $f_{xx} = 2e^y - y\sin x$, $f_{xy} = 2xe^y + \cos x$, $f_{yx} = 2xe^y + \cos x$, $f_{yy} = x^2 e^y$.

**2.** $f_x = \frac{1}{1 + (y/x)^2} \cdot (-y/x^2) = \frac{-y}{x^2 + y^2}$. $f_y = \frac{1}{1 + (y/x)^2} \cdot (1/x) = \frac{x}{x^2 + y^2}$.

**3.** Let $u = w_1 x$, $a = \sigma(u)$, $\hat{y} = w_2 a$. Then $L = \frac{1}{2}(\hat{y} - y)^2$. $\partial L / \partial \hat{y} = \hat{y} - y$, $\partial \hat{y} / \partial w_2 = a$, so $\partial L / \partial w_2 = (\hat{y} - y) \cdot a = (\hat{y} - y) \sigma(w_1 x)$.

**4.** $f_x = -2xy \sin(x^2 y)$. $f_{xy} = \frac{\partial}{\partial y}[-2xy \sin(x^2 y)] = -2x \sin(x^2 y) - 2xy \cos(x^2 y) \cdot x^2 = -2x \sin(x^2 y) - 2x^3 y \cos(x^2 y)$.

**5.** $f_x = ae^{ax} \cos(by)$, $f_{xx} = a^2 e^{ax} \cos(by)$. $f_y = -b e^{ax} \sin(by)$, $f_{yy} = -b^2 e^{ax} \cos(by)$. Laplace: $f_{xx} + f_{yy} = (a^2 - b^2) e^{ax} \cos(by) = 0$ for all $x, y$ iff $a^2 = b^2$, i.e., $a = \pm b$.

### Hard Solutions

**1.** By the Fundamental Theorem of Calculus, for $F(t)$ an antiderivative of $e^{-t^2}$: $f(x, y) = F(y) - F(x)$. Then $f_x = -F'(x) = -e^{-x^2}$, $f_y = F'(y) = e^{-y^2}$.

**2.** Let the network be: $z_1 = W_1 x + b_1$, $h_1 = \text{ReLU}(z_1)$, $z_2 = W_2 h_1 + b_2$, $h_2 = \text{ReLU}(z_2)$, $z_3 = W_3 h_2 + b_3$, $\hat{y} = z_3$, $L = \frac{1}{2}\|\hat{y} - y\|^2$. Backpropagating: $\delta_3 = \hat{y} - y$, $\partial L / \partial W_3 = \delta_3 h_2^T$, $\partial L / \partial b_3 = \delta_3$. $\delta_2 = (W_3^T \delta_3) \odot \mathbf{1}_{z_2 > 0}$ (ReLU derivative), $\partial L / \partial W_2 = \delta_2 h_1^T$, $\partial L / \partial b_2 = \delta_2$. $\delta_1 = (W_2^T \delta_2) \odot \mathbf{1}_{z_1 > 0}$, $\partial L / \partial W_1 = \delta_1 x^T$, $\partial L / \partial b_1 = \delta_1$.

**3.** For $i \neq j$, consider the second-order difference: $\Delta_{ij} = f(x + h e_i + h e_j) - f(x + h e_i) - f(x + h e_j) + f(x)$. Using the mean value theorem twice: $\Delta_{ij} = h^2 f_{xy}(\xi_{ij})$ for some point $\xi_{ij}$. Taking the limit $h \to 0$ and using continuity of $f_{xy}$ and $f_{yx}$ gives equality. More formally, from the definition: $f_{x_i x_j}(x) = \lim_{h\to 0} \lim_{k\to 0} \frac{f(x+he_i+ke_j) - f(x+he_i) - f(x+ke_j) + f(x)}{hk}$. The symmetry of the numerator in $h$ and $k$ implies $f_{x_i x_j} = f_{x_j x_i}$ when the limits can be interchanged, which is justified by the continuity of mixed partials (Clairaut's theorem).

## Related Concepts

- **Derivative** (MATH-055) — Partial derivatives generalise derivatives to multiple variables.
- **Gradient** (MATH-058) — The vector of all first-order partial derivatives.
- **Chain Rule** (MATH-057) — Essential for computing partial derivatives through compositions.
- **Limits** (MATH-053) — The definition of a partial derivative is a limit.
- **Continuity** (MATH-054) — Clairaut's theorem requires continuity of mixed partials.
- **Linear Transformation** (MATH-036) — The gradient is a linear transformation (Jacobian).

## Next Concepts

- **Directional Derivative** — The derivative in an arbitrary direction (not just coordinate axes).
- **Jacobian Matrix** — The matrix of all first-order partial derivatives of a vector-valued function.
- **Hessian Matrix** — The matrix of second-order partial derivatives for curvature analysis.
- **Partial Differential Equations** — Equations involving partial derivatives, modelling physical phenomena.
- **Automatic Differentiation** — Computational framework that computes partial derivatives efficiently.

## Summary

Partial derivatives extend the concept of differentiation to functions of several variables. For $f(x, y, \dots)$, the partial derivative $\partial f / \partial x$ is computed by treating all variables except $x$ as constants and differentiating using standard rules. Higher-order partial derivatives include mixed partials, which are equal under Clairaut's theorem when continuous. The gradient $\nabla f$ collects all first-order partial derivatives into a vector. In AI/ML, partial derivatives are the core of gradient-based optimisation: every parameter update in neural network training relies on computing $\partial L / \partial w_i$ via backpropagation. The Hessian matrix of second-order partial derivatives provides curvature information used in advanced optimisation methods.

## Key Takeaways

- $\frac{\partial f}{\partial x}$ is the derivative w.r.t. $x$ holding other variables constant.
- Treat all other variables as constants when computing partial derivatives.
- The gradient $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)$ collects all first partials.
- Mixed partials $f_{xy}$ and $f_{yx}$ are equal when continuous (Clairaut's theorem).
- The Hessian matrix $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ is symmetric for $C^2$ functions.
- Backpropagation computes all $\partial L / \partial w_i$ via the chain rule in $O(n)$ time.
- Each gradient descent step updates: $\theta_i \leftarrow \theta_i - \alpha \cdot \partial L / \partial \theta_i$.
- Second-order methods (Newton) use the Hessian for curvature-aware updates.
- Automatic differentiation frameworks compute partial derivatives efficiently.
- Partial derivatives are essential for understanding and implementing machine learning algorithms.
