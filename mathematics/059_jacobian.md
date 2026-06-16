# Concept: Jacobian Matrix and Determinant

## Concept ID

MATH-059

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define the Jacobian matrix as a matrix of all first-order partial derivatives of a vector-valued function
- Compute the Jacobian matrix and determinant for transformations between coordinate systems
- Apply the Jacobian determinant to change of variables in multiple integrals
- Analyze the Jacobian's role in neural network sensitivity analysis and backpropagation
- Understand the use of Jacobians in normalising flows and invertible neural networks
- Interpret the geometric meaning of the Jacobian as a local linear approximation

## Prerequisites

- Multivariable calculus (partial derivatives, gradients)
- Basic linear algebra (matrix multiplication, determinants)
- Vector-valued functions and their properties
- Coordinate systems (Cartesian, polar, spherical)
- Change of variables in single-variable integration

## Definition

Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ be a vector-valued function written as $\mathbf{f}(\mathbf{x}) = (f_1(x_1, x_2, \dots, x_n), f_2(x_1, x_2, \dots, x_n), \dots, f_m(x_1, x_2, \dots, x_n))$, where each $f_i$ is a real-valued function of $n$ variables. The **Jacobian matrix** of $\mathbf{f}$ is an $m \times n$ matrix whose $(i,j)$ entry is the partial derivative of $f_i$ with respect to $x_j$:

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[4pt]
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[4pt]
\vdots & \vdots & \ddots & \vdots \\[4pt]
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}_{m \times n}
$$

When $m = n$ (square matrix), the **Jacobian determinant** (often simply called the Jacobian) is the determinant of this matrix, denoted $\det(\mathbf{J})$ or $\frac{\partial(f_1, f_2, \dots, f_n)}{\partial(x_1, x_2, \dots, x_n)}$.

## Intuition

The Jacobian matrix captures how a vector-valued function changes locally. Think of $\mathbf{f}$ as a transformation that maps points from one space to another. At any point $\mathbf{x}_0$, the Jacobian $\mathbf{J}(\mathbf{x}_0)$ provides the best linear approximation of $\mathbf{f}$ near $\mathbf{x}_0$:

$$
\mathbf{f}(\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + \mathbf{J}(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)
$$

This is the multivariable analogue of the single-variable linear approximation $f(x) \approx f(x_0) + f'(x_0)(x - x_0)$.

Geometrically, the absolute value of the Jacobian determinant $|\det(\mathbf{J})|$ represents the factor by which the transformation $\mathbf{f}$ scales volumes locally. If you take a small region in the input space with volume $dV_{in}$, its image under $\mathbf{f}$ will have volume approximately $|\det(\mathbf{J})| \cdot dV_{in}$. This is why the Jacobian determinant appears in the change of variables formula for multiple integrals.

When the Jacobian determinant is zero at a point, the transformation collapses volume — the function is not locally invertible there. When it is negative, the transformation reverses orientation.

## Why This Concept Matters

The Jacobian is a cornerstone of multivariable calculus with profound implications across science and engineering. In physics, it enables coordinate transformations between Cartesian, polar, cylindrical, and spherical systems. In engineering, it is essential for robotics (mapping joint velocities to end-effector velocities) and control theory. In computer graphics, it governs texture mapping and mesh deformations. In economics, it appears in input-output models and general equilibrium theory.

For machine learning practitioners, the Jacobian has become indispensable. Neural networks compute compositions of vector-valued functions, and understanding how outputs change with respect to inputs or parameters is precisely what the Jacobian quantifies. Modern AI techniques such as normalising flows, neural ODEs, meta-learning, and sensitivity analysis all build upon Jacobian computations.

## Historical Background

The Jacobian is named after the German mathematician **Carl Gustav Jacob Jacobi** (1804–1851), a towering figure in 19th-century mathematics. Jacobi made fundamental contributions to elliptic functions, differential equations, number theory, and determinants. He introduced the Jacobian determinant in 1841 in his paper "De determinantibus functionalibus" (On Functional Determinants), where he studied how determinants of partial derivatives behave under coordinate transformations.

Jacobi was a contemporary and frequent correspondent of Gauss, Dirichlet, and Legendre. He was known for his extraordinary computational ability and his insistence on clarity and elegance in mathematical exposition. The Jacobian matrix and determinant arose from his work on functional determinants, which he used to study change of variables in multiple integrals and to solve systems of partial differential equations.

The concept was later extended and formalised by mathematicians such as Riemann, who used Jacobians in his work on manifolds and curvature, and by Sylvester, who advanced the theory of matrix algebra.

## Real World Examples

1. **Robotics: Velocity Kinematics** — A robotic arm's forward kinematics map joint angles $\boldsymbol{\theta} \in \mathbb{R}^n$ to end-effector position and orientation $\mathbf{x} \in \mathbb{R}^m$. The Jacobian $\mathbf{J}(\boldsymbol{\theta})$ relates joint velocities to end-effector velocities: $\dot{\mathbf{x}} = \mathbf{J}(\boldsymbol{\theta})\dot{\boldsymbol{\theta}}$. This is fundamental for trajectory planning, obstacle avoidance, and force control.

2. **Medical Imaging: Image Registration** — When aligning medical scans (e.g., MRI to CT), a deformation field $\mathbf{f}$ warps one image to match another. The Jacobian determinant of the deformation field indicates local expansion ($\det(\mathbf{J}) > 1$) or contraction ($\det(\mathbf{J}) < 1$), helping detect abnormal tissue changes in longitudinal studies.

3. **Computer Graphics: Non-linear Deformations** — Character rigging and skinning use Jacobians to compute how surface points move when underlying bones rotate. Artists manipulate Jacobian-based deformers to create realistic muscle bulging and cloth simulations.

4. **Economics: Input-Output Analysis** — Leontief input-output models use Jacobian matrices to compute how changes in final demand propagate through an economy's supply chain. The Jacobian of the production function captures elasticities of substitution between inputs.

5. **Meteorology: Weather Prediction** — Atmospheric models solve systems of partial differential equations. The Jacobian of the governing equations appears in numerical weather prediction schemes, particularly in semi-Lagrangian methods that track air parcels as they advect.

## AI/ML Relevance

The Jacobian is deeply embedded in modern machine learning:

1. **Backpropagation** — The chain rule for vector-valued functions is precisely a product of Jacobian matrices. When we compute $\frac{\partial L}{\partial \mathbf{W}}$ in a deep network, we multiply Jacobian matrices through the computational graph: $\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$. Each layer's backward pass computes a Jacobian-vector product.

2. **Neural Network Sensitivity Analysis** — The input-output Jacobian $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ of a neural network reveals which input features most strongly influence each output. In adversarial robustness, the Jacobian is used to craft adversarial examples through the Fast Gradient Sign Method (FGSM), which perturbs inputs in the direction that maximally changes the output.

3. **Normalising Flows** — These generative models learn invertible transformations $\mathbf{f}$ that map a simple base distribution to a complex data distribution. The change-of-variables formula requires the log-absolute-determinant of the Jacobian: $\log p(\mathbf{x}) = \log p(\mathbf{z}) - \log\left|\det\frac{\partial \mathbf{f}}{\partial \mathbf{z}}\right|$. Architectures like RealNVP, Glow, and Neural Spline Flows are designed to have efficiently computable Jacobian determinants.

4. **Invertible Neural Networks** — i-RevNet, The Reformer, and other invertible architectures use Jacobians to guarantee invertibility. By ensuring the Jacobian determinant is non-zero everywhere (e.g., through coupling layers), these networks avoid information loss and enable memory-efficient training.

5. **Optimisation** — Natural gradient descent uses the Fisher Information Matrix, which is the expected outer product of the log-likelihood Jacobian: $\mathbf{F} = \mathbb{E}\left[\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})^T\right]$. This captures the information geometry of the parameter space.

6. **Neural ODEs** — These models treat the hidden state as a continuous-time dynamical system: $\frac{d\mathbf{h}}{dt} = f(\mathbf{h}(t), t, \boldsymbol{\theta})$. The adjoint sensitivity method computes gradients by solving an ODE that involves the Jacobian $\frac{\partial f}{\partial \mathbf{h}}$.

## Mathematical Explanation

### Jacobian Matrix

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian matrix $\mathbf{J} \in \mathbb{R}^{m \times n}$ at a point $\mathbf{a} = (a_1, \dots, a_n)$ is:

$$
\mathbf{J}(\mathbf{a}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1}(\mathbf{a}) & \frac{\partial f_1}{\partial x_2}(\mathbf{a}) & \cdots & \frac{\partial f_1}{\partial x_n}(\mathbf{a}) \\[4pt]
\frac{\partial f_2}{\partial x_1}(\mathbf{a}) & \frac{\partial f_2}{\partial x_2}(\mathbf{a}) & \cdots & \frac{\partial f_2}{\partial x_n}(\mathbf{a}) \\[4pt]
\vdots & \vdots & \ddots & \vdots \\[4pt]
\frac{\partial f_m}{\partial x_1}(\mathbf{a}) & \frac{\partial f_m}{\partial x_2}(\mathbf{a}) & \cdots & \frac{\partial f_m}{\partial x_n}(\mathbf{a})
\end{bmatrix}
$$

The Jacobian exists if all partial derivatives exist and are continuous (i.e., $\mathbf{f}$ is continuously differentiable).

### Jacobian Determinant

When $m = n$, the Jacobian determinant is:

$$
\det(\mathbf{J}) = \det\left(\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_n}{\partial x_1} & \cdots & \frac{\partial f_n}{\partial x_n}
\end{bmatrix}\right)
$$

It is often denoted $\frac{\partial(f_1, \dots, f_n)}{\partial(x_1, \dots, x_n)}$.

### Change of Variables in Multiple Integrals

If $\mathbf{T}: \mathbb{R}^n \to \mathbb{R}^n$ is a continuously differentiable, invertible transformation with $\mathbf{x} = \mathbf{T}(\mathbf{u})$, then:

$$
\int_{\mathbf{T}(R)} f(\mathbf{x}) \, d\mathbf{x} = \int_R f(\mathbf{T}(\mathbf{u})) \, |\det(\mathbf{J}_{\mathbf{T}}(\mathbf{u}))| \, d\mathbf{u}
$$

where $\mathbf{J}_{\mathbf{T}}$ is the Jacobian of $\mathbf{T}$ and $|\det(\mathbf{J}_{\mathbf{T}})|$ is the absolute value of its determinant.

### Chain Rule with Jacobians

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^p$, then $(\mathbf{g} \circ \mathbf{f}): \mathbb{R}^n \to \mathbb{R}^p$ and:

$$
\mathbf{J}_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{f}(\mathbf{x})) \, \mathbf{J}_{\mathbf{f}}(\mathbf{x})
$$

This is the multivariable chain rule in matrix form.

### Inverse Function Theorem

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^n$ is continuously differentiable and $\det(\mathbf{J}_{\mathbf{f}}(\mathbf{a})) \neq 0$, then $\mathbf{f}$ is locally invertible near $\mathbf{a}$, and:

$$
\mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{f}(\mathbf{a})) = \left(\mathbf{J}_{\mathbf{f}}(\mathbf{a})\right)^{-1}
$$

## Formula(s)

1. **Jacobian Matrix Definition**:
   $$
   J_{ij} = \frac{\partial f_i}{\partial x_j}
   $$

2. **Jacobian Determinant** (square case):
   $$
   \det(\mathbf{J}) = \frac{\partial(f_1, f_2, \dots, f_n)}{\partial(x_1, x_2, \dots, x_n)}
   $$

3. **Change of Variables**:
   $$
   \int_{T(R)} f(\mathbf{x}) \, d\mathbf{x} = \int_R f(T(\mathbf{u})) \, |\det J_T(\mathbf{u})| \, d\mathbf{u}
   $$

4. **Multivariable Chain Rule**:
   $$
   \mathbf{J}_{\mathbf{g} \circ \mathbf{f}}(\mathbf{x}) = \mathbf{J}_{\mathbf{g}}(\mathbf{f}(\mathbf{x})) \cdot \mathbf{J}_{\mathbf{f}}(\mathbf{x})
   $$

5. **Inverse Function**:
   $$
   \mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y}) = \left(\mathbf{J}_{\mathbf{f}}(\mathbf{f}^{-1}(\mathbf{y}))\right)^{-1}
   $$

6. **Polar Coordinates Transformation**:
   $$
   (x, y) = (r\cos\theta, r\sin\theta), \quad |\det \mathbf{J}| = r
   $$

7. **Spherical Coordinates Transformation**:
   $$
   (x, y, z) = (r\sin\phi\cos\theta, r\sin\phi\sin\theta, r\cos\phi), \quad |\det \mathbf{J}| = r^2 \sin\phi
   $$

## Properties

1. **Linearity in Each Column**: The Jacobian determinant is multilinear and alternating in the columns (as a determinant).

2. **Multiplicativity under Composition**: If $\mathbf{h} = \mathbf{g} \circ \mathbf{f}$, then $\mathbf{J}_{\mathbf{h}} = \mathbf{J}_{\mathbf{g}} \cdot \mathbf{J}_{\mathbf{f}}$, and for square Jacobians, $\det(\mathbf{J}_{\mathbf{h}}) = \det(\mathbf{J}_{\mathbf{g}}) \cdot \det(\mathbf{J}_{\mathbf{f}})$.

3. **Inverse Relationship**: $\det(\mathbf{J}_{\mathbf{f}^{-1}}) = (\det(\mathbf{J}_{\mathbf{f}}))^{-1}$ when $\mathbf{f}$ is invertible.

4. **Geometric Interpretation**: $|\det(\mathbf{J}(\mathbf{x}))|$ is the factor by which $\mathbf{f}$ scales oriented volumes at $\mathbf{x}$.

5. **Zero Determinant**: If $\det(\mathbf{J}) = 0$ at a point, $\mathbf{f}$ is not locally invertible there (rank deficient).

6. **Gradient as Special Case**: For $f: \mathbb{R}^n \to \mathbb{R}$, the Jacobian is the transpose of the gradient: $\mathbf{J}_f = \nabla f^T$.

7. **Rectangular Case**: For $m \neq n$, the Jacobian generalises the gradient; its maximal rank determines the local geometry of the mapping.

8. **Continuity**: If all partial derivatives are continuous, the Jacobian matrix varies continuously with $\mathbf{x}$.

## Step-by-Step Worked Examples

### Example 1: Jacobian of Polar Coordinate Transformation

**Problem**: Compute the Jacobian matrix and determinant for the transformation from polar to Cartesian coordinates: $x = r\cos\theta$, $y = r\sin\theta$.

**Solution**:

The transformation is $\mathbf{T}(r, \theta) = (r\cos\theta, r\sin\theta)$, mapping $\mathbb{R}^2 \to \mathbb{R}^2$.

Step 1: Compute the four partial derivatives.

$$
\frac{\partial x}{\partial r} = \cos\theta, \quad \frac{\partial x}{\partial \theta} = -r\sin\theta
$$

$$
\frac{\partial y}{\partial r} = \sin\theta, \quad \frac{\partial y}{\partial \theta} = r\cos\theta
$$

Step 2: Assemble the Jacobian matrix.

$$
\mathbf{J}_{\mathbf{T}}(r, \theta) = \begin{bmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\[4pt]
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
\end{bmatrix} = \begin{bmatrix}
\cos\theta & -r\sin\theta \\[4pt]
\sin\theta & r\cos\theta
\end{bmatrix}
$$

Step 3: Compute the determinant.

$$
\det(\mathbf{J}) = (\cos\theta)(r\cos\theta) - (-r\sin\theta)(\sin\theta) = r\cos^2\theta + r\sin^2\theta = r(\cos^2\theta + \sin^2\theta) = r
$$

The absolute value $|\det(\mathbf{J})| = |r| = r$ (since $r \geq 0$), confirming the familiar area element $dA = r \, dr \, d\theta$.

### Example 2: Neural Network Layer Jacobian

**Problem**: Consider a single neural network layer with weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$, bias $\mathbf{b} \in \mathbb{R}^m$, and elementwise activation function $\sigma(z) = \tanh(z)$. Given input $\mathbf{x} \in \mathbb{R}^n$, the output is $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$. Compute the Jacobian $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$.

**Solution**:

Step 1: Write the forward pass. Let $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} \in \mathbb{R}^m$, so $z_i = \sum_{j=1}^n W_{ij} x_j + b_i$. Then $\mathbf{y} = \sigma(\mathbf{z})$, meaning $y_i = \sigma(z_i)$ elementwise.

Step 2: Apply the chain rule. For the Jacobian of $\mathbf{y}$ with respect to $\mathbf{x}$, we have:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{x}}
$$

Step 3: Compute $\frac{\partial \mathbf{y}}{\partial \mathbf{z}}$. Since $\sigma$ is elementwise, $\frac{\partial y_i}{\partial z_j} = 0$ when $i \neq j$, and $\frac{\partial y_i}{\partial z_i} = \sigma'(z_i)$. For $\tanh$ activation, $\sigma'(z) = 1 - \tanh^2(z)$. Thus:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{z}} = \operatorname{diag}(\sigma'(\mathbf{z})) = \begin{bmatrix}
\sigma'(z_1) & 0 & \cdots & 0 \\
0 & \sigma'(z_2) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma'(z_m)
\end{bmatrix}
$$

Step 4: Compute $\frac{\partial \mathbf{z}}{\partial \mathbf{x}}$. Since $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, $\frac{\partial z_i}{\partial x_j} = W_{ij}$, so:

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}
$$

Step 5: Multiply.

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \operatorname{diag}(\sigma'(\mathbf{z})) \cdot \mathbf{W}
$$

Explicitly, the $(i,j)$ entry is $\sigma'(\mathbf{W}_i \cdot \mathbf{x} + b_i) \cdot W_{ij}$, where $\mathbf{W}_i$ is the $i$th row of $\mathbf{W}$.

### Example 3: Change of Variables with Jacobian

**Problem**: Evaluate $\iint_R e^{\frac{y-x}{y+x}} \, dA$ where $R$ is the square region bounded by $x = 0$, $y = 0$, $x + y = 1$, and $x + y = 2$, and $y - x$ varies from $-1$ to $1$.

**Solution**:

Step 1: Choose a convenient transformation. Let $u = y - x$ and $v = y + x$. This maps the region to a rectangle in the $uv$-plane.

We need those bounds: $v = x + y$ ranges from $1$ to $2$, and $u = y - x$ ranges from $-1$ to $1$.

Step 2: Find the inverse transformation. Solve for $x$ and $y$ in terms of $u$ and $v$:

Adding: $u + v = (y - x) + (y + x) = 2y \implies y = \frac{u + v}{2}$

Subtracting: $v - u = (y + x) - (y - x) = 2x \implies x = \frac{v - u}{2}$

So $\mathbf{T}(u, v) = \left(\frac{v - u}{2}, \frac{u + v}{2}\right)$.

Step 3: Compute the Jacobian determinant.

$$
\mathbf{J}_{\mathbf{T}}(u, v) = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\[4pt]
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v}
\end{bmatrix} = \begin{bmatrix}
-\frac{1}{2} & \frac{1}{2} \\[4pt]
\frac{1}{2} & \frac{1}{2}
\end{bmatrix}
$$

$$
\det(\mathbf{J}) = \left(-\frac{1}{2}\right)\left(\frac{1}{2}\right) - \left(\frac{1}{2}\right)\left(\frac{1}{2}\right) = -\frac{1}{4} - \frac{1}{4} = -\frac{1}{2}
$$

Thus $|\det(\mathbf{J})| = \frac{1}{2}$.

Step 4: Transform the integral.

$$
\iint_R e^{\frac{y-x}{y+x}} \, dA = \int_{v=1}^{2} \int_{u=-1}^{1} e^{u/v} \cdot \frac{1}{2} \, du \, dv
$$

Step 5: Evaluate the inner integral.

$$
\int_{u=-1}^{1} e^{u/v} \, du = \left[ v e^{u/v} \right]_{u=-1}^{1} = v(e^{1/v} - e^{-1/v})
$$

Step 6: Evaluate the outer integral.

$$
\frac{1}{2} \int_{v=1}^{2} v(e^{1/v} - e^{-1/v}) \, dv = \frac{1}{2} \int_{1}^{2} 2v \sinh(1/v) \, dv = \int_{1}^{2} v \sinh(1/v) \, dv
$$

This can be evaluated numerically to approximately $0.892$.

### Example 4: Jacobian for a Normalising Flow

**Problem**: Consider an affine coupling layer used in RealNVP: $\mathbf{y} = [y_1, y_2]$ where $y_1 = x_1$ and $y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$, where $s$ and $t$ are scalar functions (neural networks), $\odot$ is elementwise multiplication, and $\mathbf{x} \in \mathbb{R}^D$ with $D$ even, split into $\mathbf{x} = (x_1, x_2)$ each of dimension $D/2$. Compute the log-determinant of the Jacobian $\log\left|\det\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right|$.

**Solution**:

Step 1: Structure the Jacobian. The transformation is:

$$
\begin{cases}
y_1 = x_1 \\
y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)
\end{cases}
$$

Step 2: Compute the Jacobian sub-blocks.

- $\frac{\partial y_1}{\partial x_1} = \mathbf{I}$ (identity matrix of size $D/2$)
- $\frac{\partial y_1}{\partial x_2} = \mathbf{0}$
- $\frac{\partial y_2}{\partial x_1}$ is complicated (depends on derivatives of $s$ and $t$), call this block $\mathbf{A}$
- $\frac{\partial y_2}{\partial x_2} = \operatorname{diag}(\exp(s(x_1)))$

Step 3: The Jacobian matrix has block-triangular form:

$$
\mathbf{J} = \begin{bmatrix}
\mathbf{I} & \mathbf{0} \\
\mathbf{A} & \operatorname{diag}(\exp(s(x_1)))
\end{bmatrix}
$$

Step 4: The determinant of a block triangular matrix is the product of the determinants of the diagonal blocks:

$$
\det(\mathbf{J}) = \det(\mathbf{I}) \cdot \det(\operatorname{diag}(\exp(s(x_1)))) = \prod_{i=1}^{D/2} \exp(s(x_1)_i) = \exp\left(\sum_{i=1}^{D/2} s(x_1)_i\right)
$$

Step 5: The log-determinant:

$$
\log|\det(\mathbf{J})| = \sum_{i=1}^{D/2} s(x_1)_i
$$

This is efficient to compute — no expensive determinant calculation required, which is why coupling layers are popular in normalising flows.

## Visual Interpretation

The Jacobian can be visualised geometrically as a local linear approximation of a non-linear transformation.

Consider a function $\mathbf{f}: \mathbb{R}^2 \to \mathbb{R}^2$. Take a small square of side length $\epsilon$ in the input space centred at $\mathbf{x}_0$, with sides parallel to the coordinate axes. Under $\mathbf{f}$, this square maps approximately to a parallelogram centred at $\mathbf{f}(\mathbf{x}_0)$.

The sides of the original square are $\epsilon \hat{\mathbf{e}}_1$ and $\epsilon \hat{\mathbf{e}}_2$ (where $\hat{\mathbf{e}}_i$ are the standard basis vectors). Their images under the linear approximation are:

$$
\mathbf{f}(\mathbf{x}_0 + \epsilon \hat{\mathbf{e}}_1) - \mathbf{f}(\mathbf{x}_0) \approx \epsilon \frac{\partial \mathbf{f}}{\partial x_1} = \epsilon \begin{bmatrix} \frac{\partial f_1}{\partial x_1} \\ \frac{\partial f_2}{\partial x_1} \end{bmatrix}
$$

$$
\mathbf{f}(\mathbf{x}_0 + \epsilon \hat{\mathbf{e}}_2) - \mathbf{f}(\mathbf{x}_0) \approx \epsilon \frac{\partial \mathbf{f}}{\partial x_2} = \epsilon \begin{bmatrix} \frac{\partial f_1}{\partial x_2} \\ \frac{\partial f_2}{\partial x_2} \end{bmatrix}
$$

These two vectors form the sides of the image parallelogram. The area of this parallelogram is:

$$
\text{Area} = \left| \det\left( \begin{bmatrix} \epsilon \frac{\partial f_1}{\partial x_1} & \epsilon \frac{\partial f_1}{\partial x_2} \\ \epsilon \frac{\partial f_2}{\partial x_1} & \epsilon \frac{\partial f_2}{\partial x_2} \end{bmatrix} \right) \right| = \epsilon^2 |\det(\mathbf{J})|
$$

Thus $|\det(\mathbf{J})|$ is the factor by which area scales locally. If we visualise a grid of points being transformed, the Jacobian tells us how the grid cells stretch, rotate, and shear. When $\det(\mathbf{J}) > 0$, orientation is preserved; when $\det(\mathbf{J}) < 0$, the transformation flips orientation (mirroring).

For a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$, the Jacobian row vector $\nabla f^T$ points in the direction of steepest ascent, and its magnitude gives the rate of change. This is why the gradient descent direction is $-\nabla f$.

## Common Mistakes

1. **Confusing Jacobian matrix with Jacobian determinant**: The term "Jacobian" is often used to refer to both the matrix and its determinant. The Jacobian matrix always exists (given differentiability), but the determinant only exists when the matrix is square ($m=n$). Always clarify which one is meant.

2. **Forgetting the absolute value in change of variables**: The change of variables formula requires $|\det(\mathbf{J})|$, not $\det(\mathbf{J})$. The absolute value ensures volumes are positive, as negative determinants indicate orientation reversal, not negative volume.

3. **Incorrectly applying the chain rule**: The chain rule for vector-valued functions involves matrix multiplication of Jacobians, and the order matters. Common errors include multiplying in the wrong order, forgetting that Jacobians must be evaluated at the correct points, or assuming elementwise multiplication.

4. **Treating the Jacobian as a gradient**: For scalar-to-scalar functions, the derivative is a single number. For vector-to-scalar functions, the gradient is a column vector. For vector-to-vector functions, the Jacobian is a matrix. Novices often confuse these and their transposes.

5. **Assuming the Jacobian determinant is always non-zero**: Many transformations have points where $\det(\mathbf{J}) = 0$ (critical points). At these points, the function is not locally invertible, and the inverse function theorem does not apply. Ignoring this can lead to incorrect conclusions about transformation properties.

6. **Incorrectly computing Jacobians for coordinate transformations**: When computing the Jacobian for, say, polar coordinates, students sometimes differentiate the wrong variables. The Jacobian is always $\frac{\partial(x,y)}{\partial(r,\theta)}$, not $\frac{\partial(r,\theta)}{\partial(x,y)}$. The two are matrix inverses.

7. **Mishandling dimensions in neural network Jacobians**: When computing $\frac{\partial \text{loss}}{\partial \mathbf{W}}$ using backpropagation, the Jacobian of a layer with respect to its weights has a more complex structure than $\frac{\partial \text{output}}{\partial \text{input}}$. The correct dimensions must be tracked carefully.

8. **Assuming the Jacobian is symmetric**: The Jacobian matrix is not necessarily symmetric. Symmetry requires $\frac{\partial f_i}{\partial x_j} = \frac{\partial f_j}{\partial x_i}$ for all $i, j$, which is not generally true for vector-valued functions.

## Interview Questions

### Beginner

1. **Q**: What is the Jacobian matrix of a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$?
   **A**: It is an $m \times n$ matrix whose $(i,j)$ entry is $\frac{\partial f_i}{\partial x_j}$, representing all first-order partial derivatives of $\mathbf{f}$.

2. **Q**: What is the geometric interpretation of the Jacobian determinant?
   **A**: Its absolute value $|\det(\mathbf{J})|$ represents the factor by which the transformation locally scales volumes (or areas in 2D, lengths in 1D).

3. **Q**: Compute the Jacobian matrix of $\mathbf{f}(x,y) = (x^2, xy, y^2)$.
   **A**: $\mathbf{J} = \begin{bmatrix} 2x & 0 \\ y & x \\ 0 & 2y \end{bmatrix}$ (a $3 \times 2$ matrix).

4. **Q**: How does the Jacobian relate to the gradient of a scalar function?
   **A**: For $f: \mathbb{R}^n \to \mathbb{R}$, the Jacobian is a $1 \times n$ row vector equal to $\nabla f^T$, the transpose of the gradient.

5. **Q**: What happens when the Jacobian determinant is zero at a point?
   **A**: The function is not locally invertible at that point — it collapses volume in at least one direction.

### Intermediate

1. **Q**: Derive the change of variables formula for double integrals using the Jacobian determinant.
   **A**: Under $\mathbf{T}(u,v) = (x(u,v), y(u,v))$, a small area element $du\,dv$ maps to a parallelogram with area $|\det(\mathbf{J})|\,du\,dv$. Thus $\iint_R f(x,y)\,dx\,dy = \iint_S f(x(u,v), y(u,v))\,|\det(\mathbf{J})|\,du\,dv$.

2. **Q**: In backpropagation, how does the Jacobian of each layer propagate gradients?
   **A**: The gradient of loss $L$ with respect to layer input $\mathbf{x}$ is $\frac{\partial L}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^T \frac{\partial L}{\partial \mathbf{y}}$, where $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is the layer's Jacobian. This is a Jacobian-vector product.

3. **Q**: For $\mathbf{f}(x,y) = (e^x\cos y, e^x\sin y)$, compute $\det(\mathbf{J})$ and interpret.
   **A**: $\mathbf{J} = \begin{bmatrix} e^x\cos y & -e^x\sin y \\ e^x\sin y & e^x\cos y \end{bmatrix}$, $\det(\mathbf{J}) = e^{2x}(\cos^2 y + \sin^2 y) = e^{2x} > 0$. The transformation preserves orientation everywhere and scales area by $e^{2x}$.

4. **Q**: How is the Jacobian used in the proof of the Inverse Function Theorem?
   **A**: The Inverse Function Theorem states that if $\det(\mathbf{J}_f(\mathbf{a})) \neq 0$, then $f$ is locally invertible near $\mathbf{a}$. The Jacobian of the inverse is the matrix inverse of $\mathbf{J}_f(\mathbf{a})$, and the theorem guarantees the inverse is differentiable.

5. **Q**: Explain why normalising flows need tractable Jacobian determinants. How do coupling layers achieve this?
   **A**: The change of variables formula in probability requires $\log|\det(\mathbf{J})|$. Coupling layers structure the transformation to have a triangular Jacobian, so the determinant equals the product of diagonal entries, avoiding expensive $O(n^3)$ computation.

### Advanced

1. **Q**: The Jacobian of a neural network's output with respect to its input is called the input-output Jacobian. How would you compute this efficiently for a deep network with $L$ layers, and what does its singular value decomposition reveal?
   **A**: By the chain rule, $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \prod_{l=1}^{L} \frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}}$, i.e., a product of per-layer Jacobians. The SVD of this product reveals directions of maximum and minimum sensitivity. The singular values indicate how much each input direction is amplified or attenuated. Networks with contracting Jacobians (all singular values $\leq 1$) are Lipschitz-constrained, which is used in adversarially robust models. In neural tangent kernel theory, the Gram matrix of the network Jacobian governs training dynamics.

2. **Q**: In Neural ODEs, the adjoint method computes gradients without storing intermediate states. Derive the adjoint ODE and show where the Jacobian appears.
   **A**: For $\frac{d\mathbf{h}}{dt} = f(\mathbf{h}(t), t, \theta)$, define the adjoint $\mathbf{a}(t) = \frac{\partial L}{\partial \mathbf{h}(t)}$. Through calculus of variations, $\frac{d\mathbf{a}}{dt} = -\mathbf{a}(t)^T \frac{\partial f}{\partial \mathbf{h}}(t)$. The final gradient $\frac{\partial L}{\partial \theta} = \int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f}{\partial \theta} \, dt$. The term $\frac{\partial f}{\partial \mathbf{h}}$ is the Jacobian of the dynamics with respect to the hidden state. Note that computing $\mathbf{a}^T \mathbf{J}$ is a Jacobian-vector product, which can be computed efficiently with reverse-mode autodiff without forming the full Jacobian.

3. **Q**: For a normalising flow with transformation $\mathbf{f}$, the log-density is $\log p(\mathbf{x}) = \log p(\mathbf{z}) - \log|\det\mathbf{J}_f(\mathbf{z})|$. In continuous normalising flows (CNFs), the instantaneous change of variables formula replaces the log-determinant with a trace. Derive this result and explain why it is computationally advantageous.
   **A**: For a continuous-time flow $\frac{d\mathbf{z}}{dt} = f(\mathbf{z}(t), t)$, the density evolves as $\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\operatorname{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$. This comes from the relationship $\frac{d}{dt} \log|\det\mathbf{J}| = \operatorname{tr}(\mathbf{J}^{-1}\frac{d\mathbf{J}}{dt})$, and applying Liouville's formula $\frac{d}{dt}\det(\mathbf{J}) = \det(\mathbf{J})\operatorname{tr}(\mathbf{J}^{-1}\frac{d\mathbf{J}}{dt})$. The trace $\operatorname{tr}(\frac{\partial f}{\partial \mathbf{z}}) = \sum_i \frac{\partial f_i}{\partial z_i}$ is an $O(D)$ operation, compared to $O(D^3)$ for the full determinant, enabling scaling to high dimensions. In practice, the trace is estimated stochastically using Hutchinson's trick: $\operatorname{tr}(\mathbf{A}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})}[\boldsymbol{\epsilon}^T \mathbf{A} \boldsymbol{\epsilon}]$, requiring only Jacobian-vector products.

## Practice Problems

### Easy

1. Compute the Jacobian matrix of $\mathbf{f}(x, y) = (3x + 2y, x - 4y)$.

2. Find the Jacobian determinant for the transformation $x = u^2 - v^2$, $y = 2uv$.

3. For $f(x, y, z) = x^2yz$, find the Jacobian (which is a $1 \times 3$ row matrix).

4. Compute $\det(\mathbf{J})$ for the transformation from cylindrical to Cartesian coordinates: $x = r\cos\theta$, $y = r\sin\theta$, $z = z$.

5. If $\mathbf{f}(t) = (t^2, t^3, e^t)$ maps $\mathbb{R} \to \mathbb{R}^3$, find its Jacobian matrix.

### Medium

1. Evaluate $\iint_R (x^2 + y^2) \, dA$ where $R$ is the disk $x^2 + y^2 \leq 4$, using a polar coordinate transformation and the Jacobian.

2. For $\mathbf{f}(u, v) = (u\cos v, u\sin v, v)$, compute the Jacobian matrix and determine where it has full rank (assuming outputs are mapped to $\mathbb{R}^2$ by dropping the third component).

3. Find the Jacobian matrix of a 2-layer neural network $f(x) = W_2 \sigma(W_1 x + b_1) + b_2$ with respect to input $x$, where $\sigma$ is the ReLU activation.

4. Compute the Jacobian determinant for the spherical-to-Cartesian transformation and verify the volume element $dV = r^2 \sin\phi \, dr \, d\phi \, d\theta$.

5. Use the Jacobian to transform $\iint_R e^{-(x^2 + y^2)} \, dA$ over the region $x^2 + y^2 \leq 1$ to polar coordinates and evaluate.

### Hard

1. Consider a normalising flow with transformation $f(x) = x + \frac{a}{1 + x^2}$ where $a$ is a parameter. Find the Jacobian (this is a scalar case). For what values of $a$ is the transformation invertible (i.e., $\frac{df}{dx} \neq 0$ for all $x$)?

2. Prove that for any invertible transformation $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^n$, the Jacobian of the inverse satisfies $\mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y}) = [\mathbf{J}_f(\mathbf{f}^{-1}(\mathbf{y}))]^{-1}$, assuming $\det(\mathbf{J}_f) \neq 0$.

3. Design a coupling layer for a 2D normalising flow that has the form $y_1 = x_1$, $y_2 = x_1^2 + x_2$. Compute its Jacobian determinant. Is this transformation always invertible? If not, find the condition on $(x_1, x_2)$ for invertibility.

## Solutions

### Easy Solutions

**Solution 1**: $\mathbf{J} = \begin{bmatrix} 3 & 2 \\ 1 & -4 \end{bmatrix}$.

**Solution 2**: $\frac{\partial x}{\partial u} = 2u$, $\frac{\partial x}{\partial v} = -2v$, $\frac{\partial y}{\partial u} = 2v$, $\frac{\partial y}{\partial v} = 2u$. Then $\det(\mathbf{J}) = (2u)(2u) - (-2v)(2v) = 4u^2 + 4v^2 = 4(u^2 + v^2)$.

**Solution 3**: $\mathbf{J} = \begin{bmatrix} 2xyz & x^2z & x^2y \end{bmatrix}$ (a $1 \times 3$ row vector).

**Solution 4**: 
$$
\mathbf{J} = \begin{bmatrix}
\cos\theta & -r\sin\theta & 0 \\
\sin\theta & r\cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}, \quad \det(\mathbf{J}) = r\cos^2\theta + r\sin^2\theta = r
$$

**Solution 5**: Since $f: \mathbb{R} \to \mathbb{R}^3$, $\mathbf{J} = \begin{bmatrix} 2t & 3t^2 & e^t \end{bmatrix}^T = \begin{bmatrix} 2t \\ 3t^2 \\ e^t \end{bmatrix}$.

### Medium Solutions

**Solution 1**: In polar coordinates $x = r\cos\theta$, $y = r\sin\theta$, $|\det(\mathbf{J})| = r$. The region is $0 \leq r \leq 2$, $0 \leq \theta \leq 2\pi$. Then $\iint (x^2 + y^2) \, dA = \int_0^{2\pi} \int_0^2 r^2 \cdot r \, dr \, d\theta = \int_0^{2\pi} \int_0^2 r^3 \, dr \, d\theta = \int_0^{2\pi} \left[\frac{r^4}{4}\right]_0^2 d\theta = \int_0^{2\pi} 4 \, d\theta = 8\pi$.

**Solution 2**: $\mathbf{f}(u,v,w) = (u\cos v, u\sin v, w)$ has Jacobian:
$$
\mathbf{J} = \begin{bmatrix}
\cos v & -u\sin v & 0 \\
\sin v & u\cos v & 0 \\
0 & 0 & 1
\end{bmatrix}
$$
$\det(\mathbf{J}) = u(\cos^2 v + \sin^2 v) = u$. Full rank when $u \neq 0$.

**Solution 3**: Let $\mathbf{z}_1 = W_1 x + b_1$, $\mathbf{h}_1 = \sigma(\mathbf{z}_1)$, $\mathbf{y} = W_2 \mathbf{h}_1 + b_2$.
Then $\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = W_2 \cdot \operatorname{diag}(\sigma'(\mathbf{z}_1)) \cdot W_1$.
For ReLU, $\sigma'(z) = 1$ if $z > 0$, $0$ otherwise, so the diagonal matrix acts as a gating mechanism.

**Solution 4**: $x = r\sin\phi\cos\theta$, $y = r\sin\phi\sin\theta$, $z = r\cos\phi$.
$$
\mathbf{J} = \begin{bmatrix}
\sin\phi\cos\theta & r\cos\phi\cos\theta & -r\sin\phi\sin\theta \\
\sin\phi\sin\theta & r\cos\phi\sin\theta & r\sin\phi\cos\theta \\
\cos\phi & -r\sin\phi & 0
\end{bmatrix}
$$
$\det(\mathbf{J}) = r^2\sin\phi$. Volume element: $dV = r^2\sin\phi \, dr \, d\phi \, d\theta$.

**Solution 5**: $\iint_{x^2+y^2 \leq 1} e^{-(x^2+y^2)} \, dA = \int_0^{2\pi} \int_0^1 e^{-r^2} \cdot r \, dr \, d\theta = 2\pi \int_0^1 re^{-r^2} \, dr$.
Let $u = -r^2$, $du = -2r \, dr$, so $= 2\pi \left[-\frac{1}{2}e^{-r^2}\right]_0^1 = \pi(1 - e^{-1})$.

### Hard Solutions

**Solution 1**: $\frac{df}{dx} = 1 + a \cdot \frac{-2x}{(1+x^2)^2} = 1 - \frac{2ax}{(1+x^2)^2}$. The minimum of $\frac{df}{dx}$ occurs when $g(x) = \frac{2x}{(1+x^2)^2}$ is maximised. The maximum of $|g(x)|$ is at $x = 1/\sqrt{3}$, where $g(1/\sqrt{3}) = \frac{3\sqrt{3}}{8} \approx 0.6495$. So for invertibility for all $x$, we need $1 - \frac{3\sqrt{3}}{8}a > 0$ and $1 + \frac{3\sqrt{3}}{8}a > 0$, giving $|a| < \frac{8}{3\sqrt{3}} \approx 1.5396$.

**Solution 2**: Since $\mathbf{f}^{-1}(\mathbf{f}(\mathbf{x})) = \mathbf{x}$ and $\mathbf{f}(\mathbf{f}^{-1}(\mathbf{y})) = \mathbf{y}$, differentiate the latter using the chain rule: $\mathbf{J}_{\mathbf{f}}(\mathbf{f}^{-1}(\mathbf{y})) \cdot \mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y}) = \mathbf{I}$. Multiply both sides by $[\mathbf{J}_{\mathbf{f}}(\mathbf{f}^{-1}(\mathbf{y}))]^{-1}$ to get $\mathbf{J}_{\mathbf{f}^{-1}}(\mathbf{y}) = [\mathbf{J}_{\mathbf{f}}(\mathbf{f}^{-1}(\mathbf{y}))]^{-1}$.

**Solution 3**: $\mathbf{y} = (x_1, x_1^2 + x_2)$.
$$
\mathbf{J} = \begin{bmatrix} 1 & 0 \\ 2x_1 & 1 \end{bmatrix}, \quad \det(\mathbf{J}) = 1
$$
The transformation is always invertible (determinant $= 1$ everywhere). The inverse is $x_1 = y_1$, $x_2 = y_2 - y_1^2$, which is always defined.

## Related Concepts

- **Gradient**: The Jacobian of a scalar-valued function (transpose of the gradient)
- **Hessian**: The Jacobian of the gradient (second-order partial derivatives)
- **Partial Derivatives**: Building blocks of the Jacobian matrix
- **Determinant**: Used to compute the Jacobian determinant; measures volume scaling
- **Change of Variables**: Integration technique that relies on the Jacobian determinant
- **Taylor Series in Multiple Variables**: The Jacobian is the coefficient of the linear term
- **Differential Forms**: The Jacobian determinant appears as the pullback of volume forms
- **Pushforward and Pullback**: The Jacobian captures the differential of a map between manifolds

## Next Concepts

- **Hessian Matrix**: The matrix of second-order partial derivatives (MATH-060)
- **Multiple Integrals**: Applications of Jacobian-based change of variables (MATH-064)
- **Vector Calculus**: Divergence, curl, and gradient operations using Jacobians
- **Differential Geometry**: The Jacobian as the differential of a map between manifolds
- **Manifold Learning**: Using Jacobians to understand data manifolds
- **Neural Tangent Kernel**: The kernel defined by the Jacobian of neural network outputs

## Summary

The Jacobian matrix collects all first-order partial derivatives of a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ into an $m \times n$ matrix. Its determinant (when $m=n$) gives the local scaling factor for volumes under the transformation. The Jacobian is the fundamental tool for linearising non-linear transformations, performing change of variables in multiple integrals, and understanding the chain rule in higher dimensions.

In machine learning, Jacobians are central to backpropagation (where gradients propagate through layer Jacobians), normalising flows (where the change-of-variables formula requires the log-determinant), sensitivity analysis (where the input-output Jacobian reveals feature importance), and Neural ODEs (where the adjoint method uses Jacobian-vector products). The efficient computation of Jacobian-related quantities — particularly Jacobian-vector products and vector-Jacobian products — is a cornerstone of modern automatic differentiation frameworks.

## Key Takeaways

- The Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function
- The Jacobian determinant (for square matrices) gives the local volume scaling factor of a transformation
- The chain rule in multivariable calculus is a product of Jacobian matrices
- The change of variables formula for multiple integrals uses $|\det(\mathbf{J})|$
- Backpropagation computes vector-Jacobian products through each network layer
- Normalising flows require tractable $\log|\det(\mathbf{J})|$ computations for density estimation
- The Jacobian determinant being zero indicates non-invertibility (rank deficiency)
- Jacobian-vector products can be computed efficiently without forming the full matrix
- The Inverse Function Theorem guarantees local invertibility when $\det(\mathbf{J}) \neq 0$
- Understanding Jacobians is essential for modern deep learning, from optimisation to generative modelling
