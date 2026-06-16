# Concept: Multiple Integrals

## Concept ID

MATH-064

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Calculus

## Learning Objectives

- Define double and triple integrals as iterated integrals over 2D and 3D regions
- Apply Fubini's theorem to evaluate multiple integrals as iterated single integrals
- Set up and evaluate double integrals in Cartesian and polar coordinates
- Set up and evaluate triple integrals in Cartesian, cylindrical, and spherical coordinates
- Use the Jacobian determinant for change of variables in multiple integrals
- Compute volumes, masses, and centres of mass using multiple integrals
- Apply multiple integrals to marginalisation and expectations in probability

## Prerequisites

- Single-variable definite integration (MATH-062)
- Partial derivatives and gradients
- The Jacobian matrix and determinant (MATH-059)
- Coordinate systems (polar, cylindrical, spherical)
- Basic probability theory

## Definition

A **multiple integral** extends the concept of a definite integral to functions of several variables. For a function $f(x, y)$ of two variables defined on a region $R \subset \mathbb{R}^2$, the **double integral** is:

$$
\iint_R f(x, y) \, dA = \iint_R f(x, y) \, dx \, dy
$$

This represents the signed volume under the surface $z = f(x, y)$ above the region $R$ in the $xy$-plane.

For a function $f(x, y, z)$ of three variables over a region $V \subset \mathbb{R}^3$, the **triple integral** is:

$$
\iiint_V f(x, y, z) \, dV = \iiint_V f(x, y, z) \, dx \, dy \, dz
$$

This represents the hypervolume (4D "volume") of the region under the graph, or more commonly, the total mass given a density function $f$.

## Intuition

Just as a single integral sums infinitesimal rectangles of area $f(x) \, dx$ to compute total area under a curve, a double integral sums infinitesimal rectangular prisms of volume $f(x, y) \, dA$ to compute total volume under a surface.

Think of slicing the region $R$ into $n \times m$ tiny rectangles of area $\Delta A = \Delta x \Delta y$. The volume of the prism above each rectangle is approximately $f(x_i^*, y_j^*) \Delta A$. The double integral is the limit of the sum of these prisms as the grid refines:

$$
\iint_R f(x, y) \, dA = \lim_{m,n \to \infty} \sum_{i=1}^n \sum_{j=1}^m f(x_i^*, y_j^*) \, \Delta A
$$

For triple integrals, we sum over 3D boxes (cubes) to compute a 4D quantity, which is harder to visualise but follows the same pattern.

## Why This Concept Matters

Multiple integrals are essential for working with multi-dimensional systems. In physics, they compute moments of inertia, gravitational potentials, and electromagnetic fields. In engineering, they determine centres of mass, fluid flow through surfaces, and heat distribution.

In probability and statistics, multiple integrals are indispensable. Joint probability distributions are functions of multiple random variables, and marginalisation requires integrating out variables. Expected values of functions of multiple random variables are multiple integrals. The entire field of Bayesian inference involves high-dimensional integrals over parameter spaces.

In machine learning, multi-dimensional integrals appear in:
- Marginal likelihood computation (Bayesian model evidence)
- Expectations over high-dimensional distributions
- Variational inference with structured latent variable models
- Integration in kernel methods and Gaussian processes

## Historical Background

The extension of integral calculus to multiple dimensions was developed throughout the 18th and 19th centuries. **Leonhard Euler** (1707–1783) worked with double and triple integrals in his studies of mechanics and hydrodynamics. **Joseph-Louis Lagrange** (1736–1813) used multiple integrals in his work on celestial mechanics.

The formalisation of multiple integration as iterated integrals is due to **Guido Fubini** (1879–1943), an Italian mathematician who proved that under appropriate conditions, a double integral can be computed as two successive single integrals. Fubini's theorem, published in 1907, is the foundation for practical computation of multiple integrals.

The change of variables formula for multiple integrals, involving the Jacobian determinant, was developed by **Carl Gustav Jacob Jacobi** (1804–1851). The geometric interpretation — that the Jacobian determinant measures local volume scaling — was refined in the 19th century with the development of differential geometry.

The theory of integration in $\mathbb{R}^n$ was substantially advanced by **Henri Lebesgue** (1875–1941), whose measure-theoretic approach generalised Riemann integration to arbitrary dimensions and domains.

## Real World Examples

1. **Physics: Centre of Mass of a 2D Plate** — The centre of mass $(\bar{x}, \bar{y})$ of a thin plate of density $\rho(x, y)$ over region $R$ is:
   $$
   \bar{x} = \frac{\iint_R x \rho(x, y) \, dA}{\iint_R \rho(x, y) \, dA}, \quad \bar{y} = \frac{\iint_R y \rho(x, y) \, dA}{\iint_R \rho(x, y) \, dA}
   $$

2. **Engineering: Moment of Inertia** — The moment of inertia of a 3D object about the $z$-axis is $I_z = \iiint_V (x^2 + y^2) \rho(x, y, z) \, dV$, which determines rotational acceleration under torque.

3. **Environmental Science: Total Rainfall over a Region** — If rainfall intensity is $I(x, y)$ mm/hour at point $(x, y)$, the total water volume over region $R$ in one hour is $\iint_R I(x, y) \, dA$.

4. **Computer Graphics: Rendering Equation** — The rendering equation computes light transport as a high-dimensional integral over surfaces, directions, and wavelengths. Path tracing evaluates these integrals numerically using Monte Carlo.

5. **Medical Imaging: CT Scan Reconstruction** — Computed tomography reconstructs 3D density from 2D X-ray projections. The reconstruction algorithm involves integrating over lines (Radon transform) and then using the inverse transform, which also involves integration.

## AI/ML Relevance

1. **Marginalisation over Multiple Variables** — Given a joint distribution $p(x, y, z)$, the marginal $p(x) = \iint p(x, y, z) \, dy \, dz$. This is a double integral. In latent variable models, we marginalise over all latent variables: $p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$, which is a high-dimensional multiple integral.

2. **Multi-Dimensional Expectations** — The expected value of a function of multiple random variables is a multiple integral:
   $$
   \mathbb{E}[g(X, Y)] = \iint g(x, y) f_{XY}(x, y) \, dx \, dy
   $$
   This appears in reinforcement learning (value functions), Bayesian decision theory, and risk estimation.

3. **Bayesian Evidence** — The marginal likelihood (model evidence) is the integral over all parameters:
   $$
   p(D) = \int p(D|\boldsymbol{\theta}) p(\boldsymbol{\theta}) \, d\boldsymbol{\theta} = \int \cdots \int p(D|\theta_1, \dots, \theta_d) p(\theta_1, \dots, \theta_d) \, d\theta_1 \cdots d\theta_d
   $$
   This $d$-dimensional integral is typically intractable, motivating approximation methods like variational inference and MCMC.

4. **Variational Inference** — The ELBO is:
   $$
   \text{ELBO}(q) = \int q(\mathbf{z}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z}
   $$
   This is a multiple integral over the latent variables. Mean-field variational inference factorises $q(\mathbf{z}) = \prod_i q_i(z_i)$, which makes the integral separable and tractable.

5. **Gaussian Processes** — Predictions in GPs involve integrals like $\int f(\mathbf{x}_*) p(f|\mathbf{X}, \mathbf{y}) \, df$, which are Gaussian integrals tractable in closed form for regression.

6. **Kernel Methods** — The kernel trick can be seen as integrating over feature maps: $k(x, y) = \int \phi(x, t) \phi(y, t) \, dt$.

## Mathematical Explanation

### Double Integrals over Rectangular Regions

For a rectangle $R = [a, b] \times [c, d]$, Fubini's theorem states:

$$
\iint_R f(x, y) \, dA = \int_a^b \int_c^d f(x, y) \, dy \, dx = \int_c^d \int_a^b f(x, y) \, dx \, dy
$$

The order of integration can be chosen based on convenience.

### Double Integrals over General Regions

**Type I** (vertical-simple): $R = \{(x, y) : a \leq x \leq b, g_1(x) \leq y \leq g_2(x)\}$

$$
\iint_R f(x, y) \, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y) \, dy \, dx
$$

**Type II** (horizontal-simple): $R = \{(x, y) : c \leq y \leq d, h_1(y) \leq x \leq h_2(y)\}$

$$
\iint_R f(x, y) \, dA = \int_c^d \int_{h_1(y)}^{h_2(y)} f(x, y) \, dx \, dy
$$

### Triple Integrals

For a region $V$ in $\mathbb{R}^3$:

$$
\iiint_V f(x, y, z) \, dV = \iiint_V f(x, y, z) \, dx \, dy \, dz
$$

These are evaluated as three successive single integrals, with appropriate limits for each variable.

### Change of Variables

For a transformation $\mathbf{T}: \mathbb{R}^2 \to \mathbb{R}^2$ mapping $(u, v)$ to $(x, y)$:

$$
\iint_R f(x, y) \, dA = \iint_S f(x(u,v), y(u,v)) \, |\det(\mathbf{J})| \, du \, dv
$$

where $\mathbf{J}$ is the Jacobian matrix of the transformation.

### Polar Coordinates

$$
x = r\cos\theta, \quad y = r\sin\theta, \quad |\det(\mathbf{J})| = r
$$

$$
\iint_R f(x, y) \, dA = \iint_S f(r\cos\theta, r\sin\theta) \, r \, dr \, d\theta
$$

### Cylindrical Coordinates

$$
x = r\cos\theta, \quad y = r\sin\theta, \quad z = z, \quad |\det(\mathbf{J})| = r
$$

$$
\iiint_V f(x, y, z) \, dV = \iiint_S f(r\cos\theta, r\sin\theta, z) \, r \, dz \, dr \, d\theta
$$

### Spherical Coordinates

$$
x = \rho\sin\phi\cos\theta, \quad y = \rho\sin\phi\sin\theta, \quad z = \rho\cos\phi, \quad |\det(\mathbf{J})| = \rho^2 \sin\phi
$$

$$
\iiint_V f(x, y, z) \, dV = \iiint_S f(\rho\sin\phi\cos\theta, \rho\sin\phi\sin\theta, \rho\cos\phi) \, \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta
$$

## Formula(s)

1. **Double Integral (Rectangular)**:
   $$
   \iint_{[a,b]\times[c,d]} f(x, y) \, dA = \int_a^b \int_c^d f(x, y) \, dy \, dx
   $$

2. **Double Integral (Type I Region)**:
   $$
   \iint_R f(x, y) \, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y) \, dy \, dx
   $$

3. **Triple Integral**:
   $$
   \iiint_V f(x, y, z) \, dV = \int \int \int f(x, y, z) \, dz \, dy \, dx
   $$

4. **Change of Variables (2D)**:
   $$
   \iint_R f(x, y) \, dA = \iint_S f(\mathbf{T}(u,v)) \, |\det(\mathbf{J}_\mathbf{T})| \, du \, dv
   $$

5. **Polar Coordinates**:
   $$
   \iint_R f(x, y) \, dA = \iint_S f(r\cos\theta, r\sin\theta) \, r \, dr \, d\theta
   $$

6. **Spherical Coordinates**:
   $$
   \iiint_V f \, dV = \iiint_S f \, \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta
   $$

7. **Fubini's Theorem**: If $f$ is continuous on a rectangle, then:
   $$
   \iint_R f(x, y) \, dA = \int_a^b \int_c^d f(x, y) \, dy \, dx = \int_c^d \int_a^b f(x, y) \, dx \, dy
   $$

8. **Volume as Double Integral**:
   $$
   V = \iint_R [f(x, y) - g(x, y)] \, dA \quad \text{(volume between two surfaces)}
   $$

## Properties

1. **Linearity**: $\iint (af + bg) \, dA = a\iint f \, dA + b\iint g \, dA$.

2. **Additivity**: If $R = R_1 \cup R_2$ with disjoint interiors, then $\iint_R f \, dA = \iint_{R_1} f \, dA + \iint_{R_2} f \, dA$.

3. **Monotonicity**: If $f(x, y) \leq g(x, y)$ on $R$, then $\iint f \, dA \leq \iint g \, dA$.

4. **Absolute Value**: $\left|\iint_R f \, dA\right| \leq \iint_R |f| \, dA$.

5. **Fubini's Theorem** (Iterated Integration): The order of integration can be swapped as long as $f$ is absolutely integrable.

6. **Volume Property**: $\iint_R 1 \, dA = \text{Area}(R)$, and $\iiint_V 1 \, dV = \text{Volume}(V)$.

7. **Mean Value Theorem**: There exists $(c, d) \in R$ such that $\iint_R f \, dA = f(c, d) \cdot \text{Area}(R)$.

8. **Change of Variables**: The Jacobian determinant provides the correct scaling factor for volume elements under coordinate transformations.

## Step-by-Step Worked Examples

### Example 1: Double Integral over a Rectangle

**Problem**: Compute $\iint_R (x^2 + 2y) \, dA$ where $R = [0, 2] \times [0, 3]$.

**Solution**:

Step 1: Write as an iterated integral using Fubini's theorem.

$$
\iint_R (x^2 + 2y) \, dA = \int_0^2 \int_0^3 (x^2 + 2y) \, dy \, dx
$$

Step 2: Evaluate the inner integral (with respect to $y$, treating $x$ as constant).

$$
\int_0^3 (x^2 + 2y) \, dy = [x^2 y + y^2]_0^3 = (3x^2 + 9) - 0 = 3x^2 + 9
$$

Step 3: Evaluate the outer integral.

$$
\int_0^2 (3x^2 + 9) \, dx = [x^3 + 9x]_0^2 = 8 + 18 = 26
$$

Result: $\iint_R (x^2 + 2y) \, dA = 26$.

### Example 2: Double Integral over a Type I Region

**Problem**: Compute $\iint_R 2xy \, dA$ where $R$ is the region bounded by $y = x^2$ and $y = 2x$.

**Solution**:

Step 1: Find the intersection points of the bounding curves.

$x^2 = 2x \implies x^2 - 2x = 0 \implies x(x-2) = 0$, so $x = 0$ and $x = 2$.

Step 2: Determine the region type. For a given $x$ between 0 and 2, $y$ ranges from $x^2$ (lower) to $2x$ (upper). This is a Type I region.

Step 3: Set up the iterated integral.

$$
\iint_R 2xy \, dA = \int_{x=0}^2 \int_{y=x^2}^{2x} 2xy \, dy \, dx
$$

Step 4: Evaluate the inner integral.

$$
\int_{y=x^2}^{2x} 2xy \, dy = [xy^2]_{y=x^2}^{2x} = x(2x)^2 - x(x^2)^2 = 4x^3 - x^5
$$

Step 5: Evaluate the outer integral.

$$
\int_0^2 (4x^3 - x^5) \, dx = \left[x^4 - \frac{x^6}{6}\right]_0^2 = 16 - \frac{64}{6} = 16 - \frac{32}{3} = \frac{48 - 32}{3} = \frac{16}{3}
$$

Result: $\iint_R 2xy \, dA = \frac{16}{3}$.

### Example 3: Double Integral in Polar Coordinates

**Problem**: Compute $\iint_R e^{-(x^2 + y^2)} \, dA$ where $R$ is the disk $x^2 + y^2 \leq a^2$.

**Solution**:

Step 1: Convert to polar coordinates: $x = r\cos\theta$, $y = r\sin\theta$, $dA = r \, dr \, d\theta$.
The region is $0 \leq r \leq a$, $0 \leq \theta \leq 2\pi$. Also $x^2 + y^2 = r^2$.

Step 2: Set up the integral.

$$
\iint_R e^{-(x^2 + y^2)} \, dA = \int_{\theta=0}^{2\pi} \int_{r=0}^a e^{-r^2} \cdot r \, dr \, d\theta
$$

Step 3: Evaluate the inner integral using substitution $u = r^2$, $du = 2r \, dr$, so $r \, dr = du/2$.

$$
\int_0^a r e^{-r^2} \, dr = \frac{1}{2} \int_0^{a^2} e^{-u} \, du = \frac{1}{2} [-e^{-u}]_0^{a^2} = \frac{1}{2}(1 - e^{-a^2})
$$

Step 4: Evaluate the outer integral.

$$
\int_0^{2\pi} \frac{1}{2}(1 - e^{-a^2}) \, d\theta = \frac{1}{2}(1 - e^{-a^2}) \cdot 2\pi = \pi(1 - e^{-a^2})
$$

Result: $\iint_R e^{-(x^2+y^2)} \, dA = \pi(1 - e^{-a^2})$.

As $a \to \infty$, this approaches $\pi$, giving the famous Gaussian integral $\iint_{\mathbb{R}^2} e^{-(x^2+y^2)} \, dA = \pi$, from which $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$ follows.

### Example 4: Triple Integral in Spherical Coordinates

**Problem**: Compute the volume of a sphere of radius $R$ using a triple integral.

**Solution**:

Step 1: The volume is $V = \iiint_V 1 \, dV$ where $V$ is the sphere $x^2 + y^2 + z^2 \leq R^2$.

Step 2: Convert to spherical coordinates: $\rho$ from $0$ to $R$, $\phi$ from $0$ to $\pi$, $\theta$ from $0$ to $2\pi$, with $dV = \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta$.

Step 3: Set up the integral.

$$
V = \int_{\theta=0}^{2\pi} \int_{\phi=0}^{\pi} \int_{\rho=0}^R \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta
$$

Step 4: Evaluate the inner integral over $\rho$.

$$
\int_0^R \rho^2 \, d\rho = \left[\frac{\rho^3}{3}\right]_0^R = \frac{R^3}{3}
$$

Step 5: Evaluate over $\phi$.

$$
\int_0^{\pi} \sin\phi \cdot \frac{R^3}{3} \, d\phi = \frac{R^3}{3} [-\cos\phi]_0^{\pi} = \frac{R^3}{3} (1 - (-1)) = \frac{2R^3}{3}
$$

Step 6: Evaluate over $\theta$.

$$
\int_0^{2\pi} \frac{2R^3}{3} \, d\theta = \frac{2R^3}{3} \cdot 2\pi = \frac{4\pi R^3}{3}
$$

Result: $V = \frac{4}{3}\pi R^3$, the familiar formula for the volume of a sphere.

### Example 5: Marginalisation in Probability

**Problem**: Let $X$ and $Y$ have joint PDF $f_{XY}(x, y) = \frac{3}{2}(x^2 + y^2)$ for $0 \leq x \leq 1$, $0 \leq y \leq 1$. Compute the marginal PDF $f_X(x)$ and $\mathbb{E}[X]$.

**Solution**:

Step 1: The marginal PDF $f_X(x)$ is obtained by integrating out $y$.

$$
f_X(x) = \int_0^1 \frac{3}{2}(x^2 + y^2) \, dy = \frac{3}{2} \left[x^2 y + \frac{y^3}{3}\right]_0^1 = \frac{3}{2}\left(x^2 + \frac{1}{3}\right) = \frac{3x^2}{2} + \frac{1}{2}
$$

for $0 \leq x \leq 1$.

Step 2: Compute $\mathbb{E}[X]$ as a single integral.

$$
\mathbb{E}[X] = \int_0^1 x \cdot f_X(x) \, dx = \int_0^1 x\left(\frac{3x^2}{2} + \frac{1}{2}\right) \, dx = \int_0^1 \left(\frac{3x^3}{2} + \frac{x}{2}\right) \, dx
$$

Step 3: Evaluate.

$$
= \left[\frac{3x^4}{8} + \frac{x^2}{4}\right]_0^1 = \frac{3}{8} + \frac{1}{4} = \frac{3}{8} + \frac{2}{8} = \frac{5}{8}
$$

Result: $f_X(x) = \frac{3x^2 + 1}{2}$, $\mathbb{E}[X] = \frac{5}{8}$.

Verification: $\int_0^1 f_X(x) \, dx = \int_0^1 \frac{3x^2 + 1}{2} \, dx = \left[\frac{x^3 + x}{2}\right]_0^1 = 1$. ✓

### Example 6: Change of Variables with Jacobian

**Problem**: Evaluate $\iint_R (x^2 + y^2) \, dA$ where $R$ is the parallelogram with vertices $(0,0)$, $(2,1)$, $(1,3)$, $(3,4)$.

**Solution**:

Step 1: Find a transformation that maps a rectangle to the parallelogram.
The sides are given by vectors $\mathbf{u} = (2, 1)$ and $\mathbf{v} = (1, 3)$. Let:

$$
x = 2u + v, \quad y = u + 3v
$$

where $(u, v) \in [0, 1] \times [0, 1]$ maps to the parallelogram.

Step 2: Compute the Jacobian.

$$
\mathbf{J} = \begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v}
\end{bmatrix} = \begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
$$

$\det(\mathbf{J}) = 2 \cdot 3 - 1 \cdot 1 = 5$. So $|\det(\mathbf{J})| = 5$.

Step 3: Express $x^2 + y^2$ in terms of $u$ and $v$.

$x^2 + y^2 = (2u+v)^2 + (u+3v)^2 = (4u^2 + 4uv + v^2) + (u^2 + 6uv + 9v^2) = 5u^2 + 10uv + 10v^2$

Step 4: Set up and evaluate the integral.

$$
\iint_R (x^2 + y^2) \, dA = \int_0^1 \int_0^1 (5u^2 + 10uv + 10v^2) \cdot 5 \, du \, dv
$$

$$
= 5 \int_0^1 \int_0^1 (5u^2 + 10uv + 10v^2) \, du \, dv
$$

Inner integral (over $u$):
$$
\int_0^1 (5u^2 + 10uv + 10v^2) \, du = \left[\frac{5u^3}{3} + 5u^2 v + 10v^2 u\right]_0^1 = \frac{5}{3} + 5v + 10v^2
$$

Outer integral (over $v$):
$$
5 \int_0^1 \left(\frac{5}{3} + 5v + 10v^2\right) \, dv = 5 \left[\frac{5v}{3} + \frac{5v^2}{2} + \frac{10v^3}{3}\right]_0^1 = 5\left(\frac{5}{3} + \frac{5}{2} + \frac{10}{3}\right)
$$

$$
= 5\left(\frac{5}{3} + \frac{10}{3} + \frac{5}{2}\right) = 5\left(5 + \frac{5}{2}\right) = 5 \cdot \frac{15}{2} = \frac{75}{2}
$$

Result: $\iint_R (x^2 + y^2) \, dA = \frac{75}{2}$.

## Visual Interpretation

A double integral $\iint_R f(x, y) \, dA$ computes the volume under the surface $z = f(x, y)$ above the region $R$ in the $xy$-plane. Visualise this as stacking vertical columns (prisms) of infinitesimal cross-section $dA = dx \, dy$ and height $f(x, y)$. The sum of all these column volumes gives the total volume.

For $\iint_R 1 \, dA$, the surface is a flat plane at $z = 1$, so the integral equals the area of $R$. This is why the double integral of 1 gives area — it's the volume under a flat surface of height 1, which numerically equals the base area.

For triple integrals, visualisation is more challenging since we are integrating over 3D space. The quantity $\iiint_V f(x, y, z) \, dV$ can be thought of as summing the values of $f$ weighted by infinitesimal volume elements $dV$.

When changing coordinates, the Jacobian determinant captures how the infinitesimal volume element changes. In polar coordinates, the area element $r \, dr \, d\theta$ can be visualised as a "curved rectangle": an infinitesimal sector of a circle with radial length $dr$ and arc length $r \, d\theta$. The area is $dr \cdot r \, d\theta = r \, dr \, d\theta$.

## Common Mistakes

1. **Incorrect limits of integration**: Setting up the wrong bounds for iterated integrals is the most common error. The outer integral's limits must be constants, while inner limits can depend on outer variables but never on inner variables. Always draw the region first.

2. **Forgetting the Jacobian in coordinate changes**: When switching to polar, cylindrical, or spherical coordinates, the extra factor ($r$, $\rho^2 \sin\phi$, etc.) from the Jacobian determinant must be included. Forgetting it leads to incorrect results.

3. **Confusing the order of integration**: In $\int_a^b \int_{g_1(x)}^{g_2(x)} f(x,y) \, dy \, dx$, the inner integral is with respect to $y$ (with limits that may depend on $x$), and the outer is with respect to $x$ (with constant limits). Reversing this without adjusting limits is incorrect.

4. **Assuming Fubini's theorem always applies**: Fubini's theorem requires absolute integrability or non-negativity of the integrand. For conditionally convergent integrals, the order of integration matters.

5. **Incorrect region description**: Describing a 2D region as Type I when it should be Type II (or splitting into multiple subregions) leads to integration limits that don't capture the region correctly. Some regions require splitting into multiple integrals.

6. **Mishandling triple integral limits**: The limits in triple integrals follow the same nesting pattern as double integrals, but with three levels. The outermost limits must be constants; middle limits can depend on the outermost variable only; innermost limits can depend on both outer variables.

7. **Confusing cylindrical and spherical coordinates**: Cylindrical $(r, \theta, z)$ uses the polar angle $\theta$ and a linear $z$ coordinate, with Jacobian $r$. Spherical $(\rho, \phi, \theta)$ uses two angles, with Jacobian $\rho^2 \sin\phi$. The angle conventions vary; always specify which convention you use.

8. **Forgetting the absolute value of the Jacobian**: The change of variables formula uses $|\det(\mathbf{J})|$, not $\det(\mathbf{J})$. The absolute value ensures that volume elements are positive.

## Interview Questions

### Beginner

1. **Q**: What is a double integral and what does it represent geometrically?
   **A**: A double integral $\iint_R f(x,y) \, dA$ sums $f$ over region $R$. When $f \geq 0$, it represents the volume under the surface $z = f(x,y)$ above $R$.

2. **Q**: State Fubini's theorem.
   **A**: For a continuous function on a rectangle $[a,b] \times [c,d]$, $\iint f \, dA = \int_a^b \int_c^d f(x,y) \, dy \, dx = \int_c^d \int_a^b f(x,y) \, dx \, dy$.

3. **Q**: Compute $\iint_R 6 \, dA$ where $R = [0, 3] \times [0, 2]$.
   **A**: $\int_0^3 \int_0^2 6 \, dy \, dx = \int_0^3 12 \, dx = 36$. Alternatively, area is $3 \times 2 = 6$, times $6$ gives $36$.

4. **Q**: What is the Jacobian for polar coordinates?
   **A**: $|\det(\mathbf{J})| = r$, so $dA = r \, dr \, d\theta$.

5. **Q**: How do you compute the area of a region using a double integral?
   **A**: $\text{Area}(R) = \iint_R 1 \, dA$.

### Intermediate

1. **Q**: Set up the double integral to compute the volume under $z = x^2 + y^2$ over the disk $x^2 + y^2 \leq 4$ in polar coordinates.
   **A**: $V = \int_0^{2\pi} \int_0^2 (r^2) \cdot r \, dr \, d\theta = \int_0^{2\pi} \int_0^2 r^3 \, dr \, d\theta = \int_0^{2\pi} 4 \, d\theta = 8\pi$.

2. **Q**: Given joint PDF $f_{XY}(x,y) = 6xy^2$ for $0 \leq x \leq 1$, $0 \leq y \leq 1$, compute the marginal $f_Y(y)$.
   **A**: $f_Y(y) = \int_0^1 6xy^2 \, dx = [3x^2 y^2]_0^1 = 3y^2$ for $0 \leq y \leq 1$.

3. **Q**: What is the change of variables formula for double integrals and why is the Jacobian needed?
   **A**: $\iint_R f(x,y) \, dA = \iint_S f(x(u,v), y(u,v)) \, |\det(\mathbf{J})| \, du \, dv$. The Jacobian measures how the transformation scales volumes locally; without it, the integral would not correctly account for the distortion of the area element.

4. **Q**: Set up the triple integral for the volume of the region bounded by $z = 4 - x^2 - y^2$ and $z = 0$.
   **A**: In cylindrical coordinates: $V = \int_0^{2\pi} \int_0^2 \int_0^{4-r^2} r \, dz \, dr \, d\theta = \int_0^{2\pi} \int_0^2 r(4-r^2) \, dr \, d\theta$.

5. **Q**: Explain how marginalisation in probability is an application of multiple integration.
   **A**: Given joint PDF $p(\mathbf{x}, \mathbf{z})$, the marginal $p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$ integrates out the latent variables $\mathbf{z}$. This is a multiple integral over all dimensions of $\mathbf{z}$, essential for computing likelihoods in latent variable models.

### Advanced

1. **Q**: In variational inference, we optimise $\text{ELBO}(q) = \int q(\mathbf{z}) \log \frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})} \, d\mathbf{z}$. Explain why this integral is typically intractable for high-dimensional $\mathbf{z}$ and how the mean-field approximation makes it tractable.
   **A**: For high-dimensional $\mathbf{z}$, the integral is over a large space with complex dependencies. The mean-field approximation assumes $q(\mathbf{z}) = \prod_i q_i(z_i)$, making the integral decompose into a sum of lower-dimensional integrals: $\text{ELBO} = \sum_i \int q_i(z_i) \mathbb{E}_{q_{-i}}[\log p(\mathbf{x},\mathbf{z})] \, dz_i - \sum_i \int q_i(z_i) \log q_i(z_i) \, dz_i + \text{const}$. Each term involves only single-variable integrals, which are tractable for conjugate exponential families through coordinate ascent updates.

2. **Q**: Derive the Gaussian integral $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$ using a double integral in polar coordinates.
   **A**: Let $I = \int_{-\infty}^{\infty} e^{-x^2} \, dx$. Then $I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2+y^2)} \, dx \, dy$. Convert to polar: $x = r\cos\theta$, $y = r\sin\theta$, $dx\,dy = r\,dr\,d\theta$. The region is the entire plane: $0 \leq r < \infty$, $0 \leq \theta \leq 2\pi$. So $I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-r^2} r \, dr \, d\theta$. The inner integral: $\int_0^{\infty} re^{-r^2} \, dr = \left[-\frac{1}{2}e^{-r^2}\right]_0^{\infty} = \frac{1}{2}$. Then $I^2 = \int_0^{2\pi} \frac{1}{2} \, d\theta = \pi$, so $I = \sqrt{\pi}$. This double-integral trick is foundational: it derives the normalising constant of the Gaussian distribution and appears throughout ML in kernel methods, Gaussian processes, and Bayesian inference.

3. **Q**: The Bayesian evidence $p(D) = \int p(D|\boldsymbol{\theta}) p(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$ is a high-dimensional integral. Explain the Laplace approximation for estimating this integral and its limitations.
   **A**: The Laplace approximation models the posterior as a Gaussian centred at the MAP estimate $\boldsymbol{\theta}_{\text{MAP}}$. A second-order Taylor expansion of $\log p(\boldsymbol{\theta}|D)$ around $\boldsymbol{\theta}_{\text{MAP}}$ gives $\log p(\boldsymbol{\theta}|D) \approx \log p(\boldsymbol{\theta}_{\text{MAP}}|D) - \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{MAP}})^T \mathbf{H}_{\text{MAP}} (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{MAP}})$, where $\mathbf{H}_{\text{MAP}} = -\nabla^2 \log p(\boldsymbol{\theta}|D)|_{\boldsymbol{\theta}_{\text{MAP}}}$ is the observed Fisher information. Then $p(D) = \frac{p(D|\boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\boldsymbol{\theta}|D)}$, and using the Gaussian approximation of $p(\boldsymbol{\theta}|D)$, we get $\log p(D) \approx \log p(D|\boldsymbol{\theta}_{\text{MAP}}) + \log p(\boldsymbol{\theta}_{\text{MAP}}) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det(\mathbf{H}_{\text{MAP}})$. Limitations: (1) The posterior may be multimodal, non-Gaussian, or skewed; (2) requires computing the Hessian determinant; (3) the approximation is poor near boundaries or for small sample sizes; (4) fails when $\mathbf{H}_{\text{MAP}}$ is near-singular.

## Practice Problems

### Easy

1. Compute $\iint_R (x + y) \, dA$ where $R = [0, 1] \times [0, 2]$.

2. Compute $\iint_R 3x^2y \, dA$ where $R = [0, 2] \times [0, 1]$.

3. Find the area of the region bounded by $y = x$ and $y = x^2$ using a double integral.

4. Convert $\iint_R f(x, y) \, dA$ to polar coordinates where $R$ is the disk $x^2 + y^2 \leq 9$.

5. Compute $\iiint_V 1 \, dV$ where $V = [0, 1] \times [0, 2] \times [0, 3]$.

### Medium

1. Compute $\iint_R (4 - x^2 - y^2) \, dA$ where $R$ is the disk $x^2 + y^2 \leq 1$ (use polar coordinates).

2. Evaluate $\int_0^1 \int_{y}^{1} x^2 e^{xy} \, dx \, dy$ by reversing the order of integration.

3. Find the volume of the region bounded by $z = x^2 + y^2$ and $z = 4$ using a triple integral in cylindrical coordinates.

4. Given joint PDF $f_{XY}(x, y) = 2e^{-x}e^{-2y}$ for $x \geq 0$, $y \geq 0$, compute $P(X < Y)$ as a double integral and evaluate.

5. Compute $\iint_R xy \, dA$ where $R$ is the triangle with vertices $(0,0)$, $(1,0)$, $(0,1)$.

### Hard

1. Compute the volume of the region inside the sphere $x^2 + y^2 + z^2 = 4$ and above the cone $z = \sqrt{x^2 + y^2}$ using spherical coordinates.

2. Evaluate $\int_0^1 \int_0^1 \int_0^1 \frac{1}{(1 + x + y + z)^3} \, dz \, dy \, dx$.

3. Use a change of variables to evaluate $\iint_R e^{(y-x)/(y+x)} \, dA$ where $R$ is the square bounded by $x=0$, $y=0$, $x+y=1$, $x+y=2$, and $y-x$ ranges from $-1$ to $1$ (use $u = y-x$, $v = y+x$).

## Solutions

### Easy Solutions

**Solution 1**: $\int_0^1 \int_0^2 (x+y) \, dy \, dx = \int_0^1 [xy + y^2/2]_0^2 \, dx = \int_0^1 (2x + 2) \, dx = [x^2 + 2x]_0^1 = 3$.

**Solution 2**: $\int_0^2 \int_0^1 3x^2y \, dy \, dx = \int_0^2 [\frac{3x^2y^2}{2}]_0^1 \, dx = \int_0^2 \frac{3x^2}{2} \, dx = [\frac{x^3}{2}]_0^2 = 4$.

**Solution 3**: Intersection: $x = x^2 \implies x = 0, 1$. $A = \int_0^1 \int_{x^2}^x 1 \, dy \, dx = \int_0^1 (x - x^2) \, dx = [\frac{x^2}{2} - \frac{x^3}{3}]_0^1 = \frac{1}{2} - \frac{1}{3} = \frac{1}{6}$.

**Solution 4**: $x = r\cos\theta$, $y = r\sin\theta$, $dA = r\,dr\,d\theta$, $0 \leq r \leq 3$, $0 \leq \theta \leq 2\pi$.
$\iint_R f(x,y)\,dA = \int_0^{2\pi} \int_0^3 f(r\cos\theta, r\sin\theta) \, r \, dr \, d\theta$.

**Solution 5**: $\iiint_V 1 \, dV = 1 \cdot 2 \cdot 3 = 6$ (volume of a rectangular box).

### Medium Solutions

**Solution 1**: $V = \int_0^{2\pi} \int_0^1 (4 - r^2) \cdot r \, dr \, d\theta = \int_0^{2\pi} \int_0^1 (4r - r^3) \, dr \, d\theta = \int_0^{2\pi} [2r^2 - r^4/4]_0^1 \, d\theta = \int_0^{2\pi} (2 - 1/4) \, d\theta = \frac{7}{4} \cdot 2\pi = \frac{7\pi}{2}$.

**Solution 2**: Original: $0 \leq y \leq 1$, $y \leq x \leq 1$. Reversed: $0 \leq x \leq 1$, $0 \leq y \leq x$.
$\int_0^1 \int_0^x x^2 e^{xy} \, dy \, dx = \int_0^1 x^2 [e^{xy}/x]_0^x \, dx = \int_0^1 x(e^{x^2} - 1) \, dx$.
Let $u = x^2$, $du = 2x\,dx$. $\frac{1}{2} \int_0^1 (e^u - 1) \, du = \frac{1}{2}[e^u - u]_0^1 = \frac{1}{2}(e - 1 - 0) = \frac{e-1}{2}$.

**Solution 3**: In cylindrical: $z = r^2$, $z = 4$. $V = \int_0^{2\pi} \int_0^2 \int_{r^2}^4 r \, dz \, dr \, d\theta = \int_0^{2\pi} \int_0^2 r(4 - r^2) \, dr \, d\theta = 2\pi [2r^2 - r^4/4]_0^2 = 2\pi(8 - 4) = 8\pi$.

**Solution 4**: $P(X < Y) = \int_0^{\infty} \int_0^y 2e^{-x}e^{-2y} \, dx \, dy = \int_0^{\infty} 2e^{-2y}[-e^{-x}]_0^y \, dy = \int_0^{\infty} 2e^{-2y}(1 - e^{-y}) \, dy = \int_0^{\infty} (2e^{-2y} - 2e^{-3y}) \, dy = [ -e^{-2y} + \frac{2}{3}e^{-3y}]_0^{\infty} = 0 - (-1 + \frac{2}{3}) = \frac{1}{3}$.

**Solution 5**: Region: $0 \leq x \leq 1$, $0 \leq y \leq 1-x$.
$\iint_R xy \, dA = \int_0^1 \int_0^{1-x} xy \, dy \, dx = \int_0^1 x [y^2/2]_0^{1-x} \, dx = \frac{1}{2} \int_0^1 x(1-x)^2 \, dx = \frac{1}{2} \int_0^1 (x - 2x^2 + x^3) \, dx = \frac{1}{2}[\frac{x^2}{2} - \frac{2x^3}{3} + \frac{x^4}{4}]_0^1 = \frac{1}{2}(\frac{1}{2} - \frac{2}{3} + \frac{1}{4}) = \frac{1}{2} \cdot \frac{6-8+3}{12} = \frac{1}{2} \cdot \frac{1}{12} = \frac{1}{24}$.

### Hard Solutions

**Solution 1**: Intersection: $z = \sqrt{x^2+y^2}$ and $x^2+y^2+z^2 = 4$ gives $\rho^2 = 4$, so $\rho = 2$, and $\rho\cos\phi = \sqrt{\rho^2\sin^2\phi} = \rho\sin\phi \implies \cos\phi = \sin\phi \implies \phi = \pi/4$.
$V = \int_0^{2\pi} \int_0^{\pi/4} \int_0^2 \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta = \int_0^{2\pi} d\theta \int_0^{\pi/4} \sin\phi \, d\phi \int_0^2 \rho^2 \, d\rho = 2\pi \cdot [-\cos\phi]_0^{\pi/4} \cdot [\rho^3/3]_0^2 = 2\pi \cdot (-\frac{\sqrt{2}}{2} + 1) \cdot \frac{8}{3} = \frac{16\pi}{3}(1 - \frac{\sqrt{2}}{2}) = \frac{8\pi}{3}(2 - \sqrt{2})$.

**Solution 2**: Let $u = 1 + x + y + z$. Then $du = dz$ (treat $x,y$ as constants while integrating over $z$).
Inner: $\int_0^1 (1+x+y+z)^{-3} \, dz = [-\frac{1}{2}(1+x+y+z)^{-2}]_0^1 = -\frac{1}{2}[(2+x+y)^{-2} - (1+x+y)^{-2}]$.
Middle over $y$: $-\frac{1}{2} \int_0^1 [(2+x+y)^{-2} - (1+x+y)^{-2}] \, dy = -\frac{1}{2}[( -1)(2+x+y)^{-1} - (-1)(1+x+y)^{-1}]_0^1 = \frac{1}{2}[(2+x+y)^{-1} - (1+x+y)^{-1}]_0^1 = \frac{1}{2}[(3+x)^{-1} - (2+x)^{-1} - ((2+x)^{-1} - (1+x)^{-1})] = \frac{1}{2}[(3+x)^{-1} - 2(2+x)^{-1} + (1+x)^{-1}]$.
Outer over $x$: $\frac{1}{2} \int_0^1 [(3+x)^{-1} - 2(2+x)^{-1} + (1+x)^{-1}] \, dx = \frac{1}{2}[\ln(3+x) - 2\ln(2+x) + \ln(1+x)]_0^1 = \frac{1}{2}[\ln 4 - 2\ln 3 + \ln 2 - (\ln 3 - 2\ln 2 + \ln 1)] = \frac{1}{2}[\ln 4 + \ln 2 - 2\ln 3 - \ln 3 + 2\ln 2] = \frac{1}{2}[2\ln 2 + \ln 2 - 3\ln 3 + 2\ln 2] = \frac{1}{2}[5\ln 2 - 3\ln 3] = \frac{1}{2}\ln(32/27)$.

**Solution 3**: Let $u = y - x$, $v = y + x$. Then $x = (v-u)/2$, $y = (u+v)/2$, $\det(\mathbf{J}) = -1/2$, so $|\det(\mathbf{J})| = 1/2$. The region $R$ maps to $1 \leq v \leq 2$, $-1 \leq u \leq 1$.
$\iint_R e^{(y-x)/(y+x)} \, dA = \int_{v=1}^2 \int_{u=-1}^1 e^{u/v} \cdot \frac{1}{2} \, du \, dv = \frac{1}{2} \int_1^2 v(e^{1/v} - e^{-1/v}) \, dv = \int_1^2 v \sinh(1/v) \, dv$.
This integral can be evaluated numerically.

## Related Concepts

- **Single-Variable Integration**: The foundation from which multiple integrals generalise (MATH-061, MATH-062)
- **Jacobian Matrix and Determinant**: Scaling factor for change of variables (MATH-059)
- **Coordinate Systems**: Polar, cylindrical, spherical coordinates simplify many multiple integrals
- **Fubini's Theorem**: Justifies evaluating multiple integrals as iterated single integrals
- **Probability Density Functions**: Multi-dimensional PDFs require multiple integrals for normalisation and marginalisation
- **Partial Derivatives**: Building blocks for describing regions and transformations

## Next Concepts

- **Vector Calculus**: Divergence, gradient, curl expressed through integrals (Green's, Stokes', Divergence theorems)
- **Line and Surface Integrals**: Integrating over curves and surfaces
- **Measure Theory**: The rigorous foundation for integration in $\mathbb{R}^n$
- **Monte Carlo Integration**: Numerical approximation of high-dimensional multiple integrals
- **Integral Transforms**: Multi-dimensional Fourier and Laplace transforms
- **Bayesian Computation**: MCMC and variational inference for high-dimensional integrals

## Summary

Multiple integrals extend the definite integral to functions of two or more variables. Double integrals $\iint_R f(x,y) \, dA$ compute volumes under surfaces, while triple integrals $\iiint_V f(x,y,z) \, dV$ extend to three dimensions. Fubini's theorem allows evaluation as iterated single integrals. Coordinate transformations with the Jacobian determinant provide flexibility in setting up integrals for complex regions.

In machine learning and statistics, multiple integrals are fundamental for multi-dimensional probability: joint distributions, marginalisation, expectations, and Bayesian evidence. High-dimensional integrals are typically intractable and require approximation methods like variational inference, MCMC, and Laplace approximation.

## Key Takeaways

- Double integrals generalise single integrals to 2D regions; triple integrals extend to 3D
- Fubini's theorem allows iterated integration in any order
- The Jacobian determinant is essential for change of variables in multiple integrals
- Polar coordinates: $dA = r \, dr \, d\theta$; cylindrical: $dV = r \, dz \, dr \, d\theta$; spherical: $dV = \rho^2 \sin\phi \, d\rho \, d\phi \, d\theta$
- Multiple integrals compute volumes, masses, centres of mass, and moments
- Marginalisation $p(x) = \int p(x,y) \, dy$ is a multiple integral
- $\mathbb{E}[g(X,Y)] = \iint g(x,y) f_{XY}(x,y) \, dx \, dy$
- Bayesian evidence $p(D) = \int p(D|\boldsymbol{\theta}) p(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$ is a high-dimensional multiple integral
- The Gaussian integral trick uses a double integral in polar coordinates to derive $\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$
- High-dimensional integrals in ML require approximation (variational inference, MCMC, Laplace)
