# Concept: Vector Projection

## Concept ID

MATH-018

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Define the projection of one vector onto another
- Compute the scalar projection and vector projection
- Decompose a vector into parallel and perpendicular components
- Apply vector projection to orthogonalisation and gradient decomposition
- Connect projection to AI/ML concepts like Gram-Schmidt and gradient descent

## Prerequisites

- Dot product (MATH-016)
- Vector magnitude and unit vectors
- Basic trigonometry: $\cos\theta$
- Understanding of parallel and perpendicular directions

## Definition

The **vector projection** of $\mathbf{u}$ onto $\mathbf{v}$, denoted $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$, is the component of $\mathbf{u}$ that lies in the direction of $\mathbf{v}$. It is the "shadow" that $\mathbf{u}$ casts onto the line spanned by $\mathbf{v}$.

**Scalar projection (component):**
$$
\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|} = \|\mathbf{u}\|\cos\theta
$$

**Vector projection:**
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\right) \mathbf{v}
$$

The **perpendicular component** (vector rejection) is:
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u} - \operatorname{proj}_{\mathbf{v}}(\mathbf{u})
$$

Together, $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ and $\operatorname{perp}_{\mathbf{v}}(\mathbf{u})$ form an orthogonal decomposition of $\mathbf{u}$.

## Intuition

Imagine standing outside on a sunny day. A vertical pole casts a shadow on the ground. The pole itself is $\mathbf{u}$, the ground direction is $\mathbf{v}$, and the shadow on the ground is the projection. The shadow tells you how much of the pole's length is "along" the ground, ignoring its height.

In vector terms, projection answers: "How much of $\mathbf{u}$ points in the direction of $\mathbf{v}$, and what is that vector?"

## Why This Concept Matters

Vector projection is the tool for decomposing vectors into meaningful components:

- In physics, splitting forces into normal and parallel components
- In computer graphics, computing shadow volumes and reflections
- In machine learning, orthogonalising feature vectors
- In optimisation, decomposing gradients into components along and perpendicular to constraints
- In signal processing, projecting signals onto basis vectors

Whenever you need to isolate the part of a vector that lies along a particular direction, projection is the answer.

## Historical Background

The concept of projection dates back to ancient Greek geometry — Euclid's "Elements" describes projecting points onto lines. The modern vector formulation was developed alongside the dot product by **Josiah Willard Gibbs** in the late 19th century. Projection became fundamental to linear algebra through the work of **David Hilbert** (1862–1943) on inner product spaces and orthogonal projections, which later proved essential for least squares, Fourier analysis, and quantum mechanics.

## Real World Examples

1. **Physics — Inclined Plane:** A weight on a ramp has its gravitational force $\mathbf{F}_g$ decomposed into a component parallel to the ramp (causing sliding) and a component perpendicular to the ramp (pressing into the surface). The parallel component is $\operatorname{proj}_{\text{ramp}}(\mathbf{F}_g)$.
2. **Computer Graphics — Shadow Mapping:** Projecting a 3D point onto a light's view plane to determine if it's in shadow.
3. **Navigation — GPS:** Decomposing velocity vectors into north/east/down components using projections onto coordinate axes.
4. **Structural Engineering:** Decomposing forces on a bridge beam into axial (along the beam) and shear (perpendicular) components.
5. **Game Physics:** Computing the normal force when a moving object collides with a surface — the velocity component perpendicular to the surface is projected to determine the bounce.

## AI/ML Relevance

Vector projection appears throughout machine learning in both theory and practice:

1. **Gram-Schmidt Orthogonalisation:** This algorithm uses projection to convert a set of vectors into an orthonormal basis. For each new vector $\mathbf{v}_k$, subtract its projection onto all previously orthogonalised vectors:
   $$
   \mathbf{u}_k = \mathbf{v}_k - \sum_{i=1}^{k-1} \operatorname{proj}_{\mathbf{u}_i}(\mathbf{v}_k)
   $$
   This is the foundation of QR decomposition, used in linear regression solvers.

2. **Gradient Components in Optimisation:** In constrained optimisation (e.g., training neural networks with constraints), the gradient is decomposed into parallel and perpendicular components relative to the constraint surface. The parallel component moves along the constraint; the perpendicular component is removed.

3. **Principal Component Analysis (PCA):** PCA projects high-dimensional data onto the directions of greatest variance. Each principal component is a direction; the projection of data onto that direction gives the "scores."

4. **Attention Mechanism:** In transformer models, the query vector can be seen as being projected onto key vectors to determine attention weights (via dot products, which are related to scalar projections).

5. **Feature Engineering:** When features are correlated, one can project out the correlated component (using projection) to create decorrelated features for more stable learning.

## Mathematical Explanation

The projection formula derives from the dot product. We want a vector $\mathbf{p} = c\mathbf{v}$ (some scalar multiple of $\mathbf{v}$) such that $\mathbf{u} - \mathbf{p}$ is perpendicular to $\mathbf{v}$. This means:

$$
(\mathbf{u} - c\mathbf{v}) \cdot \mathbf{v} = 0
$$

Expanding:
$$
\mathbf{u} \cdot \mathbf{v} - c(\mathbf{v} \cdot \mathbf{v}) = 0
$$

Solving for $c$:
$$
c = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}
$$

Thus:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}
$$

The scalar projection (length of the shadow) is:
$$
\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \|\mathbf{u}\|\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|}
$$

Note that $\operatorname{comp}_{\mathbf{v}}(\mathbf{u})$ can be negative if $\theta > 90^\circ$, indicating the projection points opposite to $\mathbf{v}$.

The perpendicular component is:
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u} - \operatorname{proj}_{\mathbf{v}}(\mathbf{u})
$$

We can verify orthogonality:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) \cdot \operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = 0
$$

## Formula(s)

**Scalar projection (component):**
$$
\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|} = \|\mathbf{u}\|\cos\theta
$$

**Vector projection:**
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}
$$

**Perpendicular component (rejection):**
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u} - \operatorname{proj}_{\mathbf{v}}(\mathbf{u})
$$

**If $\mathbf{v}$ is a unit vector ($\|\mathbf{v}\| = 1$):**
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = (\mathbf{u} \cdot \mathbf{v}) \mathbf{v}
$$

## Properties

1. **Linearity in $\mathbf{u}$:** $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}_1 + \mathbf{u}_2) = \operatorname{proj}_{\mathbf{v}}(\mathbf{u}_1) + \operatorname{proj}_{\mathbf{v}}(\mathbf{u}_2)$
2. **Not linear in $\mathbf{v}$:** $\operatorname{proj}_{\mathbf{v}}(c\mathbf{u}) = c\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$, but $\operatorname{proj}_{c\mathbf{v}}(\mathbf{u}) = \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$
3. **Idempotence:** $\operatorname{proj}_{\mathbf{v}}(\operatorname{proj}_{\mathbf{v}}(\mathbf{u})) = \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ (projecting twice does nothing new)
4. **Orthogonality:** $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) \perp \operatorname{perp}_{\mathbf{v}}(\mathbf{u})$
5. **Pythagorean theorem:** $\|\mathbf{u}\|^2 = \|\operatorname{proj}_{\mathbf{v}}(\mathbf{u})\|^2 + \|\operatorname{perp}_{\mathbf{v}}(\mathbf{u})\|^2$
6. **Zero projection:** If $\mathbf{u} \perp \mathbf{v}$, then $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \mathbf{0}$
7. **Full projection:** If $\mathbf{u}$ is parallel to $\mathbf{v}$, then $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u}$ and $\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{0}$

## Step-by-Step Worked Examples

### Example 1: Basic Vector Projection in 2D

**Problem:** Find the projection of $\mathbf{u} = \langle 3, 4 \rangle$ onto $\mathbf{v} = \langle 2, 1 \rangle$.

**Solution:**

**Step 1:** Compute $\mathbf{u} \cdot \mathbf{v}$:
$$
\mathbf{u} \cdot \mathbf{v} = 3(2) + 4(1) = 6 + 4 = 10
$$

**Step 2:** Compute $\|\mathbf{v}\|^2$:
$$
\|\mathbf{v}\|^2 = 2^2 + 1^2 = 4 + 1 = 5
$$

**Step 3:** Apply the projection formula:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{10}{5} \langle 2, 1 \rangle = 2\langle 2, 1 \rangle = \langle 4, 2 \rangle
$$

**Step 4:** Find the perpendicular component:
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \langle 3, 4 \rangle - \langle 4, 2 \rangle = \langle -1, 2 \rangle
$$

**Step 5:** Verify orthogonality:
$$
\langle 4, 2 \rangle \cdot \langle -1, 2 \rangle = 4(-1) + 2(2) = -4 + 4 = 0
$$

**Step 6:** Verify the Pythagorean theorem:
- $\|\mathbf{u}\|^2 = 3^2 + 4^2 = 9 + 16 = 25$
- $\|\operatorname{proj}\|^2 = 4^2 + 2^2 = 16 + 4 = 20$
- $\|\operatorname{perp}\|^2 = (-1)^2 + 2^2 = 1 + 4 = 5$
- $20 + 5 = 25$ ✓

### Example 2: Projection onto a Unit Vector

**Problem:** Let $\mathbf{v} = \langle 1, 0 \rangle$ (the $x$-axis unit vector). Find the projection of $\mathbf{u} = \langle 5, 12 \rangle$ onto $\mathbf{v}$.

**Solution:**

**Step 1:** Since $\mathbf{v}$ is a unit vector ($\|\mathbf{v}\| = 1$):
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = (\mathbf{u} \cdot \mathbf{v}) \mathbf{v}
$$

**Step 2:** Compute $\mathbf{u} \cdot \mathbf{v} = 5(1) + 12(0) = 5$.

**Step 3:** The projection is:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = 5 \langle 1, 0 \rangle = \langle 5, 0 \rangle
$$

**Step 4:** The perpendicular component is:
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \langle 5, 12 \rangle - \langle 5, 0 \rangle = \langle 0, 12 \rangle
$$

**Step 5:** Interpretation: The vector $\mathbf{u}$ has 5 units in the $x$-direction and 12 units in the $y$-direction. Projection extracts just the $x$-component.

### Example 3: Scalar Projection (Component)

**Problem:** Find the scalar projection (component) of $\mathbf{u} = \langle 2, -3, 6 \rangle$ onto $\mathbf{v} = \langle 1, 2, 2 \rangle$.

**Solution:**

**Step 1:** Compute $\mathbf{u} \cdot \mathbf{v}$:
$$
\mathbf{u} \cdot \mathbf{v} = 2(1) + (-3)(2) + 6(2) = 2 - 6 + 12 = 8
$$

**Step 2:** Compute $\|\mathbf{v}\|$:
$$
\|\mathbf{v}\| = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3
$$

**Step 3:** The scalar projection is:
$$
\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{8}{3} \approx 2.67
$$

**Step 4:** The vector projection is:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{8}{3^2} \langle 1, 2, 2 \rangle = \frac{8}{9} \langle 1, 2, 2 \rangle = \left\langle \frac{8}{9}, \frac{16}{9}, \frac{16}{9} \right\rangle
$$

**Step 5:** The scalar projection $\frac{8}{3}$ tells us the length of the "shadow" of $\mathbf{u}$ onto $\mathbf{v}$, while the vector projection tells us the exact vector.

### Example 4: Projection with a Negative Component

**Problem:** Find the projection of $\mathbf{u} = \langle 4, -1 \rangle$ onto $\mathbf{v} = \langle -2, 3 \rangle$.

**Solution:**

**Step 1:** Compute $\mathbf{u} \cdot \mathbf{v} = 4(-2) + (-1)(3) = -8 - 3 = -11$.

**Step 2:** Compute $\|\mathbf{v}\|^2 = (-2)^2 + 3^2 = 4 + 9 = 13$.

**Step 3:** Compute the projection:
$$
\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{-11}{13} \langle -2, 3 \rangle = \left\langle \frac{22}{13}, -\frac{33}{13} \right\rangle \approx \langle 1.692, -2.538 \rangle
$$

**Step 4:** The scalar projection is $\frac{-11}{\sqrt{13}} \approx -3.05$, which is negative. This means the projection of $\mathbf{u}$ onto $\mathbf{v}$ points in the opposite direction of $\mathbf{v}$.

**Step 5:** The perpendicular component:
$$
\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \langle 4, -1 \rangle - \left\langle \frac{22}{13}, -\frac{33}{13} \right\rangle = \left\langle \frac{52-22}{13}, \frac{-13+33}{13} \right\rangle = \left\langle \frac{30}{13}, \frac{20}{13} \right\rangle
$$

### Example 5: Decomposition in 3D

**Problem:** Decompose $\mathbf{u} = \langle 1, 2, 3 \rangle$ into components parallel and perpendicular to $\mathbf{v} = \langle 1, 1, 1 \rangle$.

**Solution:**

**Step 1:** Compute $\mathbf{u} \cdot \mathbf{v} = 1(1) + 2(1) + 3(1) = 6$.

**Step 2:** Compute $\|\mathbf{v}\|^2 = 1^2 + 1^2 + 1^2 = 3$.

**Step 3:** Parallel component:
$$
\mathbf{u}_{\parallel} = \operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{6}{3} \langle 1, 1, 1 \rangle = 2\langle 1, 1, 1 \rangle = \langle 2, 2, 2 \rangle
$$

**Step 4:** Perpendicular component:
$$
\mathbf{u}_{\perp} = \mathbf{u} - \mathbf{u}_{\parallel} = \langle 1, 2, 3 \rangle - \langle 2, 2, 2 \rangle = \langle -1, 0, 1 \rangle
$$

**Step 5:** Verify: $\mathbf{u}_{\parallel} \cdot \mathbf{u}_{\perp} = 2(-1) + 2(0) + 2(1) = -2 + 0 + 2 = 0$ ✓

**Step 6:** The original vector is $\mathbf{u} = \langle 2, 2, 2 \rangle + \langle -1, 0, 1 \rangle$.

## Visual Interpretation

Picture two vectors $\mathbf{u}$ and $\mathbf{v}$ starting from the same point. Draw a line through $\mathbf{v}$ (this is the direction $\mathbf{v}$ defines). Drop a perpendicular from the tip of $\mathbf{u}$ to this line. The point where the perpendicular meets the line is the tip of the projection vector.

The projection $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ always lies on the line of $\mathbf{v}$. The perpendicular component $\operatorname{perp}_{\mathbf{v}}(\mathbf{u})$ connects the tip of $\mathbf{u}$ to the tip of the projection, forming a right triangle with $\mathbf{u}$ as the hypotenuse.

This right triangle illustrates $\|\mathbf{u}\|^2 = \|\operatorname{proj}\|^2 + \|\operatorname{perp}\|^2$ (the Pythagorean theorem).

## Common Mistakes

1. **Forgetting to square the norm in the denominator:** The formula is $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$, not $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|} \mathbf{v}$. The scalar projection uses $\|\mathbf{v}\|$; the vector projection uses $\|\mathbf{v}\|^2$.

2. **Confusing scalar and vector projection:** The scalar projection $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|}$ is a single number (the signed length). The vector projection is an actual vector.

3. **Projecting onto the wrong vector:** $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ projects $\mathbf{u}$ onto $\mathbf{v}$, not the other way around. These are different: $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) \neq \operatorname{proj}_{\mathbf{u}}(\mathbf{v})$ in general.

4. **Thinking projection is symmetric:** Projection is not symmetric. The shadow of $\mathbf{u}$ on $\mathbf{v}$ has different length than the shadow of $\mathbf{v}$ on $\mathbf{u}$.

5. **Ignoring the sign:** The scalar projection can be negative when the angle exceeds $90^\circ$. A negative scalar projection means the projected vector points opposite to $\mathbf{v}$.

6. **Assuming the perpendicular component is zero:** Unless $\mathbf{u}$ is already parallel to $\mathbf{v}$, there is always a perpendicular component. The decomposition always yields two non-zero parts (except when $\mathbf{u}$ and $\mathbf{v}$ are parallel or perpendicular).

7. **Dividing by zero when $\mathbf{v} = \mathbf{0}$:** Projection onto the zero vector is undefined. The projection formula requires $\|\mathbf{v}\| \neq 0$.

## Interview Questions

### Beginner

1. What is the vector projection of $\mathbf{u}$ onto $\mathbf{v}$?
2. How do you compute the scalar projection of $\mathbf{u}$ onto $\mathbf{v}$?
3. What is the relationship between the scalar projection and the dot product?
4. If $\mathbf{u}$ is perpendicular to $\mathbf{v}$, what is $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$?
5. Does $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \operatorname{proj}_{\mathbf{u}}(\mathbf{v})$ in general? Explain.

### Intermediate

1. Derive the formula $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$.
2. Show that $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ and $\operatorname{perp}_{\mathbf{v}}(\mathbf{u})$ are orthogonal.
3. Prove the Pythagorean property: $\|\mathbf{u}\|^2 = \|\operatorname{proj}_{\mathbf{v}}(\mathbf{u})\|^2 + \|\operatorname{perp}_{\mathbf{v}}(\mathbf{u})\|^2$.
4. How is vector projection used in the Gram-Schmidt orthogonalisation process?
5. Given $\mathbf{u} = \langle 3, 1, -2 \rangle$ and $\mathbf{v} = \langle 1, -1, 1 \rangle$, find $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ and $\operatorname{perp}_{\mathbf{v}}(\mathbf{u})$.

### Advanced

1. Prove that the projection operator $P_{\mathbf{v}}(\mathbf{u}) = \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ is idempotent: $P_{\mathbf{v}}(P_{\mathbf{v}}(\mathbf{u})) = P_{\mathbf{v}}(\mathbf{u})$.
2. Show that the matrix representation of projection onto the line spanned by $\mathbf{v}$ is $P = \frac{\mathbf{v} \mathbf{v}^T}{\|\mathbf{v}\|^2}$.
3. Explain how projection matrices relate to least squares regression. Specifically, show that the solution to $\mathbf{X}\boldsymbol{\beta} = \mathbf{y}$ (least squares) involves a projection onto the column space of $\mathbf{X}$.

## Practice Problems

### Easy - 5 Questions

1. Find $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ for $\mathbf{u} = \langle 1, 2 \rangle$, $\mathbf{v} = \langle 3, 0 \rangle$.
2. Find the scalar projection of $\mathbf{u} = \langle 4, 3 \rangle$ onto $\mathbf{v} = \langle 1, 0 \rangle$.
3. If $\mathbf{u}$ is parallel to $\mathbf{v}$, what is $\operatorname{perp}_{\mathbf{v}}(\mathbf{u})$?
4. Compute $\operatorname{comp}_{\mathbf{v}}(\mathbf{u})$ for $\mathbf{u} = \langle 0, 5 \rangle$, $\mathbf{v} = \langle 0, -1 \rangle$.
5. Is $\operatorname{proj}_{\mathbf{v}}(2\mathbf{u})$ equal to $2\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$? Explain.

### Medium - 5 Questions

6. Decompose $\mathbf{u} = \langle 2, -1, 3 \rangle$ into components parallel and perpendicular to $\mathbf{v} = \langle 1, 2, 2 \rangle$.
7. Find $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ for $\mathbf{u} = \langle 1, 1, 1 \rangle$ and $\mathbf{v} = \langle 1, -1, 2 \rangle$.
8. For $\mathbf{u} = \langle 1, 2, 3 \rangle$ and $\mathbf{v} = \langle 4, 5, 6 \rangle$, find the scalar projection of $\mathbf{u}$ onto $\mathbf{v}$.
9. Show that $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \mathbf{0}$ if and only if $\mathbf{u} \perp \mathbf{v}$.
10. Given $\mathbf{u} = \langle 2, 1 \rangle$, find a vector $\mathbf{v}$ such that $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \langle 1, 0 \rangle$.

### Hard - 3 Questions

11. Prove that $\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ is the closest point on the line spanned by $\mathbf{v}$ to the tip of $\mathbf{u}$. That is, show that $\|\mathbf{u} - t\mathbf{v}\|$ is minimised when $t = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}$.
12. The projection matrix $P = \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}}$ projects any vector onto the line spanned by $\mathbf{v}$. Prove that $P^2 = P$ (idempotence) and $P^T = P$ (symmetry).
13. In the Gram-Schmidt process, given vectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k$, we define $\mathbf{u}_2 = \mathbf{v}_2 - \operatorname{proj}_{\mathbf{u}_1}(\mathbf{v}_2)$. Prove by induction that $\mathbf{u}_k$ is orthogonal to $\mathbf{u}_1, \dots, \mathbf{u}_{k-1}$.

## Solutions

### Easy Solutions

1. $\mathbf{u} \cdot \mathbf{v} = 1(3) + 2(0) = 3$. $\|\mathbf{v}\|^2 = 9$. $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{3}{9}\langle 3, 0 \rangle = \frac{1}{3}\langle 3, 0 \rangle = \langle 1, 0 \rangle$.

2. $\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|} = \frac{4(1) + 3(0)}{1} = 4$.

3. If $\mathbf{u}$ is parallel to $\mathbf{v}$, then $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u}$, so $\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u} - \mathbf{u} = \mathbf{0}$.

4. $\mathbf{u} \cdot \mathbf{v} = 0(0) + 5(-1) = -5$. $\|\mathbf{v}\| = 1$. $\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = -5$. The negative sign indicates the projection points opposite to $\mathbf{v}$.

5. Yes. $\operatorname{proj}_{\mathbf{v}}(2\mathbf{u}) = \frac{(2\mathbf{u}) \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v} = \frac{2(\mathbf{u} \cdot \mathbf{v})}{\|\mathbf{v}\|^2} \mathbf{v} = 2\operatorname{proj}_{\mathbf{v}}(\mathbf{u})$.

### Medium Solutions

6. $\mathbf{u} \cdot \mathbf{v} = 2(1) + (-1)(2) + 3(2) = 2 - 2 + 6 = 6$.
   $\|\mathbf{v}\|^2 = 1 + 4 + 4 = 9$.
   $\mathbf{u}_{\parallel} = \frac{6}{9}\langle 1, 2, 2 \rangle = \frac{2}{3}\langle 1, 2, 2 \rangle = \langle \frac{2}{3}, \frac{4}{3}, \frac{4}{3} \rangle$.
   $\mathbf{u}_{\perp} = \langle 2, -1, 3 \rangle - \langle \frac{2}{3}, \frac{4}{3}, \frac{4}{3} \rangle = \langle \frac{4}{3}, -\frac{7}{3}, \frac{5}{3} \rangle$.

7. $\mathbf{u} \cdot \mathbf{v} = 1(1) + 1(-1) + 1(2) = 1 - 1 + 2 = 2$.
   $\|\mathbf{v}\|^2 = 1 + 1 + 4 = 6$.
   $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{2}{6}\langle 1, -1, 2 \rangle = \frac{1}{3}\langle 1, -1, 2 \rangle = \langle \frac{1}{3}, -\frac{1}{3}, \frac{2}{3} \rangle$.

8. $\mathbf{u} \cdot \mathbf{v} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$.
   $\|\mathbf{v}\| = \sqrt{16 + 25 + 36} = \sqrt{77}$.
   $\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{32}{\sqrt{77}}$.

9. If $\mathbf{u} \perp \mathbf{v}$, then $\mathbf{u} \cdot \mathbf{v} = 0$, so $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{0}{\|\mathbf{v}\|^2} \mathbf{v} = \mathbf{0}$. Conversely, if $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \mathbf{0}$ and $\mathbf{v} \neq \mathbf{0}$, then $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} = 0$, so $\mathbf{u} \cdot \mathbf{v} = 0$, meaning $\mathbf{u} \perp \mathbf{v}$.

10. Let $\mathbf{v} = \langle v_1, v_2 \rangle$. We need $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v} = \langle 1, 0 \rangle$.
    $\mathbf{u} \cdot \mathbf{v} = 2v_1 + v_2$.
    This means $\frac{2v_1 + v_2}{v_1^2 + v_2^2} \langle v_1, v_2 \rangle = \langle 1, 0 \rangle$.
    For the second component to be zero: $\frac{2v_1 + v_2}{v_1^2 + v_2^2} \cdot v_2 = 0$, so $v_2 = 0$.
    Then the first component: $\frac{2v_1}{v_1^2} \cdot v_1 = \frac{2v_1^2}{v_1^2} = 2 \neq 1$.
    So there's a contradiction unless we adjust. With $v_2 = 0$, the formula gives $\frac{2v_1}{v_1^2} \langle v_1, 0 \rangle = \langle 2, 0 \rangle$. We want $\langle 1, 0 \rangle$, so no such $\mathbf{v}$ exists. Instead, we need $v_2 \neq 0$ and the scalar $\frac{2v_1 + v_2}{v_1^2 + v_2^2} \cdot v_2 = 0 \implies v_2 \neq 0$ but then the second component can't be zero. So we actually need $v_2 = 0$ but then we get $\langle 2, 0 \rangle$. There is no $\mathbf{v}$ that gives exactly $\langle 1, 0 \rangle$ because $\mathbf{u} = \langle 2, 1 \rangle$ has $y$-component 1 that must project somewhere. The problem has no solution — demonstrating that not every vector can be a projection of a given vector.

### Hard Solutions

11. Minimise $f(t) = \|\mathbf{u} - t\mathbf{v}\|^2 = (\mathbf{u} - t\mathbf{v}) \cdot (\mathbf{u} - t\mathbf{v}) = \|\mathbf{u}\|^2 - 2t(\mathbf{u} \cdot \mathbf{v}) + t^2\|\mathbf{v}\|^2$.
    Take derivative w.r.t. $t$: $f'(t) = -2(\mathbf{u} \cdot \mathbf{v}) + 2t\|\mathbf{v}\|^2$.
    Set $f'(t) = 0$: $-2(\mathbf{u} \cdot \mathbf{v}) + 2t\|\mathbf{v}\|^2 = 0 \implies t = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}$.
    The second derivative $f''(t) = 2\|\mathbf{v}\|^2 > 0$, so this is a minimum.
    Therefore, $t\mathbf{v} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v} = \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ is the closest point on the line.

12. Let $\mathbf{v}$ be a column vector. Then $P = \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}}$.
    $P^2 = \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}} \cdot \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}} = \frac{\mathbf{v} (\mathbf{v}^T \mathbf{v}) \mathbf{v}^T}{(\mathbf{v}^T \mathbf{v})^2} = \frac{(\mathbf{v}^T \mathbf{v}) \mathbf{v} \mathbf{v}^T}{(\mathbf{v}^T \mathbf{v})^2} = \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}} = P$.
    $P^T = \left(\frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}}\right)^T = \frac{(\mathbf{v}^T)^T \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}} = \frac{\mathbf{v} \mathbf{v}^T}{\mathbf{v}^T \mathbf{v}} = P$.
    So $P$ is both idempotent ($P^2 = P$) and symmetric ($P^T = P$), confirming it is an orthogonal projection matrix.

13. **Base case:** $\mathbf{u}_2 = \mathbf{v}_2 - \operatorname{proj}_{\mathbf{u}_1}(\mathbf{v}_2)$. Then $\mathbf{u}_2 \cdot \mathbf{u}_1 = \mathbf{v}_2 \cdot \mathbf{u}_1 - \frac{\mathbf{v}_2 \cdot \mathbf{u}_1}{\|\mathbf{u}_1\|^2} \mathbf{u}_1 \cdot \mathbf{u}_1 = \mathbf{v}_2 \cdot \mathbf{u}_1 - \mathbf{v}_2 \cdot \mathbf{u}_1 = 0$. So $\mathbf{u}_2 \perp \mathbf{u}_1$.
    **Inductive step:** Assume $\mathbf{u}_1, \dots, \mathbf{u}_{k-1}$ are pairwise orthogonal. Define $\mathbf{u}_k = \mathbf{v}_k - \sum_{i=1}^{k-1} \operatorname{proj}_{\mathbf{u}_i}(\mathbf{v}_k)$. For any $j < k$:
    $\mathbf{u}_k \cdot \mathbf{u}_j = \mathbf{v}_k \cdot \mathbf{u}_j - \sum_{i=1}^{k-1} \frac{\mathbf{v}_k \cdot \mathbf{u}_i}{\|\mathbf{u}_i\|^2} \mathbf{u}_i \cdot \mathbf{u}_j$.
    By the inductive hypothesis, $\mathbf{u}_i \cdot \mathbf{u}_j = 0$ when $i \neq j$, and $= \|\mathbf{u}_j\|^2$ when $i = j$.
    So: $\mathbf{u}_k \cdot \mathbf{u}_j = \mathbf{v}_k \cdot \mathbf{u}_j - \frac{\mathbf{v}_k \cdot \mathbf{u}_j}{\|\mathbf{u}_j\|^2} \cdot \|\mathbf{u}_j\|^2 = \mathbf{v}_k \cdot \mathbf{u}_j - \mathbf{v}_k \cdot \mathbf{u}_j = 0$.

## Related Concepts

- **Dot Product** (MATH-016): The dot product $\mathbf{u} \cdot \mathbf{v}$ is the numerator in the projection formula
- **Angle Between Vectors** (MATH-019): $\cos\theta$ relates to scalar projection
- **Distance Between Vectors** (MATH-020): Euclidean distance links to the perpendicular component
- **Cross Product** (MATH-017): The cross product gives a perpendicular vector, complementing projection

## Next Concepts

- **031 Gram-Schmidt Process**: Builds orthonormal bases using successive projections
- **034 Least Squares Regression**: The solution involves projecting $\mathbf{y}$ onto the column space of $\mathbf{X}$
- **036 Principal Component Analysis**: Projects data onto principal directions
- **042 QR Decomposition**: Factorisation built from Gram-Schmidt (projections)

## Summary

Vector projection decomposes a vector $\mathbf{u}$ into a component parallel to $\mathbf{v}$ and a component perpendicular to $\mathbf{v}$. The parallel component is $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$, and the perpendicular component is $\mathbf{u} - \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$. These components are orthogonal, and their squared lengths sum to $\|\mathbf{u}\|^2$. Projection is fundamental to orthogonalisation, linear regression, and many optimisation techniques in machine learning.

## Key Takeaways

- $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$ gives the vector component of $\mathbf{u}$ along $\mathbf{v}$
- $\operatorname{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|}$ gives the signed length
- The perpendicular component $\operatorname{perp}_{\mathbf{v}}(\mathbf{u}) = \mathbf{u} - \operatorname{proj}_{\mathbf{v}}(\mathbf{u})$ is orthogonal to $\mathbf{v}$
- $\|\mathbf{u}\|^2 = \|\operatorname{proj}\|^2 + \|\operatorname{perp}\|^2$ (Pythagorean theorem)
- Projection is idempotent: projecting twice yields the same result
- Essential for Gram-Schmidt, QR decomposition, PCA, and least squares
