# Concept: Cross Product

## Concept ID

MATH-017

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Define the cross product of two 3D vectors
- Compute the cross product using the determinant formula
- Understand the geometric meaning: magnitude equals area of parallelogram, direction follows right-hand rule
- Apply the cross product to find perpendicular vectors and areas
- Connect the cross product to 3D graphics and computer vision in AI

## Prerequisites

- Basic algebra and vector notation
- Dot product (MATH-016)
- Determinants of $2 \times 2$ and $3 \times 3$ matrices
- Right-hand rule intuition from physics
- Three-dimensional coordinate systems

## Definition

The **cross product** $\mathbf{u} \times \mathbf{v}$ of two vectors $\mathbf{u}$ and $\mathbf{v}$ in $\mathbb{R}^3$ produces a third vector that is perpendicular to both $\mathbf{u}$ and $\mathbf{v}$. Unlike the dot product (which gives a scalar), the cross product gives a vector.

**Algebraic definition:** For $\mathbf{u} = \langle u_1, u_2, u_3 \rangle$ and $\mathbf{v} = \langle v_1, v_2, v_3 \rangle$,

$$
\mathbf{u} \times \mathbf{v} = \langle u_2 v_3 - u_3 v_2,\; u_3 v_1 - u_1 v_3,\; u_1 v_2 - u_2 v_1 \rangle
$$

This can be remembered using a determinant:

$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
u_1 & u_2 & u_3 \\
v_1 & v_2 & v_3
\end{vmatrix}
$$

where $\mathbf{i}, \mathbf{j}, \mathbf{k}$ are the unit vectors along the $x$, $y$, and $z$ axes respectively.

**Geometric definition:** The magnitude of the cross product is

$$
\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta
$$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$ ($0 \leq \theta \leq \pi$). The direction is perpendicular to both $\mathbf{u}$ and $\mathbf{v}$, following the right-hand rule.

## Intuition

The cross product captures the idea of "rotational effect" or "area." If you have two vectors representing two sides of a parallelogram, the cross product gives a vector whose magnitude is the area of that parallelogram and whose direction is perpendicular to its surface (the "normal").

Imagine holding a door and pushing it open. The hinge line is the axis (direction of the cross product), and the force you apply times the distance from the hinge gives the torque — a physical example of the cross product.

## Why This Concept Matters

The cross product is essential whenever we work in 3D space:

- Computing surface normals for 3D graphics and lighting
- Calculating torque and angular momentum in physics
- Finding areas and volumes in geometry
- Determining orientation and rotation in robotics
- Building 3D computer vision systems
- Defining coordinate systems for 3D modelling

It is the standard way to produce a vector perpendicular to two given vectors.

## Historical Background

The cross product, like the dot product, emerged from **William Rowan Hamilton**'s quaternions (1843). When Hamilton multiplied two quaternions, the imaginary parts produced a term that behaved like a cross product. **Josiah Willard Gibbs** later extracted and formalised the cross product as a separate vector operation in his 1881 pamphlet "Elements of Vector Analysis." The modern notation $\mathbf{u} \times \mathbf{v}$ is attributed to Gibbs. The cross product is specific to three dimensions; in other dimensions, generalised operations like the wedge product (from geometric algebra) are used.

## Real World Examples

1. **Torque in Physics:** $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$, where $\mathbf{r}$ is the position vector from the pivot and $\mathbf{F}$ is the force. Torque is maximised when force is applied perpendicular to the lever arm.
2. **3D Graphics Lighting:** Surface normals (used for lighting calculations like Phong shading) are computed via cross products of edge vectors on a triangle.
3. **Robotics:** The cross product appears in the computation of angular velocity and the Jacobian matrix for robot arm kinematics.
4. **Flight Dynamics:** The lift force on an airplane wing involves cross products of velocity and vorticity vectors.
5. **Computer-Aided Design (CAD):** Cross products determine whether a polygon is front-facing or back-facing relative to the viewer.

## AI/ML Relevance

While the cross product is less ubiquitous in AI than the dot product, it is critical in specific domains:

1. **3D Computer Vision:** When processing point clouds or 3D meshes (e.g., in autonomous driving or robotics), surface normals are computed using the cross product of neighbouring points. These normals feed into feature extraction pipelines for object detection and segmentation.

2. **Neural Rendering:** NeRF and other neural rendering techniques use ray-surface intersections that rely on normal vectors computed via cross products.

3. **Geometric Deep Learning:** Graph neural networks operating on 3D meshes (e.g., MeshCNN, SpiralNet++) compute face normals and edge orientations using cross products to encode local geometry.

4. **Physics Simulation in AI:** Reinforcement learning environments that simulate physics (e.g., Mujoco, Isaac Gym) compute torques and forces using cross products internally.

5. **Grasp Pose Estimation:** Robot grasping algorithms use cross products to determine the orientation of gripper fingers relative to object surfaces.

## Mathematical Explanation

The cross product is defined only in $\mathbb{R}^3$. Its components can be derived from the determinant of a $3 \times 3$ matrix:

$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
u_1 & u_2 & u_3 \\
v_1 & v_2 & v_3
\end{vmatrix}
= \mathbf{i}(u_2 v_3 - u_3 v_2) - \mathbf{j}(u_1 v_3 - u_3 v_1) + \mathbf{k}(u_1 v_2 - u_2 v_1)
$$

This yields:
- **First component (x):** $u_2 v_3 - u_3 v_2$
- **Second component (y):** $-(u_1 v_3 - u_3 v_1) = u_3 v_1 - u_1 v_3$
- **Third component (z):** $u_1 v_2 - u_2 v_1$

The magnitude $\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta$ equals the area of the parallelogram spanned by $\mathbf{u}$ and $\mathbf{v}$.

Key observations:
- If $\mathbf{u}$ and $\mathbf{v}$ are parallel ($\theta = 0$ or $\pi$), $\sin\theta = 0$, so the cross product is $\mathbf{0}$.
- If $\mathbf{u}$ and $\mathbf{v}$ are perpendicular ($\theta = \pi/2$), $\sin\theta = 1$, so the magnitude is maximised.
- The direction is given by the right-hand rule: point your index finger along $\mathbf{u}$, middle finger along $\mathbf{v}$, and your thumb points along $\mathbf{u} \times \mathbf{v}$.

## Formula(s)

**Component form:**
$$
\mathbf{u} \times \mathbf{v} = \langle u_2 v_3 - u_3 v_2,\; u_3 v_1 - u_1 v_3,\; u_1 v_2 - u_2 v_1 \rangle
$$

**Determinant form:**
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
u_1 & u_2 & u_3 \\
v_1 & v_2 & v_3
\end{vmatrix}
$$

**Magnitude:**
$$
\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta
$$

**Unit normal from two vectors:**
$$
\mathbf{n} = \frac{\mathbf{u} \times \mathbf{v}}{\|\mathbf{u} \times \mathbf{v}\|}
$$

## Properties

1. **Anti-commutativity:** $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$
2. **Not associative:** $(\mathbf{u} \times \mathbf{v}) \times \mathbf{w} \neq \mathbf{u} \times (\mathbf{v} \times \mathbf{w})$ in general
3. **Distributivity:** $\mathbf{u} \times (\mathbf{v} + \mathbf{w}) = (\mathbf{u} \times \mathbf{v}) + (\mathbf{u} \times \mathbf{w})$
4. **Scalar multiplication:** $(c\mathbf{u}) \times \mathbf{v} = c(\mathbf{u} \times \mathbf{v}) = \mathbf{u} \times (c\mathbf{v})$
5. **Parallel vectors:** $\mathbf{u} \times \mathbf{v} = \mathbf{0}$ if and only if $\mathbf{u}$ and $\mathbf{v}$ are parallel (or one is zero)
6. **Self cross product:** $\mathbf{u} \times \mathbf{u} = \mathbf{0}$
7. **Orthogonality:** $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = 0$ and $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{v} = 0$ (the result is perpendicular to both inputs)
8. **BAC-CAB identity:** $\mathbf{u} \times (\mathbf{v} \times \mathbf{w}) = \mathbf{v}(\mathbf{u} \cdot \mathbf{w}) - \mathbf{w}(\mathbf{u} \cdot \mathbf{v})$
9. **Scalar triple product:** $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{w} = \mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})$ — the volume of the parallelepiped

## Step-by-Step Worked Examples

### Example 1: Basic Cross Product in 3D

**Problem:** Compute $\mathbf{u} \times \mathbf{v}$ for $\mathbf{u} = \langle 1, 2, 3 \rangle$ and $\mathbf{v} = \langle 4, 5, 6 \rangle$.

**Solution:**

**Step 1:** Write the determinant:
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
1 & 2 & 3 \\
4 & 5 & 6
\end{vmatrix}
$$

**Step 2:** Compute the $\mathbf{i}$ component (cover the $\mathbf{i}$ column):
$$
\mathbf{i}(2 \cdot 6 - 3 \cdot 5) = \mathbf{i}(12 - 15) = -3\mathbf{i}
$$

**Step 3:** Compute the $\mathbf{j}$ component (cover the $\mathbf{j}$ column, remember the minus sign):
$$
-\mathbf{j}(1 \cdot 6 - 3 \cdot 4) = -\mathbf{j}(6 - 12) = -\mathbf{j}(-6) = 6\mathbf{j}
$$

**Step 4:** Compute the $\mathbf{k}$ component (cover the $\mathbf{k}$ column):
$$
\mathbf{k}(1 \cdot 5 - 2 \cdot 4) = \mathbf{k}(5 - 8) = -3\mathbf{k}
$$

**Step 5:** Combine:
$$
\mathbf{u} \times \mathbf{v} = \langle -3, 6, -3 \rangle
$$

**Step 6:** Verify perpendicularity:
- $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = (-3)(1) + 6(2) + (-3)(3) = -3 + 12 - 9 = 0$
- $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{v} = (-3)(4) + 6(5) + (-3)(6) = -12 + 30 - 18 = 0$

### Example 2: Cross Product of Perpendicular Vectors

**Problem:** Find $\mathbf{u} \times \mathbf{v}$ for $\mathbf{u} = \langle 1, 0, 0 \rangle$ and $\mathbf{v} = \langle 0, 1, 0 \rangle$.

**Solution:**

**Step 1:** Write the determinant:
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
1 & 0 & 0 \\
0 & 1 & 0
\end{vmatrix}
$$

**Step 2:** Compute components:
- $\mathbf{i}$: $(0)(0) - (0)(1) = 0$
- $\mathbf{j}$: $-[(1)(0) - (0)(0)] = 0$
- $\mathbf{k}$: $(1)(1) - (0)(0) = 1$

**Step 3:** Result:
$$
\mathbf{u} \times \mathbf{v} = \langle 0, 0, 1 \rangle = \mathbf{k}
$$

**Step 4:** This follows the right-hand rule: index finger along $\mathbf{i}$ ($x$-axis), middle finger along $\mathbf{j}$ ($y$-axis), thumb points along $\mathbf{k}$ ($z$-axis).

**Step 5:** Magnitude: $\|\mathbf{u} \times \mathbf{v}\| = 1$, which equals $\|\mathbf{u}\|\|\mathbf{v}\|\sin(90^\circ) = 1 \cdot 1 \cdot 1 = 1$, the area of the unit square.

### Example 3: Computing Area of a Triangle

**Problem:** Find the area of the triangle with vertices $P(1, 0, 0)$, $Q(0, 2, 0)$, and $R(0, 0, 3)$.

**Solution:**

**Step 1:** Form two vectors from the triangle edges:
$$
\mathbf{u} = \overrightarrow{PQ} = \langle 0-1, 2-0, 0-0 \rangle = \langle -1, 2, 0 \rangle
$$
$$
\mathbf{v} = \overrightarrow{PR} = \langle 0-1, 0-0, 3-0 \rangle = \langle -1, 0, 3 \rangle
$$

**Step 2:** Compute the cross product:
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
-1 & 2 & 0 \\
-1 & 0 & 3
\end{vmatrix}
$$

- $\mathbf{i}$: $(2)(3) - (0)(0) = 6$
- $\mathbf{j}$: $-((-1)(3) - (0)(-1)) = -(-3 - 0) = 3$
- $\mathbf{k}$: $((-1)(0) - (2)(-1)) = 0 + 2 = 2$

So $\mathbf{u} \times \mathbf{v} = \langle 6, 3, 2 \rangle$.

**Step 3:** The area of the parallelogram is:
$$
\|\mathbf{u} \times \mathbf{v}\| = \sqrt{6^2 + 3^2 + 2^2} = \sqrt{36 + 9 + 4} = \sqrt{49} = 7
$$

**Step 4:** The area of the triangle is half the parallelogram area:
$$
\text{Area} = \frac{7}{2} = 3.5
$$

### Example 4: Finding a Unit Normal

**Problem:** Find a unit vector perpendicular to both $\mathbf{u} = \langle 2, 1, -1 \rangle$ and $\mathbf{v} = \langle 1, -2, 3 \rangle$.

**Solution:**

**Step 1:** Compute the cross product:
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
2 & 1 & -1 \\
1 & -2 & 3
\end{vmatrix}
$$

- $\mathbf{i}$: $(1)(3) - (-1)(-2) = 3 - 2 = 1$
- $\mathbf{j}$: $-[(2)(3) - (-1)(1)] = -(6 + 1) = -7$
- $\mathbf{k}$: $(2)(-2) - (1)(1) = -4 - 1 = -5$

So $\mathbf{u} \times \mathbf{v} = \langle 1, -7, -5 \rangle$.

**Step 2:** Compute the magnitude:
$$
\|\mathbf{u} \times \mathbf{v}\| = \sqrt{1^2 + (-7)^2 + (-5)^2} = \sqrt{1 + 49 + 25} = \sqrt{75} = 5\sqrt{3}
$$

**Step 3:** The unit normal is:
$$
\mathbf{n} = \frac{\mathbf{u} \times \mathbf{v}}{\|\mathbf{u} \times \mathbf{v}\|} = \left\langle \frac{1}{5\sqrt{3}}, \frac{-7}{5\sqrt{3}}, \frac{-5}{5\sqrt{3}} \right\rangle
$$

### Example 5: Checking for Parallel Vectors

**Problem:** Are $\mathbf{u} = \langle 2, -4, 6 \rangle$ and $\mathbf{v} = \langle -1, 2, -3 \rangle$ parallel?

**Solution:**

**Step 1:** Compute the cross product:
$$
\mathbf{u} \times \mathbf{v} = \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
2 & -4 & 6 \\
-1 & 2 & -3
\end{vmatrix}
$$

- $\mathbf{i}$: $(-4)(-3) - (6)(2) = 12 - 12 = 0$
- $\mathbf{j}$: $-[(2)(-3) - (6)(-1)] = -(-6 + 6) = 0$
- $\mathbf{k}$: $(2)(2) - (-4)(-1) = 4 - 4 = 0$

**Step 2:** $\mathbf{u} \times \mathbf{v} = \langle 0, 0, 0 \rangle$.

**Step 3:** Since the cross product is the zero vector, $\mathbf{u}$ and $\mathbf{v}$ are parallel. Indeed, $\mathbf{u} = -2\mathbf{v}$.

## Visual Interpretation

The cross product $\mathbf{u} \times \mathbf{v}$ corresponds to the oriented area of the parallelogram spanned by $\mathbf{u}$ and $\mathbf{v}$:

- The magnitude $\|\mathbf{u} \times \mathbf{v}\|$ is the area of the parallelogram.
- The direction is perpendicular to the plane containing $\mathbf{u}$ and $\mathbf{v}$.
- The sign (orientation) follows the right-hand rule: curl your fingers from $\mathbf{u}$ to $\mathbf{v}$; your thumb points in the direction of $\mathbf{u} \times \mathbf{v}$.

If you imagine a flat surface like a table, two vectors lying on the table have a cross product pointing straight up (or down, depending on order). The length of that upward vector tells you how big the parallelogram is.

## Common Mistakes

1. **Forgetting the negative sign on the $\mathbf{j}$ component:** The determinant expansion is $\mathbf{i}(u_2 v_3 - u_3 v_2) - \mathbf{j}(u_1 v_3 - u_3 v_1) + \mathbf{k}(u_1 v_2 - u_2 v_1)$. Many students forget the minus sign on $\mathbf{j}$.

2. **Thinking the cross product is commutative:** $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$. Swapping the order flips the direction.

3. **Applying cross product in 2D:** The cross product is defined only in $\mathbb{R}^3$. In 2D, the "cross product" usually refers to the scalar $u_1 v_2 - u_2 v_1$, which is the magnitude of the 3D cross product's $\mathbf{k}$ component.

4. **Confusing cross product with dot product:** The cross product produces a vector; the dot product produces a scalar. They are fundamentally different.

5. **Assuming associativity:** $(\mathbf{u} \times \mathbf{v}) \times \mathbf{w} \neq \mathbf{u} \times (\mathbf{v} \times \mathbf{w})$ in general. The cross product is not associative.

6. **Ignoring the right-hand rule direction:** The direction of $\mathbf{u} \times \mathbf{v}$ depends on the order of the operands. Always verify the orientation.

7. **Forgetting that parallel vectors produce zero:** If two vectors are parallel ($\theta = 0$ or $\pi$), their cross product is the zero vector, not a non-zero perpendicular vector.

## Interview Questions

### Beginner

1. What is the cross product of two vectors? How is it different from the dot product?
2. Compute $\langle 1, 0, 0 \rangle \times \langle 0, 1, 0 \rangle$.
3. What does the magnitude of the cross product represent geometrically?
4. Is the cross product commutative? Explain.
5. What is the cross product of a vector with itself?

### Intermediate

1. Derive the component formula for the cross product using the determinant of a $3 \times 3$ matrix.
2. Prove that $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = 0$.
3. Find a unit vector perpendicular to both $\mathbf{u} = \langle 1, 1, 1 \rangle$ and $\mathbf{v} = \langle 0, 1, -1 \rangle$.
4. The area of a triangle with vertices $A$, $B$, $C$ is $\frac{1}{2}\|\overrightarrow{AB} \times \overrightarrow{AC}\|$. Prove this.
5. How is the cross product used to compute surface normals in 3D graphics?

### Advanced

1. Prove the BAC-CAB identity: $\mathbf{u} \times (\mathbf{v} \times \mathbf{w}) = \mathbf{v}(\mathbf{u} \cdot \mathbf{w}) - \mathbf{w}(\mathbf{u} \cdot \mathbf{v})$.
2. Show that $(\mathbf{u} \times \mathbf{v}) \times \mathbf{w} = (\mathbf{u} \cdot \mathbf{w})\mathbf{v} - (\mathbf{v} \cdot \mathbf{w})\mathbf{u}$.
3. In geometric algebra, the cross product is generalised by the wedge product. Explain the connection between the cross product in $\mathbb{R}^3$ and the wedge product.

## Practice Problems

### Easy - 5 Questions

1. Compute $\langle 2, 1, 0 \rangle \times \langle 0, 3, 0 \rangle$.
2. Compute $\langle 1, -1, 2 \rangle \times \langle 0, 0, 1 \rangle$.
3. Find the magnitude of $\langle 3, 4, 0 \rangle \times \langle 0, 0, 1 \rangle$.
4. If $\mathbf{u} \times \mathbf{v} = \mathbf{0}$, what can you say about $\mathbf{u}$ and $\mathbf{v}$?
5. Simplify $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u}$.

### Medium - 5 Questions

6. Find a vector perpendicular to both $\mathbf{u} = \langle 1, -2, 3 \rangle$ and $\mathbf{v} = \langle 4, 0, -1 \rangle$.
7. Compute the area of the parallelogram spanned by $\mathbf{u} = \langle 2, 1, -1 \rangle$ and $\mathbf{v} = \langle 3, 2, 1 \rangle$.
8. Find the area of the triangle with vertices $A(1, 2, 3)$, $B(3, 2, 1)$, $C(1, 4, 1)$.
9. Show that $\| \mathbf{u} \times \mathbf{v} \|^2 = \|\mathbf{u}\|^2 \|\mathbf{v}\|^2 - (\mathbf{u} \cdot \mathbf{v})^2$.
10. For $\mathbf{u} = \langle 1, 2, 3 \rangle$ and $\mathbf{v} = \langle 2, -1, 0 \rangle$, verify that $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = 0$.

### Hard - 3 Questions

11. Prove Lagrange's identity: $\|\mathbf{u} \times \mathbf{v}\|^2 = \|\mathbf{u}\|^2\|\mathbf{v}\|^2 - (\mathbf{u} \cdot \mathbf{v})^2$.
12. For three vectors $\mathbf{a}, \mathbf{b}, \mathbf{c} \in \mathbb{R}^3$, prove the Jacobi identity: $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) + \mathbf{b} \times (\mathbf{c} \times \mathbf{a}) + \mathbf{c} \times (\mathbf{a} \times \mathbf{b}) = \mathbf{0}$.
13. Let $\mathbf{u}, \mathbf{v}, \mathbf{w}$ be vectors in $\mathbb{R}^3$. Prove that $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{w} = \mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})$ (the scalar triple product is cyclically invariant).

## Solutions

### Easy Solutions

1. $\langle 2, 1, 0 \rangle \times \langle 0, 3, 0 \rangle$: $\mathbf{i}(1\cdot0 - 0\cdot3) - \mathbf{j}(2\cdot0 - 0\cdot0) + \mathbf{k}(2\cdot3 - 1\cdot0) = \langle 0, 0, 6 \rangle$.
2. $\langle 1, -1, 2 \rangle \times \langle 0, 0, 1 \rangle$: $\mathbf{i}(-1\cdot1 - 2\cdot0) - \mathbf{j}(1\cdot1 - 2\cdot0) + \mathbf{k}(1\cdot0 - (-1)\cdot0) = \langle -1, -1, 0 \rangle$.
3. $\langle 3, 4, 0 \rangle \times \langle 0, 0, 1 \rangle = \langle 4\cdot1 - 0\cdot0, -(3\cdot1 - 0\cdot0), 3\cdot0 - 4\cdot0 \rangle = \langle 4, -3, 0 \rangle$. Magnitude $= \sqrt{16 + 9 + 0} = 5$.
4. If $\mathbf{u} \times \mathbf{v} = \mathbf{0}$, then $\mathbf{u}$ and $\mathbf{v}$ are parallel (or one of them is the zero vector).
5. $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = 0$ because the cross product is perpendicular to both inputs.

### Medium Solutions

6. Compute cross product: $\langle 1, -2, 3 \rangle \times \langle 4, 0, -1 \rangle$:
   - $\mathbf{i}$: $(-2)(-1) - (3)(0) = 2$
   - $\mathbf{j}$: $-[(1)(-1) - (3)(4)] = -(-1 - 12) = 13$
   - $\mathbf{k}$: $(1)(0) - (-2)(4) = 8$
   Result: $\langle 2, 13, 8 \rangle$ (any scalar multiple is also perpendicular).

7. $\mathbf{u} \times \mathbf{v} = \langle 2, 1, -1 \rangle \times \langle 3, 2, 1 \rangle$:
   - $\mathbf{i}$: $(1)(1) - (-1)(2) = 1 + 2 = 3$
   - $\mathbf{j}$: $-[(2)(1) - (-1)(3)] = -(2 + 3) = -5$
   - $\mathbf{k}$: $(2)(2) - (1)(3) = 4 - 3 = 1$
   $\| \langle 3, -5, 1 \rangle \| = \sqrt{9 + 25 + 1} = \sqrt{35}$.

8. $\overrightarrow{AB} = \langle 2, 0, -2 \rangle$, $\overrightarrow{AC} = \langle 0, 2, -2 \rangle$.
   $\overrightarrow{AB} \times \overrightarrow{AC} = \langle 0(-2) - (-2)(2), -(2(-2) - (-2)(0)), 2(2) - 0(0) \rangle = \langle 4, 4, 4 \rangle$.
   Area $= \frac{1}{2}\| \langle 4, 4, 4 \rangle \| = \frac{1}{2}\sqrt{48} = \frac{1}{2} \cdot 4\sqrt{3} = 2\sqrt{3}$.

9. $\|\mathbf{u} \times \mathbf{v}\|^2 = (\|\mathbf{u}\|\|\mathbf{v}\|\sin\theta)^2 = \|\mathbf{u}\|^2\|\mathbf{v}\|^2\sin^2\theta = \|\mathbf{u}\|^2\|\mathbf{v}\|^2(1 - \cos^2\theta) = \|\mathbf{u}\|^2\|\mathbf{v}\|^2 - \|\mathbf{u}\|^2\|\mathbf{v}\|^2\cos^2\theta = \|\mathbf{u}\|^2\|\mathbf{v}\|^2 - (\mathbf{u} \cdot \mathbf{v})^2$.

10. $\mathbf{u} \times \mathbf{v} = \langle 1, 2, 3 \rangle \times \langle 2, -1, 0 \rangle$:
    - $\mathbf{i}$: $(2)(0) - (3)(-1) = 0 + 3 = 3$
    - $\mathbf{j}$: $-[(1)(0) - (3)(2)] = -(0 - 6) = 6$
    - $\mathbf{k}$: $(1)(-1) - (2)(2) = -1 - 4 = -5$
    $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = \langle 3, 6, -5 \rangle \cdot \langle 1, 2, 3 \rangle = 3 + 12 - 15 = 0$. Verified.

### Hard Solutions

11. From the geometric definition: $\|\mathbf{u} \times \mathbf{v}\|^2 = \|\mathbf{u}\|^2\|\mathbf{v}\|^2\sin^2\theta$.
    And $\sin^2\theta = 1 - \cos^2\theta$.
    So $\|\mathbf{u} \times \mathbf{v}\|^2 = \|\mathbf{u}\|^2\|\mathbf{v}\|^2(1 - \cos^2\theta) = \|\mathbf{u}\|^2\|\mathbf{v}\|^2 - (\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta)^2 = \|\mathbf{u}\|^2\|\mathbf{v}\|^2 - (\mathbf{u} \cdot \mathbf{v})^2$.

12. Using the BAC-CAB identity twice:
    $\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = \mathbf{b}(\mathbf{a} \cdot \mathbf{c}) - \mathbf{c}(\mathbf{a} \cdot \mathbf{b})$.
    $\mathbf{b} \times (\mathbf{c} \times \mathbf{a}) = \mathbf{c}(\mathbf{b} \cdot \mathbf{a}) - \mathbf{a}(\mathbf{b} \cdot \mathbf{c})$.
    $\mathbf{c} \times (\mathbf{a} \times \mathbf{b}) = \mathbf{a}(\mathbf{c} \cdot \mathbf{b}) - \mathbf{b}(\mathbf{c} \cdot \mathbf{a})$.
    Summing all three: all terms cancel pairwise, giving $\mathbf{0}$.

13. Let $\mathbf{u} = \langle u_1, u_2, u_3 \rangle$, $\mathbf{v} = \langle v_1, v_2, v_3 \rangle$, $\mathbf{w} = \langle w_1, w_2, w_3 \rangle$.
    $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{w} = \langle u_2 v_3 - u_3 v_2, u_3 v_1 - u_1 v_3, u_1 v_2 - u_2 v_1 \rangle \cdot \langle w_1, w_2, w_3 \rangle$
    $= (u_2 v_3 - u_3 v_2)w_1 + (u_3 v_1 - u_1 v_3)w_2 + (u_1 v_2 - u_2 v_1)w_3$
    $= u_2 v_3 w_1 - u_3 v_2 w_1 + u_3 v_1 w_2 - u_1 v_3 w_2 + u_1 v_2 w_3 - u_2 v_1 w_3$.
    Meanwhile, $\mathbf{u} \cdot (\mathbf{v} \times \mathbf{w}) = \mathbf{u} \cdot \langle v_2 w_3 - v_3 w_2, v_3 w_1 - v_1 w_3, v_1 w_2 - v_2 w_1 \rangle$
    $= u_1(v_2 w_3 - v_3 w_2) + u_2(v_3 w_1 - v_1 w_3) + u_3(v_1 w_2 - v_2 w_1)$
    $= u_1 v_2 w_3 - u_1 v_3 w_2 + u_2 v_3 w_1 - u_2 v_1 w_3 + u_3 v_1 w_2 - u_3 v_2 w_1$.
    Rearranging terms shows the two expressions are equal.

## Related Concepts

- **Dot Product** (MATH-016): The scalar product $\mathbf{u} \cdot \mathbf{v}$
- **Vector Projection** (MATH-018): Decomposing vectors into parallel/perpendicular components
- **Angle Between Vectors** (MATH-019): Related to the cross product magnitude via $\sin\theta$
- **Distance Between Vectors** (MATH-020): Uses vector subtraction
- **Determinants**: The cross product formula uses a $3 \times 3$ determinant

## Next Concepts

- **022 Matrix Multiplication**: Cross products can be expressed as matrix-vector products using skew-symmetric matrices
- **027 Surface Normals and Lighting**: Applied cross products in computer graphics
- **032 Scalar Triple Product**: Volume calculations using cross and dot products

## Summary

The cross product $\mathbf{u} \times \mathbf{v}$ is a vector operation in $\mathbb{R}^3$ that produces a new vector perpendicular to both $\mathbf{u}$ and $\mathbf{v}$. Its magnitude equals the area of the parallelogram spanned by the two vectors. The direction follows the right-hand rule. The cross product is anti-commutative, non-associative, and zero when the vectors are parallel. It is essential for computing normals, areas, torques, and orientations in 3D space.

## Key Takeaways

- $\mathbf{u} \times \mathbf{v} = \langle u_2 v_3 - u_3 v_2,\; u_3 v_1 - u_1 v_3,\; u_1 v_2 - u_2 v_1 \rangle$
- $\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta$ (area of parallelogram)
- The result is perpendicular to both inputs: $(\mathbf{u} \times \mathbf{v}) \cdot \mathbf{u} = (\mathbf{u} \times \mathbf{v}) \cdot \mathbf{v} = 0$
- $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$ (anti-commutative)
- Zero cross product $\iff$ vectors are parallel
- Critical for surface normals in 3D graphics, computer vision, and geometric deep learning
