# Concept: Vector Magnitude

## Concept ID

MATH-014

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Compute the magnitude (Euclidean norm) of a vector in any dimension.
- Understand the squared norm and its relationship to the magnitude.
- Apply the properties of magnitude: non-negativity, definiteness, scaling, triangle inequality.
- Use magnitude in AI/ML contexts such as weight decay, loss magnitude, and gradient clipping.
- Distinguish between the $L^2$ norm and other norms like the $L^1$ norm.

## Prerequisites

- **Vector (MATH-002):** Understanding what a vector is and how to write its components.
- **Scalar (MATH-001):** Arithmetic operations including squaring and square roots.
- **Scalar Multiplication (MATH-013):** How scaling affects vector components.
- **Vector Addition (MATH-011):** Adding vectors component-wise.

## Definition

The **magnitude** (also called the **norm**, **length**, or **Euclidean norm**) of a vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$ is a non-negative scalar that measures the "size" or "length" of the vector. It is denoted by $\|\mathbf{v}\|$ and defined as:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

This is also called the $L^2$ norm (read as "el-two norm").

The **squared norm** (or squared magnitude) is:

$$\|\mathbf{v}\|^2 = v_1^2 + v_2^2 + \cdots + v_n^2 = \sum_{i=1}^n v_i^2$$

Both the norm and squared norm are scalars.

## Intuition

The magnitude of a vector answers the question: **How long is this arrow?**

In 2D, the magnitude $\sqrt{x^2 + y^2}$ is exactly the length of the arrow from the origin to the point $(x, y)$. This is just the Pythagorean theorem: the hypotenuse of a right triangle with sides $x$ and $y$.

In 3D, $\sqrt{x^2 + y^2 + z^2}$ is the length of the diagonal of a rectangular box with sides $x$, $y$, and $z$.

In higher dimensions (4D, 100D, 1000D), we cannot visualise the arrow, but the formula extends naturally: square every component, sum them, and take the square root. The result is still a meaningful measure of the vector's "size".

Think of magnitude as the **straight-line distance** from the origin to the point represented by the vector. It is the most fundamental way to measure how "big" a vector is.

## Why This Concept Matters

Vector magnitude is essential for measuring distances, sizes, and errors in multi-dimensional spaces. Without it, we could not quantify how large a gradient is, how far apart two data points are, or how much a prediction differs from its target.

In AI and machine learning:
- **Loss functions** are scalar magnitudes (or squared magnitudes) of error vectors.
- **Weight decay (L2 regularisation)** penalises the squared magnitude of weight vectors.
- **Gradient clipping** limits the magnitude of gradients to a threshold.
- **Normalisation** divides vectors by their magnitude to produce unit vectors.
- **Distance metrics** (Euclidean distance) between data points are magnitudes of difference vectors.
- **Feature importance** can be assessed by the magnitude of weight vectors.

The concept of magnitude bridges vectors (which are multi-dimensional) to scalars (which are single numbers), allowing us to compare, rank, and optimise vector quantities using scalar techniques.

## Historical Background

The concept of vector magnitude originates from the Pythagorean theorem (c. 500 BCE), which relates the sides of a right triangle: $a^2 + b^2 = c^2$. The length of the hypotenuse $\sqrt{a^2 + b^2}$ is the 2D magnitude.

René Descartes (1596–1650) extended this to the coordinate plane: the distance from the origin to a point $(x, y)$ is $\sqrt{x^2 + y^2}$. This gave a coordinate-based (algebraic) method for computing length.

In the 19th century, Augustin-Louis Cauchy and Hermann Schwarz generalised the concept to $n$ dimensions, leading to the Euclidean norm $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^n v_i^2}$. The notation $\|\cdot\|$ for norms was introduced by Erhard Schmidt in the early 20th century.

The generalisation of norms beyond the Euclidean norm (e.g., $L^1$, $L^\infty$, $L^p$ norms) was developed by Frigyes Riesz and others in the early 1900s as part of functional analysis.

## Real World Examples

**Example 1: Distance from origin.** A point is at coordinates $(3, 4)$ on a map. Its straight-line distance from the origin (e.g., base camp) is $\sqrt{3^2 + 4^2} = \sqrt{25} = 5$ km.

**Example 2: Speed from velocity.** A car's velocity vector is $\mathbf{v} = (40, 30)$ km/h (east and north components). The car's speed (the magnitude of velocity) is $\|\mathbf{v}\| = \sqrt{40^2 + 30^2} = \sqrt{2500} = 50$ km/h.

**Example 3: Force magnitude.** Three forces act on a point: $\mathbf{F} = (2, -1, 3)$ N. The net magnitude of the force is $\|\mathbf{F}\| = \sqrt{4 + 1 + 9} = \sqrt{14} \approx 3.74$ N.

**Example 4: Signal strength.** A signal vector in 4G communications is $\mathbf{s} = (0.3, -0.1, 0.5, 0.2)$. The signal magnitude (power) is $\|\mathbf{s}\| = \sqrt{0.09 + 0.01 + 0.25 + 0.04} = \sqrt{0.39} \approx 0.624$.

## AI/ML Relevance

Vector magnitude is used throughout machine learning for measuring, constraining, and optimising:

1. **Loss Magnitude.** The Mean Squared Error (MSE) loss for a batch of predictions is:
   $$L = \frac{1}{n} \sum_{i=1}^n \| \mathbf{y}_i - \hat{\mathbf{y}}_i \|^2$$
   Each term $\|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2$ is the squared magnitude of the error vector for one sample. The loss is the average of these squared magnitudes across all samples.

2. **Weight Decay (L2 Regularisation).** To prevent overfitting, we add a penalty proportional to the squared magnitude of the weight vector:
   $$L_{\text{reg}} = L_{\text{data}} + \frac{\lambda}{2} \|\mathbf{w}\|^2$$
   The term $\|\mathbf{w}\|^2$ penalises large weights. During backpropagation, the gradient of this term is $\lambda\mathbf{w}$, which pulls weights toward zero. This is why it is called "weight decay".

3. **Gradient Clipping.** In training deep networks, gradients can explode (become very large). Gradient clipping rescales the gradient if its norm exceeds a threshold $T$:
   $$\nabla L_{\text{clipped}} = \begin{cases} \nabla L & \text{if } \|\nabla L\| \leq T \\ \frac{T}{\|\nabla L\|} \nabla L & \text{if } \|\nabla L\| > T \end{cases}$$
   The norm $\|\nabla L\|$ is computed, compared with $T$, and used to scale the gradient.

4. **Normalisation Layers.** Batch normalisation and layer normalisation both involve computing the norm of activation vectors to standardise their distribution, improving training stability.

5. **K-Nearest Neighbours (KNN).** The Euclidean distance between data points $\mathbf{x}_i$ and $\mathbf{x}_j$ is $\|\mathbf{x}_i - \mathbf{x}_j\|$, which is the magnitude of their difference vector. KNN classifies points based on these distances.

6. **Cosine Similarity.** The cosine similarity between two vectors is:
   $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
   This normalises by the magnitudes, measuring only directional alignment. It is widely used in NLP and recommendation systems.

## Mathematical Explanation

### Euclidean Norm (L2 Norm)

For a vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$$

This is the "ordinary" distance measure — the straight-line distance from the origin to the point.

### Squared Norm

$$\|\mathbf{v}\|^2 = \sum_{i=1}^n v_i^2$$

The squared norm is often used in loss functions because it avoids the square root (making derivatives simpler) and is mathematically equivalent for optimisation purposes (minimising $\|\mathbf{v}\|$ is equivalent to minimising $\|\mathbf{v}\|^2$ since the square root is monotonic for non-negative values).

### L1 Norm (Manhattan Norm)

For reference, the $L^1$ norm (sum of absolute values) is different from the $L^2$ norm:

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

It measures distance if you can only move along axes (like city blocks). In ML, it is used in Lasso regression (L1 regularisation).

### Relationship to Dot Product

The squared norm equals the dot product of the vector with itself:

$$\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v} = \sum_{i=1}^n v_i^2$$

This relationship is important because it connects magnitude to other vector operations.

### Norm in Different Dimensions

- **1D:** $\|v\| = \sqrt{v^2} = |v|$ — the absolute value.
- **2D:** $\|(x, y)\| = \sqrt{x^2 + y^2}$ — the Pythagorean theorem.
- **3D:** $\|(x, y, z)\| = \sqrt{x^2 + y^2 + z^2}$.
- **nD:** $\|\mathbf{v}\| = \sqrt{v_1^2 + \cdots + v_n^2}$ — the general formula.

## Formula(s)

**Euclidean norm (magnitude):**

$$\|\mathbf{v}\| = \sqrt{\sum_{i=1}^n v_i^2}$$

**Squared norm:**

$$\|\mathbf{v}\|^2 = \sum_{i=1}^n v_i^2 = \mathbf{v} \cdot \mathbf{v}$$

**Norm of scaled vector:**

$$\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$$

**Distance between two vectors (Euclidean distance):**

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\| = \sqrt{\sum_{i=1}^n (u_i - v_i)^2}$$

## Properties

Let $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$ be vectors and $c \in \mathbb{R}$ be a scalar.

| Property | Formula |
|---|---|
| **Non-negativity** | $\|\mathbf{v}\| \geq 0$ |
| **Positive definiteness** | $\|\mathbf{v}\| = 0$ iff $\mathbf{v} = \mathbf{0}$ |
| **Absolute homogeneity (scaling)** | $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$ |
| **Triangle inequality** | $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ |
| **Reverse triangle inequality** | $\big|\|\mathbf{u}\| - \|\mathbf{v}\|\big| \leq \|\mathbf{u} - \mathbf{v}\|$ |
| **Cauchy-Schwarz inequality** | $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$ |
| **Relation to dot product** | $\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v}$ |

The **triangle inequality** is a geometric fact: the straight-line distance between two points is never longer than going via a third point. This is why it is called the triangle inequality — any side of a triangle is at most the sum of the other two sides.

## Step-by-Step Worked Examples

### Example 1: Magnitude in 2D and 3D

Compute the magnitude of $\mathbf{a} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$ and $\mathbf{b} = \begin{pmatrix} 2 \\ -3 \\ 6 \end{pmatrix}$.

**Step 1 — Magnitude of $\mathbf{a}$:**
$$\|\mathbf{a}\| = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10$$

**Step 2 — Magnitude of $\mathbf{b}$:**
$$\|\mathbf{b}\| = \sqrt{2^2 + (-3)^2 + 6^2} = \sqrt{4 + 9 + 36} = \sqrt{49} = 7$$

**Answer:** $\|\mathbf{a}\| = 10$, $\|\mathbf{b}\| = 7$.

### Example 2: Squared Norm and Its Use in Loss

A model predicts $\hat{\mathbf{y}} = \begin{pmatrix} 2.5 \\ 3.0 \\ 1.0 \end{pmatrix}$ and the true values are $\mathbf{y} = \begin{pmatrix} 3.0 \\ 2.5 \\ 1.5 \end{pmatrix}$. Compute the error vector, its squared magnitude, and the MSE loss (for $n=3$).

**Step 1 — Error vector:**
$$\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \begin{pmatrix} 3.0 - 2.5 \\ 2.5 - 3.0 \\ 1.5 - 1.0 \end{pmatrix} = \begin{pmatrix} 0.5 \\ -0.5 \\ 0.5 \end{pmatrix}$$

**Step 2 — Squared magnitude of error:**
$$\|\mathbf{e}\|^2 = (0.5)^2 + (-0.5)^2 + (0.5)^2 = 0.25 + 0.25 + 0.25 = 0.75$$

**Step 3 — MSE loss:**
$$L = \frac{1}{3} \|\mathbf{e}\|^2 = \frac{0.75}{3} = 0.25$$

**Answer:** The error vector is $\begin{pmatrix}0.5 & -0.5 & 0.5\end{pmatrix}^T$, its squared norm is $0.75$, and the MSE is $0.25$.

### Example 3: Scaling Property Verification

Let $\mathbf{v} = \begin{pmatrix} 3 \\ -4 \end{pmatrix}$. Compute $\|\mathbf{v}\|$, then compute $\|-2\mathbf{v}\|$ and verify $\|-2\mathbf{v}\| = |-2| \cdot \|\mathbf{v}\|$.

**Step 1 — Original magnitude:**
$$\|\mathbf{v}\| = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**Step 2 — Scaled vector:**
$$-2\mathbf{v} = \begin{pmatrix} -6 \\ 8 \end{pmatrix}$$

**Step 3 — Magnitude of scaled vector:**
$$\|-2\mathbf{v}\| = \sqrt{(-6)^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10$$

**Step 4 — Verify scaling property:**
$$|-2| \cdot \|\mathbf{v}\| = 2 \times 5 = 10 = \|-2\mathbf{v}\|$$

**Answer:** $\|\mathbf{v}\| = 5$, $\|-2\mathbf{v}\| = 10 = 2 \times 5$. ✓

### Example 4: Gradient Clipping

A gradient vector is $\nabla L = \begin{pmatrix} 3 \\ -4 \\ 12 \end{pmatrix}$. The clipping threshold is $T = 5$. Compute the clipped gradient.

**Step 1 — Compute gradient norm:**
$$\|\nabla L\| = \sqrt{3^2 + (-4)^2 + 12^2} = \sqrt{9 + 16 + 144} = \sqrt{169} = 13$$

**Step 2 — Compare with threshold:**
$13 > 5$, so clipping is needed.

**Step 3 — Compute scaling factor:**
$$\frac{T}{\|\nabla L\|} = \frac{5}{13} \approx 0.3846$$

**Step 4 — Apply scaling:**
$$\nabla L_{\text{clipped}} = \frac{5}{13} \times \begin{pmatrix} 3 \\ -4 \\ 12 \end{pmatrix} = \begin{pmatrix} 15/13 \\ -20/13 \\ 60/13 \end{pmatrix} \approx \begin{pmatrix} 1.154 \\ -1.538 \\ 4.615 \end{pmatrix}$$

**Step 5 — Verify clipped norm:**
$$\|\nabla L_{\text{clipped}}\| = \frac{5}{13} \cdot 13 = 5 = T$$

**Answer:** The clipped gradient is approximately $\begin{pmatrix}1.154 & -1.538 & 4.615\end{pmatrix}^T$, and its norm is exactly $5$.

### Example 5: L2 Regularisation Penalty

A model has weights $\mathbf{w} = \begin{pmatrix} 2.0 \\ -1.5 \\ 0.5 \\ -3.0 \end{pmatrix}$. The regularisation strength is $\lambda = 0.1$. Compute the L2 penalty: $L_{\text{reg}} = \frac{\lambda}{2} \|\mathbf{w}\|^2$.

**Step 1 — Compute squared norm:**
$$\|\mathbf{w}\|^2 = 2.0^2 + (-1.5)^2 + 0.5^2 + (-3.0)^2 = 4 + 2.25 + 0.25 + 9 = 15.5$$

**Step 2 — Compute the penalty:**
$$L_{\text{reg}} = \frac{0.1}{2} \times 15.5 = 0.05 \times 15.5 = 0.775$$

**Step 3 — Gradient of the penalty:**
$$\frac{\partial L_{\text{reg}}}{\partial \mathbf{w}} = \lambda \mathbf{w} = 0.1 \times \begin{pmatrix} 2.0 \\ -1.5 \\ 0.5 \\ -3.0 \end{pmatrix} = \begin{pmatrix} 0.2 \\ -0.15 \\ 0.05 \\ -0.30 \end{pmatrix}$$

**Answer:** The L2 regularisation penalty is $0.775$, and its gradient (used for weight decay during backpropagation) is $\begin{pmatrix}0.2 & -0.15 & 0.05 & -0.30\end{pmatrix}^T$.

## Visual Interpretation

In 2D, the magnitude $\sqrt{x^2 + y^2}$ is the length of the hypotenuse of a right triangle:

```
y
▲
│
4 ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                   ╱│
3 ─ ─ ─ ─ ─ ─ ─ ─ ╱  │
│               ╱     │
2 ─ ─ ─ ─ ─ ─ ╱       │
│           ╱  ‖v‖    │
1 ─ ─ ─ ─ ╱           │ y = 3
│       ╱             │
0 ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ▶ x
│     1   2   3   4   5
│     x = 4
```

The vector $\mathbf{v} = (4, 3)$ has magnitude $\sqrt{4^2 + 3^2} = \sqrt{25} = 5$.

In 3D, imagine a rectangular box with side lengths $x$, $y$, $z$. The magnitude is the length of the diagonal from one corner to the opposite corner.

For dimensions $n > 3$, we cannot draw the picture, but the formula gives us a consistent way to measure the "size" of a vector in any number of dimensions.

## Common Mistakes

1. **Forgetting the square root.** The magnitude is $\sqrt{v_1^2 + \cdots + v_n^2}$, not $v_1^2 + \cdots + v_n^2$. The sum of squares is the **squared norm**, not the norm itself. A common error is to say "the magnitude of $(3,4)$ is 25" when it is actually 5.

2. **Applying the norm component-wise.** The norm is a single scalar computed from all components together. It is not a vector of square roots of absolute values: $\|\mathbf{v}\| \neq (|v_1|, |v_2|, \ldots)$.

3. **Confusing $L^1$ and $L^2$ norms.** The $L^2$ norm (Euclidean) uses squares and a square root: $\sqrt{\sum v_i^2}$. The $L^1$ norm uses absolute values: $\sum |v_i|$. They give different results. For $(3, -4)$, $L^2 = 5$ while $L^1 = 7$.

4. **Thinking the norm must be an integer.** Many vectors have magnitudes that are irrational numbers. For example, $\|(1,1)\| = \sqrt{2}$ — this is perfectly valid.

5. **Believing $\|\mathbf{u} + \mathbf{v}\| = \|\mathbf{u}\| + \|\mathbf{v}\|$.** This is false in general. The triangle inequality says $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$, with equality only when $\mathbf{u}$ and $\mathbf{v}$ point in the same direction.

6. **Treating magnitude as signed.** Magnitude is always non-negative. There is no such thing as a negative length. If a computation gives a negative magnitude, something is wrong.

7. **Forgetting to square negative components properly.** $(-3)^2 = 9$, not $-9$. Every component is squared, so negative components contribute positively to the norm.

8. **Confusing $\|\mathbf{v}\|^2$ (squared norm) with $\|\mathbf{v}^2\|$ (which is undefined).** You cannot square the vector and then take the norm. Square the components individually, then sum them.

## Interview Questions

### Beginner

1. **How do you compute the magnitude of a 2D vector $\mathbf{v} = (3, 4)$?**
   *Answer: $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.*

2. **What is the difference between $\|\mathbf{v}\|$ and $\|\mathbf{v}\|^2$?**
   *Answer: $\|\mathbf{v}\|$ is the length (norm) of the vector. $\|\mathbf{v}\|^2$ is the squared length — the sum of squares without the square root. For optimisation, minimising $\|\mathbf{v}\|^2$ is equivalent to minimising $\|\mathbf{v}\|$, but $\|\mathbf{v}\|^2$ is computationally simpler (no square root).*

3. **Can the magnitude of a vector be negative?**
   *Answer: No. The magnitude is always non-negative. It is defined as $\sqrt{\sum v_i^2}$, and a square root is always $\geq 0$.*

4. **What does it mean if $\|\mathbf{v}\| = 0$?**
   *Answer: It means $\mathbf{v}$ is the zero vector $\mathbf{0}$ (all components are zero). Only the zero vector has magnitude 0.*

5. **What is the magnitude of $(-6, 8)$?**
   *Answer: $\|(-6, 8)\| = \sqrt{36 + 64} = \sqrt{100} = 10$. The negative sign in the first component disappears when squared.*

### Intermediate

1. **Prove that $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$.**
   *Answer: $\|c\mathbf{v}\| = \sqrt{(c v_1)^2 + \cdots + (c v_n)^2} = \sqrt{c^2(v_1^2 + \cdots + v_n^2)} = |c|\sqrt{v_1^2 + \cdots + v_n^2} = |c| \cdot \|\mathbf{v}\|$. The square root of $c^2$ is $|c|$, not $c$, because the norm is always non-negative.*

2. **Explain the triangle inequality using a real-world analogy.**
   *Answer: The triangle inequality $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ says the direct path from start to finish is never longer than going via an intermediate point. Imagine walking 3 km east (vector $\mathbf{u}$) then 4 km north (vector $\mathbf{v}$). The direct distance back to the start is 5 km (the magnitude of the sum), but walking back via the original path would be $3 + 4 = 7$ km. The shortcut (direct path) is always shorter or equal.*

3. **Why is the squared norm $\|\mathbf{v}\|^2$ used more often than the norm itself in loss functions?**
   *Answer: (1) It avoids the square root, reducing computational cost. (2) Its derivative is simpler: $\frac{\partial}{\partial \mathbf{v}} \|\mathbf{v}\|^2 = 2\mathbf{v}$, whereas $\frac{\partial}{\partial \mathbf{v}} \|\mathbf{v}\| = \frac{\mathbf{v}}{\|\mathbf{v}\|}$ has a division by the norm, which can cause numerical issues when the norm is near zero. (3) For optimisation, minimising $\|\mathbf{v}\|^2$ gives the same solution as minimising $\|\mathbf{v}\|$ because the square root is a monotonic function on $[0, \infty)$.*

4. **How does the magnitude of the gradient relate to the convergence of gradient descent?**
   *Answer: The magnitude $\|\nabla L\|$ measures how steep the loss landscape is at the current point. A large gradient norm indicates a steep slope (fast potential improvement), and a small norm indicates a flat region (potentially near a minimum). In practice, if the gradient norm is very large, training can be unstable (exploding gradients); if it is very small, training slows down (vanishing gradients). Monitoring $\|\nabla L\|$ is a common diagnostic in training.*

5. **What is Euclidean distance and how does it relate to vector magnitude?**
   *Answer: The Euclidean distance between two points $\mathbf{u}$ and $\mathbf{v}$ is $\|\mathbf{u} - \mathbf{v}\|$, which is the magnitude of their difference vector. It measures the straight-line distance between them in $n$-dimensional space. This is the most common distance metric in ML, used in KNN, K-means clustering, and many other algorithms.*

### Advanced

1. **Prove the Cauchy-Schwarz inequality: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$.**
   *Answer: Consider the vector $\mathbf{u} - t\mathbf{v}$ for any scalar $t$. Its squared norm is non-negative:*
   $$0 \leq \|\mathbf{u} - t\mathbf{v}\|^2 = (\mathbf{u} - t\mathbf{v}) \cdot (\mathbf{u} - t\mathbf{v}) = \|\mathbf{u}\|^2 - 2t(\mathbf{u} \cdot \mathbf{v}) + t^2\|\mathbf{v}\|^2$$
   
   *This quadratic in $t$ is always $\geq 0$, so its discriminant must be $\leq 0$:*
   $$(-2(\mathbf{u} \cdot \mathbf{v}))^2 - 4\|\mathbf{v}\|^2 \|\mathbf{u}\|^2 \leq 0$$
   $$4(\mathbf{u} \cdot \mathbf{v})^2 - 4\|\mathbf{u}\|^2 \|\mathbf{v}\|^2 \leq 0$$
   $$(\mathbf{u} \cdot \mathbf{v})^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$$
   
   *Taking square roots: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$. Equality holds when $\mathbf{u}$ and $\mathbf{v}$ are parallel (linearly dependent).*

2. **Explain how the concept of vector magnitude is used in batch normalisation.**
   *Answer: Batch normalisation normalises the activations of a layer across a batch. For each feature dimension, it computes the mean $\mu$ and standard deviation $\sigma$ across the batch, then normalises: $\hat{x}_i = (x_i - \mu)/\sigma$. The standard deviation $\sigma$ is essentially the (scaled) magnitude of the centred activation vector: $\sigma = \sqrt{\frac{1}{m}\sum_{i=1}^m (x_i - \mu)^2}$. After normalisation, the activations have zero mean and unit variance (magnitude information is removed). The layer then learns a scale parameter $\gamma$ and shift parameter $\beta$ to restore expressive power. The magnitude of activations is explicitly controlled to stabilise training.*

3. **Why does the L2 norm produce a spherical geometry while the L1 norm produces a diamond-shaped geometry, and how does this affect regularisation in ML?**
   *Answer: The L2 norm $\|\mathbf{w}\|_2 = \sqrt{\sum w_i^2}$ defines circles/spheres as level sets (all points with equal norm form a sphere). The L1 norm $\|\mathbf{w}\|_1 = \sum |w_i|$ defines diamonds/polyhedra (points with equal L1 norm form a diamond shape in 2D). In regularisation, the shape matters: L1 regularisation (Lasso) has corners on the coordinate axes, which means the optimal solution (intersection of the loss contour with the constraint region) often occurs at a corner, where some weights are exactly zero. This induces sparsity. L2 regularisation (Ridge) has a smooth spherical constraint, so weights are shrunk but rarely go to exactly zero. This geometric difference is why L1 is used for feature selection and L2 for preventing overfitting without eliminating features.*

## Practice Problems

### Easy - 5 Questions

1. Compute $\|\mathbf{v}\|$ for $\mathbf{v} = (5, 12)^T$.

2. Compute $\|\mathbf{a}\|$ for $\mathbf{a} = (1, -2, 2)^T$.

3. Find $\|\mathbf{v}\|^2$ for $\mathbf{v} = (3, -4)^T$.

4. Let $\mathbf{u} = (8, -15)^T$. Find $\|\mathbf{u}\|$.

5. Compute the distance between $\mathbf{p} = (1, 2)^T$ and $\mathbf{q} = (4, 6)^T$.

### Medium - 5 Questions

1. Let $\mathbf{v} = (2, -1, 3)^T$. Compute $\|3\mathbf{v}\|$ and verify $\|3\mathbf{v}\| = 3\|\mathbf{v}\|$.

2. The error vector for 4 predictions is $\mathbf{e} = (0.2, -0.3, 0.1, -0.4)^T$. Compute $\|\mathbf{e}\|^2$.

3. A gradient vector is $\nabla L = (1.5, -2.0, 3.0)^T$. If the clipping threshold is $T = 2.0$, compute the clipped gradient.

4. Verify the Cauchy-Schwarz inequality for $\mathbf{u} = (1, 2, 3)^T$ and $\mathbf{v} = (4, -1, 0)^T$ by computing both sides.

5. A weight vector is $\mathbf{w} = (1.0, -0.5, 2.0)^T$. The L2 regularisation penalty is $\frac{\lambda}{2}\|\mathbf{w}\|^2$ with $\lambda = 0.2$. Compute the penalty. Then compute the gradient of the penalty $\lambda\mathbf{w}$.

### Hard - 3 Questions

1. Prove the reverse triangle inequality: $\big|\|\mathbf{u}\| - \|\mathbf{v}\|\big| \leq \|\mathbf{u} - \mathbf{v}\|$.

2. The MSE loss for a batch is $L = \frac{1}{n}\sum_{i=1}^n \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2$. For $n=2$, $\mathbf{y}_1 = (1, 2)^T$, $\hat{\mathbf{y}}_1 = (1.5, 1.5)^T$, $\mathbf{y}_2 = (3, 1)^T$, $\hat{\mathbf{y}}_2 = (2.5, 1.5)^T$, compute $L$. Then compute the gradient $\nabla_{\hat{\mathbf{y}}_1} L$ (the partial derivative of $L$ with respect to $\hat{\mathbf{y}}_1$). (Hint: $\nabla_{\hat{\mathbf{y}}_i} \|\mathbf{y}_i - \hat{\mathbf{y}}_i\|^2 = -2(\mathbf{y}_i - \hat{\mathbf{y}}_i)$.)

3. Show that $\|\mathbf{u} + \mathbf{v}\|^2 + \|\mathbf{u} - \mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$ (the parallelogram law). Verify with $\mathbf{u} = (1, 2)^T$, $\mathbf{v} = (3, -1)^T$.

## Solutions

### Easy Solutions

**1.** $\|\mathbf{v}\| = \sqrt{5^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13$.

**2.** $\|\mathbf{a}\| = \sqrt{1^2 + (-2)^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$.

**3.** $\|\mathbf{v}\|^2 = 3^2 + (-4)^2 = 9 + 16 = 25$.

**4.** $\|\mathbf{u}\| = \sqrt{8^2 + (-15)^2} = \sqrt{64 + 225} = \sqrt{289} = 17$.

**5.** $\mathbf{p} - \mathbf{q} = (1-4, 2-6)^T = (-3, -4)^T$. Distance $= \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = 5$.

### Medium Solutions

**1.** $\|\mathbf{v}\| = \sqrt{2^2 + (-1)^2 + 3^2} = \sqrt{4 + 1 + 9} = \sqrt{14} \approx 3.742$.
$3\mathbf{v} = (6, -3, 9)^T$.
$\|3\mathbf{v}\| = \sqrt{6^2 + (-3)^2 + 9^2} = \sqrt{36 + 9 + 81} = \sqrt{126} = 3\sqrt{14} \approx 11.225$.
$3\|\mathbf{v}\| = 3 \times \sqrt{14} \approx 11.225 = \|3\mathbf{v}\|$. ✓

**2.** $\|\mathbf{e}\|^2 = (0.2)^2 + (-0.3)^2 + (0.1)^2 + (-0.4)^2 = 0.04 + 0.09 + 0.01 + 0.16 = 0.30$.

**3.** $\|\nabla L\| = \sqrt{1.5^2 + (-2.0)^2 + 3.0^2} = \sqrt{2.25 + 4.0 + 9.0} = \sqrt{15.25} \approx 3.905$.
$3.905 > 2.0$, so clipping needed.
Scaling factor: $2.0 / 3.905 \approx 0.512$.
$\nabla L_{\text{clipped}} = 0.512 \times (1.5, -2.0, 3.0)^T = (0.768, -1.024, 1.536)^T$.

**4.** Left side: $|\mathbf{u} \cdot \mathbf{v}| = |1 \times 4 + 2 \times (-1) + 3 \times 0| = |4 - 2 + 0| = 2$.
Right side: $\|\mathbf{u}\| \cdot \|\mathbf{v}\| = \sqrt{1 + 4 + 9} \times \sqrt{16 + 1 + 0} = \sqrt{14} \times \sqrt{17} \approx 3.742 \times 4.123 \approx 15.427$.
$2 \leq 15.427$, so $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$ holds. ✓

**5.** $\|\mathbf{w}\|^2 = 1.0^2 + (-0.5)^2 + 2.0^2 = 1 + 0.25 + 4 = 5.25$.
Penalty: $\frac{0.2}{2} \times 5.25 = 0.1 \times 5.25 = 0.525$.
Gradient of penalty: $\lambda\mathbf{w} = 0.2 \times (1.0, -0.5, 2.0)^T = (0.2, -0.1, 0.4)^T$.

### Hard Solutions

**1.** Using the triangle inequality:
$$\|\mathbf{u}\| = \|(\mathbf{u} - \mathbf{v}) + \mathbf{v}\| \leq \|\mathbf{u} - \mathbf{v}\| + \|\mathbf{v}\|$$
Therefore $\|\mathbf{u}\| - \|\mathbf{v}\| \leq \|\mathbf{u} - \mathbf{v}\|$.

Similarly:
$$\|\mathbf{v}\| = \|(\mathbf{v} - \mathbf{u}) + \mathbf{u}\| \leq \|\mathbf{v} - \mathbf{u}\| + \|\mathbf{u}\| = \|\mathbf{u} - \mathbf{v}\| + \|\mathbf{u}\|$$
Therefore $\|\mathbf{v}\| - \|\mathbf{u}\| \leq \|\mathbf{u} - \mathbf{v}\|$.

Combining: $-\|\mathbf{u} - \mathbf{v}\| \leq \|\mathbf{u}\| - \|\mathbf{v}\| \leq \|\mathbf{u} - \mathbf{v}\|$, which means:
$$\big|\|\mathbf{u}\| - \|\mathbf{v}\|\big| \leq \|\mathbf{u} - \mathbf{v}\|$$

This shows that the difference of the magnitudes is bounded by the magnitude of the difference — the reverse triangle inequality. ✓

**2.** Error vectors:
$\mathbf{e}_1 = \mathbf{y}_1 - \hat{\mathbf{y}}_1 = (1-1.5, 2-1.5)^T = (-0.5, 0.5)^T$.
$\|\mathbf{e}_1\|^2 = (-0.5)^2 + 0.5^2 = 0.25 + 0.25 = 0.50$.

$\mathbf{e}_2 = \mathbf{y}_2 - \hat{\mathbf{y}}_2 = (3-2.5, 1-1.5)^T = (0.5, -0.5)^T$.
$\|\mathbf{e}_2\|^2 = 0.5^2 + (-0.5)^2 = 0.25 + 0.25 = 0.50$.

$L = \frac{1}{2}(0.50 + 0.50) = \frac{1}{2} \times 1.0 = 0.5$.

Gradient with respect to $\hat{\mathbf{y}}_1$:
$$\nabla_{\hat{\mathbf{y}}_1} L = \frac{1}{2} \cdot (-2(\mathbf{y}_1 - \hat{\mathbf{y}}_1)) = -(\mathbf{y}_1 - \hat{\mathbf{y}}_1) = -(-0.5, 0.5)^T = (0.5, -0.5)^T$$

**3.** The parallelogram law:
$$\|\mathbf{u} + \mathbf{v}\|^2 = (\mathbf{u}+\mathbf{v})\cdot(\mathbf{u}+\mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 + 2\mathbf{u}\cdot\mathbf{v}$$
$$\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u}-\mathbf{v})\cdot(\mathbf{u}-\mathbf{v}) = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u}\cdot\mathbf{v}$$

Adding:
$$\|\mathbf{u} + \mathbf{v}\|^2 + \|\mathbf{u} - \mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2 + (2\mathbf{u}\cdot\mathbf{v} - 2\mathbf{u}\cdot\mathbf{v}) = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$$

Verification with $\mathbf{u} = (1,2)^T$, $\mathbf{v} = (3,-1)^T$:
$\mathbf{u} + \mathbf{v} = (4, 1)^T$, $\|\mathbf{u}+\mathbf{v}\|^2 = 16+1 = 17$.
$\mathbf{u} - \mathbf{v} = (-2, 3)^T$, $\|\mathbf{u}-\mathbf{v}\|^2 = 4+9 = 13$.
Sum: $17 + 13 = 30$.
Right side: $2\|\mathbf{u}\|^2 = 2(1+4) = 10$, $2\|\mathbf{v}\|^2 = 2(9+1) = 20$. Sum: $10 + 20 = 30$. ✓

The parallelogram law says the sum of squared lengths of the diagonals of a parallelogram equals the sum of squared lengths of all four sides.

## Related Concepts

- **Vector (MATH-002):** The object whose magnitude we measure.
- **Scalar (MATH-001):** The magnitude result is a scalar.
- **Unit Vector (MATH-015):)** A vector divided by its magnitude: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$.
- **Scalar Multiplication (MATH-013):** Magnitude scales with absolute value of scalar: $\|c\mathbf{v}\| = |c|\|\mathbf{v}\|$.
- **Vector Subtraction (MATH-012):)** Used to compute the difference vector whose magnitude is the distance.
- **Dot Product (MATH-016):** $\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v}$.

## Next Concepts

- **Distance Metrics (MATH-040):** Generalisations beyond Euclidean distance ($L^1$, $L^\infty$, Mahalanobis, cosine distance).
- **Normalisation (MATH-041):** Scaling vectors to have unit norm or zero mean/unit variance.
- **Orthogonality (MATH-042):)** Vectors are orthogonal if their dot product is zero, which relates to magnitude via Pythagoras: $\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$.
- **Spectral Norm (MATH-070):** The norm of a matrix, generalising the vector norm.

## Summary

The magnitude (Euclidean norm) of a vector $\mathbf{v}$ is a non-negative scalar $\|\mathbf{v}\| = \sqrt{\sum v_i^2}$ that measures the vector's length. The squared norm $\|\mathbf{v}\|^2 = \sum v_i^2$ is also commonly used. Magnitude satisfies key properties: non-negativity (only zero for the zero vector), absolute homogeneity ($\|c\mathbf{v}\| = |c|\|\mathbf{v}\|$), and the triangle inequality ($\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$). In AI/ML, magnitude is central to loss functions (MSE uses $\|\mathbf{y} - \hat{\mathbf{y}}\|^2$), weight decay (L2 regularisation penalises $\|\mathbf{w}\|^2$), gradient clipping (limits $\|\nabla L\|$), and distance-based algorithms.

## Key Takeaways

- The **magnitude** of $\mathbf{v}$ is $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^n v_i^2}$ — the Euclidean length.
- The **squared norm** $\|\mathbf{v}\|^2 = \sum v_i^2$ avoids the square root and is used in loss functions.
- Magnitude is always **non-negative**, and only the zero vector has magnitude 0.
- Scalar multiplication scales the magnitude: $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$.
- The **triangle inequality**: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.
- The **Cauchy-Schwarz inequality**: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$.
- In AI/ML, magnitude is used in loss functions, weight decay, gradient clipping, distance metrics, and normalisation.
- The **squared norm** links to the dot product: $\|\mathbf{v}\|^2 = \mathbf{v} \cdot \mathbf{v}$.
