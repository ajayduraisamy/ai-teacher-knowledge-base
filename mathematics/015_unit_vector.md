# Concept: Unit Vector

## Concept ID

MATH-015

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Define a unit vector as a vector with magnitude exactly 1.
- Compute the unit vector in the direction of any given vector using the normalisation formula.
- Identify the standard basis vectors as the fundamental unit vectors along each axis.
- Explain the role of unit vectors in representing pure direction.
- Apply unit vectors in AI/ML contexts such as weight normalisation, gradient direction, and attention mechanisms.

## Prerequisites

- **Vector (MATH-002):** Understanding that a vector has both magnitude and direction.
- **Vector Magnitude (MATH-014):** Computing $\|\mathbf{v}\|$ using the Euclidean norm.
- **Scalar Multiplication (MATH-013):)** Scaling a vector by a scalar.
- **Scalar (MATH-001):** Basic arithmetic with real numbers.

## Definition

A **unit vector** is a vector whose magnitude (length) is exactly 1. Unit vectors are used to represent **pure direction** — they tell you which way something is pointing without any information about how far.

If $\mathbf{v}$ is any non-zero vector, the unit vector in the direction of $\mathbf{v}$ is:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

This operation is called **normalisation**. The hat notation $\hat{\mathbf{v}}$ (read as "v-hat") is the standard symbol for a unit vector.

The **standard basis vectors** are the unit vectors that point along each coordinate axis. In $\mathbb{R}^3$:

$$\mathbf{i} = \hat{\mathbf{x}} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad
\mathbf{j} = \hat{\mathbf{y}} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad
\mathbf{k} = \hat{\mathbf{z}} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

## Intuition

Think of a unit vector as a **compass direction**. If someone says "head north-east", they have given you a direction but not a distance. A unit vector is the mathematical version of "north-east" — it is the pure direction without any magnitude information.

The normalisation process is like taking any arrow and adjusting its length to exactly 1 while keeping it pointing in the same direction. If you have an arrow that is 5 units long and pointing north-east, the unit vector is a 1-unit-long arrow still pointing north-east.

Another analogy: a unit vector is like a **direction on a map** without specifying how far to travel. The standard basis vectors are the cardinal directions: east ($\mathbf{i}$), north ($\mathbf{j}$), and up ($\mathbf{k}$).

## Why This Concept Matters

Unit vectors isolate direction from magnitude. This separation is crucial whenever we care about **which way** something is pointing rather than **how far** it goes.

In AI and machine learning:
- **Weight normalisation** separates a weight vector into its direction (unit vector) and its magnitude (scalar), allowing independent optimisation.
- **Gradient direction** is often more important than gradient magnitude — we want to know which way to move, and the learning rate separately determines how far.
- **Attention mechanisms** compute similarity between query and key vectors using dot products, which inherently use directional information.
- **Cosine similarity** normalises vectors to unit length before comparing, measuring only directional alignment.
- **Standard basis vectors** represent individual features or classes in one-hot encoding.

Without unit vectors, we could not separate "which way" from "how far", which is essential for many optimisation and representation techniques.

## Historical Background

The concept of a unit vector emerged naturally from the study of direction in physics and geometry. In the 17th century, René Descartes' coordinate system implicitly used unit vectors as the building blocks for specifying positions: every point was a combination of steps along the $x$ and $y$ axes.

The notation $\mathbf{i}$, $\mathbf{j}$, $\mathbf{k}$ for standard basis vectors in 3D was introduced by Sir William Rowan Hamilton in his work on quaternions (1843). These symbols became standard in physics and engineering through the vector analysis of Josiah Willard Gibbs and Oliver Heaviside in the late 19th century.

The hat notation $\hat{\mathbf{v}}$ for a normalised vector became common in the 20th century as vector notation was standardised. The concept of normalisation — dividing a vector by its magnitude — is fundamental to the idea of a "direction" vector in any normed vector space.

## Real World Examples

**Example 1: Compass directions.** A compass gives directions as unit vectors: north is $\hat{\mathbf{n}} = (0, 1)$, east is $\hat{\mathbf{e}} = (1, 0)$, north-east is $\hat{\mathbf{ne}} = (\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2})$. Each has length 1.

**Example 2: Surface normals in graphics.** In 3D computer graphics, every surface has a **normal vector** — a unit vector perpendicular to the surface. Lighting calculations use the normal to determine how light reflects. A flat surface facing upward has normal $\hat{\mathbf{n}} = (0, 0, 1)$.

**Example 3: Unit vector of velocity.** A car moves with velocity $\mathbf{v} = (40, 30)$ km/h. Its direction of travel is the unit vector $\hat{\mathbf{v}} = (0.8, 0.6)$ — the car is moving in a direction that is 0.8 of the way east and 0.6 of the way north per unit of speed.

**Example 4: Unit vector of force.** A force of 50 N is applied in the direction $\mathbf{d} = (3, 4)$. The force vector is $\mathbf{F} = 50 \cdot \hat{\mathbf{d}} = 50 \cdot (0.6, 0.8) = (30, 40)$ N. The unit vector $(0.6, 0.8)$ gives the direction, and the scalar 50 gives the magnitude.

## AI/ML Relevance

Unit vectors play a critical role in several ML techniques:

1. **Weight Normalisation.** This technique reparameterises each weight vector $\mathbf{w}$ as:
   $$\mathbf{w} = g \cdot \hat{\mathbf{v}}$$
   where $g = \|\mathbf{w}\|$ is a scalar magnitude and $\hat{\mathbf{v}} = \mathbf{w}/\|\mathbf{w}\|$ is a unit vector direction. During training, both $g$ and $\hat{\mathbf{v}}$ are learned separately. This decouples the learning rate for the magnitude and direction, often leading to faster convergence.

2. **Gradient Direction.** In gradient descent, the direction of steepest descent is the negative gradient vector $-\nabla L$, but its magnitude depends on local curvature. The **normalised gradient** $\hat{\mathbf{g}} = \nabla L / \|\nabla L\|$ gives pure direction. When combined with techniques like gradient clipping or adaptive learning rates, the direction is used with a separately controlled step size.

3. **Cosine Similarity.** The cosine similarity between two vectors normalises them to unit length:
   $$\text{cosine\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$
   This measures only the angle between the vectors (pure directional similarity), ignoring magnitude. It is widely used in NLP (sentence embeddings, word similarity) and recommendation systems.

4. **Attention Mechanisms.** In transformer attention, the attention score between query $\mathbf{q}$ and key $\mathbf{k}$ can be written as:
   $$\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}$$
   When $\mathbf{q}$ and $\mathbf{k}$ are normalised (as in some architectures), this reduces to $\hat{\mathbf{q}} \cdot \hat{\mathbf{k}}$, a pure directional comparison.

5. **One-Hot Encoding.** In classification, the target class is often represented as a one-hot vector: a unit basis vector pointing in the "class" direction. For a 3-class problem, class 2 is $\mathbf{e}_2 = (0, 1, 0)^T$, which is a standard basis vector.

6. **Spectral Normalisation.** In GANs (Generative Adversarial Networks), the weight matrices of the discriminator are normalised to have spectral norm (maximum singular value) equal to 1. This constrains the Lipschitz constant of the network and stabilises training. The normalised weight matrices operate on input vectors without changing their magnitude scaling.

## Mathematical Explanation

### Normalisation Formula

For any non-zero vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)^T$, the unit vector in the same direction is:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{1}{\sqrt{\sum_{i=1}^n v_i^2}} \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**Verification:** The magnitude of $\hat{\mathbf{v}}$ is 1:
$$\|\hat{\mathbf{v}}\| = \left\| \frac{\mathbf{v}}{\|\mathbf{v}\|} \right\| = \frac{\|\mathbf{v}\|}{\|\mathbf{v}\|} = 1$$

### Standard Basis Vectors

In $\mathbb{R}^n$, the standard basis vectors are:
$$\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad
\mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \quad \ldots, \quad
\mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix}$$

Each $\mathbf{e}_i$ has a single 1 at position $i$ and zeros elsewhere. They are unit vectors because $\|\mathbf{e}_i\| = \sqrt{0^2 + \cdots + 1^2 + \cdots + 0^2} = 1$.

In physics/engineering notation for 3D:
- $\mathbf{i} = (1, 0, 0)^T$ — points along the $x$-axis (east)
- $\mathbf{j} = (0, 1, 0)^T$ — points along the $y$-axis (north)
- $\mathbf{k} = (0, 0, 1)^T$ — points along the $z$-axis (up)

### Decomposition into Magnitude and Direction

Any vector can be expressed as the product of its magnitude and its direction:

$$\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$$

This decomposition separates the two pieces of information contained in a vector: how much ($\|\mathbf{v}\|$) and which way ($\hat{\mathbf{v}}$).

## Formula(s)

**Unit vector (normalisation):**

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{1}{\|\mathbf{v}\|} \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**Magnitude of unit vector:**

$$\|\hat{\mathbf{v}}\| = 1$$

**Decomposition into magnitude and direction:**

$$\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$$

**Standard basis vectors (in $\mathbb{R}^n$):**

$$\mathbf{e}_i = \begin{pmatrix} 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{pmatrix} \quad \text{(1 at position $i$, zeros elsewhere)}$$

**Cosine similarity (dot product of unit vectors):**

$$\cos(\theta) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

## Properties

Let $\mathbf{v}$ be a non-zero vector and $\hat{\mathbf{v}}$ its unit vector.

| Property | Formula |
|---|---|
| **Unit magnitude** | $\|\hat{\mathbf{v}}\| = 1$ |
| **Direction preserved** | $\hat{\mathbf{v}}$ points in same direction as $\mathbf{v}$ |
| **Scaling invariance** | $\widehat{c\mathbf{v}} = \hat{\mathbf{v}}$ for $c > 0$ |
| **Negative scaling** | $\widehat{-c\mathbf{v}} = -\hat{\mathbf{v}}$ for $c > 0$ |
| **Not defined for zero** | $\hat{\mathbf{0}}$ is undefined |
| **Basis representation** | $\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$ |
| **Dot product with itself** | $\hat{\mathbf{v}} \cdot \hat{\mathbf{v}} = 1$ |

The **scaling invariance** is important: multiplying a vector by any positive scalar does not change its unit vector. The direction of $(2, 2)$ is the same as $(1, 1)$ — both have unit vector $\left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right)$.

The **negative scaling** property means reversing the vector flips the direction of the unit vector.

## Step-by-Step Worked Examples

### Example 1: Normalising a 2D Vector

Let $\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$. Find $\hat{\mathbf{v}}$.

**Step 1 — Compute magnitude:**
$$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**Step 2 — Divide each component by the magnitude:**
$$\hat{\mathbf{v}} = \frac{1}{5} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$$

**Step 3 — Verify unit magnitude:**
$$\|\hat{\mathbf{v}}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$$

**Answer:** $\hat{\mathbf{v}} = \begin{pmatrix}0.6 & 0.8\end{pmatrix}^T$.

### Example 2: Normalising a 3D Vector with Negative Components

Let $\mathbf{v} = \begin{pmatrix} 2 \\ -3 \\ 6 \end{pmatrix}$. Find $\hat{\mathbf{v}}$.

**Step 1 — Compute magnitude:**
$$\|\mathbf{v}\| = \sqrt{2^2 + (-3)^2 + 6^2} = \sqrt{4 + 9 + 36} = \sqrt{49} = 7$$

**Step 2 — Divide each component by the magnitude:**
$$\hat{\mathbf{v}} = \frac{1}{7} \begin{pmatrix} 2 \\ -3 \\ 6 \end{pmatrix} = \begin{pmatrix} 2/7 \\ -3/7 \\ 6/7 \end{pmatrix} \approx \begin{pmatrix} 0.286 \\ -0.429 \\ 0.857 \end{pmatrix}$$

**Step 3 — Verify unit magnitude:**
$$\|\hat{\mathbf{v}}\| = \sqrt{(2/7)^2 + (-3/7)^2 + (6/7)^2} = \sqrt{\frac{4 + 9 + 36}{49}} = \sqrt{\frac{49}{49}} = \sqrt{1} = 1$$

**Answer:** $\hat{\mathbf{v}} \approx \begin{pmatrix}0.286 & -0.429 & 0.857\end{pmatrix}^T$.

### Example 3: Scaling Invariance of Unit Vectors

Show that $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ and $\mathbf{v} = 3\mathbf{u} = \begin{pmatrix} 3 \\ 6 \end{pmatrix}$ have the same unit vector.

**Step 1 — Unit vector of $\mathbf{u}$:**
$$\|\mathbf{u}\| = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$$
$$\hat{\mathbf{u}} = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 \\ 2 \end{pmatrix} \approx \begin{pmatrix} 0.447 \\ 0.894 \end{pmatrix}$$

**Step 2 — Unit vector of $\mathbf{v} = 3\mathbf{u}$:**
$$\|\mathbf{v}\| = \sqrt{3^2 + 6^2} = \sqrt{9 + 36} = \sqrt{45} = 3\sqrt{5}$$
$$\hat{\mathbf{v}} = \frac{1}{3\sqrt{5}} \begin{pmatrix} 3 \\ 6 \end{pmatrix} = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \hat{\mathbf{u}}$$

**Step 3 — Both equal:**
$$\hat{\mathbf{u}} = \hat{\mathbf{v}} \approx \begin{pmatrix} 0.447 \\ 0.894 \end{pmatrix}$$

**Answer:** Scaling by a positive scalar does not change the unit vector. $\hat{\mathbf{u}} = \hat{\mathbf{v}}$.

### Example 4: Decomposing a Vector into Magnitude and Direction

Decompose $\mathbf{v} = \begin{pmatrix} 5 \\ -12 \end{pmatrix}$ into magnitude $\times$ unit vector.

**Step 1 — Compute magnitude:**
$$\|\mathbf{v}\| = \sqrt{5^2 + (-12)^2} = \sqrt{25 + 144} = \sqrt{169} = 13$$

**Step 2 — Compute unit vector:**
$$\hat{\mathbf{v}} = \frac{1}{13} \begin{pmatrix} 5 \\ -12 \end{pmatrix} = \begin{pmatrix} 5/13 \\ -12/13 \end{pmatrix} \approx \begin{pmatrix} 0.385 \\ -0.923 \end{pmatrix}$$

**Step 3 — Verify decomposition:**
$$\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}} = 13 \times \begin{pmatrix} 5/13 \\ -12/13 \end{pmatrix} = \begin{pmatrix} 13 \cdot 5/13 \\ 13 \cdot (-12/13) \end{pmatrix} = \begin{pmatrix} 5 \\ -12 \end{pmatrix}$$

**Answer:** $\mathbf{v} = 13 \cdot \hat{\mathbf{v}}$ where $\hat{\mathbf{v}} \approx \begin{pmatrix}0.385 & -0.923\end{pmatrix}^T$.

### Example 5: Cosine Similarity (Dot Product of Unit Vectors)

Two word embedding vectors are:
$$\mathbf{u} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix} 0.9 \\ 0.3 \end{pmatrix}$$

Compute the cosine similarity between them.

**Step 1 — Compute magnitudes:**
$$\|\mathbf{u}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$$
Note: $\mathbf{u}$ is already a unit vector.

$$\|\mathbf{v}\| = \sqrt{0.9^2 + 0.3^2} = \sqrt{0.81 + 0.09} = \sqrt{0.90} \approx 0.949$$

**Step 2 — Compute unit vector of $\mathbf{v}$:**
$$\hat{\mathbf{v}} = \frac{1}{0.949} \begin{pmatrix} 0.9 \\ 0.3 \end{pmatrix} \approx \begin{pmatrix} 0.949 \\ 0.316 \end{pmatrix}$$

**Step 3 — Cosine similarity is the dot product of the unit vectors:**
$$\cos(\theta) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = (0.6)(0.949) + (0.8)(0.316) = 0.569 + 0.253 = 0.822$$

**Alternative using direct formula:**
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{0.6 \times 0.9 + 0.8 \times 0.3}{1 \times 0.949} = \frac{0.54 + 0.24}{0.949} = \frac{0.78}{0.949} \approx 0.822$$

**Answer:** The cosine similarity is approximately $0.822$. Since this is close to 1, the vectors point in a similar direction (angle $\theta \approx \arccos(0.822) \approx 34.7^\circ$).

## Visual Interpretation

A unit vector in 2D is an arrow of length 1 starting from the origin:

```
y
▲
│
1 ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                    ╱
│                  ╱   Unit circle (all unit vectors)
│                ╱
│              ╱   ◀── v = (0.6, 0.8)
│            ╱       ‖v‖ = 1
│          ╱
│        ╱
│      ╱
│    ╱
│  ╱
0 ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶ x
│     1
```

The set of all unit vectors in $\mathbb{R}^2$ forms the **unit circle** — every point on the circle is exactly distance 1 from the origin. In $\mathbb{R}^3$, all unit vectors form the **unit sphere**. In higher dimensions, the set of all unit vectors forms the **unit hypersphere**.

The standard basis vectors $\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$ are the unit vectors pointing along each coordinate axis. Every other vector can be built as a combination of these basis vectors.

## Common Mistakes

1. **Forgetting to divide by the magnitude.** The most common mistake: taking $\mathbf{v}$ itself as the unit vector. A unit vector must have magnitude 1, which requires dividing by $\|\mathbf{v}\|$.

2. **Dividing only some components by the magnitude.** Normalisation divides **every** component by $\|\mathbf{v}\|$. For $\mathbf{v} = (3, 4)$, $\hat{\mathbf{v}} = (0.6, 0.8)$, not $(1, 4/3)$ or $(3/5, 4)$.

3. **Trying to normalise the zero vector.** The zero vector has magnitude 0, and division by 0 is undefined. You cannot compute a unit vector for $\mathbf{v} = \mathbf{0}$.

4. **Confusing unit vectors with one-hot vectors.** In $\mathbb{R}^3$, the unit vector in the $(1, 1, 1)$ direction is $(1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3}) \approx (0.577, 0.577, 0.577)$. The one-hot vector $(0, 0, 1)$ is also a unit vector (magnitude 1), but they are different vectors. One-hot vectors are specific standard basis vectors, not the normalisation of arbitrary vectors.

5. **Thinking $\hat{\mathbf{v}}$ always has integer components.** Unit vectors often have irrational components. For example, $\hat{\mathbf{v}}$ for $(1, 1)$ is $(\sqrt{2}/2, \sqrt{2}/2)$, not $(0.5, 0.5)$. The components of a unit vector are not necessarily equal; they sum (in squared sense) to 1.

6. **Believing that $\|\mathbf{v}\|$ and $\hat{\mathbf{v}}$ are independent.** While they can be decomposed, changing the direction of $\mathbf{v}$ changes $\hat{\mathbf{v}}$. They are separate pieces of information, but together they reconstruct $\mathbf{v}$ exactly.

7. **Confusing normalisation (making a unit vector) with standardisation (making zero mean, unit variance).** Normalisation divides by the Euclidean norm to get length 1. Standardisation subtracts the mean and divides by the standard deviation. They are different operations used in different contexts.

## Interview Questions

### Beginner

1. **What is a unit vector?**
   *Answer: A unit vector is a vector with magnitude exactly 1. It represents pure direction without any magnitude information.*

2. **How do you compute the unit vector in the direction of $\mathbf{v}$?**
   *Answer: Divide $\mathbf{v}$ by its magnitude: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$. This is called normalisation.*

3. **What are the standard basis vectors in $\mathbb{R}^3$?**
   *Answer: They are $\mathbf{i} = (1, 0, 0)^T$, $\mathbf{j} = (0, 1, 0)^T$, and $\mathbf{k} = (0, 0, 1)^T$. Each has magnitude 1 and points along one coordinate axis.*

4. **Can the zero vector be converted to a unit vector?**
   *Answer: No. The zero vector has magnitude 0, and division by 0 is undefined. Unit vectors can only be computed for non-zero vectors.*

5. **If $\hat{\mathbf{v}}$ is a unit vector, what is its magnitude?**
   *Answer: $\|\hat{\mathbf{v}}\| = 1$ by definition.*

### Intermediate

1. **Show that $\widehat{c\mathbf{v}} = \hat{\mathbf{v}}$ for any $c > 0$.**
   *Answer: $\widehat{c\mathbf{v}} = \frac{c\mathbf{v}}{\|c\mathbf{v}\|} = \frac{c\mathbf{v}}{|c|\|\mathbf{v}\|} = \frac{c\mathbf{v}}{c\|\mathbf{v}\|} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \hat{\mathbf{v}}$. The factor $c$ cancels out because it appears in both the numerator and denominator (as $|c|$). This shows that the direction of a vector is independent of its magnitude.*

2. **Explain how weight normalisation separates the learning dynamics of magnitude and direction.**
   *Answer: Weight normalisation reparameterises $\mathbf{w} = g\hat{\mathbf{v}}$ where $g = \|\mathbf{w}\|$ and $\hat{\mathbf{v}} = \mathbf{w}/\|\mathbf{w}\|$. During training, gradients flow to both $g$ and $\hat{\mathbf{v}}$ separately. The gradient for $g$ is $\partial L/\partial g = \hat{\mathbf{v}} \cdot \nabla_{\mathbf{w}} L$, and the gradient for $\hat{\mathbf{v}}$ is $\partial L/\partial \hat{\mathbf{v}} = g \nabla_{\mathbf{w}} L$ (with a projection to keep $\hat{\mathbf{v}}$ on the unit sphere). This decouples the learning rate effects: $g$ can be learned with a different effective rate than $\hat{\mathbf{v}}$, often leading to faster convergence.*

3. **How would you compute a random unit vector uniformly distributed on the unit sphere in $\mathbb{R}^3$?**
   *Answer: The naive approach of sampling each component uniformly from $[-1, 1]$ and normalising gives a non-uniform distribution (biased toward the corners). The correct method is to sample each component from a standard normal distribution $\mathcal{N}(0, 1)$ and then normalise: $\mathbf{v} = (z_1, z_2, z_3)^T$ where $z_i \sim \mathcal{N}(0, 1)$, then $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$. The resulting unit vectors are uniformly distributed on the sphere surface.*

4. **What is the relationship between unit vectors and cosine similarity?**
   *Answer: Cosine similarity is defined as $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$. When both vectors are normalised to unit length, this simplifies to $\cos(\theta) = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$. This is why cosine similarity is sometimes called the "normalised dot product" — it measures directional similarity independent of magnitude.*

5. **Why are standard basis vectors used in one-hot encoding?**
   *Answer: In one-hot encoding, each category is represented by a unique standard basis vector $\mathbf{e}_i$. For a $k$-class problem, class $j$ is $\mathbf{e}_j = (0, \ldots, 1, \ldots, 0)^T$ where the 1 is at position $j$. This creates a set of $k$ mutually orthogonal unit vectors, each representing a distinct category. The orthogonal property means each class is equally "distant" from every other class in Euclidean space.*

### Advanced

1. **Prove that the unit vector $\hat{\mathbf{v}}$ minimises $\|\mathbf{v} - a\hat{\mathbf{v}}\|$ over all scalars $a$ when $a = \|\mathbf{v}\|$. What does this tell us about the decomposition $\mathbf{v} = \|\mathbf{v}\|\hat{\mathbf{v}}$?**
   *Answer: We want to find $a$ that minimises $\|\mathbf{v} - a\hat{\mathbf{v}}\|^2$. Expand:*
   $$\|\mathbf{v} - a\hat{\mathbf{v}}\|^2 = \|\mathbf{v}\|^2 - 2a\mathbf{v} \cdot \hat{\mathbf{v}} + a^2\|\hat{\mathbf{v}}\|^2$$
   $$= \|\mathbf{v}\|^2 - 2a\|\mathbf{v}\|(\hat{\mathbf{v}} \cdot \hat{\mathbf{v}}) + a^2$$
   $$= \|\mathbf{v}\|^2 - 2a\|\mathbf{v}\| + a^2$$
   
   *Take derivative with respect to $a$ and set to zero: $-2\|\mathbf{v}\| + 2a = 0$, so $a = \|\mathbf{v}\|$. The unit vector $\hat{\mathbf{v}}$ is the direction that, when scaled by $\|\mathbf{v}\|$, best approximates $\mathbf{v}$ in the least-squares sense — in fact, it reconstructs $\mathbf{v}$ exactly. This confirms that the decomposition $\mathbf{v} = \|\mathbf{v}\|\hat{\mathbf{v}}$ is optimal.*

2. **In self-attention, queries and keys are often normalised to unit length before computing attention scores. Explain why this is beneficial and what trade-off it introduces.**
   *Answer: Normalising queries and keys to unit length means the dot product $\hat{\mathbf{q}} \cdot \hat{\mathbf{k}}$ gives the cosine similarity, which ranges from $[-1, 1]$. This bounds the attention scores, preventing extreme values that could cause vanishing gradients in the softmax. It also makes the attention mechanism focus purely on directional similarity, ignoring magnitude information. The trade-off is that magnitude information can be semantically meaningful (e.g., the norm of a word embedding can indicate word frequency or importance). Normalising removes this information, which may hurt performance on tasks where magnitude matters. Some architectures use a learned temperature scalar to restore some flexibility.*

3. **The Gram-Schmidt process produces orthonormal basis vectors from a set of linearly independent vectors. Explain how unit vectors are essential to this process, and briefly describe the algorithm for $\mathbb{R}^2$.**
   *Answer: The Gram-Schmidt process takes a set of independent vectors and produces an orthonormal basis — a set of mutually perpendicular unit vectors. In $\mathbb{R}^2$ with vectors $\mathbf{a}$ and $\mathbf{b}$:*
   1. *Set $\mathbf{u}_1 = \mathbf{a}$, then $\hat{\mathbf{e}}_1 = \mathbf{u}_1/\|\mathbf{u}_1\|$ (make the first basis vector a unit vector).*
   2. *Subtract the projection of $\mathbf{b}$ onto $\hat{\mathbf{e}}_1$: $\mathbf{u}_2 = \mathbf{b} - (\mathbf{b} \cdot \hat{\mathbf{e}}_1)\hat{\mathbf{e}}_1$.*
   3. *Normalise: $\hat{\mathbf{e}}_2 = \mathbf{u}_2/\|\mathbf{u}_2\|$ (make the second basis vector a unit vector).*
   
   *The normalisation step is crucial — without it, the basis vectors would not be unit vectors. The resulting $\{\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2\}$ are orthonormal: each has magnitude 1 and dot product 0.*

## Practice Problems

### Easy - 5 Questions

1. Find the unit vector in the direction of $\mathbf{v} = (3, 4)^T$.

2. Normalise $\mathbf{v} = (1, 2, 2)^T$ to a unit vector.

3. Is $\mathbf{u} = (0.6, 0.8)^T$ a unit vector? Verify.

4. What is the unit vector in the direction of $\mathbf{v} = (-5, 12)^T$?

5. Write the standard basis vectors for $\mathbb{R}^2$ and $\mathbb{R}^3$.

### Medium - 5 Questions

1. Let $\mathbf{v} = (2, -1, 2)^T$. Compute $\hat{\mathbf{v}}$ and verify $\|\hat{\mathbf{v}}\| = 1$.

2. Decompose $\mathbf{v} = (7, -24)^T$ into magnitude and direction: $\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$.

3. Show that $\mathbf{v} = (1, 1, 1)^T$ and $\mathbf{w} = (5, 5, 5)^T$ have the same unit vector. Compute $\hat{\mathbf{v}}$.

4. The dot product of two unit vectors is 0.5. What is the angle between them?
   (Hint: $\hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = \cos\theta$)

5. Word embeddings: $\mathbf{u} = (0.2, 0.8, 0.4)^T$ and $\mathbf{v} = (0.6, 0.5, 0.6)^T$. Compute their cosine similarity.

### Hard - 3 Questions

1. Prove that $\widehat{-\mathbf{v}} = -\hat{\mathbf{v}}$ for any non-zero vector $\mathbf{v}$.

2. A neural network layer normalises its input vectors to unit length before processing. An input $\mathbf{x} = (3, 4)^T$ is passed through. What is the normalised output? If the layer then scales by $\gamma = 2$ and shifts by $\beta = 1$, what is the final output? (The operation is $\mathbf{y} = \gamma \hat{\mathbf{x}} + \beta$ where $\beta$ is a scalar added to each component.)

3. In spectral normalisation for GANs, the weight matrix $\mathbf{W}$ of a layer is divided by its spectral norm $\sigma(\mathbf{W})$ (the largest singular value): $\mathbf{W}_{\text{SN}} = \mathbf{W} / \sigma(\mathbf{W})$. Explain why this is analogous to the vector normalisation $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$. What property does $\mathbf{W}_{\text{SN}}$ have that is analogous to $\|\hat{\mathbf{v}}\| = 1$?

## Solutions

### Easy Solutions

**1.** $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = 5$.
$\hat{\mathbf{v}} = \frac{1}{5}(3, 4)^T = (0.6, 0.8)^T$.

**2.** $\|\mathbf{v}\| = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$.
$\hat{\mathbf{v}} = \frac{1}{3}(1, 2, 2)^T = \left(\frac{1}{3}, \frac{2}{3}, \frac{2}{3}\right)^T \approx (0.333, 0.667, 0.667)^T$.

**3.** $\|\mathbf{u}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$. Yes, it is a unit vector.

**4.** $\|\mathbf{v}\| = \sqrt{(-5)^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13$.
$\hat{\mathbf{v}} = \frac{1}{13}(-5, 12)^T = \left(-\frac{5}{13}, \frac{12}{13}\right)^T \approx (-0.385, 0.923)^T$.

**5.** In $\mathbb{R}^2$: $\mathbf{e}_1 = (1, 0)^T$, $\mathbf{e}_2 = (0, 1)^T$.
In $\mathbb{R}^3$: $\mathbf{e}_1 = (1, 0, 0)^T$, $\mathbf{e}_2 = (0, 1, 0)^T$, $\mathbf{e}_3 = (0, 0, 1)^T$.

### Medium Solutions

**1.** $\|\mathbf{v}\| = \sqrt{2^2 + (-1)^2 + 2^2} = \sqrt{4 + 1 + 4} = \sqrt{9} = 3$.
$\hat{\mathbf{v}} = \frac{1}{3}(2, -1, 2)^T = \left(\frac{2}{3}, -\frac{1}{3}, \frac{2}{3}\right)^T \approx (0.667, -0.333, 0.667)^T$.
Verification: $\|\hat{\mathbf{v}}\| = \sqrt{(2/3)^2 + (-1/3)^2 + (2/3)^2} = \sqrt{\frac{4+1+4}{9}} = \sqrt{\frac{9}{9}} = \sqrt{1} = 1$. ✓

**2.** $\|\mathbf{v}\| = \sqrt{7^2 + (-24)^2} = \sqrt{49 + 576} = \sqrt{625} = 25$.
$\hat{\mathbf{v}} = \frac{1}{25}(7, -24)^T = \left(\frac{7}{25}, -\frac{24}{25}\right)^T = (0.28, -0.96)^T$.
Verification: $\mathbf{v} = 25 \times (0.28, -0.96)^T = (7, -24)^T$. ✓

**3.** $\|\mathbf{v}\| = \sqrt{1+1+1} = \sqrt{3}$. $\hat{\mathbf{v}} = \frac{1}{\sqrt{3}}(1,1,1)^T$.
$\|\mathbf{w}\| = \sqrt{25+25+25} = \sqrt{75} = 5\sqrt{3}$. $\hat{\mathbf{w}} = \frac{1}{5\sqrt{3}}(5,5,5)^T = \frac{1}{\sqrt{3}}(1,1,1)^T = \hat{\mathbf{v}}$.
Both are $\left(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^T \approx (0.577, 0.577, 0.577)^T$.

**4.** $\cos\theta = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = 0.5$.
$\theta = \arccos(0.5) = 60^\circ$ (or $\pi/3$ radians).

**5.** $\|\mathbf{u}\| = \sqrt{0.2^2 + 0.8^2 + 0.4^2} = \sqrt{0.04 + 0.64 + 0.16} = \sqrt{0.84} \approx 0.9165$.
$\|\mathbf{v}\| = \sqrt{0.6^2 + 0.5^2 + 0.6^2} = \sqrt{0.36 + 0.25 + 0.36} = \sqrt{0.97} \approx 0.9849$.
$\mathbf{u} \cdot \mathbf{v} = 0.2 \times 0.6 + 0.8 \times 0.5 + 0.4 \times 0.6 = 0.12 + 0.40 + 0.24 = 0.76$.
$\cos(\theta) = \frac{0.76}{0.9165 \times 0.9849} \approx \frac{0.76}{0.9027} \approx 0.842$.

### Hard Solutions

**1.** Let $\mathbf{v}$ be a non-zero vector. Then:
$$\widehat{-\mathbf{v}} = \frac{-\mathbf{v}}{\|-\mathbf{v}\|} = \frac{-\mathbf{v}}{\|\mathbf{v}\|} = -\frac{\mathbf{v}}{\|\mathbf{v}\|} = -\hat{\mathbf{v}}$$
This holds because $\|-\mathbf{v}\| = \sqrt{(-v_1)^2 + \cdots + (-v_n)^2} = \sqrt{v_1^2 + \cdots + v_n^2} = \|\mathbf{v}\|$. The negative sign in the numerator is preserved because $|-1| = 1$ in the denominator. ✓

**2.** Normalisation:
$\|\mathbf{x}\| = \sqrt{3^2 + 4^2} = 5$.
$\hat{\mathbf{x}} = \frac{1}{5}(3, 4)^T = (0.6, 0.8)^T$.

Scale by $\gamma = 2$:
$\gamma \hat{\mathbf{x}} = 2 \times (0.6, 0.8)^T = (1.2, 1.6)^T$.

Shift by $\beta = 1$:
$\mathbf{y} = (1.2 + 1, 1.6 + 1)^T = (2.2, 2.6)^T$.

The final output is $(2.2, 2.6)^T$. Note that $\beta$ is a scalar, so it is added to every component — this is equivalent to adding the vector $(\beta, \beta)^T$.

**3.** In vector normalisation, $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$ divides the vector by its norm, resulting in a unit vector with $\|\hat{\mathbf{v}}\| = 1$. In spectral normalisation, $\mathbf{W}_{\text{SN}} = \mathbf{W}/\sigma(\mathbf{W})$ divides the matrix by its spectral norm (largest singular value), resulting in a matrix with $\sigma(\mathbf{W}_{\text{SN}}) = 1$ — the matrix has unit spectral norm.

The analogy:
- $\|\mathbf{v}\|$ (Euclidean norm) measures the maximum stretching a vector can apply in any direction.
- $\sigma(\mathbf{W})$ (spectral norm) measures the maximum stretching the matrix can apply to any unit vector.

Both normalisations ensure the resulting object does not stretch anything beyond factor 1. In GANs, this constrains the Lipschitz constant of the discriminator to 1, which stabilises training by preventing the discriminator from becoming too steep.

## Related Concepts

- **Vector Magnitude (MATH-014):** The denominator in normalisation — $\|\mathbf{v}\|$.
- **Vector (MATH-002):** The object being normalised.
- **Scalar Multiplication (MATH-013):** Normalisation is scalar multiplication by $1/\|\mathbf{v}\|$.
- **Dot Product (MATH-016):** The dot product of unit vectors gives cosine similarity.
- **Standard Basis Vectors:** The unit vectors $\mathbf{e}_i$ along each axis.
- **Vector Addition (MATH-011):)** Any vector is a combination of scaled basis vectors.

## Next Concepts

- **Orthonormal Basis (MATH-033):** A basis where all vectors are mutually orthogonal unit vectors.
- **Normalisation Layers (MATH-041):** Batch norm, layer norm, instance norm — all involve normalising vectors.
- **Projection (MATH-030):** Projecting one vector onto a unit vector gives the component in that direction: $\text{proj}_{\hat{\mathbf{u}}}(\mathbf{v}) = (\mathbf{v} \cdot \hat{\mathbf{u}})\hat{\mathbf{u}}$.
- **Cosine Similarity (MATH-045):)** The dot product of unit vectors, used extensively in NLP and recommendation systems.
- **Spectral Normalisation (MATH-076):)** The matrix analogue of vector normalisation.

## Summary

A unit vector is a vector with magnitude exactly 1, representing pure direction. It is computed by dividing a non-zero vector by its magnitude: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$. Unit vectors are scaling-invariant (multiplying by any positive scalar does not change the direction). Standard basis vectors $\mathbf{e}_i$ are the unit vectors along each coordinate axis. Any vector can be decomposed into magnitude and direction: $\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$. In AI/ML, unit vectors are used in weight normalisation, cosine similarity for comparing embeddings, attention mechanisms, gradient direction analysis, and one-hot encoding.

## Key Takeaways

- A **unit vector** has magnitude exactly 1 and represents pure direction.
- Normalisation: $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$.
- Unit vectors are **scaling-invariant**: $\widehat{c\mathbf{v}} = \hat{\mathbf{v}}$ for $c > 0$.
- **Standard basis vectors** $\mathbf{e}_i$ have a 1 in one position and 0 elsewhere.
- Any vector can be decomposed: $\mathbf{v} = \|\mathbf{v}\| \cdot \hat{\mathbf{v}}$.
- The **dot product of two unit vectors** gives the cosine of the angle between them: $\cos\theta = \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$.
- In AI/ML, unit vectors enable weight normalisation, cosine similarity, attention mechanisms, and one-hot encoding.
- The zero vector **cannot** be normalised (division by zero).
