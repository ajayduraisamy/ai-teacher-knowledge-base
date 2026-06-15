# Concept: Vector

## Concept ID

MATH-002

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Define a vector as an ordered collection of numbers with magnitude and direction.
- Distinguish between scalars, vectors, and matrices.
- Represent vectors in component form and geometric form.
- Compute vector magnitude (norm) and identify a unit vector.
- Perform basic vector operations: addition, subtraction, and scalar multiplication.
- Recognize vectors in AI/ML contexts as feature vectors, weight vectors, and embeddings.

## Prerequisites

- **Scalar (MATH-001):** Understanding that a scalar is a single numerical value.
- Basic arithmetic: addition, subtraction, multiplication, division.
- Familiarity with the coordinate plane (x-axis and y-axis) at the high-school level.

## Definition

A **vector** is an ordered collection of numbers arranged in a specific sequence. Each number in the sequence is called a **component** or **element** of the vector. Vectors are used to represent quantities that have both **magnitude** (size) and **direction**.

In contrast to a **scalar** (a single number like 5 or $-3.14$), a vector packs multiple numbers together. For example:

$$\mathbf{v} = \begin{pmatrix} 3 \\ -2 \\ 7 \end{pmatrix}$$

This is a 3-dimensional vector with components $3$, $-2$, and $7$.

A vector with $n$ components is called an **$n$-dimensional vector** and belongs to the set $\mathbb{R}^n$ (read as "R-n"), where $\mathbb{R}$ denotes the set of real numbers.

## Intuition

Think of a vector as an **arrow** pointing from one location to another. The arrow has two properties:

1. **Length** (magnitude) — how far the arrow reaches.
2. **Direction** — which way the arrow points.

If a scalar tells you "how much", a vector tells you "how much **and** in which direction". For instance:

- A scalar says: "Walk 5 kilometres."
- A vector says: "Walk 5 kilometres **north-east**."

The 5 is the magnitude (a scalar), and "north-east" is the direction. Together they form a vector.

Another helpful analogy: a vector is like a **shopping list** with categories. If you have categories [apples, bananas, oranges], then the vector $[3, 1, 4]$ means 3 apples, 1 banana, and 4 oranges. The position in the list matters — swapping positions changes the meaning.

## Why This Concept Matters

Vectors are the bridge between simple numbers and the multi-dimensional world. They are the fundamental data structure of linear algebra and the backbone of nearly every AI and machine learning algorithm.

In machine learning:

- Every data point is a **feature vector** (e.g., $[age, income, credit\_score]$).
- Every neural network layer processes **vectors of activations**.
- Every word in NLP is represented as a **word embedding vector** (e.g., GloVe, Word2Vec).
- Every image fed into a CNN is flattened or represented as a vector.
- Model parameters (weights and biases) are stored as vectors.

Without vectors, there is no way to represent multi-dimensional data, measure distances between points, compute similarities, or perform the linear algebra operations that power deep learning.

## Historical Background

The concept of vectors emerged gradually over centuries:

- **17th century:** René Descartes introduced the coordinate system (Cartesian coordinates), laying the geometric foundation for representing points in space.
- **19th century:** Sir William Rowan Hamilton discovered quaternions (1843) and formally introduced the term "vector" (from Latin *vehere*, "to carry"). Around the same time, Hermann Grassmann developed a general theory of vector spaces in his *Ausdehnungslehre* (1844).
- **Late 19th century:** Josiah Willard Gibbs and Oliver Heaviside independently developed modern vector analysis (dot products, cross products, vector calculus) for use in physics, particularly electromagnetism.
- **20th century:** Vectors became central to quantum mechanics (state vectors in Hilbert space), computer graphics (transformation vectors), and eventually machine learning (feature vectors, support vector machines).
- **21st century:** Vectors are the core data representation in deep learning — every tensor in PyTorch or TensorFlow is built from vectors.

## Real World Examples

**Example 1: GPS Navigation.** When your GPS says "head 2.3 km south-west", it is providing a vector. The magnitude is 2.3 km, and the direction is south-west. Without the direction, you would not know which way to walk.

**Example 2: Wind.** A weather report stating "winds at 25 km/h from the east" is describing a wind velocity vector. The magnitude is 25 km/h and the direction is eastward.

**Example 3: Force.** Pushing a box with 10 Newtons of force to the right is a force vector. If two people push in opposite directions, the net force vector determines whether and where the box moves.

**Example 4: Image as a Vector.** A grayscale image of size $28 \times 28$ pixels (like MNIST digits) can be flattened into a 784-dimensional vector. Each component represents the intensity of one pixel.

**Example 5: Shopping Cart.** A supermarket trip produces a vector: $[milk, eggs, bread, apples]$ where each component is the quantity bought. The vector $[2, 12, 1, 5]$ means 2 cartons of milk, 12 eggs, 1 loaf of bread, and 5 apples.

## AI/ML Relevance

Vectors are everywhere in AI and machine learning. Here are the most important uses:

1. **Feature Vectors.** Every input to a machine learning model is a vector. For a house price prediction model, each house is represented by a feature vector like $[3, 2, 1500, 0.25, 2020]$ (bedrooms, bathrooms, square footage, lot size in acres, year built). The model learns patterns from these vectors.

2. **Word Embeddings.** In natural language processing, words are mapped to dense vectors (e.g., 300-dimensional). These vectors capture semantic meaning:
   $$\text{vector("king")} - \text{vector("man")} + \text{vector("woman")} \approx \text{vector("queen")}$$

3. **Weight Vectors.** In linear regression $y = \mathbf{w} \cdot \mathbf{x} + b$, the weight vector $\mathbf{w}$ contains one weight per feature. Training is the process of finding the optimal weight vector.

4. **Gradient Vectors.** During backpropagation, the gradient $\nabla L$ is a vector pointing in the direction of steepest ascent of the loss function. Gradient descent moves parameters in the opposite direction ($-\nabla L$) to reduce loss.

5. **Support Vector Machines (SVM).** SVMs find a decision boundary (hyperplane) defined by a normal vector $\mathbf{w}$ and a scalar bias $b$. The classification of a new point depends on its position relative to this hyperplane.

6. **Attention Mechanisms.** In transformers, queries, keys, and values are all vectors. The attention score between a query vector and key vectors determines how much each part of the input influences the output.

## Mathematical Explanation

### Notation

A vector is typically written in one of these ways:

- **Bold lowercase:** $\mathbf{v}$
- **Arrow notation:** $\vec{v}$
- **Component form:** $\mathbf{v} = (v_1, v_2, \ldots, v_n)$ or $\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$

The first form is a **row vector**; the second is a **column vector**. Column vectors are more common in linear algebra and machine learning.

### Vector Components

If $\mathbf{v} = \begin{pmatrix} 3 \\ -2 \\ 7 \end{pmatrix}$, then:
- The first component is $v_1 = 3$
- The second component is $v_2 = -2$
- The third component is $v_3 = 7$

The **dimension** (or length) of a vector is the number of components it has. This vector has dimension 3.

### Special Vectors

- **Zero vector:** $\mathbf{0} = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$ — has magnitude 0 and no direction.
- **Unit vector:** A vector with magnitude 1, denoted $\hat{\mathbf{v}}$. It indicates pure direction.
- **Standard basis vectors:** In $\mathbb{R}^3$, $\mathbf{i} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$, $\mathbf{j} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$, $\mathbf{k} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$. Any vector can be expressed as a combination of these.

### Position vs. Free Vectors

A **position vector** starts at the origin $(0, 0)$ and ends at a point $(x, y)$. A **free vector** can be placed anywhere in space — only its direction and magnitude matter, not its starting point.

## Formula(s)

### Vector Magnitude (Norm)

The magnitude (length, norm) of a vector $\mathbf{v} = (v_1, v_2, \ldots, v_n)$ is:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

This is also called the **Euclidean norm** or $L^2$ norm.

### Vector Addition

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

Vectors are added **component-wise**. Both vectors must have the same dimension.

### Vector Subtraction

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} u_1 - v_1 \\ u_2 - v_2 \\ \vdots \\ u_n - v_n \end{pmatrix}$$

### Scalar Multiplication

$$c \cdot \mathbf{v} = \begin{pmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{pmatrix}$$

Each component is multiplied by the scalar $c$. If $c > 0$, the direction stays the same; if $c < 0$, the direction reverses.

### Unit Vector

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

Dividing a vector by its magnitude yields a unit vector pointing in the same direction.

### Dot Product (Preview)

The dot product of two vectors produces a scalar:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

We will cover this in detail in a later concept.

## Properties

Let $\mathbf{u}, \mathbf{v}, \mathbf{w} \in \mathbb{R}^n$ be vectors, and let $a, b \in \mathbb{R}$ be scalars.

| Property | Formula |
|---|---|
| **Commutativity of addition** | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| **Associativity of addition** | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| **Identity element (zero vector)** | $\mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| **Additive inverse** | $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| **Distributivity (scalar over vector)** | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| **Distributivity (vector over scalar)** | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |
| **Associativity of scalar multiplication** | $a(b\mathbf{v}) = (ab)\mathbf{v}$ |
| **Scalar identity** | $1 \cdot \mathbf{v} = \mathbf{v}$ |
| **Triangle inequality** | $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ |
| **Positive definiteness of norm** | $\|\mathbf{v}\| \geq 0$, and $\|\mathbf{v}\| = 0$ iff $\mathbf{v} = \mathbf{0}$ |

## Step-by-Step Worked Examples

### Example 1: Vector Addition and Magnitude

Let $\mathbf{u} = \begin{pmatrix} 3 \\ -1 \\ 4 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} -2 \\ 5 \\ 0 \end{pmatrix}$. Compute $\mathbf{u} + \mathbf{v}$, $\mathbf{u} - \mathbf{v}$, and $\|\mathbf{u}\|$.

**Step 1 — Addition:** Add corresponding components:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 3 + (-2) \\ -1 + 5 \\ 4 + 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 4 \\ 4 \end{pmatrix}$$

**Step 2 — Subtraction:** Subtract corresponding components:

$$\mathbf{u} - \mathbf{v} = \begin{pmatrix} 3 - (-2) \\ -1 - 5 \\ 4 - 0 \end{pmatrix} = \begin{pmatrix} 5 \\ -6 \\ 4 \end{pmatrix}$$

**Step 3 — Magnitude of $\mathbf{u}$:** Square each component, sum, and take the square root:

$$\|\mathbf{u}\| = \sqrt{3^2 + (-1)^2 + 4^2} = \sqrt{9 + 1 + 16} = \sqrt{26} \approx 5.099$$

**Answer:** $\mathbf{u} + \mathbf{v} = \begin{pmatrix}1 & 4 & 4\end{pmatrix}^T$, $\mathbf{u} - \mathbf{v} = \begin{pmatrix}5 & -6 & 4\end{pmatrix}^T$, $\|\mathbf{u}\| = \sqrt{26} \approx 5.099$.

### Example 2: Scalar Multiplication and Unit Vector

Let $\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix}$. Compute $2\mathbf{v}$, $-0.5\mathbf{v}$, and the unit vector $\hat{\mathbf{v}}$.

**Step 1 — Multiply by 2:** Scale each component by 2:

$$2\mathbf{v} = \begin{pmatrix} 2 \times 3 \\ 2 \times 4 \end{pmatrix} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$$

**Step 2 — Multiply by $-0.5$:** Scale each component by $-0.5$, which shrinks and reverses direction:

$$-0.5\mathbf{v} = \begin{pmatrix} -0.5 \times 3 \\ -0.5 \times 4 \end{pmatrix} = \begin{pmatrix} -1.5 \\ -2 \end{pmatrix}$$

**Step 3 — Compute magnitude:** $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.

**Step 4 — Unit vector:** Divide each component by the magnitude:

$$\hat{\mathbf{v}} = \frac{1}{5} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0.6 \\ 0.8 \end{pmatrix}$$

**Verification:** $\|\hat{\mathbf{v}}\| = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1} = 1$. ✓

**Answer:** $2\mathbf{v} = \begin{pmatrix}6 & 8\end{pmatrix}^T$, $-0.5\mathbf{v} = \begin{pmatrix}-1.5 & -2\end{pmatrix}^T$, $\hat{\mathbf{v}} = \begin{pmatrix}0.6 & 0.8\end{pmatrix}^T$.

### Example 3: Building a Feature Vector

A house has 3 bedrooms, 2 bathrooms, 1800 square feet of living area, a lot size of 0.3 acres, and was built in 2015. Construct a 5-dimensional feature vector for this house. Then compute its magnitude (as a measure of its "size" in feature space).

**Step 1 — Define features:** We decide on a consistent ordering:

$$\mathbf{x} = \begin{pmatrix} \text{bedrooms} \\ \text{bathrooms} \\ \text{sq\_ft} \\ \text{lot\_acres} \\ \text{year\_built} \end{pmatrix}$$

**Step 2 — Fill in values:**

$$\mathbf{x} = \begin{pmatrix} 3 \\ 2 \\ 1800 \\ 0.3 \\ 2015 \end{pmatrix}$$

This is the feature vector for this particular house.

**Step 3 — Compute magnitude:** The magnitude gives a sense of the "total size" of the feature vector, though it is rarely used directly because different features have different scales (year_built dominates because it is large).

$$\|\mathbf{x}\| = \sqrt{3^2 + 2^2 + 1800^2 + 0.3^2 + 2015^2}$$

$$= \sqrt{9 + 4 + 3240000 + 0.09 + 4060225}$$

$$= \sqrt{7300238.09} \approx 2701.9$$

**Note:** In practice, features are **normalised** (scaled to similar ranges) before computing distances or magnitudes. Otherwise, features with large numeric ranges (like year_built) dominate.

### Example 4: Neural Network Activation Vector

A neural network layer has 4 neurons with activations $a_1 = 0.9$, $a_2 = 0.1$, $a_3 = 0.7$, $a_4 = 0.0$. Represent these activations as a vector. Then apply a ReLU activation, which replaces negative values with 0 (none are negative here). Finally, compute the $L^1$ norm (sum of absolute values).

**Step 1 — Activation vector:**

$$\mathbf{a} = \begin{pmatrix} 0.9 \\ 0.1 \\ 0.7 \\ 0.0 \end{pmatrix}$$

**Step 2 — ReLU activation:** $\text{ReLU}(x) = \max(0, x)$. Since all components are non-negative, the vector is unchanged:

$$\mathbf{a}_{\text{out}} = \begin{pmatrix} 0.9 \\ 0.1 \\ 0.7 \\ 0.0 \end{pmatrix}$$

**Step 3 — $L^1$ norm:** Sum of absolute values:

$$\|\mathbf{a}\|_1 = |0.9| + |0.1| + |0.7| + |0.0| = 0.9 + 0.1 + 0.7 + 0.0 = 1.7$$

**Answer:** The activation vector is $\begin{pmatrix}0.9 & 0.1 & 0.7 & 0.0\end{pmatrix}^T$ and its $L^1$ norm is $1.7$.

## Visual Interpretation

A vector in 2-dimensional space ($\mathbb{R}^2$) is drawn as an **arrow** from the origin $(0, 0)$ to the point $(x, y)$.

```
y
▲
│
4 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│                      ╱
3 ─ ─ ─ ─ ─ ─ ─ ─ ─ ╱
│                  ╱
2 ─ ─ ─ ─ ─ ─ ─ ╱
│              ╱
1 ─ ─ ─ ─ ─ ╱
│          ╱
0 ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶ x
│      1   2   3   4   5
```

This arrow represents the vector $\mathbf{v} = (4, 3)$. Its length is $\sqrt{4^2 + 3^2} = 5$, and its direction is approximately $36.87^\circ$ above the horizontal axis.

When you add two vectors visually, you place the tail of the second vector at the tip of the first (the **tip-to-tail** method). The sum is the vector from the tail of the first to the tip of the second.

```
y
▲
│         ╱
│       ╱  u+v
│     ╱
│   ╱
│ ╱
└───────────▶ x
```

When you multiply a vector by a scalar $c$:
- If $c > 1$, the vector **stretches** (becomes longer).
- If $0 < c < 1$, the vector **shrinks** (becomes shorter).
- If $c < 0$, the vector **reverses direction** and stretches/shrinks by $|c|$.

## Common Mistakes

1. **Adding vectors of different dimensions.** You cannot add a 2D vector to a 3D vector. They must have the same number of components. For example, $\begin{pmatrix}1 \\ 2\end{pmatrix} + \begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix}$ is undefined.

2. **Confusing a vector with a set or a scalar.** A vector is an *ordered* tuple — $\begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix} \neq \begin{pmatrix}3 \\ 2 \\ 1\end{pmatrix}$ — while a set $\{1, 2, 3\}$ is unordered. Also, $\begin{pmatrix}5\end{pmatrix}$ (a 1D vector) is different from the scalar $5$ (they transform differently under coordinate changes).

3. **Forgetting the square root when computing magnitude.** The magnitude is $\sqrt{v_1^2 + v_2^2 + \cdots}$, not $v_1^2 + v_2^2 + \cdots$. The squared sum is the **squared norm**, not the norm itself.

4. **Thinking all vectors start at the origin.** Position vectors start at the origin, but free vectors can be placed anywhere. In physics, the force vector applied to an object is a free vector that can be drawn starting at the point of application.

5. **Confusing row vectors and column vectors.** In linear algebra, $\begin{pmatrix}1 & 2 & 3\end{pmatrix}$ (row) and $\begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix}$ (column) are different. Multiplying a matrix by a column vector is defined, but multiplying by a row vector requires the row vector to be on the left.

6. **Assuming the magnitude must be an integer.** Magnitudes often involve square roots that do not simplify to integers. For example, $\|\begin{pmatrix}1 \\ 1\end{pmatrix}\| = \sqrt{2}$ is perfectly valid.

7. **Misinterpreting negative components.** A negative component does not mean the vector has a negative magnitude. Magnitude is always non-negative. A negative component simply points in the opposite direction along that axis.

## Interview Questions

### Beginner

1. **What is a vector, and how is it different from a scalar?**
   *Answer: A vector is an ordered collection of numbers with both magnitude and direction, while a scalar is a single number with only magnitude. For example, 5 km/h (speed) is a scalar, while 5 km/h north (velocity) is a vector.*

2. **How do you compute the magnitude of a 2D vector $\mathbf{v} = (3, 4)$?**
   *Answer: The magnitude is $\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$.*

3. **What is the zero vector? Give an example in $\mathbb{R}^3$.**
   *Answer: The zero vector has all components equal to 0. In $\mathbb{R}^3$, it is $\mathbf{0} = \begin{pmatrix}0 & 0 & 0\end{pmatrix}^T$. It has magnitude 0 and undefined direction.*

4. **Can a vector have negative components? What does that mean geometrically?**
   *Answer: Yes. A negative component means the vector points in the negative direction along that axis. For example, $\mathbf{v} = \begin{pmatrix}-3 \\ 2\end{pmatrix}$ points left and up.*

5. **What is a unit vector, and how is it computed?**
   *Answer: A unit vector has magnitude 1 and indicates pure direction. It is computed by dividing a vector by its magnitude: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$.*

6. **What happens geometrically when you multiply a vector by a negative scalar?**
   *Answer: The vector reverses direction (180-degree flip) and its magnitude scales by the absolute value of the scalar.*

### Intermediate

1. **Why must vectors being added have the same dimension?**
   *Answer: Vector addition is defined component-wise: each component of the result is the sum of the corresponding components of the inputs. If the dimensions differ, there is a mismatch in the number of components, and the operation is undefined.*

2. **In machine learning, why is it important to normalise feature vectors before computing distances?**
   *Answer: Features often have different units and scales (e.g., age 0–100 vs. income $0–$200k). Without normalisation, features with larger numeric ranges dominate distance calculations (like Euclidean distance). Normalising (e.g., z-score scaling or min-max scaling) ensures each feature contributes proportionally.*

3. **Explain the difference between a position vector and a free vector. Give an example of each in AI.**
   *Answer: A position vector is anchored at the origin and represents a point in space (e.g., the coordinates of a pixel in an image). A free vector has no fixed starting point — only magnitude and direction matter (e.g., a gradient vector in neural network training, which indicates direction of steepest ascent regardless of where it is applied).*

4. **The triangle inequality states $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$. Provide an intuitive explanation.**
   *Answer: Imagine walking along $\mathbf{u}$ and then along $\mathbf{v}$. The direct path from start to finish ($\mathbf{u} + \mathbf{v}$) cannot be longer than walking both legs separately. This is why "shortcuts" are shorter — the straight-line distance is always less than or equal to the sum of the two legs.*

5. **How is a word embedding vector in NLP different from a one-hot encoded vector?**
   *Answer: A one-hot vector has length equal to vocabulary size, with a single 1 and the rest 0s (e.g., $[0,0,1,0,\ldots,0]$). It is sparse and high-dimensional. A word embedding (e.g., 300-dimensional) is dense and low-dimensional, with each component learned to capture semantic meaning. Embeddings place similar words near each other in vector space.*

### Advanced

1. **What is the $\ell^1$ norm and why is it used in Lasso regression (L1 regularisation)?**
   *Answer: The $\ell^1$ norm is $\|\mathbf{w}\|_1 = \sum_{i=1}^n |w_i|$. Lasso regression adds $\lambda\|\mathbf{w}\|_1$ to the loss function. Because the $\ell^1$ norm has a "pointy" geometry (diamond-shaped in 2D), it pushes some coefficients to exactly zero during optimisation, performing automatic feature selection.*

2. **Explain why transformers use the dot product of query and key vectors for attention rather than Euclidean distance.**
   *Answer: The dot product $\mathbf{q} \cdot \mathbf{k}$ measures both the magnitude alignment and directional alignment between query and key. It is computationally efficient (just multiply and sum) and scales well to high dimensions. In contrast, Euclidean distance only measures proximity and loses directional information. The dot product can also be scaled by $1/\sqrt{d_k}$ to control variance, which is important for stable gradients.*

3. **A neural network's gradient $\nabla L$ is a vector. Explain why the negative gradient $-\nabla L$ gives the steepest descent direction. How does this connect to directional derivatives?**
   *Answer: The directional derivative of $L$ in direction $\mathbf{d}$ is $\nabla L \cdot \mathbf{d} = \|\nabla L\| \cos\theta$, where $\theta$ is the angle between $\nabla L$ and $\mathbf{d}$. This is maximised when $\cos\theta = 1$ (moving in the gradient direction) and minimised when $\cos\theta = -1$ (moving opposite to the gradient). Since we want to *decrease* $L$, we choose $\mathbf{d} = -\nabla L/\|\nabla L\|$, giving a directional derivative of $-\|\nabla L\|$, which is the most negative (steepest descent) possible.*

## Practice Problems

### Easy

1. Let $\mathbf{a} = \begin{pmatrix} 2 \\ 5 \end{pmatrix}$ and $\mathbf{b} = \begin{pmatrix} -3 \\ 1 \end{pmatrix}$. Compute $\mathbf{a} + \mathbf{b}$ and $\mathbf{a} - \mathbf{b}$.

2. Compute the magnitude of the vector $\mathbf{v} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$.

3. Let $\mathbf{v} = \begin{pmatrix} 2 \\ -3 \\ 1 \end{pmatrix}$. Compute $3\mathbf{v}$ and $-2\mathbf{v}$.

4. Find a unit vector in the direction of $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}$.

5. A 3D vector has components $x = 1$, $y = 2$, $z = 3$. Write it in column form and compute its magnitude.

### Medium

1. Let $\mathbf{u} = \begin{pmatrix} 1 \\ -2 \\ 3 \end{pmatrix}$, $\mathbf{v} = \begin{pmatrix} 4 \\ 0 \\ -1 \end{pmatrix}$. Compute $2\mathbf{u} - 3\mathbf{v}$ and verify the triangle inequality: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.

2. A dataset has three features normalised to z-scores: $[-0.5, 1.2, 0.8]$. Construct the feature vector and compute its $L^2$ norm.

3. Given $\mathbf{v} = \begin{pmatrix} 0 \\ 3 \\ 4 \end{pmatrix}$, find a scalar $c$ such that $\|c\mathbf{v}\| = 10$.

4. Prove that $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\|$ for any scalar $c$ and vector $\mathbf{v} \in \mathbb{R}^3$. Use an explicit example to verify.

5. The gradient vector for a 2-parameter model is $\nabla L = \begin{pmatrix} 2 \\ -3 \end{pmatrix}$. The learning rate is $\alpha = 0.1$. Compute the parameter update $\Delta\mathbf{w} = -\alpha \nabla L$. What is the magnitude of this update?

### Hard

1. Consider vectors $\mathbf{u} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$. Find a vector $\mathbf{w}$ that is a linear combination $a\mathbf{u} + b\mathbf{v}$ and has magnitude exactly 1 with $a$ and $b$ being scalars you determine. (Hint: first compute $a\mathbf{u} + b\mathbf{v}$ in terms of $a$ and $b$, then set its norm to 1 and solve with a convenient choice.)

2. In a 3-class classification problem, the output logits for one sample are $\mathbf{z} = \begin{pmatrix} 2.0 \\ 1.0 \\ 0.1 \end{pmatrix}$. The softmax function produces probabilities $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Compute the probability vector $\mathbf{p}$. Then compute $\|\mathbf{p}\|$. Why is the sum of components of $\mathbf{p}$ exactly 1?

3. A model's weight vector is $\mathbf{w} = \begin{pmatrix} w_1 \\ w_2 \end{pmatrix}$. The L2 regularisation penalty is $\frac{\lambda}{2} \|\mathbf{w}\|^2$. If $\lambda = 0.5$ and $\mathbf{w} = \begin{pmatrix} 3 \\ -4 \end{pmatrix}$, compute the penalty. Then, compute the gradient of the penalty with respect to $\mathbf{w}$ (the vector of partial derivatives $\frac{\partial}{\partial w_i}$ of the penalty). Show that the gradient is $\lambda \mathbf{w}$.

## Solutions

### Easy Solutions

**1.** $\mathbf{a} + \mathbf{b} = \begin{pmatrix} 2 + (-3) \\ 5 + 1 \end{pmatrix} = \begin{pmatrix} -1 \\ 6 \end{pmatrix}$.
$\mathbf{a} - \mathbf{b} = \begin{pmatrix} 2 - (-3) \\ 5 - 1 \end{pmatrix} = \begin{pmatrix} 5 \\ 4 \end{pmatrix}$.

**2.** $\|\mathbf{v}\| = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10$.

**3.** $3\mathbf{v} = \begin{pmatrix} 3 \times 2 \\ 3 \times (-3) \\ 3 \times 1 \end{pmatrix} = \begin{pmatrix} 6 \\ -9 \\ 3 \end{pmatrix}$.
$-2\mathbf{v} = \begin{pmatrix} -2 \times 2 \\ -2 \times (-3) \\ -2 \times 1 \end{pmatrix} = \begin{pmatrix} -4 \\ 6 \\ -2 \end{pmatrix}$.

**4.** $\|\mathbf{u}\| = \sqrt{1^2 + 2^2 + 2^2} = \sqrt{1 + 4 + 4} = \sqrt{9} = 3$.
$\hat{\mathbf{u}} = \frac{1}{3} \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix} = \begin{pmatrix} 1/3 \\ 2/3 \\ 2/3 \end{pmatrix}$.

**5.** $\mathbf{v} = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$. $\|\mathbf{v}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$.

### Medium Solutions

**1.** $2\mathbf{u} = \begin{pmatrix} 2 \\ -4 \\ 6 \end{pmatrix}$, $3\mathbf{v} = \begin{pmatrix} 12 \\ 0 \\ -3 \end{pmatrix}$.
$2\mathbf{u} - 3\mathbf{v} = \begin{pmatrix} 2 - 12 \\ -4 - 0 \\ 6 - (-3) \end{pmatrix} = \begin{pmatrix} -10 \\ -4 \\ 9 \end{pmatrix}$.
Triangle inequality check:
$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 1+4 \\ -2+0 \\ 3+(-1) \end{pmatrix} = \begin{pmatrix} 5 \\ -2 \\ 2 \end{pmatrix}$.
$\|\mathbf{u} + \mathbf{v}\| = \sqrt{25 + 4 + 4} = \sqrt{33} \approx 5.745$.
$\|\mathbf{u}\| = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$.
$\|\mathbf{v}\| = \sqrt{16 + 0 + 1} = \sqrt{17} \approx 4.123$.
$3.742 + 4.123 = 7.865 \geq 5.745$. ✓

**2.** Feature vector: $\mathbf{x} = \begin{pmatrix} -0.5 \\ 1.2 \\ 0.8 \end{pmatrix}$.
$\|\mathbf{x}\| = \sqrt{(-0.5)^2 + (1.2)^2 + (0.8)^2} = \sqrt{0.25 + 1.44 + 0.64} = \sqrt{2.33} \approx 1.526$.

**3.** $\|c\mathbf{v}\| = |c| \cdot \|\mathbf{v}\| = |c| \cdot \sqrt{0^2 + 3^2 + 4^2} = |c| \cdot 5$.
Set $5|c| = 10$, so $|c| = 2$. Thus $c = 2$ or $c = -2$.

**4.** For $\mathbf{v} = (v_1, v_2, v_3)$, $c\mathbf{v} = (cv_1, cv_2, cv_3)$.
$\|c\mathbf{v}\| = \sqrt{(cv_1)^2 + (cv_2)^2 + (cv_3)^2} = \sqrt{c^2(v_1^2 + v_2^2 + v_3^2)} = |c|\sqrt{v_1^2 + v_2^2 + v_3^2} = |c|\cdot\|\mathbf{v}\|$.
Example: $\mathbf{v} = (1, 2, 3)$, $c = -2$. $\|-2\mathbf{v}\| = \|(-2, -4, -6)\| = \sqrt{4+16+36} = \sqrt{56} \approx 7.483$.
$|c|\cdot\|\mathbf{v}\| = 2 \times \sqrt{14} = 2 \times 3.742 = 7.483$. ✓

**5.** $\Delta\mathbf{w} = -0.1 \times \begin{pmatrix} 2 \\ -3 \end{pmatrix} = \begin{pmatrix} -0.2 \\ 0.3 \end{pmatrix}$.
Magnitude: $\|\Delta\mathbf{w}\| = \sqrt{(-0.2)^2 + 0.3^2} = \sqrt{0.04 + 0.09} = \sqrt{0.13} \approx 0.361$.

### Hard Solutions

**1.** $a\mathbf{u} + b\mathbf{v} = a\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} + b\begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix} = \begin{pmatrix} a + 4b \\ 2a + 5b \\ 3a + 6b \end{pmatrix}$.
We need $\|a\mathbf{u} + b\mathbf{v}\| = 1$. Choose a convenient direction; say $a = 1, b = 0$ gives $\mathbf{u}$ with magnitude $\sqrt{14} \approx 3.742$. So $\mathbf{w} = \hat{\mathbf{u}} = \frac{1}{\sqrt{14}}\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ works.
For a more interesting choice, let $a = 1, b = 1$: $\mathbf{w}_0 = \mathbf{u} + \mathbf{v} = \begin{pmatrix} 5 \\ 7 \\ 9 \end{pmatrix}$.
$\|\mathbf{w}_0\| = \sqrt{25 + 49 + 81} = \sqrt{155}$.
Normalise: $\mathbf{w} = \frac{1}{\sqrt{155}}\begin{pmatrix} 5 \\ 7 \\ 9 \end{pmatrix}$.
Then $\|\mathbf{w}\| = 1$ by construction. So one solution is $a = 5/\sqrt{155}$, $b = 5/\sqrt{155}$ (since $a+4b = 5/\sqrt{155}$ is not right — let me solve properly).

Wait, let me use $a=1, b=1$. Then the vector is $\begin{pmatrix}5 \\ 7 \\ 9\end{pmatrix}$, magnitude $\sqrt{155}$. To get a unit vector, divide by $\sqrt{155}$:
$\mathbf{w} = \frac{1}{\sqrt{155}}\begin{pmatrix}5 \\ 7 \\ 9\end{pmatrix}$.
We need $a$ and $b$ such that $a\mathbf{u} + b\mathbf{v} = \mathbf{w}$. So:
$a + 4b = 5/\sqrt{155}$
$2a + 5b = 7/\sqrt{155}$
$3a + 6b = 9/\sqrt{155}$

From the first two: subtract 2$\times$first from second: $(2a+5b) - 2(a+4b) = (7 - 10)/\sqrt{155}$, so $-3b = -3/\sqrt{155}$, thus $b = 1/\sqrt{155}$.
Then $a = (5 - 4)/\sqrt{155} = 1/\sqrt{155}$.
Check third: $3(1) + 6(1) = 9$ ✓.
So $a = 1/\sqrt{155}$, $b = 1/\sqrt{155}$.

**2.** Compute exponentials: $e^{2.0} \approx 7.389$, $e^{1.0} \approx 2.718$, $e^{0.1} \approx 1.105$.
Sum: $7.389 + 2.718 + 1.105 = 11.212$.
Probabilities:
$p_1 = 7.389 / 11.212 \approx 0.659$
$p_2 = 2.718 / 11.212 \approx 0.242$
$p_3 = 1.105 / 11.212 \approx 0.099$

Probability vector: $\mathbf{p} = \begin{pmatrix} 0.659 \\ 0.242 \\ 0.099 \end{pmatrix}$.
$\|\mathbf{p}\| = \sqrt{0.659^2 + 0.242^2 + 0.099^2} = \sqrt{0.434 + 0.059 + 0.010} = \sqrt{0.503} \approx 0.709$.

The sum of components is exactly 1 because the softmax denominator normalises each exponential by the total sum: $\sum_i p_i = \sum_i \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{\sum_i e^{z_i}}{\sum_j e^{z_j}} = 1$.

**3.** $\|\mathbf{w}\|^2 = 3^2 + (-4)^2 = 9 + 16 = 25$.
Penalty: $\frac{\lambda}{2} \|\mathbf{w}\|^2 = \frac{0.5}{2} \times 25 = 0.25 \times 25 = 6.25$.

Gradient of the penalty with respect to $\mathbf{w}$:
$\frac{\partial}{\partial \mathbf{w}} \left( \frac{\lambda}{2} \|\mathbf{w}\|^2 \right) = \frac{\lambda}{2} \cdot \frac{\partial}{\partial \mathbf{w}} (w_1^2 + w_2^2) = \frac{\lambda}{2} \cdot \begin{pmatrix} 2w_1 \\ 2w_2 \end{pmatrix} = \lambda \begin{pmatrix} w_1 \\ w_2 \end{pmatrix} = \lambda \mathbf{w}$.

With our numbers: $\lambda \mathbf{w} = 0.5 \times \begin{pmatrix} 3 \\ -4 \end{pmatrix} = \begin{pmatrix} 1.5 \\ -2 \end{pmatrix}$.

## Related Concepts

- **Scalar (MATH-001):** The building block of vectors — each component of a vector is a scalar.
- **Vector Addition (MATH-011):** The operation of adding vectors component-wise.
- **Scalar Multiplication (MATH-013):** Scaling a vector by a single number.
- **Vector Magnitude (MATH-014):** The length of a vector, computed via the Euclidean norm.
- **Unit Vector (MATH-015):** A vector of length 1 used to represent pure direction.
- **Dot Product (MATH-016):** An operation on two vectors that produces a scalar, measuring alignment.
- **Cross Product (MATH-017):** An operation on two 3D vectors that produces a vector perpendicular to both.
- **Tensor (MATH-004):** A generalisation where scalars are rank-0 tensors and vectors are rank-1 tensors.

## Next Concepts

- **Matrix (MATH-003):** A rectangular array of numbers — essentially a collection of vectors arranged as rows or columns. Understanding vectors is essential before learning matrices.
- **Vector Space (MATH-031):** A formal mathematical structure where vectors live, with axioms governing addition and scalar multiplication.
- **Basis (MATH-032):** A set of independent vectors that can represent every vector in a space uniquely.
- **Linear Independence (MATH-034):** A property of a set of vectors where no vector can be expressed as a combination of the others.
- **Gradient (MATH-058):** A vector of partial derivatives that points in the direction of steepest ascent.

## Summary

A vector is an ordered collection of numbers that has both magnitude and direction. Vectors are the fundamental data structure of linear algebra and form the backbone of modern AI and machine learning. Key operations include vector addition, subtraction, scalar multiplication, and computing magnitude. Every input to an ML model (feature vector), every set of learned parameters (weight vector), and every optimisation direction (gradient vector) is a vector. Understanding vectors is essential for progressing through matrix algebra, linear transformations, and deep learning.

## Key Takeaways

- A vector is an ordered tuple of numbers with magnitude and direction.
- Vectors can be added and subtracted component-wise; only vectors of the same dimension can be combined.
- Scalar multiplication scales each component and changes the magnitude by the factor $|c|$, reversing direction if $c < 0$.
- The magnitude (norm) of a vector $\mathbf{v}$ is $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^n v_i^2}$.
- A unit vector $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$ has magnitude 1 and indicates pure direction.
- In AI/ML, vectors represent data points (feature vectors), model parameters (weight vectors), and optimisation directions (gradient vectors).
- Vectors satisfy algebraic properties including commutativity, associativity, distributivity, and the triangle inequality.
