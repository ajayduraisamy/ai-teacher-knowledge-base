# Concept: Linear Transformation

## Concept ID

MATH-036

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Domain

Linear Algebra

## Learning Objectives

1. Define a linear transformation and verify whether a given map is linear.
2. Represent a linear transformation as a matrix.
3. Apply geometric linear transformations (rotation, scaling, shear, projection) to vectors.
4. Compute the kernel (null space) and image (column space) of a linear transformation.
5. Compose linear transformations and relate composition to matrix multiplication.
6. Connect linear transformations to machine learning models including neural networks, PCA, and attention mechanisms.

## Prerequisites

- Vector spaces and subspaces
- Linear independence and dependence (MATH-034, MATH-035)
- Matrix operations (addition, multiplication)
- Systems of linear equations
- Basis and dimension

## Definition

Let $V$ and $W$ be vector spaces over the same field $\mathbb{F}$ (typically $\mathbb{R}$). A function $T: V \rightarrow W$ is called a **linear transformation** (or **linear map**) if it satisfies two properties:

1. **Additivity (preserves vector addition):**
   $$
   T(u + v) = T(u) + T(v) \quad \text{for all } u, v \in V
   $$

2. **Homogeneity (preserves scalar multiplication):**
   $$
   T(cv) = c \cdot T(v) \quad \text{for all } c \in \mathbb{F}, v \in V
   $$

These two conditions can be combined into a single condition:

$$
T(c_1 v_1 + c_2 v_2) = c_1 T(v_1) + c_2 T(v_2) \quad \text{for all } c_1, c_2 \in \mathbb{F}, v_1, v_2 \in V
$$

A linear transformation is a function that "respects" the linear structure of vector spaces — it maps linear combinations to linear combinations.

## Intuition

A linear transformation is a function that takes vectors as input and produces vectors as output while preserving the "shape" of the vector space. Key intuitions:

- **Grid lines remain straight and parallel:** If you visualize $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$, the image of a grid of parallel lines remains a grid of parallel lines (possibly rotated, scaled, or sheared, but always linear).
- **Origin stays fixed:** $T(0) = 0$ always holds for a linear transformation (since $T(0) = T(0 \cdot v) = 0 \cdot T(v) = 0$).
- **Lines through origin map to lines through origin:** The image of any line through the origin is another line through the origin (or just the origin).
- **Linear combinations are preserved:** A linear transformation is completely determined by its action on a basis. If you know $T(v_1), \dots, T(v_n)$ for a basis $\{v_1, \dots, v_n\}$, you know $T$ everywhere.

Think of a linear transformation as a "machine" that takes in vectors, applies a matrix multiplication, and outputs transformed vectors.

## Why This Concept Matters

Linear transformations are the central objects of study in linear algebra and are ubiquitous in applied mathematics:

- **Bridge between geometry and algebra:** Every linear transformation has a matrix representation, allowing geometric operations to be computed algebraically.
- **Change of basis:** Linear transformations describe how coordinates change when switching between different bases.
- **Solving linear systems:** The equation $Ax = b$ asks for a vector $x$ that $T$ maps to $b$, where $T$ is the linear transformation represented by $A$.
- **Computer graphics:** Rotations, scaling, shearing, and projections in 2D and 3D graphics are all linear transformations.
- **Machine learning:** Neural network layers, PCA whitening, attention mechanisms, and data preprocessing transformations are all (or include) linear transformations.
- **Signal processing:** The Fourier transform, wavelet transform, and convolution are linear transformations of function spaces.

## Historical Background

The concept of a linear transformation developed over the 19th and early 20th centuries:

- **Gottfried Wilhelm Leibniz** (1646–1716) used linear substitutions in his work on systems of equations, foreshadowing the idea of linear maps.
- **Augustin-Louis Cauchy** (1789–1857) studied linear transformations in the context of determinants and eigenvalues.
- **Carl Friedrich Gauss** (1777–1855) developed Gaussian elimination, which is fundamentally about finding preimages under linear transformations.
- **Arthur Cayley** (1821–1895) formalized matrix multiplication and recognized that matrices represent linear transformations. His 1858 paper "A Memoir on the Theory of Matrices" was foundational.
- **James Joseph Sylvester** (1814–1897) coined the term "matrix" and worked on the theory of linear transformations.
- **Hermann Grassmann** (1809–1877) studied linear transformations in his *Ausdehnungslehre* and recognized them as structure-preserving maps between vector spaces.
- **David Hilbert** (1862–1943) extended the theory to infinite-dimensional spaces, leading to functional analysis.
- **John von Neumann** (1903–1957) applied linear transformations to quantum mechanics, where observables are represented as self-adjoint linear operators on Hilbert spaces.

The modern definition (additivity + homogeneity) was standardized in the early 20th century as part of the axiomatization of linear algebra.

## Real World Examples

1. **Computer Graphics:** In 3D rendering, every object undergoes a sequence of linear transformations: model transformation (rotation, scaling, translation via homogeneous coordinates), view transformation (camera orientation), and projection transformation (perspective or orthographic).

2. **Robotics:** The forward kinematics of a robot arm is a composition of rotations and translations (linear transformations in homogeneous coordinates). Each joint's motion is a linear transformation applied to the end-effector position.

3. **Image Processing:** Convolution with a kernel is a linear transformation on the space of images. Edge detection (Sobel filter), blurring (Gaussian filter), and sharpening are all linear transformations.

4. **Economics:** Input-output models use linear transformations to describe how changes in demand for one industry affect production across all industries (Leontief input-output model).

5. **GPS and Navigation:** Converting between coordinate systems (e.g., Earth-centered inertial to Earth-centered Earth-fixed, or latitude/longitude to UTM) involves linear transformations and rotations.

## AI/ML Relevance

1. **Neural Network Layers:** A fully connected (dense) layer computes $h = Wx + b$, where $Wx$ is a linear transformation followed by a bias addition and a nonlinear activation function. Deep neural networks are compositions of linear transformations and nonlinearities. The expressivity of a network depends on how these linear transformations interact with activation functions.

2. **Principal Component Analysis (PCA):** PCA finds a linear transformation $W$ that projects data onto the directions of maximum variance. The transformation $z = W^T x$ maps high-dimensional data to a lower-dimensional representation, and the inverse $x \approx W z$ reconstructs it. PCA is an orthogonal linear transformation.

3. **Whitening:** Whitening transforms data to have identity covariance: $z = S^{-1/2}U^T x$, where $U$ and $S$ come from the eigendecomposition of the covariance matrix. This is a linear transformation that decorrelates features.

4. **Attention Mechanisms:** In Transformer models, the attention mechanism computes attention scores via linear transformations of queries, keys, and values. The output is a weighted sum of value vectors, which is itself a linear transformation (though with data-dependent weights).

5. **Word Embeddings:** Methods like Word2Vec learn linear transformations between word embeddings to capture analogies: $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$. This works because the embedding space encodes semantic relationships as linear transformations.

6. **Data Augmentation:** In computer vision, random rotations, flips, and scaling are linear transformations applied to training images to increase dataset diversity.

7. **Linear Discriminant Analysis (LDA):** LDA finds a linear transformation $w$ that maximizes class separation: $w^T S_b w / w^T S_w w$, where $S_b$ and $S_w$ are between-class and within-class scatter matrices.

## Mathematical Explanation

### Matrix Representation

Every linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ can be represented by an $m \times n$ matrix $A$ such that:

$$
T(x) = Ax \quad \text{for all } x \in \mathbb{R}^n
$$

The columns of $A$ are the images of the standard basis vectors:

$$
A = \begin{bmatrix} T(e_1) & T(e_2) & \cdots & T(e_n) \end{bmatrix}
$$

where $e_i$ is the $i$-th standard basis vector (1 in position $i$, 0 elsewhere).

### Kernel and Image

**Kernel (Null Space):**

$$
\ker(T) = \{v \in V : T(v) = 0\}
$$

The kernel is a subspace of $V$. It measures how much "information is lost" by $T$. If $\ker(T) = \{0\}$, $T$ is injective (one-to-one).

**Image (Range / Column Space):**

$$
\text{Im}(T) = \{T(v) : v \in V\}
$$

The image is a subspace of $W$. It represents all possible outputs of $T$. If $\text{Im}(T) = W$, $T$ is surjective (onto).

**Rank-Nullity Theorem:**

$$
\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))
$$

### Composition

If $T: U \rightarrow V$ and $S: V \rightarrow W$ are linear transformations with matrices $A$ and $B$, then the composition $S \circ T: U \rightarrow W$ is a linear transformation with matrix $BA$:

$$
(S \circ T)(x) = S(T(x)) = B(Ax) = (BA)x
$$

This is why matrix multiplication is defined the way it is: it represents the composition of linear transformations.

### Invertibility

A linear transformation $T: V \rightarrow W$ is invertible if there exists $T^{-1}: W \rightarrow V$ such that $T^{-1} \circ T = I_V$ and $T \circ T^{-1} = I_W$. This happens iff $\ker(T) = \{0\}$ and $\text{Im}(T) = W$, i.e., $T$ is both injective and surjective (bijective). For a square matrix representation $A$, this is equivalent to $\det(A) \neq 0$.

## Formula(s)

**Linearity condition:**

$$
T(c_1 v_1 + c_2 v_2) = c_1 T(v_1) + c_2 T(v_2)
$$

**Matrix representation:**

$$
T(x) = Ax, \quad A = \begin{bmatrix} T(e_1) & T(e_2) & \cdots & T(e_n) \end{bmatrix}
$$

**Rank-Nullity Theorem:**

$$
\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))
$$

**Composition:**

$$
(S \circ T)(x) = BA x
$$

## Properties

1. $T(0) = 0$ for any linear transformation $T$.
2. $T(-v) = -T(v)$.
3. $T$ preserves linear combinations: $T(\sum c_i v_i) = \sum c_i T(v_i)$.
4. $T$ is injective $\iff \ker(T) = \{0\} \iff$ columns of $A$ are linearly independent.
5. $T$ is surjective $\iff \text{Im}(T) = W \iff$ columns of $A$ span $W$.
6. The composition of two linear transformations is linear.
7. The inverse of an invertible linear transformation is linear.
8. For finite-dimensional spaces with $\dim(V) = \dim(W)$, $T$ is injective $\iff$ $T$ is surjective $\iff$ $T$ is invertible.

## Step-by-Step Worked Examples

### Example 1: Verifying Linearity

Determine whether $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ defined by $T(x, y) = (2x + y, x - 3y)$ is a linear transformation.

**Solution:**

Check additivity: $T(u + v) = T(u) + T(v)$.

Let $u = (x_1, y_1)$, $v = (x_2, y_2)$. Then $u + v = (x_1 + x_2, y_1 + y_2)$.

$$
\begin{aligned}
T(u + v) &= (2(x_1 + x_2) + (y_1 + y_2), (x_1 + x_2) - 3(y_1 + y_2)) \\
&= (2x_1 + 2x_2 + y_1 + y_2, x_1 + x_2 - 3y_1 - 3y_2)
\end{aligned}
$$

$$
\begin{aligned}
T(u) + T(v) &= (2x_1 + y_1, x_1 - 3y_1) + (2x_2 + y_2, x_2 - 3y_2) \\
&= (2x_1 + y_1 + 2x_2 + y_2, x_1 - 3y_1 + x_2 - 3y_2) \\
&= (2x_1 + 2x_2 + y_1 + y_2, x_1 + x_2 - 3y_1 - 3y_2)
\end{aligned}
$$

Therefore $T(u + v) = T(u) + T(v)$. ?

Check homogeneity: $T(cv) = cT(v)$.

Let $v = (x, y)$. Then $cv = (cx, cy)$.

$$
T(cv) = (2(cx) + cy, cx - 3(cy)) = (c(2x + y), c(x - 3y)) = c(2x + y, x - 3y) = cT(v)
$$

?

Both conditions hold, so $T$ is linear.

**Matrix representation:** Find $T(e_1)$ and $T(e_2)$.

$T(e_1) = T(1, 0) = (2(1) + 0, 1 - 3(0)) = (2, 1)$.
$T(e_2) = T(0, 1) = (2(0) + 1, 0 - 3(1)) = (1, -3)$.

So $A = \begin{bmatrix} 2 & 1 \\ 1 & -3 \end{bmatrix}$, and $T(x) = Ax$.

### Example 2: Rotation Transformation

Find the matrix for the linear transformation $R_\theta: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ that rotates vectors counterclockwise by angle $\theta$.

**Solution:**

Find the images of the standard basis vectors under rotation:

$R_\theta(e_1)$: The vector $(1, 0)$ rotated by $\theta$ counterclockwise is $(\cos\theta, \sin\theta)$.

$R_\theta(e_2)$: The vector $(0, 1)$ rotated by $\theta$ counterclockwise is $(-\sin\theta, \cos\theta)$.

Therefore:

$$
R_\theta = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

**Verify:** Apply to $v = (1, 1)$ with $\theta = 90^\circ = \pi/2$:

$$
R_{\pi/2} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

$$
R_{\pi/2}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

The point $(1, 1)$ rotated $90^\circ$ CCW is $(-1, 1)$. ?

### Example 3: Composition of Transformations

Let $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ be a rotation by $90^\circ$ CCW, and $S: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ be reflection across the $y$-axis. Find the matrix for $S \circ T$ and $T \circ S$. Are they equal?

**Solution:**

First, find the individual matrices.

Rotation by $90^\circ$ CCW:

$$
T = \begin{bmatrix} \cos 90^\circ & -\sin 90^\circ \\ \sin 90^\circ & \cos 90^\circ \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

Reflection across the $y$-axis: $(x, y) \mapsto (-x, y)$.

$S(e_1) = S(1, 0) = (-1, 0)$.
$S(e_2) = S(0, 1) = (0, 1)$.

$$
S = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Composition $S \circ T$:** First apply $T$, then $S$.

$$
S \circ T = S T = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

So $(S \circ T)(x, y) = (y, x)$ — reflection across the line $y = x$.

**Composition $T \circ S$:** First apply $S$, then $T$.

$$
T \circ S = T S = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix}
$$

So $(T \circ S)(x, y) = (-y, -x)$ — reflection across the line $y = -x$.

**The compositions are not equal:** $S \circ T \neq T \circ S$. This demonstrates that composition of linear transformations (like matrix multiplication) is not commutative.

### Example 4: Kernel and Image

Find the kernel and image of $T: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ defined by $T(x) = Ax$ where

$$
A = \begin{bmatrix}
1 & 2 & 0 \\
0 & 1 & 1 \\
1 & 3 & 1
\end{bmatrix}
$$

**Solution:**

**Kernel:** Solve $Ax = 0$.

$$
\begin{cases}
x_1 + 2x_2 = 0 \\
x_2 + x_3 = 0 \\
x_1 + 3x_2 + x_3 = 0
\end{cases}
$$

From (2): $x_2 = -x_3$.
From (1): $x_1 = -2x_2 = 2x_3$.
Check (3): $2x_3 + 3(-x_3) + x_3 = 2x_3 - 3x_3 + x_3 = 0$. ?

So $x = (2x_3, -x_3, x_3) = x_3(2, -1, 1)$.

$$
\ker(T) = \text{span}\{(2, -1, 1)\}, \quad \dim(\ker(T)) = 1
$$

**Image:** The columns of $A$:

$$
\text{Im}(T) = \text{span}\{v_1, v_2, v_3\}, \quad v_1 = (1,0,1), v_2 = (2,1,3), v_3 = (0,1,1)
$$

Check if they are independent:

$$
\det(A) = 1(1\cdot1 - 1\cdot3) - 2(0\cdot1 - 1\cdot1) + 0 = 1(-2) - 2(-1) = -2 + 2 = 0
$$

Columns are dependent. Since $\dim(\ker(T)) = 1$, by Rank-Nullity: $\dim(\text{Im}(T)) = 3 - 1 = 2$.

Find a basis: columns 1 and 2 are independent (check: no scalar multiple relation), so:

$$
\text{Im}(T) = \text{span}\{(1,0,1), (2,1,3)\}
$$

**Verify Rank-Nullity:** $\dim(\ker) + \dim(\text{Im}) = 1 + 2 = 3 = \dim(\mathbb{R}^3)$. ?

### Example 5: Projection onto a Line

Find the matrix for the linear transformation $P: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ that projects vectors onto the line $y = 2x$ (i.e., the line through the origin with direction $v = (1, 2)$).

**Solution:**

The projection of vector $u$ onto the line spanned by $v$ is given by:

$$
P(u) = \frac{u \cdot v}{v \cdot v} v
$$

For $v = (1, 2)$, $v \cdot v = 1^2 + 2^2 = 5$.

Find $P(e_1)$:

$$
P(e_1) = P(1, 0) = \frac{(1,0) \cdot (1,2)}{5} (1, 2) = \frac{1}{5}(1, 2) = \left(\frac{1}{5}, \frac{2}{5}\right)
$$

Find $P(e_2)$:

$$
P(e_2) = P(0, 1) = \frac{(0,1) \cdot (1,2)}{5} (1, 2) = \frac{2}{5}(1, 2) = \left(\frac{2}{5}, \frac{4}{5}\right)
$$

Therefore:

$$
P = \begin{bmatrix}
\frac{1}{5} & \frac{2}{5} \\
\frac{2}{5} & \frac{4}{5}
\end{bmatrix} = \frac{1}{5} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$

**Verify:** Project $u = (3, 1)$ onto the line $y = 2x$:

$$
P(u) = \frac{1}{5} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 5 \\ 10 \end{bmatrix} = (1, 2)
$$

The projection of $(3, 1)$ onto $(1, 2)$ is indeed $(1, 2)$. Check using the formula: $\frac{(3,1)\cdot(1,2)}{5}(1,2) = \frac{5}{5}(1,2) = (1,2)$. ?

## Visual Interpretation

Linear transformations in $\mathbb{R}^2$ can be visualized by their effect on a grid or on a shape (like the unit square):

- **Identity:** $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ — leaves the grid unchanged.
- **Scaling:** $\begin{bmatrix} a & 0 \\ 0 & b \end{bmatrix}$ — stretches/shrinks along axes.
- **Rotation:** $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ — rotates the entire plane.
- **Shear (horizontal):** $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ — shifts horizontal lines, keeping vertical lines fixed.
- **Reflection:** $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ — flips across the $y$-axis.
- **Projection:** $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ — projects onto the $x$-axis (collapses $y$ dimension).

Key visual properties:
- The origin always stays fixed.
- Parallel lines remain parallel.
- The image of the unit square is a parallelogram whose area equals $|\det(A)|$.
- A zero determinant means the transformation collapses area (projects onto a lower-dimensional subspace).

## Common Mistakes

1. **Confusing linearity with the ability to factor out constants in one argument:** Students sometimes think $T(cv) = cT(v)$ alone suffices. Both additivity and homogeneity must be checked.

2. **Assuming $T(0) = 0$ is sufficient for linearity:** This is necessary but not sufficient. Many nonlinear functions also satisfy $T(0) = 0$ (e.g., $T(x) = x^2$ in $\mathbb{R}$).

3. **Forgetting that the matrix representation depends on the choice of basis:** The matrix $A$ represents $T$ with respect to specific bases of $V$ and $W$. Changing bases changes $A$ (by similarity transformation for the same space).

4. **Thinking all transformations are linear:** Most functions are not linear. For example, $T(x) = x + 1$ is not linear because $T(0) = 1 \neq 0$. $T(x) = \sin x$ is not linear because $\sin(a + b) \neq \sin a + \sin b$ generally.

5. **Mixing up the order of composition:** $(S \circ T)(x) = S(T(x))$ means apply $T$ first, then $S$. The matrix of $S \circ T$ is $BA$ (note the order reversal).

6. **Confusing kernel (null space) with eigenvectors for eigenvalue 0:** The kernel is exactly the eigenspace for eigenvalue 0, but this connection is often missed. If $Av = 0$ for nonzero $v$, then $v$ is in the kernel and is an eigenvector with eigenvalue 0.

7. **Assuming injectivity and surjectivity are equivalent for all linear transformations:** They are equivalent only when the domain and codomain have the same finite dimension. For $T: \mathbb{R}^m \rightarrow \mathbb{R}^n$ with $m \neq n$, injectivity and surjectivity are independent.

## Interview Questions

### Beginner

1. **Q:** What are the two defining properties of a linear transformation?
   **A:** Additivity: $T(u + v) = T(u) + T(v)$, and homogeneity: $T(cv) = cT(v)$. Together, they imply $T(c_1 v_1 + c_2 v_2) = c_1 T(v_1) + c_2 T(v_2)$.

2. **Q:** Why must every linear transformation satisfy $T(0) = 0$?
   **A:** Because $T(0) = T(0 \cdot v) = 0 \cdot T(v) = 0$ for any $v$, using the homogeneity property.

3. **Q:** How do you find the matrix representing a linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$?
   **A:** The columns of the $m \times n$ matrix are the images of the standard basis vectors: $A = [T(e_1) \; T(e_2) \; \cdots \; T(e_n)]$.

4. **Q:** Give an example of a function $\mathbb{R} \rightarrow \mathbb{R}$ that is not a linear transformation.
   **A:** $T(x) = x^2$ is not linear because $T(1+1) = 4 \neq T(1) + T(1) = 2$. Also $T(x) = \sin x$, $T(x) = e^x$, $T(x) = x + 1$ are not linear.

5. **Q:** What is the matrix for a $90^\circ$ counterclockwise rotation in $\mathbb{R}^2$?
   **A:** $R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$.

### Intermediate

1. **Q:** Explain the rank-nullity theorem for a linear transformation $T: V \rightarrow W$.
   **A:** $\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))$. The dimension of the domain equals the dimension of the kernel plus the dimension of the image.

2. **Q:** If $T: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ has $\ker(T) = \text{span}\{(1, 0, 1)\}$, what is the rank of $T$?
   **A:** $\dim(\ker(T)) = 1$, so by rank-nullity: $\text{rank}(T) = \dim(\text{Im}(T)) = 3 - 1 = 2$.

3. **Q:** How is matrix multiplication related to composition of linear transformations?
   **A:** If $T(x) = Ax$ and $S(x) = Bx$, then $(S \circ T)(x) = S(T(x)) = B(Ax) = (BA)x$. The matrix of the composition is the product of the individual matrices in reverse order.

4. **Q:** Is $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ defined by $T(x, y) = (x^2, y)$ a linear transformation?
   **A:** No. Check additivity: $T((1,0) + (1,0)) = T(2,0) = (4, 0)$ but $T(1,0) + T(1,0) = (1,0) + (1,0) = (2,0)$. Since $4 \neq 2$, additivity fails.

5. **Q:** What is the kernel of a linear transformation geometrically?
   **A:** The kernel is the set of all vectors that map to zero. Geometrically, it is the subspace that "collapses" to a point under $T$. For a projection, the kernel is the direction being projected out. For a rotation, the kernel is $\{0\}$.

### Advanced

1. **Q:** Prove that if $T: V \rightarrow W$ is linear and $\{v_1, \dots, v_n\}$ is linearly dependent, then $\{T(v_1), \dots, T(v_n)\}$ is also linearly dependent. Is the converse true?
   **A:** If $\{v_i\}$ is dependent, there exist $c_i$, not all zero, with $\sum c_i v_i = 0$. Then $T(\sum c_i v_i) = \sum c_i T(v_i) = T(0) = 0$, so $\{T(v_i)\}$ is dependent. The converse is false: $T$ could map independent vectors to dependent ones if $T$ is not injective (e.g., projection onto a line maps two independent vectors in $\mathbb{R}^2$ to dependent vectors on the line).

2. **Q:** In neural networks, a fully connected layer computes $h = \sigma(Wx + b)$. Why is $Wx$ called a linear transformation, and why is the bias $b$ and activation $\sigma$ necessary?
   **A:** $Wx$ is a linear transformation because $W(x_1 + x_2) = Wx_1 + Wx_2$ and $W(cx) = cWx$. The bias $b$ makes it affine (not strictly linear), shifting the output. The activation $\sigma$ introduces nonlinearity, which is essential because composition of linear transformations is still linear — without nonlinearities, a deep network would collapse to a single linear transformation and could not learn nonlinear decision boundaries.

3. **Q:** Let $T: \mathbb{R}^n \rightarrow \mathbb{R}^n$ be a linear transformation with $T^2 = T$ (idempotent, a projection). Show that $V = \ker(T) \oplus \text{Im}(T)$ (direct sum). Interpret this in terms of PCA.
   **A:** For any $v \in V$, write $v = T(v) + (v - T(v))$. Note $T(v) \in \text{Im}(T)$ and $T(v - T(v)) = T(v) - T^2(v) = 0$, so $v - T(v) \in \ker(T)$. Also, if $u \in \ker(T) \cap \text{Im}(T)$, then $u = T(w)$ for some $w$, and $0 = T(u) = T^2(w) = T(w) = u$, so the intersection is $\{0\}$. Hence $V = \ker(T) \oplus \text{Im}(T)$. In PCA, the projection onto the first $k$ principal components is an idempotent linear transformation. The image is the $k$-dimensional subspace capturing maximal variance, and the kernel is the $(n-k)$-dimensional subspace being discarded.

## Practice Problems

### Easy - 5 Questions

1. Determine whether $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$ defined by $T(x, y) = (x + y, 2x - y, x)$ is a linear transformation. If so, find its matrix.

2. Determine whether $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ defined by $T(x, y) = (x + 1, y)$ is a linear transformation.

3. Find the matrix for the linear transformation that reflects vectors across the $x$-axis in $\mathbb{R}^2$.

4. Find the matrix for the linear transformation that scales vectors by a factor of 3 in the $x$-direction and by a factor of 2 in the $y$-direction.

5. If $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ has matrix $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, find $T(2, 3)$.

### Medium - 5 Questions

1. Find the matrix for the linear transformation $T: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ that rotates by $90^\circ$ around the $z$-axis (counterclockwise when looking from above).

2. Determine the kernel and image of $T: \mathbb{R}^3 \rightarrow \mathbb{R}^2$ defined by $T(x, y, z) = (x + y + z, x - y - z)$.

3. Find the matrix for the composition of a $45^\circ$ rotation followed by a reflection across the $y$-axis in $\mathbb{R}^2$.

4. Let $T: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ be projection onto the line $y = 3x$. Find the matrix of $T$.

5. Determine whether $T: P_2 \rightarrow P_2$ (polynomials of degree $\leq 2$) defined by $T(p)(x) = p(x + 1)$ is a linear transformation. If so, find its matrix with respect to the basis $\{1, x, x^2\}$.

### Hard - 3 Questions

1. Let $T: \mathbb{R}^3 \rightarrow \mathbb{R}^3$ be defined by $T(x, y, z) = (x + y, y + z, z + x)$. Find the matrix of $T$, its kernel, its image, and verify the rank-nullity theorem.

2. Prove that a linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is injective if and only if its matrix $A$ has linearly independent columns (full column rank).

3. In the context of machine learning, consider the PCA transformation: given data matrix $X$ (centered), the transformation is $z = W^T x$ where $W$ has orthonormal columns (the principal directions). Show that $W^T W = I$ (the columns are orthonormal) and explain why this makes the transformation an orthogonal projection onto the principal subspace. Also compute the matrix $P = WW^T$ and explain its meaning.

## Solutions

### Easy Solutions

1. Check linearity:

$T(u + v) = ((x_1+x_2)+(y_1+y_2), 2(x_1+x_2)-(y_1+y_2), x_1+x_2)$
$= (x_1+y_1, 2x_1-y_1, x_1) + (x_2+y_2, 2x_2-y_2, x_2) = T(u) + T(v)$ ?

$T(cv) = (cx+cy, 2cx-cy, cx) = c(x+y, 2x-y, x) = cT(v)$ ?

$T$ is linear. Matrix: $T(e_1) = T(1,0) = (1,2,1)$, $T(e_2) = T(0,1) = (1,-1,0)$.

$$
A = \begin{bmatrix} 1 & 1 \\ 2 & -1 \\ 1 & 0 \end{bmatrix}
$$

2. $T(0, 0) = (1, 0) \neq (0, 0)$, so $T$ is not linear.

3. Reflection across $x$-axis: $(x, y) \mapsto (x, -y)$.

$T(e_1) = (1, 0)$, $T(e_2) = (0, -1)$.

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

4. Scaling: $(x, y) \mapsto (3x, 2y)$.

$T(e_1) = (3, 0)$, $T(e_2) = (0, 2)$.

$$
A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}
$$

5. 

$$
T(2, 3) = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 1(2) + 2(3) \\ 3(2) + 4(3) \end{bmatrix} = \begin{bmatrix} 2 + 6 \\ 6 + 12 \end{bmatrix} = \begin{bmatrix} 8 \\ 18 \end{bmatrix}
$$

So $T(2, 3) = (8, 18)$.

### Medium Solutions

1. Rotation around $z$-axis: $x$ and $y$ rotate as in $\mathbb{R}^2$; $z$ stays fixed.

$T(e_1) = T(1, 0, 0) = (\cos 90^\circ, \sin 90^\circ, 0) = (0, 1, 0)$.
$T(e_2) = T(0, 1, 0) = (-\sin 90^\circ, \cos 90^\circ, 0) = (-1, 0, 0)$.
$T(e_3) = T(0, 0, 1) = (0, 0, 1)$.

$$
A = \begin{bmatrix}
0 & -1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

2. Matrix of $T$:

$T(e_1) = T(1, 0, 0) = (1, 1)$.
$T(e_2) = T(0, 1, 0) = (1, -1)$.
$T(e_3) = T(0, 0, 1) = (1, -1)$.

$$
A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -1 & -1 \end{bmatrix}
$$

**Kernel:** Solve $Ax = 0$:

$$
\begin{cases}
x + y + z = 0 \\
x - y - z = 0
\end{cases}
$$

From (2): $x = y + z$. Substitute into (1): $(y+z) + y + z = 2y + 2z = 0 \implies y + z = 0 \implies z = -y$.
Then $x = y + (-y) = 0$.

So $x = (0, y, -y) = y(0, 1, -1)$.

$$
\ker(T) = \text{span}\{(0, 1, -1)\}, \quad \dim(\ker) = 1
$$

**Image:** Columns of $A$ span $\mathbb{R}^2$ (since columns 1 and 2 are independent).

$$
\text{Im}(T) = \mathbb{R}^2, \quad \dim(\text{Im}) = 2
$$

**Rank-Nullity:** $\dim(V) = 3$, $\dim(\ker) + \dim(\text{Im}) = 1 + 2 = 3$. ?

3. Rotation by $45^\circ$:

$$
R = \begin{bmatrix}
\cos 45^\circ & -\sin 45^\circ \\
\sin 45^\circ & \cos 45^\circ
\end{bmatrix} = \frac{\sqrt{2}}{2} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
$$

Reflection across $y$-axis:

$$
S = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Composition (reflect after rotating $= S \circ R$):

$$
SR = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} \cdot \frac{\sqrt{2}}{2} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} = \frac{\sqrt{2}}{2} \begin{bmatrix} -1 & 1 \\ 1 & 1 \end{bmatrix}
$$

4. Projection onto $y = 3x$ (direction $v = (1, 3)$, $v \cdot v = 10$):

$T(e_1) = \frac{(1,0)\cdot(1,3)}{10}(1,3) = \frac{1}{10}(1,3) = \left(\frac{1}{10}, \frac{3}{10}\right)$.
$T(e_2) = \frac{(0,1)\cdot(1,3)}{10}(1,3) = \frac{3}{10}(1,3) = \left(\frac{3}{10}, \frac{9}{10}\right)$.

$$
A = \frac{1}{10} \begin{bmatrix} 1 & 3 \\ 3 & 9 \end{bmatrix}
$$

5. Check linearity: $T(p+q)(x) = (p+q)(x+1) = p(x+1) + q(x+1) = T(p)(x) + T(q)(x)$. ?
$T(cp)(x) = cp(x+1) = cT(p)(x)$. ?

$T$ is linear. Find matrix with respect to $\{1, x, x^2\}$.

$T(1) = 1$ (constant polynomial): coordinates $(1, 0, 0)$.
$T(x) = x + 1$: coordinates $(1, 1, 0)$.
$T(x^2) = (x+1)^2 = x^2 + 2x + 1$: coordinates $(1, 2, 1)$.

$$
A = \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2 \\
0 & 0 & 1
\end{bmatrix}
$$

### Hard Solutions

1. **Matrix:** $T(e_1) = (1, 0, 1)$, $T(e_2) = (1, 1, 0)$, $T(e_3) = (0, 1, 1)$.

$$
A = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 1 \\
1 & 0 & 1
\end{bmatrix}
$$

**Kernel:** Solve $Ax = 0$:

$$
\begin{cases}
x + y = 0 \\
y + z = 0 \\
x + z = 0
\end{cases}
$$

From (1): $x = -y$. From (2): $z = -y$. From (3): $x + z = -y - y = -2y = 0 \implies y = 0$.
Then $x = 0$, $z = 0$. So $\ker(T) = \{(0, 0, 0)\}$, $\dim(\ker) = 0$.

**Image:** Since $\det(A) \neq 0$ (compute: $1(1\cdot1 - 1\cdot0) - 1(0\cdot1 - 1\cdot1) + 0 = 1 + 1 = 2 \neq 0$), the columns span $\mathbb{R}^3$.

$\text{Im}(T) = \mathbb{R}^3$, $\dim(\text{Im}) = 3$.

**Rank-Nullity:** $\dim(V) = 3$, $\dim(\ker) + \dim(\text{Im}) = 0 + 3 = 3$. ?

2. **($\Rightarrow$)** Suppose $T$ is injective. Assume columns of $A$ are dependent. Then there exists $c \neq 0$ such that $Ac = 0$. Then $T(c) = Ac = 0$, so $c \in \ker(T)$. Since $c \neq 0$, $\ker(T) \neq \{0\}$, contradicting injectivity. Therefore columns must be independent.

**($\Leftarrow$)** Suppose columns of $A$ are independent. If $T(x) = 0$, then $Ax = 0$. Since columns are independent, the only solution is $x = 0$. Hence $\ker(T) = \{0\}$, so $T$ is injective.

3. In PCA, $W$ is an $n \times k$ matrix whose columns are the top $k$ eigenvectors of the covariance matrix (principal directions). These eigenvectors are orthonormal, so $W^T W = I_k$ ($k \times k$ identity).

The transformation $z = W^T x$ maps $x \in \mathbb{R}^n$ to $z \in \mathbb{R}^k$. The reconstruction is $\hat{x} = Wz = WW^T x$.

The matrix $P = WW^T$ is an $n \times n$ matrix that satisfies:

- $P^T = (WW^T)^T = WW^T = P$ (symmetric)
- $P^2 = (WW^T)(WW^T) = W(W^TW)W^T = W I_k W^T = WW^T = P$ (idempotent)

So $P$ is an orthogonal projection matrix onto the column space of $W$ (the principal subspace). It projects any vector $x$ onto the $k$-dimensional principal subspace, and $z = W^T x$ gives the coordinates in that subspace.

The projection $Px = WW^T x$ is the best rank-$k$ approximation to $x$ in the least-squares sense, which is exactly what PCA provides.

## Related Concepts

- **Matrix Multiplication:** The algebraic representation of composition of linear transformations. Understanding why matrix multiplication is defined as it is requires understanding composition.
- **Kernel (Null Space):** The set of vectors mapped to zero. Its dimension measures the "information loss" of the transformation.
- **Image (Column Space):** The set of all possible outputs.
- **Rank-Nullity Theorem:** Relates the dimensions of domain, kernel, and image.
- **Eigenvalues and Eigenvectors:** For a linear transformation $T$, eigenvectors are nonzero vectors $v$ such that $T(v) = \lambda v$. These reveal the "natural directions" of the transformation.
- **Change of Basis:** Linear transformations change their matrix representation when the basis changes. This is captured by similarity transformations.
- **Determinant:** The factor by which a linear transformation scales volumes. A zero determinant means the transformation is not invertible.
- **Invertible Linear Transformation:** A bijective linear map with a linear inverse.

## Next Concepts

- **MATH-037 Change of Basis:** How to represent vectors and linear transformations in different coordinate systems.
- **MATH-038 Eigenvalues and Eigenvectors:** The fundamental directions along which a linear transformation acts as a simple scaling.
- **MATH-039 Diagonalization:** Finding a basis in which a linear transformation has a diagonal matrix representation.
- **MATH-040 Singular Value Decomposition (SVD):** A factorization of any linear transformation into rotation-scaling-rotation, revealing its fundamental structure.
- **MATH-041 Linear Transformations in Machine Learning:** Deeper exploration of how linear transformations power PCA, LDA, neural networks, and kernel methods.

## Summary

A linear transformation is a function between vector spaces that preserves addition and scalar multiplication. Every linear transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$ can be represented by an $m \times n$ matrix, where the columns are the images of the standard basis vectors. Key geometric examples include rotations, reflections, scaling, shears, and projections. The kernel and image are subspaces related by the rank-nullity theorem. Composition of linear transformations corresponds to matrix multiplication, explaining why matrix multiplication is defined the way it is. Linear transformations are fundamental to machine learning: neural network layers are compositions of linear and nonlinear transformations; PCA finds an orthogonal linear transformation to a lower-dimensional space; and attention mechanisms use learned linear transformations of queries, keys, and values.

## Key Takeaways

1. A linear transformation satisfies $T(u+v) = T(u) + T(v)$ and $T(cv) = cT(v)$ — it preserves linear combinations.
2. Every linear transformation $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ has a matrix $A$ where column $i$ is $T(e_i)$.
3. The kernel (null space) consists of vectors mapping to zero; the image (range) consists of all outputs.
4. The rank-nullity theorem: $\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))$.
5. Composition of linear transformations corresponds to matrix multiplication: $(S \circ T)(x) = BA x$.
6. Linear transformations are everywhere in AI/ML: neural network layers, PCA, whitening, attention, convolutions, and word embeddings.
