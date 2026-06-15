# Concept: Orthogonal Matrix

## Concept ID

MATH-030

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define an orthogonal matrix and state its defining property $Q^T Q = Q Q^T = I$.
- Show that $Q^{-1} = Q^T$ for an orthogonal matrix.
- Verify that the columns (and rows) of an orthogonal matrix form an orthonormal set.
- Compute $\det(Q) = \pm 1$ and explain what each sign signifies.
- Demonstrate that orthogonal matrices preserve vector lengths and angles.
- Distinguish between rotation and reflection matrices.
- Apply orthogonal matrices in the context of PCA, QR decomposition, and deep learning.

## Prerequisites

- Matrix multiplication and transpose.
- Concept of matrix inverse.
- Dot product and vector norm.
- Determinant of a square matrix.
- [Trace](mathematics/029_trace.md) (MATH-029) — helpful but not required.

## Definition

A square matrix $Q \in \mathbb{R}^{n \times n}$ is called **orthogonal** if its transpose equals its inverse:

$$
Q^T Q = Q Q^T = I_n
$$

where $I_n$ is the $n \times n$ identity matrix. An equivalent and very useful statement is:

$$
Q^{-1} = Q^T
$$

The name comes from the fact that the columns of $Q$ are *orthogonal* to each other and have unit length — they form an *orthonormal* basis. Similarly, the rows also form an orthonormal basis.

**Example:** The matrix

$$
Q = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

is orthogonal because:

$$
Q^T Q = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

## Intuition

An orthogonal matrix represents a transformation that is a "rigid motion" — it rotates or reflects space without stretching, compressing, or shearing it. If you apply an orthogonal matrix to a vector, its length stays the same. If you apply it to a shape, the shape's size and angles remain unchanged.

Think of turning a book on a table: the book's shape and size do not change, only its orientation. That is exactly what an orthogonal transformation does.

The determinant $\det(Q) = +1$ means the transformation is a pure rotation (preserves orientation); $\det(Q) = -1$ means it includes a reflection (flips orientation).

## Why This Concept Matters

Orthogonal matrices are everywhere in mathematics and applications:

- **Numerical linear algebra:** Orthogonal matrices are numerically stable because multiplying by $Q$ does not amplify rounding errors (the condition number is 1).
- **Geometry:** They describe all rotations and reflections in Euclidean space.
- **Data science:** Principal component analysis (PCA), QR decomposition, and singular value decomposition (SVD) all rely on orthogonal matrices.
- **Deep learning:** Orthogonal weight matrices help control gradient flow in very deep networks, preventing vanishing or exploding gradients.
- **Signal processing:** The discrete Fourier transform (DFT) matrix (scaled) is unitary (the complex analog of orthogonal).

## Historical Background

The study of orthogonal transformations dates back to the ancient Greeks, who studied rotations and reflections in plane geometry. The formal matrix formulation emerged in the 19th century with the development of linear algebra by Cayley, Sylvester, and others.

The term "orthogonal matrix" was crystallised in the early 20th century. The group of $n \times n$ orthogonal matrices is called the **orthogonal group**, denoted $O(n)$, and its subgroup with determinant $+1$ is the **special orthogonal group** $SO(n)$ (the rotation group). These groups are fundamental in physics (symmetries of space), robotics (kinematics), and computer graphics (camera rotations).

## Real World Examples

1. **Rigid body rotations in 3D graphics:** Every rotation of a 3D object is represented by an orthogonal matrix with determinant $+1$ (a member of $SO(3)$). Game engines use these to rotate cameras and objects.
2. **Satellite attitude control:** Orthogonal matrices describe the orientation of a satellite relative to a fixed coordinate frame.
3. **Robotics:** The rotation matrix in a robot arm's kinematics is orthogonal. It converts coordinates between the arm's joints and the world frame.
4. **GPS and navigation:** The transformation between different geographic coordinate systems is often an orthogonal matrix (rotation).
5. **Quantum computing:** Quantum gates are unitary matrices — the complex version of orthogonal matrices. For real-valued quantum states, the gates are orthogonal.

## AI/ML Relevance

Orthogonal matrices play several crucial roles in machine learning:

1. **Orthogonal weight initialisation:** In deep neural networks, initialising weight matrices to be (approximately) orthogonal helps prevent vanishing and exploding gradients. For a layer $y = Wx$, if $W$ is orthogonal, then $\|y\| = \|x\|$, so the signal magnitude is preserved through the forward pass. Similarly, during backpropagation, gradients maintain their norm. This is especially important for recurrent neural networks (RNNs) and very deep architectures.

2. **Orthogonal regularisation:** Some training schemes add a penalty term $\|W^T W - I\|_F^2$ to encourage weight matrices to be close to orthogonal, improving gradient flow.

3. **PCA loading matrices:** The principal components in PCA are the eigenvectors of the covariance matrix, which form an orthogonal matrix. This matrix rotates the data to align with the directions of maximum variance.

4. **QR decomposition:** Every real matrix $A$ can be factored as $A = QR$ where $Q$ is orthogonal and $R$ is upper triangular. This is used to solve linear least-squares problems (the workhorse of linear regression) in a numerically stable way.

5. **Householder reflections:** These are orthogonal matrices used to introduce zeros into a matrix, forming the backbone of many stable numerical algorithms (QR, Hessenberg reduction).

6. **Normalising flows:** In generative modelling, normalising flows use orthogonal transformations as building blocks for invertible, volume-preserving transformations between distributions.

## Mathematical Explanation

An orthogonal matrix encodes a linear transformation that preserves the Euclidean inner product. For any vectors $x, y \in \mathbb{R}^n$:

$$
(Qx) \cdot (Qy) = (Qx)^T (Qy) = x^T Q^T Q y = x^T I y = x^T y = x \cdot y
$$

This immediately implies two things:

- **Length preservation:** $\|Qx\| = \|x\|$ for all $x$. (Take $y = x$ in the above.)
- **Angle preservation:** The angle between $Qx$ and $Qy$ equals the angle between $x$ and $y$, since $\cos\theta = \frac{(Qx)\cdot(Qy)}{\|Qx\|\,\|Qy\|} = \frac{x \cdot y}{\|x\|\,\|y\|}$.

**Columns as orthonormal basis:** Write $Q$ in terms of its columns: $Q = [q_1 \; q_2 \; \dots \; q_n]$. Then $Q^T Q = I$ means:

$$
q_i \cdot q_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
$$

So the columns are pairwise orthogonal and each has norm 1. The same holds for the rows.

**Determinant:** From $Q^T Q = I$, taking determinants:

$$
\det(Q^T Q) = \det(I) \implies \det(Q^T)\det(Q) = 1 \implies (\det(Q))^2 = 1 \implies \det(Q) = \pm 1
$$

If $\det(Q) = +1$, $Q$ is a **rotation** (preserves orientation). If $\det(Q) = -1$, $Q$ includes a **reflection** (reverses orientation).

**Eigenvalues:** The eigenvalues of an orthogonal matrix all have absolute value 1. They lie on the unit circle in the complex plane. For real orthogonal matrices, eigenvalues come in complex conjugate pairs $e^{\pm i\theta}$ (rotation angles) plus possibly $\pm 1$.

## Formula(s)

| Formula | Description |
|---------|-------------|
| $Q^T Q = Q Q^T = I$ | Defining property |
| $Q^{-1} = Q^T$ | Inverse equals transpose |
| $q_i \cdot q_j = \delta_{ij}$ | Columns are orthonormal |
| $\|Qx\| = \|x\|$ | Length preservation |
| $(Qx) \cdot (Qy) = x \cdot y$ | Inner product (angle) preservation |
| $\det(Q) = \pm 1$ | Determinant is $\pm 1$ |
| $|\lambda| = 1$ for every eigenvalue | Eigenvalues on unit circle |

## Properties

- **Product of orthogonal matrices is orthogonal:** If $Q_1$ and $Q_2$ are orthogonal, then $Q_1 Q_2$ is orthogonal because $(Q_1 Q_2)^T (Q_1 Q_2) = Q_2^T Q_1^T Q_1 Q_2 = Q_2^T I Q_2 = I$.
- **Inverse is orthogonal:** $(Q^{-1})^T (Q^{-1}) = (Q^T)^T Q^T = Q Q^T = I$.
- **Transpose is orthogonal:** $(Q^T)^T Q^T = Q Q^T = I$.
- **Condition number is 1:** This means orthogonal matrices are perfectly well-conditioned — they do not amplify numerical errors.
- **Block diagonal form over $\mathbb{R}^2$:** Any real orthogonal matrix can be block-diagonalised into $1 \times 1$ blocks of $\pm 1$ and $2 \times 2$ rotation blocks $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.
- **The set of all $n \times n$ orthogonal matrices forms a group:** $O(n)$.

## Step-by-Step Worked Examples

### Example 1: Verifying orthogonality

Determine whether the following matrix is orthogonal:

$$
Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
$$

**Solution:**

Check $Q^T Q$:

$$
Q^T = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
$$

$$
Q^T Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
\begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{pmatrix}
$$

Entry (1,1): $\frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = \frac{1}{2} + \frac{1}{2} = 1$

Entry (1,2): $\frac{1}{\sqrt{2}}\cdot\left(-\frac{1}{\sqrt{2}}\right) + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = -\frac{1}{2} + \frac{1}{2} = 0$

Entry (2,1): $-\frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = -\frac{1}{2} + \frac{1}{2} = 0$

Entry (2,2): $-\frac{1}{\sqrt{2}}\cdot\left(-\frac{1}{\sqrt{2}}\right) + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = \frac{1}{2} + \frac{1}{2} = 1$

So $Q^T Q = I$. Also check columns:

Column 1: $\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$, norm = $\sqrt{\frac{1}{2} + \frac{1}{2}} = 1$.

Column 2: $\left(-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$, norm = $\sqrt{\frac{1}{2} + \frac{1}{2}} = 1$.

Dot product of columns: $\frac{1}{\sqrt{2}}\left(-\frac{1}{\sqrt{2}}\right) + \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} = -\frac{1}{2} + \frac{1}{2} = 0$.

Thus $Q$ is orthogonal. $\det(Q) = \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}} - \left(-\frac{1}{\sqrt{2}}\right)\cdot\frac{1}{\sqrt{2}} = \frac{1}{2} + \frac{1}{2} = 1$, so this is a rotation by $45^\circ$.

### Example 2: Length preservation

Let $Q = \frac{1}{3}\begin{pmatrix} 2 & 1 & 2 \\ -2 & 2 & 1 \\ -1 & -2 & 2 \end{pmatrix}$ (verify this is orthogonal — all columns have norm 1 and are pairwise orthogonal).

Let $x = (1, 2, 3)^T$. Compute $\|x\|$ and $\|Qx\|$ and verify they are equal.

**Solution:**

$\|x\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14}$

$$
Qx = \frac{1}{3} \begin{pmatrix}
2(1) + 1(2) + 2(3) \\
-2(1) + 2(2) + 1(3) \\
-1(1) - 2(2) + 2(3)
\end{pmatrix}
= \frac{1}{3} \begin{pmatrix}
2 + 2 + 6 \\
-2 + 4 + 3 \\
-1 - 4 + 6
\end{pmatrix}
= \frac{1}{3} \begin{pmatrix}
10 \\
5 \\
1
\end{pmatrix}
$$

$\|Qx\| = \frac{1}{3}\sqrt{10^2 + 5^2 + 1^2} = \frac{1}{3}\sqrt{100 + 25 + 1} = \frac{1}{3}\sqrt{126} = \frac{\sqrt{126}}{3}$

Simplify: $\sqrt{126} = \sqrt{9 \cdot 14} = 3\sqrt{14}$, so $\|Qx\| = \frac{3\sqrt{14}}{3} = \sqrt{14}$.

Thus $\|Qx\| = \sqrt{14} = \|x\|$. $\checkmark$

### Example 3: Recognising a reflection (Householder matrix)

The Householder reflection matrix is defined as:

$$
Q = I - 2\frac{v v^T}{v^T v}
$$

for a non-zero vector $v \in \mathbb{R}^n$. Show that $Q$ is orthogonal and symmetric, and find $\det(Q)$.

**Solution:**

First, note $Q$ is symmetric: $Q^T = I^T - 2\frac{(v v^T)^T}{v^T v} = I - 2\frac{v v^T}{v^T v} = Q$.

Now check orthogonality:

$$
Q^T Q = Q^2 = \left(I - 2\frac{v v^T}{v^T v}\right)^2
$$

Let $u = \frac{v}{\|v\|}$, so $v v^T = \|v\|^2 u u^T$ and $v^T v = \|v\|^2$. Then $Q = I - 2 u u^T$.

$$
Q^2 = (I - 2uu^T)(I - 2uu^T) = I - 2uu^T - 2uu^T + 4 u u^T u u^T
$$

Since $u^T u = 1$, we have $u u^T u u^T = u (u^T u) u^T = u (1) u^T = u u^T$.

So $Q^2 = I - 4uu^T + 4uu^T = I$.

Thus $Q^T Q = Q^2 = I$, so $Q$ is orthogonal.

For a specific example, let $v = (3, 4)^T$. Then $v^T v = 25$.

$$
Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \frac{2}{25} \begin{pmatrix} 9 & 12 \\ 12 & 16 \end{pmatrix}
= \begin{pmatrix} 1 - \frac{18}{25} & 0 - \frac{24}{25} \\ 0 - \frac{24}{25} & 1 - \frac{32}{25} \end{pmatrix}
= \begin{pmatrix} \frac{7}{25} & -\frac{24}{25} \\ -\frac{24}{25} & -\frac{7}{25} \end{pmatrix}
$$

Check: $\det(Q) = \left(\frac{7}{25}\right)\left(-\frac{7}{25}\right) - \left(-\frac{24}{25}\right)\left(-\frac{24}{25}\right) = -\frac{49}{625} - \frac{576}{625} = -\frac{625}{625} = -1$.

This confirms it is a reflection. In general, every Householder reflection has $\det(Q) = -1$.

### Example 4: Orthogonal matrix from PCA

Suppose the covariance matrix of a dataset is:

$$
S = \begin{pmatrix} 2.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix}
$$

Find the eigenvectors (principal components) and verify they form an orthogonal matrix.

**Solution:**

Eigenvalues:
$$
\det(S - \lambda I) = \det\begin{pmatrix} 2.5-\lambda & 0.5 \\ 0.5 & 0.5-\lambda \end{pmatrix}
= (2.5-\lambda)(0.5-\lambda) - 0.25
= \lambda^2 - 3\lambda + 1.25 - 0.25 = \lambda^2 - 3\lambda + 1
$$

So $\lambda = \frac{3 \pm \sqrt{9 - 4}}{2} = \frac{3 \pm \sqrt{5}}{2}$, giving $\lambda_1 \approx 2.618$, $\lambda_2 \approx 0.382$.

For $\lambda_1 = 2.618$:
$$
(S - 2.618 I) = \begin{pmatrix} -0.118 & 0.5 \\ 0.5 & -2.118 \end{pmatrix}
$$
$-0.118 v_1 + 0.5 v_2 = 0 \implies v_2 = 0.236 v_1$. So $v_1 \approx (1, 0.236)^T$. Normalise: $\|v_1\| = \sqrt{1 + 0.0557} \approx 1.027$, so $q_1 \approx (0.973, 0.230)^T$.

For $\lambda_2 = 0.382$:
$$
(S - 0.382 I) = \begin{pmatrix} 2.118 & 0.5 \\ 0.5 & 0.118 \end{pmatrix}
$$
$2.118 v_1 + 0.5 v_2 = 0 \implies v_2 = -4.236 v_1$. So $v_2 \approx (1, -4.236)^T$. Normalise: $\|v_2\| = \sqrt{1 + 17.94} \approx 4.352$, so $q_2 \approx (0.230, -0.973)^T$.

Check orthogonality: $q_1 \cdot q_2 \approx 0.973(0.230) + 0.230(-0.973) = 0.224 - 0.224 = 0$.

The matrix $Q = [q_1 \; q_2] = \begin{pmatrix} 0.973 & 0.230 \\ 0.230 & -0.973 \end{pmatrix}$ is orthogonal (approximately).

Check: $Q^T Q \approx \begin{pmatrix} 0.973^2 + 0.230^2 & 0.973(0.230) + 0.230(-0.973) \\ 0.230(0.973) + (-0.973)(0.230) & 0.230^2 + (-0.973)^2 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$.

### Example 5: Orthogonal weight matrix for gradient preservation

Consider a simple 2-layer neural network with orthogonal weight matrices. Let:

$$
W_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix},
\qquad
W_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}
$$

Verify both are orthogonal. Then for an input $x = (3, 4)^T$, show that the forward pass through one layer preserves the norm.

**Solution:**

Check $W_1$:
$$
W_1^T W_1 = \frac{1}{2} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
= \frac{1}{2} \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I
$$

Similarly for $W_2$:
$$
W_2^T W_2 = \frac{1}{2} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}
= \frac{1}{2} \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I
$$

Both are orthogonal (in fact $W_1 = W_2^T$, and $W_1$ is a rotation by $45^\circ$ with determinant $+1$).

Forward pass: $y = W_1 x = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix}
= \frac{1}{\sqrt{2}} \begin{pmatrix} 7 \\ -1 \end{pmatrix}$.

$\|x\| = \sqrt{9 + 16} = \sqrt{25} = 5$
$\|y\| = \frac{1}{\sqrt{2}}\sqrt{49 + 1} = \frac{\sqrt{50}}{\sqrt{2}} = \sqrt{25} = 5$ $\checkmark$

The norm is preserved. In a deep network with $L$ orthogonal layers, $x^{(L)}$ would have the same norm as $x^{(0)}$, preventing activations from exploding or vanishing.

## Visual Interpretation

A $2 \times 2$ orthogonal matrix $Q = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ rotates the plane by angle $\theta$. The columns are two perpendicular unit vectors that have been rotated: $(\cos\theta, \sin\theta)$ and $(-\sin\theta, \cos\theta)$.

If $\det(Q) = -1$, the matrix also flips orientation (like a mirror). For example, $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ reflects across the x-axis.

In 3D, an orthogonal matrix rotates around some axis (if $\det = +1$) or includes a reflection (if $\det = -1$). The eigenvectors corresponding to eigenvalue $+1$ point along the axis of rotation.

Picture applying an orthogonal matrix to a grid of points: the grid rotates or reflects but maintains all original distances and angles — a square remains a square of the same size, not squashed into a rectangle.

## Common Mistakes

1. **Thinking every matrix with orthonormal columns is automatically square and orthogonal.** A matrix with orthonormal columns that is not square satisfies $Q^T Q = I$ but not $Q Q^T = I$. Such a matrix is called *column-orthogonal* or *semi-orthogonal*, but it is not a true orthogonal matrix (it is not even invertible in the usual sense).

2. **Confusing orthogonal with diagonal.** An orthogonal matrix is generally not diagonal. In fact, the only diagonal orthogonal matrices are those with $\pm 1$ on the diagonal.

3. **Assuming $\det(Q) = 1$ for all orthogonal matrices.** The determinant can be $-1$ (reflection). Only matrices in $SO(n)$ have $\det = +1$.

4. **Forgetting that $Q$ must be square.** The term "orthogonal matrix" is reserved for square matrices. A rectangular matrix with orthonormal columns is not called orthogonal.

5. **Thinking orthogonal means entries are integers.** Most orthogonal matrices contain irrational entries (e.g., $\frac{1}{\sqrt{2}}$). The only orthogonal matrices with all integer entries are permutation matrices (entries $0$ or $1$).

## Interview Questions

### Beginner

**Q1:** What is the defining property of an orthogonal matrix?
**A1:** $Q^T Q = Q Q^T = I$, or equivalently $Q^{-1} = Q^T$.

**Q2:** What are the possible values of $\det(Q)$ for an orthogonal matrix $Q$?
**A2:** $\det(Q) = \pm 1$.

**Q3:** Is $I_3$ (the $3 \times 3$ identity matrix) orthogonal?
**A3:** Yes, because $I^T I = I I = I$.

**Q4:** If $Q$ is orthogonal, what is $\|Qx\|$ in terms of $\|x\|$?
**A4:** $\|Qx\| = \|x\|$ — orthogonal matrices preserve vector lengths.

**Q5:** Give an example of a $2 \times 2$ orthogonal matrix with determinant $-1$.
**A5:** The reflection matrix $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

### Intermediate

**Q1:** Prove that the product of two orthogonal matrices is orthogonal.
**A1:** If $Q_1^T Q_1 = I$ and $Q_2^T Q_2 = I$, then $(Q_1 Q_2)^T (Q_1 Q_2) = Q_2^T Q_1^T Q_1 Q_2 = Q_2^T I Q_2 = I$.

**Q2:** Show that the eigenvalues of an orthogonal matrix have absolute value 1.
**A2:** If $Qx = \lambda x$, then $\|Qx\| = |\lambda| \|x\|$. But $\|Qx\| = \|x\|$, so $|\lambda| = 1$.

**Q3:** What is the difference between $O(n)$ and $SO(n)$?
**A3:** $O(n)$ is the group of all $n \times n$ orthogonal matrices (determinant $\pm 1$). $SO(n)$ is the subgroup with determinant $+1$ (pure rotations).

**Q4:** Explain why orthogonal initialisation helps in deep learning.
**A4:** Orthogonal matrices preserve norm, so activations and gradients do not explode or vanish as they pass through layers. This stabilises training in very deep networks.

**Q5:** How do you construct a $2 \times 2$ rotation matrix for a given angle $\theta$?
**A5:** $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

### Advanced

**Q1:** Prove that any real orthogonal matrix can be block-diagonalised into $1 \times 1$ blocks of $\pm 1$ and $2 \times 2$ rotation blocks.
**A1:** The real Schur decomposition of a real orthogonal matrix yields a quasi-triangular form with $1 \times 1$ blocks (real eigenvalues $\pm 1$) and $2 \times 2$ blocks (complex conjugate eigenvalue pairs $e^{\pm i\theta}$). Since the matrix is orthogonal, the $2 \times 2$ blocks must themselves be orthogonal, hence each is a rotation $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.

**Q2:** Derive the Householder reflection matrix $H = I - 2uu^T$ and show it reflects a vector $x$ across the hyperplane orthogonal to $u$.
**A2:** For any vector $x$, write $x = (x \cdot u)u + (x - (x \cdot u)u)$ (parallel and perpendicular to $u$). Then $Hx = x - 2(x \cdot u)u = -(x \cdot u)u + (x - (x \cdot u)u)$. The component parallel to $u$ is negated; the perpendicular component is unchanged. This is exactly a reflection across the hyperplane orthogonal to $u$.

**Q3:** In PCA, the principal component matrix is orthogonal. Explain what happens geometrically when you multiply the centred data by the transpose of this orthogonal matrix.
**A3:** If $X$ is the centred data matrix and $Q$ contains the eigenvectors as columns, then $Z = X Q$ rotates the data into the principal component space. Since $Q$ is orthogonal, distances are preserved. The transformed data $Z$ has uncorrelated features (the covariance of $Z$ is diagonal), and the features are ordered by decreasing variance.

## Practice Problems

### Easy

**E1:** Determine whether $Q = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ is orthogonal.

**E2:** Compute $\det(Q)$ for $Q = \begin{pmatrix} \frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2} \end{pmatrix}$ and state whether it is a rotation or reflection.

**E3:** If $Q$ is a $4 \times 4$ orthogonal matrix, what is $Q^{-1}$ equal to?

**E4:** Let $Q = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$. Check if it is orthogonal.

**E5:** Given $Q$ orthogonal and $x = (1, 0)^T$, what is $\|Qx\|$?

### Medium

**M1:** Find the angle of rotation for $Q = \begin{pmatrix} \frac{1}{2} & -\frac{\sqrt{3}}{2} \\ \frac{\sqrt{3}}{2} & \frac{1}{2} \end{pmatrix}$.

**M2:** Construct a $3 \times 3$ orthogonal matrix where the first column is $\left(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^T$.

**M3:** For $Q = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ and $x = (2, 3)^T$, compute $\|Qx\|$ and compare to $\|x\|$.

**M4:** Show that $Q = \begin{pmatrix} \cos\theta & \sin\theta \\ \sin\theta & -\cos\theta \end{pmatrix}$ is orthogonal for any $\theta$. What type of transformation is it?

**M5:** Let $Q$ be orthogonal. Prove that $Q^T$ is also orthogonal.

### Hard

**H1:** Prove that if $\lambda$ is an eigenvalue of an orthogonal matrix, then $|\lambda| = 1$, and if $\lambda$ is real, then $\lambda = \pm 1$.

**H2:** Let $Q$ be an $n \times n$ orthogonal matrix with $\det(Q) = -1$. Show that $\lambda = -1$ is an eigenvalue of $Q$. (Hint: consider $\det(Q + I)$.)

**H3:** Construct a $3 \times 3$ rotation matrix that rotates by $30^\circ$ around the axis $(1, 1, 1)^T$ (the line $x = y = z$).

## Solutions

### Easy Solutions

**E1:**
$$
Q^T Q = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$
Yes, $Q$ is orthogonal. $\det(Q) = 0(0) - (-1)(1) = 1$, so it is a rotation by $90^\circ$.

**E2:**
$\det(Q) = \frac{\sqrt{3}}{2} \cdot \frac{\sqrt{3}}{2} - \left(-\frac{1}{2}\right)\left(\frac{1}{2}\right) = \frac{3}{4} + \frac{1}{4} = 1$.
This is a rotation (determinant $+1$).

**E3:** $Q^{-1} = Q^T$.

**E4:**
$Q^T$ is the same matrix (but transposed). In fact $Q$ is a permutation matrix (it cycles the rows: row 1 goes to row 3, etc.). Permutation matrices are orthogonal because $Q^T Q = I$. Yes, it is orthogonal.

**E5:** $\|Qx\| = \|x\| = 1$ (since $x = (1,0)^T$ has norm 1).

### Medium Solutions

**M1:**
Comparing with the standard rotation matrix:
$$
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
= \begin{pmatrix} \frac{1}{2} & -\frac{\sqrt{3}}{2} \\ \frac{\sqrt{3}}{2} & \frac{1}{2} \end{pmatrix}
$$
So $\cos\theta = \frac{1}{2}$ and $\sin\theta = \frac{\sqrt{3}}{2}$. Therefore $\theta = 60^\circ$ (or $\pi/3$ radians).

**M2:**
We need two more vectors orthonormal to $q_1 = \left(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^T$.

Choose $q_2 = \left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}, 0\right)^T$. Check: $q_1 \cdot q_2 = \frac{1}{\sqrt{6}} - \frac{1}{\sqrt{6}} + 0 = 0$, and $\|q_2\| = 1$.

For $q_3$, compute the cross product $q_1 \times q_2$:

$$
q_3 = q_1 \times q_2 = \det\begin{pmatrix}
i & j & k \\
\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0
\end{pmatrix}
$$

$$
= i\left(\frac{1}{\sqrt{3}}\cdot 0 - \frac{1}{\sqrt{3}}\left(-\frac{1}{\sqrt{2}}\right)\right)
- j\left(\frac{1}{\sqrt{3}}\cdot 0 - \frac{1}{\sqrt{3}}\cdot\frac{1}{\sqrt{2}}\right)
+ k\left(\frac{1}{\sqrt{3}}\left(-\frac{1}{\sqrt{2}}\right) - \frac{1}{\sqrt{3}}\cdot\frac{1}{\sqrt{2}}\right)
$$

$$
= i\left(\frac{1}{\sqrt{6}}\right) - j\left(-\frac{1}{\sqrt{6}}\right) + k\left(-\frac{2}{\sqrt{6}}\right)
= \left(\frac{1}{\sqrt{6}}, \frac{1}{\sqrt{6}}, -\frac{2}{\sqrt{6}}\right)^T
$$

Check $\|q_3\| = \sqrt{\frac{1}{6} + \frac{1}{6} + \frac{4}{6}} = \sqrt{1} = 1$.

So $Q = \begin{pmatrix}
\frac{1}{\sqrt{3}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \\
\frac{1}{\sqrt{3}} & 0 & -\frac{2}{\sqrt{6}}
\end{pmatrix}$ is orthogonal.

**M3:**
$\|x\| = \sqrt{4 + 9} = \sqrt{13}$.

$$
Qx = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \end{pmatrix}
= \frac{1}{\sqrt{2}} \begin{pmatrix} 5 \\ -1 \end{pmatrix}
$$

$\|Qx\| = \frac{1}{\sqrt{2}}\sqrt{25 + 1} = \frac{\sqrt{26}}{\sqrt{2}} = \sqrt{13}$.

So $\|Qx\| = \|x\| = \sqrt{13}$. $\checkmark$

**M4:**
$$
Q^T Q = \begin{pmatrix} \cos\theta & \sin\theta \\ \sin\theta & -\cos\theta \end{pmatrix}
\begin{pmatrix} \cos\theta & \sin\theta \\ \sin\theta & -\cos\theta \end{pmatrix}
= \begin{pmatrix}
\cos^2\theta + \sin^2\theta & \cos\theta\sin\theta - \sin\theta\cos\theta \\
\sin\theta\cos\theta - \cos\theta\sin\theta & \sin^2\theta + \cos^2\theta
\end{pmatrix}
= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

$\det(Q) = -\cos^2\theta - \sin^2\theta = -1$. So this is a reflection (specifically a reflection across a line through the origin at angle $\theta/2$).

**M5:**
$(Q^T)^T (Q^T) = Q Q^T = I$ (since $Q$ is orthogonal). Therefore $Q^T$ is orthogonal.

### Hard Solutions

**H1:**
Let $Qx = \lambda x$ with $x \neq 0$. Then $x^* Q^T Q x = x^* I x = x^* x = \|x\|^2$.

But also $x^* Q^T Q x = (Qx)^* (Qx) = (\lambda x)^* (\lambda x) = |\lambda|^2 \|x\|^2$.

Thus $|\lambda|^2 \|x\|^2 = \|x\|^2$, so $|\lambda|^2 = 1$, hence $|\lambda| = 1$.

If $\lambda$ is real, then $|\lambda| = 1$ implies $\lambda = \pm 1$.

**H2:**
Consider $\det(Q + I)$:
$$
\det(Q + I) = \det(Q + Q Q^T) \quad \text{(since } Q Q^T = I\text{)}
$$
$$
= \det(Q(I + Q^T)) = \det(Q) \det(I + Q^T) = \det(Q) \det((I + Q)^T) = \det(Q) \det(I + Q)
$$

So $\det(I + Q) = \det(Q) \det(I + Q)$.

If $\det(Q) = -1$, then $\det(I + Q) = -\det(I + Q)$, which implies $\det(I + Q) = 0$.

Thus $I + Q$ is singular, meaning there exists a nonzero $x$ such that $(I + Q)x = 0$, i.e., $Qx = -x$. Therefore $\lambda = -1$ is an eigenvalue.

**H3:**
The axis direction is $u = \left(\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right)^T$. The rotation angle is $\theta = 30^\circ$.

We use Rodrigues' rotation formula:
$$
R = I + (\sin\theta) K + (1 - \cos\theta) K^2
$$

where $K$ is the cross-product matrix for $u$:

$$
K = \begin{pmatrix}
0 & -u_3 & u_2 \\
u_3 & 0 & -u_1 \\
-u_2 & u_1 & 0
\end{pmatrix}
= \frac{1}{\sqrt{3}} \begin{pmatrix}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{pmatrix}
$$

$K^2 = \frac{1}{3} \begin{pmatrix}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{pmatrix}^2
= \frac{1}{3} \begin{pmatrix}
-2 & 1 & 1 \\
1 & -2 & 1 \\
1 & 1 & -2
\end{pmatrix}$

With $\theta = 30^\circ$, $\sin\theta = \frac{1}{2}$, $\cos\theta = \frac{\sqrt{3}}{2}$, $1 - \cos\theta = 1 - \frac{\sqrt{3}}{2}$.

$$
R = I + \frac{1}{2}K + \left(1 - \frac{\sqrt{3}}{2}\right)K^2
$$

$$
= I + \frac{1}{2\sqrt{3}} \begin{pmatrix}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{pmatrix}
+ \frac{1 - \frac{\sqrt{3}}{2}}{3} \begin{pmatrix}
-2 & 1 & 1 \\
1 & -2 & 1 \\
1 & 1 & -2
\end{pmatrix}
$$

Let $a = \frac{1}{2\sqrt{3}}$ and $b = \frac{1 - \frac{\sqrt{3}}{2}}{3}$:

$$
R = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
+ a \begin{pmatrix}
0 & -1 & 1 \\
1 & 0 & -1 \\
-1 & 1 & 0
\end{pmatrix}
+ b \begin{pmatrix}
-2 & 1 & 1 \\
1 & -2 & 1 \\
1 & 1 & -2
\end{pmatrix}
$$

$$
R = \begin{pmatrix}
1 - 2b & -a + b & a + b \\
a + b & 1 - 2b & -a + b \\
-a + b & a + b & 1 - 2b
\end{pmatrix}
$$

Numerically: $a \approx 0.2887$, $b \approx \frac{1 - 0.8660}{3} \approx 0.0447$.

So $1 - 2b \approx 0.9106$, $-a + b \approx -0.2440$, $a + b \approx 0.3334$.

$$
R \approx \begin{pmatrix}
0.9106 & -0.2440 & 0.3334 \\
0.3334 & 0.9106 & -0.2440 \\
-0.2440 & 0.3334 & 0.9106
\end{pmatrix}
$$

One can verify $R^T R \approx I$ and $\det(R) \approx 1$.

## Related Concepts

- **Determinant** — Orthogonal matrices have determinant $\pm 1$.
- **Eigenvalues** — Eigenvalues of orthogonal matrices lie on the unit circle.
- **Trace** — The trace of a $2 \times 2$ rotation matrix is $2\cos\theta$.
- **Change of basis** — Orthogonal matrices represent a change between orthonormal bases.
- **Unitary Matrix** — The complex analogue of an orthogonal matrix ($U^* U = I$).
- **QR Decomposition** — Factorises any matrix into an orthogonal part and a triangular part.

## Next Concepts

- **Matrix Diagonalisation** — Orthogonal matrices can diagonalise symmetric matrices.
- **Singular Value Decomposition** — Uses orthogonal matrices to factor any matrix.
- **Principal Component Analysis** — The loadings matrix is orthogonal.
- **Gram-Schmidt Process** — Constructs an orthogonal basis from any set of vectors.

## Summary

An orthogonal matrix $Q$ satisfies $Q^T Q = Q Q^T = I$, making $Q^{-1} = Q^T$. Its columns form an orthonormal basis, and multiplying by $Q$ preserves lengths and angles. The determinant is $\pm 1$: $+1$ for rotations, $-1$ for reflections. Orthogonal matrices are numerically stable, fundamental to PCA and QR decomposition, and used in deep learning to initialise weights and control gradient flow.

## Key Takeaways

- $Q^T Q = Q Q^T = I$ and $Q^{-1} = Q^T$ — the defining property.
- Columns (and rows) form an orthonormal basis: $q_i \cdot q_j = \delta_{ij}$.
- $\|Qx\| = \|x\|$ and $(Qx)\cdot(Qy) = x \cdot y$ — orthogonal matrices preserve lengths and angles.
- $\det(Q) = \pm 1$: $+1$ = rotation, $-1$ = reflection.
- The product of orthogonal matrices is orthogonal; $O(n)$ is a group.
- Orthogonal matrices are perfectly conditioned (condition number $= 1$).
- In ML, orthogonal weight initialisation prevents vanishing/exploding gradients in deep networks.
- PCA loading matrices and Householder reflections are orthogonal.
- QR decomposition factorises $A = QR$ with $Q$ orthogonal and $R$ upper triangular.
