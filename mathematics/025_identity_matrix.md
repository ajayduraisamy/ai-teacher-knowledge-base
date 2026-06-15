# Concept: Identity Matrix

## Concept ID

MATH-025

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define the identity matrix $I_n$ and describe its structure
- Understand the multiplicative property $AI = IA = A$
- Relate the identity matrix to the concept of matrix inverses
- Recognise the role of the identity matrix in deep learning architectures and numerical computation

## Prerequisites

- Basic matrix notation and multiplication
- Familiarity with the transpose (MATH-024)

## Definition

The **identity matrix** of size $n$, denoted $I_n$ (or simply $I$ when the size is clear from context), is the $n \times n$ square matrix with **1s on the main diagonal** and **0s everywhere else**:

$$
I_n = \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{pmatrix}
$$

The entry at position $(i, j)$ is given by the Kronecker delta: $(I_n)_{ij} = \delta_{ij}$, where $\delta_{ij} = 1$ if $i = j$ and $0$ otherwise.

## Intuition

The identity matrix is the matrix analogue of the number **1** in ordinary arithmetic. Just as multiplying any number by 1 leaves it unchanged ($x \cdot 1 = 1 \cdot x = x$), multiplying any compatible matrix by the identity leaves it unchanged:

$$
A I = A \quad \text{and} \quad I A = A
$$

Think of the identity as the "do nothing" transformation. When you apply the identity matrix to a vector, every component stays exactly where it is — it is the matrix version of "what you put in is what you get out."

## Why This Concept Matters

The identity matrix plays a foundational role in linear algebra. Without it, we could not define matrix inverses, solve linear systems, or discuss matrix properties like orthogonality. In applied fields, the identity appears in regularisation (ridge regression adds $\lambda I$ to $X^T X$), neural network architectures (residual connections), and numerical stability techniques.

## Historical Background

The concept of the identity matrix emerged naturally alongside matrix multiplication in the 19th century. Arthur Cayley, in his 1858 paper "A Memoir on the Theory of Matrices," recognised that there exists a matrix that acts as a multiplicative identity, analogous to the number 1. The notation $I_n$ became standard in the 20th century as matrix algebra was formalised.

## Real World Examples

- **Computer graphics**: The identity matrix represents the default transformation — no rotation, scaling, or translation has been applied. Graphics pipelines start with the identity and accumulate transformations on top of it.
- **Spreadsheets**: A correlation matrix with 1s on the diagonal and 0s off-diagonal represents completely uncorrelated variables.
- **Robotics**: When a robot arm is in its "home" position, its transformation matrix is the identity — no movement has occurred.

## AI/ML Relevance

The identity matrix is deeply woven into machine learning:

- **Residual connections (ResNets)**: The skip connection in a residual block is $y = F(x) + x$. The "pass-through" of $x$ is equivalent to adding an identity mapping. In formula terms, the derivative of a residual layer includes an identity term $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$, which helps prevent vanishing gradients in very deep networks.

- **Weight initialisation**: When initialising neural network weights, we often want the initial transformation to be close to the identity so that activations are not scaled up or down dramatically. For example, orthogonal initialisation starts with a matrix close to $I$.

- **Ridge regression (L2 regularisation)**: The normal equations become $\hat{w} = (X^T X + \lambda I)^{-1} X^T y$. Adding $\lambda I$ ensures the matrix $X^T X + \lambda I$ is always invertible even if $X^T X$ is singular.

- **Batch normalisation**: The initial scale and shift parameters are often set to $\gamma = 1$ and $\beta = 0$, effectively making the initial normalisation the identity transformation.

- **Optimisation**: The identity is used in Levenberg–Marquardt and trust-region methods where the Hessian approximation is regularised with $\mu I$.

## Mathematical Explanation

The identity matrix is characterised entirely by the property that for any $m \times n$ matrix $A$:

$$
A I_n = A \quad \text{and} \quad I_m A = A
$$

Let us verify this for a $2 \times 3$ matrix:

$$
A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix}, \quad I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

Computing $A I_3$:

$$
A I_3 = \begin{pmatrix}
a_{11}\cdot 1 + a_{12}\cdot 0 + a_{13}\cdot 0 & a_{11}\cdot 0 + a_{12}\cdot 1 + a_{13}\cdot 0 & a_{11}\cdot 0 + a_{12}\cdot 0 + a_{13}\cdot 1 \\
a_{21}\cdot 1 + a_{22}\cdot 0 + a_{23}\cdot 0 & a_{21}\cdot 0 + a_{22}\cdot 1 + a_{23}\cdot 0 & a_{21}\cdot 0 + a_{22}\cdot 0 + a_{23}\cdot 1
\end{pmatrix}
= \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix} = A
$$

Each column of $A$ is multiplied by the corresponding column of $I$, which picks out exactly one entry and leaves the rest untouched. This works because the identity has exactly one 1 per column (and per row).

## Formula(s)

**Definition by Kronecker delta**:

$$
(I_n)_{ij} = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
$$

**Multiplicative identity property**:

$$
A I_n = A \quad (\text{for } A \text{ of size } m \times n)
$$

$$
I_m A = A \quad (\text{for } A \text{ of size } m \times n)
$$

**Relation to inverse**:

$$
A A^{-1} = A^{-1} A = I
$$

## Properties

1. **Uniqueness**: For each size $n$, there is exactly one identity matrix $I_n$.

2. **Squareness**: The identity matrix is always square ($n \times n$).

3. **Symmetry**: $I_n^T = I_n$ — the identity matrix is symmetric.

4. **Idempotence**: $I_n I_n = I_n$ — multiplying the identity by itself yields itself.

5. **Inverse**: $I_n^{-1} = I_n$ — the identity is its own inverse.

6. **Trace**: $\operatorname{tr}(I_n) = n$ — the trace (sum of diagonal entries) equals the dimension.

7. **Determinant**: $\det(I_n) = 1$ — the determinant of the identity matrix is 1.

8. **Rank**: $\operatorname{rank}(I_n) = n$ — the identity matrix has full rank (all rows and columns are linearly independent).

9. **Eigenvalues**: All eigenvalues of $I_n$ are 1. Every vector is an eigenvector of the identity.

## Step-by-Step Worked Examples

### Example 1: Verifying $A I = A$

Let $A = \begin{pmatrix} 2 & -1 & 4 \\ 0 & 3 & 5 \end{pmatrix}$ and $I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$. Compute $A I_3$.

**Step 1**: Multiply row 1 of $A$ by each column of $I_3$:
- Column 1: $2\cdot 1 + (-1)\cdot 0 + 4\cdot 0 = 2$
- Column 2: $2\cdot 0 + (-1)\cdot 1 + 4\cdot 0 = -1$
- Column 3: $2\cdot 0 + (-1)\cdot 0 + 4\cdot 1 = 4$

Row 1 of result: $(2, -1, 4)$ — unchanged.

**Step 2**: Multiply row 2 of $A$ by each column of $I_3$:
- Column 1: $0\cdot 1 + 3\cdot 0 + 5\cdot 0 = 0$
- Column 2: $0\cdot 0 + 3\cdot 1 + 5\cdot 0 = 3$
- Column 3: $0\cdot 0 + 3\cdot 0 + 5\cdot 1 = 5$

Row 2 of result: $(0, 3, 5)$ — unchanged.

**Step 3**: The product is:
$$
A I_3 = \begin{pmatrix} 2 & -1 & 4 \\ 0 & 3 & 5 \end{pmatrix} = A
$$

The identity leaves $A$ unchanged, as expected.

### Example 2: Verifying $I A = A$

Let $A = \begin{pmatrix} 2 & -1 & 4 \\ 0 & 3 & 5 \end{pmatrix}$ and $I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$. Compute $I_2 A$.

**Step 1**: Multiply each row of $I_2$ by the columns of $A$:
- Row 1 of $I_2$ times $A$: $1\cdot \text{Row 1 of }A + 0\cdot \text{Row 2 of }A = (2, -1, 4)$
- Row 2 of $I_2$ times $A$: $0\cdot \text{Row 1 of }A + 1\cdot \text{Row 2 of }A = (0, 3, 5)$

**Step 2**: The result is:
$$
I_2 A = \begin{pmatrix} 2 & -1 & 4 \\ 0 & 3 & 5 \end{pmatrix} = A
$$

Verified: $I_2 A = A$.

### Example 3: Identity in the context of inverses

Given $A = \begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix}$, find $A^{-1}$ and verify $A A^{-1} = I$.

**Step 1**: Compute the determinant: $\det(A) = 2\cdot 3 - 1\cdot 5 = 6 - 5 = 1$.

**Step 2**: Use the $2 \times 2$ inverse formula:

$$
A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{1} \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix} = \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix}
$$

**Step 3**: Verify $A A^{-1} = I$:

$$
A A^{-1} = \begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix} \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix} = \begin{pmatrix} 2\cdot 3 + 1\cdot (-5) & 2\cdot (-1) + 1\cdot 2 \\ 5\cdot 3 + 3\cdot (-5) & 5\cdot (-1) + 3\cdot 2 \end{pmatrix} = \begin{pmatrix} 6-5 & -2+2 \\ 15-15 & -5+6 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2
$$

**Step 4**: Verify $A^{-1} A = I$:

$$
A^{-1} A = \begin{pmatrix} 3 & -1 \\ -5 & 2 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 5 & 3 \end{pmatrix} = \begin{pmatrix} 3\cdot 2 + (-1)\cdot 5 & 3\cdot 1 + (-1)\cdot 3 \\ -5\cdot 2 + 2\cdot 5 & -5\cdot 1 + 2\cdot 3 \end{pmatrix} = \begin{pmatrix} 6-5 & 3-3 \\ -10+10 & -5+6 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2
$$

Both products yield $I_2$, confirming that $A^{-1}$ is indeed the inverse of $A$.

### Example 4: Using the identity for regularisation

Suppose $X^T X = \begin{pmatrix} 1 & 2 \\ 2 & 5 \end{pmatrix}$ and $\lambda = 0.1$. Compute $X^T X + \lambda I_2$.

**Step 1**: Write $\lambda I_2 = 0.1 \times \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0.1 & 0 \\ 0 & 0.1 \end{pmatrix}$.

**Step 2**: Add term by term:

$$
X^T X + \lambda I_2 = \begin{pmatrix} 1 & 2 \\ 2 & 5 \end{pmatrix} + \begin{pmatrix} 0.1 & 0 \\ 0 & 0.1 \end{pmatrix} = \begin{pmatrix} 1.1 & 2 \\ 2 & 5.1 \end{pmatrix}
$$

The resulting matrix now has a larger determinant ($1.1 \times 5.1 - 2 \times 2 = 5.61 - 4 = 1.61$) than the original ($1 \times 5 - 4 = 1$), and is guaranteed to be invertible. Even if $X^T X$ were singular (determinant 0), adding $\lambda I$ would make it invertible.

## Visual Interpretation

The identity matrix is the simplest possible diagonal matrix — every diagonal entry is exactly 1, and every off-diagonal entry is exactly 0. You can picture it as a grid where only the cells on the diagonal are "on" (value 1) and everything else is "off" (value 0).

For $I_4$:

$$
I_4 = \begin{pmatrix}
\boxed{1} & 0 & 0 & 0 \\
0 & \boxed{1} & 0 & 0 \\
0 & 0 & \boxed{1} & 0 \\
0 & 0 & 0 & \boxed{1}
\end{pmatrix}
$$

Geometrically, multiplying a vector $v$ by $I_n$ returns the same vector $v$. In 2D or 3D space, the identity transformation leaves every point exactly where it is — no rotation, no scaling, no shear.

## Common Mistakes

1. **Confusing identity with zero matrix**: The identity matrix has 1s on the diagonal, not 0s. Multiplying by the zero matrix gives the zero matrix, not the original matrix.

2. **Assuming $I$ works with any dimensions**: The sizes must be compatible. $A I_n$ requires $A$ to have exactly $n$ columns; $I_m A$ requires $A$ to have exactly $m$ rows.

3. **Thinking $A I = I A$ always**: They are equal when $A$ is square, but if $A$ is $m \times n$ with $m \neq n$, then $A I_n$ yields an $m \times n$ matrix while $I_m A$ yields an $m \times n$ matrix too — but they use different-sized identity matrices. The property $AI = IA = A$ holds, but $I$ on the left and right have different dimensions.

4. **Writing $A^{-1} A = 1$ instead of $I$**: The product of a matrix and its inverse is the identity matrix $I$, not the scalar 1.

5. **Forgetting the identity when computing eigenvalues**: The eigenvalue equation is $A v = \lambda v = \lambda I v$, so it is often written as $(A - \lambda I)v = 0$. Beginners sometimes forget the $I$ and write $A - \lambda$, which is invalid because you cannot subtract a scalar from a matrix.

6. **Believing $I$ is the only matrix with $I^2 = I$**: Any projection matrix also satisfies $P^2 = P$. The identity is a special case.

7. **Confusing the identity matrix with a matrix of all ones**: The matrix $\mathbf{1}$ (all entries 1) is very different from $I$.

## Interview Questions

### Beginner

1. What is the identity matrix $I_n$? Describe its structure.
2. Why is the identity matrix called "identity"?
3. If $A$ is a $3 \times 4$ matrix, what size identity matrix do you multiply on the right? On the left?
4. Is the identity matrix symmetric? Is it invertible?
5. Compute $I_2 \cdot \begin{pmatrix} 5 & -2 \\ 3 & 7 \end{pmatrix}$.

### Intermediate

1. Show that if $A^2 = A$ (a projection matrix) and $A$ is invertible, then $A = I$.
2. Explain why ridge regression adds $\lambda I$ to $X^T X$ and how this guarantees invertibility.
3. Prove that $I_n$ is its own inverse.
4. How does the identity matrix relate to the Kronecker delta $\delta_{ij}$?
5. In the context of ResNets, how does the identity mapping help with the vanishing gradient problem?

### Advanced

1. Derive the relationship between the identity matrix and the matrix exponential: show that $e^{0} = I$ where $0$ is the zero matrix.
2. Explain the role of the identity matrix in the Cayley–Hamilton theorem: every square matrix satisfies its own characteristic polynomial, with $I$ as the constant term.
3. In numerical linear algebra, why is adding a small multiple of $I$ to a matrix a common way to improve conditioning? Provide an example from Levenberg–Marquardt optimisation.

## Practice Problems

### Easy - 5 Questions

1. Write down $I_3$ explicitly.
2. Compute $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} I_2$.
3. Compute $I_2 \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$.
4. What is $\det(I_4)$?
5. What is $\operatorname{tr}(I_5)$?

### Medium - 5 Questions

1. Let $A = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix}$. Compute $A^2 - 5A$ and express your answer in terms of $I_2$.
2. Show that $(I_n - A)(I_n + A) = I_n - A^2$.
3. If $A$ is a square matrix and $A^2 = I$, what is $A^{-1}$?
4. Find all $2 \times 2$ matrices $B$ such that $B^2 = I_2$ and $B \neq I_2$.
5. Let $A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{pmatrix}$. Compute $A - 3I_3$ and describe what operation this performs on the diagonal entries.

### Hard - 3 Questions

1. Prove that if $A$ and $B$ are $n \times n$ matrices and $AB = I$, then $BA = I$ (and therefore $B = A^{-1}$). (Hint: use properties of rank and nullspace.)
2. For any $n \times n$ matrix $A$, show that $A + A^T$ is symmetric and $A - A^T$ is skew-symmetric. Then show that $A = \frac{1}{2}(A + A^T) + \frac{1}{2}(A - A^T)$, and explain the role of $I$ in this decomposition.
3. Show that if $\lambda$ is an eigenvalue of $A$, then $1 + \lambda$ is an eigenvalue of $I + A$, and the eigenvectors are the same.

## Solutions

### Easy Solutions

1. $I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$.

2. $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} I_2 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. The identity leaves the matrix unchanged.

3. $I_2 \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Same result.

4. $\det(I_4) = 1$. The determinant of any identity matrix is 1.

5. $\operatorname{tr}(I_5) = 5$. The trace is the sum of diagonal entries: $1 + 1 + 1 + 1 + 1 = 5$.

### Medium Solutions

1. $A^2 = \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix} \begin{pmatrix} 2 & 1 \\ -1 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 5 \\ -5 & 8 \end{pmatrix}$. Then:
   $$
   A^2 - 5A = \begin{pmatrix} 3 & 5 \\ -5 & 8 \end{pmatrix} - \begin{pmatrix} 10 & 5 \\ -5 & 15 \end{pmatrix} = \begin{pmatrix} -7 & 0 \\ 0 & -7 \end{pmatrix} = -7 I_2
   $$

2. $(I_n - A)(I_n + A) = I_n I_n + I_n A - A I_n - A^2 = I_n + A - A - A^2 = I_n - A^2$.

3. If $A^2 = I$, then multiplying both sides on the left by $A^{-1}$ gives $A = A^{-1}$. Indeed, $A$ is its own inverse.

4. Many matrices satisfy $B^2 = I$:
   - $B = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$: squares to $I$.
   - $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$: squares to $I$.
   - $B = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$: this is $-I$, also squares to $I$.
   
   These are called involutory matrices.

5. $A - 3I_3 = \begin{pmatrix} 1-3 & 0 & 0 \\ 0 & 2-3 & 0 \\ 0 & 0 & 3-3 \end{pmatrix} = \begin{pmatrix} -2 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$. This operation subtracts 3 from each diagonal entry.

### Hard Solutions

1. **Proof that $AB = I \implies BA = I$**:
   
   Assume $AB = I$. Multiply both sides on the left by $A$: $A(AB) = A I \implies (AA)B = A$. But we need $BA = I$.
   
   Since $AB = I$, $A$ has a right-inverse. For square matrices, a right-inverse is also a left-inverse. Here is a proof:
   - $AB = I$ implies $A$ has full rank $n$ (since $I$ has rank $n$).
   - A full-rank square matrix is invertible, so $A^{-1}$ exists.
   - Multiply $AB = I$ on the left by $A^{-1}$: $B = A^{-1}$.
   - Then $BA = A^{-1}A = I$.
   
   Hence $BA = I$.

2. **Symmetry of $A + A^T$**: $(A + A^T)^T = A^T + (A^T)^T = A^T + A = A + A^T$. So $A + A^T$ is symmetric.
   
   **Skew-symmetry of $A - A^T$**: $(A - A^T)^T = A^T - A = -(A - A^T)$. So $A - A^T$ is skew-symmetric.
   
   **Decomposition**: $\frac{1}{2}(A + A^T) + \frac{1}{2}(A - A^T) = \frac{1}{2}A + \frac{1}{2}A^T + \frac{1}{2}A - \frac{1}{2}A^T = A$.
   
   The identity $I$ does not appear here directly, but this decomposition is analogous to writing a function as sum of even and odd parts, and similar decompositions (e.g., $A = \frac{A + A^T}{2} + \frac{A - A^T}{2}$) are fundamental in matrix theory.

3. **Eigenvalue shift**:
   
   Let $v$ be an eigenvector of $A$ with eigenvalue $\lambda$, so $A v = \lambda v$.
   
   Then $(I + A) v = I v + A v = v + \lambda v = (1 + \lambda) v$.
   
   So $v$ is also an eigenvector of $I + A$ with eigenvalue $1 + \lambda$. This property is used extensively in spectral graph theory and when analysing regularised models.

## Related Concepts

- **Zero matrix**: The additive identity for matrices (multiplying by zero matrix gives zero)
- **Diagonal matrix**: $I_n$ is the diagonal matrix where every diagonal entry is 1
- **Inverse matrix**: $A A^{-1} = I$
- **Scalar 1**: The analogue of $I$ in ordinary arithmetic
- **Kronecker delta**: $\delta_{ij}$ gives the entries of $I_n$
- **Orthogonal matrix**: A matrix $Q$ satisfying $Q^T Q = I$

## Next Concepts

- Inverse Matrix (MATH-026)
- Orthogonal matrices
- Eigendecomposition
- Singular value decomposition
- Regularisation in machine learning

## Summary

The identity matrix $I_n$ has 1s on the diagonal and 0s elsewhere. It acts as the multiplicative identity: $A I = I A = A$. It is symmetric, has determinant 1, and is its own inverse. The identity is essential for defining inverses, regularising regression problems, and is the foundation of residual connections in deep neural networks.

## Key Takeaways

- $I_n$ is an $n \times n$ matrix with $\delta_{ij}$ entries
- $A I = A$ and $I A = A$ — the identity leaves any matrix unchanged under multiplication
- $A A^{-1} = A^{-1} A = I$ — the identity defines what it means to be an inverse
- The identity matrix is symmetric ($I^T = I$), has determinant 1, and is its own inverse
- In ML, $\lambda I$ regularisation makes matrices invertible and prevents overfitting
- Residual connections in ResNets leverage the identity mapping to train very deep networks
