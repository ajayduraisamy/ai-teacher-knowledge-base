# Concept: Determinant

## Concept ID

MATH-027

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Compute the determinant of 2x2 and 3x3 matrices using formulas and cofactor expansion
- Understand the geometric meaning of the determinant as a scaling factor for area and volume
- Relate the determinant to matrix invertibility and singularity
- Apply determinant properties such as $\det(AB) = \det(A)\det(B)$ and $\det(A^T) = \det(A)$
- Recognise the role of determinants in AI/ML contexts like covariance matrices and normalising flows

## Prerequisites

- Basic matrix operations (addition, multiplication)
- Familiarity with square matrices
- Elementary algebra and equation solving
- Basic understanding of vectors and area/volume

## Definition

The determinant is a scalar value that is computed from the entries of a square matrix. For an $n \times n$ matrix $A$, the determinant is denoted $\det(A)$, or sometimes $|A|$. It encodes important algebraic and geometric properties of the matrix. A square matrix is invertible if and only if its determinant is non-zero.

## Intuition

Think of the determinant as a single number that captures the "scale factor" of the linear transformation represented by the matrix. When a matrix multiplies a region of space, the absolute value of its determinant tells you how much the area (in 2D) or volume (in 3D) of that region is stretched or shrunk. If the determinant is zero, the transformation squashes the region flat — meaning information is lost and the transformation cannot be reversed.

## Why This Concept Matters

The determinant is a gatekeeper concept in linear algebra. It tells you instantly whether a system of linear equations has a unique solution, whether a matrix has an inverse, and how a linear transformation scales space. In applied fields, determinants are used to check whether a set of vectors is linearly independent, to compute eigenvalues, and to verify properties of covariance matrices. Without the determinant, many core results in linear algebra — including the characterisation of invertible matrices — would be far more cumbersome to express.

## Historical Background

The determinant first appeared in the work of the Japanese mathematician Seki Kowa in 1683 and independently in the work of Gottfried Wilhelm Leibniz around the same time. Seki used determinants to solve systems of linear equations. In the 18th century, Gabriel Cramer developed the rule that now bears his name (Cramer's rule) for solving linear systems using determinants. The modern theory was shaped by Carl Friedrich Gauss, Augustin-Louis Cauchy, and James Joseph Sylvester, who formalised the notation and properties we use today. The term "determinant" itself was coined by Gauss in 1801.

## Real World Examples

- **Computer Graphics**: The determinant of a transformation matrix tells a renderer whether a transformation preserves area (determinant = 1) or flips orientation (determinant negative).
- **Robotics**: Determinants are used in Jacobian matrices to determine whether a robotic arm can move in all directions at a given configuration (non-zero determinant means full mobility).
- **Structural Engineering**: Determinants help analyse whether a system of forces yields a unique equilibrium configuration.
- **Economics**: Input-output models use determinants to check whether an economy's production system is solvable.
- **GPS and Navigation**: Determinants appear in the least-squares solution for position estimation from satellite signals.

## AI/ML Relevance

The determinant is critically important in several areas of machine learning:

- **Covariance Matrices**: In multivariate Gaussian distributions, the covariance matrix $\Sigma$ must be positive definite (all eigenvalues positive) to define a valid probability density. The determinant of $\Sigma$ appears directly in the formula for the multivariate Gaussian PDF:
  $$p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac12 (x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$
  If $\det(\Sigma) = 0$, the covariance matrix is singular, meaning some dimensions are perfectly correlated or have zero variance — the distribution collapses and the density is undefined.

- **Normalising Flows**: In generative modelling, normalising flows transform a simple base distribution into a complex target distribution through a series of invertible transformations. Each transformation $f$ must satisfy:
  $$p_{new}(x) = p_{old}(f^{-1}(x)) \cdot \left|\det J_{f^{-1}}(x)\right|$$
  where $J$ is the Jacobian matrix of the transformation. The determinant of the Jacobian ensures that the probability mass is correctly "stretched" or "compressed" as the transformation is applied. Modern flow architectures (RealNVP, GLOW, Neural ODEs) carefully design transformations whose Jacobian determinants are easy to compute (often triangular or diagonal).

- **Change of Variables**: In Bayesian inference and variational autoencoders, the determinant of the Jacobian appears whenever we change variables in a probability distribution.

## Mathematical Explanation

For a square matrix $A$ of size $n \times n$, the determinant is a function $\det: \mathbb{R}^{n \times n} \to \mathbb{R}$ that satisfies three key properties from which everything else follows:

1. The determinant of the identity matrix is 1: $\det(I_n) = 1$.
2. Swapping two rows flips the sign of the determinant.
3. The determinant is linear in each row.

From these properties, one can derive explicit formulas for computing determinants of matrices of any size.

### 2x2 Determinant

For a $2 \times 2$ matrix:
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

The formula is:
$$\det(A) = ad - bc$$

This is the simplest case and forms the building block for larger matrices.

### 3x3 Determinant

For a $3 \times 3$ matrix:
$$A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$$

Two common methods exist:

**Sarrus' Rule** (only for 3x3):

Copy the first two columns to the right, multiply along the six diagonals, and sum:
$$\det(A) = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}$$

**Cofactor Expansion** (works for any $n \times n$):

Choose any row $i$ or column $j$, then:
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \cdot M_{ij}$$

where $M_{ij}$ is the minor — the determinant of the submatrix obtained by deleting row $i$ and column $j$. The term $(-1)^{i+j} M_{ij}$ is called the cofactor $C_{ij}$.

For a 3x3 matrix expanded along the first row:
$$\det(A) = a_{11} \cdot \det\begin{pmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{pmatrix} - a_{12} \cdot \det\begin{pmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{pmatrix} + a_{13} \cdot \det\begin{pmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{pmatrix}$$

Each 2x2 minor is computed with the formula $ad - bc$.

## Formula(s)

**2x2 Matrix:**
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**3x3 Matrix (Sarrus):**
$$\det\begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}$$

**General nxn (Cofactor Expansion):**
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

## Properties

1. **Product Rule**: $\det(AB) = \det(A) \cdot \det(B)$ for square matrices $A$ and $B$ of the same size.
2. **Transpose**: $\det(A^T) = \det(A)$. The determinant is unchanged by transposition.
3. **Scalar Multiplication**: $\det(cA) = c^n \det(A)$ for an $n \times n$ matrix $A$.
4. **Inverse**: If $A$ is invertible, $\det(A^{-1}) = 1 / \det(A)$.
5. **Zero Determinant**: $\det(A) = 0$ if and only if $A$ is singular (not invertible).
6. **Row Operations**: Swapping two rows flips the sign. Adding a multiple of one row to another does not change the determinant. Multiplying a row by a scalar $c$ multiplies the determinant by $c$.
7. **Triangular Matrices**: If $A$ is upper or lower triangular, $\det(A)$ equals the product of the diagonal entries.
8. **Similar Matrices**: If $B = P^{-1}AP$, then $\det(B) = \det(A)$. Determinants are similarity invariants.

## Step-by-Step Worked Examples

### Example 1: 2x2 Determinant

Compute $\det(A)$ for $A = \begin{pmatrix} 3 & 7 \\ 2 & 5 \end{pmatrix}$.

**Step 1**: Identify $a=3$, $b=7$, $c=2$, $d=5$.

**Step 2**: Apply the formula $\det(A) = ad - bc$.

**Step 3**: $ad = 3 \times 5 = 15$.

**Step 4**: $bc = 7 \times 2 = 14$.

**Step 5**: $\det(A) = 15 - 14 = 1$.

Since $\det(A) \neq 0$, the matrix is invertible.

### Example 2: 3x3 Determinant Using Sarrus' Rule

Compute $\det(A)$ for $A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{pmatrix}$.

**Step 1**: Write the matrix and repeat the first two columns to the right:
$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{pmatrix} \begin{matrix} 1 & 2 \\ 0 & 4 \\ 1 & 0 \end{matrix}$$

**Step 2**: Compute the three downward diagonal products:
- $1 \times 4 \times 6 = 24$
- $2 \times 5 \times 1 = 10$
- $3 \times 0 \times 0 = 0$

Sum of downward products: $24 + 10 + 0 = 34$

**Step 3**: Compute the three upward diagonal products:
- $3 \times 4 \times 1 = 12$
- $1 \times 5 \times 0 = 0$
- $2 \times 0 \times 6 = 0$

Sum of upward products: $12 + 0 + 0 = 12$

**Step 4**: Subtract: $\det(A) = 34 - 12 = 22$.

Thus $\det(A) = 22$. The matrix is invertible.

### Example 3: 3x3 Determinant Using Cofactor Expansion

Compute $\det(A)$ for $A = \begin{pmatrix} 2 & -1 & 0 \\ 3 & 2 & 1 \\ 1 & 4 & -2 \end{pmatrix}$ by expanding along the first row.

**Step 1**: Identify entries in row 1:
- $a_{11} = 2$, $a_{12} = -1$, $a_{13} = 0$

**Step 2**: For each entry, compute the cofactor.

For $a_{11} = 2$:
- Minor $M_{11} = \det\begin{pmatrix} 2 & 1 \\ 4 & -2 \end{pmatrix} = (2)(-2) - (1)(4) = -4 - 4 = -8$
- Cofactor $C_{11} = (-1)^{1+1} M_{11} = (+1)(-8) = -8$

For $a_{12} = -1$:
- Minor $M_{12} = \det\begin{pmatrix} 3 & 1 \\ 1 & -2 \end{pmatrix} = (3)(-2) - (1)(1) = -6 - 1 = -7$
- Cofactor $C_{12} = (-1)^{1+2} M_{12} = (-1)(-7) = 7$

For $a_{13} = 0$:
- Minor $M_{13} = \det\begin{pmatrix} 3 & 2 \\ 1 & 4 \end{pmatrix} = (3)(4) - (2)(1) = 12 - 2 = 10$
- Cofactor $C_{13} = (-1)^{1+3} M_{13} = (+1)(10) = 10$

**Step 3**: Combine: $\det(A) = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}$

$\det(A) = 2(-8) + (-1)(7) + 0(10)$

$\det(A) = -16 - 7 + 0 = -23$

Thus $\det(A) = -23$. The non-zero determinant confirms the matrix is invertible.

### Example 4: Determinant of a Triangular Matrix

Compute $\det(A)$ for $A = \begin{pmatrix} 3 & 5 & 2 \\ 0 & -1 & 4 \\ 0 & 0 & 6 \end{pmatrix}$.

**Step 1**: Notice this is an upper triangular matrix (all entries below the main diagonal are zero).

**Step 2**: For a triangular matrix, the determinant is simply the product of the diagonal entries.

**Step 3**: $\det(A) = 3 \times (-1) \times 6 = -18$.

This is a powerful shortcut that saves significant computation.

### Example 5: Using the Product Rule

Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ and $B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$.

Compute $\det(A)$, $\det(B)$, $\det(AB)$, and verify $\det(AB) = \det(A)\det(B)$.

**Step 1**: Compute $\det(A) = (1)(4) - (2)(3) = 4 - 6 = -2$.

**Step 2**: Compute $\det(B) = (5)(8) - (6)(7) = 40 - 42 = -2$.

**Step 3**: Compute $AB = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 1(5)+2(7) & 1(6)+2(8) \\ 3(5)+4(7) & 3(6)+4(8) \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}$.

**Step 4**: Compute $\det(AB) = (19)(50) - (22)(43) = 950 - 946 = 4$.

**Step 5**: Verify: $\det(A)\det(B) = (-2)(-2) = 4 = \det(AB)$. The property holds.

## Visual Interpretation

Geometrically, the absolute value of the determinant of a $2 \times 2$ matrix equals the area of the parallelogram formed by its column vectors. For a $3 \times 3$ matrix, it equals the volume of the parallelepiped formed by its three column vectors.

Consider the matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$. Its columns are $v_1 = \begin{pmatrix} a \\ c \end{pmatrix}$ and $v_2 = \begin{pmatrix} b \\ d \end{pmatrix}$. The parallelogram with vertices at $(0,0)$, $v_1$, $v_2$, and $v_1+v_2$ has area $|ad - bc| = |\det(A)|$.

The sign of the determinant indicates orientation. A positive determinant means the transformation preserves orientation (right-handed coordinate system stays right-handed). A negative determinant means the transformation flips orientation (mirroring).

When $\det(A) = 0$, the columns are linearly dependent — they lie on the same line (in 2D) or the same plane (in 3D). The parallelogram collapses to a line segment, giving zero area.

## Common Mistakes

1. **Forgetting to alternate signs in cofactor expansion**: The sign pattern starts with $+$ at position $(1,1)$ and alternates: $\begin{pmatrix}+&-&+\\-&+&-\\+&-&+\end{pmatrix}$. A common error is to use all $+$ signs.

2. **Applying Sarrus' rule to 4x4 or larger matrices**: Sarrus' rule only works for $3 \times 3$ matrices. For larger matrices, cofactor expansion or row reduction must be used.

3. **Confusing $\det(cA)$ with $c\det(A)$**: For $n \times n$ matrix $A$, $\det(cA) = c^n \det(A)$, not $c\det(A)$. The scalar multiplies every row, so it factors out $n$ times.

4. **Forgetting to transpose signs correctly**: $\det(A^T) = \det(A)$, but $\det(A^*) = \overline{\det(A)}$ (conjugate) for complex matrices.

5. **Assuming $\det(A+B) = \det(A) + \det(B)$**: The determinant is not linear under addition. There is no simple formula for $\det(A+B)$.

6. **Miscomputing the 3x3 Sarrus pattern**: The three downward diagonals start from the top row and go right-down; the three upward diagonals start from the bottom row and go right-up. Mixing these up is a frequent error.

7. **Ignoring zero determinants when dividing**: If $\det(A) = 0$, then $A$ is not invertible, so expressions like $A^{-1}$ are undefined. This must be checked before using formulas involving inverses.

## Interview Questions

### Beginner

1. **Q**: What is the determinant of a $2 \times 2$ matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$?
   **A**: $\det = ad - bc$.

2. **Q**: If $\det(A) = 0$, what does that tell you about the matrix?
   **A**: The matrix is singular (not invertible). Its rows and columns are linearly dependent.

3. **Q**: What is $\det(I_3)$, the determinant of the $3 \times 3$ identity matrix?
   **A**: $\det(I_3) = 1$.

4. **Q**: How does the determinant change if you multiply every entry of a $3 \times 3$ matrix by 2?
   **A**: $\det(2A) = 2^3 \det(A) = 8\det(A)$.

5. **Q**: True or false: $\det(A+B) = \det(A) + \det(B)$ for all square matrices $A$ and $B$.
   **A**: False. The determinant is not linear under addition.

### Intermediate

1. **Q**: Prove that if $A$ is invertible, then $\det(A^{-1}) = 1 / \det(A)$.
   **A**: Since $A A^{-1} = I$, we have $\det(A A^{-1}) = \det(I) = 1$. By the product rule, $\det(A)\det(A^{-1}) = 1$, so $\det(A^{-1}) = 1 / \det(A)$.

2. **Q**: What is the geometric interpretation of a negative determinant for a $2 \times 2$ matrix?
   **A**: A negative determinant indicates that the transformation reverses orientation (i.e., it includes a reflection). The absolute value still gives the area scaling factor.

3. **Q**: Compute $\det\begin{pmatrix} 1 & 0 & 2 \\ 0 & 3 & 0 \\ 4 & 0 & 5 \end{pmatrix}$ using cofactor expansion.
   **A**: Expand along the second row (has two zeros): $\det = 3 \cdot (-1)^{2+2} \cdot \det\begin{pmatrix} 1 & 2 \\ 4 & 5 \end{pmatrix} = 3 \cdot (1\cdot5 - 2\cdot4) = 3 \cdot (5-8) = 3 \cdot (-3) = -9$.

4. **Q**: How does $\det(ABA^{-1})$ relate to $\det(B)$?
   **A**: $\det(ABA^{-1}) = \det(A)\det(B)\det(A^{-1}) = \det(A)\det(B) \cdot \frac{1}{\det(A)} = \det(B)$. So $\det(ABA^{-1}) = \det(B)$.

5. **Q**: What happens to the determinant if two rows of a matrix are identical?
   **A**: The determinant is zero. The rows are linearly dependent, so the matrix is singular.

### Advanced

1. **Q**: Explain why $\det(A) = 0$ is equivalent to the columns of $A$ being linearly dependent.
   **A**: If the columns are linearly dependent, there exists a non-zero vector $x$ such that $Ax = 0$. This means $A$ has a non-trivial nullspace, so $A$ is not invertible. The determinant of a non-invertible matrix is zero. Conversely, if $\det(A) = 0$, then by the invertible matrix theorem, $A$ is singular, meaning its columns are linearly dependent.

2. **Q**: Show that $\det(A)$ equals the product of the eigenvalues of $A$ (counting multiplicities).
   **A**: The characteristic polynomial of $A$ is $p(\lambda) = \det(A - \lambda I)$. When $\lambda = 0$, $p(0) = \det(A)$. The characteristic polynomial can also be factored as $p(\lambda) = \prod_{i=1}^n (\lambda_i - \lambda)$, where $\lambda_i$ are the eigenvalues. Setting $\lambda = 0$ gives $p(0) = \prod_{i=1}^n \lambda_i$. Therefore $\det(A) = \prod_{i=1}^n \lambda_i$.

3. **Q**: In the context of normalising flows, why are transformations with triangular Jacobians preferred?
   **A**: The determinant of a triangular matrix equals the product of its diagonal entries, which is extremely cheap to compute ($O(n)$ instead of $O(n^3)$). In normalising flows, we need the determinant of the Jacobian for the change-of-variables formula. By designing layers with triangular Jacobians (e.g., coupling layers in RealNVP), the determinant computation stays efficient even for high-dimensional data like images.

## Practice Problems

### Easy - 5 Questions

1. Compute $\det\begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$.

2. Compute $\det\begin{pmatrix} 5 & 0 \\ 0 & 7 \end{pmatrix}$.

3. Compute $\det\begin{pmatrix} 2 & 8 \\ 1 & 4 \end{pmatrix}$.

4. What is $\det(I_4)$, the $4 \times 4$ identity matrix?

5. If $\det(A) = 3$ for a $2 \times 2$ matrix $A$, what is $\det(2A)$?

### Medium - 5 Questions

1. Compute $\det\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$.

2. Compute $\det\begin{pmatrix} 2 & 0 & 0 \\ -1 & 3 & 0 \\ 4 & 5 & -2 \end{pmatrix}$ using the triangular property.

3. Let $A$ be $3 \times 3$ with $\det(A) = 5$. Compute $\det(3A)$ and $\det(A^{-1})$.

4. Use cofactor expansion along the third column to compute $\det\begin{pmatrix} 1 & 0 & 4 \\ 2 & 1 & 0 \\ -1 & 3 & 2 \end{pmatrix}$.

5. If $A$ and $B$ are $3 \times 3$ matrices with $\det(A) = 2$ and $\det(B) = -3$, compute $\det(AB)$ and $\det(2A^{-1}B)$.

### Hard - 3 Questions

1. Prove that $\det\begin{pmatrix} 1 & 1 & 1 \\ x & y & z \\ x^2 & y^2 & z^2 \end{pmatrix} = (y-x)(z-x)(z-y)$. This is the Vandermonde determinant for $n=3$.

2. For an $n \times n$ matrix $A$, show that $\det(\text{adj}(A)) = (\det(A))^{n-1}$, where $\text{adj}(A)$ is the adjugate (classical adjoint) of $A$.

3. A $3 \times 3$ matrix $A$ has eigenvalues $1$, $2$, and $3$. What is $\det(A)$? If $B = A^2 - 2A + I$, compute $\det(B)$.

## Solutions

### Easy Solutions

1. $\det = (4)(3) - (1)(2) = 12 - 2 = 10$.

2. $\det = (5)(7) - (0)(0) = 35$. (Diagonal matrix: determinant is product of diagonal entries.)

3. $\det = (2)(4) - (8)(1) = 8 - 8 = 0$. The matrix is singular.

4. $\det(I_4) = 1$.

5. For a $2 \times 2$ matrix, $\det(2A) = 2^2 \det(A) = 4 \times 3 = 12$.

### Medium Solutions

1. $\det = \begin{vmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{vmatrix}$.
   Using Sarrus: $(1)(5)(9) + (2)(6)(7) + (3)(4)(8) - (3)(5)(7) - (1)(6)(8) - (2)(4)(9)$
   $= 45 + 84 + 96 - 105 - 48 - 72 = 225 - 225 = 0$.

2. This is a lower triangular matrix. $\det = (2)(3)(-2) = -12$.

3. $\det(3A) = 3^3 \det(A) = 27 \times 5 = 135$.
   $\det(A^{-1}) = 1 / \det(A) = 1/5$.

4. Expanding along column 3 ($a_{13}=4$, $a_{23}=0$, $a_{33}=2$):
   $\det = 4(-1)^{1+3}\begin{vmatrix} 2 & 1 \\ -1 & 3 \end{vmatrix} + 0 + 2(-1)^{3+3}\begin{vmatrix} 1 & 0 \\ 2 & 1 \end{vmatrix}$
   $= 4(6 - (-1)) + 2(1 - 0) = 4(7) + 2 = 28 + 2 = 30$.

5. $\det(AB) = \det(A)\det(B) = 2 \times (-3) = -6$.
   $\det(2A^{-1}B) = 2^3 \det(A^{-1})\det(B) = 8 \cdot \frac{1}{2} \cdot (-3) = 8 \cdot (-1.5) = -12$.

### Hard Solutions

1. **Vandermonde**: $\det = \begin{vmatrix} 1 & 1 & 1 \\ x & y & z \\ x^2 & y^2 & z^2 \end{vmatrix}$.
   Subtract row 1 from row 2 and row 3 (operations that don't change determinant):
   $$\begin{vmatrix} 1 & 1 & 1 \\ 0 & y-x & z-x \\ 0 & y^2-x^2 & z^2-x^2 \end{vmatrix} = \begin{vmatrix} 1 & 1 & 1 \\ 0 & y-x & z-x \\ 0 & (y-x)(y+x) & (z-x)(z+x) \end{vmatrix}$$
   Expand along column 1: $= 1 \cdot \begin{vmatrix} y-x & z-x \\ (y-x)(y+x) & (z-x)(z+x) \end{vmatrix}$.
   Factor $(y-x)$ from column 1 and $(z-x)$ from column 2: $= (y-x)(z-x) \begin{vmatrix} 1 & 1 \\ y+x & z+x \end{vmatrix}$.
   $= (y-x)(z-x)((z+x) - (y+x)) = (y-x)(z-x)(z-y)$.

   Rearranging: $= (y-x)(z-x)(z-y) = (y-x)(z-x)(z-y)$. Notice $-(y-x)(z-x)(y-z) = (y-x)(z-x)(z-y)$. Verified.

2. The adjugate satisfies $A \cdot \text{adj}(A) = \det(A) I$. Taking determinants:
   $\det(A \cdot \text{adj}(A)) = \det(\det(A) I)$
   $\det(A) \det(\text{adj}(A)) = (\det(A))^n$
   If $\det(A) \neq 0$: $\det(\text{adj}(A)) = (\det(A))^{n-1}$.
   If $\det(A) = 0$, both sides are 0, and the identity holds by continuity.

3. $\det(A) = \lambda_1 \cdot \lambda_2 \cdot \lambda_3 = 1 \times 2 \times 3 = 6$.
   The eigenvalues of $B = A^2 - 2A + I$ are $\lambda^2 - 2\lambda + 1 = (\lambda - 1)^2$.
   So eigenvalues of $B$ are: $(1-1)^2 = 0$, $(2-1)^2 = 1$, $(3-1)^2 = 4$.
   $\det(B) = 0 \times 1 \times 4 = 0$.

## Related Concepts

- **Matrix Inverse**: Directly linked — a matrix is invertible iff its determinant is non-zero
- **Eigenvalues**: The determinant equals the product of all eigenvalues
- **Cofactor and Minor**: Building blocks for computing determinants of larger matrices
- **Cross Product**: In 3D, the cross product of two vectors can be expressed using a determinant
- **Jacobian Matrix**: The determinant of the Jacobian gives the volume scaling factor in transformations
- **Characteristic Polynomial**: Defined as $\det(A - \lambda I)$, central to eigenvalue computation
- **Cramer's Rule**: Uses determinants to solve linear systems

## Next Concepts

- Rank of a matrix (MATH-028)
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Matrix factorisations (LU, QR)

## Summary

The determinant is a scalar value associated with a square matrix that summarises key properties about the matrix: whether it is invertible, how it scales volume, and the product of its eigenvalues. For a $2 \times 2$ matrix, $\det = ad - bc$; for larger matrices, computation proceeds through cofactor expansion or row reduction. The determinant satisfies important algebraic properties including multiplicativity ($\det(AB) = \det(A)\det(B)$) and invariance under transpose ($\det(A^T) = \det(A)$). Determinants are essential in verifying positive definiteness of covariance matrices, computing probability densities in multivariate statistics, and enabling normalising flows for generative modelling.

## Key Takeaways

- The determinant is zero if and only if a square matrix is singular (non-invertible)
- For a $2 \times 2$ matrix, $\det = ad - bc$; for a $3 \times 3$ matrix, use Sarrus' rule or cofactor expansion
- The absolute value of the determinant gives the volume scaling factor of the linear transformation
- $\det(AB) = \det(A)\det(B)$ and $\det(A^T) = \det(A)$ are fundamental properties
- $\det(cA) = c^n \det(A)$ for an $n \times n$ matrix
- Determinants are vital in ML for checking covariance matrix invertibility and in normalising flows for probability density transformation
