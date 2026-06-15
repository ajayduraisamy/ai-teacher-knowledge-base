# Concept: Inverse Matrix

## Concept ID

MATH-026

## Difficulty

ADVANCED

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define the inverse matrix $A^{-1}$ and state the condition for invertibility
- Compute the inverse of $2 \times 2$ matrices using the closed-form formula
- Apply inverse matrices to solve systems of linear equations
- Understand why inverses are computationally expensive for large systems and when alternative methods are preferred
- Connect matrix inversion to key machine learning algorithms like linear regression

## Prerequisites

- Matrix multiplication (including the identity matrix $I$, MATH-025)
- Determinants (especially $2 \times 2$ determinants)
- Transpose (MATH-024)
- Basic understanding of systems of linear equations

## Definition

Let $A$ be an $n \times n$ (square) matrix. If there exists another $n \times n$ matrix $B$ such that:

$$
AB = BA = I_n
$$

then $B$ is called the **inverse** of $A$, denoted $A^{-1}$ (read "A inverse"). A matrix that has an inverse is called **invertible** or **non-singular**. A matrix that does not have an inverse is called **singular** or **non-invertible**.

The defining equations mean: multiplying $A$ by its inverse in either order yields the identity matrix $I_n$.

## Intuition

In ordinary arithmetic, the inverse of a number $x$ is $1/x$ (also written $x^{-1}$), because $x \cdot (1/x) = 1$. The inverse matrix is the matrix analogue of a reciprocal. If the identity matrix $I$ is like the number 1, then $A^{-1}$ is like $1/A$.

Another helpful intuition: if a matrix $A$ represents a transformation (stretch, rotate, shear), then $A^{-1}$ represents the **reverse transformation**. Applying $A$ and then $A^{-1}$ brings you back to where you started — that is exactly $A A^{-1} = I$, the "do nothing" transformation.

## Why This Concept Matters

Matrix inversion is one of the most important operations in linear algebra. It allows us to solve linear systems $Ax = b$ in one step: $x = A^{-1} b$. It is used throughout engineering, physics, computer graphics, statistics, and machine learning. However, because computing $A^{-1}$ for large matrices is extremely expensive ($O(n^3)$ time), understanding when and how to use inverses — and when to avoid them — is a critical skill.

## Historical Background

The concept of the matrix inverse emerged in the 19th century alongside the development of matrix algebra. Arthur Cayley introduced the notation $A^{-1}$ in his 1858 memoir. Earlier, Carl Friedrich Gauss (1777–1855) had developed Gaussian elimination for solving linear systems without explicitly computing inverses. The explicit formula for the $2 \times 2$ inverse has been known for centuries. Modern numerical analysis focuses on avoiding explicit inversion for large matrices due to stability and efficiency concerns.

## Real World Examples

- **GPS triangulation**: Solving for a receiver's position from satellite signals is a system of linear equations solved via matrix inverses (or more efficiently via least squares).
- **Cryptography**: The Hill cipher encrypts messages using an invertible matrix; decryption uses its inverse.
- **Electrical engineering**: Solving circuits with Kirchhoff's laws reduces to solving $Ax = b$, where $A$ represents the impedance matrix.
- **Economics**: Input-output models (Leontief) use matrix inverses to compute the total production needed to meet final demand.

## AI/ML Relevance

Inverse matrices appear in several foundational machine learning methods:

- **Normal equations for linear regression**: The optimal weights $\hat{w}$ for linear regression are given by:
  $$
  \hat{w} = (X^T X)^{-1} X^T y
  $$
  Here $(X^T X)^{-1}$ is the inverse of the Gram matrix $X^T X$. This is a closed-form solution — no iterative optimisation needed.

- **Ridge regression**: Adds regularisation: $\hat{w} = (X^T X + \lambda I)^{-1} X^T y$. The term $\lambda I$ guarantees that $X^T X + \lambda I$ is invertible even if $X^T X$ is singular.

- **Mahalanobis distance**: Uses the inverse of the covariance matrix $\Sigma^{-1}$ to measure distances that account for correlations between features.

- **Fisher information matrix**: Its inverse gives the Cramér–Rao lower bound on the variance of parameter estimators.

- **Why gradient descent is preferred**: For large datasets with $p$ features and $n$ samples, computing $(X^T X)^{-1}$ costs $O(p^3)$ time and $O(p^2)$ memory. When $p$ is large (say, millions), this is infeasible. Gradient descent and its variants avoid inversion entirely by iteratively updating weights, costing only $O(np)$ per iteration.

## Mathematical Explanation

For a $2 \times 2$ matrix, there is a simple closed-form formula for the inverse. Let:

$$
A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}
$$

The determinant of $A$ is:

$$
\det(A) = ad - bc
$$

If $\det(A) \neq 0$, the inverse exists and is given by:

$$
A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

Let us verify this formula:

$$
A A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

Multiply the matrices:

$$
= \frac{1}{ad - bc} \begin{pmatrix}
a d + b(-c) & a(-b) + b a \\
c d + d(-c) & c(-b) + d a
\end{pmatrix}
= \frac{1}{ad - bc} \begin{pmatrix}
ad - bc & -ab + ab \\
cd - cd & -bc + ad
\end{pmatrix}
= \frac{1}{ad - bc} \begin{pmatrix}
ad - bc & 0 \\
0 & ad - bc
\end{pmatrix}
= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I_2
$$

The verification works because the numerator $ad - bc$ cancels with the denominator, leaving the identity matrix.

For matrices larger than $2 \times 2$, computing the inverse is more involved. Methods include Gaussian elimination (Gauss–Jordan), LU decomposition, and the adjugate formula. In practice, numerical software (NumPy, MATLAB) uses highly optimised algorithms based on LU decomposition with partial pivoting.

## Formula(s)

**$2 \times 2$ inverse**:

$$
A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad
A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

**Condition for existence**:

$$
\det(A) \neq 0
$$

**General property**:

$$
A A^{-1} = A^{-1} A = I
$$

**Solving linear systems**:

$$
Ax = b \implies x = A^{-1}b
$$

## Properties

1. **Uniqueness**: If $A$ is invertible, its inverse is unique. There is exactly one matrix $A^{-1}$ satisfying $A A^{-1} = I$.

2. **Inverse of inverse**: $(A^{-1})^{-1} = A$ — the inverse of the inverse is the original matrix.

3. **Product reversal**: $(AB)^{-1} = B^{-1} A^{-1}$ — just like transposes, the order reverses when taking the inverse of a product. For three matrices: $(ABC)^{-1} = C^{-1} B^{-1} A^{-1}$.

4. **Transpose inverse**: $(A^T)^{-1} = (A^{-1})^T$ — invert then transpose is the same as transpose then invert. This matrix is sometimes denoted $A^{-T}$.

5. **Scalar multiplication**: $(cA)^{-1} = \frac{1}{c} A^{-1}$ for $c \neq 0$.

6. **Determinant of inverse**: $\det(A^{-1}) = \frac{1}{\det(A)}$ — the determinant of the inverse is the reciprocal of the original determinant.

7. **Invertibility of diagonal matrices**: If $D = \operatorname{diag}(d_1, d_2, \ldots, d_n)$ is a diagonal matrix, then $D^{-1} = \operatorname{diag}(1/d_1, 1/d_2, \ldots, 1/d_n)$, provided all $d_i \neq 0$.

8. **Inverse of identity**: $I^{-1} = I$ — the identity is its own inverse.

9. **Singular matrices**: A matrix is singular (non-invertible) if and only if $\det(A) = 0$, which is equivalent to the matrix having linearly dependent rows (or columns), or having a zero eigenvalue.

## Step-by-Step Worked Examples

### Example 1: Inverse of a $2 \times 2$ matrix

Find $A^{-1}$ for $A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$.

**Step 1**: Compute the determinant:
$$
\det(A) = 2 \cdot 4 - 3 \cdot 1 = 8 - 3 = 5
$$

Since $\det(A) = 5 \neq 0$, the inverse exists.

**Step 2**: Apply the $2 \times 2$ formula:
$$
A^{-1} = \frac{1}{5} \begin{pmatrix} 4 & -3 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} \frac{4}{5} & -\frac{3}{5} \\ -\frac{1}{5} & \frac{2}{5} \end{pmatrix}
$$

**Step 3**: Verify by multiplying $A A^{-1}$:
$$
A A^{-1} = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix} \begin{pmatrix} \frac{4}{5} & -\frac{3}{5} \\ -\frac{1}{5} & \frac{2}{5} \end{pmatrix} = \begin{pmatrix}
2\cdot\frac{4}{5} + 3\cdot(-\frac{1}{5}) & 2\cdot(-\frac{3}{5}) + 3\cdot\frac{2}{5} \\
1\cdot\frac{4}{5} + 4\cdot(-\frac{1}{5}) & 1\cdot(-\frac{3}{5}) + 4\cdot\frac{2}{5}
\end{pmatrix}
= \begin{pmatrix}
\frac{8}{5} - \frac{3}{5} & -\frac{6}{5} + \frac{6}{5} \\
\frac{4}{5} - \frac{4}{5} & -\frac{3}{5} + \frac{8}{5}
\end{pmatrix} = \begin{pmatrix}
\frac{5}{5} & 0 \\
0 & \frac{5}{5}
\end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

The verification confirms the inverse is correct.

### Example 2: Solving a linear system with an inverse

Solve the system:
$$
\begin{cases}
2x + 3y = 7 \\
x + 4y = 6
\end{cases}
$$

**Step 1**: Write in matrix form $Ax = b$:
$$
A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}, \quad x = \begin{pmatrix} x \\ y \end{pmatrix}, \quad b = \begin{pmatrix} 7 \\ 6 \end{pmatrix}
$$

**Step 2**: Use $A^{-1}$ from Example 1:
$$
x = A^{-1} b = \begin{pmatrix} \frac{4}{5} & -\frac{3}{5} \\ -\frac{1}{5} & \frac{2}{5} \end{pmatrix} \begin{pmatrix} 7 \\ 6 \end{pmatrix}
$$

**Step 3**: Multiply:
$$
x = \begin{pmatrix}
\frac{4}{5} \cdot 7 + (-\frac{3}{5}) \cdot 6 \\
-\frac{1}{5} \cdot 7 + \frac{2}{5} \cdot 6
\end{pmatrix} = \begin{pmatrix}
\frac{28}{5} - \frac{18}{5} \\
-\frac{7}{5} + \frac{12}{5}
\end{pmatrix} = \begin{pmatrix}
\frac{10}{5} \\
\frac{5}{5}
\end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$

So $x = 2$, $y = 1$.

**Step 4**: Verify by substitution:
$$
2(2) + 3(1) = 4 + 3 = 7 \quad \checkmark
$$
$$
1(2) + 4(1) = 2 + 4 = 6 \quad \checkmark
$$

### Example 3: Verifying the product reversal property

Given $A = \begin{pmatrix} 1 & 2 \\ 1 & 3 \end{pmatrix}$ and $B = \begin{pmatrix} 2 & 0 \\ 1 & 2 \end{pmatrix}$, verify that $(AB)^{-1} = B^{-1} A^{-1}$.

**Step 1**: Find $A^{-1}$:
$$
\det(A) = 1\cdot 3 - 2\cdot 1 = 3 - 2 = 1
$$
$$
A^{-1} = \frac{1}{1} \begin{pmatrix} 3 & -2 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 3 & -2 \\ -1 & 1 \end{pmatrix}
$$

**Step 2**: Find $B^{-1}$:
$$
\det(B) = 2\cdot 2 - 0\cdot 1 = 4
$$
$$
B^{-1} = \frac{1}{4} \begin{pmatrix} 2 & 0 \\ -1 & 2 \end{pmatrix} = \begin{pmatrix} \frac{1}{2} & 0 \\ -\frac{1}{4} & \frac{1}{2} \end{pmatrix}
$$

**Step 3**: Compute $AB$:
$$
AB = \begin{pmatrix} 1 & 2 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 1\cdot 2 + 2\cdot 1 & 1\cdot 0 + 2\cdot 2 \\ 1\cdot 2 + 3\cdot 1 & 1\cdot 0 + 3\cdot 2 \end{pmatrix} = \begin{pmatrix} 4 & 4 \\ 5 & 6 \end{pmatrix}
$$

**Step 4**: Compute $(AB)^{-1}$:
$$
\det(AB) = 4\cdot 6 - 4\cdot 5 = 24 - 20 = 4
$$
$$
(AB)^{-1} = \frac{1}{4} \begin{pmatrix} 6 & -4 \\ -5 & 4 \end{pmatrix} = \begin{pmatrix} \frac{3}{2} & -1 \\ -\frac{5}{4} & 1 \end{pmatrix}
$$

**Step 5**: Compute $B^{-1} A^{-1}$ (reverse order!):
$$
B^{-1} A^{-1} = \begin{pmatrix} \frac{1}{2} & 0 \\ -\frac{1}{4} & \frac{1}{2} \end{pmatrix} \begin{pmatrix} 3 & -2 \\ -1 & 1 \end{pmatrix}
= \begin{pmatrix}
\frac{1}{2}\cdot 3 + 0\cdot (-1) & \frac{1}{2}\cdot (-2) + 0\cdot 1 \\
(-\frac{1}{4})\cdot 3 + \frac{1}{2}\cdot (-1) & (-\frac{1}{4})\cdot (-2) + \frac{1}{2}\cdot 1
\end{pmatrix}
= \begin{pmatrix}
\frac{3}{2} & -1 \\
-\frac{3}{4} - \frac{1}{2} & \frac{1}{2} + \frac{1}{2}
\end{pmatrix}
= \begin{pmatrix}
\frac{3}{2} & -1 \\
-\frac{5}{4} & 1
\end{pmatrix}
$$

**Step 6**: Compare — $(AB)^{-1} = B^{-1} A^{-1}$ is verified.

### Example 4: A singular (non-invertible) matrix

Show that $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$ has no inverse.

**Step 1**: Compute the determinant:
$$
\det(A) = 1\cdot 4 - 2\cdot 2 = 4 - 4 = 0
$$

**Step 2**: Since $\det(A) = 0$, the matrix is singular and has no inverse.

**Interpretation**: The second row $(2, 4)$ is exactly twice the first row $(1, 2)$. The rows are linearly dependent. Geometrically, this matrix maps all 2D points onto a line, compressing the plane into a lower dimension. Such a transformation cannot be reversed — information is lost, and no inverse exists.

### Example 5: Normal equations for linear regression

Suppose we have a small dataset with $n = 4$ data points and $p = 2$ features (including a bias term). Let:

$$
X = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix}, \quad y = \begin{pmatrix} 2 \\ 3 \\ 3.5 \\ 5 \end{pmatrix}
$$

Find the optimal linear regression weights $\hat{w} = (X^T X)^{-1} X^T y$.

**Step 1**: Compute $X^T X$:
$$
X^T X = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 2 & 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \end{pmatrix} = \begin{pmatrix}
1+1+1+1 & 1+2+3+4 \\
1+2+3+4 & 1+4+9+16
\end{pmatrix} = \begin{pmatrix}
4 & 10 \\
10 & 30
\end{pmatrix}
$$

**Step 2**: Compute $\det(X^T X) = 4\cdot 30 - 10\cdot 10 = 120 - 100 = 20$.
Since $20 \neq 0$, the inverse exists.

**Step 3**: Compute $(X^T X)^{-1}$:
$$
(X^T X)^{-1} = \frac{1}{20} \begin{pmatrix} 30 & -10 \\ -10 & 4 \end{pmatrix} = \begin{pmatrix} 1.5 & -0.5 \\ -0.5 & 0.2 \end{pmatrix}
$$

**Step 4**: Compute $X^T y$:
$$
X^T y = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 2 & 3 & 4 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \\ 3.5 \\ 5 \end{pmatrix} = \begin{pmatrix}
2 + 3 + 3.5 + 5 \\
2 + 6 + 10.5 + 20
\end{pmatrix} = \begin{pmatrix}
13.5 \\
38.5
\end{pmatrix}
$$

**Step 5**: Compute $\hat{w} = (X^T X)^{-1} X^T y$:
$$
\hat{w} = \begin{pmatrix} 1.5 & -0.5 \\ -0.5 & 0.2 \end{pmatrix} \begin{pmatrix} 13.5 \\ 38.5 \end{pmatrix} = \begin{pmatrix}
1.5 \cdot 13.5 + (-0.5) \cdot 38.5 \\
-0.5 \cdot 13.5 + 0.2 \cdot 38.5
\end{pmatrix} = \begin{pmatrix}
20.25 - 19.25 \\
-6.75 + 7.7
\end{pmatrix} = \begin{pmatrix}
1.0 \\
0.95
\end{pmatrix}
$$

So $\hat{w} = (1.0, 0.95)$, meaning the best-fit line is $y = 1.0 + 0.95 x$.

## Visual Interpretation

Geometrically, an invertible matrix corresponds to a bijective (one-to-one and onto) linear transformation. Applying $A$ to a set of points stretches and rotates space without collapsing any dimension. The inverse $A^{-1}$ "undoes" this transformation.

For a $2 \times 2$ matrix:
- If $\det(A) > 0$, the transformation preserves orientation (no reflection).
- If $\det(A) < 0$, the transformation flips orientation (a mirroring occurs).
- If $\det(A) = 0$, space is collapsed: the entire plane maps onto a line or a point, and the transformation is not reversible.

Think of a square drawn on a rubber sheet. An invertible matrix stretches and skews the sheet but does not tear it. The inverse stretches it back to its original shape. A singular matrix collapses the sheet into a line — you cannot recover the original shape from a line, so no inverse exists.

## Common Mistakes

1. **Assuming every matrix has an inverse**: Only square matrices with non-zero determinant are invertible. Singular matrices ($\det = 0$) have no inverse.

2. **Confusing $A^{-1}$ with $A^T$**: The inverse and transpose are different operations. They are only equal for orthogonal matrices ($Q^{-1} = Q^T$), which is a special case.

3. **Forgetting the $\frac{1}{\det(A)}$ factor in the $2 \times 2$ formula**: The most common error — the inverse formula requires dividing by the determinant. Without it, you get $AA^{-1} = \det(A) I$, not $I$.

4. **Misplacing signs in the $2 \times 2$ formula**: The sign pattern is $\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$. The $a$ and $d$ keep their positions (no sign change), while $b$ and $c$ swap positions and change sign.

5. **Applying the product reversal incorrectly**: $(AB)^{-1} = B^{-1} A^{-1}$, not $A^{-1} B^{-1}$. The order reverses.

6. **Writing the inverse of a diagonal matrix incorrectly**: $D = \operatorname{diag}(d_1, \ldots, d_n)$ has inverse $\operatorname{diag}(1/d_1, \ldots, 1/d_n)$. Beginners sometimes put the reciprocals in the wrong order or forget it is still diagonal.

7. **Assuming solving $Ax = b$ always requires $A^{-1}$**: For large systems, computing $A^{-1}$ explicitly is wasteful and numerically unstable. Gaussian elimination or factorisation (LU, Cholesky) is preferred.

8. **Ignoring numerical stability**: Computing inverses on a computer introduces rounding errors. A matrix that is theoretically invertible can be nearly singular in practice (ill-conditioned), causing large numerical errors.

9. **Thinking $(A+B)^{-1} = A^{-1} + B^{-1}$**: This is false in general. There is no simple formula for the inverse of a sum.

## Interview Questions

### Beginner

1. What is the inverse of a matrix, and what notation is used for it?
2. What condition must hold for a square matrix to be invertible?
3. What is the inverse of the $2 \times 2$ identity matrix?
4. Compute the inverse of $\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$.
5. If $A$ is invertible, what is $A A^{-1}$ equal to?

### Intermediate

1. Derive the formula for the inverse of a $2 \times 2$ matrix. Show the calculation that verifies $A A^{-1} = I$.
2. State and prove the property $(AB)^{-1} = B^{-1} A^{-1}$.
3. Show that if $A$ is invertible, then $A^T$ is also invertible and $(A^T)^{-1} = (A^{-1})^T$.
4. How would you solve $Ax = b$ using an inverse? Under what circumstances would you avoid computing the inverse explicitly?
5. For a diagonal matrix $D = \operatorname{diag}(d_1, d_2, d_3)$, what is $D^{-1}$? What happens if any $d_i = 0$?

### Advanced

1. Explain why the normal equations $\hat{w} = (X^T X)^{-1} X^T y$ become computationally infeasible for very large feature spaces, and describe how gradient descent provides an alternative.
2. What is an ill-conditioned matrix? How can you detect ill-conditioning, and what is the relationship between the condition number and the reliability of the inverse?
3. The Sherman–Morrison formula states: $(A + uv^T)^{-1} = A^{-1} - \frac{A^{-1} uv^T A^{-1}}{1 + v^T A^{-1} u}$. Explain when this formula is useful in practice (e.g., in online learning or incremental updates).

## Practice Problems

### Easy - 5 Questions

1. Find $A^{-1}$ for $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$.
2. Find $B^{-1}$ for $B = \begin{pmatrix} 5 & 0 \\ 0 & 5 \end{pmatrix}$.
3. Is the matrix $C = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$ invertible? Why or why not?
4. What is $I_3^{-1}$?
5. If $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$ and $\det(A) = 7$, write the general formula for $A^{-1}$.

### Medium - 5 Questions

1. Solve the system using the inverse matrix:
   $$
   \begin{cases}
   x + 2y = 5 \\
   3x + 4y = 11
   \end{cases}
   $$

2. Given $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$, find $A^{-1}$. Then compute $A^n$ for any positive integer $n$ and use it to find $(A^n)^{-1}$.

3. Show that $\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$ is invertible for all $\theta$, and find its inverse.

4. Let $A$ be an invertible matrix. Prove that $\det(A^{-1}) = 1 / \det(A)$.
5. If $A^3 = I$, what is $A^{-1}$? Express your answer in terms of $A$.

### Hard - 3 Questions

1. For the linear regression dataset below, compute $\hat{w} = (X^T X)^{-1} X^T y$:
   $$
   X = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \quad y = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}
   $$

2. Prove that if $A$ and $B$ are invertible $n \times n$ matrices, then $A + B$ need not be invertible. Construct a counterexample.

3. A matrix $A$ is called **idempotent** if $A^2 = A$. Prove that if $A$ is idempotent and invertible, then $A = I$. Then find all idempotent $2 \times 2$ matrices that are not invertible.

## Solutions

### Easy Solutions

1. $\det(A) = 1\cdot 4 - 2\cdot 3 = 4 - 6 = -2$.
   $$
   A^{-1} = \frac{1}{-2} \begin{pmatrix} 4 & -2 \\ -3 & 1 \end{pmatrix} = \begin{pmatrix} -2 & 1 \\ \frac{3}{2} & -\frac{1}{2} \end{pmatrix}
   $$

2. $\det(B) = 5\cdot 5 - 0 = 25$.
   $$
   B^{-1} = \frac{1}{25} \begin{pmatrix} 5 & 0 \\ 0 & 5 \end{pmatrix} = \begin{pmatrix} \frac{1}{5} & 0 \\ 0 & \frac{1}{5} \end{pmatrix}
   $$
   
   Alternatively, note that $B = 5I_2$, so $B^{-1} = \frac{1}{5} I_2$.

3. $\det(C) = 1\cdot 2 - 1\cdot 2 = 0$, so $C$ is not invertible (singular). The second row is twice the first.

4. $I_3^{-1} = I_3$. The identity is its own inverse.

5. $A^{-1} = \frac{1}{7} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.

### Medium Solutions

1. $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $\det(A) = -2$.
   $$
   A^{-1} = \frac{1}{-2} \begin{pmatrix} 4 & -2 \\ -3 & 1 \end{pmatrix} = \begin{pmatrix} -2 & 1 \\ \frac{3}{2} & -\frac{1}{2} \end{pmatrix}
   $$
   $$
   x = A^{-1}b = \begin{pmatrix} -2 & 1 \\ \frac{3}{2} & -\frac{1}{2} \end{pmatrix} \begin{pmatrix} 5 \\ 11 \end{pmatrix} = \begin{pmatrix} -10 + 11 \\ \frac{15}{2} - \frac{11}{2} \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
   $$
   So $x = 1$, $y = 2$.

2. $\det(A) = 1\cdot 1 - 1\cdot 0 = 1$.
   $$
   A^{-1} = \frac{1}{1} \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}
   $$
   
   By induction, $A^n = \begin{pmatrix} 1 & n \\ 0 & 1 \end{pmatrix}$.
   
   Then $(A^n)^{-1} = \begin{pmatrix} 1 & -n \\ 0 & 1 \end{pmatrix}$, which matches $(A^{-1})^n = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}^n$.

3. Let $R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$. This is a rotation matrix.
   $\det(R) = \cos^2\theta + \sin^2\theta = 1 \neq 0$ for all $\theta$.
   
   The inverse is the rotation by $-\theta$:
   $$
   R^{-1} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} = \begin{pmatrix} \cos(-\theta) & -\sin(-\theta) \\ \sin(-\theta) & \cos(-\theta) \end{pmatrix}
   $$
   
   Note also that $R^{-1} = R^T$ (rotation matrices are orthogonal).

4. $\det(A A^{-1}) = \det(I) = 1$. But $\det(A A^{-1}) = \det(A) \det(A^{-1})$. Therefore:
   $$
   \det(A) \det(A^{-1}) = 1 \implies \det(A^{-1}) = \frac{1}{\det(A)}
   $$

5. If $A^3 = I$, multiply both sides on the left by $A^{-1}$: $A^2 = A^{-1}$. So $A^{-1} = A^2$.

### Hard Solutions

1. $X$ is $3 \times 2$, $y$ is $3 \times 1$.
   
   $X^T X = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}$.
   
   $\det(X^T X) = 3\cdot 5 - 3\cdot 3 = 15 - 9 = 6$.
   
   $(X^T X)^{-1} = \frac{1}{6} \begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix} = \begin{pmatrix} \frac{5}{6} & -\frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{pmatrix}$.
   
   $X^T y = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix} = \begin{pmatrix} 7 \\ 10 \end{pmatrix}$.
   
   $\hat{w} = \begin{pmatrix} \frac{5}{6} & -\frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{pmatrix} \begin{pmatrix} 7 \\ 10 \end{pmatrix} = \begin{pmatrix} \frac{35}{6} - 5 \\ -\frac{7}{2} + 5 \end{pmatrix} = \begin{pmatrix} \frac{35}{6} - \frac{30}{6} \\ -\frac{7}{2} + \frac{10}{2} \end{pmatrix} = \begin{pmatrix} \frac{5}{6} \\ \frac{3}{2} \end{pmatrix}$.
   
   So $\hat{w} = (\frac{5}{6}, \frac{3}{2})$, giving the best-fit line $y = \frac{5}{6} + \frac{3}{2}x$.

2. **Counterexample**: Let $A = I_2$ (invertible) and $B = -I_2$ (also invertible). Then $A + B = 0$, the zero matrix, which is singular (determinant 0). Therefore $A + B$ need not be invertible.

3. **Proof**: Assume $A$ is idempotent ($A^2 = A$) and invertible. Multiply both sides by $A^{-1}$:
   $$
   A^{-1} A^2 = A^{-1} A \implies (A^{-1} A) A = I \implies I A = I \implies A = I
   $$
   
   For non-invertible idempotent $2 \times 2$ matrices (projection matrices), examples include:
   - $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$: projects onto the x-axis.
   - $\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$: projects onto the y-axis.
   - $\begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & \frac{1}{2} \end{pmatrix}$: projects onto the line $y = x$.
   
   All projection matrices satisfy $P^2 = P$ and are singular (unless $P = I$).

## Related Concepts

- **Identity Matrix**: $A A^{-1} = I$ — the identity is central to the definition
- **Determinant**: A non-zero determinant is the condition for invertibility
- **Transpose**: $(A^T)^{-1} = (A^{-1})^T$
- **Linear systems**: $Ax = b$ solved by $x = A^{-1}b$
- **Matrix factorisation**: LU, QR, and Cholesky decompositions provide efficient alternatives to explicit inversion
- **Singular matrix**: A matrix with determinant zero, having no inverse

## Next Concepts

- **LU Decomposition**: A more efficient way to solve $Ax = b$ without computing $A^{-1}$
- **Eigenvalues and Eigenvectors**: $A v = \lambda v$; invertibility requires no zero eigenvalue
- **Singular Value Decomposition (SVD)**: $A = U \Sigma V^T$; the inverse is $A^{-1} = V \Sigma^{-1} U^T$
- **Condition Number**: Measures how close a matrix is to being singular (ill-conditioned)
- **Moore–Penrose Pseudoinverse**: A generalisation of the inverse for non-square or singular matrices

## Summary

The inverse of a square matrix $A$ is the unique matrix $A^{-1}$ such that $A A^{-1} = A^{-1} A = I$. For $2 \times 2$ matrices, $A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$, provided $\det(A) \neq 0$. Inverse matrices allow solving linear systems directly via $x = A^{-1}b$, but for large systems this approach is computationally expensive and numerically unstable. In machine learning, inverses appear in the normal equations for linear regression and in regularisation techniques, though gradient-based optimisation is preferred for large-scale problems.

## Key Takeaways

- $A^{-1}$ exists iff $\det(A) \neq 0$ (the matrix is non-singular)
- $A A^{-1} = A^{-1} A = I$
- For $2 \times 2$: $A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$
- $(AB)^{-1} = B^{-1} A^{-1}$ — order reverses
- $(A^T)^{-1} = (A^{-1})^T$
- Solving $Ax = b$ via $x = A^{-1}b$ is theoretically elegant but practically inefficient for large $n$
- The normal equations $\hat{w} = (X^T X)^{-1} X^T y$ are a closed-form solution for linear regression, but gradient descent is preferred for large feature spaces
