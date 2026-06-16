# Concept: Linear Dependence

## Concept ID

MATH-035

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Domain

Linear Algebra

## Learning Objectives

1. Define linear dependence for a set of vectors.
2. Identify whether a set of vectors is linearly dependent using multiple methods.
3. Find explicit dependence relations among dependent vectors.
4. Interpret linear dependence geometrically in $\mathbb{R}^2$ and $\mathbb{R}^3$.
5. Explain the consequences of linear dependence in machine learning contexts (multicollinearity, rank deficiency).
6. Connect linear dependence to dimensionality reduction and feature selection.

## Prerequisites

- Linear independence (MATH-034)
- Systems of linear equations and Gaussian elimination
- Matrix rank
- Determinants
- Basic vector geometry (lines and planes)

## Definition

A finite set of vectors $\{v_1, v_2, \dots, v_n\}$ in a vector space $V$ is called **linearly dependent** if there exists a nontrivial linear combination that equals the zero vector. That is, there exist scalars $c_1, c_2, \dots, c_n$, not all zero, such that

$$
c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = \mathbf{0}.
$$

A nontrivial combination means at least one $c_i \neq 0$.

Equivalently, a set is linearly dependent if at least one vector in the set can be expressed as a linear combination of the others. Specifically, if $c_k \neq 0$ in the dependence relation, then

$$
v_k = -\frac{c_1}{c_k} v_1 - \cdots - \frac{c_{k-1}}{c_k} v_{k-1} - \frac{c_{k+1}}{c_k} v_{k+1} - \cdots - \frac{c_n}{c_k} v_n.
$$

## Intuition

Linear dependence captures the idea of **redundancy**. When vectors are dependent, at least one of them is "extra" — it lies in the span of the others and contributes no new direction.

Geometric intuition:
- In $\mathbb{R}^2$, two vectors are dependent if they are collinear (lie on the same line through the origin).
- In $\mathbb{R}^3$, three vectors are dependent if they are coplanar (lie in the same plane through the origin).
- In any space, $n$ vectors are dependent if they lie in a subspace of dimension less than $n$.

Think of dependence as a "wasted dimension" — the set does not span a space as large as the number of vectors would suggest.

## Why This Concept Matters

Linear dependence is the natural counterpart to independence and is equally fundamental:

- **Redundancy detection:** Dependence reveals when features, variables, or measurements carry redundant information.
- **Rank deficiency:** Dependent columns in a matrix reduce its rank, making the matrix singular and causing problems in linear systems.
- **Multicollinearity:** In statistics and machine learning, dependent predictor variables cause unstable parameter estimates.
- **Dimensionality reduction:** Dependence motivates reducing the number of features/dimensions without losing information.
- **Overcomplete representations:** In signal processing, dependence is deliberately introduced in overcomplete dictionaries to enable sparse representations.
- **Consistency of linear systems:** A set of dependent vectors in the columns of $A$ means $Ax = b$ may have infinitely many solutions or none.

## Historical Background

The concept of linear dependence developed alongside linear independence in the 18th–19th centuries:

- **Augustin-Louis Cauchy** (1789–1857) used determinants to detect dependence among vectors and functions (the Cauchy determinant).
- **Hermann Grassmann** (1809–1877) formally introduced the concept of linear dependence in his *Ausdehnungslehre* (1844, 1862), defining it in terms of linear relations.
- **Giuseppe Peano** (1858–1932) incorporated linear dependence into his axiomatization of vector spaces (1888), providing the modern definition.
- **David Hilbert** (1862–1943) extended these ideas to infinite-dimensional spaces in his work on integral equations and functional analysis.
- **Richard Dedekind** (1831–1916) contributed to the understanding of linear dependence in the context of modules over rings.

The term "linear dependence" became standard in the early 20th century as linear algebra developed into its modern form.

## Real World Examples

1. **Traffic Flow Analysis:** In a road network, the vector of traffic counts on different roads often exhibits dependence. For example, if Road A feeds into Roads B and C, then the traffic vector on A is (approximately) the sum of the vectors on B and C. This dependence must be accounted for in traffic prediction models.

2. **Image Compression:** In the JPEG algorithm, the discrete cosine transform (DCT) basis vectors are linearly independent. But when compressing, high-frequency components that are "almost dependent" (i.e., nearly zero coefficients) are discarded, exploiting near-dependence to reduce file size.

3. **Portfolio Management:** In finance, the returns of stocks in the same sector (e.g., tech stocks) are often linearly dependent because they respond similarly to market conditions. Portfolio diversification seeks to include assets whose returns are as independent as possible.

4. **Genomics:** Gene expression levels for different genes can be linearly dependent when genes are co-regulated. Identifying these dependence relationships helps reconstruct gene regulatory networks. Dependence among gene expression vectors signals shared biological pathways.

5. **Sensor Networks:** Multiple sensors measuring the same physical quantity produce linearly dependent readings (each reading is approximately a scalar multiple of the true value plus noise). Dependence is used to detect sensor faults and to fuse redundant measurements.

## AI/ML Relevance

1. **Multicollinearity in Linear Regression:** When predictor variables in linear regression are linearly dependent (or highly correlated), the OLS estimates become unstable with large variance. The matrix $X^TX$ is ill-conditioned or singular, making $(X^TX)^{-1}$ unreliable. Detecting dependence among features is crucial before fitting regression models.

2. **Redundant Features:** In machine learning datasets, linearly dependent features add no predictive information. For example, if "temperature in Celsius" and "temperature in Fahrenheit" are both included, one is redundant. Feature selection algorithms aim to remove such dependent features.

3. **Dimensionality Reduction Motivation:** Linear dependence among high-dimensional data points motivates techniques like PCA, which finds a basis of independent principal components. The dependence structure reveals the intrinsic dimensionality of the data.

4. **Regularization:** Ridge regression ($L_2$ penalty) and Lasso ($L_1$ penalty) are designed to handle near-dependent features by shrinking coefficients, effectively stabilizing the solution when $X^TX$ is nearly singular.

5. **Autoencoders:** Undercomplete autoencoders (with a bottleneck layer smaller than the input dimension) learn a compressed representation by exploiting dependence among input features. The bottleneck forces the network to capture only the most important independent factors of variation.

6. **Sparse Coding:** Overcomplete dictionaries deliberately use more atoms than the input dimension, so the atoms are necessarily dependent. The goal is to find a sparse representation — using few atoms — which leverages the dependence structure of the dictionary.

7. **Graph Neural Networks:** Node features in graphs often exhibit dependence due to homophily (connected nodes share similar features). This dependence is both a challenge (redundancy) and an opportunity (message passing exploits it).

## Mathematical Explanation

### Finding Dependence Relations

Given a set of vectors $\{v_1, \dots, v_n\}$, to find a dependence relation:

1. Form a matrix $A$ with the vectors as **columns**.
2. Solve the homogeneous system $A c = 0$, where $c = (c_1, \dots, c_n)^T$.
3. Any nontrivial solution gives a dependence relation.

Alternatively, row reduce the matrix and identify columns without pivots — these correspond to dependent vectors that can be expressed in terms of the pivot columns.

### Geometric Interpretation

- **In $\mathbb{R}^2$:** Two vectors $u$ and $v$ are dependent $\iff$ one is a scalar multiple of the other $\iff$ they lie on the same line. Three or more vectors in $\mathbb{R}^2$ are always dependent.
- **In $\mathbb{R}^3$:** Three vectors are dependent $\iff$ they lie in the same plane through the origin $\iff$ the parallelepiped they span has zero volume $\iff$ the determinant of the $3 \times 3$ matrix they form is zero. Four or more vectors in $\mathbb{R}^3$ are always dependent.

### Relation to Rank

For a matrix $A$ with columns $\{v_1, \dots, v_n\}$:

$$
\text{columns are dependent} \iff \text{rank}(A) < n
$$

The nullity (dimension of the null space) of $A$ equals $n - \text{rank}(A)$ and represents the number of independent dependence relations among the columns.

## Formula(s)

**Linear dependence condition:**

There exist $c_1, \dots, c_n$, not all zero, such that

$$
c_1 v_1 + c_2 v_2 + \cdots + c_n v_n = \mathbf{0}
$$

**Vector as combination of others:**

If $c_k \neq 0$, then:

$$
v_k = -\sum_{i \neq k} \frac{c_i}{c_k} v_i
$$

**Rank test for dependence:**

$$
\text{columns of } A \text{ are dependent} \iff \text{rank}(A) < n
$$

**Determinant test (square):**

$$
\text{columns of } A_{n \times n} \text{ are dependent} \iff \det(A) = 0
$$

**Null space dimension:**

$$
\dim(\text{null}(A)) = n - \text{rank}(A)
$$

## Properties

1. Any set containing the zero vector is linearly dependent.
2. Any set where one vector is a scalar multiple of another is dependent.
3. A set of two nonzero vectors is dependent iff they are scalar multiples of each other.
4. If a subset is dependent, the whole set is dependent.
5. In $\mathbb{R}^m$, any set with more than $m$ vectors is dependent.
6. The columns of a matrix are dependent iff there exists a nonzero vector $x$ such that $Ax = 0$.
7. Multiplication by an invertible matrix preserves linear dependence/independence (i.e., does not change the dependence structure).
8. Linear dependence is equivalent to the existence of a nontrivial linear relation, and to at least one vector being a combination of the others (assuming more than one vector and no zero vectors).

## Step-by-Step Worked Examples

### Example 1: Detecting Dependence in $\mathbb{R}^2$

Determine whether $v_1 = (1, 2)$ and $v_2 = (2, 4)$ are linearly dependent.

**Solution:**

Check if one is a scalar multiple of the other: $v_2 = 2 \cdot (1, 2) = 2v_1$.

Set up $c_1(1, 2) + c_2(2, 4) = (0, 0)$:

$$
\begin{cases}
c_1 + 2c_2 = 0 \\
2c_1 + 4c_2 = 0
\end{cases}
$$

The second equation is $2$ times the first, so the system reduces to $c_1 + 2c_2 = 0$.

Choose $c_2 = 1$, then $c_1 = -2$. Check: $-2(1,2) + 1(2,4) = (-2+2, -4+4) = (0,0)$.

Since a nontrivial solution exists, the vectors are dependent.

The dependence relation is $-2v_1 + v_2 = 0$, so $v_2 = 2v_1$.

**Therefore, $\{v_1, v_2\}$ is linearly dependent.**

### Example 2: Finding Dependence Among Three Vectors in $\mathbb{R}^3$

Determine whether $v_1 = (1, 0, 2)$, $v_2 = (2, 1, 1)$, $v_3 = (0, 1, -3)$ are dependent. If so, find a dependence relation.

**Solution:**

Form the matrix with vectors as columns:

$$
A = \begin{bmatrix}
1 & 2 & 0 \\
0 & 1 & 1 \\
2 & 1 & -3
\end{bmatrix}
$$

Compute the determinant:

$$
\begin{aligned}
\det(A) &= 1 \cdot \det\begin{bmatrix}1 & 1 \\ 1 & -3\end{bmatrix} - 2 \cdot \det\begin{bmatrix}0 & 1 \\ 2 & -3\end{bmatrix} + 0 \\
&= 1(1\cdot(-3) - 1\cdot1) - 2(0\cdot(-3) - 1\cdot2) \\
&= 1(-3 - 1) - 2(0 - 2) \\
&= -4 + 4 \\
&= 0
\end{aligned}
$$

Since $\det(A) = 0$, the vectors are dependent.

Now find the dependence relation. Solve $c_1 v_1 + c_2 v_2 + c_3 v_3 = 0$:

$$
\begin{cases}
c_1 + 2c_2 = 0 \\
c_2 + c_3 = 0 \\
2c_1 + c_2 - 3c_3 = 0
\end{cases}
$$

From equation (1): $c_1 = -2c_2$.
From equation (2): $c_3 = -c_2$.

Substitute into equation (3):

$$
2(-2c_2) + c_2 - 3(-c_2) = -4c_2 + c_2 + 3c_2 = 0
$$

So equation (3) is automatically satisfied. Choose $c_2 = 1$, then $c_1 = -2$, $c_3 = -1$.

Dependence relation:

$$
-2v_1 + 1v_2 - 1v_3 = \mathbf{0}
$$

Thus $v_2 = 2v_1 + v_3$.

Check: $2(1,0,2) + (0,1,-3) = (2,0,4) + (0,1,-3) = (2,1,1) = v_2$. ✓

### Example 3: Dependence Among Four Vectors in $\mathbb{R}^3$

Determine whether $v_1 = (1, 1, 0)$, $v_2 = (0, 1, 1)$, $v_3 = (1, 0, 1)$, $v_4 = (1, 1, 1)$ are dependent. If so, express one as a combination of the others.

**Solution:**

Since we have 4 vectors in $\mathbb{R}^3$, by the pigeonhole principle they must be dependent. Let's find a dependence relation.

Form the matrix with vectors as columns:

$$
A = \begin{bmatrix}
1 & 0 & 1 & 1 \\
1 & 1 & 0 & 1 \\
0 & 1 & 1 & 1
\end{bmatrix}
$$

Row reduce to echelon form:

$R_2 \leftarrow R_2 - R_1$:

$$
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & -1 & 0 \\
0 & 1 & 1 & 1
\end{bmatrix}
$$

$R_3 \leftarrow R_3 - R_2$:

$$
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & -1 & 0 \\
0 & 0 & 2 & 1
\end{bmatrix}
$$

The pivots are in columns 1, 2, 3. Column 4 has no pivot, so $v_4$ is dependent on $v_1, v_2, v_3$.

Solve $c_1 v_1 + c_2 v_2 + c_3 v_3 + c_4 v_4 = 0$. From the row reduction, we can solve for $c_1, c_2, c_3$ in terms of $c_4$.

Using the reduced matrix, let $c_4 = t$. Then:

From $R_3$: $2c_3 + c_4 = 0 \implies 2c_3 + t = 0 \implies c_3 = -\frac{t}{2}$.

From $R_2$: $c_2 - c_3 = 0 \implies c_2 = c_3 = -\frac{t}{2}$.

From $R_1$: $c_1 + c_3 + c_4 = 0 \implies c_1 - \frac{t}{2} + t = 0 \implies c_1 = -\frac{t}{2}$.

Choose $t = 2$ to avoid fractions: $c_1 = -1$, $c_2 = -1$, $c_3 = -1$, $c_4 = 2$.

Dependence relation:

$$
-v_1 - v_2 - v_3 + 2v_4 = \mathbf{0}
$$

Therefore $v_4 = \frac{1}{2}(v_1 + v_2 + v_3)$.

Check: $\frac{1}{2}[(1,1,0) + (0,1,1) + (1,0,1)] = \frac{1}{2}(2,2,2) = (1,1,1) = v_4$. ✓

### Example 4: Polynomial Dependence

Determine whether the polynomials $p_1(x) = 1 + x + x^2$, $p_2(x) = 2 - x$, $p_3(x) = 1 + 4x + 3x^2$ are linearly dependent in $P_2$ (polynomials of degree $\leq 2$).

**Solution:**

Set up $c_1 p_1 + c_2 p_2 + c_3 p_3 = 0$ (the zero polynomial):

$$
c_1(1 + x + x^2) + c_2(2 - x) + c_3(1 + 4x + 3x^2) = 0
$$

Collect coefficients for each power of $x$:

- $x^0$ (constant): $c_1 + 2c_2 + c_3 = 0$
- $x^1$: $c_1 - c_2 + 4c_3 = 0$
- $x^2$: $c_1 + 0c_2 + 3c_3 = 0$

Write as a matrix system:

$$
\begin{bmatrix}
1 & 2 & 1 \\
1 & -1 & 4 \\
1 & 0 & 3
\end{bmatrix}
\begin{bmatrix} c_1 \\ c_2 \\ c_3 \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

Compute determinant:

$$
\begin{aligned}
\det &= 1((-1)\cdot3 - 4\cdot0) - 2(1\cdot3 - 4\cdot1) + 1(1\cdot0 - (-1)\cdot1) \\
&= 1(-3 - 0) - 2(3 - 4) + 1(0 + 1) \\
&= -3 - 2(-1) + 1 \\
&= -3 + 2 + 1 = 0
\end{aligned}
$$

The determinant is zero, so the vectors are dependent.

Solve the system. From $c_1 + 3c_3 = 0$, we have $c_1 = -3c_3$.

From $c_1 - c_2 + 4c_3 = 0$: $-3c_3 - c_2 + 4c_3 = 0 \implies -c_2 + c_3 = 0 \implies c_2 = c_3$.

From $c_1 + 2c_2 + c_3 = 0$: $-3c_3 + 2c_3 + c_3 = 0 \implies 0 = 0$.

Choose $c_3 = 1$: $c_1 = -3$, $c_2 = 1$, $c_3 = 1$.

Dependence relation:

$$
-3(1 + x + x^2) + 1(2 - x) + 1(1 + 4x + 3x^2) = 0
$$

Check: $(-3 - 3x - 3x^2) + (2 - x) + (1 + 4x + 3x^2) = (-3+2+1) + (-3-1+4)x + (-3+0+3)x^2 = 0$. ✓

## Visual Interpretation

In $\mathbb{R}^2$:
- Dependent vectors lie on the same line through the origin.
- The area of the parallelogram formed by two dependent vectors is zero.

In $\mathbb{R}^3$:
- Three dependent vectors lie in the same plane through the origin (or on the same line).
- The volume of the parallelepiped they span is zero.
- The determinant of the $3 \times 3$ matrix they form is zero.

In higher dimensions:
- Dependent vectors lie in a subspace of dimension less than the number of vectors.
- The $n$-dimensional volume spanned by the vectors is zero.
- The Gram matrix $G_{ij} = v_i \cdot v_j$ has zero determinant.

The "shadow" intuition: dependent vectors cast a shadow that is thinner than the number of vectors would suggest — they occupy fewer dimensions.

## Common Mistakes

1. **Thinking "dependent" means "all vectors are combinations of each other":** Dependence only guarantees that *at least one* vector is a combination of the others. For example, $\{(1,0), (2,0), (0,1)\}$ is dependent (first two are multiples), but $(0,1)$ cannot be written as a combination of the other two alone.

2. **Confusing dependence with correlation:** Statistical correlation is a measure of linear relationship between random variables. Linear dependence is an exact algebraic condition. In practice, features may be "almost dependent" (highly correlated) without being exactly dependent.

3. **Assuming dependence means the set is useless:** Overcomplete dictionaries in sparse coding are deliberately dependent. Dependence is not always bad — it depends on the application.

4. **Forgetting that a set of one nonzero vector is always independent:** A single nonzero vector's only scalar multiple that gives zero is $0 \cdot v = 0$, so $c$ must be $0$. Independence.

5. **Confusing kernel (null space) dimension with dependence:** The null space of $A$ contains all dependence relations among columns. A nonzero null space vector corresponds to a dependence relation. If $\dim(\text{null}(A)) = d$, there are $d$ independent dependence relations.

6. **Thinking row reduction must be done to reduced echelon form:** Row echelon form (not necessarily reduced) is sufficient to identify pivot columns and thus determine which columns are dependent.

7. **Assuming $n$ vectors in $\mathbb{R}^m$ with $n \leq m$ are independent:** Counterexample: $(1,0,0)$ and $(2,0,0)$ in $\mathbb{R}^3$ — two vectors in $\mathbb{R}^3$, but they are dependent.

## Interview Questions

### Beginner

1. **Q:** What does it mean for a set of vectors to be linearly dependent?
   **A:** There exists a nontrivial linear combination of the vectors that equals the zero vector. Equivalently, at least one vector can be written as a combination of the others.

2. **Q:** Can a set of two nonzero vectors be dependent? If so, how?
   **A:** Yes, if one is a scalar multiple of the other. For example, $(1,2)$ and $(2,4)$ are dependent because $(2,4) = 2(1,2)$.

3. **Q:** Is a set containing the zero vector always dependent?
   **A:** Yes, because $1 \cdot \mathbf{0} = \mathbf{0}$ is a nontrivial dependence relation.

4. **Q:** If you have 5 vectors in $\mathbb{R}^3$, what can you say about their dependence?
   **A:** They must be dependent because the maximum number of independent vectors in $\mathbb{R}^3$ is 3.

5. **Q:** What is the relationship between linear dependence and the determinant of a square matrix?
   **A:** The columns (or rows) of a square matrix are linearly dependent if and only if the determinant is zero.

### Intermediate

1. **Q:** Explain how Gaussian elimination can be used to detect linear dependence.
   **A:** Form a matrix with the vectors as columns and row reduce to echelon form. If every column has a pivot, the columns are independent. If any column lacks a pivot, that column (vector) is dependent on the columns with pivots to its left.

2. **Q:** In linear regression, what problems does multicollinearity cause? How does it relate to linear dependence?
   **A:** Multicollinearity occurs when predictor variables are approximately (or exactly) linearly dependent. It causes $X^TX$ to be near-singular, inflating the variance of coefficient estimates and making them unstable. Exact dependence means $X^TX$ is singular and the OLS solution is not unique.

3. **Q:** Given vectors $v_1 = (1,2,3)$, $v_2 = (2,4,6)$, $v_3 = (0,1,1)$, does there exist a nontrivial dependence relation? Find one.
   **A:** $v_2 = 2v_1$, so $2v_1 - v_2 = 0$ is a dependence relation. The set is dependent.

4. **Q:** How can you find a basis for the column space of a matrix with dependent columns?
   **A:** Row reduce the matrix to echelon form. The columns corresponding to pivot positions in the original matrix form a basis for the column space. The non-pivot columns are dependent on the pivot columns.

5. **Q:** If $A$ is a $5 \times 5$ matrix with $\det(A) = 0$, what can you say about the columns of $A$? About the null space?
   **A:** The columns are linearly dependent. The null space $\text{null}(A)$ is nontrivial (contains nonzero vectors), and its dimension (nullity) is at least 1.

### Advanced

1. **Q:** Prove that if $\{v_1, v_2, v_3\}$ is linearly dependent and $\{v_1, v_2\}$ is independent, then $v_3$ is a linear combination of $v_1$ and $v_2$.
   **A:** Since $\{v_1, v_2, v_3\}$ is dependent, there exist $c_1, c_2, c_3$, not all zero, with $c_1 v_1 + c_2 v_2 + c_3 v_3 = 0$. If $c_3 = 0$, then $c_1 v_1 + c_2 v_2 = 0$ with not both zero, contradicting independence of $\{v_1, v_2\}$. Hence $c_3 \neq 0$, and $v_3 = -(c_1/c_3)v_1 - (c_2/c_3)v_2$, a linear combination of $v_1$ and $v_2$.

2. **Q:** In the context of dimensionality reduction, explain why the rank of the data matrix equals the number of "true" features plus noise.
   **A:** If a dataset has $n$ samples with $m$ features, the data matrix $X$ is $n \times m$. If the features are exactly linearly dependent, $\text{rank}(X) < m$, revealing the true dimensionality. In practice, noise makes $X$ full rank, but the effective rank (number of significant singular values) indicates the intrinsic dimension. PCA finds the independent directions of maximum variance, effectively discovering the low-dimensional structure hidden in the near-dependent data.

3. **Q:** Let $A$ be an $m \times n$ matrix ($m < n$). Show that the columns of $A$ must be linearly dependent. What does this imply about the system $Ax = b$?
   **A:** The columns of $A$ are vectors in $\mathbb{R}^m$. Since $n > m$, the pigeonhole principle implies they are dependent. For $Ax = b$: either $b$ is not in the column space (no solution) or $b$ is in the column space (infinitely many solutions, since the null space has dimension $n - \text{rank}(A) \geq n - m > 0$).

## Practice Problems

### Easy - 5 Questions

1. Determine whether $v_1 = (3, 6)$ and $v_2 = (1, 2)$ are linearly dependent.

2. Determine whether $v_1 = (0, 0, 0)$ and $v_2 = (1, 2, 3)$ are linearly dependent.

3. Determine whether $v_1 = (2, -4)$ and $v_2 = (-1, 2)$ are linearly dependent.

4. Determine whether $v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (1, 1)$ are linearly dependent.

5. How many vectors in $\mathbb{R}^2$ are needed to guarantee linear dependence?

### Medium - 5 Questions

1. Determine whether $v_1 = (1, 1, 2)$, $v_2 = (1, 2, 1)$, $v_3 = (2, 3, 3)$ are linearly dependent. If so, find a dependence relation.

2. Determine whether $v_1 = (1, 0, 1, 0)$, $v_2 = (0, 1, 0, 1)$, $v_3 = (1, 1, 1, 1)$, $v_4 = (1, 0, 0, 1)$ are linearly dependent in $\mathbb{R}^4$.

3. For what value(s) of $k$ are the vectors $v_1 = (1, k, 1)$, $v_2 = (k, 1, 1)$, $v_3 = (1, 1, k)$ linearly dependent?

4. Show that $\{\sin x, \cos x, \sin 2x\}$ is linearly independent in the vector space of functions on $\mathbb{R}$.

5. Given dependent vectors $\{v_1, v_2, v_3\}$ with $v_1 = 2v_2 - v_3$, write a nontrivial linear combination that equals zero.

### Hard - 3 Questions

1. Let $A$ be an $n \times n$ matrix such that $A^2 = A$ (idempotent, also called a projection matrix). Show that if $v$ is any nonzero vector, then $\{v, Av\}$ is linearly dependent if and only if $Av = v$ or $Av = 0$.

2. Consider the vectors $v_1 = (1, 2, 3, 4)$, $v_2 = (2, 3, 4, 5)$, $v_3 = (3, 4, 5, 6)$, $v_4 = (4, 5, 6, 7)$ in $\mathbb{R}^4$. Show they are linearly dependent and find the dimension and a basis for their span.

3. In the vector space $M_{2 \times 2}$ of $2 \times 2$ matrices, determine whether the following set is linearly dependent. If so, find all dependence relations.

$$
\left\{
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},
\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix},
\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix},
\begin{bmatrix} 4 & 5 \\ 6 & 7 \end{bmatrix}
\right\}
$$

## Solutions

### Easy Solutions

1. $v_1 = 3v_2$ (or $v_1 = 3 \cdot (1, 2) = (3, 6)$). Dependent.

2. Contains the zero vector. Dependent.

3. $v_1 = -2v_2$ (since $(-1, 2) = -\frac{1}{2}(2, -4)$). Equivalently, $-2v_1 - 4v_2 = 0$. Dependent.

4. Three vectors in $\mathbb{R}^2$ are always dependent. Specifically, $v_3 = v_1 + v_2$, so $v_1 + v_2 - v_3 = 0$.

5. 3 vectors in $\mathbb{R}^2$ guarantee dependence.

### Medium Solutions

1. Form the matrix:

$$
A = \begin{bmatrix}
1 & 1 & 2 \\
1 & 2 & 3 \\
2 & 1 & 3
\end{bmatrix}
$$

Compute determinant:

$$
\begin{aligned}
\det(A) &= 1(2\cdot3 - 3\cdot1) - 1(1\cdot3 - 3\cdot2) + 2(1\cdot1 - 2\cdot2) \\
&= 1(6 - 3) - 1(3 - 6) + 2(1 - 4) \\
&= 3 + 3 - 6 = 0
\end{aligned}
$$

Dependent. Solve $c_1 v_1 + c_2 v_2 + c_3 v_3 = 0$:

$$
\begin{cases}
c_1 + c_2 + 2c_3 = 0 \\
c_1 + 2c_2 + 3c_3 = 0 \\
2c_1 + c_2 + 3c_3 = 0
\end{cases}
$$

Subtract (1) from (2): $c_2 + c_3 = 0 \implies c_2 = -c_3$.
From (1): $c_1 - c_3 + 2c_3 = 0 \implies c_1 + c_3 = 0 \implies c_1 = -c_3$.
Check (3): $2(-c_3) + (-c_3) + 3c_3 = -2c_3 - c_3 + 3c_3 = 0$. ✓
Choose $c_3 = 1$: $c_1 = -1$, $c_2 = -1$.
Relation: $-v_1 - v_2 + v_3 = 0$, so $v_3 = v_1 + v_2$.

2. Form the matrix:

$$
A = \begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 1
\end{bmatrix}
$$

Row reduce: $R_3 \leftarrow R_3 - R_1$:

$$
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 0 & -1 \\
0 & 1 & 1 & 1
\end{bmatrix}
$$

$R_4 \leftarrow R_4 - R_2$:

$$
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 0 & -1 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

$R_4 \leftarrow R_4 + R_3$:

$$
\begin{bmatrix}
1 & 0 & 1 & 1 \\
0 & 1 & 1 & 0 \\
0 & 0 & 0 & -1 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

Rank = 3 < 4, so dependent. Column 3 has no pivot. Solve: from $R_3$: $-c_4 = 0 \implies c_4 = 0$. From $R_2$: $c_2 + c_3 = 0 \implies c_2 = -c_3$. From $R_1$: $c_1 + c_3 + c_4 = c_1 + c_3 = 0 \implies c_1 = -c_3$. Choose $c_3 = 1$: $c_1 = -1$, $c_2 = -1$, $c_4 = 0$. Relation: $-v_1 - v_2 + v_3 = 0$.

3. Form determinant:

$$
\det\begin{bmatrix}
1 & k & 1 \\
k & 1 & 1 \\
1 & 1 & k
\end{bmatrix}
$$

$$
\begin{aligned}
&= 1(1\cdot k - 1\cdot1) - k(k\cdot k - 1\cdot1) + 1(k\cdot1 - 1\cdot1) \\
&= 1(k - 1) - k(k^2 - 1) + 1(k - 1) \\
&= 2(k - 1) - k(k^2 - 1) \\
&= 2(k - 1) - k(k - 1)(k + 1) \\
&= (k - 1)(2 - k(k + 1)) \\
&= (k - 1)(2 - k^2 - k) \\
&= (k - 1)(-k^2 - k + 2) \\
&= -(k - 1)(k^2 + k - 2) \\
&= -(k - 1)(k + 2)(k - 1) \\
&= -(k - 1)^2(k + 2)
\end{aligned}
$$

Set to zero: $(k - 1)^2(k + 2) = 0 \implies k = 1$ or $k = -2$.

**For $k = 1$:** all three vectors are $(1, 1, 1)$, so $v_1 - v_2 = 0$ is a relation.

**For $k = -2$:** $v_1 = (1, -2, 1)$, $v_2 = (-2, 1, 1)$, $v_3 = (1, 1, -2)$. Find relation:

$$
\begin{cases}
c_1 - 2c_2 + c_3 = 0 \\
-2c_1 + c_2 + c_3 = 0 \\
c_1 + c_2 - 2c_3 = 0
\end{cases}
$$

Add all three equations: $0 = 0$. From (1): $c_1 = 2c_2 - c_3$. Substitute into (2): $-2(2c_2 - c_3) + c_2 + c_3 = -4c_2 + 2c_3 + c_2 + c_3 = -3c_2 + 3c_3 = 0 \implies c_2 = c_3$. Then $c_1 = 2c_2 - c_2 = c_2$. So $c_1 = c_2 = c_3$. Choose $c_1 = 1$: $v_1 + v_2 + v_3 = 0$.

4. Suppose $a\sin x + b\cos x + c\sin 2x = 0$ for all $x \in \mathbb{R}$.

At $x = 0$: $a\sin 0 + b\cos 0 + c\sin 0 = b = 0 \implies b = 0$.

At $x = \pi$: $a\sin \pi + 0 + c\sin 2\pi = 0 \implies 0 = 0$ (no new info).

At $x = \frac{\pi}{2}$: $a\sin \frac{\pi}{2} + 0 + c\sin \pi = a = 0 \implies a = 0$.

So $b = 0$, $a = 0$, and the equation becomes $c\sin 2x = 0$ for all $x$, which implies $c = 0$.

Thus $a = b = c = 0$, so the set is independent.

5. $v_1 = 2v_2 - v_3 \implies v_1 - 2v_2 + v_3 = 0$. This is the required nontrivial combination.

### Hard Solutions

1. **($\Leftarrow$)** If $Av = v$, then $1 \cdot v + (-1) \cdot Av = v - v = 0$, so $\{v, Av\}$ is dependent. If $Av = 0$, then $0 \cdot v + 1 \cdot Av = 0$, so dependent.

**($\Rightarrow$)** Suppose $\{v, Av\}$ is dependent and $v \neq 0$. Then there exist $c_1, c_2$, not both zero, such that $c_1 v + c_2 Av = 0$.

If $c_2 = 0$, then $c_1 v = 0$ with $v \neq 0 \implies c_1 = 0$, contradiction. So $c_2 \neq 0$.

Then $Av = -\frac{c_1}{c_2} v$, so $v$ is an eigenvector of $A$ with eigenvalue $\lambda = -c_1/c_2$.

Since $A$ is idempotent, $A^2 = A$, so $\lambda^2 v = A^2 v = Av = \lambda v$, hence $\lambda^2 = \lambda \implies \lambda(\lambda - 1) = 0 \implies \lambda = 0$ or $\lambda = 1$.

If $\lambda = 0$, then $Av = 0$. If $\lambda = 1$, then $Av = v$.

2. Form the matrix with vectors as columns:

$$
A = \begin{bmatrix}
1 & 2 & 3 & 4 \\
2 & 3 & 4 & 5 \\
3 & 4 & 5 & 6 \\
4 & 5 & 6 & 7
\end{bmatrix}
$$

Row reduce: $R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - 3R_1$, $R_4 \leftarrow R_4 - 4R_1$:

$$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -1 & -2 & -3 \\
0 & -2 & -4 & -6 \\
0 & -3 & -6 & -9
\end{bmatrix}
$$

$R_3 \leftarrow R_3 - 2R_2$, $R_4 \leftarrow R_4 - 3R_2$:

$$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
0 & -1 & -2 & -3 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

Rank = 2. The vectors are dependent. A basis for the column space: columns 1 and 2 (pivot positions). So $\{v_1, v_2\}$ is a basis. The span has dimension 2.

We can see that $v_3 = 2v_2 - v_1$ (check: $2(2,3,4,5) - (1,2,3,4) = (4-1, 6-2, 8-3, 10-4) = (3,4,5,6)$ ✓) and $v_4 = 3v_2 - 2v_1$ (check: $3(2,3,4,5) - 2(1,2,3,4) = (6-2, 9-4, 12-6, 15-8) = (4,5,6,7)$ ✓).

3. Flatten each matrix into a vector in $\mathbb{R}^4$:

$$
v_1 = (1,2,3,4),\; v_2 = (2,3,4,5),\; v_3 = (3,4,5,6),\; v_4 = (4,5,6,7)
$$

These are the same vectors as in problem 2. They are dependent with rank 2.

From problem 2: $v_3 = 2v_2 - v_1$ and $v_4 = 3v_2 - 2v_1$.

In matrix form:

$$
\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix} = 2\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} - \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

$$
\begin{bmatrix} 4 & 5 \\ 6 & 7 \end{bmatrix} = 3\begin{bmatrix} 2 & 3 \\ 4 & 5 \end{bmatrix} - 2\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

The space of all dependence relations is 2-dimensional. The general dependence relation is:

$$
c_1 v_1 + c_2 v_2 + c_3 v_3 + c_4 v_4 = 0
$$

with $c_1 = -a - 2b$, $c_2 = 2a + 3b$, $c_3 = -a$, $c_4 = -b$ for any $a, b \in \mathbb{R}$.

## Related Concepts

- **Linear Independence (MATH-034):** The logical opposite — no nontrivial combination yields zero. Understanding both concepts is essential.
- **Rank:** The maximum number of independent columns/rows. Dependence reduces rank.
- **Column Space and Null Space:** The column space is the span of columns; the null space contains all dependence relations.
- **Basis:** A maximal independent set (and minimal spanning set). Dependence reveals when a set has more vectors than needed for a basis.
- **Dimension:** The size of any basis. Dependence means the vectors span a space of lower dimension than the number of vectors.
- **Multicollinearity:** The statistical manifestation of approximate linear dependence in regression.
- **Singular Value Decomposition (SVD):** Reveals the dependence structure of a matrix through its singular values. Zero singular values correspond to exact dependence.

## Next Concepts

- **036 Linear Transformation:** Maps between vector spaces. The kernel of a linear transformation captures dependence relations among columns of its matrix representation.
- **Eigenvalues and Eigenvectors:** For a linear transformation, eigenvectors with eigenvalue 0 lie in the kernel and indicate dependence in the transformation's action.
- **Gram-Schmidt Process:** Converts a dependent set into an orthogonal (independent) basis by sequentially removing dependence.
- **Principal Component Analysis (PCA):** Uses the dependence structure of data to find a lower-dimensional representation.

## Summary

Linear dependence describes the situation where a set of vectors contains redundancy — at least one vector can be written as a combination of the others. Dependence can be detected via the determinant (zero for square matrices), rank (less than the number of vectors), or the existence of nonzero null space vectors. In $\mathbb{R}^2$, dependence means collinearity; in $\mathbb{R}^3$, it means coplanarity. In machine learning, linear dependence among features causes multicollinearity, destabilizes regression estimates, and motivates dimensionality reduction. Dependence is not always undesirable — overcomplete representations in sparse coding deliberately use dependent atoms to achieve sparse representations.

## Key Takeaways

1. A set is linearly dependent if some nontrivial linear combination of its vectors equals zero.
2. At least one vector in a dependent set can be expressed as a linear combination of the others.
3. Testing: determinant = 0 (square matrices), rank < number of vectors, or existence of a nonzero null space vector.
4. Geometrically, dependent vectors lie in a subspace of lower dimension than the number of vectors.
5. In AI/ML, dependence causes multicollinearity, but is also exploited in sparse coding, autoencoders, and dimensionality reduction.
6. Dependence is detected by solving $Ac = 0$ for a nontrivial $c$, which is exactly finding a vector in the null space of $A$.
