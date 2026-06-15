# Concept: Rank

## Concept ID

MATH-028

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define the rank of a matrix as the dimension of its column space (or row space)
- Compute the rank via row reduction to echelon form
- Distinguish between full rank and rank-deficient matrices
- Relate rank to determinant, linear independence, and the invertible matrix theorem
- State and apply the rank-nullity theorem
- Explain the role of rank in ML contexts such as matrix factorisation and model compression

## Prerequisites

- Basic matrix operations
- Vector spaces and subspaces
- Linear independence and span
- Row reduction (Gaussian elimination)
- Understanding of determinants (MATH-027)

## Definition

The rank of a matrix $A$ is the dimension of its column space (the subspace spanned by its column vectors). Equivalently, the rank is the dimension of its row space (the subspace spanned by its row vectors). The rank is denoted $\text{rank}(A)$.

For an $m \times n$ matrix, the rank is the maximum number of linearly independent columns (or rows). It is always true that $\text{rank}(A) \leq \min(m, n)$.

**Full rank**: A matrix has full rank if $\text{rank}(A) = \min(m, n)$. A square $n \times n$ matrix has full rank if $\text{rank}(A) = n$, which means it is invertible.

**Rank deficient**: A matrix is rank deficient if $\text{rank}(A) < \min(m, n)$.

## Intuition

Think of the rank as the "effective dimensionality" of the information stored in a matrix. If you have a $1000 \times 1000$ matrix but its rank is only 10, then despite its large size, the matrix only captures information that lives in a 10-dimensional subspace — the rest is redundancy. The rank tells you how many independent directions or features the matrix truly represents.

Imagine you have 1000 photographs of faces, each with 1 million pixels. If those photos only span a 200-dimensional subspace (because faces have common structure), the rank of the data matrix is at most 200. This insight is the foundation of many compression and dimensionality reduction techniques.

## Why This Concept Matters

Rank tells you the fundamental dimensionality of the data or transformation represented by a matrix. In linear algebra, the rank determines whether a linear system has solutions, how many solutions it has, and whether a matrix can be inverted. In data science, the rank of a data matrix indicates how many independent features exist — critical for understanding whether a dataset has redundant or collinear variables. Low-rank structure appears everywhere: recommendation systems (Netflix, Spotify), image compression, natural language processing, and neural network compression all exploit the fact that many real-world matrices are approximately low-rank.

## Historical Background

The concept of rank was developed throughout the 19th century. Carl Friedrich Gauss used row reduction to solve linear systems in his work on celestial mechanics (1809), effectively computing ranks without naming them. The term "rank" was introduced by the German mathematician Ferdinand Georg Frobenius in 1879. Georg Cantor, James Joseph Sylvester, and Arthur Cayley also made significant contributions to the theory. The rank-nullity theorem, which relates the rank and nullity of a matrix to the number of columns, was formalised by Sylvester. The concept became central to linear algebra with the development of matrix theory in the 20th century.

## Real World Examples

- **Recommendation Systems**: The user-item rating matrix in Netflix or Spotify is approximately low-rank — users and items can be described by a small number of latent factors (e.g., genre preferences).
- **Image Compression**: A grayscale image can be represented as a matrix. By keeping only the top $k$ singular values (low-rank approximation), we can compress the image with minimal visual loss.
- **Sensor Networks**: If 100 sensors all measure the same physical phenomenon, the data matrix may have rank 1 (all readings are proportional). Rank reveals sensor redundancy.
- **Document Analysis**: In text mining, a term-document matrix often has rank much smaller than its dimensions because documents use a limited vocabulary of topics.
- **Genomics**: Gene expression matrices often exhibit low-rank structure — many genes are co-expressed, and the underlying biological processes are few compared to the number of genes.

## AI/ML Relevance

The rank of a matrix is a cornerstone idea across machine learning:

- **Low-Rank Matrix Factorisation**: Techniques like Non-negative Matrix Factorisation (NMF) and Singular Value Decomposition (SVD) explicitly assume that a data matrix $X$ can be approximated as $X \approx UV^T$ where $U$ and $V$ have rank $k \ll \min(m, n)$. This is the mathematical foundation behind recommendation systems (e.g., matrix factorisation for collaborative filtering), topic modelling (NMF for document decomposition), and word embeddings.

- **Overparameterised Models**: In modern deep learning, models often have far more parameters than training examples. The rank of the feature matrix (design matrix) determines whether the least-squares solution is unique. If the feature matrix has full column rank, the normal equations $X^T X \beta = X^T y$ have a unique solution. If it is rank deficient, infinitely many solutions exist — a situation addressed by regularisation (ridge regression, L2 penalty).

- **Neural Network Compression**: Weight matrices in trained neural networks often have low effective rank. Techniques like low-rank factorisation replace a large weight matrix $W$ (rank $r$) with a product $W \approx UV^T$ where $U$ and $V$ have much lower rank, reducing parameters by up to 90% with minimal accuracy loss. This is used to deploy large models on mobile devices.

- **Principal Component Analysis (PCA)**: PCA finds the directions of maximum variance in data. The number of non-zero eigenvalues of the covariance matrix equals the rank of the data matrix. Dimensionality reduction keeps only the top $k$ components, corresponding to a rank-$k$ approximation of the data.

- **Multi-task Learning**: In multi-task regression, the parameter matrix across tasks is often assumed to be low-rank, meaning tasks share a small number of underlying representations.

## Mathematical Explanation

### Row Rank Equals Column Rank

A fundamental theorem of linear algebra states that for any matrix $A$, the dimension of the row space equals the dimension of the column space. This common dimension is the rank. This is not obvious — rows and columns are different objects — but it is always true.

### Computing Rank via Row Reduction

The rank is most easily computed by reducing the matrix to row echelon form using Gaussian elimination. The rank equals the number of non-zero rows (pivot rows) in the reduced form.

### Rank and Determinant

For an $n \times n$ square matrix $A$:
- $\text{rank}(A) = n$ if and only if $\det(A) \neq 0$ (full rank, invertible)
- $\text{rank}(A) < n$ if and only if $\det(A) = 0$ (rank deficient, singular)

The rank can be understood as the size of the largest square submatrix with a non-zero determinant.

### Rank and Linear Independence

- The rank equals the maximum number of linearly independent columns.
- The rank equals the maximum number of linearly independent rows.
- If $\text{rank}(A) = r$, then $A$ has exactly $r$ linearly independent columns (and $r$ linearly independent rows).

### Rank-Nullity Theorem

For an $m \times n$ matrix $A$, let $\text{nullity}(A) = \dim(\text{nullspace}(A))$ be the dimension of the solution space of $Ax = 0$. Then:

$$\text{rank}(A) + \text{nullity}(A) = n$$

where $n$ is the number of columns. This is a fundamental relationship: the rank measures how many columns are "useful" (independent), while the nullity measures how many free variables exist.

## Formula(s)

**Rank-Nullity Theorem:**
$$\text{rank}(A) + \text{nullity}(A) = n$$

**Rank of a product:**
$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

**Rank of a sum:**
$$\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$$

**Relation to SVD:** If $A = U \Sigma V^T$ is the singular value decomposition, then $\text{rank}(A)$ equals the number of non-zero singular values on the diagonal of $\Sigma$.

## Properties

1. **Row rank equals column rank**: $\text{rank}(A) = \text{rank}(A^T)$.
2. **Range**: $0 \leq \text{rank}(A) \leq \min(m, n)$ for an $m \times n$ matrix.
3. **Full rank**: $\text{rank}(A) = \min(m, n)$ means maximum possible rank.
4. **Invertibility**: A square $n \times n$ matrix is invertible iff $\text{rank}(A) = n$.
5. **Product bound**: $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$.
6. **Rank of transpose**: $\text{rank}(A) = \text{rank}(A^T) = \text{rank}(A^T A) = \text{rank}(A A^T)$.
7. **Elementary operations**: Row operations and column operations preserve the rank.
8. **Rank of a diagonal matrix**: $\text{rank}(\text{diag}(d_1, ..., d_n))$ equals the number of non-zero diagonal entries.
9. **Rank-1 matrices**: Any matrix of the form $uv^T$ (outer product) has rank 1 (if $u, v \neq 0$).
10. **Subadditivity**: $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$.

## Step-by-Step Worked Examples

### Example 1: Rank of a 2x3 Matrix

Find the rank of $A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$.

**Step 1**: Perform row reduction to echelon form.

Replace row 2 with row 2 minus 4 times row 1:
$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & -3 & -6 \end{pmatrix}$$

**Step 2**: The matrix now has two non-zero rows. Both rows have leading non-zero entries (pivots): row 1 has pivot 1 in column 1, row 2 has pivot -3 in column 2.

**Step 3**: Count the number of non-zero rows. There are 2.

**Step 4**: Therefore $\text{rank}(A) = 2$.

Since $\min(2, 3) = 2$, this matrix has full rank.

### Example 2: Rank of a 3x3 Singular Matrix

Find the rank of $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix}$.

**Step 1**: Perform row reduction.

Row 2: $R_2 \leftarrow R_2 - 2R_1$:
$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 3 & 6 & 9 \end{pmatrix}$$

Row 3: $R_3 \leftarrow R_3 - 3R_1$:
$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

**Step 2**: Only one non-zero row remains.

**Step 3**: Therefore $\text{rank}(A) = 1$.

Notice that all rows are multiples of $(1, 2, 3)$, and all columns are multiples of $(1, 2, 3)^T$. The matrix is rank deficient ($n=3$, so full rank would be 3). Indeed $\det(A) = 0$ (the second and third rows are multiples of the first).

### Example 3: Rank and the Rank-Nullity Theorem

Let $A = \begin{pmatrix} 1 & 2 & -1 \\ 2 & 4 & -2 \\ -1 & -2 & 1 \end{pmatrix}$. Find $\text{rank}(A)$ and $\text{nullity}(A)$.

**Step 1**: Row reduce.

$R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 + R_1$:
$$\begin{pmatrix} 1 & 2 & -1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

**Step 2**: Only 1 non-zero row. $\text{rank}(A) = 1$.

**Step 3**: There are $n = 3$ columns. By the rank-nullity theorem:
$$\text{nullity}(A) = n - \text{rank}(A) = 3 - 1 = 2$$

**Step 4**: Verify by solving $Ax = 0$. From the reduced form: $x + 2y - z = 0$, so $x = -2y + z$.

Free variables: $y$ and $z$ are free (2 degrees of freedom), confirming $\text{nullity} = 2$.

The solution space (nullspace) is: $\{y\begin{pmatrix}-2\\1\\0\end{pmatrix} + z\begin{pmatrix}1\\0\\1\end{pmatrix} : y, z \in \mathbb{R}\}$, which is indeed a 2-dimensional subspace.

### Example 4: Rank from Determinant of Submatrices

Find the rank of $A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 1 & 3 & 7 \end{pmatrix}$.

**Step 1**: Check if the full matrix has non-zero determinant.

$$\det(A) = \begin{vmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 1 & 3 & 7 \end{vmatrix}$$

Expand along column 1:
$$= 1 \cdot \begin{vmatrix} 1 & 4 \\ 3 & 7 \end{vmatrix} - 0 + 1 \cdot (-1)^{3+1} \begin{vmatrix} 2 & 3 \\ 1 & 4 \end{vmatrix}$$
$$= (7 - 12) + (8 - 3) = -5 + 5 = 0$$

**Step 2**: Since $\det(A) = 0$, rank is less than 3.

**Step 3**: Check if any $2 \times 2$ submatrix has non-zero determinant. Take the top-left $2 \times 2$:
$$\det\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix} = 1(1) - 2(0) = 1 \neq 0$$

**Step 4**: Since there exists a $2 \times 2$ submatrix with non-zero determinant, $\text{rank}(A) \geq 2$. Combined with $\text{rank}(A) < 3$, we conclude $\text{rank}(A) = 2$.

### Example 5: Rank of a Product Matrix

Let $u = \begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix}$ and $v = \begin{pmatrix}4 \\ 5\end{pmatrix}$. Form the outer product $A = uv^T$ and find its rank.

**Step 1**: Compute $A = uv^T$:
$$A = \begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix} \begin{pmatrix}4 & 5\end{pmatrix} = \begin{pmatrix} 4 & 5 \\ 8 & 10 \\ 12 & 15 \end{pmatrix}$$

**Step 2**: Notice that every row is a multiple of $(4, 5)$, and every column is a multiple of $(1, 2, 3)^T$.

**Step 3**: Perform row reduction:
$$\begin{pmatrix} 4 & 5 \\ 8 & 10 \\ 12 & 15 \end{pmatrix} \xrightarrow{R_2 - 2R_1} \begin{pmatrix} 4 & 5 \\ 0 & 0 \\ 12 & 15 \end{pmatrix} \xrightarrow{R_3 - 3R_1} \begin{pmatrix} 4 & 5 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$$

**Step 4**: Only one non-zero row. $\text{rank}(A) = 1$.

Indeed, any outer product $uv^T$ has rank 1 (provided $u$ and $v$ are non-zero), confirming the property that rank-1 matrices are outer products.

## Visual Interpretation

The rank of a matrix can be visualised as the dimension of the space that the matrix's columns (or rows) span.

- **Rank 1**: All column vectors lie on a single line through the origin. Any two columns are scalar multiples of each other.
- **Rank 2**: All column vectors lie in a plane through the origin. Three or more columns exist in this plane, but at most two are linearly independent.
- **Rank $r$**: The column vectors span an $r$-dimensional subspace of $\mathbb{R}^m$.

If you think of the columns as data points in $\mathbb{R}^m$, the rank is the dimension of the subspace that contains all the data. When data is approximately low-rank, the points cluster near a lower-dimensional subspace — this is exactly what PCA exploits.

For a linear transformation $T(x) = Ax$, the rank is the dimension of the image (output space) of $T$. A rank-deficient transformation squashes some dimensions to zero, losing information.

## Common Mistakes

1. **Thinking rank always equals the number of rows or columns**: The rank cannot exceed either dimension, but it is often smaller than both. A $5 \times 3$ matrix can have rank at most 3, but it could be 2 or 1.

2. **Confusing rank with the number of non-zero entries**: The rank depends on linear independence, not on how many entries are zero. A diagonal matrix with 100 non-zero entries still has rank 100.

3. **Assuming $\text{rank}(A+B) = \text{rank}(A) + \text{rank}(B)$**: Rank is not additive. If $A = I$ and $B = -I$, then $\text{rank}(A) = \text{rank}(B) = n$ but $\text{rank}(A+B) = \text{rank}(0) = 0$.

4. **Forgetting that $\text{rank}(A^T A) = \text{rank}(A)$**: This holds for real matrices, but not generally for complex matrices (where we need $A^* A$). The Gram matrix $A^T A$ preserves the rank.

5. **Mistaking "full rank" for "full row rank" or "full column rank"**: For a non-square $m \times n$ matrix:
   - Full row rank means $\text{rank}(A) = m$ (rows are independent)
   - Full column rank means $\text{rank}(A) = n$ (columns are independent)
   - Full rank means $\text{rank}(A) = \min(m, n)$

6. **Miscounting pivots during row reduction**: A pivot must be the first non-zero entry in a row. If a row becomes all zeros, it does not contribute to the rank.

7. **Ignoring numerical rank in practice**: In floating-point computation, exact zeros are rare. We speak of "numerical rank" — the number of singular values above a tolerance threshold. This practical rank can differ from the theoretical rank.

## Interview Questions

### Beginner

1. **Q**: What is the rank of a matrix?
   **A**: The rank of a matrix is the dimension of its column space (or row space). It equals the maximum number of linearly independent columns (or rows).

2. **Q**: What is the maximum possible rank of a $3 \times 5$ matrix?
   **A**: $\min(3, 5) = 3$. The rank cannot exceed the smaller dimension.

3. **Q**: If a $3 \times 3$ matrix has $\det(A) = 0$, what can you say about its rank?
   **A**: $\text{rank}(A) < 3$. The matrix is singular and rank deficient.

4. **Q**: What is the rank of a matrix of all zeros?
   **A**: Zero. The zero matrix has rank 0.

5. **Q**: What is the rank of the $3 \times 3$ identity matrix?
   **A**: 3. The identity matrix has full rank because all rows (and columns) are linearly independent.

### Intermediate

1. **Q**: State and explain the rank-nullity theorem.
   **A**: For an $m \times n$ matrix $A$, $\text{rank}(A) + \text{nullity}(A) = n$. The rank is the dimension of the column space; the nullity is the dimension of the nullspace (solutions to $Ax = 0$). Their sum always equals the number of columns.

2. **Q**: How can you find the rank of a matrix using determinants?
   **A**: The rank is the size of the largest square submatrix with a non-zero determinant. Compute determinants of $1 \times 1$, $2 \times 2$, $3 \times 3$, ... submatrices until all submatrices of a certain size have zero determinant; the rank is one less than that size.

3. **Q**: Is it possible for a $5 \times 3$ matrix to have rank 5? Why or why not?
   **A**: No. The rank is at most $\min(5, 3) = 3$. A $5 \times 3$ matrix has only 3 columns, so at most 3 can be linearly independent.

4. **Q**: If $\text{rank}(A) = 2$ for a $3 \times 4$ matrix $A$, what is the nullity?
   **A**: By the rank-nullity theorem: $\text{nullity} = n - \text{rank} = 4 - 2 = 2$. The homogeneous system $Ax = 0$ has 2 free variables.

5. **Q**: What is the relationship between the rank of $A$ and the rank of $A^T A$?
   **A**: For a real matrix $A$, $\text{rank}(A^T A) = \text{rank}(A)$. This is because $A^T A$ and $A$ have the same nullspace (for real matrices), and by rank-nullity, the same rank.

### Advanced

1. **Q**: Explain why the rank of a matrix equals the number of non-zero singular values in its SVD.
   **A**: The SVD factorises $A = U \Sigma V^T$, where $U$ and $V$ are orthogonal and $\Sigma$ is diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$ and $\sigma_{r+1} = ... = 0$. Since orthogonal matrices are full rank and rank is preserved by multiplication, $\text{rank}(A) = \text{rank}(\Sigma)$, which equals the number of non-zero diagonal entries — i.e., the number of non-zero singular values. This gives a numerically stable way to compute rank.

2. **Q**: In neural network compression, why are weight matrices often low-rank, and how is this exploited?
   **A**: Trained neural network weights often exhibit correlations that result in low effective rank — many singular values are near zero. Low-rank factorisation replaces $W \in \mathbb{R}^{m \times n}$ (rank $r$) with $W \approx UV^T$ where $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{n \times k}$ for $k \ll r$. This reduces parameters from $mn$ to $k(m+n)$. For $m = n = 1000$ and $k = 50$, parameters drop from $10^6$ to $10^5$, a 90% reduction. The approximation is acceptable because the discarded singular dimensions carried little information.

3. **Q**: Prove that $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$.
   **A**: Let $C = A + B$. Every column of $C$ is a sum of the corresponding columns of $A$ and $B$. Therefore each column of $C$ lies in $\text{col}(A) + \text{col}(B)$, the sum of the column spaces of $A$ and $B$. The dimension of $\text{col}(A) + \text{col}(B)$ is at most $\dim(\text{col}(A)) + \dim(\text{col}(B)) = \text{rank}(A) + \text{rank}(B)$. Since $\text{col}(C) \subseteq \text{col}(A) + \text{col}(B)$, we have $\text{rank}(C) \leq \text{rank}(A) + \text{rank}(B)$.

## Practice Problems

### Easy - 5 Questions

1. Find the rank of $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$.

2. Find the rank of $A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$.

3. Find the rank of the $3 \times 3$ identity matrix.

4. What is the rank of the zero matrix of size $4 \times 5$?

5. Find the rank of $\begin{pmatrix} 2 & -1 \\ -4 & 2 \end{pmatrix}$.

### Medium - 5 Questions

1. Find the rank of $A = \begin{pmatrix} 1 & 1 & 2 \\ 2 & 3 & 5 \\ 3 & 4 & 7 \end{pmatrix}$.

2. If $A$ is a $5 \times 7$ matrix with nullity 3, what is the rank of $A$?

3. Find the rank of $\begin{pmatrix} 1 & 0 & 2 & 1 \\ 2 & 1 & 0 & 2 \\ 0 & 1 & -4 & 0 \end{pmatrix}$.

4. Determine the rank of $A = \begin{pmatrix} 1 & 1 & 1 \\ a & b & c \\ a^2 & b^2 & c^2 \end{pmatrix}$ when $a, b, c$ are distinct.

5. Let $A$ be $3 \times 3$ with $\text{rank}(A) = 2$. What is $\det(A)$? What is $\text{rank}(A^2)$?

### Hard - 3 Questions

1. Prove that for any real $m \times n$ matrix $A$, $\text{rank}(A^T A) = \text{rank}(A)$.

2. Construct a $3 \times 3$ matrix of rank 2 such that $A^2 = 0$ (nilpotent of index 2). Can a rank-1 matrix satisfy $A^2 = 0$? Explain.

3. Let $A$ be an $m \times n$ matrix and $B$ an $n \times p$ matrix. Prove that $\text{rank}(AB) \geq \text{rank}(A) + \text{rank}(B) - n$. This is the Sylvester rank inequality.

## Solutions

### Easy Solutions

1. Row reduce: already in echelon form. Two non-zero rows. $\text{rank}(A) = 2$.

2. Row reduce: $R_2 \leftarrow R_2 - 2R_1$ gives $\begin{pmatrix} 1 & 2 \\ 0 & 0 \end{pmatrix}$. One non-zero row. $\text{rank}(A) = 1$.

3. $I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$. Three non-zero rows, all independent. $\text{rank}(I_3) = 3$.

4. The zero matrix has all zero rows. $\text{rank} = 0$.

5. Row reduce: $R_2 \leftarrow R_2 + 2R_1$ gives $\begin{pmatrix} 2 & -1 \\ 0 & 0 \end{pmatrix}$. One non-zero row. $\text{rank} = 1$.

### Medium Solutions

1. Row reduce: $R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 - 3R_1$:
   $$\begin{pmatrix} 1 & 1 & 2 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}$$
   $R_3 \leftarrow R_3 - R_2$:
   $$\begin{pmatrix} 1 & 1 & 2 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$
   Two non-zero rows. $\text{rank}(A) = 2$.

2. By rank-nullity: $\text{rank}(A) = n - \text{nullity}(A) = 7 - 3 = 4$.

3. Row reduce:
   $R_2 \leftarrow R_2 - 2R_1$:
   $$\begin{pmatrix} 1 & 0 & 2 & 1 \\ 0 & 1 & -4 & 0 \\ 0 & 1 & -4 & 0 \end{pmatrix}$$
   $R_3 \leftarrow R_3 - R_2$:
   $$\begin{pmatrix} 1 & 0 & 2 & 1 \\ 0 & 1 & -4 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$
   Two non-zero rows. $\text{rank} = 2$.

4. This is the Vandermonde matrix. For distinct $a, b, c$, the determinant is $(b-a)(c-a)(c-b) \neq 0$. Therefore $\text{rank}(A) = 3$.

5. $\det(A) = 0$ because the matrix is not full rank.
   For $\text{rank}(A^2)$: Since $\text{rank}(A) = 2$, there is a 1-dimensional nullspace. The image of $A$ is 2-dimensional. $A^2$ applies $A$ again to the image. It is possible that $\text{rank}(A^2)$ could be 2 (if the image and its preimage align) or lower. Without more information, we cannot determine it uniquely. (For example, if $A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$, then $A^2 = A$ has rank 2. But if $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$, rank is 2 but $A^2$ has rank 1.)

### Hard Solutions

1. **Proof that $\text{rank}(A^T A) = \text{rank}(A)$**:

   We show $\text{nullspace}(A) = \text{nullspace}(A^T A)$ for real matrices.
   
   If $Ax = 0$, then $A^T A x = A^T 0 = 0$, so $\text{nullspace}(A) \subseteq \text{nullspace}(A^T A)$.
   
   Conversely, if $A^T A x = 0$, then $x^T A^T A x = 0$, so $(Ax)^T(Ax) = 0$, so $\|Ax\|^2 = 0$, so $Ax = 0$. Hence $\text{nullspace}(A^T A) \subseteq \text{nullspace}(A)$.
   
   Therefore $\text{nullspace}(A) = \text{nullspace}(A^T A)$, so $\text{nullity}(A) = \text{nullity}(A^T A)$.
   
   By rank-nullity: $\text{rank}(A) = n - \text{nullity}(A) = n - \text{nullity}(A^T A) = \text{rank}(A^T A)$.

2. **Construct $A$ with $\text{rank}(A) = 2$ and $A^2 = 0$**:

   Consider $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$.
   
   $\text{rank}(A) = 2$ (columns 1 and 2 are independent; column 3 is column 2 shifted).
   
   $A^2 = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \neq 0$ and $A^3 = 0$, so this has index 3. Let's try:
   
   $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$ has rank 1, $A^2 = 0$.
   $\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$ has rank 1.
   
   For rank 2 and $A^2 = 0$, consider $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$ has rank 1.
   
   Try $A = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$ — rank 2, $A^2$?
   $A^2 = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \neq 0$.
   
   We need each standard basis vector to map at most one step forward in a chain. For $A^2 = 0$, we need $A(Ax) = 0$ for all $x$, meaning $\text{col}(A) \subseteq \text{nullspace}(A)$. Then $\text{rank}(A) = \dim(\text{col}(A)) \leq \dim(\text{nullspace}(A)) = n - \text{rank}(A)$, so $2\text{rank}(A) \leq n$, meaning $\text{rank}(A) \leq n/2$. For $n=3$, we need $\text{rank}(A) \leq 1.5$, so $\text{rank}(A) \leq 1$. Therefore no $3 \times 3$ matrix with rank 2 satisfies $A^2 = 0$. A rank-1 matrix can satisfy $A^2 = 0$, e.g., $A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$.

3. **Sylvester rank inequality**: $\text{rank}(AB) \geq \text{rank}(A) + \text{rank}(B) - n$.

   Consider the linear transformation $T_B: \mathbb{R}^p \to \mathbb{R}^n$ defined by $B$, and $T_A: \mathbb{R}^n \to \mathbb{R}^m$ defined by $A$. Then $AB$ corresponds to $T_A \circ T_B$.
   
   The nullspace of $B$ is contained in the nullspace of $AB$, so $\text{nullity}(B) \leq \text{nullity}(AB)$.
   
   Also, $\text{rank}(B) = p - \text{nullity}(B)$ and $\text{rank}(AB) = p - \text{nullity}(AB)$.
   
   So $\text{rank}(AB) \geq p - \text{nullity}(B) = \text{rank}(B)$ would be the naive bound, but we need the stronger inequality.
   
   Formal proof: Consider $A$ restricted to $\text{col}(B)$. The dimension of $\text{col}(B)$ is $\text{rank}(B)$. By rank-nullity on this restriction:
   $\dim(\text{col}(B)) = \dim(A(\text{col}(B))) + \dim(\text{col}(B) \cap \text{nullspace}(A))$.
   
   But $A(\text{col}(B)) = \text{col}(AB)$, so $\dim(A(\text{col}(B))) = \text{rank}(AB)$.
   
   And $\dim(\text{col}(B) \cap \text{nullspace}(A)) \leq \dim(\text{nullspace}(A)) = n - \text{rank}(A)$.
   
   Therefore: $\text{rank}(B) \leq \text{rank}(AB) + (n - \text{rank}(A))$.
   
   Rearranging: $\text{rank}(AB) \geq \text{rank}(A) + \text{rank}(B) - n$. This is the Sylvester rank inequality.

## Related Concepts

- **Nullspace and Nullity**: The rank-nullity theorem directly relates rank to the dimension of the nullspace
- **Determinant**: For square matrices, non-zero determinant is equivalent to full rank
- **Linear Independence**: Rank counts the maximum number of linearly independent columns/rows
- **Singular Value Decomposition (SVD)**: The rank equals the number of non-zero singular values
- **Column Space and Row Space**: The rank is the dimension of both subspaces
- **Invertible Matrix Theorem**: Many conditions equivalent to full rank for square matrices
- **Eigenvalues**: The rank equals the number of non-zero eigenvalues (for diagonalisable matrices)
- **Matrix Factorisation**: NMF, SVD, and other factorisations approximate matrices with lower-rank components

## Next Concepts

- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Non-negative Matrix Factorisation (NMF)
- Low-rank approximations in machine learning
- Positive definiteness and covariance estimation

## Summary

The rank of a matrix is the dimension of its column space (equivalently, its row space). It measures the number of linearly independent columns or rows, telling us the true dimensionality of the information the matrix contains. The rank is at most the smaller of the number of rows and columns, and a square matrix is invertible if and only if it has full rank. The rank-nullity theorem ($\text{rank} + \text{nullity} = n$) connects the rank to the dimension of the solution space of $Ax = 0$. Rank is central to AI/ML through low-rank matrix factorisations (SVD, NMF) used for recommendations, compression, and topic modelling. It also determines whether overparameterised models have unique solutions and enables neural network compression via low-rank weight approximations.

## Key Takeaways

- The rank of a matrix is the maximum number of linearly independent columns (or rows)
- $\text{rank}(A) \leq \min(m, n)$ for an $m \times n$ matrix
- A square matrix is invertible iff it has full rank (rank equals its dimension)
- $\text{rank}(A) = 0$ iff $A$ is the zero matrix
- Rank-nullity theorem: $\text{rank}(A) + \text{nullity}(A) = n$
- The rank equals the number of non-zero singular values in the SVD
- Low-rank approximation is a fundamental tool in ML for compression, denoising, and factorisation
- In recommendation systems, the user-item matrix is approximated as a low-rank product of latent factor matrices
