# Concept: Matrix Addition

## Concept ID

MATH-021

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Matrix Algebra

## Learning Objectives

- Define matrix addition as an element-wise operation
- Verify dimension compatibility before adding matrices
- Apply the commutative and associative properties of matrix addition
- Identify the zero matrix as the additive identity
- Find the additive inverse of a matrix
- Use matrix addition in practical AI/ML contexts

## Prerequisites

- Basic arithmetic: comfort with adding real numbers
- Familiarity with matrix notation: a matrix $A$ of size $m \times n$ has $m$ rows and $n$ columns, with entries $a_{ij}$
- Understanding of scalar addition and negative numbers

## Definition

**Matrix addition** is the operation of adding two matrices by adding their corresponding entries. If $A$ and $B$ are both $m \times n$ matrices, their sum $C = A + B$ is also an $m \times n$ matrix where each entry is:

$$
c_{ij} = a_{ij} + b_{ij}
$$

for $i = 1, 2, \dots, m$ and $j = 1, 2, \dots, n$.

The operation is only defined when both matrices have exactly the same number of rows **and** the same number of columns. When this condition holds, we say the matrices are **conformable for addition**.

For example, if:

$$
A = \begin{bmatrix} 2 & 5 \\ 1 & -3 \end{bmatrix}, \quad
B = \begin{bmatrix} 4 & -1 \\ 0 & 7 \end{bmatrix}
$$

then:

$$
A + B = \begin{bmatrix} 2+4 & 5+(-1) \\ 1+0 & -3+7 \end{bmatrix}
= \begin{bmatrix} 6 & 4 \\ 1 & 4 \end{bmatrix}
$$

## Intuition

Think of matrix addition as adding two tables of numbers entry by entry. If a spreadsheet has sales data for January in one matrix and February in another, adding them gives the total sales for both months combined. Each cell is independent — you simply add the numbers that occupy the same position.

Visually, if you have two grids of the same shape, you overlay them and add the numbers that line up. The result is a new grid of the same shape with the combined values.

## Why This Concept Matters

Matrix addition is one of the simplest yet most essential operations in linear algebra. It serves as the foundation for:

- **Combining data**: merging datasets, adding signals, aggregating information
- **Bias terms in neural networks**: adding a bias vector (or matrix) to a weighted input
- **Ensemble methods**: averaging multiple weight matrices from different trained models
- **Error computation**: subtracting predictions from targets uses the same element-wise logic
- **Signal processing**: adding filtered signals or noise to data

Without understanding matrix addition, you cannot perform more advanced operations like matrix multiplication or solve systems of linear equations.

## Historical Background

Matrix addition, like the general concept of matrices, was formalised in the mid-19th century. The English mathematician **Arthur Cayley** (1821–1895) published "A Memoir on the Theory of Matrices" in 1858, where he defined addition and multiplication of matrices essentially as we use them today. Cayley recognised that matrices form an algebraic system with their own rules, distinct from ordinary arithmetic. His work unified many scattered ideas about linear transformations and systems of equations.

The additive structure of matrices — the fact that they form a commutative group under addition — was part of the broader development of abstract algebra in the late 19th and early 20th centuries.

## Real World Examples

1. **Budgeting**: A company's revenue matrix for Q1 and Q2 can be added to get the first-half totals. Each cell represents a product category (row) and a region (column).

2. **Image Averaging**: In photography, multiple images of the same scene taken at different times can be added and divided by the count to reduce noise (stacking).

3. **Inventory**: Two warehouse inventory matrices (items \times quantities) can be added to get the combined stock.

4. **Sports Statistics**: A player's stat matrix for two seasons added together gives career totals for points, rebounds, assists, etc.

5. **Traffic Flow**: Traffic count matrices from two different hours can be added to get the total flow over the combined period.

## AI/ML Relevance

Matrix addition appears throughout machine learning in several concrete ways.

1. **Adding Bias in Neural Networks**: In a fully-connected layer, the output is computed as $y = Wx + b$. Here $Wx$ produces a vector (or matrix for a batch), and $b$ is a **bias vector** that is added element-wise. For a batch of $m$ samples with $n$ features, $Wx$ is an $m \times n$ matrix and $b$ is a $1 \times n$ row vector that gets broadcast and added to every row. This is matrix addition (with broadcasting). Without it, neural networks would have no way to shift activations.

2. **Ensemble Averaging**: When training multiple neural networks on the same task, one common technique is to **average their weight matrices**. If two models with the same architecture are trained, their weight matrices $W_1$ and $W_2$ can be added: $W_{\text{avg}} = \frac{1}{2}(W_1 + W_2)$. This is matrix addition followed by scalar multiplication, and it often produces a more robust model.

3. **Residual Connections**: In ResNet architectures, the output of a residual block is $y = F(x) + x$, where $F(x)$ is the output of some layers and $x$ is the input. This is a matrix (or tensor) addition that helps gradients flow during backpropagation.

4. **Gradient Accumulation**: In training, gradients from multiple mini-batches are accumulated by adding gradient matrices before performing an update step: $G_{\text{total}} = G_1 + G_2 + \dots + G_k$.

5. **Loss Aggregation**: When computing the total loss over a batch, individual loss values form a matrix that is summed element-wise across samples.

## Mathematical Explanation

Matrix addition is defined for two matrices $A$ and $B$ of the same dimensions $m \times n$. The sum $C = A + B$ is computed by adding corresponding entries:

$$
C = A + B \quad \text{where} \quad c_{ij} = a_{ij} + b_{ij}
$$

This means we are performing $m \times n$ scalar additions simultaneously. If we try to add matrices of different sizes — for example, a $2 \times 3$ matrix with a $3 \times 2$ matrix — the operation is **undefined** because there is no one-to-one correspondence between entries.

### Relationship to Vector Addition

If a matrix is viewed as a collection of column vectors, adding two matrices is equivalent to adding their corresponding column vectors. Similarly for rows. This connects matrix addition to the familiar vector addition operation.

## Formula(s)

**General formula for matrix addition** (both matrices $m \times n$):

$$
(A + B)_{ij} = a_{ij} + b_{ij}
$$

**Writing out a $2 \times 2$ case:**

$$
\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
+
\begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}
=
\begin{bmatrix} a_{11}+b_{11} & a_{12}+b_{12} \\ a_{21}+b_{21} & a_{22}+b_{22} \end{bmatrix}
$$

## Properties

1. **Closure**: If $A$ and $B$ are $m \times n$ matrices, then $A + B$ is also an $m \times n$ matrix. The set of $m \times n$ matrices is closed under addition.

2. **Commutativity**: $A + B = B + A$. The order of addition does not matter because scalar addition is commutative: $a_{ij} + b_{ij} = b_{ij} + a_{ij}$.

3. **Associativity**: $(A + B) + C = A + (B + C)$. Grouping does not matter because scalar addition is associative.

4. **Additive Identity (Zero Matrix)** : There exists a zero matrix $0_{m \times n}$ (all entries are 0) such that $A + 0 = A$ for any $m \times n$ matrix $A$.

5. **Additive Inverse**: For every matrix $A$, there exists a matrix $-A$ (each entry is $-a_{ij}$) such that $A + (-A) = 0$.

6. **Compatibility with Scalar Multiplication**: $c(A + B) = cA + cB$, where $c$ is a scalar. This is the distributive property linking scalar multiplication and addition.

## Step-by-Step Worked Examples

### Example 1: Adding Two $2 \times 2$ Matrices

**Problem:** Let

$$
A = \begin{bmatrix} 3 & -1 \\ 2 & 5 \end{bmatrix}, \quad
B = \begin{bmatrix} -2 & 4 \\ 6 & -3 \end{bmatrix}
$$

Compute $A + B$.

**Solution:**

**Step 1:** Check dimensions. Both $A$ and $B$ are $2 \times 2$. They are conformable for addition.

**Step 2:** Add corresponding entries:

- $c_{11} = a_{11} + b_{11} = 3 + (-2) = 1$
- $c_{12} = a_{12} + b_{12} = -1 + 4 = 3$
- $c_{21} = a_{21} + b_{21} = 2 + 6 = 8$
- $c_{22} = a_{22} + b_{22} = 5 + (-3) = 2$

**Step 3:** Assemble the result:

$$
A + B = \begin{bmatrix} 1 & 3 \\ 8 & 2 \end{bmatrix}
$$

### Example 2: Adding Three $2 \times 3$ Matrices

**Problem:** Let

$$
A = \begin{bmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 2 & 1 & -1 \\ 0 & -2 & 4 \end{bmatrix}, \quad
C = \begin{bmatrix} -1 & 2 & 3 \\ 1 & 1 & -3 \end{bmatrix}
$$

Compute $A + B + C$.

**Solution:**

**Step 1:** Check dimensions. All three matrices are $2 \times 3$. Addition is defined.

**Step 2:** Because addition is associative, we can add them in any order. Add entry by entry:

Row 1:
- $1 + 2 + (-1) = 2$
- $0 + 1 + 2 = 3$
- $2 + (-1) + 3 = 4$

Row 2:
- $-1 + 0 + 1 = 0$
- $3 + (-2) + 1 = 2$
- $1 + 4 + (-3) = 2$

**Step 3:** Assemble the result:

$$
A + B + C = \begin{bmatrix} 2 & 3 & 4 \\ 0 & 2 & 2 \end{bmatrix}
$$

### Example 3: Using the Additive Inverse

**Problem:** Let $A = \begin{bmatrix} 5 & -3 \\ 2 & 7 \\ 0 & -1 \end{bmatrix}$. Find a matrix $B$ such that $A + B = 0$ (the zero matrix).

**Solution:**

**Step 1:** The additive inverse of $A$ is $-A$, where each entry of $A$ is negated.

**Step 2:** Compute $-A$:

$$
-A = \begin{bmatrix} -5 & 3 \\ -2 & -7 \\ 0 & 1 \end{bmatrix}
$$

**Step 3:** Verify by adding:

$$
A + (-A) = \begin{bmatrix} 5+(-5) & -3+3 \\ 2+(-2) & 7+(-7) \\ 0+0 & -1+1 \end{bmatrix}
= \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
$$

So $B = -A$.

### Example 4: Adding Matrices of Different Dimensions (Error Case)

**Problem:** Can we add $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$?

**Solution:**

**Step 1:** Check dimensions. $A$ is $2 \times 3$, $B$ is $2 \times 2$.

**Step 2:** The number of columns differs ($3 \neq 2$). Therefore, $A + B$ is **undefined** — the matrices are not conformable for addition.

This illustrates the critical rule: matrix addition requires **identical dimensions**.

### Example 5: Adding a Row Vector to Every Row of a Matrix (Broadcasting)

**Problem:** In a neural network, we compute $Z = XW + b$, where $X$ is a $3 \times 2$ matrix of input features (3 samples, 2 features), $W$ is a $2 \times 3$ weight matrix, and $b = \begin{bmatrix} 0.1 & -0.2 & 0.3 \end{bmatrix}$ is a $1 \times 3$ bias row vector. Perform the addition assuming $XW$ has been computed as:

$$
XW = \begin{bmatrix} 1.5 & 0.0 & -0.5 \\ 0.5 & 2.0 & 1.0 \\ -1.0 & 1.5 & 0.0 \end{bmatrix}
$$

**Solution:**

**Step 1:** $XW$ is $3 \times 3$, $b$ is $1 \times 3$. In matrix addition strictly defined, these dimensions do not match (3 vs 1 rows). However, in practice (NumPy, PyTorch, TensorFlow), **broadcasting** automatically adds $b$ to each row.

**Step 2:** Add $b$ to every row of $XW$:

Row 1: $[1.5+0.1,\; 0.0+(-0.2),\; -0.5+0.3] = [1.6, -0.2, -0.2]$
Row 2: $[0.5+0.1,\; 2.0+(-0.2),\; 1.0+0.3] = [0.6, 1.8, 1.3]$
Row 3: $[-1.0+0.1,\; 1.5+(-0.2),\; 0.0+0.3] = [-0.9, 1.3, 0.3]$

**Step 3:** The result is:

$$
Z = \begin{bmatrix} 1.6 & -0.2 & -0.2 \\ 0.6 & 1.8 & 1.3 \\ -0.9 & 1.3 & 0.3 \end{bmatrix}
$$

This is the output of the linear layer before applying an activation function.

## Visual Interpretation

Matrix addition can be visualised as overlaying two grids of numbers and summing each overlapping cell. If you have a $3 \times 4$ grid of temperatures from two different weather stations, adding them gives a combined temperature map.

For images, a grayscale image is a matrix of pixel intensities (0 to 255). Adding two images of the same scene brightens the result (pixel values increase). If you add a matrix of random noise to an image matrix, you get a noisy version of the image.

Another visualisation: think of each matrix as a height map (a surface). Adding two height maps produces a new surface whose height at each point is the sum of the two original heights. This is how terrain data from different sources can be combined.

## Common Mistakes

1. **Adding matrices of different sizes**: This is the most common error. Always verify that both the row count and column count match before adding.

2. **Confusing element-wise addition with matrix multiplication**: Addition simply adds entries at the same position. Multiplication is far more complex. The $+$ sign always means element-wise addition.

3. **Forgetting that scalar addition rules apply to each entry**: Since each $c_{ij} = a_{ij} + b_{ij}$ is ordinary scalar addition, the sign rules for positive and negative numbers apply normally.

4. **Assuming commutativity of subtraction**: While $A + B = B + A$, it is NOT true that $A - B = B - A$ (matrix subtraction is not commutative).

5. **Misunderstanding the zero matrix**: The zero matrix is not "nothing." It is a specific matrix of the same dimensions as $A$ with all entries equal to zero. It serves as the additive identity.

6. **Thinking addition changes the dimensions**: The sum of two $m \times n$ matrices is always $m \times n$. Addition never changes the shape.

7. **Overlooking broadcasting rules**: In pure mathematics, dimensions must match exactly. In programming libraries, broadcasting may stretch smaller matrices. Know which context you are working in.

## Interview Questions

### Beginner

1. What does it mean for two matrices to be conformable for addition?
2. Compute $\begin{bmatrix} 2 & -1 \\ 0 & 4 \end{bmatrix} + \begin{bmatrix} 3 & 2 \\ -5 & 1 \end{bmatrix}$.
3. What is the additive identity for $3 \times 2$ matrices?
4. Is matrix addition commutative? Explain briefly.
5. If $A$ is $2 \times 3$ and $B$ is $3 \times 2$, can you compute $A + B$? Why or why not?

### Intermediate

1. Show that matrix addition is associative for $2 \times 2$ matrices with a concrete example.
2. If $A + B = A$, what must be true about $B$?
3. How does matrix addition relate to adding bias terms in a neural network layer?
4. For a $3 \times 3$ matrix $A$, what is $A + (-A)$? What is this matrix called?
5. Given $A = \begin{bmatrix} 4 & 1 \\ 2 & -3 \end{bmatrix}$ and $A + B = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$, find $B$.

### Advanced

1. Prove that the set of all $m \times n$ matrices with real entries forms an abelian group under addition.
2. How does broadcasting for matrix addition work in NumPy? Give an example where broadcasting allows addition of matrices with different dimensions.
3. Explain how residual connections ($y = F(x) + x$) in ResNets use matrix addition to solve the vanishing gradient problem.

## Practice Problems

### Easy - 5 Questions

1. Compute $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$.
2. Compute $\begin{bmatrix} -1 & 0 \\ 2 & -3 \\ 4 & 5 \end{bmatrix} + \begin{bmatrix} 3 & -2 \\ -1 & 1 \\ 0 & -4 \end{bmatrix}$.
3. Find the zero matrix $0$ such that $\begin{bmatrix} 8 & -1 \\ 0 & 6 \end{bmatrix} + 0 = \begin{bmatrix} 8 & -1 \\ 0 & 6 \end{bmatrix}$.
4. What is $-A$ if $A = \begin{bmatrix} 1 & -2 & 3 \\ 0 & 4 & -5 \end{bmatrix}$?
5. If $A = \begin{bmatrix} 2 \\ 5 \\ -1 \end{bmatrix}$ (a column vector), what is $A + (-A)$?

### Medium - 5 Questions

6. Given $A = \begin{bmatrix} 3 & 0 \\ -1 & 2 \end{bmatrix}$, $B = \begin{bmatrix} -2 & 4 \\ 1 & -3 \end{bmatrix}$, $C = \begin{bmatrix} 1 & -2 \\ 0 & 5 \end{bmatrix}$, compute $A + B + C$.
7. Verify that $(A + B) + C = A + (B + C)$ for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $B = \begin{bmatrix} 0 & -1 \\ 2 & -3 \end{bmatrix}$, $C = \begin{bmatrix} -2 & 1 \\ 1 & 0 \end{bmatrix}$.
8. If $X = \begin{bmatrix} 2 & 5 \\ -1 & 0 \\ 3 & -4 \end{bmatrix}$ and $X + Y = \begin{bmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$, find $Y$.
9. In a neural network, $Z = XW + b$ where $XW$ is a $4 \times 2$ matrix and $b = [0.5, -0.3]$. Compute $Z$ given $XW = \begin{bmatrix} 1.0 & -0.5 \\ 2.5 & 0.0 \\ -1.0 & 1.5 \\ 0.0 & -2.0 \end{bmatrix}$ (use broadcasting).
10. Show that $2(A + B) = 2A + 2B$ for $A = \begin{bmatrix} 1 & -1 \\ 2 & 0 \end{bmatrix}$, $B = \begin{bmatrix} 3 & 2 \\ -1 & 4 \end{bmatrix}$.

### Hard - 3 Questions

11. Let $A$ be an $m \times n$ matrix. Prove that $A + (-A) = 0$ using the definition of matrix addition.
12. A batch of 3 data samples produces output matrix $O = XW$ (size $3 \times 2$) and bias matrix $B$ (size $1 \times 2$). If $O = \begin{bmatrix} 0.5 & 1.2 \\ -0.3 & 0.8 \\ 0.0 & -0.5 \end{bmatrix}$ and the final output after adding bias is $Z = \begin{bmatrix} 0.7 & 1.0 \\ -0.1 & 0.6 \\ 0.2 & -0.7 \end{bmatrix}$, find the bias vector $B$.
13. Suppose $A$ is a $3 \times 3$ matrix where $a_{ij} = i + j$, and $B$ is a $3 \times 3$ matrix where $b_{ij} = i - j$. Compute $A + B$ and $- (A + B)$.

## Solutions

### Easy Solutions

1. $\begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$

2. $\begin{bmatrix} -1+3 & 0+(-2) \\ 2+(-1) & -3+1 \\ 4+0 & 5+(-4) \end{bmatrix} = \begin{bmatrix} 2 & -2 \\ 1 & -2 \\ 4 & 1 \end{bmatrix}$

3. The zero matrix is $\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$.

4. $-A = \begin{bmatrix} -1 & 2 & -3 \\ 0 & -4 & 5 \end{bmatrix}$

5. $A + (-A) = \begin{bmatrix} 2 \\ 5 \\ -1 \end{bmatrix} + \begin{bmatrix} -2 \\ -5 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$, the $3 \times 1$ zero matrix.

### Medium Solutions

6. $A + B + C = \begin{bmatrix} 3-2+1 & 0+4-2 \\ -1+1+0 & 2-3+5 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 0 & 4 \end{bmatrix}$

7. First, $A + B = \begin{bmatrix} 1 & 1 \\ 5 & 1 \end{bmatrix}$, then $(A+B)+C = \begin{bmatrix} 1-2 & 1+1 \\ 5+1 & 1+0 \end{bmatrix} = \begin{bmatrix} -1 & 2 \\ 6 & 1 \end{bmatrix}$. Next, $B + C = \begin{bmatrix} -2 & 0 \\ 3 & -3 \end{bmatrix}$, then $A+(B+C) = \begin{bmatrix} 1-2 & 2+0 \\ 3+3 & 4-3 \end{bmatrix} = \begin{bmatrix} -1 & 2 \\ 6 & 1 \end{bmatrix}$. They match.

8. $Y = -X = \begin{bmatrix} -2 & -5 \\ 1 & 0 \\ -3 & 4 \end{bmatrix}$

9. $Z = \begin{bmatrix} 1.0+0.5 & -0.5+(-0.3) \\ 2.5+0.5 & 0.0+(-0.3) \\ -1.0+0.5 & 1.5+(-0.3) \\ 0.0+0.5 & -2.0+(-0.3) \end{bmatrix} = \begin{bmatrix} 1.5 & -0.8 \\ 3.0 & -0.3 \\ -0.5 & 1.2 \\ 0.5 & -2.3 \end{bmatrix}$

10. $2A = \begin{bmatrix} 2 & -2 \\ 4 & 0 \end{bmatrix}$, $2B = \begin{bmatrix} 6 & 4 \\ -2 & 8 \end{bmatrix}$, so $2A+2B = \begin{bmatrix} 8 & 2 \\ 2 & 8 \end{bmatrix}$. Meanwhile $A+B = \begin{bmatrix} 4 & 1 \\ 1 & 4 \end{bmatrix}$, and $2(A+B) = \begin{bmatrix} 8 & 2 \\ 2 & 8 \end{bmatrix}$. They are equal.

### Hard Solutions

11. By definition, $[A + (-A)]_{ij} = a_{ij} + (-a_{ij}) = 0$. This holds for every $i = 1,\dots,m$ and $j = 1,\dots,n$. Therefore $A + (-A)$ is the $m \times n$ zero matrix, denoted $0$.

12. Since $Z = O + B$ (with broadcasting), we have $B = Z - O$ (subtracting element-wise). For each column:
    - Column 1: $[0.7-0.5,\; -0.1-(-0.3),\; 0.2-0.0] = [0.2, 0.2, 0.2]$
    - Column 2: $[1.0-1.2,\; 0.6-0.8,\; -0.7-(-0.5)] = [-0.2, -0.2, -0.2]$
    Since all rows give the same result, $B = [0.2, -0.2]$.

13. For each entry $(i,j)$:
    - $a_{ij} + b_{ij} = (i+j) + (i-j) = 2i$
    So $A + B$ is a $3 \times 3$ matrix where entry $(i,j) = 2i$:
    $$
    A + B = \begin{bmatrix} 2 & 2 & 2 \\ 4 & 4 & 4 \\ 6 & 6 & 6 \end{bmatrix}
    $$
    Then $-(A+B)$ negates every entry:
    $$
    -(A + B) = \begin{bmatrix} -2 & -2 & -2 \\ -4 & -4 & -4 \\ -6 & -6 & -6 \end{bmatrix}
    $$

## Related Concepts

- **Matrix Subtraction** (MATH-022): The inverse operation of matrix addition
- **Scalar Multiplication of Matrices**: Multiplying every entry of a matrix by a constant
- **Vector Addition** (MATH-011): The 1D (or nD) analogue of matrix addition
- **Zero Matrix**: The additive identity in matrix algebra
- **Matrix Multiplication** (MATH-023): A more complex operation that is not element-wise

## Next Concepts

- **MATH-022 Matrix Subtraction**: Subtracting matrices element-wise
- **MATH-023 Matrix Multiplication**: The dot-product-based operation that powers neural networks
- **MATH-024 Transpose of a Matrix**: Flipping rows and columns

## Summary

Matrix addition is an element-wise operation that combines two matrices of identical dimensions by adding corresponding entries. It is commutative, associative, has the zero matrix as its additive identity, and every matrix has an additive inverse (its negative). Matrix addition is the simplest matrix operation and serves as the building block for more advanced operations. In AI/ML, it appears in bias addition, ensemble averaging, residual connections, and gradient accumulation.

## Key Takeaways

- $A + B$ is defined only when $A$ and $B$ have the same dimensions $m \times n$
- $(A + B)_{ij} = a_{ij} + b_{ij}$ (element-wise addition)
- Matrix addition is commutative ($A + B = B + A$) and associative ($(A + B) + C = A + (B + C)$)
- The zero matrix is the additive identity: $A + 0 = A$
- Every matrix $A$ has an additive inverse $-A$ such that $A + (-A) = 0$
- In AI/ML, matrix addition enables bias terms, ensemble averaging, and residual connections
- Programming libraries often extend pure matrix addition with broadcasting rules
