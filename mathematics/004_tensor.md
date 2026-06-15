# Concept: Tensor

## Concept ID

MATH-004

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Foundations

## Learning Objectives

- Define a tensor as a generalization of scalars, vectors, and matrices.
- Distinguish between tensor order (rank), shape, and size.
- Perform common tensor operations: slicing, reshaping, contraction, and element-wise arithmetic.
- Understand the concept of tensor indices and Einstein summation notation.
- Recognize how tensors are used in modern deep learning frameworks (PyTorch, TensorFlow).

## Prerequisites

- MATH-003 Matrix: matrix operations and dimensions.
- Basic programming familiarity with arrays.

## Definition

A **tensor** is a multi-dimensional array of numbers. The order (or rank) of a tensor is the number of axes (dimensions) it has:

- **Rank 0**: A scalar (single number).
- **Rank 1**: A vector (1D array).
- **Rank 2**: A matrix (2D array).
- **Rank 3**: A 3D array (e.g., a stack of matrices).
- **Rank $n$**: An $n$-dimensional array.

Formally, a tensor of order $p$ can be thought of as a multi-linear map from $p$ vector spaces to $\mathbb{R}$, but for practical purposes in AI/ML, it is simply a generalization of matrices to any number of dimensions.

## Intuition

If a matrix is like a spreadsheet (rows and columns), a tensor is like a stack of spreadsheets — a data cube. In machine learning, images are often represented as rank-3 tensors: height $\times$ width $\times$ color channels. A batch of images becomes a rank-4 tensor: batch_size $\times$ height $\times$ width $\times$ channels. You can think of a tensor as a container that can hold numbers in any rectangular arrangement, regardless of how many axes you need.

## Why This Concept Matters

Tensors are the native data type in every major deep learning framework (PyTorch, TensorFlow, JAX). Understanding tensors — their shapes, operations, and manipulations — is essential for building, debugging, and optimizing neural networks. Concepts like broadcasting, reshaping, and tensor contraction are used daily in model implementation. Moreover, tensor operations run efficiently on GPUs, which is what makes modern deep learning feasible.

## Historical Background

The mathematical theory of tensors was developed in the 19th and early 20th centuries by Gregorio Ricci-Curbastro and his student Tullio Levi-Civita in the context of differential geometry. The word "tensor" comes from the Latin "tensus," meaning "stretched," reflecting its use in describing stress, strain, and elasticity in physics. Albert Einstein used tensor calculus extensively in his general theory of relativity. In the 21st century, tensors found a new home in machine learning, starting with the TensorFlow library (named after tensors) released by Google in 2015.

## Real World Examples

1. **Color Images**: A single image is a rank-3 tensor of shape (height, width, channels) — e.g., (1080, 1920, 3) for an RGB photo.
2. **Video Data**: A video is a rank-4 tensor: (frames, height, width, channels).
3. **Word Embeddings**: The GloVe embedding matrix of shape (vocab_size, embedding_dim) is a rank-2 tensor. A batch of sentences becomes a rank-3 tensor (batch, sequence_length, embedding_dim).
4. **Medical Scans**: A 3D MRI scan is a rank-3 tensor representing volumetric data (depth, height, width).
5. **Physics**: The stress-energy tensor in general relativity is a rank-2 tensor describing the density and flux of energy and momentum in spacetime.

## AI/ML Relevance

Tensors are the universal data structure in deep learning:
- **Data Representation**: All input data (images, audio, text) is converted to tensors.
- **Model Parameters**: Every weight and bias in a neural network is stored as a tensor.
- **Automatic Differentiation**: Tensors track gradients for backpropagation.
- **GPU Acceleration**: Tensor operations are parallelized across thousands of GPU cores.
- **Tensor Processing Units (TPUs)** : Google's TPUs are custom ASICs designed specifically for fast tensor operations.

Popular operations include addition, multiplication, convolution, pooling, reshaping (view), transposition, and reduction (sum, mean, max).

## Mathematical Explanation

A tensor of order $p$ with dimensions $d_1, d_2, \dots, d_p$ has entries indexed by $p$ indices: $T_{i_1, i_2, \dots, i_p}$.

For a rank-3 tensor $T \in \mathbb{R}^{a \times b \times c}$, the entry at position $(i, j, k)$ is written $T_{ijk}$.

**Element-wise operations**: Two tensors of the same shape can be added, subtracted, multiplied, or divided element by element.

**Tensor contraction** (generalization of matrix multiplication): Summing over a shared index. For example, given $A \in \mathbb{R}^{m \times n \times p}$ and $B \in \mathbb{R}^{p \times q}$, a contraction over the $p$ index yields $C \in \mathbb{R}^{m \times n \times q}$ where:

$$
C_{ijk} = \sum_{l=1}^{p} A_{ijl} \, B_{lk}
$$

**Einstein summation convention**: A compact notation where repeated indices are implicitly summed over. For example, $c_{ij} = a_{ik} b_{kj}$ implies summation over $k$.

**Broadcasting**: When performing operations on tensors of different shapes, frameworks automatically expand dimensions to make shapes compatible. For example, adding a vector of shape $(3,)$ to a matrix of shape $(4, 3)$ broadcasts the vector across the rows.

## Formula(s)

**Element-wise addition** (same shape):

$$
C_{ijk} = A_{ijk} + B_{ijk}
$$

**Tensor contraction** (sum over index $k$, given $A \in \mathbb{R}^{i \times j \times k}$, $B \in \mathbb{R}^{k \times l}$):

$$
C_{ijl} = \sum_{k} A_{ijk} \, B_{kl}
$$

**Frobenius norm** (generalization of vector magnitude):

$$
\|T\|_F = \sqrt{\sum_{i_1, i_2, \dots, i_p} T_{i_1 i_2 \dots i_p}^2}
$$

**Reshaping**: Changing the shape of a tensor without changing data, preserving the total number of elements:

$$
\prod_{i=1}^{p} \text{dim}_i = \prod_{j=1}^{q} \text{dim}_j
$$

## Properties

1. **Tensors can be reshaped**: As long as the total number of elements stays the same.
2. **Broadcasting rules**: Shapes are compatible if dimensions are equal or one is 1.
3. **Contraction reduces rank**: Contracting over $k$ indices reduces the rank by $k$.
4. **No universal multiplication**: There is no single "tensor multiplication"; operations include element-wise, contraction, and outer product.
5. **Tensors support vectorization**: Operations are applied to all elements simultaneously (SIMD).
6. **Gradients flow through tensors**: In autodiff frameworks, every tensor operation builds a computational graph.

## Step-by-Step Worked Examples

### Example 1: Tensor Indexing and Shape

Let $T \in \mathbb{R}^{2 \times 3 \times 4}$ be a rank-3 tensor.

**Step 1**: Interpret the shape. This tensor has 2 "layers" (first axis), each containing a $3 \times 4$ matrix.

**Step 2**: Total elements = $2 \times 3 \times 4 = 24$.

**Step 3**: Access $T_{1,2,3}$ — row 1 (0-indexed if programming), column 2, channel 3. In 1-indexed notation, this means the element at the first layer, second row, third column.

### Example 2: Reshaping

Given a tensor of shape $(4, 6)$ with 24 elements, reshape it to $(2, 3, 4)$.

**Step 1**: Check total elements. Original: $4 \times 6 = 24$. New: $2 \times 3 \times 4 = 24$. Compatible.

**Step 2**: The data is laid out in row-major order. The first 4 elements of the original matrix's first row become the first row of the first $3 \times 4$ block in the new tensor.

**Step 3**: Continue reading elements sequentially and filling the new shape.

### Example 3: Tensor Contraction (Batch Matrix Multiplication)

Let $A \in \mathbb{R}^{2 \times 3 \times 4}$ and $B \in \mathbb{R}^{4 \times 5}$. Compute $C_{ijk} = \sum_{l=1}^{4} A_{ijl} \, B_{lk}$.

**Step 1**: Identify the contraction index: $l$ (size 4) appears in both tensors and is summed over.

**Step 2**: The result shape: $(2, 3, 5)$. The first two dimensions come from $A$, the last from $B$.

**Step 3**: For fixed $i$ and $j$, we are multiplying a row vector of length 4 (from $A$) by a matrix of size $4 \times 5$ (from $B$) to get a row vector of length 5.

For $i=1$, $j=1$:

If $A_{1,1,:} = [1, 2, 3, 4]$ and the slice of $B$ is $\begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \end{bmatrix}$, then

$C_{1,1,:} = [1(1) + 2(0) + 3(0) + 4(0),\ 1(0) + 2(1) + 3(0) + 4(0),\ \dots] = [1, 2, 3, 4, 0]$

**Step 4**: Repeat for all $i$, $j$ pairs. This is exactly what PyTorch's `torch.bmm` (batch matrix multiply) does.

## Visual Interpretation

A rank-3 tensor can be visualized as a rectangular prism (a cuboid) of numbers. The three axes correspond to depth, height, and width. Slicing along one axis gives a matrix. For example, slicing a color image tensor at the third axis gives the red, green, and blue channels individually.

For rank-4 tensors (e.g., batch of images), think of a sequence of cuboids, one per item in the batch.

## Common Mistakes

1. **Confusing tensor rank with tensor dimension**. Rank (order) is the number of axes; dimension refers to the size along a specific axis.
2. **Forgetting that broadcasting has rules**. Adding a (3, 4) tensor to a (3, 1) tensor works; adding (3, 4) to (4,) does not (by default rules).
3. **Misunderstanding reshape**. Reshape does NOT change the underlying data order; it only changes the viewing window.
4. **Attempting unsupported multiplication**. "Multiplying" two tensors usually means element-wise multiplication, not matrix multiplication.
5. **Ignoring the data layout (row-major vs. column-major)** . PyTorch and NumPy use row-major (C-style) layout; some frameworks use column-major.
6. **Confusing tensor contraction with the outer product**. Contraction reduces rank; the outer product increases rank.
7. **Treating tensors as mathematical objects with all the same algebraic properties as matrices**. For example, there is no single generalization of the determinant to tensors.

## Interview Questions

### Beginner

1. What is a tensor? How does it relate to scalars, vectors, and matrices?
2. What is the rank (order) of a tensor? Give an example of a rank-3 tensor.
3. How do you access a specific element in a rank-3 tensor?
4. What does it mean to reshape a tensor? What condition must be satisfied?
5. How are tensors used to represent images in deep learning?

### Intermediate

1. Explain the concept of broadcasting with a concrete example. What shapes are compatible?
2. What is tensor contraction? How does it generalize matrix multiplication?
3. Given a tensor of shape (2, 3, 4, 5), what is the total number of elements? What does each axis typically represent in a batch of images?
4. How does automatic differentiation work with tensors in frameworks like PyTorch?
5. Write the Einstein summation for the dot product of two vectors $a$ and $b$ of length $n$.

### Advanced

1. Explain the difference between element-wise multiplication, the outer product, and tensor contraction. Give shapes and equations.
2. How are tensor operations optimized on GPUs? Discuss memory coalescing and kernel fusion.
3. Derive the gradient of a tensor contraction $C_{ij} = \sum_k A_{ik} B_{kj}$ with respect to $A$ and $B$.

## Practice Problems

### Easy - 5 Questions

1. What is the rank of a tensor of shape (5,)? What about (3, 3, 3, 3)?
2. A tensor has shape (2, 3, 4). How many elements does it contain?
3. Create a mental picture: what does slicing a rank-3 tensor along axis 0 produce?
4. Can a tensor of shape (6,) be reshaped to (2, 3)? Why or why not?
5. If $A$ is (3, 4) and $B$ is (4, 2), what is the shape of their matrix product?

### Medium - 5 Questions

1. Given $T \in \mathbb{R}^{2 \times 3 \times 2}$ with $T_{1,1,:} = [1, 2]$, $T_{1,2,:} = [3, 4]$, $T_{1,3,:} = [5, 6]$, $T_{2,1,:} = [7, 8]$, $T_{2,2,:} = [9, 10]$, $T_{2,3,:} = [11, 12]$. Reshape to size (3, 4). What is the first row?
2. Can you add a tensor of shape (3, 1) to a tensor of shape (1, 4)? What shape is the result?
3. Let $A$ be (2, 3, 4) and $B$ be (4, 5). Perform contraction over the last axis of $A$ and first axis of $B$. What is the output shape?
4. Explain what happens when you compute the mean of a rank-3 tensor along axis 1.
5. Two tensors have shapes (2, 3, 4, 5) and (5, 6). Can they be contracted? What is the resulting shape?

### Hard - 3 Questions

1. Implement tensor contraction manually: given $A \in \mathbb{R}^{2 \times 3 \times 4}$ and $B \in \mathbb{R}^{3 \times 5}$, compute $C = \sum_j A_{ij \cdot} B_{j \cdot}$ (contract over axis 1 of $A$ and axis 0 of $B$). Show the output shape and compute for small example values.
2. Derive the backpropagation gradient for the operation $L = \sum_i (T_{ijk} - Y_{ijk})^2$ with respect to $T$, where $T$ and $Y$ are rank-3 tensors.
3. Explain the memory layout of a rank-4 tensor in row-major order. If the shape is (B, C, H, W) and you need to access the element at (b, c, h, w), what is the linear offset in memory?

## Solutions

### Easy Solutions

1. Shape (5,) has rank 1 (a vector). Shape (3, 3, 3, 3) has rank 4.
2. $2 \times 3 \times 4 = 24$ elements.
3. Slicing along axis 0 of a (2, 3, 4) tensor produces 2 matrices, each of shape (3, 4).
4. Yes, because $6 = 2 \times 3$.
5. $A$ is (3, 4), $B$ is (4, 2). Product is (3, 2).

### Medium Solutions

1. Original data in row-major order: row by row through the first layer, then the second:
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12.
   Reshaped to (3, 4): first row = [1, 2, 3, 4].

2. Yes. Broadcasting: (3, 1) + (1, 4) -> (3, 4). Compatible because each dimension is either equal or 1.

3. Contracting the last axis of $A$ (size 4) with the first axis of $B$ (size 4). Output: (2, 3, 5).

4. Taking the mean along axis 1 of a tensor of shape (a, b, c) collapses that axis, producing shape (a, c). Each slice is the average of the $b$ rows along that axis.

5. Yes, the last axis of the first tensor (size 5) matches the first axis of the second (size 5). Output shape: (2, 3, 4, 6).

### Hard Solutions

1. $A$ is (2, 3, 4). $B$ is (3, 5). Contract over axis 1 of $A$ (size 3) and axis 0 of $B$ (size 3). Output: (2, 4, 5).

   Let $A_{1,1,:} = [1,1,1,1]$, $A_{1,2,:} = [2,2,2,2]$, $A_{1,3,:} = [3,3,3,3]$.
   Let $A_{2,1,:} = [4,4,4,4]$, $A_{2,2,:} = [5,5,5,5]$, $A_{2,3,:} = [6,6,6,6]$.
   Let $B_{1,:} = [1,0,0,0,0]$, $B_{2,:} = [0,1,0,0,0]$, $B_{3,:} = [0,0,1,0,0]$.

   For $(i=1, k=1)$: $C_{1,1,:} = \sum_{j=1}^{3} A_{1,j,1} B_{j,:} = 1[1,0,0,0,0] + 2[0,1,0,0,0] + 3[0,0,1,0,0] = [1, 2, 3, 0, 0]$

   For $(i=2, k=1)$: $C_{2,1,:} = 4[1,0,0,0,0] + 5[0,1,0,0,0] + 6[0,0,1,0,0] = [4, 5, 6, 0, 0]$

   The full $C$ would have each of the 2 batches produce a 4-row, 5-column result.

2. $L = \sum_{i,j,k} (T_{ijk} - Y_{ijk})^2$. The gradient:

   $\frac{\partial L}{\partial T_{ijk}} = 2 (T_{ijk} - Y_{ijk})$

   This is an element-wise gradient. In backpropagation, this flows upstream.

3. In row-major order with shape (B, C, H, W), the element at index (b, c, h, w) has linear offset:

   $\text{offset} = b \cdot (C \cdot H \cdot W) + c \cdot (H \cdot W) + h \cdot (W) + w$

   This is because the last dimension (W) varies fastest, then H, then C, then B slowest.

## Related Concepts

- **MATH-003 Matrix**: A rank-2 tensor — the bridge between linear algebra and deep learning.
- **MATH-005 Dimension**: Understanding the size of each axis and the concept of dimensionality.
- **Vector**: A rank-1 tensor.

## Next Concepts

- **MATH-005 Dimension**: Explore the meaning of dimensionality in vector spaces and data.
- **Eigenvalues and Eigenvectors**: Fundamental matrix decompositions that extend to tensors.
- **Tensor Decomposition**: Methods like CP and Tucker decomposition for compressing and analyzing multi-way data.

## Summary

A tensor is a generalization of scalars, vectors, and matrices to any number of dimensions. The rank (order) of a tensor is its number of axes. Tensor operations including element-wise arithmetic, reshaping, broadcasting, and contraction form the computational foundation of all modern deep learning frameworks. Tensors enable efficient GPU-accelerated computation and automatic differentiation. Understanding tensor shapes, operations, and memory layout is essential for building and debugging neural network models.

## Key Takeaways

- Tensors are multi-dimensional arrays; scalars are rank 0, vectors rank 1, matrices rank 2, and beyond.
- Tensor shape must be preserved for element-wise operations; broadcasting relaxes this under specific rules.
- Reshaping changes the viewing dimensions without altering the underlying data or total element count.
- Tensor contraction sums over shared indices, generalizing matrix multiplication.
- Tensors are the fundamental data type in PyTorch, TensorFlow, and JAX, powering all deep learning computations.
