# Concept: Vector Addition

## Concept ID

MATH-011

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Add two or more vectors using component-wise addition.
- Apply the tip-to-tail rule and the parallelogram rule for geometric addition.
- Verify the commutative and associative properties of vector addition.
- Use vector addition in AI/ML contexts such as batch processing and gradient accumulation.
- Solve real-world problems that require combining vector quantities.

## Prerequisites

- **Vector (MATH-002):** Understanding that a vector is an ordered collection of numbers with magnitude and direction.
- **Scalar (MATH-001):** Basic arithmetic with real numbers.
- **Coordinate System (MATH-006):** Familiarity with the Cartesian plane and coordinates.

## Definition

**Vector addition** is the operation of combining two or more vectors to produce a new vector, called the **resultant** or **sum**. If $\mathbf{u}$ and $\mathbf{v}$ are two vectors of the same dimension $n$, their sum $\mathbf{u} + \mathbf{v}$ is computed by adding corresponding components:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

The result is a vector of the same dimension as the inputs. Vector addition is defined only for vectors of equal dimension — adding a 2D vector to a 3D vector is not allowed.

## Intuition

Think of vector addition as **walking along a path**. If you walk 3 steps east (vector $\mathbf{u}$) and then 4 steps north (vector $\mathbf{v}$), where do you end up? The sum $\mathbf{u} + \mathbf{v}$ tells you the straight-line direction and distance from your starting point to your final position.

Another way to see it: imagine two forces pulling on an object. One pulls right with 5 Newtons, the other pulls up with 12 Newtons. The object does not move right 5 and then up 12 — it moves along the diagonal, which is the vector sum of the two forces.

Vector addition is the mathematical way of asking: **What is the net effect of applying two vector quantities together?**

## Why This Concept Matters

Vector addition is the most fundamental operation in vector algebra. It is the basis for combining any kind of vector quantity — forces, velocities, displacements, and, in machine learning, gradients, activations, and embeddings.

In AI and machine learning, vector addition appears constantly:
- **Gradient accumulation:** When training with very large datasets, gradients from multiple mini-batches are added together before updating model weights.
- **Residual connections** in deep networks: ResNets add the input vector to the output of a layer ($\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$), allowing gradients to flow directly through the network.
- **Ensemble methods:** Predictions from multiple models are often combined by adding their output vectors (or averaging, which is just addition followed by scalar multiplication).
- **Word embeddings:** The famous analogy "king - man + woman ≈ queen" relies on vector addition and subtraction.

Without vector addition, there is no way to combine multi-dimensional information, and much of linear algebra and deep learning would be impossible.

## Historical Background

The concept of adding vectors is as old as the study of forces in physics. In the 16th century, Simon Stevin (1548–1620) studied the composition of forces and discovered the parallelogram law for combining them. Galileo Galilei (1564–1642) used vector addition to study projectile motion, recognising that horizontal and vertical components of velocity combine independently.

In the 17th century, Isaac Newton's Principia Mathematica (1687) relied on the parallelogram of forces to describe how multiple forces acting on a body combine. The geometric approach — placing vectors tip-to-tail — was well understood by physicists long before the algebraic component-wise definition was formalised.

The modern algebraic treatment of vector addition — adding component-by-component — emerged in the 19th century with the work of William Rowan Hamilton (quaternions, 1843) and Hermann Grassmann (vector spaces, 1844). Josiah Willard Gibbs and Oliver Heaviside later codified vector analysis in the late 1800s, giving us the notation and rules we use today.

## Real World Examples

**Example 1: Navigation.** A ship sails 30 km east (vector $\mathbf{u}$) and then 40 km north (vector $\mathbf{v}$). The resultant displacement vector $\mathbf{u} + \mathbf{v}$ gives the ship's net position relative to its starting point — 50 km at an angle of approximately $53.13^\circ$ north of east.

**Example 2: Forces on a bridge.** Two cables support a beam. Cable A exerts a force vector $\mathbf{F}_A = (2000, 500)$ N, and Cable B exerts $\mathbf{F}_B = (1500, 800)$ N. The net force on the beam is $(2000+1500, 500+800) = (3500, 1300)$ N.

**Example 3: Sports physics.** A football player runs 10 m east (vector $\mathbf{a}$) and then 6 m north-east at $45^\circ$ (vector $\mathbf{b}$). The player's total displacement is $\mathbf{a} + \mathbf{b}$, which coaches can use to analyse field coverage.

**Example 4: Video game physics.** In a game, a character experiences a gravity vector $\mathbf{g} = (0, -9.8)$ m/s$^2$, a wind vector $\mathbf{w} = (5, 0)$ m/s$^2$, and a thrust vector $\mathbf{t} = (0, 15)$ m/s$^2$. The net acceleration is the sum: $\mathbf{a}_{\text{net}} = \mathbf{g} + \mathbf{w} + \mathbf{t} = (5, 5.2)$ m/s$^2$.

## AI/ML Relevance

Vector addition is deeply woven into the fabric of deep learning:

1. **Gradient Accumulation.** When the batch size is too large to fit in GPU memory, practitioners use gradient accumulation. Instead of computing the gradient over the full batch at once, they compute gradients over smaller micro-batches and add them together:

   $$\nabla L_{\text{total}} = \nabla L_{\text{batch}_1} + \nabla L_{\text{batch}_2} + \cdots + \nabla L_{\text{batch}_k}$$

   Then a single optimisation step is taken using the accumulated gradient. Each $\nabla L_{\text{batch}_i}$ is a gradient vector, and the sum is component-wise vector addition.

2. **Residual Connections (ResNets).** In deep neural networks, the skip connection adds the input of a layer to its output:

   $$\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$$

   This is a direct vector addition. The input vector $\mathbf{x}$ and the transformed vector $F(\mathbf{x})$ are added component-wise. This simple addition allows gradients to bypass layers, solving the vanishing gradient problem and enabling networks with hundreds of layers.

3. **Bias Addition in Neural Networks.** A linear layer computes $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$. The bias vector $\mathbf{b}$ is added component-wise to the transformed input vector $\mathbf{W}\mathbf{x}$. Without vector addition, biases could not exist.

4. **Ensemble Averaging.** When averaging predictions from $k$ models, the output vectors are added and then divided by $k$:

   $$\mathbf{p}_{\text{ensemble}} = \frac{1}{k} \sum_{i=1}^k \mathbf{p}_i$$

   The sum $\sum \mathbf{p}_i$ is a vector addition operation.

5. **Word Embedding Analogies.** The classic example:

   $$\text{vector("king")} - \text{vector("man")} + \text{vector("woman")} \approx \text{vector("queen")}$$

   The subtraction and addition of word embedding vectors reveals semantic relationships captured by the vector space.

6. **Attention Mechanism.** In transformer attention, the context vector for a token is the weighted sum of value vectors:

   $$\mathbf{c}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$$

   This is a weighted vector addition (each value vector is scaled by an attention weight and then summed).

## Mathematical Explanation

### Component-Wise Addition

The rule is simple: given two vectors $\mathbf{u}$ and $\mathbf{v}$ of the same dimension $n$, their sum is:

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

### Example in 2D

$$\mathbf{u} = \begin{pmatrix} 3 \\ -2 \end{pmatrix}, \quad \mathbf{v} = \begin{pmatrix} -1 \\ 5 \end{pmatrix}$$

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 3 + (-1) \\ -2 + 5 \end{pmatrix} = \begin{pmatrix} 2 \\ 3 \end{pmatrix}$$

### Example in 3D

$$\mathbf{a} = \begin{pmatrix} 1 \\ 0 \\ -3 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} -4 \\ 2 \\ 1 \end{pmatrix}$$

$$\mathbf{a} + \mathbf{b} = \begin{pmatrix} 1 + (-4) \\ 0 + 2 \\ -3 + 1 \end{pmatrix} = \begin{pmatrix} -3 \\ 2 \\ -2 \end{pmatrix}$$

### Adding More Than Two Vectors

Vector addition is associative, so we can add any number of vectors in any order:

$$\mathbf{v}_1 + \mathbf{v}_2 + \mathbf{v}_3 = \begin{pmatrix} v_{11} + v_{21} + v_{31} \\ v_{12} + v_{22} + v_{32} \\ \vdots \end{pmatrix}$$

### Geometric Interpretation: Tip-to-Tail Rule

1. Draw the first vector $\mathbf{u}$ as an arrow starting from the origin.
2. Draw the second vector $\mathbf{v}$ starting from the **tip** (head) of $\mathbf{u}$.
3. The sum $\mathbf{u} + \mathbf{v}$ is the vector from the origin (tail of $\mathbf{u}$) to the tip of $\mathbf{v}$.

### Geometric Interpretation: Parallelogram Rule

1. Draw both vectors $\mathbf{u}$ and $\mathbf{v}$ starting from the same point (the origin).
2. Complete a parallelogram: draw a line parallel to $\mathbf{v}$ from the tip of $\mathbf{u}$, and a line parallel to $\mathbf{u}$ from the tip of $\mathbf{v}$.
3. The sum $\mathbf{u} + \mathbf{v}$ is the diagonal of the parallelogram starting from the origin.

Both methods produce the same result. The tip-to-tail rule is simpler for adding several vectors; the parallelogram rule is better for visualising the commutative property.

### The Zero Vector as Additive Identity

The zero vector $\mathbf{0}$ (all components zero) acts as the identity for addition:

$$\mathbf{v} + \mathbf{0} = \mathbf{v}$$

For any vector $\mathbf{v}$. This is because adding zero to each component leaves it unchanged.

### Additive Inverse

Every vector $\mathbf{v}$ has an additive inverse $-\mathbf{v}$ such that:

$$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$$

The additive inverse is obtained by negating every component: $-\mathbf{v} = (-v_1, -v_2, \ldots, -v_n)^T$.

## Formula(s)

**Component-wise vector addition:**

$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{pmatrix}$$

**Sum of multiple vectors:**

$$\sum_{i=1}^k \mathbf{v}_i = \mathbf{v}_1 + \mathbf{v}_2 + \cdots + \mathbf{v}_k = \begin{pmatrix} \sum_{i=1}^k v_{i1} \\ \sum_{i=1}^k v_{i2} \\ \vdots \\ \sum_{i=1}^k v_{in} \end{pmatrix}$$

where $v_{ij}$ is the $j$-th component of the $i$-th vector.

## Properties

Let $\mathbf{u}, \mathbf{v}, \mathbf{w} \in \mathbb{R}^n$ be vectors.

| Property | Formula |
|---|---|
| **Commutativity** | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| **Associativity** | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| **Identity element** | $\mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| **Additive inverse** | $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| **Closure** | $\mathbf{u} + \mathbf{v} \in \mathbb{R}^n$ |
| **Distributivity with scalar mult.** | $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$ |

**Commutativity** means the order of addition does not matter. If you walk 3 steps east then 4 steps north, you end up in the same place as walking 4 steps north then 3 steps east.

**Associativity** means grouping does not matter. When adding three vectors, you can add the first two and then the third, or add the last two and then the first — the result is the same.

## Step-by-Step Worked Examples

### Example 1: Basic 2D Vector Addition

Let $\mathbf{u} = \begin{pmatrix} 4 \\ -1 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} -2 \\ 3 \end{pmatrix}$. Find $\mathbf{u} + \mathbf{v}$.

**Step 1:** Align the vectors by dimension. Both are 2D, so addition is defined.

**Step 2:** Add the first components:
$$(\mathbf{u} + \mathbf{v})_1 = u_1 + v_1 = 4 + (-2) = 2$$

**Step 3:** Add the second components:
$$(\mathbf{u} + \mathbf{v})_2 = u_2 + v_2 = -1 + 3 = 2$$

**Step 4:** Write the result:
$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

**Answer:** $\mathbf{u} + \mathbf{v} = \begin{pmatrix}2 & 2\end{pmatrix}^T$.

### Example 2: 3D Vector Addition with Three Vectors

Let $\mathbf{a} = \begin{pmatrix} 1 \\ -3 \\ 2 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 0 \\ 4 \\ -1 \end{pmatrix}$, $\mathbf{c} = \begin{pmatrix} -2 \\ 1 \\ 3 \end{pmatrix}$. Compute $\mathbf{a} + \mathbf{b} + \mathbf{c}$.

**Step 1:** Add the first components:
$$(\mathbf{a} + \mathbf{b} + \mathbf{c})_1 = 1 + 0 + (-2) = -1$$

**Step 2:** Add the second components:
$$(\mathbf{a} + \mathbf{b} + \mathbf{c})_2 = -3 + 4 + 1 = 2$$

**Step 3:** Add the third components:
$$(\mathbf{a} + \mathbf{b} + \mathbf{c})_3 = 2 + (-1) + 3 = 4$$

**Step 4:** Write the result:
$$\mathbf{a} + \mathbf{b} + \mathbf{c} = \begin{pmatrix} -1 \\ 2 \\ 4 \end{pmatrix}$$

**Verification using associativity:** First add $\mathbf{a} + \mathbf{b} = \begin{pmatrix}1 \\ 1 \\ 1\end{pmatrix}$. Then add $\mathbf{c}$: $\begin{pmatrix}1-2 \\ 1+1 \\ 1+3\end{pmatrix} = \begin{pmatrix}-1 \\ 2 \\ 4\end{pmatrix}$. ✓

**Answer:** $\mathbf{a} + \mathbf{b} + \mathbf{c} = \begin{pmatrix} -1 & 2 & 4 \end{pmatrix}^T$.

### Example 3: Gradient Accumulation in Deep Learning

A neural network trainer uses gradient accumulation over 3 micro-batches. The gradient vectors computed are:

$$\nabla L_1 = \begin{pmatrix} 0.2 \\ -0.5 \\ 0.1 \end{pmatrix}, \quad
\nabla L_2 = \begin{pmatrix} -0.1 \\ 0.3 \\ -0.2 \end{pmatrix}, \quad
\nabla L_3 = \begin{pmatrix} 0.4 \\ 0.1 \\ -0.3 \end{pmatrix}$$

Compute the accumulated gradient $\nabla L_{\text{total}} = \nabla L_1 + \nabla L_2 + \nabla L_3$.

**Step 1:** Add the first components:
$$(\nabla L_{\text{total}})_1 = 0.2 + (-0.1) + 0.4 = 0.5$$

**Step 2:** Add the second components:
$$(\nabla L_{\text{total}})_2 = -0.5 + 0.3 + 0.1 = -0.1$$

**Step 3:** Add the third components:
$$(\nabla L_{\text{total}})_3 = 0.1 + (-0.2) + (-0.3) = -0.4$$

**Step 4:** Write the result:
$$\nabla L_{\text{total}} = \begin{pmatrix} 0.5 \\ -0.1 \\ -0.4 \end{pmatrix}$$

**Answer:** The accumulated gradient is $\begin{pmatrix}0.5 & -0.1 & -0.4\end{pmatrix}^T$. The optimiser will use this vector to update the model parameters: $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \alpha \nabla L_{\text{total}}$.

### Example 4: Parallelogram Rule Verification

Let $\mathbf{u} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$. Show that the tip-to-tail and parallelogram rules give the same result.

**Component-wise:**
$$\mathbf{u} + \mathbf{v} = \begin{pmatrix} 3+1 \\ 1+2 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$$

**Tip-to-tail:** Start at $(0,0)$. Draw $\mathbf{u}$ to $(3,1)$. From there, draw $\mathbf{v}$ to $(3+1, 1+2) = (4,3)$. The sum is the vector from $(0,0)$ to $(4,3)$. ✓

**Parallelogram:** Draw $\mathbf{u}$ from $(0,0)$ to $(3,1)$. Draw $\mathbf{v}$ from $(0,0)$ to $(1,2)$. Complete the parallelogram: the fourth vertex is at $(3+1, 1+2) = (4,3)$. The diagonal from $(0,0)$ to $(4,3)$ is $\mathbf{u}+\mathbf{v}$. ✓

**Answer:** Both methods give $\begin{pmatrix}4 & 3\end{pmatrix}^T$.

### Example 5: Adding Feature Vectors with Different Weights (Weighted Sum)

A recommendation system uses three feature vectors for a user:
- User preference vector: $\mathbf{p} = (0.8, 0.1, 0.5)^T$ (weight $w_1 = 0.6$)
- Context vector: $\mathbf{c} = (0.2, 0.9, 0.3)^T$ (weight $w_2 = 0.3$)
- Historical vector: $\mathbf{h} = (0.5, 0.4, 0.7)^T$ (weight $w_3 = 0.1$)

Compute the combined feature vector: $\mathbf{f} = w_1\mathbf{p} + w_2\mathbf{c} + w_3\mathbf{h}$.

**Step 1:** Scale each vector by its weight:
$$w_1\mathbf{p} = \begin{pmatrix} 0.6 \times 0.8 \\ 0.6 \times 0.1 \\ 0.6 \times 0.5 \end{pmatrix} = \begin{pmatrix} 0.48 \\ 0.06 \\ 0.30 \end{pmatrix}$$

$$w_2\mathbf{c} = \begin{pmatrix} 0.3 \times 0.2 \\ 0.3 \times 0.9 \\ 0.3 \times 0.3 \end{pmatrix} = \begin{pmatrix} 0.06 \\ 0.27 \\ 0.09 \end{pmatrix}$$

$$w_3\mathbf{h} = \begin{pmatrix} 0.1 \times 0.5 \\ 0.1 \times 0.4 \\ 0.1 \times 0.7 \end{pmatrix} = \begin{pmatrix} 0.05 \\ 0.04 \\ 0.07 \end{pmatrix}$$

**Step 2:** Add the scaled vectors component-wise:
$$\mathbf{f} = \begin{pmatrix} 0.48 + 0.06 + 0.05 \\ 0.06 + 0.27 + 0.04 \\ 0.30 + 0.09 + 0.07 \end{pmatrix} = \begin{pmatrix} 0.59 \\ 0.37 \\ 0.46 \end{pmatrix}$$

**Answer:** The combined feature vector is $\begin{pmatrix}0.59 & 0.37 & 0.46\end{pmatrix}^T$. This vector might be fed into a classifier to predict which item to recommend.

## Visual Interpretation

In 2D space, vector addition can be visualised in two ways:

**Tip-to-Tail Rule:**
```
y
▲
│        ╱
│      ╱  u+v
│    ╱
│  ╱
│╱
└───────────▶ x
```

Start at the origin. Follow $\mathbf{u}$ to its tip. From there, follow $\mathbf{v}$. The sum is the direct path from start to finish.

**Parallelogram Rule:**
```
y
▲
│     v ╱─────────┐
│       ╱         │
│      ╱          │
│     ╱           │ u+v
│    ╱            │
│   ╱ u           │
│  ╱──────────────┘
└────────────────────▶ x
```

Both vectors start at the same point. The diagonal of the parallelogram is the sum.

In higher dimensions (3D and beyond), we cannot draw the vectors, but the component-wise operation is exactly the same — we add numbers in corresponding positions.

## Common Mistakes

1. **Adding vectors of different dimensions.** You cannot add a 2D vector to a 3D vector because there is no way to pair the components. For example, $\begin{pmatrix}1 \\ 2\end{pmatrix} + \begin{pmatrix}1 \\ 2 \\ 3\end{pmatrix}$ is undefined.

2. **Adding components cross-wise.** Some beginners try to add the first component of $\mathbf{u}$ to the last component of $\mathbf{v}$. The rule is strict: component $i$ of $\mathbf{u}$ adds to component $i$ of $\mathbf{v}$ only.

3. **Forgetting that vector addition is commutative.** It is always true that $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$. If your geometric diagram shows otherwise, double-check your drawing.

4. **Treating vector addition like scalar addition without considering direction.** When adding vectors geometrically, you must account for both magnitude and direction — you cannot simply add the lengths. For example, $\|\mathbf{u} + \mathbf{v}\|$ is generally not equal to $\|\mathbf{u}\| + \|\mathbf{v}\|$.

5. **Confusing the tip-to-tail and parallelogram rules.** Both are valid. The tip-to-tail rule is easier for adding many vectors; the parallelogram rule is better for visualising the commutative property. They always give the same result.

6. **Adding vectors in different units or scales.** In ML applications, adding feature vectors that have not been normalised can cause features with large magnitudes to dominate. Always ensure vectors are properly scaled before addition.

7. **Misinterpreting the zero vector.** The zero vector $\mathbf{0}$ has all components equal to 0. Adding it to any vector leaves the vector unchanged. It is NOT the same as the scalar 0 — it is a vector whose components are all the scalar 0.

8. **Thinking that vector addition is element-wise multiplication.** A common confusion: $\mathbf{u} + \mathbf{v}$ means add components, not multiply them. Element-wise multiplication is a different operation called the Hadamard product.

## Interview Questions

### Beginner

1. **What is vector addition, and how is it performed?**
   *Answer: Vector addition combines two vectors of the same dimension to produce a new vector. It is performed component-wise: each component of the result is the sum of the corresponding components of the input vectors.*

2. **Can you add a vector of dimension 3 to a vector of dimension 5? Why or why not?**
   *Answer: No. Vector addition requires both vectors to have the same number of components. A 3D vector has 3 components and a 5D vector has 5 — there is no one-to-one correspondence, so the sum is undefined.*

3. **What is the tip-to-tail rule?**
   *Answer: Place the tail of the second vector at the tip of the first. The sum is the vector from the tail of the first to the tip of the second. This geometric method works for any number of vectors.*

4. **State the commutative property of vector addition in your own words.**
   *Answer: The commutative property means that the order in which you add vectors does not matter: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$. If you walk 3 km east then 4 km north, you end up at the same place as walking 4 km north then 3 km east.*

5. **What is the result of $\mathbf{v} + (-\mathbf{v})$?**
   *Answer: The result is the zero vector $\mathbf{0}$. The additive inverse $-\mathbf{v}$ is the vector with all components negated, and adding it to $\mathbf{v}$ cancels everything out, leaving the zero vector.*

### Intermediate

1. **Explain what it means for vector addition to be associative. Why does this property matter in practice?**
   *Answer: Associativity means $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$. In practice, this means we can add vectors in any grouping — useful for batch processing where we accumulate gradients from many micro-batches in any order.*

2. **How would you define vector addition for a set of vectors that are not all the same dimension? Is there any way to handle this?**
   *Answer: Strictly, you cannot add vectors of different dimensions. However, you can pad the lower-dimensional vector with zeros to match dimensions. For example, a 2D vector $(3,4)$ could be treated as $(3,4,0)$ to add with a 3D vector. This is sometimes done in ML when feature sets are augmented.*

3. **In the context of ResNets, how does the addition $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$ help with gradient flow?**
   *Answer: The addition creates a direct path for gradients to flow from the output back to the input without passing through the weight layers. During backpropagation, the gradient of the addition splits: $\partial L/\partial \mathbf{x} = \partial L/\partial F \cdot \partial F/\partial \mathbf{x} + \partial L/\partial F$. The additive term $\partial L/\partial F$ (the identity shortcut) prevents the gradient from vanishing even in very deep networks.*

4. **What is the difference between vector addition and vector concatenation?**
   *Answer: Addition combines vectors component-wise, requiring equal dimensions and producing a vector of the same dimension. Concatenation joins vectors end-to-end: concatenating $\mathbf{u} \in \mathbb{R}^2$ and $\mathbf{v} \in \mathbb{R}^3$ produces a vector in $\mathbb{R}^5$. They serve different purposes — addition combines aligned information; concatenation combines separate information into a larger vector.*

5. **Why is it important to normalise or standardise feature vectors before adding them in machine learning pipelines?**
   *Answer: If features have different scales (e.g., age 0–100 vs. income 0–200k), the feature with larger values will dominate any vector operation, including addition. Normalisation (e.g., z-score or min-max scaling) ensures that each feature contributes proportionally to the sum, which is essential for algorithms like K-nearest neighbours, SVMs, and neural networks.*

### Advanced

1. **Prove that vector addition is commutative using the component-wise definition.**
   *Answer: Let $\mathbf{u} = (u_1, \ldots, u_n)^T$ and $\mathbf{v} = (v_1, \ldots, v_n)^T$. By the component-wise definition: $\mathbf{u} + \mathbf{v} = (u_1 + v_1, \ldots, u_n + v_n)^T$. Scalar addition is commutative: $u_i + v_i = v_i + u_i$ for each $i$. Therefore $\mathbf{u} + \mathbf{v} = (v_1 + u_1, \ldots, v_n + u_n)^T = \mathbf{v} + \mathbf{u}$. The property follows directly from the commutativity of scalar addition.*

2. **In a transformer attention mechanism, the context vector is $\mathbf{c}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j$. Explain why this is a form of vector addition and how the attention weights affect the sum.**
   *Answer: The sum $\sum_{j} \alpha_{ij} \mathbf{v}_j$ is a weighted vector addition. Each value vector $\mathbf{v}_j$ is first scaled by the scalar attention weight $\alpha_{ij}$ (which is between 0 and 1 and sums to 1 over $j$). Then all scaled vectors are added component-wise. Because $\sum \alpha_{ij} = 1$, the result is a convex combination — the context vector lies inside the convex hull of the value vectors. Different attention patterns (weights) produce different context vectors, which is how the model focuses on relevant parts of the input.*

3. **Consider the update rule for stochastic gradient descent with momentum: $\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \alpha \nabla L(\mathbf{w}_t)$ and $\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_{t+1}$. Identify every vector addition in this rule and explain its role.**
   *Answer: The first equation has one vector addition: the momentum term $\mu \mathbf{v}_t$ (scaled previous velocity) added to the negative scaled gradient $-\alpha \nabla L(\mathbf{w}_t)$. The result $\mathbf{v}_{t+1}$ is the new velocity vector. The second equation adds the velocity vector $\mathbf{v}_{t+1}$ to the current weight vector $\mathbf{w}_t$ to produce the updated weights. Both are component-wise vector additions. The momentum addition accumulates past gradient directions (like a weighted sum over time), smoothing the optimisation path and helping escape local minima.*

## Practice Problems

### Easy - 5 Questions

1. Let $\mathbf{u} = (2, 5)^T$ and $\mathbf{v} = (-3, 1)^T$. Compute $\mathbf{u} + \mathbf{v}$.

2. Let $\mathbf{a} = (1, -2, 3)^T$ and $\mathbf{b} = (4, 0, -1)^T$. Compute $\mathbf{a} + \mathbf{b}$.

3. Compute $(1, 2)^T + (3, -4)^T + (-2, 1)^T$.

4. If $\mathbf{v} = (3, -1, 2, 0)^T$, what is $\mathbf{v} + \mathbf{0}$?

5. Let $\mathbf{p} = (0.5, 0.3)^T$ and $\mathbf{q} = (0.2, -0.1)^T$. Compute $\mathbf{p} + \mathbf{q}$.

### Medium - 5 Questions

1. Let $\mathbf{u} = (1, -2, 3)^T$, $\mathbf{v} = (4, 0, -1)^T$, $\mathbf{w} = (-2, 1, 0)^T$. Compute $\mathbf{u} + \mathbf{v} + \mathbf{w}$.

2. A micro-batch gradient accumulation produces $\nabla L_1 = (0.1, -0.2, 0.3)^T$, $\nabla L_2 = (-0.4, 0.5, -0.1)^T$, $\nabla L_3 = (0.2, 0.1, -0.2)^T$. Compute the total accumulated gradient.

3. Verify commutativity for $\mathbf{a} = (3, -1)^T$ and $\mathbf{b} = (-2, 4)^T$ by computing $\mathbf{a} + \mathbf{b}$ and $\mathbf{b} + \mathbf{a}$.

4. A ResNet block computes $\mathbf{y} = F(\mathbf{x}) + \mathbf{x}$. If $\mathbf{x} = (0.5, -0.3, 0.1)^T$ and $F(\mathbf{x}) = (0.2, 0.4, -0.5)^T$, compute $\mathbf{y}$.

5. The query, key, and value vectors in attention are $\mathbf{q} = (1, 0)^T$, $\mathbf{k} = (0.5, 0.5)^T$, $\mathbf{v} = (2, 1)^T$. If the attention weight for a single key is $\alpha = 0.8$, compute $\alpha \mathbf{v}$ and identify the vector addition step that occurs after all weighted values are collected.

### Hard - 3 Questions

1. Prove that vector addition is associative using the component-wise definition. Show every step.

2. In a neural network with 3 layers, the activation vectors are: $\mathbf{a}^{(1)} = (0.2, 0.8)^T$, $\mathbf{a}^{(2)} = (0.6, 0.4)^T$, $\mathbf{a}^{(3)} = (0.9, 0.1)^T$. A skip connection adds $\mathbf{a}^{(1)}$ and $\mathbf{a}^{(3)}$ before the final classification layer. Compute the combined vector. Then, if a second skip connection also adds $\mathbf{a}^{(2)}$, compute the total combined vector. Show that the associative property means these additions can be done in any order.

3. A researcher accumulates gradients from 5 micro-batches:
   $$\nabla L_1 = (0.1, -0.3)^T, \nabla L_2 = (-0.2, 0.4)^T, \nabla L_3 = (0.3, -0.1)^T, \nabla L_4 = (-0.4, 0.2)^T, \nabla L_5 = (0.5, -0.2)^T$$
   Compute the accumulated gradient. Then, if the learning rate is $\alpha = 0.01$, compute the parameter update $\Delta \mathbf{w} = -\alpha \nabla L_{\text{total}}$.

## Solutions

### Easy Solutions

**1.** $\mathbf{u} + \mathbf{v} = (2 + (-3), 5 + 1)^T = (-1, 6)^T$.

**2.** $\mathbf{a} + \mathbf{b} = (1 + 4, -2 + 0, 3 + (-1))^T = (5, -2, 2)^T$.

**3.** Add component-wise:
$$(1 + 3 + (-2), 2 + (-4) + 1)^T = (2, -1)^T$$

**4.** $\mathbf{v} + \mathbf{0} = (3 + 0, -1 + 0, 2 + 0, 0 + 0)^T = (3, -1, 2, 0)^T$. (The vector is unchanged.)

**5.** $\mathbf{p} + \mathbf{q} = (0.5 + 0.2, 0.3 + (-0.1))^T = (0.7, 0.2)^T$.

### Medium Solutions

**1.** 
$$\mathbf{u} + \mathbf{v} + \mathbf{w} = (1 + 4 + (-2), -2 + 0 + 1, 3 + (-1) + 0)^T = (3, -1, 2)^T$$

**2.** 
$$\nabla L_{\text{total}} = (0.1 + (-0.4) + 0.2, -0.2 + 0.5 + 0.1, 0.3 + (-0.1) + (-0.2))^T = (-0.1, 0.4, 0.0)^T$$

**3.** $\mathbf{a} + \mathbf{b} = (3 + (-2), -1 + 4)^T = (1, 3)^T$.
$\mathbf{b} + \mathbf{a} = (-2 + 3, 4 + (-1))^T = (1, 3)^T$.
Both equal $(1, 3)^T$, confirming commutativity. ✓

**4.** $\mathbf{y} = F(\mathbf{x}) + \mathbf{x} = (0.2 + 0.5, 0.4 + (-0.3), -0.5 + 0.1)^T = (0.7, 0.1, -0.4)^T$.

**5.** $\alpha \mathbf{v} = 0.8 \times (2, 1)^T = (1.6, 0.8)^T$.
The vector addition step is summing all $\alpha_j \mathbf{v}_j$ for all keys in the attention mechanism:
$$\mathbf{c} = \sum_{j=1}^n \alpha_j \mathbf{v}_j$$
This produces the context vector from all weighted value vectors.

### Hard Solutions

**1.** Let $\mathbf{u} = (u_1, \ldots, u_n)^T$, $\mathbf{v} = (v_1, \ldots, v_n)^T$, $\mathbf{w} = (w_1, \ldots, w_n)^T$.

Left side: $(\mathbf{u} + \mathbf{v}) + \mathbf{w}$.
First compute $\mathbf{u} + \mathbf{v} = (u_1 + v_1, \ldots, u_n + v_n)^T$.
Then add $\mathbf{w}$: $((u_1 + v_1) + w_1, \ldots, (u_n + v_n) + w_n)^T$.

Right side: $\mathbf{u} + (\mathbf{v} + \mathbf{w})$.
First compute $\mathbf{v} + \mathbf{w} = (v_1 + w_1, \ldots, v_n + w_n)^T$.
Then add $\mathbf{u}$: $(u_1 + (v_1 + w_1), \ldots, u_n + (v_n + w_n))^T$.

Since scalar addition is associative, $(u_i + v_i) + w_i = u_i + (v_i + w_i)$ for each component $i$. Therefore:
$$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$$
Associativity of vector addition follows directly from associativity of scalar addition. ✓

**2.** First skip connection (add $\mathbf{a}^{(1)}$ and $\mathbf{a}^{(3)}$):
$$\mathbf{a}^{(1)} + \mathbf{a}^{(3)} = (0.2 + 0.9, 0.8 + 0.1)^T = (1.1, 0.9)^T$$

Now add $\mathbf{a}^{(2)}$ to get the total combined vector:
$$\mathbf{a}^{(1)} + \mathbf{a}^{(3)} + \mathbf{a}^{(2)} = (1.1 + 0.6, 0.9 + 0.4)^T = (1.7, 1.3)^T$$

By associativity, we could also compute differently:
- First add $\mathbf{a}^{(1)} + \mathbf{a}^{(2)} = (0.8, 1.2)^T$, then add $\mathbf{a}^{(3)}$: $(0.8 + 0.9, 1.2 + 0.1)^T = (1.7, 1.3)^T$. ✓
- Or add $\mathbf{a}^{(2)} + \mathbf{a}^{(3)} = (1.5, 0.5)^T$, then add $\mathbf{a}^{(1)}$: $(1.5 + 0.2, 0.5 + 0.8)^T = (1.7, 1.3)^T$. ✓

All orders give the same result.

**3.** Accumulated gradient:
$$\nabla L_{\text{total}} = (0.1 - 0.2 + 0.3 - 0.4 + 0.5, -0.3 + 0.4 - 0.1 + 0.2 - 0.2)^T = (0.3, 0.0)^T$$

Parameter update:
$$\Delta \mathbf{w} = -0.01 \times (0.3, 0.0)^T = (-0.003, 0.0)^T$$

The update vector has zero change in the second parameter and a small negative change in the first parameter.

## Related Concepts

- **Vector Subtraction (MATH-012):** The inverse operation of addition — adding the negative of a vector.
- **Vector (MATH-002):** The fundamental object being added.
- **Scalar Multiplication (MATH-013):** Often combined with addition to form linear combinations.
- **Scalar (MATH-001):** The individual numbers that make up vector components.
- **Matrix Addition (MATH-003):** The same component-wise addition generalised to matrices.
- **Dot Product (MATH-016):** Combines two vectors to produce a scalar.

## Next Concepts

- **Linear Combinations (MATH-020):** Weighted sums of vectors using scalar multiplication and addition.
- **Vector Space (MATH-031):** A formal structure where vector addition and scalar multiplication obey specific axioms.
- **Basis (MATH-032):** A minimal set of vectors whose linear combinations span the entire space.
- **Linear Transformations (MATH-035):** Functions that preserve vector addition and scalar multiplication.

## Summary

Vector addition is the operation of combining two or more vectors of the same dimension by adding their corresponding components. It is the most fundamental operation in vector algebra and is used everywhere in AI and machine learning — from gradient accumulation in training to residual connections in deep networks, attention mechanisms in transformers, and ensemble averaging. Vector addition is commutative ($\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$), associative ($(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$), and has the zero vector as its identity element. Geometrically, it can be visualised using the tip-to-tail rule or the parallelogram rule.

## Key Takeaways

- Vector addition is performed **component-wise**: $\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots)^T$.
- Only vectors of the **same dimension** can be added.
- Vector addition is **commutative** ($\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$) and **associative** ($(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$).
- The **zero vector** $\mathbf{0}$ is the additive identity: $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
- Every vector $\mathbf{v}$ has an **additive inverse** $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
- Geometric interpretations include the **tip-to-tail rule** and the **parallelogram rule**.
- In AI/ML, vector addition is used in gradient accumulation, residual connections, ensemble averaging, attention mechanisms, and embedding analogies.
- Vector addition is the foundation of **linear combinations**, which are central to all of linear algebra and deep learning.
