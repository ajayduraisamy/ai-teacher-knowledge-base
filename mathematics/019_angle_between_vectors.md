# Concept: Angle Between Vectors

## Concept ID

MATH-019

## Difficulty

INTERMEDIATE

## Domain

Mathematics

## Module

Vector Algebra

## Learning Objectives

- Compute the angle between two vectors using the dot product
- Understand cosine similarity and its interpretation
- Identify orthogonal vectors ($\theta = 90^\circ$)
- Apply angle formulas to semantic search and recommendation systems
- Connect cosine similarity to embedding-based machine learning

## Prerequisites

- Dot product (MATH-016)
- Vector magnitude: $\|\mathbf{u}\| = \sqrt{\sum u_i^2}$
- Basic trigonometry: $\cos\theta$, $\cos^{-1}$ (arccos)
- Understanding of radians and degrees

## Definition

The **angle between two vectors** $\mathbf{u}$ and $\mathbf{v}$ is the smaller angle $\theta$ ($0 \leq \theta \leq \pi$) formed when the vectors are placed tail-to-tail. It is given by:

$$
\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

Solving for $\theta$:

$$
\theta = \cos^{-1}\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)
$$

The quantity $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ is called the **cosine similarity**, often denoted $\cos(\mathbf{u}, \mathbf{v})$ or simply $\cos\theta$.

## Intuition

The angle between vectors tells you how closely their directions align. Two vectors pointing the same way have angle $0^\circ$ (cosine = 1). Perpendicular vectors have angle $90^\circ$ (cosine = 0). Opposite vectors have angle $180^\circ$ (cosine = -1).

Think of it like two arrows on a compass: one pointing north-east, the other pointing north. The angle between them is small (they mostly agree on direction). One pointing north and one pointing south have a large angle (they oppose each other).

## Why This Concept Matters

Angle measurement between vectors is the foundation of similarity in high-dimensional spaces:

- It normalises for vector length, giving a pure measure of directional agreement
- Cosine similarity is the most widely used similarity metric in NLP and information retrieval
- Orthogonality (angle $= 90^\circ$) is a core concept in linear algebra, indicating independence
- Many machine learning algorithms rely on vector similarity for classification, clustering, and retrieval

## Historical Background

The relationship between the dot product and the angle between vectors was formalised in the 19th century. **Augustin-Louis Cauchy** (1789–1857) proved the Cauchy-Schwarz inequality, which provides the theoretical justification: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$, ensuring that $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ always lies between $-1$ and $1$. The cosine similarity became the standard similarity measure in vector space models of information retrieval (Salton, 1970s) and was adopted by the NLP community for word embeddings (Mikolov et al., 2013).

## Real World Examples

1. **Navigation:** The angle between a ship's heading vector and the wind vector determines sailing efficiency. Close to $0^\circ$ means a tailwind; close to $90^\circ$ means a crosswind.
2. **Astronomy:** The angle between vectors from Earth to two stars determines their angular separation in the sky.
3. **Sports Analytics:** The angle between a player's running direction and the vector to the goal determines shot quality.
4. **Robotics:** A robot's arm orientation relative to a target is measured by the angle between direction vectors.
5. **Geography:** The angle between two GPS position vectors (from Earth's centre) determines the great-circle distance between two points on the globe.

## AI/ML Relevance

Cosine similarity (which is exactly the cosine of the angle between vectors) is one of the most important similarity measures in AI:

1. **Semantic Search:** When documents and queries are embedded into vector space, the search ranks documents by cosine similarity between the query embedding and document embeddings. The smaller the angle, the more semantically related the content.

2. **Recommendation Systems:** User and item embeddings are compared using cosine similarity. A small angle between a user vector and a movie vector indicates the user will like that movie. Systems like YouTube and Netflix use this approach.

3. **Word Embeddings (Word2Vec, GloVe):** The classic example is $\cos(\text{"king"} - \text{"man"} + \text{"woman"}, \text{"queen"}) \approx 1$, showing that analogy relationships are captured by vector arithmetic and angle measurement.

4. **Clustering (Spherical k-Means):** When clustering text data, cosine distance (1 - cosine similarity) is used instead of Euclidean distance because document vectors are typically length-normalised.

5. **Face Recognition:** Face embeddings (e.g., from FaceNet) are compared using cosine similarity. Two face images of the same person have embeddings with a small angle between them.

6. **Anomaly Detection:** Data points whose angle from the mean vector exceeds a threshold may be flagged as anomalies.

## Mathematical Explanation

From the dot product, we have two equivalent definitions:
$$
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
$$

Rearranging:
$$
\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

The angle is then $\theta = \cos^{-1}\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)$.

The Cauchy-Schwarz inequality guarantees $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$, so:
$$
-1 \leq \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} \leq 1
$$

This means:
- $\cos\theta = 1$ when $\theta = 0^\circ$: vectors are parallel and in the same direction
- $\cos\theta = 0$ when $\theta = 90^\circ$: vectors are orthogonal (perpendicular)
- $\cos\theta = -1$ when $\theta = 180^\circ$: vectors are parallel but opposite

The angle is always measured as the smaller angle between the two vectors, so $\theta \in [0, \pi]$.

**Cosine distance** is defined as $1 - \cos\theta$, which ranges from 0 (identical direction) to 2 (opposite direction).

## Formula(s)

**Angle between vectors:**
$$
\theta = \cos^{-1}\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)
$$

**Cosine similarity:**
$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

**Cosine distance:**
$$
d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \cos(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

**Orthogonality condition:**
$$
\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0 \iff \cos\theta = 0 \iff \theta = 90^\circ
$$

## Properties

1. **Range:** $0 \leq \theta \leq \pi$ (or $0^\circ \leq \theta \leq 180^\circ$)
2. **Cosine similarity range:** $-1 \leq \cos\theta \leq 1$
3. **Symmetry:** $\cos(\mathbf{u}, \mathbf{v}) = \cos(\mathbf{v}, \mathbf{u})$
4. **Scale invariance:** $\cos(c\mathbf{u}, \mathbf{v}) = \cos(\mathbf{u}, \mathbf{v})$ for $c > 0$ (the angle does not depend on vector magnitude)
5. **Orthogonality:** $\cos\theta = 0$ iff $\mathbf{u} \perp \mathbf{v}$
6. **Parallel:** $\cos\theta = 1$ iff $\mathbf{u}$ and $\mathbf{v}$ point in the same direction; $\cos\theta = -1$ iff they point in opposite directions
7. **Triangle inequality for angles:** The angle from $\mathbf{u}$ to $\mathbf{w}$ is at most the sum of angles from $\mathbf{u}$ to $\mathbf{v}$ and $\mathbf{v}$ to $\mathbf{w}$
8. **Relation to Euclidean distance for unit vectors:** If $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$, then $\|\mathbf{u} - \mathbf{v}\|^2 = 2(1 - \cos\theta)$

## Step-by-Step Worked Examples

### Example 1: Angle Between 2D Vectors

**Problem:** Find the angle between $\mathbf{u} = \langle 3, 4 \rangle$ and $\mathbf{v} = \langle 1, 2 \rangle$.

**Solution:**

**Step 1:** Compute the dot product:
$$
\mathbf{u} \cdot \mathbf{v} = 3(1) + 4(2) = 3 + 8 = 11
$$

**Step 2:** Compute magnitudes:
$$
\|\mathbf{u}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$
$$
\|\mathbf{v}\| = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5}
$$

**Step 3:** Compute cosine similarity:
$$
\cos\theta = \frac{11}{5 \times \sqrt{5}} = \frac{11}{5\sqrt{5}}
$$

**Step 4:** Rationalise: $\frac{11}{5\sqrt{5}} = \frac{11\sqrt{5}}{25} \approx 0.9839$.

**Step 5:** Compute the angle:
$$
\theta = \cos^{-1}(0.9839) \approx 10.3^\circ
$$

**Step 6:** Interpretation: The vectors point in very similar directions (small angle).

### Example 2: Orthogonal Vectors

**Problem:** Verify that $\mathbf{u} = \langle 2, -1, 3 \rangle$ and $\mathbf{v} = \langle 4, 5, -1 \rangle$ are orthogonal and find the angle between them.

**Solution:**

**Step 1:** Compute dot product:
$$
\mathbf{u} \cdot \mathbf{v} = 2(4) + (-1)(5) + 3(-1) = 8 - 5 - 3 = 0
$$

**Step 2:** Since the dot product is zero, the vectors are orthogonal.

**Step 3:** The angle is:
$$
\cos\theta = \frac{0}{\|\mathbf{u}\|\|\mathbf{v}\|} = 0 \implies \theta = \cos^{-1}(0) = 90^\circ
$$

**Step 4:** Confirmation: We did not even need to compute the magnitudes. The angle is exactly $90^\circ$.

### Example 3: Cosine Similarity in NLP (Word Embeddings)

**Problem:** Three word embeddings (hypothetical 3D vectors) are:
- "king" = $\langle 0.8, 0.5, 0.3 \rangle$
- "queen" = $\langle 0.7, 0.6, 0.4 \rangle$
- "apple" = $\langle 0.1, 0.3, 0.9 \rangle$

Compute the cosine similarity between "king" and "queen", and between "king" and "apple". Which pair is more semantically similar?

**Solution:**

**Step 1:** Cosine similarity between "king" and "queen":
- Dot: $0.8(0.7) + 0.5(0.6) + 0.3(0.4) = 0.56 + 0.30 + 0.12 = 0.98$
- $\|\text{king}\| = \sqrt{0.64 + 0.25 + 0.09} = \sqrt{0.98} \approx 0.990$
- $\|\text{queen}\| = \sqrt{0.49 + 0.36 + 0.16} = \sqrt{1.01} \approx 1.005$
- $\cos\theta_{\text{king,queen}} = \frac{0.98}{0.990 \times 1.005} \approx 0.985$

**Step 2:** Cosine similarity between "king" and "apple":
- Dot: $0.8(0.1) + 0.5(0.3) + 0.3(0.9) = 0.08 + 0.15 + 0.27 = 0.50$
- $\|\text{apple}\| = \sqrt{0.01 + 0.09 + 0.81} = \sqrt{0.91} \approx 0.954$
- $\cos\theta_{\text{king,apple}} = \frac{0.50}{0.990 \times 0.954} \approx 0.530$

**Step 3:** Interpretation: The cosine similarity between "king" and "queen" (0.985) is much higher than between "king" and "apple" (0.530). This matches our intuition — kings and queens are semantically related royalty terms, while apples are unrelated.

### Example 4: Finding the Angle in 3D

**Problem:** Find $\theta$ between $\mathbf{u} = \langle 1, -1, 2 \rangle$ and $\mathbf{v} = \langle 3, 0, 4 \rangle$.

**Solution:**

**Step 1:** Dot product:
$$
\mathbf{u} \cdot \mathbf{v} = 1(3) + (-1)(0) + 2(4) = 3 + 0 + 8 = 11
$$

**Step 2:** Magnitudes:
$$
\|\mathbf{u}\| = \sqrt{1^2 + (-1)^2 + 2^2} = \sqrt{1 + 1 + 4} = \sqrt{6}
$$
$$
\|\mathbf{v}\| = \sqrt{3^2 + 0^2 + 4^2} = \sqrt{9 + 0 + 16} = \sqrt{25} = 5
$$

**Step 3:** Cosine similarity:
$$
\cos\theta = \frac{11}{\sqrt{6} \times 5} = \frac{11}{5\sqrt{6}} = \frac{11\sqrt{6}}{30} \approx 0.898
$$

**Step 4:** Angle:
$$
\theta = \cos^{-1}(0.898) \approx 26.0^\circ
$$

### Example 5: Opposite Direction Vectors

**Problem:** Find the angle between $\mathbf{u} = \langle -2, 4, -6 \rangle$ and $\mathbf{v} = \langle 1, -2, 3 \rangle$.

**Solution:**

**Step 1:** Dot product:
$$
\mathbf{u} \cdot \mathbf{v} = (-2)(1) + 4(-2) + (-6)(3) = -2 - 8 - 18 = -28
$$

**Step 2:** Magnitudes:
$$
\|\mathbf{u}\| = \sqrt{4 + 16 + 36} = \sqrt{56} = 2\sqrt{14}
$$
$$
\|\mathbf{v}\| = \sqrt{1 + 4 + 9} = \sqrt{14}
$$

**Step 3:** Cosine similarity:
$$
\cos\theta = \frac{-28}{(2\sqrt{14})(\sqrt{14})} = \frac{-28}{2 \times 14} = \frac{-28}{28} = -1
$$

**Step 4:** Angle:
$$
\theta = \cos^{-1}(-1) = 180^\circ = \pi \text{ radians}
$$

**Step 5:** Interpretation: The vectors point in exactly opposite directions. Indeed, $\mathbf{u} = -2\mathbf{v}$.

## Visual Interpretation

Imagine two vectors as arrows from the origin. The angle between them is the smallest rotation needed to align one arrow with the other.

In 2D, this is straightforward: you can measure $\theta$ with a protractor. The cosine of $\theta$ tells you the fraction of one vector's length that you get when you take the dot product. 

For unit vectors (length 1), the cosine similarity is simply the dot product. The vectors live on a unit circle (in 2D) or a unit sphere (in 3D), and the cosine similarity is the $x$-coordinate of the second vector when the first is aligned with the $x$-axis.

In high dimensions (used in AI), we cannot visualise the space, but the cosine similarity still works identically: it measures the directional agreement between vectors regardless of their magnitudes.

## Common Mistakes

1. **Forgetting to divide by magnitudes:** Cosine similarity is $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$, not just $\mathbf{u} \cdot \mathbf{v}$. A large dot product does not necessarily mean a small angle — the vectors might just be long.

2. **Using degrees instead of radians:** If using a calculator, ensure the mode matches the unit you want. The formula $\theta = \cos^{-1}(x)$ typically returns radians in mathematical software.

3. **Assuming $\theta$ is always acute:** The angle between vectors can be up to $180^\circ$. The formula automatically gives the correct obtuse angle when $\cos\theta$ is negative.

4. **Thinking angle is symmetric with magnitude:** The angle is independent of magnitude (scale-invariant), but the dot product itself is not. Doubling the length of $\mathbf{u}$ doubles the dot product but does not change the angle.

5. **Confusing cosine similarity with cosine distance:** Cosine similarity $\cos\theta$ ranges from $-1$ to $1$. Cosine distance $1 - \cos\theta$ ranges from $0$ to $2$. A cosine similarity of $0.5$ corresponds to a cosine distance of $0.5$, not an angle of $0.5$ radians.

6. **Using Euclidean distance when cosine similarity is appropriate:** In text analysis, document vectors are often compared by angle (cosine), not by straight-line distance (Euclidean), because two documents with different lengths may have identical topic proportions.

7. **Forgetting the domain of arccos:** $\cos^{-1}(x)$ is defined only for $x \in [-1, 1]$. Due to floating-point errors, computed values outside this range should be clamped.

## Interview Questions

### Beginner

1. How do you compute the angle between two vectors?
2. What is cosine similarity? What range of values can it take?
3. If two vectors are orthogonal, what is their cosine similarity?
4. What angle corresponds to a cosine similarity of 1? Of -1?
5. Compute $\cos\theta$ for $\mathbf{u} = \langle 1, 0 \rangle$ and $\mathbf{v} = \langle 0, 1 \rangle$.

### Intermediate

1. Prove that $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$.
2. Why is cosine similarity preferred over Euclidean distance for comparing text documents?
3. If $\cos(\mathbf{u}, \mathbf{v}) = 0.6$ and $\|\mathbf{u}\| = 2$, $\|\mathbf{v}\| = 3$, find $\mathbf{u} \cdot \mathbf{v}$.
4. Show that for unit vectors, $\|\mathbf{u} - \mathbf{v}\| = \sqrt{2(1 - \cos\theta)}$.
5. How does cosine similarity relate to the attention mechanism in transformers?

### Advanced

1. Prove the Cauchy-Schwarz inequality $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$ and explain why it guarantees $\cos\theta \in [-1, 1]$.
2. For a set of $n$ vectors, the Gram matrix $G$ has entries $G_{ij} = \mathbf{v}_i \cdot \mathbf{v}_j$. Show that the cosine similarity matrix $C_{ij} = \frac{G_{ij}}{\sqrt{G_{ii}G_{jj}}}$ is positive semidefinite.
3. In transformer models, scaled dot-product attention uses $\frac{QK^T}{\sqrt{d_k}}$ instead of raw dot products. How does this scaling relate to stabilising cosine similarity values for high-dimensional vectors?

## Practice Problems

### Easy - 5 Questions

1. Find $\theta$ between $\mathbf{u} = \langle 1, 1 \rangle$ and $\mathbf{v} = \langle 1, 0 \rangle$.
2. Compute cosine similarity between $\mathbf{u} = \langle 2, 3 \rangle$ and $\mathbf{v} = \langle -1, 2 \rangle$.
3. Are $\mathbf{u} = \langle 1, 2, -3 \rangle$ and $\mathbf{v} = \langle 2, -2, 0 \rangle$ orthogonal?
4. What is the angle between $\mathbf{u} = \langle 5, 0 \rangle$ and $\mathbf{v} = \langle -3, 0 \rangle$?
5. If $\|\mathbf{u}\| = 4$, $\|\mathbf{v}\| = 3$, and $\mathbf{u} \cdot \mathbf{v} = 6$, find $\cos\theta$.

### Medium - 5 Questions

6. Find the angle between $\mathbf{u} = \langle 1, -2, 1 \rangle$ and $\mathbf{v} = \langle 2, 1, -1 \rangle$.
7. For unit vectors $\mathbf{u}$ and $\mathbf{v}$, if $\|\mathbf{u} - \mathbf{v}\| = 1$, what is $\cos\theta$?
8. Compute cosine similarity between $\mathbf{u} = \langle 0.2, 0.8, 0.4, 0.1 \rangle$ and $\mathbf{v} = \langle 0.1, 0.3, 0.9, 0.2 \rangle$.
9. Verify that $\cos(2\mathbf{u}, 3\mathbf{v}) = \cos(\mathbf{u}, \mathbf{v})$.
10. Two vectors have cosine similarity 0.8. What is the cosine distance? What is $\theta$ in degrees?

### Hard - 3 Questions

11. Show that for any non-zero vectors $\mathbf{u}$ and $\mathbf{v}$, $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \|\mathbf{u}\|\cos\theta \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$.
12. Given three vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$ with pairwise cosine similarities $\cos(\mathbf{a}, \mathbf{b}) = 0.5$, $\cos(\mathbf{b}, \mathbf{c}) = 0.5$, and $\cos(\mathbf{a}, \mathbf{c}) = 0.5$, find the angles between them. Are these values consistent? (Hint: consider the Gram matrix and check positive semidefiniteness.)
13. In an inner product space, the angle between two vectors is defined by $\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|\|\mathbf{v}\|}$. Prove the law of cosines: $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$.

## Solutions

### Easy Solutions

1. $\mathbf{u} \cdot \mathbf{v} = 1(1) + 1(0) = 1$. $\|\mathbf{u}\| = \sqrt{2}$, $\|\mathbf{v}\| = 1$. $\cos\theta = \frac{1}{\sqrt{2}} = \frac{\sqrt{2}}{2}$. $\theta = \cos^{-1}(\frac{\sqrt{2}}{2}) = 45^\circ$.

2. $\mathbf{u} \cdot \mathbf{v} = 2(-1) + 3(2) = -2 + 6 = 4$. $\|\mathbf{u}\| = \sqrt{4 + 9} = \sqrt{13}$. $\|\mathbf{v}\| = \sqrt{1 + 4} = \sqrt{5}$. $\cos\theta = \frac{4}{\sqrt{65}} \approx 0.496$.

3. $\mathbf{u} \cdot \mathbf{v} = 1(2) + 2(-2) + (-3)(0) = 2 - 4 + 0 = -2 \neq 0$. Not orthogonal.

4. $\mathbf{u} \cdot \mathbf{v} = 5(-3) + 0(0) = -15$. $\|\mathbf{u}\| = 5$, $\|\mathbf{v}\| = 3$. $\cos\theta = \frac{-15}{15} = -1$. $\theta = 180^\circ$.

5. $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \frac{6}{4 \times 3} = \frac{6}{12} = 0.5$.

### Medium Solutions

6. $\mathbf{u} \cdot \mathbf{v} = 1(2) + (-2)(1) + 1(-1) = 2 - 2 - 1 = -1$.
   $\|\mathbf{u}\| = \sqrt{1 + 4 + 1} = \sqrt{6}$.
   $\|\mathbf{v}\| = \sqrt{4 + 1 + 1} = \sqrt{6}$.
   $\cos\theta = \frac{-1}{6}$.
   $\theta = \cos^{-1}(-\frac{1}{6}) \approx 99.6^\circ$.

7. $\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v} = 1 + 1 - 2\cos\theta = 2 - 2\cos\theta$.
   Given $\|\mathbf{u} - \mathbf{v}\| = 1$: $1 = 2 - 2\cos\theta$.
   So $2\cos\theta = 2 - 1 = 1$, giving $\cos\theta = 0.5$.

8. $\mathbf{u} \cdot \mathbf{v} = 0.2(0.1) + 0.8(0.3) + 0.4(0.9) + 0.1(0.2) = 0.02 + 0.24 + 0.36 + 0.02 = 0.64$.
   $\|\mathbf{u}\| = \sqrt{0.04 + 0.64 + 0.16 + 0.01} = \sqrt{0.85} \approx 0.922$.
   $\|\mathbf{v}\| = \sqrt{0.01 + 0.09 + 0.81 + 0.04} = \sqrt{0.95} \approx 0.975$.
   $\cos\theta = \frac{0.64}{0.922 \times 0.975} \approx 0.712$.

9. $\cos(2\mathbf{u}, 3\mathbf{v}) = \frac{(2\mathbf{u}) \cdot (3\mathbf{v})}{\|2\mathbf{u}\|\|3\mathbf{v}\|} = \frac{6(\mathbf{u} \cdot \mathbf{v})}{6\|\mathbf{u}\|\|\mathbf{v}\|} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \cos(\mathbf{u}, \mathbf{v})$.

10. Cosine distance $= 1 - 0.8 = 0.2$. $\theta = \cos^{-1}(0.8) \approx 36.87^\circ$.

### Hard Solutions

11. $\operatorname{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v} = \frac{\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta}{\|\mathbf{v}\|^2} \mathbf{v} = \|\mathbf{u}\|\cos\theta \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$.
    This shows the projection has length $\|\mathbf{u}\||\cos\theta|$ and direction $\frac{\mathbf{v}}{\|\mathbf{v}\|}$ (the unit vector in the $\mathbf{v}$ direction).

12. All angles are $\theta = \cos^{-1}(0.5) = 60^\circ$.
    The Gram matrix for three unit vectors with pairwise cosine $0.5$ is:
    $$
    G = \begin{pmatrix}
    1 & 0.5 & 0.5 \\
    0.5 & 1 & 0.5 \\
    0.5 & 0.5 & 1
    \end{pmatrix}
    $$
    Check positive semidefiniteness: the eigenvalues are $\lambda_1 = 2$ (multiplicity 1), $\lambda_2 = \lambda_3 = 0.5$ (multiplicity 2). All are non-negative, so $G$ is PSD. The vectors can exist (e.g., in $\mathbb{R}^3$, place them at equal angles of $60^\circ$ from each other, like the vertices of a regular tetrahedron from the centre).

13. $\|\mathbf{u} - \mathbf{v}\|^2 = (\mathbf{u} - \mathbf{v}) \cdot (\mathbf{u} - \mathbf{v}) = \mathbf{u} \cdot \mathbf{u} - 2(\mathbf{u} \cdot \mathbf{v}) + \mathbf{v} \cdot \mathbf{v} = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$.
    This is the vector form of the law of cosines, where $c^2 = a^2 + b^2 - 2ab\cos C$ is the familiar trigonometric law of cosines for a triangle with sides $a, b, c$ and angle $C$ opposite side $c$.

## Related Concepts

- **Dot Product** (MATH-016): The numerator in the angle formula
- **Vector Projection** (MATH-018): $\|\mathbf{u}\|\cos\theta$ is the scalar projection
- **Distance Between Vectors** (MATH-020): Euclidean distance relates to angle for unit vectors
- **Cross Product** (MATH-017): $\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\|\|\mathbf{v}\|\sin\theta$

## Next Concepts

- **021 Cosine Similarity and Distance Metrics**: Deep dive into similarity measures
- **035 Embeddings and Vector Spaces**: How angles define semantic relationships
- **040 k-Nearest Neighbours**: Classification based on vector similarity
- **045 Recommendation Systems**: Collaborative filtering using cosine similarity

## Summary

The angle between vectors $\mathbf{u}$ and $\mathbf{v}$ is given by $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$, where $\theta \in [0, \pi]$. This quantity, known as cosine similarity, ranges from $-1$ to $1$ and measures purely directional agreement independent of vector magnitude. A zero dot product implies orthogonality ($90^\circ$). Cosine similarity is the dominant similarity metric in NLP, information retrieval, recommendation systems, and embedding-based machine learning.

## Key Takeaways

- $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ defines the angle between vectors
- Cosine similarity is scale-invariant (independent of vector length)
- $\cos\theta = 0 \iff \mathbf{u} \perp \mathbf{v}$ (orthogonality)
- $\cos\theta = 1$: same direction; $\cos\theta = -1$: opposite direction
- Cosine similarity is the foundation of semantic search, word embeddings, and recommendation systems
- Cosine distance $= 1 - \cos\theta$ is a proper distance metric for unit vectors
