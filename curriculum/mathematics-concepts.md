# Mathematics Curriculum — 100 Concepts

## Overview

This curriculum covers the foundational mathematics required for AI, Machine Learning, and Deep Learning. It is organized into 7 modules progressing from basic building blocks to advanced optimization algorithms.

**Total Concepts:** 100
**Difficulty Range:** Beginner → Intermediate → Advanced
**Prerequisite Structure:** Each module builds on the previous; concepts within a module are ordered from foundational to applied.

---

## Module 1: Foundations (001–010)

*Building blocks of mathematical notation and representation.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 001 | Scalar                | Beginner    | A single numerical value; the simplest mathematical object |
| 002 | Vector                | Beginner    | An ordered array of numbers representing magnitude and direction |
| 003 | Matrix                | Beginner    | A rectangular array of numbers arranged in rows and columns |
| 004 | Tensor                | Intermediate| A multi-dimensional generalization of scalars, vectors, and matrices |
| 005 | Dimension             | Beginner    | The number of coordinates or axes needed to represent an object |
| 006 | Coordinate System     | Beginner    | A system for uniquely identifying points in space |
| 007 | Number Systems        | Beginner    | Natural, integer, rational, real, and complex numbers |
| 008 | Real Numbers          | Beginner    | Numbers representing continuous quantities on the number line |
| 009 | Complex Numbers       | Intermediate| Numbers of the form $a + bi$ extending the real number line |
| 010 | Mathematical Notation | Beginner    | Standard symbols and conventions for mathematical expression |

---

## Module 2: Vector Algebra (011–020)

*Operations and properties of vectors — the language of geometry and physics in ML.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 011 | Vector Addition       | Beginner    | Combining vectors component-wise |
| 012 | Vector Subtraction    | Beginner    | Difference between vectors component-wise |
| 013 | Scalar Multiplication | Beginner    | Scaling a vector by a real number |
| 014 | Vector Magnitude      | Beginner    | The length or norm of a vector |
| 015 | Unit Vector           | Beginner    | A vector with magnitude 1, indicating direction only |
| 016 | Dot Product           | Intermediate| Scalar product of two vectors measuring alignment |
| 017 | Cross Product         | Intermediate| Vector product yielding a vector perpendicular to both inputs |
| 018 | Vector Projection     | Intermediate| Projecting one vector onto another |
| 019 | Angle Between Vectors | Intermediate| Geometric angle derived from the dot product |
| 020 | Distance Between Vectors | Intermediate | Euclidean and other distance metrics |

---

## Module 3: Matrix Algebra (021–030)

*Matrix operations — the workhorse of linear algebra in ML.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 021 | Matrix Addition       | Beginner    | Element-wise addition of matrices |
| 022 | Matrix Subtraction    | Beginner    | Element-wise subtraction of matrices |
| 023 | Matrix Multiplication | Intermediate| Dot product of rows and columns across matrices |
| 024 | Transpose             | Beginner    | Flipping a matrix over its diagonal |
| 025 | Identity Matrix       | Beginner    | The multiplicative identity in matrix algebra |
| 026 | Inverse Matrix        | Advanced    | The matrix analog of division |
| 027 | Determinant           | Intermediate| Scalar encoding volume scaling and invertibility |
| 028 | Rank                  | Intermediate| The dimension of the column/row space |
| 029 | Trace                 | Intermediate| Sum of diagonal elements |
| 030 | Orthogonal Matrix     | Intermediate| A matrix whose rows/columns form an orthonormal basis |

---

## Module 4: Linear Algebra (031–043)

*The conceptual foundation of linear algebra — spaces, transformations, and decompositions.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 031 | Vector Space          | Intermediate| A set of vectors closed under addition and scalar multiplication |
| 032 | Basis                 | Intermediate| A minimal set of independent vectors spanning a space |
| 033 | Span                  | Intermediate| All linear combinations of a set of vectors |
| 034 | Linear Independence   | Intermediate| Vectors that cannot be expressed as combinations of each other |
| 035 | Linear Dependence     | Intermediate| Vectors where at least one is a combination of others |
| 036 | Linear Transformation | Intermediate| A function preserving vector addition and scalar multiplication |
| 037 | Kernel                | Advanced    | Vectors mapped to zero by a linear transformation |
| 038 | Image                 | Advanced    | The set of all outputs of a linear transformation |
| 039 | Eigenvalue            | Advanced    | The scaling factor of an eigenvector under a transformation |
| 040 | Eigenvector           | Advanced    | A non-zero vector that only scales under a transformation |
| 041 | Diagonalization       | Advanced    | Factoring a matrix into a diagonal form using eigenvectors |
| 042 | SVD                   | Advanced    | Singular Value Decomposition — factoring any matrix |
| 043 | PCA                   | Advanced    | Principal Component Analysis — dimensionality reduction via SVD |

---

## Module 5: Functions (044–063)

*Functions are the core of mathematical modeling and neural networks.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 044 | Function              | Beginner    | A mapping from inputs to outputs |
| 045 | Domain                | Beginner    | The set of valid inputs to a function |
| 046 | Range                 | Beginner    | The set of possible outputs of a function |
| 047 | Composite Function    | Intermediate| Applying one function to the output of another |
| 048 | Inverse Function      | Intermediate| A function that reverses another function |
| 049 | Polynomial Function   | Beginner    | Functions of the form $f(x) = a_n x^n + \dots + a_0$ |
| 050 | Exponential Function  | Intermediate| Functions of the form $f(x) = a^x$ |
| 051 | Logarithmic Function  | Intermediate| The inverse of exponential functions |
| 052 | Trigonometric Function| Intermediate| Sine, cosine, tangent and their properties |

---

## Module 6: Calculus (053–064)

*The mathematics of change — essential for optimization and training.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 053 | Limits                | Intermediate| The behavior of a function as input approaches a value |
| 054 | Continuity            | Intermediate| Functions without jumps or breaks |
| 055 | Derivative            | Intermediate| Instantaneous rate of change |
| 056 | Partial Derivative    | Intermediate| Derivative with respect to one variable while holding others constant |
| 057 | Chain Rule            | Intermediate| Derivative of composite functions |
| 058 | Gradient              | Intermediate| Vector of all partial derivatives |
| 059 | Jacobian              | Advanced    | Matrix of all first-order partial derivatives |
| 060 | Hessian               | Advanced    | Matrix of all second-order partial derivatives |
| 061 | Integral              | Intermediate| Accumulation of quantities or area under a curve |
| 062 | Definite Integral     | Intermediate| Integral over a specific interval |
| 063 | Indefinite Integral   | Intermediate| General antiderivative with constant of integration |
| 064 | Multiple Integrals    | Advanced    | Integration over multi-dimensional domains |

---

## Module 7: Probability (065–076)

*Quantifying uncertainty — fundamental for probabilistic ML and statistics.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 065 | Probability           | Beginner    | A number between 0 and 1 quantifying likelihood |
| 066 | Sample Space          | Beginner    | The set of all possible outcomes |
| 067 | Event                 | Beginner    | A subset of the sample space |
| 068 | Conditional Probability| Intermediate| Probability of an event given another has occurred |
| 069 | Bayes Theorem         | Intermediate| Updating probabilities based on new evidence |
| 070 | Random Variable       | Intermediate| A variable whose values are outcomes of a random process |
| 071 | Probability Distribution | Intermediate| A function describing the likelihood of different outcomes |
| 072 | Bernoulli Distribution | Intermediate| Distribution for binary outcomes |
| 073 | Binomial Distribution | Intermediate| Distribution for count of successes in independent trials |
| 074 | Poisson Distribution  | Intermediate| Distribution for count of rare events |
| 075 | Normal Distribution   | Intermediate| The bell curve — central distribution in statistics |
| 076 | Central Limit Theorem | Advanced    | Sum of independent variables approximates a normal distribution |

---

## Module 8: Statistics (077–087)

*Summarizing and drawing conclusions from data.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 077 | Mean                  | Beginner    | The arithmetic average |
| 078 | Median                | Beginner    | The middle value of a sorted dataset |
| 079 | Mode                  | Beginner    | The most frequent value |
| 080 | Variance              | Intermediate| Average squared deviation from the mean |
| 081 | Standard Deviation    | Intermediate| Square root of variance — spread of data |
| 082 | Covariance            | Intermediate| How two variables vary together |
| 083 | Correlation           | Intermediate| Normalized covariance between -1 and 1 |
| 084 | Skewness              | Intermediate| Asymmetry of a distribution |
| 085 | Kurtosis              | Intermediate| Tailedness of a distribution |
| 086 | Confidence Interval   | Advanced    | Range containing the true parameter with a given confidence |
| 087 | Hypothesis Testing    | Advanced    | Framework for making data-driven decisions |

---

## Module 9: Information Theory (088–092)

*Quantifying information — fundamental for loss functions and model evaluation.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 088 | Entropy               | Intermediate| Average information content or uncertainty |
| 089 | Cross Entropy         | Intermediate| Expected surprise using an incorrect distribution |
| 090 | KL Divergence         | Advanced    | Measure of how one distribution diverges from another |
| 091 | Mutual Information    | Advanced    | Reduction in uncertainty of one variable given another |
| 092 | Information Gain      | Intermediate| Reduction in entropy used in decision trees |

---

## Module 10: Optimization (093–100)

*Finding the best parameters — the engine of ML training.*

| ID  | Concept               | Difficulty  | Description |
|-----|-----------------------|-------------|-------------|
| 093 | Convex Function       | Intermediate| Functions where any local minimum is global |
| 094 | Optimization          | Intermediate| The process of minimizing or maximizing a function |
| 095 | Gradient Descent      | Intermediate| Iterative first-order optimization |
| 096 | SGD                   | Intermediate| Stochastic Gradient Descent — using random subsets |
| 097 | Momentum              | Advanced    | Accelerating gradient descent with velocity |
| 098 | RMSProp               | Advanced    | Adaptive learning rate using root mean square |
| 099 | Adam                  | Advanced    | Adaptive Moment Estimation — combines momentum and RMSProp |
| 100 | Learning Rate Scheduling | Advanced | Strategies for adjusting the learning rate during training |

---

## Concept File Naming Convention

Files are named with a zero-padded three-digit ID followed by a lowercase underscore-separated slug:

```
{id:03d}_{slug}.md
```

**Examples:**

```
001_scalar.md
016_dot_product.md
039_eigenvalue.md
095_gradient_descent.md
```

## Module Directory Structure

```
mathematics/
├── 001_scalar.md
├── 002_vector.md
├── 003_matrix.md
├── 004_tensor.md
├── 005_dimension.md
├── 006_coordinate_system.md
├── 007_number_systems.md
├── 008_real_numbers.md
├── 009_complex_numbers.md
├── 010_mathematical_notation.md
├── 011_vector_addition.md
├── 012_vector_subtraction.md
├── 013_scalar_multiplication.md
├── 014_vector_magnitude.md
├── 015_unit_vector.md
├── 016_dot_product.md
├── 017_cross_product.md
├── 018_vector_projection.md
├── 019_angle_between_vectors.md
├── 020_distance_between_vectors.md
├── 021_matrix_addition.md
├── 022_matrix_subtraction.md
├── 023_matrix_multiplication.md
├── 024_transpose.md
├── 025_identity_matrix.md
├── 026_inverse_matrix.md
├── 027_determinant.md
├── 028_rank.md
├── 029_trace.md
├── 030_orthogonal_matrix.md
├── 031_vector_space.md
├── 032_basis.md
├── 033_span.md
├── 034_linear_independence.md
├── 035_linear_dependence.md
├── 036_linear_transformation.md
├── 037_kernel.md
├── 038_image.md
├── 039_eigenvalue.md
├── 040_eigenvector.md
├── 041_diagonalization.md
├── 042_singular_value_decomposition.md
├── 043_principal_component_analysis.md
├── 044_function.md
├── 045_domain.md
├── 046_range.md
├── 047_composite_function.md
├── 048_inverse_function.md
├── 049_polynomial_function.md
├── 050_exponential_function.md
├── 051_logarithmic_function.md
├── 052_trigonometric_function.md
├── 053_limits.md
├── 054_continuity.md
├── 055_derivative.md
├── 056_partial_derivative.md
├── 057_chain_rule.md
├── 058_gradient.md
├── 059_jacobian.md
├── 060_hessian.md
├── 061_integral.md
├── 062_definite_integral.md
├── 063_indefinite_integral.md
├── 064_multiple_integrals.md
├── 065_probability.md
├── 066_sample_space.md
├── 067_event.md
├── 068_conditional_probability.md
├── 069_bayes_theorem.md
├── 070_random_variable.md
├── 071_probability_distribution.md
├── 072_bernoulli_distribution.md
├── 073_binomial_distribution.md
├── 074_poisson_distribution.md
├── 075_normal_distribution.md
├── 076_central_limit_theorem.md
├── 077_mean.md
├── 078_median.md
├── 079_mode.md
├── 080_variance.md
├── 081_standard_deviation.md
├── 082_covariance.md
├── 083_correlation.md
├── 084_skewness.md
├── 085_kurtosis.md
├── 086_confidence_interval.md
├── 087_hypothesis_testing.md
├── 088_entropy.md
├── 089_cross_entropy.md
├── 090_kl_divergence.md
├── 091_mutual_information.md
├── 092_information_gain.md
├── 093_convex_function.md
├── 094_optimization.md
├── 095_gradient_descent.md
├── 096_stochastic_gradient_descent.md
├── 097_momentum.md
├── 098_rmsprop.md
├── 099_adam.md
├── 100_learning_rate_scheduling.md
```

---

## How to Use This Curriculum

### For Self-Learners

1. Start with Module 1 (Foundations) if you are new to mathematical notation.
2. Progress sequentially — each concept lists its prerequisites.
3. Complete all practice problems and review the worked examples.
4. Use the interview questions for self-assessment.

### For Educators

1. Assign concepts by module for course structuring.
2. Each concept file is self-contained and ready for lecture material.
3. Practice problems include full solutions for grading reference.
4. Difficulty levels help differentiate assignments.

### For AI/ML Practitioners

1. Use AI/ML Relevance sections to connect theory to practice.
2. Focus on Modules 3-4 (Linear Algebra) and 6 (Calculus) for understanding neural networks.
3. Module 10 (Optimization) is essential for training algorithms.
4. Modules 7-9 (Probability, Statistics, Information Theory) are key for evaluation and uncertainty.

---

## Concept File Template

Every concept file in this curriculum follows this exact structure:

```markdown
# Concept: {Concept Name}

## Concept ID

{three-digit ID}

## Difficulty

{BEGINNER | INTERMEDIATE | ADVANCED}

## Domain

Mathematics

## Learning Objectives

- Objective 1
- Objective 2
- Objective 3

## Prerequisites

- [Concept Name](...)

## Definition

Concise definition of the concept.

## Intuition

Simple language explanation building intuition before formulas.

## Why This Concept Matters

Why this concept is important in mathematics and beyond.

## Real World Examples

Practical, relatable examples.

## AI/ML Relevance

How this concept is used in AI and Machine Learning.

## Mathematical Explanation

Formal mathematical treatment with derivations.

## Formula(s)

Key formulas with explanation of each symbol.

## Step-by-Step Worked Example

At least one complete example with intermediate calculations shown.

## Visual Interpretation

Description of what the concept looks like geometrically or graphically.

## Common Mistakes

Frequent errors and how to avoid them.

## Interview Questions

### Beginner Questions

### Intermediate Questions

### Advanced Questions

## Practice Problems

### Easy

### Medium

### Hard

## Solutions

Complete solutions to all practice problems.

## Related Concepts

Other concepts closely connected to this one.

## Next Concepts

Concepts that build on this one.

## Summary

Brief recap of the most important points.

## Key Takeaways

3-5 bullet points with the most essential information.
```

---

## Prerequisite Graph (Concept Dependencies)

```
Module 1 (001-010) ─── Foundations
    │
    ├──► Module 2 (011-020) ─── Vector Algebra
    │
    ├──► Module 3 (021-030) ─── Matrix Algebra
    │
    ├──► Module 5 (044-052) ─── Functions
    │
    └──► Module 7 (065-076) ─── Probability
             │
Module 3 ───► Module 4 (031-043) ─── Linear Algebra
             │
Module 5 ───► Module 6 (053-064) ─── Calculus
             │
Module 7 ───► Module 8 (077-087) ─── Statistics
             │
Module 8 ───► Module 9 (088-092) ─── Information Theory
             │
Module 6+8 ─► Module 10 (093-100) ─── Optimization
```

---

## Progress Tracking

| Module | Concepts | Status |
|--------|----------|--------|
| 1: Foundations | 001–010 | 🟡 Planned |
| 2: Vector Algebra | 011–020 | 🟡 Planned |
| 3: Matrix Algebra | 021–030 | 🟡 Planned |
| 4: Linear Algebra | 031–043 | 🟡 Planned |
| 5: Functions | 044–052 | 🟡 Planned |
| 6: Calculus | 053–064 | 🟡 Planned |
| 7: Probability | 065–076 | 🟡 Planned |
| 8: Statistics | 077–087 | 🟡 Planned |
| 9: Information Theory | 088–092 | 🟡 Planned |
| 10: Optimization | 093–100 | 🟡 Planned |

**Legend:** 🟢 Complete | 🟡 Planned | 🔴 Not Started

---

*Last updated: June 2026*
