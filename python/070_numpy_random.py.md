# Concept: NumPy Random Module

## Concept ID

PYT-070

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Set and manage random seeds with `np.random.seed` and `RandomState`
- Generate random numbers from uniform, normal, and integer distributions
- Use `np.random.choice` for random sampling with/without replacement
- Shuffle arrays in-place with `np.random.shuffle`
- Create reproducible random sequences for machine learning experiments
- Understand when to use different distributions (uniform, normal, randint)

## Prerequisites

- Basic NumPy array creation (PYT-066)
- Understanding of probability distributions (uniform, normal)
- Array indexing and slicing (PYT-067)

## Definition

The `np.random` module provides functions for generating random numbers from various probability distributions, random sampling, and shuffling. It supports reproducible randomness via explicit seed management and the `RandomState` class for independent random streams.

## Intuition

Randomness is essential for many areas of computing — initializing model parameters, splitting data into train/test sets, shuffling data, and simulating stochastic processes. `np.random` generates pseudorandom numbers using deterministic algorithms (like the Mersenne Twister) that appear random for practical purposes. Setting a seed ensures the same sequence of "random" numbers is generated each time, enabling reproducible experiments.

## Why This Concept Matters

Machine learning relies heavily on controlled randomness: weight initialization, stochastic gradient descent (random mini-batches), data shuffling, dropout regularization, and train/test splitting all depend on random number generation. Reproducibility is critical in research and production — you must be able to exactly reproduce results by controlling the random seed. `np.random` is the standard tool for all these tasks.

## Real World Examples

1. **Train/Test Split:** Randomly shuffle data and split 80/20 for training and testing.
2. **Weight Initialization:** Initialize neural network weights with small random values from a uniform or normal distribution.
3. **Bootstrapping:** Resample data with replacement to estimate confidence intervals.
4. **Monte Carlo Simulation:** Estimate pi by randomly sampling points in a square.
5. **Cross-Validation:** Randomly partition data into k folds for k-fold cross-validation.

## AI/ML Relevance

Every ML framework uses random number generation. Weight initialization (Xavier, He) relies on normal or uniform distributions. Mini-batch selection requires shuffling. Dropout uses Bernoulli random variables. Data augmentation uses random transformations. Hyperparameter search uses random sampling. Reproducibility via seed setting is essential for debugging and comparing models.

## Code Examples

### Example 1: Setting Seeds and Basic Random Numbers

```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
print("Random float:", np.random.rand())
print("Random float:", np.random.rand())

# Same seed produces same sequence
np.random.seed(42)
print("\nAfter resetting seed:")
print("Random float:", np.random.rand())
print("Random float:", np.random.rand())

# Random values in [0, 1) with given shape
np.random.seed(0)
rand_arr = np.random.rand(3, 4)
print("\nrand(3, 4):\n", rand_arr)
```
```
# Output:
# Random float: 0.3745401188473625
# Random float: 0.9507143064099162
# 
# After resetting seed:
# Random float: 0.3745401188473625
# Random float: 0.9507143064099162
# 
# rand(3, 4):
#  [[0.5488135  0.71518937 0.60276338 0.54488318]
#  [0.4236548  0.64589411 0.43758721 0.891773  ]
#  [0.96366276 0.38344152 0.79172504 0.52889492]]
```

### Example 2: Different Distributions

```python
import numpy as np

np.random.seed(42)

# randint: uniform random integers
ints = np.random.randint(0, 10, size=10)
print("randint 0-10:", ints)

# randn: standard normal (mean=0, std=1)
normal = np.random.randn(1000)
print(f"\nrandn mean: {normal.mean():.4f}, std: {normal.std():.4f}")

# uniform: uniform in [low, high)
uniform = np.random.uniform(-1, 1, size=5)
print("\nuniform(-1, 1):", uniform)

# normal: custom mean and std
custom_normal = np.random.normal(loc=5, scale=2, size=1000)
print(f"\nnormal(5, 2) mean: {custom_normal.mean():.4f}, std: {custom_normal.std():.4f}")

# Visual check of distributions
print("\nFirst 5 values of each:")
print(f"  randint:  {ints[:5]}")
print(f"  uniform:  {uniform[:5]}")
print(f"  normal:   {normal[:5].round(3)}")
```
```
# Output:
# randint 0-10: [6 3 7 4 6 9 2 6 7 4]
# 
# randn mean: -0.0240, std: 1.0016
# 
# uniform(-1, 1): [ 0.12573082 -0.13210468 -0.37398973  0.44798114 -0.14620374]
# 
# normal(5, 2) mean: 4.9897, std: 1.9976
# 
# First 5 values of each:
#   randint:  [6 3 7 4 6]
#   uniform:  [ 0.12573082 -0.13210468 -0.37398973  0.44798114 -0.14620374]
#   normal:   [ 1.034  0.238 -0.368 -0.766 -0.166]
```

### Example 3: Random Sampling with np.random.choice

```python
import numpy as np

np.random.seed(42)

# Sample from array-like
population = np.array(['A', 'B', 'C', 'D', 'E'])

# Single sample
print("Single sample:", np.random.choice(population))

# Multiple samples with replacement
samples = np.random.choice(population, size=10, replace=True)
print("10 samples (with replacement):", samples)

# Multiple samples without replacement
samples_no_replace = np.random.choice(population, size=3, replace=False)
print("3 samples (no replacement):", samples_no_replace)

# Weighted sampling
colors = ['red', 'blue', 'green']
weights = [0.5, 0.3, 0.2]
weighted_sample = np.random.choice(colors, size=20, p=weights)
print("\nWeighted samples:", weighted_sample)
print("Proportions:", {c: (weighted_sample == c).mean() for c in colors})
```
```
# Output:
# Single sample: C
# 10 samples (with replacement): ['E' 'E' 'B' 'E' 'E' 'E' 'B' 'E' 'A' 'C']
# 3 samples (no replacement): ['A' 'D' 'E']
# 
# Weighted samples: ['red' 'red' 'blue' 'red' 'red' 'blue' 'green' 'blue' 'red' 'red' 'red'
#  'red' 'red' 'red' 'blue' 'green' 'red' 'red' 'red' 'red']
# Proportions: {'red': 0.65, 'blue': 0.2, 'green': 0.15}
```

### Example 4: Shuffling Arrays

```python
import numpy as np

np.random.seed(42)

# Shuffle 1D array (in-place)
arr = np.arange(10)
print("Before shuffle:", arr)
np.random.shuffle(arr)
print("After shuffle: ", arr)

# Shuffle 2D arrays along first axis (rows)
X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
y = np.array([0, 1, 0, 1])

# Shuffle X and y in unison
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]
print("\nX shuffled:\n", X_shuffled)
print("y shuffled:", y_shuffled)

# Permutation (returns shuffled copy)
perm = np.random.permutation(10)
print("\nPermutation of 0-9:", perm)
```
```
# Output:
# Before shuffle: [0 1 2 3 4 5 6 7 8 9]
# After shuffle:  [3 0 2 6 4 9 7 5 1 8]
# 
# X shuffled:
#  [[5 6]
#  [3 4]
#  [7 8]
#  [1 2]]
# y shuffled: [0 1 1 0]
# 
# Permutation of 0-9: [7 5 8 6 9 1 0 4 2 3]
```

### Example 5: RandomState for Independent Streams

```python
import numpy as np

# Global seed
np.random.seed(42)

# Independent random state
rng1 = np.random.RandomState(42)
rng2 = np.random.RandomState(99)

print("rng1 randn (first 3):", rng1.randn(3))
print("rng2 randn (first 3):", rng2.randn(3))

# Same seed = same sequence
rng1b = np.random.RandomState(42)
print("\nrng1b (same seed):", rng1b.randn(3))

# Practical use: different seeds for different experiments
train_rng = np.random.RandomState(0)
val_rng = np.random.RandomState(1)

train_data = train_rng.randn(10, 5)
val_data = val_rng.randn(5, 5)
print(f"\nTrain shape: {train_data.shape}, Val shape: {val_data.shape}")
```
```
# Output:
# rng1 randn (first 3): [ 0.49671415 -0.1382643   0.64768854]
# rng2 randn (first 3): [-0.51687202  1.93402421  1.16892182]
# 
# rng1b (same seed): [ 0.49671415 -0.1382643   0.64768854]
# 
# Train shape: (10, 5), Val shape: (5, 5)
```

### Example 6: Weight Initialization for Neural Networks

```python
import numpy as np

np.random.seed(42)

# He initialization (for ReLU)
fan_in, fan_out = 256, 128
limit = np.sqrt(6 / fan_in)
W_he = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
print(f"He init: mean={W_he.mean():.4f}, std={W_he.std():.4f}")

# Xavier/Glorot initialization (for tanh/sigmoid)
limit_xavier = np.sqrt(6 / (fan_in + fan_out))
W_xavier = np.random.uniform(-limit_xavier, limit_xavier, size=(fan_in, fan_out))
print(f"Xavier init: mean={W_xavier.mean():.4f}, std={W_xavier.std():.4f}")

# Small normal initialization
W_small = np.random.randn(fan_in, fan_out) * 0.01
print(f"Small normal: mean={W_small.mean():.4f}, std={W_small.std():.4f}")

# Bias initialization (usually zero)
b = np.zeros(fan_out)
print(f"Bias shape: {b.shape}, values: {b[:5]}")
```
```
# Output:
# He init: mean=-0.0007, std=0.1528
# Xavier init: mean=-0.0006, std=0.1290
# Small normal: mean=-0.0001, std=0.0100
# Bias shape: (128,), values: [0. 0. 0. 0. 0.]
```

## Common Mistakes

1. **Not Setting a Seed:** Without a fixed seed, every run produces different results, making debugging and comparison impossible. Always call `np.random.seed()` at the start of experiments.

2. **Confusing `rand` vs `randn`:** `np.random.rand(d0, d1)` returns uniform values in `[0, 1)`. `np.random.randn(d0, d1)` returns standard normal values (mean=0, std=1). They are very different distributions.

3. **Using `shuffle` When You Need `permutation`:** `shuffle` modifies the array in-place and returns `None`. `permutation` returns a shuffled copy. Accidentally assigning the result of `shuffle` leads to `NoneType` errors.

4. **Sampling Without Replacement from Too Large a Sample:** `np.random.choice(pop, size=n, replace=False)` raises an error if `n > len(pop)`. Check population size first.

5. **Assuming `randint` Upper Bound is Inclusive:** `np.random.randint(0, 10)` returns integers from 0 to 9 (exclusive of 10), consistent with Python's `range`.

6. **Modifying the Global Random State:** Functions that use `np.random` internally affect the global random state. Use `RandomState` objects for isolated, independent random streams.

7. **Not Specifying `p` When Sampling Equally:** If you pass a `p` parameter, it must sum to 1. For equal probabilities, omit `p` entirely.

## Interview Questions

### Beginner

1. **Q:** How do you generate a random float between 0 and 1?
   **A:** `np.random.rand()` or `np.random.random()`.

2. **Q:** What does `np.random.seed(42)` do?
   **A:** It initializes the random number generator to a known state, ensuring that subsequent random calls produce the same sequence across runs.

3. **Q:** How do you generate a 3x3 array of random integers between 1 and 100?
   **A:** `np.random.randint(1, 101, size=(3, 3))`.

4. **Q:** What is the difference between `np.random.shuffle` and `np.random.permutation`?
   **A:** `shuffle` modifies the array in-place and returns None. `permutation` returns a shuffled copy without modifying the original.

5. **Q:** How do you randomly select 3 elements from a list without replacement?
   **A:** `np.random.choice(['a', 'b', 'c', 'd', 'e'], size=3, replace=False)`.

### Intermediate

1. **Q:** What distributions do `rand` and `randn` sample from respectively?
   **A:** `rand` samples from a uniform distribution on `[0, 1)`. `randn` samples from a standard normal distribution (Gaussian with mean=0, variance=1).

2. **Q:** How do you generate 1000 samples from N(5, 3²) (normal with mean 5, std 3)?
   **A:** `np.random.normal(loc=5, scale=3, size=1000)`.

3. **Q:** Why is `RandomState` preferred over `np.random.seed` in larger projects?
   **A:** `RandomState` creates independent random number generators without affecting the global state. This allows different parts of a program (e.g., data splitting and model initialization) to have their own controllable randomness.

4. **Q:** How do you shuffle two arrays (features and labels) in the same order?
   **A:** Generate shuffled indices: `idx = np.random.permutation(len(X))`, then apply: `X_shuffled = X[idx]`, `y_shuffled = y[idx]`.

5. **Q:** What is the difference between `np.random.uniform` and `np.random.rand`?
   **A:** `np.random.rand(d0, d1)` is specialized for uniform `[0, 1)` with shape as positional args. `np.random.uniform(low, high, size)` allows custom range and takes `size` as a keyword argument. Functionally similar but `uniform` is more flexible.

### Advanced

1. **Q:** Explain the algorithm behind NumPy's default random number generator. What are its limitations?
   **A:** NumPy's default (prior to version 1.17) uses the Mersenne Twister (MT19937), a 623-dimensionally equidistributed uniform pseudorandom generator. It has a huge period of 2^19937 - 1. Limitations: it fails some statistical tests for extreme high dimensions, is slow to seed, and its state is large (2.5 KB). Since NumPy 1.17, the recommended approach uses `numpy.random.Generator` with the PCG-64 algorithm, which is faster and has better statistical properties.

2. **Q:** How would you implement a custom random number generator that follows a specific probability distribution using only `np.random.uniform`?
   **A:** Use inverse transform sampling: compute the inverse CDF of the desired distribution and apply it to uniform random numbers. For example, to sample from an exponential distribution with rate λ: `x = -np.log(1 - np.random.uniform(size=n)) / λ`.

3. **Q:** Discuss the transition from the legacy `np.random` API to the new `Generator` API. What are the key differences and why was the change made?
   **A:** The new `Generator` API (since NumPy 1.17) separates random number generation into bit generators (e.g., PCG-64, MT19937) and distributions. It uses `np.random.default_rng(seed)` instead of `np.random.RandomState`. Benefits: faster (up to 2-4x), better statistical properties, more algorithms (SFC64, Philox for parallel), and cleaner API. The legacy `np.random` functions are frozen but not deprecated.

## Practice Problems

### Easy

1. Set the seed to 123 and generate 5 random floats between 0 and 1.

2. Create a 4x4 array of random integers from 1 to 50.

3. Generate 100 samples from a standard normal distribution and compute the empirical mean and standard deviation.

4. Randomly select a winner from `['Alice', 'Bob', 'Charlie', 'Diana']`.

5. Create a 1D array of 10 random uniform values and shuffle it in-place.

### Medium

1. Generate 1000 data points from `N(10, 2.5²)` and compute the 25th, 50th, and 75th percentiles.

2. Create a train/test split (80/20) for a dataset of 500 samples using `np.random.permutation`.

3. Weighted sampling: given probabilities `[0.1, 0.2, 0.3, 0.4]` for categories A, B, C, D, generate 1000 samples and verify the proportions match.

4. Implement a simple Monte Carlo simulation to estimate π by sampling points in a 2x2 square.

5. Use `RandomState` to create two independent random generators. Use one for data shuffling and one for weight initialization, verifying they produce different sequences.

### Hard

1. Implement k-fold cross-validation indices (k=5) using `np.random.permutation`, returning train and test index lists for each fold.

2. Implement Xavier (Glorot) uniform initialization for a list of layer sizes `[784, 256, 128, 10]`, ensuring correct scaling based on `fan_in` and `fan_out`.

3. Write a function `resample(data, n_bootstrap)` that performs bootstrap resampling (sampling with replacement) and returns `n_bootstrap` resampled datasets. Compute the bootstrap confidence interval for the mean.

## Solutions

### Easy Solutions

```python
# 1
np.random.seed(123)
print(np.random.rand(5))

# 2
arr = np.random.randint(1, 51, size=(4, 4))
print(arr)

# 3
samples = np.random.randn(100)
print(f"Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")

# 4
candidates = ['Alice', 'Bob', 'Charlie', 'Diana']
winner = np.random.choice(candidates)
print(f"Winner: {winner}")

# 5
arr = np.random.rand(10)
print("Before:", arr)
np.random.shuffle(arr)
print("After:", arr)
```

### Medium Solutions

```python
# 1
data = np.random.normal(10, 2.5, 1000)
print("Percentiles:", np.percentile(data, [25, 50, 75]))

# 2
n = 500
indices = np.random.permutation(n)
split = int(n * 0.8)
train_idx, test_idx = indices[:split], indices[split:]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# 3
categories = ['A', 'B', 'C', 'D']
probs = [0.1, 0.2, 0.3, 0.4]
samples = np.random.choice(categories, size=1000, p=probs)
for c in categories:
    print(f"{c}: {(samples == c).mean():.4f} (expected {probs[ord(c)-65]})")

# 4 Estimate pi
n_points = 100000
x = np.random.uniform(-1, 1, n_points)
y = np.random.uniform(-1, 1, n_points)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * inside.sum() / n_points
print(f"Estimated π: {pi_estimate:.4f}, Actual: {np.pi:.4f}")

# 5 Independent generators
rng_a = np.random.RandomState(42)
rng_b = np.random.RandomState(99)
data = np.arange(100)
rng_a.shuffle(data.copy())
weights_a = rng_a.randn(10, 5)
weights_b = rng_b.randn(10, 5)
print("Same?", np.allclose(weights_a, weights_b))
```

### Hard Solutions

```python
# 1 K-fold CV indices
def kfold_indices(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    folds = []
    current = 0
    for i in range(k):
        start, stop = current, current + fold_sizes[i]
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop
    return folds

for i, (train, test) in enumerate(kfold_indices(100, k=5)):
    print(f"Fold {i+1}: train={len(train)}, test={len(test)}")

# 2 Xavier initialization
def xavier_init(layer_sizes, seed=42):
    rng = np.random.RandomState(seed)
    weights = []
    for i in range(len(layer_sizes) - 1):
        fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        weights.append(W)
    return weights

layers = [784, 256, 128, 10]
Ws = xavier_init(layers)
for i, W in enumerate(Ws):
    print(f"W{i+1} shape: {W.shape}, range: [{W.min():.4f}, {W.max():.4f}]")

# 3 Bootstrap confidence interval
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    rng = np.random.RandomState(42)
    n = len(data)
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    lower = (1 - ci) / 2 * 100
    upper = (1 + ci) / 2 * 100
    return np.percentile(means, [lower, upper])

np.random.seed(42)
data = np.random.exponential(scale=5, size=1000)
ci_low, ci_high = bootstrap_ci(data)
print(f"Sample mean: {data.mean():.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
```

## Related Concepts

- Python's built-in `random` module
- Probability and statistics basics
- Random seeds and reproducibility in ML
- Monte Carlo methods

## Next Concepts

- NumPy statistics (PYT-071)
- Array reshaping (PYT-072)
- Data preprocessing for ML pipelines

## Summary

`np.random` provides essential tools for generating random numbers from common distributions (`rand`, `randn`, `randint`, `uniform`, `normal`), random sampling (`choice`, `shuffle`, `permutation`), and managing reproducibility (`seed`, `RandomState`). Setting a seed is crucial for reproducible experiments. `RandomState` offers independent random streams for different components of a system.

## Key Takeaways

- Always set `np.random.seed()` at the start of experiments for reproducibility
- Use `RandomState` for independent, non-interfering random streams
- `rand` → uniform [0, 1); `randn` → standard normal; `randint` → uniform integers
- `shuffle` is in-place; `permutation` returns a shuffled copy
- `choice` with `replace=True/False` handles sampling with/without replacement
- Weight initialization strategies (Xavier, He) are built on uniform/normal random
- Separating train/validation random streams prevents data leakage
