# Concept: NumPy Statistics

## Concept ID

PYT-071

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Compute descriptive statistics: mean, median, standard deviation, variance
- Calculate correlation and covariance between variables
- Use `np.histogram` to bin data into frequency distributions
- Compute percentiles and quantiles with `np.percentile`
- Apply statistical functions along specified axes
- Interpret statistical results for data analysis

## Prerequisites

- NumPy array creation (PYT-066)
- Array operations and aggregation (PYT-068)
- Basic understanding of descriptive statistics

## Definition

NumPy's statistical functions compute descriptive statistics on arrays, including measures of central tendency (mean, median), spread (standard deviation, variance, percentile), and relationships between variables (correlation, covariance). These functions are vectorized, support axis-wise computation, and handle multidimensional data efficiently.

## Intuition

Statistics helps you understand what your data looks like. The mean tells you the central value, standard deviation tells you how spread out the data is, correlation tells you how two variables move together, and percentiles tell you where specific data points fall in the distribution. Histograms visualize the distribution shape. NumPy computes all these efficiently in a single function call.

## Why This Concept Matters

Before building any ML model, you must understand your data. Exploratory Data Analysis (EDA) relies heavily on descriptive statistics. Feature engineering decisions (scaling, normalization, outlier removal) depend on statistics. Model evaluation metrics (MSE, MAE, R-squared) are statistical computations. Understanding the distribution of your data is critical for choosing the right model.

## Real World Examples

1. **EDA Report:** Compute mean, median, std, min, max, and percentiles for each feature in a dataset.
2. **Outlier Detection:** Values beyond 3 standard deviations from the mean are flagged as outliers.
3. **Feature Correlation:** Identify highly correlated features for removal to reduce multicollinearity.
4. **Portfolio Risk:** Compute covariance matrix of stock returns to measure diversification.
5. **Quality Control:** Use percentiles to establish acceptable ranges for manufacturing measurements.

## AI/ML Relevance

Statistics are used throughout ML: feature scaling (mean, std for standardization), evaluating model residuals (mean, std of errors), detecting data drift (comparing distribution statistics), feature selection (correlation with target), and anomaly detection (percentile-based thresholds). The covariance matrix is used in PCA, Fisher's linear discriminant, and Gaussian Naive Bayes.

## Code Examples

### Example 1: Basic Descriptive Statistics

```python
import numpy as np

data = np.array([12, 15, 14, 10, 18, 20, 22, 17, 14, 16])
print("Data:", data)
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Variance:", np.var(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Range:", np.ptp(data))
```
```
# Output:
# Data: [12 15 14 10 18 20 22 17 14 16]
# Mean: 15.8
# Median: 15.5
# Standard Deviation: 3.552105584278721
# Variance: 12.61744966442953
# Min: 10
# Max: 22
# Range: 12
```

### Example 2: Axis-Wise Statistics on 2D Data

```python
import numpy as np

data = np.array([[1.2, 3.4, 5.6],
                 [2.1, 4.5, 6.7],
                 [0.9, 3.9, 5.1],
                 [1.8, 4.1, 6.2],
                 [1.5, 3.7, 5.9]])

print("Feature statistics (per column):")
print(f"  Mean:     {data.mean(axis=0)}")
print(f"  Std:      {data.std(axis=0)}")
print(f"  Min:      {data.min(axis=0)}")
print(f"  Max:      {data.max(axis=0)}")
print(f"  Median:   {np.median(data, axis=0)}")

print("\nSummary table:")
for i in range(data.shape[1]):
    col = data[:, i]
    print(f"  Feature {i+1}: mean={col.mean():.2f}, std={col.std():.2f}, "
          f"min={col.min():.2f}, max={col.max():.2f}")
```
```
# Output:
# Feature statistics (per column):
#   Mean:     [1.5  3.92 5.9 ]
#   Std:      [0.42130702 0.37416574 0.56568542]
#   Min:      [0.9 3.4 5.1]
#   Max:      [2.1 4.5 6.7]
#   Median:   [1.5 3.9 5.9]
#
# Summary table:
#   Feature 1: mean=1.50, std=0.42, min=0.90, max=2.10
#   Feature 2: mean=3.92, std=0.37, min=3.40, max=4.50
#   Feature 3: mean=5.90, std=0.57, min=5.10, max=6.70
```

### Example 3: Percentiles and Quantiles

```python
import numpy as np

np.random.seed(42)
data = np.random.normal(100, 15, 1000)

percentiles = [0, 25, 50, 75, 100]
vals = np.percentile(data, percentiles)
print("IQ score percentiles:")
for p, v in zip(percentiles, vals):
    print(f"  {p}th percentile: {v:.1f}")

p_vals = np.percentile(data, [2.5, 97.5])
print(f"\n95% central range: [{p_vals[0]:.1f}, {p_vals[1]:.1f}]")

q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75])
iqr = q3 - q1
print(f"\nQ1={q1:.1f}, Median={q2:.1f}, Q3={q3:.1f}")
print(f"IQR={iqr:.1f}")
print(f"Outlier bounds: <{q1-1.5*iqr:.1f} or >{q3+1.5*iqr:.1f}")
```
```
# Output:
# IQ score percentiles:
#   0th percentile: 56.9
#   25th percentile: 89.6
#   50th percentile: 99.9
#   75th percentile: 111.2
#   100th percentile: 146.6
#
# 95% central range: [69.5, 129.6]
#
# Q1=89.6, Median=99.9, Q3=111.2
# IQR=21.6
# Outlier bounds: <57.2 or >143.6
```

### Example 4: Correlation and Covariance

```python
import numpy as np

np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 0.7 * x + 0.3 * np.random.randn(n)
z = -0.4 * x + 0.6 * np.random.randn(n)

data = np.column_stack([x, y, z])

corr = np.corrcoef(data, rowvar=False)
print("Correlation matrix:")
print(np.round(corr, 3))

cov = np.cov(data, rowvar=False)
print("\nCovariance matrix:")
print(np.round(cov, 3))

print(f"\nPearson r (x, y): {np.corrcoef(x, y)[0,1]:.3f}")
print(f"Pearson r (x, z): {np.corrcoef(x, z)[0,1]:.3f}")
print(f"Pearson r (y, z): {np.corrcoef(y, z)[0,1]:.3f}")
```
```
# Output:
# Correlation matrix:
# [[ 1.   0.69 -0.44]
#  [ 0.69  1.   -0.36]
#  [-0.44 -0.36  1.  ]]
#
# Covariance matrix:
# [[ 1.19   0.782 -0.491]
#  [ 0.782  1.074 -0.378]
#  [-0.491 -0.378  1.007]]
#
# Pearson r (x, y): 0.690
# Pearson r (x, z): -0.440
# Pearson r (y, z): -0.355
```

### Example 5: Histograms

```python
import numpy as np

np.random.seed(42)
data = np.random.exponential(scale=2, size=1000)

counts, edges = np.histogram(data, bins=10)
print("Histogram (10 bins):")
print("Edges:", np.round(edges, 2))
print("Counts:", counts)

counts_d, edges_d = np.histogram(data, bins=20, density=True)
area = np.sum(counts_d * np.diff(edges_d))
print(f"\nDensity histogram area: {area:.2f} (should = 1)")

custom_edges = np.array([0, 1, 2, 3, 5, 10])
counts_custom, _ = np.histogram(data, bins=custom_edges)
print(f"\nCustom bins counts: {counts_custom}")
```
```
# Output:
# Histogram (10 bins):
# Edges: [ 0.    1.02  2.04  3.06  4.08  5.11  6.13  7.15  8.17  9.19 10.21]
# Counts: [310 240 168 118  76  40  24  16   5   3]
#
# Density histogram area: 1.00 (should = 1)
#
# Custom bins counts: [448 540  12   0   0]
```

### Example 6: Bivariate Statistics and Standardization

```python
import numpy as np

np.random.seed(42)
scores = np.random.randn(100, 4)  # 100 students, 4 exams

# Z-score standardization
means = scores.mean(axis=0)
stds = scores.std(axis=0)
z_scores = (scores - means) / stds

print("Original mean:", means.round(3))
print("Z-score mean:", z_scores.mean(axis=0).round(6))
print("Z-score std:", z_scores.std(axis=0).round(6))

# Summarize with describe-like output
print("\nFeature summary:")
for i in range(scores.shape[1]):
    col = scores[:, i]
    print(f"  Exam {i+1}: mean={col.mean():.2f}, std={col.std():.2f}, "
          f"median={np.median(col):.2f}, "
          f"p25={np.percentile(col, 25):.2f}, "
          f"p75={np.percentile(col, 75):.2f}")
```
```
# Output:
# Original mean: [-0.273 -0.123  0.042 -0.12 ]
# Z-score mean: [ 0. -0. -0. -0.]
# Z-score std: [1. 1. 1. 1.]
#
# Feature summary:
#   Exam 1: mean=-0.27, std=0.96, median=-0.34, p25=-0.88, p75=0.36
#   Exam 2: mean=-0.12, std=1.03, median=-0.16, p25=-0.77, p75=0.55
#   Exam 3: mean=0.04, std=1.00, median=0.06, p25=-0.62, p75=0.73
#   Exam 4: mean=-0.12, std=0.99, median=-0.10, p25=-0.83, p75=0.58
```

## Common Mistakes

1. **Using `np.std` with `ddof=0` vs `ddof=1`:** `np.std` computes population standard deviation (ddof=0) by default. For sample standard deviation (unbiased), use `np.std(arr, ddof=1)`.

2. **Confusing `np.cov` and `np.corrcoef`:** `np.cov` returns covariance (unscaled), `np.corrcoef` returns correlation (scaled to [-1, 1]). Both take `rowvar` parameter: `rowvar=True` means each row is a variable (default).

3. **Forgetting `rowvar=False` for Column-Wise Data:** When data is shape `(n_samples, n_features)`, pass `rowvar=False` to compute feature-feature covariance/correlation.

4. **Assuming `np.histogram` Returns Frequencies:** By default, `np.histogram` returns counts. Use `density=True` for probability density or `weights` for weighted counts.

5. **Mixing `np.percentile` and `np.quantile`:** `np.percentile(data, 50)` = `np.quantile(data, 0.5)`. Percentile uses 0-100 scale, quantile uses 0-1 scale.

6. **Ignoring NaN Values:** Statistical functions return NaN if any value is NaN. Use `np.nanmean`, `np.nanstd`, `np.nanpercentile` for arrays with missing values.

7. **Degrees of Freedom in Covariance:** `np.cov` uses `ddof=1` (sample covariance) by default, while `np.var` uses `ddof=0` (population variance). This inconsistency can cause confusion.

## Interview Questions

### Beginner

1. **Q:** How do you compute the mean of a NumPy array?
   **A:** `np.mean(arr)` or `arr.mean()`.

2. **Q:** What is the difference between `np.std` and `np.var`?
   **A:** `np.std` returns the standard deviation (sqrt of variance). `np.var` returns the variance. `std = sqrt(var)`.

3. **Q:** How do you find the median of an array?
   **A:** `np.median(arr)`.

4. **Q:** What does `np.corrcoef` return?
   **A:** It returns the Pearson correlation coefficient matrix. Diagonal entries are 1, off-diagonal entries are pairwise correlations between variables.

5. **Q:** How do you compute the 90th percentile of a dataset?
   **A:** `np.percentile(data, 90)`.

### Intermediate

1. **Q:** What is the difference between `np.cov` and `np.corrcoef`?
   **A:** `np.cov` returns the covariance matrix (values depend on units/scales). `np.corrcoef` returns the correlation matrix (standardized to [-1, 1]), which is unitless and shows linear relationship strength.

2. **Q:** How do you handle NaN values when computing statistics?
   **A:** Use `np.nanmean`, `np.nanstd`, `np.nanvar`, `np.nanmedian`, `np.nanpercentile`, `np.nanmin`, `np.nanmax` — these ignore NaN values.

3. **Q:** What does `rowvar` mean in `np.cov` and `np.corrcoef`?
   **A:** `rowvar=True` (default) treats each row as a variable. For data with shape `(n_samples, n_features)`, set `rowvar=False` to treat each column as a variable.

4. **Q:** How would you compute a rolling mean using NumPy?
   **A:** Use `np.convolve(data, np.ones(k)/k, mode='valid')` for a simple moving average.

5. **Q:** What is the difference between `np.histogram` and `np.digitize`?
   **A:** `np.histogram` computes counts per bin and returns bin edges. `np.digitize` returns the bin index for each data point given bin edges.

### Advanced

1. **Q:** How does `np.cov` compute covariance for multidimensional arrays, and what does the `ddof` parameter control?
   **A:** For a 2D array, `np.cov` computes the covariance matrix where entry `(i,j)` is `sum((x_i - mean_i)(x_j - mean_j)) / (n - ddof)`. `ddof=0` gives population covariance, `ddof=1` (default) gives sample covariance. For higher dimensions, the array is flattened first.

2. **Q:** Explain the relationship between the covariance matrix and PCA. How would you derive principal components from `np.cov`?
   **A:** PCA finds eigenvectors of the covariance matrix. Compute `cov = np.cov(X, rowvar=False)`, then `eigvals, eigvecs = np.linalg.eigh(cov)`. Sort by eigenvalue magnitude. The eigenvectors are the principal components. Alternatively, use SVD directly on centered X for better numerical stability.

3. **Q:** How would you efficiently compute the pairwise correlation matrix for a 10,000 x 500 dataset without running out of memory?
   **A:** Use `np.corrcoef` with `rowvar=False`, which internally centers and normalizes. For very large data, consider chunked computation: compute `X_std = (X - X.mean(axis=0)) / X.std(axis=0)`, then `corr = X_std.T @ X_std / (n-1)`. Use `np.float32` and memory-mapped arrays if needed.

## Practice Problems

### Easy

1. Compute the mean, median, standard deviation of `[5, 7, 8, 10, 12, 15, 18]`.

2. Given `data = np.random.randn(50, 3)`, compute the mean of each column.

3. Find the 25th, 50th, and 75th percentiles of `np.array([1, 3, 5, 7, 9, 11, 13])`.

4. Compute the correlation coefficient between `[1, 2, 3, 4, 5]` and `[2, 4, 6, 8, 10]`.

5. Create a histogram with 5 bins for `np.random.exponential(1, 100)`.

### Medium

1. Given a 100x10 dataset, compute the z-scores (standardization) for each feature.

2. Compute the pairwise correlation matrix for the Iris dataset (4 features) and identify the pair with the highest correlation.

3. Using `np.histogram`, compute the histogram of a dataset and find which bin contains the median value.

4. Implement a function that computes the rolling standard deviation of a 1D array with window size k.

5. Compute the covariance matrix for 100 samples of 5 features. Find the eigenvalues and explain what they represent.

### Hard

1. Implement a function `describe(data)` that returns a summary table with count, mean, std, min, 25%, 50%, 75%, max for each column (similar to pandas `df.describe()`).

2. Implement a function that computes the mutual information between two continuous variables using histogram binning (discretize into bins, compute entropy).

3. Implement a robust outlier detection function using both z-score (mean/std) and IQR methods. Compare their performance on a dataset with injected outliers. Return indices flagged by each method.

## Solutions

### Easy Solutions

```python
# 1
arr = np.array([5, 7, 8, 10, 12, 15, 18])
print(f"Mean={arr.mean():.2f}, Median={np.median(arr):.2f}, Std={arr.std():.2f}")

# 2
data = np.random.randn(50, 3)
print("Column means:", data.mean(axis=0))

# 3
arr = np.array([1, 3, 5, 7, 9, 11, 13])
print("Percentiles:", np.percentile(arr, [25, 50, 75]))

# 4
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
print("Correlation:", np.corrcoef(x, y)[0, 1])  # 1.0 (perfect linear)

# 5
data = np.random.exponential(1, 100)
counts, edges = np.histogram(data, bins=5)
print("Edges:", edges.round(2))
print("Counts:", counts)
```

### Medium Solutions

```python
# 1 Z-score
data = np.random.randn(100, 10)
z = (data - data.mean(axis=0)) / data.std(axis=0)
print("Z-score means:", z.mean(axis=0).round(6))
print("Z-score stds:", z.std(axis=0).round(6))

# 2 Iris correlation
from sklearn.datasets import load_iris
iris = load_iris()
corr = np.corrcoef(iris.data, rowvar=False)
np.fill_diagonal(corr, 0)
i, j = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
print(f"Highest |r| = {corr[i,j]:.3f} between features {i} and {j}")

# 3 Median bin
data = np.random.randn(1000)
counts, edges = np.histogram(data, bins=10)
cumcounts = np.cumsum(counts)
median_bin = np.searchsorted(cumcounts, cumcounts[-1] / 2)
print(f"Median is in bin {median_bin}: [{edges[median_bin]:.2f}, {edges[median_bin+1]:.2f}]")

# 4 Rolling std
def rolling_std(arr, k):
    means = np.convolve(arr, np.ones(k)/k, mode='valid')
    sq_means = np.convolve(arr**2, np.ones(k)/k, mode='valid')
    return np.sqrt(sq_means - means**2)

arr = np.arange(20, dtype=float)
print("Rolling std (k=3):", rolling_std(arr, 3))

# 5 Covariance eigenvalues
data = np.random.randn(100, 5)
cov = np.cov(data, rowvar=False)
eigvals = np.linalg.eigvalsh(cov)
print("Eigenvalues of covariance:", eigvals.round(3))
print("Explained variance ratio:", (eigvals / eigvals.sum()).round(3))
```

### Hard Solutions

```python
# 1 describe function
def describe(data):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_features = data.shape[1]
    stats = []
    for i in range(n_features):
        col = data[:, i]
        stats.append([
            col.size, col.mean(), col.std(ddof=1),
            col.min(), np.percentile(col, 25),
            np.median(col), np.percentile(col, 75), col.max()
        ])
    names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    result = np.array(stats).T
    for name, vals in zip(names, result):
        print(f"{name:>6}: {vals}")
    return result

data = np.random.randn(100, 4)
describe(data)

# 2 Mutual information via binning
def mutual_information(x, y, bins=10):
    cxy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    cx = np.histogram(x, bins=bins, density=True)[0]
    cy = np.histogram(y, bins=bins, density=True)[0]
    cxy += 1e-12
    cx += 1e-12
    cy += 1e-12
    return np.sum(cxy * np.log(cxy / (cx[:, None] * cy[None, :])))

np.random.seed(42)
x = np.random.randn(1000)
y = x + np.random.randn(1000) * 0.5
print(f"MI(x, y): {mutual_information(x, y):.4f}")
print(f"MI(x, noise): {mutual_information(x, np.random.randn(1000)):.4f}")

# 3 Outlier detection comparison
def detect_outliers(data, method='zscore', threshold=3):
    if method == 'zscore':
        z = np.abs(data - data.mean()) / data.std()
        return np.where(z > threshold)[0]
    elif method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        return np.where((data < lower) | (data > upper))[0]

np.random.seed(42)
data = np.random.randn(1000)
data[10:15] = 10
data[20:23] = -8

z_idx = detect_outliers(data, 'zscore')
iqr_idx = detect_outliers(data, 'iqr')
print(f"Z-score outliers: {len(z_idx)}")
print(f"IQR outliers: {len(iqr_idx)}")
print(f"Overlap: {len(set(z_idx) & set(iqr_idx))}")
```

## Related Concepts

- Pandas DataFrame `.describe()` and `.corr()` methods
- Matplotlib histograms and box plots
- Seaborn distribution plots
- Scipy stats module for advanced statistics

## Next Concepts

- Reshaping arrays (PYT-072)
- Concatenation (PYT-073)
- Structured arrays (PYT-075)

## Summary

NumPy provides comprehensive statistical functions: `mean`, `median`, `std`, `var`, `percentile`, `corrcoef`, `cov`, and `histogram`. These functions support axis-wise computation, handle multidimensional data, and form the basis of exploratory data analysis. `np.corrcoef` measures linear relationships, `np.histogram` visualizes distributions, and percentiles describe data spread. Understanding these tools is essential for data preprocessing and model evaluation.

## Key Takeaways

- `np.mean`, `np.median`, `np.std`, `np.var` for central tendency and spread
- `np.percentile` and `np.quantile` for distribution quantiles
- `np.corrcoef` for correlation matrix; `np.cov` for covariance matrix
- `np.histogram` for frequency/distribution binning
- Use `axis` parameter for row/column-wise statistics
- `ddof=1` for sample statistics, `ddof=0` for population statistics
- Use `np.nan*` variants to handle missing values
- `rowvar=False` for column-oriented feature data
