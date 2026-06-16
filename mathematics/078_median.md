# Concept: Median

## Concept ID

MATH-078

## Difficulty

BEGINNER

## Domain

Mathematics

## Module

Statistics

## Learning Objectives

- Define and compute the median for a given dataset
- Distinguish between odd and even $n$ cases
- Explain why the median is robust to outliers
- Understand the relationship between median and quartiles
- Apply the median in robust statistical methods for AI/ML

## Prerequisites

- Sorting numbers in ascending order
- Basic arithmetic
- Understanding of the mean (MATH-077)

## Definition

The **median** is the middle value of a dataset when the values are arranged in ascending order. It divides the dataset into two equal halves — 50% of the values lie below the median and 50% lie above it.

For a sorted dataset $x_{(1)} \leq x_{(2)} \leq \dots \leq x_{(n)}$:

- If $n$ is odd: $\text{Median} = x_{((n+1)/2)}$
- If $n$ is even: $\text{Median} = \frac{x_{(n/2)} + x_{(n/2 + 1)}}{2}$

The median is also known as the 50th percentile or the second quartile ($Q_2$).

## Intuition

Imagine a group of people standing on a number line according to their heights. The median is the height of the person in the exact middle when everyone lines up from shortest to tallest. If a very tall person joins the group, the median might not change at all because the middle person could remain the same.

The median answers the question: "What is the typical value that splits the group into two equal halves?" Unlike the mean, which is pulled by extreme values, the median stays anchored to the middle of the ordered data.

## Why This Concept Matters

The median is the preferred measure of central tendency for skewed distributions and datasets with outliers. When reporting typical income, home prices, or any data with extreme values, the median gives a more honest picture than the mean.

For example, in a neighbourhood where nine families earn $50,000 and one family earns $50,000,000, the mean income is approximately $5,045,000 — not representative of anyone's experience. The median income is $50,000, accurately reflecting the typical family.

## Historical Background

The concept of the median dates back to ancient times. The term "median" was introduced by Francis Galton in the late 19th century. Galton, a pioneer of statistics, used the median in his work on heredity and correlation. The median gained prominence in the 20th century through the work of George Box and John Tukey, who championed robust statistics. Tukey's work on exploratory data analysis (EDA) heavily featured the median in tools like box plots and median polish.

The median has ancient roots — Greek mathematicians understood the concept of the "middle" value, and the median was used implicitly in early census and tax records to determine typical wealth.

## Real World Examples

**Real estate:** Median home prices are reported instead of mean prices because a few mansions would inflate the mean, misrepresenting the market.

**Income:** Government agencies report median household income to describe the financial status of the typical family.

**Education:** The median test score tells educators how the average student performed without being skewed by extreme high or low scores.

**Healthcare:** Median survival time in clinical trials is a standard endpoint, preferred over mean survival when some patients live much longer than others.

**Sports:** Median player salary in a sports league better represents what a typical player earns than the mean, which is pulled upward by superstar contracts.

## AI/ML Relevance

**Robust feature scaling:** RobustScaler in scikit-learn uses the median and interquartile range (IQR) to scale features. Unlike StandardScaler (which uses mean and variance), RobustScaler is not affected by outliers:
$$
x_{\text{scaled}} = \frac{x - \text{Median}(x)}{\text{IQR}(x)}
$$

**Median absolute deviation (MAD):** A robust measure of spread:
$$
\text{MAD} = \text{Median}(|x_i - \text{Median}(x)|)
$$
MAD is used in outlier detection, anomaly detection, and as a robust alternative to standard deviation.

**Decision trees:** Some decision tree implementations use the median as a splitting threshold for numerical features, as it guarantees balanced splits.

**Ensemble methods:** Median averaging is used in random forests for regression. Combining predictions via the median is more robust than the mean when individual trees produce extreme outliers.

**Missing value imputation:** For features with outliers, the median is preferred over the mean for imputing missing values.

**Exploratory data analysis (EDA):** Box plots display the median (as the centre line), providing a visual summary of central tendency and spread.

## Mathematical Explanation

The median is the solution to a minimisation problem involving absolute deviations:

$$
\text{Median} = \arg\min_{c} \sum_{i=1}^n |x_i - c|
$$

This is in contrast to the mean, which minimises squared deviations. The absolute value makes the median robust — large deviations are penalised linearly rather than quadratically.

**Quartiles:** The median divides data into two halves. Each half can be further divided:

- $Q_1$ (first quartile): median of the lower half
- $Q_2$ (second quartile): the median
- $Q_3$ (third quartile): median of the upper half

The interquartile range is $\text{IQR} = Q_3 - Q_1$, a robust measure of spread.

**Properties for odd vs even $n$:**

- Odd $n$ ($n = 2k+1$): Median $= x_{(k+1)}$. Exactly one middle value.
- Even $n$ ($n = 2k$): Median $= (x_{(k)} + x_{(k+1)})/2$. The average of the two middle values.

## Formula(s)

**Median for odd $n$:**
$$
\text{Median} = x_{((n+1)/2)}
$$

**Median for even $n$:**
$$
\text{Median} = \frac{x_{(n/2)} + x_{(n/2+1)}}{2}
$$

**Median from grouped data:**
$$
\text{Median} = L + \frac{(n/2 - F)}{f} \times h
$$
where $L$ is the lower boundary of the median class, $F$ is the cumulative frequency before the median class, $f$ is the frequency of the median class, and $h$ is the class width.

## Properties

- **Robustness:** The median has a breakdown point of 50% — nearly half the data can be arbitrarily corrupted without making the median unbounded.
- **Uniqueness:** The median is unique for odd $n$ and can be any value between the two middle values for even $n$ (conventionally their average).
- **Optimality:** The median minimises the sum of absolute deviations.
- **Scale equivariance:** $\text{Median}(aX + b) = a\,\text{Median}(X) + b$.
- **Order statistic:** The median is a specific order statistic — the $\lceil n/2 \rceil$-th order statistic.
- **Non-parametric:** The median is distribution-free — it does not assume any underlying distribution.
- **Comparison with mean:** For symmetric distributions, median $=$ mean. For positively skewed distributions, median $<$ mean. For negatively skewed distributions, median $>$ mean.

## Step-by-Step Worked Examples

### Example 1: Odd Number of Values

**Problem:** Find the median of $\{12, 7, 3, 15, 9\}$.

**Solution:**

Step 1: Sort the data in ascending order.
$$
3, 7, 9, 12, 15
$$

Step 2: Count the values.
$n = 5$ (odd)

Step 3: Find the middle position.
Position $= (5+1)/2 = 3$

Step 4: The third value in the sorted list is 9.

The median is 9.

### Example 2: Even Number of Values

**Problem:** Find the median of $\{18, 25, 12, 30, 22, 15\}$.

**Solution:**

Step 1: Sort ascending.
$$
12, 15, 18, 22, 25, 30
$$

Step 2: Count values.
$n = 6$ (even)

Step 3: Identify the two middle positions.
$n/2 = 3$, $n/2 + 1 = 4$
The 3rd value is 18. The 4th value is 22.

Step 4: Average the two middle values.
$$
\text{Median} = \frac{18 + 22}{2} = 20
$$

The median is 20.

### Example 3: Outlier Resistance

**Problem:** Compare the mean and median of $\{4, 5, 6, 7, 100\}$.

**Solution:**

Sorted: $4, 5, 6, 7, 100$

Mean:
$$
\bar{x} = \frac{4 + 5 + 6 + 7 + 100}{5} = \frac{122}{5} = 24.4
$$

Median:
$n = 5$, middle position $= 3$, Median $= 6$

The mean (24.4) is pulled far right by the outlier 100, while the median (6) remains at the centre of the majority of the data. The median better represents the typical value.

## Visual Interpretation

On a box plot, the median is the horizontal line inside the box. It shows where the centre of the data lies.

In a histogram, the median is the point where the total area to the left equals the total area to the right (50% each). For symmetric distributions this coincides with the peak; for skewed distributions the median lies between the mode and the mean.

For skewed-right distributions: Mode $<$ Median $<$ Mean. For skewed-left: Mode $>$ Median $>$ Mean. This relationship is known as the Pearson mode skewness rule.

## Common Mistakes

1. **Forgetting to sort:** Computing the median from unsorted data gives a meaningless result.

2. **Using the wrong formula for even $n$:** Some students pick just the lower middle value or the upper middle value instead of averaging them.

3. **Confusing median with mean:** The median is not the same as the average. For skewed data, the difference can be large.

4. **Applying median to categorical data:** The median requires ordered data. For nominal (unordered) categorical data, the median is undefined.

5. **Assuming median equals the middle of the range:** The median is not the midpoint between min and max. The median depends on the distribution, not the extremes.

6. **Rounding incorrectly:** When $n$ is odd, the median is exactly the middle value. When $n$ is even, the median is the average of two values and may be a non-integer.

7. **Misinterpreting the median as the average:** The median does not use all values in its computation, making it less efficient than the mean for normally distributed data.

## Interview Questions

### Beginner - 5

1. **Q:** What is the median of $\{3, 7, 2, 9, 5\}$?
   **A:** Sorted: $2, 3, 5, 7, 9$. $n=5$ odd, middle position $=3$, median $=5$.

2. **Q:** How is the median different from the mean?
   **A:** The mean is the arithmetic average; the median is the middle value when sorted. The median is robust to outliers; the mean is not.

3. **Q:** What is the median of $\{10, 20, 30, 40\}$?
   **A:** Even $n=4$. Two middle values: 20 and 30. Median $= (20+30)/2 = 25$.

4. **Q:** When do we use the median instead of the mean?
   **A:** When data has outliers or is skewed, such as with income data, home prices, or reaction times.

5. **Q:** What does the median tell us about a dataset?
   **A:** It tells us the value below which 50% of the data falls — the centre of the ordered distribution.

### Intermediate - 5

1. **Q:** Prove that the median minimises the sum of absolute deviations.
   **A:** $L(c) = \sum |x_i - c|$. The subgradient is $\partial L/\partial c = -\sum \text{sign}(x_i - c)$. Setting to zero shows the number of points left of $c$ equals the number right of $c$, making $c$ the median.

2. **Q:** How are the median and quartiles related?
   **A:** $Q_2$ is the median. $Q_1$ is the median of the lower half; $Q_3$ is the median of the upper half.

3. **Q:** What is the breakdown point of the median?
   **A:** Approximately 50%. You can corrupt up to half the data points arbitrarily without making the median unbounded.

4. **Q:** How does the median change under monotonic transformations?
   **A:** $\text{Median}(f(X)) = f(\text{Median}(X))$ for any monotonic function $f$, unlike the mean.

5. **Q:** In decision trees, why might the median be preferred over the mean for split points?
   **A:** The median guarantees a balanced split (equal numbers on each side), which can produce more balanced trees.

### Advanced - 3

1. **Q:** Compare the influence function of the median versus the mean.
   **A:** The influence function of the mean is unbounded (IF$(x) = x - \mu$), while the median has a bounded influence function (IF$(x) \propto \text{sign}(x-\theta)/f(\theta)$). This makes the median robust.

2. **Q:** Explain the concept of the median in multivariate data (geometric median).
   **A:** The geometric median minimises $\sum \|x_i - c\|_2$. It is a generalisation to higher dimensions, computed iteratively via Weiszfeld's algorithm.

3. **Q:** How is the median used in the Random Forest algorithm for regression?
   **A:** Individual tree predictions are aggregated using the median instead of the mean to reduce the impact of extreme outlier predictions from individual trees.

## Practice Problems

### Easy - 5

1. Find the median of $\{2, 5, 8, 11, 14\}$.

2. Find the median of $\{3, 7, 3, 9, 5, 7\}$.

3. A dataset with $n=7$ has sorted values $\{1, 3, 5, 7, 9, 11, 13\}$. What is the median?

4. Find the median of $\{100, 200, 300, 400\}$.

5. The sorted values $\{a, b, c, d, e\}$ have median 10. If $c = 10$, what are the values of $a,b,d,e$?

### Medium - 5

1. Given the mean = 20 and median = 25 for a dataset, is the distribution skewed left or right?

2. Find the median of: $\{7, 2, 9, 4, 11, 6, 15, 3, 8\}$.

3. A dataset has median 50. If we add 10 to each value, what is the new median?

4. Compute both mean and median for $\{1, 2, 3, 4, 100\}$. Which is more representative?

5. For the dataset $\{x_1, x_2, x_3, x_4\}$ with $x_1 < x_2 < x_3 < x_4$, express the median in terms of $x_2$ and $x_3$.

### Hard - 3

1. Derive the median of a continuous distribution with PDF $f(x)$.

2. Given the grouped frequency table: Class $0-10$ (freq 5), $10-20$ (freq 12), $20-30$ (freq 8), $30-40$ (freq 5), compute the median using the grouped data formula.

3. Prove that among all values $c$, the sum of absolute deviations $\sum |x_i - c|$ is minimised when $c$ is any median of the dataset.

## Solutions

**Easy:**

1. $n=5$, middle $=8$.

2. Sorted: $3,3,5,7,7,9$, $n=6$, median $=(5+7)/2=6$.

3. $n=7$, middle position $4$, median $=7$.

4. $n=4$, median $=(200+300)/2=250$.

5. Any values with $a \leq b \leq 10 \leq d \leq e$ and $a,b \leq 10 \leq d,e$.

**Medium:**

1. Mean (20) $<$ Median (25), so distribution is negatively skewed (skewed left).

2. Sorted: $2,3,4,6,7,8,9,11,15$, $n=9$, middle $=7$.

3. Median becomes $50+10=60$ (scale equivariance).

4. Mean $=110/5=22$, Median $=3$. The median better represents the data.

5. Median $= (x_2 + x_3)/2$.

**Hard:**

1. For a continuous distribution, median $m$ satisfies $\int_{-\infty}^{m} f(x)\,dx = 0.5$.

2. $n=30$, $n/2=15$. Cumulative: $5, 17, 25, 30$. The median class is $10-20$. $L=10$, $F=5$, $f=12$, $h=10$. Median $= 10 + (15-5)/12 \times 10 = 10 + 100/12 \approx 18.33$.

3. Let $L(c) = \sum |x_i - c|$. The subgradient is $\partial L/\partial c = -\sum \text{sign}(x_i - c)$. For $c$ less than all $x_i$, $\partial L/\partial c = -n$ (decreasing $c$ increases $L$). For $c$ greater than all $x_i$, $\partial L/\partial c = n$ (increasing $c$ increases $L$). The minimum occurs when the derivative crosses zero — i.e., when the number of $x_i$ below $c$ equals the number above $c$. This is exactly the median condition.

## Related Concepts

- Mean (MATH-077) — arithmetic average, sensitive to outliers
- Mode (MATH-079) — most frequent value
- Quartiles — generalisation of median to 25th and 75th percentiles
- Interquartile Range (IQR) — robust spread measure using Q1 and Q3
- Robust Statistics — statistical methods resistant to outliers
- Percentiles — generalisation of median to any percentage

## Next Concepts

- Mode (MATH-079) — third measure of central tendency
- Variance (MATH-080) — spread around the mean
- Standard Deviation (MATH-081) — scale of variability
- Skewness (MATH-084) — asymmetry measured via mean-median relationship

## Summary

The median is the middle value of an ordered dataset, splitting the data into two equal halves. For odd $n$, it is the central value; for even $n$, the average of the two middle values. The median is robust to outliers with a 50% breakdown point, making it the preferred measure of central tendency for skewed distributions. It minimises the sum of absolute deviations, unlike the mean which minimises squared deviations. In AI/ML, the median is used in robust scaling (RobustScaler), median absolute deviation (MAD) for outlier detection, ensemble aggregation, and decision tree splitting.

## Key Takeaways

- The median is the middle value when data is sorted: $x_{(n+1)/2}$ for odd $n$, $(x_{(n/2)} + x_{(n/2+1)})/2$ for even $n$.
- The median is robust to outliers with a 50% breakdown point.
- The median minimises the sum of absolute deviations.
- For skewed distributions, Mode $<$ Median $<$ Mean (right skew) or reverse (left skew).
- The median is used in RobustScaler, MAD, and ensemble methods in ML.
- The median is a specific quartile ($Q_2$) and percentile (50th).
- The median is preferred for income, housing, and any data with extreme values.
