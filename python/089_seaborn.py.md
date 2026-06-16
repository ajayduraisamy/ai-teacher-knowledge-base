# Concept: Seaborn Statistical Visualizations

## Concept ID

PYT-089

## Difficulty

Intermediate

## Domain

Python

## Module

Visualization

## Learning Objectives

- Create statistical visualizations using Seaborn's high-level API: scatter, line, bar, box, violin, heatmap, pairplot, and histogram
- Understand how Seaborn integrates with Pandas DataFrames for "tidy data" plotting
- Customize Seaborn aesthetics with `sns.set_theme()` and palette management
- Apply Seaborn to visualize ML datasets for exploratory data analysis

## Prerequisites

- PYT-086 — Matplotlib Basics (Seaborn renders through Matplotlib)
- PYT-088 — Bar, Histogram, Box Plots (underlying statistical concepts)
- Pandas DataFrames (Seaborn operates natively on DataFrames)
- Basic understanding of categorical and continuous data types

## Definition

Seaborn is a Python statistical visualization library built on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. Key features include:

- **Tidy data API:** Functions accept Pandas DataFrames with `data=` parameter and use column names for `x`, `y`, `hue`, etc.
- **Statistical defaults:** Automatically computes and displays statistical summaries (e.g., `sns.barplot` shows mean + CI)
- **Color palettes:** Purpose-built palettes for categorical, sequential, and diverging data
- **Faceting:** `sns.FacetGrid` and `sns.relplot`/`sns.catplot` create multi-plot grids by conditioning on additional variables
- **Theme system:** `sns.set_theme()` applies a consistent, publication-ready style to all plots

Key functions covered:
- `sns.scatterplot()` / `sns.relplot(kind='scatter')` — scatter plots with optional hue/style/size
- `sns.lineplot()` / `sns.relplot(kind='line')` — line plots with confidence bands
- `sns.barplot()`, `sns.boxplot()`, `sns.violinplot()` — categorical estimates
- `sns.histplot()` / `sns.kdeplot()` / `sns.ecdfplot()` — distribution plots (replacing deprecated `sns.distplot`)
- `sns.heatmap()` — matrix heatmaps
- `sns.pairplot()` — pairwise scatter plot matrix

## Intuition

Seaborn assumes you have "tidy" data — every column is a variable, every row is an observation. This lets you express plots declaratively:

```python
sns.scatterplot(data=df, x='weight', y='height', hue='gender')
```

Instead of manually splitting data into groups and calling Matplotlib repeatedly, you tell Seaborn which columns map to which visual channels. It handles aggregation, color assignment, legend creation, and faceting automatically.

The philosophy: "I have this DataFrame and I want to see the relationship between these columns." Seaborn figures out the rest.

## Why This Concept Matters

- **Rapid EDA:** Seaborn lets you explore relationships in a dataset with a single line of code
- **Statistical Orientation:** It shows uncertainty (confidence intervals, bootstrapped error bars) by default
- **Publication Quality:** Seaborn's default theme and color palettes produce attractive plots with minimal customization
- **ML Workflow:** Seaborn is the standard tool for visualizing ML datasets — examining feature distributions, class balance, correlations, and outliers
- **Industry Standard:** Seaborn is widely used in data science, ML research, and industry analysis

## Real World Examples

1. **Customer Churn Analysis:** A telco analyst uses `sns.pairplot()` with `hue='churn'` to visualize feature differences between churned and retained customers.
2. **Medical Study:** A researcher uses `sns.boxplot()` to compare biomarker levels across control/treatment/dose groups, with `sns.swarmplot()` overlaid for raw data.
3. **Financial Risk:** A quantitative analyst uses `sns.heatmap(corr)` to visualize the correlation matrix of 30 stock returns.
4. **E-commerce:** An analyst uses `sns.barplot()` to show average cart value by traffic source with bootstrapped confidence intervals.
5. **ML Pipeline:** A data scientist uses `sns.histplot()` with `hue='label'` on each feature before training a classifier, detecting which features separate classes well.

## AI/ML Relevance

- **EDA Before ML:** Seaborn is the go-to for exploring feature distributions, target balance, and feature-target relationships
- **Correlation Analysis:** `sns.heatmap(df.corr())` is essential for detecting multicollinearity
- **Feature Separation:** `sns.pairplot(hue='target')` shows which features separate classes
- **Model Diagnostics:** Residual distributions, prediction vs actual, and feature importance visualization
- **Hyperparameter Analysis:** Heatmaps of accuracy across two hyperparameter dimensions

## Code Examples

### Example 1: Basic scatter and line plots with `sns.scatterplot` / `sns.lineplot`
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style='whitegrid')

df = pd.DataFrame({
    'x': np.linspace(0, 10, 50),
    'y': np.sin(np.linspace(0, 10, 50)) + np.random.randn(50) * 0.2,
    'group': ['A'] * 25 + ['B'] * 25
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.scatterplot(data=df, x='x', y='y', hue='group', style='group', ax=ax1)
ax1.set_title('Scatter Plot')

sns.lineplot(data=df, x='x', y='y', hue='group', style='group', ax=ax2)
ax2.set_title('Line Plot with CI Band')

plt.tight_layout()
plt.show()
```
```
# Output: Two subplots — scatter (left) with points colored by group, line (right) with confidence bands.
```

### Example 2: Categorical plots — bar, box, violin
```python
tips = sns.load_dataset('tips')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sns.barplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[0])
axes[0].set_title('Bar Plot (mean ± CI)')

sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[1])
axes[1].set_title('Box Plot')

sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, ax=axes[2])
axes[2].set_title('Violin Plot (split)')

plt.tight_layout()
plt.show()
```
```
# Output: Three panels — bar with error bars, box plots, and split violins by day/sex.
```

### Example 3: Distribution plots — histplot and kdeplot
```python
penguins = sns.load_dataset('penguins')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sns.histplot(data=penguins, x='flipper_length_mm', hue='species', ax=axes[0])
axes[0].set_title('Histogram by Species')

sns.kdeplot(data=penguins, x='flipper_length_mm', hue='species', fill=True, ax=axes[1])
axes[1].set_title('KDE Plot')

sns.ecdfplot(data=penguins, x='flipper_length_mm', hue='species', ax=axes[2])
axes[2].set_title('ECDF Plot')

plt.tight_layout()
plt.show()
```
```
# Output: Three distribution views — histogram, KDE, and ECDF of flipper length per species.
```

### Example 4: Heatmap for correlation
```python
np.random.seed(42)
corr_data = pd.DataFrame(np.random.randn(100, 6),
                         columns=['Feature_A', 'Feature_B', 'Feature_C',
                                  'Feature_D', 'Feature_E', 'Feature_F'])
corr_matrix = corr_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```
```
# Output: A heatmap of the 6×6 correlation matrix with numeric annotations, red-blue colormap.
```

### Example 5: Pairplot for multi-dimensional EDA
```python
iris = sns.load_dataset('iris')

sns.pairplot(iris, hue='species', diag_kind='kde',
             markers=['o', 's', 'D'],
             palette='Set2')
plt.show()
```
```
# Output: 4×4 grid of scatter plots (lower triangle) and KDE (diagonal), colored by species.
```

### Example 6: Relplot with faceting
```python
mpg = sns.load_dataset('mpg').dropna()

g = sns.relplot(data=mpg, x='horsepower', y='mpg',
                hue='origin', col='cylinders',
                kind='scatter', col_wrap=3,
                height=3.5, aspect=1.2)
g.fig.suptitle('MPG vs Horsepower by Cylinder Count', y=1.02)
plt.show()
```
```
# Output: Multiple scatter plots arranged by cylinder count, colored by origin.
```

### Example 7: Customizing themes and palettes
```python
sns.set_theme(style='darkgrid', palette='muted', font='monospace')

df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [23, 45, 56, 78]
})

sns.barplot(data=df, x='category', y='value', palette='viridis')
plt.title('Dark Grid Theme with Viridis Palette')
plt.show()

# Reset to default style
sns.set_theme()
```
```
# Output: Bar chart with dark background grid and green-to-yellow color mapping.
```

### Example 8: Catplot for multi-faceted categorical plots
```python
exercise = sns.load_dataset('exercise')

g = sns.catplot(data=exercise, x='time', y='pulse',
                hue='kind', col='diet',
                kind='box', height=4, aspect=0.8)
g.fig.suptitle('Pulse by Exercise Kind and Diet', y=1.05)
g.set_axis_labels('Time', 'Pulse')
plt.show()
```
```
# Output: Box plots of pulse split by diet (columns) and exercise kind (hue), over time.
```

### Example 9: Jointplot for bivariate + marginal distributions
```python
penguins = sns.load_dataset('penguins').dropna()

g = sns.jointplot(data=penguins, x='bill_length_mm', y='bill_depth_mm',
                  hue='species', kind='kde',
                  marginal_ticks=True, palette='Set1')
g.fig.suptitle('Bill Dimensions by Species', y=1.02)
plt.show()
```
```
# Output: Central KDE contour plot with marginal KDEs on top and right, colored by species.
```

### Example 10: Clustered heatmap (advanced)
```python
flights = sns.load_dataset('flights')
pivot = flights.pivot(index='month', columns='year', values='passengers')

g = sns.clustermap(pivot, cmap='YlGnBu',
                   standard_scale=1,
                   row_cluster=True, col_cluster=True,
                   figsize=(8, 6),
                   linewidths=0.5)
g.fig.suptitle('Clustered Heatmap of Flight Passengers', y=1.02)
plt.show()
```
```
# Output: Heatmap with dendrograms showing clustering of months and years by passenger volume.
```

## Common Mistakes

1. **Using deprecated `sns.distplot()`.** It was removed in Seaborn 0.14. Use `sns.histplot()` or `sns.kdeplot()` instead.
2. **Forgetting `data=` parameter and passing raw arrays.** Seaborn is designed for DataFrames. Passing arrays loses column-name-based legend and labeling.
3. **Overriding Seaborn's statistics inadvertently.** `sns.barplot()` shows mean ± bootstrapped CI by default. If you want raw values, use `sns.stripplot()` or `sns.swarmplot()`.
4. **Not using `hue` for grouping.** Seaborn's `hue` parameter automatically separates groups with distinct colors and creates a legend. Manually filtering data is almost always worse.
5. **Setting global style with `sns.set_style()` inside scripts that are imported.** This affects all downstream plotting. Use `sns.set_theme()` once at the top of a script, or use `with sns.axes_style('darkgrid'):` for local scope.
6. **Calling `plt.figure()` before a Seaborn plot that uses `plt.subplots()` internally.** Many Seaborn functions create their own figure. Use the `ax=` parameter to direct output to a specific axes.
7. **Misusing `sns.pairplot()` on datasets with many columns (>10).** It becomes too dense to read. Use `sns.heatmap(df.corr())` for a compact overview instead.

## Interview Questions

### Beginner - 5

1. **Q:** What is the main advantage of Seaborn over Matplotlib?  
   **A:** Seaborn provides a high-level, declarative API that works directly with Pandas DataFrames, automatically handles statistical aggregation, produces attractive defaults, and requires far less code for common statistical plots.

2. **Q:** How do you create a scatter plot colored by a categorical variable in Seaborn?  
   **A:** `sns.scatterplot(data=df, x='col1', y='col2', hue='category')`. The `hue` parameter automatically assigns colors and creates a legend.

3. **Q:** What does `sns.pairplot()` show?  
   **A:** A matrix of scatter plots for every pair of numeric columns in a DataFrame, with histograms/KDEs on the diagonal. Colored by `hue` if specified.

4. **Q:** How do you change the overall Seaborn style?  
   **A:** Call `sns.set_theme(style='darkgrid', palette='muted')` before plotting. Other styles: `'whitegrid'`, `'dark'`, `'white'`, `'ticks'`.

5. **Q:** What is the difference between `sns.barplot()` and `plt.bar()`?  
   **A:** `sns.barplot()` automatically computes the mean of `y` for each `x` category and shows a bootstrapped confidence interval. `plt.bar()` shows raw values as-is.

### Intermediate - 5

1. **Q:** How does `sns.catplot()` differ from `sns.boxplot()`?  
   **A:** `sns.catplot()` is a figure-level function that creates a `FacetGrid`, supporting `col=` and `row=` parameters for subplots. `sns.boxplot()` is an axes-level function for a single panel.

2. **Q:** Explain the `kind` parameter in `sns.relplot()` and `sns.catplot()`.  
   **A:** For `relplot()`, `kind` can be `'scatter'` (default) or `'line'`. For `catplot()`, `kind` can be `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, or `'count'`.

3. **Q:** What is the difference between `sns.histplot()` and `sns.kdeplot()`?  
   **A:** `histplot()` creates a binned histogram (discrete bars). `kdeplot()` estimates and plots a smooth continuous density curve. Both can show univariate or bivariate distributions.

4. **Q:** How would you overlay raw data points on a box plot in Seaborn?  
   **A:** Combine `sns.boxplot()` with `sns.stripplot()` or `sns.swarmplot()` on the same axes:  
   `ax = sns.boxplot(...); sns.stripplot(..., ax=ax, color='black', alpha=0.5, jitter=True)`.

5. **Q:** What palette types are available in Seaborn?  
   **A:** Categorical (`'Set1'`, `'Set2'`, `'Paired'`, `'husl'`, `'pastel'`), sequential (`'viridis'`, `'Blues'`, `'YlOrRd'`), diverging (`'RdBu'`, `'coolwarm'`, `'vlag'`). Created with `sns.color_palette()`.

### Advanced - 3

1. **Q:** How does Seaborn calculate the confidence interval in `sns.barplot()` and `sns.pointplot()`?  
   **A:** By default, it uses bootstrapping (resampling with replacement 10000 times) to compute the 95% confidence interval of the mean. This can be changed via `ci=68` (1 std) or `ci=None` to disable.

2. **Q:** Explain how `sns.FacetGrid` works and when to use it instead of `plt.subplots()`.  
   **A:** `FacetGrid` takes a DataFrame and column names for `row`, `col`, and `hue`. It creates a grid of subplots and maps a plotting function (`map()` or `map_dataframe()`) to each cell. Use it when you want to condition a visualization on categorical variables without manually looping over subplots.

3. **Q:** How would you create a custom diverging palette centered at a specific value (e.g., 0.5 for a correlation where 0.5 is neutral)?  
   **A:** Use `sns.diverging_palette(250, 15, center='light', as_cmap=True)` for a custom palette. For an existing palette, use a `DivergingNorm` (now `TwoSlopeNorm` in Matplotlib 3.4+) with `vcenter=0.5`.

## Practice Problems

### Easy - 5

1. **E1:** Load the `tips` dataset and create a scatter plot of `total_bill` vs `tip`.
2. **E2:** Create a histogram of `total_bill` from the `tips` dataset.
3. **E3:** Create a box plot of `tip` by `day` from the `tips` dataset.
4. **E4:** Create a heatmap of the correlation matrix of the numeric columns in `iris`.
5. **E5:** Create a bar plot of `class` survival counts from `titanic`.

### Medium - 5

1. **M1:** Create a pairplot of `penguins` colored by `species` using KDE on the diagonal.
2. **M2:** Create a 2×2 FacetGrid of scatter plots: `bill_length_mm` vs `bill_depth_mm` by `species` and `island`.
3. **M3:** Overlay a swarm plot on a violin plot showing `total_bill` by `day` from `tips`.
4. **M4:** Create a joint plot with KDE and marginal histograms of `sepal_length` vs `sepal_width` from `iris`.
5. **M5:** Create a clustered heatmap from the `flights` dataset and interpret the clustering.

### Hard - 3

1. **H1:** Create a custom pairplot that shows scatter in the lower triangle, correlation coefficients in the upper triangle, and histograms on the diagonal.
2. **H2:** Build a function that takes a DataFrame and creates a grid of `sns.ecdfplot()` for every numeric column, faceted by a categorical column of choice.
3. **H3:** Use `sns.FacetGrid` to create a multi-panel plot with different plot types per facet (e.g., box plot in one facet, violin in another).

## Solutions

### E1 Solution
```python
tips = sns.load_dataset('tips')
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.show()
```

### E2 Solution
```python
sns.histplot(data=tips, x='total_bill', bins=20)
plt.show()
```

### E3 Solution
```python
sns.boxplot(data=tips, x='day', y='tip')
plt.show()
```

### E4 Solution
```python
iris = sns.load_dataset('iris')
sns.heatmap(iris.select_dtypes('number').corr(), annot=True)
plt.show()
```

### E5 Solution
```python
titanic = sns.load_dataset('titanic')
sns.barplot(data=titanic, x='class', y='survived')
plt.show()
```

### M1 Solution
```python
penguins = sns.load_dataset('penguins')
sns.pairplot(penguins, hue='species', diag_kind='kde')
plt.show()
```

### M2 Solution
```python
penguins = sns.load_dataset('penguins')
g = sns.FacetGrid(penguins, col='species', row='island', height=2.5)
g.map_dataframe(sns.scatterplot, x='bill_length_mm', y='bill_depth_mm')
plt.show()
```

### M3-M5 Solutions follow the patterns in the code examples.

## Related Concepts

- 086 — Matplotlib Basics (Seaborn is built on Matplotlib)
- 088 — Bar, Histogram, Box Plots (statistical concepts)
- 091 — Subplots (FacetGrid internally uses subplots)
- 092 — Customizing Plots (Matplotlib customization applies to Seaborn)

## Next Concepts

- 090 — Plotly (interactive alternative to Seaborn)
- 093 — sklearn Basics (EDA + ML pipeline)
- 094 — Preprocessing (understanding feature distributions)

## Summary

Seaborn is a high-level statistical visualization library that works directly with Pandas DataFrames. It provides declarative APIs for scatter plots, line plots, categorical plots, distribution plots, heatmaps, and pairwise plots with automatic statistical aggregation, attractive defaults, and faceting support. Seaborn is the standard tool for EDA in the Python ML ecosystem.

## Key Takeaways

- Seaborn assumes tidy data (DataFrame with columns as variables)
- Use `sns.set_theme()` once at the top for consistent styling
- `hue`, `style`, `size`, `col`, `row` provide powerful multi-dimensional encoding
- Figure-level functions (`relplot`, `catplot`, `displot`) return `FacetGrid`; axes-level functions (`scatterplot`, `boxplot`) work on specific axes
- Seaborn is ideal for EDA before ML, revealing feature distributions, correlations, and class separation
- Always use `sns.histplot()` or `sns.kdeplot()` — never the deprecated `sns.distplot()`
