# Concept: Line and Scatter Plots

## Concept ID

PYT-087

## Difficulty

Intermediate

## Domain

Python

## Module

Visualization

## Learning Objectives

- Create line plots with `plt.plot()` and scatter plots with `plt.scatter()`
- Customize markers, line styles, colors, and transparency
- Add error bars with `plt.errorbar()` and shaded regions with `plt.fill_between()`
- Understand when to use line plots versus scatter plots depending on data type

## Prerequisites

- PYT-086 — Matplotlib Basics (figure/axes creation, show/save)
- NumPy arrays and basic operations
- Understanding of x-y coordinate data

## Definition

Line plots and scatter plots are the two most fundamental chart types in data visualization:

- **Line plot** (`plt.plot(x, y)`): Points are connected by straight line segments. Used primarily for continuous data where the order of points matters — time series, mathematical functions, sequential measurements.
- **Scatter plot** (`plt.scatter(x, y)`): Individual points are plotted without connections. Used for examining relationships between two continuous variables — correlation, clustering, outliers.

Both are `Axes` methods that add `Line2D` or `PathCollection` artists to the axes. They support extensive customization of visual properties including color, marker style, size, transparency, and line width.

Extended variants include:
- **Error bars** (`plt.errorbar()`): Scatter points with vertical/horizontal error indicators
- **Fill between** (`plt.fill_between()`): Shaded region between two curves or a curve and a baseline

## Intuition

When you have data that flows — stock prices by minute, temperature by hour, a mathematical function — the line helps the eye follow the sequence. The line acts as a visual guide for continuity and trend.

When you have paired observations — height vs weight, advertising spend vs sales — you want to see individual data points. Scatter plots reveal density, gaps, clusters, and outliers that a line would conceal.

Error bars communicate uncertainty: "our measurement is 5.2 ± 0.3." Fill between shows a range — a confidence band, a min-max envelope, or the area between two curves.

## Why This Concept Matters

Line and scatter plots are everywhere:
- **Exploratory data analysis:** First thing you do with any two-column dataset is scatter it
- **Model evaluation:** Plot predicted vs actual values as a scatter; plot loss curves as lines
- **Scientific communication:** Nearly every paper uses these plots to show results
- **Time series analysis:** Lines are the default visualization for temporal data

Customizing markers, colors, and styles transforms a default plot into a clear, publication-ready figure. Knowing every option for line style (`-`, `--`, `-.`, `:`), marker (circle, square, triangle, diamond), and color (named, hex, RGB) makes your visualizations more informative and professional.

## Real World Examples

1. **Finance:** A line plot of daily closing prices for a stock over 5 years, with moving average overlay in a different style.
2. **Medicine:** A scatter plot of drug dosage vs patient response, with error bars representing standard deviation at each dose level.
3. **Meteorology:** A fill-between plot showing the daily temperature range (min/max) with the average as a central line.
4. **Sports Analytics:** Shot location scatter plot on a basketball court, with marker size proportional to points scored per shot.
5. **ML Diagnostics:** Two line plots overlaid on the same axes — training loss (solid) and validation loss (dashed) — to detect overfitting.

## AI/ML Relevance

- **Loss Curves:** Training vs validation loss plotted as lines over epochs is the most common ML diagnostic
- **Prediction vs Actual:** Scatter plots of ground truth vs predictions reveal bias, variance, and outliers
- **Error Analysis:** `plt.errorbar()` for reporting metrics with confidence intervals
- **Decision Boundaries:** Overlay scatter points on contour plots to show classification regions
- **Hyperparameter Sensitivity:** Line plots of performance vs hyperparameter value

## Code Examples

### Example 1: Basic line plot with default vs customized styles
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 4))
plt.plot(x, y1, label='default style')
plt.plot(x, y2, 'r--o', label='custom: r--o', linewidth=2, markersize=6)
plt.title('Default vs Customized Line Plot')
plt.legend()
plt.show()
```
```
# Output: Two lines — one default blue solid, one red dashed with circle markers.
```

### Example 2: Full customization of line and marker properties
```python
x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)

plt.plot(x, y,
         color='#8E44AD',
         linestyle='-.',
         linewidth=2.5,
         marker='s',
         markersize=8,
         markerfacecolor='#F1C40F',
         markeredgecolor='#E67E22',
         markeredgewidth=2,
         alpha=0.8,
         label='custom sin')
plt.legend()
plt.title('Fully Customized Line/Marker')
plt.show()
```
```
# Output: Purple dash-dot line with square markers; gold fill, orange edge, 80% opacity.
```

### Example 3: Scatter plot with varying size and color
```python
np.random.seed(42)
n = 200
x = np.random.randn(n)
y = np.random.randn(n)
colors = np.sqrt(x**2 + y**2)  # distance from origin
sizes = np.random.randint(20, 200, n)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y,
                      c=colors,
                      s=sizes,
                      cmap='viridis',
                      alpha=0.6,
                      edgecolors='black',
                      linewidth=0.5)
plt.colorbar(scatter, label='Distance from origin')
plt.title('Scatter Plot with Variable Size & Color')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: A "bubble chart" scatter with 200 points; color and size encode a third dimension.
```

### Example 4: Error bars with `plt.errorbar()`
```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.3, 4.1, 3.8, 5.2, 4.9])
y_err = np.array([0.5, 0.4, 0.6, 0.3, 0.5])
x_err = np.array([0.1, 0.1, 0.2, 0.1, 0.15])

plt.errorbar(x, y,
             yerr=y_err,
             xerr=x_err,
             fmt='o',
             color='#2E86C1',
             ecolor='#E74C3C',
             capsize=4,
             capthick=2,
             elinewidth=2,
             markersize=8,
             label='measurements')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data with Error Bars')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Five points with red error bars (caps, thickness=2) in both x and y directions.
```

### Example 5: Fill between with confidence bands
```python
np.random.seed(42)
x = np.linspace(0, 10, 30)
y_true = 2 * x + 1
y_measured = y_true + np.random.randn(30) * 2
y_upper = y_measured + 1.96 * 2
y_lower = y_measured - 1.96 * 2

plt.figure(figsize=(10, 5))
plt.plot(x, y_true, 'k-', label='True line', linewidth=2)
plt.plot(x, y_measured, 'ro', label='Measurements', alpha=0.6)
plt.fill_between(x, y_lower, y_upper,
                 color='blue',
                 alpha=0.15,
                 label='95% CI')
plt.fill_between(x, y_measured - 2, y_measured + 2,
                 color='gray',
                 alpha=0.1,
                 label='±2 band')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line with Confidence Band (fill_between)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: True line, measured points, 95% CI band (blue, light), and ±2 band (gray).
```

### Example 6: Custom marker styles gallery
```python
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
x = np.arange(len(markers))
y = np.ones(len(markers))

fig, ax = plt.subplots(figsize=(10, 3))
for i, marker in enumerate(markers):
    ax.plot(x[i], y[i], marker=marker, markersize=12,
            color='#2C3E50', label=f"'{marker}'")
ax.set_ylim(0.5, 1.5)
ax.set_yticks([])
ax.legend(loc='upper center', ncol=7, fontsize=9)
ax.set_title('Common Matplotlib Markers')
plt.tight_layout()
plt.show()
```
```
# Output: A horizontal row of 14 different marker types with their format string labels.
```

### Example 7: Step plot for discrete data
```python
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 2, 2])

plt.step(x, y, where='post', label='step (post)', linewidth=2)
plt.step(x, y + 0.5, where='pre', label='step (pre)', linewidth=2, linestyle='--')
plt.title('Step Plots')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Two stair-step functions — one with steps after the x-value (post) and one before (pre).
```

### Example 8: Stem plot
```python
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)

plt.stem(x, y, linefmt='C3-', markerfmt='C3o', basefmt='gray')
plt.title('Stem Plot (Discrete Samples)')
plt.show()
```
```
# Output: Vertical lines from baseline to each (x, sin(x)) point, with circle markers at the top.
```

## Common Mistakes

1. **Using line plots for categorical or unordered data.** If x-values are categories (Apple, Banana, Cherry) or unordered, a line suggests a continuity that doesn't exist. Use scatter or bar plots instead.
2. **Plotting too many points with `plt.plot()` vs `plt.scatter()`.** For large datasets (>10k points), `plt.plot(marker='.', linestyle='none')` is more efficient than `plt.scatter()`. For very large datasets (>100k), use rasterization (`rasterized=True`).
3. **Ignoring marker edge colors.** Default markers on a busy background can be hard to see. Add `markeredgecolor='black'` or `markeredgewidth=0.5` for visibility.
4. **Setting `linestyle='-` when data is not sequential.** If data points are not ordered by x, a connecting line will zigzag confusingly. Always sort x or omit the line.
5. **Error bars that overlap with marker.** Increase the marker size or use `capsize` to ensure error bars are visible above the marker.
6. **Forgetting `alpha` on overlapping scatter points.** Without transparency, dense scatter regions appear as solid black blobs. Always add `alpha=0.5` or lower for dense data.
7. **Using the default blue for all elements.** Color is a critical encoding channel. Use it intentionally to distinguish groups, highlight key points, or show a third dimension.

## Interview Questions

### Beginner - 5

1. **Q:** What's the difference between `plt.plot(x, y)` and `plt.scatter(x, y)`?  
   **A:** `plot()` connects points with lines by default and is best for sequential/continuous data. `scatter()` creates individual points, supports varying size/color per point, and is better for relationship analysis.

2. **Q:** How do you change the marker style in a line plot?  
   **A:** Pass the format string: `'ro--'` gives red circles with dashed line. Alternatively use keyword arguments: `marker='o', linestyle='--', color='red'`.

3. **Q:** What does `fmt='o'` mean in `plt.errorbar()`?  
   **A:** It specifies the data point marker format — in this case, circles. If omitted, points are not shown (only error bars). You can combine: `'rs--'` for red squares with dashed line.

4. **Q:** How can you make a scatter plot where point size represents a third variable?  
   **A:** Pass an array to the `s` parameter: `plt.scatter(x, y, s=area_array)`. Each point gets its own size.

5. **Q:** What does `alpha` control in plotting functions?  
   **A:** `alpha` controls transparency, from 0 (invisible) to 1 (fully opaque). Useful for overlapping points and fill regions.

### Intermediate - 5

1. **Q:** How do you create a scatter plot with a colorbar where color encodes a continuous variable?  
   **A:** Pass the variable to `c=` and a colormap to `cmap=`, then call `plt.colorbar()`: `sc = plt.scatter(x, y, c=z, cmap='viridis'); plt.colorbar(sc)`.

2. **Q:** What is the difference between `plt.fill_between()` and `plt.fill_betweenx()`?  
   **A:** `fill_between()` fills vertically between two y-values across a range of x. `fill_betweenx()` fills horizontally between two x-values across a range of y.

3. **Q:** How do you plot multiple datasets with shared axes but different y-ranges?  
   **A:** Use `ax.twinx()` to create a second y-axis sharing the same x-axis: `ax2 = ax.twinx()`.

4. **Q:** What are the valid `where` arguments for `plt.step()` and how do they differ?  
   **A:** `'pre'` (default): the step changes value at the x-coordinate, holding the previous value until then. `'post'`: the step changes after the x-coordinate. `'mid'`: the step changes at the midpoint between points.

5. **Q:** How can you create a scatter plot with 1 million points without memory issues?  
   **A:** Use `rasterized=True` in `ax.scatter()` or use `plt.plot(x, y, ',')` (pixel marker) for the fastest rendering. Alternatively, downsample, use hexbin, or use a 2D histogram.

### Advanced - 3

1. **Q:** How does Matplotlib handle marker path customization? Can you create custom markers?  
   **A:** Yes, by creating a `Path` object from vertices and codes, then using it with `marker=CustomPath`. You can also use `MarkerStyle` with a custom transform for rotated markers.

2. **Q:** Explain the performance trade-offs between `Line2D`, `PathCollection`, and `PathPatch` for rendering many points.  
   **A:** `Line2D` (from `plot()`) is a single artist optimized for connected paths. `PathCollection` (from `scatter()`) can handle varying properties per point but has overhead for large n. `PathPatch` is for individual shapes. For maximum performance on large scatter data, use `plot()` with pixel marker `','`.

3. **Q:** How would you create an interactive point-highlighting scatter plot where hovering a point shows its label?  
   **A:** Use `mplcursors` or implement via `mpl_connect('motion_notify_event')` with `contains()` point-in-test on the scatter `PathCollection`. Update an annotation artist on hover.

## Practice Problems

### Easy - 5

1. **E1:** Plot y = x³ for x in [−3, 3] as a solid red line with circle markers.
2. **E2:** Scatter 100 random points with `np.random.randn(100, 2)` using blue circles, alpha=0.5.
3. **E3:** Create an error bar plot with 5 data points: x=[1,2,3,4,5], y=[10,12,9,11,13], yerr=[1,2,0.5,1.5,1].
4. **E4:** Plot two lines (sine and cosine) on the same axes with different styles: sine is solid, cosine is dashed.
5. **E5:** Generate a step plot of a simple random walk (cumulative sum of random ±1 steps).

### Medium - 5

1. **M1:** Create a scatter plot with 500 points where color represents a continuous variable (e.g., distance from origin) and size represents another (e.g., absolute x-value). Add a colorbar.
2. **M2:** Plot y = sin(x) with a 95% confidence band created from 100 bootstrap resamples at each x point.
3. **M3:** Create a "forecast" plot: actual data (solid line), forecast (dashed line), and 80% prediction interval (shaded region) for 30 time steps.
4. **M4:** Make a scatter matrix (pairs plot) manually for 3 variables using `plt.subplots()` and nested loops.
5. **M5:** Plot three overlapping line styles using `axvline` and `axhline` to create a "crosshair" effect intersecting at the data maximum.

### Hard - 3

1. **H1:** Implement an interactive lasso-select tool for a scatter plot that highlights selected points.
2. **H2:** Build a "connected scatter plot" that animates over time, showing trajectories of particles.
3. **H3:** Create a scatter plot with non-uniform marker scaling based on a log-transformed third variable, using a custom `Normalize` instance.

## Solutions

### E1 Solution
```python
x = np.linspace(-3, 3, 50)
plt.plot(x, x**3, 'ro-')
plt.xlabel('x'); plt.ylabel('y')
plt.title('y = x³')
plt.grid(True)
plt.show()
```

### E2 Solution
```python
data = np.random.randn(100, 2)
plt.scatter(data[:,0], data[:,1], alpha=0.5)
plt.title('Random Scatter')
plt.show()
```

### E3-E5 Solutions follow the same pattern shown in examples above.

### M1 Solution
```python
np.random.seed(42)
n = 500
x = np.random.randn(n)
y = np.random.randn(n)
dist = np.sqrt(x**2 + y**2)
sizes = np.abs(x) * 100
sc = plt.scatter(x, y, c=dist, s=sizes, cmap='plasma', alpha=0.6)
plt.colorbar(sc, label='Distance')
plt.title('Multi-dimensional scatter')
plt.show()
```

### M2-M5 Solutions are extensions of the example code shown above.

### H1-H3 Solutions are beyond the scope of brief code blocks.

## Related Concepts

- 086 — Matplotlib Basics (figure/axes setup, savefig)
- 088 — Bar, Histogram, Box Plots (alternative visualizations for distributions)
- 091 — Subplots (placing line/scatter in multi-panel layouts)
- 092 — Customizing Plots (LaTeX labels, color maps, rcParams)

## Next Concepts

- 089 — Seaborn (statistical visualizations built on Matplotlib)
- 090 — Plotly (interactive line/scatter plots)
- 092 — Customizing Plots (advanced styling)

## Summary

Line plots (`plt.plot()`) are for sequential/continuous data where order matters. Scatter plots (`plt.scatter()`) are for paired observations where individual points matter. Both support extensive customization of markers, colors, styles, and transparency. Error bars (`plt.errorbar()`) add uncertainty visualization, and `plt.fill_between()` shades confidence regions. These are the workhorses of every data scientist's visualization toolkit.

## Key Takeaways

- Use line plots for time series, functions, and ordered data; use scatter plots for correlation/relationship analysis
- Format strings (`'ro--'`) provide concise marker/line/color control
- `scatter()` supports per-point size and color — use it for multi-dimensional data
- Always sort x-data before line plotting to avoid zigzag artifacts
- `alpha` is essential for dense scatter plots to reveal overlapping structure
