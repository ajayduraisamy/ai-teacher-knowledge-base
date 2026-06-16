# Concept: Subplots and Complex Layouts

## Concept ID

PYT-091

## Difficulty

Intermediate

## Domain

Python

## Module

Visualization

## Learning Objectives

- Create multi-panel figure layouts using `plt.subplots()`, `GridSpec`, and `add_subplot()`
- Share axes across subplots to improve visual comparison
- Manage spacing with `tight_layout()` and `constrained_layout`
- Create inset axes within existing plots
- Build publication-quality multi-panel figures

## Prerequisites

- PYT-086 — Matplotlib Basics (Figure/Axes creation)
- NumPy array manipulation
- Basic understanding of plotting functions

## Definition

Subplots are the arrangement of multiple plotting regions (Axes) within a single Figure. Matplotlib provides several approaches:

1. **`plt.subplots(nrows, ncols)`:** The simplest — creates a grid of Axes in one call. Returns `(fig, axes)` where `axes` is a 2D array of Axes objects.
2. **`GridSpec`:** Fine-grained control over subplot geometry — individual cells can span multiple rows/columns.
3. **`fig.add_subplot()`:** Add one subplot at a time to an existing figure.
4. **`fig.add_axes()`:** Create an inset axes at an arbitrary position in figure coordinates.

Key parameters:
- `sharex` / `sharey`: Synchronize axis limits across subplots (options: `True`, `False`, `'row'`, `'col'`)
- `gridspec_kw`: Pass additional kwargs (width_ratios, height_ratios, hspace, wspace) to GridSpec
- `subplot_kw`: Pass kwargs to each subplot (e.g., `projection='polar'`)

## Intuition

Think of a figure as a page layout. `plt.subplots(2, 2)` divides the page into a 2×2 grid. GridSpec is like a table layout where a cell can span multiple rows or columns — creating a "featured" main plot with smaller supporting plots.

The axes sharing mechanism (`sharex`, `sharey`) ensures that when you zoom or pan in one subplot, the others follow. This is essential for comparing time series across different conditions on the same time scale.

## Why This Concept Matters

- **Research Publications:** Nearly all papers have multi-panel figures (e.g., Fig 1A, Fig 1B, Fig 1C) showing different aspects of the same experiment
- **Model Diagnostics:** Combine loss curves, confusion matrix, and feature importance on one figure
- **Comprehensive EDA:** Show distributions, pairwise relationships, and summary stats in a single canvas
- **Dashboards:** Arrange multiple related charts for at-a-glance comparison
- **Efficiency:** One figure with 6 subplots is easier to save, share, and discuss than 6 separate figures

## Real World Examples

1. **ML Research Paper:** A 3×2 figure with training/validation loss, accuracy curves, confusion matrix, ROC curve, feature importances, and sample predictions.
2. **Climate Report:** A 2×1 layout with a map in the top panel and a time series line plot in the bottom panel.
3. **Financial Analysis:** A 3-panel layout: candlestick chart (top, takes 2/3 width), volume bars (middle), and RSI indicator (bottom) with shared x-axis for synchronized panning.
4. **Medical Study:** Subplots showing biomarker distributions for control and treatment groups, each with shared y-axis ranges.
5. **Data Quality Report:** A dashboard with histograms of every numeric column arranged in a grid using GridSpec for flexible sizing.

## AI/ML Relevance

- **Multi-panel Model Evaluation:** Confusion matrix + ROC + precision-recall curve + loss curves in one figure
- **Feature Analysis:** Grid of histograms for every feature, colored by target class
- **Hyperparameter Visualization:** Heatmap of accuracy across two parameters in the main panel, with marginal line plots on top and right
- **Error Analysis:** Scatter of predicted vs actual in main panel, residual histogram in a smaller inset

## Code Examples

### Example 1: Basic `plt.subplots()` grid
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')
axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')
axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_ylim(-5, 5)
axes[1, 0].set_title('tan(x)')
axes[1, 1].plot(x, np.sinh(x))
axes[1, 1].set_title('sinh(x)')

fig.suptitle('2x2 Grid of Subplots', fontsize=14)
plt.tight_layout()
plt.show()
```
```
# Output: A 2×2 grid showing four trigonometric functions, each in its own axes.
```

### Example 2: Shared axes
```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x) + 1
y3 = np.sin(2 * x)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
axes[0].plot(x, y1)
axes[0].set_ylabel('sin(x)')
axes[1].plot(x, y2)
axes[1].set_ylabel('cos(x)+1')
axes[2].plot(x, y3)
axes[2].set_ylabel('sin(2x)')
axes[2].set_xlabel('X')

fig.suptitle('Three Subplots with Shared X-Axis')
plt.tight_layout()
plt.show()
```
```
# Output: Three vertically stacked plots sharing the same x-axis limits and tick labels.
```

### Example 3: GridSpec with spanning cells
```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 0.5])

ax1 = fig.add_subplot(gs[0, :])    # top row, all 3 columns
ax2 = fig.add_subplot(gs[1, 0])    # middle-left
ax3 = fig.add_subplot(gs[1, 1:])   # middle-right, spans 2 columns
ax4 = fig.add_subplot(gs[2, :])    # bottom row, all 3 columns

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x)); ax1.set_title('Main Plot (spans all columns)')
ax2.scatter(np.random.randn(50), np.random.randn(50)); ax2.set_title('Scatter')
ax3.plot(x, np.cos(x)); ax3.set_title('Cos (spans 2 cols)')
ax4.bar(['A', 'B', 'C'], [3, 7, 2]); ax4.set_title('Bar')

fig.suptitle('GridSpec with Variable Spans')
plt.tight_layout()
plt.show()
```
```
# Output: A figure with subplots of different sizes — some spanning multiple grid cells.
```

### Example 4: Inset axes
```python
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), 'b-', label='full range')
ax.set_title('Main Plot with Inset')

# Create an inset axes
inset = fig.add_axes([0.6, 0.6, 0.25, 0.25])
x_zoom = np.linspace(4, 6, 50)
inset.plot(x_zoom, np.sin(x_zoom), 'r-', linewidth=2)
inset.set_title('Zoomed Region', fontsize=9)
inset.set_xlim(4, 6)
inset.set_ylim(-1.1, 1.1)

# Indicate the zoom region on the main plot
ax.axvspan(4, 6, color='red', alpha=0.1)
plt.show()
```
```
# Output: Main sine plot with a red-highlighted region and an inset axes showing the zoomed view.
```

### Example 5: `add_subplot()` for irregular positioning
```python
fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)  # bottom, spans both columns

x = np.linspace(0, 10, 100)
ax1.plot(x, x**2); ax1.set_title('Square')
ax2.plot(x, x**3); ax2.set_title('Cube')
ax3.plot(x, x**4, 'r-'); ax3.set_title('Fourth Power (wide)')
ax3.set_xlabel('X')

plt.tight_layout()
plt.show()
```
```
# Output: Three subplots — two small ones on top, one wide one below spanning the full width.
```

### Example 6: Constrained Layout vs Tight Layout
```python
fig, axes = plt.subplots(2, 2, figsize=(8, 6),
                         constrained_layout=True)
for i, ax in enumerate(axes.flat):
    ax.plot([0, 1], [0, 1])
    ax.set_title(f'Subplot {i+1}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
print("constrained_layout=True handles spacing automatically.")
plt.show()

# Compare with tight_layout
fig2, axes2 = plt.subplots(2, 2, figsize=(8, 6))
for i, ax in enumerate(axes2.flat):
    ax.plot([0, 1], [0, 1])
    ax.set_title(f'Subplot {i+1}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
fig2.tight_layout()
print("tight_layout() also works but is a post-processing step.")
plt.show()
```
```
# Output: Two identical-looking 2x2 grids — constrained_layout is set at creation, tight_layout is called after plotting.
```

### Example 7: Nested GridSpec
```python
fig = plt.figure(figsize=(10, 8))
outer = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)

# Top-left is a 2x2 grid itself
inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0])
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(inner[i, j])
        ax.plot(np.random.randn(10))
        ax.set_title(f'Inner ({i},{j})', fontsize=8)

# Other outer cells are simple plots
ax1 = fig.add_subplot(outer[0, 1])
ax1.bar(['A', 'B'], [3, 7])
ax1.set_title('Bar', fontsize=10)

ax2 = fig.add_subplot(outer[1, :])  # bottom spans both columns
ax2.plot(np.cumsum(np.random.randn(100)))
ax2.set_title('Random Walk', fontsize=10)

fig.suptitle('Nested GridSpec')
plt.show()
```
```
# Output: Complex layout with a 2×2 mini-grid inside the top-left cell, a bar chart top-right, and a wide line plot across the bottom.
```

### Example 8: Setting common axis labels
```python
fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
for ax in axes.flat:
    ax.scatter(np.random.randn(20), np.random.randn(20), alpha=0.6)

# Set common labels using figure.text
fig.text(0.5, 0.02, 'Common X Label', ha='center', fontsize=12)
fig.text(0.02, 0.5, 'Common Y Label', va='center', rotation=90, fontsize=12)
fig.suptitle('Shared Axes with Common Labels')

plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.show()
```
```
# Output: 2×3 scatter subplots sharing axes, with one centered x-label and one y-label for the whole figure.
```

### Example 9: Adding a colorbar to multi-subplot figures
```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x = np.linspace(0, 10, 50)
for ax in axes:
    sc = ax.scatter(x, np.sin(x), c=np.cos(x), cmap='viridis')
    ax.set_title('Sine')

fig.colorbar(sc, ax=axes, label='cos(x) value', shrink=0.8)
plt.tight_layout()
plt.show()
```
```
# Output: Three identical sine scatter plots with a single shared colorbar on the right.
```

### Example 10: Hiding unused subplot axes
```python
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 5:  # only 5 plots, hide the 6th
        ax.plot(np.random.randn(10))
        ax.set_title(f'Plot {i+1}')
    else:
        ax.set_visible(False)
plt.tight_layout()
plt.show()
```
```
# Output: 5 active subplots with the 6th cell hidden (no axes drawn).
```

## Common Mistakes

1. **Indexing the axes array incorrectly.** For 1-row or 1-col subplots, `axes` is 1D, not 2D. For 2+ rows and cols, it's 2D. Use `axes.flat` to always get a flat iterator.
2. **Forgetting `tight_layout()` or `constrained_layout=True`.** Without them, subplot titles and labels overlap.
3. **Using `sharex=True` when subplots have different x-ranges.** Shared axes force all subplots to the same x-limits. Use `sharex=False` or adjust limits after sharing.
4. **Mismatched `GridSpec` indices.** GridSpec uses 0-based indexing and Python slicing: `gs[0:2, 1]` spans rows 0-1 in column 1.
5. **Creating subplots in a loop without referencing the correct axes.** In a 2×2 grid, `axes[1, 0]` is bottom-left. Drawing to the wrong index is a common bug.
6. **Not accounting for colorbar space.** A colorbar added to a subplot layout compresses existing subplots. Use `fig.subplots_adjust(right=0.85)` to reserve space on the right.
7. **Using `fig.add_subplot(2, 2, 1)` after `plt.subplots(2, 2)`.** This creates a new subplot on top of existing ones. Either use the pre-created axes from `subplots()` or use `add_subplot()` on a fresh `plt.figure()`.

## Interview Questions

### Beginner - 5

1. **Q:** What does `plt.subplots(2, 3)` return?  
   **A:** A tuple `(Figure, ndarray of Axes)`. The axes array has shape `(2, 3)`, accessible as `axes[0, 0]`, `axes[0, 1]`, etc.

2. **Q:** How do you make a figure with 4 subplots (2 rows × 2 columns)?  
   **A:** `fig, axes = plt.subplots(2, 2, figsize=(10, 8))`.

3. **Q:** What does `sharex='col'` mean in `plt.subplots()`?  
   **A:** Subplots in the same column share the same x-axis limits. When you zoom or pan one, all subplots in that column follow.

4. **Q:** How do you prevent axis labels from overlapping between subplots?  
   **A:** Call `plt.tight_layout()` after plotting, or set `constrained_layout=True` in `plt.subplots()`.

5. **Q:** What is `GridSpec`?  
   **A:** `GridSpec` from `matplotlib.gridspec` allows flexible subplot layouts where cells can span multiple rows or columns, unlike the uniform grid of `plt.subplots()`.

### Intermediate - 5

1. **Q:** How do you add a single colorbar for multiple subplots?  
   **A:** Pass all subplot axes to the `ax` parameter: `fig.colorbar(sc, ax=axes, shrink=0.8)`. Using `ax=axes` (the array) ensures the colorbar spans all subplots.

2. **Q:** What is the difference between `tight_layout()` and `constrained_layout=True`?  
   **A:** `tight_layout()` is a post-creation adjustment that minimizes whitespace. `constrained_layout=True` is set at figure creation and dynamically adjusts during drawing. Constrained layout is more flexible with colorbars and legends but slightly slower.

3. **Q:** How would you create a figure where the top subplot is twice the height of the bottom subplot?  
   **A:** Use `gridspec_kw={'height_ratios': [2, 1]}` in `plt.subplots()`:  
   `fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})`.

4. **Q:** How do you create subplots with different projections (e.g., 3D, polar)?  
   **A:** Pass `subplot_kw={'projection': '3d'}` or `{'projection': 'polar'}`:  
   `fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})`.

5. **Q:** What does `fig.add_axes([0.1, 0.1, 0.5, 0.3])` do?  
   **A:** It adds a new axes at the position [left, bottom, width, height] in figure coordinates (0-1 normalized). Used for placing axes at arbitrary positions, including insets.

### Advanced - 3

1. **Q:** Explain the `GridSpecFromSubplotSpec` class and when you'd use it.  
   **A:** It creates a nested GridSpec within a cell of an outer GridSpec, enabling subplots-within-subplots for complex publication layouts (e.g., a 2×2 mini-grid inside the top-left panel of a 3×3 outer grid).

2. **Q:** How does `constrained_layout` handle colorbars and legends compared to `tight_layout()`?  
   **A:** `constrained_layout` uses a constraint solver and tracks colorbar/legend artists during the draw phase, automatically reserving space. `tight_layout()` only considers axis labels and titles — you must manually add `fig.subplots_adjust()` for colorbars.

3. **Q:** Design a layout for a research paper figure that includes: a main scatter plot taking 60% width on the left, a vertical column of 3 small plots (histogram, box plot, violin plot) on the right, spanning the same height.  
   **A:** Use `GridSpec` with 3 rows and 2 columns, `width_ratios=[3, 1]`. The left column uses `gs[:, 0]` spanning all 3 rows for the main plot. The right column has 3 separate cells `gs[0, 1]`, `gs[1, 1]`, `gs[2, 1]` for the three small plots.

## Practice Problems

### Easy - 5

1. **E1:** Create a 1×3 subplot grid showing sin, cos, and tan in each subplot.
2. **E2:** Create a 2×2 grid of scatter plots with random data.
3. **E3:** Create 3 vertically stacked subplots sharing the x-axis.
4. **E4:** Create a figure with a 2×2 grid and add a figure-level title.
5. **E5:** Create a 1×2 subplot where the left is a line plot and the right is a bar chart.

### Medium - 5

1. **M1:** Use `GridSpec` to create a figure with a main plot spanning 2 columns on top, and 2 smaller plots below.
2. **M2:** Create a figure with a colorbar shared across 2×2 subplots of imshow data.
3. **M3:** Build an inset zoom plot: main plot of y = sin(x) over [0, 10] with an inset zoomed to [4, 6].
4. **M4:** Create a 3×3 subplot grid where only the 5 active plots are visible (hide the remaining 4).
5. **M5:** Use `height_ratios` and `width_ratios` to create a layout with a dominant central plot and smaller surrounding plots.

### Hard - 3

1. **H1:** Build a multi-panel figure using nested `GridSpecFromSubplotSpec` with 4 different plot types at varying sizes.
2. **H2:** Create a linked-brushing system where selecting data in one subplot highlights matching data in all other subplots.
3. **H3:** Build a figure with `add_axes` insets that automatically position based on the parent axes' data limits (transforms-based).

## Solutions

### E1 Solution
```python
x = np.linspace(0, 2*np.pi, 100)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].plot(x, np.sin(x)); axes[0].set_title('sin')
axes[1].plot(x, np.cos(x)); axes[1].set_title('cos')
axes[2].plot(x, np.tan(x)); axes[2].set_ylim(-5, 5); axes[2].set_title('tan')
plt.tight_layout()
plt.show()
```

### E2 Solution
```python
fig, axes = plt.subplots(2, 2)
for ax in axes.flat:
    ax.scatter(np.random.randn(30), np.random.randn(30))
plt.tight_layout()
plt.show()
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions follow the GridSpec and spanning techniques shown in the examples.

### H1 Solution (Outline)
```python
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
fig = plt.figure(figsize=(12, 10))
outer = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0])
# ... add subplots to inner grid ...
# ... add remaining outer cells ...
```

## Related Concepts

- 086 — Matplotlib Basics (figure/axes creation)
- 087 — Line and Scatter (content for subplots)
- 088 — Bar, Histogram, Box Plots (content for subplots)
- 092 — Customizing Plots (styling subplot elements)

## Next Concepts

- 092 — Customizing Plots (advanced styling of subplot elements)
- 089 — Seaborn (FacetGrid uses subplots internally)
- 090 — Plotly (make_subplots, interactive multi-panel)

## Summary

Subplots allow multiple plots in a single figure. `plt.subplots()` creates uniform grids; `GridSpec` enables cells of varying sizes and spans; `add_subplot()` adds one at a time; `add_axes()` creates insets. Sharing axes synchronizes zoom/pan across subplots. `tight_layout()` and `constrained_layout` manage spacing. Mastering subplots is essential for publication-quality multi-panel figures.

## Key Takeaways

- Use `plt.subplots()` for uniform grids; `GridSpec` for irregular layouts
- Index axes carefully: `axes.flat` flattens any-shaped array
- `sharex`/`sharey` synchronize axis limits across subplots
- Always call `tight_layout()` or use `constrained_layout=True` to prevent overlap
- Colorbars can span multiple subplots with `fig.colorbar(sc, ax=axes)`
- `fig.add_axes([l, b, w, h])` creates insets and arbitrary-position plots
