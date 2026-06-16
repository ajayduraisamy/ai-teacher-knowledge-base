# Concept: Customizing Plots

## Concept ID

PYT-092

## Difficulty

Intermediate

## Domain

Python

## Module

Visualization

## Learning Objectives

- Add LaTeX-formatted labels, titles, and annotations to plots
- Configure legends with custom positions, outside the axes, and multi-column layouts
- Apply named colors, hex codes, and Matplotlib colormaps effectively
- Use `rcParams` for global default customization
- Apply pre-built stylesheets with `plt.style.use()`
- Add annotations, text boxes, and reference lines with `axvline`/`axhline`

## Prerequisites

- PYT-086 — Matplotlib Basics
- PYT-087 — Line and Scatter
- Basic familiarity with LaTeX math syntax

## Definition

Customizing plots transforms default Matplotlib output into publication-ready, visually clear, and brand-consistent figures. Key customization areas include:

- **Labels and Titles:** Plain text, LaTeX math expressions (`$\\alpha$`, `$\\beta$`), font size/weight/family
- **Legends:** Location codes, outside-the-axes placement, title, column count, frame styling
- **Colors:** Named colors (140+ CSS colors), hex strings (`#FF5733`), RGB tuples, colormaps (sequential, diverging, qualitative)
- **rcParams:** Global configuration dictionary for matplotlib defaults — set figure size, DPI, font family, tick size, grid style
- **Stylesheets:** Pre-defined style packages (`'seaborn-v0_8'`, `'ggplot'`, `'bmh'`, `'dark_background'`, `'fivethirtyeight'`)
- **Annotations:** Text annotations with arrows, bounding boxes, and precise positioning
- **Reference Lines:** `axvline()` (vertical), `axhline()` (horizontal), `axvspan()`/`axhspan()` (shaded spans)

## Intuition

Think of a Matplotlib plot as a blank canvas where every element has configurable properties. The defaults (blue line, white background, auto-legend placement) are designed for quick exploration, but customization is how you make the plot your own.

`rcParams` is like a master style sheet: set it once at the top of a script and every subsequent plot inherits those defaults. Stylesheets are pre-made rcParams bundles for specific aesthetics (e.g., FiveThirtyEight's bold infographic style).

LaTeX rendering (when enabled with `usetex=True`) produces publication-quality mathematical typesetting. For systems without LaTeX, Matplotlib's built-in math parser handles most common expressions using `$...$` delimiters.

## Why This Concept Matters

- **Publication Requirements:** Journals require specific fonts, sizes, color schemes, and resolution
- **Brand Consistency:** Companies want plots that match their brand guidelines (colors, fonts, logo presence)
- **Clarity:** Good labels, annotations, and reference lines guide the reader to the key insight
- **Accessibility:** Choosing colorblind-safe colormaps and sufficient contrast ensures your work reaches more readers
- **Automation:** Setting `rcParams` in a script ensures all generated plots share a consistent look without repeating style code

## Real World Examples

1. **Journal Article:** A figure with LaTeX axis labels (`$E = mc^2$`), Times New Roman font, and a legend placed outside the plot area.
2. **Company Report:** All plots use the corporate color palette (`#003366`, `#CC3333`, etc.) set via a custom `.mplstyle` file.
3. **Scientific Poster:** A dark-background plot with neon-colored lines and bold annotations for a conference poster session.
4. **Automated Reports:** A scripts that generates 50 plots daily — all consistent because `rcParams` is configured once in the main function.
5. **Data Journalism:** A FiveThirtyEight-style infographic with bold titles, annotated key events, and reference lines marking averages.

## AI/ML Relevance

- **Consistent Experiment Tracking:** Set `rcParams` once so every training curve plot looks the same, making visual comparison easy
- **Publication Figures:** ML research papers demand high-quality, customized figures
- **Error Annotations:** Annotate misclassification examples with arrows and text
- **Reference Lines:** `axhline(y=0.5)` for chance level in ROC plots, `axvline(x=epoch)` for early stopping point
- **Custom Colormaps:** Highlight model attention or saliency with custom diverging colormaps

## Code Examples

### Example 1: LaTeX labels and titles
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y, label=r'$\sin(x)$')
plt.title(r'Trigonometric Function: $y = \sin(x)$', fontsize=14)
plt.xlabel(r'$x$ (radians)', fontsize=12)
plt.ylabel(r'$\sin(x)$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Sine plot with LaTeX-formatted title, axis labels, and legend entry.
```

### Example 2: Advanced LaTeX (requires usetex or built-in parser)
```python
plt.figure(figsize=(8, 4))
x = np.linspace(0, 3, 100)
plt.plot(x, np.exp(-x) * np.sin(2*np.pi*x), label=r'$e^{-x} \sin(2\pi x)$')
plt.title(r'Damped Oscillation: $e^{-x} \sin(2\pi x)$', fontsize=14)
plt.xlabel(r'$t$ (s)', fontsize=12)
plt.ylabel(r'Amplitude', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Damped sine wave with LaTeX formula in title and legend.
```

### Example 3: Legend customization
```python
x = np.linspace(0, 10, 50)
plt.plot(x, np.sin(x), label='sin', color='#E74C3C', linewidth=2)
plt.plot(x, np.cos(x), label='cos', color='#3498DB', linewidth=2)
plt.plot(x, np.sin(x) + np.cos(x), label='sum', color='#2ECC71', linewidth=2)

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1),  # outside axes
           title='Functions', title_fontsize=12,
           frameon=True, shadow=True, fancybox=True,
           ncol=1, fontsize=10)
plt.title('Legend Outside the Axes')
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # make room for legend
plt.show()
```
```
# Output: Plot with legend placed to the right of the axes, with title and shadow frame.
```

### Example 4: Named, hex, and RGB colors
```python
x = np.linspace(0, 10, 50)

plt.plot(x, x, color='tomato', linewidth=2, label='named color')
plt.plot(x, x**2, color='#2E86C1', linewidth=2, label='hex color')
plt.plot(x, x**3, color=(0.2, 0.6, 0.2), linewidth=2, label='RGB tuple')
plt.plot(x, x**4, color='C3', linewidth=2, label='cycle color C3')

plt.legend()
plt.title('Different Color Specifications')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Four lines using named (tomato), hex (#2E86C1), RGB (0.2,0.6,0.2), and cycle (C3) colors.
```

### Example 5: Colormaps
```python
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(Z, cmap='viridis', aspect='auto', origin='lower')
axes[0].set_title('Sequential: viridis')
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(Z, cmap='RdBu_r', aspect='auto', origin='lower')
axes[1].set_title('Diverging: RdBu_r')
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(Z, cmap='twilight', aspect='auto', origin='lower')
axes[2].set_title('Cyclic: twilight')
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
```
```
# Output: Three heatmaps showing sequential (viridis), diverging (RdBu_r), and cyclic (twilight) colormaps.
```

### Example 6: rcParams for global customization
```python
print("Default figure size:", plt.rcParams['figure.figsize'])
print("Default font size:", plt.rcParams['font.size'])

# Change defaults globally
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title('Plot with Custom rcParams')
plt.show()

# Reset to defaults
plt.rcParams.update(plt.rcParamsDefault)
```
```
# Output: Plot with custom defaults applied globally — larger figure, bigger font, grid on, spines removed.
```

### Example 7: Stylesheets (plt.style.use)
```python
styles = ['default', 'seaborn-v0_8', 'ggplot', 'bmh', 'fivethirtyeight', 'dark_background']
x = np.linspace(0, 10, 100)
y = np.sin(x)

for i, style in enumerate(styles[:3]):
    plt.style.use(style)
    plt.figure(figsize=(8, 3))
    plt.plot(x, y)
    plt.title(f'Style: {style}')
    plt.tight_layout()
    plt.show()
    print(f"Applied style: {style}")

# Restore default at the end
plt.style.use('default')
```
```
# Output: Three figures showing the same data in default, seaborn, and ggplot styles.
```

### Example 8: Annotations with arrows and text boxes
```python
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
peak_x = np.pi / 2
peak_y = 1

plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(peak_x, color='green', linestyle='--', alpha=0.5, label='peak x')

# Basic annotation with arrow
plt.annotate('Maximum',
             xy=(peak_x, peak_y),
             xytext=(peak_x + 1, peak_y + 0.5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=2,
                            headwidth=8, headlength=10),
             fontsize=12, fontweight='bold', color='red')

# Text box annotation
plt.text(4.5, -0.5, 'Note: Sine wave\nwith annotated peak',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
         fontsize=10)

plt.title('Annotated Sine Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
```
# Output: Sine wave with a green dashed line at π/2, a red arrow annotation at the peak, and a yellow text box.
```

### Example 9: Reference lines with axvline and axhline
```python
np.random.seed(42)
data = np.random.randn(100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Histogram with reference lines
ax1.hist(data, bins=20, color='#3498DB', edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(data):.2f}')
ax1.axvline(np.median(data), color='green', linestyle=':', linewidth=2, label=f'Median={np.median(data):.2f}')
ax1.axvline(np.mean(data) + np.std(data), color='orange', linestyle='-.', linewidth=2, label='+1 Std')
ax1.axvline(np.mean(data) - np.std(data), color='orange', linestyle='-.', linewidth=2, label='-1 Std')
ax1.legend()
ax1.set_title('Histogram with Reference Lines')

# Scatter with horizontal reference
ax2.scatter(range(len(data)), data, alpha=0.5)
ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
ax2.axhline(1.96, color='red', linestyle='--', linewidth=1, label='95% CI upper')
ax2.axhline(-1.96, color='red', linestyle='--', linewidth=1, label='95% CI lower')
ax2.legend()
ax2.set_title('Scatter with CI Reference Lines')

plt.tight_layout()
plt.show()
```
```
# Output: Left — histogram with mean, median, and ±1 std lines. Right — scatter with zero line and 95% CI bounds.
```

### Example 10: Creating a custom `.mplstyle` file programmatically
```python
import matplotlib as mpl

# Define custom style as a dict
custom_style = {
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.shadow': True,
}

# Apply custom style
mpl.rcParams.update(custom_style)

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), label='sin')
plt.plot(x, np.cos(x), label='cos')
plt.title('Custom Style Applied')
plt.legend()
plt.show()

# Reset
mpl.rcParams.update(mpl.rcParamsDefault)
```
```
# Output: Plot with serif font, 150 DPI, grid on, and styled legend — all set via rcParams dict.
```

### Example 11: Spines and tick customization
```python
fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(2*x), linewidth=2)

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

# Customize ticks
ax.tick_params(axis='both', which='major', labelsize=10,
               length=6, width=1.5, direction='inout')
ax.tick_params(axis='both', which='minor', length=3, width=1)

# Add minor ticks
ax.set_xticks(np.arange(0, 11, 2))
ax.set_xticks(np.arange(1, 10, 2), minor=True)
ax.set_yticks(np.arange(-1, 1.5, 0.5))
ax.set_yticks(np.arange(-0.75, 1, 0.5), minor=True)

ax.set_title('Custom Spines and Tick Marks')
ax.grid(True, which='both', alpha=0.3)
plt.show()
```
```
# Output: Plot with only left/bottom spines (offset outward), customized tick direction and minor ticks.
```

## Common Mistakes

1. **Using `usetex=True` without LaTeX installed.** This causes a runtime error. Instead, use Matplotlib's built-in math parser with `$...$` which handles most common expressions.
2. **Overriding `rcParams` in a library or importable module.** This affects all users of the module. Use `plt.style.context('style')` for temporary style changes.
3. **Legends placed behind data.** If the legend is inside the plot, set `legend(framealpha=1)` or use `zorder` to ensure the legend is on top.
4. **Forgetting to escape underscores in LaTeX.** `$x_1$` works but in regular text, underscores trigger subscript. Use raw strings `r'x\_1'` for regular text with underscores.
5. **Using colorblind-unfriendly colormaps.** Avoid jet/rainbow. Use viridis, plasma, or cividis for sequential; RdBu or coolwarm for diverging.
6. **Too many colors.** Humans distinguish ~8-10 categories by color. Beyond that, use patterns, markers, and shapes.
7. **Hardcoding colors in loops.** Use `plt.cm.tab10(i)` or `plt.rcParams['axes.prop_cycle'].by_key()['color']` to cycle through colors programmatically.

## Interview Questions

### Beginner - 5

1. **Q:** How do you add LaTeX math to a plot title?  
   **A:** Use raw strings with `$...$`: `plt.title(r'$y = \sin(x)$')`. The `r` prefix prevents backslash escaping issues.

2. **Q:** What's the difference between a named color like `'red'` and a hex color like `'#FF0000'`?  
   **A:** Both specify the same color. Named colors (140 CSS names) are convenient; hex colors give precise control. Matplotlib accepts both.

3. **Q:** How do you place a legend outside the plot area?  
   **A:** Use `bbox_to_anchor=(x, y)` with `loc='upper left'`. For example: `plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))`.

4. **Q:** What is `plt.style.use('ggplot')`?  
   **A:** It applies the ggplot-themed style sheet, making Matplotlib plots resemble R's ggplot2 library — gray background, white grid, specific color palette.

5. **Q:** How would you add a vertical line at x=5 to an existing plot?  
   **A:** Call `plt.axvline(x=5, color='red', linestyle='--', linewidth=2)` or `ax.axvline(x=5)` on the specific Axes.

### Intermediate - 5

1. **Q:** How does `plt.rcParams` work and how would you save a custom configuration?  
   **A:** `plt.rcParams` is a dict-like object holding all default parameters. Modify it directly or use `plt.rc()` / `plt.rcParams.update()`. Save as a `.mplstyle` file using `plt.style.use('path/to/style.mplstyle')`.

2. **Q:** What are the three main categories of colormaps and when should each be used?  
   **A:** Sequential (viridis, Blues) — ordered data from low to high. Diverging (RdBu, coolwarm) — data with a meaningful midpoint (e.g., correlation -1 to 1). Qualitative (Set1, tab10) — categorical data without order.

3. **Q:** How do you annotate a point with an arrow?  
   **A:** Use `plt.annotate('text', xy=(x_data, y_data), xytext=(x_offset, y_offset), arrowprops=dict(arrowstyle='->'))`.

4. **Q:** Explain the difference between `axvline` and `axvspan`.  
   **A:** `axvline(x=5)` draws a vertical line at position x=5 spanning the full y-range. `axvspan(xmin=4, xmax=6)` shades the region between x=4 and x=6.

5. **Q:** How can you create a custom color cycle for line plots?  
   **A:** Set `plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#E74C3C', '#3498DB', '#2ECC71'])` or create a custom `Cycler` object.

### Advanced - 3

1. **Q:** How would you create a custom colormap from a list of colors?  
   **A:** Use `from matplotlib.colors import LinearSegmentedColormap; cmap = LinearSegmentedColormap.from_list('mycmap', ['red', 'blue', 'green'])`.

2. **Q:** Explain the `transform` parameter in `annotate` and `text`. What is the difference between `'axes fraction'`, `'data'`, and `'figure fraction'`?  
   **A:** `transform=ax.transData` (default) uses data coordinates; `transform=ax.transAxes` uses fraction of axes (0=left/bottom, 1=right/top); `transform=fig.transFigure` uses fraction of figure. Essential for positioning annotations relative to the figure rather than data.

3. **Q:** How would you build a reusable `mplstyle` file and bundle it with a package?  
   **A:** Create a `stylelib/` directory inside the package with `.mplstyle` files. Users apply with `plt.style.use('package.stylelib.mystyle')`. Register with `plt.style.core.USER_LIBRARY_PATHS` or use `entry_points` in `setup.py`.

## Practice Problems

### Easy - 5

1. **E1:** Create a plot with the title set to `r'$E = mc^2$'` using LaTeX.
2. **E2:** Plot a line with hex color `#E67E22` and a dashed style.
3. **E3:** Add a legend to a plot and place it in the lower right corner.
4. **E4:** Use `plt.style.use('dark_background')` and create a simple line plot.
5. **E5:** Add a horizontal line at y=0 to an existing plot.

### Medium - 5

1. **M1:** Create a plot with a legend placed outside the axes on the right side.
2. **M2:** Use `plt.annotate()` to label the global maximum of y = -x² + 4x on [0, 4].
3. **M3:** Create a 2×2 subplot grid where each subplot uses a different colormap for its scatter points.
4. **M4:** Use `rcParams` to set all future plots to have: figsize=(12, 5), font.size=14, and grid on.
5. **M5:** Create a plot with custom spines (only left and bottom visible, offset outward by 10 points).

### Hard - 3

1. **H1:** Create a custom diverging colormap centered at white for values from -1 to 1 and apply it to a heatmap.
2. **H2:** Build a reusable `.mplstyle` file and apply it to a complex multi-panel figure.
3. **H3:** Create a plot annotation that uses a `FancyBboxPatch` with a custom path effect (shadow, glow).

## Solutions

### E1 Solution
```python
plt.plot([1, 2, 3], [1, 4, 9])
plt.title(r'$E = mc^2$')
plt.show()
```

### E2 Solution
```python
plt.plot([1, 2, 3], [4, 5, 6], color='#E67E22', linestyle='--')
plt.show()
```

### E3-E5 Solutions follow patterns in examples.

### M1-M5 Solutions follow the techniques shown in examples above.

### H1-H3 Solutions involve advanced colormap creation and are beyond brief code blocks.

## Related Concepts

- 086 — Matplotlib Basics (foundation)
- 087 — Line and Scatter (marker/line customization)
- 088 — Bar, Histogram, Box Plots (coloring statistical plots)
- 091 — Subplots (arranging customized plots)

## Next Concepts

- 089 — Seaborn (built-in styling via set_theme)
- 090 — Plotly (template system and layout customization)
- 093 — sklearn Basics (applying visualization to ML pipelines)

## Summary

Customizing Matplotlib plots involves LaTeX labels, flexible legends, precise color specification, global `rcParams` configuration, pre-built stylesheets, annotations with arrows, and reference lines. These techniques transform default plots into publication-ready, accessible, and brand-consistent visualizations.

## Key Takeaways

- Use `r'$...$'` for LaTeX math expressions in labels and titles
- Place legends outside with `bbox_to_anchor=(x, y)`
- Use perceptually uniform colormaps (viridis, plasma) over jet/rainbow
- Set global defaults with `plt.rcParams.update()` for consistent multi-plot output
- `plt.style.use('style_name')` instantly applies a predefined aesthetic
- Annotate key features with `plt.annotate()` with arrow properties
- Use `axvline`/`axhline` for reference lines; `axvspan`/`axhspan` for shaded regions
