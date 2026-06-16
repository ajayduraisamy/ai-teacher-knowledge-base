# Concept: Plotly for Interactive Visualization

## Concept ID

PYT-090

## Difficulty

Intermediate

## Domain

Python

## Module

Visualization

## Learning Objectives

- Create interactive scatter, line, bar, histogram, and image plots using Plotly Express
- Understand the difference between Plotly Express (high-level) and Plotly Graph Objects (low-level)
- Build interactive dashboards with hover tooltips, zoom, and pan capabilities
- Apply Plotly for interactive ML model output exploration

## Prerequisites

- PYT-086 — Matplotlib Basics
- Pandas DataFrames
- Basic understanding of HTML/web rendering (helpful but not required)

## Definition

Plotly is an interactive, web-based visualization library for Python. It produces HTML/JavaScript-based charts that support zoom, pan, hover, click events, and animation — all without any web programming. Plotly offers two main APIs:

1. **Plotly Express (`px`):** High-level, concise API similar to Seaborn. One function call creates a fully interactive chart with sensible defaults. Best for exploratory analysis and rapid prototyping.
2. **Plotly Graph Objects (`go`):** Low-level, explicit API that gives full control over every chart element. Best for complex custom layouts, subplots, and production dashboards.

Plotly charts are rendered as HTML `<div>` elements that can be displayed in Jupyter notebooks, saved as standalone HTML files, or embedded in web applications (Dash, Flask, Django).

Key features:
- **Interactivity by default:** Hover tooltips, zoom with mouse wheel, pan, box/lasso select, auto-scale
- **Export formats:** HTML, PNG, SVG, PDF
- **Animation:** Frame-based animation for time series
- **Dashboards:** Integrates with Dash for building full web applications
- **3D charts:** Built-in 3D scatter, surface, and mesh plots
- **Maps:** Built-in support for Mapbox, GeoJSON, and choropleth maps

## Intuition

Think of Plotly as "visualization that behaves like a web app." When you hover over a point in a Matplotlib chart, nothing happens. In Plotly, a tooltip appears with the exact data values. You can zoom into a dense region, pan across time, or click a legend item to toggle a trace on/off.

Plotly Express abstracts away all the details: you say "I want a scatter plot of `x` vs `y` colored by `z`" and it creates a complete interactive chart. Graph Objects gives you lower-level control: you explicitly create `Scatter`, `Layout`, and `Figure` objects, useful when you need fine-grained customization.

## Why This Concept Matters

- **Data Exploration:** Interactive zoom and hover make it dramatically easier to explore complex datasets
- **Client-Facing Dashboards:** Plotly charts embedded in Dash apps are the industry standard for Python-based BI tools
- **ML Model Debugging:** Hover over predictions to see input features; zoom into error regions; toggle between model outputs
- **Presentation:** Interactive charts engage audiences far more than static images
- **Reproducible Reports:** Save standalone HTML files that anyone can open in a browser without Python

## Real World Examples

1. **Financial Dashboard:** A Plotly candlestick chart with 5-year stock data, overlay of moving averages, and volume bars — all zoomable to any time range.
2. **Geospatial Analytics:** A Mapbox scatter plot of 10,000 delivery locations colored by delivery time, with hover showing address and status.
3. **ML Model Comparison:** A bar chart comparing accuracy, precision, recall, and F1 for 8 models, with hover showing confidence intervals and sample sizes.
4. **Animated Time Series:** An animated scatter plot showing COVID-19 cases vs deaths per country over time, with a slider to control the date.
5. **3D Hyperparameter Search:** A 3D scatter plot of learning rate vs batch size vs validation accuracy, colored by accuracy — interactive rotation reveals the optimal region.

## AI/ML Relevance

- **Exploration of Model Outputs:** Scatter plots of predictions vs actual with hover showing feature values (identify failure modes)
- **SHAP/Feature Importance:** Interactive bar charts of feature importance values
- **Hyperparameter Tuning:** 3D surface plots of accuracy over hyperparameter space
- **Animated Training:** Plot training curves that animate as epochs progress
- **Confusion Matrix:** Interactive heatmaps where clicking a cell shows misclassified examples

## Code Examples

### Example 1: Basic scatter plot with Plotly Express
```python
import plotly.express as px
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.uniform(5, 30, 100)
})

fig = px.scatter(df, x='x', y='y', color='category',
                 size='size', hover_data=['category'],
                 title='Interactive Scatter Plot')
fig.show()
```
```
# Output: Interactive scatter plot with colored categories, variable marker sizes, hover tooltips.
```

### Example 2: Line plot with Plotly Express
```python
import plotly.express as px
import numpy as np
import pandas as pd

t = np.linspace(0, 10, 200)
df = pd.DataFrame({
    'time': np.concatenate([t, t]),
    'value': np.concatenate([np.sin(t), np.cos(t)]),
    'series': ['sin'] * 200 + ['cos'] * 200
})

fig = px.line(df, x='time', y='value', color='series',
              title='Trigonometric Functions',
              labels={'time': 'Time (s)', 'value': 'Amplitude'})
fig.show()
```
```
# Output: Interactive line chart with togglable legend, zoom, and hover showing exact coordinates.
```

### Example 3: Bar chart with Plotly Express
```python
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'sales': [450, 320, 780, 540, 290],
    'region': ['North', 'South', 'North', 'East', 'West']
})

fig = px.bar(df, x='product', y='sales', color='region',
             title='Sales by Product (colored by region)',
             text_auto='.0f')
fig.update_traces(textposition='outside')
fig.show()
```
```
# Output: Interactive bar chart with colored bars, value annotations, and hover details.
```

### Example 4: Histogram with multiple distributions
```python
df = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(2, 1.5, 500),
        np.random.normal(-1, 0.8, 500)
    ]),
    'group': ['A']*500 + ['B']*500 + ['C']*500
})

fig = px.histogram(df, x='value', color='group',
                   marginal='box',  # add box plot above
                   title='Distribution Comparison',
                   barmode='overlay',
                   opacity=0.7)
fig.show()
```
```
# Output: Overlaid interactive histograms for three groups, with box plot marginals.
```

### Example 5: Heatmap / image display with px.imshow
```python
import plotly.express as px
import numpy as np

# Confusion matrix-like data
cm = np.array([[95, 3, 2],
               [5, 88, 7],
               [0, 9, 91]])
labels = ['Class A', 'Class B', 'Class C']

fig = px.imshow(cm,
                x=labels, y=labels,
                color_continuous_scale='Blues',
                title='Confusion Matrix',
                text_auto=True,
                aspect='auto')
fig.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
fig.show()
```
```
# Output: Interactive heatmap with numeric annotations, color scale, and hover cell values.
```

### Example 6: Using Plotly Graph Objects for custom control
```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers',
                         name='sin', line=dict(color='red', width=2),
                         marker=dict(symbol='circle', size=6)))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines',
                         name='cos', line=dict(color='blue', width=2, dash='dash')))

fig.update_layout(
    title='Graph Objects Example',
    xaxis_title='X',
    yaxis_title='Y',
    legend=dict(x=0.8, y=1),
    template='plotly_white',
    hovermode='x unified'
)
fig.show()
```
```
# Output: Customized chart with explicit trace definitions, dashed lines, and unified hover.
```

### Example 7: 3D Scatter Plot
```python
df = pd.DataFrame({
    'x': np.random.randn(200),
    'y': np.random.randn(200),
    'z': np.random.randn(200),
    'label': np.random.choice(['Group1', 'Group2', 'Group3'], 200)
})

fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                    title='3D Scatter Plot (interactive rotation)',
                    opacity=0.7)
fig.show()
```
```
# Output: 3D scatter plot that can be rotated, zoomed, and panned with mouse interactions.
```

### Example 8: Faceted plots (small multiples)
```python
tips = px.data.tips()

fig = px.scatter(tips, x='total_bill', y='tip',
                 color='sex', facet_col='day',
                 facet_col_wrap=4,
                 title='Tips by Day and Sex',
                 trendline='ols')
fig.show()
```
```
# Output: Four scatter plots (one per day) with OLS trend lines, colored by sex.
```

### Example 9: Animated plot
```python
gapminder = px.data.gapminder()

fig = px.scatter(gapminder, x='gdpPercap', y='lifeExp',
                 size='pop', color='continent',
                 hover_name='country',
                 log_x=True, size_max=60,
                 animation_frame='year',
                 title='Global Development Over Time',
                 range_x=[200, 100000], range_y=[25, 90])
fig.show()
```
```
# Output: Animated bubble chart showing country development from 1952-2007, with play/pause slider.
```

### Example 10: Saving and exporting
```python
fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title='Export Demo')

# Save as standalone HTML
fig.write_html('plot.html')
print("Saved as HTML — opens in any browser, interactivity preserved.")

# Save as static image (requires kaleido or orca)
fig.write_image('plot.png', width=800, height=500)
print("Saved as PNG.")

# Save as JSON for later loading
fig.write_json('plot.json')
print("Saved as JSON.")
```
```
# Output:
# Saved as HTML — opens in any browser, interactivity preserved.
# Saved as PNG.
# Saved as JSON.
```

## Common Mistakes

1. **Forgetting that `fig.show()` only works in certain environments.** In Jupyter, it renders inline. In a script, it opens a browser tab. In headless servers, neither works — use `fig.write_html()` or `fig.write_image()` instead.
2. **Calling `plt.show()` after `fig.show()` in the same cell.** This creates a blank Matplotlib figure. Don't mix Plotly and Matplotlib show calls in the same output cell.
3. **Not using `update_layout()` for global settings.** Trace-specific settings go in the trace constructor; figure-wide settings (title, axis labels, template, legend position) go in `update_layout()`.
4. **Using Plotly Express for very large datasets (>100k points).** Plotly renders all data points to HTML, which becomes slow. Use `sample()` to downsample, or use Datashader + Plotly for big data.
5. **Overloading hover_data.** Too many hover fields create a cluttered tooltip. Limit to 3–5 most important fields.
6. **Forgetting to install `kaleido` for static image export.** `fig.write_image()` requires the kaleido engine: `pip install kaleido`.
7. **Ignoring the `template` parameter.** Plotly's default template has a gray background that looks dated. Use `template='simple_white'`, `'plotly_white'`, or `'plotly_dark'` for a cleaner look.

## Interview Questions

### Beginner - 5

1. **Q:** What is Plotly Express?  
   **A:** Plotly Express (`px`) is a high-level Python API for creating interactive Plotly charts with minimal code. It's similar to Seaborn but produces interactive web-based visualizations.

2. **Q:** How is Plotly different from Matplotlib?  
   **A:** Plotly produces interactive HTML/JavaScript charts (zoom, pan, hover, click). Matplotlib produces static images (PNG, PDF, SVG). Plotly has a different API and is better suited for web dashboards.

3. **Q:** How do you create a scatter plot in Plotly Express?  
   **A:** `px.scatter(data_frame=df, x='col1', y='col2', color='col3')`. It automatically adds hover, zoom, and a legend.

4. **Q:** How do you save a Plotly figure as an interactive HTML file?  
   **A:** `fig.write_html('filename.html')`. The file can be opened in any web browser without Python.

5. **Q:** What is the difference between `px` and `go`?  
   **A:** `px` (Plotly Express) is high-level and concise. `go` (Graph Objects) is low-level and explicit, giving full control over every trace and layout element.

### Intermediate - 5

1. **Q:** How do you create subplots in Plotly?  
   **A:** Use `plotly.subplots.make_subplots()`: `from plotly.subplots import make_subplots; fig = make_subplots(rows=2, cols=2)`. Then add traces with `fig.add_trace(go.Scatter(...), row=1, col=1)`.

2. **Q:** What is the `update_layout()` method used for?  
   **A:** It modifies figure-wide properties: title, axis titles, legend position, template, width/height, margins, hovermode, and more.

3. **Q:** How do you add an OLS trendline to a Plotly Express scatter plot?  
   **A:** Pass `trendline='ols'` to `px.scatter()`. For more complex models, use `trendline_options=dict(log_x=True)`.

4. **Q:** How does Plotly handle animations?  
   **A:** Pass `animation_frame='column_name'` to a Plotly Express function. The column should contain discrete time steps. Plotly interpolates between frames and provides a play/pause slider.

5. **Q:** What is the `template` parameter and what are common options?  
   **A:** `template` sets the overall chart style. Common options: `'plotly'` (default), `'plotly_white'`, `'plotly_dark'`, `'simple_white'`, `'seaborn'`, `'ggplot2'`.

### Advanced - 3

1. **Q:** Explain the Plotly figure schema (Figure → data → traces, layout, config).  
   **A:** A `Figure` is a dict-like object with three top-level keys: `data` (list of `Trace` objects — Scatter, Bar, Heatmap, etc.), `layout` (Layout object with plot-wide settings), and `config` (configuration for the interactive mode bar, scroll zoom, etc.).

2. **Q:** How would you create a custom callback in a Dash app that updates a Plotly figure based on dropdown selection?  
   **A:** Define a `@app.callback(Output('fig', 'figure'), Input('dropdown', 'value'))` that filters data and returns a new `px` figure or a `go.Figure` with updated traces.

3. **Q:** How do you handle large datasets (>1M points) efficiently in Plotly?  
   **A:** Options: (a) Downsample using `df.sample(n=100000)`, (b) use `scattergl` trace (WebGL accelerated via `go.Scattergl`), (c) use Datashader to rasterize then overlay as an image on a Plotly map, (d) aggregate into bins (hexbin or 2D histogram) before plotting.

## Practice Problems

### Easy - 5

1. **E1:** Create an interactive scatter plot of `iris` dataset: sepal_length vs sepal_width, colored by species.
2. **E2:** Create a line plot showing the `gapminder` life expectancy over time for 5 countries.
3. **E3:** Create a bar chart of average tip by day from the `tips` dataset (`px.data.tips()`).
4. **E4:** Create a histogram of `total_bill` from `tips` with 30 bins.
5. **E5:** Create a heatmap (imshow) of a random 5×5 matrix and save it as HTML.

### Medium - 5

1. **M1:** Create a 3D scatter plot of the `iris` dataset using all four numeric features.
2. **M2:** Create a line plot with markers for `sin(x)` and `cos(x)` using Graph Objects.
3. **M3:** Create a faceted scatter plot of `tips`: total_bill vs tip, faceted by time (Lunch/Dinner), colored by sex.
4. **M4:** Build an animated scatter plot of the `gapminder` dataset with continent as color, population as size.
5. **M5:** Create a dashboard-style figure with 2 subplots: a scatter plot and a histogram of the same data.

### Hard - 3

1. **H1:** Use `make_subplots` with `specs` to create a figure with a 3D scatter on the left and a 2D projection on the right.
2. **H2:** Build a custom `go.Figure` with multiple traces, secondary y-axes (`yaxis2`), and linked brushing.
3. **H3:** Create an interactive Dash app with two linked plots: selecting points in a scatter plot highlights corresponding bars in a bar chart.

## Solutions

### E1 Solution
```python
iris = px.data.iris()
fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species')
fig.show()
```

### E2 Solution
```python
gapminder = px.data.gapminder()
countries = gapminder[gapminder['country'].isin(['India', 'China', 'USA', 'UK', 'Japan'])]
fig = px.line(countries, x='year', y='lifeExp', color='country')
fig.show()
```

### E3 Solution
```python
tips = px.data.tips()
fig = px.bar(tips, x='day', y='tip', title='Average Tip by Day')
fig.show()
```

### E4 Solution
```python
fig = px.histogram(tips, x='total_bill', nbins=30)
fig.show()
```

### E5 Solution
```python
import numpy as np
fig = px.imshow(np.random.rand(5, 5), text_auto=True)
fig.write_html('random_heatmap.html')
```

### M1 Solution
```python
iris = px.data.iris()
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_length',
                    color='species', size='petal_width')
fig.show()
```

### M2-M5 Solutions follow the patterns shown in the code examples.

## Related Concepts

- 086 — Matplotlib Basics (static plotting counterpart)
- 089 — Seaborn (high-level statistical counterpart to Plotly Express)
- 091 — Subplots (make_subplots, faceting)
- 092 — Customizing Plots (similar concepts in Plotly's update_layout)

## Next Concepts

- 093 — sklearn Basics (ML pipeline with interactive visualization)
- 095 — Model Evaluation (confusion matrices, ROC curves in Plotly)
- 100 — Project Structure (building Dash apps)

## Summary

Plotly is an interactive, web-based visualization library with two APIs: Plotly Express (high-level, concise) and Graph Objects (low-level, explicit). It produces HTML/JavaScript charts with built-in zoom, pan, hover, and animation. Plotly is the industry standard for interactive data exploration, client-facing dashboards (via Dash), and ML model output analysis where point-by-point inspection is valuable.

## Key Takeaways

- Use Plotly Express for rapid interactive exploration; use Graph Objects for fine-grained control
- Hover, zoom, and pan are free — no configuration needed
- `fig.write_html('file.html')` creates a portable interactive report
- Use `template='plotly_white'` for cleaner default styling
- For datasets >100K points, downsample or use `scattergl`
- Plotly + Dash = full-stack data apps without JavaScript
