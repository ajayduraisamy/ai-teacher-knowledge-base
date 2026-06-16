# Concept: Time Series Analysis

## Concept ID

PYT-084

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Create and manipulate DatetimeIndex objects
- Convert strings to datetime with `pd.to_datetime()`
- Generate date ranges with `pd.date_range()`
- Resample time series data (downsampling and upsampling)
- Compute rolling window statistics with `rolling()`
- Create lagged features using `shift()` and `diff()`
- Use the `.dt` accessor for datetime component extraction

## Prerequisites

- Series and DataFrame fundamentals (PYT-076, PYT-077)
- Indexing with `.loc` (PYT-079)
- GroupBy operations (PYT-081) — resample is a time-based groupby
- Basic understanding of datetime concepts (time zones, frequencies)

## Definition

**Time series analysis** in Pandas refers to working with data indexed by dates, times, or timestamps. Pandas provides specialized tools for:

- **Parsing**: Converting strings/dates to `datetime64` dtype
- **Indexing**: Using DatetimeIndex for label-based selection
- **Resampling**: Changing the frequency of observations (downsampling = aggregating to lower frequency; upsampling = filling to higher frequency)
- **Windowing**: Computing rolling or expanding statistics
- **Shifting**: Creating lagged or leading values
- **Accessors**: Extracting year, month, day, etc. with `.dt`

```python
# Key functions
ts = pd.to_datetime('2024-01-01')
rng = pd.date_range('2024-01-01', periods=100, freq='D')
df.set_index('date', inplace=True)
df.resample('M').mean()
df['value'].rolling(window=7).mean()
df['lag_1'] = df['value'].shift(1)
df['change'] = df['value'].diff()
df['year'] = df.index.year
```

## Intuition

Think of time series operations along a spectrum:

- **`to_datetime`**: Teaching Pandas what strings represent dates.
- **`date_range`**: Creating a calendar of timestamps.
- **`resample`**: A time-aware GroupBy — group by weeks/months/years and aggregate.
- **`rolling`**: A sliding window that computes statistics over N consecutive observations.
- **`shift`**: Moving values forward/backward in time to create lag features.
- **`diff`**: Computing the change between consecutive observations.
- **`.dt`**: Extracting calendar components from datetime columns.

## Why This Concept Matters

Time series data is everywhere: stock prices, sensor readings, web traffic, sales, weather data. Pandas' time series capabilities are among the most mature in the Python ecosystem. Mastering them enables:

- Financial analysis (returns, moving averages, volatility)
- Demand forecasting (aggregation to weekly/monthly)
- Anomaly detection (rolling statistics, deviation from moving average)
- Feature engineering for ML models (lags, rolling features, time-based aggregates)

## Real World Examples

1. **Stock Market**: Daily closing prices, 20-day moving average, daily returns.
2. **IoT Sensors**: Temperature readings every 5 minutes, resampled to hourly averages.
3. **Web Traffic**: Daily page views, 7-day rolling average, week-over-week change.
4. **Sales Forecasting**: Monthly sales, lagged features for predicting next month.
5. **Energy Consumption**: Smart meter readings at 15-minute intervals, resampled to daily totals.

## AI/ML Relevance

- **Forecasting**: Feature engineering for ARIMA, Prophet, LSTMs, Transformers.
- **Feature Engineering**: Lag features (shift), rolling statistics (rolling), time-based aggregates (resample).
- **Train/Test Split**: Time-based split (never shuffle time series).
- **Evaluation**: Time series cross-validation (expanding window, sliding window).
- **Anomaly Detection**: Rolling mean and std to identify outliers.

## Code Examples

### Example 1: `pd.to_datetime()` — parsing dates

```python
import pandas as pd
import numpy as np

print(pd.to_datetime('2024-01-15'))
# Output: 2024-01-15 00:00:00

dates = pd.Series(['2024-01-01', '2024-02-01', '2024-03-01'])
parsed = pd.to_datetime(dates)
print(parsed)
# Output:
# 0   2024-01-01
# 1   2024-02-01
# 2   2024-03-01
# dtype: datetime64[ns]

print(pd.to_datetime('01/15/2024', format='%m/%d/%Y'))
# Output: 2024-01-15 00:00:00

print(pd.to_datetime(['2024-01-01', 'invalid', '2024-03-01'], errors='coerce'))
# Output:
# DatetimeIndex(['2024-01-01', 'NaT', '2024-03-01'], dtype='datetime64[ns]', freq=None)
```

### Example 2: `pd.date_range()` — generating date ranges

```python
rng = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
print(rng)
# Output:
# DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
#                '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08',
#                '2024-01-09', '2024-01-10'],
#               dtype='datetime64[ns]', freq='D')

bng = pd.date_range(start='2024-01-01', periods=5, freq='B')
print(bng)
# Output:
# DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
#                '2024-01-05'],
#               dtype='datetime64[ns]', freq='B')

hr = pd.date_range('2024-01-01', periods=4, freq='h')
print(hr)
# Output:
# DatetimeIndex(['2024-01-01 00:00:00', '2024-01-01 01:00:00',
#                '2024-01-01 02:00:00', '2024-01-01 03:00:00'],
#               dtype='datetime64[ns]', freq='h')

me = pd.date_range('2024-01-01', periods=3, freq='M')
print(me)
# Output:
# DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'], dtype='datetime64[ns]', freq='M')
```

### Example 3: Setting DatetimeIndex and selecting by date

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10, freq='D'),
    'value': np.random.randn(10)
})
df.set_index('date', inplace=True)
print(df.head(3))
# Output:
#                 value
# date
# 2024-01-01  0.123456
# 2024-01-02 -0.789012
# 2024-01-03  1.234567

# Select by date label
print(df.loc['2024-01-03':'2024-01-05'])
# Output:
#                 value
# date
# 2024-01-03  1.234567
# 2024-01-04 -0.345678
# 2024-01-05  0.901234

# Partial string indexing
print(df.loc['2024-01'])
# Output:
#                 value
# date
# 2024-01-01  0.123456
# 2024-01-02 -0.789012
# ...
```

### Example 4: `resample()` — downsampling and upsampling

```python
# Create 30 days of hourly data
dates = pd.date_range('2024-01-01', periods=30*24, freq='h')
ts = pd.Series(np.random.randn(len(dates)), index=dates)

# Downsample to daily mean
daily = ts.resample('D').mean()
print(daily.head(3))
# Output:
# 2024-01-01    0.012345
# 2024-01-02   -0.123456
# 2024-01-03    0.234567
# Freq: D, dtype: float64

# Downsample to weekly sum
weekly = ts.resample('W').sum()
print(weekly.head(3))
# Output:
# 2024-01-07    1.234567
# 2024-01-14   -0.567890
# 2024-01-21    0.890123
# Freq: W-SUN, dtype: float64

# Upsample from daily to hourly with forward fill
daily = pd.Series([100, 200, 300], index=pd.date_range('2024-01-01', periods=3, freq='D'))
hourly = daily.resample('h').ffill()
print(hourly.head(5))
# Output:
# 2024-01-01 00:00:00    100
# 2024-01-01 01:00:00    100
# 2024-01-01 02:00:00    100
# 2024-01-01 03:00:00    100
# 2024-01-01 04:00:00    100
# Freq: h, dtype: int64

# Custom aggregation
result = ts.resample('D').agg(['mean', 'std', 'min', 'max'])
print(result.head(2))
# Output:
#               mean       std       min      max
# 2024-01-01  0.0123  1.234567 -2.345678  2.34567
# 2024-01-02 -0.1234  0.987654 -2.123456  1.98765
```

### Example 5: `rolling()` — window statistics

```python
s = pd.Series(np.random.randn(20), index=pd.date_range('2024-01-01', periods=20, freq='D'))

# 7-day rolling mean
rolling_mean = s.rolling(window=7).mean()
result = pd.DataFrame({'original': s, 'rolling_mean': rolling_mean})
print(result.head(10))
# Output:
#             original  rolling_mean
# 2024-01-01  0.123456           NaN
# 2024-01-02 -0.789012           NaN
# 2024-01-03  1.234567           NaN
# 2024-01-04 -0.345678           NaN
# 2024-01-05  0.901234           NaN
# 2024-01-06 -0.567890           NaN
# 2024-01-07  0.123456      0.097155
# 2024-01-08 -0.234567    0.045873
# 2024-01-09  1.345678    0.351114
# 2024-01-10 -0.123456    0.157111

# Rolling with min_periods
roll = s.rolling(window=7, min_periods=3).mean()
print(roll.head(7))
# Output:
# 2024-01-01         NaN
# 2024-01-02         NaN
# 2024-01-03    0.189670
# 2024-01-04    0.055083
# 2024-01-05    0.224913
# 2024-01-06    0.092779
# 2024-01-07    0.097155
# Freq: D, dtype: float64

# Rolling with multiple functions
roll_stats = s.rolling(7).agg(['mean', 'std'])
print(roll_stats.head(8))
# Output:
#                mean       std
# 2024-01-01      NaN       NaN
# ...
# 2024-01-07  0.097155  0.803456
```

### Example 6: `shift()` and `diff()` — lags and changes

```python
s = pd.Series([100, 105, 103, 108, 112, 110],
              index=pd.date_range('2024-01-01', periods=6, freq='D'))

# Lag features
df = pd.DataFrame({'value': s})
df['lag_1'] = s.shift(1)
df['lag_2'] = s.shift(2)
df['lead_1'] = s.shift(-1)
print(df)
# Output:
#            value  lag_1  lag_2  lead_1
# 2024-01-01    100    NaN    NaN   105.0
# 2024-01-02    105  100.0    NaN   103.0
# 2024-01-03    103  105.0  100.0   108.0
# 2024-01-04    108  103.0  105.0   112.0
# 2024-01-05    112  108.0  103.0   110.0
# 2024-01-06    110  112.0  108.0     NaN

# Difference (first-order)
df['diff_1'] = s.diff(1)
df['pct_change'] = s.pct_change() * 100
print(df[['value', 'diff_1', 'pct_change']])
# Output:
#            value  diff_1  pct_change
# 2024-01-01    100     NaN         NaN
# 2024-01-02    105     5.0    5.000000
# 2024-01-03    103    -2.0   -1.904762
# 2024-01-04    108     5.0    4.854369
# 2024-01-05    112     4.0    3.703704
# 2024-01-06    110    -2.0   -1.785714
```

### Example 7: `.dt` accessor — datetime component extraction

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=5, freq='h')
})
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['quarter'] = df['timestamp'].dt.quarter
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
print(df)
# Output:
#            timestamp  year  month  day  hour  dayofweek  quarter  is_weekend
# 0 2024-01-01 00:00:00  2024      1    1     0          0        1       False
# 1 2024-01-01 01:00:00  2024      1    1     1          0        1       False
# 2 2024-01-01 02:00:00  2024      1    1     2          0        1       False
# 3 2024-01-01 03:00:00  2024      1    1     3          0        1       False
# 4 2024-01-01 04:00:00  2024      1    1     4          0        1       False

# Date-based filtering
df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
print(df[['timestamp', 'days_since_start']])
# Output:
#            timestamp  days_since_start
# 0 2024-01-01 00:00:00                 0
# 1 2024-01-01 01:00:00                 0
# 2 2024-01-01 02:00:00                 0
# 3 2024-01-01 03:00:00                 0
# 4 2024-01-01 04:00:00                 0
```

### Example 8: Time zone handling

```python
# Localize a naive timestamp
ts = pd.Timestamp('2024-01-01 12:00')
ts_ny = ts.tz_localize('America/New_York')
print(ts_ny)
# Output: 2024-01-01 12:00:00-05:00

# Convert to another time zone
ts_london = ts_ny.tz_convert('Europe/London')
print(ts_london)
# Output: 2024-01-01 17:00:00+00:00

# Create range in UTC
rng_utc = pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
print(rng_utc)
# Output:
# DatetimeIndex(['2024-01-01 00:00:00+00:00', '2024-01-02 00:00:00+00:00',
#                '2024-01-03 00:00:00+00:00'],
#               dtype='datetime64[ns, UTC]', freq='D')
```

## Common Mistakes

1. **Forgetting to set the datetime column as the index.** Many time series functions (`resample`, `rolling`) expect a DatetimeIndex.
2. **Using `rolling` without checking for NaN at the start.** The first `window - 1` values are NaN. Use `min_periods` to control this.
3. **`shift()` creating NaN for the first lagged value.** Plan for this by dropping or filling the first row.
4. **Confusing `resample('M')` (month end) with `resample('MS')` (month start).** 'M' aggregates to the last calendar day; 'MS' aggregates to the first.
5. **Applying `diff()` on non-stationary series.** Multiple `diff()` calls may be needed for stationarity.

## Interview Questions

### Beginner

1. How do you convert a string column to datetime in Pandas?
2. What is `pd.date_range()` used for?
3. How do you compute a 7-day rolling average?
4. What does `shift(1)` do?
5. How do you extract the month from a datetime column?

### Intermediate

1. What is the difference between `resample('M')` and `resample('MS')`?
2. How would you create lag features for the last 3 time steps?
3. What is the difference between `ffill` and `bfill` in `resample`?
4. How do you handle time zones in Pandas time series?
5. How can you compute a rolling z-score for anomaly detection?

### Advanced

1. Compare the performance of `rolling().apply()` with custom Cython/numba-accelerated rolling functions on a 10M-row dataset.
2. Explain how Pandas handles gaps in a DatetimeIndex during resampling and how to fill them correctly.
3. Implement a custom resampling function that handles irregular time intervals by doing volume-weighted average pricing for financial tick data.

## Practice Problems

### Easy

1. Create a date range from '2024-01-01' to '2024-12-31' with monthly frequency.
2. Parse a column of strings like '01/15/2024' into datetime.
3. Compute the 7-day rolling mean of a daily time series.
4. Create a lag-1 feature from a time series column.
5. Extract the year, month, and day of week from a datetime column.

### Medium

1. Given hourly temperature data, resample it to daily average and daily max.
2. Compute the 30-day rolling volatility (standard deviation of daily returns) for a stock price series.
3. Create features: lag-1, lag-7, 7-day rolling mean, and 7-day rolling std for a daily time series.
4. Upsample quarterly GDP data to monthly frequency using cubic interpolation.
5. Build a function that detects anomalies by marking points where the value deviates more than 3 standard deviations from the 30-day rolling mean.

### Hard

1. Implement a time series cross-validation splitter that preserves temporal order and creates expanding window train/test splits.
2. Build a feature engineering pipeline that automatically generates 50+ time series features (lags, rolling stats, expanding stats, fourier terms) from a single datetime column.
3. Design a streaming time series class that maintains rolling statistics incrementally without storing the full window (using exponential moving statistics).

## Solutions

```python
# E1
pd.date_range('2024-01-01', '2024-12-31', freq='M')

# E2
pd.to_datetime(df['date_col'], format='%m/%d/%Y')

# E3
s.rolling(7).mean()

# E4
s.shift(1)

# E5
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dow'] = df['date'].dt.dayofweek

# M1
hourly = pd.Series(temp_data, index=date_index)
daily_avg = hourly.resample('D').mean()
daily_max = hourly.resample('D').max()

# M2
returns = prices.pct_change().dropna()
volatility = returns.rolling(30).std() * np.sqrt(252)

# M3
features = pd.DataFrame({'value': s})
features['lag_1'] = s.shift(1)
features['lag_7'] = s.shift(7)
features['roll_mean_7'] = s.rolling(7).mean()
features['roll_std_7'] = s.rolling(7).std()

# M4
quarterly.resample('M').interpolate(method='cubic')

# M5
rolling_mean = s.rolling(30).mean()
rolling_std = s.rolling(30).std()
anomaly = (s - rolling_mean).abs() > 3 * rolling_std
```

## Related Concepts

- Python `datetime` module (standard library)
- NumPy datetime64 data type
- Statsmodels (ARIMA, SARIMAX)
- Prophet (Facebook time series forecasting)
- Time series-specific ML models (LSTM, N-BEATS, Transformer)

## Next Concepts

- Applying functions with `apply()` for custom feature engineering
- Combining time series features with other data via merge/join
- Visualizing time series with Matplotlib/Seaborn

## Summary

Time series analysis in Pandas provides comprehensive tools for working with datetime-indexed data. Key capabilities include parsing dates with `pd.to_datetime()`, generating regular date ranges with `pd.date_range()`, resampling to different frequencies, computing rolling window statistics, creating lagged features with `shift()` and `diff()`, and extracting datetime components with the `.dt` accessor. These tools are essential for financial analysis, sensor data processing, demand forecasting, and ML feature engineering.

## Key Takeaways

- Use `pd.to_datetime()` to parse strings to datetime64.
- `pd.date_range()` generates regular datetime sequences.
- `resample()` is a time-based GroupBy for frequency conversion.
- `rolling()` computes sliding window statistics (mean, std, etc.).
- `shift()` creates lag/lead features; `diff()` computes changes.
- `.dt` accessor extracts year, month, day, hour, dayofweek, etc.
- DatetimeIndex enables label-based slicing like `df.loc['2024-01']`.
- Always set datetime as the index (`set_index`) before resampling.
- First `window - 1` values of a rolling operation are NaN.
- For ML, create lag features, rolling features, and datetime components as inputs.
