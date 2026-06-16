# Concept: Reading and Writing Data

## Concept ID

PYT-078

## Difficulty

INTERMEDIATE

## Domain

Python

## Module

Pandas

## Learning Objectives

- Read data from CSV, Excel, JSON, SQL, Parquet, and HTML formats
- Write DataFrames to CSV, Excel, JSON, and Parquet
- Handle compression and encoding during I/O operations
- Read large files in chunks to manage memory
- Understand format-specific parameters and trade-offs

## Prerequisites

- DataFrame fundamentals (PYT-077)
- Basic understanding of file paths and file systems
- Familiarity with SQL queries (for `read_sql`)
- Installing optional dependencies: `openpyxl`, `sqlalchemy`, `pyarrow`, `lxml`

## Definition

Pandas provides a comprehensive suite of I/O (input/output) functions to read data from various file formats into a DataFrame and write DataFrames back to those formats. The main functions follow the naming convention `pd.read_<format>()` for reading and `df.to_<format>()` for writing.

```python
# Reading
df = pd.read_csv('path/to/file.csv')
df = pd.read_excel('path/to/file.xlsx', sheet_name='Sheet1')
df = pd.read_json('path/to/file.json')
df = pd.read_sql('SELECT * FROM table', connection)
df = pd.read_parquet('path/to/file.parquet')
df = pd.read_pickle('path/to/file.pkl')
df = pd.read_html('path/to/page.html')

# Writing
df.to_csv('path/to/output.csv', index=False)
df.to_excel('path/to/output.xlsx', sheet_name='Sheet1', index=False)
df.to_json('path/to/output.json')
df.to_parquet('path/to/output.parquet')
```

## Intuition

Think of I/O functions as **adapters** between Pandas' in-memory DataFrame and persistent storage formats. Each format has strengths:

- **CSV**: Universal, human-readable, slow for large data, no schema.
- **Parquet**: Columnar, compressed, fast, schema-rich — ideal for big data.
- **Excel**: Business-friendly, multi-sheet, but slow and limited rows.
- **JSON**: Web-friendly, nested structures, semi-structured.
- **SQL**: Database-native, supports queries, ACID guarantees.
- **Pickle**: Python-native, fast, but not portable across languages.
- **HTML**: Scraping tables from web pages.

## Why This Concept Matters

Real-world data lives in files and databases. The ability to efficiently read and write data is the gateway to any data analysis or ML pipeline. Choosing the right format and parameters can mean the difference between:

- A script that runs in 2 seconds vs 2 minutes
- A file that is 10 MB vs 1 GB on disk
- A pipeline that crashes vs one that gracefully handles 100 GB datasets

Understanding I/O trade-offs — compression, chunking, encoding, schema preservation — is a hallmark of an experienced data engineer or data scientist.

## Real World Examples

1. **ETL Pipeline**: Read CSV from S3, transform, write Parquet to data warehouse.
2. **Financial Reporting**: Read Excel reports from stakeholders, consolidate, export to CSV for analysis.
3. **Web Scraping**: Use `read_html` to extract tables from Wikipedia or documentation pages.
4. **Database Export**: Use `read_sql` to pull customer data from PostgreSQL for churn modeling.
5. **API Data**: Read JSON responses from REST APIs into DataFrames for analysis.

## AI/ML Relevance

- **Training Data**: Most ML training starts with `pd.read_csv()` or `pd.read_parquet()`.
- **Feature Store**: Parquet is the standard format for feature stores due to columnar access and compression.
- **Large Datasets**: Chunked reading enables out-of-core preprocessing for datasets that don't fit in RAM.
- **Model Outputs**: Predictions are often saved to CSV or Parquet for downstream consumption.
- **Pipeline Artifacts**: Pickle is used to serialize preprocessor objects and small models.

## Code Examples

### Example 1: Reading and writing CSV

```python
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'id': range(1, 6),
    'value': [10.5, 20.3, 15.8, np.nan, 30.1],
    'category': ['A', 'B', 'A', 'B', 'C']
})

# Write to CSV
df.to_csv('sample.csv', index=False)

# Read back
df_read = pd.read_csv('sample.csv')
print(df_read)
# Output:
#    id  value category
# 0   1   10.5        A
# 1   2   20.3        B
# 2   3   15.8        A
# 3   4    NaN        B
# 4   5   30.1        C
```

### Example 2: CSV with custom parameters

```python
# Write without header, with custom separator
df.to_csv('no_header.csv', index=False, header=False, sep='|')

# Read with custom separator, skipping first row, setting column names
df_custom = pd.read_csv('no_header.csv', sep='|',
                        names=['idx', 'val', 'cat'], skiprows=0)
print(df_custom)
# Output:
#    idx   val cat
# 0    1  10.5   A
# 1    2  20.3   B
# 2    3  15.8   A
# 3    4   NaN   B
# 4    5  30.1   C
```

### Example 3: Handling missing values on read

```python
# Write with a custom NA marker
df.to_csv('custom_na.csv', index=False, na_rep='NULL')

# Read back, specifying which strings represent NA
df_na = pd.read_csv('custom_na.csv', na_values=['NULL'])
print(df_na)
# Output:
#    id  value category
# 0   1   10.5        A
# 1   2   20.3        B
# 2   3   15.8        A
# 3   4    NaN        B
# 4   5   30.1        C
```

### Example 4: Reading Excel files

```python
# Requires openpyxl or xlrd
# Write to Excel
df.to_excel('sample.xlsx', sheet_name='Sheet1', index=False)

# Read back
df_excel = pd.read_excel('sample.xlsx', sheet_name='Sheet1')
print(df_excel)
# Output:
#    id  value category
# 0   1   10.5        A
# 1   2   20.3        B
# 2   3   15.8        A
# 3   4    NaN        B
# 4   5   30.1        C

# Read specific columns
df_subset = pd.read_excel('sample.xlsx', usecols=['id', 'value'])
print(df_subset)
# Output:
#    id  value
# 0   1   10.5
# 1   2   20.3
# 2   3   15.8
# 3   4    NaN
# 4   5   30.1
```

### Example 5: Reading JSON

```python
# Write to JSON
df.to_json('sample.json', orient='records', lines=True)

# Read back
df_json = pd.read_json('sample.json', orient='records', lines=True)
print(df_json)
# Output:
#    id  value category
# 0   1   10.5        A
# 1   2   20.3        B
# 2   3   15.8        A
# 3   4    NaN        B
# 4   5   30.1        C
```

### Example 6: Reading data with compression

```python
# Write compressed CSV
df.to_csv('sample.csv.gz', index=False, compression='gzip')

# Read compressed CSV — Pandas detects .gz extension automatically
df_gz = pd.read_csv('sample.csv.gz')
print(df_gz.shape)
# Output:
# (5, 3)

# Also supports: 'bz2', 'xz', 'zip'
```

### Example 7: Chunking large files

```python
# Simulate a large file by writing many rows
big_df = pd.DataFrame({'x': range(10000), 'y': np.random.randn(10000)})
big_df.to_csv('large.csv', index=False)

# Read in chunks of 1000 rows
chunk_size = 1000
total_sum = 0
for chunk in pd.read_csv('large.csv', chunksize=chunk_size):
    total_sum += chunk['y'].sum()

print(f"Sum of all y values: {total_sum:.2f}")
# Output:
# Sum of all y values: -12.34  (varies)
```

### Example 8: Reading Parquet

```python
# Requires pyarrow or fastparquet
df.to_parquet('sample.parquet', index=False)

df_pq = pd.read_parquet('sample.parquet')
print(df_pq)
# Output:
#    id  value category
# 0   1   10.5        A
# 1   2   20.3        B
# 2   3   15.8        A
# 3   4    NaN        B
# 4   5   30.1        C
```

### Example 9: Reading HTML tables

```python
# read_html returns a list of DataFrames (one per <table> tag)
# This example uses a URL with an HTML table
url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
tables = pd.read_html(url)
if tables:
    print(f"Found {len(tables)} tables")
    print(tables[0].head(3))
# Output:
# Found 3 tables
#   Rank  Country/Region   GDP(US$million)  ...
# 0    1   United States    ...                ...
```

## Common Mistakes

1. **Forgetting `index=False` when saving.** The default writes the index as a column; on re-read it becomes a duplicate column.
2. **Using `read_excel` without installing `openpyxl`.** Install with `pip install openpyxl`.
3. **Assuming all CSV separators are commas.** Many datasets use `;` or `\t` — use `sep` parameter.
4. **Reading all data into memory when chunking is needed.** Use `chunksize` for files larger than available RAM.
5. **Not specifying `encoding` for non-UTF-8 files.** Common encodings: `'latin1'`, `'cp1252'`, `'utf-8-sig'`.

## Interview Questions

### Beginner

1. How do you read a CSV file into a DataFrame?
2. What does `index=False` do when saving a DataFrame to CSV?
3. How do you read only specific columns from a CSV file?
4. What is the purpose of the `sep` parameter in `read_csv`?
5. How do you handle files with no header row?

### Intermediate

1. Explain how chunking works in `pd.read_csv` and when you would use it.
2. What are the trade-offs between CSV and Parquet for storing data?
3. How would you read a JSON file with nested structures into a flat DataFrame?
4. How do you write multiple DataFrames to different sheets in the same Excel file?
5. What is the difference between `orient='records'` and `orient='split'` in `to_json`?

### Advanced

1. Implement a streaming CSV parser that processes a 50 GB file using chunking and writes out summary statistics per chunk, combining them at the end.
2. Compare the performance of `read_csv`, `read_parquet`, and `read_feather` on a 1 GB dataset. What accounts for the differences?
3. Design a custom I/O adapter class that reads from a SQL database in chunks, applies a transform, and writes to Parquet with partition columns.

## Practice Problems

### Easy

1. Read `data.csv` and display the first 5 rows.
2. Save a DataFrame to CSV without the index column.
3. Read a CSV file specifying that missing values are marked as `'NA'` or `'null'`.
4. Read only columns `'A'` and `'C'` from a 10-column CSV file.
5. Write a DataFrame to an Excel file with sheet name `'Report'`.

### Medium

1. Read a CSV file with 500k rows in chunks of 50k and compute the mean of a column per chunk, then compute the global mean.
2. Convert a CSV file to Parquet format, ensuring all datetime columns are properly typed.
3. Read a JSON file with `orient='records'` lines format and flatten a nested `'address'` field into separate columns.
4. Download an HTML table from Wikipedia and clean it (remove footnotes, convert numbers to numeric).
5. Write a script that reads a SQL table, applies a filter, and writes the result to a compressed CSV.

### Hard

1. Build a function that automatically detects the delimiter, encoding, and header row of an unknown text file and returns a properly parsed DataFrame.
2. Implement a memory-mapped reader that lazily reads a large Parquet file column-by-column without loading the full file into memory.
3. Create a pipeline that reads from a PostgreSQL database in chunks, parallelizes transformation across 4 processes, and writes results to partitioned Parquet files.

## Solutions

```python
# E1
df = pd.read_csv('data.csv')
print(df.head())

# E2
df.to_csv('output.csv', index=False)

# E3
df = pd.read_csv('data.csv', na_values=['NA', 'null'])

# E4
df = pd.read_csv('data.csv', usecols=['A', 'C'])

# E5
df.to_excel('output.xlsx', sheet_name='Report', index=False)

# M1
chunk_sum = 0
chunk_count = 0
for chunk in pd.read_csv('large.csv', chunksize=50000):
    chunk_sum += chunk['value'].sum()
    chunk_count += len(chunk)
global_mean = chunk_sum / chunk_count

# M2
df = pd.read_csv('data.csv', parse_dates=['date_col'])
df.to_parquet('data.parquet', index=False)

# M3
df = pd.read_json('data.json', orient='records', lines=True)
import pandas as pd
address_df = pd.json_normalize(df['address'])
df = pd.concat([df.drop('address', axis=1), address_df], axis=1)

# M4
tables = pd.read_html('https://en.wikipedia.org/wiki/...')
df = tables[0]
# Clean: strip footnotes, convert to numeric
df.iloc[:, 1:] = df.iloc[:, 1:].replace(r'\[\d+\]', '', regex=True)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

## Related Concepts

- File system operations (`os`, `pathlib`)
- SQL databases and connection engines (SQLAlchemy)
- Data serialization formats (Avro, Arrow, Protocol Buffers)
- Compression algorithms (gzip, brotli, snappy, zstd)

## Next Concepts

- Indexing and selection for exploring loaded data
- Handling missing data that comes in during import
- Merging and joining datasets from multiple sources

## Summary

Pandas I/O functions (`pd.read_csv`, `pd.to_csv`, `read_excel`, `read_json`, `read_sql`, `read_parquet`, `read_pickle`, `read_html`) bridge the gap between disk storage and in-memory DataFrames. Each format has strengths: CSV for portability, Parquet for performance, Excel for business users, JSON for web APIs. Key considerations include compression, chunking for large files, encoding, and schema handling.

## Key Takeaways

- `pd.read_csv()` is the most common entry point for data analysis.
- Use `index=False` when saving to avoid redundant columns.
- Chunking (`chunksize`) enables processing files larger than RAM.
- Parquet is preferred for large datasets due to columnar storage and compression.
- Each format has specific parameters (`sep`, `encoding`, `na_values`) for correct parsing.
- Install optional dependencies (`openpyxl`, `pyarrow`, `sqlalchemy`) as needed.
- Compression (gzip, bz2) reduces disk and network usage.
- `read_html` extracts tables from web pages for quick scraping.
