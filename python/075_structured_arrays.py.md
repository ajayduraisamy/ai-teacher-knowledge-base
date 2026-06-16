# Concept: Structured Arrays

## Concept ID

PYT-075

## Difficulty

Advanced

## Domain

Python

## Module

NumPy

## Learning Objectives

- Create structured arrays with named fields using `dtype`
- Define complex dtypes with multiple field types and shapes
- Access and modify data by field name
- Understand `recarray` for attribute-style field access
- Compare structured arrays with pandas DataFrames
- Apply structured arrays for heterogeneous tabular data

## Prerequisites

- NumPy array creation and dtype understanding (PYT-066)
- Basic indexing (PYT-067)
- Familiarity with tabular data concepts (rows, columns, data types)

## Definition

A structured array is a NumPy array with a compound `dtype` that defines named fields, each potentially with a different data type. Unlike regular homogeneous arrays, structured arrays can store heterogeneously-typed data (e.g., integers, floats, strings) in a single array, with field access by name. This makes them similar to a table or a pandas DataFrame but with lower overhead.

## Intuition

Think of a structured array as a table where each column has a name and a type. A row might contain a name (string), an age (integer), and a salary (float). Unlike a regular NumPy array which requires all elements to be the same type, a structured array packs multiple types together in contiguous memory. Accessing a field returns all values for that column, while accessing a row returns a tuple of values.

## Why This Concept Matters

Structured arrays bridge the gap between raw NumPy arrays and high-level data structures like pandas DataFrames. They are useful when you need the performance and memory efficiency of NumPy but need to work with heterogeneous data. They are commonly used in scientific data formats (HDF5, NetCDF), reading CSV files with `np.genfromtxt`, and for interchanging data with C/Fortran code that uses structs.

## Real World Examples

1. **CSV Data Loading:** `np.genfromtxt('data.csv', names=True, dtype=None)` loads CSV rows into a structured array.
2. **Sensor Data:** Store timestamp (int64), temperature (float32), humidity (float32), and status (bool) from IoT sensors.
3. **Particle Physics:** Track particle ID (int32), position (float64, 3), momentum (float64, 3), and energy (float64).
4. **Database Records:** Load query results directly into structured arrays for fast numerical processing.
5. **Binary File I/O:** Read/write binary structs in C-compatible memory layout for inter-language communication.

## AI/ML Relevance

Structured arrays are less common in ML pipelines (pandas is preferred), but they are important when working with raw sensor data, scientific datasets, and memory-constrained environments. They serve as an efficient intermediate format when preprocessing data before conversion to ML-ready arrays. Many ML benchmark datasets (e.g., from UCI repository) can be loaded as structured arrays.

## Code Examples

### Example 1: Creating Structured Arrays

```python
import numpy as np

# Define dtype with field names and types
dtype = np.dtype([('name', 'U10'),
                  ('age', 'i4'),
                  ('salary', 'f8')])

# Create from list of tuples
employees = np.array([
    ('Alice', 30, 75000.0),
    ('Bob', 25, 62000.0),
    ('Charlie', 35, 95000.0),
    ('Diana', 28, 68000.0)
], dtype=dtype)

print("Employees:")
print(employees)
print("\nShape:", employees.shape)
print("dtype:", employees.dtype)
print("dtype names:", employees.dtype.names)
```
```
# Output:
# Employees:
# [('Alice', 30, 75000.) ('Bob', 25, 62000.)
#  ('Charlie', 35, 95000.) ('Diana', 28, 68000.)]
#
# Shape: (4,)
# dtype: [('name', '<U10'), ('age', '<i4'), ('salary', '<f8')]
# dtype names: ('name', 'age', 'salary')
```

### Example 2: Accessing Fields and Rows

```python
import numpy as np

dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('salary', 'f8')])
employees = np.array([
    ('Alice', 30, 75000.0),
    ('Bob', 25, 62000.0),
    ('Charlie', 35, 95000.0),
    ('Diana', 28, 68000.0)
], dtype=dtype)

# Access field (column) — returns a regular array
ages = employees['age']
print("Ages:", ages)
print("Age dtype:", ages.dtype)
print("Mean age:", ages.mean())

# Access multiple fields
name_salary = employees[['name', 'salary']]
print("\nName and salary:\n", name_salary)

# Access row — returns a tuple
print("\nFirst row:", employees[0])
print("Second row name:", employees[1]['name'])

# Boolean indexing
high_earners = employees[employees['salary'] > 70000]
print("\nHigh earners:\n", high_earners)

# Modify field values
employees['salary'] *= 1.05  # 5% raise!
print("\nAfter raise:\n", employees)
```
```
# Output:
# Ages: [30 25 35 28]
# Age dtype: int32
# Mean age: 29.5
#
# Name and salary:
#  [('Alice', 75000.) ('Bob', 62000.) ('Charlie', 95000.)
#  ('Diana', 68000.)]
#
# First row: ('Alice', 30, 75000.)
# Second row name: Bob
#
# High earners:
#  [('Alice', 30, 75000.) ('Charlie', 35, 95000.)]
#
# After raise:
#  [('Alice', 30, 78750.) ('Bob', 25, 65100.)
#  ('Charlie', 35, 99750.) ('Diana', 28, 71400.)]
```

### Example 3: Complex Dtype Definitions

```python
import numpy as np

# Nested fields
dtype_complex = np.dtype([
    ('point', [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
    ('label', 'U10'),
    ('confidence', 'f4')
])

points = np.array([
    ((1.0, 2.0, 3.0), 'A', 0.95),
    ((4.0, 5.0, 6.0), 'B', 0.87),
    ((7.0, 8.0, 9.0), 'C', 0.92)
], dtype=dtype_complex)

print("Points:\n", points)
print("\nAll x coordinates:", points['point']['x'])
print("Mean confidence:", points['confidence'].mean())

# Field with shape
dtype_shape = np.dtype([
    ('particle_id', 'i4'),
    ('position', 'f8', (3,)),   # 3-element array per row
    ('energy', 'f8')
])

particles = np.array([
    (0, [1.0, 2.0, 3.0], 100.0),
    (1, [4.0, 5.0, 6.0], 200.0)
], dtype=dtype_shape)

print("\nParticles:\n", particles)
print("Particle 0 position:", particles[0]['position'])
print("All positions:\n", particles['position'])
```
```
# Output:
# Points:
#  [((1., 2., 3.), 'A', 0.95) ((4., 5., 6.), 'B', 0.87)
#  ((7., 8., 9.), 'C', 0.92)]
#
# All x coordinates: [1. 4. 7.]
# Mean confidence: 0.91333336
#
# Particles:
#  [(0, [1., 2., 3.], 100.) (1, [4., 5., 6.], 200.)]
# Particle 0 position: [1. 2. 3.]
# All positions:
#  [[1. 2. 3.]
#  [4. 5. 6.]]
```

### Example 4: recarray — Record Array

```python
import numpy as np

# Create from structured array
dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('salary', 'f8')])
employees = np.array([
    ('Alice', 30, 75000.0),
    ('Bob', 25, 62000.0),
    ('Charlie', 35, 95000.0),
], dtype=dtype)

# Convert to record array for attribute-style access
rec = employees.view(np.recarray)
print("Name via attribute:", rec.name)
print("Ages:", rec.age)
print("Mean salary:", rec.salary.mean())

# Create directly with np.rec.array
rec2 = np.rec.array([
    ('X', 10, 1.5),
    ('Y', 20, 2.5),
    ('Z', 30, 3.5)
], names=['id', 'count', 'value'])
print("\nFrom np.rec.array:")
print("rec2.id:", rec2.id)
print("rec2[rec2.count > 15]:", rec2[rec2.count > 15])

# Modify via attribute
rec.salary *= 1.1
print("\nAfter 10% raise via attribute:")
print(f"  {rec[0].name}: {rec[0].salary:.0f}")
```
```
# Output:
# Name via attribute: ['Alice' 'Bob' 'Charlie']
# Ages: [30 25 35]
# Mean salary: 77333.33333333333
#
# From np.rec.array:
# rec2.id: ['X' 'Y' 'Z']
# rec2[rec2.count > 15]: [('Y', 20, 2.5) ('Z', 30, 3.5)]
#
# After 10% raise via attribute:
#   Alice: 82500
```

### Example 5: Structured Array Operations

```python
import numpy as np

dtype = np.dtype([('city', 'U15'), ('temp', 'f4'), ('humidity', 'f4')])
weather = np.array([
    ('New York', 22.5, 65.0),
    ('London', 18.0, 80.0),
    ('Tokyo', 28.0, 70.0),
    ('Sydney', 15.0, 55.0),
    ('Paris', 20.0, 60.0)
], dtype=dtype)

# Sorting by field
sorted_by_temp = np.sort(weather, order='temp')
print("Sorted by temperature (ascending):")
print(sorted_by_temp[['city', 'temp']])

# Sorting by multiple fields
sorted_multi = np.sort(weather, order=['humidity', 'temp'])
print("\nSorted by humidity, then temp:")
print(sorted_multi[['city', 'temp', 'humidity']])

# Adding a field (create a new dtype and copy)
new_dtype = np.dtype(weather.dtype.descr + [('category', 'U10')])
weather_extended = np.empty(weather.shape, dtype=new_dtype)
for field in weather.dtype.names:
    weather_extended[field] = weather[field]
weather_extended['category'] = np.where(weather['temp'] > 20, 'Warm', 'Cool')
print("\nWith category:")
print(weather_extended[['city', 'temp', 'category']])
```
```
# Output:
# Sorted by temperature (ascending):
# [('Sydney', 15., 55.) ('London', 18., 80.) ('Paris', 20., 60.)
#  ('New York', 22.5, 65.) ('Tokyo', 28., 70.)]
#
# Sorted by humidity, then temp:
# [('Sydney', 15., 55.) ('Paris', 20., 60.) ('New York', 22.5, 65.)
#  ('Tokyo', 28., 70.) ('London', 18., 80.)]
#
# With category:
# [('New York', 22.5, 65., 'Warm') ('London', 18., 80., 'Cool')
#  ('Tokyo', 28., 70., 'Warm') ('Sydney', 15., 55., 'Cool')
#  ('Paris', 20., 60., 'Cool')]
```

### Example 6: Loading CSV Data with genfromtxt

```python
import numpy as np

# Create a sample CSV in-memory
import io
csv_data = b"""name,age,score,year
Alice,30,85.5,2020
Bob,25,92.3,2021
Charlie,35,78.0,2019
Diana,28,95.1,2022
Eve,32,88.7,2021"""

# Load into structured array
data = np.genfromtxt(io.BytesIO(csv_data),
                     delimiter=',',
                     names=True,
                     dtype=['U10', 'i4', 'f8', 'i4'])

print("Loaded data:\n", data)
print("\nColumn names:", data.dtype.names)
print("Mean score:", data['score'].mean())
print("Oldest person:", data[data['age'].argmax()]['name'])

# Filter: students with score > 85
top_students = data[data['score'] > 85]
print("\nTop students (score > 85):")
print(top_students[['name', 'score']])
```
```
# Output:
# Loaded data:
#  [('Alice', 30, 85.5, 2020) ('Bob', 25, 92.3, 2021)
#  ('Charlie', 35, 78. , 2019) ('Diana', 28, 95.1, 2022)
#  ('Eve', 32, 88.7, 2021)]
#
# Column names: ('name', 'age', 'score', 'year')
# Mean score: 87.92
# Oldest person: Charlie
#
# Top students (score > 85):
# [('Alice', 85.5) ('Bob', 92.3) ('Diana', 95.1) ('Eve', 88.7)]
```

### Example 7: Structured Array vs Regular Array Conversion

```python
import numpy as np

dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('label', 'i4')])
structured = np.array([
    (1.0, 2.0, 0),
    (3.0, 4.0, 1),
    (5.0, 6.0, 0)
], dtype=dtype)

# Extract fields as separate arrays
X = np.column_stack([structured['x'], structured['y']])
y = structured['label']
print("Feature matrix X:\n", X)
print("Labels y:", y)

# Convert to homogeneous array (lose field names)
flat = structured.view(np.float64).reshape(-1, 3)
print("\nViewed as flat float64:\n", flat)
print("Note: label field is now float!")

# Pack back from separate arrays
new_struct = np.zeros(3, dtype=dtype)
new_struct['x'] = X[:, 0]
new_struct['y'] = X[:, 1]
new_struct['label'] = y
print("\nReconstructed:\n", new_struct)
```
```
# Output:
# Feature matrix X:
#  [[1. 2.]
#  [3. 4.]
#  [5. 6.]]
# Labels y: [0 1 0]
#
# Viewed as flat float64:
#  [[1. 2. 0.]
#  [3. 4. 1.]
#  [5. 6. 0.]]
# Note: label field is now float!
#
# Reconstructed:
#  [(1., 2., 0) (3., 4., 1) (5., 6., 0)]
```

## Common Mistakes

1. **Using Field Names as Index (Without Quotes):** Fields must be accessed with string names: `arr['field']`, not `arr.field`. For attribute-style access, use `recarray`.

2. **Incorrect Dtype String Format:** `'U10'` means a Unicode string of up to 10 characters. `'i4'` is a 4-byte integer. `'f8'` is an 8-byte float. `'S10'` is a byte string. These codes differ from pandas.

3. **Forgetting That String Fields Have Fixed Width:** Strings in structured arrays are fixed-width. 'U10' truncates strings longer than 10 characters and pads shorter ones. This can waste memory or truncate data.

4. **Byte Order Markers ('<' and '>'):** Dtype strings may show `'<U10'` (little-endian) or `'>U10'` (big-endian). These are platform-specific but usually not a concern unless exchanging binary data.

5. **Adding Fields Is Not Trivial:** Unlike pandas DataFrames where `df['new'] = ...` adds a column, structured arrays require creating a new dtype and copying all data.

6. **Sorting Fields by Name vs by Value:** `np.sort(struct_arr, order='field')` sorts by the given field. Without `order`, sorting treats the array as a homogeneous byte sequence (usually meaningless).

7. **Performance of Field Access:** Accessing a single field (`arr['field']`) is fast as it returns a view. Accessing multiple fields (`arr[['f1', 'f2']]`) returns a copy and is slower.

## Interview Questions

### Beginner

1. **Q:** What is a structured array in NumPy?
   **A:** A structured array is a NumPy array with a compound dtype that has named fields of potentially different types, allowing heterogeneous data in a single array.

2. **Q:** How do you access a field named "age" in a structured array?
   **A:** `arr['age']` returns all values for that field as a regular NumPy array.

3. **Q:** How do you create a structured array with fields "name" (string) and "age" (integer)?
   **A:** Define `dtype = np.dtype([('name', 'U20'), ('age', 'i4')])`, then `np.array([('Alice', 30)], dtype=dtype)`.

4. **Q:** What is the difference between a structured array and a regular NumPy array?
   **A:** Regular arrays store homogeneous data (all same type). Structured arrays store heterogeneous data with named fields, similar to a table.

5. **Q:** How do you view a structured array with attribute-style field access?
   **A:** Convert to record array: `rec = arr.view(np.recarray)`, then access fields as `rec.fieldname`.

### Intermediate

1. **Q:** How do you add a new field to an existing structured array?
   **A:** Create a new dtype that includes the new field, allocate `np.empty` with the new dtype, copy existing fields, then set the new field. There is no in-place field addition.

2. **Q:** How does `np.sort` work with structured arrays, and what does the `order` parameter do?
   **A:** `np.sort(struct_arr, order='field')` sorts by the specified field name(s). Without `order`, sorting is lexicographic over the bytes, which is usually not meaningful.

3. **Q:** What are the advantages of structured arrays over pandas DataFrames?
   **A:** Structured arrays are more memory efficient (no index overhead), faster for numerical operations, directly compatible with NumPy functions, and have lower serialization overhead. They work in environments where pandas isn't available.

4. **Q:** What is the difference between `'U10'` and `'S10'` in dtype specifications?
   **A:** `'U10'` is a Unicode string (4 bytes per character, handles any Unicode). `'S10'` is a byte string (1 byte per character, ASCII only). Use 'U' for text data, 'S' for binary or known-ASCII data.

5. **Q:** How do you load a CSV file into a structured array?
   **A:** Use `np.genfromtxt('file.csv', names=True, delimiter=',', dtype=None)`. Set `dtype=None` to auto-detect types, or provide explicit dtype.

### Advanced

1. **Q:** Explain the internal memory layout of a structured array. How does it differ from a regular array with multiple columns?
   **A:** Structured arrays use a compound dtype where fields are laid out in memory sequentially per row (row-major). Field access returns a view using strides. This is equivalent to a C struct. A regular 2D array uses a single type for all elements. The structured layout is more cache-friendly for row access; the homogeneous layout is better for column operations.

2. **Q:** How would you handle byte-order issues when exchanging structured arrays between systems with different endianness?
   **A:** Use `dtype.newbyteorder()` to swap byte order. For portable data, use network byte order (big-endian, '>') or write to text formats (CSV, HDF5). NumPy's `np.array(..., dtype=dtype.newbyteorder('>'))` ensures consistent byte order across platforms.

3. **Q:** Compare the performance characteristics of structured arrays vs a dictionary of homogeneous arrays for tabular data. When would you choose one over the other?
   **A:** A dictionary of arrays (`{'x': arr_x, 'y': arr_y}`) allows adding columns dynamically and provides faster column access. Structured arrays are faster for row access, use less total memory (single array object vs multiple), and provide contiguous row layout. Choose dict-of-arrays for column-heavy workloads (ML feature matrices); choose structured arrays for row-heavy workloads (database records, row-by-row processing) or when interfacing with C/Fortran structs.

## Practice Problems

### Easy

1. Create a structured array for 5 students with fields: name (U10), grade (f8), passed (bool).

2. Access and print the "grade" field from the array above. Compute the mean grade.

3. Add a student to the array (hint: use `np.append`).

4. Select all students who passed (grade >= 60).

5. Convert the structured array to a regular array using `.view()`.

### Medium

1. Load the following CSV data into a structured array and compute statistics per field:
   ```
   product,price,quantity,in_stock
   Widget,10.99,100,True
   Gadget,24.99,50,False
   Doohickey,5.49,200,True
   ```

2. Sort a structured array with fields ('name', 'salary', 'department') by salary descending, then by name alphabetically.

3. Given a structured array with fields ('x', 'y', 'label'), compute the centroid (mean x, mean y) for each label.

4. Add a new field "score_squared" to a structured array with a "score" field.

5. Create a recarray from a structured array and use attribute access to filter rows where salary > 50000.

### Hard

1. Implement a function `structured_to_dataframe(struct_arr)` that converts a structured NumPy array to a dictionary suitable for pandas DataFrame construction.

2. Implement a function that reads a binary file containing C structs (e.g., 4-byte int + 8-byte double + 1-byte char) into a structured array.

3. Implement a function that performs a SQL-style JOIN on two structured arrays based on a key field, supporting both inner and left joins. The function should return a new structured array with combined fields.

## Solutions

### Easy Solutions

```python
# 1
dtype = np.dtype([('name', 'U10'), ('grade', 'f8'), ('passed', '?')])
students = np.array([
    ('Alice', 85.5, True),
    ('Bob', 72.0, True),
    ('Charlie', 45.0, False),
    ('Diana', 91.0, True),
    ('Eve', 58.0, False)
], dtype=dtype)
print(students)

# 2
grades = students['grade']
print("Grades:", grades)
print("Mean grade:", grades.mean())

# 3
new_student = np.array([('Frank', 76.0, True)], dtype=dtype)
students = np.append(students, new_student)
print("After append:\n", students)

# 4
passing = students[students['passed']]
print("Passing students:", passing['name'])

# 5
as_float = students[['grade', 'passed']].view(np.float64)
print(as_float)
```

### Medium Solutions

```python
# 1 CSV loading
import io
csv = b"product,price,quantity,in_stock\nWidget,10.99,100,True\nGadget,24.99,50,False\nDoohickey,5.49,200,True"
data = np.genfromtxt(io.BytesIO(csv), delimiter=',', names=True, dtype=['U10', 'f8', 'i4', '?'])
print("Prices:", data['price'])
print("Mean price:", data['price'].mean())
print("Total quantity:", data['quantity'].sum())
print("All in stock:", data['in_stock'].all())

# 2 Multi-field sort
dtype = np.dtype([('name', 'U10'), ('salary', 'f8'), ('dept', 'U10')])
emps = np.array([
    ('Alice', 75000, 'Eng'),
    ('Bob', 62000, 'Sales'),
    ('Charlie', 95000, 'Eng'),
    ('Diana', 75000, 'Sales')
], dtype=dtype)
sorted_emps = np.sort(emps, order=['salary', 'name'])
print("Sorted by salary asc, name asc:", sorted_emps[['name', 'salary']])

# 3 Group centroids
dtype = np.dtype([('x', 'f8'), ('y', 'f8'), ('label', 'i4')])
points = np.array([
    (1.0, 2.0, 0), (2.0, 3.0, 0), (10.0, 10.0, 1), (12.0, 8.0, 1), (3.0, 1.0, 0)
], dtype=dtype)
for label in [0, 1]:
    cluster = points[points['label'] == label]
    centroid = (cluster['x'].mean(), cluster['y'].mean())
    print(f"Label {label}: centroid = {centroid}")

# 4 Add field
arr = np.array([(1, 10.0), (2, 20.0), (3, 30.0)], dtype=[('id', 'i4'), ('score', 'f8')])
new_dtype = np.dtype(arr.dtype.descr + [('score_squared', 'f8')])
new_arr = np.empty(arr.shape, dtype=new_dtype)
for f in arr.dtype.names:
    new_arr[f] = arr[f]
new_arr['score_squared'] = arr['score'] ** 2
print(new_arr)

# 5 Recarray filter
dtype = np.dtype([('name', 'U10'), ('salary', 'f8')])
emps = np.array([('Alice', 75000), ('Bob', 45000), ('Charlie', 95000)], dtype=dtype)
rec = emps.view(np.recarray)
high = rec[rec.salary > 50000]
print("High earners:", high.name)
```

### Hard Solutions

```python
# 1 Structured array to DataFrame-compatible dict
def structured_to_dict(struct_arr):
    result = {}
    for name in struct_arr.dtype.names:
        result[name] = struct_arr[name]
    return result

dtype = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
data = np.array([('Alice', 30, 85.5), ('Bob', 25, 92.0)], dtype=dtype)
d = structured_to_dict(data)
print("Dict:", d)
# import pandas as pd
# df = pd.DataFrame(d)

# 2 Binary C-struct reader
def read_c_structs(filepath, count, dtype):
    with open(filepath, 'rb') as f:
        raw = f.read(count * dtype.itemsize)
    return np.frombuffer(raw, dtype=dtype)

# Example usage:
# dtype = np.dtype([('id', '>i4'), ('value', '>f8'), ('code', 'S1')])
# data = read_c_structs('data.bin', 100, dtype)

# 3 SQL-style JOIN
def join_structured(left, right, on, how='inner'):
    left_key = left[on]
    right_key = right[on]
    
    # Build index mapping
    left_indices = []
    right_indices = []
    
    for i, lk in enumerate(left_key):
        matches = np.where(right_key == lk)[0]
        for j in matches:
            left_indices.append(i)
            right_indices.append(j)
    
    # Build result dtype (combine fields, avoid duplicate key)
    right_fields = [(name, right.dtype[name]) for name in right.dtype.names if name != on]
    result_dtype = np.dtype(left.dtype.descr + right_fields)
    
    result = np.empty(len(left_indices), dtype=result_dtype)
    
    for i, (li, ri) in enumerate(zip(left_indices, right_indices)):
        for name in left.dtype.names:
            result[name][i] = left[name][li]
        for name, _ in right_fields:
            result[name][i] = right[name][ri]
    
    if how == 'left':
        # Add unmatched left rows
        matched_right = set(right_indices)
        for li in range(len(left)):
            if li not in set(left_indices):
                new_row = np.empty(1, dtype=result_dtype)
                for name in left.dtype.names:
                    new_row[name] = left[name][li]
                result = np.append(result, new_row)
    
    return result

# Test the join
left = np.array([(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')],
                dtype=[('id', 'i4'), ('name', 'U10')])
right = np.array([(1, 75000), (2, 62000), (1, 80000)],
                 dtype=[('id', 'i4'), ('salary', 'f8')])

joined = join_structured(left, right, 'id')
print("Inner join:")
print(joined)

left_joined = join_structured(left, right, 'id', how='left')
print("\nLeft join:")
print(left_joined)
```

## Related Concepts

- Pandas DataFrame (higher-level abstraction)
- NumPy dtype system
- C struct memory layout
- HDF5 compound types
- SQL tables and database records

## Next Concepts

- (Advanced NumPy topics: Masked arrays, memory mapping)
- Pandas DataFrames for tabular data analysis

## Summary

Structured arrays extend NumPy to handle heterogeneous tabular data with named fields of different types. They provide efficient row-oriented storage with field access by name, sorting by field, and loading CSV data via `np.genfromtxt`. `recarray` adds attribute-style access. Structured arrays bridge the gap between raw NumPy arrays and high-level data structures like pandas DataFrames, offering performance and memory efficiency for structured data.

## Key Takeaways

- Structured arrays use compound dtypes with named fields of different types
- Access fields with `arr['fieldname']`; rows return tuples
- Use `view(np.recarray)` for attribute-style access (e.g., `arr.name`)
- `np.genfromtxt` loads CSV data directly into structured arrays
- Strings are fixed-width ('U10' = Unicode, 'S10' = bytes)
- Adding fields requires creating a new dtype and copying data
- Sorting structured arrays requires the `order` parameter
- Structured arrays are more memory efficient than pandas for simple tabular data
- They are ideal for interfacing with C/Fortran structs and binary data formats
- Field access returns a view (fast); multi-field access returns a copy (slower)
