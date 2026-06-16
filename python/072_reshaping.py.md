# Concept: Reshaping Arrays

## Concept ID

PYT-072

## Difficulty

Intermediate

## Domain

Python

## Module

NumPy

## Learning Objectives

- Change array shape with `reshape` and `resize`
- Flatten arrays into 1D with `ravel` and `flatten`
- Transpose arrays with `transpose` and `.T`
- Swap axes with `swapaxes`
- Add and remove dimensions with `expand_dims`, `squeeze`, and `newaxis`
- Understand the difference between views and copies in reshaping operations
- Prepare data for neural network inputs by reshaping

## Prerequisites

- NumPy array creation and shape attributes (PYT-066)
- Array indexing (PYT-067)
- Understanding of array dimensions and axes

## Definition

Reshaping refers to changing the dimensions of an array without changing its data. NumPy provides functions to rearrange elements into new shapes, flatten multidimensional arrays, transpose axes, add or remove singleton dimensions, and reorder axes. These operations are essential for data preparation and compatibility between different numerical libraries.

## Intuition

Think of a NumPy array as a sequence of elements stored in contiguous memory, and the shape as a "view" that interprets this sequence as a grid of a certain dimensionality. Reshaping rearranges this interpretation without moving the data (when possible). Flattening unravels a multidimensional array into a linear sequence. Transposing swaps the role of rows and columns. Expanding/squeezing dimensions is like adding or removing a bracket level around the data.

## Why This Concept Matters

Data rarely arrives in the exact shape needed for computation. Machine learning models have specific input shape requirements: neural networks expect batches of shape `(batch_size, channels, height, width)` or `(batch_size, height, width, channels)`. Reshaping is the bridge between your dataset shape and the model's expected input shape. Transposing is critical for matrix operations and broadcasting.

## Real World Examples

1. **Image Data:** Flatten a batch of 32x32 RGB images from `(100, 32, 32, 3)` to `(100, 3072)` for a fully connected network.
2. **Sequence Data:** Reshape time series of shape `(1000, 10)` to `(1000, 10, 1)` for an LSTM input (adding a feature dimension).
3. **Matrix Operations:** Transpose a weight matrix from shape `(features, neurons)` to `(neurons, features)` for correct matrix multiplication.
4. **Broadcasting:** Add a dimension with `newaxis` to align arrays of different shapes for broadcasting.
5. **Batch Processing:** Reshape a flat array of predictions into `(n_batches, batch_size, n_classes)`.

## AI/ML Relevance

Neural network frameworks (TensorFlow, PyTorch) all require specific tensor shapes. Understanding NumPy reshaping is directly transferable to tensor reshaping. Data preprocessing pipelines frequently reshape data between different representations: flat feature vectors, 2D matrices, 3D sequences, and 4D image tensors. Transposing is used for correct matrix alignment in backpropagation.

## Code Examples

### Example 1: Basic reshape

```python
import numpy as np

arr = np.arange(12)
print("Original (1D):", arr)

# Reshape to 3x4
reshaped = arr.reshape(3, 4)
print("\nReshaped to (3, 4):\n", reshaped)

# Reshape to 2x2x3 (3D)
reshaped_3d = arr.reshape(2, 2, 3)
print("\nReshaped to (2, 2, 3):\n", reshaped_3d)

# Auto-detect dimension with -1
auto = arr.reshape(2, -1)
print("\nReshaped to (2, -1):\n", auto)
print("Shape:", auto.shape)

# -1 infers the remaining dimension
auto2 = arr.reshape(-1, 6)
print("\nReshaped to (-1, 6):\n", auto2)
print("Shape:", auto2.shape)
```
```
# Output:
# Original (1D): [ 0  1  2  3  4  5  6  7  8  9 10 11]
#
# Reshaped to (3, 4):
#  [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
#
# Reshaped to (2, 2, 3):
#  [[[ 0  1  2]
#   [ 3  4  5]]
#
#  [[ 6  7  8]
#   [ 9 10 11]]]
#
# Reshaped to (2, -1):
#  [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
# Shape: (2, 6)
#
# Reshaped to (-1, 6):
#  [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
# Shape: (2, 6)
```

### Example 2: Flatten and Ravel

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# flatten returns a copy
flat_copy = arr.flatten()
flat_copy[0] = 99
print("Original after flatten mod:\n", arr)
print("Flat copy:", flat_copy)

# ravel returns a view (when possible)
flat_view = arr.ravel()
flat_view[0] = 99
print("\nOriginal after ravel mod:\n", arr)
print("Flat view:", flat_view)

# C-order (row-major) vs Fortran-order (column-major)
print("\nC-order flatten:", arr.flatten(order='C'))
print("F-order flatten:", arr.flatten(order='F'))
```
```
# Output:
# Original after flatten mod:
#  [[1 2 3]
#  [4 5 6]]
# Flat copy: [99  2  3  4  5  6]
#
# Original after ravel mod:
#  [[99  2  3]
#  [ 4  5  6]]
# Flat view: [99  2  3  4  5  6]
#
# C-order flatten: [99  2  3  4  5  6]
# F-order flatten: [99  4  2  5  3  6]
```

### Example 3: Transpose and T

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Original shape:", arr.shape)
print("Original:\n", arr)
print("\nTranspose (T):\n", arr.T)
print("Transpose shape:", arr.T.shape)

# transpose with axis ordering for 3D
arr3d = np.arange(24).reshape(2, 3, 4)
print("\n3D shape:", arr3d.shape)
print("Original 3D [0, :, :]:\n", arr3d[0, :, :])

# Transpose axes: (1, 2, 0) -> new axis order
t3d = np.transpose(arr3d, (1, 2, 0))
print("\nTransposed (1, 2, 0) shape:", t3d.shape)
print("t3d[0, :, :]:\n", t3d[0, :, :])

# .T only reverses axes (last 2 for 2D, all reversed for ND)
print("\n.T on 3D reverses all axes:", arr3d.T.shape)
```
```
# Output:
# Original shape: (2, 3)
# Original:
#  [[1 2 3]
#  [4 5 6]]
#
# Transpose (T):
#  [[1 4]
#  [2 5]
#  [3 6]]
# Transpose shape: (3, 2)
#
# 3D shape: (2, 3, 4)
# Original 3D [0, :, :]:
#  [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
#
# Transposed (1, 2, 0) shape: (3, 4, 2)
# t3d[0, :, :]:
#  [[ 0 12]
#  [ 1 13]
#  [ 2 14]
#  [ 3 15]]
#
# .T on 3D reverses all axes: (4, 3, 2)
```

### Example 4: expand_dims, squeeze, and newaxis

```python
import numpy as np

arr = np.array([1, 2, 3])  # shape (3,)
print("Original shape:", arr.shape)

# Add dimension at axis 0
expanded_0 = np.expand_dims(arr, axis=0)
print("expand_dims axis=0:", expanded_0.shape, expanded_0)

# Add dimension at axis 1
expanded_1 = np.expand_dims(arr, axis=1)
print("expand_dims axis=1:", expanded_1.shape, "\n", expanded_1)

# Using newaxis (more concise)
via_newaxis = arr[np.newaxis, :]
print("\nnewaxis axis=0:", via_newaxis.shape, via_newaxis)

via_newaxis2 = arr[:, np.newaxis]
print("newaxis axis=1:", via_newaxis2.shape, "\n", via_newaxis2)

# squeeze removes single-dimensional entries
arr_squeeze = np.array([[[1], [2], [3]]])  # shape (1, 3, 1)
print("\nOriginal shape:", arr_squeeze.shape)
print("Squeezed shape:", arr_squeeze.squeeze().shape)
print("Partial squeeze:", arr_squeeze.squeeze(axis=0).shape)
```
```
# Output:
# Original shape: (3,)
# expand_dims axis=0: (1, 3) [[1 2 3]]
# expand_dims axis=1: (3, 1)
#  [[1]
#  [2]
#  [3]]
#
# newaxis axis=0: (1, 3) [[1 2 3]]
# newaxis axis=1: (3, 1)
#  [[1]
#  [2]
#  [3]]
#
# Original shape: (1, 3, 1)
# Squeezed shape: (3,)
# Partial squeeze: (3, 1)
```

### Example 5: swapaxes

```python
import numpy as np

arr = np.arange(24).reshape(2, 3, 4)
print("Original shape:", arr.shape)
print("Element at [0,1,2]:", arr[0, 1, 2])

# Swap axes 0 and 2
swapped = np.swapaxes(arr, 0, 2)
print("Swapped (0,2) shape:", swapped.shape)
print("Element at [2,1,0]:", swapped[2, 1, 0])  # same value as arr[0,1,2]

# swapaxes is useful for reordering channels in image data
images = np.random.rand(32, 64, 64, 3)  # NHWC format
print("\nImages NHWC:", images.shape)
# Convert to NCHW format
images_nchw = np.swapaxes(images, 1, 3)
images_nchw = np.swapaxes(images_nchw, 2, 3)
print("Images NCHW:", images_nchw.shape)

# Alternative: transpose
images_nchw2 = np.transpose(images, (0, 3, 1, 2))
print("Images NCHW (transpose):", images_nchw2.shape)
```
```
# Output:
# Original shape: (2, 3, 4)
# Element at [0,1,2]: 6
# Swapped (0,2) shape: (4, 3, 2)
# Element at [2,1,0]: 6
#
# Images NHWC: (32, 64, 64, 3)
# Images NCHW: (32, 3, 64, 64)
# Images NCHW (transpose): (32, 3, 64, 64)
```

### Example 6: Resize (Changing Number of Elements)

```python
import numpy as np

arr = np.array([[1, 2],
                [3, 4]])

# resize with more elements (repeats data)
resized_more = np.resize(arr, (3, 4))
print("Original:\n", arr)
print("\nResize to (3, 4) — repeats:\n", resized_more)

# resize with fewer elements (truncates)
resized_less = np.resize(arr, (2, 1))
print("\nResize to (2, 1) — truncates:\n", resized_less)

# In-place resize (modifies the array's shape directly)
arr.resize(1, 4)
print("\nIn-place resize to (1, 4):\n", arr)
```
```
# Output:
# Original:
#  [[1 2]
#  [3 4]]
#
# Resize to (3, 4) — repeats:
#  [[1 2 3 4]
#  [1 2 3 4]
#  [1 2 3 4]]
#
# Resize to (2, 1) — truncates:
#  [[1]
#  [2]]
#
# In-place resize to (1, 4):
#  [[1 2 3 4]]
```

### Example 7: Preparing Data for Neural Networks

```python
import numpy as np

# Simulate image data: 1000 flattened 32x32 grayscale images
flat_images = np.random.randn(1000, 1024)  # 32*32 = 1024

# Reshape back to images for CNN input: (batch, height, width, channels)
images_cnn = flat_images.reshape(-1, 32, 32, 1)
print("Flat input:", flat_images.shape)
print("CNN input:", images_cnn.shape)

# Simulate time series: 500 samples, each with 10 time steps, 3 features
ts_data = np.random.randn(500, 30)  # 10 * 3 = 30
ts_reshaped = ts_data.reshape(-1, 10, 3)
print("\nFlat TS:", ts_data.shape)
print("Reshaped TS (samples, timesteps, features):", ts_reshaped.shape)

# Add feature dimension with newaxis for LSTM
ts_lstm = ts_data.reshape(-1, 10, 3, 1)
print("LSTM input (with extra dim):", ts_lstm.shape)

# Flatten batch of images for fully connected layer
batch_size = 128
images_batch = np.random.randn(batch_size, 3, 64, 64)  # NCHW
fc_input = images_batch.reshape(batch_size, -1)
print(f"\nConv output: {images_batch.shape}")
print(f"FC input: {fc_input.shape}")
```
```
# Output:
# Flat input: (1000, 1024)
# CNN input: (1000, 32, 32, 1)
#
# Flat TS: (500, 30)
# Reshaped TS (samples, timesteps, features): (500, 10, 3)
#
# LSTM input (with extra dim): (500, 10, 3, 1)
#
# Conv output: (128, 3, 64, 64)
# FC input: (128, 12288)
```

## Common Mistakes

1. **Forgetting Total Elements Must Match:** `arr.reshape(3, 5)` fails if `arr` has 12 elements. Total elements must stay the same unless using `resize`.

2. **Confusing `flatten` and `ravel`:** `flatten` always returns a copy. `ravel` returns a view when possible. Modifying the result of `ravel` can affect the original array.

3. **Using `resize` When `reshape` Is Needed:** `resize` can change the total number of elements (by truncation or repetition). `reshape` requires the same number of elements.

4. **Misunderstanding C vs Fortran Order:** `reshape` defaults to C-order (row-major). To fill column-wise, use `order='F'`. This is critical when interfacing with Fortran-based libraries.

5. **Assuming `.T` Works Like `transpose` for ND Arrays:** `.T` reverses all axes for any-dimensional array. For specific axis reordering, use `np.transpose(arr, axes)`.

6. **Adding Unnecessary Dimensions:** Adding singleton dimensions with `newaxis` or `expand_dims` when broadcasting would handle it automatically can waste memory and complicate code.

7. **In-Place Resize Failing:** `arr.resize(new_shape)` works only if the array doesn't share memory with another array. Use `np.resize(arr, new_shape)` for a safe copy-based resize.

## Interview Questions

### Beginner

1. **Q:** How do you change a 1D array of 12 elements into a 3x4 matrix?
   **A:** `arr.reshape(3, 4)` or `arr.reshape(-1, 4)`.

2. **Q:** What is the difference between `flatten` and `ravel`?
   **A:** `flatten` always returns a copy. `ravel` returns a view when possible (more memory efficient).

3. **Q:** How do you transpose a 2D array?
   **A:** `arr.T` or `np.transpose(arr)`.

4. **Q:** What does `-1` mean in `reshape`?
   **A:** `-1` tells NumPy to infer that dimension's size automatically based on the total number of elements and the other specified dimensions.

5. **Q:** How do you add a batch dimension to a 2D array of shape `(64, 64)`?
   **A:** `arr[np.newaxis, :, :]` → shape `(1, 64, 64)` or `np.expand_dims(arr, axis=0)`.

### Intermediate

1. **Q:** What is the difference between `reshape` and `resize`?
   **A:** `reshape` requires the same total number of elements and returns a view if possible. `resize` can change the total number of elements (truncating or repeating data) and always returns a new array.

2. **Q:** Explain C-order vs Fortran-order in reshaping.
   **A:** C-order (row-major) fills the last axis fastest. Fortran-order (column-major) fills the first axis fastest. `arr.reshape(3, 4, order='C')` fills across rows first; `order='F'` fills down columns first.

3. **Q:** How would you convert an image from NHWC (batch, height, width, channels) to NCHW format?
   **A:** `np.transpose(images, (0, 3, 1, 2))` or with `swapaxes`: `np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)`.

4. **Q:** When does `reshape` return a view vs a copy?
   **A:** `reshape` returns a view when the array is contiguous in memory (C-order or F-order). It returns a copy when the array has non-contiguous memory (e.g., after transposition or slicing with steps).

5. **Q:** What does `squeeze` do and when is it useful?
   **A:** `squeeze` removes all single-dimensional entries from shape. It's useful after reduction operations with `keepdims=True` or when reading data that has unnecessary singleton dimensions.

### Advanced

1. **Q:** How does NumPy determine if a `reshape` can return a view, and what is the role of strides in this decision?
   **A:** NumPy checks if the array's memory layout is compatible with the new shape. For C-contiguous arrays, any reshape is a view. For non-contiguous arrays, a view is only possible if the new shape can be represented with the existing strides. Otherwise, a copy is made. The stride information determines how to step through memory for each axis.

2. **Q:** Explain the internal mechanics of `np.transpose` with axis permutation. How does it affect strides without copying data?
   **A:** `np.transpose` creates a new view of the array by reordering the shape and strides tuples without copying any data. For example, transposing a (3, 4) C-contiguous array creates a view with shape (4, 3) and strides swapped (from (4*8, 8) to (8, 4*8) for float64). This is why transposed arrays are not contiguous.

3. **Q:** How would you implement a memory-efficient `flatten` that handles non-contiguous arrays without making a copy?
   **A:** `np.ascontiguousarray(arr).ravel()` first makes a contiguous copy if needed, then returns a view. For truly zero-copy flattening of non-contiguous arrays, you must accept that you get a copy (flatten always copies non-contiguous data). The only way to avoid copying is to iterate in the original layout using strides.

## Practice Problems

### Easy

1. Create a 1D array of 16 elements and reshape it into a 4x4 matrix.

2. Flatten a 3x3 matrix into a 1D array using both `flatten` and `ravel`.

3. Transpose a 2x5 matrix into 5x2.

4. Add a new axis to a 1D array of shape (10,) to make it (10, 1).

5. Remove all singleton dimensions from `np.array([[[1], [2], [3]]])`.

### Medium

1. Create a 3D array of shape (2, 3, 4) and reshape it to (2, 12). Then reshape it back to (2, 3, 4). Verify the data is preserved.

2. Convert a batch of 100 RGB images of size 28x28 from NHWC (100, 28, 28, 3) to NCHW (100, 3, 28, 28) using both `transpose` and `swapaxes`.

3. Given a 1D array of 144 elements, reshape it into all possible 2D shapes (list them) using -1 inference.

4. Create a non-contiguous array by slicing a matrix with a step, then compare the behavior of `flatten` and `ravel` on it.

5. Use `resize` to create a 3x3 array from `[1, 2, 3]` (repeating elements). Then use it to create a 2x2 array (truncating).

### Hard

1. Implement a function that can convert between any two image data formats (NHWC, NCHW, CHW, HWC) given an `src_format` and `dst_format` string.

2. Implement a function `block_reshape(arr, blocksize)` that reshapes a 2D array into blocks (e.g., a (6, 6) array into 4 (3, 3) blocks along the first two axes).

3. Implement a function that rechunks a 3D array: given an array of shape (12, 12, 12), reshape it so that each 4x4x4 block becomes a single element (using reshape and transpose combinations).

## Solutions

### Easy Solutions

```python
# 1
arr = np.arange(16)
reshaped = arr.reshape(4, 4)
print(reshaped)

# 2
mat = np.ones((3, 3))
print("flatten:", mat.flatten())
print("ravel:", mat.ravel())

# 3
mat = np.arange(10).reshape(2, 5)
print("Original:", mat.shape)
print("Transposed:", mat.T.shape)

# 4
arr = np.zeros(10)
expanded = arr[:, np.newaxis]
print(expanded.shape)

# 5
arr = np.array([[[1], [2], [3]]])
print(arr.squeeze())
```

### Medium Solutions

```python
# 1 Roundtrip
arr = np.arange(24).reshape(2, 3, 4)
flat = arr.reshape(2, 12)
back = flat.reshape(2, 3, 4)
print("Data preserved:", np.allclose(arr, back))

# 2 NHWC to NCHW
batch = np.random.rand(100, 28, 28, 3)
nchw_t = np.transpose(batch, (0, 3, 1, 2))
nchw_s = np.swapaxes(np.swapaxes(batch, 1, 3), 2, 3)
# Actually simpler:
nchw_s2 = batch.transpose(0, 3, 1, 2)
print("NCHW shape:", nchw_t.shape)
print("Match:", np.allclose(nchw_t, nchw_s2))

# 3 All 2D shapes
arr = np.arange(144)
factors = [(i, 144//i) for i in range(1, 145) if 144 % i == 0]
for r, c in factors:
    print(f"({r}, {c})", end=" ")
print(f"\nTotal: {len(factors)} shapes")

# 4 Non-contiguous ravel vs flatten
mat = np.arange(16).reshape(4, 4)
slice_view = mat[::2, ::2]
print("Slice shape:", slice_view.shape)
print("Slice is contiguous?", slice_view.flags.c_contiguous)
r = slice_view.ravel()
r[0] = 999
print("Original modified via ravel?", mat[0, 0] == 999)  # May not modify!

# 5 Resize
arr = np.array([1, 2, 3])
big = np.resize(arr, (3, 3))
small = np.resize(arr, (2, 2))
print("Big:\n", big)
print("Small:\n", small)
```

### Hard Solutions

```python
# 1 Format converter
def convert_image_format(img, src, dst):
    axes = {'N': 0, 'C': 1, 'H': 2, 'W': 3} if len(src) == 4 else {'C': 0, 'H': 1, 'W': 2}
    src_axes = [axes[c] for c in src]
    dst_axes = [axes[c] for c in dst]
    # Map current positions to positions in src
    perm = [src_axes.index(i) for i in range(len(src))]
    # Reorder to standard (src order), then to dst
    # Simpler: create permutation that maps src order to dst order
    perm = [src.find(c) for c in dst]
    return np.transpose(img, perm)

img_nhwc = np.random.rand(10, 32, 32, 3)
img_nchw = convert_image_format(img_nhwc, 'NHWC', 'NCHW')
print("NHWC -> NCHW:", img_nchw.shape)

# 2 Block reshape
def block_reshape(arr, blocksize):
    h, w = arr.shape
    bh, bw = blocksize
    return arr.reshape(h//bh, bh, w//bw, bw).transpose(0, 2, 1, 3)

arr = np.arange(36).reshape(6, 6)
blocks = block_reshape(arr, (3, 3))
print("Block shape:", blocks.shape)
print("First block:\n", blocks[0, 0])

# 3 Rechunk 3D array
arr = np.arange(12**3).reshape(12, 12, 12)
bs = 4
rechunked = (arr.reshape(12//bs, bs, 12//bs, bs, 12//bs, bs)
             .transpose(0, 2, 4, 1, 3, 5)
             .reshape(3, 3, 3, -1))
print("Rechunked shape:", rechunked.shape)
block_000 = rechunked[0, 0, 0].reshape(bs, bs, bs)
print("Block (0,0,0) corner:", block_000[0, 0, :4])
```

## Related Concepts

- NumPy broadcasting rules
- Array strides and memory layout
- Tensor reshaping in PyTorch/TensorFlow
- Einstein summation (np.einsum)

## Next Concepts

- Concatenation and splitting (PYT-073)
- Structured arrays (PYT-075)
- Universal functions (PYT-074)

## Summary

Reshaping operations are fundamental for preparing data in the correct format for computation. `reshape` changes dimensions (same data count), `resize` can add/remove elements, `flatten`/`ravel` convert to 1D, `transpose`/`.T` reorder axes, `expand_dims`/`squeeze`/`newaxis` manage singleton dimensions, and `swapaxes` exchanges two axes. Understanding when operations return views vs copies is critical for memory management.

## Key Takeaways

- `reshape(-1, n)` auto-infers one dimension
- `flatten` returns a copy; `ravel` returns a view (memory efficient)
- `.T` reverses all axes; `transpose(arr, axes)` for custom ordering
- `np.newaxis` and `expand_dims` add singleton dimensions
- `squeeze` removes all size-1 dimensions
- `resize` can change total element count (with truncation/repetition)
- C-order (row-major) vs F-order (column-major) determines fill direction
- Non-contiguous arrays (after transpose/step slicing) copy on reshape
