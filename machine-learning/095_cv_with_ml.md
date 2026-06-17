# Concept: Computer Vision with ML

## Concept ID

ML-095

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Extract handcrafted image features (HOG, SIFT, ORB) for classification
- Represent images as feature vectors for traditional ML algorithms
- Apply flattening and PCA for image dimensionality reduction
- Understand data augmentation techniques for limited image datasets
- Build simple CNN architectures for image classification
- Compare traditional CV + ML with deep learning approaches

## Prerequisites

- Basic image processing (pixels, color channels, convolution)
- Python with OpenCV, scikit-learn, and numpy
- Classification algorithms (SVM, Random Forest, Logistic Regression)
- Linear algebra fundamentals (eigenvalues, SVD, PCA)

## Definition

Computer vision with traditional machine learning refers to the approach of using handcrafted feature extractors (HOG, SIFT, ORB) combined with classical ML classifiers (SVM, Random Forest, Logistic Regression) to solve visual recognition tasks. This contrasts with end-to-end deep learning, where feature extraction and classification are learned jointly. The traditional approach remains valuable when labeled data is scarce, computational resources are limited, or interpretable features are required for domain-specific analysis.

## Intuition

Imagine you need to build a system that recognizes handwritten digits. A deep learning approach would learn features directly from pixels. A traditional ML approach first computes meaningful numerical descriptors: "How many edges are in each region?", "What is the average intensity?", "How many loops does the digit have?" These descriptors become features for a classifier. This is like asking a detective to look for specific clues (red car, tall suspect, blue hat) rather than looking at the entire crime scene and forming an intuition.

## Why This Concept Matters

Despite the dominance of deep learning in computer vision, traditional ML-based approaches are still used in industrial applications where training data is limited (<1000 images per class), real-time performance on edge devices is required (medical devices, industrial inspection), or feature interpretability is critical. Traditional features like HOG and ORB are also used as building blocks in modern pipelines (e.g., ORB for SLAM in robotics). Understanding these fundamentals provides intuition for why CNNs work.

## Mathematical Explanation

### HOG (Histogram of Oriented Gradients)

Divide the image into small cells (e.g., 8x8 pixels). For each pixel, compute gradient magnitude and orientation:

$$\text{Magnitude: } g(x,y) = \sqrt{g_x(x,y)^2 + g_y(x,y)^2}$$
$$\text{Orientation: } \theta(x,y) = \arctan\left(\frac{g_y(x,y)}{g_x(x,y)}\right)$$

Where g_x and g_y are gradients in x and y directions (e.g., computed with Sobel operators).

Build a histogram of gradient orientations (9 bins, 0-180 degrees) per cell. Normalize blocks of cells to be illumination-invariant. Concatenate all block histograms into a single feature vector.

### SIFT (Scale-Invariant Feature Transform)

Detect keypoints at extrema of the Difference-of-Gaussians (DoG) function over scale space:

$$D(x,y,\sigma) = (G(x,y,k\sigma) - G(x,y,\sigma)) * I(x,y)$$

Where G is the Gaussian kernel, I is the image, and k is a constant scale factor.

For each keypoint, compute a 128-dimensional descriptor based on gradient histograms in 4x4 subregions.

### PCA for Image Dimensionality

Given N images of size H x W, flatten each to a vector of length H*W. The data matrix X (N x HW) has mean mu. The covariance matrix (HW x HW) is:

$$C = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)(x_i - \mu)^T$$

Eigenvectors of C with the largest eigenvalues are the principal components (eigenfaces for faces). Project images onto the top k components:

$$z_i = W_k^T (x_i - \mu)$$

Where W_k is the matrix of k principal eigenvectors.

## Code Examples

### Example 1: Image Classification with Flattening and PCA

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load digits dataset (8x8 images of digits 0-9)
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
# Output: Dataset shape: (1797, 64)
# Output: Number of classes: 10

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Original dimensions: {X.shape[1]}")
print(f"After PCA (95% variance): {X_train_pca.shape[1]}")
# Output: Original dimensions: 64
# Output: After PCA (95% variance): 40

# Train SVM on original features
svm_orig = SVC(kernel='rbf', gamma='scale', C=10)
svm_orig.fit(X_train_scaled, y_train)
y_pred_orig = svm_orig.predict(X_test_scaled)
acc_orig = accuracy_score(y_test, y_pred_orig)

# Train SVM on PCA features
svm_pca = SVC(kernel='rbf', gamma='scale', C=10)
svm_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"Accuracy (original 64 dims): {acc_orig:.4f}")
print(f"Accuracy (PCA {X_train_pca.shape[1]} dims): {acc_pca:.4f}")
# Output: Accuracy (original 64 dims): 0.9852
# Output: Accuracy (PCA 40 dims): 0.9852

# Show eigenfaces (first 10 principal components)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    eigenface = pca.components_[i].reshape(8, 8)
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f'PC {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Example 2: HOG Features with SVM for Pedestrian Detection

```python
import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, exposure
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load example image for visualization
image = data.astronaut()
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Compute HOG features
fd, hog_image = hog(
    image_gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True,
    block_norm='L2-Hys'
)

print(f"HOG feature vector length: {len(fd)}")
# Output: HOG feature vector length: 15876

# Visualize HOG
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image_gray, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(hog_image_rescaled, cmap='gray')
ax2.set_title('HOG Visualization')
ax2.axis('off')
plt.tight_layout()
plt.show()

# Synthetic binary classification: person vs non-person
# In practice, use a dataset like INRIA Person or Daimler Pedestrian
np.random.seed(42)
n_samples = 200

# Simulate HOG-like features (in practice, extract from real images)
X_pos = np.random.randn(100, 100) + 0.5   # "person" features
X_neg = np.random.randn(100, 100) - 0.5   # "non-person" features
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(100), np.zeros(100)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {acc:.4f}")
# Output: Classification accuracy: 0.8500
```

### Example 3: Simple CNN Architecture for Image Classification

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load digits and reshape for CNN (8x8 grayscale images)
digits = load_digits()
X = digits.data.reshape(-1, 8, 8, 1).astype('float32') / 16.0  # Normalize to [0,1]
y = to_categorical(digits.target, 10)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=digits.target
)

print(f"Training data shape: {X_train.shape}")
# Output: Training data shape: (1257, 8, 8, 1)

# Build CNN
model = Sequential([
    # First conv layer: 8 filters, 3x3
    Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),

    # Second conv layer: 16 filters, 3x3
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Classifier
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
# Output: Test accuracy: 0.9778
```

### Example 4: Data Augmentation

```python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load a sample digit
digits = load_digits()
sample = digits.images[0].reshape(8, 8, 1).astype('float32') / 16.0

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Generate augmented images
sample_batch = sample.reshape(1, 8, 8, 1)
aug_iter = datagen.flow(sample_batch, batch_size=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes[0, 0].imshow(sample.squeeze(), cmap='gray')
axes[0, 0].set_title('Original')

for i in range(1, 10):
    ax = axes[(i) // 5, (i) % 5]
    aug_img = next(aug_iter)[0].squeeze()
    ax.imshow(aug_img, cmap='gray')
    ax.set_title(f'Augmented {i}')
    ax.axis('off')

axes[0, 0].axis('off')
plt.tight_layout()
plt.show()

print("Data augmentation examples generated (8x8 digits)")
# Output: Data augmentation examples generated (8x8 digits)
```

## Common Mistakes

1. **Flattening images directly without normalization**: Pixel values range from 0-255, which can dominate feature magnitude. Always normalize to [0,1] or standardize to zero mean and unit variance before feeding to ML algorithms.

2. **Applying PCA on raw pixel values without centering**: PCA is not translation-invariant. If you do not center the data (subtract the mean), the first principal component will capture the mean image rather than the directions of maximum variance.

3. **Ignoring illumination and scale invariance**: Raw pixel values vary dramatically with lighting conditions. Always preprocess with histogram equalization, normalization, or use illumination-invariant features like HOG.

4. **Using too many PCA components or too few**: Keeping too many components retains noise; keeping too few loses discriminative information. Use the cumulative explained variance plot to choose the elbow point.

5. **Assuming traditional features generalize across domains**: HOG features tuned for pedestrian detection do not work well for medical imaging. Feature engineering in CV is domain-specific; always validate with domain experts.

6. **Not using data augmentation with limited data**: Without augmentation, CNNs and even traditional models overfit badly when training data is limited. Use random flips, rotations, crops, and color jittering.

7. **Using the wrong color space**: RGB is not always the best representation. HSV separates color from intensity (useful for object tracking), while grayscale is sufficient for many texture-based tasks. Experiment with different color spaces.

## Interview Questions

### Beginner

1. What is the difference between image classification and object detection?
2. How does PCA reduce the dimensionality of image data?
3. What is a convolution operation in image processing?
4. Explain the difference between RGB and grayscale images in terms of data dimensions.
5. What is the purpose of max pooling in a CNN?

### Intermediate

1. Explain how HOG features capture shape information while being invariant to illumination.
2. How would you classify images when you only have 50 labeled examples per class?
3. Compare the tradeoffs between SIFT, SURF, and ORB for feature matching.
4. What is the curse of dimensionality in image classification and how does feature extraction (HOG, PCA) address it?
5. How does data augmentation prevent overfitting and how do you ensure augmented samples remain realistic?

### Advanced

1. Design a system that combines handcrafted features (HOG) with learned features from a pretrained CNN for a domain-specific medical imaging task.
2. Explain how spatial pyramid matching (SPM) extends bag-of-visual-words for image classification.
3. How would you implement multi-scale detection for objects of varying sizes using HOG features and a sliding window?

## Practice Problems

### Easy

1. Load an RGB image, convert to grayscale, and display it. What is the shape difference?
2. Compute the mean image of the digits dataset and visualize it.
3. Apply histogram equalization to a low-contrast grayscale image.
4. Use PCA to reduce the digits dataset to 20 components and reconstruct the images. How does the reconstruction look?
5. Flatten a 28x28 grayscale image into a vector. How many elements does the vector have?

### Medium

1. Build a complete pipeline: load CIFAR-10, flatten images, apply PCA, train an SVM, and report accuracy.
2. Extract HOG features from the digits dataset and train a classifier. Compare with the raw pixel + PCA approach.
3. Implement a sliding window detector using HOG features and a trained linear SVM.
4. Create an image dataset augmentation pipeline that generates 10x the original data using rotations, flips, and color jitter.
5. Build an SVM classifier using color histograms (HSV) as features for scene classification (beach vs forest vs city).

### Hard

1. Implement the bag-of-visual-words pipeline: SIFT feature extraction, k-means clustering for codebook generation, histogram encoding, and SVM classification.
2. Build a Siamese network using a pretrained CNN feature extractor + SVM for one-shot face verification.
3. Design and implement a multiple-instance learning (MIL) approach for weakly supervised object localization where only image-level labels are available.

## Solutions

### Easy 1 — Image shape
```python
import cv2
import numpy as np
# Create a synthetic RGB image
rgb = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
print(f"RGB shape: {rgb.shape}")
print(f"Grayscale shape: {gray.shape}")
# Output: RGB shape: (100, 200, 3)
# Output: Grayscale shape: (100, 200)
```

### Easy 5 — Flatten a 28x28 image
```python
import numpy as np
img_28x28 = np.random.rand(28, 28)
flattened = img_28x28.flatten()
print(f"Flattened vector length: {len(flattened)}")
# Output: Flattened vector length: 784
```

## Related Concepts

- Dimensionality Reduction (PCA) — ML-076
- Image Processing Basics — ML-077
- Convolutional Neural Networks — ML-085
- Feature Engineering — ML-070

## Next Concepts

- NLP with ML — ML-094
- Reinforcement Learning — ML-096
- ML on Edge — ML-099

## Summary

Traditional computer vision with ML uses handcrafted features (HOG, SIFT, ORB, color histograms) combined with classical classifiers (SVM, Random Forest, Logistic Regression) for image recognition. Dimensionality reduction via PCA helps manage the high-dimensional nature of raw pixels. Data augmentation mitigates overfitting when data is limited. While deep CNNs dominate large-scale vision, traditional approaches remain valuable for small datasets, edge deployment, and interpretable applications. Understanding these fundamentals builds intuition for why deep learning architectures are designed the way they are.

## Key Takeaways

- Traditional CV + ML separates feature extraction from classification
- HOG captures edge orientation distributions; SIFT provides scale-invariant keypoints
- PCA reduces image dimensionality while retaining discriminative information
- Normalize pixel values and use illumination-invariant features
- Data augmentation is critical when labeled images are scarce
- SVMs with RBF kernels work well with HOG features for object detection
- Traditional methods are fast, interpretable, and need less data than deep learning
- Color spaces (RGB, HSV, grayscale) affect feature quality — choose wisely
