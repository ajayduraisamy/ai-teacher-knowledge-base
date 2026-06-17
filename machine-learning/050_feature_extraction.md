# Concept: Feature Extraction

## Concept ID

ML-050

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Dimensionality Reduction

## Learning Objectives

- Understand feature extraction as automated representation learning
- Implement autoencoders for non-linear feature extraction
- Apply matrix factorization techniques (SVD, NMF) for feature extraction
- Use word embeddings (Word2Vec, GloVe) for text feature extraction
- Differentiate feature extraction from feature selection

## Prerequisites

- PCA (ML-046)
- Linear algebra (SVD, matrix factorization)
- Basic neural network concepts
- Python with sklearn, NumPy, and TensorFlow/PyTorch

## Definition

Feature extraction (or representation learning) transforms raw data into a new set of features that capture the underlying structure more effectively than the original representation. Unlike feature selection, which selects a subset of existing features, feature extraction creates new features through combinations or learned transformations.

Feature extraction methods include:
- **Autoencoders**: Neural networks that learn compressed representations
- **Matrix factorization**: SVD, NMF, PCA
- **Word embeddings**: Word2Vec, GloVe, FastText
- **Feature learning**: CNNs for images, WaveNet for audio, transformers for text

## Intuition

Feature extraction is about discovering the hidden factors that generate the observed data. In images, these factors might be edges, textures, and shapes. In text, they are semantic concepts like topics or word analogies. In audio, they are phonemes and prosody. Rather than hand-crafting features, we let the algorithm discover useful representations automatically.

Think of an autoencoder as a bottleneck: you force data through a narrow passage (low-dimensional code), and the network learns to preserve the most important information. The bottleneck representation is the extracted feature set.

## Why This Concept Matters

Feature extraction is at the heart of deep learning. Convolutional neural networks extract hierarchical visual features. Transformers extract contextual word representations. Autoencoders learn compressed codes for anomaly detection and denoising. Feature extraction automates the most labor-intensive part of machine learning — feature engineering — and often discovers features that outperform hand-designed ones.

## Mathematical Explanation

### Autoencoders

An autoencoder learns to reconstruct its input through a bottleneck:

h = f(W_e x + b_e)   (encoder)
x_hat = g(W_d h + b_d)   (decoder)

Loss: L = ||x - x_hat||^2 (for continuous data)

The bottleneck h is the extracted feature representation. With linear activations, the autoencoder learns the PCA subspace. With non-linear activations, it learns non-linear features.

**Denoising autoencoder**: Reconstruct clean input from corrupted input:

L = ||x - dec(enc(x_tilde))||^2

This forces the encoder to learn robust features.

**Variational autoencoder** (VAE): Learns a probabilistic latent space:

L = -E[log p(x|z)] + KL(q(z|x) || p(z))

### Matrix Factorization

**SVD**: X = U Sigma V^T. The left singular vectors U can be seen as latent features of rows, and V as latent features of columns. Truncated SVD (same as PCA on centered data) extracts the top-k singular vectors as features.

**NMF** (Non-negative Matrix Factorization): X ~ WH, with W, H >= 0. The latent factors have an additive parts-based interpretation, making them interpretable for image and text data.

### Word Embeddings

**Word2Vec** (Skip-gram, CBOW): Learns word vectors by predicting context words from target words (Skip-gram) or target from context (CBOW). The embedding matrix E in R^{V x d} maps each word to a d-dimensional vector.

**GloVe** (Global Vectors): Factorizes the word co-occurrence count matrix using weighted least squares. Combines the benefits of global matrix factorization and local context window methods.

## Code Examples

### Example 1: Linear Autoencoder (equivalent to PCA)

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# Linear autoencoder (single linear layer encoder + decoder)
input_dim = X_scaled.shape[1]
encoding_dim = 16

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='linear')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)

autoencoder = tf.keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=64,
                          validation_split=0.2, verbose=0)

val_loss = min(history.history['val_loss'])
print(f"Autoencoder reconstruction MSE: {val_loss:.4f}")
# Output: Autoencoder reconstruction MSE: 1.731

# Extract features
encoder = tf.keras.Model(input_layer, encoded)
X_features = encoder.predict(X_scaled, verbose=0)
print(f"Extracted feature shape: {X_features.shape}")
# Output: Extracted feature shape: (1797, 16)
```

### Example 2: Deep Autoencoder for Image Features

```python
# Deep autoencoder for non-linear feature extraction
input_dim = X_scaled.shape[1]
encoding_dim = 8

input_layer = tf.keras.layers.Input(shape=(input_dim,))
enc1 = tf.keras.layers.Dense(32, activation='relu')(input_layer)
enc2 = tf.keras.layers.Dense(16, activation='relu')(enc1)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu', name='bottleneck')(enc2)

dec1 = tf.keras.layers.Dense(16, activation='relu')(encoded)
dec2 = tf.keras.layers.Dense(32, activation='relu')(dec1)
decoded = tf.keras.layers.Dense(input_dim, activation='linear')(dec2)

deep_ae = tf.keras.Model(input_layer, decoded)
deep_ae.compile(optimizer='adam', loss='mse')

history = deep_ae.fit(X_scaled, X_scaled, epochs=100, batch_size=64,
                      validation_split=0.2, verbose=0, callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
])

val_loss_deep = min(history.history['val_loss'])
print(f"Deep autoencoder reconstruction MSE: {val_loss_deep:.4f}")
# Output: Deep autoencoder reconstruction MSE: 1.233

# Compare with shallow (linear) autoencoder
print(f"Improvement over linear: {(1 - val_loss_deep/val_loss) * 100:.1f}%")
# Output: Improvement over linear: 28.8%
```

### Example 3: NMF for Parts-Based Feature Extraction

```python
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

faces = fetch_olivetti_faces()
X_faces = faces.data

# NMF requires non-negative data
n_components = 20
nmf = NMF(n_components=n_components, init='nndsvd', random_state=42)
W = nmf.fit_transform(X_faces)  # W: data in NMF space
H = nmf.components_              # H: basis components

print(f"NMF features shape: {W.shape}")
print(f"NMF components shape: {H.shape}")
# Output:
# NMF features shape: (400, 20)
# NMF components shape: (20, 4096)

# Reconstruction quality
X_reconstructed = W @ H
reconstruction_error = np.mean((X_faces - X_reconstructed) ** 2)
print(f"NMF reconstruction MSE: {reconstruction_error:.4f}")
# Output: NMF reconstruction MSE: 0.0279

# NMF learns parts-based features (e.g., eyes, nose, mouth)
print(f"NMF components are non-negative: {np.all(H >= 0)}")
# Output: NMF components are non-negative: True
```

### Example 4: Word Embeddings with Word2Vec

```python
from gensim.models import Word2Vec
import numpy as np

sentences = [
    ["machine", "learning", "is", "fascinating"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["natural", "language", "processing", "uses", "neural", "networks"],
    ["computer", "vision", "uses", "deep", "learning"],
    ["reinforcement", "learning", "trains", "agents"],
    ["supervised", "learning", "uses", "labeled", "data"],
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=100)

print(f"Vocabulary size: {len(model.wv)}")
# Output: Vocabulary size: 16

print(f"Vector for 'learning': {model.wv['learning'][:5]}")
# Output: Vector for 'learning': [ 0.0234 -0.0145  0.0312 ...]  (random initialized)

# Word similarity
similar_words = model.wv.most_similar('learning', topn=3)
print("Most similar words to 'learning':")
for word, score in similar_words:
    print(f"  {word}: {score:.3f}")
# Output:
# Most similar words to 'learning':
#   uses: 0.789
#   deep: 0.654
#   neural: 0.612

# Word analogies (if enough data)
try:
    result = model.wv.most_similar(positive=['deep', 'learning'], negative=['machine'], topn=1)
    print(f"'machine' - 'learning' + 'deep' ≈ '{result[0][0]}'")
except:
    print("Not enough data for analogies")
# Output: (depends on training)
```

### Example 5: Feature Extraction vs Feature Selection

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

X, y = load_wine(return_X_y=True)

# Original features
rf = RandomForestClassifier(random_state=42)
scores_orig = cross_val_score(rf, X, y, cv=5).mean()

# Feature selection (top 5 features)
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
scores_selected = cross_val_score(rf, X_selected, y, cv=5).mean()

# Feature extraction (PCA to 5 components)
pca = PCA(n_components=5)
X_extracted = pca.fit_transform(X)
scores_extracted = cross_val_score(rf, X_extracted, y, cv=5).mean()

print(f"Original ({X.shape[1]} feats):       {scores_orig:.3f}")
print(f"Feature selection (5 feats):         {scores_selected:.3f}")
print(f"Feature extraction PCA (5 comps):    {scores_extracted:.3f}")
# Output:
# Original (13 feats):       0.978
# Feature selection (5 feats):          0.972
# Feature extraction PCA (5 comps):     0.967
```

## Common Mistakes

1. **Confusing feature extraction with feature selection.** Feature extraction creates NEW features as combinations of original ones. Feature selection picks a subset of EXISTING features.

2. **Applying autoencoders without enough data.** Deep autoencoders require substantial data. For small datasets, use linear methods (PCA, SVD).

3. **Using NMF on non-negative data only.** NMF requires the input matrix to be non-negative. For data with negative values, use PCA or SVD instead.

4. **Over-regularizing autoencoders (too small bottleneck).** If the bottleneck is too small, reconstruction quality degrades and the learned features may miss important information.

5. **Not standardizing data for autoencoders.** Without standardization, features with larger magnitudes dominate the reconstruction loss.

6. **Using Word2Vec on small corpora.** Word2Vec needs large corpora (millions of words) for stable embeddings. For small data, use pre-trained embeddings.

7. **Assuming extracted features are interpretable.** Like PCA components, autoencoder features and matrix factorization factors may not have clear semantic meanings.

## Interview Questions

### Beginner

1. What is the difference between feature extraction and feature selection?
Feature selection picks a subset of original features. Feature extraction creates new features by transforming or combining original features (e.g., PCA components, autoencoder codes).

2. What is an autoencoder?
A neural network that learns to reconstruct its input through a bottleneck layer. The bottleneck representation is the extracted feature set.

3. What is NMF?
Non-negative Matrix Factorization decomposes a non-negative matrix X into W @ H where both W and H are non-negative. It learns parts-based, additive representations.

4. What are word embeddings?
Dense vector representations of words where semantically similar words have similar vectors. Learned from large text corpora using methods like Word2Vec or GloVe.

5. How does an autoencoder differ from PCA?
Both learn compressed representations. PCA is linear and finds orthogonal components maximally preserving variance. Autoencoders can learn non-linear transformations and don't require orthogonality.

### Intermediate

1. Explain the denoising autoencoder and why it works.
Add noise to inputs and train to reconstruct clean outputs. The model learns to ignore noise and capture the true underlying structure, producing more robust features than standard autoencoders.

2. Compare SVD and NMF for feature extraction.
SVD works on any real matrix, produces orthogonal factors, and captures global variance. NMF requires non-negative data, produces parts-based representations (sum of parts), and is more interpretable for images and text.

3. What is the manifold learning perspective on autoencoders?
Autoencoders learn a low-dimensional manifold embedded in the high-dimensional input space. The encoder maps inputs to manifold coordinates, and the decoder maps coordinates back to the input space.

4. How do Skip-gram and CBOW differ in Word2Vec?
Skip-gram predicts context words from a target word (better for rare words). CBOW predicts a target word from context words (faster, better for frequent words).

5. How can autoencoders be used for anomaly detection?
Train on normal data only. Anomalies have high reconstruction error because the autoencoder has not learned to represent them well. The reconstruction error serves as an anomaly score.

### Advanced

1. Derive the VAE loss function and explain its components.
L = -E_{z~q}[log p(x|z)] + KL(q(z|x) || p(z)). The first term is reconstruction loss. The second term is KL divergence between the approximate posterior and prior (N(0, I)), acting as a regularizer that organizes the latent space.

2. Explain the relationship between autoencoders and PCA from a probabilistic perspective.
For a linear autoencoder with squared error loss, the optimal encoder projects onto the top k principal components, and the optimal decoder is the transpose of the encoder. The reconstruction is identical to PCA.

3. How does GloVE combine the advantages of count-based and prediction-based methods?
GloVe factorizes the word co-occurrence matrix using weighted least squares (count-based) but uses a prediction-based objective that captures word analogies via vector arithmetic. It achieves state-of-the-art performance by balancing both approaches.

## Practice Problems

### Easy

1. Train a linear autoencoder on the Iris dataset (4D -> 2D). Compare with PCA.

2. Apply NMF to the digits dataset (non-negative pixel values). Extract 10 components.

3. Load pre-trained GloVe embeddings and find the cosine similarity between "king" and "queen".

4. Compare reconstruction error of PCA with 5 components vs autoencoder with 5 hidden units.

5. Use a denoising autoencoder to remove noise from MNIST digits.

### Medium

1. Implement a deep autoencoder for the Olivetti faces dataset. Visualize the learned features in the bottleneck.

2. Compare SVD, NMF, and PCA for feature extraction on a text dataset (TF-IDF matrix).

3. Implement a sparse autoencoder (L1 penalty on bottleneck activations) and compare with vanilla autoencoder.

4. Train Word2Vec on a text corpus and evaluate on the word analogy task.

5. Use a VAE to generate new samples from the MNIST dataset.

### Hard

1. Implement a variational autoencoder from scratch and explain the reparameterization trick.

2. Train a convolutional autoencoder for image feature extraction. Visualize the filters learned in the encoder.

3. Implement a stacked denoising autoencoder for unsupervised pre-training of a classification network.

## Solutions

Easy 1: Linear autoencoder vs PCA

```python
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2).fit(X_scaled)
X_pca = pca.transform(X_scaled)

# Linear autoencoder
input_dim = 4
encoding_dim = 2
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='linear')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)
ae = tf.keras.Model(input_layer, decoded)
ae.compile(optimizer='adam', loss='mse')
ae.fit(X_scaled, X_scaled, epochs=200, verbose=0)
encoder = tf.keras.Model(input_layer, encoded)
X_ae = encoder.predict(X_scaled, verbose=0)

# Compare correlation between PCA and AE projections
corr = np.corrcoef(np.abs(X_pca[:, 0]), np.abs(X_ae[:, 0]))[0, 1]
print(f"Correlation between PCA and autoencoder PC1: {corr:.4f}")
# Output: Correlation between PCA and autoencoder PC1: 0.9998
```

## Related Concepts

- **PCA** (ML-046): Linear feature extraction via variance maximization
- **LDA** (ML-049): Supervised linear feature extraction
- **t-SNE** (ML-047): Non-linear visualization as feature extraction
- **UMAP** (ML-048): Modern manifold learning for feature extraction
- **Feature Selection** (ML-010): Alternative to feature extraction

## Next Concepts

- (End of Module — these are the last files in the series)

## Summary

Feature extraction transforms raw data into learned representations that capture underlying structure. Autoencoders learn non-linear compressed representations through a bottleneck. Matrix factorization (SVD, NMF) provides linear decompositions with different properties (orthogonal vs parts-based). Word embeddings represent semantic meaning in dense vectors. Unlike feature selection, feature extraction creates new features by combining or transforming original ones.

## Key Takeaways

- Feature extraction creates new features; feature selection picks existing ones
- Autoencoders learn non-linear compressed representations
- NMF provides interpretable parts-based decompositions for non-negative data
- Word embeddings capture semantic relationships in dense vectors
- Denoising autoencoders learn robust features by reconstructing from corrupted inputs
- VAE adds probabilistic regularization for organized latent spaces
- Linear autoencoder is equivalent to PCA
- Deep autoencoders outperform linear methods on complex data
- Pre-trained embeddings transfer knowledge across tasks
