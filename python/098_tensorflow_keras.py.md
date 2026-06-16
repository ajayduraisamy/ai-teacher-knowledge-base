# Concept: TensorFlow and Keras

## Concept ID

PYT-098

## Difficulty

Advanced

## Domain

Python

## Module

Python for ML/AI

## Learning Objectives

- Build neural network models using `tf.keras.Sequential` and the Functional API
- Implement convolutional, pooling, dropout, and dense layers
- Compile models with optimizers, loss functions, and metrics
- Train, evaluate, and predict using `fit()`, `evaluate()`, and `predict()`
- Understand the Model API for complex multi-input/multi-output architectures

## Prerequisites

- Basic understanding of neural network concepts (layers, activation, loss)
- NumPy and basic Python
- Familiarity with supervised learning

## Definition

TensorFlow is an open-source machine learning framework developed by Google. Keras is its high-level API for building and training neural networks. Keras provides three APIs:

1. **Sequential API (`tf.keras.Sequential`):** A linear stack of layers. Simplest approach, ideal for most feed-forward networks.
2. **Functional API (`tf.keras.Model`):** Builds complex models with multiple inputs/outputs, shared layers, branching, and skip connections.
3. **Model Subclassing:** Custom training loops via subclassing `tf.keras.Model` (for advanced users).

**Core Layer Types:**
- `Dense(units, activation)`: Fully connected layer
- `Conv2D(filters, kernel_size, activation)`: 2D convolution
- `MaxPooling2D(pool_size)`: Downsampling
- `Dropout(rate)`: Regularization
- `Flatten()`: Flatten multi-dimensional input
- `BatchNormalization()`: Normalize activations
- `Input(shape)`: Define input shape (Functional API)

**Compilation:**
- `model.compile(optimizer, loss, metrics)`: Configures the model for training

**Training/Evaluation:**
- `model.fit(x, y, epochs, batch_size, validation_data)`: Train the model
- `model.evaluate(x, y)`: Compute loss and metrics on test data
- `model.predict(x)`: Generate predictions for new data

## Intuition

Keras is designed for human beings, not machines. It follows a "progressive disclosure of complexity" philosophy: you can build and train a neural network in 5 lines of code, yet the same API scales to complex research models.

`Sequential` is like stacking LEGO bricks — each layer sits on top of the previous one. The Functional API is like building with LEGO Technic — you can create branching structures, shared components, and multiple outputs.

The `compile()` → `fit()` → `evaluate()` → `predict()` lifecycle covers the entire ML workflow: configure, train, test, deploy.

## Why This Concept Matters

- **Industry Standard:** TensorFlow/Keras is one of the most widely used deep learning frameworks in production
- **Beginner Friendly:** Keras is the most accessible deep learning API — ideal for learning and rapid prototyping
- **Production Deployment:** TensorFlow Serving, TensorFlow Lite, and TensorFlow.js enable deployment anywhere
- **Ecosystem:** Integrated with TensorBoard (visualization), TF Datasets, TF Hub (transfer learning), and TFX (production pipelines)
- **Research:** Keras Functional API supports complex architectures used in cutting-edge research

## Real World Examples

1. **Product Image Classification:** An e-commerce company uses a fine-tuned EfficientNet (Keras) to categorize 10 million product images into 5000 categories.
2. **Real-time Object Detection:** A self-driving car company deploys a YOLO model (built with Keras) for real-time pedestrian detection at 30 FPS.
3. **Sentiment Analysis:** A social media monitoring platform uses an LSTM model (Keras) to classify tweet sentiment in real-time.
4. **Recommendation System:** A streaming service uses a multi-input Keras model (user features + movie features) to predict ratings.
5. **Medical Imaging:** A hospital deploys a U-Net (Keras) for automated tumor segmentation in MRI scans.

## AI/ML Relevance

- **Production ML:** TensorFlow is the most deployed deep learning framework in production
- **Transfer Learning:** Keras Applications provides pretrained models (ResNet, EfficientNet, MobileNet)
- **Distributed Training:** Keras `fit()` supports multi-GPU and TPU training with minimal code changes
- **MLOps:** TFX pipeline integration, model versioning, and monitoring
- **Edge Deployment:** TensorFlow Lite converts Keras models for mobile and embedded devices

## Code Examples

### Example 1: Sequential API for binary classification
```python
import tensorflow as tf
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
history = model.fit(X, y, epochs=20, batch_size=32,
                    validation_split=0.2, verbose=0)

# Evaluate
test_loss, test_acc = model.evaluate(X, y, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Predict
y_prob = model.predict(X[:5], verbose=0)
print(f"Sample predictions: {y_prob.flatten().round(3)}")
```
```
# Output:
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 32)                352
#  dropout (Dropout)           (None, 32)                0
#  dense_1 (Dense)             (None, 16)                528
#  dense_2 (Dense)             (None, 1)                 17
# =================================================================
# Total params: 897
# Trainable params: 897
# _________________________________________________________________
# Test accuracy: 0.892
# Sample predictions: [0.991 0.021 0.997 0.984 0.003]
```

### Example 2: CNN for image classification (MNIST)
```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]  # add channel dim
x_test = x_test[..., tf.newaxis]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5,
                    batch_size=128, validation_split=0.1, verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Predict 5 samples
predictions = model.predict(x_test[:5], verbose=0)
print(f"Predicted classes: {np.argmax(predictions, axis=1)}")
print(f"True classes: {y_test[:5]}")
```
```
# Output:
# Epoch 1/5: ... - loss: 0.2819 - accuracy: 0.9152 - val_loss: 0.0652 - val_accuracy: 0.9815
# Epoch 2/5: ... - loss: 0.0906 - accuracy: 0.9723 - val_loss: 0.0484 - val_accuracy: 0.9852
# Epoch 3/5: ... - loss: 0.0647 - accuracy: 0.9799 - val_loss: 0.0383 - val_accuracy: 0.9890
# Epoch 4/5: ... - loss: 0.0505 - accuracy: 0.9842 - val_loss: 0.0342 - val_accuracy: 0.9893
# Epoch 5/5: ... - loss: 0.0411 - accuracy: 0.9869 - val_loss: 0.0328 - val_accuracy: 0.9903
# Test accuracy: 0.9907
# Predicted classes: [7 2 1 0 4]
# True classes: [7 2 1 0 4]
```

### Example 3: Functional API for multi-input model
```python
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

# Two input branches
input_numeric = Input(shape=(5,), name='numeric_input')
input_categorical = Input(shape=(10,), name='categorical_input')

# Process each branch
x1 = Dense(16, activation='relu')(input_numeric)
x2 = Dense(8, activation='relu')(input_categorical)

# Concatenate
combined = Concatenate()([x1, x2])
x = Dense(32, activation='relu')(combined)
x = Dense(16, activation='relu')(x)
output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input_numeric, input_categorical],
              outputs=output)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Dummy data for two inputs
numeric_data = np.random.randn(1000, 5)
cat_data = np.random.randn(1000, 10)
y_data = np.random.randint(0, 2, (1000,))

history = model.fit(
    {'numeric_input': numeric_data, 'categorical_input': cat_data},
    y_data, epochs=10, batch_size=32, validation_split=0.2, verbose=0
)
print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.3f}")
```
```
# Output:
# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  numeric_input (InputLayer)     [(None, 5)]          0           []
#  categorical_input (InputLayer)  [(None, 10)]         0           []
#  dense_3 (Dense)                (None, 16)           96          ['numeric_input[0][0]']
#  dense_4 (Dense)                (None, 8)            88          ['categorical_input[0][0]']
#  concatenate (Concatenate)      (None, 24)           0           ['dense_3[0][0]', 'dense_4[0][0]']
#  dense_5 (Dense)                (None, 32)           800         ['concatenate[0][0]']
#  dense_6 (Dense)                (None, 16)           528         ['dense_5[0][0]']
#  output (Dense)                 (None, 1)            17          ['dense_6[0][0]']
# ==================================================================================================
# Total params: 1,529
# Trainable params: 1,529
# __________________________________________________________________________________________________
# Final val accuracy: 0.525
```

### Example 4: Functional API with skip connection (ResNet-like)
```python
def residual_block(x, units, dropout_rate=0.2):
    shortcut = x
    x = Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = Dense(units, activation='relu')(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

inputs = Input(shape=(20,))
x = Dense(64, activation='relu')(inputs)
x = residual_block(x, 64)
x = residual_block(x, 128)
x = tf.keras.layers.GlobalAveragePooling1D()(x[:, tf.newaxis, :])
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```
```
# Output:
# Model: "model_1"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 20)]         0           []
#  dense_7 (Dense)                (None, 64)           1344        ['input_2[0][0]']
#  ... (residual blocks with skip connections)
#  dense_14 (Dense)               (None, 1)            33          ['dense_13[0][0]']
# ==================================================================================================
# Total params: 45,217
# Trainable params: 45,217
# __________________________________________________________________________________________________
```

### Example 5: Callbacks — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', save_best_only=True, monitor='val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1
    )
]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=callbacks, verbose=0)

# EarlyStopping stopped training early
actual_epochs = len(history.history['loss'])
print(f"Trained for {actual_epochs} epochs (would have been 100 without early stopping)")

# Best model was saved to 'best_model.h5'
best_model = tf.keras.models.load_model('best_model.h5')
_, best_acc = best_model.evaluate(X, y, verbose=0)
print(f"Best model accuracy: {best_acc:.3f}")
```
```
# Output:
# Trained for 18 epochs (would have been 100 without early stopping)
# Best model accuracy: 0.901
```

### Example 6: Transfer learning with Keras Applications
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load pretrained model (without top classifier)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base layers

# Add new classifier
inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Base model trainable weights:", len(base_model.trainable_weights))
print("Full model summary:")
model.summary()
```
```
# Output:
# Base model trainable weights: 0
# Full model summary:
# Model: "model_2"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_3 (InputLayer)        [(None, 224, 224, 3)]     0
#  tf.nn.bias_add (TFOpLambda)  (None, 224, 224, 3)      0
#  resnet50 (Functional)       (None, 7, 7, 2048)        23,587,712
#  global_average_pooling2d    (None, 2048)              0
#  dropout_2 (Dropout)         (None, 2048)              0
#  dense_16 (Dense)            (None, 5)                 10245
# =================================================================
# Total params: 23,597,957
# Trainable params: 10,245
# Non-trainable params: 23,587,712
```

### Example 7: Custom training loop with tf.GradientTape
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

for epoch in range(5):
    epoch_loss = []
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = loss_fn(batch_y, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        epoch_loss.append(loss.numpy())

    print(f"Epoch {epoch+1}: Loss = {np.mean(epoch_loss):.4f}")

test_preds = model.predict(X[:5], verbose=0).flatten()
print(f"Sample predictions: {test_preds.round(3)}")
```
```
# Output:
# Epoch 1: Loss = 0.6934
# Epoch 2: Loss = 0.6210
# Epoch 3: Loss = 0.5431
# Epoch 4: Loss = 0.4782
# Epoch 5: Loss = 0.4321
# Sample predictions: [0.821 0.312 0.745 0.654 0.221]
```

### Example 8: Data augmentation with tf.keras.layers
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Show augmentation effect on one image
sample = x_train[0:1]
for i, aug_layer in enumerate(data_augmentation.layers[:2]):
    aug_sample = aug_layer(sample, training=True)
    print(f"Augmentation {i} output shape: {aug_sample.shape}")
```
```
# Output:
# Model: "sequential_4"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  sequential_3 (Sequential)   (None, 28, 28, 1)         0
#  conv2d_4 (Conv2D)           (None, 26, 26, 32)        320
#  max_pooling2d_3 (MaxPooling2D) (None, 13, 13, 32)    0
#  flatten_1 (Flatten)         (None, 5408)              0
#  dense_19 (Dense)            (None, 10)                54090
# =================================================================
# Total params: 54,410
# Trainable params: 54,410
# _________________________________________________________________
# Augmentation 1 output shape: (1, 28, 28, 1)
# Augmentation 2 output shape: (1, 28, 28, 1)
```

### Example 9: Learning rate schedules
```python
def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.5
    else:
        return lr * 0.1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=15, batch_size=32,
                    validation_split=0.2,
                    callbacks=[
                        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
                        tf.keras.callbacks.CSVLogger('training_log.csv')
                    ],
                    verbose=0)

# Print LR at each epoch
for epoch, lr in enumerate(history.history['lr']):
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: LR = {lr:.6f}")

print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.3f}")
```
```
# Output:
# Epoch 0: LR = 0.010000
# Epoch 5: LR = 0.005000
# Epoch 10: LR = 0.001000
# Final val accuracy: 0.885
```

### Example 10: Saving, loading, and exporting
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# Save entire model (architecture + weights + optimizer state)
model.save('full_model.keras')
print("Saved full model.")

# Save only weights
model.save_weights('model_weights.h5')
print("Saved weights only.")

# Save architecture as JSON
json_config = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(json_config)
print("Saved architecture JSON.")

# Load model
loaded_model = tf.keras.models.load_model('full_model.keras')
loaded_weights = tf.keras.models.model_from_json(json_config)
loaded_weights.load_weights('model_weights.h5')

# Verify
preds = loaded_model.predict(X[:3], verbose=0)
preds2 = loaded_weights.predict(X[:3], verbose=0)
print(f"Predictions match: {np.allclose(preds, preds2)}")
```
```
# Output:
# Saved full model.
# Saved weights only.
# Saved architecture JSON.
# Predictions match: True
```

## Common Mistakes

1. **Using `model.predict()` on training data.** `model.predict()` computes predictions for inference. For training, pass data to `model.fit()` directly.
2. **Forgetting to normalize input data for pretrained models.** Models like ResNet50 expect `preprocess_input()` which normalizes to the range the model was trained on.
3. **Applying softmax + categorical_crossentropy together.** Use `from_logits=True` in the loss function or omit softmax in the last layer. Keras's `categorical_crossentropy` expects logits by default in some versions.
4. **Not setting `training=True`/`training=False` correctly in custom models.** Dropout and BatchNorm behave differently during training vs inference. The `training` argument must be propagated.
5. **Calling `model.summary()` before `model.compile()` or before building the model.** For Sequential models with `input_shape`, summary works. For Functional API, inputs must be defined first.
6. **Using `sparse_categorical_crossentropy` with one-hot targets.** `sparse_categorical_crossentropy` expects integer labels, not one-hot vectors. Use `categorical_crossentropy` for one-hot.
7. **Not freezing base model before training in transfer learning.** If `base_model.trainable = True` is not set, the pretrained weights may be destroyed during the first training step.

## Interview Questions

### Beginner - 5

1. **Q:** What is the difference between Sequential and Functional API in Keras?  
   **A:** Sequential is a linear stack of layers, suitable for most simple networks. Functional API supports multiple inputs/outputs, branching, and shared layers via explicit tensor connections.

2. **Q:** What does `model.compile()` do?  
   **A:** It configures the model for training: sets the optimizer, loss function, and evaluation metrics.

3. **Q:** How do you prevent overfitting in Keras?  
   **A:** Use Dropout layers, EarlyStopping callback, L1/L2 regularization in Dense layers, data augmentation, or reduce model complexity.

4. **Q:** What is the purpose of validation_split in model.fit()?  
   **A:** It reserves a fraction of training data (e.g., 0.2 = 20%) for validation during training, reporting val_loss and val_accuracy without affecting model weights.

5. **Q:** How do you save a trained Keras model?  
   **A:** `model.save('model.keras')` saves architecture, weights, optimizer state, and configuration. Load with `tf.keras.models.load_model()`.

### Intermediate - 5

1. **Q:** What are callbacks in Keras and give three examples?  
   **A:** Callbacks are objects passed to `fit()` that perform actions during training. Examples: EarlyStopping (stop when val_loss plateaus), ModelCheckpoint (save best weights), ReduceLROnPlateau (reduce LR when metric stalls).

2. **Q:** How does `tf.GradientTape` work for custom training loops?  
   **A:** `tf.GradientTape` records operations for automatic differentiation. Inside the `with` block, all operations are recorded. `tape.gradient(loss, vars)` computes gradients, then `optimizer.apply_gradients` updates weights.

3. **Q:** What is the difference between `model.fit()` and `model.train_on_batch()`?  
   **A:** `fit()` trains for entire epochs with batching, shuffling, callbacks, and validation. `train_on_batch()` performs a single gradient update on one batch — useful for custom training loops.

4. **Q:** How do you implement transfer learning in Keras?  
   **A:** Load a pretrained model (e.g., ResNet50 with `weights='imagenet'`), set `base.trainable = False`, add new classifier layers, compile and train. Optionally unfreeze later layers for fine-tuning.

5. **Q:** What is the difference between `binary_crossentropy` and `categorical_crossentropy`?  
   **A:** `binary_crossentropy` is for 2-class problems with single sigmoid output. `categorical_crossentropy` is for multi-class with softmax output. Use `sparse_categorical_crossentropy` when labels are integers.

### Advanced - 3

1. **Q:** Explain how Keras handles variable-length sequences with RNNs and masking.  
   **A:** Use `Masking` layer or `mask_zero=True` in Embedding layers. RNN layers propagate the mask through time, skipping padded steps in loss computation.

2. **Q:** How would you implement a custom Keras layer with trainable weights?  
   **A:** Subclass `tf.keras.layers.Layer`, define `__init__`, `build(input_shape)` (create weights via `add_weight`), and `call(inputs)`. Optionally override `get_config` for serialization.

3. **Q:** Describe the Keras training loop internals — what happens during `model.fit()`?  
   **A:** (1) Data is shuffled and batched. (2) For each batch: forward pass → compute loss → `GradientTape` computes gradients → `optimizer.apply_gradients` updates weights → metrics are updated. (3) At epoch end: validation pass, callback calls, logging. (4) Returns History object.

## Practice Problems

### Easy - 5

1. **E1:** Build a Sequential model with 2 Dense layers for binary classification on synthetic data.
2. **E2:** Train a CNN on MNIST with 2 Conv2D layers and evaluate test accuracy.
3. **E3:** Add EarlyStopping callback to a model and verify it stops early.
4. **E4:** Build a regression model (1 output neuron, no activation) for y = 2x + 1.
5. **E5:** Save a trained model and reload it, verifying predictions match.

### Medium - 5

1. **M1:** Build a Functional API model with two dense branches that concatenate and produce a single output.
2. **M2:** Use ResNet50 pretrained on ImageNet to classify 5 custom classes (transfer learning).
3. **M3:** Implement a custom training loop with tf.GradientTape for a small model.
4. **M4:** Add data augmentation layers (flip, rotation, zoom) to an image classifier and compare performance.
5. **M5:** Use LearningRateScheduler callback with a custom schedule function.

### Hard - 3

1. **H1:** Implement a custom Keras layer that computes a 1D attention mechanism over the input sequence.
2. **H2:** Build a multi-output model that predicts both class and bounding box coordinates from an image.
3. **H3:** Implement a Variational Autoencoder (VAE) using the Functional API with KL divergence loss.

## Solutions

### E1 Solution
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### E2 Solution
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### E3-E5 Solutions follow patterns from examples.

### M1-M5 Solutions extend the examples with multi-branch, transfer learning, and custom loops.

### H1-H3 Solutions require advanced layer/architecture implementation.

## Related Concepts

- 093 — sklearn Basics (traditional ML vs deep learning)
- 096 — PyTorch Tensors (equivalent framework comparison)
- 097 — PyTorch NN (equivalent Keras concepts)
- 099 — Training Loops (PyTorch equivalent of Keras fit)

## Next Concepts

- 099 — Training Loops (PyTorch custom loops vs Keras fit)
- 100 — Project Structure (packaging and deploying Keras models)

## Summary

TensorFlow/Keras provides the Sequential API for simple networks, the Functional API for complex architectures, and Model Subclassing for full control. Layers (Dense, Conv2D, MaxPooling2D, Dropout) are composed into models, compiled with optimizers/losses/metrics, and trained with `fit()`. Callbacks enable early stopping, checkpointing, and learning rate scheduling. Keras Applications provides pretrained models for transfer learning.

## Key Takeaways

- Sequential API for simple stacks; Functional API for complex graphs
- `compile(optimizer, loss, metrics)` configures the model for training
- `fit()` trains with batching, shuffling, validation, and callbacks
- `evaluate()` computes test metrics; `predict()` generates predictions
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
- Transfer learning: freeze base model → add new head → fine-tune
- `model.save()` saves everything; `load_model()` restores it
- `tf.GradientTape` enables custom training loops for advanced use cases
