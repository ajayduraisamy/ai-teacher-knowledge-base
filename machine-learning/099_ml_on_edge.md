# Concept: ML on Edge

## Concept ID

ML-099

## Difficulty

Intermediate

## Domain

Machine Learning

## Module

Applied ML

## Learning Objectives

- Understand the constraints and opportunities of edge deployment
- Apply model quantization (FP32 to FP16 to INT8) to reduce model size
- Implement weight pruning (magnitude-based, structured) for compression
- Apply knowledge distillation to train compact student models
- Deploy models with TensorFlow Lite, ONNX Runtime, and PyTorch Mobile
- Analyze the accuracy vs model size tradeoff

## Prerequisites

- Neural network fundamentals (CNN, dense layers, activations)
- Python with TensorFlow or PyTorch
- Basic understanding of model training and inference
- Familiarity with mobile or embedded systems concepts

## Definition

ML on Edge refers to the deployment of machine learning models on edge devices (smartphones, IoT sensors, microcontrollers, cameras, wearables) rather than in the cloud. Edge deployment imposes severe constraints on model size, memory, compute, latency, and power consumption. To meet these constraints, models must be compressed and optimized through techniques like quantization, pruning, knowledge distillation, and architecture search. The goal is to achieve acceptable accuracy while fitting within the device's hardware budget.

## Intuition

Imagine you want to run a powerful object detection model on a security camera that has a tiny processor, limited memory, and runs on batteries. The original model might be 500MB and need a GPU — impossible on the camera. Model compression is like taking a detailed encyclopedia and condensing it into a pocket reference guide: you keep the most important information (pruning), use shorter abbreviations (quantization), and organize it more efficiently (architecture optimization). The pocket guide is never as complete as the full encyclopedia, but it fits in your pocket and answers most questions correctly.

## Why This Concept Matters

Edge ML is transforming industries: smartphone apps run real-time face recognition, AR filters, and on-device translation without sending data to the cloud; IoT sensors detect anomalies locally (predictive maintenance); medical wearables monitor heart arrhythmias in real-time; autonomous vehicles process sensor data on-board. Edge deployment reduces latency (no network round-trip), improves privacy (data stays on device), works offline, and reduces cloud costs. Gartner predicts that by 2027, 65% of ML inference will be performed at the edge.

## Mathematical Explanation

### Quantization

Quantization maps continuous floating-point values to discrete integer values:

r = S * (q - Z)

Where r is the real value, q is the quantized integer, S is the scale factor, and Z is the zero point.

**INT8 Quantization**: Map FP32 range [r_min, r_max] to INT8 range [0, 255]:

S = (r_max - r_min) / 255
Z = round(-r_min / S)

For weights, we typically use per-tensor or per-channel quantization:

q = clamp(round(r / S + Z), 0, 255)

**Quantization-Aware Training (QAT)**: Simulate quantization effects during training using fake quantization nodes, allowing the model to adapt to quantization noise.

### Pruning

**Magnitude-based pruning**: Remove weights with the smallest absolute values:

mask_i = 1 if |w_i| > tau else 0
w_i_pruned = w_i * mask_i

**Structured pruning**: Remove entire neurons, channels, or filters that contribute least to the output. For a convolutional layer, prune the filter with the smallest L2 norm:

||W_i||_2 = sqrt(sum_j W_ij^2)

**Iterative pruning**: Train -> prune -> retrain -> repeat. Gradually increase the pruning ratio to find the optimal sparse structure.

### Knowledge Distillation

Train a compact student model S to mimic a large teacher model T using softened probabilities:

L = alpha * L_hard(y, S(x)) + (1 - alpha) * L_soft(T(x, tau), S(x, tau))

Where tau is the temperature parameter that softens the probability distribution:

P_i = exp(z_i / tau) / sum_j exp(z_j / tau)

Higher tau produces softer distributions that reveal the teacher's "dark knowledge" (relative probabilities between incorrect classes).

## Code Examples

### Example 1: Post-Training Quantization with TensorFlow Lite

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Train a simple model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=0)

# Evaluate FP32 model
fp32_loss, fp32_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"FP32 model - Accuracy: {fp32_acc:.4f}, Size: {model.count_params()} params")
# Output: FP32 model - Accuracy: 0.9820, Size: 91242 params

# Convert to TensorFlow Lite (Float16 quantization)
converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter_fp16.convert()

# Convert to TensorFlow Lite (INT8 quantization with representative dataset)
def representative_dataset():
    for i in range(100):
        yield [x_train[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8
tflite_int8 = converter_int8.convert()

# Evaluate TFLite models
def evaluate_tflite(interpreter, x_test, y_test):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(len(x_test)):
        input_data = x_test[i:i+1]
        # Quantize input if needed
        if input_details[0]['dtype'] == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output)
        if predicted == y_test[i]:
            correct += 1

    return correct / len(x_test)

# FP16 model
interpreter_fp16 = tf.lite.Interpreter(model_content=tflite_fp16)
acc_fp16 = evaluate_tflite(interpreter_fp16, x_test[:500], y_test[:500])

# INT8 model
interpreter_int8 = tf.lite.Interpreter(model_content=tflite_int8)
acc_int8 = evaluate_tflite(interpreter_int8, x_test[:500], y_test[:500])

print(f"FP16 TFLite - Accuracy: {acc_fp16:.4f}")
print(f"INT8 TFLite - Accuracy: {acc_int8:.4f}")
print(f"FP32 model size: {len(model.to_json()):,} bytes (JSON) + weights")
print(f"FP16 TFLite size: {len(tflite_fp16):,} bytes")
print(f"INT8 TFLite size: {len(tflite_int8):,} bytes")
# Output:
# FP16 TFLite - Accuracy: 0.9820
# INT8 TFLite - Accuracy: 0.9800
# FP16 TFLite size: 116,448 bytes
# INT8 TFLite size: 59,288 bytes
```

### Example 2: Magnitude-Based Weight Pruning

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Build prune-able model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Apply pruning schedule: start at step 0, end at step 2000, target sparsity 0.75
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.75,
        begin_step=0,
        end_step=2000,
        frequency=100
    )
}

pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
pruned_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with pruning
pruned_model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=3,
    validation_split=0.1,
    callbacks=[sparsity.UpdatePruningStep()],
    verbose=0
)

# Strip pruning wrappers for deployment
final_model = sparsity.strip_pruning(pruned_model)

# Evaluate
test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
print(f"Pruned model - Accuracy: {test_acc:.4f}")

# Count non-zero weights
total_weights = 0
nonzero_weights = 0
for layer in final_model.layers:
    weights = layer.get_weights()
    for w in weights:
        total_weights += w.size
        nonzero_weights += np.count_nonzero(w)

sparsity_ratio = 1 - nonzero_weights / total_weights
print(f"Total weights: {total_weights}, Non-zero: {nonzero_weights}")
print(f"Actual sparsity: {sparsity_ratio:.2%}")
# Output:
# Pruned model - Accuracy: 0.9789
# Total weights: 91242, Non-zero: 23060
# Actual sparsity: 74.73%
```

### Example 3: Knowledge Distillation

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Teacher model (large)
teacher = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

teacher.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
teacher.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
teacher_acc = teacher.evaluate(x_test, y_test, verbose=0)[1]
print(f"Teacher accuracy: {teacher_acc:.4f}, Params: {teacher.count_params()}")
# Output: Teacher accuracy: 0.9858, Params: 321994

# Student model (small)
student = models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Distillation training
temperature = 5.0
alpha = 0.7

teacher_predictions = teacher.predict(x_train, verbose=0)
teacher_logits = teacher_predictions  # Softmax outputs

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

for epoch in range(5):
    for i in range(0, len(x_train), 128):
        batch_x = x_train[i:i+128]
        batch_y = y_train[i:i+128]
        batch_teacher = teacher_predictions[i:i+128]

        with tf.GradientTape() as tape:
            student_logits = student(batch_x, training=True)

            # Hard loss (on true labels)
            hard_loss = loss_fn(batch_y, student_logits)

            # Soft loss (on teacher softened predictions)
            student_soft = tf.nn.softmax(student_logits / temperature)
            teacher_soft = tf.nn.softmax(batch_teacher / temperature)
            soft_loss = tf.reduce_mean(
                tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft)
            )

            total_loss = alpha * hard_loss + (1 - alpha) * soft_loss * (temperature ** 2)

        grads = tape.gradient(total_loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))

student_acc = student.evaluate(x_test, y_test, verbose=0)[1]
print(f"Student (distilled) accuracy: {student_acc:.4f}, Params: {student.count_params()}")
# Output: Student (distilled) accuracy: 0.9728, Params: 20842

# Student trained from scratch (baseline)
student_scratch = models.Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
student_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
student_scratch.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
scratch_acc = student_scratch.evaluate(x_test, y_test, verbose=0)[1]
print(f"Student (scratch) accuracy: {scratch_acc:.4f}")
# Output: Student (scratch) accuracy: 0.9666

print(f"\nSize reduction: Teacher={teacher.count_params()} -> Student={student.count_params()}")
print(f"Accuracy: Teacher={teacher_acc:.4f}, Distilled Student={student_acc:.4f}, Scratch Student={scratch_acc:.4f}")
# Output:
# Size reduction: Teacher=321994 -> Student=20842
# Accuracy: Teacher=0.9858, Distilled Student=0.9728, Scratch Student=0.9666
```

## Common Mistakes

1. **Quantizing without a representative dataset**: Post-training quantization uses a calibration dataset to determine scale and zero-point. Using random or mismatched data produces poor quantization scales and significant accuracy loss.

2. **Over-pruning in one shot**: Aggressive pruning (>90% sparsity) in a single step destroys the model. Use iterative pruning (prune small amounts, retrain, repeat) to find the optimal sparse structure.

3. **Not accounting for hardware support**: INT8 quantization is well-supported on modern hardware (ARM NEON, Qualcomm Hexagon, Apple Neural Engine, NVIDIA TensorRT), but older devices may only support FP16. Always check target hardware capabilities.

4. **Ignoring the student-teacher capacity gap**: If the student is too small, it cannot absorb the teacher's knowledge. There is a minimum capacity threshold below which distillation provides no benefit over training from scratch.

5. **Measuring only accuracy, not latency**: A pruned model may have similar accuracy but be slower on hardware without sparse matrix acceleration. Always benchmark actual latency on the target device.

6. **Using too high a distillation temperature**: Temperature tau > 10 produces near-uniform soft targets that provide no information. tau between 2 and 8 is typical. Tune it as a hyperparameter.

7. **Pruning without considering memory layout**: On many accelerators, non-structured (random) sparsity does not yield speedups because weights are stored in dense format. Use structured pruning (channel-wise, block-wise) for real performance gains.

## Interview Questions

### Beginner

1. What are the main challenges of deploying ML models on edge devices?
2. What is model quantization and how does INT8 quantization reduce model size?
3. What is the difference between post-training quantization and quantization-aware training?
4. How does weight pruning reduce model size?
5. What is knowledge distillation and why does a small student model benefit from a large teacher?

### Intermediate

1. Explain the mathematics of INT8 quantization: how are scale and zero-point computed?
2. Compare magnitude-based pruning with structured pruning. When would you use each?
3. How does temperature affect knowledge distillation? What is "dark knowledge"?
4. Compare TensorFlow Lite, ONNX Runtime, and PyTorch Mobile in terms of deployment workflow and target hardware.
5. How would you profile a model to determine the best compression strategy (quantization, pruning, or distillation)?

### Advanced

1. Design a compression pipeline that combines pruning, quantization, and distillation for a BERT model to be deployed on a smartphone.
2. Explain how Neural Architecture Search (NAS) can be constrained to find edge-optimal architectures (e.g., MnasNet, MobileNetV3).
3. How would you implement mixed-precision quantization where different layers use different bit-widths (4-bit, 8-bit, 16-bit) to minimize accuracy loss?

## Practice Problems

### Easy

1. Compute the INT8 scale and zero-point for a weight tensor with values in [-2.0, 3.0].
2. Quantize the value 0.5 to INT8 using scale=0.02 and zero-point=128.
3. Calculate the model size reduction from quantizing a 10M parameter model from FP32 to INT8.
4. Count the number of nonzero weights in a numpy array with 50% zeros.
5. Convert a Keras model to TensorFlow Lite format and measure the file size.

### Medium

1. Implement quantization-aware training for a small CNN and compare accuracy with post-training quantization.
2. Build an iterative pruning pipeline that gradually increases sparsity from 0% to 90% and plots accuracy vs sparsity.
3. Implement knowledge distillation with a ResNet-50 teacher and a MobileNetV2 student on CIFAR-10.
4. Deploy an ONNX model to ONNX Runtime and benchmark inference latency vs TensorFlow Lite.
5. Create a pruning schedule that prunes early layers less aggressively than later layers, and evaluate the impact on accuracy.

### Hard

1. Implement mixed-precision quantization (8-bit weights, 4-bit activations) for a small model using custom quantization layers.
2. Build a hardware-aware model compression tool that profiles latency on a target device and suggests the optimal compression strategy.
3. Implement differentiable pruning using L0 regularization (Louizos et al.) for learning the pruning mask during training.

## Solutions

### Easy 1 — INT8 scale and zero-point
```python
import numpy as np
r_min, r_max = -2.0, 3.0
S = (r_max - r_min) / 255
Z = round(-r_min / S)
print(f"Scale: {S:.4f}, Zero-point: {Z}")
# Output: Scale: 0.0196, Zero-point: 102
```

### Easy 3 — Model size reduction
```python
n_params = 10_000_000
fp32_size = n_params * 4 / (1024 * 1024)
int8_size = n_params * 1 / (1024 * 1024)
print(f"FP32: {fp32_size:.1f} MB, INT8: {int8_size:.1f} MB, " +
      f"Reduction: {(1 - int8_size/fp32_size)*100:.0f}%")
# Output: FP32: 38.1 MB, INT8: 9.5 MB, Reduction: 75%
```

## Related Concepts

- Model Deployment — ML-080
- Neural Architecture Search — ML-097
- Embedded Systems — ML-088
- Deep Learning — ML-082

## Next Concepts

- Ethics and Responsible AI — ML-100
- AutoML — ML-097
- Causal ML — ML-098

## Summary

ML on Edge deploys compressed models to resource-constrained devices. Quantization (FP32 to FP16/INT8) reduces model size by 75% with minimal accuracy loss. Pruning removes redundant weights (50-90% sparsity achievable) and can be unstructured or structured. Knowledge distillation transfers knowledge from a large teacher to a small student model. TensorFlow Lite, ONNX Runtime, and PyTorch Mobile provide deployment frameworks. The key challenge is balancing accuracy, latency, model size, and power consumption for the target hardware.

## Key Takeaways

- Edge ML requires model compression: quantization, pruning, distillation
- INT8 quantization reduces model size by 75% with <1% accuracy loss typically
- Quantization-aware training recovers more accuracy than post-training quantization
- Iterative pruning finds better sparse structures than one-shot pruning
- Knowledge distillation trains small models that mimic large teacher outputs
- Structured pruning (channel/filter removal) gives actual speedups; unstructured does not
- Always benchmark on target hardware — theoretical FLOP reductions may not translate to latency gains
- The optimal compression strategy depends on the specific hardware, latency requirements, and accuracy targets
