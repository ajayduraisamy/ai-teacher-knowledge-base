# Concept: Deep Learning Hardware

## Concept ID

DL-015

## Difficulty

Beginner

## Domain

Deep Learning

## Module

Deep Learning Foundations

## Learning Objectives

- Distinguish between CPU, GPU, and TPU for deep learning workloads
- Explain why GPUs are preferred for neural network training
- Understand the role of CUDA, tensor cores, and memory management
- Identify when CPU training is sufficient

## Prerequisites

- Basic computer architecture knowledge
- Training Pipeline (DL-010)
- Forward/Backward Pass (DL-012, DL-013)

## Definition

Deep learning hardware refers to the computational devices used to train and deploy neural networks. The three primary types are:

**CPU (Central Processing Unit):** General-purpose processor with a few powerful cores (4-16) optimized for sequential, latency-sensitive tasks. Good for data preprocessing, small models, and inference on low-power devices.

**GPU (Graphics Processing Unit):** Specialized processor with thousands of small cores designed for parallel computation. Originally built for graphics rendering, GPUs excel at the matrix operations central to neural networks. Key features: CUDA cores (NVIDIA), stream processors (AMD), tensor cores (NVIDIA Volta+).

**TPU (Tensor Processing Unit):** Google's custom ASIC (Application-Specific Integrated Circuit) designed specifically for neural network inference and training. Optimized for TensorFlow computations with matrix multiplication units (MXUs).

## Intuition

Think of a CPU as a few expert chefs in a kitchen — they can handle any recipe, coordinate complex tasks, and adapt to changing conditions quickly. A GPU is like a thousand line cooks — each can only do simple tasks (chop vegetables, stir pots), but working in parallel, they process massive amounts of ingredients much faster than a few experts.

Neural network training is mostly matrix multiplications: the same operation applied to millions of numbers simultaneously. This is the line cook's dream — thousands of simple, identical operations that can be done in parallel. A CPU with 8 cores does 8 computations at once; a GPU with 3000 cores does 3000 computations at once.

A TPU is like a factory that does only one thing extremely well: high-throughput matrix multiplication. It strips away all general-purpose features to maximize speed for this single operation.

## Why This Concept Matters

Hardware choices directly affect:

- **Training Time:** What takes a week on CPU might take hours on GPU and minutes on TPU
- **Model Size:** GPU memory (VRAM) limits the maximum model size that can be trained
- **Cost:** Cloud GPU/TPU rental can be expensive; choosing the right hardware saves money
- **Deployment:** Edge devices (phones, IoT) require different hardware considerations than cloud servers
- **Accessibility:** Understanding hardware requirements helps practitioners choose feasible projects

## Real World Examples

1. **GPT-3 Training:** Trained on a cluster of 10,000 NVIDIA V100 GPUs for approximately 3.4 exaflops-days of computation. Would have taken decades on CPUs.

2. **Edge AI Inference:** Running a small model (MobileNet, TinyML) on a Raspberry Pi or smartphone CPU for real-time object detection. GPUs are too power-hungry for battery-powered devices.

3. **TPU for Research:** Google trained BERT on 64 TPUv2 chips in 4 days. The same training on a single high-end GPU would take months.

## AI/ML Relevance

- **CUDA:** NVIDIA's parallel computing platform. Most deep learning frameworks (PyTorch, TensorFlow) use CUDA for GPU acceleration.
- **cuDNN:** NVIDIA's CUDA Deep Neural Network library — optimized primitives for convolutions, pooling, normalization, and activations.
- **Mixed Precision Training:** Using FP16 (half precision) for most operations and FP32 for critical accumulations, enabled by tensor cores for 2x speedup.
- **Distributed Training:** Data parallelism (each GPU processes a subset of data) and model parallelism (each GPU processes a subset of layers).
- **VRAM Management:** Training large models requires careful memory management — gradient checkpointing, gradient accumulation, and memory-efficient optimizers.

## Mathematical Explanation

### Computational Requirements

For a forward pass through a layer with input $d_{in}$ and output $d_{out}$ on batch size $B$:

- **FLOPs (multiply-add):** $B \cdot d_{in} \cdot d_{out}$ for the matrix multiply, $B \cdot d_{out}$ for bias add
- **Memory (weights):** $d_{in} \cdot d_{out} + d_{out}$ parameters × 4 bytes (FP32)
- **Memory (activations):** $B \cdot d_{out}$ × 4 bytes (stored for backward pass)

### GPU vs CPU Performance

For matrix multiplication $C = A \times B$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$:

- **CPU theoretical peak:** cores × clock_speed × FLOPS_per_cycle
  - Example: AMD Ryzen 9 7950X (16 cores, 5.7 GHz, AVX-512): ~1.5 TFLOPS
- **GPU theoretical peak:** cores × clock_speed × 2 (FMA)
  - Example: NVIDIA RTX 4090 (16,384 CUDA cores, 2.5 GHz): ~82 TFLOPS

GPU is ~50x faster for matrix multiply for a single consumer card.

### Memory Bandwidth

- CPU DDR5: ~50-80 GB/s
- GPU GDDR6X: ~1000 GB/s (RTX 4090)
- HBM2e (A100): ~2000 GB/s

Higher bandwidth means faster data movement between memory and compute units, which is often the bottleneck in neural network training.

### Arithmetic Intensity

Arithmetic intensity = FLOPs / bytes_transferred

- Compute-bound: high arithmetic intensity (large matrices, batch size)
- Memory-bound: low arithmetic intensity (small matrices, element-wise ops)

GPUs excel at compute-bound operations; CPUs can be competitive for memory-bound operations.

## Code Examples

### Example 1: Checking GPU Availability and Basic Operations

```python
import torch
import time

# Check device availability
print(f"CUDA available: {torch.cuda.is_available()}")
# Output: CUDA available: True (or False)

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
# Output: GPU count: 1
# Output: Current GPU: NVIDIA GeForce RTX 4090
# Output: CUDA version: 12.1

# Compare CPU vs GPU matrix multiplication speed
device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

size = 5000
A_cpu = torch.randn(size, size, device=device_cpu)
B_cpu = torch.randn(size, size, device=device_cpu)

start = time.time()
C_cpu = A_cpu @ B_cpu
cpu_time = time.time() - start

if torch.cuda.is_available():
    A_gpu = torch.randn(size, size, device=device_gpu)
    B_gpu = torch.randn(size, size, device=device_gpu)
    # Warm up GPU
    _ = A_gpu @ B_gpu
    torch.cuda.synchronize()

    start = time.time()
    C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s, Speedup: {cpu_time/gpu_time:.1f}x")
# Output: CPU time: 0.8934s, GPU time: 0.0123s, Speedup: 72.6x
```

### Example 2: Measuring GPU Memory Usage

```python
import torch
import torch.nn as nn

def print_memory_usage(stage=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

print_memory_usage("start")
# Output: [start] Allocated: 0.00 GB, Cached: 0.00 GB

# Create a large model
model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1000)
).cuda()

print_memory_usage("after model creation")
# Output: [after model creation] Allocated: 0.82 GB, Cached: 0.88 GB

# Create input and compute forward pass
x = torch.randn(64, 4096).cuda()
output = model(x)
print_memory_usage("after forward pass")
# Output: [after forward pass] Allocated: 1.28 GB, Cached: 1.50 GB

# Memory increases because activations are stored for backward pass
loss = output.sum()
loss.backward()
print_memory_usage("after backward pass")
# Output: [after backward pass] Allocated: 1.30 GB, Cached: 1.50 GB

# Model size calculation
param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
print(f"Model parameters in memory: {param_size:.4f} GB")
# Output: Model parameters in memory: 0.82 GB
```

### Example 3: Mixed Precision Training for Speed

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

class LargeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1000)
        )
    def forward(self, x):
        return self.net(x)

model = LargeMLP().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

x = torch.randn(128, 2048).cuda()
y = torch.randint(0, 1000, (128,)).cuda()

# FP32 training
def train_fp32():
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Warm up
for _ in range(10):
    _ = train_fp32()

# Time FP32
start = time.time()
for _ in range(50):
    _ = train_fp32()
fp32_time = time.time() - start
print(f"FP32 time for 50 steps: {fp32_time:.4f}s")
# Output: FP32 time for 50 steps: 1.2345s

# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

def train_amp():
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = criterion(output, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

# Warm up
for _ in range(10):
    _ = train_amp()

# Time AMP
start = time.time()
for _ in range(50):
    _ = train_amp()
amp_time = time.time() - start
print(f"AMP time for 50 steps: {amp_time:.4f}s, Speedup: {fp32_time/amp_time:.2f}x")
# Output: AMP time for 50 steps: 0.7654s, Speedup: 1.61x
```

## Common Mistakes

1. **Running out of GPU memory:** Batch size too large, too many layers, or not clearing unused tensors. Use `torch.cuda.empty_cache()`, gradient accumulation, or gradient checkpointing.

2. **Using CPU when GPU is available:** Typing `tensor.cpu()` unnecessarily or forgetting `.cuda()` causes inadvertent CPU computation. Use `model.to(device)` with a device variable.

3. **Ignoring data transfer bottlenecks:** Loading data on CPU and transferring to GPU for each batch can be slow. Use `DataLoader(pin_memory=True)` and `num_workers>0` for efficient data loading.

4. **Not synchronizing CUDA for timing:** CUDA operations are asynchronous. `time.time()` before and after GPU operations does not give correct timing without `torch.cuda.synchronize()`.

5. **Over-relying on GPU for tiny models:** If your model and batch are small (< 100K parameters, batch size < 32), CPU may be faster due to GPU kernel launch overhead.

## Interview Questions

### Beginner

1. Why are GPUs preferred over CPUs for deep learning?
2. What is CUDA and why is it important for deep learning?
3. What is VRAM and why does it matter for training?
4. When would you use a CPU instead of a GPU for deep learning?
5. What is the difference between FP32, FP16, and mixed precision training?

### Intermediate

1. Explain the concept of arithmetic intensity and how it determines whether an operation is compute-bound or memory-bound.
2. What are tensor cores and how do they accelerate training?
3. How does data parallelism work across multiple GPUs? What is the role of all-reduce?
4. What is gradient accumulation and how does it help with limited GPU memory?
5. Compare GPU memory consumption for training vs inference — why does training require more memory?

### Advanced

1. Design a memory optimization strategy for training a large language model on a single GPU with limited VRAM (gradient checkpointing, offloading, activation compression).
2. Analyze the roofline model for a given neural network architecture and determine whether it is compute-bound or memory-bound on specific hardware.
3. Explain the communication overhead in distributed training — how does the all-reduce algorithm scale with the number of GPUs, and what techniques reduce communication?

## Practice Problems

### Easy

1. Write code to check if CUDA is available and print GPU specifications.
2. Compare CPU vs GPU execution time for a 1000x1000 matrix multiplication.
3. Move a PyTorch model and tensors to GPU using `.cuda()` and `.to(device)`.
4. Calculate the VRAM needed for a model with 1M parameters trained with batch size 64, input 784, hidden 256.
5. Convert a training script to use a device-agnostic pattern (dynamically choose CPU or GPU).

### Medium

1. Implement mixed precision training using `torch.cuda.amp` and compare training speed and memory usage against FP32.
2. Measure GPU memory usage with `torch.cuda.memory_summary()` and identify which operations consume the most memory.
3. Implement gradient accumulation to train with an effective batch size of 256 when GPU can only fit batch size 32.
4. Profile a training script to identify bottlenecks (data loading, forward pass, backward pass, optimizer step) using PyTorch profiler.
5. Implement data parallelism using `torch.nn.DataParallel` or `DistributedDataParallel` on a multi-GPU setup (simulate if only one GPU).

### Hard

1. Implement model parallelism: split a large model across two GPUs, passing activations from one to the other.
2. Design and implement a memory-efficient training loop using gradient checkpointing for a 20-layer MLP.
3. Implement a distributed training pipeline using NCCL backend and gradient compression techniques (e.g., Top-K sparsification or QSGD).

## Solutions

### Easy 1
```python
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, VRAM: {props.total_memory / 1024**3:.1f} GB")
```

### Easy 4
Parameters: 784*256 + 256 + 256*10 + 10 = ~202K parameters
FP32: 202K * 4 bytes = ~0.8 MB for weights
Activations (backward): batch_size * hidden * layers * 4 bytes = 64 * 256 * 2 * 4 = ~0.13 MB
Total ≈ 1 MB — very small, but intermediate values scale with batch size and layers.

### Medium 1
```python
scaler = torch.cuda.amp.GradScaler()
for data, target in loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Related Concepts

- Mixed Precision Training
- Distributed Training
- Gradient Checkpointing
- Model Parallelism
- Data Pipeline Optimization

## Next Concepts

- Distributed Data Parallel
- Pipeline Parallelism
- Tensor Parallelism
- Compression Techniques
- Hardware-Aware Neural Architecture Search

## Summary

Deep learning hardware includes CPUs (general-purpose, few cores), GPUs (parallel, many cores optimized for matrix operations), and TPUs (ASICs for tensor computations). GPUs dominate training due to their parallel architecture (thousands of cores) and high memory bandwidth. Key considerations include VRAM capacity, CUDA/tensor core availability, mixed precision support, and data transfer bottlenecks. For small models or low-power deployment, CPUs remain relevant. Distributed and memory-efficient training techniques enable scaling beyond single-device limits.

## Key Takeaways

- GPUs excel at parallel matrix operations central to neural network training
- CPU: few powerful cores, good for data preprocessing and small models
- GPU: thousands of simple cores, 50-100x faster for large matrix operations
- TPU: custom ASIC for TensorFlow, even faster for specific workloads
- VRAM limits maximum model size and batch size
- Mixed precision (FP16/FP32 with tensor cores) provides ~2x speedup with minimal accuracy loss
- Data loading and CPU-GPU transfer are common bottlenecks
