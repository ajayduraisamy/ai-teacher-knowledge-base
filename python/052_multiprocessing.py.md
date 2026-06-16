# Concept: Multiprocessing

## Concept ID

PYT-052

## Difficulty

Advanced

## Domain

Python

## Module

Concurrency and Performance

## Learning Objectives

- Create and manage separate processes using `multiprocessing.Process`
- Use `Pool.map` and `ProcessPoolExecutor` for parallel task execution
- Share data between processes using Queue, Pipe, and shared memory
- Understand when to use multiprocessing over threading
- Handle process synchronization with locks and barriers
- Measure and compare performance gains from parallel execution

## Prerequisites

- Understanding of threads (PYT-051) and the GIL
- Familiarity with functions and iterables
- Basic understanding of operating system processes

## Definition

Multiprocessing is a technique that uses multiple processes — each with its own Python interpreter and memory space — to achieve true parallelism. The `multiprocessing` module in Python provides APIs similar to `threading` but spawns separate OS processes, bypassing the Global Interpreter Lock entirely.

## Intuition

If threading is like a single chef rapidly switching between tasks, multiprocessing is like hiring multiple chefs, each working in their own kitchen with their own set of tools. They can all cook simultaneously without interfering. The cost is more overhead — each kitchen needs its own setup — but the benefit is true parallel execution.

## Why This Concept Matters

CPU-bound tasks (image processing, numerical simulations, hash cracking) cannot benefit from threads due to the GIL. Multiprocessing solves this by distributing work across multiple CPU cores. Modern machines have 4, 8, 16 or more cores; tapping into them is essential for performance. Multiprocessing is also crucial for fault isolation — a crash in one process does not bring down others.

## Real World Examples

1. **Image Processing:** Applying filters to thousands of images in parallel.
2. **Data Analysis:** Processing large CSV/Parquet files chunk by chunk across cores.
3. **Web Scraping:** Distributing URL fetching across processes to avoid DNS bottlenecks.
4. **Scientific Computing:** Running Monte Carlo simulations with independent trials.
5. **Video Encoding:** Splitting a video into segments and encoding them in parallel.

## AI/ML Relevance

- **Parallel Hyperparameter Search:** Running multiple model configurations simultaneously across CPU cores.
- **Ensemble Training:** Training separate models (bagging, boosting) in parallel processes.
- **Data Augmentation:** Applying transformations to training data in parallel before feeding to GPU.
- **Feature Extraction:** Computing features for large datasets using `Pool.map`.
- **Cross-Validation:** Executing k-fold validation folds in parallel.

## Code Examples

### Example 1: Basic Process Creation

```python
import multiprocessing
import os

def worker(name):
    print(f"Worker {name} (PID: {os.getpid()})")

if __name__ == "__main__":
    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes done")
# Output:
# Worker 0 (PID: 12345)
# Worker 1 (PID: 12346)
# Worker 2 (PID: 12347)
# Worker 3 (PID: 12348)
# All processes done
```

### Example 2: Pool.map for Parallel Map

```python
import multiprocessing

def square(n):
    return n * n

if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(square, range(10))
    print(results)
# Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Example 3: ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

if __name__ == "__main__":
    numbers = list(range(100000, 100200))
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(is_prime, numbers))
    print(f"Found {sum(results)} primes")
# Output: Found 20 primes
```

### Example 4: Queue for Interprocess Communication

```python
import multiprocessing

def producer(queue):
    for i in range(5):
        queue.put(i)
    queue.put(None)  # Sentinel

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")

if __name__ == "__main__":
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=producer, args=(q,))
    p2 = multiprocessing.Process(target=consumer, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
# Output:
# Consumed: 0
# Consumed: 1
# Consumed: 2
# Consumed: 3
# Consumed: 4
```

### Example 5: Pipe Communication

```python
import multiprocessing

def sender(conn):
    conn.send("Hello from sender")
    conn.close()

def receiver(conn):
    msg = conn.recv()
    print(f"Receiver got: {msg}")

if __name__ == "__main__":
    parent_conn, child_conn = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target=sender, args=(parent_conn,))
    p2 = multiprocessing.Process(target=receiver, args=(child_conn,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
# Output: Receiver got: Hello from sender
```

### Example 6: Shared Memory with Value

```python
import multiprocessing

def increment(counter):
    for _ in range(10000):
        counter.value += 1

if __name__ == "__main__":
    counter = multiprocessing.Value("i", 0)
    processes = [multiprocessing.Process(target=increment, args=(counter,)) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(f"Counter: {counter.value} (expected 40000)")
# Output: Counter: 40000 (expected 40000)
```

### Example 7: CPU vs I/O Bound Benchmark

```python
import time
import threading
import multiprocessing

def cpu_work(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    n = 10000000

    start = time.time()
    for _ in range(4):
        cpu_work(n)
    print(f"Sequential: {time.time() - start:.2f}s")

    start = time.time()
    threads = [threading.Thread(target=cpu_work, args=(n,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"Threading: {time.time() - start:.2f}s")

    start = time.time()
    with multiprocessing.Pool(4) as pool:
        pool.map(cpu_work, [n] * 4)
    print(f"Multiprocessing: {time.time() - start:.2f}s")
# Output (varies by machine):
# Sequential: 2.10s
# Threading: 2.08s
# Multiprocessing: 0.55s
```

## Common Mistakes

1. **Forgetting `if __name__ == "__main__":`:** On Windows, multiprocessing re-imports the module, causing infinite recursion without the guard.
2. **Using Multiprocessing for I/O-bound Tasks:** Process overhead outweighs benefits for I/O work — threads or asyncio are better.
3. **Pickling Limitations:** Arguments and return values must be picklable; lambdas, nested functions, and some objects are not.
4. **Excessive Shared State:** Overusing shared memory or queues defeats the isolation benefit and creates bottlenecks.
5. **Starting Too Many Processes:** Creating more processes than CPU cores causes context-switching overhead.
6. **Ignoring Start Methods:** The default `fork` (Unix) vs `spawn` (Windows) behave differently with global state.
7. **Not Handling Exceptions in Children:** An exception in a child process is silently lost unless explicitly caught.

## Interview Questions

### Beginner

1. How do you create a new process in Python?
2. What is the difference between `multiprocessing.Process` and `threading.Thread`?
3. What must you always guard the process-spawning code with?
4. How does `Pool.map` work?
5. How do you pass data between processes?

### Intermediate

1. Explain why multiprocessing bypasses the GIL.
2. What are the differences between `Queue`, `Pipe`, and shared `Value`/`Array`?
3. How does `ProcessPoolExecutor` differ from `multiprocessing.Pool`?
4. What is the impact of the `spawn`, `fork`, and `forkserver` start methods?
5. How would you implement a progress bar for tasks running in a `Pool`?

### Advanced

1. Design a distributed task queue using `multiprocessing` (master-worker pattern with result collection).
2. How would you handle graceful shutdown of child processes on `Ctrl+C`?
3. Implement a lock-free shared counter using `multiprocessing.RawValue` and compare its performance to a locked version.

## Practice Problems

### Easy

1. **Parallel Squares:** Use `Pool.map` to compute squares of numbers 1–100.
2. **Process Count:** Write a program that prints the PID of the parent and child process.
3. **File Line Counter:** Count total lines in multiple files using a process pool.
4. **Parallel Sleep:** Start 4 processes that each sleep for 2 seconds and print a message — measure total time.
5. **Word Frequency:** Count word frequencies across 5 text files in parallel.

### Medium

1. **Parallel Merge Sort:** Implement a merge sort that splits into subprocesses at each level.
2. **Matrix Multiplication:** Multiply two large matrices by splitting row computation across processes.
3. **Web Scraper with Pool:** Scrape 50 URLs using `ProcessPoolExecutor` and collect response times.
4. **Shared Counter:** Use `multiprocessing.Value` to maintain a shared counter incremented by 10 processes.
5. **Pipe Chat:** Implement a two-way chat between parent and child using `Pipe`.

### Hard

1. **Process Pool from Scratch:** Implement a reusable process pool using `Queue` and `Process`.
2. **Parallel MapReduce:** Implement a simple MapReduce framework — split input, map in parallel, reduce results.
3. **Fault-Tolerant Worker:** Design a system where if a child process crashes, its work is reassigned to another worker.

## Solutions

### Solution to Easy 1: Parallel Squares

```python
import multiprocessing

def square(n):
    return n * n

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        print(pool.map(square, range(1, 101)))
# Output: [1, 4, 9, 16, ..., 10000]
```

### Solution to Medium 1: Parallel Merge Sort

```python
import multiprocessing

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_merge_sort(arr, depth=0, max_depth=2):
    if len(arr) <= 1:
        return arr
    if depth >= max_depth:
        return merge_sort(arr)
    mid = len(arr) // 2
    with multiprocessing.Pool(2) as pool:
        left, right = pool.map(parallel_merge_sort, [arr[:mid], arr[mid:]],
                               [depth + 1, depth + 1], [max_depth, max_depth])
    return merge(left, right)

if __name__ == "__main__":
    import random
    data = [random.randint(0, 1000) for _ in range(100)]
    sorted_data = parallel_merge_sort(data)
    print(sorted_data == sorted(data))
# Output: True
```

### Solution to Hard 1: Process Pool from Scratch

```python
import multiprocessing
import queue

class SimplePool:
    def __init__(self, num_workers):
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=self._worker)
            p.start()
            self.workers.append(p)

    def _worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            func, args = task
            try:
                result = func(*args)
                self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put(e)

    def submit(self, func, args):
        self.task_queue.put((func, args))

    def shutdown(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join()
```

## Related Concepts

- **Multithreading (PYT-051):** Thread-based concurrency within a single process.
- **Asyncio (PYT-053):** Single-threaded cooperative multitasking.
- **Concurrency Patterns (PYT-054):** Higher-level patterns combining threading, multiprocessing, and asyncio.

## Next Concepts

- **053 — Asyncio:** Event-loop-based concurrency for I/O-bound tasks.
- **054 — Concurrency Patterns:** Design patterns for concurrent systems.
- **055 — Performance Optimization:** Profiling and accelerating Python code.

## Summary

Multiprocessing enables true parallel execution by spawning multiple OS processes, each with its own Python interpreter. The `multiprocessing` module provides `Process`, `Pool`, `Queue`, `Pipe`, and shared memory primitives. Unlike threading, multiprocessing bypasses the GIL, making it suitable for CPU-bound tasks. Communication between processes requires serialization (pickling), and data sharing must be explicit.

## Key Takeaways

- Use multiprocessing for CPU-bound tasks to achieve true parallelism across cores.
- Always guard process-spawning code with `if __name__ == "__main__":`.
- Prefer `Pool.map` and `ProcessPoolExecutor` over manual process management.
- Use `Queue` or `Pipe` for inter-process communication.
- Avoid excessive shared state — prefer message passing.
- Each process has its own GIL, enabling true parallel execution.
- Process creation overhead is significant — reuse pools rather than creating short-lived processes.
