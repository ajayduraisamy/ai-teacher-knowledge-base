# Concept: Concurrency Patterns

## Concept ID

PYT-054

## Difficulty

Advanced

## Domain

Python

## Module

Concurrency and Performance

## Learning Objectives

- Identify and implement common concurrency patterns: producer-consumer, thread pool, futures
- Compare thread pools vs process pools and choose appropriately
- Use `concurrent.futures` for high-level asynchronous execution
- Recognize and prevent deadlocks in concurrent systems
- Understand lock-free programming principles
- Coordinate concurrent tasks with barriers

## Prerequisites

- Solid understanding of threading (PYT-051) and multiprocessing (PYT-052)
- Familiarity with asyncio (PYT-053) basics
- Understanding of synchronization primitives (Lock, Semaphore, Queue)

## Definition

Concurrency patterns are reusable solutions to common problems in concurrent programming. They provide structured ways to coordinate work across threads, processes, or coroutines. Patterns like producer-consumer, thread pool, and futures abstract away low-level synchronization details, making concurrent code more maintainable and less error-prone.

## Intuition

Building concurrent software is like running a factory. You have workers (threads/processes), assembly lines (pipelines), and inventory (queues). Without a blueprint, workers bump into each other, inventory piles up, or some stations sit idle. Concurrency patterns are the factory blueprints that ensure smooth, efficient operation.

## Why This Concept Matters

Writing correct concurrent code is notoriously difficult — race conditions, deadlocks, and livelocks are subtle and hard to debug. Concurrency patterns provide battle-tested solutions that avoid these pitfalls. Understanding these patterns is essential for building scalable systems, from web servers to data processing pipelines to distributed ML training frameworks.

## Real World Examples

1. **Web Servers:** Thread pool pattern handles incoming HTTP requests with a fixed worker pool.
2. **Data Pipelines:** Producer-consumer pattern stages data processing steps (fetch → transform → load).
3. **Image Processing:** Process pool distributes CPU-intensive filter operations across cores.
4. **Chat Servers:** Fan-out pattern broadcasts messages to all connected clients.
5. **Build Systems:** Parallel task execution with dependency graphs (like Make or Bazel).

## AI/ML Relevance

- **Distributed Training:** Parameter server pattern with workers computing gradients and a server updating weights.
- **Data Loading Pipeline:** Producer-consumer with prefetching: producer reads from disk, consumer augments and batches.
- **Ensemble Evaluation:** Futures pattern to run multiple model inferences concurrently.
- **Hyperparameter Search:** Fan-out pattern distributes training jobs across a cluster.
- **Model Serving:** Thread pool handles concurrent prediction requests with batching.

## Code Examples

### Example 1: Producer-Consumer with Queue

```python
import threading
import queue
import time

def producer(q, items):
    for item in items:
        q.put(item)
        print(f"Produced: {item}")
        time.sleep(0.1)
    q.put(None)  # Sentinel

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        print(f"Consumed: {item}")
        q.task_done()

q = queue.Queue(maxsize=3)
t1 = threading.Thread(target=producer, args=(q, range(5)))
t2 = threading.Thread(target=consumer, args=(q,))
t1.start()
t2.start()
t1.join()
t2.join()
# Output: Interleaved produce/consume with bounded buffer
```

### Example 2: Thread Pool vs Process Pool

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def io_bound(n):
    time.sleep(0.5)
    return n * 2

def cpu_bound(n):
    return sum(i * i for i in range(n))

# I/O-bound: ThreadPoolExecutor is faster
with ThreadPoolExecutor(max_workers=4) as ex:
    start = time.time()
    results = list(ex.map(io_bound, range(8)))
    print(f"ThreadPool I/O: {time.time() - start:.2f}s")

# CPU-bound: ProcessPoolExecutor is faster
with ProcessPoolExecutor(max_workers=4) as ex:
    start = time.time()
    results = list(ex.map(cpu_bound, [5_000_000] * 4))
    print(f"ProcessPool CPU: {time.time() - start:.2f}s")
# Output (varies):
# ThreadPool I/O: 1.02s
# ProcessPool CPU: 0.45s
```

### Example 3: Futures and Callbacks

```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(url):
    time.sleep(1)
    return f"Data from {url}"

def on_finished(future):
    print(f"Callback: {future.result()}")

with ThreadPoolExecutor(max_workers=3) as executor:
    future = executor.submit(fetch_data, "http://example.com")
    future.add_done_callback(on_finished)
    print("Main continues while task runs...")
    result = future.result()  # Block until done
    print(f"Got: {result}")
# Output:
# Main continues while task runs...
# Callback: Data from http://example.com
# Got: Data from http://example.com
```

### Example 4: Barrier Synchronization

```python
import threading

barrier = threading.Barrier(3)

def worker(name):
    print(f"{name} working...")
    import time
    time.sleep(hash(name) % 3)
    print(f"{name} waiting at barrier")
    barrier.wait()
    print(f"{name} passed barrier")

threads = [threading.Thread(target=worker, args=(f"W{i}",)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Output (all pass barrier simultaneously):
# W0 working...
# W1 working...
# W2 working...
# W0 waiting at barrier
# W1 waiting at barrier
# W2 waiting at barrier
# W0 passed barrier
# W1 passed barrier
# W2 passed barrier
```

### Example 5: Deadlock Prevention with Ordered Locking

```python
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def transfer(from_lock, to_lock, amount):
    # Always acquire locks in the same order (by id)
    first, second = (from_lock, to_lock) if id(from_lock) < id(to_lock) else (to_lock, from_lock)
    with first:
        with second:
            print(f"Transferred {amount}")

t1 = threading.Thread(target=transfer, args=(lock_a, lock_b, 100))
t2 = threading.Thread(target=transfer, args=(lock_b, lock_a, 50))
t1.start()
t2.start()
t1.join()
t2.join()
# Output (no deadlock):
# Transferred 100
# Transferred 50
```

### Example 6: Pipeline Pattern with Queues

```python
import threading
import queue
import time

def stage1(in_q, out_q):
    for i in range(5):
        item = in_q.get()
        result = item * 2
        time.sleep(0.1)
        out_q.put(result)
        print(f"Stage1: {item} -> {result}")

def stage2(in_q, out_q):
    for i in range(5):
        item = in_q.get()
        result = item + 1
        time.sleep(0.1)
        out_q.put(result)
        print(f"Stage2: {item} -> {result}")

input_q = queue.Queue()
middle_q = queue.Queue()
output_q = queue.Queue()

for i in range(5):
    input_q.put(i)

t1 = threading.Thread(target=stage1, args=(input_q, middle_q))
t2 = threading.Thread(target=stage2, args=(middle_q, output_q))
t1.start()
t2.start()
t1.join()
t2.join()
# Output:
# Stage1: 0 -> 0
# Stage2: 0 -> 1
# Stage1: 1 -> 2
# Stage2: 2 -> 3
# ... (sequential pipeline processing)
```

### Example 7: Fan-Out / Fan-In

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_chunk(chunk):
    time.sleep(0.5)
    return sum(chunk)

data = list(range(100))
chunks = [data[i:i + 20] for i in range(0, 100, 20)]

results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_chunk, c): c for c in chunks}
    for future in as_completed(futures):
        results.append(future.result())

print(f"Total: {sum(results)} (expected 4950)")
# Output: Total: 4950 (expected 4950)
```

## Common Mistakes

1. **Mixing Thread Pools with CPU-Bound Work:** Thread pool won't speed up CPU-bound tasks due to GIL.
2. **Not Handling Future Exceptions:** Calling `future.result()` can raise exceptions that were not caught.
3. **Deadlock with Nested Locks:** Acquiring locks in inconsistent order across threads.
4. **Fixed-Size Pool Starvation:** A pool that is too small can cause task starvation and deadlock.
5. **Ignoring Thread Safety in Callbacks:** Callbacks registered with `add_done_callback` must be thread-safe.
6. **Using Sentinels Incorrectly:** Multiple consumers need multiple sentinels or a poison pill pattern.
7. **Oversubscribing Workers:** Creating more workers than CPU cores for CPU-bound tasks harms performance.

## Interview Questions

### Beginner

1. What is the producer-consumer pattern?
2. What is a thread pool and why use one?
3. What does `concurrent.futures.Future` represent?
4. How does `ThreadPoolExecutor.map` work?
5. What is a sentinel value in queue-based patterns?

### Intermediate

1. Compare thread pools and process pools — when do you use each?
2. Explain the fan-out / fan-in pattern with an example.
3. What is a barrier in concurrent programming?
4. How do you prevent deadlocks when multiple locks are needed?
5. What is the difference between `submit` and `map` in `ThreadPoolExecutor`?

### Advanced

1. Design a pipeline pattern that allows dynamic addition/removal of stages.
2. Implement a lock-free ring buffer for single-producer, single-consumer.
3. How would you implement work-stealing in a thread pool?

## Practice Problems

### Easy

1. **Thread Pool:** Use `ThreadPoolExecutor` to download 10 URLs and print their sizes.
2. **Producer-Consumer:** Implement a simple producer that generates numbers and a consumer that prints them.
3. **Parallel Map:** Use `ProcessPoolExecutor.map` to compute factorials of 20 numbers.
4. **Future Wait:** Submit 5 tasks and wait for the first one to complete.
5. **Barrier:** Synchronize 4 threads to all print "Ready" simultaneously.

### Medium

1. **Pipeline:** Implement a 3-stage pipeline (read → process → write) using queues.
2. **Bounded Buffer:** Implement a thread-safe bounded buffer with blocking put/get.
3. **Retry Pattern:** Wrap a task in a future that retries on failure up to 3 times.
4. **Priority Queue Worker:** Implement a worker pool that processes tasks by priority.
5. **Rate-Limited Worker:** Limit a thread pool to at most 10 tasks per second.

### Hard

1. **Work-Stealing Scheduler:** Implement a thread pool where idle threads steal tasks from busy threads.
2. **Distributed MapReduce:** Simulate a distributed MapReduce with concurrent mappers and reducers.
3. **Lock-Free Hash Table:** Implement a thread-safe, lock-free hash table using atomic operations.

## Solutions

### Solution to Easy 1: Thread Pool Downloader

```python
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def download(url):
    with urllib.request.urlopen(url) as response:
        return len(response.read())

urls = ["http://example.com", "http://python.org", "http://github.com"]
with ThreadPoolExecutor(max_workers=3) as ex:
    sizes = list(ex.map(download, urls))
for url, size in zip(urls, sizes):
    print(f"{url}: {size} bytes")
# Output:
# http://example.com: 1256 bytes
# http://python.org: 49257 bytes
# http://github.com: 95831 bytes
```

### Solution to Medium 1: 3-Stage Pipeline

```python
import threading
import queue
import time

def read_stage(in_q, out_q):
    for item in iter(in_q.get, None):
        out_q.put(f"read:{item}")

def process_stage(in_q, out_q):
    for item in iter(in_q.get, None):
        out_q.put(f"proc:{item.upper()}")

def write_stage(in_q):
    for item in iter(in_q.get, None):
        print(f"Write: {item}")

input_q = queue.Queue()
mid_q = queue.Queue()
output_q = queue.Queue()

threads = [
    threading.Thread(target=read_stage, args=(input_q, mid_q)),
    threading.Thread(target=process_stage, args=(mid_q, output_q)),
    threading.Thread(target=write_stage, args=(output_q,)),
]
for t in threads:
    t.start()

for word in ["hello", "world", "python"]:
    input_q.put(word)
for q in [input_q, mid_q, output_q]:
    q.put(None)

for t in threads:
    t.join()
# Output:
# Write: proc:READ:HELLO
# Write: proc:READ:WORLD
# Write: proc:READ:PYTHON
```

### Solution to Hard 1: Work-Stealing Thread Pool

```python
import threading
import queue
from collections import deque

class WorkStealingPool:
    def __init__(self, num_workers):
        self.queues = [deque() for _ in range(num_workers)]
        self.locks = [threading.Lock() for _ in range(num_workers)]
        self.workers = []
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, args=(i,))
            t.start()
            self.workers.append(t)

    def submit(self, task, worker_id=0):
        with self.locks[worker_id]:
            self.queues[worker_id].append(task)

    def _worker(self, wid):
        while True:
            task = self._get_task(wid)
            if task is None:
                break
            task()

    def _get_task(self, wid):
        # Try own queue first, then steal from others
        with self.locks[wid]:
            if self.queues[wid]:
                return self.queues[wid].popleft()
        for i in range(len(self.queues)):
            if i != wid:
                with self.locks[i]:
                    if self.queues[i]:
                        return self.queues[i].popleft()
        return None
```

## Related Concepts

- **Multithreading (PYT-051):** Low-level threading primitives used in patterns.
- **Multiprocessing (PYT-052):** Process-based parallelism for CPU-bound work.
- **Asyncio (PYT-053):** Coroutine-based concurrency for I/O-bound work.
- **Performance Optimization (PYT-055):** Measuring and improving concurrent performance.

## Next Concepts

- **055 — Performance Optimization:** Profiling concurrent code and optimizing bottlenecks.
- **056 — Big O Notation:** Analyzing algorithmic efficiency.

## Summary

Concurrency patterns provide reusable solutions to common coordination problems in multithreaded and multiprocess programs. Key patterns include producer-consumer, thread pool vs process pool, futures and callbacks, barriers, fan-out/fan-in, and pipelines. The `concurrent.futures` module offers high-level abstractions (`ThreadPoolExecutor`, `ProcessPoolExecutor`, `Future`) that simplify concurrent programming and reduce errors.

## Key Takeaways

- Use `ThreadPoolExecutor` for I/O-bound tasks and `ProcessPoolExecutor` for CPU-bound tasks.
- Producer-consumer with `queue.Queue` is the foundational pattern for pipelining.
- Always acquire multiple locks in a consistent global order to prevent deadlocks.
- Use `as_completed` to process futures as they finish, not in submission order.
- Barriers synchronize groups of threads at a common point.
- Prefer high-level abstractions (`concurrent.futures`) over raw threads when possible.
- Profile before optimizing — concurrency adds complexity, ensure it actually improves performance.
