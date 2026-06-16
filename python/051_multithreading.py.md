# Concept: Multithreading

## Concept ID

PYT-051

## Difficulty

Advanced

## Domain

Python

## Module

Concurrency and Performance

## Learning Objectives

- Understand the difference between concurrency and parallelism
- Create and manage threads using the `threading` module
- Explain the Global Interpreter Lock (GIL) and its implications
- Use synchronization primitives (Lock, RLock, Semaphore) to prevent race conditions
- Implement thread-safe code with proper locking strategies
- Use `ThreadPoolExecutor` for high-level thread management
- Recognize and debug common threading issues like deadlocks and data races

## Prerequisites

- Python syntax fundamentals (functions, classes, exceptions)
- Basic understanding of operating system processes
- Familiarity with `time` and `random` modules
- Understanding of function arguments and closures

## Definition

Multithreading is a programming technique where multiple threads of execution exist within a single process, sharing the same memory space. In Python, the `threading` module provides a high-level interface for working with threads. Each thread runs concurrently, and the operating system scheduler determines when each thread gets CPU time.

## Intuition

Imagine a restaurant kitchen with a single chef (the CPU). The chef can handle only one task at a time, but they can switch between tasks rapidly — chopping vegetables, then stirring a sauce, then plating a dish. This interleaving of tasks creates the illusion of doing multiple things at once. Threads work similarly: they allow a program to juggle multiple tasks by rapidly switching between them.

## Why This Concept Matters

Multithreading enables programs to remain responsive while performing blocking operations like I/O. A web server can serve thousands of clients without dedicating a full process to each. A GUI application can keep its interface responsive while downloading data in the background. Understanding threading is critical for building performant, responsive systems, even with the Python GIL constraint.

## Real World Examples

1. **Web Servers:** Handling multiple HTTP requests concurrently using a thread pool.
2. **File Processing:** Reading and validating multiple files simultaneously.
3. **Network Applications:** Managing multiple client connections in a chat server.
4. **GUI Applications:** Keeping the UI responsive while performing background tasks.
5. **Data Pipelines:** Fetching data from multiple APIs concurrently.

## AI/ML Relevance

- **Parallel Data Loading:** Loading and preprocessing training batches in parallel while the GPU computes gradients.
- **Async Inference:** Serving ML model predictions concurrently for multiple requests.
- **Hyperparameter Monitoring:** Logging metrics from a training run on a separate thread.
- **Model Ensemble:** Running multiple model evaluations concurrently.
- **Reinforcement Learning:** Collecting environment interactions on separate threads.

## Code Examples

### Example 1: Basic Thread Creation

```python
import threading
import time

def worker(name):
    print(f"Worker {name} starting")
    time.sleep(1)
    print(f"Worker {name} finished")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All workers done")
# Output:
# Worker 0 starting
# Worker 1 starting
# Worker 2 starting
# Worker 0 finished
# Worker 1 finished
# Worker 2 finished
# All workers done
```

### Example 2: Race Condition

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(100000):
        counter += 1

t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter} (expected 200000)")
# Output: Counter: 132847 (expected 200000)
# The exact value varies each run due to race conditions.
```

### Example 3: Using Lock for Thread Safety

```python
import threading

counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

t1 = threading.Thread(target=safe_increment)
t2 = threading.Thread(target=safe_increment)
t1.start()
t2.start()
t1.join()
t2.join()

print(f"Counter: {counter} (expected 200000)")
# Output: Counter: 200000 (expected 200000)
```

### Example 4: RLock (Reentrant Lock)

```python
import threading

lock = threading.RLock()

def outer():
    with lock:
        print("Outer acquired lock")
        inner()

def inner():
    with lock:  # RLock allows re-acquisition by same thread
        print("Inner acquired lock")

t = threading.Thread(target=outer)
t.start()
t.join()
# Output:
# Outer acquired lock
# Inner acquired lock
```

### Example 5: Semaphore

```python
import threading
import time

semaphore = threading.Semaphore(2)  # Allow 2 concurrent accesses

def limited_resource(name):
    with semaphore:
        print(f"{name} acquired semaphore")
        time.sleep(1)
    print(f"{name} released semaphore")

threads = [threading.Thread(target=limited_resource, args=(f"Thread-{i}",)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Output (timing shows at most 2 run concurrently):
# Thread-0 acquired semaphore
# Thread-1 acquired semaphore
# Thread-0 released semaphore
# Thread-1 released semaphore
# Thread-2 acquired semaphore
# Thread-3 acquired semaphore
# Thread-2 released semaphore
# Thread-3 released semaphore
# Thread-4 acquired semaphore
# Thread-4 released semaphore
```

### Example 6: ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_url(url):
    time.sleep(0.5)
    return f"Data from {url}"

urls = ["http://example.com", "http://python.org", "http://github.com"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_url, urls))

for r in results:
    print(r)
# Output:
# Data from http://example.com
# Data from http://python.org
# Data from http://github.com
```

### Example 7: Daemon Threads

```python
import threading
import time

def background_monitor():
    while True:
        print("Monitoring...")
        time.sleep(0.5)

daemon = threading.Thread(target=background_monitor, daemon=True)
daemon.start()

time.sleep(1.5)
print("Main program ending")
# Output:
# Monitoring...
# Monitoring...
# Monitoring...
# Main program ending
# (Daemon thread terminates when main exits.)
```

## Common Mistakes

1. **Forgetting `join()`:** The main thread exits before worker threads complete, potentially terminating them prematurely.
2. **Assuming Thread Safety:** Believing simple operations like `x += 1` are atomic — they are not in Python.
3. **Overusing Locks:** Acquiring locks unnecessarily or holding them too long, reducing concurrency.
4. **Deadlock:** Two threads each hold a lock the other needs, causing both to wait forever.
5. **Mutable Shared State Without Locks:** Passing mutable objects between threads without synchronization.
6. **Believing the GIL Makes Threading Useless:** The GIL only protects Python objects; I/O-bound tasks still benefit greatly.
7. **Creating Too Many Threads:** Each thread consumes memory (stack space). Thousands of threads can overwhelm the system.

## Interview Questions

### Beginner

1. What is the difference between a thread and a process?
2. How do you create a thread in Python using the `threading` module?
3. What does the `join()` method do on a thread?
4. What is a daemon thread?
5. How do you pass arguments to a thread's target function?

### Intermediate

1. Explain the Global Interpreter Lock (GIL). How does it affect threading?
2. What is a race condition? Give a Python example.
3. What is the difference between `Lock` and `RLock`?
4. How does `ThreadPoolExecutor` differ from manually managing threads?
5. What is a semaphore and when would you use one?

### Advanced

1. How would you implement a thread-safe singleton pattern in Python?
2. Explain how Python's GIL is released during I/O operations. What operations release the GIL?
3. Design a thread pool from scratch using only `threading` primitives.

## Practice Problems

### Easy

1. **Print Numbers:** Create two threads — one prints odd numbers 1-99, the other prints even numbers 2-100.
2. **Delayed Greeting:** Write a program that starts 5 threads, each printing "Hello from thread N" after N seconds.
3. **Word Counter:** Count the total number of words across multiple text files using threads.
4. **Timer:** Implement a countdown timer that prints remaining seconds every second using a separate thread.
5. **Parallel Sum:** Compute the sum of a large list by splitting it across 4 threads.

### Medium

1. **Thread-Safe Queue:** Implement a thread-safe bounded queue using `threading.Condition`.
2. **Reader-Writer Lock:** Implement a reader-writer lock that allows concurrent reads but exclusive writes.
3. **Parallel Web Scraper:** Fetch 20 URLs concurrently using `ThreadPoolExecutor` and collect response sizes.
4. **BarrierSync:** Use `threading.Barrier` to synchronize 5 threads to start a computation phase simultaneously.
5. **Thread Pool:** Implement a simple thread pool with a fixed number of workers and a task queue.

### Hard

1. **Deadlock Detection:** Write a program that deliberately creates a deadlock, then detect and recover from it.
2. **Lock-Free Stack:** Implement a thread-safe stack using atomic operations (via `ctypes` or `mmap`).
3. **Parallel Merge Sort:** Implement merge sort that spawns threads for sub-array sorting with a configurable depth limit.

## Solutions

### Solution to Easy 1: Print Numbers

```python
import threading

def print_odd():
    for i in range(1, 100, 2):
        print(i, end=" ")

def print_even():
    for i in range(2, 101, 2):
        print(i, end=" ")

t1 = threading.Thread(target=print_odd)
t2 = threading.Thread(target=print_even)
t1.start()
t2.start()
t1.join()
t2.join()
# Output: 1 2 3 4 ... 99 100 (order may vary)
```

### Solution to Medium 1: Thread-Safe Queue

```python
import threading
import collections

class ThreadSafeQueue:
    def __init__(self, maxsize=0):
        self.queue = collections.deque()
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.lock:
            while self.maxsize and len(self.queue) >= self.maxsize:
                self.not_full.wait()
            self.queue.append(item)
            self.not_empty.notify()

    def get(self):
        with self.lock:
            while not self.queue:
                self.not_empty.wait()
            item = self.queue.popleft()
            self.not_full.notify()
            return item
```

### Solution to Hard 1: Deadlock Detection

```python
import threading
import time

lock_a = threading.Lock()
lock_b = threading.Lock()

def thread_1():
    with lock_a:
        print("Thread 1: acquired lock_a")
        time.sleep(0.1)
        with lock_b:
            print("Thread 1: acquired lock_b")

def thread_2():
    with lock_b:
        print("Thread 2: acquired lock_b")
        time.sleep(0.1)
        with lock_a:
            print("Thread 2: acquired lock_a")

t1 = threading.Thread(target=thread_1)
t2 = threading.Thread(target=thread_2)
t1.start()
t2.start()
# Program hangs. Use timeout-based detection:
t1.join(timeout=2)
t2.join(timeout=2)
if t1.is_alive() or t2.is_alive():
    print("Deadlock detected!")
# Output:
# Thread 1: acquired lock_a
# Thread 2: acquired lock_b
# Deadlock detected!
```

## Related Concepts

- **Global Interpreter Lock (GIL):** The mechanism limiting true parallel execution in CPython.
- **Multiprocessing:** Using separate processes to bypass the GIL for CPU-bound work.
- **Asyncio:** Cooperative multitasking for I/O-bound tasks without threads.
- **Concurrent Futures:** High-level abstraction for thread and process pools.

## Next Concepts

- **052 — Multiprocessing:** Bypassing the GIL with separate processes.
- **053 — Asyncio:** Single-threaded concurrency with coroutines.
- **054 — Concurrency Patterns:** Advanced patterns for managing concurrent execution.

## Summary

Multithreading in Python allows concurrent execution of tasks within a single process. The `threading` module provides `Thread`, `Lock`, `RLock`, `Semaphore`, and `ThreadPoolExecutor`. The GIL limits threads to executing one Python bytecode instruction at a time, making threads ideal for I/O-bound rather than CPU-bound tasks. Synchronization primitives like locks prevent race conditions on shared data.

## Key Takeaways

- Threads share the same memory space and are lightweight compared to processes.
- The GIL prevents true parallelism for CPU-bound Python code but does not affect I/O-bound tasks.
- Always use locks (`Lock`, `RLock`, `Semaphore`) when accessing mutable shared state.
- Use `ThreadPoolExecutor` for managing groups of threads.
- Make daemon threads for background tasks that should not block program exit.
- Prefer `with lock:` context manager over manual `acquire()`/`release()`.
