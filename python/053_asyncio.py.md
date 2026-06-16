# Concept: Async/Await (Asyncio)

## Concept ID

PYT-053

## Difficulty

Advanced

## Domain

Python

## Module

Concurrency and Performance

## Learning Objectives

- Understand the event loop and cooperative multitasking model
- Write coroutines using `async def` and `await`
- Schedule and gather multiple coroutines with `create_task` and `gather`
- Use `asyncio.run()` to execute the top-level coroutine
- Distinguish between coroutines, awaitables, futures, and tasks
- Perform async I/O operations with libraries like `aiohttp`
- Handle timeouts and cancellation in async code

## Prerequisites

- Understanding of functions and generators
- Familiarity with I/O-bound operations
- Basic knowledge of threading (PYT-051) for comparison

## Definition

Asyncio is Python's library for writing concurrent code using the `async`/`await` syntax. It uses an event loop to manage cooperative multitasking: tasks voluntarily yield control at `await` points, allowing other tasks to run. This enables efficient concurrency for I/O-bound workloads without the overhead of threads or processes.

## Intuition

Think of a single chef with a timer for each dish. Instead of staring at a pot waiting for water to boil (blocking), the chef sets a timer, starts chopping vegetables, sets another timer for the sauce, and responds to each timer as it rings. The chef is doing one thing at a time but never idle-waits. Asyncio works the same way: a single thread runs an event loop that schedules coroutines, and whenever a coroutine awaits I/O, the loop switches to another ready coroutine.

## Why This Concept Matters

Modern applications spend most of their time waiting — for network responses, database queries, file reads, API calls. Asyncio allows handling thousands of such operations concurrently with minimal overhead. A thread-based web server might handle 5000 connections with 5000 threads (heavy), but an async server can handle 50000 connections on a single thread (lightweight). Asyncio is the foundation of modern Python web frameworks (FastAPI, aiohttp) and is increasingly important in data engineering and AI pipelines.

## Real World Examples

1. **Web Servers:** FastAPI and aiohttp use asyncio to handle thousands of concurrent requests.
2. **Web Scraping:** Fetching hundreds of URLs concurrently with asynchronous HTTP clients.
3. **Database Access:** Async database drivers (asyncpg, aiomysql) handle connection pooling without threads.
4. **Real-Time Services:** WebSockets, chat servers, and streaming data pipelines.
5. **Microservices:** Coordinating calls to multiple downstream services concurrently.

## AI/ML Relevance

- **Async Data Preprocessing:** Loading, transforming, and augmenting data batches while the model trains.
- **Concurrent API Calls:** Fetching predictions from multiple model endpoints simultaneously.
- **Streaming Inference:** Processing real-time data streams with async iterators.
- **Distributed Training Coordination:** Using async patterns for parameter server communication.
- **Experiment Tracking:** Logging metrics and artifacts to remote servers without blocking training.

## Code Examples

### Example 1: Basic Coroutine

```python
import asyncio

async def greet(name):
    print(f"Hello, {name}!")
    await asyncio.sleep(1)
    print(f"Goodbye, {name}!")

async def main():
    await greet("Alice")

asyncio.run(main())
# Output:
# Hello, Alice!
# Goodbye, Alice!
```

### Example 2: Concurrent Execution with create_task

```python
import asyncio

async def fetch_data(delay, name):
    print(f"Fetching {name}...")
    await asyncio.sleep(delay)
    print(f"Finished {name}")
    return f"Data from {name}"

async def main():
    task1 = asyncio.create_task(fetch_data(2, "source A"))
    task2 = asyncio.create_task(fetch_data(1, "source B"))
    result1 = await task1
    result2 = await task2
    print(result1, result2)

asyncio.run(main())
# Output:
# Fetching source A...
# Fetching source B...
# Finished source B
# Finished source A
# Data from source A Data from source B
```

### Example 3: asyncio.gather

```python
import asyncio

async def worker(n):
    await asyncio.sleep(n)
    return f"Worker {n} done"

async def main():
    results = await asyncio.gather(worker(3), worker(1), worker(2))
    print(results)

asyncio.run(main())
# Output:
# Worker 1 done
# Worker 2 done
# Worker 3 done
# ['Worker 3 done', 'Worker 1 done', 'Worker 2 done']
```

### Example 4: Async Context Manager

```python
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.5)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.5)

    async def use(self):
        print("Using resource")

async def main():
    async with AsyncResource() as res:
        await res.use()

asyncio.run(main())
# Output:
# Acquiring resource
# Using resource
# Releasing resource
```

### Example 5: Async Iterator

```python
import asyncio

class AsyncCounter:
    def __init__(self, limit):
        self.limit = limit

    def __aiter__(self):
        self.n = 0
        return self

    async def __anext__(self):
        if self.n >= self.limit:
            raise StopAsyncIteration
        await asyncio.sleep(0.2)
        self.n += 1
        return self.n

async def main():
    async for num in AsyncCounter(5):
        print(num)

asyncio.run(main())
# Output:
# 1
# 2
# 3
# 4
# 5
```

### Example 6: Timeout with asyncio.wait_for

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "Done"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2)
        print(result)
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(main())
# Output: Operation timed out!
```

### Example 7: Async HTTP with aiohttp

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        "http://example.com",
        "http://httpbin.org/get",
        "http://jsonplaceholder.typicode.com/posts/1"
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    for url, content in zip(urls, results):
        print(f"{url}: {len(content)} bytes")

asyncio.run(main())
# Output:
# http://example.com: 1256 bytes
# http://httpbin.org/get: 312 bytes
# http://jsonplaceholder.typicode.com/posts/1: 292 bytes
```

## Common Mistakes

1. **Blocking the Event Loop:** Calling `time.sleep()` instead of `await asyncio.sleep()` blocks the entire loop.
2. **Forgetting `await`:** Calling an `async` function without `await` returns a coroutine object, not the result.
3. **Mixing Sync and Async I/O:** Using blocking file reads or database drivers inside async code defeats concurrency.
4. **Not Using `create_task`:** Awaiting coroutines sequentially instead of scheduling them as tasks.
5. **Ignoring Exception Handling:** An unhandled exception in a task is silently swallowed unless the task is awaited.
6. **Creating Too Many Tasks:** Submitting thousands of tasks without rate limiting can overwhelm resources.
7. **Using `asyncio.run()` Multiple Times:** Calling `asyncio.run()` more than once in the same process is an error.

## Interview Questions

### Beginner

1. What does `async def` do?
2. What does the `await` keyword do?
3. How do you run a top-level coroutine?
4. What is the difference between a coroutine and a task?
5. What happens if you call an `async` function without `await`?

### Intermediate

1. Explain the event loop. How does cooperative multitasking work?
2. How does `asyncio.gather` differ from awaiting tasks individually?
3. What is the difference between `asyncio.create_task` and `asyncio.ensure_future`?
4. How do you implement an async context manager?
5. What is `asyncio.wait_for` and how do you handle timeouts?

### Advanced

1. Implement a simple event loop from scratch using generators.
2. How would you build an async rate limiter that allows N requests per second?
3. Explain how async/await is implemented at the CPython level (C stack, yield from, etc.).

## Practice Problems

### Easy

1. **Async Counter:** Write an async function that counts from 1 to 10 with a 0.5-second delay between each.
2. **Parallel Greetings:** Create 3 coroutines that print greeting messages after different delays.
3. **Fetch Sizes:** Use `aiohttp` to fetch 5 URLs and print their content lengths.
4. **Timer:** Implement an async countdown timer that prints seconds remaining every second.
5. **Async Sum:** Create two coroutines that each sum half of a list and return the result.

### Medium

1. **Async Queue:** Implement a producer-consumer pattern using `asyncio.Queue`.
2. **Rate Limiter:** Write a decorator that limits an async function to N calls per second.
3. **Concurrent Web Scraper:** Scrape 20 URLs concurrently and save results, respecting `robots.txt`.
4. **Async File Reader:** Read 10 large files asynchronously using `aiofiles`.
5. **Retry with Backoff:** Implement an async retry wrapper with exponential backoff.

### Hard

1. **Custom Event Loop:** Implement a minimal event loop using `selectors` and callbacks.
2. **Async Task Scheduler:** Build a scheduler that runs tasks at specific times or intervals.
3. **Distributed Async Map:** Implement an async `map` that distributes work across multiple machines.

## Solutions

### Solution to Easy 1: Async Counter

```python
import asyncio

async def count():
    for i in range(1, 11):
        print(i)
        await asyncio.sleep(0.5)

asyncio.run(count())
# Output: 1 2 3 4 5 6 7 8 9 10 (with 0.5s delays)
```

### Solution to Medium 1: Async Queue

```python
import asyncio

async def producer(queue):
    for i in range(10):
        await queue.put(i)
        print(f"Produced: {i}")
        await asyncio.sleep(0.2)
    await queue.put(None)

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        await asyncio.sleep(0.3)

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(producer(queue), consumer(queue))

asyncio.run(main())
# Output: Interleaved produce/consume messages
```

### Solution to Hard 1: Minimal Event Loop

```python
import selectors
import time
from collections import deque

class SimpleLoop:
    def __init__(self):
        self.ready = deque()
        self.selector = selectors.DefaultSelector()

    def call_soon(self, callback):
        self.ready.append(callback)

    def run_forever(self):
        while self.ready or self.selector.get_map():
            if not self.ready:
                events = self.selector.select(timeout=1)
                for key, _ in events:
                    self.ready.append(key.data)
            while self.ready:
                callback = self.ready.popleft()
                callback()

loop = SimpleLoop()

def hello():
    print("Hello")
    loop.call_soon(world)

def world():
    print("World")

loop.call_soon(hello)
loop.run_forever()
# Output: Hello World
```

## Related Concepts

- **Multithreading (PYT-051):** Thread-based concurrency with preemptive scheduling.
- **Multiprocessing (PYT-052):** Parallelism via separate OS processes.
- **Concurrency Patterns (PYT-054):** Higher-level patterns for coordinating async tasks.
- **Generators:** The foundation of coroutines — `yield` and `yield from`.

## Next Concepts

- **054 — Concurrency Patterns:** Combining async with other concurrency models.
- **055 — Performance Optimization:** Profiling async and sync code.

## Summary

Asyncio provides a framework for writing single-threaded concurrent code using `async`/`await`. The event loop schedules coroutines cooperatively: each coroutine runs until it hits an `await`, at which point the loop switches to another ready coroutine. Asyncio excels at I/O-bound workloads where tasks spend most of their time waiting, making it much more efficient than threads for managing thousands of concurrent connections.

## Key Takeaways

- Use `async def` to define coroutines and `await` to yield control.
- Run the top-level entry point with `asyncio.run(main())`.
- Use `create_task` to schedule coroutines for concurrent execution.
- Use `asyncio.gather` to await multiple tasks and collect results.
- Never call blocking functions inside coroutines — use async equivalents.
- The event loop is single-threaded — no GIL issues, no race conditions on shared data.
- Always handle exceptions in tasks; unhandled exceptions are silently lost.
