# Concept: Stacks and Queues

## Concept ID

PYT-059

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Understand LIFO (stack) and FIFO (queue) principles
- Implement a stack using Python's list (append/pop)
- Implement a queue using `collections.deque`
- Use `queue.PriorityQueue` for priority-based ordering
- Understand circular buffer implementation
- Recognize real-world applications of stacks and queues

## Prerequisites

- Basic Python lists and methods
- Understanding of linked lists (PYT-058) is helpful
- Familiarity with the `collections` module

## Definition

A stack is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end (top). A queue is a First-In-First-Out (FIFO) data structure where elements are added at the rear and removed from the front. Both are abstract data types that can be implemented using arrays or linked lists.

## Intuition

A stack is like a stack of plates: you add plates to the top and take plates from the top. The last plate you put on is the first one you take off. A queue is like a line at a coffee shop: the first person in line gets served first, and new people join at the back.

## Why This Concept Matters

Stacks and queues are fundamental to computer science. Stacks manage function calls (call stack), expression evaluation, and undo/redo functionality. Queues handle task scheduling, breadth-first search, and buffering. Understanding these structures is crucial for algorithm design and system architecture.

## Real World Examples

1. **Undo/Redo:** Text editor undo uses a stack of actions.
2. **Call Stack:** Function calls are managed with the call stack.
3. **Print Spooler:** Print jobs are queued FIFO.
4. **Task Scheduling:** Operating system process queues.
5. **Browser Back Button:** Navigation history is a stack.

## AI/ML Relevance

- **BFS/DFS in Tree Search:** Queue for BFS, stack for DFS in game AI and state-space search.
- **Backtracking in Optimization:** Stack-based backtracking for constraint satisfaction problems.
- **Reinforcement Learning:** Experience replay buffers are deques.
- **Data Loading:** Prefetch queues for training data pipelines.
- **Beam Search:** Priority queue for sequence generation in NLP.

## Code Examples

### Example 1: Stack Using List

```python
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
print(stack.pop())
print(stack.pop())
print(stack.pop())
# Output:
# 3
# 2
# 1
```

### Example 2: Queue Using deque

```python
from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
print(queue.popleft())
print(queue.popleft())
print(queue.popleft())
# Output:
# 1
# 2
# 3
```

### Example 3: Balanced Parentheses (Stack Application)

```python
def is_balanced(s):
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return len(stack) == 0

print(is_balanced("()[]{}"))
print(is_balanced("([)]"))
print(is_balanced("((()))"))
# Output:
# True
# False
# True
```

### Example 4: Priority Queue

```python
import queue

pq = queue.PriorityQueue()
pq.put((2, "medium priority"))
pq.put((1, "high priority"))
pq.put((3, "low priority"))

while not pq.empty():
    priority, task = pq.get()
    print(f"Priority {priority}: {task}")
# Output:
# Priority 1: high priority
# Priority 2: medium priority
# Priority 3: low priority
```

### Example 5: Circular Buffer (Ring Buffer)

```python
class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0

    def enqueue(self, item):
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.size
        if self.count < self.size:
            self.count += 1
        else:
            self.head = (self.head + 1) % self.size

    def dequeue(self):
        if self.count == 0:
            raise IndexError("Empty buffer")
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.size
        self.count -= 1
        return item

    def __iter__(self):
        for i in range(self.count):
            yield self.buffer[(self.head + i) % self.size]

cb = CircularBuffer(3)
for x in [1, 2, 3, 4, 5]:
    cb.enqueue(x)
print(list(cb))
# Output: [3, 4, 5]
```

### Example 6: Stack Implementation using collections.deque

```python
from collections import deque

class Stack:
    def __init__(self):
        self._data = deque()

    def push(self, item):
        self._data.append(item)

    def pop(self):
        return self._data.pop()

    def peek(self):
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)

s = Stack()
s.push(10)
s.push(20)
s.push(30)
print(s.pop())
print(s.peek())
print(s.size())
# Output:
# 30
# 20
# 2
```

### Example 7: Queue Implementation Using Two Stacks

```python
class QueueWithTwoStacks:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def enqueue(self, item):
        self.stack_in.append(item)

    def dequeue(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        if not self.stack_out:
            raise IndexError("Empty queue")
        return self.stack_out.pop()

    def is_empty(self):
        return not self.stack_in and not self.stack_out

q = QueueWithTwoStacks()
for x in [1, 2, 3]:
    q.enqueue(x)
print(q.dequeue())
print(q.dequeue())
q.enqueue(4)
print(q.dequeue())
print(q.dequeue())
# Output:
# 1
# 2
# 3
# 4
```

## Common Mistakes

1. **Using List as Queue:** `list.pop(0)` is O(n) — use `collections.deque.popleft()` for O(1).
2. **Empty Structure Operations:** Calling `pop()` or `popleft()` on an empty structure raises `IndexError`.
3. **Confusing Stack and Queue Order:** Using stack where queue is needed (or vice versa) gives wrong results.
4. **Priority Queue with Comparable Items:** Items must be comparable — wrap in a tuple if needed.
5. **Circular Buffer Overwrite:** Not checking if buffer is full before overwriting oldest data.
6. **Mutable Default Items:** Sharing mutable objects across queue items causes aliasing bugs.
7. **Unbounded Growth:** Using a plain list as a queue in a long-running program causes unbounded memory growth.

## Interview Questions

### Beginner

1. What is the difference between a stack and a queue?
2. How do you implement a stack in Python?
3. How do you implement a queue in Python using `collections.deque`?
4. What is LIFO and FIFO?
5. What is the time complexity of push and pop operations on a stack?

### Intermediate

1. Implement a queue using two stacks.
2. How would you implement a priority queue in Python?
3. What is a circular buffer and when would you use it?
4. Write a function to reverse a string using a stack.
5. How do you implement a deque from scratch using a linked list?

### Advanced

1. Design a thread-safe bounded queue using `threading.Condition`.
2. Implement a min-stack that supports push, pop, and retrieving the minimum element in O(1).
3. How would you implement a monotonic queue for sliding window maximum problems?

## Practice Problems

### Easy

1. **Reverse String:** Use a stack to reverse a string.
2. **Valid Parentheses:** Check if parentheses in a string are balanced.
3. **Queue from Stack:** Implement a queue using two stacks.
4. **Stack Sort:** Sort a stack using only one additional stack.
5. **Recent Calls:** Count recent requests within a time window using a queue.

### Medium

1. **Min Stack:** Implement a stack that tracks the minimum element.
2. **Sliding Window Maximum:** Find maximum in each sliding window using a deque.
3. **Browser History:** Implement forward/back navigation using two stacks.
4. **Task Scheduler:** Schedule tasks with cooldown periods using a queue.
5. **Circular Deque:** Implement a circular deque from scratch.

### Hard

1. **Max Stack:** Implement a stack that supports push, pop, and retrieving the maximum in O(1).
2. **LRU Cache:** Implement LRU cache using a doubly linked list and a dict.
3. **Thread-Safe Queue:** Implement a thread-safe bounded queue for multi-producer, multi-consumer.

## Solutions

### Solution to Easy 1: Reverse String

```python
def reverse_string(s):
    stack = list(s)
    result = ""
    while stack:
        result += stack.pop()
    return result

print(reverse_string("hello"))
# Output: olleh
```

### Solution to Medium 1: Min Stack

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            return self.stack.pop()

    def top(self):
        return self.stack[-1] if self.stack else None

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

ms = MinStack()
ms.push(3)
ms.push(5)
print(ms.get_min())
ms.push(2)
ms.push(1)
print(ms.get_min())
ms.pop()
print(ms.get_min())
# Output:
# 3
# 1
# 2
```

### Solution to Hard 1: Max Stack

```python
from collections import deque

class MaxStack:
    def __init__(self):
        self.stack = deque()
        self.max_stack = deque()

    def push(self, x):
        self.stack.append(x)
        if not self.max_stack or x >= self.max_stack[-1]:
            self.max_stack.append(x)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.max_stack[-1]:
                self.max_stack.pop()
            return val

    def top(self):
        return self.stack[-1] if self.stack else None

    def get_max(self):
        return self.max_stack[-1] if self.max_stack else None

    def pop_max(self):
        max_val = self.get_max()
        buffer = deque()
        while self.top() != max_val:
            buffer.append(self.pop())
        self.pop()
        while buffer:
            self.push(buffer.pop())
        return max_val
```

## Related Concepts

- **Recursion (PYT-057):** The call stack is the underlying mechanism for recursion.
- **Trees (PYT-060):** BFS uses a queue, DFS uses a stack (or recursion).
- **Heaps (PYT-061):** Priority queues are typically implemented with heaps.
- **Graph Algorithms (PYT-063):** BFS/DFS traversal of graphs.

## Next Concepts

- **060 — Trees:** Non-linear data structures traversed with stacks and queues.
- **061 — Heaps:** Tree-based structure for priority queues.

## Summary

Stacks are LIFO structures implemented efficiently with `list.append()`/`list.pop()` or `deque`. Queues are FIFO structures best implemented with `collections.deque.popleft()`. Priority queues order elements by priority and are available via `queue.PriorityQueue`. Circular buffers overwrite old data when full, useful for fixed-size buffers.

## Key Takeaways

- Use `list` for stacks (append/pop), `collections.deque` for queues (append/popleft).
- Never use `list.pop(0)` for queues — it is O(n).
- `queue.PriorityQueue` provides thread-safe priority ordering.
- Circular buffers are memory-efficient for fixed-size streams.
- Stacks model nested structures (parentheses, function calls).
- Queues model waiting lines (task scheduling, BFS).
- A queue can be implemented with two stacks for amortized O(1) operations.
