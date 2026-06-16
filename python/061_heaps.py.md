# Concept: Heaps

## Concept ID

PYT-061

## Difficulty

Advanced

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Understand the min-heap property and complete binary tree structure
- Use the heapq module (heappush, heappop, heapify, nlargest, nsmallest)
- Implement a priority queue using a heap
- Analyze heap operation complexities (O(log n) for push/pop)
- Build a heap from a list in O(n) time
- Distinguish between min-heap and max-heap

## Prerequisites

- Understanding of trees (PYT-060) — heaps are complete binary trees
- Knowledge of lists and indexing
- Basic Big O notation (PYT-056)

## Definition

A heap is a specialized complete binary tree that satisfies the heap property: in a min-heap, every parent node is less than or equal to its children (the smallest element is at the root). In a max-heap, every parent is greater than or equal to its children. Python's heapq module implements min-heaps using lists.

## Intuition

Imagine a pyramid of numbers where the smallest number sits at the top. If you remove the top, the next smallest rises to take its place. Adding a new number might cause it to bubble up or down until the pyramid property is restored. That is a min-heap. It is like a self-organizing pile where you always have instant access to the smallest item.

## Why This Concept Matters

Heaps provide efficient priority queue operations: O(log n) insertion and O(log n) removal of the smallest element, with O(1) peek. This is critical for scheduling, graph algorithms (Dijkstra, A*), merge operations, and streaming median computation. Python's heapq is a standard library module used extensively in production systems.

## Real World Examples

1. **Task Scheduling:** Operating systems prioritize processes using heaps.
2. **Dijkstra's Algorithm:** Finds shortest paths using a min-heap.
3. **Median Maintenance:** Streaming median using two heaps.
4. **Merging Sorted Streams:** Merge K sorted files with a heap.
5. **Event Simulation:** Discrete event simulators use heaps for event ordering.

## AI/ML Relevance

- **Beam Search:** Heaps maintain top-K candidate sequences in NLP generation.
- **k-NN Search:** Heaps track the K nearest neighbors during search.
- **Priority Experience Replay:** Sampling important transitions more frequently.
- **Huffman Coding:** Building optimal prefix codes using a min-heap.
- **A* Search:** Heuristics-guided pathfinding with open set as a heap.

## Code Examples

### Example 1: Basic Heap Operations

`python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
heapq.heappush(heap, 1)
heapq.heappush(heap, 9)

print(heap)
print(heapq.heappop(heap))
print(heapq.heappop(heap))
print(heapq.heappop(heap))
# Output:
# [1, 3, 7, 5, 9]
# 1
# 3
# 5
`

### Example 2: heapify

`python
import heapq

data = [9, 5, 3, 1, 7, 2, 8, 4, 6]
heapq.heapify(data)
print(data)

sorted_data = [heapq.heappop(data) for _ in range(len(data))]
print(sorted_data)
# Output:
# [1, 4, 2, 5, 7, 3, 8, 9, 6]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]
`

### Example 3: nlargest and nsmallest

`python
import heapq
import random

data = [random.randint(0, 1000) for _ in range(100)]
print(f"3 largest: {heapq.nlargest(3, data)}")
print(f"3 smallest: {heapq.nsmallest(3, data)}")
print(f"Full sorted check: {sorted(data)[:3]} == {heapq.nsmallest(3, data)}")
# Output:
# 3 largest: [997, 982, 965]
# 3 smallest: [2, 7, 15]
# Full sorted check: [2, 7, 15] == [2, 7, 15]
`

### Example 4: Priority Queue with Tuples

`python
import heapq

class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, item, priority):
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        return heapq.heappop(self._heap)[-1]

    def is_empty(self):
        return len(self._heap) == 0

pq = PriorityQueue()
pq.push("low task", 3)
pq.push("high task", 1)
pq.push("medium task", 2)

while not pq.is_empty():
    print(pq.pop())
# Output:
# high task
# medium task
# low task
`

### Example 5: Max-Heap via Negation

`python
import heapq

data = [4, 1, 7, 3, 8, 5]
max_heap = [-x for x in data]
heapq.heapify(max_heap)

print(f"Max element: {-max_heap[0]}")
print(f"Sorted descending: {[-heapq.heappop(max_heap) for _ in range(len(max_heap))]}")
# Output:
# Max element: 8
# Sorted descending: [8, 7, 5, 4, 3, 1]
`

### Example 6: Merge K Sorted Lists

`python
import heapq

def merge_k_sorted(lists):
    heap = []
    result = []

    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        value, list_idx, elem_idx = heapq.heappop(heap)
        result.append(value)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result

lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted(lists))
# Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
`

### Example 7: Running Median

`python
import heapq

class RunningMedian:
    def __init__(self):
        self.low = []  # max-heap (store negatives)
        self.high = []  # min-heap

    def add(self, value):
        if not self.low or value <= -self.low[0]:
            heapq.heappush(self.low, -value)
        else:
            heapq.heappush(self.high, value)

        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2.0

rm = RunningMedian()
for v in [5, 15, 1, 3, 10]:
    rm.add(v)
    print(f"After {v}: median = {rm.median()}")
# Output:
# After 5: median = 5
# After 15: median = 10.0
# After 1: median = 5
# After 3: median = 4.0
# After 10: median = 5
`

## Common Mistakes

1. **Forgetting heapify:** Using a list as a heap without calling heapify first.
2. **Modifying Heap Elements:** Changing a heap element's value after insertion breaks the heap property.
3. **Using heapq for Max-Heap Without Negation:** heapq only supports min-heaps natively.
4. **Confusing Priority Order:** Lower priority value comes out first in a min-heap.
5. **Not Handling Ties in Priority Queue:** Without a tiebreaker, items with equal priority are ordered by the next tuple element.
6. **Assuming Heap Sort Is Stable:** Heap sort is not a stable sort.
7. **Building Heap by Repeated Pushes:** Creating a heap by pushing N items is O(N log N); heapify is O(N).

## Interview Questions

### Beginner

1. What is a heap data structure?
2. What is the difference between a min-heap and a max-heap?
3. How do you push an element onto a heap?
4. How do you pop the smallest element from a heap?
5. What is the time complexity of heap push and pop?

### Intermediate

1. How does heapify work in O(n) time?
2. How would you implement a max-heap using heapq?
3. How do you find the K largest elements in a list using a heap?
4. What is the difference between a heap and a sorted list?
5. How would you merge K sorted lists using a heap?

### Advanced

1. Implement a heap from scratch using an array (including sift-up and sift-down).
2. How would you implement a priority queue with decrease-key operation?
3. Design an algorithm to find the median of a stream of numbers using two heaps.

## Practice Problems

### Easy

1. **Kth Smallest:** Find the Kth smallest element in a list.
2. **Heap Sort:** Implement heap sort using heapq.
3. **Min Cost to Connect Sticks:** Connect sticks with min cost (always combine smallest two).
4. **Kth Largest:** Find the Kth largest element using a heap.
5. **Top K Frequent:** Find the K most frequent elements in a list.

### Medium

1. **Merge K Sorted Lists:** Merge K sorted linked lists.
2. **Find Median from Data Stream:** Implement a class for streaming median.
3. **Task Scheduler:** Rearrange tasks so same tasks are K apart.
4. **Sliding Window Maximum:** Find max in each sliding window (heap + lazy deletion).
5. **Ugly Number II:** Find the nth ugly number using a heap.

### Hard

1. **Heap Implementation:** Implement a heap from scratch with sift-up and sift-down.
2. **Skyline Problem:** Find the skyline of a city using a heap.
3. **Dijkstra's Algorithm:** Implement shortest path with a heap-based priority queue.

## Solutions

### Solution to Easy 1: Kth Smallest

`python
import heapq

def kth_smallest(nums, k):
    return heapq.nsmallest(k, nums)[-1]

print(kth_smallest([3, 1, 4, 1, 5, 9, 2, 6], 3))
# Output: 2
`

### Solution to Medium 1: Merge K Sorted Lists

`python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode()
    current = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = ListNode(val)
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next

# Test
l1 = ListNode(1, ListNode(4, ListNode(7)))
l2 = ListNode(2, ListNode(5, ListNode(8)))
l3 = ListNode(3, ListNode(6, ListNode(9)))

merged = merge_k_lists([l1, l2, l3])
result = []
while merged:
    result.append(merged.val)
    merged = merged.next
print(result)
# Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
`

### Solution to Hard 1: Heap Implementation

`python
class MinHeap:
    def __init__(self):
        self.heap = []

    def _parent(self, i): return (i - 1) // 2
    def _left(self, i): return 2 * i + 1
    def _right(self, i): return 2 * i + 2

    def _sift_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self._parent(i)]:
            self.heap[i], self.heap[self._parent(i)] = self.heap[self._parent(i)], self.heap[i]
            i = self._parent(i)

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left = self._left(i)
            right = self._right(i)
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest != i:
                self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
                i = smallest
            else:
                break

    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            raise IndexError("pop from empty heap")
        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        if self.heap:
            self._sift_down(0)
        return min_val

    def peek(self):
        return self.heap[0] if self.heap else None

h = MinHeap()
for v in [5, 3, 7, 1, 9]:
    h.push(v)
print([h.pop() for _ in range(5)])
# Output: [1, 3, 5, 7, 9]
`

## Related Concepts

- **Trees (PYT-060):** Heaps are complete binary trees stored in arrays.
- **Priority Queues (PYT-059):** Heaps are the standard implementation.
- **Sorting (PYT-064):** Heap sort is an O(n log n) sorting algorithm.
- **Graph Algorithms (PYT-063):** Dijkstra and Prim use heaps extensively.

## Next Concepts

- **062 — Hash Tables:** Key-value storage with O(1) lookup.
- **063 — Graph Algorithms:** Heaps enable efficient shortest path algorithms.

## Summary

A heap is a complete binary tree satisfying the heap property. Python's heapq implements min-heaps using lists. Key operations: heappush (O(log n)), heappop (O(log n)), heapify (O(n)), 
largest/
smallest. Heaps are the standard implementation of priority queues and are essential for scheduling, graph algorithms, and streaming computations.

## Key Takeaways

- heapq provides min-heap operations on regular Python lists.
- Use heapq.heapify(list) to turn a list into a heap in O(n) time.
- Push: heapq.heappush(heap, item) — O(log n).
- Pop: heapq.heappop(heap) — returns and removes smallest — O(log n).
- Use negative values for max-heap behavior.
- heapq.nlargest(k, iterable) and heapq.nsmallest(k, iterable) for top/bottom K.
- Heaps are not sorted lists — only the smallest element is guaranteed to be at index 0.
