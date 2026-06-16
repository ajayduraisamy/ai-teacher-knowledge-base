# Concept: Linked Lists

## Concept ID

PYT-058

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Define a Node class for linked list elements
- Implement singly and doubly linked lists
- Perform insert, delete, search, and reverse operations
- Understand the tradeoffs between linked lists and Python lists
- Use sentinel nodes to simplify edge case handling
- Analyze the time complexity of linked list operations

## Prerequisites

- Understanding of classes and objects in Python
- Basic knowledge of references and memory
- Familiarity with Big O notation (PYT-056)

## Definition

A linked list is a linear data structure where elements (nodes) are stored independently and connected by references (pointers). Each node contains data and a reference to the next node (and optionally the previous node for doubly linked lists). Unlike Python lists (dynamic arrays), linked lists do not store elements in contiguous memory.

## Intuition

Imagine a treasure hunt where each clue tells you where to find the next clue. You cannot jump to clue #5 — you must follow the chain from clue #1 through #2, #3, #4 to reach it. That is a linked list. Adding a new clue between two existing ones is easy: just change two references. But finding a specific clue requires walking through the whole chain.

## Why This Concept Matters

While Python's built-in list is often the better choice, linked lists teach fundamental concepts about pointers, dynamic memory, and data structure design. They are the foundation for more complex structures like trees, graphs, and hash table buckets. Understanding linked lists is also essential for low-level programming and system design interviews.

## Real World Examples

1. **Music Player Playlist:** A doubly linked list for next/previous track navigation.
2. **Image Viewer:** Navigating through photos with next/previous.
3. **Browser History:** Forward/backward navigation via doubly linked list.
4. **Blockchain:** Each block references the previous block by hash.
5. **Memory Allocation:** Free lists track available memory blocks in operating systems.

## AI/ML Relevance

- **LRU Cache:** Linked list + hash map combo for cache eviction policies.
- **Graph Adjacency Lists:** Linked lists store neighbors of each node.
- **Memory Pool Management:** Free lists in custom GPU memory allocators.
- **Sequence Models:** Attention layers sometimes use learned "pointers" to input elements.
- **Token Bucket Algorithms:** Linked lists manage rate-limited request queues.

## Code Examples

### Example 1: Singly Linked List Node

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        node = Node(data)
        if not self.head:
            self.head = node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

ll = SinglyLinkedList()
for x in [1, 2, 3]:
    ll.append(x)
print(ll.display())
# Output: [1, 2, 3]
```

### Example 2: Insert at Position

```python
def insert(self, data, position):
    node = Node(data)
    if position == 0:
        node.next = self.head
        self.head = node
        return
    current = self.head
    for _ in range(position - 1):
        if not current:
            raise IndexError("Position out of range")
        current = current.next
    node.next = current.next
    current.next = node

# Assuming the SinglyLinkedList from above with insert method added
ll2 = SinglyLinkedList()
for x in [1, 2, 4]:
    ll2.append(x)
ll2.insert(3, 2)
print(ll2.display())
# Output: [1, 2, 3, 4]
```

### Example 3: Delete by Value

```python
def delete(self, value):
    if not self.head:
        return False
    if self.head.data == value:
        self.head = self.head.next
        return True
    current = self.head
    while current.next:
        if current.next.data == value:
            current.next = current.next.next
            return True
        current = current.next
    return False

# Using delete with example list
ll3 = SinglyLinkedList()
for x in [1, 2, 3, 4, 5]:
    ll3.append(x)
ll3.delete(3)
print(ll3.display())
ll3.delete(1)
print(ll3.display())
ll3.delete(5)
print(ll3.display())
# Output:
# [1, 2, 4, 5]
# [2, 4, 5]
# [2, 4]
```

### Example 4: Reverse a Linked List

```python
def reverse(self):
    prev = None
    current = self.head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    self.head = prev

ll4 = SinglyLinkedList()
for x in [1, 2, 3, 4]:
    ll4.append(x)
ll4.reverse()
print(ll4.display())
# Output: [4, 3, 2, 1]
```

### Example 5: Doubly Linked List

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, data):
        node = DNode(data)
        if not self.head:
            self.head = node
            self.tail = node
            return
        self.tail.next = node
        node.prev = self.tail
        self.tail = node

    def display_forward(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

    def display_backward(self):
        elements = []
        current = self.tail
        while current:
            elements.append(current.data)
            current = current.prev
        return elements

dll = DoublyLinkedList()
for x in [1, 2, 3]:
    dll.append(x)
print(dll.display_forward())
print(dll.display_backward())
# Output:
# [1, 2, 3]
# [3, 2, 1]
```

### Example 6: Sentinel Node (Dummy Head)

```python
class SentinelList:
    def __init__(self):
        self.sentinel = Node(None)
        self.tail = self.sentinel

    def append(self, data):
        node = Node(data)
        self.tail.next = node
        self.tail = node

    def display(self):
        elements = []
        current = self.sentinel.next
        while current:
            elements.append(current.data)
            current = current.next
        return elements

sl = SentinelList()
for x in [10, 20, 30]:
    sl.append(x)
print(sl.display())
# Output: [10, 20, 30]
```

### Example 7: Linked List vs Python List Performance

```python
import time

# Linked list append
ll = SinglyLinkedList()
start = time.time()
for i in range(100000):
    ll.append(i)
print(f"Linked list append: {time.time() - start:.3f}s")

# Python list append
py_list = []
start = time.time()
for i in range(100000):
    py_list.append(i)
print(f"Python list append: {time.time() - start:.3f}s")

# Python list insert at beginning (O(n))
start = time.time()
for i in range(1000):
    py_list.insert(0, i)
print(f"List insert at 0 (1000x): {time.time() - start:.3f}s")
# Output (varies):
# Linked list append: 0.045s
# Python list append: 0.008s
# List insert at 0 (1000x): 0.025s
```

## Common Mistakes

1. **Losing the Reference:** Overwriting `current.next` before saving it causes list breakage.
2. **Off-by-One Errors:** Incorrect loop bounds when inserting at a specific position.
3. **Not Updating `tail` in Doubly Linked Lists:** Appending without updating tail breaks backward traversal.
4. **Memory Leaks:** Not removing references to deleted nodes (though Python's GC handles this).
5. **Confusing Index vs Value:** Searching by index vs value requires different logic.
6. **Not Handling Empty List:** Operations on empty lists cause `AttributeError` on `None`.
7. **Infinite Loops:** Creating a cycle in the list (A → B → A) causes infinite traversal.

## Interview Questions

### Beginner

1. What is a linked list? How does it differ from an array?
2. How do you traverse a linked list?
3. Write a function to find the length of a linked list.
4. How do you insert a node at the beginning of a linked list?
5. What is the time complexity of searching for an element in a singly linked list?

### Intermediate

1. How do you reverse a singly linked list in place?
2. What is a doubly linked list? What are its advantages?
3. How would you detect a cycle in a linked list? (Floyd's algorithm)
4. What is the tradeoff between arrays and linked lists?
5. What is a sentinel node and why use one?

### Advanced

1. Implement a function to merge two sorted linked lists.
2. How would you implement an LRU cache using a doubly linked list and a hash map?
3. Design a lock-free linked list for concurrent access.

## Practice Problems

### Easy

1. **Find Middle:** Find the middle element of a singly linked list in one pass.
2. **Nth from End:** Find the nth node from the end of a linked list.
3. **Remove Duplicates:** Remove duplicate values from an unsorted linked list.
4. **Count Occurrences:** Count occurrences of a given value in a linked list.
5. **Max Element:** Find the maximum value in a linked list.

### Medium

1. **Detect Cycle:** Detect if a linked list has a cycle using Floyd's tortoise and hare.
2. **Intersection Point:** Find the node where two linked lists intersect.
3. **Palindrome Check:** Check if a linked list is a palindrome.
4. **Rotate List:** Rotate a linked list by k positions to the right.
5. **Partition List:** Partition a linked list around a value x.

### Hard

1. **Reverse in Groups:** Reverse a linked list in groups of size k.
2. **LRU Cache:** Implement an LRU cache with O(1) get and put using a doubly linked list + dict.
3. **Flatten Multi-Level List:** Flatten a linked list where nodes can have child lists.

## Solutions

### Solution to Easy 1: Find Middle

```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow.data if slow else None

ll = SinglyLinkedList()
for x in [1, 2, 3, 4, 5]:
    ll.append(x)
print(find_middle(ll.head))
# Output: 3
```

### Solution to Medium 1: Detect Cycle

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False

# Create a list with a cycle
ll5 = SinglyLinkedList()
for x in [1, 2, 3, 4, 5]:
    ll5.append(x)
# Create cycle: 5 -> 3
ll5.tail = ll5.head
while ll5.tail.next:
    ll5.tail = ll5.tail.next
ll5.tail.next = ll5.head.next.next
print(has_cycle(ll5.head))
# Output: True
```

### Solution to Hard 1: Reverse in Groups of K

```python
def reverse_k_group(head, k):
    count = 0
    current = head
    while current and count < k:
        current = current.next
        count += 1
    if count < k:
        return head

    prev = None
    current = head
    for _ in range(k):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    head.next = reverse_k_group(current, k)
    return prev

ll6 = SinglyLinkedList()
for x in [1, 2, 3, 4, 5, 6]:
    ll6.append(x)
ll6.head = reverse_k_group(ll6.head, 2)
print(ll6.display())
# Output: [2, 1, 4, 3, 6, 5]
```

## Related Concepts

- **Big O Notation (PYT-056):** Complexity analysis for linked list operations.
- **Recursion (PYT-057):** Recursive approaches to linked list problems.
- **Stacks and Queues (PYT-059):** Implemented using linked lists.
- **Hash Tables (PYT-062):** Chaining collision resolution uses linked lists.

## Next Concepts

- **059 — Stacks and Queues:** Linear structures built on linked lists or arrays.
- **060 — Trees:** Non-linear data structures using linked nodes.

## Summary

A linked list stores elements in nodes connected by pointers. Singly linked lists have a single `next` pointer per node; doubly linked lists also have a `prev` pointer. Insertion and deletion at known positions are O(1), but search and indexed access are O(n). Sentinels simplify edge case handling. Linked lists are the foundation for stacks, queues, and graph adjacency lists.

## Key Takeaways

- Linked lists provide O(1) insertion/deletion at ends, O(n) indexed access.
- Doubly linked lists support O(1) deletion from both ends.
- Always handle edge cases: empty list, single node, head/tail operations.
- Use Floyd's algorithm for cycle detection.
- Sentinel (dummy) nodes eliminate special cases for head/tail operations.
- Python's built-in list is usually more practical — use linked lists only when you need constant-time insert/delete at both ends or the structure is inherently linked (graph adjacencies).
