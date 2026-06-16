# Concept: Trees

## Concept ID

PYT-060

## Difficulty

Intermediate

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Define tree terminology: root, node, leaf, parent, child, subtree
- Implement a binary tree and a binary search tree (BST)
- Perform tree traversals: inorder, preorder, postorder, level-order
- Implement BST insert, search, and delete operations
- Understand tree depth, height, and balancing
- Compare balanced vs unbalanced trees

## Prerequisites

- Understanding of recursion (PYT-057)
- Familiarity with linked lists (PYT-058)
- Basic knowledge of stacks and queues (PYT-059)

## Definition

A tree is a hierarchical, non-linear data structure consisting of nodes connected by edges. Each tree has a root node, and every node (except the root) has exactly one parent. A binary tree restricts each node to at most two children (left and right). A binary search tree (BST) maintains the ordering property: all left descendants are less than the node, and all right descendants are greater.

## Intuition

Think of a family tree or an organizational chart. The CEO is at the top (root). Direct reports branch out beneath, and each of them has their own team. To find someone, you start at the top and follow the chain of command. A BST is like a well-organized file cabinet: everything in the left drawer comes before everything in the right drawer, so finding a file is fast.

## Why This Concept Matters

Trees are everywhere in computing. File systems are trees. Database indexes use B-trees. HTML/XML documents are tree structures (DOM). Compilers parse code into abstract syntax trees. Decision trees power many ML models. Understanding trees is essential for building efficient search, sorting, and hierarchical storage systems.

## Real World Examples

1. **File System:** Directories and files form a tree.
2. **DOM:** Web pages are structured as the Document Object Model tree.
3. **Compilers:** Source code is parsed into an Abstract Syntax Tree.
4. **Routing:** Network routing tables use tree structures.
5. **Merkle Trees:** Git and blockchain use hash trees for integrity.

## AI/ML Relevance

- **Decision Trees:** Recursive partitioning of feature space.
- **Random Forests:** Ensemble of decision trees.
- **Gradient Boosting:** XGBoost, LightGBM use tree ensembles.
- **Tree-Based Feature Importance:** Measuring split frequency.
- **AST Neural Networks:** Learning over code ASTs.

## Code Examples

### Example 1: Binary Tree Node

`python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
`

### Example 2: Tree Traversals

`python
def inorder(node):
    if node:
        inorder(node.left)
        print(node.value, end=" ")
        inorder(node.right)

def preorder(node):
    if node:
        print(node.value, end=" ")
        preorder(node.left)
        preorder(node.right)

def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        print(node.value, end=" ")

print("Inorder:", end=" ")
inorder(root)
print()
print("Preorder:", end=" ")
preorder(root)
print()
print("Postorder:", end=" ")
postorder(root)
print()
# Output:
# Inorder: 4 2 5 1 3
# Preorder: 1 2 4 5 3
# Postorder: 4 5 2 3 1
`

### Example 3: Level-Order Traversal

`python
from collections import deque

def level_order(root):
    if not root:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.value, end=" ")
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

print("Level-order:", end=" ")
level_order(root)
print()
# Output: Level-order: 1 2 3 4 5
`

### Example 4: BST Insert and Search

`python
class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
            return
        self._insert(self.root, value)

    def _insert(self, node, value):
        if value < node.value:
            if node.left:
                self._insert(node.left, value)
            else:
                node.left = TreeNode(value)
        else:
            if node.right:
                self._insert(node.right, value)
            else:
                node.right = TreeNode(value)

    def search(self, value):
        return self._search(self.root, value)

    def _search(self, node, value):
        if not node:
            return False
        if node.value == value:
            return True
        if value < node.value:
            return self._search(node.left, value)
        return self._search(node.right, value)

    def inorder(self):
        self._inorder(self.root)
        print()

    def _inorder(self, node):
        if node:
            self._inorder(node.left)
            print(node.value, end=" ")
            self._inorder(node.right)

bst = BST()
for v in [5, 3, 7, 2, 4, 6, 8]:
    bst.insert(v)
bst.inorder()
print(bst.search(4))
print(bst.search(9))
# Output:
# 2 3 4 5 6 7 8
# True
# False
`

### Example 5: BST Delete

`python
def delete(self, value):
    self.root = self._delete(self.root, value)

def _delete(self, node, value):
    if not node:
        return None
    if value < node.value:
        node.left = self._delete(node.left, value)
    elif value > node.value:
        node.right = self._delete(node.right, value)
    else:
        if not node.left:
            return node.right
        if not node.right:
            return node.left
        min_node = self._find_min(node.right)
        node.value = min_node.value
        node.right = self._delete(node.right, min_node.value)
    return node

def _find_min(self, node):
    while node.left:
        node = node.left
    return node

bst2 = BST()
for v in [5, 3, 7, 2, 4, 8]:
    bst2.insert(v)
bst2.delete(3)
bst2.inorder()
# Output: 2 4 5 7 8
`

### Example 6: Depth and Height

`python
def depth(node, value, d=0):
    if not node:
        return -1
    if node.value == value:
        return d
    if value < node.value:
        return depth(node.left, value, d + 1)
    return depth(node.right, value, d + 1)

def height(node):
    if not node:
        return -1
    return 1 + max(height(node.left), height(node.right))

bst3 = BST()
for v in [10, 5, 15, 3, 7, 20]:
    bst3.insert(v)
print(f"Depth of 7: {depth(bst3.root, 7)}")
print(f"Depth of 3: {depth(bst3.root, 3)}")
print(f"Height of tree: {height(bst3.root)}")
# Output:
# Depth of 7: 2
# Depth of 3: 2
# Height of tree: 2
`

### Example 7: Balanced vs Unbalanced BST

`python
import time

def build_bst(values):
    bst = BST()
    for v in values:
        bst.insert(v)
    return bst

balanced = build_bst([5, 3, 7, 2, 4, 6, 8])
unbalanced = build_bst([1, 2, 3, 4, 5, 6, 7, 8])

print(f"Balanced height: {height(balanced.root)}")
print(f"Unbalanced height: {height(unbalanced.root)}")

start = time.time()
balanced.search(8)
print(f"Balanced search: {time.time() - start:.6f}s")

start = time.time()
unbalanced.search(8)
print(f"Unbalanced search: {time.time() - start:.6f}s")
# Output:
# Balanced height: 2
# Unbalanced height: 7
# Balanced search: 0.000010s
# Unbalanced search: 0.000025s
`

## Common Mistakes

1. **Not Handling None:** Forgetting to check if a node is None before attribute access.
2. **Modifying Tree During Traversal:** Deleting nodes while iterating over them.
3. **Wrong Traversal Order:** Using preorder when inorder is needed.
4. **BST Property Violation:** Not maintaining left < node < right.
5. **Incorrect Delete Logic:** Not handling the two-child case properly.
6. **Stack Overflow:** Recursive traversals on very deep trees.
7. **Ignoring Self-Balancing:** Sorted input creates degenerate BST (O(n) operations).

## Interview Questions

### Beginner

1. What is a binary tree?
2. What is the difference between a tree and a linked list?
3. What are the three depth-first traversal orders?
4. What is a binary search tree?
5. How do you find the height of a tree?

### Intermediate

1. Implement a function to check if a binary tree is a valid BST.
2. How do you delete a node from a BST? Describe the three cases.
3. What is the difference between BFS and DFS traversal?
4. How would you serialize and deserialize a binary tree?
5. What is a balanced tree and why is it important?

### Advanced

1. Implement an AVL tree with rotations.
2. Find the lowest common ancestor of two nodes in a binary tree.
3. Implement a segment tree for range sum queries.

## Practice Problems

### Easy

1. **Max Depth:** Find the maximum depth of a binary tree.
2. **Count Nodes:** Count all nodes in a binary tree.
3. **Leaf Count:** Count the number of leaf nodes.
4. **Tree Sum:** Sum all values in a binary tree.
5. **Mirror Tree:** Create a mirror image of a binary tree.

### Medium

1. **Validate BST:** Check if a tree satisfies BST properties.
2. **Level Order Traversal:** Return level-by-level node values.
3. **Lowest Common Ancestor:** Find LCA of two nodes in a BST.
4. **Tree Diameter:** Find the longest path between any two nodes.
5. **Path Sum:** Determine if a root-to-leaf path sums to a target.

### Hard

1. **AVL Tree:** Implement a self-balancing AVL tree.
2. **Serialize/Deserialize:** Convert a binary tree to a string and back.
3. **Morris Traversal:** O(1) space inorder traversal.

## Solutions

### Solution to Easy 1: Max Depth

`python
def max_depth(node):
    if not node:
        return 0
    return 1 + max(max_depth(node.left), max_depth(node.right))

print(max_depth(root))
# Output: 3
`

### Solution to Medium 1: Validate BST

`python
def is_valid_bst(node, low=float("-inf"), high=float("inf")):
    if not node:
        return True
    if node.value <= low or node.value >= high:
        return False
    return (is_valid_bst(node.left, low, node.value) and
            is_valid_bst(node.right, node.value, high))

invalid = TreeNode(5)
invalid.left = TreeNode(3)
invalid.right = TreeNode(7)
invalid.left.right = TreeNode(6)

print(is_valid_bst(bst.root))
print(is_valid_bst(invalid))
# Output:
# True
# False
`

### Solution to Hard 1: AVL Tree Rotations

`python
class AVLNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def _height(self, node):
        return node.height if node else 0

    def _balance(self, node):
        return self._height(node.left) - self._height(node.right) if node else 0

    def _rotate_right(self, y):
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        return x

    def _rotate_left(self, x):
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        return y

    def insert(self, node, value):
        if not node:
            return AVLNode(value)
        if value < node.value:
            node.left = self.insert(node.left, value)
        elif value > node.value:
            node.right = self.insert(node.right, value)
        else:
            return node

        node.height = 1 + max(self._height(node.left), self._height(node.right))
        balance = self._balance(node)

        if balance > 1 and value < node.left.value:
            return self._rotate_right(node)
        if balance < -1 and value > node.right.value:
            return self._rotate_left(node)
        if balance > 1 and value > node.left.value:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1 and value < node.right.value:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

avl = AVLTree()
root = None
for v in [10, 20, 30, 40, 50, 25]:
    root = avl.insert(root, v)
print("AVL tree built with rotations")
# Output: AVL tree built with rotations
`

## Related Concepts

- **Recursion (PYT-057):** Tree traversals are naturally recursive.
- **Stacks and Queues (PYT-059):** BFS uses a queue, DFS uses a stack.
- **Heaps (PYT-061):** Complete binary tree for priority queues.
- **Graph Algorithms (PYT-063):** Trees are a subset of graphs.

## Next Concepts

- **061 — Heaps:** Tree-based structure for priority queues.
- **062 — Hash Tables:** Key-value storage with O(1) average access.

## Summary

Trees are hierarchical structures with nodes connected by parent-child edges. Binary trees restrict nodes to two children. BSTs maintain the ordering property for efficient search (O(log n) average, O(n) worst case). Tree traversals include inorder, preorder, postorder (DFS), and level-order (BFS). Self-balancing trees (AVL, Red-Black) maintain O(log n) operations.

## Key Takeaways

- Trees have a root, leaves, and levels — depth and height measure positions.
- BST property: left < node < right for all nodes in subtrees.
- Inorder traversal of BST produces sorted order.
- Recursive traversals are elegant but deep trees overflow the call stack.
- Unbalanced BSTs degenerate to O(n) — use self-balancing variants for production.
- Level-order traversal uses a queue (BFS).
- Trees are the foundation of many advanced data structures (heaps, tries, segment trees).
