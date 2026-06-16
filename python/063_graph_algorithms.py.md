# Concept: Graph Algorithms

## Concept ID

PYT-063

## Difficulty

Advanced

## Domain

Python

## Module

Data Structures and Algorithms

## Learning Objectives

- Represent graphs using adjacency lists and adjacency matrices
- Implement BFS (breadth-first search) and DFS (depth-first search)
- Implement Dijkstra's shortest path algorithm
- Understand A* search for heuristic-guided pathfinding
- Implement topological sort for DAGs
- Analyze graph algorithm complexities

## Prerequisites

- Understanding of queues and stacks (PYT-059)
- Familiarity with trees (PYT-060)
- Knowledge of heaps (PYT-061) for Dijkstra
- Basic recursion (PYT-057)

## Definition

A graph is a collection of vertices (nodes) and edges connecting them. Graphs can be directed or undirected, weighted or unweighted. Graph algorithms solve problems like finding paths, connectivity, shortest routes, and ordering. Common representations include adjacency lists (dict of neighbors) and adjacency matrices (2D array).

## Intuition

Think of a map with cities (vertices) connected by roads (edges). BFS explores like a wave spreading from a starting city, visiting all cities 1 step away, then 2 steps, and so on. DFS explores like a traveler going down one road as far as possible before backtracking. Dijkstra is like GPS navigation — it finds the shortest route considering road lengths (weights).

## Why This Concept Matters

Graphs model relationships and networks, which are everywhere: social networks, transportation, the internet, protein interactions, and knowledge bases. Graph algorithms power Google Maps, Facebook friend suggestions, page ranking, and dependency resolution. Understanding graph algorithms is essential for any software engineer working with connected data.

## Real World Examples

1. **GPS Navigation:** Dijkstra's algorithm finds the shortest driving route.
2. **Social Networks:** BFS finds degrees of separation between users.
3. **Web Crawling:** DFS crawls linked pages recursively.
4. **Package Management:** Topological sort resolves dependency order.
5. **Network Routing:** OSPF routing protocol uses Dijkstra-like algorithms.

## AI/ML Relevance

- **Knowledge Graphs:** Google's Knowledge Graph is a massive graph queried with graph algorithms.
- **Graph Neural Networks (GNNs):** Message passing over graph structures.
- **Dependency Parsing:** Syntactic dependency trees for NLP.
- **Recommendation Systems:** Collaborative filtering as a bipartite graph.
- **Reinforcement Learning:** State-space search, Monte Carlo tree search.

## Code Examples

### Example 1: Graph Representation

`python
# Adjacency List
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"]
}

# Adjacency Matrix
nodes = ["A", "B", "C", "D", "E", "F"]
n = len(nodes)
matrix = [[0] * n for _ in range(n)]
edges = [("A","B"), ("A","C"), ("B","D"), ("B","E"), ("C","F"), ("E","F")]
for u, v in edges:
    i, j = nodes.index(u), nodes.index(v)
    matrix[i][j] = 1
    matrix[j][i] = 1

print("Adjacency list:")
for node, neighbors in graph.items():
    print(f"  {node}: {neighbors}")
# Output:
# Adjacency list:
#   A: ['B', 'C']
#   B: ['A', 'D', 'E']
#   C: ['A', 'F']
#   D: ['B']
#   E: ['B', 'F']
#   F: ['C', 'E']
`

### Example 2: BFS (Breadth-First Search)

`python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

print("BFS from A:", bfs(graph, "A"))
# Output: BFS from A: ['A', 'B', 'C', 'D', 'E', 'F']
`

### Example 3: DFS (Depth-First Search)

`python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    order = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order

def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    order = [node]
    for neighbor in graph[node]:
        if neighbor not in visited:
            order.extend(dfs_recursive(graph, neighbor, visited))
    return order

print("DFS iterative from A:", dfs_iterative(graph, "A"))
print("DFS recursive from A:", dfs_recursive(graph, "A"))
# Output:
# DFS iterative from A: ['A', 'B', 'D', 'E', 'F', 'C']
# DFS recursive from A: ['A', 'B', 'D', 'E', 'F', 'C']
`

### Example 4: Dijkstra's Shortest Path

`python
import heapq

def dijkstra(graph, start):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

weighted_graph = {
    "A": [("B", 4), ("C", 2)],
    "B": [("A", 4), ("C", 1), ("D", 5)],
    "C": [("A", 2), ("B", 1), ("D", 8), ("E", 10)],
    "D": [("B", 5), ("C", 8), ("E", 2)],
    "E": [("C", 10), ("D", 2)]
}

print(dijkstra(weighted_graph, "A"))
# Output: {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
`

### Example 5: Topological Sort

`python
from collections import defaultdict, deque

def topological_sort(vertices, edges):
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for v in vertices:
        in_degree[v] = 0

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([v for v in vertices if in_degree[v] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(vertices):
        raise ValueError("Graph has a cycle")
    return result

vertices = ["A", "B", "C", "D", "E", "F"]
edges = [("A", "D"), ("F", "B"), ("B", "D"), ("F", "A"), ("D", "C"), ("E", "C")]
print(topological_sort(vertices, edges))
# Output: ['E', 'F', 'A', 'B', 'D', 'C'] (order may vary)
`

### Example 6: A* Search

`python
import heapq

def a_star(graph, heuristics, start, goal):
    open_set = [(heuristics[start], 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for neighbor, weight in graph[current]:
            tentative_g = current_cost + weight
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristics[neighbor]
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None

heuristics = {"A": 6, "B": 4, "C": 4, "D": 2, "E": 0, "S": 5}
a_star_graph = {
    "S": [("A", 2), ("B", 3)],
    "A": [("S", 2), ("C", 3)],
    "B": [("S", 3), ("D", 2)],
    "C": [("A", 3), ("E", 3)],
    "D": [("B", 2), ("E", 1)],
    "E": [("C", 3), ("D", 1)]
}

print(a_star(a_star_graph, heuristics, "S", "E"))
# Output: ['S', 'B', 'D', 'E']
`

### Example 7: Detect Cycle in Directed Graph

`python
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False

cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
acyclic = {"A": ["B"], "B": ["C"], "C": []}
print(has_cycle(cyclic))
print(has_cycle(acyclic))
# Output:
# True
# False
`

## Common Mistakes

1. **Not Tracking Visited Nodes:** BFS/DFS without visited set causes infinite loops in cyclic graphs.
2. **Using DFS for Shortest Path:** DFS does not find the shortest path in unweighted graphs — BFS does.
3. **Incorrect Dijkstra with Negative Weights:** Dijkstra fails with negative edge weights; use Bellman-Ford.
4. **Forgetting Graph May Be Disconnected:** Only exploring from one start node misses unreachable components.
5. **Inefficient Adjacency Matrix for Sparse Graphs:** O(V^2) memory is wasteful for sparse graphs — use adjacency lists.
6. **Not Handling Self-Loops:** Edges from a node to itself can cause infinite loops in some algorithms.
7. **Assuming Undirected:** Always clarify whether the graph is directed or undirected before choosing algorithms.

## Interview Questions

### Beginner

1. What is a graph? What is the difference between directed and undirected?
2. What are adjacency lists and adjacency matrices?
3. Explain BFS and when to use it.
4. Explain DFS and when to use it.
5. What is a topological sort?

### Intermediate

1. How does Dijkstra's algorithm work? What is its time complexity?
2. How do you detect a cycle in a directed graph?
3. What is the difference between BFS and DFS in terms of space complexity?
4. How would you find all connected components in an undirected graph?
5. What is the difference between Dijkstra and A*?

### Advanced

1. Implement Bellman-Ford for graphs with negative weights.
2. Explain Floyd-Warshall and when you would use it over Dijkstra.
3. Design a shortest path algorithm for a graph with millions of nodes (discuss optimizations).

## Practice Problems

### Easy

1. **Graph from Edges:** Build an adjacency list from a list of edges.
2. **BFS Traversal:** Return BFS order from a given start node.
3. **DFS Traversal:** Return DFS order from a given start node.
4. **Connected Components:** Count connected components in an undirected graph.
5. **Has Path:** Check if there is a path between two nodes.

### Medium

1. **Shortest Path (BFS):** Find shortest path in an unweighted graph.
2. **Detect Cycle:** Detect a cycle in a directed graph.
3. **Topological Sort:** Perform topological sort on a DAG.
4. **Clone Graph:** Deep copy a graph with connected nodes.
5. **Word Ladder:** Find shortest transformation sequence between words.

### Hard

1. **Dijkstra's Algorithm:** Implement from scratch with a heap.
2. **A* Search:** Implement A* with Manhattan distance heuristic.
3. **Course Schedule II:** Find order to take courses given prerequisites (topological sort with cycle detection).

## Solutions

### Solution to Easy 1: Build Adjacency List

`python
def build_graph(edges, directed=False):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph

edges = [(1, 2), (2, 3), (3, 4), (4, 2)]
print(build_graph(edges, directed=True))
# Output: {1: [2], 2: [3], 3: [4], 4: [2]}
`

### Solution to Medium 1: Shortest Path BFS

`python
from collections import deque

def shortest_path(graph, start, goal):
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

print(shortest_path(graph, "A", "F"))
# Output: ['A', 'C', 'F']
`

### Solution to Hard 1: Dijkstra's Algorithm

`python
import heapq

def dijkstra_full(graph, start, goal):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    came_from = {}

    while pq:
        current_dist, node = heapq.heappop(pq)
        if node == goal:
            path = []
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return list(reversed(path))

        if current_dist > distances[node]:
            continue

        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                came_from[neighbor] = node
                heapq.heappush(pq, (distance, neighbor))

    return None

print(dijkstra_full(weighted_graph, "A", "E"))
# Output: ['A', 'C', 'B', 'D', 'E'] or ['A', 'B', 'D', 'E']
`

## Related Concepts

- **Trees (PYT-060):** Trees are acyclic connected graphs.
- **Heaps (PYT-061):** Used in Dijkstra and A* for priority queues.
- **Stacks and Queues (PYT-059):** BFS uses a queue, DFS uses a stack.
- **Recursion (PYT-057):** DFS is naturally recursive.

## Next Concepts

- **064 — Sorting:** Comparing sorting algorithms with graph-based analysis.
- **065 — Searching:** Graph search (BFS/DFS) vs list search algorithms.

## Summary

Graphs represent relationships between entities. Key representations are adjacency lists (efficient for sparse graphs) and adjacency matrices (efficient for dense graphs). BFS finds shortest paths in unweighted graphs, DFS explores deeply for topological sorting and cycle detection. Dijkstra finds shortest paths in weighted graphs with non-negative weights. A* adds heuristics for faster goal-directed search.

## Key Takeaways

- Adjacency lists are the most common Python graph representation.
- BFS uses a queue, DFS uses a stack (or recursion).
- BFS finds shortest paths in unweighted graphs; Dijkstra for weighted.
- Dijkstra requires non-negative edge weights; use Bellman-Ford otherwise.
- Topological sort only works on DAGs (directed acyclic graphs).
- A* is Dijkstra with a heuristic for goal-directed search.
- Always track visited nodes to avoid infinite loops in cyclic graphs.
