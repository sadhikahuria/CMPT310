"""
Uniform Cost Search
- uninformed
- uses lowest cost to reach the node
- nodes are expanded according to the minimum cost

Implementation
- insert the root node in the priority queue
- pop the node with the lowest cost
- FIFO queue

"""
import heapq

class graph:
    def init(self):
        self.graph = {}
    def add_edge(self, start, end, cost):
        if start not in self.graph:
            self.graph[start] = []
        self.graph[start].append((end, cost))

