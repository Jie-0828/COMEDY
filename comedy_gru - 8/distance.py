import sys
import heapq

import networkx as nx
from scipy.sparse import csr_matrix
from collections import deque

# from get_gc import *


class BFS:
    def __init__(self, vertices):
        self.V = vertices

    def bfs_shortest_path(self,start,graph):
        start=int(start)

        distances = [float('inf') ] * self.V
        distances[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()

            neighs_index =get_in_neighbors(node,graph)+get_out_neighbors(node,graph)

            for neighbor in set(neighs_index):

                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

def get_in_neighbors(node,graph):

    in_neighbors = []
    for edge_ in graph.in_edges(node):
        in_neighbors.append(edge_[0])
    return in_neighbors

def get_out_neighbors(node,graph):

    in_neighbors = []
    for edge_ in graph.out_edges(node):
        in_neighbors.append(edge_[0])
    return in_neighbors

