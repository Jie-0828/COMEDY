import sys
import heapq

import networkx as nx
from scipy.sparse import csr_matrix
from collections import deque

from get_gc import get_in_neighbors, get_out_neighbors


class BFS:
    def __init__(self, vertices):
        self.V = vertices

    def bfs_shortest_path(self,start,graph):
        start=int(start)
        # Initializes the distance array with the distance of
        # the starting node being 0 and the distance of the other nodes being infinity
        distances = [float('inf') ] * self.V
        distances[start] = 0

        queue = deque([start])

        while queue:
            node = queue.popleft()

            # Traverse the neighbor nodes of the current node
            neighs_index =get_in_neighbors(node,graph)+get_out_neighbors(node,graph)

            for neighbor in set(neighs_index):
                # If the distance of the neighbor node is greater than
                # the distance of the current node plus 1,
                # the distance of the neighbor node is updated
                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        return distances

