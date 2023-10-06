import math

import networkx as nx
import numpy as np
import torch
from networkx import single_source_dijkstra_path_length, single_source_dijkstra_path


def gravity_centrality(G,tar_node,alpha=0.5):
    tar_node=int(tar_node)
    GC_NODE=0
    if tar_node in G.nodes:
        length = single_source_dijkstra_path_length(G,tar_node)#distance
        tar_degree=G.degree()[tar_node]#degree
        for neigh in list(G.nodes()):
            if neigh in length and 0<length[neigh] and tar_node!=neigh:
                neigh_degree = G.degree()[neigh]
                GC_NODE+=(tar_degree*neigh_degree)/pow(length[neigh],alpha)
    return GC_NODE
