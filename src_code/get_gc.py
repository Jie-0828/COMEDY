import copy

import numpy as np
import torch

import dataloader
from distance import get_in_neighbors
from tqdm import tqdm
import pickle



def get_GC(train_edge,train_labels,G,adj_time,dataset,threshold,bfs,divide,test_radio=0.01,flag='train'):
    dict_gc_all={}
    number=0
    pbar = tqdm(range(len(train_edge)), total =len(train_edge))
    for index in pbar:
        edge=train_edge[index]
        label=train_labels[index]
        time = edge[2]
        G.add_nodes_from(edge[:2])

        # graph_remain = G.copy()
        if label == 1 and flag=='train':
            pass
        else:
            for node in set(G.nodes()):
                time_interval = time - adj_time[node]
                if time_interval > threshold and node != edge[0] and node != edge[1]:
                    G.remove_node(node)
            G.add_edge(edge[0], edge[1])

        neighs_index_all = edge[:2]+get_in_neighbors(edge[0],G)+get_in_neighbors(edge[1],G)
        dict_gc = {}
        for j in list(set(neighs_index_all)):
            dict_gc[int(j)] = short_lengh(G, j,bfs)
        dict_gc_all[number]=dict_gc
        number+=1

        if label == 1 and flag == 'train':
            pass
        else:
            adj_time[edge[1]] = time
            adj_time[edge[0]] = time

    if flag=='train':
        with open(dataset+"_"+flag+"_"+str(threshold)+str(divide)+".pkl", "wb") as tf:
            pickle.dump(dict_gc_all, tf)
    else:
        with open(dataset+"_"+flag+'_'+str(test_radio)+"_"+str(threshold)+str(divide)+".pkl", "wb") as tf:
            pickle.dump(dict_gc_all, tf)


def short_lengh(G,tar_node,bfs):
    tar_node=int(tar_node)
    length=bfs.bfs_shortest_path(tar_node,G)
    tar_degree = G.in_degree(tar_node)+G.out_degree(tar_node)
    LGC_tar = 0

    for neigh in range(len(length)):
        if length[neigh]!=float('inf') and tar_node != neigh:
            neigh_degree = G.in_degree(neigh)+G.out_degree(neigh)
            LGC_tar+=(tar_degree*neigh_degree)/pow(length[neigh],2)


    return LGC_tar




