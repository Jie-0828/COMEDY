import csv
import random
from tqdm import tqdm
import numpy as np
import torch


def negative_sample_str(old_edges,all_edge,set_node):
    old_edges_=np.asarray(old_edges)[:,:2].tolist()
    all_edge=np.asarray(all_edge)[:,:2].tolist()
    # list_edge=[]
    # set_node=set()
    set_time=set()
    set_edge=set()
    for old_edge in old_edges:
        # set_node.add(int(old_edge[0]))
        # set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    # list_node=list(set_node)
    list_time=list(set_time)

    # 将选择出的边进行替换
    list_edge_new=[]
    labels=[]
    print('negative_creation')
    for item in tqdm(old_edges):
        index = 0
        old_edge = item[:2]
        old_time = item[2]

        while index < 1:
            select_node_idx = random.randrange(2)
            # not_select_node_idx = {0, 1} - {select_node_idx}
            new_time = random.uniform(min(list_time), max(list_time))
            old_edge[select_node_idx] = list(set_node)[random.randrange(len(set_node))]
            new_edge_ = old_edge + [int(new_time)]
            if old_edge not in old_edges_:
                list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2])])
                index += 1

    print(len(old_edges),len(labels))
    return list_edge_new

def negative_sample_test_str(old_edges,all_edge,set_node):
    old_edges_ = np.asarray(old_edges)[:, :2].tolist()
    list_edge=[]
    # set_node=set()
    set_time=set()
    for old_edge in old_edges:
        # set_node.add(int(old_edge[0]))
        # set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_node=list(set_node)
    list_time=list(set_time)
    labels=[]
    list_edge_new=[]
    # 将选择出的边进行替换
    # dict_time_node={}
    # dict_node_neigh={}
    print('negative_creation_test')
    for item in tqdm(old_edges):
        index = 0
        old_edge = item[:2]
        old_time = item[2]

        while index < 1:
            select_node_idx = random.randrange(2)
            # not_select_node_idx = {0, 1} - {select_node_idx}
            new_time = random.uniform(min(list_time), max(list_time))
            old_edge[select_node_idx] = list(set_node)[random.randrange(len(set_node))]
            new_edge_ = old_edge + [int(new_time)]
            if old_edge not in old_edges_:
                list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2])])
                index += 1

    print(len(list_edge),int(len(old_edges)*0.1),len(labels))
    return list_edge_new

