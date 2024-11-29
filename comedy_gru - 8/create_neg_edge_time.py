import csv
import random
from tqdm import tqdm
import numpy as np
import torch


def negative_sample_time(old_edges,all_edge):
    all_edge=np.asarray(all_edge)[:,:2].tolist()
    set_node=set()
    set_time=set()
    for old_edge in old_edges:
        set_node.add(int(old_edge[0]))
        set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_node=list(set_node)
    list_time=list(set_time)

    list_edge_new=[]
    list_edge=[]
    print('negative_creation')
    for item in tqdm(old_edges):
        index = 0
        new_edge = item[:2]
        old_time = item[2]
        list_edge.append(item[:2])
        while index < 1:
            new_time = random.uniform(min(list_time), int(max(list_time) * 1))
            new_edge_ = new_edge + [int(new_time)]
            if new_edge_ not in old_edges:
                list_edge_new.append([new_edge_[0],new_edge_[1],float(new_edge_[2])])
                index += 1
    return list_edge_new,list_edge

def negative_sample_test_time(old_edges,all_edge,list_edge):
    list_edge_new=[]
    set_node=set()
    set_time=set()
    for old_edge in old_edges:
        set_node.add(int(old_edge[0]))
        set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_node=list(set_node)
    list_time=list(set_time)


    print('negative_creation_test')
    for item in tqdm(old_edges):
        index = 0
        new_edge = item[:2]
        old_time = item[2]
        list_edge.append(new_edge)

        while index < 1:
            new_time = random.uniform(min(list_time),int(max(list_time)*1))

            new_edge_ =  new_edge + [int(new_time)]
            if new_edge_ not in old_edges:
                list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2])])
                index += 1

    return list_edge_new


