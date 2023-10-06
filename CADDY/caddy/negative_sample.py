import csv
import random
from tqdm import tqdm
import numpy as np
import torch


def negative_sample(old_edges,all_edge):
    all_edge=np.asarray(all_edge)[:,:2].tolist()
    list_edge=[]
    set_node=set()
    set_time=set()
    for old_edge in old_edges:
        set_node.add(int(old_edge[0]))
        set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_node=list(set_node)
    list_time=list(set_time)

    # Randomly sample positive edges and generate negative edges
    print('negative_creation')
    for item in tqdm(old_edges):
        index = 0
        while index < 1:
            new_edge=item[:2]
            new_edge[random.randrange(2)] = list_node[random.randrange(len(set_node))]
            new_time=list_time[random.randrange(len(set_time))]
            if new_edge not in all_edge and new_edge !=item[:2]:
                new_edge_ = new_edge + [new_time]
                list_edge.append(new_edge_)
                index += 1

    #Positive and negative sample synthesis training set
    random.shuffle(list_edge)
    list_edge_new=old_edges+list_edge
    dict_edge={}
    labels=[]
    edge_order=[]
    for i in range(len(list_edge_new)):
        if i <len(old_edges):
            label=0
        else:
            label=1
        timestampe=list_edge_new[i][2]
        user_id=list_edge_new[i][0]
        item_id=list_edge_new[i][1]
        if timestampe not in dict_edge:
            dict_edge[timestampe]=[[user_id,item_id,timestampe,label]]
        else:
            dict_edge[timestampe].append([user_id, item_id, timestampe,label])

    for key in sorted(dict_edge.keys()):
        random.shuffle(dict_edge[key])
        for i in dict_edge[key]:
            edge_order.append(i[:3])
            labels.append(i[3])

    return edge_order,labels,set_node

def negative_sample_test(old_edges,all_edge,test_radio):
    list_edge=[]
    set_node=set()
    set_time=set()
    for old_edge in old_edges:
        set_node.add(int(old_edge[0]))
        set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_node=list(set_node)
    list_time=list(set_time)

    # Randomly sample positive edges and generate negative edges
    for item in old_edges:
        index = 0
        while index < 1:
            new_edge=item[:2]
            new_edge[random.randrange(2)] = list_node[random.randrange(len(set_node))]
            new_time=list_time[random.randrange(len(set_time))]
            new_edge_ = new_edge+[new_time]
            if new_edge_ not in all_edge and new_edge_ !=item:
                list_edge.append(new_edge_)
                index += 1

    # The test set selects only negative edges with a certain proportion of exceptions
    random.shuffle(list_edge)
    list_edge_new = old_edges + list_edge[:int(len(list_edge)*test_radio)]
    dict_edge = {}
    labels = []
    edge_order = []
    for i in range(len(list_edge_new)):
        if i < len(old_edges):
            label = 0
        else:
            label = 1
        timestampe = list_edge_new[i][2]
        user_id = list_edge_new[i][0]
        item_id = list_edge_new[i][1]
        if timestampe not in dict_edge:
            dict_edge[timestampe] = [[user_id, item_id, timestampe, label]]
        else:
            dict_edge[timestampe].append([user_id, item_id, timestampe, label])

    for key in sorted(dict_edge.keys()):
        random.shuffle(dict_edge[key])
        for i in dict_edge[key]:
            edge_order.append(i[:3])
            labels.append(i[3])

    return edge_order, labels,set_node
