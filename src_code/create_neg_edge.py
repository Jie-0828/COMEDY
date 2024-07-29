import csv
import random
from tqdm import tqdm
import numpy as np
import torch


def negative_sample_spatial(old_edges,all_edge,set_node):
    old_edges_=np.asarray(old_edges)[:,:2].tolist()
    set_time=set()
    for old_edge in old_edges:
        set_time.add(float(old_edge[2]))
    list_time=list(set_time)
    list_node=list(range(len(set_node)))

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
            old_edge[select_node_idx] = list_node[random.randrange(len(list_node))]
            new_edge_ = old_edge + [int(new_time)]
            if old_edge not in old_edges_:
                list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2])])
                index += 1
    print(len(old_edges),len(labels))
    return list_edge_new


def negative_sample_time(old_edges,all_edge):
    all_edge=np.asarray(all_edge)[:,:2].tolist()
    set_node=set()
    set_time=set()
    for old_edge in old_edges:
        set_node.add(int(old_edge[0]))
        set_node.add(int(old_edge[1]))
        set_time.add(float(old_edge[2]))
    list_time=list(set_time)

    list_edge_new=[]
    list_edge=[]
    print('negative_creation')
    for item in tqdm(old_edges):
        index = 0
        old_time = item[2]
        list_edge.append(item[:2])

        while index < 1:
            new_edge=list_edge[random.randrange(len(list_edge))]
            new_time = random.uniform(min(list_time), max(list_time))
            new_edge_ = new_edge + [int(new_time)]
            if new_edge_ not in old_edges:
                list_edge_new.append([new_edge_[0],new_edge_[1],float(new_edge_[2])])
                index += 1
    return list_edge_new


def negative_sample_hist(old_edges,all_edge,dict_time_node,dict_node_neigh):
    old_edges_=np.asarray(old_edges)[:,:2].tolist()
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
    labels=[]
    print('negative_creation')
    for item in tqdm(old_edges):
        index = 0
        old_edge = item[:2]
        old_time = item[2]
        dict_time_node[item[0]] = old_time
        if item[0] not in dict_node_neigh:
            dict_node_neigh[item[0]]=set()
        dict_node_neigh[item[0]].add(item[1])

        dict_time_node[item[1]] = old_time
        if item[1] not in dict_node_neigh:
            dict_node_neigh[item[1]]=set()
        dict_node_neigh[item[1]].add(item[0])


        while index < 1:
            select_node_idx = random.randrange(1)
            dict_neigh = {}
            for neigh in list(dict_node_neigh[item[select_node_idx]]):
                his_time = dict_time_node[neigh]
                if  old_time-his_time >0:
                    if his_time not in dict_neigh:
                        dict_neigh[his_time] = [neigh]
                    else:
                        dict_neigh[his_time].append(neigh)

            for time_node in sorted(dict_neigh.keys()):
                for i in dict_neigh[time_node]:
                    old_edge[select_node_idx] = i
                    new_edge_ = old_edge + [int(old_time)]

                    if old_edge not in old_edges_:
                        list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2]), 1])
                        labels.append(1)
                        index += 1
                        break
                if index == 1:
                    break

            if index < 1:
                not_select_node_idx = {0, 1} - {select_node_idx}
                not_select_node_idx=list(not_select_node_idx)[0]
                dict_neigh = {}
                for neigh in list(dict_node_neigh[item[not_select_node_idx]]):
                    his_time = dict_time_node[neigh]
                    if old_time-his_time >0:
                        if his_time not in dict_neigh:
                            dict_neigh[his_time] = [neigh]
                        else:
                            dict_neigh[his_time].append(neigh)

                for time_node in sorted(dict_neigh.keys()):
                    for i in dict_neigh[time_node]:
                        old_edge[not_select_node_idx] = i
                        new_edge_ = old_edge + [int(old_time)]
                        if old_edge not in old_edges_:
                            list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2]), 1])
                            labels.append(1)
                            index += 1
                            break
                    if index == 1:
                        break

            if index < 1:
                while index <1:
                    new_time = random.uniform(min(list_time), max(list_time))
                    old_edge[select_node_idx] = list(set_node)[random.randrange(len(set_node))]
                    new_edge_ = old_edge + [int(old_time)]
                    if old_edge not in old_edges_:
                        list_edge_new.append([new_edge_[0], new_edge_[1], float(new_edge_[2])])
                        index += 1

    print(len(old_edges),len(labels))
    return list_edge_new,dict_time_node,dict_node_neigh

def merge_edges(old_edges, list_edge):
    # random.shuffle(list_edge)
    list_edge_new = old_edges + list_edge
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
            dict_edge[timestampe] = [[user_id, item_id, float(timestampe), label]]
        else:
            dict_edge[timestampe].append([user_id, item_id, float(timestampe), label])

    for key in sorted(dict_edge.keys()):
        random.shuffle(dict_edge[key])
        for i in dict_edge[key]:
            edge_order.append(i[:3])
            labels.append(i[3])

    return edge_order, labels
