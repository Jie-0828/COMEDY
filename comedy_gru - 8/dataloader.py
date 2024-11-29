import random
import numpy as np
import torch
import torch.nn.functional as F

def load_data_node(path,dataset,flag='negative'):  # file_path,devide
    set_node = set()
    set_time=set()
    dict_edge= {}
    with open(path,'r')as fd:
        if dataset in ['UCI', 'DIGG']:
            fd.readline()
        for line in fd:
            if line=='\n':
                break
            if dataset in ['UCI', 'DIGG']:
                line = line.strip('\n').split(' ')
                timestampe = float(line[3])
            elif dataset in ['email','mathoverflow','askubuntu','ia-contacts_dublin']:
                line = line.strip('\n').split(' ')
                timestampe = float(line[2])
            elif dataset in ['fb-forum']:
                line = line.strip('\n').split(',')
                timestampe = float(line[2])
            else:
                line = line.strip('\n').split(',')
                timestampe = float(line[3])
            user_id = int(line[0])
            item_id = int(line[1])
            set_node.add(user_id)
            set_node.add(item_id)
            set_time.add(timestampe)
            if flag == 'negative':
                label=1
            else:
                label=0
            if user_id != item_id:
                if (user_id, item_id, label) not in dict_edge:
                    dict_edge[(user_id, item_id, label)] = [timestampe]
                else:
                    dict_edge[(user_id, item_id, label)].append(timestampe)

        dict_edge_new={}
        for key, value in dict_edge.items():
            new_time = np.random.choice(value, size=1)
            if new_time[0] not in dict_edge_new:
                dict_edge_new[new_time[0]] = [[key[0], key[1], new_time[0],key[2]]]
            else:
                dict_edge_new[new_time[0]].append([key[0],key[1],new_time[0], key[2]])

        edge_order_new = []
        for key in sorted(dict_edge_new.keys()):
            random.shuffle(dict_edge_new[key])
            for i in dict_edge_new[key]:
                edge_order_new.append([i[0], i[1], float(i[2]),i[3]])

    return edge_order_new,list(set_time),set_node


def merge_edge(edge_order,dict_node,ori_time):
    list_labels=[]
    dict_edge={}

    for edge in edge_order:
        user_id=edge[0]
        item_id=edge[1]
        timestampe=edge[2]
        label=edge[3]

        if user_id != item_id:
            if (user_id, item_id, label) not in dict_edge:
                dict_edge[(user_id, item_id, label)] = [timestampe]
            else:
                dict_edge[(user_id, item_id, label)].append(timestampe)


    dict_edge_new={}
    for key, value in dict_edge.items():
        new_time = np.random.choice(value, size=1)
        if new_time[0] not in dict_edge_new:
            dict_edge_new[new_time[0]] = [[dict_node[key[0]], dict_node[key[1]], key[2], new_time[0]]]
        else:
            dict_edge_new[new_time[0]].append([dict_node[key[0]], dict_node[key[1]], key[2], new_time[0]])

    edge_order_new = []
    final_time = sorted(dict_edge_new.keys())[-1]
    for key in sorted(dict_edge_new.keys()):
        random.shuffle(dict_edge_new[key])
        for i in dict_edge_new[key]:
            edge_order_new.append([i[0], i[1], float(i[3] - ori_time)])
            list_labels.append(i[2])

        edge_order_new.append([dict_node[user_id],dict_node[item_id],float(timestampe-ori_time)])
        list_labels.append(label)

    return edge_order_new,list_labels

def reindex(set_node):
    dict_node={}
    np_node=np.array(sorted(list(set_node)))
    np_node=np.unique(np_node, return_index=True, axis=0)
    for i in range(len(np_node[0])):
        dict_node[np_node[0][i]]=np_node[1][i]

    return dict_node

def merge_edges(old_edges, list_edge, dict_node,ori_time):
    list_edge_new = old_edges + list_edge
    dict_edge = {}
    labels = []
    edge_order = []
    set_time = set()
    for i in range(len(list_edge_new)):
        if i < len(old_edges):
            label = 0
        else:
            label = 1
        timestampe = list_edge_new[i][2]
        set_time.add(timestampe)
        user_id = list_edge_new[i][0]
        item_id = list_edge_new[i][1]
        if timestampe not in dict_edge:
            dict_edge[timestampe] = [[dict_node[user_id], dict_node[item_id], float(timestampe-ori_time), label]]
        else:
            dict_edge[timestampe].append([dict_node[user_id], dict_node[item_id], float(timestampe-ori_time), label])

    for key in sorted(dict_edge.keys()):
        random.shuffle(dict_edge[key])
        for i in dict_edge[key]:
            edge_order.append(i[:3])
            labels.append(i[3])
    max_time = sorted(list(set_time))[-1]

    return edge_order, labels,max_time