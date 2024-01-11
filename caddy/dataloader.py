import random
import numpy as np
import torch
import torch.nn.functional as F

def load_data_node(path,dataset):  # file_path,devide
    edge_order = []
    # edge_features = []
    dict_edge={}
    set_node=set()
    dict_edge_new={}
    dict_node={}
    number=0

    with open(path,'r')as fd:
        if dataset in ['UCI', 'DIGG','Reddit']:
            fd.readline()
        for line in fd:
            if line=='\n':
                break
            if dataset in ['UCI', 'DIGG']:
                line = line.strip('\n').split(' ')
                timestampe = float(line[3])
            elif dataset in ['fb-forum','ia-contacts_dublin']:
                line = line.strip('\n').split(',')
                timestampe = float(line[2])
            else:
                line = line.strip('\n').split(',')
                timestampe = float(line[3])
            user_id = int(line[0])
            item_id = int(line[1])
            set_node.add(user_id)
            set_node.add(item_id)
            if user_id!=item_id:
                if (user_id, item_id) not in dict_edge:
                    dict_edge[(user_id, item_id)] = [timestampe]
                else:
                    dict_edge[(user_id, item_id)].append(timestampe)

    np_node=np.array(sorted(list(set_node)))
    np_node=np.unique(np_node, return_index=True, axis=0)
    for i in range(len(np_node[0])):
        dict_node[np_node[0][i]]=np_node[1][i]


    for key,value in dict_edge.items():
        new_time=np.random.choice(value, size=1)
        if new_time[0] not in dict_edge_new:
            dict_edge_new[new_time[0]]=[[dict_node[key[0]],dict_node[key[1]],new_time[0]]]
        else:
            dict_edge_new[new_time[0]].append([dict_node[key[0]],dict_node[key[1]],new_time[0]])


    ori_time=sorted(dict_edge_new.keys())[0]
    final_time=sorted(dict_edge_new.keys())[-1]
    for key in sorted(dict_edge_new.keys()):
        random.shuffle(dict_edge_new[key])
        for i in dict_edge_new[key]:
            edge_order.append([i[0],i[1],float(i[2]-ori_time)])

    return edge_order,len(dict_node)
