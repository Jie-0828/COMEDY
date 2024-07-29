import random
import numpy as np
import torch
import torch.nn.functional as F

def load_data_node(path,dataset):  # file_path,devide
    set_node = set()
    set_time=set()
    dict_edge= {}
    with open(path,'r')as fd:
        if dataset in ['uci', 'digg']:
            fd.readline()
        for line in fd:
            if line=='\n':
                break
            if dataset in ['uci', 'digg']:
                line = line.strip('\n').split(' ')
                timestampe = float(line[3])
            elif dataset in ['email','mathoverflow','askubuntu','ia-contacts_dublin']:
                line = line.strip('\n').split(' ')
                timestampe = float(line[2])
            else:
                line = line.strip('\n').split(',')
                timestampe = float(line[3])
            user_id = int(line[0])
            item_id = int(line[1])
            set_node.add(user_id)
            set_node.add(item_id)
            label=0
            # set_time.add(timestampe)
            if user_id != item_id:  # remove self-loop
                if (user_id, item_id, label) not in dict_edge:
                    dict_edge[(user_id, item_id, label)] = [timestampe]
                else:
                    dict_edge[(user_id, item_id, label)].append(timestampe)

        dict_node=reindex(set_node)


        # Remove duplicate edges
        dict_edge_new={}
        for key, value in dict_edge.items():
            new_time = np.random.choice(value, size=1)
            set_time.add(new_time[0])
            if new_time[0] not in dict_edge_new:
                dict_edge_new[new_time[0]] = [[key[0], key[1], new_time[0],key[2]]]
            else:
                dict_edge_new[new_time[0]].append([key[0],key[1],new_time[0], key[2]])

        edge_order_new = []
        ori_time=sorted(list(set_time))[0]
        for key in sorted(dict_edge_new.keys()):
            random.shuffle(dict_edge_new[key])
            for i in dict_edge_new[key]:
                edge_order_new.append([dict_node[i[0]], dict_node[i[1]], float(i[2]-ori_time),i[3]])

    return edge_order_new,list(set_time),set_node

def reindex(set_node):
    dict_node={}
    np_node=np.array(sorted(list(set_node)))
    np_node=np.unique(np_node, return_index=True, axis=0)
    for i in range(len(np_node[0])):
        dict_node[np_node[0][i]]=np_node[1][i]

    return dict_node