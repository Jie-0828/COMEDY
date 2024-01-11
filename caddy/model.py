import copy

import networkx as nx
import torch
import torch.nn as nn
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict, deque
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm

from decayer import Decayer
from attention import Attention
from spatial_encoding import get_in_neighbors
from time2vec import Time2Vec


class CADDY(nn.Module):
    def __init__(self, num_nodes, embedding_dims, hidden_size, dropout, device,threshold,window_size, gcn=False):
        super(Model, self).__init__()
        self.decayer=Decayer(2)
        self.decayer_feature = Decayer(2,"rev")
        self.num_nodes = num_nodes
        self.embedding_dims = embedding_dims
        self.attention = Attention(self.embedding_dims, self.embedding_dims,embedding_dims).to(device)
        self.gcn = gcn
        self.device = device
        self.threshold =threshold
        self.window_size = window_size  # The length of the feature queue
        self.linear_time = nn.Linear(1, embedding_dims).to(device)
        # self.time_encode = TimeEncode(embedding_dims)
        self.linear_node = nn.Linear(1, embedding_dims).to(device)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, self.embedding_dims*2)).to(device)
        nn.init.xavier_uniform_(self.weight)
        self.edge=0#Record the current interactive edge
        self.feat=0#Record the current node features
        self.graph_remain=0

    def initialize(self):
        self.feature_bank = {}  # historical information bank
        self.adj_time = torch.zeros([self.num_nodes]).to(self.device)  # record The time of the last interaction
        self.graph=nx.DiGraph()
    def forward(self, edge,label,dict_gc):
        time = edge[2]
        self.edge=edge
        self.graph.add_nodes_from(edge[:2])


        #remove inactive nodes and their edges
        for node in set(self.graph.nodes()):
            time_interval = time - self.adj_time[node]
            if time_interval > self.threshold and node != edge[0] and node != edge[1]:
                self.graph.remove_node(node)

        if label == 0:
            self.graph.add_edge(edge[0], edge[1])

        for i in range(2):
            if edge[i] not in self.feature_bank:
                self.feature_bank[edge[i]]={}
                self.feature_bank[edge[i]]["feature"]=deque(maxlen=self.window_size)
                self.feature_bank[edge[i]]["time"] = deque(maxlen=self.window_size)


            neigh_feat = torch.zeros(self.embedding_dims).to(self.device).unsqueeze(dim=0)
            temp_node = self.linear_time(torch.abs(torch.Tensor([time-self.adj_time[edge[i]]])).to(self.device))  # 时间编码
            eff_tar = self.linear_node(torch.Tensor([dict_gc[int(edge[i])]]).to(self.device))
            node_feat =eff_tar.unsqueeze(dim=0)+temp_node.unsqueeze(dim=0)

            # node_feat = F.relu(node_feat)
            node_feat = F.normalize(node_feat)

            in_neighbors=get_in_neighbors(edge[i],self.graph)
            for neigh in in_neighbors:
                neigh_time = self.adj_time[neigh]
                time_score=self.decayer(torch.FloatTensor([time - neigh_time]))
                temp_neigh = self.linear_time(torch.abs(torch.Tensor([time - neigh_time])).to(self.device))  # 时间编码
                # temp_neigh = self.time_encode(torch.abs(torch.Tensor([time - neigh_time])).to(self.device))  # 时间编码
                eff_neigh = self.linear_node(torch.Tensor([dict_gc[int(neigh)]]).to(self.device))
                neigh_feat_i =eff_neigh.unsqueeze(dim=0)+temp_neigh.unsqueeze(dim=0)

                # neigh_feat_i = F.relu(neigh_feat_i)
                neigh_feat_i = F.normalize(neigh_feat_i)

                attention_score = self.attention(node_feat.squeeze(dim=0), neigh_feat_i.squeeze(dim=0))
                score =F.leaky_relu(attention_score+time_score)
                neigh_feat +=(score/len(in_neighbors))*neigh_feat_i

            if not self.gcn:
                combined = torch.cat([node_feat.squeeze(dim=0), neigh_feat.squeeze(dim=0)])
                # combined = node_feat.squeeze(dim=0)+ neigh_feat.squeeze(dim=0)
            else:
                combined = neigh_feat.squeeze(dim=0)  # 不聚合自身节点特征
                # self.feature[edge[i]] += combined

            #将队列之前的所有的特征按时间衰减相加，在加上当前的节点特征
            fearure_i = torch.zeros(self.embedding_dims*2).to(self.device)
            for index in range(len(self.feature_bank[edge[i]]["feature"])):
                weight=self.decayer_feature(torch.FloatTensor([time - self.feature_bank[edge[i]]["time"][index]]).to(self.device))
                fearure_i+=(weight)*self.feature_bank[edge[i]]["feature"][index]
            fearure_i+=combined

            if i == 0:
                combined_i = fearure_i
                self.feat = combined
            else:
                combined_all = torch.stack((combined_i, fearure_i), dim=0)
                self.feat = torch.stack((self.feat, combined), dim=0)

        self.update_network()

        del neigh_feat
        combined_all=combined_all.mm(self.weight.t())


        return torch.relu(combined_all)
        # return combined_all

    def update_network(self):
        time = self.edge[2]
        for i in range(2):
            with torch.no_grad():
                self.feature_bank[self.edge[i]]["feature"].append(self.feat[i].data)
                self.feature_bank[self.edge[i]]["time"].append(time)
                self.adj_time[self.edge[i]] = time
