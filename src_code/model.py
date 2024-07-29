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
from attention import NodeFeatureAggregator, Attention
from get_gc import get_in_neighbors
from time2vec import Time2Vec, t2v


class Model(nn.Module):
    def __init__(self, num_nodes, embedding_dims, hidden_size, dropout, device,threshold,window_size, gcn=False):
        super(Model, self).__init__()
        self.decayer=Decayer(2,"log")
        self.decayer_feature = Decayer(2,"rev")
        self.num_nodes = num_nodes
        self.embedding_dims = embedding_dims
        self.attention = Attention(self.embedding_dims, self.embedding_dims).to(device)
        self.gcn = gcn
        self.device = device
        self.threshold =threshold#100000
        self.window_size = window_size
        self.linear_time = Time2Vec("cos",embedding_dims).to(device)
        self.linear_node = nn.Linear(1, embedding_dims).to(device)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, self.embedding_dims*4).to(device))
        nn.init.xavier_uniform_(self.weight)
        self.time=0#Record the time of the current interaction

    def initialize(self):
        self.feature_dict = {}
        self.adj_time = torch.zeros([self.num_nodes]).to(self.device)
        self.graph=nx.DiGraph()

    def forward(self, edge,label,dict_gc,flag='test'):
        self.time  = float(edge[2])
        self.graph.add_nodes_from(edge[:2])

        #outdated information filter
        if label == 1 and flag=='train':
            pass
        else:
            for node in set(self.graph.nodes()):
                time_interval = self.time - self.adj_time[node]
                if time_interval > self.threshold and node != edge[0] and node != edge[1]:
                    self.graph.remove_node(node)
            self.graph.add_edge(edge[0], edge[1])


        for i in range(2):
            #initialize historical information bank
            if edge[i] not in self.feature_dict:
                self.feature_dict[edge[i]]={}
                self.feature_dict[edge[i]]["feature"]=deque(maxlen=self.window_size)
                self.feature_dict[edge[i]]["time"] = deque(maxlen=self.window_size)

            node_feat= self.node_encoding(edge[i], dict_gc)

            #attention-temporal layer
            neigh_feat = torch.zeros(self.embedding_dims).to(self.device).unsqueeze(dim=0)
            in_neighbors=get_in_neighbors(edge[i],self.graph)
            for neigh in in_neighbors:
                neigh_time = self.adj_time[neigh]
                time_score=self.decayer(torch.FloatTensor([self.time - neigh_time]))
                neigh_feat_i =self.node_encoding(neigh, dict_gc)

                attention_score = self.attention(node_feat.squeeze(dim=0), neigh_feat_i.squeeze(dim=0))
                score =F.leaky_relu(time_score+attention_score)
                neigh_feat +=(score)*neigh_feat_i

            if not self.gcn:
                combined = torch.cat([node_feat.squeeze(dim=0), neigh_feat.squeeze(dim=0)])
            else:
                combined = neigh_feat.squeeze(dim=0)

            # historical information bank
            fearure_i = torch.zeros(self.embedding_dims*2).to(self.device)
            for index in range(len(self.feature_dict[edge[i]]["feature"])):
                weight=self.decayer_feature(torch.FloatTensor([self.time - self.feature_dict[edge[i]]["time"][index]]).to(self.device))
                if self.time - self.feature_dict[edge[i]]["time"][index]<0:
                     print('time',self.time, self.feature_dict[edge[i]]["time"][index],weight)
                fearure_i+=(weight)*self.feature_dict[edge[i]]["feature"][index]

            fearure_i = torch.cat((fearure_i, combined), dim=0)

            if i == 0:
                combined_i = fearure_i
                self.feat = combined
            else:
                combined_all = torch.stack((combined_i, fearure_i), dim=0)
                self.feat = torch.stack((self.feat, combined), dim=0)

        #update network time
        if label == 1 and flag=='train':
            pass
        else:
            for item in range(2):
                self.adj_time[edge[item]] = self.time


        for item in range(2):
            with torch.no_grad():
                self.feature_dict[edge[item]]["feature"].append(self.feat[item].data)
                self.feature_dict[edge[item]]["time"].append(self.time)

        combined_all=combined_all.mm(self.weight.t())


        return torch.relu(combined_all)

    def node_encoding(self,node,dict_gc):
        temp_node = self.linear_time(torch.abs(torch.Tensor([self.time - self.adj_time[node]])).to(self.device))
        eff_tar = self.linear_node(torch.Tensor([dict_gc[node]]).to(self.device))
        node_feat = eff_tar.unsqueeze(dim=0) + temp_node.unsqueeze(dim=0)
        node_feat = F.relu(node_feat)
        node_feat = F.normalize(node_feat)

        return node_feat