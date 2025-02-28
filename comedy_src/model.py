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
from distance import get_in_neighbors, getCC
from time2vec import Time2Vec, t2v


class Model(nn.Module):
    def __init__(self, num_nodes, embedding_dims, hidden_size,k, device, gcn=False):
        super(Model, self).__init__()
        self.decayer=Decayer(2,"rev")
        self.num_nodes = num_nodes
        self.embedding_dims = embedding_dims
        self.attention = Attention(self.embedding_dims, self.embedding_dims,embedding_dims).to(device)
        self.gcn = gcn
        self.device = device
        self.linear_time = Time2Vec("cos",embedding_dims).to(device)
        self.linear_node = nn.Linear(1, embedding_dims).to(device)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, self.embedding_dims*4).to(device))
        nn.init.xavier_uniform_(self.weight)
        self.updater = nn.GRU(embedding_dims*2, embedding_dims*2)
        self.feat=0
        self.graph_remain=0
        self.k=k

    def initialize(self):
        self.feature_dict = {}
        self.adj_time = torch.zeros([self.num_nodes]).to(self.device)
        self.recent_node= deque(maxlen=self.k)
        self.graph=nx.DiGraph()

    def forward(self, edge,label,flag='test'):
        time = edge[2]
        self.graph.add_nodes_from(edge[:2])
        self.edge=edge

        if len(self.recent_node)<self.k:
            subgraph = self.graph.copy()
        else:
            nodes=list(self.recent_node)+edge[:2]
            subgraph = self.graph.subgraph(nodes).copy()

        if label == 1 and flag=='train':
            pass
        else:
            self.graph.add_edge(edge[0], edge[1])
        subgraph.add_edge(edge[0], edge[1])

        for i in range(2):
            neigh_feat = torch.zeros(self.embedding_dims).to(self.device).unsqueeze(dim=0)
            temp_node = self.linear_time(torch.abs(torch.Tensor([time-self.adj_time[edge[i]]])).to(self.device))
            degree_node = (subgraph.in_degree(edge[i]) + subgraph.out_degree(edge[i]))/(len(subgraph.nodes)-1)
            clustering_coeff = getCC(subgraph, edge[i])
            eff_tar=self.linear_node(torch.Tensor([clustering_coeff]))
            dre_tar=self.linear_node(torch.Tensor([degree_node]))

            node_feat =dre_tar.unsqueeze(dim=0)+temp_node.unsqueeze(dim=0)+eff_tar.unsqueeze(dim=0)
            node_feat = F.normalize(node_feat)

            in_neighbors=get_in_neighbors(edge[i],subgraph)
            for neigh in in_neighbors:
                neigh_time = self.adj_time[neigh]
                time_score=self.decayer(torch.FloatTensor([time - neigh_time]))
                temp_neigh = self.linear_time(torch.abs(torch.Tensor([time - neigh_time])).to(self.device))
                degree_neigh = (subgraph.in_degree(neigh) + subgraph.out_degree(neigh))/(len(subgraph.nodes)-1)
                clustering_neigh = getCC(subgraph,neigh)
                eff_neigh = self.linear_node(torch.Tensor([clustering_neigh]))
                dre_neigh = self.linear_node(torch.Tensor([degree_neigh]))
                neigh_feat_i =dre_neigh.unsqueeze(dim=0)+temp_neigh.unsqueeze(dim=0)+eff_neigh.unsqueeze(dim=0)

                # neigh_feat_i = F.relu(neigh_feat_i)
                neigh_feat_i = F.normalize(neigh_feat_i)

                attention_score = self.attention(node_feat.squeeze(dim=0), neigh_feat_i.squeeze(dim=0))
                score =F.leaky_relu(time_score)*attention_score
                neigh_feat +=(score/len(in_neighbors))*neigh_feat_i

            if not self.gcn:
                combined = torch.cat([node_feat.squeeze(dim=0), neigh_feat.squeeze(dim=0)])
            else:
                combined = neigh_feat.squeeze(dim=0)

            if edge[i] not in self.feature_dict:
                fearure_hist = torch.zeros(self.embedding_dims*2).to(self.device)
            else:
                fearure_hist = self.feature_dict[edge[i]].squeeze(dim=0)
            fearure_i = torch.cat((combined, fearure_hist), dim=0)
            # fearure_i = combined+fearure_hist

            if i == 0:
                combined_i = fearure_i
                self.feat = combined
            else:
                combined_all = torch.stack((combined_i, fearure_i), dim=0)
                self.feat = torch.stack((self.feat, combined), dim=0)



        if label == 1 and flag=='train':
            pass
        else:
            for item in range(2):
                if edge[item] in self.recent_node:
                    self.recent_node.remove(edge[item])
                self.recent_node.append(edge[item])
                self.adj_time[edge[item]] = time

            for item in range(2):
                if edge[item] not in self.feature_dict:
                    with torch.no_grad():
                        self.feature_dict[edge[item]] = self.feat[item]
                else:
                    output,h_n = self.updater(self.feat[item].unsqueeze(dim=0),self.feature_dict[edge[item]].unsqueeze(dim=0))
                    with torch.no_grad():
                        self.feature_dict[edge[item]] = F.normalize(h_n).squeeze(dim=0)
                            # self.feature_dict[edge[item]] = h_n.squeeze(dim=0)
        #
        del neigh_feat
        combined_all=combined_all.mm(self.weight.t())

        return torch.relu(combined_all)