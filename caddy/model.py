import networkx as nx
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm

from decayer import Decayer
from attention import Attention
from updata_adj import gravity_centrality


class CADDY(nn.Module):
    def __init__(self, num_nodes, embedding_dims, hidden_size, dropout,alpha, device, gcn=False):
        super(CADDY, self).__init__()
        self.decayer=Decayer(2)
        self.num_nodes = num_nodes
        self.embedding_dims = embedding_dims
        self.attention = Attention(self.embedding_dims, self.embedding_dims).to(device)
        self.gcn = gcn
        self.device = device
        self.threshold = 50000#100000
        self.alpha=alpha

        self.linear_time = nn.Linear(1, self.embedding_dims).to(device)
        self.linear_node = nn.Linear(1, self.embedding_dims).to(device)
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, self.embedding_dims)).to(device)
        nn.init.xavier_uniform_(self.weight)

    def initialize(self):
        # Initialize an empty graph
        self.graph = nx.Graph()
        self.adj_time = torch.zeros([self.num_nodes]).to(self.device)  # Saves the time of the last interaction
        self.adj = torch.zeros([self.num_nodes, self.num_nodes]).to(self.device)

    def forward(self, edge,label):
        # Update network structure
        if label==0:
            self.adj[edge[1]][edge[0]]+=1
            self.graph.add_edge(edge[0], edge[1])
        time = edge[2]

        for i in range(2):  # Two nodes on one side
            neigh_feat = torch.zeros(self.embedding_dims).unsqueeze(dim=0).to(self.device)

            # Find the neighbor node of the target node at this time
            neighs_index = torch.nonzero(self.adj[edge[i]] != 0).view(1, -1).squeeze(dim=0)

            #Relative Temporal Encoding
            temp_node = self.linear_time(torch.abs(torch.Tensor([time-self.adj_time[edge[i]]])).to(self.device))

            #Spatial Encoding
            spa_node = self.linear_node(torch.Tensor([gravity_centrality(self.graph,edge[i])])).to(self.device)

            #Node Encoding Fusion
            node_feat =torch.relu(temp_node.unsqueeze(dim=0)+spa_node.unsqueeze(dim=0))
            node_feat = F.normalize(node_feat)

            if len(neighs_index) != 0:
                for neigh in neighs_index:
                    neigh_time = self.adj_time[neigh]
                    if (time - neigh_time) <= self.threshold:
                        # Relative Temporal Encoding
                        temp_neigh = self.linear_time(torch.abs(torch.Tensor([time - neigh_time])).to(self.device))

                        # Spatial Encoding
                        spa_neigh = self.linear_node(torch.Tensor([gravity_centrality(self.graph,neigh)])).to(self.device)

                        # Node Encoding Fusion
                        neigh_feat_i = torch.relu(temp_neigh.unsqueeze(dim=0)+spa_neigh.unsqueeze(dim=0))
                        neigh_feat_i = F.normalize(neigh_feat_i)

                        #ATAgg
                        time_score = self.decayer((time - neigh_time))#Temporal Decay Function
                        score = self.attention(node_feat.squeeze(dim=0), neigh_feat_i.squeeze(dim=0))#Attention Score
                        neigh_feat += (time_score*score)*neigh_feat_i

            if not self.gcn:
                combined = node_feat.squeeze(dim=0)+neigh_feat.squeeze(dim=0)#Aggregate its own node features
            else:
                combined = neigh_feat.squeeze(dim=0)  # Not aggregate its own node features

            if i == 0:
                combined_i = combined
            else:
                combined_all = torch.stack((combined_i, combined), dim=0)

        # Update node interaction time
        for i in range(2):
            self.adj_time[edge[i]] = time


        del neigh_feat
        combined_all=combined_all.mm(self.weight.t())#Linear Layer


        return combined_all
