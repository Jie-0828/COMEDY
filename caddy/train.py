import argparse
import math
import os
import pickle
import time

import networkx as nx
from scipy.sparse import lil_matrix
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import dataloader
from tqdm import tqdm
from classifier import Classification
from distance import BFS
from spatial_encoding import get_GC
from learn_graph import train_model, evaluate
from model import CADDY
from negative_sample import *

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, bitcoinotc, UCI,DIGG,bitcoinalpha or Reddit', default='fb-forum')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--hidden_size', type=int, default=32, help='Dimentions of the node hidden size')
parser.add_argument('--node_dim', type=int, default=8, help='Dimentions of the time embedding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--ratio', type=str,help='the ratio of training sets', default=0.3)
parser.add_argument('-dropout', type=float, help='dropout', default=0)
parser.add_argument('-anomaly_radio', type=float, help='test_radio', default=0.1)
parser.add_argument('-threshold', type=float, help='inactive nodes threshold', default=10000)
parser.add_argument('-window_size', type=float, help='the queue size of the historical information bank', default=10)

args = parser.parse_args()
dataset = args.data
ratio=args.ratio
node_dim=args.node_dim
hidden_size=args.hidden_size
anomaly_radio=args.anomaly_radio
print(args)

#Random seed
seed=0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
if dataset == 'bitcoinotc':
    path = r'dataset\soc-sign-bitcoinotc.csv'  # dataset pathpath
elif dataset == 'UCI':
    path = r'dataset\out.opsahl-ucsocial'  # dataset path
elif dataset == 'DIGG':
    path = r'dataset/out.munmun_digg_reply'
elif dataset == 'bitcoinalpha':
    path = r'dataset\soc-sign-bitcoinalpha.csv'
elif dataset == 'ia-contacts_dublin':
    path = r'dataset\ia-contacts_dublin.edges'
elif dataset == 'fb-forum':
    path = r'dataset\fb-forum_old.edges'


edge_order,num_nodes= dataloader.load_data_node(path,dataset)
# random.shuffle(data)
train_edge_pos=edge_order[:int(len(edge_order)*ratio)]
test_edge_pos=edge_order[int(len(edge_order)*ratio):]

print('train：'+str(len(train_edge_pos))+','+'test：'+str(len(test_edge_pos)))
num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create the model and initialize it
caddy = CADDY(num_nodes,node_dim,hidden_size,args.dropout,device,args.threshold,args.window_size).to(device)
if args.edge_agg != "activation" and args.edge_agg != "origin":
    # classification = Classification((node_dim+1), num_labels).to(device) # Create a classifier
    classification = Classification(hidden_size, 2).to(device) # Create a classifier
else:
    classification = Classification(hidden_size*2, 1).to(device)  # Create a classifier
bfs=BFS(num_nodes)

models = [caddy,classification]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=args.lr)# Create optimizer

# Training
train_edge,train_labels,set_node=negative_sample(train_edge_pos,edge_order)
if not os.path.exists(dataset+'_train_'+str(args.threshold)+str(ratio)+'.pkl'):
    caddy.initialize()
    print('spatial encoding')
    G=caddy.graph
    adj_time=caddy.adj_time
    get_GC(train_edge,train_labels,G,adj_time,dataset,args.threshold,bfs,ratio,anomaly_radio,flag='train')
    del adj_time


with open(dataset+'_train_'+str(args.threshold)+str(ratio)+'.pkl', "rb") as tf:
    dict_gc=pickle.load(tf)

print('Model with Supervised Learning')
for epoch in range(args.n_epoch):
    caddy.initialize()
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    loss_all, caddy,classification=train_model(train_labels,train_edge,caddy,optimizer,classification,device,args.edge_agg,dict_gc,models)
    print("epoch :"+str(epoch),'loss',loss_all/ len(train_labels))

# Testing
print('Testing')
test_edge,test_labels,set_node_test=negative_sample_test(test_edge_pos,edge_order,anomaly_radio)
if not os.path.exists(dataset+'_test_'+str(anomaly_radio)+"_"+str(args.threshold)+str(ratio)+'.pkl'):
    print('spatial encoding')
    G=caddy.graph
    adj_time=caddy.adj_time
    get_GC(test_edge,test_labels,G,adj_time,dataset,args.threshold,bfs,ratio,anomaly_radio,flag='test')

with open(dataset+'_test_'+str(anomaly_radio)+"_"+str(args.threshold)+str(ratio)+'.pkl', "rb") as tf:
    dict_gc_test = pickle.load(tf)

time.sleep(0.01)
predicts_socre_all,predict_test= evaluate(test_edge,test_labels,caddy, classification, args.edge_agg,dict_gc_test)

predicts_test_all=[]
for i in range(0, len(predicts_socre_all)):
    if predicts_socre_all[i] > 0.5:
        predicts_test_all.append(1)
    else:
        predicts_test_all.append(0)

labels = np.array(test_labels)
print(labels)
scores = np.array(predicts_test_all)
print(scores)

print("AUC: "+str(metrics.roc_auc_score(labels ,predicts_socre_all))+"\n")

