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
from create_neg_edge import *
from distance import BFS
from get_gc import get_GC
from learn_graph import train_model, evaluate
from model import Model
from option import args

# Argument and global variables

print(args)
dataset = args.data
divide=args.divide
embedding_dims=args.embedding_dims
hidden_size=args.hidden_size
batch_size=args.bs
test_radio=args.test_radio


#Random seed
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
path = r'D:\model_C\dataset\UCI\out.opsahl-ucsocial'  # dataset pathpath


#Processing dataset
edge_order,list_time_pos,set_node= dataloader.load_data_node(path,dataset)
train_edge_pos=edge_order[:int(len(edge_order)*divide)]
test_edge_pos=edge_order[int(len(edge_order)*divide):]
dict_node=dataloader.reindex(set_node)


#negative sampling
list_time=sorted(list_time_pos)
if not os.path.exists(dataset + "_neg_edges_train" +str(args.divide)+ '.txt'):
    if args.negative_sample=='time':
        train_edges_neg=negative_sample_time(train_edge_pos,edge_order)
    elif  args.negative_sample=='hist':
        train_edges_neg,dict_time_node,dict_node_neigh=negative_sample_hist(train_edge_pos,edge_order, {}, {})
    else:
        train_edges_neg=negative_sample_spatial(train_edge_pos,edge_order,set_node)
    with open(dataset + "_neg_edges_train" +str(args.divide)+ '.txt', "w+") as f_train:
        for neg_edge_train in train_edges_neg:
            f_train.write(str(neg_edge_train[0])+" "+str(neg_edge_train[1])+' '+str(neg_edge_train[2])+'\n')
else:
    train_edges_neg=[]
    train_labels=[]
    with open(dataset + "_neg_edges_train" +str(args.divide)+ '.txt', "r+") as f_train:
        for neg_edge_train in f_train:
            neg_edge_train=neg_edge_train.strip('\n').split(' ')
            train_edges_neg.append([int(neg_edge_train[0]),int(neg_edge_train[1]),int(float(neg_edge_train[2]))])
train_edges,train_labels=merge_edges(train_edge_pos,train_edges_neg)
#
if not os.path.exists(dataset + "_neg_edges_test" +str(args.divide)+ '.txt'):
    if args.negative_sample == 'time':
        test_edge_neg = negative_sample_time(test_edge_pos, edge_order)
    elif args.negative_sample == 'hist':
        test_edge_neg,dict_time_node,dict_node_neigh = negative_sample_hist(test_edge_pos, dict_time_node,dict_node_neigh)
    else:
        test_edge_neg = negative_sample_spatial(test_edge_pos, edge_order, set_node)
    with open(dataset + "_neg_edges_test" +str(args.divide)+'.txt', "w+") as f_neg:
        for neg_edge_test in test_edge_neg:
            f_neg.write(str(neg_edge_test[0])+" "+str(neg_edge_test[1])+' '+str(neg_edge_test[2])+'\n')
else:
    test_edge_neg=[]
    with open(dataset + "_neg_edges_test" +str(args.divide)+ '.txt', "r+") as f_neg:
        for neg_edge_test in f_neg:
            neg_edge_test = neg_edge_test.strip('\n').split(' ')
            test_edge_neg.append([int(neg_edge_test[0]),int(neg_edge_test[1]),int(float(neg_edge_test[2]))])


random.shuffle(test_edge_neg)
test_edge_neg_ = test_edge_neg[:int(len(test_edge_pos) * test_radio)]
test_edge,test_labels=merge_edges(test_edge_pos,test_edge_neg_)


num_nodes=len(set_node)
num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create the model and initialize it
model_ = Model(num_nodes,embedding_dims,hidden_size,args.dropout,device,args.threshold,args.window_size).to(device)
if args.edge_agg != "activation" and args.edge_agg != "origin":
    classification = Classification(hidden_size, 2).to(device) # Create a classifier
else:
    classification = Classification(hidden_size*2, 1).to(device)  # Create a classifier

bfs=BFS(num_nodes)
models = [model_,classification]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=args.lr,weight_decay=args.weight_decay)# Create optimizer
#Preserve experimental results
f = open(str(dataset)+str(divide)+'.txt', 'a+')
f.write(str(args)+'\n')


#Calculate the gravity centrality of the node and save
if not os.path.exists(dataset+'_train_'+str(args.threshold)+str(args.divide)+'.pkl'):
    model_.initialize()
    print('get train dict_gc')
    G=model_.graph
    adj_time=model_.adj_time
    get_GC(train_edges,train_labels,G,adj_time,dataset,args.threshold,bfs,divide,test_radio,flag='train')


with open(dataset+'_train_'+str(args.threshold)+str(args.divide)+'.pkl', "rb") as tf:
    dict_gc=pickle.load(tf)


# Training
print('Model with Supervised Learning')
for epoch in range(args.n_epoch):
    model_.initialize()
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    loss_all, model_,classification=train_model(train_labels,train_edges,model_,optimizer,classification,device,args.edge_agg,models,dict_gc)
    print("epoch :"+str(epoch),'loss',loss_all/ len(train_labels))
    f.write(str(epoch) + ' epoch----' + str(loss_all / len(train_labels)) + '\n')

# Specify a path to save to
PATH = "save_model\model"+dataset+".pt"
torch.save({
    'modelA_state_dict': model_.state_dict(),
    'modelB_state_dict': classification.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_adj':model_.adj_time,
    'model_feature':model_.feature_dict,
    'model_G':model_.graph,
}, PATH)


# # Testing
print('Testing')
if not os.path.exists(dataset+'_test_'+str(test_radio)+"_"+str(args.threshold)+str(args.divide)+'.pkl'):
    print('get test dict_gc')
    G=model_.graph
    adj_time=model_.adj_time.clone()
    get_GC(test_edge,test_labels,G,adj_time,dataset,args.threshold,bfs,divide,test_radio,flag='test')

with open(dataset+'_test_'+str(test_radio)+"_"+str(args.threshold)+str(args.divide)+'.pkl', "rb") as tf:
    dict_gc_test = pickle.load(tf)
#
time.sleep(0.01)
predicts_socre_all,predict_test= evaluate(test_edge,test_labels,model_, classification, args.edge_agg,dict_gc_test)

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
f.write('AUC:' + str(metrics.roc_auc_score(labels ,predicts_socre_all))+"\n")
