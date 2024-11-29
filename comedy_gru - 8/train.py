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
from create_neg_edge_hist import negative_sample_hist, negative_sample_test_hist
from create_neg_edge_str import  negative_sample_str, negative_sample_test_str
from create_neg_edge_time import negative_sample_time, negative_sample_test_time
from distance import BFS
# from get_gc import get_GC
from learn_graph import train_model, evaluate
from model2 import Model
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
if dataset == 'UCI':
    path = r'D:\model_C\dataset\UCI\out.opsahl-ucsocial'  # dataset pathpath


edge_order,list_time_pos,set_node= dataloader.load_data_node(path,dataset,flag='positive')
train_edge_pos=edge_order[:int(len(edge_order)*divide)]
test_edge_pos=edge_order[int(len(edge_order)*divide):]
dict_node=dataloader.reindex(set_node)


list_time=sorted(list_time_pos)
if not os.path.exists(dataset + "_neg_edges_train_" +args.sampling+ '.txt'):
    if args.sampling=='time':
        train_edges_neg,list_edge_time=negative_sample_time(train_edge_pos,edge_order)
    elif args.sampling=='hist':
        train_edges_neg,dict_time_node,dict_node_neigh=negative_sample_hist(train_edge_pos,edge_order)
    else:
        train_edges_neg = negative_sample_str(train_edge_pos, edge_order, set_node)

    with open(dataset + "_neg_edges_train_" +args.sampling+ '.txt', "w+") as f_train:
        for neg_edge_train in train_edges_neg:
            f_train.write(str(neg_edge_train[0])+" "+str(neg_edge_train[1])+' '+str(neg_edge_train[2])+'\n')
else:
    train_edges_neg=[]
    train_labels=[]
    with open(dataset + "_neg_edges_train_" +args.sampling+ '.txt', "r+") as f_train:
        for neg_edge_train in f_train:
            neg_edge_train=neg_edge_train.strip('\n').split(' ')
            train_edges_neg.append([int(neg_edge_train[0]),int(neg_edge_train[1]),int(float(neg_edge_train[2]))])
train_edges,train_labels,train_max_time= dataloader.merge_edges(train_edge_pos, train_edges_neg, dict_node, list_time[0])
print('train_max_time',train_max_time)


#
if not os.path.exists(dataset + "_neg_edges_test_" +args.sampling+ '.txt'):
    if args.sampling=='time':
        test_edge_neg=negative_sample_test_time(test_edge_pos,edge_order,list_edge_time)
    elif args.sampling=='hist':
        test_edge_neg=negative_sample_test_hist(test_edge_pos,edge_order,dict_time_node,dict_node_neigh)
    else:
        test_edge_neg = negative_sample_test_str(test_edge_pos, edge_order, set_node)
    with open(dataset + "_neg_edges_test_" +args.sampling+'.txt', "w+") as f_neg:
        for neg_edge_test in test_edge_neg:
            f_neg.write(str(neg_edge_test[0])+" "+str(neg_edge_test[1])+' '+str(neg_edge_test[2])+'\n')
else:
    test_edge_neg=[]
    with open(dataset + "_neg_edges_test_" +args.sampling+ '.txt', "r+") as f_neg:
        for neg_edge_test in f_neg:
            neg_edge_test = neg_edge_test.strip('\n').split(' ')
            test_edge_neg.append([int(neg_edge_test[0]),int(neg_edge_test[1]),int(float(neg_edge_test[2]))])


random.shuffle(test_edge_neg)
test_edge_neg_ = test_edge_neg[:int(len(test_edge_pos) * test_radio)]
test_edge,test_labels,test_max_time= dataloader.merge_edges(test_edge_pos, test_edge_neg_, dict_node, list_time[0])
#
num_nodes=len(set_node)
num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create the model and initialize it
model_ = Model(num_nodes,embedding_dims,hidden_size,args.threshold,device).to(device)
if args.edge_agg != "activation" and args.edge_agg != "origin":
    classification = Classification(hidden_size, 2).to(device) #Create a classifier
else:
    classification = Classification(hidden_size*4, 1).to(device)  #Create a classifier
bfs=BFS(num_nodes)

models = [model_,classification]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=args.lr,weight_decay=args.weight_decay)
f = open(str(dataset)+str(divide)+str(args.sampling)+'.txt', 'a+')
f.write(str(args)+'\n')
#
#
auc=[]
print('Model with Supervised Learning')
for epoch in range(args.n_epoch):
    model_.initialize()
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    loss_all, model_,classification=train_model(train_labels,train_edges,model_,optimizer,classification,device,args.edge_agg,models)
    print("epoch :"+str(epoch),'loss',loss_all/ len(train_labels))
    f.write(str(epoch) + ' epoch----' + str(loss_all / len(train_labels)) + '\n')

    predicts_socre_all, predict_test = evaluate(test_edge, test_labels, model_, classification, args.edge_agg)
    auc.append(metrics.roc_auc_score(np.array(test_labels), predicts_socre_all))
    print('AUC:' + str(metrics.roc_auc_score(np.array(test_labels), predicts_socre_all)) + "\n")
    f.write('AUC:' + str(metrics.roc_auc_score(np.array(test_labels), predicts_socre_all)) + "\n")
f.write("\n")

print("best AUC: " + str(max(auc)))


# Specify a path to save to
PATH = "save_model\model"+dataset+".pt"
torch.save({
    'modelA_state_dict': model_.state_dict(),
    'modelB_state_dict': classification.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_adj':model_.adj_time,
    'model_feature':model_.feature_dict,
    'model_G':model_.graph,
    'model_renode':model_.recent_node,
}, PATH)

# # Testing
print('Testing')

