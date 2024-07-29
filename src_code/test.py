# coding=gb2312

# Create the model and initialize it
import os
import pickle
import random

import numpy as np
import torch

import dataloader
from classifier import Classification
from create_neg_edge import *
from distance import BFS
from get_gc import get_GC
from learn_graph import evaluate
from model import Model
from sklearn import metrics
from option import args


print(args)
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Load dataset
path = r''  # dataset path

edge_order,list_time_pos,set_node= dataloader.load_data_node(path,args.data)
train_edge_pos=edge_order[:int(len(edge_order)*args.divide)]
print(len(train_edge_pos))
test_edge_pos=edge_order[int(len(edge_order)*args.divide):]
dict_node=dataloader.reindex(set_node)

test_edge_neg=[]
with open(args.data + "_neg_edges_test" +str(args.divide)+ '.txt', "r+") as f_neg:
    for neg_edge_test in f_neg:
        neg_edge_test = neg_edge_test.strip('\n').split(' ')
        test_edge_neg.append([int(neg_edge_test[0]),int(neg_edge_test[1]),int(float(neg_edge_test[2]))])
random.shuffle(test_edge_neg)
test_edge_neg_ = test_edge_neg[:int(len(test_edge_pos) * args.test_radio)]
test_edge,test_labels=merge_edges(test_edge_pos,test_edge_neg_)


num_nodes=len(dict_node)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels=2

model_ = Model(num_nodes,args.embedding_dims,args.hidden_size,args.dropout,device,args.threshold,args.window_size).to(device)
if args.edge_agg != "activation" and args.edge_agg != "origin":
    classification = Classification(args.hidden_size, num_labels).to(device) # Create a classifier
else:
    classification = Classification(args.hidden_size*2, num_labels).to(device)  # Create a classifier

models = [model_,classification]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)


PATH = "save_model\model"+args.data+".pt"
checkpoint = torch.load(PATH)
model_.load_state_dict(checkpoint['modelA_state_dict'])
classification.load_state_dict(checkpoint['modelB_state_dict'])
Model.adj_time = checkpoint['model_adj']
Model.graph = checkpoint['model_G']
Model.feature_dict= checkpoint['model_feature']


# Testing
model_.eval()
classification.eval()
print('Testing')
f = open(str(args.data)+str(args.divide)+'.txt', 'a+')
bfs=BFS(num_nodes)
if not os.path.exists(args.data+'_test_'+str(args.test_radio)+"_"+str(args.threshold)+str(args.divide)+'.pkl'):
    print('get test dict_gc')
    G=model_.graph.copy()
    adj_time=model_.adj_time.clone()
    get_GC(test_edge,test_labels,G,adj_time,args.data,args.threshold,bfs,args.divide,args.test_radio,flag='test')


with open(args.data+'_test_'+str(args.test_radio)+"_"+str(args.threshold)+str(args.divide)+'.pkl', "rb") as tf:
    dict_gc_test = pickle.load(tf)
predicts_socre_all,predict_test= evaluate(test_edge,test_labels,model_,classification, args.edge_agg,dict_gc_test)

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
