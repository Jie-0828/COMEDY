# coding=gb2312

# Create the model and initialize it
import os
import pickle
import random

import numpy as np
import torch

import dataloader
from classifier import Classification
from create_neg_edge_hist import negative_sample_test_hist, negative_sample_hist
from create_neg_edge_str import negative_sample_test_str, negative_sample_str
from create_neg_edge_time import negative_sample_test_time, negative_sample_time
from distance import BFS
from learn_graph import evaluate
from model2 import Model
from sklearn import metrics
from option import args


print(args)
seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Load dataset
if args.data == 'bitcoinotc':
    path = r'D:\model_C\dataset\archive\soc-sign-bitcoinotc.csv'  # dataset pathpath
    path_negative_train = r'D:\model_C\dataset\archive\soc-sign-bitcoinotc_neg_train.csv'  # dataset pathpath
    path_negative_test = r'D:\model_C\dataset\archive\soc-sign-bitcoinotc_neg_test.csv'  # dataset pathpath
elif args.data == 'UCI':
    path = r'D:\model_C\dataset\UCI\out.opsahl-ucsocial'  # dataset path
    path_negative_train=r'D:\model_C\dataset\UCI\uci_neg_train_hist.txt'
    path_negative_test=r'D:\model_C\dataset\UCI\uci_neg_test_hist.txt'
elif args.data == 'DIGG':
    path = r'D:\model_C\dataset\DIGG/digg_.txt'
    path_negative_train = r'D:\model_C\dataset\DIGG/digg_neg_train.txt'
    path_negative_test = r'D:\model_C\dataset\DIGG/digg_neg_test.txt'
elif args.data == 'bitcoinalpha':
    path = r'D:\model_C\dataset\archive\soc-sign-bitcoinalpha.csv'
elif args.data == 'ia-contacts_dublin':
    path = r'D:\model_C\dataset\ia-contacts_dublin\ia-contacts_dublin_.edges'
    path_negative_train=r'D:\model_C\dataset\ia-contacts_dublin\ia-contacts_dublin_neg_train.edges'
    path_negative_test=r'D:\model_C\dataset\ia-contacts_dublin\ia-contacts_dublin_neg_test.edges'
elif args.data == 'fb-forum':
    path = r'D:\model_C\dataset\fb-forum\fb-forum_old.edges'
elif args.data == 'email':
    path = r'D:\model_C\dataset\email-Eu-core-temporal\email-Eu-core-temporal_.txt'
    path_negative_train=r'D:\model_C\dataset\email-Eu-core-temporal.txt\email-Eu-core-temporal_neg_train.txt'
    path_negative_test=r'D:\model_C\dataset\email-Eu-core-temporal.txt\email-Eu-core-temporal_neg_test.txt'
elif args.data == 'mathoverflow':
    path = r'D:\model_C\dataset\sx-mathoverflow.txt\sx-mathoverflow.txt'
    path_negative_train = r'D:\model_C\dataset\sx-mathoverflow.txt\sx-mathoverflow_neg_train.txt'
    path_negative_test = r'D:\model_C\dataset\sx-mathoverflow.txt\sx-mathoverflow_neg_test.txt'
elif args.data == 'askubuntu':
    path = r'D:\model_C\dataset\sx-askubuntu.txt\sx-askubuntu_.txt'
    path_negative_train = r'D:\model_C\dataset\sx-askubuntu.txt\sx-askubuntu_neg_train.txt'
    path_negative_test = r'D:\model_C\dataset\sx-askubuntu.txt\sx-askubuntu_neg_test.txt'


edge_order,list_time_pos,set_node= dataloader.load_data_node(path,args.data,flag='positive')
train_edge_pos=edge_order[:int(len(edge_order)*args.divide)]
print(len(train_edge_pos))
test_edge_pos=edge_order[int(len(edge_order)*args.divide):]
dict_node=dataloader.reindex(set_node)

list_time=sorted(list_time_pos)
if not os.path.exists(args.data + "_neg_edges_train_" +args.sampling+ '.txt'):
    if args.sampling=='time':
        train_edges_neg,list_edge_time=negative_sample_time(train_edge_pos,edge_order)
    elif args.sampling=='hist':
        train_edges_neg,dict_time_node,dict_node_neigh=negative_sample_hist(train_edge_pos,edge_order)
    else:
        train_edges_neg = negative_sample_str(train_edge_pos, edge_order, set_node)
    with open(args.data + "_neg_edges_train_" +args.sampling+ '.txt', "w+") as f_train:
        for neg_edge_train in train_edges_neg:
            f_train.write(str(neg_edge_train[0])+" "+str(neg_edge_train[1])+' '+str(neg_edge_train[2])+'\n')
else:
    train_edges_neg=[]
    train_labels=[]
    with open(args.data + "_neg_edges_train_" +args.sampling+ '.txt', "r+") as f_train:
        for neg_edge_train in f_train:
            neg_edge_train=neg_edge_train.strip('\n').split(' ')
            train_edges_neg.append([int(neg_edge_train[0]),int(neg_edge_train[1]),int(float(neg_edge_train[2]))])
train_edges,train_labels,train_max_time= dataloader.merge_edges(train_edge_pos, train_edges_neg, dict_node, list_time[0])
# print(train_labels)


if not os.path.exists(args.data + "_neg_edges_test_" +args.sampling+ '.txt'):
    if args.sampling=='time':
        test_edge_neg=negative_sample_test_time(test_edge_pos,edge_order,list_edge_time)
    elif args.sampling=='hist':
        test_edge_neg=negative_sample_test_hist(test_edge_pos,edge_order,dict_time_node,dict_node_neigh)
    else:
        test_edge_neg = negative_sample_test_str(test_edge_pos, edge_order, set_node)
    with open(args.data + "_neg_edges_test_" +args.sampling+'.txt', "w+") as f_neg:
        for neg_edge_test in test_edge_neg:
            f_neg.write(str(neg_edge_test[0])+" "+str(neg_edge_test[1])+' '+str(neg_edge_test[2])+'\n')
else:
    test_edge_neg=[]
    with open(args.data + "_neg_edges_test_" +args.sampling+ '.txt', "r+") as f_neg:
        for neg_edge_test in f_neg:
            neg_edge_test = neg_edge_test.strip('\n').split(' ')
            test_edge_neg.append([int(neg_edge_test[0]),int(neg_edge_test[1]),int(float(neg_edge_test[2]))])


random.shuffle(test_edge_neg)
test_edge_neg_ = test_edge_neg[:int(len(test_edge_pos) * args.test_radio)]
# print(test_edge_neg_)
test_edge,test_labels,test_max_time= dataloader.merge_edges(test_edge_pos, test_edge_neg_, dict_node, list_time[0])
print('test_max_time',test_max_time)


num_nodes=len(dict_node)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels=2

model_ = Model(num_nodes,args.embedding_dims,args.hidden_size,args.threshold,device).to(device)
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
Model.recent_node= checkpoint['model_renode']

model_.eval()
classification.eval()

# Testing
print('Testing')
predicts_socre_all,predict_test= evaluate(test_edge,test_labels,model_,classification, args.edge_agg)

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