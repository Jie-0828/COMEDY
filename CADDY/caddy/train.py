import argparse
import math
import time

import networkx as nx
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
from learn_graph import train_model, evaluate
from model import CADDY
from negative_sample import *

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TP-GCN experiments on graph classification task')
parser.add_argument('-d', '--data', type=str, help='dataset to use, e.g.UCI,DIGG,bitcoinotc, bitcoinalpha,ia-contacts-dublin or fb-forum', default='UCI')
parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--hidden_size', type=int, default=32, help='Dimentions of the node hidden size')
parser.add_argument('--alpha_', type=int, default=0.5, help='Distance factor in spatial encoding')
parser.add_argument('--lambda_reg', type=int, default=0.0001, help=' Regularized term hyperparameter')
parser.add_argument('--node_dim', type=int, default=16, help='Dimentions of the node encoding')
parser.add_argument('--edge_agg', type=str, choices=['mean', 'had', 'w1','w2', 'activate'], help='EdgeAgg method', default='mean')
parser.add_argument('--ratio', type=str,help='the ratio of training sets', default=0.3)
parser.add_argument('-dropout', type=float, help='dropout', default=0)
parser.add_argument('-anomaly_ratio', type=float, help='the ratio of anomalous edges in testing set', default=0.1)

args = parser.parse_args()
print(args)

#Random seed
seed=824
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load dataset
if args.data == 'UCI':
    path = r'dataset/out.opsahl-ucsocial'  # dataset path
elif args.data == 'DIGG':
    path = r''
elif args.data == 'bitcoinotc':
    path = r'dataset/soc-sign-bitcoinotc.csv'
elif args.data == 'bitcoinalpha':
    path = r''
elif args.data == 'ia-contacts_dublin':
    path = r''
elif args.data == 'fb-forum':
    path = r''

edge_order,num_nodes= dataloader.load_data_node(path,args.data)
train_edge_pos=edge_order[:int(len(edge_order)*args.ratio)]#Divide the training set and test set
test_edge_pos=edge_order[int(len(edge_order)*args.ratio):]

print('train edges：'+str(len(train_edge_pos))+','+'test edges：'+str(len(test_edge_pos)))
num_labels = 2  # Binary classification task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model and initialize it
caddy = CADDY(num_nodes,args.node_dim,args.hidden_size,args.dropout,args.alpha_,device).to(device)
if args.edge_agg != "activation" and args.edge_agg != "origin":
    detector = Classification(args.hidden_size, num_labels).to(device) # Create a anomaly detector object
else:
    detector = Classification(args.hidden_size*2, num_labels).to(device)

models = [caddy,detector]
params = []
for model in models:
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

optimizer = torch.optim.Adam(params, lr=args.lr)# Create optimizer
# f = open(str(args.data)+str(args.ratio)+'.txt', 'a+')
# f.write(str(args.hidden_size) + ',' + str(args.node_dim) + '\n')

# Training
train_edge,train_labels,set_node=negative_sample(train_edge_pos,edge_order)#negative_sample for training set
print('CADDY Training')
for epoch in range(args.n_epoch):
    caddy.initialize()
    time.sleep(0.0001)
    print('----------------------EPOCH %d-----------------------' % epoch)
    loss_all,caddy,detector=train_model(train_edge,train_labels,caddy,optimizer,detector,device,args.edge_agg,args.lambda_reg)
    print("epoch :"+str(epoch),'loss',loss_all/len(train_labels))
    # f.write(str(epoch) + ' epoch----' + str(loss_all / len(train_labels)) + '\n')

# Specify a path to save model
PATH = "save_model\caddy_"+args.data+".pt"
torch.save({
    'modelA_state_dict': caddy.state_dict(),
    'modelB_state_dict': detector.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_adj':caddy.adj,
    'model_G':caddy.graph,
    'model_adj_time':caddy.adj_time
}, PATH)

# Testing
print('CADDY Testing')
test_edge,test_labels,set_node_test=negative_sample_test(test_edge_pos,edge_order,args.anomaly_ratio)#negative_sample for test set
predicts_socre_all,predicts_test_all= evaluate(test_edge,test_labels,caddy, detector, args.edge_agg)


# Print result
labels = np.array(test_labels)
print(labels)
scores = np.array(predicts_test_all)
print(scores)

print("AUC: "+str(metrics.roc_auc_score(labels ,predicts_socre_all))+"\n")
# f.write('AUC:' + str(metrics.roc_auc_score(labels ,predicts_socre_all))+"\n")
# f.write("\n")


