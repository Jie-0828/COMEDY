# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_agg import edge_agg_function
from model import *
from tqdm import tqdm


def evaluate(test_edge,test_labels,caddy, detector, edge_agg):
    """
    Test the performance of the model
    Parameters:
        dataset: test_edge,test_labels
        caddy：Well trained caddy model object
        detector: Well trained anoamly detector object
        optimizer:Adam optimizer
        edge_agg: EdgeAgg function
    """
    with torch.no_grad():
        predicts_test_all = []  # Store the model prediction results
        predicts_socre_all = []  # Store the model prediction scores

        pbar = tqdm(range(len(test_edge)), total=len(test_edge))
        for index in pbar:
            edge = test_edge[index]
            embeddings = caddy(edge,test_labels[index])
            edge_embed = edge_agg_function(edge_agg, embeddings).view(1, -1)#Converted to edge embedding

            logists_edge = detector(edge_embed)

            _, predicts_test = torch.max(logists_edge, 1)
            predicts_test_all.append(predicts_test.data[0].cpu())
            predicts_socre_all.append(logists_edge[0][1].cpu())
    return predicts_socre_all,predicts_test_all

def train_model(train_edge,train_labels,caddy,optimizer,detector,device,edge_agg,lambda_reg):
    """
    Traing the caddy and anomaly detector
    Parameters:
        dataset: train_edge,train_labels
        caddy：caddy model object
        detector: anoamly detector object
        optimizer:Adam optimizer
        edge_agg: EdgeAgg function
    """
    models = [caddy,detector]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer.zero_grad()
    for model in models:
        model.zero_grad()

    loss_all = 0
    pbar = tqdm(range(len(train_edge)), total =len(train_edge))
    for index in pbar:
        edge=train_edge[index]
        embeddings=caddy(edge,train_labels[index])
        edge_embed=edge_agg_function(edge_agg,embeddings).view(1,-1)#Node embedding is converted to edge embedding

        logists_edge = detector(edge_embed)#anomaly detection
        loss_edge = -torch.sum(logists_edge[range(logists_edge.size(0)), train_labels[index]], 0)

        regularization_loss = 0
        for model in models:
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))#regularization

        assert not torch.isnan(loss_edge).any()
        loss=(loss_edge+lambda_reg*regularization_loss)
        loss_all+=loss

        loss.backward()


    for model in models:
        nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    return loss_all, caddy,detector



