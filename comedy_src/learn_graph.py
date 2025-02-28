# In[1]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_agg import edge_agg_function
from model import *
from tqdm import tqdm, trange


def evaluate(test_edge,test_labels,model_, classification, edge_agg):
    """
    Test the performance of the model
    Parameters:
        datacenter: datacenter object
        graphSage：Well trained model object
        classification: Well trained classificator object
    """
    with torch.no_grad():
        predicts_test_all = []  # Store the model prediction results
        predicts_socre_all = []  # Store the model prediction scores

        pbar = tqdm(range(len(test_edge)), total=len(test_edge))
        for index in pbar:
            edge = test_edge[index]

            embeddings = model_(edge,test_labels[index])
            edge_embed = edge_agg_function(edge_agg, embeddings).view(1, -1)

            logists_edge = torch.softmax(classification(edge_embed),1)

            test_probabilities_np = logists_edge[:, 1]


            _, predicts_test = torch.max(logists_edge, 1)
            predicts_test_all.append(predicts_test.data[0].cpu())
            predicts_socre_all.append(test_probabilities_np[0].cpu())


    return predicts_socre_all,predicts_test_all

def train_model(train_labels,train_edge,model_,optimizer,classification,device,edge_agg,models):
    criterion=nn.CrossEntropyLoss()
    lambda_reg=0
    loss_all = 0

    pbar = tqdm(range(len(train_edge)), total =len(train_edge))
    for index in pbar:
        edge=train_edge[index]

        embeddings=model_(edge,train_labels[index],flag='train')
        edge_embed=edge_agg_function(edge_agg,embeddings).view(1,-1)

        logists_edge = classification(edge_embed)

        label=torch.LongTensor([train_labels[index]]).to(device)
        loss_edge = criterion(logists_edge,label)

        regularization_loss =0
        for model in models:
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))

        assert not torch.isnan(loss_edge).any()
        loss=(loss_edge+lambda_reg*regularization_loss)
        loss_all+=loss
        optimizer.zero_grad()
        for model in models:
            model.zero_grad()
        loss.backward()


        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return loss_all, model_,classification



