import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decayer import Decayer
from time2vec import Time2Vec


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels,time_dim):
        super(Attention, self).__init__()
        self.in_channels = in_channels  #
        self.out_channels = out_channels
        # self.time2vec = Time2Vec('sin', time_dim)

        self.weight_w = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.weight_a = nn.Parameter(torch.FloatTensor(self.out_channels*2, 1))

        # self.leakyrelu = nn.LeakyReLU()

        # self.decayer = Decayer(w, decay_method)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_w)
        nn.init.xavier_uniform_(self.weight_a)

    def forward(self, x_i,x_j):
        wh1 = torch.mm(x_i.unsqueeze(dim=0), self.weight_w)
        wh2 = torch.mm(x_j.unsqueeze(dim=0), self.weight_w)
        x_cat = torch.cat([wh1, wh2], dim=1)

        e = torch.mm(x_cat, self.weight_a)


        return e



