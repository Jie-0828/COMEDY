import torch
import torch.nn as nn
from torch.nn import init

class Decayer(nn.Module):
    def __init__(self, w, decay_method='exp'):
        super(Decayer,self).__init__()
        self.decay_method = decay_method
        # self.linear = nn.Linear(1,1)
        self.w = w
        # self.w =nn.Parameter(torch.Tensor(1)).unsqueeze(dim=0)
        # nn.init.xavier_uniform_(self.w)

    def exponetial_decay(self, delta_t):
        # return self.linear(torch.exp(-self.w*delta_t).unsqueeze(dim=0))
        return torch.exp(-self.w*delta_t)
    def log_decay(self, delta_t):
        return 1/torch.log(2.7183 + self.w*delta_t)
    def rev_decay(self, delta_t):
        return 1/(1 + self.w*delta_t)

    def forward(self,delta_t):
        if self.decay_method == 'exp':
            return self.exponetial_decay(delta_t)
        elif self.decay_method == 'log':
            return self.log_decay(delta_t)
        elif self.decay_method == 'rev':
            return self.rev_decay(delta_t)
        else:
            return self.exponetial_decay(delta_t)
