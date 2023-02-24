
"""
Aggregation-distillation module
"""

from __future__ import absolute_import, print_function
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np


class ADUnit(nn.Module):
    def __init__(self, lan_dim, fea_dim, r=1, shrink_thres=0.0025):
        super(AD, self).__init__()
        self.lan_dim = lan_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.lan_dim, self.fea_dim))  
        self.bias = None
        self.shrink_thres= shrink_thres
        self.r= r   # distillation radius
        self.reset_parameters()
        
    def reset_parameters(self):                    
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, r=1):
        att_weight = F.linear(input, self.weight) 
        att_weight = F.softmax(att_weight, dim=1)  
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)
        lan_trans = self.weight.permute(1, 0)  
        output = F.linear(att_weight, lan_trans) 
        
        #find the nearest landmark for each latent feature（self.weight）
        nl = []
        ind = att_weight.argmax(1)
        for i in range(input.shape[0]):
            weight_w = self.weight.detach().numpy()
            nl.append(weight_w[ind[i]])
        for i in range(input.shape[0]):
            att_weight2 = att_weight.detach().numpy()
            att_weight2[i,ind[i]] = min(att_weight[i])
        #the second nearest landmark
        nl2 = []
        ind2 = att_weight2.argmax(1)
        for i in range(input.shape[0]):
            nl2.append(weight_w[ind2[i]])
        #find the centre of the landmarks
        cen = 0
        for i in range(self.weight.shape[0]):
            cen = cen + self.weight[i]
        cen = cen/self.weight.shape[0]
        cen = cen.detach().numpy()
        #record the anomalies
        col=[]
        for i in range(input.shape[0]):
            sample = input[i]
            if np.sqrt(np.sum((sample.detach().numpy() - cen) ** 2)) < r:
                col.append(i)
        return {'output': output, 'att': att_weight, 'landmark': self.weight , 'landmark_nearest': nl, 'landmark_nearest2': nl2 ,'col':col}  # output, att_weight

    def extra_repr(self):
        return 'lan_dim={}, fea_dim={}'.format(
            self.lan_dim, self.fea_dim is not None
        )


class ADModule(nn.Module):
    def __init__(self, lan_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(ADModule, self).__init__()
        self.lan_dim = lan_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.lanad = ADUnit(self.lan_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)
        
        if l == 2:
            x = input.permute(0,1)

        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
            
        x= torch.Tensor(x)    
        x = x.contiguous()    
        x = x.view(-1, s[1])         
        y_and = self.lanad(x)  
        y = y_and['output']         
        att = y_and['att']          
        lan = y_and['landmark']       
        nl = y_and['landmark_nearest'] 
        nl2 = y_and['landmark_nearest2'] 
        col = y_and['col']
        
        if l ==2:
            att = att
            lan = lan
        elif l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.lan_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.lan_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.lan_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att, 'lan':lan , 'landmark_nearest':nl,'landmark_nearest2':nl2, 'col':col }

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

