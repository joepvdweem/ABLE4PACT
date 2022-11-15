# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:10:34 2020

@author: Joepi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Antirectifier(nn.Module):
    def __init__(self):
        super(Antirectifier, self).__init__()
        
    def forward(self, X):
        return torch.cat((F.relu(X), -F.relu(X)), dim=-1)

class Able(nn.Module):
    def __init__(self, input_shape):
        super(Able, self).__init__()
        
        self.input_shape = input_shape
        
        N = input_shape[1]
        self.able = nn.Sequential(
            nn.Linear(N/2, N),
            Antirectifier(),
            nn.Dropout(0.2),
            
            nn.Linear(2*N, N//4),
            Antirectifier(),
            nn.Dropout(0.2),
            
            nn.Linear(N//2, N//4),
            Antirectifier(),
            nn.Dropout(0.2),
        
            nn.Linear(N//2, N)      
            )
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        weights = self.able(x)
        x = weights * x
        x = x.sum(-1)
        
        return x

if __name__ == "__main__":
    input_shape = [1, 512, 512, 512]

    able = Able(input_shape)
    
    input = torch.rand(input_shape);
    
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        output = able(input)
    
    print(output.size())
    
    print(prof.key_averages().table(sort_by = "cpu_time"))
