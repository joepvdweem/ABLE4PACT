# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:39:21 2020

@author: Joepi
"""

from Able import Able
from dataloader import Dataset
from delayLayer import DelayLayer
from Logger import Logger, makeFigure
import ssim

import scipy.io as sio
import torch
import numpy as np

PI = 3.141592
device = "cpu"

epochs = 1

EVAL_MODE = 1

output_path = ".\\Results"

dataset_params = {
        "input_file": "D:/0. TUE/Stage Spul/WFM/wfm/Preprocessed/Data.h5" ,
        "input_dims": [512, 2167],
        "target_shape": [512,512]
        }

valdataset_params = {
        "input_file": "D:/0. TUE/Stage Spul/WFM/wfm/Preprocessed/valData.h5" ,
        "input_dims": [512, 2167],
        "target_shape": [512,512]
        }

dataloader_params = {
        "batch_size": 1,
        "shuffle": True,
        "pin_memory": True,
    }

train_dataset = Dataset(dataset_params, 'Train', eval_mode = EVAL_MODE)
test_dataset = Dataset(dataset_params, 'Test', eval_mode = EVAL_MODE)
val_dataset = Dataset(valdataset_params, 'val', eval_mode = EVAL_MODE)

train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
test_dataloader = torch.utils.data.DataLoader(test_dataset, **dataloader_params)
val_dataloader = torch.utils.data.DataLoader(val_dataset, **dataloader_params)

delaylayer = DelayLayer(device=device)
able = Able(dataset_params["target_shape"])

network = torch.nn.Sequential(delaylayer,able)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters())

metrics = {"L1": torch.nn.L1Loss(),
           "ssim": ssim.SSIM(),
           }
trainprog = Logger(train_dataloader, metrics)
testprog = Logger(test_dataloader, metrics)

for epoch in range(epochs):
    print("============= EPOCH {} / {} ==============".format(epoch, epochs))

    for i, data in enumerate(trainprog.getProgBar()):
        optimizer.zero_grad()
        
        X, y = data
        
        X = X.to(device)
        y = y.to(device)
        
        X = network(X)
        
        loss = criterion(y, X)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            trainprog.on_batch_end(loss.detach(), X.cpu(), y.cpu())
    
    trainprog.on_epoch_end()
    network.eval()
    with torch.no_grad():
        for i, data in enumerate(testprog.getProgBar()):
            optimizer.zero_grad()
            
            X, y = data
            
            X = X.to(device)
            y = y.to(device)
            
            X = network(X)
            
            loss = criterion(y, X)
                
            testprog.on_batch_end(loss.detach().cpu(), X.cpu(), y.cpu())
    
        testprog.on_epoch_end()
        
        
        if (epoch + 1) % 5 == 0 or EVAL_MODE:
            figure = np.empty((7, 3, 512, 512))
            
            for i, data in enumerate(val_dataloader):
                X, y = data
                
                X = X.to(device)
                y = y.to(device)
                
                X_das = delaylayer(X).sum(1).squeeze()
                
                X_able = network(X)
                
                figure[i, 0] = y.cpu().numpy()
                figure[i, 1] = X_das.cpu().numpy()
                figure[i, 2] = X_able.cpu().numpy()
            
            makeFigure(output_path + "\\ValImages\\epoch{}.png".format(epoch), figure)
            torch.save(network.state_dict(), output_path + "\\models\\epoch{}.pt".format(epoch))
            
    testprog.on_epoch_end()
    
testlog = testprog.on_training_end()
trainlog = trainprog.on_training_end()

sio.savemat("trainLog.mat", trainlog)
sio.savemat("testLog.mat", testlog)
    
