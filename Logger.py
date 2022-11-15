# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:50:19 2020

@author: Joepi
"""

import torch
from tqdm import tqdm
import numpy as np

class Logger:
    def __init__(self, process, metrics):
        self.process = process
        self.logger = {"loss": []}
        self.deepLogger = {"loss": []}
        
        self.metrics = metrics
        for key in metrics.keys():
            self.logger[key] = []
            self.deepLogger[key] = []
        
     
    def logMetrics(self, X, y):
        for key in self.metrics.keys():
            self.logger[key].append(np.array(self.metrics[key](X, y)))
    
    def updateProgbar(self):
        string = ""
        for key in self.logger.keys():
            string = string + key + ": {:.2e} | ".format(np.mean(self.logger[key]))
        self.progbar.set_description(string)
    
    def getProgBar(self):
        self.progbar = tqdm(self.process)
        return self.progbar
    
    def on_batch_end(self, loss, X, y):
        self.logger["loss"].append(np.array(loss))
        self.logMetrics(X, y)
        self.updateProgbar()
        
    def on_epoch_end(self):
        for key in self.logger.keys():
            self.deepLogger[key].append(np.mean(self.logger[key]))
            self.logger[key] = []
        self.progbar.reset()

            
    def on_training_end(self):
        return self.deepLogger

import matplotlib.pyplot as plt

def makeFigure(figures, path):
    plt.figure(figsize=[np.shape(figures)[1]*2, np.shape(figures)[0]*2])
    
    ss = np.shape(figures)
    
    columnTitles = ["Ground Truth", "Rectengular", "DL"]
    rowTitles = ["Derenzo", "PAT", "vb", "data1", "data100", "patient3_158", "signal_slice"]
    
    for ii, xx in enumerate(figures):
        for jj, yy in enumerate(xx):
            ax = plt.subplot2grid([ss[0], ss[1]], [ii, jj])
            ax.imshow(yy)
            ax.axis("off")
            if ii == 0:
                ax.set_title(columnTitles[jj])
            if jj == 0:
                ax.set_ylabel(rowTitles[ii])
    
    plt.saveFig(path, dpi=144)
            
    

if __name__ == "__main__":
    metrics = {"something": lambda x, y:x}
    
    logger = Logger(range(10), metrics)
    
    for i in logger.getProgBar():
        logger.on_batch_end(10, i, 1)
    logger.on_epoch_end()
    
    for j in logger.getProgBar():
        logger.on_batch_end(10, j, 1)
    logger.on_epoch_end()
    
    print(logger.on_training_end())
    
    rand = np.random.rand(7,3,512, 512)
    
    makeFigure(rand)
    
    
    
    