# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:55:14 2020

@author: Joepi
"""

import torch

PI = 3.141592



params = {
    "c": 1500,
    "t_fixed": 0,
    "t_dt": 25e-9,
    "t_max": 2166*25e-9,
    "t_min": 2.33e-5,
    "s_num": 512,
    "s_rad": 110e-3,
    "s_ang": PI,
    "g_Nx": 512,
    "g_Ny": 512,
    "g_dx": 150e-3/512,
    "g_dy": 150e-3/512,
    "batch_size": 32
}

class DelayLayer(torch.nn.Module):
    def __init__(self, args = params, device="cuda:0"):
        super(DelayLayer, self).__init__()
        phi = torch.linspace(0, 2*PI, args["s_num"], device=device)
        
        self.nr_of_sensors = torch.tensor(args["s_num"], device = device)
        self.c = torch.tensor(args["c"], device = device)
        self.t0 = torch.tensor(args["t_fixed"] - args["t_min"], device = device)
        self.dt = torch.tensor(args["t_dt"], device = device)
        self.t_max = torch.tensor((args["t_max"] - args["t_fixed"])/args["t_dt"], device=device) 
        
        self.g_Nx = args["g_Nx"]
        self.g_Ny = args["g_Ny"]
        
        self.sensor_x =  args["s_rad"]*torch.cos(phi + args["s_ang"]).view(-1, 1, 1)
        self.sensor_y =  args["s_rad"]*torch.sin(phi + args["s_ang"]).view(-1, 1, 1)
    
        self.grid_x = (torch.linspace(-args["g_Nx"]/2, args["g_Nx"]/2, args["g_Nx"], device = device) * args["g_dx"]).view(1, -1, 1)
        self.grid_y = (torch.linspace(-args["g_Ny"]/2, args["g_Ny"]/2, args["g_Ny"], device = device) * args["g_dy"]).view(1, 1, -1)
    
        self.sub_x = torch.zeros(args["g_Nx"], device = device).view(1, 1, -1)
        self.sub_y = torch.zeros(args["g_Ny"], device = device).view(1, -1, 1)
        
        self.zero = torch.zeros(1, device=device)
        
        self.output = torch.zeros((1, args["s_num"], args["g_Nx"], args["g_Ny"]), device = device)
        
        self.batch_size = args["batch_size"]
        
        self.batch_indexes = torch.arange(0, self.batch_size, device=device, dtype=torch.long) 
        self.batch_steps = torch.arange(0, self.nr_of_sensors//self.batch_size, device = device, dtype=torch.long)
        

    def forward(self, x):
        x[:, :, 0]= 0
        
        for ii in self.batch_steps:
            batch_idx = ii*self.batch_size + self.batch_indexes
            
            signal_batch = x[:, batch_idx]
            sensor_x_batch = self.sensor_x[batch_idx]
            sensor_y_batch = self.sensor_y[batch_idx]
            
            distance = torch.stack(((self.grid_x - sensor_x_batch + self.sub_x), (self.grid_y - sensor_y_batch + self.sub_y)), -1)
            distance = torch.norm(distance, dim=-1)
    
            idx = (distance/ self.c + self.t0) / self.dt
            idx = torch.where(idx > self.t_max, self.zero, idx)
            idx = torch.where(idx < 200, self.zero, idx)
            idx = idx.view(1, self.batch_size, self.g_Nx*self.g_Ny)
    
            d0 = torch.floor(idx) 
            d1 = d0 + 1
    
            #get the weights to get the "inbetween value"
            Wa = (d1 - idx)
            Wb = (idx - d0)
    
            # #get the values of both indexes
            y0 = torch.gather(signal_batch,2, d0.long())
            y1 = torch.gather(signal_batch,2, d1.long())
    
            # making the indexes workable
            self.output[:, batch_idx] = (Wa * y0 + Wb * y1).view(self.batch_size, self.g_Nx, self.g_Ny).sum(0)

        return self.output

if __name__ == "__main__":
    import scipy.io as sio
    
    das = DelayLayer()
    data = sio.loadmat("D:/0. TUE/Stage Spul/reverse_beamform/output.mat")["out"]
    data = torch.tensor(data).to("cuda:0").unsqueeze(0)    
    
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        out = das(data)
        
    sio.savemat("das.mat", {"out": out.detach().cpu().numpy()})
    print(out.size())
    
    print(prof.key_averages().table(sort_by = "cuda_time"))
    
    