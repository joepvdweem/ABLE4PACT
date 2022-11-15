import h5py
import numpy as np

class Dataset:
    def __init__(self, args, split, subsample_factor = 1, eval_mode = 0):
        self.args = args
        self.eval_mode = eval_mode
        self.sensor_idx = np.arange(0, args["input_dims"][0], subsample_factor)
        
        self.f = h5py.File(args["input_file"], 'r')
        if split == 'val':
            self.dataset = self.f
        else:
            self.dataset = self.f[split]
        self.len = len(self.dataset["X"])
        
    def __getitem__(self, i):
        x = self.dataset["X"][i, self.sensor_idx, :]
        y = self.dataset["y"][i]
        
        return (x, y)
    
    def __len__(self):
        if self.eval_mode:
            return 1
        else:
            return self.len
        
if __name__ == "__main__":
    args = {
        "input_file": "D:/0. TUE/Stage Spul/WFM/wfm/Preprocessed/Data.h5" ,
        "input_dims": [512, 2167]
        }
    
    k = Dataset(args, 'Test')
    print(len(k))