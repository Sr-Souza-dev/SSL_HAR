import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from data.datas import Sets, Datas
class HarDataset(Dataset):
    def __init__(self, transforms: list = [], file:Datas = Datas.PAMAP.value, set:Sets = Sets.TRAIN.value):
        xy = pd.read_csv(f'./data/{file}_{set}.dat', sep=" ", dtype=np.float32 ).values
        xy = xy.reshape(-1, 6, 60)

        self.transforms = transforms
        self.n_sample = xy.shape[0]
        self.data = torch.tensor(xy, dtype=torch.float32)

    def __getitem__(self, index):
        if(len(self.transforms) > 0):
            idx = random.randint(0, len(self.transforms)-1)
            return self.transforms[idx](self.data[index]), idx
        return self.data[index], 0
    
    def __len__ (self):
        return self.n_sample
    
    def getLabel(self, index):
        if(len(self.transforms) > 0):
            return self.transforms[index].getName()
        return "-Undefined-"
            
