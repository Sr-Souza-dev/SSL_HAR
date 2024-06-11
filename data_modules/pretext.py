import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset


class HarDataset(Dataset):
    def __init__(self, transforms: list = [], path = "train"):
        # initialize the download if you don't have 
        xy = pd.read_csv(f'./data/{path}.csv').values
        x = xy[:, :-2].astype(np.float32)
        y = xy[:, [-1]]

        self.transforms = transforms
        self.n_sample = xy.shape[0]
        self.x_data = torch.from_numpy(x).unsqueeze(1)
        self.labels = {value: indice for indice, value in  enumerate(np.unique(y))}           # Salva todos os labels e discretiza em valores correspondentes ao index
        self.y_data = torch.from_numpy(np.array([self.labels.get(value[0]) for value in y]))  # Salva apenas o index do label 

    def __getitem__(self, index):
        if(len(self.transforms) > 0):
            idx = random.randint(0, len(self.transforms)-1)
            return self.transforms[idx](self.x_data[index]), idx
        return self.x_data[index], self.y_data[index]
    
    def __len__ (self):
        return self.n_sample
    
    def getLabel(self, index):
        # Converte valor em label
        if(len(self.transforms) > 0):
            return self.transforms[index].getName()
        for label, idx in self.labels.items():
            if idx == index:
                return label
            
    def getLabelbyElement(self, index):
        # Recupera o label uma determinada posição
        label_index = self.y_data[index][0]
        for label, idx in self.labels.items():
            if idx == label_index:
                return label


class PamapDataset(Dataset):
    def __init__(self, transforms: list = [], file = "subject101", set = "train"):
        xy = pd.read_csv(f'./data/data_{set}_{file}.dat', sep=" ", dtype=np.float32 ).values
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
        # Converte valor em label
        if(len(self.transforms) > 0):
            return self.transforms[index].getName()
        return "-Undefined-"
            
