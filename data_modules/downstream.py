import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

def extrair_texto(string):
    try:
        parte_intermediaria = string.split('.')[1]
        resultado = parte_intermediaria.split('/')[0]
        return resultado
    except IndexError:
        return "Padrão não encontrado"
    
class HarDataset(Dataset):
    def __init__(self, path = "train"):
        # initialize the download if you don't have 
        xy = pd.read_csv(f'./data/har/{path}.csv', sep=',')
        y = xy['csv'].apply(extrair_texto).values
        x = xy.iloc[:, 1:361].values
        x = x.reshape((-1, 6, 60))

        aux = {value: indice for indice, value in  enumerate(np.unique(y))} 
        self.labels = {indice: value for indice, value in  enumerate(np.unique(y))} 
        self.n_sample = xy.shape[0]
        self.x_data = torch.from_numpy(x).float()
        self.y_data = torch.from_numpy(np.array([aux.get(value) for value in y]))       

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__ (self):
        return self.n_sample
    
    def getLabel(self, index):
        return self.labels[index]
            
    def getLabelbyElement(self, index):
        label_index = self.y_data[index][0]
        return self.labels[label_index]
            