import torch
from pathlib import Path
from typing import Union
import lightning as L
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.enums import Sets, Datas
from utils.fetch_data import fetch_data
import math

def extrair_texto(string):
    try:
        parte_intermediaria = string.split('.')[1]
        resultado = parte_intermediaria.split('/')[0]
        return resultado
    except IndexError:
        return "Padrão não encontrado"

# É utilizado apenas para garantir que os mesmos dados são pegados para percentuais diferentes
# columns = [45,0,26,39,49,54,22,53,32,15,24,5,19,29,17,4,11,58,6,2,37,42,31,8,35,20,50,25,43,21,1,23,40,44,7,59,30,28,46,48,41,55,12,47,14,16,38,52,18,56,51,36,57,34,27,13,10,3,9,33]
class HarDataset(Dataset):
    def __init__(self, path, with_flatter=True, percent = 1):
        # read data
        xy = pd.read_csv(f'{path}', sep=',')
        xy = xy.sample(frac=percent)

        if percent < 1:
            xy = xy.sample(frac=percent, random_state=42)
            # total = math.ceil(percent * len(xy))
            # y = xy.iloc[columns[0:total]]

        x = xy.iloc[:, 1:361].values

        if(with_flatter):
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
            
class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()
        self.mainData = Datas.HAR
        root_data_dir = f"data/{self.mainData.value}"
        self.root_data_dir = Path(root_data_dir)

        self.batch_size = batch_size
        fetch_data(
            root_data_dir = self.root_data_dir,
            type = self.mainData.type,
            files = ['train', 'validation', 'test'],
            zip_name = f'{self.mainData.value}.zip',
            url = self.mainData.url
        )        

    def _get_dataset_dataloader(self, path: Path, shuffle: bool, with_flatter:bool, percent=1) -> DataLoader[HarDataset]:
        dataset = HarDataset(path, with_flatter, percent)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
        )
        return dataloader, dataset
    
    def get_dataloader(self, set:Sets, with_flatter=True, shuffle=True, percent=1):
        return self._get_dataset_dataloader(
            f"{self.root_data_dir}/{set}.{self.mainData.type}", 
            shuffle=shuffle,
            with_flatter = with_flatter,
            percent = percent
        )
        