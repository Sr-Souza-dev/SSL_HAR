import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import lightning as L
from torch.utils.data import Dataset
from utils.enums import Sets, Datas
from torch.utils.data import DataLoader, Dataset
from utils.enums import Sets
from utils.fetch_data import fetch_data

class HarDataset(Dataset):
    def __init__(self, path, sep= " ", transforms: list = [], with_flatter=True):
        xy = pd.read_csv(f'{path}', sep=sep, dtype=np.float32 ).values

        if with_flatter:
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


class HarDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        main_data = Datas.MOBIT
    ):
        super().__init__()
        self.main_data = main_data
        root_data_dir = f"data/{self.main_data.value}"
        self.root_data_dir = Path(root_data_dir)

        self.batch_size = batch_size
        fetch_data(
            root_data_dir = self.root_data_dir,
            type = self.main_data.type,
            files = ['train', 'test'],
            zip_name = f'{self.main_data.value}.zip',
            url = self.main_data.url
        )        

    def _get_dataset_dataloader(self, path: Path, shuffle: bool, transforms:list, with_flatter:bool) -> DataLoader[HarDataset]:
        dataset = HarDataset(
            path = path,
            sep = self.main_data.sep,
            transforms = transforms,
            with_flatter = with_flatter
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
        )
        return dataloader, dataset
    
    def get_dataloader(self, set:Sets, transforms:list, with_flatter=True, shuffle=True):
        return self._get_dataset_dataloader(
            f"{self.root_data_dir}/{set}.{self.main_data.type}", 
            shuffle=shuffle,
            transforms = transforms,
            with_flatter = with_flatter
        )
            