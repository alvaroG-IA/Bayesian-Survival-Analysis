import pandas as pd
import numpy as np
import torch

class HeartHealthDataset(torch.utils.data.DataSet):
    def __init__(self, data_path_cvs: str):

        self.data_path_csv = data_path_cvs

        df = pd.read_csv(self.data_path_csv)
        
        self.labels = torch.from_numpy(df['DEATH_EVENT'].astype(int))
        self.data = torch.from_numpy(df.drop(['DEATH_EVENT'], axis=1).astype(float))

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y
    