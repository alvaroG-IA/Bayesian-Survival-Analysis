import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class HeartHealthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_cvs: str):

        self.data_path_csv = data_path_cvs

        df = pd.read_csv(self.data_path_csv)
        
        self.labels = df['DEATH_EVENT'].astype(int)
        self.data = df.drop(['DEATH_EVENT'], axis=1).astype(float)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y
    
    def preproces(self, mode: str = 'std'):
        if mode == 'std':
            self.std = StandardScaler()
        elif mode == 'min_max':
            self.std = MinMaxScaler()

        self.data_processed = self.std.fit_transform(self.data)
        
        return self.data_processed, self.labels
    