import pandas as pd
import torch


class HeartHealthDataset(torch.utils.data.Dataset):
    def __init__(self, data_path_cvs: str):

        self.data_path_csv = data_path_cvs

        df = pd.read_csv(self.data_path_csv)
        
        self.labels = df['DEATH_EVENT'].astype(int)
        self.data = df.drop(['DEATH_EVENT'], axis=1).astype(float)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        X = self.data.iloc[idx].values
        y = self.labels.iloc[idx]
        return X, y
    
    def get_raw_data(self):
        return self.data, self.labels

    def get_col_names(self):
        return self.data.columns.tolist()
    