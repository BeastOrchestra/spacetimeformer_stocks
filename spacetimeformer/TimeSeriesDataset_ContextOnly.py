import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset_ContextOnly(Dataset):
    def __init__(self, folder_name, file_name, context_length):
        self.context_length = context_length
        self.data_file = self.load_csv_file(folder_name, file_name)
        self.cumulative_length = len(self.data_file) - self.context_length

    def load_csv_file(self, folder, file_name):
        file_path = os.path.join(folder, file_name)
        return pd.read_csv(file_path, index_col=0).values  # Treat the first column as index

    def __len__(self):
        return self.cumulative_length

    def __getitem__(self, idx):
        context = self.data_file[idx:idx+self.context_length]
        return torch.tensor(context, dtype=torch.float)

# Example usage
# dataset = TimeSeriesDataset(folder_name='path_to_folder', file_name='data.csv', context_length=10)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

