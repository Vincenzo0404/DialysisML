import numpy as np
import torch
from torch.utils.data import Dataset


class NumDaysDataset(Dataset):
    def __init__(self, X_numpy: np.ndarray, y_numpy: np.ndarray):

        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        # returns the number of windows
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RiskDataset(Dataset):
    def __init__(self, X_numpy: np.ndarray, y_numpy: np.ndarray):

        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.float32)

    def __len__(self):
        # returns the number of windows
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
