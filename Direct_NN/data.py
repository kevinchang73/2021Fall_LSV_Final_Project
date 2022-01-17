import numpy as np
import torch
from torch.utils.data import Dataset

class TLNDataset(Dataset):
    def __init__(self, X, Y):
        self.data = torch.from_numpy(X).bool()
        Y = Y.astype(np.bool)
        self.label = torch.from_numpy(Y).bool()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
