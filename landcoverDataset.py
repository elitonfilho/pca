from pathlib import Path

import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        data = np.load(Path(path))
        self.splits = np.split(data, data.shape[0], axis=0)[:100]

    def __getitem__(self, i):
        data = self.splits[i][..., :-1].squeeze().transpose(2,0,1)
        mask = self.splits[i][..., -1].squeeze()
        return tensor(data), tensor(mask)

    def __len__(self):
        return len(self.splits)

if __name__ == '__main__':
    dataset = LandCoverDataset(r'/home/users/jeafilho/dataset/original1.npy')
    minV = float('inf')
    maxV = float('-inf')
    for data, mask in dataset:
        maxV = max(maxV, torch.max(data))
        minV = min(minV, torch.min(data))
    print(minV, maxV)
