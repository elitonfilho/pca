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
        data = self.splits[i][:,:-1,...].squeeze()
        mask = self.splits[i][:,-1,...].squeeze()
        return tensor(data), tensor(mask)

    def __len__(self):
        return len(self.splits)

if __name__ == '__main__':
    dataset = LandCoverDataset(r'C:\Users\eliton\Documents\ml\pca\dataSourceV2\original.npy')
    minV = 10000
    maxV = -10000
    for img1, img2 in dataset:
        # print(img1.shape, img2.shape)
        # print(torch.min(img1), torch.max(img1))
        maxV = max(maxV, torch.max(img2))
        minV = min(minV, torch.min(img2))
    print(minV, maxV)
