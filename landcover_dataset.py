from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
from PIL import Image
from sklearn import decomposition
from pathlib import Path
from torch import tensor

class LandCoverDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        root = Path(root)
        self.data = np.load(root)
        self.splits = np.split(self.data, self.data.shape[0], axis=0)

    def __getitem__(self, i):
        data, mask = self.splits[i][:,0:-1,:,:], self.splits[i][:,-1:,:,:]
        data = (data/32768.0).squeeze()
        mask = mask.squeeze()
        return tensor(data), tensor(mask)

    def __len__(self):
        return len(self.splits)

if __name__ == '__main__':
    dataset = LandCoverDataset(r'C:\Users\eliton\Documents\ml\pca\datasets\8-la.npy')
    next(iter(dataset))