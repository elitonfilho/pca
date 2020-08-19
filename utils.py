from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.io import loadmat
from PIL import Image
from sklearn import decomposition


class TrainDataset(Dataset):

    def __init__(self, mat_path='rit18_data.mat', patch=(256, 256), typeD='train', ncomp=3, pca=False):
        super().__init__()
        dataset = loadmat(mat_path)
        self.patch = patch
        self.typeD = typeD
        train_data, train_mask, train_labels = dataset[f'{typeD}_data'][
            :6], dataset[f'{typeD}_data'][-1], dataset[f'{typeD}_labels']
        train_data, train_mask, train_labels = np.array(train_data, dtype='int'), np.array(
            train_mask, dtype='float'), np.array(train_labels, dtype='float')
        train_data = self.pca(train_data, ncomp) if pca else train_data
        self.train_data, self.train_mask, self.train_label = self.crop(
            train_data, train_mask, train_labels)

    def crop(self, train_data, train_mask, train_labels):
        data = []
        mask = []
        labels = []
        tsize = train_mask.shape
        for i in range(tsize[0] // self.patch[0]):
            for j in range(tsize[1] // self.patch[1]):
                data_patch = train_data[:, self.patch[0]*i:self.patch[0]
                                        * (i+1), self.patch[1]*j:self.patch[1]*(j+1)]
                mask_patch = train_mask[self.patch[0]*i:self.patch[0]
                                        * (i+1), self.patch[1]*j:self.patch[1]*(j+1)]
                labels_patch = train_labels[self.patch[0]*i:self.patch[0]
                                            * (i+1), self.patch[1]*j:self.patch[1]*(j+1)]
                if self.check_image(mask_patch):
                    data.append(data_patch)
                    mask.append(mask_patch)
                    labels.append(labels_patch)
                else:
                    continue
        return data, mask, labels

    @staticmethod
    def pca(data, ncomp):
        shape = data.shape
        data = data.reshape(-1, data.shape[0])
        pca = decomposition.PCA(n_components=ncomp)
        red = pca.fit_transform(data)
        red = red.reshape((ncomp, shape[-2], -1))
        return red

    @staticmethod
    def check_image(mask):
        mask = (mask > 0)
        mask = mask.reshape(-1)
        count = np.bincount(mask)
        if len(count) == 1:
            return False
        return count[1] > count[0]


    def __getitem__(self, i):
        return self.train_data[i], self.train_mask[i], self.train_label[i]

    def __len__(self):
        return len(self.train_data)


def saveImage(typeD='train'):
    dataset = loadmat('rit18_data.mat')
    train_data, train_mask, train_labels = dataset[f'{typeD}_data'][:
                                                                    6], dataset[f'{typeD}_data'][-1], dataset[f'{typeD}_labels']
    im_label = Image.fromarray(train_labels)
    im_label.save(f'{typeD}.png')


if __name__ == "__main__":
    # saveImage('train')
    train_dataset = TrainDataset(typeD='train')
    train_loader = DataLoader(
        train_dataset, num_workers=1, batch_size=5, shuffle=True)
    print(len(train_loader))
