from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from pathlib import Path

root = Path(r'D:\datasets\landcover\val')
nComp = 8
results = []

for x in root.iterdir():
    with np.load(root / x) as array:
        results.append(array['arr_0'][0])

imgs_np = np.array(results)
print('Original shape:', imgs_np.shape)

to_pca = imgs_np[:,10:19,:,:]
mask = imgs_np[:,8,:,:]
mask = np.expand_dims(mask, axis=1)
to_pca = to_pca.reshape(-1, to_pca.shape[1])

pca = decomposition.PCA(n_components=nComp)
red = pca.fit_transform(to_pca)

red = red.reshape(-1, nComp, 256, 256)
print('After PCA shape:', red.shape)

red = np.append(red, mask, axis=1)
print('After appending mask:', red.shape)

np.save(r'C:\Users\eliton\Documents\ml\pca\datasets\8-la.npy', red)