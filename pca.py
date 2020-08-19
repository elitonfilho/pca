from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat
from readRIT import getTrainData

nComp = 3
dataset = loadmat('rit18_data.mat')

#Load Training Data and Labels
train_data = dataset['train_data']
train_mask = train_data[-1]
train_data = train_data[:6]
train_data = train_data.reshape(-1, train_data.shape[0])

pca = decomposition.PCA(n_components=nComp)
red = pca.fit_transform(train_data)
red = red.reshape((nComp,9393,-1))
dataset['train_data'] = np.concatenate([red, train_mask.reshape((1,train_mask.shape[0], -1))])
print('New shape train:', dataset['train_data'].shape)

#Load Validation Data and Labels
val_data = dataset['val_data']
val_mask = val_data[-1]
val_data = val_data[:6]
val_data = val_data.reshape(-1, val_data.shape[0])
pca = decomposition.PCA(n_components=nComp)
red = pca.fit_transform(val_data)
red = red.reshape((nComp,val_data.shape[-2],-1))
dataset['val_data'] = np.concatenate(red, val_mask.reshape((1, val_mask.shape[0], -1)))
print(dataset['val_sata'].shape)

#Load Test Data
test_data = dataset['test_data']
test_mask = test_data[-1]
test_data = test_data[:6]
test_data = test_data.reshape(-1, test_data.shape[0])
pca = decomposition.PCA(n_components=nComp)
red = pca.fit_transform(test_data)
red = red.reshape((nComp,test_data.shape[-2],-1))
dataset['test_data'] = np.concatenate(red, test_mask.reshape((1, test_mask.shape[0], -1)))
print(dataset['test_data'].shape)

savemat('3band.mat' , dataset)

# data, labels = getTrainData()
# print(data.shape)
# data = data.reshape(-1, data.shape[0])

# pca = decomposition.PCA(n_components=nComp)
# red = pca.fit_transform(data)
# inv = pca.inverse_transform(red)
# print(red.shape, inv.shape)

red = red.reshape((3,9393,-1))

plt.imshow(red)
plt.show()


# img =np.array(Image.open('lena.jpg'), np.int)
# img = img.transpose((2,0,1))
# a = img[0,:,:].squeeze()
# pca = decomposition.PCA(n_components=nComp)
# red = pca.fit_transform(a)
# rec = pca.inverse_transform(red)
# print(rec.shape)
# # rec = np.dot(pca.transform(a)[:,:nComp], pca.components_[:nComp,:])
# plt.imshow(rec, cmap=plt.cm.gray)
# plt.show()