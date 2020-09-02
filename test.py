import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from PIL import Image
from model import *
from utils import *
from metrics import Metrics
import matplotlib.pyplot as plt
import cv2

def save_img(img, pred, label, ch, counter):
    lc = 0
    savedir = f'output/images/{ch}/'
    for img, lp, lt in zip(img, pred, label):
        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        ax0.axis('off')
        ax1.axis('off')
        ax2.axis('off')
        ax0.imshow(lp)
        ax1.imshow(lt)
        ax2.imshow(img[0].squeeze())
        plt.savefig(savedir + f'{counter*10 + lc}.png',pad_inches=0.1, bbox_inches='tight')
        plt.close(fig=fig)
        lc += 1



if __name__ == "__main__":
    nch = 1
    counter = 0
    mat_path = f'rit18_data.mat'
    model_path= f'output/train/unet{nch}_250'
    unet_resnet = UNetResNet(num_classes=19, in_channels=nch)
    
    pretrained_model = torch.load(model_path)
    for name, tensor in pretrained_model.items():
        unet_resnet.state_dict()[name].copy_(tensor)

    val_dataset = TrainDataset(typeD='val', pca=True, mat_path=mat_path, ncomp = nch)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=10, shuffle=False)

    unet_resnet.cuda()
    unet_resnet.eval()
    softmax2d = torch.nn.Softmax2d()

    metric = Metrics(nbands=nch)
    for img, mask , label in val_loader:

        with torch.no_grad():
            pred = unet_resnet(img.float().cuda())
            pred = softmax2d(pred)
            pred = pred.argmax(1)

            pred = pred.data.cpu().numpy()
            mask = mask.data.cpu().numpy()
            label = label.data.cpu().numpy()
            img = img.clone().data.cpu().numpy()

            metric.update(pred, label)

            save_img(img.astype(np.uint8), pred.astype(np.uint8), label.astype(np.uint8), nch, counter)
            counter+=1

    fig, ax = plt.subplots(3,5)

    print(metric.metrics(mask, pred, label))

        # bz = 5
        # fig, ax = plt.subplots(3,5)
        # for i in range(bz):
        #     ax[0,i].imshow(img[i,0,:,:].squeeze())
        #     ax[1,i].imshow(label[i].squeeze())
        #     # ax[1,i].imshow(mask[i].squeeze())
        #     ax[2,i].imshow(pred[i], cmap='gray')

        # plt.show()