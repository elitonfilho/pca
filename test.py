import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from model import *
from utils import *
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    unet_resnet = UNetResNet(num_classes=19, in_channels=6)
    model_path= './unet-final'
    pretrained_model = torch.load(model_path)
    for name, tensor in pretrained_model.items():
        unet_resnet.state_dict()[name].copy_(tensor)

    val_dataset = TrainDataset(typeD='val', pca=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=5, shuffle=False)

    # unet_resnet.cuda()
    unet_resnet.eval()
    softmax2d = torch.nn.Softmax2d()

    count = 0
    for img, mask , label in val_loader:
    # img = cv2.imread('./data/val/hr/2953-3-SO_300_HR.png')
    # img = np.expand_dims(img, axis=0)
    # norm_img = np.zeros((256,256))
    # img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
    # img = img.transpose(0, 3, 1, 2)
    # img = torch.FloatTensor(img)
    # img = img.cuda()

        with torch.no_grad():
            pred = unet_resnet(img.float())
            pred = softmax2d(pred)
            pred = pred.argmax(1)
        pred = pred.data.cpu().numpy()

        # for i in range(19):
        #     plt.subplot(1,19,i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(pred[:,i,:,:].squeeze(), cmap='gray')

        bz = 5

        fig, ax = plt.subplots(3,5)
        for i in range(bz):
            ax[0,i].imshow(img[i,0,:,:].squeeze())
            ax[1,i].imshow(label[i].squeeze())
            # ax[1,i].imshow(mask[i].squeeze())
            ax[2,i].imshow(pred[i], cmap='gray')

        # plt.show()

        # fig, ax = plt.subplots(1,3)
        # print(img.shape)
        # ax[0].imshow(img.squeeze()[0,:,:])
        # ax[1].imshow(mask.squeeze())
        # ax[2].imshow(pred, cmap='gray')

        plt.show()

        # cv2.imwrite(f'output/mask{count}.png', pred)
        # count+= 1
    
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for data in val_loader:
    #         inputs, labels = data
    #         out = unet_resnet(img)
    #         pred = softmax2d(out)
    #         _, predicted = torch.max(pred.data, 1)
    #         # print(torch.histc(predicted))
    #         # print(torch.histc(labels))
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         print(correct, total)

    #     print('Accuracy of the network on test images: %0.3f %%' % (
    #     100 * correct / total))