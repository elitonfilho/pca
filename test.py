import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from model import *
from utils import *
from metrics import Metrics
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    unet_resnet = UNetResNet(num_classes=19, in_channels=6)
    model_path= './unet6'
    pretrained_model = torch.load(model_path)
    for name, tensor in pretrained_model.items():
        unet_resnet.state_dict()[name].copy_(tensor)

    val_dataset = TrainDataset(typeD='val', pca=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=5, shuffle=False)

    # unet_resnet.cuda()
    unet_resnet.eval()
    softmax2d = torch.nn.Softmax2d()

    count = 0
    metric = Metrics()
    for img, mask , label in val_loader:

        with torch.no_grad():
            pred = unet_resnet(img.float())
            pred = softmax2d(pred)
            pred = pred.argmax(1)

            pred = pred.data.cpu().numpy()
            mask = mask.data.cpu().numpy()
            label = label.data.cpu().numpy()

            metric.update(pred, label)
            print(metric.metrics(mask, pred, label))

        # bz = 5
        # fig, ax = plt.subplots(3,5)
        # for i in range(bz):
        #     ax[0,i].imshow(img[i,0,:,:].squeeze())
        #     ax[1,i].imshow(label[i].squeeze())
        #     # ax[1,i].imshow(mask[i].squeeze())
        #     ax[2,i].imshow(pred[i], cmap='gray')

        # plt.show()


    
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