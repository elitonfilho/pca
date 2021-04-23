import torch, torchvision
from model import *
from utils import *
from landcover_dataset import LandCoverDataset

if __name__ == "__main__":
    train_dataset = LandCoverDataset(root='data_source/8-la.npy')
    # val_dataset = TrainDataset(typeD='val')
    # test_dataset = TrainDataset(typeD='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=1, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=1, shuffle=False)

    unet_resnet = UNetResNet(num_classes=7, in_channels=8)
    unet_resnet = unet_resnet.cuda()
    unet_resnet.train()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_resnet.parameters(), lr=0.0001, weight_decay=0.001)

    for epoch_idx in range(100):

        loss_batches = []
        for batch_idx, data in enumerate(train_loader):
        
            imgs, masks = data
            imgs = imgs.cuda()
            masks = masks.long().cuda()

            y = unet_resnet(imgs)
            loss = cross_entropy_loss(y, masks.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batches.append(loss.data.cpu().numpy())

        print('epoch: ' + str(epoch_idx) + ' training loss: ' + str(np.sum(loss_batches)))

    model_file = './t1.pth'
    unet_resnet = unet_resnet.cpu()
    torch.save(unet_resnet.state_dict(), model_file)
    # unet_resnet = unet_resnet.cuda()
    print('model saved')