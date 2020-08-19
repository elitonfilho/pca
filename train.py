import torch, torchvision
from model import *
from utils import *


if __name__ == "__main__":
    train_dataset = TrainDataset(typeD='train')
    val_dataset = TrainDataset(typeD='val')
    # test_dataset = TrainDataset(typeD='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=1, batch_size=1, shuffle=False)

    unet_resnet = UNetResNet(num_classes=19, in_channels=6)
    unet_resnet = unet_resnet.cuda()
    unet_resnet.train()
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unet_resnet.parameters(), lr=0.0001, weight_decay=0.0001)

    for epoch_idx in range(10):

        loss_batches = []
        for batch_idx, data in enumerate(train_loader):
        
            imgs, _ , masks = data
            # imgs = torch.tensor(imgs).cuda()
            # masks = torch.tensor(masks).long().cuda()
            imgs = torch.autograd.Variable(imgs).float().cuda()
            masks = torch.autograd.Variable(masks).cuda()

            y = unet_resnet(imgs)
            loss = cross_entropy_loss(y, masks.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batches.append(loss.data.cpu().numpy())

        print('epoch: ' + str(epoch_idx) + ' training loss: ' + str(np.sum(loss_batches)))

    model_file = './unet-final'
    unet_resnet = unet_resnet.cpu()
    torch.save(unet_resnet.state_dict(), model_file)
    # unet_resnet = unet_resnet.cuda()
    print('model saved')