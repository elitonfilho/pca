# From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py

import torch
from torch import nn
from torch import Tensor
from torchsummary import summary

class VanillaVAE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 d: list = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        if d is None:
            d = [32,16,8,4]

        self.enc1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=d[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[0]),
                        nn.LeakyReLU())
        self.enc2 = nn.Sequential(
                        nn.Conv2d(d[0], out_channels=d[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[1]),
                        nn.LeakyReLU())
        self.enc3 = nn.Sequential(
                        nn.Conv2d(d[1], out_channels=d[2], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[2]),
                        nn.LeakyReLU())
        self.enc4 = nn.Sequential(
                        nn.Conv2d(d[2], out_channels=d[3], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[3]),
                        nn.LeakyReLU())
        self.bottleneck = nn.Sequential(
                        nn.Conv2d(d[3], out_channels=d[3], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[3]),
                        nn.LeakyReLU())        
        self.dec1 = nn.Sequential(
                        nn.Conv2d(d[2], out_channels=d[2], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[2]),
                        nn.LeakyReLU())
        self.dec2 = nn.Sequential(
                        nn.Conv2d(d[1], out_channels=d[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[1]),
                        nn.LeakyReLU())
        self.dec3 = nn.Sequential(
                        nn.Conv2d(d[0], out_channels=d[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[0]),
                        nn.LeakyReLU())
        self.dec4 = nn.Sequential(
                        nn.Conv2d(d[0], out_channels=in_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(in_channels),
                        nn.Tanh())
        
    def forward(self, x: Tensor, **kwargs):
        x = self.enc1(x)
        x2 = self.enc2(x)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        bot = self.bottleneck(x4)
        x = self.dec1(torch.cat((bot,x4), 1))
        x = self.dec2(torch.cat((x,x3), 1))
        x = self.dec3(torch.cat((x,x2), 1))
        x = self.dec4(x)
        return x

if __name__ == '__main__':
    model = VanillaVAE(9)
    # v = torch.zeros((1,8,256,256))
    summary(model.cuda(), (9,256,256))
    # # model(v)