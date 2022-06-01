# From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py

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

        self.enc11 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=d[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[0]),
                        nn.LeakyReLU())
        self.enc12 = nn.Sequential(
                        nn.Conv2d(d[0], d[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[0]),
                        nn.LeakyReLU())
        self.enc21 = nn.Sequential(
                        nn.Conv2d(d[0], out_channels=d[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[1]),
                        nn.LeakyReLU())
        self.enc22 = nn.Sequential(
                        nn.Conv2d(d[1], out_channels=d[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[1]),
                        nn.LeakyReLU())
        self.enc31 = nn.Sequential(
                        nn.Conv2d(d[1], out_channels=d[2], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[2]),
                        nn.LeakyReLU())
        self.enc32 = nn.Sequential(
                        nn.Conv2d(d[2], out_channels=d[2], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[2]),
                        nn.LeakyReLU())
        self.enc41 = nn.Sequential(
                        nn.Conv2d(d[2], out_channels=d[3], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[3]),
                        nn.LeakyReLU())
        self.enc42 = nn.Sequential(
                        nn.Conv2d(d[3], out_channels=d[3], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[3]),
                        nn.LeakyReLU())
        self.bottleneck = nn.Sequential(
                        nn.Conv2d(d[3], out_channels=d[3], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[3]),
                        nn.LeakyReLU())        
        self.dec1 = nn.Sequential(
                        nn.Conv2d(d[3], out_channels=d[2], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[2]),
                        nn.LeakyReLU())
        self.dec2 = nn.Sequential(
                        nn.Conv2d(d[2], out_channels=d[1], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[1]),
                        nn.LeakyReLU())
        self.dec3 = nn.Sequential(
                        nn.Conv2d(d[1], out_channels=d[0], kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(d[0]),
                        nn.LeakyReLU())
        self.dec4 = nn.Sequential(
                        nn.Conv2d(d[0], out_channels=in_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(in_channels),
                        nn.Tanh())
        
    def forward(self, x: Tensor, **kwargs):
        k = 0.1
        _x = self.enc11(x)
        x = self.enc12(_x) + k*_x
        _x = self.enc21(x)
        x = self.enc22(_x) + k*_x
        _x = self.enc31(x)
        x = self.enc32(_x) + k*_x
        _x = self.enc41(x)
        x = self.enc42(_x) + k*_x
        x = self.bottleneck(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x

if __name__ == '__main__':
    model = VanillaVAE(9)
    # v = torch.zeros((1,8,256,256))
    summary(model.cuda(), (9,256,256))
    # # model(v)