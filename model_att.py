import torch
import torchvision
from torch import nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )     

class UNet(nn.Module):

    def __init__(self,in_ch, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(in_ch, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.att = SelfAttention(512)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)

        x = self.att(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_ch, k=8):
        super().__init__()
        self.in_ch = in_ch
        self.k = k
        self.convf = nn.Conv2d(in_ch, in_ch//k, 1)
        self.convg = nn.Conv2d(in_ch, in_ch//k, 1)
        self.convh = nn.Conv2d(in_ch, in_ch, 1)
        self.convv = nn.Conv2d(in_ch, in_ch, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, ch, w, h = x.shape
        f = self.convf(x)
        f = f.view(bs, -1, h*w).transpose(1, 2)
        g = self.convg(x)
        g = g.view(bs, -1, h*w)
        att = torch.bmm(f, g)
        att = self.softmax(att).transpose(1, 2)
        _h = self.convh(x)
        _h = _h.view(bs, -1, h*w)
        att_h = torch.bmm(_h, att)
        att_h = att_h.view((bs, ch, w, h))
        att_h = self.convv(att_h)
        return x + self.gamma*att_h


def convrelu2x(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, 1),
        nn.ReLU(inplace=True)
    )


def convreluatt(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1),
        nn.ReLU(inplace=True),
        # nn.Conv2d(out_ch,out_ch,3,1),
        # nn.ReLU(inplace=True),
        SelfAttention(out_ch)
    )


class UnetAttention(nn.Module):
    def __init__(self, n_class, in_ch):
        super().__init__()
        sequence = [convreluatt(x, x-1) for x in range(in_ch, 4, -1)]
        print(sequence)
        self.blk = nn.Sequential(*sequence)

    def forward(self, x):
        return self.blk(x)


if __name__ == '__main__':
    nch = 5
    unet = UNet(nch,nch)
    # unet = UnetAttention(n_class=7, in_ch=nch)
    unet = unet.cuda()
    teste = torch.ones((2, nch, 64, 64)).cuda()
    print(unet(teste).shape)
    # params = sum((x.numel() for x in unet.parameters()))
