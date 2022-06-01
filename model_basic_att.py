# From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        inch = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 4, 2, 1]

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU(),
                    SelfAttention(hidden_dims[0])
                    )

        # Build Encoder
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], out_channels=hidden_dims[i+1],
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.att = SelfAttention(8)

        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i],
                              hidden_dims[i + 1],
                              kernel_size=3,
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1],
                      inch,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh())

    def forward(self, input: Tensor, **kwargs):
        x = self.first_layer(input)
        x = self.encoder(x)
        x = self.att(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x


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


if __name__ == '__main__':
    model = VanillaVAE(9, 256*256).cuda()
    v = torch.zeros((1,9,256,256)).cuda()
    # summary(model.cuda(), (9,256,256))
    print(model(v).shape)
