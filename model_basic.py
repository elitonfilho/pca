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
            hidden_dims = [8,4,2,1]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3, stride = 1, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 1,
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

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) list of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs):
        x = self.encoder(input)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

if __name__ == '__main__':
    model = VanillaVAE(9,256*256)
    # v = torch.zeros((1,8,256,256))
    # summary(model.cuda(), (9,256,256))
    # # model(v)