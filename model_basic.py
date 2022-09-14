# From https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py

import torch
from torch import Tensor, nn

class VanillaAE(nn.Module):
    '''Simple AE implementation that does not reduce input spatial shape.
    In other words, for input images of shape (cin, win, hin), the bottleneck has shape (hidden_dims[-1], w_in, h_in)
    '''

    def __init__(self,
                 inch: int,
                 hidden_dims: list[int] = None,
                 outch=9,
                 **kwargs) -> None:

        super(VanillaAE, self).__init__()

        modules = []

        if hidden_dims is None:
            hidden_dims = [8, 4, 2, 1]

        # Build Encoder
        for hch in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(inch, hch,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hch),
                    nn.LeakyReLU())
            )
            inch = hch

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
                              stride=1,
                              padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1],
                      outch,
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
    model = VanillaAE(2)
    v = torch.zeros((1,2,256,256))
    print(model(v).shape)